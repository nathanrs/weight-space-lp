# weight-space-lp: implementation plan

> **Status (2026-11-07).** This is the plan as written before any experiment ran, kept unedited below this box as a
> record. Results and the write-up are in the [README](../README.md); numbers live in [`results/`](../results/).
>
> | Item | Status | What actually happened |
> |---|---|---|
> | E0 reproduce | **done**, all gates pass | sha256 rebuild, rank-1, u re-derived (cos 1.0000), TF metric −0.1032 vs −0.1031; harness: MATH-500 Astra 50.0 / base 7.0 (×1 sample; logs 48.1 / 7.9 ×4). [results/reproduction](../results/reproduction/README.md) |
> | E1 data | **done**, changed | Activations from MIT's 452 released base traces plus 512 of our own base generations on practice problems. Loop labels use 16-token windows and an 80%-duplicate rule. Cached only layers 15/19 (not all 24); no off-domain P_gen. |
> | E2 sparsity sweep | **done**, changed | LP-A with a constant δ (Astra's mean suppression), not per-token. HiGHS IPM + cutting planes were too slow on dense rows → replaced by a PDHG (PDLP-style) solver on MPS/GPU. L15/L19 solved independently on base activations (no sequential recollection). The TF metric did **not** predict generation, so selection used dev generation only. [results/dev_sweep](../results/dev_sweep/README.md) |
> | E3 overlap | **done** (partial) | LP supports barely overlap Astra's 64 (3/8 at L15, 1/8 at L19 for `shift_k8`). LP-B restricted to Astra's support not run. [overlap.json](../results/analysis/overlap.json) |
> | E4 baselines | **partial** | Done: base, Astra, recipe-no-search, random control (at Astra's sparsity), steering vector. Not done: α-ablation sweep, ridge/L2, sampling penalties, LoRA. |
> | E5 ablations | **partial** | Done: direction source (Astra's u vs. real-loop mean diff; not the probe direction), blind strength 2×/4×/8×. Not done: onset vs. during (all fits used R_during), single layer, blind layer choice, preservation set (IFEval checked instead). |
> | E6 budget-matched pipeline | **not done** | |
> | E7 stretch | **not done** | |
> | Full finalist evals | **done** | MATH-500 ×4, GSM8K test ×2, IFEval ×1 (plan said ×2), plus 512 untouched practice problems. [results/benchmarks](../results/benchmarks/README.md) |
> | Compute | | Measured ~7.2k generated tok/s on one H100 PCIe; ~18 H100-hours + ~1 A100-hour total, ~$65 ([spend](../results/spend.md)). |
>
> Kill criteria: none triggered. "Astra recipe, no search" reached 44.5 vs. Astra's 59.4 on dev, so the search
> mattered, and the LP's claim rests on matching that with 8× fewer weights and one solve.

**Thesis.** The MIT "weight-space programming" Experiment 8 edit (GPT-6 Astra fixing Qwen3.5-0.8B's thinking loops)
is a rank-1 edit whose two free choices (which neurons, how much) can be posed as a linear program over cached
activations. Diagnosis (direction + layers) is the hard part; compiling it into weights is a solver's job.
We test whether an LP, given the same data budget, matches or beats Astra's edit on sparsity, collateral drift,
and forward-pass budget.

Source of truth for the original: `RikiyaT/lm_understand-logs/exp8_math` (cloned to scratch; key files cited below).

---

## 0. What Astra actually did (facts from the logs)

| Item | Value | Where |
|---|---|---|
| Final edit (p50c, 50 passes) | `W = W0 − 6 · outer(u, mask64 ⊙ (uᵀW0))` on `layers.{15,19}.mlp.down_proj.weight` | `runs/p50c/work_notes/final_report.md` |
| Shape | down_proj is `[1024, 3584]`; 64 of 3584 columns changed per layer → 2 × 64 × 1024 = 131,072 weights | `edit/manifest.json` |
| Neurons | explicit index lists, 64 per layer | `work_notes/frozen_weights_manifest.json` |
| Direction `u` | unit vector, mean diff of residual stream (hidden_states 16 / 20 = outputs of layers 15 / 19) between **4 hand-written productive vs. repetitive text pairs** | `requests/fwd_4.json.claimed`, `experiment_history.md` |
| Neuron selection | top-64 writers of the repetitive-minus-productive contrast on those 4 pairs (~91% / 98% of contrast, 50% / 40% of the directional weight norm) | `experiment_history.md` |
| α = 6 | **not** ablation: the u-component of each selected column goes from c to −5c. Dose curve α = 1, 2, 3, 4, 6 kept improving | `experiment_history.md` |
| TF metric (top64 α6) | repetitive-branch logp −0.103 nats/tok, productive −0.009 | same, pass 29 |
| Random control | norm-matched orthogonal rank-1: +1.6 pp (SE 3.5) | pass 23 |
| Eval protocol | vLLM, thinking on, T=0.6, top_p 0.95, top_k 20, **presence_penalty 1.5**, max 32,768 new tokens, last `\boxed{}` graded with math-verify | `README.md`, `code/profiles.py` (`think35`) |
| Results | MATH-500 7.9 → 48.1, GSM8K 13.8 → 67.1, IFEval 21.0 → 62.3, MATH-500 cut share 93.0 → 33.1 | `eval/*/merged.json` |
| Other budgets | p10c: α=1.5 rank-1 removal across 30 matrices (12.3 MATH); p25c: zero L15 attn head 7 (14.1 MATH) | `README.md` |

Implications for the design:
- Astra's edit is exactly `ΔW = u vᵀ` with `v_j = −6 · uᵀw_j` for j ∈ S, else 0. **It is a feasible point of
  the LP below**, so we can score it on our own constraints and compare like-for-like.
- A rank-1 edit along u changes the residual stream **only along u**: Δresid_t = u · (vᵀh_t). Every other
  direction is preserved exactly, so the LP's local constraints are tight descriptions of the edit's local effect.
- The repetition-penalty baseline is partly done already: the protocol uses presence_penalty 1.5 and the base
  model still loops 93% of the time. Our sampling baselines have to go further than that (see E4).
- Layers 15 and 19 are both `full_attention` layers in a hybrid (Gated DeltaNet + full attention every 4th) stack.
  That's worth noting but doesn't change the MLP edit.
- `evaluate.py` imports an unreleased `weightenv` package, so we reimplement the eval harness against the
  documented protocol and reuse `mathqa.py`'s grading where possible.

---

## 1. The LP

Per layer ℓ, using cached down_proj inputs `h_t ∈ R^3584` (= `act(gate) * up`), fixed unit direction `u ∈ R^1024`:

- base signal `s_t = uᵀ W0 h_t` (how much the MLP writes u at token t)
- edit `ΔW = u vᵀ`, unknown `v ∈ R^3584`, split `v = v⁺ − v⁻`, both ≥ 0
- slack `ξ_t ≥ 0`

**LP-A (sparsity: match Astra's suppression, minimize weights touched)**
```
min   Σ(v⁺ + v⁻) + C · Σ ξ
s.t.  vᵀh_t ≤ −δ_t + ξ_t          t ∈ R   (repetitive / loop-onset tokens)
      |vᵀh_t| ≤ ε + ξ_t           t ∈ P   (productive tokens)
```
`δ_t` takes one of three forms: (i) Astra-matched, `δ_t = −v_Astraᵀh_t`, i.e. at least as much suppression as
Astra on every token; (ii) relative, `δ_t = κ · s_t`, the analog of α; (iii) absolute, push `s_t + vᵀh_t` below the
productive-token mean of s. `ε` defaults to Astra's own productive-token spread, `max_P |v_Astraᵀh|` at some
quantile, so LP-A is "same suppression, same or better collateral, fewer weights".

**LP-B (preservation: match suppression, minimize collateral)**
```
min   ε + λ · Σ(v⁺ + v⁻)
s.t.  vᵀh_t ≤ −δ_t + ξ_t, t ∈ R;   |vᵀh_t| ≤ ε, t ∈ P;   optionally supp(v) ⊆ S_Astra
```
This is a Chebyshev (minimax) objective on productive drift. With `supp(v) ⊆ S_Astra` it asks: given Astra's
neurons, what are the optimal per-neuron strengths?

**Effective sparsity.** The model is evaluated in bf16. Entries `u_i · v_j` that are small relative to `W0_ij`
round back to the original value, so we always report L0 after the bf16 cast (Astra: 130,188 of 131,072 actually
changed). Note: one nonzero `v_j` touches a full 1024-entry column, so "weights changed" ≈ 1024 · ‖v‖₀.

**Scale.** 3584 × 2 variables. Constraints = labeled tokens. Dense rows: 20k tokens is about 72M nonzeros. Plan:
1. HiGHS (via `highspy`) on a stratified subsample (~4k R + ~8k P), solved with IPM.
2. **Cutting planes**: check all cached tokens (~10⁵–10⁶, a cheap matmul), add the worst violators, re-solve, and
   repeat until the violation rate on held-out tokens is below 1%.
3. If that is too slow, use PDLP (HiGHS's built-in PDLP, with a GPU build if available), which only needs mat-vecs.

**Two layers.** Solve L15 first. Then recollect L19 activations with the L15 edit applied (one TF pass) and
solve L19. This avoids the stale-activation problem without needing a joint nonlinear solve.

---

## 2. Repo layout

```
weight-space-lp/
  PLAN.md
  pyproject.toml              # torch, transformers>=5.17, vllm (Qwen3.5 support), highspy, scipy, numpy,
                              #   math-verify, datasets, safetensors, ifeval checkers (lm-eval or google ifeval pkg)
  src/wslp/
    model.py                  # load Qwen3.5-0.8B text model @ rev 2fc0636; hooks on mlp.down_proj input & resid
    astra.py                  # rebuild p50c edit (port apply_edit.py), extract u and S, compute v_Astra
    loops.py                  # loop detection (duplicate 8-gram tails, like Astra's metric); token labeling
    collect.py                # teacher-forced activation caching: h_t (3584) and s_t per layer, labels, sharded .npy
    lp.py                     # LP-A / LP-B builders, HiGHS solve, cutting-plane loop, solution → ΔW
    edits.py                  # apply rank-1 edits, bf16 cast, effective L0, write safetensors checkpoint
    baselines.py              # ablation α-sweep, top-k α (Astra recipe), random norm-matched, ridge/L2, steering
    tfmetrics.py              # Astra's branch-logp metric on transfer_probe texts; KL-to-base on held-out text
    eval/
      run_vllm.py             # think35 protocol exactly; sharded; writes per-problem json + cut flags
      grade.py                # last \boxed{} + math-verify (port mathqa.py); IFEval strict on post-</think>
      merge.py                # acc, cut_share, mean_tokens, CIs
  scripts/                    # one script per experiment below (e0_repro.py ... e6_budget.py)
  configs/                    # yaml per run
  data/                       # practice splits (copied from exp8_math/practice), probe texts
  results/                    # run outputs, figures
```

---

## 3. Experiments

Each one has a deliverable and a gate. Dev eval means paired base vs. edited on held-out practice problems
(64 GSM8K + 64 MATH, 1 sample, 32k tokens), mirroring Astra's own confirmation protocol. Full eval
(MATH-500 ×4, GSM8K ×2, IFEval ×2) is only for finalists.

### E0. Reproduce (must pass before anything else)
1. Download the base model at the pinned revision. Rebuild p50c with ported `apply_edit.py` and check every sha256.
   Cross-check against `RikiyaT/Qwen3.5-0.8B-weight-space-programming` on HF (only 2 tensors should differ).
2. Verify the structure: every changed column satisfies `ΔW[:, j] ∥ u` (rank-1, residual < 1e-5). Extract u and S.
   Confirm the selected neurons match `frozen_weights_manifest.json`.
3. Re-derive u from the 4 `fwd_4` texts on the base model and get cosine with the extracted u. A high cosine
   means we understand their pipeline.
4. Eval harness check: run base and p50c on MATH-500 (4 samples). **Gate:** cut share is ~93% / ~33% and accuracy
   is 7.9 / 48.1 within about ±2 pp. If this fails, fix the harness before going further.
5. Reproduce Astra's TF metric on `transfer_probe_request.json` / `transfer_probe_bounds.json`.
   **Gate:** ≈ −0.103 / −0.009.

### E1. Data and activations
1. Base model generates on ~1,000 practice problems with the eval protocol at 32k tokens. Expect ~90% loops, so
   the data is plentiful.
2. Labeling (`loops.py`):
   - **repetitive (R_during)**: tokens inside a loop after the 2nd occurrence of a repeated ≥8-gram
   - **onset (R_onset)**: the window of ~32 tokens just before the loop's first repetition starts
   - **productive (P)**: all tokens of finished correct traces, plus loop-trace prefixes ending ≥256 tokens before
     onset
   - **off-domain P (P_gen)**: a few thousand tokens of generic chat / Wikipedia, to guard non-math behavior
3. Teacher-force all traces through the base model. Cache `h_t` (down_proj input) for all 24 layers on a
   stratified token subsample (~50k R, 100k P per layer, fp16 ≈ 1 GB/layer), plus residual outputs for direction
   finding.
4. Splits: an LP train-token set, a held-out-token set (same traces, other positions), and held-out *problems*
   for dev eval. Problem-level splits are disjoint from anything that touches the LP.

### E2. Core result: sparsity sweep (the headline chart)
- Use Astra's u and layers 15/19 (the "given the diagnosis" setting).
- Sweep the L1 weight (equivalently, cap ‖v‖₀ at 1, 2, 4, 8, 16, 32, 64, 128 neurons per layer) for LP-A with the
  Astra-matched δ.
- For each point: effective L0 after bf16, TF metric, KL-to-base on P and P_gen, dev-eval cut share and accuracy.
- **Chart:** weights changed (log x) vs. cut share, with Astra as one point and the random control as another.
- First, a free check: score Astra's own v on our constraints, i.e. how much of our R set it suppresses and its
  ε on P.

### E3. Neuron overlap with Astra
- Jaccard of supp(v_LP) vs. S_Astra at k = 64, plus mass-weighted overlap (Σ|v_j| over j ∈ S_Astra / Σ|v_j|).
- LP-B with `supp ⊆ S_Astra`: are Astra's per-neuron strengths (uniform α = 6 × projection) close to optimal?
- Read it both ways. High overlap means the same mechanism. Low overlap with an equal fix means many solutions,
  i.e. the loop circuit is redundant.

### E4. Baselines and controls (reviewers will ask for these)
| Baseline | What it tests |
|---|---|
| Base model, protocol sampling | reference |
| Astra p50c | the target |
| **Astra recipe, no search**: u from real-loop mean diff, top-64 by contrast, α = 6 | whether Astra's 50-pass search mattered vs. a one-shot recipe (Goedecke's point) |
| Full-column directional ablation, α ∈ {1, 2, 4, 6} | abliteration-style |
| Norm-matched random rank-1 (orthogonal to u), at the LP's sparsity | Astra's control, repeated |
| Ridge / least-squares version of LP-A (same constraints, L2 objective) | whether the L1/LP formulation matters vs. a ROME-style L2 edit |
| Inference-time steering: subtract β·u at resid 16/20 | same mechanism without a weight edit; upper reference |
| Sampling: repetition_penalty 1.1–1.3, frequency_penalty, n-gram blocking (custom logits processor), forced `</think>` at a budget | whether you need weights at all; note presence_penalty 1.5 is already in the protocol |
| *(stretch)* LoRA SFT on ~1k of the model's own finished correct traces | "training is still the right tool" |

### E5. Design ablations (dev eval only)
1. **Onset vs. during**: constrain R_onset vs. R_during vs. both. Hypothesis: onset needs fewer neurons and
   generalizes better at generation time.
2. **Direction source**: Astra's synthetic-4-pairs u vs. real-loop mean diff vs. a logistic-probe direction.
   Report cosines and eval.
3. **Strength**: sweep κ (the analog of α); does the LP reproduce Astra's α-dose curve?
4. **Single layer** (L15 only, L19 only) vs. both, sequential vs. independent solves.
5. **Blind layer choice** (no Astra hindsight): run E2's solver on all 24 layers. Rank by LP objective / held-out
   R–P separation; pick the top 2. Does it find 15 and 19?
6. **Preservation set**: math-only P vs. P + P_gen (IFEval / MMLU-Pro drift).

### E6. Budget-matched pipeline (the second headline)
Emulate Astra's forward-pass accounting: 1 generation pass = up to 512 problems (or 256 paired "both"); 1 TF pass
= up to 1,024 texts. Kernel compute (the LP solve) is free, as it was for Astra.
- Pass 1: base generates 512 practice problems.
- Pass 2: TF over the resulting traces, giving activations.
- Solve the LPs (blind layer choice from E5.5, real-loop u) and pick ≤3 candidates by held-out-token metrics.
- Passes 3+: paired "both" generations to select and confirm.
- Run at 3, 5 and 10 total passes. Compare to Astra's budget table (10 passes: 87.2% cut; 25: 86.4%; 50: 33.1%).
- **Headline if it holds:** LP at ≤10 passes ≈ Astra at 50. We'll state the hindsight caveat explicitly (we know
  the problem is loops). The blind-layer variant is the one we headline.

### E7. Stretch
- **Rank > 1 / free u**: let ΔW = Σ_k u_k v_kᵀ with u from the top-k contrast PCs, or a full-block LP on a
  64 × 1024 sub-block (PDLP on GPU). Does extra freedom buy anything?
- **Gemma 3 270M negative control**: no loops (8.5% cut), so the LP should find nothing to fix. That backs up the
  "removes obstacles, can't add ability" claim.
- **Certificate check**: report the fraction of held-out tokens that violate the constraints, plus how drift at
  the edited layer propagates to final-layer logits (KL), to quantify how much the local guarantee says about the
  output.

---

## 4. Metrics (every run)
- Edit: ‖v‖₀ per layer, effective changed weights after bf16, relative Frobenius change
- Local: held-out R suppression, held-out P |vᵀh| distribution, constraint-violation rate, AUC of vᵀh between
  R and P (does the edit act as a detector?)
- TF: Astra's branch-logp deltas on their transfer-probe texts; next-token KL to base on P and P_gen
- Generation (dev): paired acc, cut share, rescues / regressions, accuracy on originally-finished outputs (should
  be unchanged), median correct length (Astra: 9,172; checks we aren't just forcing early stops), whitespace-tail
  degeneration rate
- Generation (full, finalists): MATH-500, GSM8K, IFEval, cut share. Optional: AMC23, Minerva, OlympiadBench,
  MMLU-Pro-1k, GPQA-D

## 5. Compute
- Activation caching and LP solves: one GPU for TF (minutes), CPU for HiGHS (minutes per solve, mostly in the
  cutting-plane loop).
- Eval dominates. Base-model samples run ~30k tokens each because of the loops. Full eval ≈ 2k MATH + 2.6k GSM8K
  + 1.1k IFEval samples ≈ 100–150M generated tokens per model. That's roughly 1–3 H100-hours per model with vLLM
  (to be measured in E0). Dev eval (128 samples) is about 4M tokens, minutes. **Budget:** ~6–8 finalists on full
  eval and ~100 dev evals.
- Use a 16k token cap for quick screening, as Astra did, but confirm at 32k.

## 6. Order of work and kill criteria
1. **E0** (repro + harness). Stop and debug if MATH-500 numbers don't match.
2. **E1** + LP-A on Astra's u/layers + **E2** sweep (the first chart). *Kill / pivot:* if no LP point beats the
   random control on the TF metric, the local-constraint framing is wrong. Pivot to an onset-only or
   logit-level spec.
3. **E3, E4** (overlap + baselines). If the "Astra recipe, no search" baseline already matches p50c, that's the
   Goedecke-style finding. The LP's claim then has to rest on sparsity / drift / budget, not on the fix itself.
4. **E5** ablations → **E6** budget pipeline.
5. Full evals on finalists. Write-up.

## 7. Open risks
- Local guarantees ≠ generation behavior. Loops are a dynamical property, and fixing activations at one layer on
  sampled tokens may not stop loops at generation time. E5.1 (onset) and E7 (certificate check) measure this gap.
- Astra's α = 6 overshoot suggests the fix needs *active* anti-repetition pressure, not neutralization. The
  Astra-matched δ preserves that, but the "absolute τ" variant may under-correct.
- bf16 rounding can erase small v entries. Always evaluate post-cast.
- The eval harness is reimplemented (`weightenv` isn't public), so small protocol mismatches are possible. The
  E0 gate catches this.
- Batch-dependent vLLM RNG: compare only paired-within-run (Astra hit this too).
