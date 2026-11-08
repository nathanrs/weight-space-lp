# weight-space-lp

**Spec and solve: compiling an LLM's weight-space diagnosis into an edit with a linear program.**

In [MIT CSAIL's weight-space programming Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math),
an LLM agent ("Astra") spent 50 passes (about 20 hours) of trial and error hand-editing Qwen3.5-0.8B's weights to stop
its thinking loops. Its final edit took MATH-500 from 7.9% to 48.1%. That edit turns out to be **rank-1**, so once you
know the direction it acts along, choosing *which neurons and how much* is a linear program.

Given Astra's direction and layers, **one LP solve (about a minute on a laptop) matches Astra's fix while changing 8×
fewer weights**. With no LLM in the loop at all, using a direction measured from real loops, the same LP recovers
about 93% of Astra's gain. A constant steering vector along the same direction, at the dose the LP picks, also
matches, so the fix is essentially "push away from `u` by the right amount". The LP finds that amount in one solve
and bakes it into ordinary weights.

![benchmarks](results/figures/benchmarks.png)

| Model | Weights changed | GSM8K test (×2) | MATH-500 (×4) | Fresh practice (512) | IFEval strict |
|---|---|---|---|---|---|
| Base Qwen3.5-0.8B | 0 | 14.1 | 7.0 ¹ | 12.1 | 22.9 |
| Astra p50c (50-pass search) | 130,188 | 68.2 | 50.9 | 53.3 | 61.9 |
| **LP `shift_k8`** (Astra's u, 8 neurons/layer) | **16,291** | **69.6** (+1.4 ± 0.9) | **51.7** (+0.8 ± 1.1) | **56.6** (+3.3 ± 1.9) | 59.7 (−2.2) |
| **Blind LP `realshift_k8`** (no LLM) | 16,271 | 65.4 (−2.8 ± 0.9) | 47.8 (−3.1 ± 1.1) | 54.9 (+1.6 ± 1.9) | 60.4 (−1.5) |
| Steering vector, `shift_k8`'s mean write (no weight edit) ² | 0 | — | 50.8 ¹ (+0.8 ± 2.0) | — | — |

Parentheses: paired difference vs. Astra ± SE. ¹ 1 sample (steering: paired vs. Astra's seed-0 sample, A100).
² see [steering](#is-it-just-a-steering-vector).
All runs use the authors' `think35` protocol (vLLM, thinking on, T = 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
32,768 new tokens; last `\boxed{}` graded with math-verify).

## Why it is a linear program

Astra's edit to `mlp.down_proj` in layers 15 and 19 is

```
W = W0 − 6 · u (mask64 ⊙ uᵀW0)      i.e.  ΔW = u vᵀ,  v_j = −6·uᵀw_j on 64 chosen neurons, 0 elsewhere
```

`u` is a unit direction in the 1024-d hidden state: the mean difference between 4 hand-written repetitive and
productive texts. At token `t` the edit adds `(v · h_t) · u` to the residual stream, where `h_t` is the 3,584-d
`down_proj` input, and nothing else changes. That is **linear in `v`**. With activations recorded from the model's
own traces, labeled as looping tokens `R` and productive tokens `P`:

```
LP-A   min_v  Σ_j |v_j|  +  C · Σ slack
       s.t.   v · h_t ≤ −δ        t ∈ R    (push loop tokens away from u by at least δ)
              |v · h_t| ≤ ε       t ∈ P    (leave productive tokens within ε)
```

The L1 objective makes `v` sparse, so the LP chooses the neurons. For exactly `k` neurons, solve once, keep the top
`k` by `|v_j|·std(h_j)`, then re-solve on that support. HiGHS took over 20 minutes per solve on these dense rows, so
[`src/wslp/lp.py`](src/wslp/lp.py) uses **PDHG** (Chambolle–Pock, the algorithm under PDLP) on MPS/CUDA. It reaches
HiGHS's optimum to within 0.2% in about a minute.

## What we did

1. **Reproduced Astra's edit** ([results/reproduction](results/reproduction/README.md)). Rebuilt it from the released
   diff (sha256 match) and confirmed it is rank-1. Re-derived `u` from Astra's 4 text pairs (cos 1.0000) and matched
   Astra's teacher-forced metric (−0.1032 vs. −0.1031). Built a vLLM eval harness that reproduces the logged MATH-500
   numbers within noise (Astra 50.0 vs. 48.1, base 7.0 vs. 7.9).
2. **Collected loop data.** A 16-token-window duplicate detector finds loop onsets (median about 680 tokens into the
   reasoning). We cached layer 15/19 activations for productive, loop-onset and in-loop tokens from MIT's released
   base traces and from 512 of our own.
3. **Swept 14 edits on a dev set** of 128 held-out practice problems ([results/dev_sweep](results/dev_sweep/README.md)),
   varying the source of `u` (Astra's or measured from real loops), the number of neurons (8–3,584), and loose vs.
   tight protection of productive tokens (ε = 1.0 "shift" vs. 0.25 "sel"). Controls: a random direction, and
   Astra's recipe run once without its search.
4. **Confirmed on untouched data** ([results/benchmarks](results/benchmarks/README.md)): 512 fresh practice problems,
   GSM8K test, MATH-500 ×4, IFEval.
5. **Analysis and baselines**: neuron overlap, blind push strength, and a steering-vector baseline
   (it matches: see below).

![dev sweep](results/figures/dev_sweep.png)

## Findings

- **Given the diagnosis, the LP matches Astra with 8× fewer weights.** `shift_k8` is within about a point of Astra on
  every benchmark, all within noise. The claim is "matches", not "beats".
- **The search was worth something, and the LP replaces it.** Astra's recipe run once without its search scores 44.5
  on dev vs. Astra's 59.4. One LP solve on the same direction scores 62.5.
- **The blind LP (no LLM) gets about 93% of the way.** It is about 3 points behind Astra on GSM8K test and MATH-500
  and ties it on IFEval. The gap is the direction, not the strength: the blind push at 2× / 4× / 8× scores
  54.7 / 54.7 / 55.5 on dev. The real-loop direction has cos ≈ 0.6 with Astra's.
- **A plain steering vector does the same job.** Adding the edit's mean write as a constant vector gets 50.8 on
  MATH-500 (Astra 50.0), so the per-token structure adds little. The LP's value is choosing the dose in one solve and
  compiling it into 8 ordinary neurons per layer.
- **The fix is bias-like.** Fewer neurons was better (8 > 32 > 128). Edits restricted to fire only on loops did no
  better than a near-constant push. At layer 15 the LP's 8 neurons are active on 89% of tokens (median neuron:
  16%), and their push has a coefficient of variation of 0.17 (Astra's: 1.1). Loops are linearly detectable once
  under way (probe AUC 0.93) but not at onset (0.73), so a selective edit fires too late to help.
- **Different neurons, same fix.** `shift_k8` shares only 3/8 (L15) and 1/8 (L19) neurons with Astra's 64. Many
  sparse solutions exist; what matters is the write along `u`.
- **No collateral damage on IFEval.** Base fails IFEval mainly by looping inside its thinking (73% cut). All edits
  fix that and land within about 1 SE of each other.
- **Teacher-forced metrics mislead.** Astra's own branch-logprob metric ranked several edits wrongly, so every
  selection here used generation.

### Is it just a steering vector?

Mostly, yes. Since the LP's best edit acts almost like a constant, we added its **mean** residual write
(`mean_t(v·h_t) · u`, norm 0.89 at L15 and 0.67 at L19) as a fixed vector to the *unedited* model at layers 15 and 19,
through a small vLLM plugin ([`src/wslp/eval/steer.py`](src/wslp/eval/steer.py)):

| Steering (unedited model) | Dev acc (128) | MATH-500 ×1 |
|---|---|---|
| 0.5 × `shift_k8` mean write | 51.6 | |
| **1 × `shift_k8` mean write** | **57.8** | **50.8** (Astra 50.0, `shift_k8` 53.2 on the same seed) |
| 2 × `shift_k8` mean write | 31.2 (loops stop, reasoning breaks) | |
| 1 × Astra's mean write | 57.0 | |
| *for reference: `shift_k8` weight edit / Astra / base* | *62.5 / 59.4 / 10.9* | |

A constant push along `u` recovers essentially the whole fix: within noise of Astra and of `shift_k8`
(−2.4 ± 1.9 vs. `shift_k8` on MATH-500). The strength matters and the window is narrow: half is too weak, double
breaks reasoning. So the honest framing is that **the diagnosis is the direction `u` and the dose**, and the
per-token variation of the edit adds little or nothing. What the LP contributes is (a) choosing the dose from data in
one solve, with no search, and (b) compiling that push into 8 neurons per layer of ordinary weights, so the fix ships
inside a standard checkpoint, with no inference-time hooks. Astra's search found the same dose by trial and error.

## Caveats

- One model (Qwen3.5-0.8B) and one behavior (math reasoning loops). Generality is untested.
- `shift_k8` was picked from 14 candidates on a 128-problem dev set, then confirmed on data never used for selection.
  The headline numbers are from the confirmation sets only.
- The harness is a reimplementation (the authors' `weightenv` package is unreleased). E0 shows it matches within
  sampling noise. IFEval uses lm-eval's port of the official checkers, scored on the text after `</think>`.
- Plan items not done: forward-pass-budget pipeline, blind layer choice, sampling-penalty and L2 baselines. See the
  status box in [docs/PLAN.md](docs/PLAN.md).

## Reproduce

```bash
uv sync                                  # Python 3.12; MPS or CUDA for activation caching and PDHG
bash scripts/fetch_exp8.sh               # MIT's Experiment 8 logs -> third_party/ (CC BY 4.0)
python scripts/e0_astra_structure.py     # rebuild Astra's edit, extract (u, S, v)  -> cache/astra_L{15,19}.npz
python scripts/export_astra_edit.py      # edits/astra_p50c.safetensors

# activations: released traces, plus our base generations on fit_gen (GPU job scripts/gpu/e0_job.sh)
python scripts/e1_collect_released.py
python scripts/e1_collect_fitgen.py

# LP sweep: eps, k list, activations, u source (astra|real), [blind push multiplier]
python scripts/e2_sparsity_sweep.py 1.0 8,32,128 acts_fitgen astra
python scripts/e2_make_edits_sweep.py acts_fitgen_astra_eps1.0 8,32,128 shift    # -> edits/shift_k{8,32,128}

# generation evals on a cloud GPU (SkyPilot; any >= 24 GB CUDA GPU)
sky launch -c wslp sky/eval.yaml --env JOB=scripts/gpu/e3_finalists.sh -i 20 --down -y
python scripts/benchmark_tables.py && python scripts/make_figures.py
```

`python -m wslp.eval.run_vllm <ckpt> <out> --sets math500 --n 4` runs one eval;
`python -m wslp.eval.ckpt <out> <edit.safetensors>` builds an edited checkpoint.

## Layout

```
src/wslp/      lp.py (LP-A, cutting planes, PDHG) · astra.py · collect.py · loops.py · edits.py · eval/ (vLLM, grading, IFEval, steering plugin)
scripts/       e0_* reproduce · e1_* activations · e2_* LP sweep · e3_overlap · benchmark_tables · make_figures · gpu/ cloud jobs
edits/         every evaluated edit (.json metadata; .safetensors are gitignored, rebuilt by the scripts)
steer/         steering vectors
results/       reproduction/ · dev_sweep/ · benchmarks/ · analysis/ · figures/ · gpu/ (per-run scores) · spend.md
docs/PLAN.md   the original plan, with a status box
```

**Compute:** about 18 H100-hours and 1.5 A100-hours on Lambda via SkyPilot, about $63 total ([ledger](results/spend.md)).
LP solves and activation caching ran on an M1 Pro.

## Credit

Experiment 8, the released edit, traces and protocol are from MIT CSAIL's weight-space programming work
([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), logs CC BY 4.0). Their logs are fetched
into `third_party/`, not redistributed here.
