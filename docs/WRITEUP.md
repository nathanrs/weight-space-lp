# Write-up

## Background

In [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., MIT CSAIL;
[logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)), an LLM agent ("Astra") spent 50 passes (about 20 hours) hand-editing Qwen3.5-0.8B's weights to stop its thinking
loops. Its final edit (`p50c`) took MATH-500 from 7.9% to 48.1% in the authors' logs. That edit turns out to be
**rank-1**, so once you know the direction it acts along, choosing *which neurons and how much* is a linear program.
This project tests whether one LP solve can stand in for the search.

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
[`src/wslp/lp.py`](../src/wslp/lp.py) uses **PDHG** (Chambolle–Pock, the algorithm under PDLP) on MPS/CUDA. It reaches
HiGHS's optimum to within 0.2% in about a minute.

## What we did

1. **Reproduced Astra's edit** ([results/reproduction](../results/reproduction/README.md)). Rebuilt it from the
   released diff (sha256 match) and confirmed it is rank-1. Re-derived `u` from Astra's 4 text pairs (cos 1.0000) and
   matched Astra's teacher-forced metric (−0.1032 vs. −0.1031). Built a vLLM eval harness that reproduces the logged
   MATH-500 numbers within noise (Astra 50.0 vs. 48.1, base 7.0 vs. 7.9).
2. **Collected loop data.** A 16-token-window duplicate detector finds loop onsets (median about 680 tokens into the
   reasoning). We cached layer 15/19 activations for productive, loop-onset and in-loop tokens from the paper's released
   base traces and from 512 of our own.
3. **Swept 14 edits on a dev set** of 128 held-out practice problems ([results/dev_sweep](../results/dev_sweep/README.md)),
   varying the source of `u` (Astra's or measured from real loops), the number of neurons (8–3,584), and loose vs.
   tight protection of productive tokens (ε = 1.0 "shift" vs. 0.25 "sel"). Controls: a random direction, and
   Astra's recipe run once without its search.
4. **Confirmed on untouched data** ([results/benchmarks](../results/benchmarks/README.md)): 512 fresh practice
   problems, GSM8K test, MATH-500 ×4, IFEval.
5. **Analysis and baselines**: neuron overlap, blind push strength, and a steering-vector baseline.

![dev sweep](../results/figures/dev_sweep.png)

![benchmarks](../results/figures/benchmarks.png)

## Findings

- **Given the diagnosis, the LP matches Astra with 8× fewer weights.** `shift_k8` is within about a point of Astra on
  every benchmark, all within noise.
- **The search was worth something, and the LP replaces it.** Astra's recipe run once without its search scores 44.5
  on dev vs. Astra's 59.4. One LP solve on the same direction scores 62.5.
- **The blind LP (no LLM) gets about 93% of the way.** It is about 3 points behind Astra on GSM8K test and MATH-500
  and ties it on IFEval. The gap is the direction, not the strength: the blind push at 2× / 4× / 8× scores
  54.7 / 54.7 / 55.5 on dev. The real-loop direction has cos ≈ 0.6 with Astra's.
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

## A steering vector does the same job

Since the LP's best edit acts almost like a constant, we added its **mean** residual write
(`mean_t(v·h_t) · u`, norm 0.89 at L15 and 0.67 at L19) as a fixed vector to the *unedited* model at layers 15 and 19,
through a small vLLM plugin ([`src/wslp/eval/steer.py`](../src/wslp/eval/steer.py)):

| Steering (unedited model) | Dev acc (128) | MATH-500 ×1 |
|---|---|---|
| 0.5 × `shift_k8` mean write | 51.6 | |
| **1 × `shift_k8` mean write** | **57.8** | **50.8** (Astra 50.0, `shift_k8` 53.2 on the same seed) |
| 2 × `shift_k8` mean write | 31.2 (loops stop, reasoning breaks) | |
| 1 × Astra's mean write | 57.0 | |
| *for reference: `shift_k8` weight edit / Astra / base* | *62.5 / 59.4 / 10.9* | |

A constant push along `u` recovers essentially the whole fix: within noise of Astra and of `shift_k8`
(−2.4 ± 1.9 vs. `shift_k8` on MATH-500). The strength matters and the window is narrow: half is too weak, double
breaks reasoning. So the diagnosis is **a direction `u` and a dose**, and the per-token variation of the edit adds
little. What the LP contributes is (a) choosing the dose from data in one solve, with no search, and (b) compiling
that push into 8 neurons per layer of ordinary weights, so the fix ships inside a standard checkpoint with no
inference-time hooks. Astra's search found the same dose by trial and error.

## Evaluation protocol

All runs use the authors' `think35` protocol: vLLM, thinking on, T = 0.6, top-p 0.95, top-k 20, presence penalty 1.5,
32,768 new tokens; the last `\boxed{}` is graded with math-verify. IFEval uses lm-eval's port of the official
checkers, scored on the text after the last `</think>` (empty if the thinking was cut off). Differences are paired
per problem against Astra.

## Caveats

- One model (Qwen3.5-0.8B) and one behavior (math reasoning loops). Generality is untested.
- `shift_k8` was picked from 14 candidates on a 128-problem dev set, then confirmed on data never used for selection.
  The headline numbers are from the confirmation sets only.
- The harness is a reimplementation (the authors' `weightenv` package is unreleased); it matches the logged numbers
  within sampling noise.
- Not done: forward-pass-budget pipeline, blind layer choice, sampling-penalty and L2 baselines. See the status box
  in [PLAN.md](PLAN.md).
