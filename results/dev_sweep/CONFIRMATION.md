# First confirmation on untouched problems

The LP edit (agent's direction, broad, 8 neurons/layer; `shift_k8`) was picked from about 14 candidates on the dev
set. These problems were never used to fit or pick anything. The fuller confirmation, with more samples and the
measured-direction edit, is in [benchmarks](../benchmarks/README.md).

## Fresh practice (256 GSM8K-train + 256 MATH-train, 1 sample), compared to the base model

| Edit | Changed weights | GSM8K | MATH | Pooled | Gain vs base (SE) | Rescues / losses | Truncated % | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| Base model | 0 | 15.6 | 8.6 | 12.1 | — | — | 89.1 | 29,717 |
| Agent edit (50-pass search) | 130,188 | 64.5 | 42.2 | 53.3 | +41.2 (2.2) | 215 / 4 | 21.3 | 17,004 |
| **LP, agent's direction** (8 neurons/layer) | **16,291** | **66.4** | **46.9** | **56.6** | **+44.5 (2.3)** | 231 / 3 | 21.7 | 15,195 |

Head-to-head, paired, n = 512: **LP edit − agent edit = +3.3 points (SE 1.9)**; 54 problems only the LP edit solves,
37 only the agent edit solves.

## MATH-500 (1 sample)

| Model | Changed weights | Acc | Truncated % |
|---|---|---|---|
| Base model | 0 | 7.0 | 93.2 |
| Agent edit | 130,188 | 50.0 | 30.6 |
| **LP, agent's direction** | **16,291** | **53.2** | 37.2 |

Reading: given the agent's direction and layers, the LP recovers the agent's fix with 8× fewer weights and equal or
better accuracy (+3.3 points, about 1.7 SE: suggestive, not significant).
