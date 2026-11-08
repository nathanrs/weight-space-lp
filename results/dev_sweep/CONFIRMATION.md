# Confirmation on untouched problems

`shift_k8` was chosen among ~14 candidates on dev_a. These problems were never used to fit or select anything.

## final_practice (256 GSM8K-train + 256 MATH-train, 1 sample, 32k, think35), paired vs base

| Edit | Changed weights | GSM8K | MATH | Pooled | Gain vs base (SE) | Rescues / losses | Cut % | Mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 15.6 | 8.6 | 12.1 | — | — | 89.1 | 29,717 |
| Astra p50c (50-pass search) | 130,188 | 64.5 | 42.2 | 53.3 | +41.2 (2.2) | 215 / 4 | 21.3 | 17,004 |
| **LP shift_k8** (Astra's u, 8 neurons/layer) | **16,291** | **66.4** | **46.9** | **56.6** | **+44.5 (2.3)** | 231 / 3 | 21.7 | 15,195 |

Head-to-head, paired, n = 512: **shift_k8 − Astra = +3.3 pp (SE 1.9)**; 54 problems only shift_k8 solves, 37 only Astra solves.

## MATH-500 (× 1 sample)

| Model | Changed weights | Acc | Cut % |
|---|---|---|---|
| base | 0 | 7.0 | 93.2 |
| Astra p50c | 130,188 | 50.0 | 30.6 |
| **LP shift_k8** | **16,291** | **53.2** | 37.2 |

Reading: an LP given Astra's direction and layers recovers Astra's fix with 8× fewer weights and equal-or-better
accuracy (+3.3 pp, ~1.7 SE: suggestive, not significant). Not yet run: GSM8K test, IFEval, 4-sample MATH-500.
