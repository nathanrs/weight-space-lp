# E3 finalist benchmarks

## final_practice

| Model | Samples/problem | Acc | Cut % | Mean tokens |
|---|---|---|---|---|
| base | 1 | 12.1 | 89.1 | 29,717 |
| astra_p50c | 1 | 53.3 | 21.3 | 17,004 |
| shift_k8 | 1 | 56.6 | 21.7 | 15,196 |
| realshift_k8 | 1 | 54.9 | 38.1 | 18,268 |

| Paired difference | Δ pp (SE) | n |
|---|---|---|
| astra_p50c − base | +41.2 (2.2) | 512 |
| shift_k8 − base | +44.5 (2.3) | 512 |
| realshift_k8 − base | +42.8 (2.3) | 512 |
| shift_k8 − astra_p50c | +3.3 (1.9) | 512 |
| realshift_k8 − astra_p50c | +1.6 (1.9) | 512 |
| realshift_k8 − shift_k8 | -1.8 (2.0) | 512 |

## gsm8k_test

| Model | Samples/problem | Acc | Cut % | Mean tokens |
|---|---|---|---|---|
| base | 1 | 14.1 | 87.7 | 29,352 |
| astra_p50c | 2 | 68.2 | 6.6 | 11,708 |
| shift_k8 | 2 | 69.6 | 4.6 | 9,658 |
| realshift_k8 | 2 | 65.4 | 23.2 | 13,505 |

| Paired difference | Δ pp (SE) | n |
|---|---|---|
| astra_p50c − base | +54.1 (1.3) | 1319 |
| shift_k8 − base | +55.5 (1.3) | 1319 |
| realshift_k8 − base | +51.3 (1.3) | 1319 |
| shift_k8 − astra_p50c | +1.4 (0.9) | 1319 |
| realshift_k8 − astra_p50c | -2.8 (0.9) | 1319 |
| realshift_k8 − shift_k8 | -4.2 (0.9) | 1319 |

## math500

| Model | Samples/problem | Acc | Cut % | Mean tokens |
|---|---|---|---|---|
| base | 1 | 7.0 | 93.2 | 30,978 |
| astra_p50c | 4 | 50.9 | 31.1 | 21,762 |
| shift_k8 | 4 | 51.7 | 39.2 | 20,641 |
| realshift_k8 | 4 | 47.8 | 48.5 | 22,301 |

| Paired difference | Δ pp (SE) | n |
|---|---|---|
| astra_p50c − base | +44.0 (1.9) | 500 |
| shift_k8 − base | +44.7 (2.0) | 500 |
| realshift_k8 − base | +40.8 (1.9) | 500 |
| shift_k8 − astra_p50c | +0.8 (1.1) | 500 |
| realshift_k8 − astra_p50c | -3.1 (1.1) | 500 |
| realshift_k8 − shift_k8 | -3.9 (1.1) | 500 |

## IFEval (541 prompts, think35, 1 sample; lm-eval port of the official checkers; response = text after last </think>)

| Model | Prompt strict | Prompt loose | Inst strict | Inst loose | Cut % | Mean tokens |
|---|---|---|---|---|---|---|
| base | 22.9 | 23.1 | 21.0 | 21.1 | 73.4 | 24,940 |
| astra_p50c | 61.9 | 64.5 | 71.5 | 73.9 | 1.7 | 4,052 |
| shift_k8 | 59.7 | 63.6 | 67.0 | 70.3 | 5.7 | 4,349 |
| realshift_k8 | 60.4 | 63.6 | 65.3 | 67.6 | 11.8 | 7,280 |

Base fails IFEval mostly by looping inside its thinking (73% cut → empty response); every edit fixes that and lands
at ~60% strict, within ~1 SE (≈2.1 pp) of each other. No sign the edits damage instruction following.

## Blind push strength (dev_a, realshift k=8; δ = mult × loop-token excess along u)

| mult | GSM8K | MATH | Pooled | Cut % |
|---|---|---|---|---|
| 2× | 62.5 | 46.9 | 54.7 | 37.5 |
| 4× (realshift_k8) | 62.5 | 46.9 | 54.7 | 36.7 |
| 8× (target only ~25% reachable at k=8) | 64.1 | 46.9 | 55.5 | 37.5 |

Flat: the blind edit's gap to Astra's direction (and its higher cut rate) is not a push-strength problem; it comes
from the direction u itself (cos ≈ 0.6 to Astra's).
