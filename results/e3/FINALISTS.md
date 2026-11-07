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
