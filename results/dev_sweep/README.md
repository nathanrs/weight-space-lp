# Dev-set sweep

Dev set: 128 held-out practice problems (64 GSM8K-train, 64 MATH-train), 1 sample each, the authors' sampling settings
with a 32,768-token limit. Every edit is compared to the base model on the same problems. The LP edit used in the
headline ("LP, agent's direction, broad, 8 neurons/layer") was picked here from 14 candidates, so differences on this
set are only suggestive; the confirmation is in [benchmarks](../benchmarks/README.md). Terms (broad, selective,
measured direction, push) are defined in the [write-up](../../docs/WRITEUP.md#what-we-did); file IDs in [results/README.md](../README.md). Rescues / losses = problems
the edit solves that the base model doesn't, and vice versa.

Table built by `python scripts/paired_table.py`.

| Edit | ID | Changed weights | GSM8K acc | MATH acc | Pooled acc | Gain vs base (SE) | Rescues / losses | Truncated % | Mean tokens |
|---|---|---|---|---|---|---|---|---|---|
| Base model | `base` | 0 | 10.9 | 10.9 | 10.9 | +0.0 (0.0) | 0 / 0 | 89.1 | 29,868 |
| Agent edit | `astra_p50c` | 130,188 | 70.3 | 48.4 | 59.4 | +48.4 (4.7) | 64 / 2 | 12.5 | 16,502 |
| Agent's recipe on the measured direction, no search | `recipe_nosearch` | 130,152 | 53.1 | 35.9 | 44.5 | +33.6 (4.2) | 43 / 0 | 48.4 | 20,719 |
| Random direction (control) | `random_ctrl` | 130,285 | 9.4 | 10.9 | 10.2 | -0.8 (2.1) | 3 / 4 | 92.2 | 30,661 |
| LP, agent's direction, broad, 8 neurons/layer | `shift_k8` | 16,291 | 71.9 | 53.1 | 62.5 | +51.6 (4.6) | 67 / 1 | 21.9 | 14,501 |
| LP, agent's direction, broad, 32 neurons/layer | `shift_k32` | 64,927 | 70.3 | 46.9 | 58.6 | +47.7 (4.6) | 62 / 1 | 25.0 | 14,278 |
| LP, agent's direction, broad, 128 neurons/layer | `shift_k128` | 256,623 | 60.9 | 46.9 | 53.9 | +43.0 (4.7) | 57 / 2 | 25.0 | 14,716 |
| LP, agent's direction, broad, no neuron limit | `lp_shift` | 772,252 | 67.2 | 46.9 | 57.0 | +46.1 (4.7) | 61 / 2 | 24.2 | 14,214 |
| LP, agent's direction, selective, 64 neurons/layer | `sel_k64` | 129,458 | 68.8 | 48.4 | 58.6 | +47.7 (4.6) | 62 / 1 | 21.1 | 15,475 |
| LP, agent's direction, selective, 512 neurons/layer | `sel_k512` | 1,029,493 | 68.8 | 48.4 | 58.6 | +47.7 (4.6) | 62 / 1 | 22.7 | 16,253 |
| LP, agent's direction, selective, no neuron limit | `lp_sel` | 1,284,815 | 71.9 | 54.7 | 63.3 | +52.3 (4.4) | 67 / 0 | 21.9 | 16,774 |
| LP, measured direction, broad, 8 neurons/layer | `realshift_k8` | 16,271 | 62.5 | 46.9 | 54.7 | +43.8 (4.5) | 57 / 1 | 36.7 | 17,425 |
| LP, measured direction, broad, 8 neurons/layer, 2× push | `realshift_d2_k8` | 16,056 | 62.5 | 46.9 | 54.7 | +43.8 (4.5) | 57 / 1 | 37.5 | 17,491 |
| LP, measured direction, broad, 8 neurons/layer, 8× push | `realshift_d8_k8` | 16,202 | 64.1 | 46.9 | 55.5 | +44.5 (4.5) | 58 / 1 | 37.5 | 16,815 |
| LP, measured direction, broad, 64 neurons/layer | `realshift_k64` | 123,121 | 65.6 | 46.9 | 56.2 | +45.3 (4.6) | 59 / 1 | 35.9 | 17,079 |
| LP, measured direction, selective, 64 neurons/layer | `realsel_k64` | 128,710 | 57.8 | 45.3 | 51.6 | +40.6 (4.5) | 53 / 1 | 45.3 | 20,042 |
| LP, measured direction, selective, 512 neurons/layer | `realsel_k512` | 1,025,282 | 56.2 | 42.2 | 49.2 | +38.3 (4.3) | 49 / 0 | 49.2 | 20,971 |
| Steering vector, 1× mean write of: Agent edit | `steer_astra_x1` | 0 | 70.3 | 43.8 | 57.0 | +46.1 (4.8) | 62 / 3 | 33.6 | 16,590 |
| Steering vector, 0.5× mean write of: LP, agent's direction | `steer_shift_k8_x0.5` | 0 | 65.6 | 37.5 | 51.6 | +40.6 (4.5) | 53 / 1 | 42.2 | 18,387 |
| Steering vector, 1× mean write of: LP, agent's direction | `steer_shift_k8_x1` | 0 | 65.6 | 50.0 | 57.8 | +46.9 (4.8) | 63 / 3 | 23.4 | 14,853 |
| Steering vector, 2× mean write of: LP, agent's direction | `steer_shift_k8_x2` | 0 | 31.2 | 31.2 | 31.2 | +20.3 (4.5) | 32 / 6 | 10.2 | 10,314 |

Solver diagnostics per sweep (`sweep_*.json`): file names give the activation set, direction (`astra` = agent's,
`real` = measured), ε (1.0 = broad, 0.25 = selective) and push multiplier. `lp_proto_L{15,19}.json` are the first
LP prototype runs.
