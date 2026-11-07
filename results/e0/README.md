# E0 (local part): reproduction of Astra's p50c edit

| Check | Ours | Astra / logs |
|---|---|---|
| Edit rebuilt from released diff, sha256 | match | `edit/manifest.json` |
| Rank-1: σ2/σ1 of ΔW (L15 / L19) | 5.6e-4 / 6.4e-4 | formula W0 − 6·u·(mask⊙uᵀW0) |
| ‖ΔW − u v_Astraᵀ‖/‖ΔW‖ (bf16 rounding only) | 0.23% / 0.29% | |
| Changed columns = manifest neurons | yes (64 / 64) | `frozen_weights_manifest.json` |
| Relative Frobenius change | 10.636% / 8.106% | 10.6357% / 8.1058% |
| cos(u re-derived from 4 hand-written pairs, edit's u) | 1.0000 / 1.0000 | tool calls 9/16 |
| fwd_4 token lengths | identical | [124,123,131,136,107,146,98,135] |
| TF transfer probe, repetitive branch Δlogp | −0.1032 | −0.1031 |
| TF transfer probe, productive branch Δlogp | −0.0075 | −0.0090 (case 7 shortened in Astra's run) |
| Base premature-close logp | −19.76 / −19.72 | −19.77 / −19.69 |
| cos(u15, u19) | 0.849 | |

Local MPS throughput (backbone only, fallback linear-attention kernels): ~900–1200 tok/s up to 16k context, 5.6 GB at 16k.

## Generation-side harness check (vLLM 0.33, H100 PCIe, think35 protocol, MATH-500 × 1 sample)

| Model | Acc | Cut share | Mean tokens | Logs (× 4 samples) |
|---|---|---|---|---|
| Astra p50c | 50.0 | 30.6 | 21,493 | 48.1 / 33.1 |
| Base | (running) | | | 7.9 / 93.0 |

Within sampling noise (SE ≈ 2.2 pp at n = 500). Throughput ≈ 7.2k generated tok/s on one H100 PCIe.
Lambda driver 570 (CUDA 12.8) needed NVIDIA's cuda-compat-13-0 forward-compat libs for current vLLM wheels.

## Activations from released traces (E1)

440 / 452 cut base-model traces (OlympiadBench, Minerva, AMC23, AIME) have a detected loop; onset median ~680 tokens.
Cached L15/L19: 86,662 P, 27,840 R_onset, 18,765 R_during positions.

| Held-out AUC (R vs P) | L15 linear probe on h | L15 base u-write | L15 Astra −v·h | L19 probe | L19 Astra |
|---|---|---|---|---|---|
| R_during | 0.933 | 0.738 | 0.733 | 0.918 | 0.735 |
| R_onset | 0.727 | 0.620 | 0.622 | 0.698 | 0.608 |

Astra's edit is unconditional: mean shift of the u-write on P tokens is −0.42 (L15) vs −0.94 on loop tokens.
cos(Astra's synthetic u, real-loop mean-diff direction) ≈ 0.60 (L15), 0.62 (L19).
