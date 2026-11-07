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

Not yet done: the generation-side check (MATH-500 cut share 93% → 33%), which needs vLLM on GPU.
