# Reproducing the agent edit

## The edit itself (local)

| Check | Ours | Agent / logs |
|---|---|---|
| Edit rebuilt from the released diff, sha256 | match | `edit/manifest.json` |
| Rank-1: σ2/σ1 of ΔW (layer 15 / 19) | 5.6e-4 / 6.4e-4 | formula W0 − 6·u·(mask⊙uᵀW0) |
| ‖ΔW − u vᵀ‖/‖ΔW‖ (bf16 rounding only) | 0.23% / 0.29% | |
| Changed columns = the agent's chosen neurons | yes (64 / 64) | `frozen_weights_manifest.json` |
| Relative Frobenius change | 10.636% / 8.106% | 10.6357% / 8.1058% |
| Cosine(`u` re-derived from the 4 hand-written text pairs, the edit's `u`) | 1.0000 / 1.0000 | tool calls 9/16 |
| Token lengths of the agent's 8 test texts | identical | [124,123,131,136,107,146,98,135] |
| Agent's teacher-forced metric, repetitive continuation Δlogp | −0.1032 | −0.1031 |
| Same, productive continuation Δlogp | −0.0075 | −0.0090 (one text shortened in the agent's run) |
| Base model, log-prob of closing the thinking early | −19.76 / −19.72 | −19.77 / −19.69 |
| Cosine(`u` at layer 15, `u` at layer 19) | 0.849 | |

Local MPS throughput (backbone only, fallback linear-attention kernels): ~900–1200 tok/s up to 16k context, 5.6 GB at 16k.

## The eval harness (vLLM 0.33, H100 PCIe, authors' sampling settings, MATH-500, 1 sample)

| Model | Acc | Truncated % | Mean tokens | Logs (4 samples): acc / truncated % |
|---|---|---|---|---|
| Agent edit | 50.0 | 30.6 | 21,493 | 48.1 / 33.1 |
| Base model | 7.0 | 93.2 | 30,978 | 7.9 / 93.0 |

Within sampling noise (SE ≈ 2.2 points at n = 500). Throughput ≈ 7.2k generated tok/s on one H100 PCIe.
Lambda's driver 570 (CUDA 12.8) needed NVIDIA's cuda-compat-13-0 forward-compatibility libraries for current vLLM wheels.

## Activations from the released base-model traces

440 of 452 truncated base-model traces (OlympiadBench, Minerva, AMC23, AIME) contain a detected loop; the median loop
starts about 680 tokens in. Cached at layers 15 and 19: 86,662 productive tokens, 27,840 loop-start tokens and
18,765 tokens inside loops.

How well each signal separates looping from productive tokens (held-out AUC):

| | Layer 15 linear probe | Layer 15 base model's component along `u` | Layer 15 agent edit's push | Layer 19 probe | Layer 19 agent edit's push |
|---|---|---|---|---|---|
| Inside a loop | 0.933 | 0.738 | 0.733 | 0.918 | 0.735 |
| At loop start | 0.727 | 0.620 | 0.622 | 0.698 | 0.608 |

The agent edit pushes on every token, not just loops: its average push along `u` is −0.42 on productive tokens
(layer 15) vs. −0.94 on looping tokens. Cosine(agent's direction, measured direction) ≈ 0.60 (layer 15), 0.62
(layer 19).
