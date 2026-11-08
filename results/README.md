# Results

| Folder | What | Key file |
|---|---|---|
| [reproduction/](reproduction/README.md) | E0: Astra's edit rebuilt, harness reproduces the logged MATH-500 numbers, loop statistics and probe AUCs | `README.md` |
| [dev_sweep/](dev_sweep/README.md) | E2: 14 edits on 128 held-out practice problems; LP prototype and sparsity-sweep diagnostics; first confirmation | `README.md`, `CONFIRMATION.md`, `sweep_*.json` |
| [benchmarks/](benchmarks/README.md) | Finalists on untouched data: GSM8K test ×2, MATH-500 ×4, 512 fresh practice, IFEval; paired differences; blind push strength; steering baseline | `README.md` (built by `scripts/benchmark_tables.py`, then hand-appended sections) |
| [analysis/](analysis/overlap.json) | E3: neuron overlap with Astra, neuron activity, push mean / variability per edit | `overlap.json` |
| [figures/](figures/) | README figures (`scripts/make_figures.py`) | `benchmarks.png`, `dev_sweep.png` |
| [gpu/](gpu/) | Raw eval outputs, one folder per run: `scores.json`, per-problem correctness, every sampled text (`*_samples.jsonl.gz`) | |
| [logs/](logs/) | Local logs from activation collection and LP sweeps | |
| [spend.md](spend.md) | Cloud cost ledger | |

Run-folder naming in `gpu/`: `dev_a_<edit>` (dev set), `final_<edit>` (512 fresh practice problems),
`gsm8k_<edit>`, `e0_<edit>_math500` + `math500s1_<edit>` (MATH-500 seeds 0 and 1), `ifeval_<edit>`,
`*_steer_*` (steering-vector baseline on the unedited model).
