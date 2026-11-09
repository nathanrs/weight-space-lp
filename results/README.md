# Results

Tables use readable names; file and folder names use short IDs:

| ID | Meaning |
|---|---|
| `base` | Base model |
| `astra_p50c` | Agent edit (the agent's run `p50c`, with 50 forward passes) |
| `shift_k<N>` / `sel_k<N>` | LP with the agent's direction, broad (ε = 1.0) / selective (ε = 0.25), N neurons per layer |
| `lp_shift` / `lp_sel` | Same, with no neuron limit |
| `real…` | Same LPs with the heuristic direction (from automatically detected loops); `_d2` / `_d8` = push 2× / 8× the loop excess (default 4×) |
| `recipe_nosearch`, `random_ctrl` | Agent's recipe (column scoring, 6× rule) applied once to the heuristic direction, no search; random direction (control) |
| `steer_<edit>_x<s>` | Unedited model plus s × that edit's average push as a constant vector |
| `dev_a`, `final_practice` | Dev set (128 problems), fresh practice (512 problems) |

The headline edits are `shift_k8` ("LP, agent's direction") and `realshift_k8` ("LP, heuristic direction").

| Folder | What | Key file |
|---|---|---|
| [reproduction/](reproduction/README.md) | The agent edit rebuilt and checked; the eval harness reproduces the logged MATH-500 numbers; loop statistics and probe AUCs | `README.md` |
| [dev_sweep/](dev_sweep/README.md) | All edits on the 128-problem dev set, LP solver diagnostics, and the first confirmation | `README.md`, `CONFIRMATION.md`, `sweep_*.json` |
| [benchmarks/](benchmarks/README.md) | The two LP edits vs. the agent edit on untouched data (GSM8K test, MATH-500, fresh practice, IFEval), push strength, steering-vector baseline | `README.md` (top part built by `scripts/benchmark_tables.py`) |
| [analysis/](analysis/overlap.json) | Neuron overlap with the agent edit, neuron activity, average push and its variability per edit | `overlap.json` |
| [figures/](figures/) | Write-up figures (`scripts/make_figures.py`) | `benchmarks.png`, `dev_sweep.png` |
| [gpu/](gpu/) | Eval outputs, one folder per run: `scores.json` and `*_per_problem.json` (correctness per sample), which drive every table and figure. The generated texts (`*_samples.jsonl.gz`, ~160 MB) are gitignored, except the base-model traces the LP is fit on (`e1_base_fitgen/`) | |
| [spend.md](spend.md) | Cloud cost ledger | |

Run folders in `gpu/` are named `<benchmark>_<edit id>`: `dev_a_*` (dev set), `final_*` (fresh practice),
`gsm8k_*`, `e0_*_math500` and `math500s1_*` (MATH-500, seeds 0 and 1), `ifeval_*`, and `*_steer_*` (steering vector
on the unedited model). In `scores.json`, `cut_share` is the share of generations truncated at the token limit.
