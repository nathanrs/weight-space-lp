"""Read eval run folders (results/gpu/<run>) from the small tracked files only: per-problem correctness from
<set>_per_problem.json and run totals from scores.json. The raw sample texts (*_samples.jsonl.gz) are not needed."""
import json

import numpy as np

from ..paths import RESULTS

G = RESULTS / "gpu"


def per_problem(runs, sets=None) -> dict[str, list[float]]:
    """{problem id: [0/1 per sample]} pooled over runs (e.g. two MATH-500 seeds). IFEval: prompt-level strict."""
    out = {}
    for run in [runs] if isinstance(runs, str) else runs:
        for f in sorted((G / run).glob("*_per_problem.json")):
            if sets and f.name.removesuffix("_per_problem.json") not in sets:
                continue
            for r in json.load(open(f)):
                out.setdefault(r["id"], []).extend(map(float, r["correct"]))
    return out


def mean_acc(runs, sets=None) -> dict[str, float]:
    return {i: float(np.mean(v)) for i, v in per_problem(runs, sets).items()}


def totals(runs, sets=None) -> dict:
    """Sample-weighted cut share and mean generated tokens over the runs' sets."""
    n = cut = tok = 0
    for run in [runs] if isinstance(runs, str) else runs:
        for k, v in json.load(open(G / run / "scores.json")).items():
            if k.startswith("_") or (sets and k not in sets):
                continue
            m = v.get("total", v["n_problems"])
            n += m
            cut += v["cut_share"] * m
            tok += v["mean_tokens"] * m
    return {"samples": n, "cut_share": cut / n, "mean_tokens": tok / n}


FIXED = {"base": "Base model", "astra_p50c": "Agent edit", "recipe_nosearch": "Agent's recipe, no search",
         "random_ctrl": "Random direction (control)"}
HEADLINE = {"shift_k8": "LP, agent's direction", "realshift_k8": "LP, measured direction"}


def label(edit: str, short=True) -> str:
    """Readable name for an edit id (file names keep the ids). Ids: [real](shift|sel)_k<N> or lp_(shift|sel):
    real = direction measured from the model's own loops (else the agent's), shift = broad push (loose bound on
    productive tokens), sel = selective (tight bound), k = neurons per layer."""
    if edit in FIXED:
        return FIXED[edit]
    if short and edit in HEADLINE:
        return HEADLINE[edit]
    if edit.startswith("steer_"):  # steer_<edit>_x<scale>: constant vector = scale × that edit's mean write
        src, _, scale = edit.removeprefix("steer_").rpartition("_x")
        src = "astra_p50c" if src == "astra" else src
        return f"Steering vector, {scale}× mean write of: {label(src)}"
    real = edit.startswith("real")
    rest = edit.removeprefix("real").removeprefix("lp_")
    kind, _, k = rest.partition("_k")
    kind, _, push = kind.partition("_d")  # _d<m>: measured direction pushed m× the loop excess (default 4×)
    return (f"LP, {"measured" if real else "agent's"} direction, "
            f"{'broad' if kind == 'shift' else 'selective'}, {k + ' neurons/layer' if k else 'no neuron limit'}"
            + (f", {push}× push" if push else ""))
