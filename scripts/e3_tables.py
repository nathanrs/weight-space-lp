"""E3 finalist tables: per-benchmark accuracy and paired differences (per-problem mean over samples, paired SE)."""
import gzip
import json
from collections import defaultdict

import numpy as np

from wslp.paths import ROOT

G = ROOT / "results/gpu"
RUNS = {  # benchmark -> model -> run dirs whose samples are pooled
    "final_practice": {m: [f"final_{m}"] for m in ("base", "astra_p50c", "shift_k8", "realshift_k8")},
    "gsm8k_test": {m: [f"gsm8k_{m}"] for m in ("base", "astra_p50c", "shift_k8", "realshift_k8")},
    "math500": {"base": ["e0_base_math500"], "astra_p50c": ["e0_astra_math500", "math500s1_astra_p50c"],
                "shift_k8": ["e0_shift_k8_math500", "math500s1_shift_k8"], "realshift_k8": ["math500_realshift_k8"]},
}


def load(dirs):
    c, cut, tok = defaultdict(list), [], []
    for d in dirs:
        for f in (G / d).glob("*_samples.jsonl.gz"):
            for r in map(json.loads, gzip.open(f)):
                c[r["id"]].append(r["correct"])
                cut.append(r["cut"])
                tok.append(r["tokens"])
    return {i: np.mean(v) for i, v in c.items()}, np.mean(cut), np.mean(tok), {i: len(v) for i, v in c.items()}


out = ["# E3 finalist benchmarks", ""]
for bench, models in RUNS.items():
    data = {m: load(d) for m, d in models.items() if all((G / x).exists() for x in d)}
    out += [f"## {bench}", "", "| Model | Samples/problem | Acc | Cut % | Mean tokens |", "|---|---|---|---|---|"]
    for m, (acc, cut, tok, n) in data.items():
        out.append(f"| {m} | {np.median(list(n.values())):g} | {100 * np.mean(list(acc.values())):.1f} | "
                   f"{100 * cut:.1f} | {tok:,.0f} |")
    out += ["", "| Paired difference | Δ pp (SE) | n |", "|---|---|---|"]
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = sorted(set(data[a][0]) & set(data[b][0]))
            d = np.array([data[b][0][k] - data[a][0][k] for k in ids])
            out.append(f"| {b} − {a} | {100 * d.mean():+.1f} ({100 * d.std(ddof=1) / np.sqrt(len(d)):.1f}) | {len(d)} |")
    out.append("")
txt = "\n".join(out)
print(txt)
(ROOT / "results/e3").mkdir(exist_ok=True)
open(ROOT / "results/e3/FINALISTS.md", "w").write(txt)
