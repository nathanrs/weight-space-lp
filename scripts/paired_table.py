"""Paired comparison of dev runs vs base: acc, cut, rescues/losses, paired SE, changed weights."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from wslp.paths import ROOT

prefix = sys.argv[1] if len(sys.argv) > 1 else "dev_a"
G = ROOT / "results/gpu"


def load(run):
    rows = {}
    for s in ("gsm8k_train", "math_train"):
        f = G / run / f"{s}_samples.jsonl.gz"
        if f.exists():
            for r in map(json.loads, gzip.open(f)):
                rows[r["id"]] = r
    return rows


base = load(f"{prefix}_base")
lines = ["| Edit | Changed weights | GSM8K acc | MATH acc | Pooled acc | Gain (SE) | Rescues / losses | Cut % | Mean tokens |",
         "|---|---|---|---|---|---|---|---|---|"]
for d in sorted(G.glob(f"{prefix}_*")):
    run = d.name[len(prefix) + 1:]
    r = load(d.name)
    if not r:
        continue
    ids = sorted(set(r) & set(base))
    a = np.array([base[i]["correct"] for i in ids], float)
    b = np.array([r[i]["correct"] for i in ids], float)
    g = [i for i in ids if i.startswith("gsm8k")]
    m = [i for i in ids if i.startswith("math")]
    diff = b - a
    info = ROOT / "edits" / f"{run}.json"
    w = sum(v["changed_weights_bf16"] for v in json.load(open(info))["layers"].values()) if info.exists() else (130188 if run == "astra_p50c" else 0)
    lines.append(f"| {run} | {w:,} | {100*np.mean([r[i]['correct'] for i in g]):.1f} | {100*np.mean([r[i]['correct'] for i in m]):.1f} | "
                 f"{100*b.mean():.1f} | {100*diff.mean():+.1f} ({100*diff.std(ddof=1)/np.sqrt(len(diff)):.1f}) | "
                 f"{int((diff>0).sum())} / {int((diff<0).sum())} | {100*np.mean([r[i]['cut'] for i in ids]):.1f} | "
                 f"{int(np.mean([r[i]['tokens'] for i in ids])):,} |")
print("\n".join(lines))
