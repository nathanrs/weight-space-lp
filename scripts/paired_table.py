"""Paired comparison of dev runs vs base: acc, cut, rescues/losses, paired SE, changed weights."""
import json
import sys

import numpy as np

from wslp.eval.results import G, mean_acc, totals
from wslp.paths import ROOT

prefix = sys.argv[1] if len(sys.argv) > 1 else "dev_a"
SETS = ("gsm8k_train", "math_train")


def load(run):
    return mean_acc(run, SETS)


base = load(f"{prefix}_base")
lines = ["| Edit | Changed weights | GSM8K acc | MATH acc | Pooled acc | Gain (SE) | Rescues / losses | Cut % | Mean tokens |",
         "|---|---|---|---|---|---|---|---|---|"]
for d in sorted(G.glob(f"{prefix}_*")):
    run = d.name[len(prefix) + 1:]
    r = load(d.name)
    if not r:
        continue
    ids = sorted(set(r) & set(base))
    a = np.array([base[i] for i in ids], float)
    b = np.array([r[i] for i in ids], float)
    g = [i for i in ids if i.startswith("gsm8k")]
    m = [i for i in ids if i.startswith("math")]
    diff = b - a
    info = ROOT / "edits" / f"{run}.json"
    w = sum(v["changed_weights_bf16"] for v in json.load(open(info))["layers"].values()) if info.exists() else (130188 if run == "astra_p50c" else 0)
    t = totals(d.name, SETS)
    lines.append(f"| {run} | {w:,} | {100*np.mean([r[i] for i in g]):.1f} | {100*np.mean([r[i] for i in m]):.1f} | "
                 f"{100*b.mean():.1f} | {100*diff.mean():+.1f} ({100*diff.std(ddof=1)/np.sqrt(len(diff)):.1f}) | "
                 f"{int((diff>0).sum())} / {int((diff<0).sum())} | {100*t['cut_share']:.1f} | {int(t['mean_tokens']):,} |")
print("\n".join(lines))
