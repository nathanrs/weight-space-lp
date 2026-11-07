"""E1: cache activations at layers 15/19 from the authors' released base-model traces on non-headline benchmarks
(OlympiadBench, Minerva, AMC23, AIME24/25: never used for our MATH-500/GSM8K/IFEval evaluation)."""
import glob
import gzip
import json
import sys

import numpy as np

from wslp.collect import collect
from wslp.eval.datasets import load as load_set
from wslp.model import load
from wslp.paths import CACHE, EXP8

layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "15,19").split(",")]
traces = []
for s in ("olympiad", "minerva", "amc23", "aime24", "aime25"):
    probs = {r["id"]: r["problem"] for r in load_set(s)}
    for f in sorted(glob.glob(str(EXP8 / f"eval/unedited/parts/{s}_*/{s}_first_samples.jsonl.gz"))):
        for r in map(json.loads, gzip.open(f)):
            traces.append({"id": r["id"], "problem": probs[r["id"]], "text": r["text"],
                           "cut": "</think>" not in r["text"]})
print(len(traces), "traces;", sum(t["cut"] for t in traces), "cut")
tok, model = load()
H, RES, meta = collect(model, tok, traces, layers)
out = CACHE / "acts_released"
out.mkdir(parents=True, exist_ok=True)
for l in layers:
    np.save(out / f"h_L{l}.npy", H[l])
    np.save(out / f"resid_L{l}.npy", RES[l])
np.savez(out / "meta.npz", label=meta["label"], trace=meta["trace"], pos=meta["pos"])
json.dump({"ids": [t["id"] for t in traces], "onset": meta["onset"]}, open(out / "traces.json", "w"))
print({k: int((meta["label"] == k).sum()) for k in (0, 1, 2)}, "positions per label (P, R_onset, R_during)")
