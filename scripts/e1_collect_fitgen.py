"""E1: cache L15/L19 activations from our base-model generations on the 512 fit_gen practice problems."""
import gzip
import json

import numpy as np

from wslp.collect import collect
from wslp.eval.datasets import load as load_set
from wslp.model import load
from wslp.paths import CACHE, RESULTS

probs = {r["id"]: r["problem"] for s in ("gsm8k_train", "math_train") for r in load_set(s)}
traces = []
for s in ("gsm8k_train", "math_train"):
    for r in map(json.loads, gzip.open(RESULTS / f"gpu/e1_base_fitgen/{s}_samples.jsonl.gz")):
        traces.append({"id": r["id"], "problem": probs[r["id"]], "text": r["text"], "cut": r["cut"], "correct": r["correct"]})
print(len(traces), "traces;", sum(t["cut"] for t in traces), "cut;", sum(t["correct"] for t in traces), "correct")
tok, model = load()
H, RES, meta = collect(model, tok, traces, [15, 19])
out = CACHE / "acts_fitgen"
out.mkdir(parents=True, exist_ok=True)
for l in (15, 19):
    np.save(out / f"h_L{l}.npy", H[l])
    np.save(out / f"resid_L{l}.npy", RES[l])
np.savez(out / "meta.npz", label=meta["label"], trace=meta["trace"], pos=meta["pos"])
json.dump({"ids": [t["id"] for t in traces], "onset": meta["onset"]}, open(out / "traces.json", "w"))
print({k: int((meta["label"] == k).sum()) for k in (0, 1, 2)}, "positions per label (P, R_onset, R_during)")
