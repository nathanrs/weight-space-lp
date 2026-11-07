"""E2 prototype: LP-A on one layer, constant suppression target, eps sweep; held-out-trace diagnostics."""
import json
import sys
import time

import numpy as np

from wslp import acts, lp
from wslp.paths import CACHE, RESULTS

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 15
eps_list = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1.0,0.5,0.25").split(",")]
H, RES, meta = acts.load()
lab, ho = meta["label"], meta["heldout"]
h = np.asarray(H[layer], np.float32)
A = np.load(CACHE / f"astra_L{layer}.npz")
vA, SA = A["v"], A["S"]
fitR, fitP = (lab == 2) & ~ho, (lab == 0) & ~ho
d = (h[fitR].mean(0) - h[fitP].mean(0)) / (h[fitP | fitR].std(0) + 1e-6)
cand = np.union1d(np.argsort(-np.abs(d))[:1024], SA)
delta = float(-(h[fitR] @ vA).mean())


def diag(v):
    z, zA = h[ho] @ v, h[ho] @ vA
    out = {}
    for k, n in acts.NAMES.items():
        m = lab[ho] == k
        out[n] = {"dz_mean": round(float(z[m].mean()), 3), "astra_dz_mean": round(float(zA[m].mean()), 3),
                  "abs_p90": round(float(np.percentile(np.abs(z[m]), 90)), 3)}
    mR = lab[ho] == 2
    out["R_during_frac_suppressed>=delta"] = round(float((z[mR] <= -delta).mean()), 3)
    out["astra_R_during_frac>=delta"] = round(float((zA[mR] <= -delta).mean()), 3)
    return out


report = {"layer": layer, "delta": delta, "n_candidates": len(cand), "runs": []}
for eps in eps_list:
    t = time.time()
    res = lp.solve(h[fitR], h[fitP], lp.Spec(delta_R=np.full(fitR.sum(), delta), eps=eps, support=cand),
                   n_init=(1500, 3000), add=(500, 1000), max_rounds=4)
    v = res.v
    nz = np.flatnonzero(np.abs(v) > 1e-6)
    run = {"eps": eps, "secs": round(time.time() - t), "nnz": len(nz), "l1": round(float(np.abs(v).sum()), 3),
           "overlap_with_astra64": len(np.intersect1d(nz, SA)), "rounds": res.rounds, "heldout": diag(v)}
    report["runs"].append(run)
    np.save(CACHE / f"lp_proto_L{layer}_eps{eps}.npy", v)
    print(json.dumps({k: run[k] for k in ("eps", "secs", "nnz", "l1", "overlap_with_astra64")}), flush=True)
    print("  ", json.dumps(run["heldout"]), flush=True)
(RESULTS / "e2").mkdir(parents=True, exist_ok=True)
json.dump(report, open(RESULTS / f"e2/lp_proto_L{layer}.json", "w"), indent=1)
