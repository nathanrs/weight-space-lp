"""E2 prototype: LP-A via PDHG on one layer; grid over L1 weight (sparsity) and productive band eps.
Constant suppression target delta = Astra's mean suppression on fit R_during tokens. Held-out-trace diagnostics."""
import json
import sys
import time

import numpy as np

from wslp import acts, lp
from wslp.paths import CACHE, RESULTS

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 15
lams = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,10,100").split(",")]
epss = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.0,0.25").split(",")]
H, RES, meta = acts.load()
lab, ho = meta["label"], meta["heldout"]
h = np.asarray(H[layer], np.float32)
A = np.load(CACHE / f"astra_L{layer}.npz")
vA, SA = A["v"], A["S"]
rng = np.random.default_rng(0)
iR = rng.choice(np.flatnonzero((lab == 2) & ~ho), 4000, replace=False)
iP = rng.choice(np.flatnonzero((lab == 0) & ~ho), 8000, replace=False)
delta = float(-(h[(lab == 2) & ~ho] @ vA).mean())
zA_ho = h[ho] @ vA


def diag(v):
    z = h[ho] @ v
    out = {}
    for k, n in acts.NAMES.items():
        m = lab[ho] == k
        out[n] = [round(float(z[m].mean()), 3), round(float(np.percentile(np.abs(z[m]), 90)), 3)]
    mR = lab[ho] == 2
    out["R_during_hit"] = round(float((z[mR] <= -delta).mean()), 3)
    return out


astra = diag(vA)
print("delta", round(delta, 3), "| Astra held-out [mean dz, |dz| p90]:", json.dumps(astra), flush=True)
runs = []
for eps in epss:
    for lam in lams:
        t = time.time()
        spec = lp.Spec(delta_R=np.full(len(iR), delta), eps=eps, C=10.0, cost=np.full(h.shape[1], lam))
        r = lp.pdhg(h[iR], h[iP], spec, iters=20000, omega=100.0)
        nz = np.flatnonzero(r.v)
        run = {"eps": eps, "lam": lam, "secs": round(time.time() - t), "nnz": len(nz),
               "overlap_astra64": len(np.intersect1d(nz, SA)), "heldout": diag(r.v)}
        runs.append(run)
        np.save(CACHE / f"lp_proto_L{layer}_eps{eps}_lam{lam}.npy", r.v)
        print(json.dumps(run), flush=True)
(RESULTS / "dev_sweep").mkdir(parents=True, exist_ok=True)
json.dump({"layer": layer, "delta": delta, "astra": astra, "runs": runs}, open(RESULTS / f"dev_sweep/lp_proto_L{layer}.json", "w"), indent=1)
