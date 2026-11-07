"""Check PDHG against HiGHS (simplex = exact vertex) on a small LP-A instance."""
import time
import numpy as np
from wslp import acts, lp
from wslp.paths import CACHE
H, RES, meta = acts.load()
lab, ho = meta["label"], meta["heldout"]
h = np.asarray(H[15], np.float32); vA = np.load(CACHE / "astra_L15.npz")["v"]
rng = np.random.default_rng(0)
iR = rng.choice(np.flatnonzero((lab == 2) & ~ho), 500, replace=False)
iP = rng.choice(np.flatnonzero((lab == 0) & ~ho), 1000, replace=False)
d = (h[lab == 2].mean(0) - h[lab == 0].mean(0)) / (h.std(0) + 1e-6)
cols = np.sort(np.argsort(-np.abs(d))[:512])
delta = float(-(h[lab == 2] @ vA).mean())
spec = lp.Spec(delta_R=np.full(500, delta), eps=0.5, C=10.0, support=cols)
obj = lambda v: np.abs(v).sum() + spec.C * (np.maximum(0, h[iR] @ v + delta).sum() + np.maximum(0, np.abs(h[iP] @ v) - 0.5).sum())
t = time.time(); vh, _, msg = lp._solve(h[iR], spec.delta_R, h[iP], spec, cols); th = time.time() - t
print(f"HiGHS ipm {th:.1f}s obj={obj(vh):.4f} nnz(>1e-4)={(np.abs(vh)>1e-4).sum()}", flush=True)
for om in (0.01, 0.1, 1.0, 10.0, 100.0):
    t = time.time(); r = lp.pdhg(h[iR], h[iP], spec, iters=20000, omega=om); tp = time.time() - t
    print(f"PDHG omega={om:<6} {tp:.1f}s obj={obj(r.v):.4f} nnz={(r.v!=0).sum()}  {r.status}", flush=True)

