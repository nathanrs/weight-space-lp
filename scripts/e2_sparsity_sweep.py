"""E2 sparsity sweep: exactly-k-neuron LP edits (L1 path -> top-k by |v_j|*std(h_j) -> re-solve on that support).

  python scripts/e2_sparsity_sweep.py <eps> <k1,k2,...> [acts_name] [u_source: astra|real] [delta_mult]
Writes cache/sweep_<tag>_L{l}_k{k}.npy and results/e2/sweep_<tag>.json with held-out diagnostics.
"""
import json
import sys

import numpy as np

from wslp import acts, lp
from wslp.astra import base_tensor
from wslp.paths import CACHE, RESULTS, down_proj

eps = float(sys.argv[1])
ks = [int(x) for x in sys.argv[2].split(",")]
name = sys.argv[3] if len(sys.argv) > 3 else "acts_released"
usrc = sys.argv[4] if len(sys.argv) > 4 else "astra"
mult = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0   # real-u only: push loop tokens mult x their excess
tag = f"{name}_{usrc}_eps{eps}" + (f"_d{mult:g}" if mult != 4.0 else "")
H, RES, meta = acts.load(name)
lab, ho = meta["label"], meta["heldout"]
rng = np.random.default_rng(0)
report = {"eps": eps, "acts": name, "u": usrc, "layers": {}}
for l in (15, 19):
    h = np.asarray(H[l], np.float32)
    A = np.load(CACHE / f"astra_L{l}.npz")
    fitR, fitP = (lab == 2) & ~ho, (lab == 0) & ~ho
    if usrc == "astra":
        u, vref = A["u"], A["v"]
        delta = float(-(h[fitR] @ vref).mean())            # Astra's mean suppression on loop tokens
    else:
        R = np.asarray(RES[l], np.float32)
        u = R[fitR].mean(0) - R[fitP].mean(0); u /= np.linalg.norm(u)
        wu = u @ base_tensor(down_proj(l))
        s = h @ wu
        delta = float(mult * (s[fitR].mean() - s[fitP].mean()))  # push loop tokens mult x their excess below normal
    iR = rng.choice(np.flatnonzero(fitR), min(4000, fitR.sum()), replace=False)
    iP = rng.choice(np.flatnonzero(fitP), min(8000, fitP.sum()), replace=False)
    base_spec = lp.Spec(delta_R=np.full(len(iR), delta), eps=eps, C=10.0)
    r_full = lp.pdhg(h[iR], h[iP], lp.Spec(**{**base_spec.__dict__, "cost": np.full(h.shape[1], 1.0)}), omega=100.0)
    scale = h[fitP | fitR].std(0)
    rows = []
    for k in ks:
        sup = np.sort(np.argsort(-np.abs(r_full.v) * scale)[:k])
        r = lp.pdhg(h[iR], h[iP], lp.Spec(**{**base_spec.__dict__, "support": sup, "cost": np.full(h.shape[1], 1e-3)}),
                    omega=100.0)
        z = h[ho] @ r.v
        row = {"k": k, "overlap_astra64": len(np.intersect1d(np.flatnonzero(r.v), A["S"]))}
        for kk, n in acts.NAMES.items():
            m = lab[ho] == kk
            row[n] = round(float(z[m].mean()), 3)
        row["P_abs_p90"] = round(float(np.percentile(np.abs(z[lab[ho] == 0]), 90)), 3)
        row["R_during_hit"] = round(float((z[lab[ho] == 2] <= -delta).mean()), 3)
        rows.append(row)
        np.save(CACHE / f"sweep_{tag}_L{l}_k{k}.npy", np.stack([np.pad(u, (0, len(r.v) - len(u))), r.v]))
        print(f"L{l}", json.dumps(row), flush=True)
    report["layers"][l] = {"delta": delta, "rows": rows}
(RESULTS / "e2").mkdir(parents=True, exist_ok=True)
json.dump(report, open(RESULTS / f"e2/sweep_{tag}.json", "w"), indent=1)
