"""How Astra's edit acts on real loop / productive tokens (from released traces), per layer."""
import json

import numpy as np

from wslp import acts
from wslp.astra import base_tensor
from wslp.paths import CACHE, down_proj

H, RES, meta = acts.load()
lab = meta["label"]
out = {}
for l in (15, 19):
    A = np.load(CACHE / f"astra_L{l}.npz")
    u, vA = A["u"], A["v"]
    wu = u @ base_tensor(down_proj(l))          # s_t = wu . h_t (the MLP's write onto u)
    h = np.asarray(H[l], np.float32)
    s, zA = h @ wu, h @ vA
    r = np.asarray(RES[l], np.float32) @ u      # residual stream projection on u after the layer
    row = {}
    for k, n in acts.NAMES.items():
        m = lab == k
        row[n] = {"n": int(m.sum()), "s_mean": float(s[m].mean()), "resid_u_mean": float(r[m].mean()),
                  "astra_dz_mean": float(zA[m].mean()), "astra_dz_p10": float(np.percentile(zA[m], 10)),
                  "astra_|dz|_p90": float(np.percentile(np.abs(zA[m]), 90))}
    # real-loop direction vs Astra's synthetic one
    R = np.asarray(RES[l], np.float32)
    d = R[lab == 2].mean(0) - R[lab == 0].mean(0)
    d_on = R[lab == 1].mean(0) - R[lab == 0].mean(0)
    row["cos(u_astra, u_real_during)"] = float(u @ d / np.linalg.norm(d))
    row["cos(u_astra, u_real_onset)"] = float(u @ d_on / np.linalg.norm(d_on))
    out[l] = row
print(json.dumps(out, indent=1))
