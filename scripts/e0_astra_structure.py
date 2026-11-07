"""E0.1-2: rebuild p50c, verify rank-1 structure, save u / S / v_Astra."""
import json

import numpy as np

from wslp.astra import structure
from wslp.paths import CACHE

CACHE.mkdir(exist_ok=True)
st = structure()
report = {}
for layer, d in st.items():
    recon = np.outer(d["u"], d["v"])
    err = np.linalg.norm(recon - d["dw"]) / np.linalg.norm(d["dw"])
    report[layer] = {
        "changed_columns": int(len(d["cols"])),
        "cols_match_manifest": bool(set(d["cols"].tolist()) == set(d["S"].tolist())),
        "sv_ratio_2_to_1": float(d["sv"][1] / d["sv"][0]),
        "rel_err_dW_vs_-6u(uTW0)": float(err),
        "rel_frobenius_change": float(np.linalg.norm(d["dw"]) / np.linalg.norm(d["w0"])),
    }
    np.savez(CACHE / f"astra_L{layer}.npz", u=d["u"], S=d["S"], v=d["v"])
print(json.dumps(report, indent=1))
print("cos(u15, u19) =", float(st[15]["u"] @ st[19]["u"]))
