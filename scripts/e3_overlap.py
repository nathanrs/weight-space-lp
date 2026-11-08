"""E3: which neurons do the edits use? Overlap between Astra's 64, the LP's k-neuron supports and the blind supports,
plus how 'always on' the chosen neurons are (the bias-like reading), and each edit's mean push along u.

Also writes steer/<name>.npz ({layer: c*u}) = each rank-1 edit's mean write on fit tokens, i.e. the steering
vector that reproduces the edit's average effect (E4 steering baseline).
"""
import json

import numpy as np

from wslp import acts
from wslp.paths import CACHE, RESULTS, ROOT

H, RES, meta = acts.load("acts_fitgen")
lab, ho = meta["label"], meta["heldout"]
fit = ~ho
EDITS = {"shift_k8": "acts_fitgen_astra_eps1.0", "shift_k32": "acts_fitgen_astra_eps1.0",
         "sel_k64": "acts_fitgen_astra_eps0.25", "realshift_k8": "acts_fitgen_real_eps1.0",
         "realshift_k64": "acts_fitgen_real_eps1.0"}
out, steer = {}, {}
for l in (15, 19):
    h = np.asarray(H[l], np.float32)
    A = np.load(CACHE / f"astra_L{l}.npz")
    sets = {"astra_p50c": (A["u"], A["v"])}
    for name, tag in EDITS.items():
        a = np.load(CACHE / f"sweep_{tag}_L{l}_k{name.split('_k')[1]}.npy")
        sets[name] = (a[0][:1024], a[1])
    active = (np.abs(h[fit]) > 0.1).mean(0)  # share of fit tokens on which a neuron is "on" (|h| > 0.1)
    rows = {}
    for name, (u, v) in sets.items():
        S = np.flatnonzero(v)
        z = h @ v
        rows[name] = {
            "n_neurons": len(S),
            "overlap_astra64": len(np.intersect1d(S, A["S"])),
            "cos_u_astra": round(float(u @ A["u"]), 3),
            "active_share_median": round(float(np.median(active[S])), 3),
            "push_P": round(float(z[(lab == 0) & fit].mean()), 3),
            "push_R_during": round(float(z[(lab == 2) & fit].mean()), 3),
            "push_cv": round(float(z[fit].std() / abs(z[fit].mean())), 3),  # small = nearly constant (bias-like)
        }
        steer.setdefault(name, {})[str(l)] = (float(z[fit].mean()) * u).astype(np.float32)
    S8, S8r = np.flatnonzero(sets["shift_k8"][1]), np.flatnonzero(sets["realshift_k8"][1])
    rows["_overlap_shift_k8_realshift_k8"] = len(np.intersect1d(S8, S8r))
    rows["_active_share_all_neurons_median"] = round(float(np.median(active)), 3)
    out[l] = rows
    print(f"L{l}", json.dumps(rows, indent=1))
for name, d in steer.items():
    np.savez(ROOT / "steer" / f"{name}.npz", **d)
(RESULTS / "e3").mkdir(exist_ok=True)
json.dump(out, open(RESULTS / "e3/overlap.json", "w"), indent=1)
