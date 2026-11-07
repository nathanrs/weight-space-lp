"""First dev-eval batch: LP (selective / near-uniform), Astra recipe without search, norm-matched random control."""
import json

import numpy as np

from wslp import acts, edits
from wslp.astra import base_tensor
from wslp.paths import CACHE, down_proj

H, RES, meta = acts.load()
lab, ho = meta["label"], meta["heldout"]
fitR, fitP = (lab == 2) & ~ho, (lab == 0) & ~ho
rng = np.random.default_rng(1)
specs = {"lp_sel": {}, "lp_shift": {}, "recipe_nosearch": {}, "random_ctrl": {}}
for l in (15, 19):
    A = np.load(CACHE / f"astra_L{l}.npz")
    u, vA = A["u"], A["v"]
    specs["lp_sel"][l] = (u, np.load(CACHE / f"lp_proto_L{l}_eps0.25_lam100.0.npy"))
    specs["lp_shift"][l] = (u, np.load(CACHE / f"lp_proto_L{l}_eps1.0_lam100.0.npy"))
    # Astra's recipe, one shot, real loops: u = mean-diff of layer output (R_during - P); top-64 writers; alpha 6
    R = np.asarray(RES[l], np.float32)
    ur = R[fitR].mean(0) - R[fitP].mean(0); ur /= np.linalg.norm(ur)
    w0 = base_tensor(down_proj(l))
    h = np.asarray(H[l], np.float32)
    score = (h[fitR].mean(0) - h[fitP].mean(0)) * (ur @ w0)
    S = np.sort(np.argsort(-score)[:64])
    vr = np.zeros(w0.shape[1]); vr[S] = -6.0 * (ur @ w0)[S]
    specs["recipe_nosearch"][l] = (ur, vr)
    print(f"L{l}: recipe overlap with Astra's 64 = {len(np.intersect1d(S, A['S']))}, cos(u_real, u_astra) = {ur @ u:.3f}")
    # Astra's control: same columns and |v|, random unit direction orthogonal to u (Frobenius-matched)
    q = rng.standard_normal(len(u)); q -= (q @ u) * u; q /= np.linalg.norm(q)
    specs["random_ctrl"][l] = (q, vA)
for name, sp in specs.items():
    print(name, json.dumps(edits.write(name, sp)["layers"]))
