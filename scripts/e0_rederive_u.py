"""E0.3: recompute Astra's repetition direction from its 4 hand-written pairs (fwd_4) and compare with the edit's u.

Astra (tool call 9/16): u_l = normalize(mean_pairs(mean_last36(resid_l | repetitive) - mean_last36(resid_l | productive))),
resid_l = output of layer l (hidden_states[l+1]); texts alternate productive (even) / repetitive (odd).
"""
import json

import numpy as np
import torch

from wslp.model import capture, load
from wslp.paths import CACHE, EXP8

req = json.load(open(EXP8 / "runs/p50c/requests/fwd_4.json.claimed"))
texts = req["texts"]
tok, model = load()
ids = [tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device) for t in texts]
print("token lengths", [x.shape[1] for x in ids], "(Astra: [124,123,131,136,107,146,98,135])")

layers = list(range(24))
means = {l: [] for l in layers}
mlp_means = {l: [] for l in layers}
with torch.no_grad():
    for x in ids:
        with capture(model, layers) as st:
            model.model(x)
        for l in layers:
            means[l].append(st[l]["resid"][-36:].float().mean(0).cpu().numpy())
            mlp_means[l].append(st[l]["h"][-36:].float().mean(0).cpu().numpy())
out = {}
for l in layers:
    m = np.stack(means[l])
    d = (m[1::2] - m[::2]).mean(0)
    out[l] = d / np.linalg.norm(d)
for l in (15, 19):
    ua = np.load(CACHE / f"astra_L{l}.npz")["u"]
    print(f"L{l}: cos(rederived u, Astra edit u) = {float(out[l] @ ua):.4f}")
np.savez(CACHE / "fwd4_directions.npz", **{f"u{l}": out[l] for l in layers},
         **{f"mlp{l}": np.stack(mlp_means[l]) for l in layers})
