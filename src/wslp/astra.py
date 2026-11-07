"""Rebuild Astra's p50c edit from the released diff and pull out its rank-1 structure (u, S, v)."""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

from .paths import BASE_ST, EDIT_LAYERS, EXP8, down_proj

sys.path.insert(0, str(EXP8 / "code"))
import apply_edit  # noqa: E402  (authors' MIT-licensed rebuild code)
import st_numpy as S  # noqa: E402

RUN = EXP8 / "runs" / "p50c"


def base_tensor(name: str) -> np.ndarray:
    with safe_open(str(BASE_ST), "pt") as f:
        return f.get_tensor(name).float().numpy()


def edited_tensors() -> dict[str, np.ndarray]:
    """Edited bf16 tensors (as float32), rebuilt bit-for-bit and hash-checked like apply_edit.py."""
    import hashlib
    man = json.load(open(RUN / "edit" / "manifest.json"))
    z = np.load(RUN / "edit" / "edit.npz")
    tensors = S.load(str(BASE_ST), [t["name"] for t in man["tensors"]])
    out = {}
    for i, t in enumerate(man["tensors"]):
        _, b = tensors[t["name"]]
        assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == t["base_sha256"]
        e = apply_edit.rebuild(b, t, z, i)
        assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"]
        out[t["name"]] = S.bf16_to_f32(e)
    return out


def structure(alpha: float = 6.0) -> dict[int, dict]:
    """Per layer: unit direction u (1024), selected neurons S, and v_Astra (3584) with dW ≈ u v^T.

    Astra's formula: W = W0 - alpha * outer(u, mask * (u @ W0)), so v = -alpha * mask * (u @ W0).
    u is recovered as the top left singular vector of the (bf16-rounded) diff.
    """
    manifest = json.load(open(RUN / "work_notes" / "frozen_weights_manifest.json"))
    edited = edited_tensors()
    out = {}
    for layer in EDIT_LAYERS:
        name = down_proj(layer)
        w0 = base_tensor(name)
        dw = edited[name] - w0
        cols = np.flatnonzero(np.abs(dw).sum(0) > 0)
        U, s, _ = np.linalg.svd(dw[:, cols].astype(np.float64), full_matrices=False)
        u = U[:, 0]
        mask = np.zeros(w0.shape[1], bool)
        sel = np.array(manifest[str(layer)]["selected_neurons"])
        mask[sel] = True
        proj = u @ w0
        if (proj[sel] * (u @ dw)[sel]).sum() > 0:  # fix SVD sign so dW = -alpha u (u^T W0)
            u = -u
            proj = -proj
        v = -alpha * mask * proj
        out[layer] = dict(u=u, S=sel, v=v, w0=w0, dw=dw, cols=cols, sv=s)
    return out
