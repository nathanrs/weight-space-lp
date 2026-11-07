"""Load cached activations and split them by trace into fit / held-out."""
import numpy as np

from .paths import CACHE

NAMES = {0: "P", 1: "R_onset", 2: "R_during"}


def load(name="acts_released", layers=(15, 19), heldout_frac=0.2, seed=0):
    d = CACHE / name
    meta = dict(np.load(d / "meta.npz"))
    traces = np.unique(meta["trace"])
    rng = np.random.default_rng(seed)
    held = set(rng.choice(traces, int(len(traces) * heldout_frac), replace=False).tolist())
    meta["heldout"] = np.array([t in held for t in meta["trace"]])
    H = {l: np.load(d / f"h_L{l}.npy", mmap_mode="r") for l in layers}
    RES = {l: np.load(d / f"resid_L{l}.npy", mmap_mode="r") for l in layers}
    return H, RES, meta
