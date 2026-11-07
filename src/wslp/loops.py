"""Token-level loop detection and labeling of generated traces.

dup[t] = 1 if the K-token window ending at t already occurred earlier in the response (verbatim copy).
Loop onset o = first t such that the suffix from t is >= FRAC duplicated and dup[t] = 1 (the trace never recovers).
Labels (positions index the hidden state that predicts token t+1):
  R_onset  : [o - 16, o + 48)   the decision to start copying
  R_during : t >= o + 64 with dup[t] = 1
  P        : response tokens < o - GAP (loop traces), all response tokens (finished traces)
"""
from dataclasses import dataclass

import numpy as np

K, FRAC, GAP = 16, 0.8, 256


def dup_mask(ids: np.ndarray, k: int = K) -> np.ndarray:
    seen, out = set(), np.zeros(len(ids), bool)
    for t in range(k - 1, len(ids)):
        w = ids[t - k + 1:t + 1].tobytes()
        out[t] = w in seen
        seen.add(w)
    return out


def onset(dup: np.ndarray, frac: float = FRAC, min_tail: int = 512) -> int | None:
    n = len(dup)
    if n < min_tail:
        return None
    suffix = np.cumsum(dup[::-1])[::-1] / np.arange(n, 0, -1)
    ok = np.flatnonzero((suffix >= frac) & dup & (np.arange(n) <= n - min_tail))
    return int(ok[0]) if len(ok) else None


@dataclass
class Labels:
    onset: int | None          # in response-token coordinates
    R_onset: np.ndarray        # response-token positions
    R_during: np.ndarray
    P: np.ndarray


def label(resp_ids: np.ndarray, cut: bool) -> Labels:
    dup = dup_mask(resp_ids)
    o = onset(dup) if cut else None
    n = len(resp_ids)
    if o is None:
        P = np.arange(n) if not cut else np.arange(0)   # cut traces without a detected loop: unlabeled
        return Labels(None, np.arange(0), np.arange(0), P)
    R_on = np.arange(max(o - 16, 0), min(o + 48, n))
    t = np.arange(o + 64, n)
    return Labels(o, R_on, t[dup[o + 64:]], np.arange(0, max(o - GAP, 0)))
