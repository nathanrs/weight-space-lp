"""Astra's teacher-forced transfer-probe metric (p50c tool call 82), so our edits are scored on its exact yardstick.

14 texts: 6 cases x (productive, repetitive) branches sharing a prefix, plus 2 premature-</think> controls.
branch_delta = mean over branch tokens of (edited - base) log p; reported per condition.
"""
import json

import numpy as np

from .model import token_logprobs
from .paths import EXP8

NOTES = EXP8 / "runs/p50c/work_notes"


def probe():
    texts = json.load(open(NOTES / "transfer_probe_request.json"))["texts"]
    bounds = json.load(open(NOTES / "transfer_probe_bounds.json"))
    return texts, bounds


def logprobs_all(model, tok, texts):
    return [token_logprobs(model, tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0]).numpy()
            for t in texts]


def summary(lp, base, bounds):
    out = {}
    for cond in ("productive", "repetitive"):
        ix = [i for i, b in enumerate(bounds) if b["condition"] == cond]
        d = [float((lp[i] - base[i])[bounds[i]["start"] - 1:bounds[i]["length"] - 1].mean()) for i in ix]
        out[cond] = {"branch_delta": float(np.mean(d)), "per_case": np.round(d, 4).tolist()}
    out["premature_close_logp"] = [float(lp[i][-1]) for i in (12, 13)]
    return out
