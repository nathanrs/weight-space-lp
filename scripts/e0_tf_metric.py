"""E0.5: reproduce Astra's TF transfer metric for its final edit (Astra reported rep -0.1031 / prod -0.0090)."""
import json

import torch

from wslp import tfmetrics as TF
from wslp.astra import edited_tensors
from wslp.model import load, set_down_proj
from wslp.paths import down_proj

tok, model = load()
texts, bounds = TF.probe()
lens = [tok(t, add_special_tokens=False).input_ids.__len__() for t in texts]
print("lengths", lens, "bounds lengths", [b["length"] for b in bounds])
base = TF.logprobs_all(model, tok, texts)
ed = edited_tensors()
with set_down_proj(model, {l: torch.from_numpy(ed[down_proj(l)]) for l in (15, 19)}):
    lp = TF.logprobs_all(model, tok, texts)
print(json.dumps(TF.summary(lp, base, bounds), indent=1))
print("base premature-close logp", [float(base[i][-1]) for i in (12, 13)], "(Astra: -19.77/-19.69)")
