"""Score edit files on Astra's TF transfer probe (repetitive / productive branch delta logp, premature close)."""
import json
import sys

from safetensors.torch import load_file

from wslp import tfmetrics as TF
from wslp.model import layers, load, set_down_proj
from wslp.paths import ROOT

tok, model = load()
texts, bounds = TF.probe()
base = TF.logprobs_all(model, tok, texts)
out = {}
for name in sys.argv[1:]:
    t = load_file(str(ROOT / "edits" / f"{name}.safetensors"))
    with set_down_proj(model, {int(k.split(".")[3]): v for k, v in t.items()}):
        s = TF.summary(TF.logprobs_all(model, tok, texts), base, bounds)
    out[name] = {"rep": round(s["repetitive"]["branch_delta"], 4), "prod": round(s["productive"]["branch_delta"], 4),
                 "close": [round(x, 2) for x in s["premature_close_logp"]]}
    print(name, json.dumps(out[name]), flush=True)
