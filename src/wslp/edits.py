"""Turn rank-1 specs {layer: (u, v)} into full bf16 down_proj tensors (the evaluated form) + bookkeeping."""
import json

import numpy as np
import torch
from safetensors.torch import save_file

from .astra import base_tensor
from .paths import ROOT, down_proj


def write(name: str, rank1: dict[int, tuple[np.ndarray, np.ndarray]], note: str = "") -> dict:
    tensors, info = {}, {"name": name, "note": note, "layers": {}}
    for l, (u, v) in rank1.items():
        w0 = torch.from_numpy(base_tensor(down_proj(l)))
        w = (w0.double() + torch.outer(torch.from_numpy(u).double(), torch.from_numpy(v).double())).to(torch.bfloat16)
        w0b = w0.to(torch.bfloat16)
        changed = (w != w0b)
        tensors[down_proj(l)] = w
        info["layers"][l] = {"nnz_v": int((v != 0).sum()), "changed_weights_bf16": int(changed.sum()),
                             "changed_columns": int(changed.any(0).sum()),
                             "rel_frobenius": float((w.float() - w0b.float()).norm() / w0b.float().norm())}
    (ROOT / "edits").mkdir(exist_ok=True)
    save_file(tensors, str(ROOT / "edits" / f"{name}.safetensors"))
    json.dump(info, open(ROOT / "edits" / f"{name}.json", "w"), indent=1)
    return info
