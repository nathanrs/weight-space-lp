"""Build a text-only Qwen3.5-0.8B checkpoint (as the authors evaluated: Qwen3_5ForCausalLM, model.language_model.*)
with optional replaced tensors from an edit file (safetensors of full bf16 tensors)."""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..paths import BASE_MODEL, BASE_ST, ROOT

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")


def build(out: Path, edit: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "generation_config.json"):
        shutil.copyfile(ROOT / "data/model_text" / f, out / f)
    for f in TOKENIZER_FILES:
        shutil.copyfile(BASE_MODEL / f, out / f)
    sd = {}
    with safe_open(str(BASE_ST), "pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                sd[k] = f.get_tensor(k)
    changed = {}
    if edit is not None:
        for k, v in load_file(str(edit)).items():
            assert k in sd and sd[k].shape == v.shape, k
            v = v.to(torch.bfloat16)
            changed[k] = {"rel_change": float((v.float() - sd[k].float()).norm() / sd[k].float().norm()),
                          "changed_entries": int((v != sd[k]).sum())}
            sd[k] = v.contiguous()
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    json.dump({"edit": str(edit) if edit else None, "changed": changed}, open(out / "edits.json", "w"), indent=1)
    return changed


if __name__ == "__main__":
    print(build(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None))
