"""Write Astra's p50c edited tensors (exact bf16) to edits/astra_p50c.safetensors for GPU evaluation."""
import torch
from safetensors.torch import save_file

from wslp.astra import edited_tensors
from wslp.paths import ROOT

t = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in edited_tensors().items()}
save_file(t, str(ROOT / "edits/astra_p50c.safetensors"))
print(list(t))
