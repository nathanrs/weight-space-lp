"""Load the Qwen3.5-0.8B text model and capture MLP down_proj inputs / residual outputs with hooks."""
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import BASE_MODEL


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load(dtype=torch.bfloat16, dev: str | None = None):
    dev = dev or device()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(dev).eval()
    return tok, model


def layers(model):
    m = model.model
    return (m.language_model if hasattr(m, "language_model") else m).layers


@contextmanager
def capture(model, layer_ids, what=("h", "resid")):
    """Record per-layer down_proj input ('h', [T,3584]) and layer output ('resid', [T,1024]) for batch 1."""
    store = {l: {} for l in layer_ids}
    hooks = []
    L = layers(model)
    for l in layer_ids:
        if "h" in what:
            hooks.append(L[l].mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, l=l: store[l].__setitem__("h", args[0][0].detach())))
        if "resid" in what:
            hooks.append(L[l].register_forward_hook(
                lambda mod, args, out, l=l: store[l].__setitem__("resid", (out[0] if isinstance(out, tuple) else out)[0].detach())))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def rank1_edit(model, edits: dict[int, tuple[torch.Tensor, torch.Tensor]]):
    """Temporarily set down_proj W <- bf16(W0 + u v^T) per layer {layer: (u, v)}; restores W0 on exit."""
    L = layers(model)
    saved = {}
    for l, (u, v) in edits.items():
        W = L[l].mlp.down_proj.weight
        saved[l] = W.data.clone()
        W.data = (W.data.float() + torch.outer(u.float(), v.float()).to(W.device)).to(W.dtype)
    try:
        yield
    finally:
        for l, w in saved.items():
            L[l].mlp.down_proj.weight.data = w


@contextmanager
def set_down_proj(model, tensors: dict[int, torch.Tensor]):
    """Temporarily replace whole down_proj weights {layer: W} (cast to the model dtype)."""
    L = layers(model)
    saved = {}
    for l, w in tensors.items():
        W = L[l].mlp.down_proj.weight
        saved[l] = W.data.clone()
        W.data = w.to(W.device, W.dtype)
    try:
        yield
    finally:
        for l, w in saved.items():
            L[l].mlp.down_proj.weight.data = w


@torch.no_grad()
def token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    """lp[t-1] = log p(ids[t] | ids[:t]) for a single sequence ids [T]. Same indexing as Astra's tool."""
    logits = model(ids[None].to(model.device)).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[1:, None].to(logits.device))[:, 0].cpu()
