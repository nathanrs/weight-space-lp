"""Steering-vector baseline as a vLLM general plugin: add a fixed vector to chosen layers' MLP output (= the layer's
write into the residual stream, exactly where a rank-1 down_proj edit writes u * (v.h)).

Enabled by env WSLP_STEER=<npz with {"15": vec, "19": vec}> and optional WSLP_STEER_SCALE (default 1). Registered
through the "vllm.general_plugins" entry point, so it runs in every vLLM process before the model is built; the
vector is a module buffer, so torch.compile / CUDA graphs capture it like any weight. No-op when WSLP_STEER is unset.
"""
import os


def register():
    path = os.environ.get("WSLP_STEER")
    if not path:
        return
    import numpy as np
    import torch
    from vllm.model_executor.models import qwen3_5

    scale = float(os.environ.get("WSLP_STEER_SCALE", "1"))
    vecs = {int(k): (v * scale).astype(np.float32) for k, v in np.load(path).items()}
    cls = qwen3_5.Qwen3_5DecoderLayer
    if getattr(cls, "_wslp_steer", False):
        return
    init, fwd = cls.__init__, cls.forward

    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        self.wslp_steer = self.layer_idx in vecs
        if self.wslp_steer:
            p = next(self.parameters())  # build on the layer's own device/dtype: no host copies under CUDA graphs
            self.register_buffer("wslp_vec", torch.tensor(vecs[self.layer_idx], device=p.device, dtype=p.dtype),
                                 persistent=False)
            print(f"[wslp-steer] layer {self.layer_idx}: |vec| = {float(self.wslp_vec.norm()):.3f} (scale {scale})",
                  flush=True)

    def forward(self, *a, **kw):
        hidden_states, residual = fwd(self, *a, **kw)
        if self.wslp_steer:
            hidden_states = hidden_states + self.wslp_vec
        return hidden_states, residual

    cls.__init__, cls.forward, cls._wslp_steer = __init__, forward, True
