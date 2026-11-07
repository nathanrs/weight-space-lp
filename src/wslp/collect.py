"""Teacher-force labeled traces through the (optionally edited) model and cache down_proj inputs h and layer outputs.

Each trace: {"id", "problem", "text", "cut"}. Context = chat prompt + response truncated to onset + TAIL
(loop traces) or MAX_FINISHED tokens (finished traces). Per trace we keep at most N_P / N_ON / N_DUR positions.
Output (npz per layer): h [N,3584] fp16, resid [N,1024] fp16; meta: label (0=P,1=R_onset,2=R_during), trace, pos.
"""
import numpy as np
import torch
from tqdm import tqdm

from .eval import grade as G
from .loops import label
from .model import capture

TAIL, MAX_FINISHED, MAX_CTX = 1024, 8192, 16384
N_P, N_ON, N_DUR = 256, 64, 64


def prompt_ids(tok, problem):
    p = tok.apply_chat_template([{"role": "user", "content": G.build_prompt(problem)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    return tok(p, add_special_tokens=False).input_ids


@torch.no_grad()
def collect(model, tok, traces, layers, seed=0, edits_ctx=None):
    rng = np.random.default_rng(seed)
    H = {l: [] for l in layers}
    RES = {l: [] for l in layers}
    meta = {"label": [], "trace": [], "pos": [], "onset": {}}
    for ti, tr in enumerate(tqdm(traces)):
        pids = prompt_ids(tok, tr["problem"])
        rids = np.array(tok(tr["text"], add_special_tokens=False).input_ids, np.int64)
        L = label(rids, tr["cut"])
        if L.onset is None and tr["cut"]:
            continue
        end = L.onset + TAIL if L.onset is not None else MAX_FINISHED
        end = min(end, len(rids), MAX_CTX - len(pids))
        pick = []
        for lab, arr, k in ((0, L.P, N_P), (1, L.R_onset, N_ON), (2, L.R_during, N_DUR)):
            arr = arr[arr < end]
            if len(arr) > k:
                arr = np.sort(rng.choice(arr, k, replace=False))
            pick += [(lab, int(t)) for t in arr]
        if not pick:
            continue
        meta["onset"][tr["id"]] = L.onset
        ids = torch.tensor(pids + rids[:end].tolist())[None].to(model.device)
        pos = torch.tensor([len(pids) + t for _, t in pick], device=model.device)
        with capture(model, layers) as st:
            model.model(ids)
        for l in layers:
            H[l].append(st[l]["h"][pos].to(torch.float16).cpu().numpy())
            RES[l].append(st[l]["resid"][pos].to(torch.float16).cpu().numpy())
        meta["label"] += [lab for lab, _ in pick]
        meta["trace"] += [ti] * len(pick)
        meta["pos"] += [t for _, t in pick]
    return ({l: np.concatenate(H[l]) for l in layers}, {l: np.concatenate(RES[l]) for l in layers},
            {k: (np.array(v) if isinstance(v, list) else v) for k, v in meta.items()})
