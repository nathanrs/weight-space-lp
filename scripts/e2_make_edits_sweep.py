"""Edit files from sparsity-sweep solutions: sweep_<tag>_L{l}_k{k}.npy holds [u (padded), v]."""
import json
import sys

import numpy as np

from wslp import edits
from wslp.paths import CACHE

tag = sys.argv[1]           # e.g. acts_fitgen_astra_eps1.0
ks = [int(x) for x in sys.argv[2].split(",")]
short = sys.argv[3]          # e.g. shift
for k in ks:
    sp = {}
    for l in (15, 19):
        a = np.load(CACHE / f"sweep_{tag}_L{l}_k{k}.npy")
        sp[l] = (a[0][:1024], a[1])
    info = edits.write(f"{short}_k{k}", sp, note=f"{tag} k={k}")
    print(f"{short}_k{k}", {l: d["changed_weights_bf16"] for l, d in info["layers"].items()})
