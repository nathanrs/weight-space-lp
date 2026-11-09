"""README figures from results/gpu: (1) dev accuracy vs. changed weights, (2) finalist benchmarks with paired SEs."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wslp.eval.results import mean_acc
from wslp.paths import ROOT

OUT = ROOT / "results/figures"
OUT.mkdir(exist_ok=True)
ASTRA_W = 130188


def weights(name):
    if name == "astra_p50c":
        return ASTRA_W
    return sum(v["changed_weights_bf16"] for v in json.load(open(ROOT / f"edits/{name}.json"))["layers"].values())


def dev_acc(name):
    return 100 * np.mean(list(mean_acc(f"dev_a_{name}").values()))


# ---- figure 1: dev_a accuracy vs changed weights
fig, ax = plt.subplots(figsize=(7, 4.2))
series = {
    "LP, agent's direction, broad": (["shift_k8", "shift_k32", "shift_k128", "lp_shift"], "C0", "o"),
    "LP, agent's direction, selective": (["sel_k64", "sel_k512", "lp_sel"], "C1", "s"),
    "LP, measured direction, broad": (["realshift_k8", "realshift_k64"], "C2", "o"),
    "LP, measured direction, selective": (["realsel_k64", "realsel_k512"], "C3", "s"),
}
for label, (names, col, mk) in series.items():
    x = [weights(n) for n in names]
    y = [dev_acc(n) for n in names]
    ax.plot(x, y, marker=mk, color=col, label=label)
for n, txt, col in [("astra_p50c", "Agent edit (50 rounds of search)", "k"), ("recipe_nosearch", "Agent's recipe, no search", "gray")]:
    ax.scatter([weights(n)], [dev_acc(n)], marker="*", s=180, color=col, zorder=5, label=txt)
ax.axhline(dev_acc("base"), color="gray", ls=":", lw=1)
ax.text(1.2e4, dev_acc("base") + 1, "base model", color="gray", fontsize=8)
ax.axhline(dev_acc("random_ctrl"), color="red", ls=":", lw=1)
ax.text(1.2e4, dev_acc("random_ctrl") - 3.5, "random direction (control)", color="red", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("changed weights (layers 15 + 19, bf16)")
ax.set_ylabel("dev-set accuracy, % (128 problems; SE ≈ 4.5)")
ax.set_title("Fewer neurons is as good or better")
ax.set_ylim(0, 70)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "dev_sweep.png", dpi=160)

# ---- figure 2: finalist benchmarks
BENCH = {
    "GSM8K test\n(1319 × 2)": "gsm8k_{}",
    "MATH-500\n(500 × 4)": None,
    "Fresh practice\n(512 × 1)": "final_{}",
    "IFEval strict\n(541 × 1)": "ifeval_{}",
}
MATH = {"base": ["e0_base_math500"], "astra_p50c": ["e0_astra_math500", "math500s1_astra_p50c"],
        "shift_k8": ["e0_shift_k8_math500", "math500s1_shift_k8"], "realshift_k8": ["math500_realshift_k8"]}
MODELS = [("base", "Base model", "lightgray"), ("astra_p50c", "Agent edit (130k weights changed)", "k"),
          ("shift_k8", "LP, agent's direction (16k)", "C0"), ("realshift_k8", "LP, measured direction (16k)", "C2")]
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.2
for j, (b, pat) in enumerate(BENCH.items()):
    data = {}
    for m, _, _ in MODELS:
        dirs = MATH[m] if pat is None else [pat.format(m)]
        data[m] = mean_acc(dirs)  # IFEval per-problem "correct" = prompt-level strict
    ref = data["astra_p50c"]
    for i, (m, label, col) in enumerate(MODELS):
        v = np.array(list(data[m].values()))
        ids = sorted(set(data[m]) & set(ref))
        se = 0 if m in ("astra_p50c", "base") else 100 * np.std([data[m][k] - ref[k] for k in ids], ddof=1) / np.sqrt(len(ids))
        ax.bar(j + (i - 1.5) * w, 100 * v.mean(), w, color=col, label=label if j == 0 else None,
               yerr=se, capsize=2, edgecolor="k", lw=0.4)
        ax.text(j + (i - 1.5) * w, 100 * v.mean() + se + 1, f"{100 * v.mean():.0f}", ha="center", fontsize=7)
ax.set_xticks(range(len(BENCH)), list(BENCH))
ax.set_ylabel("accuracy, %")
ax.set_ylim(0, 85)
ax.set_title("Accuracy: base model vs. the agent's 50-round edit vs. one LP solve")
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "benchmarks.png", dpi=160)
print("wrote", list(OUT.glob("*.png")))
