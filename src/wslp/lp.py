"""LP compilation of a rank-1 down_proj edit dW = u v^T (only v is unknown).

The edit changes the layer's residual write by u * (v . h_t) at token t, so every constraint is linear in v:

  LP-A  min  sum_j c_j |v_j| + C * sum slack
        s.t. v . h_t <= -delta_t + s_t            t in R   (suppress the u-write on loop tokens)
             |v . h_t| <= eps + s_t               t in P   (leave productive tokens alone)

  LP-B  min  eps + lam * sum_j c_j |v_j| + C * sum slack   (same constraints, eps a variable)

Optional: support restricted to a neuron set (e.g. Astra's 64), or exactly-k support via L1 -> top-k -> re-solve.
Solved with HiGHS (IPM) through scipy; constraints are added in rounds from the full token pool (cutting planes).
"""
from dataclasses import dataclass, field

import highspy
import numpy as np
import scipy.sparse as sp


@dataclass
class Spec:
    delta_R: np.ndarray            # required suppression per R token (> 0 means push the u-write down)
    eps: float | None = None       # productive band; None -> LP-B (minimize eps)
    C: float = 100.0               # slack penalty
    lam: float = 1e-3              # L1 weight in LP-B
    cost: np.ndarray | None = None  # per-neuron L1 weights (default 1)
    support: np.ndarray | None = None
    v_max: float = 10.0


@dataclass
class Result:
    v: np.ndarray
    eps: float
    status: str
    rounds: list = field(default_factory=list)


def _solve(HR, dR, HP, spec: Spec, cols: np.ndarray):
    """One LP over the given rows. Variables: [v+ (k), v- (k), sR (nR), sP (nP), (eps)]."""
    k, nR, nP = len(cols), len(HR), len(HP)
    HRc, HPc = HR[:, cols].astype(np.float64), HP[:, cols].astype(np.float64)
    lpb = spec.eps is None
    cost = np.ones(k) if spec.cost is None else spec.cost[cols]
    l1w = spec.lam if lpb else 1.0
    c = np.concatenate([l1w * cost, l1w * cost, spec.C * np.ones(nR), spec.C * np.ones(nP), [1.0] if lpb else []])
    # R rows:  HR v+ - HR v- - sR <= -dR
    # P rows:  +-(HP v+ - HP v-) - sP - eps <= eps0
    IR, IP = sp.identity(nR, format="csr"), sp.identity(nP, format="csr")
    zRP, zPR = sp.csr_matrix((nR, nP)), sp.csr_matrix((nP, nR))
    e = [sp.csr_matrix((nR, 1))] if lpb else []
    eP = [-sp.csr_matrix(np.ones((nP, 1)))] if lpb else []
    A = sp.vstack([
        sp.hstack([sp.csr_matrix(HRc), sp.csr_matrix(-HRc), -IR, zRP] + e),
        sp.hstack([sp.csr_matrix(HPc), sp.csr_matrix(-HPc), zPR, -IP] + eP),
        sp.hstack([sp.csr_matrix(-HPc), sp.csr_matrix(HPc), zPR, -IP] + eP),
    ], format="csc")
    b = np.concatenate([-dR, np.full(2 * nP, 0.0 if lpb else spec.eps)])
    upper = np.concatenate([np.full(2 * k, spec.v_max), np.full(nR + nP + int(lpb), highspy.kHighsInf)])
    x, msg = _highs(c, A, b, upper)
    if x is None:
        return None, None, msg
    v = np.zeros(HR.shape[1])
    v[cols] = x[:k] - x[k:2 * k]
    return v, (x[-1] if lpb else spec.eps), msg


def _highs(c, A, b_ub, upper, time_limit=1800.0):
    """min c.x s.t. A x <= b_ub, 0 <= x <= upper. HiGHS IPM without crossover (crossover dominates on dense rows)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    h.setOptionValue("time_limit", time_limit)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, np.zeros(A.shape[1]), upper
    lp.row_lower_, lp.row_upper_ = np.full(A.shape[0], -highspy.kHighsInf), b_ub
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status not in ("Optimal", "Time limit reached"):
        return None, status
    return np.array(h.getSolution().col_value), status


def solve(H_R, H_P, spec: Spec, n_init=(2000, 4000), add=(1000, 2000), max_rounds=6, tol=1e-3, seed=0) -> Result:
    """Cutting planes: start from a random subset, add the most violated R / P rows from the full pool each round."""
    rng = np.random.default_rng(seed)
    cols = np.arange(H_R.shape[1]) if spec.support is None else np.asarray(spec.support)
    iR = rng.choice(len(H_R), min(n_init[0], len(H_R)), replace=False)
    iP = rng.choice(len(H_P), min(n_init[1], len(H_P)), replace=False)
    rounds = []
    for it in range(max_rounds):
        v, eps, msg = _solve(H_R[iR], spec.delta_R[iR], H_P[iP], spec, cols)
        if v is None:
            return Result(np.zeros(H_R.shape[1]), np.nan, msg, rounds)
        zR = H_R.astype(np.float32) @ v.astype(np.float32)
        zP = H_P.astype(np.float32) @ v.astype(np.float32)
        violR = zR + spec.delta_R - tol
        violP = np.abs(zP) - eps - tol
        rounds.append({"round": it, "rows": (len(iR), len(iP)), "eps": float(eps), "l1": float(np.abs(v).sum()),
                       "nnz": int((np.abs(v) > 1e-6).sum()), "R_viol": float((violR > 0).mean()),
                       "P_viol": float((violP > 0).mean()), "msg": msg})
        newR = np.setdiff1d(np.argsort(-violR)[:add[0]], iR)
        newP = np.setdiff1d(np.argsort(-violP)[:add[1]], iP)
        newR, newP = newR[violR[newR] > 0], newP[violP[newP] > 0]
        if len(newR) == 0 and len(newP) == 0:
            break
        iR, iP = np.concatenate([iR, newR]), np.concatenate([iP, newP])
    return Result(v, float(eps), "ok", rounds)


def top_k_resolve(H_R, H_P, spec: Spec, v_l1: np.ndarray, k: int, scale: np.ndarray | None = None, **kw) -> Result:
    """Exactly-k support: rank neurons by |v_j| * scale_j from an L1 solution, re-solve restricted to the top k."""
    score = np.abs(v_l1) * (1.0 if scale is None else scale)
    sup = np.sort(np.argsort(-score)[:k])
    return solve(H_R, H_P, Spec(**{**spec.__dict__, "support": sup}), **kw)


# ----------------------------------------------------------------------------------------------- first-order solver
def pdhg(H_R, H_P, spec: Spec, iters=20000, tol=1e-5, device=None, log_every=0, v0=None, omega=1.0):
    """The same LP-A, solved exactly by primal-dual hybrid gradient (Chambolle-Pock; PDLP's base algorithm).

    Eliminating the slacks gives the equivalent convex problem
        min_v  sum_j c_j|v_j| + C sum_R max(0, h_t.v + delta_t) + C sum_P max(0, |h_t.v| - eps),
    i.e. min_v g(v) + f(Kv) with K = [H_R; H_P]. Only mat-vecs with K, K^T are needed, so all tokens fit at once on
    GPU/MPS (no cutting planes) and the L1 prox (soft threshold) returns exactly sparse v.
    Duals y_R in [0, C], y_P in [-C, C] are the LP's constraint multipliers.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    f32 = torch.float32
    cols = np.arange(H_R.shape[1]) if spec.support is None else np.asarray(spec.support)
    K = torch.tensor(np.concatenate([H_R[:, cols], H_P[:, cols]]), dtype=f32, device=dev)
    nR = len(H_R)
    delta = torch.tensor(spec.delta_R, dtype=f32, device=dev)
    eps = float(spec.eps)
    C = float(spec.C)
    cost = torch.ones(len(cols), dtype=f32, device=dev) if spec.cost is None else torch.tensor(spec.cost[cols], dtype=f32, device=dev)
    # ||K|| by power iteration; diagonal-free step sizes with tau*sigma*||K||^2 = 0.9
    x = torch.randn(len(cols), device=dev)
    for _ in range(50):
        x = K.T @ (K @ x)
        x /= x.norm()
    L = float((K @ x).norm())
    tau, sigma = 0.95 / (L * omega), 0.95 * omega / L   # omega: PDLP's primal weight
    v = torch.zeros(len(cols), device=dev) if v0 is None else torch.tensor(v0[cols], dtype=f32, device=dev)
    vbar, y = v.clone(), torch.zeros(K.shape[0], device=dev)
    hist = []

    def objective(v):
        z = K @ v
        return float((cost * v.abs()).sum() + C * torch.relu(z[:nR] + delta).sum() + C * torch.relu(z[nR:].abs() - eps).sum())

    best = (np.inf, v.clone())
    for it in range(iters):
        w = y + sigma * (K @ vbar)
        y = torch.cat([(w[:nR] + sigma * delta).clamp(0, C),
                       torch.sign(w[nR:]) * (w[nR:].abs() - sigma * eps).clamp(min=0).clamp(max=C)])
        u = v - tau * (K.T @ y)
        vn = torch.sign(u) * (u.abs() - tau * cost).clamp(min=0)
        if spec.v_max:
            vn = vn.clamp(-spec.v_max, spec.v_max)
        vbar = 2 * vn - v
        step = float((vn - v).norm() / (vn.norm() + 1e-3))
        v = vn
        if it % 200 == 0 or it == iters - 1:
            obj = objective(v)
            if obj < best[0]:
                best = (obj, v.clone())
            hist.append((it, obj, step))
            if log_every and it % log_every == 0:
                print(f"  pdhg it={it} obj={obj:.4f} step={step:.2e} nnz={int((v != 0).sum())}", flush=True)
            if it > 2000 and len(hist) > 5 and abs(hist[-5][1] - obj) < tol * max(1.0, abs(obj)):
                break
    out = np.zeros(H_R.shape[1])
    out[cols] = best[1].cpu().numpy()
    return Result(out, eps, f"pdhg it={it}", [{"history": hist}])
