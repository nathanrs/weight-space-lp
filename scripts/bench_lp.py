"""Benchmark HiGHS solvers on the LP-A instance (layer 15, Astra-matched suppression)."""
import sys, time
import numpy as np
import highspy
import scipy.sparse as sp
from wslp import acts
from wslp.paths import CACHE

nR, nP, ncols = map(int, sys.argv[1:4])
solvers = sys.argv[4].split(",")
H, RES, meta = acts.load()
lab, ho = meta["label"], meta["heldout"]
h = np.asarray(H[15], np.float32); vA = np.load(CACHE / "astra_L15.npz")["v"]
rng = np.random.default_rng(0)
iR = rng.choice(np.flatnonzero((lab == 2) & ~ho), nR, replace=False)
iP = rng.choice(np.flatnonzero((lab == 0) & ~ho), nP, replace=False)
# candidate neurons: largest |mean_R - mean_P| / std
d = (h[lab == 2].mean(0) - h[lab == 0].mean(0)) / (h.std(0) + 1e-6)
cols = np.sort(np.argsort(-np.abs(d))[:ncols])
HR, HP = h[iR][:, cols].astype(np.float64), h[iP][:, cols].astype(np.float64)
dR = -(h[iR] @ vA).astype(np.float64)
eps, C = 0.3, 100.0
k = len(cols)
# vars: v+ v- sR sP ; rows: R (<= -dR), P+ (<= eps), P- (<= eps)
A = sp.vstack([sp.hstack([HR, -HR, -sp.identity(nR), sp.csr_matrix((nR, nP))]),
               sp.hstack([HP, -HP, sp.csr_matrix((nP, nR)), -sp.identity(nP)]),
               sp.hstack([-HP, HP, sp.csr_matrix((nP, nR)), -sp.identity(nP)])]).tocsc()
c = np.concatenate([np.ones(2 * k), C * np.ones(nR + nP)])
ub_row = np.concatenate([-dR, np.full(2 * nP, eps)])
for s in solvers:
    hh = highspy.Highs(); hh.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = c; lp.col_lower_ = np.zeros(A.shape[1]); lp.col_upper_ = np.concatenate([np.full(2 * k, 10.0), np.full(nR + nP, highspy.kHighsInf)])
    lp.row_lower_ = np.full(A.shape[0], -highspy.kHighsInf); lp.row_upper_ = ub_row
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    hh.passModel(lp)
    name = s
    if s == "ipm_nox": hh.setOptionValue("solver", "ipm"); hh.setOptionValue("run_crossover", "off")
    else: hh.setOptionValue("solver", s)
    hh.setOptionValue("time_limit", 300.0)
    t = time.time(); hh.run(); dt = time.time() - t
    x = np.array(hh.getSolution().col_value); v = x[:k] - x[k:2 * k]
    print(f"{name:8s} {dt:6.1f}s status={hh.modelStatusToString(hh.getModelStatus())} obj={hh.getInfo().objective_function_value:.4f} nnz={(np.abs(v)>1e-6).sum()} l1={np.abs(v).sum():.3f}", flush=True)
