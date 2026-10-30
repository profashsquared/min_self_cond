#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Numerical experiments for
#   A. Ashrafi, "A Minimum-Self-Conductance Characterization of Kron Reduction
#   in Resistive Networks," IEEE Transactions on Circuits and Systems I:
#   Regular Papers (accepted, 2026).
#
# Author:    Ashkan Ashrafi, Department of Electrical and Computer Engineering,
#            San Diego State University, San Diego, CA, USA (ashrafi@sdsu.edu)
# Copyright: (c) 2026 Ashkan Ashrafi. Released under the MIT License; see the
#            LICENSE file in this repository.
# -----------------------------------------------------------------------------
"""Numerical experiments for the eight-node example of the manuscript
"A Minimum-Self-Conductance Characterization of Kron Reduction in Resistive
Networks" (Section VI). Python >= 3.10.

Rational-arithmetic audit of the printed matrices, exact-data joint and
row-wise programs with the optimal-face uniqueness check, and the finite-SNR
comparison of the regularized joint estimator with the direct pseudoinverse.
The trace-penalty weight is chosen by K-fold cross-validation (default K = 10,
which is leave-one-out at M = 10) over a grid that includes eta = 0 and extends
to eta = 100. See README.md for usage.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import zipfile

import numpy as np

N, S = 8, 5
VERSION = "1.3.0-kfold"
DEFAULT_SEED = 20260921
# Preserve every candidate used in v1.0, add the unregularized constrained
# estimator, and extend the upper end after pilot/full-run boundary checks.
ETAS = np.concatenate((
    np.array([0.0]),
    np.logspace(-6, 0, 9),
    np.array([10.0**0.75, 10.0**1.5, 100.0]),
))
DEFAULT_KAPPAS = (1, 50, 100)
# Curve colors for the three default condition numbers.
KAPPA_COLORS = ("tab:blue", "tab:red", "tab:orange")
METHODS = ("direct", "joint_cv")
BRANCHES = ((1, 2, 2), (1, 6, 1), (2, 3, 3), (2, 7, 2),
            (3, 4, 1), (4, 5, 2), (4, 8, 1), (5, 8, 3),
            (6, 7, 1), (7, 8, 2))
PRINTED_G = [[3,-2,0,0,0,-1,0,0], [-2,7,-3,0,0,0,-2,0],
             [0,-3,4,-1,0,0,0,0], [0,0,-1,4,-2,0,0,-1],
             [0,0,0,-2,5,0,0,-3], [-1,0,0,0,0,2,-1,0],
             [0,-2,0,0,0,-1,5,-2], [0,0,0,-1,-3,0,-2,6]]
PRINTED_INV = {
    "shunt_free": (46, [[26,6,2], [6,12,4], [2,4,9]]),
    "shunted": (70, [[38,6,2], [6,12,4], [2,4,13]])}
PRINTED_KRON = {
    "shunt_free": (46, [[112,-104,0,-2,-6], [-104,274,-138,-8,-24],
                        [0,-138,184,-46,0], [-2,-8,-46,175,-119],
                        [-6,-24,0,-119,149]]),
    "shunted": (70, [[172,-152,0,-2,-6], [-152,512,-210,-8,-24],
                     [0,-210,280,-70,0], [-2,-8,-70,267,-179],
                     [-6,-24,0,-179,233]])}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inverse_fraction(a):
    n = len(a)
    x = [[F(v) for v in row] + [F(i == j) for j in range(n)]
         for i, row in enumerate(a)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if x[i][k])
        x[k], x[pivot] = x[pivot], x[k]
        scale = x[k][k]
        x[k] = [v / scale for v in x[k]]
        for i in range(n):
            if i != k:
                scale = x[i][k]
                x[i] = [v - scale*w for v, w in zip(x[i], x[k])]
    return [row[n:] for row in x]


@dataclass
class Network:
    name: str
    g: np.ndarray
    gamma: np.ndarray
    transfer: np.ndarray
    target: np.ndarray


def networks_and_audit():
    """Reconstruct from branch values; compare independent printed arrays."""
    nets, audit = {}, []
    for name in ("shunt_free", "shunted"):
        g = [[F(0) for _ in range(N)] for _ in range(N)]
        for a, b, c in BRANCHES:
            a -= 1; b -= 1
            g[a][a] += c; g[b][b] += c
            g[a][b] -= c; g[b][a] -= c
        printed_g = [row.copy() for row in PRINTED_G]
        if name == "shunted":
            g[1][1] += 1; g[6][6] += 2
            printed_g[1][1] += 1; printed_g[6][6] += 2
        inv = inverse_fraction([row[S:] for row in g[S:]])
        gamma = [[g[i][j] - sum(g[i][S+k]*inv[k][l]*g[S+l][j]
                  for k in range(N-S) for l in range(N-S))
                  for j in range(S)] for i in range(S)]
        den, values = PRINTED_INV[name]
        printed_inv = [[F(x, den) for x in row] for row in values]
        den, values = PRINTED_KRON[name]
        printed_gamma = [[F(x, den) for x in row] for row in values]
        full_sums = [sum(row) for row in g]
        expected_full_sums = [0,1,0,0,0,0,2,0] if name == "shunted" else [0]*N
        sums = [sum(row) for row in gamma]
        expected_sums = [F(x,70) for x in [12,118,0,8,24]] if name == "shunted" else [F(0)]*S
        comparisons = (("full_matrix", g, printed_g),
                       ("unsourced_inverse", inv, printed_inv),
                       ("kron_matrix", gamma, printed_gamma),
                       ("full_row_sums", [full_sums], [expected_full_sums]),
                       ("kron_row_sums", [sums], [expected_sums]))
        for quantity, calculated, printed in comparisons:
            error = max(abs(a-b) for rowa, rowb in zip(calculated, printed)
                        for a, b in zip(rowa, rowb))
            require(error == 0, f"Exact arithmetic mismatch: {name}/{quantity}")
            audit.append(dict(network=name, quantity=quantity,
                              maximum_absolute_difference_exact=str(error),
                              checked_entries=sum(len(row) for row in calculated)))
        gf, gammaf = np.asarray(g, float), np.asarray(gamma, float)
        transfer = np.vstack((np.eye(S), -np.linalg.solve(gf[S:, S:], gf[S:, :S])))
        target = np.zeros((N, N)); target[:S, :S] = gammaf
        require(np.max(np.abs((gf-target) @ transfer)) < 1e-12, "KCL identity failed")
        nets[name] = Network(name, gf, gammaf, transfer, target)
    return nets, audit


def voltage_matrices(m, kappas, seed):
    # Same construction and global source RMS as the original joint script.
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(S, S)))
    w, _ = np.linalg.qr(rng.normal(size=(m, S)))
    matrices = {}
    for k in kappas:
        vs = u @ np.diag(np.geomspace(1., 1./k, S)) @ w.T
        vs *= np.sqrt(S*m) / np.linalg.norm(vs, "fro")
        require(np.isclose(np.linalg.cond(vs), k, rtol=1e-10), "Condition number mismatch")
        matrices[k] = vs
    return matrices


def clean_data(net, vs):
    v = net.transfer @ vs
    # Currents generated from the full network, not from the target Kron matrix.
    q = net.g @ v
    require(np.max(np.abs(q[S:])) < 1e-11, "Unsourced KCL failed")
    q[S:] = 0.  # Exact known zero injections; no sensors are assumed there.
    return v, q


def symmetric_basis(n=N):
    pairs = [(i, j) for i in range(n) for j in range(i+1)]
    basis = np.zeros((len(pairs), n, n))
    for a, (i, j) in enumerate(pairs):
        basis[a, i, j] = 1.
        basis[a, j, i] = 1.
    return pairs, basis


PAIRS, BASIS = symmetric_basis()
TRACE = np.array([float(i == j and i < S) for i, j in PAIRS])
ROWSUM = BASIS.sum(axis=2).T


def quadratic_terms(v, q):
    m = v.shape[1]
    p, r = v @ v.T / m, q @ v.T / m
    h = np.einsum("aij,jk,bik->ab", BASIS, p, BASIS, optimize=True)
    h = (h + h.T) * 0.5  # Only remove floating-point asymmetry.
    c = np.einsum("ij,aij->a", r, BASIS)
    return h, c


def compress_exact_data(v, q):
    """Equivalent exact equalities G U_r = Q W_r Sigma_r^-1.

    Uses measured V and Q only. All residuals are checked against the original
    uncompressed data. Never use this compression for finite-noise fitting.
    """
    u, singular, vh = np.linalg.svd(v, full_matrices=False)
    cutoff = max(v.shape) * np.finfo(float).eps * singular[0]
    r = int(np.sum(singular > cutoff))
    require(r > 0, "Zero-rank voltage matrix")
    right = vh[:r].T
    projected_q = (q @ right) @ right.T
    require(np.linalg.norm(q-projected_q) <= 1e-10*max(1., np.linalg.norm(q)),
            "Q is inconsistent with the row space of exact V")
    return u[:, :r], (q @ right) / singular[:r], r


def direct_fit(v, q):
    gamma = q[:S] @ np.linalg.pinv(v[:S], rcond=1e-15)
    g = np.zeros((N, N)); g[:S, :S] = gamma
    return g


class GurobiSolver:
    """One licensed environment per process; each fit starts a fresh model."""
    def __init__(self, threads=1):
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError("gurobipy is missing in this Python environment. See README.md.") from exc
        self.gp, self.threads = gp, threads
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    def close(self):
        self.env.dispose()

    def model(self):
        model = self.gp.Model(env=self.env)
        model.Params.OutputFlag = 0
        model.Params.Threads = self.threads
        model.Params.Seed = 0
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        model.Params.BarConvTol = 1e-10
        model.Params.NonConvex = 0
        return model

    def variables(self, model):
        gp = self.gp
        lb = np.array([0. if i == j else -gp.GRB.INFINITY for i, j in PAIRS])
        ub = np.array([gp.GRB.INFINITY if i == j else 0. for i, j in PAIRS])
        x = model.addMVar(len(PAIRS), lb=lb, ub=ub, name="g")
        model.addConstr(ROWSUM @ x >= np.zeros(N), name="passive_rowsums")
        return x

    def qp(self, v, q, lam):
        h, c = quadratic_terms(v, q)
        with self.model() as model:
            x = self.variables(model)
            model.setObjective(0.5 * (x @ h @ x) + (lam*TRACE-c) @ x,
                               self.gp.GRB.MINIMIZE)
            model.optimize()
            require(model.Status == self.gp.GRB.OPTIMAL,
                    f"QP failed: Gurobi status {model.Status}")
            g = np.einsum("a,aij->ij", x.X, BASIS)
            return g, float(model.Runtime)

    def exact_joint(self, v, q):
        u, y, rank = compress_exact_data(v, q)
        a = np.einsum("aij,jt->ita", BASIS, u).reshape(-1, len(PAIRS))
        with self.model() as model:
            x = self.variables(model)
            model.addConstr(a @ x == y.reshape(-1))
            model.setObjective(TRACE @ x, self.gp.GRB.MINIMIZE)
            model.optimize()
            require(model.Status == self.gp.GRB.OPTIMAL,
                    f"Exact joint LP failed: Gurobi status {model.Status}")
            g = np.einsum("a,aij->ij", x.X, BASIS)
        require(np.max(np.abs(g @ v-q)) < 1e-7, "Uncompressed exact joint residual too large")
        return g, rank

    def exact_rows_and_uniqueness(self, v, q, tolerance=1e-7):
        u, y, _ = compress_exact_data(v, q)
        recovered, records = [], []
        for n in range(S):
            with self.model() as model:
                ub = np.zeros(N); ub[n] = self.gp.GRB.INFINITY
                x = model.addMVar(N, lb=-self.gp.GRB.INFINITY, ub=ub)
                model.addConstr(u.T @ x == y[n])
                model.addConstr(np.ones(N) @ x >= 0.)
                obj = np.eye(N)[n]
                model.setObjective(obj @ x, self.gp.GRB.MINIMIZE)
                model.optimize()
                require(model.Status == self.gp.GRB.OPTIMAL, "Row LP failed")
                row = x.X.copy(); optimum = float(row[n])
                recovered.append(row)
                model.addConstr(obj @ x == optimum)
                lower, upper = [], []
                for i in range(N):
                    for sense, destination in ((self.gp.GRB.MINIMIZE, lower),
                                               (self.gp.GRB.MAXIMIZE, upper)):
                        model.setObjective(np.eye(N)[i] @ x, sense)
                        model.optimize()
                        require(model.Status == self.gp.GRB.OPTIMAL,
                                f"Optimal-face LP failed for sourced row {n+1}")
                        destination.append(float(x.X[i]))
                width = float(np.max(np.abs(np.array(upper)-lower)))
                records.append(dict(node=n+1, optimum_S=optimum,
                                    lower_bounds_S=lower, upper_bounds_S=upper,
                                    max_width_S=width,
                                    numerically_unique_within_tolerance=width <= tolerance,
                                    tolerance_S=tolerance))
        recovered = np.asarray(recovered)
        require(np.max(np.abs(recovered @ v-q[:S])) < tolerance,
                "Uncompressed exact row residual too large")
        return recovered, records


def fit_cv(solver, v, q, rng, folds):
    """K-fold cross-validation over the eta grid, then a refit on all columns.

    The columns are permuted once and split into `folds` nearly equal groups.
    Each candidate is fitted on the other groups and scored by its squared KCL
    error on the held-out group; the score is the total over all groups, so
    every column is validated exactly once. With folds == M this is
    leave-one-out. The penalty scale uses the cross-correlation of all M
    measured columns; it is a data-dependent normalization only.
    """
    start = time.perf_counter()
    m = v.shape[1]
    groups = np.array_split(rng.permutation(m), folds)
    # Explicit entrywise maximum, NOT the induced matrix infinity norm.
    scale = max(float(np.max(np.abs(q @ v.T / m))), 1e-12)
    candidates = ETAS * scale
    sse = np.zeros(len(candidates)); optimize_seconds = 0.
    for valid in groups:
        train = np.setdiff1d(np.arange(m), valid)
        for a, lam in enumerate(candidates):
            g, duration = solver.qp(v[:, train], q[:, train], float(lam))
            optimize_seconds += duration
            sse[a] += float(np.sum((g @ v[:, valid]-q[:, valid])**2))
    errors = (sse / (v.shape[0]*m)).tolist()
    selected = int(np.argmin(errors))
    selection_seconds = time.perf_counter()-start
    refit_start = time.perf_counter()
    g, duration = solver.qp(v, q, float(candidates[selected]))
    optimize_seconds += duration
    refit_seconds = time.perf_counter()-refit_start
    return g, dict(lambda_value=float(candidates[selected]), eta_index=selected,
                   eta_value=float(ETAS[selected]), scale=scale, validation_mse=errors,
                   seconds_total=time.perf_counter()-start,
                   seconds_selection=selection_seconds, seconds_refit=refit_seconds,
                   seconds_gurobi_optimize=optimize_seconds)


def diagnostics(g, tolerance):
    symmetry = float(np.max(np.abs(g-g.T)))
    off = g[~np.eye(g.shape[0], dtype=bool)]
    sign = max(0., float(np.max(off)))
    rowsum = max(0., -float(np.min(g.sum(axis=1))))
    minimum_eigenvalue = float(np.linalg.eigvalsh((g+g.T)/2).min())
    return dict(symmetry_max_S=symmetry, positive_offdiag_max_S=sign,
                negative_rowsum_max_S=rowsum, symmetric_part_min_eigenvalue_S=minimum_eigenvalue,
                physical_violation=int(max(symmetry, sign, rowsum) > tolerance))


def measurements(net, vs, experiment, kappa, snr, trial, seed):
    v, q = clean_data(net, vs)
    net_id = 0 if net.name == "shunt_free" else 1
    exp_id = 0 if experiment == "voltage_only" else 1
    # Separate streams: split/candidate changes cannot change the noisy inputs.
    common = [seed, vs.shape[1], net_id, exp_id, int(kappa), int(snr), trial]
    noise_rng = np.random.default_rng(np.random.SeedSequence(common+[0]))
    cv_rng = np.random.default_rng(np.random.SeedSequence(common+[1]))
    sv = np.sqrt(np.mean(v*v))*10.**(-snr/20.)
    sq = np.sqrt(np.mean(q[:S]*q[:S]))*10.**(-snr/20.) if exp_id else 0.
    vt = v + noise_rng.normal(0., sv, v.shape)
    qt = q.copy()
    if exp_id:
        qt[:S] += noise_rng.normal(0., sq, q[:S].shape)
    return v, q, vt, qt, cv_rng, float(sv), float(sq)


def atomic_npz(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def run_condition(path, net, vs, experiment, kappa, snr, cfg, solver):
    estimates = np.empty((cfg.trials, 2, N, N))
    durations = np.empty((cfg.trials, 2))
    selection = np.empty(cfg.trials); refit = np.empty(cfg.trials)
    optimize = np.empty(cfg.trials); lambdas = np.empty(cfg.trials)
    etas = np.empty(cfg.trials, dtype=int); cv_errors = np.empty((cfg.trials, len(ETAS)))
    scales = np.empty(cfg.trials); observed_kappa = np.empty(cfg.trials)
    residuals = np.empty((cfg.trials, 2)); clean_residuals = np.empty((cfg.trials, 2))
    for t in range(cfg.trials):
        v, q, vt, qt, rng, sv, sq = measurements(net, vs, experiment, kappa, snr, t, cfg.seed)
        observed_kappa[t] = np.linalg.cond(vt[:S])
        # Alternate order to reduce systematic cache/order bias in paired timing.
        for method in ((0, 1) if t % 2 == 0 else (1, 0)):
            if method == 0:
                start = time.perf_counter()
                g = direct_fit(vt, qt)
                durations[t, method] = time.perf_counter()-start
            else:
                g, info = fit_cv(solver, vt, qt, rng, cfg.folds)
                durations[t, method] = info["seconds_total"]
                selection[t], refit[t] = info["seconds_selection"], info["seconds_refit"]
                optimize[t], lambdas[t] = info["seconds_gurobi_optimize"], info["lambda_value"]
                etas[t], scales[t] = info["eta_index"], info["scale"]
                cv_errors[t] = info["validation_mse"]
            require(np.all(np.isfinite(g)), "Nonfinite estimate")
            estimates[t, method] = g
            # Same terminal-current target and all measured voltage columns.
            residuals[t, method] = np.linalg.norm((g @ vt-qt)[:S])/max(np.linalg.norm(qt[:S]), 1e-300)
            clean_residuals[t, method] = np.linalg.norm((g @ v-q)[:S])/np.linalg.norm(q[:S])
        if (t+1) % 25 == 0:
            print(f"    {t+1}/{cfg.trials} trials", flush=True)
    atomic_npz(path, estimates=estimates, seconds=durations, selection_seconds=selection,
               refit_seconds=refit, gurobi_optimize_seconds=optimize, lambda_values=lambdas,
               eta_indices=etas, lambda_scales=scales, cv_validation_mse=cv_errors,
               observed_kappa=observed_kappa, measured_terminal_relative_residual=residuals,
               clean_terminal_relative_residual=clean_residuals, sigma_v=sv, sigma_q=sq)


def summarize_condition(path, net, m, experiment, kappa, snr, tolerance):
    with np.load(path, allow_pickle=False) as loaded:
        d = {key: loaded[key] for key in loaded.files}
    estimates = d["estimates"]
    require(estimates.ndim == 4 and estimates.shape[1:] == (2,N,N), "Invalid checkpoint")
    require(all(np.all(np.isfinite(a)) for a in d.values()), "Nonfinite checkpoint data")
    summary, nodewise = [], []
    base = dict(samples=m, network=net.name, experiment=experiment, kappa_clean=kappa, snr_db=snr)
    target_rows = net.target[:S]
    denom = np.linalg.norm(net.gamma, "fro")
    diag = np.diag(net.gamma)
    for method_index, method in enumerate(METHODS):
        gs = estimates[:, method_index]
        diagonal = np.diagonal(gs, axis1=1, axis2=2)[:, :S]
        diag_error = diagonal-diag
        rmse = np.sqrt(np.mean(diag_error**2, axis=0))
        sourced_sq = np.sum((gs[:, :S, :S]-net.gamma)**2, axis=(1,2))
        rows_sq = np.sum((gs[:, :S]-target_rows)**2, axis=(1,2))
        full_sq = np.sum((gs-net.target)**2, axis=(1,2))
        coupling_sq = np.sum(gs[:, :S, S:]**2, axis=(1,2))
        diags = [diagnostics(g, tolerance) for g in gs]
        times = d["seconds"][:, method_index]
        entry = dict(**base, method=method, trials=len(gs),
                     worst_node_relative_rmse=float(np.max(rmse/diag)),
                     sourced_block_relative_frobenius_rmse=float(np.sqrt(np.mean(sourced_sq))/denom),
                     sourced_rows_relative_frobenius_rmse=float(np.sqrt(np.mean(rows_sq))/denom),
                     full_target_relative_frobenius_rmse=float(np.sqrt(np.mean(full_sq))/denom),
                     coupling_frobenius_rms_S=float(np.sqrt(np.mean(coupling_sq))),
                     max_abs_sourced_entry_error_S=float(np.max(np.abs(gs[:, :S]-target_rows))),
                     max_abs_unsourced_coupling_S=float(np.max(np.abs(gs[:, :S,S:]))),
                     mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
                     p90_seconds=float(np.quantile(times, .9)),
                     physical_violation_fraction=float(np.mean([a["physical_violation"] for a in diags])),
                     max_symmetry_violation_S=max(a["symmetry_max_S"] for a in diags),
                     max_positive_offdiagonal_S=max(a["positive_offdiag_max_S"] for a in diags),
                     max_negative_rowsum_S=max(a["negative_rowsum_max_S"] for a in diags),
                     min_symmetric_part_eigenvalue_S=min(a["symmetric_part_min_eigenvalue_S"] for a in diags),
                     mean_measured_terminal_relative_residual=float(d["measured_terminal_relative_residual"][:,method_index].mean()),
                     mean_clean_terminal_relative_residual=float(d["clean_terminal_relative_residual"][:,method_index].mean()),
                     mean_observed_kappa=float(d["observed_kappa"].mean()),
                     mean_selection_seconds=float(d["selection_seconds"].mean()) if method_index else 0.,
                     mean_refit_seconds=float(d["refit_seconds"].mean()) if method_index else 0.,
                     mean_gurobi_optimize_seconds=float(d["gurobi_optimize_seconds"].mean()) if method_index else 0.,
                     mean_lambda=float(d["lambda_values"].mean()) if method_index else 0.,
                     smallest_eta_fraction=float(np.mean(d["eta_indices"] == 0)) if method_index else 0.,
                     largest_eta_fraction=float(np.mean(d["eta_indices"] == len(ETAS)-1)) if method_index else 0.)
        summary.append(entry)
        for n in range(S):
            nodewise.append(dict(**base, method=method, trials=len(gs), node=n+1,
                                 target_S=float(diag[n]), mean_estimate_S=float(diagonal[:,n].mean()),
                                 bias_S=float(diag_error[:,n].mean()), rmse_S=float(rmse[n]),
                                 relative_rmse=float(rmse[n]/diag[n])))
    # A paired bootstrap CI for the *difference* in matrix RMSE, joint minus direct.
    # Fixed postprocessing seed; no estimator or data-generation randomness is affected.
    rng = np.random.default_rng(np.random.SeedSequence([7919,m,kappa,snr,
              0 if net.name == "shunt_free" else 1, 0 if experiment == "voltage_only" else 1]))
    indices = rng.integers(len(estimates), size=(2000, len(estimates)))
    sq = np.sum((estimates[:, :, :S, :S]-net.gamma)**2, axis=(2,3))/denom**2
    boot = np.sqrt(sq[indices,1].mean(axis=1))-np.sqrt(sq[indices,0].mean(axis=1))
    paired = dict(**base, trials=len(estimates),
                  joint_minus_direct_matrix_rmse=summary[1]["sourced_block_relative_frobenius_rmse"]-summary[0]["sourced_block_relative_frobenius_rmse"],
                  paired_bootstrap_ci95_low=float(np.quantile(boot,.025)),
                  paired_bootstrap_ci95_high=float(np.quantile(boot,.975)),
                  joint_to_direct_ratio_of_median_seconds=summary[1]["median_seconds"]/summary[0]["median_seconds"],
                  ci_note="pilot only" if len(estimates) < 30 else "conditional on fixed clean excitation")
    return summary, nodewise, paired


def exact_study(nets, matrices, solver, tolerance):
    records, face_records, matrices_out = [], [], {}
    for name, net in nets.items():
        for kappa, vs in matrices.items():
            v, q = clean_data(net, vs)
            start = time.perf_counter(); direct = direct_fit(v, q)
            direct_seconds = time.perf_counter()-start
            start = time.perf_counter(); joint, rank = solver.exact_joint(v, q)
            joint_seconds = time.perf_counter()-start
            start = time.perf_counter(); rows, faces = solver.exact_rows_and_uniqueness(v, q, tolerance)
            rows_and_certificate_seconds = time.perf_counter()-start
            for method, g, duration in (("direct", direct, direct_seconds),
                    ("joint_exact", joint, joint_seconds)):
                records.append(dict(network=name, samples=vs.shape[1], kappa_clean=kappa,
                    method=method, rank_V=rank,
                    max_sourced_row_error_S=float(np.max(np.abs(g[:S]-net.target[:S]))),
                    max_full_matrix_error_S=float(np.max(np.abs(g-net.target))),
                    max_full_KCL_residual_A=float(np.max(np.abs(g @ v-q))),
                    max_unsourced_coupling_S=float(np.max(np.abs(g[:S,S:]))),
                    seconds=duration))
            for face in faces:
                n = face["node"]-1
                face_records.append(dict(network=name, samples=vs.shape[1], kappa_clean=kappa,
                    **face, max_row_error_S=float(np.max(np.abs(rows[n]-net.target[n]))),
                    max_KCL_residual_A=float(np.max(np.abs(rows[n] @ v-q[n]))),
                    all_rows_and_certificate_seconds=rows_and_certificate_seconds))
            alternative = 0.5*(net.g+net.target)
            require(np.max(np.abs(alternative @ v-q)) < 1e-10, "Alternative exact fit failed")
            matrices_out[f"{name}_k{kappa}"] = dict(target=net.target.tolist(),
                    recovered_joint=joint.tolist(), recovered_rows=rows.tolist(),
                    direct=direct.tolist(), alternative=alternative.tolist(),
                    alternative_max_residual_A=float(np.max(np.abs(alternative @ v-q))),
                    original_diagonal_S=np.diag(net.g)[:S].tolist(),
                    alternative_diagonal_S=np.diag(alternative)[:S].tolist())
    return dict(exact_summary=records, row_uniqueness=face_records, matrices=matrices_out)


def plot_results(summary, output, samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"direct": ("--", "s"), "joint_cv": ("-", "o")}
    experiments = list(dict.fromkeys(row["experiment"] for row in summary))
    for metric, ylabel, stem in (
        ("worst_node_relative_rmse", "Worst-node relative RMSE", "diagonal_comparison"),
        ("sourced_block_relative_frobenius_rmse", "Relative sourced-block matrix RMSE", "matrix_comparison")):
        fig, axes = plt.subplots(len(experiments), 2, figsize=(9, 3.4*len(experiments)),
                                 squeeze=False, sharey=True, layout="constrained")
        for row_index, experiment in enumerate(experiments):
            for col, name in enumerate(("shunt_free", "shunted")):
                ax = axes[row_index,col]
                rows = [r for r in summary if r["experiment"] == experiment and r["network"] == name]
                kappas = sorted({r["kappa_clean"] for r in rows})
                colors = (KAPPA_COLORS if len(kappas) <= len(KAPPA_COLORS)
                          else plt.get_cmap("tab10")(np.linspace(0, .9, len(kappas))))
                for color, k in zip(colors, kappas):
                    for method in METHODS:
                        group = sorted([r for r in rows if r["method"] == method and r["kappa_clean"] == k],
                                       key=lambda r:r["snr_db"])
                        if not group:
                            continue
                        ls, marker = styles[method]
                        values = [r[metric] if r[metric] > 0 else np.nan for r in group]
                        method_label = "Direct" if method == "direct" else "Joint + CV"
                        ax.semilogy([r["snr_db"] for r in group], values, linestyle=ls, marker=marker,
                                    markersize=3, color=color, label=f"{method_label}, k={k}")
                label = "voltage + current noise" if experiment == "voltage_current" else "voltage noise only"
                ax.set_title(f"{label} | {name.replace('_','-')}", fontsize=9)
                ax.set_xlabel("Measurement SNR (dB)"); ax.grid(True, which="both", alpha=.25)
                ax.legend(fontsize=6, ncol=2)
                if col == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle(f"M={samples}; finite-noise estimates (exact LP results reported separately)", fontsize=10)
        for suffix in ("png", "pdf"):
            fig.savefig(output/f"{stem}.{suffix}", dpi=300)
        plt.close(fig)


def self_test():
    nets, audit = networks_and_audit()
    rng = np.random.default_rng(11)
    for m in (10, 1000):
        for k, vs in voltage_matrices(m, (1,100), DEFAULT_SEED).items():
            for net in nets.values():
                v, q = clean_data(net, vs)
                require(np.max(np.abs(direct_fit(v,q)-net.target)) < 1e-10, "Direct exact check failed")
                u, y, rank = compress_exact_data(v, q)
                require(rank == S and np.max(np.abs(net.g @ u-y)) < 1e-10, "SVD equality check failed")
                x = rng.normal(size=len(PAIRS)); g = np.einsum("a,aij->ij", x, BASIS)
                h, c = quadratic_terms(v,q)
                direct_objective = .5*np.sum((g @ v-q)**2)/m
                expanded = .5*x @ h @ x-c @ x+.5*np.sum(q*q)/m
                require(np.isclose(direct_objective, expanded, rtol=1e-12), "QP expansion check failed")
                require(np.allclose(ROWSUM @ x, g.sum(axis=1)), "Passivity map failed")
                require(np.isclose(TRACE @ x, np.trace(g[:S,:S])), "Trace map failed")
                require(diagnostics(net.target,1e-7)["physical_violation"] == 0, "Target passivity failed")
    # Independent HiGHS row LPs check the exact recovery and the optimal face.
    from scipy.optimize import linprog
    for net in nets.values():
        v,q = clean_data(net, voltage_matrices(10,(1,),DEFAULT_SEED)[1])
        u,y,_ = compress_exact_data(v,q)
        for n in range(S):
            objective = np.eye(N)[n]
            kwargs = dict(A_eq=u.T, b_eq=y[n], A_ub=-np.ones((1,N)), b_ub=[0.],
                          bounds=[(None,None) if i == n else (None,0.) for i in range(N)],
                          method="highs")
            fit = linprog(objective,**kwargs)
            require(fit.success and np.max(np.abs(fit.x-net.target[n])) < 1e-8,
                    "Independent HiGHS row recovery failed")
            kwargs["A_eq"] = np.vstack((u.T,objective))
            kwargs["b_eq"] = np.r_[y[n],fit.fun]
            for i in range(N):
                lo = linprog(np.eye(N)[i],**kwargs)
                hi = linprog(-np.eye(N)[i],**kwargs)
                require(lo.success and hi.success and abs(lo.fun+hi.fun) < 1e-8,
                        "Independent optimal-face check failed")
    print(f"PASS: {len(audit)} rational checks; QP expansion, SVD equalities, passivity, direct recovery, and HiGHS row/uniqueness checks.")


def check_gurobi(cfg):
    self_test()
    solver = GurobiSolver(cfg.threads)
    try:
        net = networks_and_audit()[0]["shunted"]
        vs = voltage_matrices(10,(1,),cfg.seed)[1]
        v,q = clean_data(net,vs)
        joint,_ = solver.exact_joint(v,q)
        require(np.max(np.abs(joint[:S]-net.target[:S])) < 1e-6, "Exact joint solver check failed")
        noisy = v + np.random.default_rng(123).normal(0,.001,v.shape)
        qp,_ = solver.qp(noisy,q,1e-5)
        require(diagnostics(qp,cfg.physical_tolerance)["physical_violation"] == 0, "QP passivity check failed")
        print("PASS: Gurobi", ".".join(map(str,solver.gp.gurobi.version())), "small exact LP and noisy QP.")
    finally:
        solver.close()


def package_results(output):
    archive = output.with_name(output.name+"_results.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                z.write(path, Path(output.name)/path.relative_to(output))
    print(f"Results archive: {archive.resolve()}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--samples", nargs="+", type=int, default=[10, 1000])
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--kappas", nargs="+", type=int, default=list(DEFAULT_KAPPAS))
    p.add_argument("--snrs", nargs="+", type=int, default=[0,10,20,30,40,50])
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--folds", type=int, default=10, help="cross-validation folds (folds == M gives leave-one-out)")
    p.add_argument("--physical-tolerance", type=float, default=1e-7, help="diagnostic tolerance, siemens")
    p.add_argument("--output", type=Path, default=Path("revision_full_refined"))
    p.add_argument("--resume", action="store_true", help="resume a matching interrupted run")
    p.add_argument("--include-voltage-only", action="store_true", help="also run the voltage-noise-only experiment")
    p.add_argument("--exact-only", action="store_true", help="rational audit and exact-data programs only")
    p.add_argument("--self-test", action="store_true", help="algebra tests without Gurobi, no output files")
    p.add_argument("--check", action="store_true", help="algebra and two tiny Gurobi fits, no output files")
    p.add_argument("--summarize-only", action="store_true", help="regenerate summaries/plots from saved checkpoints")
    cfg = p.parse_args()
    if cfg.folds < 2 or any(cfg.folds > m for m in cfg.samples):
        p.error("folds must be at least 2 and at most the smallest sample count")
    if any(m - -(-m // cfg.folds) < S for m in cfg.samples):
        p.error("each training fold must keep at least S=5 columns")
    if cfg.trials < 1 or cfg.threads < 1 or cfg.seed < 0 or cfg.physical_tolerance <= 0:
        p.error("trials/threads/tolerance must be positive and seed nonnegative")
    if any(k < 1 for k in cfg.kappas) or any(s < 0 for s in cfg.snrs):
        p.error("kappas must be positive integers; SNRs must be nonnegative integers")
    cfg.samples = sorted(set(cfg.samples)); cfg.kappas = sorted(set(cfg.kappas)); cfg.snrs = sorted(set(cfg.snrs))
    return cfg


def main():
    cfg = parse_args()
    if cfg.self_test:
        self_test(); return
    if cfg.check:
        check_gurobi(cfg); return
    output = cfg.output.resolve()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scientific_config = {key:getattr(cfg,key) for key in ("samples","trials","kappas","snrs","seed",
                         "threads","folds","physical_tolerance","include_voltage_only","exact_only")}
    signature = dict(version=VERSION, script_sha256=script_hash, configuration=scientific_config)
    metadata_file = output/"metadata.json"
    if output.exists():
        require(cfg.resume or cfg.summarize_only,
                f"Output folder exists: {output}. Use a new --output or --resume.")
        require(metadata_file.is_file(), "Existing folder has no benchmark metadata")
        metadata = json.loads(metadata_file.read_text())
        require(metadata["signature"] == signature,
                "Configuration or script changed. Use original flags for resume, or a new output folder.")
    else:
        require(not cfg.resume and not cfg.summarize_only, "Cannot resume a nonexistent output folder")
        output.mkdir(parents=True)
        packages = {}
        for package in ("numpy","scipy","matplotlib","gurobipy"):
            try: packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: packages[package] = "not installed"
        metadata = dict(signature=signature, created_utc=datetime.now(timezone.utc).isoformat(),
                        platform=platform.platform(), machine=platform.machine(), python=sys.version,
                        processor=platform.processor(), logical_cpus=os.cpu_count(), packages=packages,
                        completed=False, etas=ETAS.tolist(),
                        lambda_scale="max absolute entry of Q V.T / M over all measured columns; floor 1e-12",
                        pinv_rcond=1e-15, estimator_order="alternating by trial",
                        eta_grid_note="includes eta=0; preserves the v1.0 grid and extends its upper end to 100",
                        timing=f"perf_counter wall time; includes model construction, {len(ETAS)*cfg.folds} CV fits ({cfg.folds}-fold) and refit; excludes license startup, data generation, metrics, saving and plotting",
                        lambda_scale_note="entrywise max of Q V^T / M over all measured columns",
                        exact_equalities="SVD compression using V,Q only; checked on all original columns",
                        randomization="fixed clean excitation per M,kappa; independent noise and CV seeds per trial",
                        information_sets="direct: terminal V,Q; joint: all nodal V and terminal Q, plus known zero interior injections",
                        target_warning="Ghat_ss is a block, not the noisy Ghat Schur complement; matrix errors target the zero-padded Kron realization",
                        solver_parameters=dict(Threads=cfg.threads, Seed=0, FeasibilityTol=1e-8,
                                               OptimalityTol=1e-8, BarConvTol=1e-10, NonConvex=0))
        write_json(metadata_file, metadata)
    nets,audit = networks_and_audit()
    write_csv(output/"exact_arithmetic_audit.csv", audit)
    solver = None
    try:
        if not cfg.summarize_only:
            solver = GurobiSolver(cfg.threads)
            current_version = list(solver.gp.gurobi.version())
            if "gurobi_runtime_version" in metadata:
                require(metadata["gurobi_runtime_version"] == current_version, "Gurobi version changed during resume")
            metadata["gurobi_runtime_version"] = current_version
            write_json(metadata_file, metadata)
            # Warm up both implementations; exclude license and lazy API startup from timing.
            v,q = clean_data(nets["shunted"],voltage_matrices(10,(1,),cfg.seed)[1])
            direct_fit(v,q); solver.qp(v,q,1e-5)
        total = (len(cfg.samples)*2*len(cfg.snrs)*len(cfg.kappas)*(2 if cfg.include_voltage_only else 1)
                 if not cfg.exact_only else 0)
        print(f"{total} finite-noise conditions; {cfg.trials} trials each; "
              f"{len(ETAS)*cfg.folds+1} QP fits per trial ({cfg.folds}-fold CV).", flush=True)
        done = 0
        for m in cfg.samples:
            folder = output/f"M{m}"; folder.mkdir(exist_ok=True)
            checkpoints = folder/"checkpoints"; checkpoints.mkdir(exist_ok=True)
            matrices = voltage_matrices(m,cfg.kappas,cfg.seed)
            inputs = {f"Vs_k{k}":vs for k,vs in matrices.items()}
            for name,net in nets.items():
                inputs[f"{name}_G"] = net.g; inputs[f"{name}_target"] = net.target
            atomic_npz(folder/"clean_inputs.npz", **inputs)
            exact_file = folder/"exact_results.json"
            if not exact_file.exists():
                require(solver is not None, "Exact checkpoint missing; run without --summarize-only")
                print(f"M={m}: exact-data joint LPs, row LPs, and optimal-face checks", flush=True)
                write_json(exact_file, exact_study(nets,matrices,solver,cfg.physical_tolerance))
            exact = json.loads(exact_file.read_text())
            write_csv(folder/"exact_recovery.csv",exact["exact_summary"])
            write_csv(folder/"uniqueness.csv",[{k:v for k,v in r.items() if not isinstance(v,list)} for r in exact["row_uniqueness"]])
            if cfg.exact_only:
                continue
            summary, nodewise, comparisons = [], [], []
            for experiment,kappas in (("voltage_only",cfg.kappas if cfg.include_voltage_only else []),
                    ("voltage_current",cfg.kappas)):
                for name,net in nets.items():
                    for k in kappas:
                        for snr in cfg.snrs:
                            checkpoint = checkpoints/f"{name}_{experiment}_k{k}_snr{snr}.npz"
                            done += 1
                            print(f"[{done}/{total}] M={m} {name} {experiment} k={k} SNR={snr}"+
                                  (" [saved]" if checkpoint.exists() else ""), flush=True)
                            if not checkpoint.exists():
                                require(solver is not None, "Missing checkpoint; resume the run first")
                                run_condition(checkpoint,net,matrices[k],experiment,k,snr,cfg,solver)
                            a,b,c = summarize_condition(checkpoint,net,m,experiment,k,snr,cfg.physical_tolerance)
                            require(a[0]["trials"] == cfg.trials, "Checkpoint trial count mismatch")
                            summary.extend(a); nodewise.extend(b); comparisons.append(c)
            write_csv(folder/"comparison_summary.csv", summary)
            write_csv(folder/"nodewise_rmse.csv", nodewise)
            write_csv(folder/"paired_comparison.csv", comparisons)
            plot_results(summary,folder,m)
        metadata["completed"] = True
        metadata["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata_file,metadata)
        package_results(output)
    finally:
        if solver is not None:
            solver.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed conditions are saved; repeat the same command with --resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("For a full traceback, set CIRCUIT_GRAPH_TRACEBACK=1.", file=sys.stderr)
        if os.environ.get("CIRCUIT_GRAPH_TRACEBACK"):
            raise
        sys.exit(1)
