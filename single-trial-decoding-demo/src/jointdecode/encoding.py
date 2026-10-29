"""Training-only baseline scaling and mutually controlled position encoding GLMs."""
from __future__ import annotations
import numpy as np
from scipy.linalg import helmert


def fit_baseline_scaler(X, times, window):
    """Fit ROI mean/population SD across unique training trials and baseline frames."""
    X = np.asarray(X, dtype=np.float64)
    times = np.asarray(times)
    mask = (times >= window[0]) & (times < window[1])
    if not mask.any():
        raise ValueError('Baseline window contains no native frames')
    baseline = X[:, :, mask]
    center = baseline.mean(axis=(0, 2))
    scale = baseline.std(axis=(0, 2), ddof=0)
    degenerate = scale < 1e-10
    scale[degenerate] = 1.
    return center, scale, degenerate


def design_matrix(y0, order, K):
    """Build intercept, R Helmert position blocks, and original trial order."""
    y0 = np.asarray(y0)
    order = np.asarray(order, dtype=np.float64)
    if y0.ndim != 2 or order.shape != (len(y0),):
        raise ValueError('Labels/order shape mismatch')
    if not np.issubdtype(y0.dtype, np.integer) or np.any((y0 < 0) | (y0 >= K)):
        raise ValueError('GLM requires zero-based integer position labels')
    C = helmert(K, full=False).T
    return np.column_stack([np.ones(len(y0)), *[C[y0[:, r]] for r in range(y0.shape[1])], order])


def target_columns(r, K):
    return np.arange(1 + r * (K - 1), 1 + (r + 1) * (K - 1))


def fit_rank_glms(Z, y0, order, K, ridge=1e-3):
    """Fit one inverse-frequency weighted controlled GLM for each target.

    Ridge does not repair confounded target blocks: each target is first
    residualized against all nuisance columns and checked for identifiability.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y0 = np.asarray(y0)
    N, P, T = Z.shape
    R = y0.shape[1]
    if N != len(y0) or not np.isfinite(Z).all():
        raise ValueError('Invalid GLM responses')
    D = design_matrix(y0, order, K)
    C = helmert(K, full=False).T
    counts = np.stack([np.bincount(y0[:, r], minlength=K) for r in range(R)])
    if np.any(counts == 0):
        raise ValueError('Every training rank must contain all position classes')
    penalty = np.eye(D.shape[1]); penalty[0, 0] = 0.
    response = Z.reshape(N, P * T)
    betas, B, ranks, nuisance_columns = [], [], [], []
    tol = max(D.shape) * np.finfo(float).eps * np.linalg.norm(D, 2)
    for r in range(R):
        cols = target_columns(r, K)
        nuisance = np.setdiff1d(np.arange(D.shape[1]), cols)
        residual = D[:, cols] - D[:, nuisance] @ np.linalg.lstsq(D[:, nuisance], D[:, cols], rcond=None)[0]
        rank = np.linalg.matrix_rank(residual, tol=tol)
        if rank != K - 1:
            raise ValueError(f'Target {r + 1} GLM is not identifiable: residual rank {rank}, required {K - 1}')
        weights = N / (K * counts[r, y0[:, r]])
        DW = D.T * weights
        beta = np.linalg.solve(DW @ D + ridge * penalty, DW @ response).reshape(D.shape[1], P, T)
        betas.append(beta)
        B.append(np.einsum('kd,dpt->kpt', C, beta[cols]))
        ranks.append(rank)
        nuisance_columns.append(nuisance)
    names = ['intercept'] + [f'target_{r+1}_helmert_{d+1}' for r in range(R) for d in range(K-1)] + ['trial_order']
    return dict(B=np.stack(B), beta=np.stack(betas), design_columns=names,
                class_counts=counts, identifiable_ranks=np.array(ranks),
                nuisance_columns=nuisance_columns, ridge=float(ridge))


def conditional_residuals(Z, y0, order, glm, K):
    """Supervised TG-only residuals, never used for outer-test decoding."""
    D = design_matrix(y0, order, K)
    Z = np.asarray(Z, dtype=np.float64)
    result = []
    for r, beta in enumerate(glm['beta']):
        cols = glm['nuisance_columns'][r]
        pred = (D[:, cols] @ beta[cols].reshape(len(cols), -1)).reshape(Z.shape)
        result.append(Z - pred)
    return np.stack(result)
