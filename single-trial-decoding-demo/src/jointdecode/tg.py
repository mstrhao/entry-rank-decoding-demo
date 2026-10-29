"""Nested controlled-centroid temporal generalization, using no outer-test data."""
from __future__ import annotations
import numpy as np
from .encoding import fit_baseline_scaler, fit_rank_glms, conditional_residuals


def bin_native(X, bin_frames=2):
    if X.shape[-1] % bin_frames:
        raise ValueError('Native time length must be divisible by prior_bin_frames')
    return X.reshape(*X.shape[:-1], X.shape[-1] // bin_frames, bin_frames).mean(axis=-1)


def fit_weighted_centroids(residual, labels, K):
    """Return native-coordinate score weights (U,K,P) and intercepts (U,K).

    Every train time has its own weighted neuron standardization. Folding that
    standardization into centroid weights is algebraically exact.
    """
    N, P, U = residual.shape
    counts = np.bincount(labels, minlength=K)
    if np.any(counts == 0):
        raise ValueError('Centroid training data must contain all classes')
    weights = N / (K * counts[labels])
    mean = np.einsum('n,npu->pu', weights, residual) / weights.sum()
    var = np.einsum('n,npu->pu', weights, (residual - mean[None]) ** 2) / weights.sum()
    for u in range(U):
        bad = var[:, u] <= 1e-12
        good = var[:, u][~bad]
        var[bad, u] = np.median(good) if len(good) else 1.
    scale = np.sqrt(var)
    standardized = (residual - mean[None]) / scale[None]
    centroids = np.stack([np.einsum('n,npu->pu', weights[labels == k], standardized[labels == k]) / weights[labels == k].sum() for k in range(K)])
    W = (centroids / scale[None]).transpose(2, 0, 1)
    bias = (-np.einsum('kpu,pu->ku', centroids, mean / scale) - .5 * np.sum(centroids ** 2, axis=1)).T
    return W, bias


def centroid_tg_counts(rtrain, rval, ytrain, yval, K, chunk_size=15):
    """Class hit counts (R,R,U,V,K), sample counts (R,R,K)."""
    R, _, P, U = rtrain.shape
    _, Nval, _, V = rval.shape
    hits = np.zeros((R, R, U, V, K), dtype=np.int64)
    counts = np.zeros((R, R, K), dtype=np.int64)
    validation_flat = [rval[s].transpose(1, 0, 2).reshape(P, -1) for s in range(R)]
    for r in range(R):
        W, bias = fit_weighted_centroids(rtrain[r], ytrain[:, r], K)
        for s in range(R):
            labels = yval[:, s]
            counts[r, s] = np.bincount(labels, minlength=K)
            for start in range(0, U, chunk_size):
                stop = min(start + chunk_size, U)
                scores = (W[start:stop].reshape(-1, P) @ validation_flat[s]).reshape(stop-start, K, Nval, V)
                scores += bias[start:stop, :, None, None]
                pred = scores.argmax(axis=1)
                for k in range(K):
                    hits[r, s, start:stop, :, k] = (pred[:, labels == k, :] == k).sum(axis=1)
    return hits, counts


def nested_tg(X, y0, order, folds, outer, config):
    """Pool inner-validation class hits before balanced-accuracy averaging.

    Returned audit indices refer to original input rows. Caller can attach IDs
    and group IDs without inferring source identity from row numbers.
    """
    K = len(config.position_labels)
    times = np.asarray(config.times_seconds)
    folds = np.asarray(folds)
    total_hits = total_counts = None
    audit = []
    for inner in sorted(set(folds.tolist()) - {outer}):
        itr = np.flatnonzero((folds != outer) & (folds != inner))
        iva = np.flatnonzero(folds == inner)
        center, scale, degenerate = fit_baseline_scaler(X[itr], times, config.baseline_window_seconds)
        ztrain = bin_native((X[itr] - center[None, :, None]) / scale[None, :, None], config.prior_bin_frames)
        zval = bin_native((X[iva] - center[None, :, None]) / scale[None, :, None], config.prior_bin_frames)
        glm = fit_rank_glms(ztrain, y0[itr], order[itr], K)
        rtrain = conditional_residuals(ztrain, y0[itr], order[itr], glm, K)
        rval = conditional_residuals(zval, y0[iva], order[iva], glm, K)
        hits, counts = centroid_tg_counts(rtrain, rval, y0[itr], y0[iva], K)
        if total_hits is None:
            total_hits, total_counts = hits, counts
        else:
            total_hits += hits; total_counts += counts
        audit.append(dict(inner_fold=int(inner), train_indices=itr, validation_indices=iva,
                          train_class_counts=glm['class_counts'],
                          validation_class_counts=np.stack([np.bincount(y0[iva, r], minlength=K) for r in range(y0.shape[1])]),
                          center=center, scale=scale, degenerate=degenerate,
                          identifiable_ranks=glm['identifiable_ranks']))
    if total_hits is None or np.any(total_counts == 0):
        raise ValueError('Insufficient inner validation classes for TG prior')
    TG = (total_hits / total_counts[:, :, None, None, :]).mean(axis=-1)
    return dict(TG=TG, hits=total_hits, counts=total_counts, audit=audit,
                bin_times=bin_native(times, config.prior_bin_frames))
