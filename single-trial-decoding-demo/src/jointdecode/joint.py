"""Shared-entry / private-rank soft decomposition, with explicit objective scales.

This alternating, shrinkage-based algorithm is not a single convex optimization.
No held-out labels or ground-truth simulation quantities enter these routines.
"""
from itertools import combinations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit


def _get(config, name, default):
    return getattr(config, name, default)


def derive_priors(TG, times, onsets, bin_frames=2, K=6):
    """Return (pE, pR), shape (R,T), from all directed controlled TGs.

    ``times`` is the original native-frame time axis, not bin centers.
    Raises ValueError rather than substituting fixed windows for zero priors.
    """
    TG = np.asarray(TG, dtype=float)
    times = np.asarray(times, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    R = len(onsets)
    if R < 2 or times.size % bin_frames:
        raise ValueError('Need R >= 2 and T divisible by prior_bin_frames')
    tau = times.reshape(-1, bin_frames).mean(axis=1)
    if TG.shape != (R, R, len(tau), len(tau)):
        raise ValueError('TG dimensions do not match R and native times')
    return _derive_priors(TG, tau, onsets, bin_frames, K)


def _derive_priors(TG, tau, onsets, bin_frames, K):
    R = len(onsets)
    H = np.maximum((TG - 1 / K) / (1 - 1 / K), 0)
    gate = expit(np.clip((tau[None, :] - onsets[:, None]) / .05, -60, 60))
    raw_e = np.zeros((R, len(tau)))
    for r, s in combinations(range(R), 2):
        cross = .5 * (H[r, s] + H[s, r].T)
        delta = (tau - onsets[r])[:, None] - (tau - onsets[s])[None, :]
        support = cross * np.exp(-.5 * (delta / .10) ** 2)
        support *= gate[r, :, None] * gate[s, None, :]
        raw_e[r] += support.sum(axis=1) / (R - 1)
        raw_e[s] += support.sum(axis=0) / (R - 1)
    raw_r = np.zeros_like(raw_e)
    for r in range(R):
        support = (.5 * (H[r, r] + H[r, r].T)).sum(axis=1)
        den = raw_e[r] @ raw_e[r]
        alpha = max(raw_e[r] @ support / den, 0) if den > 1e-12 else 0.
        raw_r[r] = np.maximum(support - alpha * raw_e[r], 0) * gate[r]
    out = []
    for name, raw in [('Entry', raw_e), ('Rank', raw_r)]:
        p = np.maximum(gaussian_filter1d(raw, .75, axis=-1, mode='nearest'), 0)
        norms = np.linalg.norm(p, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError(f'{name} time prior cannot be estimated for ranks '
                             f'{np.flatnonzero(norms <= 1e-12).tolist()}')
        p /= norms[:, None]
        p = np.repeat(p, bin_frames, axis=1)
        p /= np.linalg.norm(p, axis=1)[:, None]
        out.append(p)
    return tuple(out)


def recurrence_pairs(priors, onsets, sfreq, floor=1e-12):
    """All rank-pair onset-aligned indices and unit-sum positive weights."""
    R, T = priors.shape[0] // 2, priors.shape[1]
    pairs = []
    for r, s in combinations(range(R), 2):
        exact = (onsets[s] - onsets[r]) * sfreq
        if not np.isclose(exact, np.rint(exact), atol=1e-8, rtol=0):
            raise ValueError('Noninteger onset shift requires an explicit interpolation rule')
        shift = int(np.rint(exact))
        first = np.arange(max(0, -shift), min(T, T - shift))
        second = first + shift
        w = np.sqrt(priors[r, first] * priors[s, second])
        valid = w > floor
        w = w[valid]
        if len(w):
            w = w / w.sum()
        pairs.append((r, s, first[valid], second[valid], w))
    return pairs


def _objective_factory(B, templates, priors, onsets, sfreq, config):
    R, K, P, T = B.shape
    E, private = templates[0], templates[1:]
    ee = np.mean(E * E)
    rr = np.mean(private * private, axis=(1, 2))
    er = np.mean(private * E, axis=(1, 2))
    be = np.einsum('rkpt,kp->rt', B, E, optimize=True) / (K * P)
    br = np.einsum('rkpt,rkp->rt', B, private, optimize=True) / (K * P)
    bb = np.mean(B * B)
    energies = np.maximum(np.r_[np.repeat(ee, R), rr], np.finfo(float).eps)
    pairs = recurrence_pairs(priors, onsets, sfreq, _get(config, 'floor', 1e-12))
    lp = _get(config, 'tgm_prior_strength', 1.)
    ls = _get(config, 'temporal_smooth_strength', 2.)
    lr = _get(config, 'entry_recurrence_strength', 1.)

    def objective(x):
        profiles = np.asarray(x).reshape(2 * R, T)
        a, b = profiles[:R], profiles[R:]
        value = bb + np.sum(ee*a*a + rr[:,None]*b*b + 2*er[:,None]*a*b
                            - 2*be*a - 2*br*b) / (R*T)
        ga = 2 * (ee*a + er[:,None]*b - be) / (R*T)
        gb = 2 * (rr[:,None]*b + er[:,None]*a - br) / (R*T)
        gradient = np.concatenate([ga, gb])
        delta = profiles - priors
        value += lp * np.sum(energies[:,None]*delta**2) / (2*R*T)
        gradient += 2*lp*energies[:,None]*delta / (2*R*T)
        d2 = np.diff(profiles, n=2, axis=1)
        value += ls * np.sum(energies[:,None]*d2**2) / (2*R*(T-2))
        gd2 = 2*ls*energies[:,None]*d2 / (2*R*(T-2))
        gradient[:, :-2] += gd2
        gradient[:, 1:-1] -= 2*gd2
        gradient[:, 2:] += gd2
        for r, s, first, second, w in pairs:
            diff = a[r, first] - a[s, second]
            weight = lr * energies[0] / len(pairs)
            value += weight * np.sum(w * diff**2)
            g = 2 * weight * w * diff
            gradient[r, first] += g
            gradient[s, second] -= g
        return float(value), gradient.ravel()
    return objective


def profile_objective(x, B, templates, priors, onsets, sfreq, config):
    """Objective and flattened analytic gradient; x is [a_1..a_R,b_1..b_R]."""
    return _objective_factory(B, templates, priors, onsets, sfreq, config)(x)


def fit_templates(B, profiles, config):
    R, K, P, T = B.shape
    design = np.zeros((R*T, R+1))
    for r in range(R):
        design[r*T:(r+1)*T, 0] = profiles[r]
        design[r*T:(r+1)*T, r+1] = profiles[R+r]
    responses = B.reshape(R, K*P, T).transpose(0, 2, 1).reshape(R*T, K*P)
    templates = np.linalg.solve(design.T @ design + _get(config, 'template_ridge', 1e-6)*np.eye(R+1),
                                design.T @ responses).reshape(R+1, K, P)
    templates -= templates.mean(axis=1, keepdims=True)
    e = templates[0].ravel()
    den = e @ e
    strength = _get(config, 'private_overlap_strength', .25)
    if den > _get(config, 'floor', 1e-12):
        for r in range(1, R+1):
            v = templates[r].ravel()
            v -= strength/(1+strength) * e * (e @ v) / den
    templates -= templates.mean(axis=1, keepdims=True)
    return templates


def normalize_profiles(profiles, templates, floor=1e-12):
    """Preserve reconstruction, sharing one normalization across Entry gains."""
    profiles, templates = profiles.copy(), templates.copy()
    R = profiles.shape[0] // 2
    scales = np.r_[np.sqrt(np.sum(profiles[:R]**2)/R),
                   np.linalg.norm(profiles[R:], axis=1)]
    if np.any(scales <= floor):
        raise ValueError('Component collapse during profile normalization')
    profiles[:R] /= scales[0]
    profiles[R:] /= scales[1:, None]
    templates *= scales[:, None, None]
    return profiles, templates


def fit_joint(B, pE, pR, config):
    """Alternate template shrinkage and jointly constrained profile optimization."""
    B = np.asarray(B, dtype=float)
    R, K, P, T = B.shape
    if R < 2 or T < 3:
        raise ValueError('Need at least two ranks and three time points')
    priors = np.concatenate([pE, pR]).astype(float)
    if priors.shape != (2*R, T) or not np.all(np.isfinite(priors)):
        raise ValueError('Invalid prior shape or nonfinite prior')
    profiles = priors.copy()
    onsets = np.asarray(config.target_onsets_seconds)
    sfreq = config.sfreq
    history, records = [], []
    converged = False
    for iteration in range(_get(config, 'max_iter', 80)):
        templates = fit_templates(B, profiles, config)
        objective = _objective_factory(B, templates, priors, onsets, sfreq, config)
        opt = minimize(objective, profiles.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(0, None)] * profiles.size,
                       options={'maxiter': _get(config, 'profile_max_iter', 300),
                                'ftol': _get(config, 'profile_tol', 1e-9),
                                'gtol': _get(config, 'profile_tol', 1e-9), 'maxls': 40})
        records.append({'iteration': iteration+1, 'success': bool(opt.success),
                        'message': str(opt.message), 'nit': int(opt.nit),
                        'fun': float(opt.fun), 'status': int(opt.status)})
        profiles, templates = normalize_profiles(opt.x.reshape(2*R, T), templates,
                                                 _get(config, 'floor', 1e-12))
        templates = fit_templates(B, profiles, config)
        value, _ = profile_objective(profiles, B, templates, priors, onsets, sfreq, config)
        history.append(value)
        if len(history) >= 2 and abs(history[-2]-value) <= _get(config, 'tol', 1e-7)*max(1, abs(history[-2])):
            converged = True
            break
    return {'templates': templates, 'profiles': profiles, 'a': profiles[:R],
            'b': profiles[R:], 'objective_history': np.array(history),
            'converged': converged, 'optimizer_records': records,
            'all_profile_optimizers_success': all(r['success'] for r in records)}


def row_basis(template):
    """Return (P x d orthonormal column basis, all singular values)."""
    centered = template - template.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    d = int(np.sum(s > s[0]*1e-10)) if len(s) and s[0] > 0 else 0
    return vt[:d].T, s


def simultaneous_transform(bases, ridge=.25):
    """Return simultaneous ridge coordinate transform A and block cuts."""
    cuts = np.cumsum([0] + [b.shape[1] for b in bases])
    Q = np.concatenate(bases, axis=1)
    A = np.linalg.solve(Q.T @ Q + ridge*np.eye(Q.shape[1]), Q.T).T
    return A, cuts
