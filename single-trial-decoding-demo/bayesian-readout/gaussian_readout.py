"""Gaussian plug-in class posterior: explicit prior, training-only shared covariance.

This is NOT Bayesian integration over parameter uncertainty. No test labels,
per-time fitting, time gains, or test calibration enter prediction.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from jointdecode.config import frame_mask


def fit_gaussian(features, labels, K, weights=None, latent_mean=None, latent_scale=None):
    """Fit class means and LW pooled within-class covariance to training rows.

Nonuniform sample weights are deliberately unsupported: sklearn LedoitWolf
has no sample_weight. The frozen ring design has equal weights, checked below.
"""
    x=np.asarray(features,float);y=np.asarray(labels,int)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('Invalid training rows')
    if not np.array_equal(np.unique(y),np.arange(K)):raise ValueError('Every class is required')
    w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(x),) or not np.isfinite(w).all() or np.any(w<=0) or not np.allclose(w,w[0],rtol=1e-12,atol=1e-12):
        raise ValueError('Nonuniform weights require an explicitly weighted covariance estimator; not silently ignored')
    d=x.shape[1]
    if latent_mean is None and d==0:latent_mean=latent_scale=np.empty(0)
    if latent_mean is None:
        scaler=StandardScaler().fit(x,sample_weight=w);latent_mean=scaler.mean_;latent_scale=scaler.scale_
    mean=np.asarray(latent_mean,float);scale=np.asarray(latent_scale,float)
    if mean.shape!=(d,) or scale.shape!=(d,) or np.any(scale<=0):raise ValueError('Invalid latent scaler')
    z=(x-mean)/scale;mu=np.array([z[y==k].mean(axis=0) for k in range(K)])
    residual=z-mu[y]
    if d:
        lw=LedoitWolf(assume_centered=True).fit(residual);cov=lw.covariance_
        eigen,U=np.linalg.eigh(cov);floor=max(float(np.trace(cov)/d),1.)*1e-8
        stabilized=np.maximum(eigen,floor);cov=(U*stabilized)@U.T;cholesky=np.linalg.cholesky(cov)
        shrinkage=float(lw.shrinkage_);logdet=float(np.log(stabilized).sum())
    else:cov=np.empty((0,0));cholesky=cov.copy();floor=0.;shrinkage=0.;logdet=0.;eigen=np.empty(0)
    return dict(class_means=mu,covariance=cov,cholesky=cholesky,logdet=logdet,latent_mean=mean,latent_scale=scale,
                prior=np.full(K,1/K),K=K,shrinkage=shrinkage,eigenvalue_floor=floor,raw_eigenvalues=eigen,n_floored_eigenvalues=int(np.sum(eigen<floor)),
                training_rows=len(x),class_counts=np.bincount(y,minlength=K),equal_sample_weights=True)


def predict_logposterior(readout,features,prior=None):
    """Accept features only, no labels. Last axis is the coordinate dimension."""
    x=np.asarray(features,float);K=readout['K'];p=np.asarray(readout['prior'] if prior is None else prior,float)
    if p.shape!=(K,) or np.any(p<=0) or not np.isfinite(p).all() or not np.isclose(p.sum(),1):raise ValueError('Prior must be positive and sum to one')
    z=(x-readout['latent_mean'])/readout['latent_scale'];delta=z[...,None,:]-readout['class_means']
    # Shared Gaussian normalizer cancels exactly in Bayes normalization.
    if delta.shape[-1]:
        flat=delta.reshape(-1,delta.shape[-1]);white=solve_triangular(readout['cholesky'],flat.T,lower=True).T
        mahalanobis=(white*white).sum(axis=1).reshape(delta.shape[:-1])
    else:mahalanobis=np.zeros(delta.shape[:-1])
    logjoint=-.5*mahalanobis+np.log(p)
    return logjoint-logsumexp(logjoint,axis=-1,keepdims=True)


def training_rows(X_train,y_train,upstream,config,component):
    """Exact window-mean samples/labels/weights from the existing logistic fit."""
    R=y_train.shape[1];K=len(config.position_labels);N=len(y_train);j=component
    counts=np.array([np.bincount(y_train[:,r],minlength=K) for r in range(R)])
    ranks=range(R) if j==0 else [j-1]
    windows=config.entry_readout_windows_seconds if j==0 else config.rank_readout_windows_seconds
    feats=[];labels=[];weights=[]
    for r in ranks:
        mean=X_train[:,:,frame_mask(config.times_seconds,windows[r])].mean(axis=2)
        coords=((mean-upstream['center'])/upstream['scale'])@upstream['A'][:,upstream['cuts'][j]:upstream['cuts'][j+1]]
        feats.append(coords);labels.append(y_train[:,r]);weights.append(1/((R if j==0 else 1)*K*counts[r,y_train[:,r]]))
    y=np.concatenate(labels)
    return np.concatenate(feats),y,np.concatenate(weights)*len(y)


def fit_bayesian_readouts(X_train,y_train,upstream,config):
    fitted=[]
    for j in range(y_train.shape[1]+1):
        x,y,w=training_rows(X_train,y_train,upstream,config,j)
        diag=upstream['readouts'][j]['diagnostics']
        if x.shape[1]:
            # Verify our recovered rows are exactly consistent with the frozen scaler.
            scaler=StandardScaler().fit(x,sample_weight=w)
            np.testing.assert_allclose(scaler.mean_,diag['latent_mean'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(scaler.scale_,diag['latent_scale'],rtol=1e-12,atol=1e-12)
            mu,sd=diag['latent_mean'],diag['latent_scale']
        else:mu=sd=np.empty(0)
        fitted.append(fit_gaussian(x,y,len(config.position_labels),w,mu,sd))
    return fitted


def predict_bayesian_fold(upstream,readouts,X_test):
    """Return [component, location, trial, time] log-posterior; no labels."""
    Z=(np.asarray(X_test,float)-upstream['center'][None,:,None])/upstream['scale'][None,:,None]
    coords=np.einsum('npt,pd->ntd',Z,upstream['A'],optimize=True)
    return np.array([predict_logposterior(ro,coords[:,:,upstream['cuts'][j]:upstream['cuts'][j+1]]).transpose(2,0,1) for j,ro in enumerate(readouts)])
