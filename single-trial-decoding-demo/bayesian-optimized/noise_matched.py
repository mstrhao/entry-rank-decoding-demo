"""Frame-matched likelihood plus baseline-like null. Prediction has no time gates."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
CORE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CORE/'src'));sys.path.insert(0,str(CORE/'bayesian-readout'))
from jointdecode.config import frame_mask
from gaussian_readout import fit_gaussian


def frame_training_rows(X,y,upstream,config,j):
    R=y.shape[1];K=len(config.position_labels);ranks=range(R) if j==0 else [j-1]
    windows=config.entry_readout_windows_seconds if j==0 else config.rank_readout_windows_seconds
    counts=np.array([np.bincount(y[:,r],minlength=K) for r in range(R)])
    Z=(X-upstream['center'][None,:,None])/upstream['scale'][None,:,None]
    A=upstream['A'][:,upstream['cuts'][j]:upstream['cuts'][j+1]]
    coords=np.einsum('npt,pd->ntd',Z,A,optimize=True);features=[];labels=[];weights=[]
    for r in ranks:
        mask=frame_mask(config.times_seconds,windows[r]);L=int(mask.sum())
        if not L:raise ValueError('Empty training window')
        features.append(coords[:,mask,:].reshape(len(X)*L,A.shape[1]));labels.append(np.repeat(y[:,r],L))
        weights.append(np.repeat(1/((R if j==0 else 1)*K*counts[r,y[:,r]]*L),L))
    label=np.concatenate(labels)
    return np.concatenate(features),label,np.concatenate(weights)*len(label),coords


def fit_frame_and_null(X_train,y_train,upstream,config):
    """Only supplied outer/inner training trials; baseline negatives have no labels."""
    baseline=frame_mask(config.times_seconds,config.baseline_window_seconds)
    half_support=3/config.sfreq if config.preprocessing=='baseline_subtract_smooth7' else 0.
    if config.baseline_window_seconds[1]+half_support>min(config.target_onsets_seconds):
        raise ValueError('Baseline training support may overlap first stimulus')
    content=[];null=[]
    for j in range(y_train.shape[1]+1):
        x,y,w,coords=frame_training_rows(X_train,y_train,upstream,config,j)
        diag=upstream['readouts'][j]['diagnostics'];d=x.shape[1]
        mu=diag['latent_mean'] if d else np.empty(0);sd=diag['latent_scale'] if d else np.empty(0)
        content.append(fit_gaussian(x,y,len(config.position_labels),w,mu,sd))
        neg=coords[:,baseline,:].reshape(-1,d) if d else np.empty((len(coords)*int(baseline.sum()),0))
        null.append(fit_gaussian(neg,np.zeros(len(neg),int),1,None,mu,sd))
    return dict(content=content,null=null,prior_content=.5)


def gaussian_loglikelihood(model,features):
    """Full Gaussian log density, including determinant: needed across null/content."""
    z=(np.asarray(features)-model['latent_mean'])/model['latent_scale'];delta=z[...,None,:]-model['class_means'];d=delta.shape[-1]
    if d:
        white=solve_triangular(model['cholesky'],delta.reshape(-1,d).T,lower=True).T
        mahal=(white*white).sum(1).reshape(delta.shape[:-1])
    else:mahal=np.zeros(delta.shape[:-1])
    # Both distributions use SAME scaler. Its Jacobian cancels in their ratio.
    return -.5*(mahal+model['logdet']+d*np.log(2*np.pi))


def posterior_from_features(model,features):
    ll=gaussian_loglikelihood(model['content'],features);ln=gaussian_loglikelihood(model['null'],features)[...,0]
    pi=model.get('prior_content',.5)
    if not 0<pi<1:raise ValueError('Prior content must be strictly between zero and one')
    conditional=ll-logsumexp(ll,axis=-1,keepdims=True)
    K=ll.shape[-1];raw=np.concatenate([(ln+np.log1p(-pi))[...,None],ll+np.log(pi)-np.log(K)],axis=-1)
    joint=raw-logsumexp(raw,axis=-1,keepdims=True)
    presence=np.exp(logsumexp(joint[...,1:],axis=-1))
    return conditional,joint,presence


def predict_frame_and_null(upstream,fitted,X_test):
    """No labels, timestamps, generator gains or ground-truth absence at prediction."""
    Z=(np.asarray(X_test,float)-upstream['center'][None,:,None])/upstream['scale'][None,:,None]
    coords=np.einsum('npt,pd->ntd',Z,upstream['A'],optimize=True);cs=[];js=[];ps=[]
    for j,(content,null) in enumerate(zip(fitted['content'],fitted['null'])):
        x=coords[:,:,upstream['cuts'][j]:upstream['cuts'][j+1]]
        c,q,p=posterior_from_features(dict(content=content,null=null,prior_content=fitted['prior_content']),x)
        cs.append(c.transpose(2,0,1));js.append(q.transpose(2,0,1));ps.append(p)
    return np.array(cs),np.array(js),np.array(ps)


def crossfit_thresholds(X,y,folds,outer,upstream,config):
    """Readout-only cross-fit, conditional on frozen outer-training representation.

The upstream model saw inner validation trials; this is NOT fully nested
representation validation. The final independent outer test is never used.
"""
    baseline=frame_mask(config.times_seconds,config.baseline_window_seconds);scores=[[] for _ in range(y.shape[1]+1)];audit=[]
    for inner in sorted(set(folds)-{outer}):
        tr=np.flatnonzero((folds!=outer)&(folds!=inner));va=np.flatnonzero(folds==inner)
        assert not set(tr)&set(va) and not np.any(folds[tr]==outer) and not np.any(folds[va]==outer)
        fit=fit_frame_and_null(X[tr],y[tr],upstream,config);_,_,p=predict_frame_and_null(upstream,fit,X[va])
        for j in range(len(scores)):scores[j].append(p[j][:,baseline].ravel())
        audit.append(dict(inner_fold=int(inner),train_indices=tr,validation_indices=va))
    scores=[np.concatenate(x) for x in scores]
    thresholds=np.array([np.quantile(x,.95,method='higher') for x in scores])
    diagnostics=[dict(threshold=float(t),crossfit_baseline_fpr=float(np.mean(x>t)),baseline_samples=len(x),decision='strict > threshold',quantile=.95,quantile_method='higher') for x,t in zip(scores,thresholds)]
    return thresholds,diagnostics,audit
