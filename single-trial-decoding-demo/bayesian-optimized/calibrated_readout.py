"""Six exclusive-location conditionals: frame matching + training-only temperature.

No baseline is treated as an empty-content label. Prediction keeps all times.
"""
import numpy as np
from scipy.special import log_softmax
from scipy.optimize import minimize_scalar
from noise_matched import frame_training_rows
from gaussian_readout import fit_gaussian,predict_logposterior
from jointdecode.config import frame_mask


def fit_frame_readouts(X_train,y_train,upstream,config):
    fitted=[]
    for j in range(y_train.shape[1]+1):
        x,y,w,_=frame_training_rows(X_train,y_train,upstream,config,j)
        d=x.shape[1];diag=upstream['readouts'][j]['diagnostics']
        fitted.append(fit_gaussian(x,y,len(config.position_labels),w,diag['latent_mean'] if d else np.empty(0),diag['latent_scale'] if d else np.empty(0)))
    return fitted


def predict_frame_readouts(upstream,readouts,X_test):
    Z=(np.asarray(X_test,float)-upstream['center'][None,:,None])/upstream['scale'][None,:,None]
    coords=np.einsum('npt,pd->ntd',Z,upstream['A'],optimize=True)
    return np.array([predict_logposterior(ro,coords[:,:,upstream['cuts'][j]:upstream['cuts'][j+1]]).transpose(2,0,1) for j,ro in enumerate(readouts)])


def temperature_objective(logits,labels,T):
    if T<=0:raise ValueError('Temperature must be positive')
    lp=log_softmax(np.asarray(logits)/T,axis=-1);y=np.asarray(labels,int)
    return float(-lp[np.arange(len(y)),y].mean())


def fit_temperature(logits,labels):
    """One scalar; predeclared bounds .05..20; deterministic no-worse-than-T1 rule."""
    logits=np.asarray(logits,float);labels=np.asarray(labels,int)
    if logits.ndim!=2 or labels.shape!=(len(logits),) or not np.isfinite(logits).all():raise ValueError('Invalid calibration rows')
    base=temperature_objective(logits,labels,1.)
    result=minimize_scalar(lambda u:temperature_objective(logits,labels,np.exp(u)),bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-6})
    candidate=float(np.exp(result.x));loss=temperature_objective(logits,labels,candidate)
    fallback=not result.success or not np.isfinite(loss) or loss>base
    T=1. if fallback else candidate
    return T,dict(temperature=T,optimizer_success=bool(result.success),fallback_to_T1=fallback,bounds=[.05,20.],near_bound=bool(T<=.0505 or T>=19.8),inner_nll_before=base,inner_nll_after=temperature_objective(logits,labels,T),rows=len(labels))


def crossfit_temperatures(X,y,folds,outer,upstream,config):
    """Four original inner groups, conditional on frozen supervised upstream."""
    J=y.shape[1]+1;K=len(config.position_labels);scores=[[] for _ in range(J)];targets=[[] for _ in range(J)];audit=[]
    for inner in sorted(set(folds)-{outer}):
        tr=np.flatnonzero((folds!=outer)&(folds!=inner));va=np.flatnonzero(folds==inner)
        assert not set(tr)&set(va) and not np.any(folds[tr]==outer)
        ro=fit_frame_readouts(X[tr],y[tr],upstream,config);lp=predict_frame_readouts(upstream,ro,X[va])
        for j in range(J):
            ranks=range(J-1) if j==0 else [j-1];windows=config.entry_readout_windows_seconds if j==0 else config.rank_readout_windows_seconds
            lengths=[]
            for r in ranks:
                mask=frame_mask(config.times_seconds,windows[r]);L=int(mask.sum());lengths.append(L)
                counts=np.bincount(y[va,r],minlength=K);assert np.all(counts==counts[0])
                scores[j].append(lp[j][:,:,mask].transpose(1,2,0).reshape(-1,K));targets[j].append(np.repeat(y[va,r],L))
            assert len(set(lengths))==1, 'This fixed balanced design uses equal per-rank frame weights'
        audit.append(dict(inner_fold=int(inner),train_indices=tr,validation_indices=va))
    fitted=[fit_temperature(np.concatenate(s),np.concatenate(t)) for s,t in zip(scores,targets)]
    return np.array([f[0] for f in fitted]),[f[1] for f in fitted],audit


def apply_temperature(logp,temperatures):
    T=np.asarray(temperatures,float)
    if T.shape!=(len(logp),) or np.any(T<=0) or not np.isfinite(T).all():raise ValueError('One positive temperature per component')
    return log_softmax(np.asarray(logp)/T[:,None,None,None],axis=1)
