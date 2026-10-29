"""Auditable outer-fivefold / inner-fourfold simulation pipeline."""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedGroupKFold
from .config import frame_mask
from .encoding import fit_baseline_scaler,fit_rank_glms
from .tg import nested_tg
from .joint import derive_priors,fit_joint,row_basis
from .readout import fit_readouts,predict_fold,predict_fold_folded


def validate_inputs(epochs,target_labels,config,trial_ids,groups,folds):
    X=np.asarray(epochs,dtype=float); raw=np.asarray(target_labels)
    R=len(config.target_onsets_seconds); K=len(config.position_labels)
    if X.ndim!=3 or raw.shape!=(len(X),R): raise ValueError('Expected epochs[N,P,T], complete target_labels[N,R]')
    if not np.isfinite(X).all() or not np.issubdtype(raw.dtype,np.number) or not np.isfinite(raw).all(): raise ValueError('Missing/nonfinite data or labels are unsupported')
    if not np.array_equal(raw,raw.astype(int)) or not np.isin(raw,config.position_labels).all(): raise ValueError('Labels must be complete integers in 1..K; zero is not a location')
    if tuple(config.position_labels)!=tuple(range(1,K+1)) or K<3 or R<2: raise ValueError('Need shared vocabulary 1..K (K>=3), R>=2')
    t=np.asarray(config.times_seconds)
    if len(t)!=X.shape[2] or len(t)<3 or not np.allclose(np.diff(t),1/config.sfreq): raise ValueError('Explicit uniformly sampled times must match T and sfreq')
    if len(t)%config.prior_bin_frames: raise ValueError('Time length must be divisible by prior_bin_frames')
    if len(config.entry_readout_windows_seconds)!=R or len(config.rank_readout_windows_seconds)!=R: raise ValueError('One readout window per target required')
    windows=[config.baseline_window_seconds,*config.entry_readout_windows_seconds,*config.rank_readout_windows_seconds]
    if any(w[1]<=w[0] or not frame_mask(t,w).any() for w in windows): raise ValueError('Empty/invalid half-open time window')
    if config.baseline_window_seconds[1]>min(config.target_onsets_seconds): raise ValueError('Shared baseline must end before first target')
    n=len(X); ids=np.arange(n) if trial_ids is None else np.asarray(trial_ids); group=ids.copy() if groups is None else np.asarray(groups)
    if ids.shape!=(n,) or len(np.unique(ids))!=n or group.shape!=(n,): raise ValueError('Unique trial_ids and length-N groups required')
    y=raw.astype(int)-1  # The sole external-to-internal label conversion.
    if folds is None:
        folds=np.full(n,-1,dtype=int)
        strata=np.array(['-'.join(map(str,row)) for row in y])
        splitter=StratifiedGroupKFold(5,shuffle=True,random_state=config.seed_split)
        for k,(_,test) in enumerate(splitter.split(X,strata,group)): folds[test]=k
    folds=np.asarray(folds)
    if folds.shape!=(n,) or set(folds.tolist())!=set(range(5)): raise ValueError('Exactly five nonempty outer folds 0..4 required')
    for g in np.unique(group):
        if len(np.unique(folds[group==g]))!=1: raise ValueError('A dependency group crosses folds')
    order=np.arange(n)/max(n-1,1) if config.original_trial_order is None else np.asarray(config.original_trial_order,dtype=float)
    if order.shape!=(n,) or not np.isfinite(order).all(): raise ValueError('Explicit original_trial_order must have N finite values')
    return X,y,ids,group,folds,order


def preprocess(X,config):
    modes=('as_is','baseline_subtract_smooth7','baseline_subtract_unsmoothed')
    if config.preprocessing not in modes: raise ValueError(f'preprocessing must be one of {modes}')
    X=X.copy()
    if config.preprocessing!='as_is':
        X-=X[:,:,frame_mask(config.times_seconds,config.baseline_window_seconds)].mean(axis=2,keepdims=True)
        if config.preprocessing=='baseline_subtract_smooth7': X=savgol_filter(X,7,0,axis=-1,mode='interp')
    return X


def run_oof_decoding(epochs,target_labels,config,trial_ids=None,groups=None,outer_fold=None,fold_callback=None):
    X,y,ids,groups,folds,order=validate_inputs(epochs,target_labels,config,trial_ids,groups,outer_fold)
    X=preprocess(X,config); N,P,T=X.shape; R=y.shape[1]; K=len(config.position_labels)
    logits=np.empty((R+1,K,N,T)); probabilities=np.empty_like(logits); filled=np.zeros(N,int); models=[]
    errors=[]
    for outer in range(5):
        tr=np.flatnonzero(folds!=outer); te=np.flatnonzero(folds==outer)
        center,scale,degenerate=fit_baseline_scaler(X[tr],config.times_seconds,config.baseline_window_seconds)
        encoding=fit_rank_glms((X[tr]-center[None,:,None])/scale[None,:,None],y[tr],order[tr],K)
        nested=nested_tg(X,y,order,folds,outer,config)
        pE,pR=derive_priors(nested['TG'],np.asarray(config.times_seconds),config.target_onsets_seconds,config.prior_bin_frames,K=K)
        joint=fit_joint(encoding['B'],pE,pR,config)
        bases=[]; singular=[]
        for template in joint['templates']:
            q,s=row_basis(template); bases.append(q); singular.append(s)
        cuts=np.cumsum([0]+[q.shape[1] for q in bases]); Q=np.concatenate(bases,axis=1)
        A=np.linalg.solve(Q.T@Q+config.coordinate_ridge*np.eye(Q.shape[1]),Q.T).T
        readouts=fit_readouts(X[tr],y[tr],center,scale,A,cuts,config)
        model=dict(center=center,scale=scale,degenerate=degenerate,encoding=encoding,nested=nested,pE=pE,pR=pR,
                   joint=joint,bases=bases,singular_values=singular,cuts=cuts,A=A,readouts=readouts,
                   train_ids=ids[tr],test_ids=ids[te],train_groups=groups[tr],test_groups=groups[te],outer_fold=outer,
                   train_indices=tr,test_indices=te,
                   train_class_counts=np.array([np.bincount(y[tr,r],minlength=K) for r in range(R)]),
                   test_class_counts=np.array([np.bincount(y[te,r],minlength=K) for r in range(R)]))
        for audit in nested['audit']:
            for key in ('train','validation'):
                index=np.asarray(audit[key+'_indices'],dtype=int)
                audit[key+'_ids']=ids[index]; audit[key+'_groups']=groups[index]
        ll,pp,_=predict_fold(model,X[te]); err=float(np.max(np.abs(ll-predict_fold_folded(model,X[te])))); errors.append(err)
        if err>1e-8: raise AssertionError('Coordinate/folded logits mismatch')
        logits[:,:,te,:]=ll; probabilities[:,:,te,:]=pp; filled[te]+=1; models.append(model)
        if fold_callback: fold_callback(outer,model)
    assert np.all(filled==1)
    from .evaluation import evaluate
    result=dict(logits=logits,probabilities=probabilities,predicted_location=probabilities.argmax(axis=1)+1,
                trial_ids=ids,group_ids=groups,outer_fold=folds,target_labels=y+1,times_seconds=np.asarray(config.times_seconds),
                class_labels=np.asarray(config.position_labels),models=models,filled=filled,folded_max_errors=np.array(errors))
    result.update(evaluate(probabilities,y+1,config))
    return result
