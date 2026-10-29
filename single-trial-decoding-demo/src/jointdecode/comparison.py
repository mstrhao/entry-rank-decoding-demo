"""Matched-readout hard templates. No ground truth or held-out labels in fit.

This is NOT historical cosine-toy reproduction: all templates have up to K-1
rows of signal, and fixed multinomial readout matches joint-soft exactly.
"""
import json
from pathlib import Path
import numpy as np
from .config import frame_mask
from .joint import row_basis
from .readout import fit_readouts

METHOD='hard_full_span_matched_readout_v1'


def load_tree(path):
    path=Path(path)
    with np.load(str(path)+'.npz',allow_pickle=False) as z:
        def decode(obj):
            if isinstance(obj,dict):
                if '__array__' in obj: return z[obj['__array__']].copy()
                if '__list__' in obj: return obj['__list__']
                return {k:decode(v) for k,v in obj.items()}
            if isinstance(obj,list): return [decode(x) for x in obj]
            return obj
        return decode(json.loads(Path(str(path)+'.json').read_text()))


def hard_templates(B,config):
    """Only outer-training GLM coefficients and predefined windows are used."""
    R,K,P,T=B.shape
    if R!=len(config.target_onsets_seconds): raise ValueError('Rank count mismatch')
    E=np.mean([B[r][:,:,frame_mask(config.times_seconds,w)].mean(axis=-1)
               for r,w in enumerate(config.entry_readout_windows_seconds)],axis=0)
    E=E-E.mean(axis=0,keepdims=True)
    q,_=row_basis(E)
    Ds=np.array([B[r][:,:,frame_mask(config.times_seconds,w)].mean(axis=-1)
                 for r,w in enumerate(config.rank_readout_windows_seconds)])
    residual=Ds-(Ds@q)@q.T
    templates=np.concatenate([E[None],residual],axis=0)
    return templates-templates.mean(axis=1,keepdims=True)


def fit_hard_from_training(B,X_train,y_train,center,scale,config):
    """Fit accepts only training rows, not any truth/test labels/projector."""
    templates=hard_templates(B,config)
    bases=[];singular=[]
    for s in templates:
        q,v=row_basis(s);bases.append(q);singular.append(v)
    cuts=np.cumsum([0]+[q.shape[1] for q in bases]); Q=np.concatenate(bases,axis=1)
    A=np.linalg.solve(Q.T@Q+config.coordinate_ridge*np.eye(Q.shape[1]),Q.T).T
    readouts=fit_readouts(X_train,y_train,center,scale,A,cuts,config)
    return dict(center=center.copy(),scale=scale.copy(),joint={'templates':templates},bases=bases,
                singular_values=singular,cuts=cuts,A=A,readouts=readouts,method=METHOD)


def oracle_residual_targets(truth,scale,config):
    """Evaluation only: actual processed gains and active Entry geometry.

For each GLM target r, D_true,r is its full Entry+Rank coefficient averaged
in r's fixed Rank window, in the fold's standardized coordinates.
"""
    from scipy.signal import savgol_filter
    E=truth['templates'][0]/scale
    gains=np.concatenate([truth['entry_gain'],truth['rank_gain']]).copy()
    if config.preprocessing!='as_is':
        gains-=gains[:,frame_mask(config.times_seconds,config.baseline_window_seconds)].mean(axis=1,keepdims=True)
        if config.preprocessing=='baseline_subtract_smooth7': gains=savgol_filter(gains,7,0,axis=-1,mode='interp')
    R=len(config.target_onsets_seconds)
    q,_=row_basis(E) if np.any(np.abs(truth['entry_gain'])>config.floor) else (np.zeros((len(scale),0)),None)
    rows=[]
    for r,w in enumerate(config.rank_readout_windows_seconds):
        mask=frame_mask(config.times_seconds,w)
        D=gains[r,mask].mean()*E+gains[R+r,mask].mean()*truth['templates'][r+1]/scale
        residual=D-(D@q)@q.T
        rows.append((D,residual))
    return rows
