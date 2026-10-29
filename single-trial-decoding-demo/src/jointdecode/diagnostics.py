"""Evaluation-only source attribution; never imported by model fitting.

All geometries use the fold's training-scaled ROI coordinates. Absent-source
alignment is a separate diagnostic, never successful source recovery.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles
from .joint import row_basis
from .config import frame_mask


def load_tree(path):
    """Read numeric NPZ + JSON model without pickle."""
    path=Path(path)
    with np.load(str(path)+'.npz',allow_pickle=False) as arrays:
        def decode(x):
            if isinstance(x,dict):
                if '__array__' in x: return arrays[x['__array__']].copy()
                if '__list__' in x: return x['__list__']
                return {k:decode(v) for k,v in x.items()}
            if isinstance(x,list): return [decode(v) for v in x]
            return x
        return decode(json.loads(Path(str(path)+'.json').read_text()))


def source_activity(ground_truth,floor=1e-12):
    gains=[np.asarray(ground_truth['entry_gain'])]+list(np.asarray(ground_truth['rank_gain']))
    return [dict(source=j,source_active=bool(np.linalg.norm(g)>floor),
                 gain_norm=float(np.linalg.norm(g)),unused_basis_available=bool(j<len(ground_truth['templates'])))
            for j,g in enumerate(gains)]


def geometry(estimated,truth,floor=1e-12):
    """General-dimensional row-template geometry; zero/rank mismatch explicit.

Projector distance is ||Pe-Pt||F/sqrt(rank_e+rank_t), including unmatched
ranks. Angles cover only min(rank_e,rank_t) directions, not missing ones.
"""
    e=np.asarray(estimated,float); t=np.asarray(truth,float)
    e=e-e.mean(axis=0,keepdims=True); t=t-t.mean(axis=0,keepdims=True)
    qe,_=row_basis(e); qt,_=row_basis(t); de=qe.shape[1]; dt=qt.shape[1]
    ne=float(np.linalg.norm(e)); nt=float(np.linalg.norm(t))
    capture_t=float(np.linalg.norm(t@qe)**2/nt**2) if nt>floor else None
    capture_e=float(np.linalg.norm(e@qt)**2/ne**2) if ne>floor else None
    cosine=float(np.sum(e*t)/(ne*nt)) if ne>floor and nt>floor and e.shape==t.shape else None
    d=float(np.linalg.norm(qe@qe.T-qt@qt.T)/np.sqrt(de+dt)) if de+dt else 0.
    return dict(estimated_rank=de,truth_rank=dt,status='collapsed' if de==0 else ('zero_truth' if dt==0 else 'available'),
                signed_template_cosine=cosine,estimated_span_truth_energy_capture=capture_t,
                truth_span_estimated_energy_capture=capture_e,projector_distance=d,
                principal_angles_degrees=np.rad2deg(subspace_angles(qe,qt)).tolist() if de and dt else [],
                projection_relative_error=float(np.sqrt(max(0.,1.-capture_t))) if capture_t is not None else None,
                angles_cover_matched_directions_only=bool(de!=dt))


def template_attribution(model,ground_truth):
    activity=source_activity(ground_truth); records=[]
    for i,e in enumerate(model['joint']['templates']):
        for j,t in enumerate(ground_truth['templates']):
            g=geometry(e,t/np.asarray(model['scale'])[None,:]); active=activity[j]['source_active']
            record=dict(fold=int(model['outer_fold']),learned_space=i,truth_source=j,source_active=active,
                        target_present=active,gain_norm=activity[j]['gain_norm'],**g)
            record['unused_basis_alignment']=None
            if not active:
                record['unused_basis_alignment']=g
                for key in ('signed_template_cosine','estimated_span_truth_energy_capture','truth_span_estimated_energy_capture',
                            'projector_distance','principal_angles_degrees','projection_relative_error'):
                    record[key]=None
                record['status']='absent'
            records.append(record)
    return records


def window_cross_target_scores(probabilities,labels,times,windows,method='joint_soft',fold='pooled'):
    """Framewise class-balanced BA, then window average (not mean-prob argmax).

labels use public 1..K. prediction_matches_target is unbalanced framewise
accuracy, separately named from BA. No fitted quantity is changed here.
"""
    p=np.asarray(probabilities); y=np.asarray(labels)-1
    J,K,N,T=p.shape
    if y.shape[0]!=N or len(times)!=T or not np.isin(y,np.arange(K)).all(): raise ValueError('Incompatible labels/times')
    pred=p.argmax(axis=1); rows=[]
    for wid,win in windows.items():
        mask=frame_mask(times,win)
        if not mask.any(): raise ValueError('Empty diagnostic window')
        for j in range(J):
            for r in range(y.shape[1]):
                class_scores=[(pred[j,y[:,r]==k][:,mask]==k).mean() for k in range(K) if np.any(y[:,r]==k)]
                if len(class_scores)!=K: raise ValueError('BA requires all scoring classes')
                rows.append(dict(method=method,fold=fold,decoder=j,window_id=wid,window_start=float(win[0]),window_end=float(win[1]),
                    scoring_target=r+1,balanced_accuracy=float(np.mean(class_scores)),
                    mean_true_probability=float(p[j,y[:,r],np.arange(N)][:,mask].mean()),
                    mean_max_probability=float(p[j].max(axis=0)[:,mask].mean()),
                    prediction_matches_target=float((pred[j][:,mask]==y[:,r,None]).mean()),
                    prediction_matches_no_target=float((~np.any(pred[j,:,None,:]==y[:,:,None],axis=1))[:,mask].mean())))
    return rows
