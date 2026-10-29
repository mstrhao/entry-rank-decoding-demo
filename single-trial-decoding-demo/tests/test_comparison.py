"""No performance-threshold tuning: compare estimands, interfaces and isolation."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from jointdecode.config import Config
from jointdecode.comparison import hard_templates,fit_hard_from_training,oracle_residual_targets
from jointdecode.readout import predict_fold,predict_fold_folded
from jointdecode.simulation import simulate
from jointdecode.pipeline import preprocess,validate_inputs
from jointdecode.encoding import fit_baseline_scaler,fit_rank_glms
from jointdecode.joint import row_basis

@pytest.mark.parametrize('R',[2,3])
def test_shared_training_and_label_free_prediction(R):
    cfg=Config();cfg=replace(cfg,target_onsets_seconds=cfg.target_onsets_seconds[:R],entry_readout_windows_seconds=cfg.entry_readout_windows_seconds[:R],rank_readout_windows_seconds=cfg.rank_readout_windows_seconds[:R])
    d=simulate(cfg);X,y,ids,groups,f,order=validate_inputs(d['epochs'],d['target_labels'],cfg,d['trial_ids'],d['groups'],d['outer_fold'])
    X=preprocess(X,cfg);tr=f!=0;te=f==0
    c,s,_=fit_baseline_scaler(X[tr],cfg.times_seconds,cfg.baseline_window_seconds)
    B=fit_rank_glms((X[tr]-c[None,:,None])/s[None,:,None],y[tr],order[tr],6)['B']
    m=fit_hard_from_training(B,X[tr],y[tr],c,s,cfg)
    ll,p,_=predict_fold(m,X[te]);permuted=y[te][::-1].copy()
    np.testing.assert_array_equal(p,predict_fold(m,X[te])[1]);np.testing.assert_allclose(ll,predict_fold_folded(m,X[te]),atol=1e-8)
    assert list(inspect.signature(predict_fold).parameters)==['model','X_test']
    assert 'truth' not in inspect.signature(fit_hard_from_training).parameters
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    q=m['bases'][0]
    for template in m['joint']['templates'][1:]:np.testing.assert_allclose(template@q,0,atol=1e-11)
    assert all(q.shape[1]<=5 for q in m['bases'])
    assert set(ids[tr]).isdisjoint(ids[te])


def test_general_dimension_not_legacy_cap():
    rng=np.random.default_rng(8);cfg=Config();B=rng.normal(size=(3,6,30,180));T=hard_templates(B,cfg)
    assert row_basis(T[0])[0].shape[1]==5
    assert all(row_basis(x)[0].shape[1]==5 for x in T)


def test_absent_entry_oracle_does_not_remove_unused_basis():
    cfg=Config();d=simulate(cfg,entry_amplitude=0)
    for full,residual in oracle_residual_targets(d['ground_truth'],np.ones(60),cfg):np.testing.assert_array_equal(full,residual)


def test_no_training_mutation():
    cfg=Config();B=np.random.default_rng(7).normal(size=(3,6,20,180));before=B.copy();hard_templates(B,cfg);np.testing.assert_array_equal(B,before)
