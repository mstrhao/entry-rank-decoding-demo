from dataclasses import replace
import inspect
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from jointdecode import Config,simulate,run_oof_decoding,predict_fold
from jointdecode.pipeline import validate_inputs,preprocess
from jointdecode.readout import predict_fold_folded


def small_config(ranks=2):
    return replace(Config(),sfreq=10.,times_seconds=tuple(np.arange(60)/10),target_onsets_seconds=(1.,2.,3.)[:ranks],
                   entry_readout_windows_seconds=((1.2,1.6),(2.2,2.6),(3.2,3.6))[:ranks],rank_readout_windows_seconds=((4.5,5.5),)*ranks,
                   position_labels=tuple(range(1,ranks+2)),preprocessing='baseline_subtract_unsmoothed')

@pytest.mark.parametrize('ranks',[2,3])
def test_end_to_end(ranks):
    cfg=small_config(ranks); d=simulate(cfg,neurons=(ranks+1)*ranks)
    with threadpool_limits(1):
        result=run_oof_decoding(d['epochs'],d['target_labels'],cfg,trial_ids=d['trial_ids'],groups=d['groups'],outer_fold=d['outer_fold'])
    P=result['probabilities']; assert P.shape==(ranks+1,ranks+1,len(d['epochs']),60)
    assert np.isfinite(P).all() and np.all(P>=0); np.testing.assert_allclose(P.sum(axis=1),1,atol=1e-14)
    assert np.all(result['filled']==1)
    assert 'target_labels' not in inspect.signature(predict_fold).parameters
    X=preprocess(d['epochs'],cfg)
    for model in result['models']:
        tr=model['train_ids']; te=model['test_ids']; assert not set(tr)&set(te)
        for a in model['nested']['audit']:
            assert set(a['train_ids'])<=set(tr) and set(a['validation_ids'])<=set(tr)
            assert not set(a['train_groups'])&set(a['validation_groups'])
        before=predict_fold(model,X[te]); shuffled=d['target_labels'][te].copy(); np.random.default_rng(9).shuffle(shuffled)
        after=predict_fold(model,X[te]); assert np.array_equal(before[0],after[0]); assert np.array_equal(before[1],after[1])
        np.testing.assert_allclose(before[0],predict_fold_folded(model,X[te]),atol=1e-8)
        np.testing.assert_allclose(model['joint']['templates'].mean(axis=1),0,atol=1e-12)
        for q in model['bases']:
            assert q.shape[1]<=len(cfg.position_labels)-1
            np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-12)


def test_invalid_inputs_and_group_leakage():
    cfg=small_config(); d=simulate(cfg,neurons=6); labels=d['target_labels'].astype(float); labels[0,0]=np.nan
    with pytest.raises(ValueError): validate_inputs(d['epochs'],labels,cfg,None,None,d['outer_fold'])
    groups=np.zeros(len(labels),int)
    with pytest.raises(ValueError): validate_inputs(d['epochs'],d['target_labels'],cfg,None,groups,d['outer_fold'])


def test_simulation_reproducible():
    cfg=small_config(); a=simulate(cfg,neurons=6); b=simulate(cfg,neurons=6)
    assert np.array_equal(a['epochs'],b['epochs']); assert np.array_equal(a['outer_fold'],b['outer_fold'])
    for combo in np.unique(a['target_labels'],axis=0):
        idx=np.all(a['target_labels']==combo,axis=1); assert set(a['outer_fold'][idx])==set(range(5))
