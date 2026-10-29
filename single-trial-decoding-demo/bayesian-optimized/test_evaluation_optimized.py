import numpy as np
from types import SimpleNamespace
from evaluate_optimized import activity_masks,coverage_risk,class_metrics


def test_activity_gray_mixed_unique():
    # Unique strong, mixed strong, gray, wholly inactive, then other active.
    entry=np.array([[1,1,.3,0,0],[0,1,0,0,1],[0,0,0,0,0]],float)
    rank=np.array([[1,.3,0,0,0],[0,0,1,.3,0],[0,0,0,0,1]],float)
    cfg=SimpleNamespace(preprocessing='as_is')
    state,target,norm=activity_masks(dict(entry_gain=entry,rank_gain=rank),cfg,.5,.1)
    assert state[0].tolist()==[1,-1,-1,0,1]
    assert target[0].tolist()==[0,-1,-1,-1,1]
    assert state[1].tolist()==[1,-1,0,0,0]
    assert np.all(np.isfinite(norm))


def test_inactive_emission_is_risk_even_if_location_matches_placeholder():
    confidence=np.array([.99,.95,.4]);prediction=np.array([0,1,2]);truth=np.array([0,-1,2]);supported=np.ones(3,bool)
    r=coverage_risk(confidence,prediction,truth,supported,[.9])[0]
    assert r['n_emitted']==2 and r['coverage']==2/3 and r['risk']==.5
    r=coverage_risk(confidence,prediction,truth,supported,[.99])[0]
    assert r['n_emitted']==0 and r['risk'] is None # strict >


def test_normalized_underflow_nll_and_circular_error():
    from scipy.special import log_softmax
    logits=np.zeros((6,6));logits[np.arange(6),np.arange(6)]=-1000
    lp=log_softmax(logits,axis=0);s=class_metrics(lp,np.arange(6))
    assert np.allclose(np.exp(lp).sum(0),1)
    assert s['nll']>1000 and np.isfinite(s['nll'])
    lp=np.full((6,6),-1000.);lp[(np.arange(6)+1)%6,np.arange(6)]=0
    s=class_metrics(log_softmax(lp,axis=0),np.arange(6))
    assert s['circular_error_steps']==1 and s['circular_error_degrees']==60


def test_exact_coverage_ties_and_cumulative_counts():
    c=np.array([.9,.9,.8]);p=np.array([0,1,2]);y=np.array([0,-1,2]);support=np.ones(3,bool)
    curve=coverage_risk(c,p,y,support)
    at={r['threshold']:r for r in curve}
    assert at[.8]['n_emitted']==2 and at[.8]['risk']==.5
    assert at[.9]['n_emitted']==0 and at[.9]['risk'] is None
    assert at[0.]['n_emitted']==3 and at[0.]['risk']==1/3
    rng=np.random.default_rng(7);c=rng.random(3000);p=rng.integers(0,6,3000);y=rng.integers(-1,6,3000)
    curve=coverage_risk(c,p,y,np.ones(3000,bool))
    assert len(curve)<=250
    for row in curve:
        m=c>row['threshold'];assert row['n_emitted']==m.sum()
        if m.any():assert np.isclose(row['risk'],((y[m]<0)|(p[m]!=y[m])).mean())
