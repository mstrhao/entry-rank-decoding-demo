import numpy as np
from evaluate_comparison import score_window, probability_difference, paired_bootstrap


def test_uniform_scores():
    y=np.tile(np.arange(1,7),4);p=np.full((6,len(y),3),1/6)
    s,t,b=score_window(p,np.log(p),y)
    assert np.isclose(s['balanced_accuracy'],1/6)
    assert np.isclose(s['accuracy'],1/6)
    assert np.isclose(s['nll'],np.log(6))
    assert np.isclose(s['brier'],5/6)
    assert np.isclose(s['entropy'],np.log(6))
    assert np.isclose(s['ece'],0)
    assert sum(x['count'] for x in b)==len(y)*3


def test_perfect_and_log_domain():
    y=np.arange(1,7);p=np.eye(6)[:,:,None]
    with np.errstate(divide='ignore'):lp=np.log(p)
    s,_,_=score_window(p,lp,y)
    assert all(np.isclose(s[k],0) for k in ['nll','brier','entropy','ece'])
    assert s['balanced_accuracy']==1
    p=np.full((6,6,1),1/6);lp=np.log(p);lp[0,0,0]=-1000
    s,_,_=score_window(p,lp,y)
    assert np.isclose(s['nll'],(1000+5*np.log(6))/6)


def test_identical_paired_metrics_and_bootstrap():
    y=np.tile(np.arange(1,7),20);p=np.full((6,len(y),2),1/6)
    _,t,_=score_window(p,np.log(p),y)
    tv,js=probability_difference(p,p)
    assert np.all(tv==0) and np.all(js==0)
    draws=np.random.default_rng(2026).integers(0,len(y),(100,len(y)))
    b=paired_bootstrap(t,t,y,draws,6)
    for k in ['accuracy','balanced_accuracy','nll','brier']:
        assert b[k]['lower']==b[k]['upper']==0
