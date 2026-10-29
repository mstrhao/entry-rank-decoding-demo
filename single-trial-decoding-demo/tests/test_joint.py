from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from jointdecode.joint import (profile_objective, normalize_profiles, recurrence_pairs,
                              derive_priors, row_basis, simultaneous_transform, fit_joint)


def setup_problem(R):
    rng = np.random.default_rng(891)
    K, P, T = 6, 9, 14
    B = rng.normal(size=(R,K,P,T))
    templates = rng.normal(size=(R+1,K,P))
    priors = rng.uniform(.01, .5, size=(2*R,T))
    x = rng.uniform(.01,.5,size=priors.shape)
    onsets = np.arange(R)*.2
    cfg = SimpleNamespace(target_onsets_seconds=onsets, sfreq=10.)
    return B, templates, priors, x, onsets, cfg


@pytest.mark.parametrize('R',[2,3])
def test_analytic_gradient(R):
    B,S,p,x,onsets,cfg = setup_problem(R)
    fun = lambda xx: profile_objective(xx,B,S,p,onsets,10.,cfg)
    val,g = fun(x)
    numerical = approx_derivative(lambda xx: fun(xx)[0],x.ravel(),method='3-point').ravel()
    np.testing.assert_allclose(g,numerical,atol=1e-8,rtol=1e-6)
    # Independent direct, uncontracted objective checks every denominator.
    a,b=x[:R],x[R:]
    resid=a[:,None,None,:]*S[0,:,:,None]+b[:,None,None,:]*S[1:,:,:,None]-B
    energy=np.r_[np.repeat(np.mean(S[0]**2),R),np.mean(S[1:]**2,axis=(1,2))]
    direct=np.mean(resid**2)+np.mean(energy[:,None]*(x-p)**2)
    direct+=2*np.mean(energy[:,None]*np.diff(x,n=2,axis=1)**2)
    pairs=recurrence_pairs(p,onsets,10.)
    assert len(pairs)==R*(R-1)//2
    direct+=energy[0]*sum(np.sum(w*(a[r,i]-a[s,j])**2) for r,s,i,j,w in pairs)/len(pairs)
    np.testing.assert_allclose(val,direct,atol=1e-12,rtol=0)


@pytest.mark.parametrize('R',[2,3])
def test_normalization_invariant(R):
    B,S,p,x,onsets,cfg=setup_problem(R)
    reconstruct=lambda x,S:x[:R,None,None,:]*S[0,:,:,None]+x[R:,None,None,:]*S[1:,:,:,None]
    xn,Sn=normalize_profiles(x,S)
    np.testing.assert_allclose(reconstruct(x,S),reconstruct(xn,Sn),atol=1e-14)
    np.testing.assert_allclose(np.sum(xn[:R]**2)/R,1)
    np.testing.assert_allclose(np.linalg.norm(xn[R:],axis=1),1)
    # Entry relative amplitudes must survive public normalization.
    np.testing.assert_allclose(np.linalg.norm(xn[0])/np.linalg.norm(xn[1]),
                               np.linalg.norm(x[0])/np.linalg.norm(x[1]))


def test_priors_use_third_pair_and_reject_missing():
    R,T=3,30
    rng=np.random.default_rng(17)
    tg=rng.uniform(.2,.8,(R,R,T//2,T//2))
    times=np.arange(T)/10
    onsets=np.array([.2,.6,1.])
    pe,pr=derive_priors(tg,times,onsets)
    np.testing.assert_allclose(np.linalg.norm(pe,axis=1),1)
    np.testing.assert_allclose(np.linalg.norm(pr,axis=1),1)
    changed=tg.copy();changed[1,2]=1/6;changed[2,1]=1/6
    pe2,_=derive_priors(changed,times,onsets)
    assert not np.allclose(pe[1:],pe2[1:])
    with pytest.raises(ValueError,match='cannot be estimated'):
        derive_priors(np.full_like(tg,1/6),times,onsets)
    with pytest.raises(ValueError,match='Noninteger'):
        recurrence_pairs(np.r_[pe,pr],[.2,.65,1.],10.)


def test_geometry_and_finite_ridge():
    rng=np.random.default_rng(7)
    bases=[row_basis(rng.normal(size=(6,25)))[0] for _ in range(4)]
    for q in bases:
        assert q.shape[1]<=5
        np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-14)
    A,cuts=simultaneous_transform(bases)
    Q=np.concatenate(bases,axis=1)
    np.testing.assert_allclose(A@(Q.T@Q+.25*np.eye(20)),Q,atol=1e-14)
    assert cuts.tolist()==[0,5,10,15,20]
    zero,_=row_basis(np.zeros((6,25)))
    assert zero.shape==(25,0)


@pytest.mark.parametrize('R',[2,3])
def test_shared_implementation_smoke(R):
    B,S,p,x,onsets,cfg=setup_problem(R)
    cfg.max_iter=3;cfg.profile_max_iter=30
    out=fit_joint(B,p[:R],p[R:],cfg)
    assert out['templates'].shape==(R+1,6,9)
    assert out['profiles'].shape==(2*R,14)
    assert np.all(out['profiles']>=0)
    np.testing.assert_allclose(out['templates'].mean(axis=1),0,atol=1e-14)
    assert len(out['optimizer_records'])==len(out['objective_history'])
    assert all('success' in record for record in out['optimizer_records'])
