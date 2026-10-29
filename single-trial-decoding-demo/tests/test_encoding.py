import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from jointdecode.encoding import fit_baseline_scaler, design_matrix, fit_rank_glms, conditional_residuals
from jointdecode.tg import bin_native, fit_weighted_centroids, centroid_tg_counts, nested_tg


def test_scaler_unique_trials_and_ddof():
    X = np.arange(5*3*8, dtype=float).reshape(5,3,8)
    center, scale, bad = fit_baseline_scaler(X, np.arange(8), [2,5])
    np.testing.assert_allclose(center, X[:,:,2:5].mean(axis=(0,2)))
    np.testing.assert_allclose(scale, X[:,:,2:5].std(axis=(0,2)))
    assert not bad.any()
    _, scale, bad = fit_baseline_scaler(np.ones_like(X), np.arange(8), [2,5])
    assert bad.all() and np.all(scale == 1)


@pytest.mark.parametrize('R', [2,3])
def test_glm_control_and_conditional_residual(R):
    rng = np.random.default_rng(3)
    K=4
    y = np.array(list(itertools.permutations(range(K), R)) * 3)
    order = np.linspace(0,1,len(y))
    D = design_matrix(y,order,K)
    beta = rng.normal(size=(D.shape[1],2,5))
    Z = (D @ beta.reshape(D.shape[1],-1)).reshape(len(y),2,5)
    glm = fit_rank_glms(Z,y,order,K,ridge=0)
    np.testing.assert_allclose(glm['beta'],np.broadcast_to(beta,glm['beta'].shape),atol=1e-12)
    np.testing.assert_allclose(glm['B'].mean(axis=1),0,atol=1e-12)
    residual = conditional_residuals(Z,y,order,glm,K)
    for r in range(R):
        cols = np.arange(1+r*(K-1),1+(r+1)*(K-1))
        expected = (D[:,cols] @ beta[cols].reshape(K-1,-1)).reshape(Z.shape)
        np.testing.assert_allclose(residual[r],expected,atol=1e-12)


def test_confounded_labels_rejected_before_ridge():
    y = np.tile(np.arange(4),10)
    Y = np.column_stack([y,y,y])
    with pytest.raises(ValueError,match='not identifiable'):
        fit_rank_glms(np.random.default_rng(1).normal(size=(40,3,8)),Y,np.linspace(0,1,40),4)


def test_centroid_folded_scores_and_tg_brute_force():
    rng = np.random.default_rng(4)
    R,N,M,P,U,K = 2,17,11,5,4,3
    train = rng.normal(size=(R,N,P,U)); val=rng.normal(size=(R,M,P,U))
    train[:,:,0,0]=0  # exercise replacement variance
    yt=rng.integers(K,size=(N,R)); yv=rng.integers(K,size=(M,R))
    hits,counts=centroid_tg_counts(train,val,yt,yv,K,chunk_size=2)
    for r in range(R):
        weights=N/(K*np.bincount(yt[:,r],minlength=K)[yt[:,r]])
        for u in range(U):
            mean=np.average(train[r,:,:,u],axis=0,weights=weights)
            var=np.average((train[r,:,:,u]-mean)**2,axis=0,weights=weights)
            good=var[var>1e-12]; var[var<=1e-12]=np.median(good) if len(good) else 1
            scale=np.sqrt(var)
            standardized=(train[r,:,:,u]-mean)/scale
            cent=np.stack([np.average(standardized[yt[:,r]==k],axis=0,weights=weights[yt[:,r]==k]) for k in range(K)])
            for s in range(R):
                for v in range(U):
                    scores=((val[s,:,:,v]-mean)/scale) @ cent.T - .5*(cent**2).sum(axis=1)
                    pred=scores.argmax(axis=1)
                    expected=np.array([np.sum((pred==k)&(yv[:,s]==k)) for k in range(K)])
                    np.testing.assert_array_equal(hits[r,s,u,v],expected)
                np.testing.assert_array_equal(counts[r,s],np.bincount(yv[:,s],minlength=K))


def test_nested_tg_outer_test_isolation():
    rng=np.random.default_rng(42)
    K,R=4,3
    tuples=np.array(list(itertools.permutations(range(K),R)))
    Y=np.repeat(tuples,5,axis=0); folds=np.tile(np.arange(5),len(tuples))
    X=rng.normal(size=(len(Y),5,8)); order=np.linspace(0,1,len(Y))
    cfg=SimpleNamespace(position_labels=tuple(range(1,K+1)),times_seconds=np.arange(8),baseline_window_seconds=(0,2),prior_bin_frames=2)
    out=nested_tg(X,Y,order,folds,0,cfg)
    altered=X.copy(); altered[folds==0]+=1e6
    otherY=Y.copy(); otherY[folds==0]=np.roll(Y[folds==0],1,axis=0)
    out2=nested_tg(altered,otherY,order,folds,0,cfg)
    np.testing.assert_array_equal(out['TG'],out2['TG'])
    for audit in out['audit']:
        assert not np.any(folds[audit['train_indices']]==0)
        assert not np.any(folds[audit['validation_indices']]==0)
        assert not set(audit['train_indices']) & set(audit['validation_indices'])
    np.testing.assert_allclose(out['TG'],(out['hits']/out['counts'][:,:,None,None,:]).mean(axis=-1))


def test_bin_native_no_silent_truncation():
    with pytest.raises(ValueError,match='divisible'):
        bin_native(np.ones((2,3,7)))
    np.testing.assert_array_equal(bin_native(np.arange(6)),[.5,2.5,4.5])
