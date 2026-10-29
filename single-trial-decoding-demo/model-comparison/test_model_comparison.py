import numpy as np
import pytest
from compare_models import reconstruction_scores,pooled_reconstruction,circular_error


def test_exact_projection_reference_and_pooling():
    rng=np.random.default_rng(4);train=rng.normal(size=(10,4,5));test=rng.normal(size=(7,4,5));bases=[np.eye(4)[:,j:j+1] for j in range(4)]
    r=reconstruction_scores(train,test,np.eye(4),bases);mu=train.mean((0,2));W=np.moveaxis(test,1,-1)-mu
    np.testing.assert_allclose(r['training_reference_mean'],mu)
    assert r['sse']['Joint ridge']==0 and r['union_dimension']==4
    for j in range(4):np.testing.assert_allclose(r['sse'][f'Drop {j}'],np.sum(W[...,j]**2))
    pp=pooled_reconstruction([r,r]);assert pp['r2']['Joint ridge']==1
    assert sum(pp['drop_block_delta_r2'].values())==pytest.approx(1)


def test_negative_retained_and_train_reference_no_test_influence():
    rng=np.random.default_rng(2);train=rng.normal(size=(9,4,2));test=rng.normal(size=(7,4,2));bases=[np.eye(4)[:,j:j+1] for j in range(4)]
    a=reconstruction_scores(train,test,np.eye(4)*4,bases);b=reconstruction_scores(train,test+100,np.eye(4)*4,bases)
    np.testing.assert_array_equal(a['training_reference_mean'],b['training_reference_mean'])
    pp=pooled_reconstruction([a]);assert pp['r2']['Joint ridge']<0
    assert pp['r2']['Orthogonal union']==pytest.approx(1)


def test_pooled_sse_not_mean_fold_ratios():
    names=['Joint ridge','Orthogonal union']+[f'{kind} {j}' for kind in ['Only','Drop'] for j in range(4)]
    a={'sst':1.,'sse':dict.fromkeys(names,.5)};b={'sst':9.,'sse':dict.fromkeys(names,0.)}
    assert pooled_reconstruction([a,b])['r2']['Joint ridge']==pytest.approx(.95)


def test_circular_wrap_and_opposite():
    np.testing.assert_array_equal(circular_error([6,2,4,1],[1,1,1,1]),[60,60,180,0])


def test_overlapping_blocks_not_additive_variance():
    rng=np.random.default_rng(5);train=rng.normal(size=(10,4,3));test=rng.normal(size=(8,4,3));I=np.eye(4)
    bases=[I[:,:1],I[:,:1],I[:,1:2],I[:,2:3]];Q=np.concatenate(bases,axis=1)
    A=np.linalg.solve(Q.T@Q+.25*np.eye(4),Q.T).T
    r=pooled_reconstruction([reconstruction_scores(train,test,A,bases)])
    assert not np.isclose(sum(r['drop_block_delta_r2'].values()),r['r2']['Joint ridge'])
    assert not np.isclose(sum(r['r2'][f'Only {j}'] for j in range(4)),r['r2']['Joint ridge'])
