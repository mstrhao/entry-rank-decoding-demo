import inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from scipy.special import logsumexp
from gaussian_readout import fit_gaussian,predict_logposterior,fit_bayesian_readouts,predict_bayesian_fold,training_rows


def test_manual_bayes_prior_and_normalization():
    x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([0,0,1,1]);m=fit_gaussian(x,y,2)
    test=np.array([[-10.],[0.],[1e5]])
    lp=predict_logposterior(m,test,prior=[.25,.75]);z=(test-m['latent_mean'])/m['latent_scale']
    raw=-.5*(z[:,None,:]-m['class_means'])[:,:,0]**2/m['covariance'][0,0]+np.log([.25,.75])
    np.testing.assert_allclose(lp,raw-logsumexp(raw,axis=-1,keepdims=True))
    np.testing.assert_allclose(np.exp(lp).sum(axis=1),1,atol=1e-12)
    np.testing.assert_allclose(np.exp(lp[1]),[.25,.75])
    assert np.isfinite(lp).all() and np.linalg.eigvalsh(m['covariance']).min()>0


def test_zero_dimensional_and_constant_covariance():
    y=np.tile(np.arange(3),3)
    m=fit_gaussian(np.empty((9,0)),y,3,latent_mean=np.empty(0),latent_scale=np.empty(0))
    np.testing.assert_allclose(np.exp(predict_logposterior(m,np.empty((2,0)),[.2,.3,.5])),[[.2,.3,.5]]*2)
    m=fit_gaussian(np.ones((9,2)),y,3)
    assert m['n_floored_eigenvalues']==2
    np.testing.assert_allclose(np.exp(predict_logposterior(m,np.ones((2,2)))),1/3)


def test_weights_and_prior_validation():
    x=np.arange(12).reshape(6,2);y=np.arange(6)%3
    with pytest.raises(ValueError,match='Nonuniform'):fit_gaussian(x,y,3,weights=[1,1,2,1,1,1])
    m=fit_gaussian(x,y,3)
    for prior in ([0,.5,.5],[1,1,1],[.2,.8]):
        with pytest.raises(ValueError):predict_logposterior(m,x,prior)
    with pytest.raises(ValueError,match='Every class'):fit_gaussian(x,np.zeros(6,int),3)


def test_prediction_has_no_labels_and_fit_uses_only_supplied_rows():
    rng=np.random.default_rng(1);train=rng.normal(size=(60,3));labels=np.arange(60)%3
    m=fit_gaussian(train,labels,3);before=predict_logposterior(m,train[:4]);unrelated_labels=np.arange(40)%3
    rng.shuffle(unrelated_labels)
    np.testing.assert_array_equal(before,predict_logposterior(m,train[:4]))
    assert list(inspect.signature(predict_bayesian_fold).parameters)==['upstream','readouts','X_test']
    changed=train.copy();changed[labels==0]+=1
    m2=fit_gaussian(changed,labels,3)
    assert not np.allclose(m['class_means'],m2['class_means'])
