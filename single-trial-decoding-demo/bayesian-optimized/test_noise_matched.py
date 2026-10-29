import inspect
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from noise_matched import gaussian_loglikelihood,posterior_from_features,predict_frame_and_null,crossfit_thresholds,fit_frame_and_null,frame_training_rows
from gaussian_readout import fit_gaussian
from jointdecode.config import Config,frame_mask


def fitted_example(d=2):
    rng=np.random.default_rng(12);y=np.repeat(np.arange(6),30)
    means=np.c_[3*np.cos(2*np.pi*np.arange(6)/6),3*np.sin(2*np.pi*np.arange(6)/6)][:,:d]
    x=means[y]+rng.normal(0,.2,(len(y),d));neg=rng.normal(0,.15,(180,d))
    return dict(content=fit_gaussian(x,y,6,latent_mean=np.zeros(d),latent_scale=np.ones(d)),null=fit_gaussian(neg,np.zeros(180,int),1,latent_mean=np.zeros(d),latent_scale=np.ones(d)),prior_content=.5)


def test_full_gaussian_and_joint_normalization():
    m=fitted_example();x=np.array([[0.,0.],[3.,0.]])
    ll=gaussian_loglikelihood(m['content'],x)
    manual=np.array([multivariate_normal.logpdf(x,mean=mu,cov=m['content']['covariance']) for mu in m['content']['class_means']]).T
    np.testing.assert_allclose(ll,manual,atol=1e-12)
    c,j,p=posterior_from_features(m,x)
    np.testing.assert_allclose(np.exp(j).sum(-1),1);np.testing.assert_allclose(p,np.exp(j[:,1:]).sum(-1))
    np.testing.assert_allclose(j[:,1:]-logsumexp(j[:,1:],axis=-1,keepdims=True),c,atol=1e-12)
    assert np.exp(j[0,0])>.99 and p[1]>.99


def test_zero_dimensional_is_prior():
    m=fitted_example(d=0);c,j,p=posterior_from_features(m,np.empty((3,0)))
    np.testing.assert_allclose(np.exp(c),1/6);np.testing.assert_allclose(p,.5)
    np.testing.assert_allclose(np.exp(j[:,0]),.5);np.testing.assert_allclose(np.exp(j[:,1:]),.5/6)
    assert list(inspect.signature(predict_frame_and_null).parameters)==['upstream','fitted','X_test']


def synthetic_design():
    rng=np.random.default_rng(9);y=np.tile(np.arange(6),5);y=np.stack([y,(y+1)%6,(y+2)%6],axis=1);folds=np.repeat(np.arange(5),6)
    X=rng.normal(size=(30,4,180));u=dict(center=np.zeros(4),scale=np.ones(4),A=np.eye(4),cuts=np.arange(5),
       readouts=[dict(diagnostics=dict(latent_mean=np.zeros(1),latent_scale=np.ones(1))) for _ in range(4)])
    return X,y,folds,u,Config()


def test_training_rows_are_frames_and_baseline_support():
    X,y,folds,u,cfg=synthetic_design();tr=folds!=0
    fit=fit_frame_and_null(X[tr],y[tr],u,cfg)
    for j in range(4):
        x,label,w,coords=frame_training_rows(X[tr],y[tr],u,cfg,j)
        assert len(x)==24*(3*12 if j==0 else 30)
        np.testing.assert_allclose(w,1)
        assert fit['null'][j]['training_rows']==24*15
        assert fit['content'][j]['training_rows']==len(x)
    assert cfg.baseline_window_seconds[1]+3/cfg.sfreq<cfg.target_onsets_seconds[0]


def test_crossfit_whole_trials_and_outer_test_isolation():
    X,y,folds,u,cfg=synthetic_design();t,diag,audit=crossfit_thresholds(X,y,folds,0,u,cfg)
    changed=X.copy();changed[folds==0]=1e6;newlabels=y.copy();newlabels[folds==0]=5-y[folds==0]
    tt,_,_=crossfit_thresholds(changed,newlabels,folds,0,u,cfg);np.testing.assert_array_equal(t,tt)
    for a in audit:
        assert not set(a['train_indices'])&set(a['validation_indices'])
        assert np.all(folds[a['train_indices']]!=0) and np.all(folds[a['validation_indices']]!=0)
    assert all(d['crossfit_baseline_fpr']<=.05 for d in diag)
