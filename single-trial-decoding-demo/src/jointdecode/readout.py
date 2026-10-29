"""Fixed weighted multinomial readouts, with label-free per-frame inference."""
import warnings
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .config import frame_mask


def fit_readouts(X,y,center,scale,A,cuts,config):
    R=y.shape[1]; K=len(config.position_labels); N=len(y); result=[]
    counts=np.array([np.bincount(y[:,r],minlength=K) for r in range(R)])
    for j in range(R+1):
        ranks=range(R) if j==0 else [j-1]
        windows=config.entry_readout_windows_seconds if j==0 else config.rank_readout_windows_seconds
        feats=[]; labels=[]; weights=[]
        for r in ranks:
            mean=X[:,:,frame_mask(config.times_seconds,windows[r])].mean(axis=2)
            coords=((mean-center)/scale)@A[:,cuts[j]:cuts[j+1]]
            feats.append(coords); labels.append(y[:,r])
            weights.append(1/((R if j==0 else 1)*K*counts[r,y[:,r]]))
        feat=np.concatenate(feats); label=np.concatenate(labels); weight=np.concatenate(weights)*len(label)
        if feat.shape[1]==0:
            V=np.zeros((K,0)); bias=np.zeros(K); diag={'zero_dimensional':True,'converged':True}
        else:
            scaler=StandardScaler().fit(feat,sample_weight=weight)
            # sklearn >= 1.8 removed multi_class; lbfgs with K>=3 is multinomial.
            kwargs=dict(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,fit_intercept=True)
            import inspect
            if 'multi_class' in inspect.signature(LogisticRegression).parameters:
                kwargs['multi_class']='multinomial'
            clf=LogisticRegression(**kwargs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                clf.fit(scaler.transform(feat),label,sample_weight=weight)
            if not np.array_equal(clf.classes_,np.arange(K)): raise ValueError('Missing readout classes')
            V=clf.coef_/scaler.scale_[None,:]; bias=clf.intercept_-V@scaler.mean_
            diag={'zero_dimensional':False,'converged':not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                  'warnings':[str(w.message) for w in caught],'n_iter':clf.n_iter_.tolist(),'estimator_params':clf.get_params(),
                  'latent_mean':scaler.mean_,'latent_scale':scaler.scale_}
        Wstd=V@A[:,cuts[j]:cuts[j+1]].T
        Winput=Wstd/scale[None,:]; binput=bias-Winput@center
        result.append(dict(V=V,bias=bias,Winput=Winput,binput=binput,diagnostics=diag))
    return result


def predict_fold(model,X_test):
    """No label argument, conditional residuals, time-varying gain, or trial averaging."""
    Z=(np.asarray(X_test,dtype=float)-model['center'][None,:,None])/model['scale'][None,:,None]
    coords=np.einsum('npt,pd->ndt',Z,model['A'],optimize=True)
    logits=[]
    for j,readout in enumerate(model['readouts']):
        block=coords[:,model['cuts'][j]:model['cuts'][j+1],:]
        logits.append(np.einsum('kd,ndt->knt',readout['V'],block,optimize=True)+readout['bias'][:,None,None])
    logits=np.array(logits); probabilities=softmax(logits,axis=1)
    return logits,probabilities,np.argmax(probabilities,axis=1)+1


def predict_fold_folded(model,X_test):
    return np.array([np.einsum('kp,npt->knt',r['Winput'],X_test,optimize=True)+r['binput'][:,None,None]
                     for r in model['readouts']])
