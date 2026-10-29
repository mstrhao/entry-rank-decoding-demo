import numpy as np
from calibrated_readout import fit_temperature,apply_temperature,crossfit_temperatures
from test_noise_matched import synthetic_design


def test_temperature_corrects_synthetic_overconfidence_and_keeps_argmax():
    y=np.tile(np.arange(3),100);pred=y.copy();pred[::3]=(pred[::3]+1)%3
    logits=np.zeros((300,3));logits[np.arange(300),pred]=8
    T,d=fit_temperature(logits,y)
    assert T>1 and d['inner_nll_after']<d['inner_nll_before']
    p=logits.T[None,:,:,None];out=apply_temperature(p,[T])
    np.testing.assert_array_equal(p.argmax(1),out.argmax(1));np.testing.assert_allclose(np.exp(out).sum(1),1)


def test_temperature_outer_test_data_and_labels_isolation():
    X,y,f,u,c=synthetic_design();T,diag,audit=crossfit_temperatures(X,y,f,0,u,c)
    X2=X.copy();X2[f==0]=1e6;y2=y.copy();y2[f==0]=5-y[f==0]
    TT,_,_=crossfit_temperatures(X2,y2,f,0,u,c);np.testing.assert_array_equal(T,TT)
    assert np.all(T>0) and all(d['inner_nll_after']<=d['inner_nll_before'] for d in diag)
    for a in audit:assert not set(a['train_indices'])&set(a['validation_indices'])
