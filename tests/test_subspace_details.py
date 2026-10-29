import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('details',Path(__file__).resolve().parents[1]/'scripts'/'plot_subspace_details.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_pca_preserves_two_dimensional_template_and_ratios():
    S=np.array([[2.,0.,0.],[-2.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
    p=m.template_pca(S)
    np.testing.assert_allclose(p['explained_variance_ratio'],[.8,.2,0.],atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(p['scores'],axis=1),np.linalg.norm(S,axis=1))
    np.testing.assert_allclose(p['scores'].mean(axis=0),0,atol=1e-14)


def test_directional_vaf_not_symmetric_or_basis_overlap():
    S=np.array([[3.,0.,0.],[-3.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
    T=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    Q=[np.eye(3)[:,:2],np.eye(3)[:,:1]]
    v,_,p=m.space_relations([S,T],Q)
    np.testing.assert_allclose(v,[[1,.9],[1,1]])
    assert p[0]['source_dimension']==2 and p[0]['target_dimension']==1
    assert len(p[0]['angles_degrees'])==1


def test_full_angle_spectrum_and_basis_rotation_invariance():
    rng=np.random.default_rng(88);A,_=np.linalg.qr(rng.normal(size=(15,5)));B,_=np.linalg.qr(rng.normal(size=(15,5)));O,_=np.linalg.qr(rng.normal(size=(5,5)))
    S=rng.normal(size=(6,5))@A.T;T=rng.normal(size=(6,5))@B.T
    v,a,p=m.space_relations([S,T],[A,B]);v2,a2,p2=m.space_relations([S,T],[A@O,B@O])
    np.testing.assert_allclose(v,v2,atol=1e-14);np.testing.assert_allclose(a,a2,atol=1e-12)
    assert len(p[0]['angles_degrees'])==5
    np.testing.assert_allclose(p[0]['angles_degrees'],p2[0]['angles_degrees'],atol=1e-12)
    assert np.all((v>=0)&(v<=1+1e-14))


def test_degenerate_templates_and_empty_target_are_explicit():
    zero=np.zeros((6,4));p=m.template_pca(zero)
    assert np.isfinite(p['scores']).all() and np.isfinite(p['explained_variance_ratio']).all()
    np.testing.assert_array_equal(p['explained_variance_ratio'],0)
    S=np.array([[1.,0.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
    v,a,pairs=m.space_relations([zero,S],[np.zeros((4,0)),np.eye(4)[:,:1]])
    assert np.isnan(v[0]).all()  # No source energy: undefined, not zero recovery.
    assert v[1,0]==0 and v[1,1]==1
    assert np.isnan(a[0]).all() and pairs[0]['angles_degrees']==[]
