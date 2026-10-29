import numpy as np
from scipy.linalg import helmert
from jointdecode.diagnostics import geometry,source_activity,template_attribution,window_cross_target_scores


def test_five_dimensional_geometry_and_rank_penalty():
    t=helmert(6,full=False).T
    g=geometry(t,t)
    assert g['estimated_rank']==5
    assert g['projector_distance']<1e-14
    assert abs(g['estimated_span_truth_energy_capture']-1)<1e-14
    e=t.copy();e[:,2:]=0
    g=geometry(e,t)
    assert g['projector_distance']>0
    assert g['angles_cover_matched_directions_only']
    assert len(g['principal_angles_degrees'])==2


def test_basis_rotation_and_scaled_truth():
    rng=np.random.default_rng(8);t=rng.normal(size=(6,12));t-=t.mean(0)
    h=helmert(6,full=False).T
    rotation=np.linalg.qr(rng.normal(size=(5,5)))[0]
    e=h@rotation@h.T@t
    assert abs(geometry(e,t)['estimated_span_truth_energy_capture']-1)<1e-12
    scale=np.arange(1,13)
    model=dict(outer_fold=0,scale=scale,joint=dict(templates=np.array([t/scale,t/scale])))
    truth=dict(templates=np.array([t,t]),entry_gain=np.ones((1,5)),rank_gain=np.ones((1,5)))
    assert abs(template_attribution(model,truth)[0]['signed_template_cosine']-1)<1e-12


def test_absent_nonzero_basis_and_collapsed():
    t=helmert(6,full=False).T
    truth=dict(templates=np.array([t,t]),entry_gain=np.zeros((1,5)),rank_gain=np.ones((1,5)))
    model=dict(outer_fold=0,scale=np.ones(5),joint=dict(templates=np.array([t,np.zeros_like(t)])))
    assert not source_activity(truth)[0]['source_active']
    rec=template_attribution(model,truth)
    assert rec[0]['status']=='absent' and rec[0]['projector_distance'] is None
    assert abs(rec[0]['unused_basis_alignment']['signed_template_cosine']-1)<1e-12
    assert rec[3]['status']=='collapsed'
    assert rec[3]['signed_template_cosine'] is None
    assert rec[3]['estimated_span_truth_energy_capture']==0
    assert geometry(np.zeros_like(t),np.zeros_like(t))['projector_distance']==0


def test_scores_framewise_not_mean_probability_and_label_isolation():
    labels=np.array([[1,2],[2,3],[3,1]])
    p=np.zeros((1,3,3,2))
    for n in range(3):p[0,n,n,:]=1
    frozen=p.copy();rows=window_cross_target_scores(p,labels,[0.,1.],{'all':(0.,2.)})
    assert rows[0]['balanced_accuracy']==1
    assert rows[1]['balanced_accuracy']==0
    assert rows[0]['prediction_matches_no_target']==0
    shuffled=window_cross_target_scores(p,labels[::-1],[0.,1.],{'all':(0.,2.)})
    assert np.array_equal(p,frozen)
    assert shuffled[0]['balanced_accuracy']!=rows[0]['balanced_accuracy']

