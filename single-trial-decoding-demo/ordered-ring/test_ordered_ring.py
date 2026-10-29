import inspect
import numpy as np
from run_ordered_ring import named_simulate,ring_contrast
from jointdecode.config import Config
from jointdecode import simulation,cli
from jointdecode.readout import predict_fold
from plot_ordered_ring import topology


def test_ring_geometry_and_paired_random_stream():
    old=simulation.helmert;runner=cli.simulate
    a=named_simulate(Config());b=simulation.simulate(Config(),overlap=0.)
    assert simulation.helmert is old and cli.simulate is runner
    for key in ['target_labels','outer_fold','trial_ids','groups','simulation_order']:np.testing.assert_array_equal(a[key],b[key])
    for key in ['noise','entry_gain','rank_gain']:np.testing.assert_array_equal(a['ground_truth'][key],b['ground_truth'][key])
    target=np.cos(2*np.pi*(np.arange(6)[:,None]-np.arange(6)[None,:])/6)
    S=a['ground_truth']['templates']
    for s in S:
        assert np.linalg.matrix_rank(s)==2
        np.testing.assert_allclose(s@s.T/60,target,atol=1e-14)
        np.testing.assert_allclose(np.mean(s*s),1,atol=1e-14)
    for i in range(4):
        for j in range(i):np.testing.assert_allclose(S[i]@S[j].T,0,atol=1e-12)
    assert list(inspect.signature(predict_fold).parameters)==['model','X_test']


def test_topology_is_diagnostic_not_relabeling():
    xy=ring_contrast(6).T[:,:2];original=xy.copy()
    for v in [xy,xy[:,::-1],np.roll(xy,2,axis=0)]:
        t=topology(v);assert t['cyclic_order_up_to_rotation_reflection'];assert abs(abs(t['winding_number'])-1)<1e-12;assert not t['nonadjacent_edge_crossings']
    assert not topology(xy[[0,2,1,3,4,5]])['cyclic_order_up_to_rotation_reflection']
    np.testing.assert_array_equal(xy,original)
