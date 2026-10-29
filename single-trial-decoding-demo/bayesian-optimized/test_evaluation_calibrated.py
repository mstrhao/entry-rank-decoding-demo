import inspect
import numpy as np
from scipy.special import log_softmax
from evaluate_calibrated import evaluate,score_window,paired_bootstrap


def test_conditional_evaluator_has_no_truth_gain_or_presence_input():
    assert list(inspect.signature(evaluate).parameters)==['logps','labels','times','config','outdir']


def test_positive_temperature_preserves_decisions_not_probabilities():
    rng=np.random.default_rng(2026);logits=rng.normal(size=(6,120,4))*3
    l0=log_softmax(logits,axis=0);l1=log_softmax(logits/2,axis=0);y=np.tile(np.arange(1,7),20)
    s0,t0,_=score_window(np.exp(l0),l0,y);s1,t1,_=score_window(np.exp(l1),l1,y)
    assert np.array_equal(l0.argmax(0),l1.argmax(0))
    assert s0['accuracy']==s1['accuracy'] and s0['balanced_accuracy']==s1['balanced_accuracy']
    assert s0['entropy']<s1['entropy']
    draws=rng.integers(0,len(y),(100,len(y)));ci=paired_bootstrap(t0,t1,y,draws,6)
    assert ci['balanced_accuracy']['lower']==ci['balanced_accuracy']['upper']==0
