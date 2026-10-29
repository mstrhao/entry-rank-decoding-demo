"""Balanced independent simulation; ground truth is never supplied to fitting."""
from itertools import permutations
import numpy as np
from scipy.linalg import helmert
from scipy.special import expit


def simulate(config, *, neurons=60, repetitions=5, overlap=.25, noise_sd=1., entry_amplitude=1.5, rank_amplitude=1.):
    R=len(config.target_onsets_seconds); K=len(config.position_labels)
    if repetitions != 5:
        raise ValueError('The balanced fivefold simulation requires five repetitions.')
    if neurons < (R+1)*(K-1) or not 0 <= overlap < 1:
        raise ValueError('Need P >= (R+1)*(K-1) and 0 <= overlap < 1.')
    rng=np.random.default_rng(config.seed_data)
    split_rng=np.random.default_rng(config.seed_split)
    combos=np.array(list(permutations(range(1,K+1),R)))
    labels=np.repeat(combos,5,axis=0)
    folds=np.concatenate([split_rng.permutation(5) for _ in combos])
    perm=rng.permutation(len(labels)); labels=labels[perm]; folds=folds[perm]
    n=len(labels); times=np.asarray(config.times_seconds); T=len(times)
    Q,rr=np.linalg.qr(rng.normal(size=(neurons,(R+1)*(K-1))))
    Q*=np.where(np.diag(rr)<0,-1,1)[None,:]
    C=helmert(K,full=False).T; qe=Q[:,:K-1]
    templates=[C@qe.T]
    for r in range(R):
        qr=Q[:,(r+1)*(K-1):(r+2)*(K-1)]
        templates.append(C@(overlap*qe+np.sqrt(1-overlap**2)*qr).T)
    templates=np.array(templates); templates-=templates.mean(axis=1,keepdims=True)
    templates/=np.sqrt(np.mean(templates**2,axis=(1,2),keepdims=True))
    on=np.asarray(config.target_onsets_seconds)[:,None]; dt=times[None,:]-on
    e=(dt>=0)*np.exp(-.5*((dt-.25)/.13)**2)
    q=(dt>=0)*expit((dt-.55)/.18)*expit((5.5-times[None,:])/.12)
    b=rng.normal(0,.3,size=(n,neurons,1))
    noise=rng.normal(size=(n,neurons,T))
    for t in range(1,T): noise[:,:,t]=.6*noise[:,:,t-1]+.8*noise[:,:,t]
    noise*=noise_sd
    X=100+b+noise
    for r in range(R):
        X += entry_amplitude*templates[0][labels[:,r]-1,:,None]*e[r]
        X += rank_amplitude*templates[r+1][labels[:,r]-1,:,None]*q[r]
    return dict(epochs=X,target_labels=labels,outer_fold=folds,trial_ids=np.arange(n),groups=np.arange(n),
                simulation_order=perm,ground_truth=dict(templates=templates,entry_gain=entry_amplitude*e,rank_gain=rank_amplitude*q,noise=noise),
                simulation_parameters=dict(overlap=overlap,noise_sd=noise_sd,entry_amplitude=entry_amplitude,rank_amplitude=rank_amplitude,neurons=neurons))
