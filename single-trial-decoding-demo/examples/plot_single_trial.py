"""Display fixed original input row 0, never select a trial for good appearance."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--oof',default='ordered-ring/runs/ordered_ring_transient_shared_r3/oof.npz')
parser.add_argument('--output',default='ordered-ring/runs/ordered_ring_transient_shared_r3/single_trial_0.png')
args=parser.parse_args()
d=np.load(args.oof,allow_pickle=False); idx=0
P=d['probabilities'][:,:,idx,:];times=d['times_seconds'];labels=d['target_labels'][idx]
tid=d['trial_ids'][idx];fold=d['outer_fold'][idx]
assert P.shape==(4,6,180), 'This presentation example requires the default three-target demo.'
assert np.isfinite(P).all() and np.allclose(P.sum(axis=1),1)
fig,axes=plt.subplots(4,1,figsize=(12,8),sharex=True,layout='constrained')
for j,ax in enumerate(axes):
    im=ax.imshow(P[j],origin='lower',aspect='auto',extent=(times[0]-.5/30,times[-1]+.5/30,.5,6.5),vmin=.2,vmax=1,cmap='viridis',interpolation='nearest')
    for onset in [1.,2.3,3.6]: ax.axvline(onset,color='white',ls='--',lw=1,alpha=.8)
    ax.set_ylabel(['Entry','Rank 1','Rank 2','Rank 3'][j]+'\nLocation'); ax.set_yticks(range(1,7))
axes[-1].set_xlabel('Original epoch time (s)')
fig.colorbar(im,ax=axes,label='Softmax probability (display 0.2–1; values <0.2 use bottom color)',shrink=.9)
fig.suptitle(f'OOF single-trial example — fixed input row 0, trial ID {tid}, outer fold {fold}\nTrue locations: T1={labels[0]}, T2={labels[1]}, T3={labels[2]} (labels for display only; not inputs to prediction)',fontsize=12)
fig.savefig(args.output,dpi=160)
