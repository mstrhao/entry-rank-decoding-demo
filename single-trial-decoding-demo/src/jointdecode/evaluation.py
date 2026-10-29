"""Post-prediction descriptive metrics and geometry (no truth used in fit)."""
import numpy as np
from scipy.linalg import subspace_angles
from .config import frame_mask
from .joint import row_basis


def evaluate(probabilities,labels,config):
    J,K,N,T=probabilities.shape; R=labels.shape[1]; y=labels-1
    pred=probabilities.argmax(axis=1)
    ba=np.empty((J,R,T)); true=np.empty((J,R,N,T))
    for j in range(J):
        for r in range(R):
            ba[j,r]=np.mean([(pred[j,y[:,r]==k]==k).mean(axis=0) for k in range(K)],axis=0)
            true[j,r]=probabilities[j,y[:,r],np.arange(N),:]
    own=np.stack([true[0],np.array([true[r+1,r] for r in range(R)])])
    metrics=[]
    for r in range(R):
        for name,j,win in [('Entry',0,config.entry_readout_windows_seconds[r]),('Rank',r+1,config.rank_readout_windows_seconds[r])]:
            mask=frame_mask(config.times_seconds,win)
            metrics.append(dict(target=r+1,space=name,window=list(win),balanced_accuracy=float(ba[j,r,mask].mean()),true_probability=float(true[j,r,:,mask].mean())))
    return dict(balanced_accuracy=ba,true_probability=true,own_target_probability=own,window_metrics=metrics)


def recovery(models,ground_truth):
    from .diagnostics import template_attribution
    records=[]
    for model in models:
        for record in template_attribution(model,ground_truth):
            if record['learned_space']==record['truth_source']:
                records.append(dict(record,space=record['learned_space']))
    return records


def plots(result,config,directory,geometry=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    directory=Path(directory); directory.mkdir(exist_ok=True)
    t=np.asarray(config.times_seconds); R=len(config.target_onsets_seconds); K=len(config.position_labels)
    names=['Entry']+[f'Rank {r+1}' for r in range(R)]
    fig,axes=plt.subplots(R+1,R,figsize=(12,9),sharex=True,sharey=True,squeeze=False)
    for j in range(R+1):
        for r in range(R):
            ax=axes[j,r]; ax.plot(t,result['balanced_accuracy'][j,r],lw=1.2); ax.axhline(1/K,color='gray',ls='--')
            for on in config.target_onsets_seconds: ax.axvline(on,color='gray',alpha=.25)
            ax.set_title(f'{names[j]} → target {r+1}'); ax.set_ylim(0,1); ax.set_xlabel('Time (s)')
    fig.suptitle('OOF balanced accuracy — descriptive, not significance'); fig.tight_layout(); fig.savefig(directory/'balanced_accuracy.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,R,figsize=(12,3.5),squeeze=False)
    for r,ax in enumerate(axes[0]):
        ax.plot(t,result['own_target_probability'][0,r].mean(axis=0),label='Entry')
        ax.plot(t,result['own_target_probability'][1,r].mean(axis=0),label=f'Rank {r+1}')
        ax.axhline(1/K,color='gray',ls='--'); ax.set(title=f'Target {r+1}',xlabel='Time (s)',ylabel='True-location probability'); ax.legend()
    fig.tight_layout(); fig.savefig(directory/'true_probability.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(5,R,figsize=(12,12),squeeze=False)
    for f,model in enumerate(result['models']):
        for r in range(R):
            ax=axes[f,r]; p=model['joint']['profiles']; ax.plot(t,p[r],label='a'); ax.plot(t,p[R+r],label='b')
            ax.plot(t,model['pE'][r],ls='--',label='pE'); ax.plot(t,model['pR'][r],ls='--',label='pR')
            ax.set_title(f'Fold {f}, target {r+1}'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(directory/'gains_priors.png',dpi=140); plt.close(fig)
    overlaps=np.array([[np.linalg.norm(a.T@b,'fro')**2/max(min(a.shape[1],b.shape[1]),1) for b in result['models'][0]['bases']] for a in result['models'][0]['bases']])
    fig,axes=plt.subplots(1,2,figsize=(11,4)); im=axes[0].imshow(overlaps,vmin=0,vmax=1); fig.colorbar(im,ax=axes[0]); axes[0].set_title('Fold 0 basis overlap (not unique information)'); axes[0].set_xticks(range(R+1),names); axes[0].set_yticks(range(R+1),names)
    if geometry:
        for j in range(R+1):
            values=[g['projection_relative_error'] for g in geometry if g['space']==j]
            axes[1].plot(range(5),values,'o-',label=names[j])
        axes[1].legend(); axes[1].set(title='Truth projection relative error',xlabel='Outer fold',ylabel='Relative Frobenius error')
    else: axes[1].axis('off')
    fig.tight_layout(); fig.savefig(directory/'geometry.png',dpi=140); plt.close(fig)
