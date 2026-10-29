"""Conditional six-location evaluation; no absence model or truth-gain gating.

Reuses the original per-frame score/reliability and whole-trial bootstrap code.
Prestimulus predictions remain visible; neither uniform nor sharp location
probabilities establish absence/presence of imagined content.
"""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import xlogy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bayesian-readout'))
from evaluate_comparison import score_window,paired_bootstrap

KEYS=('logistic','mean_gaussian','frame_gaussian','calibrated_gaussian')
NAMES=('M0 Logistic','M1 Mean Gaussian','M2 Frame Gaussian','M2 + train-only temperature')
COLORS=('#2878B5','#D95F02','#2A9D55','#8B3DAD')
COMPONENTS=('Entry','Rank 1','Rank 2','Rank 3')


def _get(config,key):
    return config[key] if isinstance(config,dict) else getattr(config,key)


def _csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate(logps,labels,times,config,outdir):
    """Compare four [component,location,trial,time] conditional log posteriors."""
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    logs=[np.asarray(logps[k],float) for k in KEYS];probs=[np.exp(l) for l in logs]
    labels=np.asarray(labels,int);times=np.asarray(times);J,K,N,T=logs[0].shape
    if (J,K)!=(4,6) or labels.shape!=(N,3) or times.shape!=(T,):raise ValueError('Aligned R3/K6 axes required')
    if any(l.shape!=(J,K,N,T) for l in logs):raise ValueError('All methods must have exactly the same axes')
    if any(not np.allclose(p.sum(1),1) for p in probs):raise ValueError('Six-way normalized conditionals required')
    draws=np.random.default_rng(2026).integers(0,N,size=(1000,N))
    windows=[];rows=[];binrows=[]
    for family,windowkey in [('Entry','entry_readout_windows_seconds'),('Rank','rank_readout_windows_seconds')]:
        for r,window in enumerate(_get(config,windowkey)):
            j=0 if family=='Entry' else r+1;mask=(times>=window[0])&(times<window[1])
            if not mask.any():raise ValueError('Empty prespecified score window')
            methods={};trial=[]
            for key,name,p,l in zip(KEYS,NAMES,probs,logs):
                score,tr,bins=score_window(p[j][:,:,mask],l[j][:,:,mask],labels[:,r]);methods[key]=score;trial.append(tr)
                rows.append(dict(window=f'{family} / T{r+1}',method=key,**score))
                binrows.extend(dict(window=f'{family} / T{r+1}',method=key,**b) for b in bins)
            delta={key:methods['calibrated_gaussian'][key]-methods['frame_gaussian'][key] for key in ('balanced_accuracy','accuracy','nll','brier','ece','entropy','true_probability')}
            windows.append(dict(name=f'{family} / T{r+1}',component=j,target=r+1,window=list(window),methods=methods,
                calibrated_minus_frame=delta,paired_trial_bootstrap_95=paired_bootstrap(trial[2],trial[3],labels[:,r],draws,K)))
    report=dict(methods=dict(zip(KEYS,NAMES)),shape=[J,K,N,T],windows=windows,
        definitions=dict(scope='All four outputs are conditional probabilities over the SAME exclusive six-location category set. No null class, no source-gain masks, no presence threshold, no prestimulus suppression.',
          scoring='Original six prespecified Entry/own-Rank windows; per-frame scores computed before trial/time averaging; no window-averaged posterior scoring.',
          nll='Negative true-class log posterior, natural-log units, direct log-domain values without clipping.',
          brier='Mean sum over six classes of squared probability error, range 0..2.',
          calibration='Top-label ECE with 10 fixed equal-width confidence bins; last includes 1; counts in reliability.csv. This is calibration under the specified class problem, not animal confidence or content presence.',
          temperature='One positive scalar temperature per component and outer fold, fitted only on inner-readout-OOF original-window training data, not held-out outer predictions. Readout-only crossfit conditional on fixed upstream; not fully nested upstream refitting. Positive scalar temperature preserves each argmax.',
          bootstrap='1000 whole-trial paired resamples, seed2026; same draws for both methods and all six windows. Percentile95% interval for calibrated-minus-frame conditional on fixed OOF models/simulation, not retraining or across-animal inference; no iid-frame significance.',
          interpretation='Uniform probabilities do not establish absence. Sharp probabilities do not prove real or imagined content. Six mutually exclusive classes cannot identify simultaneous coactivation of multiple locations.',
          display='Fixed original trial0, known outer fold4; true labels for display only. Full epoch including prestimulus retained. Heatmaps use .2–1; <.2 bottom color, values unchanged. Entropy shown over all times is descriptive, not an absence detector.'))
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');_csv(out/'window_metrics.csv',rows);_csv(out/'reliability.csv',binrows)
    _plots(probs,labels,times,config,windows,binrows,out)
    return report


def _plots(probs,labels,times,config,windows,binrows,out):
    x=np.arange(6);wn=[w['name'] for w in windows]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['balanced_accuracy','accuracy','nll','brier','ece','entropy'],
                           ['Balanced accuracy ↑','Accuracy ↑','NLL (nats) ↓','Multiclass Brier ↓','Top-label ECE ↓','Conditional entropy (nats)']):
        for mi,k in enumerate(KEYS):ax.bar(x+(mi-1.5)*.2,[w['methods'][k][key] for w in windows],width=.2,color=COLORS[mi],label=NAMES[mi])
        ax.set_xticks(x,wn,rotation=30,ha='right');ax.set_title(title);ax.set_ylim(bottom=0)
        if key in ('balanced_accuracy','accuracy'):ax.set_ylim(0,1.05)
    axes.flat[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Four conditional six-location readouts: same frozen upstream, trials, and original scoring windows\nTemperature is training-only; lower entropy is not automatically better calibration')
    fig.savefig(out/'conditional_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(18,10),sharex=True,sharey=True,layout='constrained')
    for j in range(4):
        for mi,p in enumerate(probs):
            ax=axes[j,mi];im=ax.imshow(p[j,:,0],origin='lower',aspect='auto',extent=(times[0],times[-1]+np.median(np.diff(times)),.5,6.5),vmin=.2,vmax=1,cmap='viridis')
            for onset in _get(config,'target_onsets_seconds'):ax.axvline(onset,color='white',ls='--',alpha=.7,lw=.7)
            ax.set_yticks(range(1,7))
            if mi==0:ax.set_ylabel(COMPONENTS[j]+'\nLocation')
            if j==0:ax.set_title(NAMES[mi])
            if j==3:ax.set_xlabel('Epoch time (s)')
    fig.colorbar(im,ax=axes,label='Conditional location probability (display .2–1; <.2 bottom color)',shrink=.8)
    fig.suptitle(f'Fixed original trial0 / outer fold4; true locations {labels[0].tolist()} for display only\nAll columns sum to one over six locations; prestimulus retained, no null class or truth gating')
    fig.savefig(out/'single_trial_conditional.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True,layout='constrained')
    for ax,w in zip(axes.flat,windows):
        ax.plot([0,1],[0,1],'k:',lw=1)
        for mi,k in enumerate(KEYS):
            b=[v for v in binrows if v['window']==w['name'] and v['method']==k and v['count']]
            ax.plot([v['confidence'] for v in b],[v['accuracy'] for v in b],'-o',ms=3,color=COLORS[mi],label=NAMES[mi])
        ax.set(title=w['name'],xlabel='Mean top-class confidence',ylabel='Observed top-class accuracy',xlim=(0,1),ylim=(0,1))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Original-window reliability: 10 equal-width confidence bins, empty bins omitted\nCounts in reliability.csv; correlated frames, not independent replicates')
    fig.savefig(out/'conditional_reliability.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True,sharey=True,layout='constrained')
    for j,ax in enumerate(axes):
        for mi,p in enumerate(probs):ax.plot(times,-xlogy(p[j,:,0],p[j,:,0]).sum(0),color=COLORS[mi],label=NAMES[mi],lw=1)
        for onset in _get(config,'target_onsets_seconds'):ax.axvline(onset,color='.6',ls='--',lw=.7)
        ax.axhline(np.log(6),color='.6',ls=':',lw=.7);ax.set(title=COMPONENTS[j],ylabel='Entropy (nats)',ylim=(0,np.log(6)*1.04))
    axes[0].legend(fontsize=8,loc='upper right');axes[-1].set_xlabel('Epoch time (s)')
    fig.suptitle('Fixed trial0: full-epoch six-location uncertainty, including prestimulus\nUniform entropy log(6) is not absence; low entropy is not evidence of experienced/imagined content')
    fig.savefig(out/'single_trial_entropy.png',dpi=160);plt.close(fig)
