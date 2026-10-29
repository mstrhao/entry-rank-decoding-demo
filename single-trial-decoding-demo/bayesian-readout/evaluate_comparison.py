"""Paired, window-prespecified evaluation of frozen OOF class probabilities.

Scores are computed per frame before averaging; the resampling unit is the whole
trial, with the SAME trial indices for both methods and all six target/windows.
No fitting, probability calibration, or threshold selection happens here.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.special import xlogy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('Logistic', 'Gaussian Bayes')
COLORS = ('#2878B5', '#D95F02')


def reliability(prob, labels, bins=10):
    """Top-label ECE; equal-width bins [lo,hi), last bin includes 1."""
    K,N,T=prob.shape
    y=np.asarray(labels,dtype=int)-1
    prediction=prob.argmax(axis=0); confidence=prob.max(axis=0)
    correct=(prediction==y[:,None])
    assignment=np.minimum((confidence*bins).astype(int),bins-1)
    records=[]; ece=0.
    for b in range(bins):
        selected=assignment==b; count=int(selected.sum())
        accuracy=float(correct[selected].mean()) if count else None
        conf=float(confidence[selected].mean()) if count else None
        gap=abs(accuracy-conf) if count else None
        if count:ece+=count/(N*T)*gap
        records.append(dict(bin=b,lower=b/bins,upper=(b+1)/bins,count=count,
                            accuracy=accuracy,confidence=conf,absolute_gap=gap))
    return float(ece),records


def score_window(prob, logp, labels):
    """prob/logp [K,N,T]; labels [N] 1-based. Preserve log-domain NLL."""
    prob=np.asarray(prob,float); logp=np.asarray(logp,float)
    K,N,T=prob.shape; y=np.asarray(labels,dtype=int)-1
    if logp.shape!=prob.shape or y.shape!=(N,) or np.any((y<0)|(y>=K)):
        raise ValueError('Probability/log-probability/label shape mismatch')
    if not np.isfinite(prob).all() or np.any(prob<0) or not np.allclose(prob.sum(0),1):
        raise ValueError('Invalid normalized probabilities')
    correct=(prob.argmax(0)==y[:,None]).astype(float)
    true_p=prob[y,np.arange(N),:]
    nll=-logp[y,np.arange(N),:]
    brier=np.square(prob).sum(0)-2*true_p+1
    entropy=-xlogy(prob,prob).sum(0)
    trial=dict(accuracy=correct.mean(1),nll=nll.mean(1),brier=brier.mean(1),
               entropy=entropy.mean(1),true_probability=true_p.mean(1))
    per_class=[float(correct[y==k].mean()) for k in range(K) if np.any(y==k)]
    if len(per_class)!=K:raise ValueError('All classes required for balanced accuracy')
    ece,bins=reliability(prob,labels)
    summary={k:float(v.mean()) for k,v in trial.items()}
    summary.update(balanced_accuracy=float(np.mean(per_class)),ece=ece,
                   n_trials=N,n_frames=T,n_observations=N*T)
    return summary,trial,bins


def probability_difference(a,b):
    """Label-free total variation and Jensen-Shannon divergence, per trial/frame."""
    middle=(a+b)/2
    tv=.5*np.abs(a-b).sum(0)
    # xlogy(0,0) is zero; zero middle can occur only when both masses are zero.
    ar=np.divide(a,middle,out=np.ones_like(a),where=middle>0)
    br=np.divide(b,middle,out=np.ones_like(b),where=middle>0)
    js=.5*(xlogy(a,ar)+xlogy(b,br)).sum(0)
    return tv,js


def paired_bootstrap(trial_a,trial_b,labels,draws,K):
    """Fixed-OOF conditional uncertainty; do NOT interpret as retraining CI."""
    y=np.asarray(labels)-1; samples={}
    for metric in ('accuracy','nll','brier'):
        samples[metric]=(trial_b[metric]-trial_a[metric])[draws].mean(1)
    delta=(trial_b['accuracy']-trial_a['accuracy'])[draws]
    yy=y[draws]; class_delta=[]; missing=np.zeros(len(draws),bool)
    for k in range(K):
        mask=yy==k; count=mask.sum(1);missing|=count==0
        class_delta.append(np.divide((delta*mask).sum(1),count,
                            out=np.full(len(draws),np.nan),where=count>0))
    samples['balanced_accuracy']=np.mean(class_delta,axis=0)
    result={}
    for metric,sample in samples.items():
        finite=sample[np.isfinite(sample)]
        result[metric]=dict(lower=float(np.quantile(finite,.025)),
                            upper=float(np.quantile(finite,.975)),
                            valid_replicates=int(len(finite)))
    result['replicates_missing_class']=int(missing.sum())
    return result


def _cfg(config,key):
    return config[key] if isinstance(config,dict) else getattr(config,key)


def _write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate(logistic_prob,bayes_prob,logistic_logp,bayes_logp,labels,times,config,outdir):
    """Generate metrics.json/CSVs and five PNGs; return JSON-safe full report."""
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    probs=[np.asarray(logistic_prob),np.asarray(bayes_prob)]
    logs=[np.asarray(logistic_logp),np.asarray(bayes_logp)]
    labels=np.asarray(labels);times=np.asarray(times)
    J,K,N,T=probs[0].shape;R=labels.shape[1]
    if J!=R+1 or J!=4 or K!=6 or labels.shape[0]!=N or len(times)!=T:
        raise ValueError('This fixed display requires R3/K6 with aligned trial/time axes')
    if any(a.shape!=(J,K,N,T) for a in probs+logs):raise ValueError('Aligned method arrays required')
    specs=[]
    for family in ('Entry','Rank'):
        windows=_cfg(config,'entry_readout_windows_seconds' if family=='Entry' else 'rank_readout_windows_seconds')
        for r in range(R):
            mask=(times>=windows[r][0])&(times<windows[r][1])
            if not mask.any():raise ValueError('Empty prespecified evaluation window')
            specs.append(dict(name=f'{family} / T{r+1}',family=family,target=r,
                              component=0 if family=='Entry' else r+1,window=list(windows[r]),mask=mask))
    # Single shared resampling matrix across ALL windows and methods.
    draws=np.random.default_rng(2026).integers(0,N,size=(1000,N))
    rows=[];binrows=[];records=[]
    for spec in specs:
        j=spec['component'];r=spec['target'];mask=spec['mask'];scores=[];trials=[]
        for method,prob,logp in zip(METHODS,probs,logs):
            score,trial,bins=score_window(prob[j][:,:,mask],logp[j][:,:,mask],labels[:,r])
            scores.append(score);trials.append(trial)
            rows.append(dict(window=spec['name'],component=j,target=r+1,method=method,**score))
            binrows.extend(dict(window=spec['name'],method=method,**b) for b in bins)
        tv,js=probability_difference(probs[0][j][:,:,mask],probs[1][j][:,:,mask])
        delta={k:scores[1][k]-scores[0][k] for k in ('balanced_accuracy','accuracy','nll','brier','ece','entropy','true_probability')}
        records.append(dict(name=spec['name'],component=j,target=r+1,window=spec['window'],
                            methods=dict(zip(METHODS,scores)),bayes_minus_logistic=delta,
                            paired_trial_bootstrap_95=paired_bootstrap(*trials,labels[:,r],draws,K),
                            total_variation_mean=float(tv.mean()),jensen_shannon_nats_mean=float(js.mean())))
    report=dict(methods=list(METHODS),shape=[J,K,N,T],windows=records,
        definitions=dict(evaluation='Per-frame score before equal trial/time averaging; only prespecified target-present windows',
         balanced_accuracy='Mean class recall; pooled trial-frame predictions within each window',
         nll='Mean negative log true-class posterior in natural-log units; from log-domain values, not clipped probabilities',
         brier='Mean sum over all K classes of (probability - onehot)^2; range 0..2',
         ece='Top-label confidence ECE: 10 equal-width bins [b/10,(b+1)/10), last includes 1; pooled correlated trial-frames, no independence claim',
         entropy='Mean -sum p log p, natural-log units; not animal confidence',
         total_variation='0.5 sum_k |p_Bayes-p_logistic|; range 0..1; label-free',
         jensen_shannon='0.5 KL(p_logistic||m)+0.5 KL(p_Bayes||m), m=(p_logistic+p_Bayes)/2; natural-log units',
         bootstrap='1000 paired resamples of entire trials, seed 2026, same index matrix across both methods and every window; percentile 95% CI on Bayes-minus-logistic. Conditional on this simulation and fixed fitted OOF models; no retraining, no animal/generalization inference, no multiple-comparison significance claim',
         display='Fixed original row/trial 0; heatmaps 0.2..1 (values below .2 bottom color), numerical outputs unchanged; all other plots use their own metric scales'))
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    _write_csv(out/'window_metrics.csv',rows);_write_csv(out/'reliability.csv',binrows)
    _plots(probs,labels,times,config,specs,records,binrows,out)
    return report


def _plots(probs,labels,times,config,specs,records,binrows,out):
    x=np.arange(6);names=[s['name'] for s in specs]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['balanced_accuracy','accuracy','nll','brier','ece','entropy'],
                           ['Balanced accuracy ↑','Accuracy ↑','NLL (nats) ↓','Multiclass Brier ↓','Top-label ECE ↓','Posterior entropy (nats)']):
        for mi,method in enumerate(METHODS):
            ax.bar(x+(mi-.5)*.36,[r['methods'][method][key] for r in records],width=.36,color=COLORS[mi],label=method)
        ax.set_xticks(x,names,rotation=30,ha='right');ax.set_title(title);ax.set_ylim(bottom=0)
        if key in ('balanced_accuracy','accuracy'):ax.set_ylim(0,1.05)
    axes.flat[0].legend(loc='lower left');fig.suptitle('Same frozen OOF trials and windows; only the readout differs\nPer-frame scoring before window averaging; entropy is not an accuracy score',fontsize=14)
    fig.savefig(out/'window_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=False,sharey=True,layout='constrained')
    for ax,s in zip(axes.flat,specs):
        for mi,p in enumerate(probs):
            pred=p[s['component']].argmax(0);y=labels[:,s['target']]-1
            ba=np.mean([(pred[y==k]==k).mean(0) for k in range(6)],axis=0)
            ax.plot(times,np.where(s['mask'],ba,np.nan),color=COLORS[mi],label=METHODS[mi])
        ax.axhline(1/6,color='.6',ls=':',lw=1);ax.axvspan(*s['window'],color='.8',alpha=.2)
        ax.set(title=s['name'],xlim=tuple(s['window']),ylim=(0,1.05),xlabel='Epoch time (s)',ylabel='Balanced accuracy')
    axes.flat[0].legend();fig.suptitle('Time-resolved accuracy: only prespecified target-present windows shown')
    fig.savefig(out/'window_time_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,10),sharex=True,sharey=True,layout='constrained')
    for j in range(4):
        for mi,p in enumerate(probs):
            ax=axes[j,mi];im=ax.imshow(p[j,:,0,:],origin='lower',aspect='auto',extent=(times[0],times[-1]+np.median(np.diff(times)),.5,6.5),vmin=.2,vmax=1,cmap='viridis')
            for onset in _cfg(config,'target_onsets_seconds'):ax.axvline(onset,color='white',ls='--',alpha=.7)
            ax.set_yticks(range(1,7));ax.set_ylabel(('Entry' if j==0 else f'Rank {j}')+'\nLocation')
            if j==0:ax.set_title(METHODS[mi])
            if j==3:ax.set_xlabel('Epoch time (s)')
    fig.colorbar(im,ax=axes,label='Class posterior (display .2–1; <.2 bottom color)',shrink=.85)
    fig.suptitle(f'Fixed original trial 0, OOF: true locations {labels[0].tolist()} (display only)\nSame frozen upstream representation; neither predictor receives these labels')
    fig.savefig(out/'single_trial_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=False,sharey=True,layout='constrained')
    for ax,s,rec in zip(axes.flat,specs,records):
        tv,_=probability_difference(probs[0][s['component']],probs[1][s['component']])
        ax.plot(times,np.where(s['mask'],tv.mean(0),np.nan),color='#7B3294');ax.axvspan(*s['window'],color='.8',alpha=.2)
        ax.set(title=f"{s['name']}; window mean={rec['total_variation_mean']:.3f}",xlim=tuple(s['window']),ylim=(0,1),xlabel='Epoch time (s)',ylabel='Mean total variation')
    fig.suptitle('Probability disagreement (label-free): total variation of all six location probabilities\nShown in the same prespecified windows; disagreement is not evidence of which method is right')
    fig.savefig(out/'probability_disagreement.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
    for ax,s in zip(axes.flat,specs):
        ax.plot([0,1],[0,1],ls=':',color='.5')
        for mi,method in enumerate(METHODS):
            rr=[r for r in binrows if r['window']==s['name'] and r['method']==method and r['count']]
            ax.plot([r['confidence'] for r in rr],[r['accuracy'] for r in rr],'-o',color=COLORS[mi],label=method,markersize=4)
        ax.set(title=s['name'],xlim=(0,1),ylim=(0,1),xlabel='Mean predicted top-class confidence',ylabel='Observed top-class accuracy')
    axes.flat[0].legend();fig.suptitle('Reliability diagrams: 10 fixed equal-width confidence bins\nEmpty bins omitted; counts in reliability.csv; correlated frames, not independent replicates')
    fig.savefig(out/'reliability.png',dpi=160);plt.close(fig)
