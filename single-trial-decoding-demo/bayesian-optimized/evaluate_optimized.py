"""Evaluation-only source-activity masks and paired readout diagnostics.

True gains define prespecified evaluation subsets, never fitted predictors.
All framewise scores are descriptive: correlated frames are not iid replicates.
"""
from pathlib import Path
from types import SimpleNamespace
import json,csv
import numpy as np
from scipy.special import xlogy,logsumexp
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jointdecode.pipeline import preprocess

KEYS=('logistic','mean_gaussian','frame_gaussian')
NAMES=('M0 Logistic','M1 Mean Gaussian','M2 Frame Gaussian','M3 + baseline-like null')
COMPONENTS=('Entry','Rank 1','Rank 2','Rank 3')
COLORS=('#2878B5','#D95F02','#2A9D55','#8B3DAD')
SENSITIVITY=((.25,.05),(.5,.1),(.75,.15))


def _cfg(config):
    return SimpleNamespace(**config) if isinstance(config,dict) else config


def activity_masks(ground_truth,config,strong=.5,weak=.1):
    """state [4,T]: 1 active, 0 inactive, -1 mixed/gray; Entry target rank map."""
    cfg=_cfg(config)
    entry=np.asarray(ground_truth['entry_gain'],float)
    rank=np.asarray(ground_truth['rank_gain'],float)
    if entry.shape!=rank.shape or entry.shape[0]!=3:raise ValueError('R3 gains required')
    gains=preprocess(np.concatenate([entry,rank])[None,:,:],cfg)[0]
    peak=np.max(np.abs(gains),axis=1,keepdims=True)
    norm=np.divide(np.abs(gains),peak,out=np.zeros_like(gains),where=peak>0)
    e,r=norm[:3],norm[3:];T=e.shape[1]
    state=np.full((4,T),-1,int);target=np.full((4,T),-1,int)
    state[0,np.all(e<=weak,axis=0)]=0
    for k in range(3):
        active=(e[k]>=strong)&np.all(np.delete(e,k,axis=0)<=weak,axis=0)
        state[0,active]=1;target[0,active]=k
        state[k+1,r[k]<=weak]=0;state[k+1,r[k]>=strong]=1
        target[k+1,state[k+1]==1]=k
    return state,target,norm


def class_metrics(logp,y,K=6):
    """logp [K,S], y[S] zero-based; no averaging probabilities before scores."""
    if logp.shape!=(K,len(y)) or not len(y):return None
    p=np.exp(logp);prediction=np.argmax(p,axis=0);correct=prediction==y
    confidence=p.max(axis=0);bi=np.minimum((confidence*10).astype(int),9)
    bins=[];ece=0.
    for b in range(10):
        m=bi==b;n=int(m.sum());a=float(correct[m].mean()) if n else None;c=float(confidence[m].mean()) if n else None
        bins.append(dict(bin=b,lower=b/10,upper=(b+1)/10,count=n,accuracy=a,confidence=c))
        if n:ece+=n/len(y)*abs(a-c)
    difference=np.abs(prediction-y);steps=np.minimum(difference,K-difference)
    present=[k for k in range(K) if np.any(y==k)]
    return dict(n_observations=len(y),accuracy=float(correct.mean()),
        balanced_accuracy=float(np.mean([correct[y==k].mean() for k in present])),n_classes_present=len(present),
        nll=float(-logp[y,np.arange(len(y))].mean()),
        brier=float((np.square(p).sum(axis=0)-2*p[y,np.arange(len(y))]+1).mean()),
        ece=float(ece),entropy=float(-xlogy(p,p).sum(axis=0).mean()),
        circular_error_steps=float(steps.mean()),circular_error_degrees=float(steps.mean()*360/K),reliability=bins)


def coverage_risk(confidence,prediction,truth,supported,threshold_grid=None):
    """Exact strict-threshold curve with whole confidence ties kept together.

    Default: compute every unique-confidence operating point, then deterministically
    retain at most 250 points for the report/plot. This is not threshold fitting.
    """
    c=np.asarray(confidence)[supported];p=np.asarray(prediction)[supported];y=np.asarray(truth)[supported]
    wrong=(y<0)|(p!=y)
    if not len(c):return [dict(threshold=1.,coverage=None,risk=None,n_emitted=0,n_supported=0)]
    order=np.argsort(c,kind='stable');sorted_c=c[order]
    prefix=np.r_[0,np.cumsum(wrong[order],dtype=np.int64)]
    default=threshold_grid is None
    thresholds=np.unique(np.r_[0.,c,1.]) if default else np.asarray(threshold_grid,float)
    # side=right excludes every observation exactly equal to the threshold.
    cut=np.searchsorted(sorted_c,thresholds,side='right')
    count=len(c)-cut;errors=prefix[-1]-prefix[cut]
    selected=np.arange(len(thresholds))
    if default and len(selected)>250:
        selected=np.unique(np.linspace(0,len(selected)-1,250).astype(int))
    return [dict(threshold=float(thresholds[i]),coverage=float(count[i]/len(c)),
                 risk=float(errors[i]/count[i]) if count[i] else None,
                 n_emitted=int(count[i]),n_supported=int(len(c))) for i in selected]


def _detection(score,active,supported):
    s=score[supported];y=active[supported].astype(int)
    record=dict(n=int(len(y)),n_active=int(y.sum()),n_inactive=int(len(y)-y.sum()),
                prevalence=float(y.mean()) if len(y) else None)
    if len(np.unique(y))<2:return dict(**record,auroc=None,average_precision=None,roc=[],pr=[])
    fpr,tpr,threshold=roc_curve(y,s);precision,recall,threshold_pr=precision_recall_curve(y,s)
    # Decimate curve points for a compact report; AUROC/AP use all samples.
    ridx=np.unique(np.linspace(0,len(fpr)-1,min(250,len(fpr))).astype(int))
    pidx=np.unique(np.linspace(0,len(precision)-1,min(250,len(precision))).astype(int))
    return dict(**record,auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),
        roc=[dict(fpr=float(fpr[i]),tpr=float(tpr[i])) for i in ridx],
        pr=[dict(recall=float(recall[i]),precision=float(precision[i])) for i in pidx])


def _rate(mask,event):
    n=int(mask.sum())
    return dict(n=int(n),rate=float(event[mask].mean()) if n else None)


def evaluate(logps,logjoint7,presence,thresholds,labels,times,config,ground_truth,outdir):
    """Write metrics + CSV + six figures; return report. Arrays stay unmodified."""
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);cfg=_cfg(config)
    logs=[np.asarray(logps[k],float) for k in KEYS];probs=[np.exp(l) for l in logs]
    logjoint7=np.asarray(logjoint7);presence=np.asarray(presence);thresholds=np.asarray(thresholds)
    labels=np.asarray(labels,int);times=np.asarray(times);J,K,N,T=logs[0].shape
    if (J,K)!=(4,6) or any(l.shape!=(J,K,N,T) for l in logs) or logjoint7.shape!=(4,7,N,T):raise ValueError('Aligned R3/K6 arrays required')
    if presence.shape!=(4,N,T) or thresholds.shape!=(4,N) or labels.shape!=(N,3):raise ValueError('Aligned trial arrays required')
    if not all(np.allclose(p.sum(1),1) for p in probs) or not np.allclose(np.exp(logjoint7).sum(1),1):raise ValueError('Normalized posteriors required')
    if not np.allclose(presence,1-np.exp(logjoint7[:,0]),atol=1e-10):raise ValueError('Presence must match joint seven-class posterior')
    records=[];csvrows=[]
    for strong,weak in SENSITIVITY:
        state,target,norm=activity_masks(ground_truth,cfg,strong,weak);components=[]
        for j in range(4):
            active=np.broadcast_to(state[j]==1,(N,T));inactive=np.broadcast_to(state[j]==0,(N,T));supported=active|inactive
            truth=np.full((N,T),-1,int)
            for r in range(3):
                mask=(state[j]==1)&(target[j]==r);truth[:,mask]=labels[:,r,None]-1
            active_scores={key:class_metrics(log[j][:,active],truth[active]) for key,log in zip(KEYS,logs)}
            submasks=dict(all_inactive=inactive,
                baseline_inactive=inactive&((times>=cfg.baseline_window_seconds[0])&(times<cfg.baseline_window_seconds[1]))[None,:],
                pre_first_onset_inactive=inactive&(times<cfg.target_onsets_seconds[0])[None,:],
                post_first_onset_inactive=inactive&(times>=cfg.target_onsets_seconds[0])[None,:])
            inactive_rates={};detection={};curves={}
            for mi,key in enumerate(KEYS):
                confidence=probs[mi][j].max(0);prediction=probs[mi][j].argmax(0)
                inactive_rates[key]={name:_rate(mask,confidence>=.9) for name,mask in submasks.items()}
                detection[key]=_detection(confidence,active,supported)
                curves[key]=coverage_risk(confidence,prediction,truth,supported)
            joint6=np.exp(logjoint7[j,1:]);event_joint=joint6.max(0)>=.9
            inactive_rates['null_joint_location_ge_0.9']={name:_rate(mask,event_joint) for name,mask in submasks.items()}
            inactive_rates['null_presence_gt_train_q95']={name:_rate(mask,presence[j]>thresholds[j,:,None]) for name,mask in submasks.items()}
            detection['null_presence']=_detection(presence[j],active,supported)
            curves['null_presence']=coverage_risk(presence[j],joint6.argmax(0),truth,supported)
            # M3 conditional class posterior equals M2; never duplicate it as another winner.
            if active.any():
                active_lp=logjoint7[j,1:][:,active];active_y=truth[active]
                joint_active_nll=float(-active_lp[active_y,np.arange(len(active_y))].mean())
            else:joint_active_nll=None
            emit=presence[j]>thresholds[j,:,None]
            right=joint6.argmax(0)==truth
            log_content=logsumexp(logjoint7[j,1:],axis=0)
            binary_y=active[supported].astype(float)
            binary_p=presence[j][supported]
            binary_logp=np.where(active,log_content,logjoint7[j,0])[supported]
            operating=dict(active_true_positive_rate=_rate(active,emit),
                active_correct_location_emission_rate=_rate(active,emit&right),
                active_emitted_location_accuracy=_rate(active&emit,right),
                supported_binary_brier=float(np.square(binary_p-binary_y).mean()),
                supported_binary_nll=float(-binary_logp.mean()),
                supported_active_prevalence=float(binary_y.mean()))
            component=dict(component=j,name=COMPONENTS[j],active_frames=int((state[j]==1).sum()),
                inactive_frames=int((state[j]==0).sum()),gray_frames=int((state[j]<0).sum()),
                active_metrics=active_scores,inactive_metrics=inactive_rates,detection=detection,
                coverage_risk=curves,null_operating_point=operating,null_joint_active_nll=joint_active_nll,
                null_presence_active_mean=float(presence[j][active].mean()) if active.any() else None,
                null_presence_inactive_mean=float(presence[j][inactive].mean()) if inactive.any() else None)
            components.append(component)
            for key,score in active_scores.items():
                if score:csvrows.append(dict(strong=strong,inactive=weak,component=COMPONENTS[j],method=key,**{k:v for k,v in score.items() if k!='reliability'}))
        records.append(dict(strong_threshold=strong,inactive_threshold=weak,components=components))
    fixed_windows=[];fixed_rows=[]
    for family,windows in [('Entry',cfg.entry_readout_windows_seconds),('Rank',cfg.rank_readout_windows_seconds)]:
        for r,window in enumerate(windows):
            j=0 if family=='Entry' else r+1;mask=(times>=window[0])&(times<window[1])
            y=np.broadcast_to(labels[:,r,None]-1,(N,int(mask.sum()))).reshape(-1)
            metrics={key:class_metrics(log[j][:,:,mask].reshape(6,-1),y) for key,log in zip(KEYS,logs)}
            fixed_windows.append(dict(name=f'{family} / T{r+1}',window=list(window),component=j,target=r+1,metrics=metrics))
            for key,score in metrics.items():fixed_rows.append(dict(window=f'{family} / T{r+1}',method=key,**{k:v for k,v in score.items() if k!='reliability'}))
    report=dict(original_prespecified_windows=fixed_windows,main_thresholds=dict(strong=.5,inactive=.1),sensitivity=records,
        definitions=dict(activity='Evaluation-only true source gains undergo exactly the configured baseline subtraction and smoothing; absolute value normalized separately by each source/rank peak. Active >= strong, inactive <= weak. Entry requires one strong rank and all other ranks weak; mixed/gray frames excluded. Rank uses its own source. Masks never enter fitting or prediction.',
        active_scores='Per-frame class scores over supported strong-active trial-frames only; M0/M1/M2 six-way conditionals. M3 six-way conditional is exactly M2, so not repeated. NLL from log domain, Brier=sum over six classes, circular error=min(|pred-true|,6-|pred-true|) location steps.',
        ece='10 fixed equal-width max-confidence bins, top-label calibration only; not animal confidence. All frames correlated; no iid significance or animal inference.',
        inactive='M0/M1/M2 event is conditional max location probability >=.9. M3 location event is max of six JOINT probabilities >=.9; separate M3 presence event is P(content)>train-readout-OOF baseline q95 threshold. These are different events, not interchangeable accuracy measures.',
        baseline='SIMULATION-ONLY assumption diagnostic. Prestimulus can contain imagined content; do not apply this null as a real absence label. Baseline-like null is not a verified absence model. Its prior .5 is a design choice, not calibrated true prevalence. Baseline-only and post-first-onset inactive subsets are reported separately; baseline rejection does not guarantee post-onset rejection.',
        detection='AUROC/AP label strong-active versus weak-inactive, exclude gray. M0/M1/M2 use max conditional posterior as diagnostic score, not a learned absence detector. M3 uses P(content). AP no-skill reference is subset active prevalence. Curves may be decimated for display; scalar metrics use all observations.',
        binary_presence_scores='Binary Brier=mean(Pcontent-active_indicator)^2 (not the two-class summed variant); binary NLL uses logsumexp of the six joint location logits or log null posterior without clipping. Evaluated on active+inactive support; depends on gain thresholds and true prevalence. Prior .5 is not necessarily that prevalence.',
        operating_point='At exactly the frozen strict P(content)>train q95 decision, report active TPR, correctly emitted location fraction of ALL active rows, and location accuracy among emitted active rows. False-alarm reductions must be read together with these power metrics.',
        original_windows='Original six Entry/own-Rank windows are also scored for M0/M1/M2, without truth-activity masking, preserving direct comparison to the previous version.',
        threshold='Strict P(content)>q95; threshold estimated from readout-only inner crossfit conditional on fixed outer-training upstream, not fully nested upstream refitting. No promised 5% held-out/OOD FPR.',
        coverage_risk='For each threshold, emit a location if max conditional (M0-M2) or P(content) (M3) exceeds threshold. Coverage=emitted/all active+inactive supported observations. Risk=wrong class OR emitted on inactive among emitted. All unique confidence cutpoints are computed with strict > and intact ties; at most 250 deterministic points retained for display. No threshold chosen using this evaluation.',
        display='Original row/trial 0, known outer fold 4. First three heatmaps conditional six-way; M3 six JOINT locations (sum=Pcontent), separate presence/null plot. All heatmaps .2..1; values below .2 bottom color; no truth gating.'),
        class_order=list(range(1,7)),joint_order=['null',1,2,3,4,5,6],n_trials=N)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'active_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    with (out/'original_window_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fixed_rows[0]));w.writeheader();w.writerows(fixed_rows)
    _plots(report,probs,logjoint7,presence,thresholds,labels,times,cfg,out)
    return report


def _plots(report,probs,logjoint,presence,thresholds,labels,times,cfg,out):
    main=report['sensitivity'][1]['components'];x=np.arange(4)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['balanced_accuracy','nll','brier','ece','entropy','circular_error_degrees'],
            ['Active balanced accuracy ↑','Active NLL (nats) ↓','Active multiclass Brier ↓','Active top-label ECE ↓','Active posterior entropy (nats)','Active circular error (degrees) ↓']):
        for mi,k in enumerate(KEYS):ax.bar(x+(mi-1)*.25,[c['active_metrics'][k][key] for c in main],width=.25,label=NAMES[mi],color=COLORS[mi])
        ax.set_xticks(x,COMPONENTS);ax.set_title(title);ax.set_ylim(bottom=0)
        if key=='balanced_accuracy':ax.set_ylim(0,1.05)
    axes.flat[0].legend(loc='lower left');fig.suptitle('SIMULATION-ONLY diagnostic: gain ≥ .5; other Entry ranks ≤ .1\nM3 conditional locations = M2; baseline-like null does NOT establish real content absence')
    fig.savefig(out/'active_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,subset in zip(axes.flat[:2],['baseline_inactive','post_first_onset_inactive']):
        keys=[*KEYS,'null_joint_location_ge_0.9','null_presence_gt_train_q95']
        for mi,key in enumerate(keys):
            vals=[c['inactive_metrics'][key][subset]['rate'] for c in main]
            ax.bar(x+(mi-2)*.16,[np.nan if v is None else v for v in vals],width=.16,label=([*NAMES[:3],'M3 joint location ≥ .9','M3 presence > train q95'][mi]),color=[*COLORS[:3],COLORS[3],'#777777'][mi])
        ax.set(title=subset.replace('_',' '),ylabel='False emission rate',ylim=(0,1.05));ax.set_xticks(x,COMPONENTS)
    for ax,key,title in zip(axes.flat[2:],['auroc','average_precision'],['Active vs inactive AUROC','Active vs inactive average precision']):
        for mi,k in enumerate([*KEYS,'null_presence']):ax.bar(x+(mi-1.5)*.2,[c['detection'][k][key] for c in main],width=.2,label=NAMES[mi],color=COLORS[mi])
        if key=='average_precision':ax.plot(x,[c['detection']['null_presence']['prevalence'] for c in main],'k_',ms=14,label='AP no-skill prevalence')
        ax.set(title=title,ylim=(0,1.05));ax.set_xticks(x,COMPONENTS)
    axes.flat[0].legend(fontsize=8);axes.flat[3].legend(fontsize=8)
    fig.suptitle('SIMULATION-ONLY: known generator low-gain versus strong-gain states (.5, .1)\nBaseline-like null is NOT a real absence label; post-onset threshold transfer fails')
    fig.savefig(out/'inactive_and_detection.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
    for j,c in enumerate(main):
        for mi,k in enumerate([*KEYS,'null_presence']):
            d=c['detection'][k]
            axes[j,0].plot([v['fpr'] for v in d['roc']],[v['tpr'] for v in d['roc']],color=COLORS[mi],label=NAMES[mi])
            axes[j,1].plot([v['recall'] for v in d['pr']],[v['precision'] for v in d['pr']],color=COLORS[mi],label=NAMES[mi])
        axes[j,0].plot([0,1],[0,1],'k:',lw=1);axes[j,1].axhline(c['detection']['null_presence']['prevalence'],color='k',ls=':',lw=1)
        axes[j,0].set(title=COMPONENTS[j]+' ROC',xlabel='Inactive false-positive rate',ylabel='Active true-positive rate',xlim=(0,1),ylim=(0,1))
        axes[j,1].set(title=COMPONENTS[j]+' precision–recall',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    axes[0,0].legend(fontsize=8);fig.suptitle('SIMULATION-ONLY discrimination, gray/mixed states excluded\nM3 model-assumed alternative vs baseline-like posterior; not real presence; PR dotted: prevalence')
    fig.savefig(out/'presence_roc_pr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,c in zip(axes.flat,main):
        for mi,k in enumerate([*KEYS,'null_presence']):
            curve=[v for v in c['coverage_risk'][k] if v['risk'] is not None]
            ax.plot([v['coverage'] for v in curve],[v['risk'] for v in curve],color=COLORS[mi],label=NAMES[mi])
        ax.set(title=c['name'],xlabel='Coverage of active + inactive observations',ylabel='Risk among emitted locations',xlim=(0,1),ylim=(0,1))
    axes.flat[0].legend(fontsize=8);fig.suptitle('SIMULATION-ONLY risk: emission on a known low-gain source counts as error\nNot valid absence scoring for imagined real content; no threshold selected on these curves')
    fig.savefig(out/'coverage_risk.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(18,10),sharex=True,sharey=True,layout='constrained')
    displayed=[*probs,np.exp(logjoint[:,1:])]
    for j in range(4):
        for mi,p in enumerate(displayed):
            ax=axes[j,mi];im=ax.imshow(p[j,:,0],origin='lower',aspect='auto',extent=(times[0],times[-1]+np.median(np.diff(times)),.5,6.5),vmin=.2,vmax=1,cmap='viridis')
            for onset in cfg.target_onsets_seconds:ax.axvline(onset,color='white',ls='--',alpha=.7,lw=.7)
            ax.set_yticks(range(1,7))
            if mi==0:ax.set_ylabel(COMPONENTS[j]+'\nLocation')
            if j==0:ax.set_title(NAMES[mi]+('\nJOINT location probabilities' if mi==3 else '\nSix exclusive-location conditional'))
            if j==3:ax.set_xlabel('Epoch time (s)')
    fig.colorbar(im,ax=axes,label='Probability (display .2–1; <.2 bottom color)',shrink=.8)
    fig.suptitle(f'SIMULATION-ONLY null-model diagnostic: fixed trial 0 / fold 4, labels {labels[0].tolist()}\nM3 six locations sum to model-assumed alternative mass; baseline-like does NOT mean no imagination')
    fig.savefig(out/'single_trial_four_methods.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    for j,ax in enumerate(axes):
        ax.plot(times,presence[j,0],color=COLORS[3],label='P(model alternative to baseline-like)');ax.plot(times,np.exp(logjoint[j,0,0]),color='.5',label='P(baseline-like null)')
        ax.axhline(thresholds[j,0],color='#C62828',ls=':',label='Train-only baseline q95 threshold')
        for onset in cfg.target_onsets_seconds:ax.axvline(onset,color='.6',ls='--',lw=.7)
        ax.set(title=COMPONENTS[j],ylim=(0,1.03),ylabel='Posterior mass')
    axes[0].legend(loc='upper left',fontsize=8);axes[-1].set_xlabel('Epoch time (s)')
    fig.suptitle('SIMULATION-ONLY trial 0: alternative versus baseline-like model posterior\nNot true presence/absence: prestimulus may contain imagined content; prior .5 is assumed')
    fig.savefig(out/'single_trial_presence.png',dpi=160);plt.close(fig)
