"""Separate shared neural representation reconstruction from readout label scores."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CORE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CORE/'src'))
from jointdecode.config import Config,frame_mask
from jointdecode.pipeline import preprocess
from jointdecode.diagnostics import load_tree


def reconstruction_scores(train,test,A,bases):
    """Arrays [trial,neuron,time], already outer-training baseline standardized.

A and bases are frozen. Window-specific mean uses training rows only, with no
labels. This affine recentering is a diagnostic, NOT a new decoder forward pass.
Return SSE/SST sufficient statistics, not averages of fold ratios.
"""
    train=np.asarray(train,float);test=np.asarray(test,float)
    mu=train.mean(axis=(0,2));W=np.moveaxis(test,1,-1)-mu
    Q=np.concatenate(bases,axis=1);cuts=np.cumsum([0]+[b.shape[1] for b in bases]);coords=W@A
    blocks=[coords[...,cuts[j]:cuts[j+1]]@q.T for j,q in enumerate(bases)]
    full=sum(blocks);recon={'Joint ridge':full}
    for j,b in enumerate(blocks):recon[f'Only {j}']=b;recon[f'Drop {j}']=full-b
    U,s,_=np.linalg.svd(Q,full_matrices=False);tol=max(Q.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    U=U[:,s>tol];recon['Orthogonal union']=W@U@U.T
    sst=float(np.sum(W*W));sse={name:float(np.sum((W-pred)**2)) for name,pred in recon.items()}
    return dict(sst=sst,sse=sse,union_dimension=U.shape[1],component_dimensions=[q.shape[1] for q in bases],training_reference_mean=mu)


def pooled_reconstruction(records):
    sst=sum(r['sst'] for r in records)
    if sst<=0:raise ValueError('R2 undefined for zero reference energy')
    scores={name:1-sum(r['sse'][name] for r in records)/sst for name in records[0]['sse']}
    return dict(reference_sst=sst,r2=scores,drop_block_delta_r2={str(j):scores['Joint ridge']-scores[f'Drop {j}'] for j in range(4)})


def circular_error(predicted,true,K=6):
    delta=np.abs(np.asarray(predicted)-np.asarray(true));steps=np.minimum(delta,K-delta)
    return steps*(360./K)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(source,comparison,out):
    source=Path(source);comparison=Path(comparison);out=Path(out)
    if out.exists():raise FileExistsError('Immutable output already exists')
    out.mkdir(parents=True)
    cfg=Config(**json.loads((source/'config.json').read_text()));data=np.load(source/'simulation_inputs.npz',allow_pickle=False)
    X=preprocess(data['epochs'],cfg);folds=data['outer_fold'];records=[];windows=list(cfg.entry_readout_windows_seconds)+[cfg.rank_readout_windows_seconds[0]]
    assert all(tuple(w)==tuple(windows[-1]) for w in cfg.rank_readout_windows_seconds)
    names=['Entry window T1','Entry window T2','Entry window T3','Rank late window'];bywindow=[[] for _ in windows]
    sourcefiles=[source/'config.json',source/'simulation_inputs.npz',comparison/'comparison_oof.npz']+[source/f'fold_{f}.{ext}' for f in range(5) for ext in ('npz','json')]
    before={('comparison/' if f.parent==comparison else 'upstream/')+f.name:digest(f) for f in sourcefiles}
    for fold in range(5):
        m=load_tree(source/f'fold_{fold}');tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
        np.testing.assert_array_equal(m['train_indices'],tr);np.testing.assert_array_equal(m['test_indices'],te)
        Z=(X-m['center'][None,:,None])/m['scale'][None,:,None]
        for i,window in enumerate(windows):
            mask=frame_mask(cfg.times_seconds,window);rec=reconstruction_scores(Z[tr][:,:,mask],Z[te][:,:,mask],m['A'],m['bases'])
            rec.pop('training_reference_mean');bywindow[i].append(rec)
            records.append(dict(fold=fold,window=names[i],**rec))
    neural=[dict(name=name,window=list(window),**pooled_reconstruction(rr)) for name,window,rr in zip(names,windows,bywindow)]
    d=np.load(comparison/'comparison_oof.npz',allow_pickle=False);np.testing.assert_array_equal(d['target_labels'],data['target_labels']);np.testing.assert_array_equal(d['outer_fold'],folds)
    readout=[]
    for j,r,window in [(0,r,cfg.entry_readout_windows_seconds[r]) for r in range(3)]+[(r+1,r,cfg.rank_readout_windows_seconds[r]) for r in range(3)]:
        mask=frame_mask(cfg.times_seconds,window);y=d['target_labels'][:,r];name=f"{'Entry' if j==0 else 'Rank'} / T{r+1}"
        for method,key in [('Logistic','logistic'),('Gaussian Bayes','bayesian')]:
            lp=d[key+'_logp'][j][:,:,mask];prob=d[key+'_probabilities'][j][:,:,mask];pred=prob.argmax(0)+1
            nll=float(-lp[y-1,np.arange(len(y)),:].mean());err=circular_error(pred,y[:,None])
            readout.append(dict(window=name,method=method,mean_absolute_circular_error_degrees=float(err.mean()),
                                logscore_gain_vs_uniform_nats=float(np.log(6)-nll),nll_nats=nll))
    after={('comparison/' if f.parent==comparison else 'upstream/')+f.name:digest(f) for f in sourcefiles};assert before==after
    report=dict(neural_representation=neural,neural_fold_statistics=records,readout_scores=readout,
        definitions=dict(neural_level='Identical upstream for both readouts. These scores cannot distinguish Logistic from Gaussian Bayes.',
            reference='Per outer fold and fixed window, mu is mean of outer-training standardized neural data over training trials and window frames, without labels; SST=sum_test||Z-mu||^2.',
            operational_reconstruction='W=Z_test-mu; frozen joint coordinates W A; reconstructed Zhat=mu+W A Q^T. Diagnostic window recentering only; original decoder predictions unchanged.',
            r2='1-pooled_test_SSE/pooled_test_SST, pool sufficient statistics across folds, never mean fold ratios. Negative values retained. Same input is encoded/reconstructed; NOT independent or future-neural prediction.',
            orthogonal_union='Training Q concatenation orthonormalized by SVD, same learned union; mu+W U U^T. Energy VAF/R2 for this centered orthogonal projection, distinct from operational ridge reconstruction.',
            component='Only j uses its frozen joint-coordinate block, not a separately refitted projection. Drop j subtracts that block without refitting. Delta=R2_full-R2_drop_j; NOT unique information or additive variance partition.',
            label_logscore='log(6)-NLL (nats per frame), improvement against explicit uniform prior baseline. Can be negative. Not neural variance, mutual information, Bayes factor, or parameter evidence.',
            circular_error='Argmax class shortest circular distance on the fixed 1..6 ring, degrees in [0,180]. Per frame then averaged; only same six predefined target-present windows.',
            distinction='No posterior-weighted neural-template reconstruction or AIC/BIC; those would require new likelihood/observation assumptions.'),
        provenance=dict(source_run=source.name,comparison_run=comparison.name,source_sha256=before,source_unchanged=True,
                        script_sha256=digest(__file__),upstream_and_readout_commit='c25e511744a056244a48b99c681ac88564cfd45b'))
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(out/'readout_scores.csv',readout)
    write_csv(out/'neural_reconstruction.csv',[dict(window=r['name'],model=k,r2=v) for r in neural for k,v in r['r2'].items()])
    write_csv(out/'drop_block_delta.csv',[dict(window=r['name'],component=['Entry','Rank 1','Rank 2','Rank 3'][int(k)],delta_r2=v) for r in neural for k,v in r['drop_block_delta_r2'].items()])
    plots(report,out);return report


def plots(report,out):
    neural=report['neural_representation'];names=[x['name'] for x in neural];x=np.arange(4)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for off,key,color in [(-.18,'Joint ridge','#2878B5'),(.18,'Orthogonal union','#777777')]:
        axes[0].bar(x+off,[r['r2'][key] for r in neural],width=.34,label=key,color=color)
    axes[0].set_title('Whole learned union: input reconstruction');axes[0].legend(fontsize=8);axes[0].set_ylabel('Held-out reconstruction R²')
    for j,label in enumerate(['Entry','Rank 1','Rank 2','Rank 3']):
        off=(j-1.5)*.2
        axes[1].bar(x+off,[r['r2'][f'Only {j}'] for r in neural],width=.19,label=label)
        axes[2].bar(x+off,[r['drop_block_delta_r2'][str(j)] for r in neural],width=.19,label=label)
    axes[1].set_title('Single block, joint coordinates frozen');axes[1].set_ylabel('Reconstruction R² (not additive)')
    axes[2].set_title('Drop-block ΔR² = full − ablated');axes[2].set_ylabel('Conditional drop, no refit / not unique variance');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,names,rotation=30,ha='right');ax.axhline(0,color='.3',lw=.7)
    fig.suptitle('SHARED upstream representation: identical for Logistic and Gaussian Bayes\nTraining-mean reference; same X encoded and reconstructed, NOT prediction of independent neural data',fontsize=12)
    fig.savefig(out/'neural_reconstruction.png',dpi=160);plt.close(fig)
    rows=report['readout_scores'];windows=list(dict.fromkeys(r['window'] for r in rows));x=np.arange(6)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for mi,method in enumerate(['Logistic','Gaussian Bayes']):
        rr=[r for r in rows if r['method']==method]
        for ax,key in zip(axes,['mean_absolute_circular_error_degrees','logscore_gain_vs_uniform_nats']):ax.bar(x+(mi-.5)*.36,[r[key] for r in rr],width=.36,label=method)
    axes[0].set(title='Circular decoding error ↓',ylabel='Mean shortest argmax error (degrees)')
    axes[1].set(title='Label log-score improvement over uniform ↑',ylabel='log(6) − NLL (nats/frame; not neural VAF)')
    for ax in axes:ax.set_xticks(x,windows,rotation=30,ha='right');ax.axhline(0,color='.3',lw=.7)
    axes[0].legend();fig.suptitle('READOUT level: same frozen OOF trials, six predefined target-present windows')
    fig.savefig(out/'readout_extended_scores.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.source,a.comparison,a.output)))
