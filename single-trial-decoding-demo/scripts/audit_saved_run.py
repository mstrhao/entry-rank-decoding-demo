"""Read-only audit of a frozen run, writing into a new directory only."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jointdecode.diagnostics import load_tree,template_attribution,window_cross_target_scores,source_activity
from jointdecode.cli import write_json,source_hash


def write_csv(path,records):
    if not records: return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records)


def audit(run,out):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=json.loads((run/'config.json').read_text())
    with np.load(run/'ground_truth.npz',allow_pickle=False) as z: truth={k:z[k] for k in ('templates','entry_gain','rank_gain')}
    models=[load_tree(run/f'fold_{i}') for i in range(5)]; attribution=sum([template_attribution(m,truth) for m in models],[])
    windows={f'entry_{i+1}':w for i,w in enumerate(config['entry_readout_windows_seconds'])}
    windows.update({f'rank_{i+1}':w for i,w in enumerate(config['rank_readout_windows_seconds'])})
    with np.load(run/'oof.npz',allow_pickle=False) as z:
        rows=window_cross_target_scores(z['probabilities'],z['target_labels'],z['times_seconds'],windows)
    write_csv(out/'source_attribution.csv',attribution);write_csv(out/'window_cross_target.csv',rows)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir()) if p.name in ['oof.npz','ground_truth.npz','config.json','provenance.json'] or p.name.startswith('fold_')}
    write_json(out/'diagnostics.json',dict(schema_version=1,input_run=run.name,input_sha256=hashes,
        diagnostic_source_sha256=source_hash(),source_activity=source_activity(truth),
        interpretation='Absent source recovery is null; unused_basis_alignment is not recovery. Angles omit unmatched directions; projector distance includes them. One seed, descriptive only.',
        source_attribution=attribution,window_cross_target=rows))
    plot(attribution,rows,truth,out,run.name)
    return attribution,rows


def plot(attribution,rows,truth,out,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    J=len(truth['templates']);names=['Entry']+[f'R{i}' for i in range(1,J)]
    active=[a['source_active'] for a in source_activity(truth)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title in zip(axes,['signed_template_cosine','estimated_span_truth_energy_capture'],['Template cosine','Truth energy captured by learned span']):
        matrix=np.full((J,J),np.nan)
        for i in range(J):
            for j in range(J):
                vals=[a[key] for a in attribution if a['learned_space']==i and a['truth_source']==j and a[key] is not None]
                if vals:matrix[i,j]=np.mean(vals)
        cmap=plt.get_cmap('coolwarm' if key=='signed_template_cosine' else 'viridis').copy();cmap.set_bad('lightgray')
        im=ax.imshow(matrix,cmap=cmap,vmin=-1 if key=='signed_template_cosine' else 0,vmax=1);fig.colorbar(im,ax=ax)
        for i in range(J):
            for j in range(J):ax.text(j,i,f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else ('absent' if not active[j] else 'undefined'),ha='center',va='center',fontsize=8)
        ax.set(xticks=range(J),yticks=range(J),xticklabels=[n if a else n+'\nABSENT' for n,a in zip(names,active)],yticklabels=names,xlabel='True source (fold-standardized)',ylabel='Learned template',title=title)
    fig.suptitle(name+' — mean of five fold diagnostics, not independent replicates');fig.tight_layout();fig.savefig(out/'source_attribution.png',dpi=160);plt.close(fig)
    wins=list(dict.fromkeys(r['window_id'] for r in rows));fig,axes=plt.subplots(1,len(wins),figsize=(3*len(wins),4),squeeze=False)
    for ax,win in zip(axes[0],wins):
        m=np.array([[next(r['balanced_accuracy'] for r in rows if r['window_id']==win and r['decoder']==j and r['scoring_target']==k+1) for k in range(J-1)] for j in range(J)])
        ax.imshow(m,vmin=0,vmax=1,cmap='viridis')
        for j in range(J):
            for k in range(J-1):ax.text(k,j,f'{m[j,k]:.3f}',ha='center',va='center',color='white' if m[j,k]<.5 else 'black',fontsize=8)
        ax.set(xticks=range(J-1),xticklabels=[f'T{k+1}' for k in range(J-1)],yticks=range(J),yticklabels=names,title=win,xlabel='Scoring target')
    fig.suptitle(name+' — pooled OOF framewise balanced accuracy');fig.tight_layout();fig.savefig(out/'cross_target_ba.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.run,a.out)
