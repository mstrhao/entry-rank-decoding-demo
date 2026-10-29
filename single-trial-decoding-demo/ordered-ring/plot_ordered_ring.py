"""Read-only geometry diagnostics: independent raw PCA and fixed label edges."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from run_ordered_ring import CORE
spec=importlib.util.spec_from_file_location('generic_plots',CORE.parent/'scripts'/'plot_subspace_details.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from jointdecode.diagnostics import load_tree


def topology(xy):
    xy=np.asarray(xy);angles=np.arctan2(xy[:,1],xy[:,0]);delta=(np.roll(angles,-1)-angles+np.pi)%(2*np.pi)-np.pi
    order=np.argsort(angles);steps=(np.roll(order,-1)-order)%len(order);cross=[]
    def orient(a,b,c):
        u=b-a;v=c-a;return float(u[0]*v[1]-u[1]*v[0])
    n=len(xy)
    for i in range(n):
        for j in range(i+1,n):
            if j==(i+1)%n or i==(j+1)%n:continue
            a,b=xy[i],xy[(i+1)%n];c,d=xy[j],xy[(j+1)%n]
            if orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0:cross.append([i+1,j+1])
    return dict(angular_order_labels=(order+1).tolist(),cyclic_order_up_to_rotation_reflection=bool(np.all(steps==1) or np.all(steps==n-1)),winding_number=float(delta.sum()/(2*np.pi)),nonadjacent_edge_crossings=cross)


def generate(run,out):
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(out);report=base.generate(run,out)
    model=load_tree(run/'fold_0');S=model['joint']['templates'];pcas=[base.template_pca(s) for s in S]
    names=['Entry','Rank 1','Rank 2','Rank 3'];colors=plt.cm.tab10(np.arange(6))
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for j,ax in enumerate(axes.flat):
        xy=pcas[j]['scores'];ev=pcas[j]['explained_variance_ratio'];loop=np.r_[np.arange(6),0]
        ax.plot(xy[loop,0],xy[loop,1],color='.55',lw=1.2,zorder=0)
        for k in range(6):
            ax.scatter(*xy[k],color=colors[k],s=75,label=f'Position {k+1}');ax.annotate(str(k+1),xy[k],xytext=(6,6),textcoords='offset points')
        ax.set_aspect('equal',adjustable='datalim');ax.margins(.25)
        ax.set(title=f'{names[j]} — PC1+PC2: {100*ev[:2].sum():.2f}%',xlabel=f'PC1 ({100*ev[0]:.2f}%)',ylabel=f'PC2 ({100*ev[1]:.2f}%)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6)
    fig.suptitle('Ordered-ring simulation: fixed fold 0 TRAINING templates\nIndependent unaligned PCA; fixed position edges 1→2→3→4→5→6→1',fontsize=12)
    fig.savefig(out/'location_template_pca2.png',dpi=160);plt.close(fig)
    leading=[np.linalg.svd(s-s.mean(axis=0,keepdims=True),full_matrices=False)[2][:2].T for s in S]
    vaf,mean,pairs=base.space_relations(S,leading)
    report['leading_two_diagnostic']=dict(label='Dimensionality-truncated diagnostic only; full source-template energy denominator, NOT inference/readout bases',angles=pairs,template_vaf=vaf.tolist())
    report['wrapper_source_sha256']=hashlib.sha256((Path(__file__).parent/'run_ordered_ring.py').read_bytes()).hexdigest()
    report['ring_plot_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['true_template_rank']=2;report['learned_dimensions_unchanged']=[q.shape[1] for q in model['bases']]
    config=json.loads((run/'config.json').read_text());tg=[];topologies=[];ratios=[];within=[]
    for fold in range(5):
        m=load_tree(run/f'fold_{fold}');t=m['nested']['bin_times'];M=m['nested']['TG'];early=[(t>=o+.2)&(t<o+.6) for o in config['target_onsets_seconds']];late=(t>=4.5)&(t<5.5)
        within.extend(float(M[r,r][np.ix_(late,late)].mean()) for r in range(3))
        blocks={}
        for a,b in [('early','early'),('early','late'),('late','early'),('late','late')]:
            blocks[a+'_'+b]=[float(M[r,s][np.ix_(early[r] if a=='early' else late,early[s] if b=='early' else late)].mean()) for r in range(3) for s in range(3) if r!=s]
        tg.append(blocks);pp=[base.template_pca(s) for s in m['joint']['templates']];ratios.append([float(p['explained_variance_ratio'][:2].sum()) for p in pp]);topologies.append([topology(p['scores']) for p in pp])
    report['within_rank_TG_late_late_mean']=float(np.mean(within));report['shared_TG_mean']={k:float(np.mean([d[k] for d in tg])) for k in tg[0]}
    report['shared_TG_each_fold_direction']=tg;report['all_fold_pca2_ratios']=ratios;report['all_fold_topology']=topologies
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args();print(json.dumps(generate(args.run,args.output)))
