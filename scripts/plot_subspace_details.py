"""Read-only training-fold TG, template PCA, angles and directional template VAF."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'single-trial-decoding-demo'/'src'))
from jointdecode.diagnostics import load_tree


def template_pca(S):
    S=np.asarray(S,float);S=S-S.mean(axis=0,keepdims=True)
    U,s,Vt=np.linalg.svd(S,full_matrices=False)
    # Deterministic sign convention; no cross-space alignment is implied.
    for i in range(len(s)):
        if Vt[i,np.argmax(np.abs(Vt[i]))]<0: U[:,i]*=-1;Vt[i]*=-1
    energy=np.sum(s*s)
    return dict(scores=U[:,:2]*s[:2],explained_variance_ratio=s*s/energy if energy>0 else np.zeros_like(s),singular_values=s)


def space_relations(templates,bases):
    J=len(templates);vaf=np.full((J,J),np.nan);mean=np.zeros((J,J));pairs=[]
    for i,S in enumerate(templates):
        S=S-S.mean(axis=0,keepdims=True);energy=np.sum(S*S)
        for j,Q in enumerate(bases):
            if energy>0:vaf[i,j]=np.sum((S@Q)**2)/energy
            a=np.rad2deg(subspace_angles(bases[i],Q))
            mean[i,j]=a.mean() if len(a) else np.nan
            if i<j:pairs.append(dict(source=i,target=j,angles_degrees=np.sort(a).tolist(),source_dimension=bases[i].shape[1],target_dimension=Q.shape[1]))
    return vaf,mean,pairs


def write_csv(path,fields,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def generate(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=json.loads((run/'config.json').read_text());m=load_tree(run/'fold_0')
    assert m['outer_fold']==0
    S=m['joint']['templates'];Q=m['bases'];J=len(S);K=S.shape[1];R=J-1
    assert J==4 and K==6, 'This fixed presentation requires R=3 and K=6; use a separately configured layout otherwise.'
    names=['Entry']+[f'Rank {r}' for r in range(1,J)]
    pcas=[template_pca(x) for x in S];vaf,angle_mean,pairs=space_relations(S,Q)
    TG=m['nested']['TG'];tau=m['nested']['bin_times'];assert TG.shape==(R,R,len(tau),len(tau))
    assert all(q.shape[1]==K-1 for q in Q), 'This fixed plot expects the observed full K-1 dimensional bases.'
    # Existing nested TG contains pooled inner-validation class-balanced accuracy, not outer-test TG.
    fig,axes=plt.subplots(R,R,figsize=(10,9),sharex=True,sharey=True,layout='constrained')
    width=(tau[1]-tau[0])/2
    for r in range(R):
        for s in range(R):
            ax=axes[r,s];im=ax.imshow(TG[r,s],origin='lower',aspect='equal',extent=(tau[0]-width,tau[-1]+width,tau[0]-width,tau[-1]+width),vmin=.2,vmax=1,cmap='viridis',interpolation='nearest')
            ax.axhline(config['target_onsets_seconds'][r],color='white',lw=.7,ls='--');ax.axvline(config['target_onsets_seconds'][s],color='white',lw=.7,ls='--')
            ax.set(title=f'Train Rank {r+1} → test Rank {s+1}',xlabel='Test time (s)',ylabel='Train time (s)')
    fig.colorbar(im,ax=axes,label=f'Inner-validation balanced accuracy (display 0.2–1; values <0.2 use bottom color; chance = 1/{K})',shrink=.85)
    fig.suptitle('Fixed outer fold 0: training-only controlled TG\nTwo-frame bins; pooled inner-validation class hits, not outer-test evaluation',fontsize=12)
    fig.savefig(out/'training_tg.png',dpi=160);plt.close(fig)
    colors=plt.cm.tab10(np.arange(K))
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for j,ax in enumerate(axes.flat):
        p=pcas[j];xy=p['scores'];ev=p['explained_variance_ratio'];ax.axhline(0,color='gray',alpha=.25);ax.axvline(0,color='gray',alpha=.25)
        for k in range(K):
            ax.scatter(*xy[k],color=colors[k],s=70,label=f'Location {k+1}');ax.annotate(str(k+1),xy[k],xytext=(5,5),textcoords='offset points')
        ax.margins(.22);ax.set_aspect('equal',adjustable='datalim');ax.set(title=f'{names[j]} — PC1+PC2: {100*ev[:2].sum():.1f}%',xlabel=f'PC1 ({100*ev[0]:.1f}% template variance)',ylabel=f'PC2 ({100*ev[1]:.1f}% template variance)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=K)
    fig.suptitle('Fixed outer fold 0: six TRAINING location templates in two PCs\nEach panel has independent PC axes; not held-out activity or a temporal trajectory',fontsize=12)
    fig.savefig(out/'location_template_pca2.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    im=axes[0].imshow(angle_mean,vmin=0,vmax=90,cmap='magma');fig.colorbar(im,ax=axes[0],label='Mean principal angle (degrees)')
    axes[0].set(title='Full 5D spaces: mean principal angle',xticks=range(J),xticklabels=names,yticks=range(J),yticklabels=names)
    for i in range(J):
        for j in range(J):axes[0].text(j,i,f'{angle_mean[i,j]:.1f}',ha='center',va='center',color='white' if angle_mean[i,j]<55 else 'black')
    for n,pair in enumerate(pairs):
        angles=pair['angles_degrees'];axes[1].plot(np.full(len(angles),n),angles,'o',ms=5,alpha=.75)
        axes[1].plot([n,n],[min(angles),max(angles)],color='gray',lw=1)
    axes[1].set(xticks=range(len(pairs)),xticklabels=[f"{names[p['source']]}\nvs {names[p['target']]}" for p in pairs],ylim=(0,90),ylabel='All five principal angles (degrees)',title='Six pairs: complete angle spectra')
    im=axes[2].imshow(vaf,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=axes[2],label='Fraction of source template energy captured')
    axes[2].set(title='Directional TEMPLATE VAF (not unique variance)',xticks=range(J),xticklabels=names,yticks=range(J),yticklabels=names,xlabel='Target subspace',ylabel='Source centered template')
    for i in range(J):
        for j in range(J):axes[2].text(j,i,f'{vaf[i,j]:.3f}',ha='center',va='center',color='white' if vaf[i,j]<.6 else 'black')
    fig.suptitle('Fixed outer fold 0; common training-standardized neuron coordinates',fontsize=12)
    fig.savefig(out/'angles_and_template_vaf.png',dpi=160);plt.close(fig)
    write_csv(out/'pca_coordinates.csv',['component','location','PC1','PC2','PC1_variance_ratio','PC2_variance_ratio'],[dict(component=names[j],location=k+1,PC1=float(pcas[j]['scores'][k,0]),PC2=float(pcas[j]['scores'][k,1]),PC1_variance_ratio=float(pcas[j]['explained_variance_ratio'][0]),PC2_variance_ratio=float(pcas[j]['explained_variance_ratio'][1])) for j in range(J) for k in range(K)])
    write_csv(out/'principal_angles.csv',['space_i','space_j','angle_index_ascending','angle_degrees','dimension_i','dimension_j'],[dict(space_i=names[p['source']],space_j=names[p['target']],angle_index_ascending=k+1,angle_degrees=a,dimension_i=p['source_dimension'],dimension_j=p['target_dimension']) for p in pairs for k,a in enumerate(p['angles_degrees'])])
    write_csv(out/'template_vaf.csv',['source_template','target_subspace','VAF'],[dict(source_template=names[i],target_subspace=names[j],VAF=float(vaf[i,j])) for i in range(J) for j in range(J)])
    report=dict(fixed_outer_fold=0,component_names=names,dimensions=[q.shape[1] for q in Q],source_run=run.name,
                definitions=dict(TG_axes='[train_rank,test_rank,train_bin,test_bin]; native bins=2 frames; pooled inner-validation balanced accuracy; other target labels used only inside outer-training conditional analysis',PCA='Centered K x P training template in outer-training standardized ROI coordinates; separate SVD PCA per component, scores=U*s; variance ratio=s^2/sum(s^2); no trajectory and no aligned axes across components',principal_angles='All principal angles between full column-orthonormal row-space bases, sorted ascending; not computed on PCA2',VAF='VAF[i,j]=||center(S_i) Q_j||_F^2 / ||center(S_i)||_F^2; rows source template, columns target full subspace; diagonal 1; directional, not unique/additive variance and not held-out trial VAF'),
                pca_variance_ratios=[p['explained_variance_ratio'].tolist() for p in pcas],pca_2d_total_variance=[float(p['explained_variance_ratio'][:2].sum()) for p in pcas],principal_angle_pairs=pairs,template_vaf=vaf.tolist(),
                frozen_input_sha256={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['fold_0.json','fold_0.npz','config.json','provenance.json']},plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='One fixed fold selected in advance; descriptive simulation. TG not a significance map; PCA2 omits remaining dimensions. Low angles/high VAF do not prove shared source identity. Existing Rank-only source-allocation failure remains.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(generate(a.run,a.output)))
