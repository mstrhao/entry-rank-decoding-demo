"""Frozen-upstream Bayesian ablation; ground-truth gains are evaluation-only."""
import argparse,json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import scipy,sklearn
CORE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CORE/'src'))
from jointdecode.config import Config
from jointdecode.pipeline import preprocess
from jointdecode.diagnostics import load_tree
from jointdecode.cli import save_tree,write_json
from noise_matched import fit_frame_and_null,predict_frame_and_null,crossfit_thresholds
from gaussian_readout import predict_logposterior


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(source,baseline,out):
    source=Path(source);baseline=Path(baseline);out=Path(out)
    if out.exists():raise FileExistsError('Immutable output already exists')
    out.mkdir(parents=True)
    cfg=Config(**json.loads((source/'config.json').read_text()));d=np.load(source/'simulation_inputs.npz',allow_pickle=False)
    X=preprocess(d['epochs'],cfg);y=d['target_labels']-1;folds=d['outer_fold'];old=np.load(baseline/'comparison_oof.npz',allow_pickle=False)
    np.testing.assert_array_equal(old['target_labels'],y+1);np.testing.assert_array_equal(old['outer_fold'],folds)
    files=[source/'config.json',source/'simulation_inputs.npz',baseline/'comparison_oof.npz']+[source/f'fold_{f}.{ext}' for f in range(5) for ext in ('json','npz')]+[baseline/f'gaussian_fold_{f}.{ext}' for f in range(5) for ext in ('json','npz')]
    hashes={('baseline/' if f.parent==baseline else 'upstream/')+f.name:digest(f) for f in files}
    lp=np.empty_like(old['logistic_logp']);lj=np.empty((4,7,len(y),X.shape[-1]));pc=np.empty((4,len(y),X.shape[-1]));thresholds=np.empty((4,len(y)));filled=np.zeros(len(y),int);audits=[]
    for f in range(5):
        model=load_tree(source/f'fold_{f}');tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
        np.testing.assert_array_equal(model['train_indices'],tr);np.testing.assert_array_equal(model['test_indices'],te)
        assert not set(d['groups'][tr])&set(d['groups'][te])
        threshold,caldiag,calaudit=crossfit_thresholds(X,y,folds,f,model,cfg)
        fitted=fit_frame_and_null(X[tr],y[tr],model,cfg)
        old_fit=load_tree(baseline/f'gaussian_fold_{f}')['readouts']
        mean_errors=[float(np.max(np.abs(a['class_means']-b['class_means']))) for a,b in zip(fitted['content'],old_fit)]
        assert max(mean_errors)<1e-10
        a,b,c=predict_frame_and_null(model,fitted,X[te])
        np.testing.assert_allclose(np.exp(a).sum(1),1,atol=1e-12);np.testing.assert_allclose(np.exp(b).sum(1),1,atol=1e-12)
        # Conditional location posterior is exactly the content model, not changed by null.
        np.testing.assert_allclose(b[:,1:]-scipy.special.logsumexp(b[:,1:],axis=1,keepdims=True),a,rtol=1e-10,atol=1e-10)
        lp[:,:,te,:]=a;lj[:,:,te,:]=b;pc[:,te,:]=c;thresholds[:,te]=threshold[:,None];filled[te]+=1
        save_tree(out/f'optimized_fold_{f}',dict(readouts=fitted,thresholds=threshold,calibration=caldiag,inner_audit=calaudit,train_indices=tr,test_indices=te))
        audits.append(dict(fold=f,class_mean_difference_vs_windowmean=mean_errors,thresholds=threshold.tolist(),calibration=caldiag,content_training_rows=[m['training_rows'] for m in fitted['content']],null_training_rows=[m['training_rows'] for m in fitted['null']],
                           content_shrinkage=[m['shrinkage'] for m in fitted['content']],null_shrinkage=[m['shrinkage'] for m in fitted['null']],content_n_floored=[m['n_floored_eigenvalues'] for m in fitted['content']],null_n_floored=[m['n_floored_eigenvalues'] for m in fitted['null']]))
    assert np.all(filled==1)
    after={('baseline/' if f.parent==baseline else 'upstream/')+f.name:digest(f) for f in files};assert hashes==after
    logps=dict(logistic=old['logistic_logp'],mean_gaussian=old['bayesian_logp'],frame_gaussian=lp)
    np.savez_compressed(out/'optimized_oof.npz',**{k+'_logp':v for k,v in logps.items()},logjoint7=lj,presence=pc,thresholds=thresholds,target_labels=y+1,times_seconds=np.asarray(cfg.times_seconds),outer_fold=folds,trial_ids=d['trial_ids'])
    provenance=dict(method='frame_matched_gaussian_plus_baseline_null_v1',source_run=source.name,prior_content=.5,source_sha256=hashes,source_unchanged=True,
        script_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},historical_source_commit='7b9fac5fc2199a1859bf6a8c2f853653d463b3b3',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),outer_folds=audits,
        calibration='Four existing outer-training folds, whole-trial readout-only cross-fit conditional on fixed upstream/scaler. Not fully nested representation validation. q95 baseline, strict >; final refit/OOD FPR not guaranteed.',
        activity_threshold_pairs=[[.25,.05],[.5,.1],[.75,.15]],activity_primary_pair=[.5,.1],
        limitations=['Null is baseline-like distribution, not proof of absent internal content.','Baseline subtraction uses the same baseline window and can suppress baseline noise relative to later inactive epochs.','Prior content .5 is an explicit design assumption, not estimated true prevalence.','No time/gain hard gate. Ground truth is read only after every fit and prediction is completed.'])
    write_json(out/'provenance.json',provenance)
    # Evaluation-only: never pass this object to any fitting/prediction function.
    gt=np.load(source/'ground_truth.npz',allow_pickle=False)
    write_json(out/'evaluation_provenance.json',dict(ground_truth_sha256=digest(source/'ground_truth.npz'),use='Evaluation only, opened after all fits/predictions; no threshold selection from gain truth',activity_threshold_pairs=[[.25,.05],[.5,.1],[.75,.15]]))
    from evaluate_optimized import evaluate
    return evaluate(logps,lj,pc,thresholds,y+1,np.asarray(cfg.times_seconds),cfg,gt,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.source,a.baseline,a.output)))
