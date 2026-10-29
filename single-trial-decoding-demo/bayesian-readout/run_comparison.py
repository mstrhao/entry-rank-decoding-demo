"""Readout-only comparison using frozen upstream models; never retrains subspaces."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import scipy,sklearn,platform
from scipy.special import log_softmax
CORE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CORE/'src'))
from jointdecode.config import Config
from jointdecode.pipeline import preprocess
from jointdecode.readout import predict_fold
from jointdecode.diagnostics import load_tree
from jointdecode.cli import save_tree,write_json
from gaussian_readout import fit_bayesian_readouts,predict_bayesian_fold


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(source,out):
    source=Path(source);out=Path(out)
    if out.exists():raise FileExistsError('Immutable output directory already exists')
    out.mkdir(parents=True)
    cfg=Config(**json.loads((source/'config.json').read_text()))
    inputs=np.load(source/'simulation_inputs.npz',allow_pickle=False);saved=np.load(source/'oof.npz',allow_pickle=False)
    X=preprocess(inputs['epochs'],cfg);y=inputs['target_labels']-1;folds=inputs['outer_fold']
    assert np.array_equal(saved['target_labels'],y+1) and np.array_equal(saved['outer_fold'],folds)
    lp=np.empty_like(saved['probabilities']);bp=np.empty_like(lp);filled=np.zeros(len(X),int);audit=[]
    source_files=['config.json','simulation_inputs.npz','oof.npz']+[f'fold_{f}.{ext}' for f in range(5) for ext in ('json','npz')]
    before={f:digest(source/f) for f in source_files}
    for f in range(5):
        model=load_tree(source/f'fold_{f}');tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
        assert np.array_equal(model['train_indices'],tr) and np.array_equal(model['test_indices'],te)
        assert not set(inputs['groups'][tr])&set(inputs['groups'][te])
        logits,p,_=predict_fold(model,X[te]);np.testing.assert_array_equal(p,saved['probabilities'][:,:,te,:])
        lp[:,:,te,:]=log_softmax(logits,axis=1)
        fitted=fit_bayesian_readouts(X[tr],y[tr],model,cfg)
        bp[:,:,te,:]=predict_bayesian_fold(model,fitted,X[te]);filled[te]+=1
        save_tree(out/f'gaussian_fold_{f}',dict(readouts=fitted,train_indices=tr,test_indices=te))
        audit.append(dict(fold=f,train_trials=len(tr),test_trials=len(te),baseline_replay_max_error=float(np.max(np.abs(p-saved['probabilities'][:,:,te,:]))),
                          component_training_rows=[r['training_rows'] for r in fitted],equal_weights=[r['equal_sample_weights'] for r in fitted],shrinkage=[r['shrinkage'] for r in fitted],n_floored_eigenvalues=[r['n_floored_eigenvalues'] for r in fitted],minimum_raw_covariance_eigenvalue=[float(r['raw_eigenvalues'].min()) if len(r['raw_eigenvalues']) else None for r in fitted]))
    assert np.all(filled==1) and np.isfinite(bp).all()
    np.testing.assert_allclose(np.exp(bp).sum(axis=1),1,atol=1e-12)
    after={f:digest(source/f) for f in source_files};assert before==after
    np.savez_compressed(out/'comparison_oof.npz',logistic_logp=lp,bayesian_logp=bp,logistic_probabilities=np.exp(lp),bayesian_probabilities=np.exp(bp),target_labels=y+1,times_seconds=saved['times_seconds'],trial_ids=saved['trial_ids'],outer_fold=folds)
    provenance=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),method='gaussian_plugin_bayes_lw_v1',source_run=source.name,source_file_sha256=before,source_unchanged_after_comparison=True,
        script_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},fold_audit=audit,prior=[1/6]*6,
        likelihood='Training-window-mean class Gaussian; shared LedoitWolf pooled within-class residual covariance; same latent scaler as logistic',
        inference='Plug-in class posterior; no integration over parameters; no test calibration; same per-frame upstream coordinates',
        feature_scale_mismatch='Both methods train on window means but predict single frames. Gaussian covariance of means is not a frame-noise model; calibration may worsen.',
        source_inference_commit='779a6d021aba8095264d6e2880e5abcb4da82029')
    write_json(out/'provenance.json',provenance)
    from evaluate_comparison import evaluate
    report=evaluate(np.exp(lp),np.exp(bp),lp,bp,y+1,saved['times_seconds'],cfg,out)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();print(json.dumps(run(args.source,args.output)))
