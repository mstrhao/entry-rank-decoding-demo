"""Primary optimization: six-location frame likelihood + readout-only calibration.

No null class, no absence labels, no generator-gain or time gate at inference.
"""
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import scipy,sklearn
CORE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CORE/'src'))
from jointdecode.config import Config
from jointdecode.pipeline import preprocess
from jointdecode.diagnostics import load_tree
from jointdecode.cli import save_tree,write_json
from calibrated_readout import fit_frame_readouts,predict_frame_readouts,crossfit_temperatures,apply_temperature


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(source,baseline,out,reference_m2=None):
    source=Path(source);baseline=Path(baseline);out=Path(out)
    if out.exists():raise FileExistsError('Immutable output already exists')
    out.mkdir(parents=True)
    cfg=Config(**json.loads((source/'config.json').read_text()));d=np.load(source/'simulation_inputs.npz',allow_pickle=False)
    X=preprocess(d['epochs'],cfg);y=d['target_labels']-1;folds=d['outer_fold'];old=np.load(baseline/'comparison_oof.npz',allow_pickle=False)
    np.testing.assert_array_equal(old['target_labels'],y+1);np.testing.assert_array_equal(old['outer_fold'],folds)
    files=[source/'config.json',source/'simulation_inputs.npz',baseline/'comparison_oof.npz']+[source/f'fold_{f}.{ext}' for f in range(5) for ext in ('json','npz')]
    before={('baseline/' if f.parent==baseline else 'upstream/')+f.name:digest(f) for f in files}
    raw=np.empty_like(old['logistic_logp']);cal=np.empty_like(raw);filled=np.zeros(len(y),int);audits=[]
    for f in range(5):
        m=load_tree(source/f'fold_{f}');tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
        np.testing.assert_array_equal(m['train_indices'],tr);np.testing.assert_array_equal(m['test_indices'],te)
        assert not set(d['groups'][tr])&set(d['groups'][te])
        T,diag,inner=crossfit_temperatures(X,y,folds,f,m,cfg);readouts=fit_frame_readouts(X[tr],y[tr],m,cfg)
        lp=predict_frame_readouts(m,readouts,X[te]);lc=apply_temperature(lp,T)
        np.testing.assert_array_equal(lp.argmax(1),lc.argmax(1));np.testing.assert_allclose(np.exp(lc).sum(1),1,atol=1e-12)
        raw[:,:,te,:]=lp;cal[:,:,te,:]=lc;filled[te]+=1
        save_tree(out/f'calibrated_fold_{f}',dict(readouts=readouts,temperatures=T,calibration=diag,inner_audit=inner,train_indices=tr,test_indices=te))
        audits.append(dict(fold=f,calibration=diag,temperatures=T.tolist(),content_shrinkage=[r['shrinkage'] for r in readouts],training_rows=[r['training_rows'] for r in readouts],n_floored_eigenvalues=[r['n_floored_eigenvalues'] for r in readouts]))
    assert np.all(filled==1)
    replay=None
    if reference_m2:
        reference=np.load(reference_m2,allow_pickle=False)['frame_gaussian_logp'];replay=float(np.max(np.abs(raw-reference)));np.testing.assert_allclose(raw,reference,atol=1e-10,rtol=1e-10)
    after={('baseline/' if f.parent==baseline else 'upstream/')+f.name:digest(f) for f in files};assert before==after
    logps=dict(logistic=old['logistic_logp'],mean_gaussian=old['bayesian_logp'],frame_gaussian=raw,calibrated_gaussian=cal)
    np.savez_compressed(out/'calibrated_oof.npz',**{k+'_logp':v for k,v in logps.items()},target_labels=y+1,times_seconds=np.asarray(cfg.times_seconds),trial_ids=d['trial_ids'],outer_fold=folds)
    write_json(out/'provenance.json',dict(method='frame_matched_gaussian_temperature_v1',source_run=source.name,source_sha256=before,source_unchanged=True,
        scripts_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if 'protocol' not in p.name},historical_source_commit='7b9fac5fc2199a1859bf6a8c2f853653d463b3b3',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),fold_audit=audits,
        frame_model_replay_max_logp_error=replay,calibration='4 whole-trial inner folds; readout-only cross-fit conditional on fixed outer-training supervised representation, NOT fully nested upstream validation. Original predefined class windows only; no presence/absence labels.',
        inference='One positive scalar temperature per component per fold; six exclusive-location conditional probabilities; all times including prestimulus retained. No null class or time gate.',
        limitation='Uniform six-location posterior does not identify no content versus equal coactivation or across-trial balancing; ideal centered cos/sin ring may cancel equal coactivation. Probabilities and signal magnitude do not prove imagination content.'))
    from evaluate_calibrated import evaluate
    return evaluate(logps,y+1,np.asarray(cfg.times_seconds),cfg,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--reference-m2');a=p.parse_args();print(json.dumps(run(a.source,a.baseline,a.output,a.reference_m2)))
