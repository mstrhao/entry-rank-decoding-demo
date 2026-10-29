"""Portable command line, write-once run directories, explicit provenance."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from .config import Config
from .simulation import simulate
from .pipeline import run_oof_decoding
from .evaluation import recovery,plots


def jsonable(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [jsonable(v) for v in obj]
    return obj


def write_json(path,data):
    Path(path).write_text(json.dumps(jsonable(data),indent=2,allow_nan=False)+'\n')


def save_tree(path,tree):
    """No pickle: numeric arrays in NPZ with recursive JSON structure."""
    arrays={}
    def encode(obj,prefix='root'):
        if isinstance(obj,np.ndarray):
            if obj.dtype.hasobject: return {'__list__':obj.tolist()}
            arrays[prefix]=obj; return {'__array__':prefix}
        if isinstance(obj,dict): return {k:encode(v,prefix+'/'+str(k)) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)): return [encode(v,prefix+'/'+str(i)) for i,v in enumerate(obj)]
        return jsonable(obj)
    structure=encode(tree); np.savez_compressed(str(path)+'.npz',**arrays); write_json(str(path)+'.json',structure)


def source_hash():
    h=hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')): h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def main(argv=None):
    ap=argparse.ArgumentParser(description='Joint static subspace decoding; simulation only by default')
    ap.add_argument('--output',required=True,help='New directory; existing output is never overwritten')
    ap.add_argument('--mode',choices=['baseline_subtract_smooth7','baseline_subtract_unsmoothed','as_is'],default=None)
    ap.add_argument('--ranks',type=int,choices=[2,3],default=3)
    ap.add_argument('--epochs'); ap.add_argument('--labels'); ap.add_argument('--config'); ap.add_argument('--folds'); ap.add_argument('--groups'); ap.add_argument('--trial-ids')
    ap.add_argument('--control',choices=['positive','noise','shuffle','entry-only','rank-only','overlap-high','noise-high'],default='positive')
    ap.add_argument('--threads',type=int,default=1)
    args=ap.parse_args(argv); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    cfg=Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    if args.epochs:
        if not args.labels or not args.config: ap.error('External epochs require --labels and explicit --config')
        cfg=replace(cfg,preprocessing=args.mode or cfg.preprocessing)
        data=dict(epochs=np.load(args.epochs,allow_pickle=False),target_labels=np.load(args.labels,allow_pickle=False))
        for flag,key in [('folds','outer_fold'),('groups','groups'),('trial_ids','trial_ids')]:
            if getattr(args,flag): data[key]=np.load(getattr(args,flag),allow_pickle=False)
    else:
        cfg=replace(cfg,preprocessing=args.mode or 'baseline_subtract_smooth7',input_processing_declaration='Raw additive simulation, no prior baseline subtraction or smoothing')
        if args.ranks==2:
            cfg=replace(cfg,target_onsets_seconds=cfg.target_onsets_seconds[:2],entry_readout_windows_seconds=cfg.entry_readout_windows_seconds[:2],rank_readout_windows_seconds=cfg.rank_readout_windows_seconds[:2])
        kwargs={}
        if args.control=='noise': kwargs.update(entry_amplitude=0.,rank_amplitude=0.)
        elif args.control=='entry-only': kwargs.update(rank_amplitude=0.)
        elif args.control=='rank-only': kwargs.update(entry_amplitude=0.)
        elif args.control=='overlap-high': kwargs.update(overlap=.7)
        elif args.control=='noise-high': kwargs.update(noise_sd=2.)
        data=simulate(cfg,**kwargs)
        if args.control=='shuffle':
            # Whole label tuples permuted; fold allocation is fixed before the null transformation.
            permutation=np.random.default_rng(cfg.seed_split+1).permutation(len(data['target_labels']))
            data['target_labels']=data['target_labels'][permutation]
            data['label_row_permutation']=permutation
        np.savez_compressed(out/'ground_truth.npz',**data['ground_truth'])
        np.savez_compressed(out/'simulation_inputs.npz',**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
        write_json(out/'simulation.json',data['simulation_parameters'])
    write_json(out/'config.json',cfg.to_dict())
    import scipy,sklearn,matplotlib
    provenance=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,source_sha256=source_hash(),control=args.control,threads=args.threads,
                    input_sha256={k:hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest() for k in ['epochs','target_labels']},
                    caveat='Softmax outputs are not Bayesian posterior or demonstrated calibrated probabilities.')
    try: provenance['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError: provenance['git_commit']=None
    write_json(out/'provenance.json',provenance)
    start=time.monotonic()
    def callback(f,model):
        save_tree(out/f'fold_{f}',model)
        print(json.dumps(dict(fold=f,elapsed_seconds=round(time.monotonic()-start,2),converged=model['joint']['converged'],profile_optimizers_success=model['joint']['all_profile_optimizers_success'],iterations=len(model['joint']['objective_history']))),flush=True)
    try:
        with threadpool_limits(args.threads):
            result=run_oof_decoding(data['epochs'],data['target_labels'],cfg,trial_ids=data.get('trial_ids'),groups=data.get('groups'),outer_fold=data.get('outer_fold'),fold_callback=callback)
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}; np.savez_compressed(out/'oof.npz',**arrays)
        geometry=recovery(result['models'],data['ground_truth']) if 'ground_truth' in data else None
        report=dict(status='completed',elapsed_seconds=time.monotonic()-start,shape=list(result['probabilities'].shape),window_metrics=result['window_metrics'],
                    outer_converged=[m['joint']['converged'] for m in result['models']],profile_optimizers_success=[m['joint']['all_profile_optimizers_success'] for m in result['models']],
                    readouts_converged=[[r['diagnostics']['converged'] for r in m['readouts']] for m in result['models']],
                    folded_max_error=float(result['folded_max_errors'].max()),probability_sum_max_error=float(np.max(np.abs(result['probabilities'].sum(axis=1)-1))),
                    oof_exactly_once=bool(np.all(result['filled']==1)),recovery=geometry)
        write_json(out/'report.json',report); plots(result,cfg,out,geometry)
        print(json.dumps(jsonable(report)),flush=True)
    except Exception as exc:
        write_json(out/'failure.json',dict(status='failed',error_type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-start)); raise

if __name__=='__main__': main()
