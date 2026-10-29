"""Named ordered first-harmonic generator; unchanged full K-1 inference.

All original QR columns are drawn to preserve baseline/noise random streams.
Only the generation contrast is replaced, within a scoped try/finally.
"""
import argparse,sys
from pathlib import Path
import numpy as np
CORE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CORE/'src'))
from jointdecode import cli,simulation
NAME='ordered_ring_transient_shared_r3'
SPEC=dict(name=NAME,overlap=0.,position_geometry='C[k]=[cos(2*pi*k/6),sin(2*pi*k/6)]; k=0..5 maps positions 1..6; zero-padded into unchanged QR blocks',true_template_rank=2,inference_dimension='Full K-1 categorical GLM; no forced rank-2 basis or readout',interpretation='Simulation hypothesis, not empirical-data reanalysis')

def ring_contrast(n,full=False):
    if n!=6 or full:raise ValueError('Named ring run requires six positions and reduced contrast')
    C=np.zeros((n-1,n));theta=2*np.pi*np.arange(n)/n
    C[0]=np.cos(theta);C[1]=np.sin(theta)
    return C

def named_simulate(config,**kwargs):
    if kwargs or len(config.target_onsets_seconds)!=3 or len(config.position_labels)!=6:
        raise ValueError('Named ring run is R3 K6 positive simulation only')
    original=simulation.helmert
    try:
        simulation.helmert=ring_contrast
        result=simulation.simulate(config,overlap=0.)
    finally:simulation.helmert=original
    result['simulation_parameters'].update(name=NAME,position_geometry=SPEC['position_geometry'],true_template_rank=2)
    return result

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--epochs');parser.add_argument('--control',default='positive');parser.add_argument('--output')
    known,_=parser.parse_known_args(args)
    if known.epochs or known.control!='positive':raise ValueError('Named entry point is simulation-only positive control')
    original=cli.simulate
    try:
        cli.simulate=named_simulate;cli.main(args)
    finally:cli.simulate=original
    cli.write_json(Path(known.output)/'simulation_config.json',SPEC)
if __name__=='__main__':main()
