"""Joint shared/private subspace decoding. Probabilistic softmax, not Bayesian inference."""
from .config import Config
from .pipeline import run_oof_decoding
from .readout import predict_fold
from .simulation import simulate
__all__=['Config','run_oof_decoding','predict_fold','simulate']
