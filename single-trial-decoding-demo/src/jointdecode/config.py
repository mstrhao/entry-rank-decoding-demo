"""Explicit timing and immutable algorithm configuration."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Config:
    sfreq: float = 30.
    times_seconds: tuple = tuple(np.arange(180)/30)
    target_onsets_seconds: tuple = (1., 2.3, 3.6)
    baseline_window_seconds: tuple = (.3, .8)
    entry_readout_windows_seconds: tuple = ((1.2,1.6),(2.5,2.9),(3.8,4.2))
    rank_readout_windows_seconds: tuple = ((4.5,5.5),)*3
    position_labels: tuple = (1,2,3,4,5,6)
    preprocessing: str = 'as_is'
    input_processing_declaration: str = 'User-supplied processed epochs; no additional preprocessing'
    seed_data: int = 20261028
    seed_split: int = 20261029
    prior_bin_frames: int = 2
    tgm_prior_strength: float = 1.
    temporal_smooth_strength: float = 2.
    entry_recurrence_strength: float = 1.
    private_overlap_strength: float = .25
    template_ridge: float = 1e-6
    coordinate_ridge: float = .25
    max_iter: int = 80
    profile_max_iter: int = 300
    tol: float = 1e-7
    profile_tol: float = 1e-9
    floor: float = 1e-12
    original_trial_order: tuple | None = None

    def to_dict(self):
        return asdict(self)


def frame_mask(times, window):
    times=np.asarray(times)
    return (times >= window[0]) & (times < window[1])
