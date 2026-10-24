from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    time: np.ndarray
    th_stim: np.ndarray
    target_stim: np.ndarray

    rate_v1: np.ndarray
    spike_v1: np.ndarray
    agg_spike_v1: np.ndarray

    rate_fef: np.ndarray
    spike_fef: np.ndarray
    agg_spike_fef: np.ndarray

    rate_sc: np.ndarray
    spike_sc: np.ndarray

    th_hat: np.ndarray
    th_hat_behav: np.ndarray
    sigma: np.ndarray
    dv_fef: np.ndarray

    choice: np.ndarray
    t_choice: np.ndarray
    forced: np.ndarray

    @property
    def response(self):
        return self.choice * self.target_stim
