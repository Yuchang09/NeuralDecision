from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    stim_dur: float = 0.5
    dt: float = 0.001
    dth: float = 1.75
    n_rep: int = 50

    n_neurons_v1: int = 20
    n_neurons_fef: int = 20
    n_neurons_sc: int = 20

    var_gain: float = 0.5

    dtau_v1: float = 0.05
    dtau_fef: float = 0.05
    threshold_fef: float = 15.0
    tau_threshold: float = 0.4
    ch_decay_fef: float = 0.99

    @property
    def time(self):
        # MATLAB colon is endpoint-inclusive; use integer steps to avoid drift.
        n = int(round(self.stim_dur / self.dt))
        return np.arange(n + 1, dtype=float) * self.dt

    @property
    def T(self):
        return self.time.size

    @property
    def th(self):
        # MATLAB -3*dth:dth:3*dth
        return np.arange(-3, 4, dtype=float) * self.dth

    @property
    def target(self):
        return np.array([-1, 1], dtype=int)

    @property
    def n_trials(self):
        return self.th.size * self.target.size * self.n_rep

    @property
    def tau_bin_v1(self):
        n = int(round(self.stim_dur / self.dtau_v1))
        return np.arange(n + 1, dtype=float) * self.dtau_v1

    @property
    def tau_bin_fef(self):
        n = int(round(self.stim_dur / self.dtau_fef))
        return np.arange(n + 1, dtype=float) * self.dtau_fef

    @property
    def tau_threshold_index(self):
        return int(round(self.tau_threshold / self.dt))
