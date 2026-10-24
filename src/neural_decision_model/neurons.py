from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import gamma, lognorm, norm

from .math_utils import orientation_tuning, temporal_profile


class NeuralPopulation(ABC):
    """Common storage and Bernoulli spike generation for a population."""

    def __init__(self, n_neurons, n_trials, T, dt, rng):
        self.n_neurons = int(n_neurons)
        self.n_trials = int(n_trials)
        self.T = int(T)
        self.dt = float(dt)
        self.rng = rng

        self.rate = np.zeros((T, n_neurons, n_trials), dtype=float)
        self.spikes = np.zeros((T, n_neurons, n_trials), dtype=bool)

    def generate_spikes(self, trial, t):
        p = self.rate[t, :, trial] * self.dt
        self.spikes[t, :, trial] = self.rng.random(self.n_neurons) < p

    @abstractmethod
    def compute_rate(self, trial, t):
        raise NotImplementedError


class V1Population(NeuralPopulation):
    def __init__(self, config, th_stim, rng):
        super().__init__(
            config.n_neurons_v1, config.n_trials, config.T, config.dt, rng
        )
        self.config = config
        self.th_stim = np.asarray(th_stim)

        self.theta_pref = rng.uniform(-90.0, 90.0, self.n_neurons)
        self.alpha = 1.0
        self.amplitude = lognorm.rvs(
            s=0.125, scale=np.exp(4.5), size=self.n_neurons,
            random_state=rng
        )
        self.epsilon = lognorm.rvs(
            s=0.8, scale=np.exp(1.0), size=self.n_neurons,
            random_state=rng
        )
        self.beta = lognorm.rvs(
            s=1.0, scale=np.exp(1.5), size=self.n_neurons,
            random_state=rng
        )

        # MATLAB gain: min(4, Gamma(shape=1/varGain, scale=varGain))
        gains = gamma.rvs(
            a=1.0 / config.var_gain,
            scale=config.var_gain,
            size=(self.n_neurons, self.n_trials),
            random_state=rng
        )
        self.gain = np.minimum(4.0, gains)

        self.tmp_profile = temporal_profile(config.time)
        self.agg_profile = np.cumsum(self.tmp_profile)

        # One tuning value per trial and neuron.
        # One tuning value per trial and neuron. The trial axis is explicit.
        tuning = orientation_tuning(
            self.th_stim[:, None],
            self.theta_pref[None, :],
            self.alpha,
            self.beta[None, :],
            self.amplitude[None, :],
            axis=0
        )
        # Shape: (time, neuron, trial).
        self.drive = (
            self.epsilon[None, :, None]
            + tuning.T[None, :, :] * self.tmp_profile[:, None, None]
        )

        self.rate[:] = self.drive * self.gain[None, :, :]

    def compute_rate(self, trial, t):
        self.rate[t, :, trial] = (
            self.drive[t, :, trial] * self.gain[:, trial]
        )

    def generate_trial_spikes(self, trial):
        self.rate[:, :, trial] = (
            self.drive[:, :, trial] * self.gain[:, trial][None, :]
        )
        p = self.rate[:, :, trial] * self.dt
        self.spikes[:, :, trial] = self.rng.random(
            (self.T, self.n_neurons)
        ) < p


class FEFPopulation(NeuralPopulation):
    def __init__(self, config, rng):
        super().__init__(
            config.n_neurons_fef, config.n_trials, config.T, config.dt, rng
        )
        self.config = config
        self.ch_pref = -1 + 2 * rng.integers(0, 2, self.n_neurons)
        self.mt_pref = -1 + 2 * rng.integers(0, 2, self.n_neurons)

        gains = gamma.rvs(
            a=1.0 / config.var_gain,
            scale=config.var_gain,
            size=(self.n_neurons, self.n_trials),
            random_state=rng
        )
        self.gain = np.minimum(4.0, gains)

        self.mt_evidence = 60.0 * norm.cdf(
            config.time, loc=3 * 0.075, scale=0.075
        )
        self.amplitude_mt = lognorm.rvs(
            s=0.5, scale=np.exp(2.5), size=self.n_neurons,
            random_state=rng
        )
        self.epsilon = lognorm.rvs(
            s=0.25, scale=np.exp(3.25), size=self.n_neurons,
            random_state=rng
        )
        self.w_ch = lognorm.rvs(
            s=0.5, scale=np.exp(-2.0), size=self.n_neurons,
            random_state=rng
        )
        self.w_mt = lognorm.rvs(
            s=0.5, scale=np.exp(0.0), size=self.n_neurons,
            random_state=rng
        )

        self.ch_decay = config.ch_decay_fef
        self.threshold = config.threshold_fef

    def compute_rate(self, trial, t):
        raise RuntimeError("FEF rate requires current categorical and motor DVs.")

    def set_rate_from_dvs(self, trial, t, categorical_dv, motor_dv):
        drive = np.maximum(
            0.0,
            self.w_ch * self.ch_pref * categorical_dv
            + self.w_mt * motor_dv
            + self.epsilon
        )
        self.rate[t, :, trial] = drive * self.gain[:, trial]


class SCPopulation(NeuralPopulation):
    def __init__(self, config, rng):
        super().__init__(
            config.n_neurons_sc, config.n_trials, config.T, config.dt, rng
        )
        self.config = config
        self.mt_pref = -1 + 2 * rng.integers(0, 2, self.n_neurons)

        self.gain = gamma.rvs(
            a=1.0 / config.var_gain,
            scale=config.var_gain,
            size=(self.n_neurons, self.n_trials),
            random_state=rng
        )
        self.d_motor = lognorm.rvs(
            s=0.5, scale=np.exp(0.0), size=self.n_neurons,
            random_state=rng
        )
        self.epsilon = lognorm.rvs(
            s=0.8, scale=np.exp(1.0), size=self.n_neurons,
            random_state=rng
        )

    def compute_rate(self, trial, t):
        raise RuntimeError("SC rate is set from the experiment's motor state.")
