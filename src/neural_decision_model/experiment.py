import numpy as np
from scipy.stats import gamma

from .config import SimulationConfig
from .stimuli import StimulusGenerator
from .neurons import V1Population, FEFPopulation, SCPopulation
from .math_utils import orientation_tuning, poisson_log_likelihood, circular_std_deg
from .results import SimulationResult


class Experiment:
    """Coordinates populations and preserves the source's trial/time update order."""

    def __init__(self, config=None, seed=None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)

        self.th_stim, self.target_stim = StimulusGenerator(
            self.config
        ).generate()

        self.v1 = V1Population(self.config, self.th_stim, self.rng)
        self.fef = FEFPopulation(self.config, self.rng)
        self.sc = SCPopulation(self.config, self.rng)

        self.candidates = np.arange(-90, 90, dtype=float)

        # Decoder tuning: candidates x neurons.
        self.candidate_tuning = orientation_tuning(
            self.candidates[:, None],
            self.v1.theta_pref[None, :],
            self.v1.alpha,
            self.v1.beta[None, :],
            self.v1.amplitude[None, :],
            axis=0
        )

        T, ntr = self.config.T, self.config.n_trials
        self.agg_spike_v1 = np.zeros_like(self.v1.rate)
        self.agg_spike_fef = np.zeros_like(self.fef.rate)

        self.th_hat = np.zeros((T, ntr))
        self.th_hat_behav = np.zeros(ntr)
        self.sigma = np.zeros((T, ntr))
        self.dv_fef = np.zeros((T, ntr))

        self.choice = np.zeros(ntr, dtype=int)
        self.t_choice = np.zeros(ntr, dtype=int)
        self.forced = np.zeros(ntr, dtype=bool)

        self.mt_fef_state = np.zeros(self.fef.n_neurons)
        self.mt_sc_state = np.zeros(self.sc.n_neurons)

    def _decode(self, trial, time_indices, spike_counts):
        """Decode over a temporal slice, returning MLE and circular uncertainty."""
        profile = self.v1.tmp_profile[time_indices]
        gains = self.v1.gain[:, trial]

        # For each candidate and neuron, integrate instantaneous rate over time.
        tuning = self.candidate_tuning  # candidate x neuron
        base = self.v1.epsilon[None, :] + tuning * profile[:, None].mean(axis=0) if False else None

        # Exact temporal integration: sum_t (eps + tuning * profile[t]) * gain_i * dt
        profile_sum = np.sum(profile)
        n_time = len(np.arange(self.config.T)[time_indices]) if isinstance(time_indices, slice) else len(time_indices)
        # The source gain is time-invariant per neuron/trial, repeated over time.
        lambdas = (
            self.v1.epsilon[None, :] * n_time
            + tuning * profile_sum
        ) * gains[None, :] * self.config.dt

        log_likelihood = poisson_log_likelihood(spike_counts, lambdas)
        best = int(np.argmax(log_likelihood))
        probabilities = np.exp(log_likelihood - np.max(log_likelihood))
        sigma = circular_std_deg(self.candidates, weights=probabilities, axis=0)

        return self.candidates[best], sigma, log_likelihood

    def _run_trial(self, trial):
        cfg = self.config

        # V1 spikes are generated for the whole trial, as in the MATLAB code.
        self.v1.generate_trial_spikes(trial)

        # Behavioral/full-window V1 decoder.
        full_counts = self.v1.spikes[:, :, trial].sum(axis=0)
        theta_behav, _, _ = self._decode(
            trial, slice(0, cfg.T), full_counts
        )
        self.th_hat_behav[trial] = theta_behav

        # Initial V1 decoder uses the first sample only.
        initial_counts = self.v1.spikes[0, :, trial].astype(float)
        theta0, sigma0, _ = self._decode(
            trial, slice(0, 1), initial_counts
        )
        self.agg_spike_v1[0, :, trial] = initial_counts
        self.th_hat[0, trial] = theta0
        self.sigma[0, trial] = sigma0

        self.dv_fef[0, trial] = theta0 / sigma0

        # FEF baseline at first time sample; initial spikes remain zero as source.
        self.fef.rate[0, :, trial] = (
            self.fef.epsilon * self.fef.gain[:, trial]
        )
        self.agg_spike_fef[0, :, trial] = self.fef.spikes[0, :, trial]

        mt_fef = np.zeros(self.fef.n_neurons)
        mt_sc = np.zeros(self.sc.n_neurons)

        for t in range(1, cfg.T):

            # V1 sliding-window count and decoder.
            window = slice(max(t - int(round(cfg.dtau_v1 / cfg.dt)), 0), t + 1)
            counts = self.v1.spikes[window, :, trial].sum(axis=0)
            self.agg_spike_v1[t, :, trial] = counts

            theta_hat, sigma, _ = self._decode(
                trial, window, counts
            )
            self.th_hat[t, trial] = theta_hat
            self.sigma[t, trial] = sigma

            # Categorical DV accumulates until choice; afterwards it decays.
            if self.choice[trial] == 0:
                self.dv_fef[t, trial] = (
                    self.dv_fef[t - 1, trial] + theta_hat / sigma
                )
            else:
                self.dv_fef[t, trial] = (
                    self.dv_fef[t - 1, trial] * self.fef.ch_decay
                )

                elapsed = t - self.t_choice[trial]
                evidence_index = elapsed
                evidence = self.fef.mt_evidence[evidence_index]

                mt_fef = (
                    self.choice[trial]
                    * self.target_stim[trial]
                    * self.fef.mt_pref
                    * evidence
                )

                congruent = (
                    self.choice[trial]
                    * self.target_stim[trial]
                    * self.sc.mt_pref > 0
                )
                mt_sc[congruent] = evidence

            # FEF spikes/rates.
            self.fef.set_rate_from_dvs(
                trial, t, self.dv_fef[t, trial], mt_fef
            )
            self.fef.generate_spikes(trial, t)

            # SC spikes/rates.
            self.sc.rate[t, :, trial] = mt_sc * self.sc.gain[:, trial]
            self.sc.generate_spikes(trial, t)

            # FEF count in sliding decoder window.
            fef_window = slice(
                max(t - int(round(cfg.dtau_fef / cfg.dt)), 0),
                t + 1
            )
            self.agg_spike_fef[t, :, trial] = (
                self.fef.spikes[fef_window, :, trial].sum(axis=0)
            )

            # Choice evaluation: retain original positive/negative channel tests.
            if self.choice[trial] == 0:
                counts_adjusted = (
                    self.agg_spike_fef[t, :, trial]
                    - self.fef.epsilon * cfg.dtau_fef
                )

                positive = self.fef.ch_pref > 0
                negative = self.fef.ch_pref < 0

                positive_score = np.sum(
                    (1.0 / self.fef.w_ch[positive])
                    * counts_adjusted[positive]
                )
                negative_score = np.sum(
                    (1.0 / self.fef.w_ch[negative])
                    * counts_adjusted[negative]
                )

                if positive_score > self.fef.threshold * np.sum(positive):
                    self.choice[trial] = 1
                    self.t_choice[trial] = t
                elif negative_score > self.fef.threshold * np.sum(negative):
                    self.choice[trial] = -1
                    self.t_choice[trial] = t

                # Forced choice at threshold time if no decision has occurred.
                if self.choice[trial] == 0 and t >= cfg.tau_threshold_index:
                    if positive_score > negative_score:
                        self.choice[trial] = 1
                    else:
                        self.choice[trial] = -1

                    self.t_choice[trial] = t
                    self.forced[trial] = True

    def run(self):
        for trial in range(self.config.n_trials):
            self._run_trial(trial)

        return SimulationResult(
            time=self.config.time,
            th_stim=self.th_stim.copy(),
            target_stim=self.target_stim.copy(),

            rate_v1=self.v1.rate,
            spike_v1=self.v1.spikes,
            agg_spike_v1=self.agg_spike_v1,

            rate_fef=self.fef.rate,
            spike_fef=self.fef.spikes,
            agg_spike_fef=self.agg_spike_fef,

            rate_sc=self.sc.rate,
            spike_sc=self.sc.spikes,

            th_hat=self.th_hat,
            th_hat_behav=self.th_hat_behav,
            sigma=self.sigma,
            dv_fef=self.dv_fef,

            choice=self.choice,
            t_choice=self.t_choice,
            forced=self.forced
        )
