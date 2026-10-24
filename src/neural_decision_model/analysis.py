import numpy as np

from .math_utils import circular_mean_deg, circular_std_deg


class Analysis:
    def __init__(self, result):
        self.result = result

    def decode_orientation_at_time(self, time_index):
        r = self.result
        orientations = np.unique(r.th_stim)

        means = np.array([
            circular_mean_deg(r.th_hat[time_index, r.th_stim == th])
            for th in orientations
        ])

        stds = np.array([
            circular_std_deg(r.th_hat[time_index, r.th_stim == th])
            for th in orientations
        ])

        choice_v1 = r.th_hat[time_index] > 0

        return {
            "orientations": orientations,
            "mean_decoded_orientation": means,
            "std_decoded_orientation": stds,
            "choice_v1": choice_v1,
        }

    def compute_behavior(self, rng=None):
        r = self.result
        rng = np.random.default_rng() if rng is None else rng

        correct = r.th_stim * r.choice > 0
        zero_stim = r.th_stim == 0
        if np.any(zero_stim):
            correct[zero_stim] = rng.random(np.sum(zero_stim)) < 0.5

        orientations = np.unique(r.th_stim)

        p_correct = np.array([
            np.mean(correct[r.th_stim == th])
            for th in orientations
        ])

        p_choice_positive = np.array([
            np.mean(r.choice[r.th_stim == th] > 0)
            for th in orientations
        ])

        choice_behav = r.th_stim * r.th_hat_behav > 0

        p_choice_positive_behav = np.array([
            np.mean(choice_behav[r.th_stim == th])
            for th in orientations
        ])

        return {
            "orientations": orientations,
            "correct": correct,
            "p_correct": p_correct,
            "p_choice_positive": p_choice_positive,
            "choice_behavioral_decoder": choice_behav,
            "p_choice_positive_behavioral_decoder": p_choice_positive_behav,
        }

    def compute_choice_motor_dvs(self):
        r = self.result
        T, n_neurons, n_trials = r.agg_spike_fef.shape

        ch_w = np.zeros((n_neurons, T))
        mt_w = np.zeros((n_neurons, T))

        for t in range(T):
            # MATLAB zscore(..., [], 2) across trials for each neuron.
            x = r.agg_spike_fef[t, :, :].astype(float)
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, ddof=0, keepdims=True)
            std[std == 0] = 1.0
            z = (x - mean) / std

            # MATLAB Sigma = cov(zSpikeFEF'); columns are neurons.
            sigma = np.cov(z, bias=False)

            choice_pos = r.choice == 1
            choice_neg = r.choice == -1
            response = r.response
            response_pos = response == 1
            response_neg = response == -1

            ch_s = (
                z[:, choice_pos].mean(axis=1)
                - z[:, choice_neg].mean(axis=1)
            )
            mt_s = (
                z[:, response_pos].mean(axis=1)
                - z[:, response_neg].mean(axis=1)
            )

            # MATLAB backslash: solve Sigma * w = signal.
            ch_w[:, t] = np.linalg.solve(sigma, ch_s)
            mt_w[:, t] = np.linalg.solve(sigma, mt_s)

        # In source, zSpikeFEF after loop corresponds to the final time point.
        ch_dv = ch_w.T @ z
        mt_dv = mt_w.T @ z

        unsigned_ch_dv = np.concatenate(
            [ch_dv[:, r.choice == 1], -ch_dv[:, r.choice == -1]],
            axis=1
        )
        unsigned_mt_dv = np.concatenate(
            [mt_dv[:, r.response == 1], -mt_dv[:, r.response == -1]],
            axis=1
        )

        return {
            "ch_w": ch_w,
            "mt_w": mt_w,
            "ch_dv": ch_dv,
            "mt_dv": mt_dv,
            "unsigned_ch_dv": unsigned_ch_dv,
            "unsigned_mt_dv": unsigned_mt_dv,
        }
