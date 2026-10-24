import numpy as np


class StimulusGenerator:
    """Generate trials in the same order as MATLAB's reshape/repmat block."""

    def __init__(self, config):
        self.config = config

    def generate(self):
        th = self.config.th
        target = self.config.target
        n_rep = self.config.n_rep

        # MATLAB reshape(repmat(th,[length(target) 1 nRep]), ...)
        # creates orientation-major, target-minor ordering per repetition.
        th_one_rep = np.repeat(th, target.size)
        target_one_rep = np.tile(target, th.size)

        th_stim = np.tile(th_one_rep, n_rep)
        target_stim = np.tile(target_one_rep, n_rep)

        return th_stim, target_stim
