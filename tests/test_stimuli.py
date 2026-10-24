import numpy as np
from neural_decision_model.config import SimulationConfig
from neural_decision_model.stimuli import StimulusGenerator


def test_trial_order_and_count():
    config = SimulationConfig(n_rep=2)
    th, target = StimulusGenerator(config).generate()

    assert len(th) == config.n_trials
    assert len(target) == config.n_trials
    assert np.array_equal(th[:14], np.repeat(config.th, 2))
    assert np.array_equal(target[:14], np.tile([-1, 1], 7))
