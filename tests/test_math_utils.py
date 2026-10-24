import numpy as np
from neural_decision_model.math_utils import (
    orientation_tuning,
    poisson_log_likelihood,
    temporal_profile,
)


def test_tuning_shape():
    candidates = np.arange(-90, 90)[:, None]
    prefs = np.array([0.0, 30.0])
    y = orientation_tuning(
        candidates, prefs, 1.0,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        axis=0
    )
    assert y.shape == (180, 2)


def test_poisson_likelihood_shape():
    counts = np.array([0, 2])
    lambdas = np.array([[1.0, 2.0], [2.0, 3.0]])
    result = poisson_log_likelihood(counts, lambdas)
    assert result.shape == (2,)


def test_profile_bounds():
    profile = temporal_profile(np.arange(0, 0.501, 0.001))
    assert profile[0] == 0.0
    assert np.isclose(profile.min(), 0.0)
    assert np.isclose(profile.max(), 1.0)
