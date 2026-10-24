import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def matlab_rescale(x, new_min=0.0, new_max=1.0, axis=None):
    """Explicit min-max rescale; axis=None uses the whole array."""
    x = np.asarray(x, dtype=float)
    lo = np.min(x, axis=axis, keepdims=True)
    hi = np.max(x, axis=axis, keepdims=True)
    span = hi - lo
    safe = np.where(span == 0, 1.0, span)
    out = (x - lo) / safe
    out = np.where(span == 0, 0.0, out)
    return new_min + out * (new_max - new_min)


def orientation_tuning(theta, theta_pref, alpha, beta, amplitude, axis=0):
    """MATLAB tuningCurve, with explicit broadcastable inputs.

    Normalize raw tuning values over `axis`. For candidate-angle arrays shaped
    (n_candidates, n_neurons), axis=0 normalizes each neuron's candidate curve.
    """
    theta = np.asarray(theta, dtype=float)
    theta_pref = np.asarray(theta_pref, dtype=float)
    beta = np.asarray(beta, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)

    c = np.cos(np.deg2rad(2.0 * (theta - theta_pref)))
    raw = c * np.exp(-0.5 * c**2 * (1.0 - alpha**2))
    scaled = matlab_rescale(raw, 0.0, 1.0, axis=axis)
    return amplitude * scaled**beta


def temporal_profile(time):
    """Construct the MATLAB profile: rescale([0; normcdf(t,.1,.025)/t^.9])."""
    time = np.asarray(time, dtype=float)
    profile = np.zeros_like(time)
    positive = time > 0
    profile[positive] = (
        norm.cdf(time[positive], loc=0.1, scale=0.025)
        / time[positive]**0.9
    )
    return matlab_rescale(profile, 0.0, 1.0, axis=None)


def poisson_log_likelihood(spike_counts, lambdas):
    """Poisson log likelihood across neurons.

    Parameters
    ----------
    spike_counts : (n_neurons,)
    lambdas : (n_candidates, n_neurons)

    Returns
    -------
    (n_candidates,) log likelihoods
    """
    n = np.asarray(spike_counts, dtype=float)[None, :]
    lam = np.asarray(lambdas, dtype=float)
    # MATLAB log(0) can produce -inf. For observed n=0, n*log(0) is
    # mathematically 0; use the standard xlogy convention.
    from scipy.special import xlogy
    return np.sum(-lam + xlogy(n, lam) - gammaln(n + 1), axis=1)


def circular_mean_deg(angles_deg, axis=None):
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    s = np.mean(np.sin(a), axis=axis)
    c = np.mean(np.cos(a), axis=axis)
    return np.rad2deg(np.arctan2(s, c))


def circular_std_deg(angles_deg, weights=None, axis=None):
    """Circular std in degrees; weights, when given, are nonnegative masses."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    if weights is None:
        r = np.hypot(
            np.mean(np.sin(a), axis=axis),
            np.mean(np.cos(a), axis=axis)
        )
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w, axis=axis, keepdims=True)
        r = np.hypot(
            np.sum(w * np.sin(a), axis=axis),
            np.sum(w * np.cos(a), axis=axis)
        )
    r = np.clip(r, 0.0, 1.0)
    # Circular standard deviation: sqrt(-2 log R), in radians.
    return np.rad2deg(np.sqrt(-2.0 * np.log(np.maximum(r, 1e-15))))
