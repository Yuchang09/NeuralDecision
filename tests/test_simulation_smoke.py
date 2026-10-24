from neural_decision_model import SimulationConfig, Experiment


def test_small_simulation_runs():
    config = SimulationConfig(n_rep=1)
    result = Experiment(config=config, seed=4).run()

    assert result.choice.shape == (config.n_trials,)
    assert result.th_hat.shape == (config.T, config.n_trials)
    assert result.rate_v1.shape == (
        config.T, config.n_neurons_v1, config.n_trials
    )
