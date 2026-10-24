import argparse
import time
import numpy as np

from neural_decision_model import SimulationConfig, Experiment, Analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=str, default="simulation_result.npz")
    args = parser.parse_args()

    config = SimulationConfig(n_rep=args.repetitions)

    start = time.perf_counter()
    experiment = Experiment(config=config, seed=args.seed)
    result = experiment.run()
    elapsed = time.perf_counter() - start

    np.savez_compressed(
        args.output,
        time=result.time,
        th_stim=result.th_stim,
        target_stim=result.target_stim,
        rate_v1=result.rate_v1,
        spike_v1=result.spike_v1,
        agg_spike_v1=result.agg_spike_v1,
        rate_fef=result.rate_fef,
        spike_fef=result.spike_fef,
        agg_spike_fef=result.agg_spike_fef,
        rate_sc=result.rate_sc,
        spike_sc=result.spike_sc,
        th_hat=result.th_hat,
        th_hat_behav=result.th_hat_behav,
        sigma=result.sigma,
        dv_fef=result.dv_fef,
        choice=result.choice,
        t_choice=result.t_choice,
        forced=result.forced,
    )

    analysis = Analysis(result)
    behavior = analysis.compute_behavior(rng=np.random.default_rng(args.seed))

    print(f"Trials: {config.n_trials}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Mean accuracy: {behavior['p_correct'].mean():.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
