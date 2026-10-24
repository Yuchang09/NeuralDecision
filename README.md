# Neural decision model

A modular Python refactor of the supplied MATLAB V1–FEF–SC simulation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

## Run

```bash
python scripts/run_simulation.py --seed 123 --repetitions 50
```

The default simulation is computationally substantial. Use `--repetitions 1` while developing.

## Architecture

- `config.py`: simulation parameters and derived time/stimulus arrays
- `stimuli.py`: MATLAB-order trial construction
- `math_utils.py`: tuning curves, Poisson likelihood, circular statistics
- `neurons.py`: population-level V1, FEF, and SC classes
- `results.py`: output dataclass
- `experiment.py`: trial/time loops and state transitions
- `analysis.py`: behavior and choice/motor decision-variable analysis
- `scripts/run_simulation.py`: command-line entry point

## Equivalence notes

This project preserves the source equations, population axes, trial ordering, and the order of updates as closely as practical. MATLAB and NumPy do not share a random stream, so equal seeds do not produce identical random draws. A deterministic MATLAB-vs-Python comparison should compare intermediate deterministic arrays first.

A few MATLAB expressions are dimension-sensitive. The implementation uses explicit broadcasting and explicit circular-statistics routines. The `rescale`-style tuning curve is normalized over the candidate-angle axis for the decoder and over the supplied orientation values for stimulus drives. If your MATLAB `rescale` behavior differs due to input shape/version, validate this detail against exported MATLAB arrays.

The original script leaves `thhatBehav` un-preallocated and uses the same decoder formula over the full trial window. This implementation stores that behavior estimate explicitly.

The source's `choice` can be assigned by the decision threshold and then overwritten by the forced-choice condition in the same time step only if the forced-choice branch is reached while `choice` is still zero. This implementation retains the same conditional order.

## Tests

```bash
pytest
```
# NeuralDecision
