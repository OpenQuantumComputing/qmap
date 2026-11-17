# qmap

## Overview

*qmap* is a toolkit for solving the generalized Qubit Mapping Problem (QMP) exactly. It implements a custom branch-and-bound approach that allows the compiler to re-order gates freely and to account for gate durations directly, while always respecting hardware connectivity. The resulting qubit schedules can be computed to minimize either the total execution time or the number of swaps.

The solver can enforce restrictions in the structure of the circuit (layered versus non-layered) and implements a few heuristic strategies (beam search, diving, etc.). Additional restrictions, features, or heuristics can be easily integrated. 

The algorithms and benchmarks implemented here accompany the paper *An Exact Branch and Bound Algorithm for the generalized Qubit Mapping Problem* ([arXiv:2508.21718](https://arxiv.org/abs/2508.21718)).

## Requirements

The project supports Python 3.12 and 3.13. Install the dependencies with
Poetry; the exact package versions are specified in `pyproject.toml`.

## Input

Problems are described as JSON dictionaries containing the hardware topology, two-qubit gate sequence with durations, and swap-time constant. Example instances are provided in `example_problem.json`

## Running the Solver

A minimal test example is available in `test_qiskit_circuit.py`.

## Benchmark

Benchmark instances, solutions and scripts are available in
[`src/benchmark`](./src/benchmark).

### Directory layout

- `instances/`: JSON solver inputs and the corresponding QPY circuits.
- `solutions/`: generated solver solutions.
- `analysis/`: generated evaluation data and plots.
- `solve_examples.py`: solves instances for the available hardware graphs.
- `run_7qubit_benchmarks.py`: runs the full 7-qubit benchmark matrix.
- `generate_random_instances.py`: generates JSON/QPY benchmark pairs.
- `evaluate_solutions.py`: computes metrics for generated solutions.
- `make_pictures.py`: creates comparison plots.

All benchmark paths are resolved relative to `src/benchmark`, so the scripts
can be run from any current working directory.

### Setup

From the repository root:

```powershell
poetry install
```

### Run benchmarks

Run the default benchmark matrix:

```powershell
poetry run python src/benchmark/solve_examples.py
```

Run the 7-qubit benchmark matrix:

```powershell
poetry run python src/benchmark/run_7qubit_benchmarks.py
```

Generate new instances:

```powershell
poetry run python src/benchmark/generate_random_instances.py
```

Evaluate solutions and create plots:

```powershell
poetry run python src/benchmark/evaluate_solutions.py
poetry run python src/benchmark/make_pictures.py
```

The evaluation uses `src/benchmark/basis_gates.pkl` and
`src/benchmark/noise_model.pkl`. These files must be present before running
the evaluation. A local noise model can be regenerated with
`circuit_metrics.generate_new_noise_model()`. Access to an IBM backend is
optional and requires the `QISKIT_IBM_TOKEN` environment variable.

## Citing

If you build on this code or some of the ideas within, please consider citing:

```
An Exact Branch and Bound Algorithm for the generalized Qubit Mapping Problem.
arXiv:2508.21718, 2025.
```

## License

Distributed under the MIT License. See `LICENSE` for details.