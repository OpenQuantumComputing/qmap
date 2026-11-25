# Benchmarks

This directory contains the benchmark instances, solver scripts, and generated
solutions for `qmap`.

## Directory layout

- `instances/`: input problem files in JSON format and the corresponding QPY
  circuits.
- `solutions/`: solver output. This directory is created or extended when a
  benchmark is run.
- `solve_examples.py`: solves the JSON instances for one or more hardware
  graphs.
- `run_7qubit_benchmarks.py`: runs the full 7-qubit benchmark matrix.
- `generate_random_instances.py`: generates JSON/QPY benchmark pairs.
- `evaluate_solutions.py`: adds circuit metrics to solver output.
- `make_pictures.py`: creates comparison plots from evaluated solutions.

All paths are resolved relative to this directory, so the scripts do not
depend on the current working directory or a local checkout path.

## Setup

From the repository root, install the Poetry environment:

```powershell
poetry install
```

The solver modules are loaded from `src/qmap` automatically by the benchmark
scripts.

## Run a benchmark

To run the default example matrix:

```powershell
poetry run python src/benchmark/solve_examples.py
```

To run the 7-qubit benchmark matrix:

```powershell
poetry run python src/benchmark/run_7qubit_benchmarks.py
```

Results are written below `src/benchmark/solutions`. Existing result files
and entries in `run_7qubit_run_list.json` are skipped unless the script is
changed to use `overwrite=True`.

## Generate new instances

The generator writes both the solver input (`.json`) and the original circuit
(`.qpy`) below `src/benchmark/instances`:

```powershell
poetry run python src/benchmark/generate_random_instances.py
```

The helper functions in the module can also be imported to select a graph,
number of qubits, depth, seed, and output directory.

## Evaluate and plot

The evaluation uses the local `circuit_metrics.py` module and requires the
support files `basis_gates.pkl` and `noise_model.pkl`. Plotting additionally
requires Matplotlib:

```powershell
poetry run python src/benchmark/evaluate_solutions.py
poetry run python src/benchmark/make_pictures.py
```

Generated analysis files are written below `src/benchmark/analysis`.
