# qmap

## Overview

*qmap* is a toolkit for solving the generalized Qubit Mapping Problem (QMP) exactly. It implements a custom branch-and-bound approach that allows the compiler to re-order gates freely and to account for gate durations directly, while always respecting hardware connectivity. The resulting qubit schedules can be computed to minimize either the total execution time or the number of swaps.

The solver can enforce restrictions in the structure of the circuit (layered versus non-layered) and implements a few heuristic strategies (beam search, diving, etc.). Additional restrictions, features, or heuristics can be easily integrated. 

The algorithms and benchmarks implemented here accompany the paper *An Exact Branch and Bound Algorithm for the generalized Qubit Mapping Problem* ([arXiv:2508.21718](https://arxiv.org/abs/2508.21718)).

## Requirements

The project targets Python 3.11.3 and currently depends on Qiskit 0.42.1 (see also `pyproject.toml`).

## Input

Problems are described as JSON dictionaries containing the hardware topology, two-qubit gate sequence with durations, and swap-time constant. Example instances are provided in `example_problem.json`

## Running the Solver

A minimal test example is available in `test_qiskit_circuit.py`.

## Citing

If you build on this code or some of the ideas within, please consider citing:

```
An Exact Branch and Bound Algorithm for the generalized Qubit Mapping Problem.
arXiv:2508.21718, 2025.
```

## License

Distributed under the MIT License. See `LICENSE` for details.