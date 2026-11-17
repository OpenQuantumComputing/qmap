import json
from pathlib import Path

from common import INSTANCES_ROOT, SOLUTIONS_ROOT, add_qmap_to_import_path
from qiskit import qpy

add_qmap_to_import_path()
import converter

try:
    import circuit_metrics
except ModuleNotFoundError:
    circuit_metrics = None


def _require_metrics() -> None:
    if circuit_metrics is None:
        raise ModuleNotFoundError(
            "circuit_metrics is required for solution evaluation but is not "
            "included in this repository."
        )


def _print_metrics(metrics: dict) -> None:
    print(f"duration:    {metrics['duration']}")
    print(f"noise_delta: {metrics['noise_delta']}")
    print(f"gate_count:  {metrics['gate_count']}")
    print(f"hop:         {metrics['heavy_output_prob']}")


def evaluate_solutions_dynamic_graph_in_folder(
    problem_directory: Path,
    solution_directory: Path,
    overwrite: bool = False,
) -> None:
    _require_metrics()
    for solution_path in sorted(solution_directory.glob("*.json")):
        with solution_path.open(encoding="utf-8") as file:
            solution = json.load(file)
        dynamic_solution = solution.get("dynamic_graph_solution")
        if dynamic_solution is None or (
            "metrics" in dynamic_solution and not overwrite
        ):
            continue

        qpy_path = problem_directory / f"{solution_path.stem}.qpy"
        with qpy_path.open("rb") as file:
            circuit = qpy.load(file)[0]
        representation = converter.GateRepresentation(circuit)
        result_circuit = representation.to_circuit(dynamic_solution)
        metrics = circuit_metrics.evaluate_metrics(
            result_circuit,
            solution["problem"]["topology"],
            42,
            42,
        )
        dynamic_solution["metrics"] = metrics
        with solution_path.open("w", encoding="utf-8") as file:
            json.dump(solution, file, indent=4)
        _print_metrics(metrics)


def evaluate_solutions_qiskit_in_folder(
    solution_directory: Path,
    overwrite: bool = False,
) -> None:
    _require_metrics()
    for qpy_path in sorted(solution_directory.glob("*.qpy")):
        output_path = solution_directory / f"{qpy_path.stem}.json"
        if output_path.exists() and not overwrite:
            continue
        with qpy_path.open("rb") as file:
            circuit = qpy.load(file)[0]
        metrics = circuit_metrics.evaluate_metrics(circuit, None, 42, 42)
        solution = {
            "problem": {"gates": [], "topology": []},
            "dynamic_graph_solution": {"solution": [], "metrics": metrics},
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(solution, file, indent=4)
        _print_metrics(metrics)


def main() -> None:
    for objective in ("swaps", "depth"):
        for graph in ("linear", "grid", "Y"):
            problem_directory = INSTANCES_ROOT / graph
            for layering in (False, True):
                mode = "WithLayering" if layering else "NoLayering"
                solution_directory = SOLUTIONS_ROOT / graph / objective / mode
                evaluate_solutions_dynamic_graph_in_folder(
                    problem_directory,
                    solution_directory,
                )


if __name__ == "__main__":
    main()
