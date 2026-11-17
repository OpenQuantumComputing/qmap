import json
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
QMAP_SOURCE_ROOT = BENCHMARK_ROOT.parent / "qmap"
if str(QMAP_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(QMAP_SOURCE_ROOT))

import run_dynamic_graph as dg_solver
from dynamic_graph import DivingParameters, DynamicGraphParameters, NodeGrouping


def _load_run_list(run_list_path: Path | None) -> dict:
    if run_list_path is None or not run_list_path.exists():
        return {}
    try:
        with run_list_path.open(encoding="utf-8") as run_list_file:
            run_list = json.load(run_list_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return run_list if isinstance(run_list, dict) else {}


def _save_run_list(run_list_path: Path | None, run_list: dict) -> None:
    if run_list_path is None:
        return
    run_list_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = run_list_path.with_name(f"{run_list_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as run_list_file:
        json.dump(run_list, run_list_file, indent=2)
    temporary_path.replace(run_list_path)


def _solver_parameters(objective: str, layering: bool) -> DynamicGraphParameters:
    if objective not in {"swaps", "depth"}:
        raise ValueError("objective must be either 'swaps' or 'depth'")

    return DynamicGraphParameters(
        DivingParameters(2000, 20, 6),
        int(objective == "swaps"),
        int(objective == "depth"),
        0,
        0,
        0,
        NodeGrouping.NONE,
        layering,
        500,
    )


def _problem_metadata(problem_path: Path) -> tuple[int, int]:
    with problem_path.open(encoding="utf-8") as problem_file:
        problem = json.load(problem_file)["problem"]

    num_gates = len(problem["gates"])
    qubits = {qubit for edge in problem["topology"] for qubit in edge}
    return num_gates, len(qubits)


def solve_examples(
    graph: str,
    layering: bool = False,
    objective: str = "swaps",
    overwrite: bool = False,
    min_num_qubits_to_solve: int = 2,
    max_num_qubits_to_solve: int = 6,
    progress_label: str | None = None,
    progress_total: int | None = None,
    run_list_path: Path | None = None,
    run_list_prefix: str | None = None,
    instances_root: Path = BENCHMARK_ROOT / "instances",
    solutions_root: Path = BENCHMARK_ROOT / "solutions",
) -> None:
    parameters = _solver_parameters(objective, layering)
    timeout_seconds = 500
    path_to_examples = instances_root / graph
    mode = "WithLayering" if layering else "NoLayering"
    path_to_solutions = solutions_root / graph / objective / mode

    if not path_to_examples.is_dir():
        raise FileNotFoundError(
            f"Instance directory does not exist: {path_to_examples}"
        )
    path_to_solutions.mkdir(parents=True, exist_ok=True)

    parameters_as_dict = parameters.__dict__.copy()
    parameters_as_dict["diving_parameters"] = parameters_as_dict[
        "diving_parameters"
    ].__dict__
    parameters_as_dict["statistics_grouping"] = 0
    with (path_to_solutions / "settings.json").open("w", encoding="utf-8") as settings:
        json.dump(parameters_as_dict, settings, indent=4)

    run_list = _load_run_list(run_list_path)
    problems = sorted(path_to_examples.glob("*.json"))
    progress_index = 0

    for problem_path in problems:
        num_gates, num_qubits = _problem_metadata(problem_path)
        try:
            depth = int(problem_path.stem.split("_")[-2])
        except ValueError:
            depth = 0

        if not (
            1 <= num_gates <= 1000
            and 0 <= depth <= 1000
            and min_num_qubits_to_solve <= num_qubits <= max_num_qubits_to_solve
        ):
            continue

        progress_index += 1
        progress = (
            f"[{progress_index}/{progress_total}]"
            if progress_total is not None
            else f"[{progress_index}]"
        )
        if progress_label is not None:
            progress += f" {progress_label}"
        print(f"{progress} Try solving: {problem_path.name}", flush=True)

        solution_path = path_to_solutions / problem_path.name
        run_key = (
            f"{run_list_prefix}/{problem_path.name}"
            if run_list_prefix is not None
            else problem_path.name
        )
        if not overwrite and solution_path.exists():
            run_list[run_key] = {"status": "done", "source": "existing_output"}
            _save_run_list(run_list_path, run_list)
            print(
                f"{progress} skipped (already exists): {problem_path.name}", flush=True
            )
            continue

        known_status = run_list.get(run_key, {}).get("status")
        if not overwrite and known_status is not None:
            print(
                f"{progress} skipped (run list: {known_status}): {problem_path.name}",
                flush=True,
            )
            continue

        run_list[run_key] = {"status": "running", "problem": problem_path.name}
        _save_run_list(run_list_path, run_list)
        start_time = time.monotonic()
        try:
            dg_solver.solve_single_problem(
                str(problem_path), str(solution_path), parameters
            )
            elapsed_seconds = time.monotonic() - start_time
            if solution_path.exists():
                status = "done"
            elif elapsed_seconds >= timeout_seconds:
                status = "timeout"
            else:
                status = "no_output"
            run_list[run_key] = {
                "status": status,
                "problem": problem_path.name,
                "elapsed_seconds": elapsed_seconds,
            }
            _save_run_list(run_list_path, run_list)
            print(f"{progress} {status}: {problem_path.name}", flush=True)
        except KeyboardInterrupt:
            run_list[run_key] = {
                "status": "interrupted",
                "problem": problem_path.name,
            }
            _save_run_list(run_list_path, run_list)
            print(f"{progress} interrupted: {problem_path.name}", flush=True)
            raise
        except Exception as error:
            run_list[run_key] = {
                "status": "failed",
                "problem": problem_path.name,
                "error": f"{type(error).__name__}: {error}",
            }
            _save_run_list(run_list_path, run_list)
            print(f"{progress} failed: {problem_path.name}: {error}", flush=True)


if __name__ == "__main__":
    for graph in ("linear", "grid", "Y"):
        for layering in (False, True):
            solve_examples(
                graph,
                layering=layering,
                objective="swaps",
                overwrite=False,
            )
