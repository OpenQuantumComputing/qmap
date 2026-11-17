import json
from pathlib import Path

from solve_examples import solve_examples

BENCHMARK_ROOT = Path(__file__).resolve().parent
RUN_LIST_PATH = BENCHMARK_ROOT / "run_7qubit_run_list.json"
GRAPHS = ["linear", "Y"]
OBJECTIVES = ["swaps", "depth"]
LAYERING_MODES = [False, True]
DEPTHS = [10, 15, 20]
INSTANCE_IDS = range(100)


def _branch_prefix(graph, objective, layering):
    mode = "WithLayering" if layering else "NoLayering"
    return f"{graph}/{objective}/{mode}"


def _problem_names(graph):
    hardware_name = f"{graph}_7"
    return sorted(
        f"problem_random_{hardware_name}_{depth}_{instance_id}.json"
        for depth in DEPTHS
        for instance_id in INSTANCE_IDS
    )


def _solution_path(graph, objective, layering, problem_name):
    mode = "WithLayering" if layering else "NoLayering"
    return BENCHMARK_ROOT / "solutions" / graph / objective / mode / problem_name


def _load_run_list():
    if not RUN_LIST_PATH.exists():
        return {}
    try:
        with RUN_LIST_PATH.open("r") as run_list_file:
            run_list = json.load(run_list_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return run_list if isinstance(run_list, dict) else {}


def _save_run_list(run_list):
    RUN_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RUN_LIST_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as run_list_file:
        json.dump(run_list, run_list_file, indent=2)
    temporary_path.replace(RUN_LIST_PATH)


def initialize_run_list():
    run_list = _load_run_list()

    for graph in GRAPHS:
        for objective in OBJECTIVES:
            for layering in LAYERING_MODES:
                prefix = _branch_prefix(graph, objective, layering)
                for problem_name in _problem_names(graph):
                    output_path = _solution_path(
                        graph, objective, layering, problem_name
                    )
                    if output_path.exists():
                        run_list[f"{prefix}/{problem_name}"] = {
                            "status": "done",
                            "source": "existing_output",
                        }

    completed_branch = _branch_prefix("linear", "swaps", False)
    for problem_name in _problem_names("linear"):
        key = f"{completed_branch}/{problem_name}"
        if key not in run_list:
            run_list[key] = {
                "status": "previously_attempted",
                "source": "known_completed_branch",
            }

    _save_run_list(run_list)
    print(f"Run list initialized: {len(run_list)} entries", flush=True)


def main():
    initialize_run_list()
    for graph in GRAPHS:
        for objective in OBJECTIVES:
            for layering in LAYERING_MODES:
                mode = "WithLayering" if layering else "NoLayering"
                print(f"Starting {graph} / {objective} / {mode}", flush=True)
                solve_examples(
                    graph,
                    layering=layering,
                    objective=objective,
                    overwrite=False,
                    min_num_qubits_to_solve=7,
                    max_num_qubits_to_solve=7,
                    progress_label=f"{graph} / {objective} / {mode}",
                    progress_total=300,
                    run_list_path=str(RUN_LIST_PATH),
                    run_list_prefix=_branch_prefix(graph, objective, layering),
                )


if __name__ == "__main__":
    main()
