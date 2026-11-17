import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import ANALYSIS_ROOT, SOLUTIONS_ROOT


def _output_path(folder, filename):
    folder_path = Path(folder)
    if not folder_path.is_absolute():
        if folder_path.parts and folder_path.parts[0] == "pictures":
            folder_path = ANALYSIS_ROOT.joinpath(*folder_path.parts[1:])
        else:
            folder_path = ANALYSIS_ROOT / folder_path
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / filename


def readInData(path_to_sol1, path_to_sol2, timeout=300):
    solution_directory = Path(path_to_sol1)
    comparison_directory = Path(path_to_sol2)
    instances = sorted(solution_directory.glob("*.json"))
    metrics1 = list()
    metrics2 = list()
    solution_metrics1 = list()
    solution_metrics2 = list()

    compare_depth_bigger = list()
    compare_depth_smaller = list()
    compare_swaps_bigger = list()
    compare_swaps_smaller = list()
    problem_statistics = list()

    nr_solutions_identical = 0
    nr_solutions_different = 0

    for instance_path in instances:
        instance = instance_path.name

        comparison_path = comparison_directory / instance
        if comparison_path.exists():
            with instance_path.open(encoding="utf-8") as fh:
                cur_solution1 = json.load(fh)
            with comparison_path.open(encoding="utf-8") as fh:
                cur_solution2 = json.load(fh)

            if (
                "dynamic_graph_solution" in cur_solution1
                and "dynamic_graph_solution" in cur_solution2
            ):
                if (
                    cur_solution1["dynamic_graph_solution"]["solution"]
                    == cur_solution2["dynamic_graph_solution"]["solution"]
                ):
                    nr_solutions_identical += 1
                else:
                    nr_solutions_different += 1

                if "runtime" in cur_solution1["dynamic_graph_solution"]:
                    solution_metrics1.append(
                        {
                            "runtime": cur_solution1["dynamic_graph_solution"][
                                "runtime"
                            ],
                            "depth": cur_solution1["dynamic_graph_solution"]["depth"],
                            "swaps": cur_solution1["dynamic_graph_solution"]["swaps"],
                            "timeout": cur_solution1["dynamic_graph_solution"][
                                "runtime"
                            ]
                            >= timeout,
                        }
                    )
                else:
                    solution_metrics1.append(
                        {"runtime": -1, "depth": 0, "swaps": 0, "timeout": False}
                    )
                if "runtime" in cur_solution2["dynamic_graph_solution"]:
                    solution_metrics2.append(
                        {
                            "runtime": cur_solution2["dynamic_graph_solution"][
                                "runtime"
                            ],
                            "depth": cur_solution2["dynamic_graph_solution"]["depth"],
                            "swaps": cur_solution2["dynamic_graph_solution"]["swaps"],
                            "timeout": cur_solution2["dynamic_graph_solution"][
                                "runtime"
                            ]
                            >= timeout,
                        }
                    )
                else:
                    solution_metrics2.append(
                        {"runtime": -1, "depth": 0, "swaps": 0, "timeout": False}
                    )
                if (
                    "metrics" in cur_solution1["dynamic_graph_solution"]
                    and "metrics" in cur_solution2["dynamic_graph_solution"]
                ):
                    metrics1.append(cur_solution1["dynamic_graph_solution"]["metrics"])
                    metrics2.append(cur_solution2["dynamic_graph_solution"]["metrics"])

                    if (
                        "depth" in cur_solution1["dynamic_graph_solution"]["metrics"]
                        and "depth"
                        in cur_solution2["dynamic_graph_solution"]["metrics"]
                    ):
                        if (
                            cur_solution1["dynamic_graph_solution"]["metrics"]["depth"]
                            > cur_solution2["dynamic_graph_solution"]["metrics"][
                                "depth"
                            ]
                        ):
                            compare_depth_bigger.append(instance)
                        elif (
                            cur_solution1["dynamic_graph_solution"]["metrics"]["depth"]
                            < cur_solution2["dynamic_graph_solution"]["metrics"][
                                "depth"
                            ]
                        ):
                            compare_depth_smaller.append(instance)
                    if (
                        "gate_count"
                        in cur_solution1["dynamic_graph_solution"]["metrics"]
                        and "gate_count"
                        in cur_solution2["dynamic_graph_solution"]["metrics"]
                    ):
                        if (
                            cur_solution1["dynamic_graph_solution"]["metrics"][
                                "gate_count"
                            ]
                            > cur_solution2["dynamic_graph_solution"]["metrics"][
                                "gate_count"
                            ]
                        ):
                            compare_swaps_bigger.append(instance)
                        elif (
                            cur_solution1["dynamic_graph_solution"]["metrics"][
                                "gate_count"
                            ]
                            < cur_solution2["dynamic_graph_solution"]["metrics"][
                                "gate_count"
                            ]
                        ):
                            compare_swaps_smaller.append(instance)
                    qubits = set()
                    for edge in cur_solution1["problem"]["topology"]:
                        qubits.update(edge)

                    splittedString = instance.split("_")
                    cur_statistics = {
                        "num_qubits": len(qubits),
                        "depth": splittedString[-2],
                        "num_gate": len(cur_solution1["problem"]["gates"]),
                        "name": instance,
                    }
                    problem_statistics.append(cur_statistics)
        # except Exception as e:
        #    print(f"Ein Fehler ist aufgetreten beim laden von {instance}: {e}")

    print("Compare Swaps Smaller: ")
    print(compare_swaps_smaller)

    print("Compare Swaps Bigger: ")
    print(compare_swaps_bigger)

    print("solutions different: " + str(nr_solutions_different))
    print("solutions identical: " + str(nr_solutions_identical))
    return (
        solution_metrics1,
        solution_metrics2,
        metrics1,
        metrics2,
        compare_depth_bigger,
        compare_swaps_bigger,
        problem_statistics,
    )


def save_data_to_csv(data_1, data_2, title, file_subtitle, folder="pictures", **kwargs):
    # save data_1, data_2 as columns
    with _output_path(folder, f"{file_subtitle}.csv").open("w", encoding="utf-8") as f:
        f.write(f"{title}0;{title}1\n")
        for i in range(len(data_1)):
            f.write(f"{data_1[i]};{data_2[i]}\n")


def make_general_parity_plot(
    data_1,
    x_label,
    data_2,
    y_label,
    title,
    file_subtitle,
    min_value=None,
    max_value=None,
    folder="pictures",
    **kwargs,
):
    if min_value is None:
        min_value = min([min(data_1), min(data_2)])
    if max_value is None:
        max_value = max([max(data_1), max(data_2)])

    crossed_x = kwargs.get("crossed_x", None)
    crossed_y = kwargs.get("crossed_y", None)

    plt.figure()
    plt.scatter(data_1, data_2, color="blue", label="Data Points")

    if crossed_x is not None:
        assert len(crossed_x) == len(data_1)
        plt.scatter(
            [data_1[i] for i in range(len(data_1)) if crossed_x[i] == 1],
            [data_2[i] for i in range(len(data_1)) if crossed_x[i] == 1],
            color="red",
            marker="x",
            label="x Timeout Solutions",
        )

    if crossed_y is not None:
        assert len(crossed_y) == len(data_2)
        plt.scatter(
            [data_1[i] for i in range(len(data_1)) if crossed_y[i] == 1],
            [data_2[i] for i in range(len(data_1)) if crossed_y[i] == 1],
            color="purple",
            marker="|",
            label="y Timeout Solutions",
        )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--",
        label="Ideal Line",
    )
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=15, weight="bold")
    plt.savefig(_output_path(folder, f"{file_subtitle}.png"))


def make_parity_plot(
    data_1,
    data_2,
    title,
    file_subtitle,
    min_value=None,
    max_value=None,
    folder="pictures",
):
    if min_value is None:
        min_value = min([min(data_1), min(data_2)])
    if max_value is None:
        max_value = max([max(data_1), max(data_2)])

    plt.figure()
    plt.scatter(data_1, data_2, color="blue", label="Data Points")
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--",
        label="Ideal Line",
    )
    plt.xlabel("NoLayering", fontsize=15)
    plt.ylabel("WithLayering", fontsize=15)
    plt.title(title, fontsize=15, weight="bold")
    plt.savefig(_output_path(folder, f"{file_subtitle}.png"))
    # plt.show()


def plot_difference_over_problem_size(
    data_1,
    data_2,
    title,
    file_subtitle,
    size,
    title_size,
    count_difference=False,
    sense=1,
    folder="pictures",
):
    max_size = max(size)
    min_size = min(size)
    plt.figure()
    difference = [data_2[i] - data_1[i] for i in range(len(data_1))]

    plt.scatter(
        [size[i] for i in range(len(data_1)) if sense * difference[i] < 0],
        [difference[i] for i in range(len(data_1)) if sense * difference[i] < 0],
        color="red",
        label="Data Points",
    )
    plt.scatter(
        [size[i] for i in range(len(data_1)) if sense * difference[i] >= 0],
        [difference[i] for i in range(len(data_1)) if sense * difference[i] >= 0],
        color="green",
        label="Data Points",
    )
    if count_difference:
        print(title)
        print(
            "Number better without layering: "
            + str(
                len(
                    [
                        difference[i]
                        for i in range(len(data_1))
                        if sense * difference[i] > 0
                    ]
                )
            )
        )
        print(
            "Number better with layering: "
            + str(
                len(
                    [
                        difference[i]
                        for i in range(len(data_1))
                        if sense * difference[i] < 0
                    ]
                )
            )
        )

    # print([i for i in range(len(data_1)) if difference[i]<0])
    # print([size[i] for i in range(len(data_1)) if difference[i]<0])

    plt.plot(
        [min_size, max_size], [0, 0], color="blue", linestyle="--", label="Ideal Line"
    )
    plt.xlabel(title_size)
    plt.ylabel("difference" + title)
    plt.title(title)
    plt.savefig(_output_path(folder, f"{title}_{file_subtitle}.png"))
    # plt.show()


def plot_values_over_problem_size(
    data_1, data_2, title, file_subtitle, size1, size2, title_size, folder="pictures"
):
    plt.figure()

    plt.scatter(size1, data_1, color="blue", label="Data Points")
    plt.scatter(size2, data_2, color="black", label="Data Points")

    plt.xlabel(title_size)
    plt.ylabel("difference" + title)
    plt.title(title)
    plt.savefig(_output_path(folder, f"{file_subtitle}_values_{title_size}.png"))
    # plt.show()


def plot_histogram(data1, data2, title, folder="pictures"):
    # Example data sets
    # Calculate the relative difference
    sameData = [abs(d1 - d2) < 1.0e-6 for d1, d2 in zip(data1, data2)]
    relative_difference = [(d2 - d1) / d1 for d1, d2 in zip(data1, data2)]

    relative_difference = [
        relative_difference[i]
        for i in range(len(relative_difference))
        if not sameData[i]
    ]

    # Calculate the mean of the relative differences
    mean_relative_difference = np.mean(relative_difference)
    plt.figure()
    # Plotting the histogram

    # Calculate histogram data
    counts, bins = np.histogram(relative_difference, bins=9, range=(-0.15, 0.15))

    # Calculate percentages
    percentages = counts / len(relative_difference)

    plt.hist(bins[:-1], bins, weights=percentages, edgecolor="black", alpha=0.7)
    # plt.axvline(mean_relative_difference, color='r', linestyle='dashed', linewidth=1)
    # plt.text(mean_relative_difference, plt.ylim()[1]*0.9, f'Mean: {mean_relative_difference:.5f}', color='r', ha='center')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.title("scheduling time", fontsize=15, weight="bold")
    plt.xlabel("Relative Difference", fontsize=15)
    plt.ylabel("Percentage", fontsize=15)
    plt.ylim(0, 0.5)
    # plt.grid(True)
    plt.savefig(_output_path(folder, f"{title}_time_hist.png"))


def plot_metrics(
    path_to_sol1, path_to_sol2, file_subtitle, folder="pictures", timeout=300, **kwargs
):

    (
        solution_metrics1,
        solution_metrics2,
        metrics1,
        metrics2,
        compare_depth,
        compare_swaps,
        problem_statistics,
    ) = readInData(path_to_sol1, path_to_sol2, timeout=timeout)

    size_num_gates = [d["num_gate"] for d in problem_statistics]
    size_num_qubits = [d["num_qubits"] for d in problem_statistics]

    noise_delta1 = [d["noise_delta"] for d in metrics1]
    noise_delta2 = [d["noise_delta"] for d in metrics2]
    make_parity_plot(
        noise_delta1,
        noise_delta2,
        "noise_delta",
        "noise_delta_" + file_subtitle,
        min_value=0,
        max_value=2,
        folder=folder,
    )
    # plot_difference_over_problem_size(noise_delta1, noise_delta2,'noise_delta', 'noise_delta_num_gates'+file_subtitle,size_num_gates,'number gates',True, folder=folder)
    # plot_difference_over_problem_size(noise_delta1, noise_delta2,'noise_delta', 'noise_delta_num_qubits'+file_subtitle,size_num_qubits,'number qubits', folder=folder)
    # print(np.mean(noise_delta1))
    # print(np.mean(noise_delta2))
    # for i in range(min(size_num_qubits),max(size_num_qubits)+1):
    #    print(i)
    #    print(np.mean([noise_delta1[j]  for j in range(len(noise_delta1)) if  size_num_qubits[j]==i]))
    #    print(np.mean([noise_delta2[j]  for j in range(len(noise_delta1)) if  size_num_qubits[j]==i]))

    hop1 = [d["heavy_output_prob"] for d in metrics1]
    hop2 = [d["heavy_output_prob"] for d in metrics2]

    validIndices = [i for i in range(len(hop1)) if hop1[i] > 0.01 and hop2[i] > 0.01]

    hop1 = [hop1[i] for i in validIndices]
    hop2 = [hop2[i] for i in validIndices]

    make_parity_plot(
        hop1,
        hop2,
        "heavy_output_prob (noise)",
        "heavy_output_prob_(noise)_" + file_subtitle,
        min_value=0,
        max_value=1,
        folder=folder,
    )
    # plot_difference_over_problem_size(hop1, hop2,'heavy_output_prob (noise)', 'heavy_output_prob_(noise)_num_gates_'+file_subtitle,size_num_gates,'number gates',True,sense=-1, folder=folder)
    # plot_difference_over_problem_size(hop1, hop2,'heavy_output_prob (noise)', 'heavy_output_prob_(noise)_num_qubits_'+file_subtitle,size_num_qubits,'number qubits',  sense=-1, folder=folder)
    # print(np.mean(hop1))
    # print(np.mean(hop2))
    # for i in range(min(size_num_qubits),max(size_num_qubits)+1):
    #    print(i)
    #    print(np.mean([hop1[j]  for j in range(len(hop1)) if  size_num_qubits[j]==i]))
    #    print(np.mean([hop2[j]  for j in range(len(hop2)) if  size_num_qubits[j]==i]))

    time1 = [d["duration"] for d in metrics1]
    time2 = [d["duration"] for d in metrics2]
    make_parity_plot(
        time1, time2, "pulse time", "pulse_time_" + file_subtitle, folder=folder
    )
    # plot_difference_over_problem_size(time1, time2,'pulse time', 'pulse_time_num_gates_'+file_subtitle,size_num_gates,'number gates',True, folder=folder)
    # plot_difference_over_problem_size(time1, time2,'pulse time', 'pulse_time_num_qubits'+file_subtitle,size_num_qubits,'number qubits', folder=folder)
    # plot_histogram(time1, time2, file_subtitle, folder=folder)
    # for i in range(len(time1)):
    #     if time1[i]>time2[i]:
    #         print(time1[i]-time2[i])
    #         print(time1[i])
    #         print(time2[i])
    #         print(problem_statistics[i]['name'])

    num_Swaps1 = [d["gate_count"] for d in metrics1]
    num_Swaps2 = [d["gate_count"] for d in metrics2]
    make_parity_plot(
        num_Swaps1,
        num_Swaps2,
        "number swaps",
        "number_swaps_" + file_subtitle,
        folder=folder,
    )
    # plot_difference_over_problem_size(num_Swaps1, num_Swaps2,'number swaps', 'number_swaps_num_gates'+file_subtitle,size_num_gates,'number gates',True, folder=folder)
    # plot_difference_over_problem_size(num_Swaps1, num_Swaps2,'number swaps', 'number_swaps_num_gates'+file_subtitle,size_num_qubits,'number qubits', folder=folder)

    depth1 = [d["depth"] for d in solution_metrics1]
    depth2 = [d["depth"] for d in solution_metrics2]
    make_parity_plot(depth1, depth2, "depth", "depth_" + file_subtitle, folder=folder)
    # plot_difference_over_problem_size(depth1, depth2,'depth', 'depth_num_gates'+file_subtitle,size_num_gates,'number gates',True, folder=folder)
    # plot_difference_over_problem_size(depth1, depth2,'depth', 'depth_num_qubits'+file_subtitle,size_num_qubits,'number qubits', folder=folder)

    depth1 = [depth1[i] for i in validIndices]
    depth2 = [depth2[i] for i in validIndices]
    # plot_values_over_problem_size(hop1,hop2,'hop_depth', 'hop'+file_subtitle,depth1,depth2,'depth', folder=folder)


def make_paper_plots(
    path_to_sol1,
    x_label,
    path_to_sol2,
    y_label,
    file_subtitle,
    folder="pictures",
    **kwargs,
):

    (
        solution_metrics1,
        solution_metrics2,
        metrics1,
        metrics2,
        compare_depth,
        compare_swaps,
        problem_statistics,
    ) = readInData(path_to_sol1, path_to_sol2)
    if kwargs.get("cross_x", False):
        crossed_x = [0 for _ in range(len(solution_metrics1))]
        for i in range(len(solution_metrics1)):
            if solution_metrics1[i]["timeout"]:
                crossed_x[i] = 1
    else:
        crossed_x = None
    if kwargs.get("cross_y", False):
        crossed_y = [0 for _ in range(len(solution_metrics2))]
        for i in range(len(solution_metrics2)):
            if solution_metrics2[i]["timeout"]:
                crossed_y[i] = 1
    else:
        crossed_y = None

    noise_delta1 = [d["noise_delta"] for d in metrics1]
    noise_delta2 = [d["noise_delta"] for d in metrics2]
    make_general_parity_plot(
        noise_delta1,
        x_label,
        noise_delta2,
        y_label,
        "noise_delta",
        "noise_delta_" + file_subtitle,
        min_value=0,
        max_value=2,
        folder=folder,
        crossed_x=crossed_x,
        crossed_y=crossed_y,
    )
    save_data_to_csv(
        noise_delta1,
        noise_delta2,
        "noise_delta",
        "noise_delta_" + file_subtitle,
        folder=folder,
    )

    hop1 = [d["heavy_output_prob"] for d in metrics1]
    hop2 = [d["heavy_output_prob"] for d in metrics2]

    validIndices = [i for i in range(len(hop1)) if hop1[i] > 0.01 and hop2[i] > 0.01]
    hop1 = [hop1[i] for i in validIndices]
    hop2 = [hop2[i] for i in validIndices]
    hop1mean = np.mean(hop1)
    hop2mean = np.mean(hop2)
    print("xlabel: ", x_label)
    print(f"Mean hop1: {hop1mean}")
    print("ylabel: ", y_label)
    print(f"Mean hop2: {hop2mean}")
    make_general_parity_plot(
        hop1,
        x_label,
        hop2,
        y_label,
        "heavy_output_prob (noise)",
        "heavy_output_prob_(noise)_" + file_subtitle,
        min_value=0,
        max_value=1,
        folder=folder,
        crossed_x=crossed_x,
        crossed_y=crossed_y,
    )
    save_data_to_csv(
        hop1,
        hop2,
        "heavy_output_prob (noise)",
        "heavy_output_prob_(noise)_" + file_subtitle,
        folder=folder,
    )

    # time1=[d['duration'] for d in metrics1]
    # time2=[d['duration'] for d in metrics2]
    # make_general_parity_plot(time1, x_label, time2, y_label, 'pulse time', 'pulse_time_'+file_subtitle, folder=folder, crossed_x=crossed_x, crossed_y=crossed_y)
    # save_data_to_csv(time1, time2, 'pulse time','pulse_time_'+file_subtitle, folder=folder)

    num_Swaps1 = [d["gate_count"] for d in metrics1]
    num_Swaps2 = [d["gate_count"] for d in metrics2]
    make_general_parity_plot(
        num_Swaps1,
        x_label,
        num_Swaps2,
        y_label,
        "number swaps",
        "number_swaps_" + file_subtitle,
        folder=folder,
        crossed_x=crossed_x,
        crossed_y=crossed_y,
    )
    save_data_to_csv(
        num_Swaps1,
        num_Swaps2,
        "number swaps",
        "number_swaps_" + file_subtitle,
        folder=folder,
    )

    duration1 = [d["depth"] for d in solution_metrics1]
    duration2 = [d["depth"] for d in solution_metrics2]
    make_general_parity_plot(
        duration1,
        x_label,
        duration2,
        y_label,
        "duration",
        "duration" + "_" + file_subtitle,
        folder=folder,
        crossed_x=crossed_x,
        crossed_y=crossed_y,
    )
    save_data_to_csv(
        duration1,
        duration2,
        "duration",
        "duration" + "_" + file_subtitle,
        folder=folder,
    )

    depth1 = [d["depth"] for d in metrics1]
    depth2 = [d["depth"] for d in metrics2]
    make_general_parity_plot(
        depth1,
        x_label,
        depth2,
        y_label,
        "depth",
        "depth" + "_" + file_subtitle,
        folder=folder,
        crossed_x=crossed_x,
        crossed_y=crossed_y,
    )
    save_data_to_csv(
        depth1, depth2, "depth", "depth" + "_" + file_subtitle, folder=folder
    )


if __name__ == "__main__":

    all_sols_paths_and_abbr = {"algorithm": {}}
    for graph in ("linear", "grid", "Y"):
        all_sols_paths_and_abbr["algorithm"][graph] = {}
        for objective in ("swaps", "depth"):
            all_sols_paths_and_abbr["algorithm"][graph][objective] = {
                "layering": (
                    SOLUTIONS_ROOT / graph / objective / "WithLayering",
                    f"ALG-{graph}-{objective.upper()}-LAYER",
                ),
                "no_layering": (
                    SOLUTIONS_ROOT / graph / objective / "NoLayering",
                    f"ALG-{graph}-{objective.upper()}-NOLAYER",
                ),
            }

    # plot_metrics(path_to_sol1=all_sols_paths_and_abbr["algorithm"]["linear"]["depth"]["no_layering"][0],
    #             path_to_sol2=all_sols_paths_and_abbr["algorithm"]["linear"]["depth"]["layering"][0],
    #             file_subtitle=all_sols_paths_and_abbr["algorithm"]["linear"]["depth"]["no_layering"][1] + "_vs_" + all_sols_paths_and_abbr["algorithm"]["linear"]["depth"]["layering"][1])

    for graph in ["linear", "grid", "Y"]:
        for objective in ["swaps", "depth"]:
            print("OBJECTIVE: ", objective)
            make_paper_plots(
                path_to_sol1=all_sols_paths_and_abbr["algorithm"][graph][objective][
                    "no_layering"
                ][0],
                x_label="ALG-" + graph + "-" + objective + "-" + "NOLAYER",
                path_to_sol2=all_sols_paths_and_abbr["algorithm"][graph][objective][
                    "layering"
                ][0],
                y_label="ALG-" + graph + "-" + objective + "-" + "LAYER",
                file_subtitle=all_sols_paths_and_abbr["algorithm"][graph][objective][
                    "no_layering"
                ][1]
                + "_vs_"
                + all_sols_paths_and_abbr["algorithm"][graph][objective]["layering"][1],
                folder=f"pictures//{graph}",
            )
    # make_paper_plots(path_to_sol1=all_sols_paths_and_abbr["algorithm"][graph]["swaps"][layering_str][0],
    #            x_label="ALG-"+graph+"-"+objective+"-"+layering_short,
    #            path_to_sol2=all_sols_paths_and_abbr["algorithm"][graph]["depth"][layering_str][0],
    #            y_label="qiskit_swaps_sol",
    #            file_subtitle=f"{layering}_swaps_{graph}" + "_vs_" + f"{layering}_depth_{graph}",
    #            folder=f"pictures//{graph}")

    # make_paper_plots(path_to_sol1=all_sols_paths_and_abbr["algorithm"][graph][objective][layering_str][0],
    #                 x_label=F"alg_{objective}_sol_{layering_str}",
    #                 path_to_sol2=all_sols_paths_and_abbr["algorithm"][graph][objective][layering_str][0],
    #                 y_label="layout_sabre",
    #                 file_subtitle=f"ALG_{objective}_{layering_str}" + "_vs_" + f"routing and layout_sabre",
    #                 folder=f"pictures//{graph}")
