from generate_random_instances import gen_instances_linear, gen_instances_Y


def main():
    print("Generating Linear-7 instances: 3 depths x 100 instances", flush=True)
    gen_instances_linear(different_qubits=[7], show_progress=True)
    print("Linear-7 generation complete", flush=True)
    print("Generating Y-7 instances: 3 depths x 100 instances", flush=True)
    gen_instances_Y(different_qubits=[7], show_progress=True)
    print("Y-7 generation complete", flush=True)


if __name__ == "__main__":
    main()
