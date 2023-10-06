import math
import logging

import numpy as np
from qiskit.circuit.library import QuantumVolume

from .runner import Runner, SimulatorRunner
from .prob_distr import ProbDistr


logger = logging.getLogger("qpu_bench")


def _calculate_heavy_output(prob_distr: ProbDistr) -> list[str]:
    """
    Calculates the heavy output of a probability distribution.
    A heavy output is an output with probability greater than or equal to the median.

    Args:
        prob_distr (ProbDistr): The probability distribution.

    Returns:
        list[str]: The heavy outputs.
    """
    median = np.median(list(prob_distr.values()))
    return [k for k, v in prob_distr.items() if v >= median]


def _calculate_heavy_output_probability(
    sim_result: ProbDistr, noisy_result: ProbDistr
) -> float:
    """Calculates the probability of heavy output.

    Args:
        sim_result (ProbDistr): The results of perfect simulation.
        noisy_result (ProbDistr): The noisy results.

    Returns:
        float: The probability of heavy output.
    """
    sim_heavy_output = _calculate_heavy_output(sim_result)
    return sum(noisy_result.get(k, 0.0) for k in sim_heavy_output)


def _run_qv_experiment(
    noisy_runner: Runner,
    num_qubits: int,
    num_trials: int = 100,
    shots: int = 100,
    z: int = 2,
) -> bool:
    """
    Runs a single quantum volume experiment.
    Returns true, if the experiment passes for the specific number of qubits.

    Args:
        noisy_runner (Runner): The runner to benchmark.
        num_qubits (int): The number of qubits to test.
        num_trials (int, optional): The number of trials in the test. Defaults to 100.
        shots (int, optional): The number of shots for each trial. Defaults to 100.

    Returns:
        bool: If the experiment passes.
    """
    qv_circs = [QuantumVolume(num_qubits) for _ in range(num_trials)]
    for qv_circ in qv_circs:
        qv_circ.measure_active()

    # simulate the QV circuits without noise for the comparison
    sim_runner = SimulatorRunner()
    sim_results = sim_runner.run(qv_circs, shots=shots)

    # run the QV circuits on the noisy QPU
    noisy_results = noisy_runner.run(qv_circs, shots=shots)

    # calculate the probability of heavy output for each trial
    hops = []
    for sim_res, noisy_res in zip(sim_results, noisy_results):
        hops.append(_calculate_heavy_output_probability(sim_res, noisy_res))

    # calculate the mean hop, and threshold that must be exceeded
    mean_hop = np.mean(hops)
    sigma_hop = (mean_hop * ((1.0 - mean_hop) / num_trials)) ** 0.5
    threshold = 2 / 3 + z * sigma_hop

    logger.info(f"Mean hop: {mean_hop}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Sigma hop: {sigma_hop}")

    # return true if the mean hop is greater than the threshold
    if mean_hop < threshold:
        return False
    return True


def find_quantum_volume(
    noisy_runner: Runner,
    num_trials: int = 100,
    max_num_qubits: int = 7,
    shots: int = 100,
    z: int = 2,
) -> int:
    """Finds the quantum volume of a noisy backend using binary search.

    Args:
        noisy_runner (Runner): The runner for the noisy backend.
        num_trials (int, optional): The number of trials for each circuit. Defaults to 100.
        max_num_qubits (int, optional): The maximum number of qubits to test. Defaults to 7.
        shots (int, optional): The number of shots for each circuit. Defaults to 1024.

    Returns:
        int: The quantum volume of the noisy backend.
    """
    lower_bound = 1
    upper_bound = max_num_qubits
    while lower_bound != upper_bound:
        mid = (lower_bound + upper_bound) // 2
        logger.info(f"Trying depth {mid}")
        if _run_qv_experiment(
            noisy_runner,
            mid,
            num_trials=num_trials,
            shots=shots,
            z=z,
        ):
            lower_bound = mid + 1
            logger.info(f"Quantum volume {2 ** mid} passed")
        else:
            upper_bound = mid
            logger.info(f"Quantum volume {2 ** mid} failed")
    return 2 ** (lower_bound - 1)
