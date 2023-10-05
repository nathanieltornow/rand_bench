import math
import logging

import numpy as np
from qiskit.circuit.library import QuantumVolume

from .runner import Runner, SimulatorRunner
from .prob_distr import ProbDistr


def _calculate_heavy_output(prob_distr: ProbDistr) -> list[str]:
    median = np.median(list(prob_distr.values()))
    return [k for k, v in prob_distr.items() if v >= median]


def _calculate_heavy_output_probability(
    sim_result: ProbDistr, noisy_result: ProbDistr
) -> float:
    sim_heavy_output = _calculate_heavy_output(sim_result)
    return sum(noisy_result.get(k, 0.0) for k in sim_heavy_output)


def _run_qv_experiment(
    noisy_runner: Runner, num_qubits: int, num_trials: int = 100, shots: int = 1024
) -> bool:
    qv_circs = [QuantumVolume(num_qubits) for _ in range(num_trials)]
    for qv_circ in qv_circs:
        qv_circ.measure_active()

    sim_runner = SimulatorRunner()
    sim_results = sim_runner.run(qv_circs, shots=shots)

    noisy_results = noisy_runner.run(qv_circs, shots=shots)

    hops = []
    for sim_res, noisy_res in zip(sim_results, noisy_results):
        hops.append(_calculate_heavy_output_probability(sim_res, noisy_res))

    mean_hop = np.mean(hops)
    sigma_hop = (mean_hop * ((1.0 - mean_hop) / num_trials)) ** 0.5

    z = 2
    threshold = 2 / 3 + z * sigma_hop
    z_value = (mean_hop - 2 / 3) / sigma_hop
    confidence = 0.5 * (1 + math.erf(z_value / 2**0.5))

    if confidence < 0.977:
        return False

    if mean_hop < threshold:
        return False

    return True


def find_quantum_volume(
    noisy_runner: Runner,
    num_trials: int = 100,
    max_num_qubits: int = 7,
    shots: int = 1024,
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
        logging.info(f"Trying depth {mid}")
        if _run_qv_experiment(noisy_runner, mid, num_trials=num_trials, shots=shots):
            lower_bound = mid + 1
            logging.info(f"Quantum volume {2 ** mid} passed")
        else:
            upper_bound = mid
    return 2 ** (lower_bound - 1)
