import numpy as np
from scipy.optimize import curve_fit

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford

from .runner import Runner


def _generate_rb_circuits(
    num_qubits: int, max_length: int = 100, step: int = 10
) -> list[QuantumCircuit]:
    """
    Generates a list of RB circuits with a number of Cliffords
    in range [1, max_length] with step size "step".

    Args:
        num_qubits (int): The number of qubits.
        max_length (int, optional): The maximal number of cliffords. Defaults to 100.
        step (int, optional): The stepsize. Defaults to 10.

    Returns:
        list[QuantumCircuit]: The RB circuits.
    """
    circs = []
    for i in range(1, max_length + 1, step):
        circ = QuantumCircuit(num_qubits)
        for _ in range(i):
            circ.append(random_clifford(num_qubits).to_instruction(), range(num_qubits))
        inv = Clifford.from_circuit(circ).adjoint().to_instruction()
        circ.compose(inv, inplace=True)
        circ.measure_all()
        circs.append(circ)
    return circs


def _fit_to_exp_decay(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fits the experimental data to an exponential decay function.
    Returns the decay constant alpha.

    Args:
        x (np.ndarray): The x values.
        y (np.ndarray): The y values.

    Returns:
        float: The decay constant alpha.
    """

    def _exp_decay(x, a, alpha, b):
        return a * alpha**x + b

    popt, _ = curve_fit(_exp_decay, x, y)
    return popt[1]


def run_randomized_benchmarking(
    noisy_runner: Runner,
    num_qubits: int,
    max_length: int = 100,
    step: int = 10,
    shots: int = 100,
) -> int:
    """
    Runs standard randomized benchmarking on a noisy backend
    and returns the error per Clifford.

    Args:
        noisy_runner (Runner): The runner for the noisy backend.
        num_qubits (int): The number of qubits to benchmark.
        max_length (int, optional): Max. number of cliffords. Defaults to 100.
        step (int, optional): Stepsize of clifford generation. Defaults to 10.
        shots (int, optional): Number of shots to run each RB circuit. Defaults to 100.

    Returns:
        int: The error per Clifford.
    """

    circuits = _generate_rb_circuits(num_qubits, max_length, step)

    # Run the circuits
    prob_dists = noisy_runner.run(circuits, shots=shots)

    # fit the data to an exponential decay
    y = np.array([probs["0" * num_qubits] for probs in prob_dists])
    x = np.arange(1, max_length, step)
    alpha = _fit_to_exp_decay(x, y)

    # calculate the error per Clifford
    epc = (1 - alpha) * (2**num_qubits - 1) / (2**num_qubits)
    return epc
