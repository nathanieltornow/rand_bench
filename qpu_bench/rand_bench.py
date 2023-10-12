import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info import random_clifford, Clifford

from .runner import Runner


def _generate_clifford_sequences(
    num_qubits: int,
    length_range: tuple[int, int, int] = (1, 100, 10),
    num_samples: int = 3,
) -> list[list[Clifford]]:
    """
    Generates a list of Clifford sequences with a number of Cliffords
    in range [1, max_length] with step size "step".

    Args:
        num_qubits (int): The number of qubits.
        length_range (tuple[int, int, int], optional): A range to specify which legths the
            clifford-sequences should have. Defaults to (1, 100, 10).
        num_samples (int, optional): Number of samples per sequence. Defaults to 3.

    Returns:
        list[list[Clifford]]: The Clifford sequences.
    """
    sequences = []
    for i in range(*length_range):
        for _ in range(num_samples):
            sequences.append([random_clifford(num_qubits) for _ in range(i)])
    return sequences


def _clifford_sequences_to_circuits(
    clifford_sequences: list[list[Clifford]],
    num_qubits: int,
    interleaved_gate: Gate | None = None,
) -> list[QuantumCircuit]:
    """Converts a list of Clifford sequences to a list of QuantumCircuits.

    Args:
        clifford_sequences (list[list[Clifford]]): The Clifford sequences.
        num_qubits (int): The number of qubits of the cliffords.
        interleaved_gate (Gate | None, optional): A optional interleaved gate. Defaults to None.

    Raises:
        ValueError: If the interleaved gate is not a Clifford.

    Returns:
        list[QuantumCircuit]: The list of QuantumCircuits.
    """
    if interleaved_gate is not None:
        # check whether the interleaved gate is a Clifford
        try:
            interleaved_clifford = Clifford(interleaved_gate)
        except QiskitError:
            raise ValueError("The interleaved gate must be a Clifford.")

        # interleave the gate into all sequences
        interleaved_sequences = []
        for sequence in clifford_sequences:
            interleaved_sequence = []
            for clifford in sequence:
                interleaved_sequence.append(clifford)
                interleaved_sequence.append(interleaved_clifford)
            interleaved_sequences.append(interleaved_sequence)
        clifford_sequences = interleaved_sequences

    circs = []
    for sequence in clifford_sequences:
        circ = QuantumCircuit(num_qubits)
        for clifford in sequence:
            circ.append(clifford.to_instruction(), range(num_qubits))
        inv = Clifford.from_circuit(circ).adjoint().to_instruction()
        circ.compose(inv, inplace=True)
        circ.measure_all()
        circs.append(circ)
    return circs


def run_standard_rb(
    noisy_runner: Runner,
    num_qubits: int,
    length_range: tuple[int, int, int] = (1, 100, 10),
    shots: int = 100,
    num_samples: int = 3,
    ax: plt.Axes | None = None,
) -> int:
    """
    Runs a randomized benchmark on a noisy backend
    and returns the error per Clifford.

    Args:
        noisy_runner (Runner): The runner for the noisy backend.
        num_qubits (int): The number of qubits.
        length_range (tuple[int, int, int], optional): A range to specify which legths the
            clifford-sequences should have. Defaults to (1, 100, 10).
        shots (int, optional): Number of shots for each RB circuit. Defaults to 100.
        num_samples (int, optional): Number of samples per sequence. Defaults to 3.
        ax (plt.Axes | None, optional): A axis to plot the experiment data to. Defaults to None.

    Returns:
        int: The error per Clifford.
    """

    # Generate the random clifford sequences
    clifford_sequences = _generate_clifford_sequences(
        num_qubits, length_range, num_samples
    )
    circuits = _clifford_sequences_to_circuits(clifford_sequences, num_qubits)

    # Run the circuits
    prob_dists = noisy_runner.run(circuits, shots=shots)

    x = np.arange(*length_range)
    x = np.repeat(x, num_samples)
    y = np.array([probs.get("0" * num_qubits, 0.0) for probs in prob_dists])

    # fit the data to an exponential decay
    params, _ = curve_fit(_exp_decay, x, y, bounds=(0, 1))

    if ax is not None:
        ax.plot(x, y, ".", label="RB data")
        ax.plot(x, _exp_decay(x, *params), "-", label="Fit")
        ax.set_xlabel("Clifford length")
        ax.set_ylabel("P(0)")

    # calculate the error per Clifford
    d = 2**num_qubits
    alpha = params[1]
    epc = (d - 1) * (1 - alpha) / d
    return epc


def run_interleaved_rb(
    noisy_runner: Runner,
    gate: Gate,
    length_range: tuple[int, int, int] = (1, 100, 10),
    num_samples: int = 3,
    shots: int = 100,
    ax: plt.Axes | None = None,
) -> float:
    """
    Runs a interleaved randomized benchmark on a noisy backend
    to get the error of a given clifford gate.

    Args:
        noisy_runner (Runner): The runner for the noisy backend.
        gate (Gate): The gate to benchmark.
        length_range (tuple[int, int, int], optional): A range to specify which legths the
            clifford-sequences should have. Defaults to (1, 100, 10).
        num_samples (int, optional): Number of samples per sequence. Defaults to 3.
        shots (int, optional): Number of shots for each RB circuit. Defaults to 100.
        ax (plt.Axes | None, optional): A axis to plot the experiment data to. Defaults to None.

    Raises:
        ValueError: If the gate is not a clifford.

    Returns:
        float: The error of the given clifford gate.
    """
    try:
        _ = Clifford(gate)
    except ValueError:
        raise ValueError("The interleaved gate must be a Clifford.")

    num_qubits = gate.num_qubits

    # Generate the random clifford sequences
    clifford_sequences = _generate_clifford_sequences(
        num_qubits, length_range, num_samples
    )
    circuits = _clifford_sequences_to_circuits(clifford_sequences, num_qubits)
    inter_circuits = _clifford_sequences_to_circuits(
        clifford_sequences, num_qubits, gate
    )

    # Run the clifford circuits on the noisy backend
    prob_dists = noisy_runner.run(circuits, shots=shots)
    inter_prob_dists = noisy_runner.run(inter_circuits, shots=shots)

    # store the probabilities of measuring 0 into numpy arrays
    x = np.arange(*length_range)
    x = np.repeat(x, num_samples)
    y = np.array([probs.get("0" * num_qubits, 0.0) for probs in prob_dists])
    inter_y = np.array([probs.get("0" * num_qubits, 0.0) for probs in inter_prob_dists])

    # fit the data to an exponential decay
    params, _ = curve_fit(_exp_decay, x, y, bounds=(0, 1))
    inter_params, _ = curve_fit(_exp_decay, x, inter_y, bounds=(0, 1))

    # calculate the error of the gate
    d = 2**num_qubits
    alpha, alpha_c = params[1], inter_params[1]
    gate_error = ((d - 1) * (1 - alpha_c / alpha)) / d

    # plot the onto the given axis
    if ax is not None:
        ax.plot(x, y, ".", label="Standard")
        ax.plot(x, inter_y, ".", label="Interleaved")
        ax.plot(x, _exp_decay(x, *params), "-", label="Standard fit")
        ax.plot(x, _exp_decay(x, *inter_params), "--", label="Interleaved fit")
        ax.set_xlabel("Clifford length")
        ax.set_ylabel("P(0)")

    return gate_error


def _exp_decay(x, a, alpha, b):
    """The exponential decay function."""
    return a * alpha**x + b
