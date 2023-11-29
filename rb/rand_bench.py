from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info import random_clifford, Clifford

from .runner import Runner


@dataclass
class RBResult:
    num_qubits: int
    sequence_lengths: np.array
    mean_probs: np.array
    err_probs: np.array


def _plot_rb_probs(
    ax: plt.Axes,
    rb_result: RBResult,
    color: str = "blue",
    label: str = "Probs",
) -> None:
    ax.errorbar(
        rb_result.sequence_lengths,
        rb_result.mean_probs,
        yerr=rb_result.err_probs,
        fmt="o",
        color=color,
        capsize=3,
        label=label,
    )


def _plot_rb_fit(
    ax: plt.Axes, rb_result: RBResult, label: str = "Fit", color: str = "red"
) -> None:
    params, _ = curve_fit(
        _exp_decay, rb_result.sequence_lengths, rb_result.mean_probs, bounds=(0, 1)
    )
    print(params)
    ax.plot(
        rb_result.sequence_lengths,
        _exp_decay(rb_result.sequence_lengths, *params),
        "-",
        label=label,
        color=color,
        linewidth=3,
    )


def _generate_clifford_sequences(
    num_qubits: int,
    sequence_lengths: np.array,
) -> list[list[Clifford]]:
    sequences = []
    for i in sequence_lengths:
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


def calculate_standard_epc(rb_result: RBResult) -> float:
    # fit the data to an exponential decay
    params, _ = curve_fit(
        _exp_decay, rb_result.sequence_lengths, rb_result.mean_probs, bounds=(0, 1)
    )
    # calculate the error per Clifford
    d = 2**rb_result.num_qubits
    alpha = params[1]
    epc = (d - 1) * (1 - alpha) / d
    return epc


def calculate_interleaved_epc(standard_rb: RBResult, interleaved_rb: RBResult) -> float:
    # fit the data to an exponential decay
    params, _ = curve_fit(
        _exp_decay, standard_rb.sequence_lengths, standard_rb.mean_probs, bounds=(0, 1)
    )
    inter_params, _ = curve_fit(
        _exp_decay,
        interleaved_rb.sequence_lengths,
        interleaved_rb.mean_probs,
        bounds=(0, 1),
    )

    # calculate the error of the gate
    d = 2**interleaved_rb.num_qubits
    alpha, alpha_c = params[1], inter_params[1]
    gate_error = ((d - 1) * (1 - alpha_c / alpha)) / d
    return gate_error


def run_standard_rb(
    noisy_runner: Runner,
    num_qubits: int,
    sequence_lengths: np.ndarray,
    shots: int = 100,
    num_samples: int = 3,
) -> int:
    all_circuits = []

    for _ in range(num_samples):
        # Generate the random clifford sequences
        clifford_sequences = _generate_clifford_sequences(num_qubits, sequence_lengths)
        circuits = _clifford_sequences_to_circuits(clifford_sequences, num_qubits)
        all_circuits += circuits

    # Run the circuits
    prob_dists = noisy_runner.run(all_circuits, shots=shots)

    y = np.array([probs.get("0" * num_qubits, 0.0) for probs in prob_dists]).reshape(
        (num_samples, -1)
    )
    y_mean = np.mean(y, axis=0)
    y_err = np.std(y, axis=0)

    return RBResult(
        num_qubits,
        sequence_lengths,
        y_mean,
        y_err,
    )


def run_interleaved_rb(
    noisy_runner: Runner,
    gate: Gate,
    sequence_lengths: np.ndarray,
    shots: int = 100,
    num_samples: int = 3,
) -> tuple[RBResult, RBResult]:
    try:
        _ = Clifford(gate)
    except ValueError:
        raise ValueError("The interleaved gate must be a Clifford.")

    num_qubits = gate.num_qubits

    all_standard_circuits = []
    all_interleaved_circuits = []

    for _ in range(num_samples):
        # Generate the random clifford sequences
        clifford_sequences = _generate_clifford_sequences(num_qubits, sequence_lengths)
        circuits = _clifford_sequences_to_circuits(clifford_sequences, num_qubits)
        inter_circuits = _clifford_sequences_to_circuits(
            clifford_sequences, num_qubits, gate
        )

        all_standard_circuits += circuits
        all_interleaved_circuits += inter_circuits

    # Run the clifford circuits on the noisy backend
    prob_dists = noisy_runner.run(all_standard_circuits, shots=shots)
    inter_prob_dists = noisy_runner.run(all_interleaved_circuits, shots=shots)

    probs = np.array(
        [probs.get("0" * num_qubits, 0.0) for probs in prob_dists]
    ).reshape((num_samples, -1))
    inter_probs = np.array(
        [probs.get("0" * num_qubits, 0.0) for probs in inter_prob_dists]
    ).reshape((num_samples, -1))

    y = np.mean(probs, axis=0)
    y_err = np.std(probs, axis=0)
    inter_y = np.mean(inter_probs, axis=0)
    inter_y_err = np.std(inter_probs, axis=0)

    std_res = RBResult(
        num_qubits,
        sequence_lengths,
        y,
        y_err,
    )
    inter_res = RBResult(
        num_qubits,
        sequence_lengths,
        inter_y,
        inter_y_err,
    )
    return std_res, inter_res


def _exp_decay(x, a, alpha, b):
    """The exponential decay function."""
    return a * alpha**x + b
