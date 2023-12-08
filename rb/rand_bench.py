from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.providers import BackendV2
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import random_clifford, Clifford

from .util import counts_to_probs

OPT_LVL = 1
NUM_SAMPLES = 5

SEQUENCE_LENGTHS_1Q = np.arange(2, 2000, 400)
SEQUENCE_LENGTHS_2Q = np.arange(2, 200, 20)


@dataclass
class RBResult:
    num_qubits: int
    sequence_lengths: np.array
    mean_probs: np.array
    err_probs: np.array


def exp_decay(x, a, alpha, b):
    """The exponential decay function."""
    return a * alpha**x + b


def benchmark_qpu_1q(qpu: BackendV2, gate: Gate | None = None) -> list[float]:
    errors = []
    for qubit in range(qpu.num_qubits):
        epc = get_errors_on_qubits(qpu, [qubit], SEQUENCE_LENGTHS_1Q, gate)
        errors.append(epc * 100)
    return errors


def benchmark_qpu_2q(
    qpu: BackendV2, gate: Gate | None = None
) -> dict[tuple[int, int], float]:
    errors = {}
    coupling_map: CouplingMap = qpu.coupling_map

    done_edges = set()
    for qubit1, qubit2 in coupling_map.get_edges():
        # skip if the edge has already been done
        if (qubit2, qubit1) in done_edges:
            continue
        done_edges.add((qubit1, qubit2))

        epc = get_errors_on_qubits(qpu, [qubit1, qubit2], SEQUENCE_LENGTHS_2Q, gate)
        errors[(qubit1, qubit2)] = epc * 100
    return errors


def get_errors_on_qubits(
    qpu: BackendV2,
    qubits: list[int],
    sequence_lengths: np.ndarray,
    interleaved_gate: Gate | None = None,
) -> float:
    if interleaved_gate is None:
        return _errors_on_qubits_standard(
            qpu, qubits, sequence_lengths=sequence_lengths
        )
    return _errors_on_qubits_interleaved(
        qpu, qubits, interleaved_gate, sequence_lengths
    )


def _errors_on_qubits_standard(
    qpu: BackendV2, qubits: list[int], sequence_lengths: np.ndarray
) -> float:
    result = run_standard_rb(
        qpu,
        qubits,
        sequence_lengths=sequence_lengths,
    )
    return calculate_standard_epc(result)


def _errors_on_qubits_interleaved(
    qpu: BackendV2, qubits: list[int], gate: Gate, sequence_lengths: np.ndarray
) -> float:
    if gate.num_qubits != len(qubits):
        raise ValueError("Gate has to have the same number of qubits as the qubits")

    res_std, res_inter = run_interleaved_rb(
        qpu,
        qubits,
        gate=gate,
        sequence_lengths=sequence_lengths,
    )
    return calculate_interleaved_epc(res_std, res_inter)


def plot_rb_probs(
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
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(r"$P(|0\rangle)$")


def plot_rb_fit(
    ax: plt.Axes, rb_result: RBResult, label: str = "Fit", color: str = "red"
) -> None:
    params, _ = curve_fit(
        exp_decay,
        rb_result.sequence_lengths,
        rb_result.mean_probs,
        bounds=(0, 1),
        p0=_guesses(rb_result.mean_probs),
    )
    ax.plot(
        rb_result.sequence_lengths,
        exp_decay(rb_result.sequence_lengths, *params),
        "-",
        label=label,
        color=color,
        linewidth=3,
    )


def generate_clifford_sequences(
    num_qubits: int,
    sequence_lengths: np.array,
) -> list[list[Clifford]]:
    sequences = []
    for i in sequence_lengths:
        sequences.append([random_clifford(num_qubits) for _ in range(i)])
    return sequences


def clifford_sequences_to_circuits(
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
            circ.barrier(range(num_qubits))
        inv = Clifford.from_circuit(circ).adjoint().to_instruction()
        circ.compose(inv, inplace=True)
        circ.measure_all()
        circs.append(circ)
    return circs


def _guesses(probs: np.array) -> tuple[float, float, float]:
    a = probs[0] - probs[-1]
    b = probs[-1]
    alpha = 0.99
    return a, alpha, b


def calculate_standard_epc(rb_result: RBResult) -> float:
    # fit the data to an exponential decay
    params, _ = curve_fit(
        exp_decay,
        rb_result.sequence_lengths,
        rb_result.mean_probs,
        bounds=(0, 1),
        p0=_guesses(rb_result.mean_probs),
    )
    # calculate the error per Clifford
    d = 2**rb_result.num_qubits
    alpha = params[1]
    epc = (d - 1) * (1 - alpha) / d
    return epc


def calculate_interleaved_epc(standard_rb: RBResult, interleaved_rb: RBResult) -> float:
    # fit the data to an exponential decay
    params, _ = curve_fit(
        exp_decay,
        standard_rb.sequence_lengths,
        standard_rb.mean_probs,
        bounds=(0, 1),
        p0=_guesses(standard_rb.mean_probs),
    )
    inter_params, _ = curve_fit(
        exp_decay,
        interleaved_rb.sequence_lengths,
        interleaved_rb.mean_probs,
        bounds=(0, 1),
        p0=_guesses(interleaved_rb.mean_probs),
    )

    # calculate the error of the gate
    d = 2**interleaved_rb.num_qubits
    alpha, alpha_c = params[1], inter_params[1]
    gate_error = ((d - 1) * (1 - alpha_c / alpha)) / d
    return gate_error


def run_standard_rb(
    qpu: BackendV2,
    qubits: list[int],
    sequence_lengths: np.ndarray,
    shots: int = 100,
    num_samples: int = NUM_SAMPLES,
) -> RBResult:
    all_circuits = []

    num_qubits = len(qubits)

    for _ in range(num_samples):
        # Generate the random clifford sequences
        clifford_sequences = generate_clifford_sequences(num_qubits, sequence_lengths)
        circuits = clifford_sequences_to_circuits(clifford_sequences, num_qubits)
        circuits = transpile(
            circuits, backend=qpu, initial_layout=qubits, optimization_level=OPT_LVL
        )
        all_circuits += circuits

    # Run the circuits
    counts = qpu.run(all_circuits, shots=shots).result().get_counts()
    prob_dists = counts_to_probs(counts)

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
    qpu: BackendV2,
    qubits: list[int],
    gate: Gate,
    sequence_lengths: np.ndarray,
    shots: int = 100,
    num_samples: int = NUM_SAMPLES,
) -> tuple[RBResult, RBResult]:
    try:
        _ = Clifford(gate)
    except ValueError:
        raise ValueError("The interleaved gate must be a Clifford.")

    assert len(qubits) == gate.num_qubits

    num_qubits = gate.num_qubits

    all_standard_circuits = []
    all_interleaved_circuits = []

    for _ in range(num_samples):
        # Generate the random clifford sequences
        clifford_sequences = generate_clifford_sequences(num_qubits, sequence_lengths)
        circuits = clifford_sequences_to_circuits(clifford_sequences, num_qubits)
        circuits = transpile(
            circuits, backend=qpu, initial_layout=qubits, optimization_level=OPT_LVL
        )

        inter_circuits = clifford_sequences_to_circuits(
            clifford_sequences, num_qubits, gate
        )
        inter_circuits = transpile(
            inter_circuits,
            backend=qpu,
            initial_layout=qubits,
            optimization_level=OPT_LVL,
        )

        all_standard_circuits += circuits
        all_interleaved_circuits += inter_circuits

    prob_dists = counts_to_probs(
        qpu.run(all_standard_circuits, shots=shots).result().get_counts()
    )
    inter_prob_dists = counts_to_probs(
        qpu.run(all_interleaved_circuits, shots=shots).result().get_counts()
    )

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
