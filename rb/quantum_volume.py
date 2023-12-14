import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import QuantumVolume
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from .util import counts_to_probs, CustomTranspiler, BackendTranspiler


def heavy_output(prob_distr: dict[str, float]) -> set[str]:
    """Calculates the heavy output set of a probability distribution.

    Args:
        prob_distr (dict[str, float]): The probability distribution.

    Returns:
        set[str]: The heavy output set.
    """
    median = np.median(list(prob_distr.values()))
    return set(k for k, v in prob_distr.items() if v >= median)


def heavy_output_probability(
    sim_result: dict[str, float], noisy_result: dict[str, float]
) -> float:
    """Calculates the heavy output probability of a noisy result.

    Args:
        sim_result (dict[str, float]): The ideal result.
        noisy_result (dict[str, float]): The noisy result.

    Returns:
        float: The heavy output probability.
    """
    sim_heavy_output = heavy_output(sim_result)
    return sum(noisy_result.get(k, 0.0) for k in sim_heavy_output)


def run_qv_experiment(
    qpu: BackendV2,
    num_qubits: int,
    num_trials: int = 100,
    shots: int = 100,
    optimization_level: int = 3,
    custom_transpiler: CustomTranspiler | None = None,
) -> np.array:
    """Runs a single quantum volume experiment.

    Args:
        qpu (BackendV2): The noisy QPU.
        num_qubits (int): The number of qubits to test $d$.
        num_trials (int, optional): Number of trials. Defaults to 100.
        shots (int, optional): Number of shots for each trial. Defaults to 100.
        optimization_level (int, optional): Optimization level for transpiling. Defaults to 3.
        custom_transpiler (CustomTranspiler | None, optional): A custom transpiler. Defaults to None.

    Returns:
        np.array: The heavy output probabilities for each trial.
    """
    if custom_transpiler is None:
        custom_transpiler = BackendTranspiler(
            qpu, optimization_level=optimization_level
        )

    qv_circs = [QuantumVolume(num_qubits).decompose() for _ in range(num_trials)]
    for qv_circ in qv_circs:
        qv_circ.measure_active()

    qv_circs = custom_transpiler.run(qv_circs)

    # simulate the QV circuits without noise for the comparison
    sim = AerSimulator()
    sim_results = counts_to_probs(sim.run(qv_circs, shots=shots).result().get_counts())

    # run the QV circuits on the noisy QPU
    noisy_results = counts_to_probs(
        qpu.run(qv_circs, shots=shots).result().get_counts()
    )

    return np.array(
        [
            heavy_output_probability(sim_res, noisy_res)
            for sim_res, noisy_res in zip(sim_results, noisy_results)
        ]
    )


def is_successful(hops: np.array) -> bool:
    """Determines if a quantum volume experiment is successful.

    Args:
        hops (np.array): The heavy output probabilities for each trial.

    Returns:
        bool: If the experiment is successful.
    """
    num_trials = hops.shape[0]
    mean_hop = np.mean(hops)
    sigma_hop = (mean_hop * ((1.0 - mean_hop) / num_trials)) ** 0.5
    threshold = 2 / 3 + 2 * sigma_hop
    return mean_hop > threshold


def find_quantum_volume(
    qpu: BackendV2,
    num_trials: int = 100,
    shots: int = 100,
    max_num_qubits: int = 6,
    custom_transpiler: CustomTranspiler | None = None,
) -> int:
    """Find the quantum volume of a noisy runner by binary search.

    Args:
        noisy_runner (Runner): The runner to benchmark.
        num_trials (int, optional):
            The number of trials for each QV circuit size. Defaults to 100.
        shots (int, optional): The number of shots for each run. Defaults to 100.
        max_num_qubits (int, optional):
            The maximal number of qubits to constrain the binary search. Defaults to 10.
        custom_transpiler (CustomTranspiler | None, optional): A custom transpiler. Defaults to None.

    Returns:
        int: The quantum volume.
    """
    results = {}
    lower_bound = 1
    upper_bound = max_num_qubits
    while lower_bound != upper_bound:
        mid = (lower_bound + upper_bound) // 2
        print(f"-------\nTrying for QV {2 ** mid}...")
        qv_result = run_qv_experiment(
            qpu,
            mid,
            num_trials=num_trials,
            shots=shots,
            custom_transpiler=custom_transpiler,
        )
        results[mid] = qv_result

        if is_successful(qv_result):
            lower_bound = mid + 1
            print(f"✅")
        else:
            upper_bound = mid
            print(f"❌")
    print(f"QV: {2 ** (lower_bound - 1)} 🎉")
    return 2 ** (lower_bound - 1)


def plot_qv_experiment(
    hops: np.array,
    fig_size: tuple[float, float] = (12, 3.8),
) -> plt.Figure:
    """Plot the results of a quantum volume experiment.

    Args:
        hops (np.array): The heavy output probabilities for each trial.
        fig_size (tuple[float, float], optional):
            The size of the plot. Defaults to (12, 3.8).

    Returns:
        plt.Figure: The figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    _plot_qv_probs(ax1, hops)
    _plot_qv_distr(ax2, hops)
    ax1.set_title(f"(a) HOPs per Trial", fontweight="bold")
    ax2.set_title(f"(b) HOPs Distribution", fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=10,
    )
    fig.tight_layout()
    return fig


def _plot_qv_distr(ax: plt.Axes, hops: np.array, z: int = 2):
    bins = np.arange(0, 1.05, 0.05)
    hist, _ = np.histogram(hops, bins=bins)
    hist = hist / np.sum(hist)

    num_trials = hops.shape[0]
    mean_hop = np.mean(hops)
    sigma_hop = (mean_hop * ((1.0 - mean_hop) / num_trials)) ** 0.5
    threshold = 2 / 3 + z * sigma_hop

    # plot the histogram
    x_hist = bins[:-1] + 0.025
    ax.bar(x_hist, hist, width=0.05, align="center", color="gray")
    ax.set_xlabel("Heavy Output Probability")
    ax.set_ylabel("Frequency")
    ax.axvline(threshold, color="red", label="Threshold", linewidth=3, linestyle="--")
    ax.axvline(mean_hop, color="blue", label="Mean", linewidth=3)

    # plot a interpolated curve over the histogram
    x_interp = np.linspace(0, 1, 1000)
    y = np.interp(x_interp, x_hist, hist)
    ax.plot(x_interp, y, color="black", linewidth=3, linestyle="--")


def _plot_qv_probs(ax: plt.Axes, hops: np.array, z: int = 2):
    num_trials = hops.shape[0]
    mean_hop = np.mean(hops)
    sigma_hop = (mean_hop * ((1.0 - mean_hop) / num_trials)) ** 0.5
    threshold = 2 / 3 + z * sigma_hop

    x = np.arange(1, num_trials + 1)
    ax.plot(x, hops, "o", color="black")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Heavy Output Probability")
    ax.axhline(threshold, color="red", label="Threshold", linewidth=3, linestyle="--")
    ax.axhline(mean_hop, color="blue", label="Mean", linewidth=3)
