import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import QuantumVolume

from .runner import Runner, SimulatorRunner
from .prob_distr import ProbDistr


def run_qv_experiment(
    noisy_runner: Runner,
    num_qubits: int,
    num_trials: int = 100,
    shots: int = 100,
) -> np.array:
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
    qv_circs = [QuantumVolume(num_qubits).decompose() for _ in range(num_trials)]
    for qv_circ in qv_circs:
        qv_circ.measure_active()

    # simulate the QV circuits without noise for the comparison
    sim_runner = SimulatorRunner()
    sim_results = sim_runner.run(qv_circs, shots=shots)

    # run the QV circuits on the noisy QPU
    noisy_results = noisy_runner.run(qv_circs, shots=shots)

    return np.array(
        [
            calculate_heavy_output_probability(sim_res, noisy_res)
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
    noisy_runner: Runner,
    num_trials: int = 100,
    shots: int = 100,
    max_num_qubits: int = 6,
) -> int:
    """Find the quantum volume of a noisy runner by binary search.

    Args:
        noisy_runner (Runner): The runner to benchmark.
        num_trials (int, optional):
            The number of trials for each QV circuit size. Defaults to 100.
        shots (int, optional): The number of shots for each run. Defaults to 100.
        max_num_qubits (int, optional):
        The maximal number of qubits to constrain the binary search. Defaults to 10.

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
            noisy_runner,
            mid,
            num_trials=num_trials,
            shots=shots,
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
    """Runs and plots a single quantum volume experiment.

    Args:
        noisy_runner (Runner): The runner to benchmark.
        num_qubits (int): The number of qubits to test.
        num_trials (int, optional): The number of trials. Defaults to 100.
        shots (int, optional): The number of shots per trial. Defaults to 100.
        fig_size (tuple[float, float], optional): The size of the plot. Defaults to (12, 3.8).

    Returns:
        plt.Figure: The plot as a figure.
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


def calculate_heavy_output(prob_distr: ProbDistr) -> set[str]:
    """
    Calculates the heavy output of a probability distribution.
    A heavy output is an output with probability greater than or equal to the median.

    Args:
        prob_distr (ProbDistr): The probability distribution.

    Returns:
        list[str]: The heavy outputs.
    """
    median = np.median(list(prob_distr.values()))
    return set(k for k, v in prob_distr.items() if v >= median)


def calculate_heavy_output_probability(
    sim_result: ProbDistr, noisy_result: ProbDistr
) -> float:
    """Calculates the probability of heavy output.

    Args:
        sim_result (ProbDistr): The results of perfect simulation.
        noisy_result (ProbDistr): The noisy results.

    Returns:
        float: The probability of heavy output.
    """
    sim_heavy_output = calculate_heavy_output(sim_result)
    return sum(noisy_result.get(k, 0.0) for k in sim_heavy_output)
