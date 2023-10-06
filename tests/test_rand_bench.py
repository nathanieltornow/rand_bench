from qpu_bench.rand_bench import _generate_rb_circuits
from qpu_bench.runner import SimulatorRunner


def test_clifford_generation():
    circuits = _generate_rb_circuits(1, 40, 1)
    runner = SimulatorRunner()
    prob_dists = runner.run(circuits)
    assert all(probs["0"] == 1.0 for probs in prob_dists)
