import logging

from qiskit.providers.fake_provider import FakeMumbaiV2

from qpu_bench.rand_bench import run_randomized_benchmarking
from qpu_bench.runner import SimulatedBackendRunner

logger = logging.getLogger("qpu_bench")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    # create a noisy simulator
    backend = FakeMumbaiV2()

    # create a runner of a noisy simulator
    runner = SimulatedBackendRunner(
        backend, compiler_options={"optimization_level": 0, "initial_layout": [8, 9]}
    )
    epc = run_randomized_benchmarking(runner, 2, 1000, 20)
    
    print("-----------------------------")
    print("The Error Per Clifford is: ", epc)
    print("-----------------------------")


if __name__ == "__main__":
    main()
