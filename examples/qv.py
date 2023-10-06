import logging

from qiskit.providers.fake_provider import FakeMumbaiV2

from qpu_bench.quantum_volume import find_quantum_volume
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
    runner = SimulatedBackendRunner(backend)

    # run the quantum volume algorithm, which return the quantum volume of the device
    quantum_volume = find_quantum_volume(runner)

    print("----------------------------------------")
    print(f"The quantum volume of {backend.name} is: {quantum_volume}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
