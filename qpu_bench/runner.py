import abc

from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from .prob_distr import ProbDistr


class Runner(abc.ABC):
    @abc.abstractmethod
    def run(self, circuits: list[QuantumCircuit], shots: int = 100) -> list[ProbDistr]:
        ...


class BackendRunner(Runner):
    def __init__(self, backend: BackendV2, opt_level: int = 3) -> None:
        super().__init__()
        self._backend = backend
        self._opt_level = opt_level

    def run(self, circuits: list[QuantumCircuit], shots: int = 100) -> list[ProbDistr]:
        circuits = transpile(
            circuits, backend=self._backend, optimization_level=self._opt_level
        )
        counts = self._backend.run(circuits, shots=shots).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [ProbDistr.from_counts(count) for count in counts]


class SimulatorRunner(BackendRunner):
    def __init__(self) -> None:
        super().__init__(AerSimulator())


class SimulatedBackendRunner(BackendRunner):
    def __init__(self, backend: BackendV2) -> None:
        super().__init__(AerSimulator.from_backend(backend))


class IBMRunner(Runner):
    def __init__(self, provider: IBMProvider, backend_name: str) -> None:
        super().__init__(provider.get_backend(backend_name))
