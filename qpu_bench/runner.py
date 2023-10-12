import abc
from dataclasses import dataclass, asdict

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
    def __init__(self, backend: BackendV2, compile_options: dict | None = None) -> None:
        super().__init__()
        self._backend = backend
        if compile_options is None:
            compile_options = {"optimization_level": 3}
        self._compile_options = compile_options

    def run(self, circuits: list[QuantumCircuit], shots: int = 100) -> list[ProbDistr]:
        circuits = transpile(circuits, backend=self._backend, **self._compile_options)
        counts = self._backend.run(circuits, shots=shots).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [ProbDistr.from_counts(count) for count in counts]


class SimulatorRunner(BackendRunner):
    def __init__(self) -> None:
        super().__init__(AerSimulator())


class SimulatedBackendRunner(BackendRunner):
    def __init__(self, backend: BackendV2, compiler_options: dict) -> None:
        super().__init__(AerSimulator.from_backend(backend), compiler_options)


class IBMRunner(BackendRunner):
    def __init__(self, provider: IBMProvider, backend_name: str) -> None:
        super().__init__(provider.get_backend(backend_name))
