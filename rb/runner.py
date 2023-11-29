import abc
from dataclasses import dataclass, asdict

from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

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


class NoiseModelRunner(Runner):
    def __init__(
        self,
        noise_model: NoiseModel,
        coupling_map: CouplingMap,
        basis_gates: list[str],
        optimization_level: int = 3,
    ) -> None:
        self._noise_model = noise_model
        self._coupling_map = coupling_map
        self._basis_gates = basis_gates
        self._optimization_level = optimization_level
        self._sim = AerSimulator(
            noise_model=noise_model,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
        )
        super().__init__()

    def run(self, circuits: list[QuantumCircuit], shots: int = 100) -> list[ProbDistr]:
        circuits = transpile(
            circuits,
            basis_gates=self._basis_gates,
            optimization_level=self._optimization_level,
            coupling_map=self._coupling_map,
        )
        counts = self._sim.run(circuits, shots=shots).result().get_counts()
        counts = [counts] if isinstance(counts, dict) else counts
        return [ProbDistr.from_counts(count) for count in counts]


class IBMNoiseSimulator(NoiseModelRunner):
    def __init__(self, optimization_level: int = 3) -> None:
        gates_1q = ["x", "sx", "rz"]
        gates_2q = ["cx"]
        basis_gates = gates_1q + gates_2q
        coupling_map = CouplingMap.from_heavy_hex(3)
        noise_model = create_noise_model(
            gates_1q=gates_1q,
            gates_2q=gates_2q,
            error_1q=0.0001,
            error_2q=0.001,
            t1=100,
            t2=70,
            time_1q=0.035,
        )
        super().__init__(
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )


class IonQNoiseSimulator(NoiseModelRunner):
    def __init__(self, optimization_level: int = 3) -> None:
        gates_1q = ["x", "sx", "rz"]
        gates_2q = ["cx"]
        basis_gates = gates_1q + gates_2q
        coupling_map = CouplingMap.from_full(21)
        noise_model = create_noise_model(
            gates_1q=gates_1q,
            gates_2q=gates_2q,
            error_1q=0.0001,
            error_2q=0.001,
            t1=100,
            t2=70,
            time_1q=0.035,
        )
        super().__init__(
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )


class SimulatedBackendRunner(BackendRunner):
    def __init__(self, backend: BackendV2, compiler_options: dict) -> None:
        super().__init__(AerSimulator.from_backend(backend), compiler_options)


class IBMRunner(BackendRunner):
    def __init__(self, provider: IBMProvider, backend_name: str) -> None:
        super().__init__(provider.get_backend(backend_name))


def create_noise_model(
    gates_1q: list[str],
    gates_2q: list[str],
    error_1q: float,
    error_2q: float,
    t1: float,
    t2: float,
    time_1q: float,
) -> NoiseModel:
    nm = NoiseModel(gates_1q + gates_2q)
    nm.add_all_qubit_quantum_error(
        depolarizing_error(error_1q, 1), gates_1q, warnings=False
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(error_2q, 2), gates_2q, warnings=False
    )
    nm.add_all_qubit_quantum_error(
        thermal_relaxation_error(t1, t2, time_1q), gates_1q, warnings=False
    )
    return nm
