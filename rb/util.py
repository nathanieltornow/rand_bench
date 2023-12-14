import abc

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile


def counts_to_probs(
    all_counts: dict[str, int] | list[dict[str, int]]
) -> list[dict[str, float]]:
    all_counts = all_counts if isinstance(all_counts, list) else [all_counts]
    return [
        {state: count / sum(counts.values()) for state, count in counts.items()}
        for counts in all_counts
    ]


class CustomTranspiler(abc.ABC):
    @abc.abstractmethod
    def run(
        self, circuits: QuantumCircuit | list[QuantumCircuit]
    ) -> list[QuantumCircuit]:
        ...


class BackendTranspiler(CustomTranspiler):
    def __init__(self, backend: BackendV2, optimization_level: int = 3) -> None:
        self._backend = backend
        self._optimization_level = optimization_level

    def run(
        self, circuits: QuantumCircuit | list[QuantumCircuit]
    ) -> list[QuantumCircuit]:
        circuits = circuits if isinstance(circuits, list) else [circuits]
        return transpile(
            circuits, backend=self._backend, optimization_level=self._optimization_level
        )


class ModelTranspiler(CustomTranspiler):
    def __init__(
        self,
        coupling_map: CouplingMap,
        basis_gates: list[str],
        optimization_level: int = 3,
    ) -> None:
        self._coupling_map = coupling_map
        self._basis_gates = basis_gates
        self._optimization_level = optimization_level

    def run(
        self, circuits: QuantumCircuit | list[QuantumCircuit]
    ) -> list[QuantumCircuit]:
        circuits = circuits if isinstance(circuits, list) else [circuits]
        return transpile(
            circuits,
            coupling_map=self._coupling_map,
            basis_gates=self._basis_gates,
            optimization_level=self._optimization_level,
        )
