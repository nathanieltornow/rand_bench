# Benchmaring Quantum Devices

## Quantum Volume

Quantum Volume is a single-number metric to benchmark the performance of QPUs. It is defined as the largest random circuit of equal width and depth that the QPU successfully implements.


### Running the Quantum Volume Benchmark

The Quantum Volume benchmark is implemented in the [quantum_volume.py](qpu_bench/quantum_volume.py) file. It can be run as follows:

```python
from qiskit.providers.fake_provider import FakeMontreal

from qpu_bench.quantum_volume import find_quantum_volume
from qpu_bench.runner import SimulatedBackendRunner

# create a noisy simulator
backend = FakeMontreal()

# create a runner of a noisy simulator
runner = SimulatedBackendRunner(backend)

# run the quantum volume algorithm, which return the quantum volume of the device
quantum_volume = find_quantum_volume(runner)
print(quantum_volume)
```

See [examples/qv.py](examples/qv.py) for a complete example.


