import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from src.cnry import append_cnry
from src.simulation import SimulationMode, simulate_circuit
from src.utility import DecompositionTargets


def _test_cnry():
    n = 3
    control_state = 3
    theta = np.pi

    # We propagate all relevant basis states (auxiliary |00>)
    for i in range(2 ** (n + 1)):
        bv = np.zeros((2 ** (n + 3),), dtype=complex)
        bv[i] = 1

        prim_circuit = QuantumCircuit(n + 3)  # n control, 1 target, 1 reset aux, 1 clean aux
        prim_circuit.initialize(bv)
        abstract_circuit = QuantumCircuit(n + 3)  # n control, 1 target, 1 reset aux, 1 clean aux
        abstract_circuit.initialize(bv)

        append_cnry(prim_circuit, theta, list(range(n)), n, n + 1, n + 2, DecompositionTargets.ALL, control_state)
        append_cnry(abstract_circuit, theta, list(range(n)), n, n + 1, n + 2, DecompositionTargets.NONE, control_state)

        prim_circuit.save_statevector()
        abstract_circuit.save_statevector()

        prim_sv = np.array(simulate_circuit(prim_circuit, SimulationMode.STATEVECTOR))
        abstract_sv = np.array(simulate_circuit(abstract_circuit, SimulationMode.STATEVECTOR))
        assert np.allclose(prim_sv, abstract_sv)


if __name__ == "__main__":
    _test_cnry()
