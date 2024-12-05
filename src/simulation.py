from enum import Enum, auto

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator


class SimulationMode(Enum):
    """
    Enumerates supported simulation modes.
    """
    SHOTS = auto()
    STATEVECTOR = auto()
    UNITARY = auto()


def simulate_circuit(circuit: QuantumCircuit, mode: SimulationMode, shots: int | None = None):
    """
    Utility function to simplify simulating a Quantum Circuit using the aer-backend.
    :param circuit: The circuit to simulate. Must be correctly configured with an initial state and instructions to save the results.
    :param mode: The simulation mode.
    :param shots: The number of shots if the simulation mode is SHOTS. Default value is 1024 and it should not be specified for other modes.
    :return: The simulation result, depending on the mode. For SHOTS: a dictionary of basis states and measurement counts, for STATEVECTOR: a qiskit.Statevector object, for UNITARY: a np.ndarray.
    """
    if mode == SimulationMode.SHOTS:
        if shots is None:
            shots = 1024

        # Transpile for simulator
        simulator = AerSimulator()
        circuit = transpile(circuit, simulator)

        # Run and get counts
        result = simulator.run(circuit, shots=shots).result()
        counts = result.get_counts(0)
        return [(label, count) for label, count in counts.items()]

    elif mode == SimulationMode.STATEVECTOR:
        if shots is not None:
            raise ValueError("Shots parameter is only supported for the SHOTS simulation method.")

        # Transpile for simulator
        simulator = AerSimulator(method="statevector")
        circuit = transpile(circuit, simulator)

        # Run and get statevector
        result = simulator.run(circuit).result()
        return result.get_statevector(circuit)

    elif mode == SimulationMode.UNITARY:
        if shots is not None:
            raise ValueError("Shots parameter is only supported for the SHOTS simulation method.")

        # Transpile for simulator
        simulator = AerSimulator(method="unitary")
        circuit = transpile(circuit, simulator)

        # Run and get statevector
        result = simulator.run(circuit).result()
        return np.array(result.get_unitary(circuit), dtype=complex)

    else:
        raise ValueError("Unrecognized simulation mode.")


def simulate_gate(gate: Gate, mode: SimulationMode, initial: np.ndarray | str | None = None, shots: int | None = None):
    """
    Utility function to simplify simulating a Quantum Gate using the aer-backend.
    :param gate: The gate to simulate.
    :param mode: The simulation mode.
    :param initial: The initial statevector as a np.ndarray of basis state label (bottom to top). |0.....0> is used by default.
    :param shots: The number of shots if the simulation mode is SHOTS. Default value is 1024 and it should not be specified for other modes.
    :return: The simulation result, depending on the mode. For SHOTS: a dictionary of basis states and measurement counts, for STATEVECTOR: a qiskit.Statevector object, for UNITARY: a np.ndarray.
    """
    if mode == SimulationMode.SHOTS:
        if shots is None:
            shots = 1024

        circuit = QuantumCircuit(gate.num_qubits)
        if initial is not None:
            circuit.initialize(initial)
        circuit.append(gate, range(gate.num_qubits))
        circuit.measure_all()

        # Transpile for simulator
        simulator = AerSimulator()
        circuit = transpile(circuit, simulator)

        # Run and get counts
        result = simulator.run(circuit, shots=shots).result()
        counts = result.get_counts(0)
        return [(label, count) for label, count in counts.items()]

    elif mode == SimulationMode.STATEVECTOR:
        if shots is not None:
            raise ValueError("Shots parameter is only supported for the SHOTS simulation method.")

        circuit = QuantumCircuit(gate.num_qubits)
        if initial is not None:
            circuit.initialize(initial)
        circuit.append(gate, range(gate.num_qubits))
        circuit.save_statevector()

        # Transpile for simulator
        simulator = AerSimulator(method="statevector")
        circuit = transpile(circuit, simulator)

        # Run and get statevector
        result = simulator.run(circuit).result()
        return result.get_statevector(circuit)

    elif mode == SimulationMode.UNITARY:
        if shots is not None:
            raise ValueError("Shots parameter is only supported for the SHOTS simulation method.")
        if initial is not None:
            raise ValueError("Initial statevectors are not supported when calculating the entire gate unitary matrix.")

        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, range(gate.num_qubits))
        circuit.save_unitary()

        # Transpile for simulator
        simulator = AerSimulator(method="unitary")
        circuit = transpile(circuit, simulator)

        # Run and get statevector
        result = simulator.run(circuit).result()
        return np.array(result.get_unitary(circuit), dtype=complex)

    else:
        raise ValueError("Unrecognized simulation mode.")