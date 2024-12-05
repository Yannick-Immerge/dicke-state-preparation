import math
from enum import IntFlag, auto

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit


class DecompositionTargets(IntFlag):
    """
    A function that supports decompositions will apply these for all selected targets.
    """
    NONE = 0
    TOFFOLI = auto()
    CNX = auto()
    CNRY = auto()
    TRANSP = auto()
    I_OP = auto()

    ALL = TOFFOLI | CNX | CNRY | TRANSP | I_OP


class QuantumGates(IntFlag):
    """
    Enumeration of all used Quantum Gates.
    """
    # Primitive Gates
    X = auto()
    H = auto()
    T = auto()
    T_DG = auto()
    CX = auto()
    CRY = auto()
    ALL_PRIMITIVE = X | H | T | T_DG | CX | CRY

    # Abstract Gates
    TOFFOLI = auto()
    CNX = auto()
    CNRY = auto()
    TRANSP = auto()
    CTRANSP = auto()
    CU_INC = auto()
    CU_DEC = auto()
    ALL_ABSTRACT = TOFFOLI | CNX | CNRY | TRANSP | CTRANSP | CU_INC | CU_DEC

    # General
    NONE = 0
    ALL = ALL_PRIMITIVE |  ALL_ABSTRACT

    def to_string(self) -> str:
        return {
            QuantumGates.X: "X", QuantumGates.H: "H", QuantumGates.T: "T", QuantumGates.T_DG: "T^dg",
            QuantumGates.CX: "cX", QuantumGates.CRY: "cR_Y", QuantumGates.TOFFOLI: "Toffoli", QuantumGates.CNX: "c^nX",
            QuantumGates.CNRY: "c^nR_Y", QuantumGates.TRANSP: "Tau", QuantumGates.CTRANSP: "cTau",
            QuantumGates.CU_INC: "cInc", QuantumGates.CU_DEC: "cDec"
        }[self.value]


class CircuitSpec:
    """
    Object passed through during circuit construction, keeping track of layer (circuit-depth) and gate count.
    """

    _gate_counts: dict[QuantumGates, int]

    def __init__(self):
        self._gate_counts = {}

    def notify_gates(self, gate_type: QuantumGates, n: int):
        """
        Notifies about a number of constructed gates of a certain type.
        :param gate_type: The gate type of the constructed gates (single flag required).
        :param n: The number of gates that have been created.
        """
        if len(gate_type) != 1:
            raise ValueError("Each call can only notify about a single gate type.")
        if gate_type not in self._gate_counts:
            self._gate_counts[gate_type] = 0
        self._gate_counts[gate_type] += n

    def get_gate_count(self, gate_types: QuantumGates = QuantumGates.ALL) -> int:
        """
        Gets the accumulated gate count for all given gate types.
        :param gate_types: A flag collection of gates to count.
        :return: The total number of gates of the given types.
        """
        acc = 0
        for gate_type in gate_types:
            acc += self._gate_counts.get(gate_type, 0)
        return acc

    def used_gate_types(self) -> QuantumGates:
        """
        Returns the collection of gate types with one or more occurrence.
        :return: A flag collection of all gate types.
        """
        used = QuantumGates.NONE
        for gate_type, n in self._gate_counts.items():
            if n > 0:
                used |= gate_type
        return used


def produce_gate_count_table(specs: list[tuple[int, int, CircuitSpec, QuantumCircuit]]) -> pd.DataFrame:
    """
    Produces a neat data frame containing the Gate counts for a number of Dicke preparation circuits.
    :param specs: A collection of (n, k, spec, circuit) where spec is the CircuitSpec object for the construction of the Dicke preparation circuit.
    :return: The data frame.
    """
    data = {}
    circuit_column = []
    depth_column = []
    qubit_count_column = []
    gate_types = QuantumGates.NONE
    for n, k, spec, circuit in specs:
        circuit_column.append(f"D^{n}_{k}")
        depth_column.append(circuit.depth())
        qubit_count_column.append(circuit.num_qubits)
        gate_types |= spec.used_gate_types()
    data["Circuit"] = circuit_column
    data["Layer Depth"] = depth_column
    data["Qubit Count"] = qubit_count_column

    for gate_type in gate_types:
        gate_type_column = [spec.get_gate_count(gate_type) for _, _, spec, _ in specs]
        data[gate_type.to_string()] = gate_type_column

    return pd.DataFrame(data)


def calculate_t(chi: int) -> int:
    """
    Calculates the minimum qubits required to emulate a qudit with chi levels.
    :param chi: The levels of the qudit.
    :return: The minimum number of qubits required.
    """
    return max(1, math.ceil(np.log2(chi)))


def binary_string(i: int, t: int) -> str:
    """
    Produces a strings of 0s and 1s of length t that is the binary representation of i.
    :param i: The number to represent in binary.
    :param t: The number of binary digits.
    :return: The binary representation as a string of 0s and 1s starting with the MSB.
    """
    return "{:0{t}b}".format(i, t=t)


def hamming_weight(binary: str) -> int:
    """
    Calculates the hamming weight of a binary string, i.e. the number of 1s.
    :param binary: The binary representation as a string of 0s and 1s starting with the MSB.
    :return: The number of 1s (hamming weight).
    """
    acc = 0
    for c in binary:
        acc += int(c)
    return acc


def basis_state(binary: str) -> np.ndarray:
    """
    Constructs a statevector for a basis state specified by the given binary string.
    :param binary: The binary representation of the basis state as a string of 0s and 1s starting with the MSB. The length of this string (including leading 0s) is used to determine the number of qubits.
    :return:
    """
    index = int(binary, 2)
    state = np.zeros((2 ** len(binary),), dtype=complex)
    state[index] = 1
    return state

def mismatching_qubits(binary_a: str, binary_b: str) -> set[int]:
    """
    Interprets the given binary strings as basis vectors and determines all qubit indices (lowest index <-> last binary digit <-> LSB of the basis state) where they mismatch.
    :param binary_a: The binary representation of the first basis state as a string of 0s and 1s starting with the MSB.
    :param binary_b: The binary representation of the second basis state as a string of 0s and 1s starting with the MSB.
    :return: A set of all mismatching qubit indices (= string indices of the reversed strings)
    """
    n = len(binary_a)
    if n != len(binary_b):
        raise ValueError("Both binary strings have to have the same length.")
    mismatching = set()

    # Note that the MSB is the first character in the string, but the last qubit
    for i in range(n):
        if binary_a[i] != binary_b[i]:
            mismatching.add(n - 1 - i)
    return mismatching
