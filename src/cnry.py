"""
Contains decomposable constructions for C^nR_Y(theta) gates and all required sub-gates.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, RYGate

from src.utility import DecompositionTargets, binary_string, mismatching_qubits, CircuitSpec, QuantumGates


def _append_toffoli_abstract(circuit: QuantumCircuit, c1, c2, target, spec: CircuitSpec | None):
    circuit.ccx(c1, c2, target)

    if spec is not None:
        spec.notify_gates(QuantumGates.TOFFOLI, 1)


def _append_toffoli_primitive(circuit: QuantumCircuit, c1, c2, target, spec: CircuitSpec | None):
    circuit.h(target)
    circuit.cx(c2, target)
    circuit.tdg(target)
    circuit.cx(c1, target)
    circuit.t(target)
    circuit.cx(c2, target)
    circuit.tdg(target)
    circuit.cx(c1, target)
    circuit.t(c2)
    circuit.t(target)
    circuit.h(target)
    circuit.cx(c1, c2)
    circuit.t(c1)
    circuit.tdg(c2)
    circuit.cx(c1, c2)

    if spec is not None:
        spec.notify_gates(QuantumGates.H, 2)
        spec.notify_gates(QuantumGates.T, 4)
        spec.notify_gates(QuantumGates.T_DG, 3)
        spec.notify_gates(QuantumGates.CX, 6)


def append_toffoli(circuit: QuantumCircuit, c1, c2, target,
                   decomposition : DecompositionTargets = DecompositionTargets.NONE, spec: CircuitSpec | None = None):
    """
    Append a Toffoli gate with a controllable decomposition level.

    :param circuit: The quantum circuit to append the Toffoli gate.
    :param c1: The first control qubit.
    :param c2: The second control qubit.
    :param target: The target qubit.
    :param decomposition: An optional collection of targets that will be decomposed if they appear throughout the construction.
    :param spec: An optional introspection object that counts gates during the construction.
    :return: Nothing, appends to circuit in-place.
    """
    if DecompositionTargets.TOFFOLI in decomposition:
        _append_toffoli_primitive(circuit, c1, c2, target, spec)
    else:
        _append_toffoli_abstract(circuit, c1, c2, target, spec)


def _append_transp_operator_abstract(circuit: QuantumCircuit, l: int, l_prime: int, targets: list[int], spec: CircuitSpec | None,
                                     control: int | None):
    n = len(targets)
    if n == 0:
        raise ValueError("Expected at least one target qubit.")

    gate_unitary = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(2 ** n):
        if i == l:
            target = l_prime
        elif i == l_prime:
            target = l
        else:
            target = i
        gate_unitary[target, i] = 1

    gate_circuit = QuantumCircuit(n)
    gate_circuit.unitary(gate_unitary, range(n))
    gate = gate_circuit.to_gate(label="$\\mathcal{T}_{" + str(l) + ", " + str(l_prime) + "}$")

    if control is None:
        circuit.append(gate, targets)
    else:
        circuit.append(gate.control(1), [control] + targets)

    if spec is not None:
        spec.notify_gates(QuantumGates.TRANSP if control is None else QuantumGates.CTRANSP, 1)


def _append_transp_operator_primitive(circuit: QuantumCircuit, l: int, l_prime: int, targets: list[int], spec: CircuitSpec | None,
                                     control: int | None):
    n = len(targets)
    if n == 0:
        raise ValueError("Expected at least one target qubit.")

    a = binary_string(l, n)
    b = binary_string(l_prime, n)
    x_targets = mismatching_qubits(a, b)
    for x_target in x_targets:
        if control is None:
            circuit.x(targets[x_target])
        else:
            circuit.cx(control, targets[x_target])

    if spec is not None:
        spec.notify_gates(QuantumGates.X if control is None else QuantumGates.CX, len(x_targets))


def append_transp_operator(circuit: QuantumCircuit, l: int, l_prime: int, targets: list[int],
                           decomposition: DecompositionTargets = DecompositionTargets.NONE, spec: CircuitSpec | None = None, control: int | None = None):
    """
    Appends a transposition operator, i.e. the simplest X-gate construction that achieves |l> <-> <l'| for the basis states of the targets register.
    Note that the first qubit in targets represents the LSB of the basis state.
    The decomposition level can be controlled and a single control qubit can optionally be specified.

    :param circuit: The quantum circuit to append the transp-operator gate.
    :param l: The basis state to swap with l'.
    :param l_prime: The basis state to swap with l.
    :param targets: The register of target qubits.
    :param decomposition: An optional collection of targets that will be decomposed if they appear throughout the construction.
    :param spec: An optional introspection object that counts gates during the construction.
    :param control: An optional control qubit that controls the transp-operator.
    :return: Nothing, appends to circuit in-place.
    """
    if DecompositionTargets.TRANSP in decomposition:
        _append_transp_operator_primitive(circuit, l, l_prime, targets, spec, control)
    else:
        _append_transp_operator_abstract(circuit, l, l_prime, targets, spec, control)


def _append_cnx_abstract(circuit: QuantumCircuit, controls: list[int], target: int, spec: CircuitSpec | None, control_state: int | None):
    x = XGate().control(len(controls), ctrl_state=control_state)
    circuit.append(x, controls + [target])

    if spec is not None:
        spec.notify_gates(QuantumGates.CNX, 1)


def _append_cnx_primitive_borrowed(circuit: QuantumCircuit, controls: list[int], borrow: list[int], target: int,
                                   decomposition: DecompositionTargets, spec: CircuitSpec | None):
    n = len(controls)
    if n == 0:
        raise ValueError("Expect at least a single control qubit.")
    elif n == 1:
        circuit.cx(controls[0], target)
    elif n == 2:
        append_toffoli(circuit, controls[0], controls[1], target, decomposition, spec)
    else:
        if len(borrow) < n - 2:
            raise ValueError("This decomposition requires at least n - 2 borrowed ancilla qubits.")

        # Arrange
        arranged = [controls[0], controls[1]]
        for i in range(n - 2):
            arranged.append(borrow[i])
            arranged.append(controls[i + 2])
        arranged.append(target)

        for _ in range(2):
            for i in range(2 * n - 2, 0, -2):
                append_toffoli(circuit, arranged[i - 2], arranged[i - 1], arranged[i], decomposition, spec)
            for i in range(4, 2 * (n - 1) - 1, 2):
                append_toffoli(circuit, arranged[i - 2], arranged[i - 1], arranged[i], decomposition, spec)

    if spec is not None and n == 1:
        spec.notify_gates(QuantumGates.X, 1)


def _append_cnx_primitive_clean(circuit: QuantumCircuit, controls: list[int], target: int, clean_aux: int,
                               decomposition: DecompositionTargets, spec: CircuitSpec | None, control_state: int | None):
    n = len(controls)
    if n == 0:
        raise ValueError("Expect at least a single control qubit.")

    if control_state is not None:
        append_transp_operator(circuit, control_state, 2 ** n - 1, controls, decomposition, spec)

    if n == 1:
        circuit.cx(controls[0], target)
    elif n == 2:
        append_toffoli(circuit, controls[0], controls[1], target, decomposition, spec)
    else:
        n_1 = int(n / 2)
        n_0 = n - n_1
        x_register = [controls[i] for i in range(n_0)]
        y_register = [controls[n_0 + i] for i in range(n_1)]
        _append_cnx_primitive_borrowed(circuit, x_register, y_register, clean_aux, decomposition, spec)
        _append_cnx_primitive_borrowed(circuit, y_register + [clean_aux], x_register, target, decomposition, spec)
        _append_cnx_primitive_borrowed(circuit, x_register, y_register, clean_aux, decomposition, spec)

    if control_state is not None:
        append_transp_operator(circuit, control_state, 2 ** n - 1, controls, decomposition, spec)

    if spec is not None and n == 1:
        spec.notify_gates(QuantumGates.X, 1)


def append_cnx(circuit: QuantumCircuit, controls: list[int], target: int, clean_aux: int,
               decomposition: DecompositionTargets = DecompositionTargets.NONE, spec: CircuitSpec | None = None,
               control_state: int | None = None):
    """
    Appends a c^nX gate with a controllable decomposition level (potentially requiring a clean ancilla).
    The control state (basis state in the control register that activates the X-gate) can also be specified.

    :param circuit: The quantum circuit to append the c^nX gate.
    :param controls: The control register.
    :param target: The target qubit.
    :param clean_aux: The clean ancilla qubit that might be used if the gate should be decomposed.
    :param decomposition: An optional collection of targets that will be decomposed if they appear throughout the construction.
    :param spec: An optional introspection object that counts gates during the construction.
    :param control_state: If provided, the basis state |control_state> is used to determine when to activate the X gate. By default, this is 2^t - 1, where t = len(controls).
    :return: Nothing, appends to circuit in-place.
    """
    if DecompositionTargets.CNX in decomposition:
        _append_cnx_primitive_clean(circuit, controls, target, clean_aux, decomposition, spec, control_state)
    else:
        _append_cnx_abstract(circuit, controls, target, spec, control_state)

def _append_cnry_abstract(circuit: QuantumCircuit, theta: float, controls: list[int], target: int, spec: CircuitSpec | None, control_state: int | None):
    n = len(controls)
    if n == 0:
        raise ValueError("Expected at least one control qubit.")
    ry = RYGate(theta).control(n, ctrl_state=control_state)
    circuit.append(ry, controls + [target])

    if spec is not None:
        spec.notify_gates(QuantumGates.CNRY, 1)


def _append_cnry_primitive(circuit: QuantumCircuit, theta: float, controls: list[int], target: int, clean_aux_cry: int,
                          clean_aux_cx: int, decomposition: DecompositionTargets, spec: CircuitSpec | None, control_state: int | None):
    n = len(controls)
    if n == 0:
        raise ValueError("Expected at least one qubit.")

    append_cnx(circuit, controls, clean_aux_cry, clean_aux_cx, decomposition, spec, control_state)
    circuit.cry(theta, clean_aux_cry, target)
    append_cnx(circuit, controls, clean_aux_cry, clean_aux_cx, decomposition, spec, control_state)

    if spec is not None:
        spec.notify_gates(QuantumGates.CRY, 1)


def append_cnry(circuit: QuantumCircuit, theta: float, controls: list[int], target: int, clean_aux_cry: int,
                clean_aux_cx: int, decomposition: DecompositionTargets = DecompositionTargets.NONE, spec: CircuitSpec | None = None, control_state: int | None = None):
    """
    Appends a c^nR_Y(theta) gate with a controllable decomposition level (potentially requiring a clean ancilla for the cR_Y(theta) and c^nX operations).
    The control state (basis state in the control register that activates the R_y(theta)-gate) can also be specified.

    :param circuit: The quantum circuit to append the c^nR_Y(theta) gate.
    :param theta: Rotation angle for the y-rotation of the target qubit.
    :param controls: The control register.
    :param target: The target qubit.
    :param clean_aux_cry: The clean ancilla qubit that might be used for the cR_Y(theta)-operation if the gate should be decomposed.
    :param clean_aux_cx: The clean ancilla qubit that might be used for the c^nX-operation if the gate should be decomposed.
    :param decomposition: An optional collection of targets that will be decomposed if they appear throughout the construction.
    :param spec: An optional introspection object that counts gates during the construction.
    :param control_state: If provided, the basis state |control_state> is used to determine when to activate the R_Y(theta) gate. By default, this is 2^t - 1, where t = len(controls).
    :return: Nothing, appends to circuit in-place.
    """
    if DecompositionTargets.CNRY in decomposition:
        _append_cnry_primitive(circuit, theta, controls, target, clean_aux_cry, clean_aux_cx, decomposition, spec, control_state)
    else:
        _append_cnry_abstract(circuit, theta, controls, target, spec, control_state)
