import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.cnry import append_cnry, append_transp_operator
from src.utility import calculate_t, DecompositionTargets, CircuitSpec, QuantumGates, produce_gate_count_table


def _gamma_j_r_0(j, r, n, k) -> float:
    """
    Calculates gamma_{r, 0}^(j)
    :param j:
    :param r:
    :param n:
    :param k:
    :return:
    """
    return 0 if k - r > n - j + 1 else np.sqrt(1 + (r - k) / (n-j+1))


def create_inc_unitary(chi, invert: bool = False) -> np.ndarray:
    t = calculate_t(chi)
    unitary = np.zeros((2 ** t, 2 ** t), dtype=complex)
    for i in range(2 ** t):
        if i < chi:
            unitary[(i + 1) % chi, i] = 1
        else:
            unitary[i, i] = 1
    if invert:
        unitary = unitary.T
    return unitary


def create_inc_gate(chi, invert: bool = False) -> Gate:
    """
    Produces an increment gate modulus chi for the minimum number of qubits required to represent a chi-level qudit.
    :return:
    """
    t = calculate_t(chi)
    gate_circuit = QuantumCircuit(t)
    gate_circuit.unitary(create_inc_unitary(chi, invert), range(t))
    op_name = "\\ominus" if invert else "\\oplus"
    return gate_circuit.to_gate(label=f"$U_{op_name}^\\ast$")


def _append_i_operator_abstract(circuit: QuantumCircuit, l, j, sites: list[int], qudit: list[int], chi, clean_aux_cry: int, clean_aux_cx: int,
                                decomposition: DecompositionTargets, spec: CircuitSpec | None):
    n = len(sites)
    k = chi - 1
    l_plus_1 = (l + 1) % chi
    theta = 2 * np.arccos(_gamma_j_r_0(j, l, n, k))

    inc_gate = create_inc_gate(chi)
    dec_gate = create_inc_gate(chi, invert=True)

    circuit.append(inc_gate.control(1, ctrl_state="0"), [j - 1] + qudit)
    append_cnry(circuit, theta, qudit, j - 1, clean_aux_cry, clean_aux_cx, decomposition, spec, control_state=l_plus_1)
    circuit.append(dec_gate.control(1, ctrl_state="0"), [j - 1] + qudit)

    if spec is not None:
        spec.notify_gates(QuantumGates.CU_INC, 1)
        spec.notify_gates(QuantumGates.CU_DEC, 1)


def _append_i_operator_primitive(circuit: QuantumCircuit, l: int, j: int, sites: list[int], qudit: list[int], chi: int,
                                 clean_aux_cry: int, clean_aux_cx: int, decomposition: DecompositionTargets,
                                 spec: CircuitSpec | None = None):
    n = len(sites)
    k = chi - 1
    l_plus_1 = (l + 1) % chi
    theta = 2 * np.arccos(_gamma_j_r_0(j, l, n, k))

    circuit.x(j - 1)
    append_transp_operator(circuit, l, l_plus_1, qudit, decomposition, spec, control=j - 1)
    circuit.x(j - 1)

    append_cnry(circuit, theta, qudit, j - 1, clean_aux_cry, clean_aux_cx, decomposition, spec, control_state=l_plus_1)

    circuit.x(j - 1)
    append_transp_operator(circuit, l, l_plus_1, qudit, decomposition, spec, control=j - 1)
    circuit.x(j - 1)

    if spec is not None:
        spec.notify_gates(QuantumGates.X, 4)


def append_i_operator(circuit: QuantumCircuit, l: int, j: int, sites: list[int], qudit: list[int], chi: int,
                      clean_aux_cry: int, clean_aux_cx: int,
                      decomposition: DecompositionTargets = DecompositionTargets.NONE,
                      spec: CircuitSpec | None = None):
    """
    Produces I_l^(j).
    :return:
    """
    if DecompositionTargets.I_OP in decomposition:
        _append_i_operator_primitive(circuit, l, j, sites, qudit, chi, clean_aux_cry, clean_aux_cx, decomposition, spec)
    else:
        _append_i_operator_abstract(circuit, l, j, sites, qudit, chi, clean_aux_cry, clean_aux_cx, decomposition, spec)


def append_u_operator(circuit: QuantumCircuit, j: int, sites: list[int], qudit: list[int], chi: int, clean_aux_cry: int,
                      clean_aux_cx: int, decomposition: DecompositionTargets = DecompositionTargets.NONE,
                      spec: CircuitSpec | None = None):
    n = len(sites)
    k = chi - 1
    for l in range(max(0, j - n + k - 1), min(j - 1, k - 1) + 1):
        append_i_operator(circuit, l, j, sites, qudit, chi, clean_aux_cry, clean_aux_cx, decomposition, spec)


def append_dicke_circuit(circuit: QuantumCircuit, sites: list[int], qudit: list[int], chi: int, clean_aux_cry: int,
                         clean_aux_cx: int, decomposition: DecompositionTargets = DecompositionTargets.NONE,
                         spec: CircuitSpec | None = None):
    n = len(sites)
    for j in range(1, n + 1):
        append_u_operator(circuit, j, sites, qudit, chi, clean_aux_cry, clean_aux_cx, decomposition, spec)


def create_dicke_circuit(n: int, k: int,
                         decomposition: DecompositionTargets = DecompositionTargets.NONE,
                         spec: CircuitSpec | None = None) -> QuantumCircuit:
    """
    Produces a composite qiskit gate that prepares the
    """
    # Depending on the decomposition level we might need auxiliary qubits.
    additional = 0
    if DecompositionTargets.CNRY in decomposition:
        additional += 1
    if DecompositionTargets.CNX in decomposition:
        additional += 1

    chi = k + 1
    t = calculate_t(chi)
    sites = list(range(n))
    qudit = list(range(n, n + t))
    clean_aux_cry = n + t
    clean_aux_cx = n + t + 1

    circuit = QuantumCircuit(n + t + additional)

    append_dicke_circuit(circuit, sites, qudit, chi, clean_aux_cry, clean_aux_cx, decomposition, spec)
    return circuit


def print_dicke_circuit(n: int, k: int,
                         decomposition: DecompositionTargets = DecompositionTargets.NONE):
    circuit = create_dicke_circuit(n, k, decomposition)
    print(circuit)


def jupyter_produce_counts(decomposition: DecompositionTargets) -> pd.DataFrame:
    spec_2_2 = CircuitSpec()
    spec_3_2 = CircuitSpec()
    spec_3_3 = CircuitSpec()
    spec_5_4 = CircuitSpec()

    create_dicke_circuit(2, 2, decomposition, spec_2_2)
    create_dicke_circuit(3, 2, decomposition, spec_3_2)
    create_dicke_circuit(3, 3, decomposition, spec_3_3)
    create_dicke_circuit(5, 4, decomposition, spec_5_4)

    return produce_gate_count_table([
        (2, 2, spec_2_2),
        (3, 2, spec_3_2),
        (3, 3, spec_3_3),
        (5, 4, spec_5_4)
    ])


if __name__ == "__main__":
    jupyter_produce_counts(DecompositionTargets.ALL)
    print_dicke_circuit(5, 4, DecompositionTargets.ALL)