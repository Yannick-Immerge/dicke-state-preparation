import sys
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import pytest
from qiskit import QuantumCircuit

# Fix Python Path
_ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(_ROOT_DIR))

from src.dicke import append_dicke_circuit
from src.simulation import SimulationMode, simulate_circuit
from src.utility import calculate_t, basis_state, binary_string, hamming_weight, DecompositionTargets


def _generate_n_k_tuples(n_values: Iterable[int]) -> Generator[tuple[int, int], None, None]:
    for n in n_values:
        for k in range(n + 1):
            yield n, k



def _test_dicke_states_with_decomposition_level(n, k, decomposition_targets: DecompositionTargets):
    chi = k + 1
    t = calculate_t(chi)
    qudit_register = basis_state(binary_string(k, t))

    dicke_register = np.zeros((2 ** n,), dtype=complex)
    total = 0
    for i in range(2 ** n):
        bin_i = binary_string(i, n)
        if hamming_weight(bin_i) == k:
            dicke_register += basis_state(bin_i)
            total += 1
    dicke_register *= 1 / np.sqrt(total)

    aux_register = np.zeros((4,), dtype=complex)
    aux_register[0] = 1
    expected_statevector = np.kron(aux_register, np.kron(qudit_register, dicke_register))

    chi = k + 1
    t = calculate_t(chi)
    dicke_circuit = QuantumCircuit(n + t + 2)
    append_dicke_circuit(dicke_circuit, list(range(n)), list(range(n, n + t)), chi, n + t, n + t + 1, (
        decomposition_targets
    ))
    dicke_circuit.save_statevector()

    actual_statevector = np.array(simulate_circuit(dicke_circuit, SimulationMode.STATEVECTOR))

    assert np.allclose(actual_statevector, expected_statevector)


@pytest.mark.parametrize("n, k", list(_generate_n_k_tuples(range(1, 6))))
def test_dicke_states_abstract(n, k):
    _test_dicke_states_with_decomposition_level(n, k, DecompositionTargets.NONE)


@pytest.mark.parametrize("n, k", list(_generate_n_k_tuples(range(1, 6))))
def test_dicke_states_primitive(n, k):
    _test_dicke_states_with_decomposition_level(n, k, DecompositionTargets.ALL)


def jupyter_test_status() -> str:
    try:
        for n, k in _generate_n_k_tuples(range(1, 5)):
            test_dicke_states_abstract(n, k)
            test_dicke_states_primitive(n, k)
        return "SUCCESS: The Dicke States are produced correctly!"
    except Exception:
        return "FAIL: Some tests fail!"
