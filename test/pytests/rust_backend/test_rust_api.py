"""Parity and safety tests for the Rust state-vector backend.

The tests import the compiled extension from ``QUANTAR_RUST_TEST_DIR`` when
that environment variable is set (so a freshly built ``.pyd`` can be tested in
place), otherwise from the installed ``quantum.quantar_kernels`` module.

Build first with::

    cd rust
    set PYO3_PYTHON=<venv>\\Scripts\\python.exe
    cargo build
    set QUANTAR_RUST_TEST_DIR=<dir containing quantar_kernels.pyd>
    python -m unittest discover -s test/rust_backend -v
"""

from __future__ import annotations

import gc
import os
import sys
import unittest

import numpy as np

# Allow testing a freshly built module without replacing the shipped .pyd.
_test_dir = os.environ.get("QUANTAR_RUST_TEST_DIR")
if _test_dir:
    sys.path.insert(0, _test_dir)
    import quantar_kernels as rk  #type: ignore
else:
    from quantum import quantar_kernels as rk

import quantum.gate_matrix as gm
from quantum.gate_abs import Gate as PyGate
from quantum.qubit_abs import QuantumComputer as PyQC


def assert_close(test: unittest.TestCase, actual, expected, label: str, atol: float = 1e-9) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape, f"{label}: shape mismatch")
    if not np.allclose(actual, expected, atol=atol):
        test.fail(f"{label}: max diff {np.max(np.abs(actual - expected))}")


class GateTests(unittest.TestCase):
    def test_attributes_and_copies(self) -> None:
        matrix = gm.h_MAT()
        rust = rk.Gate(matrix, [], [0], name="h")
        python = PyGate(matrix, [], [0], name="h")
        for attr in ("matrix", "controls", "targets", "name", "params", "inverted"):
            ra, pa = np.asarray(getattr(rust, attr)), np.asarray(getattr(python, attr))
            self.assertEqual(ra.shape, pa.shape, attr)
            self.assertTrue(np.all(ra == pa), attr)

        inverse, python_inverse = rust.inverse(), python.inverse()
        assert_close(self, inverse.matrix, python_inverse.matrix, "inverse")
        self.assertTrue(inverse.inverted and python_inverse.inverted)

        remapped, python_remapped = rust.remap([2, 1, 0]), python.remap([2, 1, 0])
        self.assertEqual(remapped.targets, python_remapped.targets)
        assert_close(self, remapped.matrix, python_remapped.matrix, "remap")

    def test_non_unitary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rk.Gate(np.eye(2) * 2, [], [0])

    def test_mutable_attributes(self) -> None:
        gate = rk.Gate(gm.h_MAT(), [], [0], name="h")
        gate.name = "x"
        gate.inverted = True
        gate.controls = [1]
        gate.targets = [2]
        gate.params = [0.5]
        self.assertEqual((gate.name, gate.inverted, gate.controls, gate.targets, gate.params),
                         ("x", True, [1], [2], [0.5]))


class ApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.python = PyQC(3)
        self.rust = rk.QuantumComputer(3)

    def test_matrix_operand(self) -> None:
        self.rust.apply(gm.h_MAT(), [self.rust[0]])
        self.python.apply(gm.h_MAT(), [self.python[0]])
        assert_close(self, self.rust.state_vector, self.python.state_vector, "apply(matrix)")
        self.assertEqual(len(self.rust.op_hist), len(self.python.op_hist))

    def test_gate_operand(self) -> None:
        rust_gate = self.rust.gate(gm.px_MAT(), [0])
        self.rust.apply(rust_gate)
        self.python.apply(PyGate(gm.px_MAT(), [], [0]))
        assert_close(self, self.rust.state_vector, self.python.state_vector, "apply(Gate)")

    def test_python_gate_operand_on_rust_backend(self) -> None:
        self.rust.apply(PyGate(gm.px_MAT(), [], [0]))
        self.python.apply(PyGate(gm.px_MAT(), [], [0]))
        assert_close(self, self.rust.state_vector, self.python.state_vector, "apply(Python Gate)")

    def test_program_with_fusion(self) -> None:
        programs = []
        for qc in (self.rust, self.python):
            programs.append([
                qc.gate(gm.swap_MAT(), [0, 2]),
                qc.gate(gm.swap_MAT(), [1, 2]),
                qc.gate(gm.p_MAT(0.3), [0]),
                qc.gate(gm.p_MAT(0.5), [0], [2]),
            ])
        self.rust.apply(programs[0])
        self.python.apply(programs[1])
        assert_close(self, self.rust.state_vector, self.python.state_vector, "apply(program)")
        self.assertEqual(len(self.rust.op_hist), len(self.python.op_hist))

    def test_history_suppression(self) -> None:
        with self.rust.suppress_history():
            self.rust.apply(gm.h_MAT(), [0])
        with self.python.suppress_history():
            self.python.apply(gm.h_MAT(), [0])
        self.assertEqual(len(self.rust.op_hist), len(self.python.op_hist), 0)
        assert_close(self, self.rust.state_vector, self.python.state_vector, "suppressed state")

    def test_uncompute(self) -> None:
        for backend, qubits in ((self.rust, self.rust.qubits), (self.python, self.python.qubits)):
            backend.apply(gm.h_MAT(), [qubits[0]])  #type: ignore
            backend.apply(gm.px_MAT(), [qubits[1]])  #type: ignore
            backend.apply(gm.h_MAT(), [qubits[0]])  #type: ignore
        cp_r, cp_p = self.rust.checkpoint(), self.python.checkpoint()
        self.assertEqual(cp_r, cp_p)
        self.rust.uncompute([self.rust[0]], cp_r - 2)
        self.python.uncompute([self.python[0]], cp_p - 2)
        assert_close(self, self.rust.state_vector, self.python.state_vector, "uncompute")


class AlgorithmTests(unittest.TestCase):
    def test_qft_roundtrip(self) -> None:
        for n in (2, 3, 4):
            python, rust = PyQC(n), rk.QuantumComputer(n)
            for i in range(n):
                matrix = gm.h_MAT() if i % 2 == 0 else gm.ry_MAT(0.4 + 0.1 * i)
                python.apply(matrix, [i])
                rust.apply(matrix, [i])
            python.qft(python.qubits) #type: ignore
            rust.qft(rust.qubits) #type: ignore
            assert_close(self, rust.state_vector, python.state_vector, f"qft({n})")
            python.iqft(python.qubits)  #type: ignore
            rust.iqft(rust.qubits)  #type: ignore
            assert_close(self, rust.state_vector, python.state_vector, f"iqft({n})", atol=1e-8)

    def test_measure_pauli(self) -> None:
        for pauli in ("Z", "X", "Y", "ZZ", "XY", "IZ"):
            python, rust = PyQC(len(pauli)), rk.QuantumComputer(len(pauli))
            rust.apply(gm.ry_MAT(0.7), [0])
            python.apply(gm.ry_MAT(0.7), [0])
            if len(pauli) > 1:
                rust.apply(gm.h_MAT(), [1])
                python.apply(gm.h_MAT(), [1])
            rust_value = rust.measure_pauli(pauli, rust.qubits, shots=20000)  #type: ignore
            python_value = python.measure_pauli(pauli, python.qubits, shots=20000)  #type: ignore
            self.assertLess(abs(rust_value - python_value), 0.05, f"measure_pauli({pauli})")
            assert_close(self, rust.state_vector, python.state_vector, f"pauli state {pauli}")

    def test_sample(self) -> None:
        def circuit(qc):
            qc.apply(gm.h_MAT(), [0])
            qc.apply(gm.px_MAT(), [1])

        for sampler in ("sample", "sample_parallel"):
            python, rust = PyQC(2), rk.QuantumComputer(2)
            rust_counts = getattr(rust, sampler)(circuit, 500)
            python_counts = getattr(python, sampler)(circuit, 500)
            self.assertEqual(sum(rust_counts.values()), 500)
            self.assertEqual(set(rust_counts), set(python_counts), sampler)

    def test_openqasm3(self) -> None:
        python, rust = PyQC(3, "demo"), rk.QuantumComputer(3, "demo")
        for backend, qubits in ((rust, rust.qubits), (python, python.qubits)):
            backend.apply_gate(gm.h_MAT(), [qubits[0]], qasm_name="h") #type: ignore
            backend.apply_gate(gm.px_MAT(), [qubits[2]], [qubits[0]], qasm_name="x") #type: ignore
        self.assertEqual(rust.to_openqasm3(), python.to_openqasm3())


class SafetyTests(unittest.TestCase):
    def test_invalid_inputs_raise(self) -> None:
        qc = rk.QuantumComputer(3)
        with self.assertRaises(ValueError):
            qc.apply_gate(gm.swap_MAT(), [qc[0], qc[0]])
        with self.assertRaises(ValueError):
            qc.apply_gate(gm.px_MAT(), [qc[0]], [qc[0]])
        with self.assertRaises(ValueError):
            qc.apply_gate(gm.px_MAT(), [0, 1])
        with self.assertRaises(IndexError):
            qc[5]
        with self.assertRaises(KeyError):
            qc["nope"]
        with self.assertRaises(ValueError):
            rk.Gate(np.eye(2) * 2, [], [0])

    def test_measurement_locks_qubit(self) -> None:
        qc = rk.QuantumComputer(2)
        qc[0].measure()
        with self.assertRaises(ValueError):
            qc[0].measure()
        with self.assertRaises(ValueError):
            qc.apply(gm.h_MAT(), [qc[0]])

    def test_history_is_a_real_list(self) -> None:
        qc = rk.QuantumComputer(2)
        self.assertIsInstance(qc.op_hist, list)
        gate = qc.gate(gm.h_MAT(), [0], qasm_name="h")
        qc.op_hist.append(gate)
        self.assertEqual(qc.checkpoint(), 1)
        self.assertIsInstance(qc._custom_operators, dict)

    def test_garbage_collection(self) -> None:
        gc.collect()
        for _ in range(200):
            qc = rk.QuantumComputer(3)
            qc.op_hist.append(qc.gate(gm.h_MAT(), [0]))
            qc.register_operator("h", gm.h_MAT)
            _ = qc.qubits[0].computer
            del qc
        gc.collect()


class SeedTests(unittest.TestCase):
    def setUp(self) -> None:
        import quantum

        self._quantum = quantum
        self._previous_seed = quantum.get_seed()

    def tearDown(self) -> None:
        self._quantum.set_seed(self._previous_seed)

    def test_seed_helpers_share_python_state(self) -> None:
        self.assertTrue(hasattr(rk, "set_seed") and hasattr(rk, "get_seed"))
        rk.set_seed(4242)
        self.assertEqual(self._quantum.get_seed(), 4242)
        self.assertEqual(rk.get_seed(), 4242)

    def test_sampling_is_reproducible(self) -> None:
        def circuit(qc):
            qc.apply(gm.h_MAT(), [0])
            qc.apply(gm.h_MAT(), [1])

        self._quantum.set_seed(1234)
        first = rk.QuantumComputer(2).sample(circuit, 500)
        self._quantum.set_seed(1234)
        second = rk.QuantumComputer(2).sample(circuit, 500)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
