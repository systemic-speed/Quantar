"""Small manual benchmark comparing the Python and Rust simulators.

Run from the repository root::

    python test/test_speed.py

Use fewer qubits/samples while iterating; the defaults below are intentionally
heavy enough to show the compiled backend's advantage.
"""

from __future__ import annotations

import time
from typing import Any

import quantum
from quantum import gate_matrix as gm
from stl.quantum_algs import grovers_alg


def grover_benchmark(qc: Any) -> None:
    """Mark the all-ones data register with the last qubit as ancilla."""

    def mark_all_ones(qc: Any, data_qubits: list, ancilla_qubit: Any) -> None:
        with qc.suppress_history():
            qc.apply_gate(
                gm.px_MAT(), [ancilla_qubit], control_qubits=data_qubits
            )

    grovers_alg(qc, mark_all_ones)


def run(num_qubits: int = 15, num_samples: int = 100_000) -> None:
    for backend_name in ("python", "rust"):
        quantum.use_backend(backend_name)
        qc = quantum.QuantumComputer(num_qubits)

        start = time.perf_counter_ns()
        counts = qc.sample_parallel(grover_benchmark, num_samples)
        elapsed = time.perf_counter_ns() - start

        print(f"{backend_name}'s results: {counts}")
        print(f"{backend_name} took {elapsed} ns")

    quantum.use_backend("python")


if __name__ == "__main__":
    run()
