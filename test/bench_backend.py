"""Compare the Python simulator with the installed Rust backend.

Run from the repo root with::

    python test/bench_backend.py

The script warms both backends, reports the median of several runs, and uses
the extension installed at ``quantum.quantar_kernels``.
"""

from __future__ import annotations

import statistics
import time

import quantum.gate_matrix as gm
from quantum import quantar_kernels as RustQC
from quantum.qubit_abs import QuantumComputer as PythonQC


def bench(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def single_h(QC, n: int, count: int):
    qc = QC(n)
    h = gm.h_MAT()
    target = 0

    def run():
        nonlocal target
        for _ in range(count):
            qc.apply_gate(h, [target])
            target = (target + 1) % n

    return run


def h_program(QC, n: int, count: int):
    qc = QC(n)
    h = gm.h_MAT()
    program = [qc.gate(h, [i % n]) for i in range(count)]

    def run():
        qc.apply(program)

    return run


def diagonal_program(QC, n: int, depth: int):
    qc = QC(n)
    program = [
        qc.gate(gm.cp_MAT(0.01 * (i + 1)), [i % n, (i + 1) % n])
        for i in range(depth)
    ]

    def run():
        qc.apply(program)

    return run


def swap_program(QC, n: int, pairs: int):
    qc = QC(n)
    program = [qc.gate(gm.swap_MAT(), [2 * i, 2 * i + 1]) for i in range(pairs)]

    def run():
        qc.apply(program)

    return run


def qft(QC, n: int):
    qc = QC(n)
    target = [qc[i] for i in range(n)]

    def run():
        qc.qft(target, record=False)

    return run


WORKLOADS = [
    ("10q: 100 H gates", lambda QC: single_h(QC, 10, 100), 40),
    ("16q: 100 H gates", lambda QC: single_h(QC, 16, 100), 20),
    ("20q: 100 H gates", lambda QC: single_h(QC, 20, 100), 10),
    ("16q: 100 H via apply()", lambda QC: h_program(QC, 16, 100), 20),
    ("20q: 100 H via apply()", lambda QC: h_program(QC, 20, 100), 10),
    ("16q: 64 fused CP", lambda QC: diagonal_program(QC, 16, 64), 20),
    ("20q: 64 fused CP", lambda QC: diagonal_program(QC, 20, 64), 10),
    ("16q: 8 fused SWAP", lambda QC: swap_program(QC, 16, 8), 30),
    ("12q: QFT (no record)", lambda QC: qft(QC, 12), 20),
]


def main() -> None:
    print(f"{'workload':<24} {'python':>12} {'rust':>12} {'speedup':>10}")
    print("-" * 62)
    for label, factory, repeats in WORKLOADS:
        py = bench(factory(PythonQC), repeats)
        rust = bench(factory(RustQC.QuantumComputer), repeats)
        print(f"{label:<24} {py * 1e3:>10.3f}ms {rust * 1e3:>10.3f}ms {py / rust:>9.2f}x")


if __name__ == "__main__":
    main()
