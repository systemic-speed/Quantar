"""Compare compiled physics kernels with their NumPy twins.

Run from the repo root with::

    python test/bench_physics.py

The script reports the median of several runs. It uses
``stl.physics_algs.kernels`` so no code changes are needed in user programs.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from stl.physics_algs import kernels


def bench(fn, repeats: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    rng = np.random.default_rng(42)
    n, dim = 200_000, 3

    positions = rng.normal(size=(n, dim))
    velocities = rng.normal(size=(n, dim))
    acceleration = -positions
    fields = (
        rng.normal(size=(n, 3)),
        rng.normal(size=(n, 3)),
        rng.normal(size=(n, 3)),
    )
    offsets = rng.normal(size=(2 * dim, n, dim))

    cases = {
        "kinetic_energy": lambda use_rust: (
            kernels.kinetic_energy if use_rust else kernels.kinetic_energy_numpy
        )(velocities, np.ones(n)),
        "lorentz_force": lambda use_rust: (
            kernels.lorentz_force if use_rust else kernels.lorentz_force_numpy
        )(1.5, *fields),
        "jacobian_reduction": lambda use_rust: (
            kernels.jacobian_from_offsets
            if use_rust
            else kernels.jacobian_from_offsets_numpy
        )(offsets, n, dim, dim, 1e-5),
        "verlet_position": lambda use_rust: (
            kernels.verlet_position if use_rust else kernels.verlet_position_numpy
        )(positions, velocities, acceleration, 0.01),
    }

    print(f"points={n} dim={dim} rust_available={kernels.rust_available()}")
    print(f"{'kernel':<20} {'numpy (ms)':>12} {'rust (ms)':>12} {'speedup':>9}")
    for name, run in cases.items():
        numpy_time = bench(lambda: run(False), repeats=3, warmup=1)
        rust_time = bench(lambda: run(True), repeats=3, warmup=1)
        speedup = numpy_time / rust_time if rust_time else float("inf")
        print(
            f"{name:<20} {numpy_time * 1e3:12.2f} {rust_time * 1e3:12.2f} {speedup:8.2f}x"
        )


if __name__ == "__main__":
    main()
