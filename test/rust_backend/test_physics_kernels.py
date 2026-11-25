"""Direct tests for the Rust physics kernels.

Uses ``QUANTAR_RUST_TEST_DIR`` when set, just like ``test_rust_api.py``, so a
freshly built ``.pyd`` can be tested without replacing the installed one.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_test_dir = os.environ.get("QUANTAR_RUST_TEST_DIR")
if _test_dir:
    sys.path.insert(0, _test_dir)
    import quantar_kernels as rk  # type: ignore
else:
    from quantum import quantar_kernels as rk


class PhysicsKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(12345)

    def test_required_names_exist(self) -> None:
        for name in (
            "kinetic_energy",
            "lorentz_force",
            "jacobian_from_offsets",
            "verlet_position",
            "verlet_velocity",
            "verlet_update",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(rk, name)))

    def test_kinetic_energy(self) -> None:
        velocities = self.rng.normal(size=(6, 2))
        self.assertAlmostEqual(
            rk.kinetic_energy(velocities, 2.0),
            float(np.sum(velocities**2)),
            places=10,
        )
        masses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(
            rk.kinetic_energy(velocities, masses),
            float(0.5 * np.sum(masses[:, None] * velocities**2)),
            places=10,
        )

    def test_lorentz_force(self) -> None:
        electric = self.rng.normal(size=(4, 3))
        magnetic = self.rng.normal(size=(4, 3))
        velocity = self.rng.normal(size=(4, 3))
        actual = np.asarray(rk.lorentz_force(1.5, electric, magnetic, velocity))
        expected = 1.5 * (electric + np.cross(velocity, magnetic))
        self.assertEqual(actual.shape, (4, 3))
        self.assertTrue(np.allclose(actual, expected))

    def test_jacobian_from_offsets(self) -> None:
        n, dim, out_dim, h = 5, 2, 3, 1e-3
        values = self.rng.normal(size=(2 * dim, n, out_dim))
        actual = np.asarray(rk.jacobian_from_offsets(values, n, dim, out_dim, h))
        expected = ((values[0::2] - values[1::2]) / (2.0 * h)).transpose(1, 2, 0)
        self.assertEqual(actual.shape, (n, out_dim, dim))
        self.assertTrue(np.allclose(actual, expected))

    def test_verlet_steps(self) -> None:
        x = self.rng.normal(size=(7, 2))
        v = self.rng.normal(size=(7, 2))
        a = self.rng.normal(size=(7, 2))
        next_a = self.rng.normal(size=(7, 2))
        dt = 0.02
        new_x = np.asarray(rk.verlet_position(x, v, a, dt))
        new_v = np.asarray(rk.verlet_velocity(v, a, next_a, dt))
        both_x, both_v = rk.verlet_update(x, v, a, next_a, dt)
        self.assertTrue(np.allclose(new_x, x + v * dt + 0.5 * a * dt * dt))
        self.assertTrue(np.allclose(new_v, v + 0.5 * (a + next_a) * dt))
        self.assertTrue(np.allclose(both_x, new_x))
        self.assertTrue(np.allclose(both_v, new_v))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            rk.kinetic_energy([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            rk.lorentz_force(1.0, np.zeros(3), np.zeros(3), np.zeros(4))
        with self.assertRaises(ValueError):
            rk.jacobian_from_offsets(np.zeros(4), 1, 1, 1, -1.0)
        with self.assertRaises(ValueError):
            rk.verlet_position(np.zeros(3), np.zeros(2), np.zeros(3), 0.1)


if __name__ == "__main__":
    unittest.main()
