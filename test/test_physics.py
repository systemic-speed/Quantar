"""Tests for Rust-backed physics kernels and the higher-level helpers."""

from __future__ import annotations

import contextlib
import unittest

import numpy as np

from stl.physics_algs import (
    Dual,
    SpaceMetric,
    Trajectory,
    VectorField,
    angular_momentum,
    gradient,
    hessian,
    kinetic_energy,
    laplacian,
    momentum,
    rust_available,
    velocity_verlet,
)
from stl.physics_algs import kernels


@contextlib.contextmanager
def physics_backend(enabled: bool):
    """Run a block with the requested kernel path, then restore the setting."""
    previous = kernels._enabled
    kernels.use_rust_physics(enabled)
    try:
        yield
    finally:
        kernels.use_rust_physics(previous)


class KernelSelectionTests(unittest.TestCase):
    def test_extension_is_detected_without_forcing_a_build(self) -> None:
        # Missing/stale extensions are fine; the flag only reports availability.
        self.assertIsInstance(rust_available(), bool)

    def test_numpy_twins_are_always_importable(self) -> None:
        for name in (
            "kinetic_energy_numpy",
            "lorentz_force_numpy",
            "jacobian_from_offsets_numpy",
            "verlet_position_numpy",
            "verlet_velocity_numpy",
            "verlet_update_numpy",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(kernels, name)))

    def test_toggle_changes_the_setting(self) -> None:
        previous = kernels._enabled
        try:
            kernels.use_rust_physics(False)
            self.assertFalse(kernels.use_rust_physics())
            kernels.use_rust_physics(True)
            self.assertEqual(kernels.use_rust_physics(), rust_available())
        finally:
            kernels.use_rust_physics(previous)


class KernelParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20261125)

    def test_kinetic_energy_mass_broadcasting(self) -> None:
        velocities = self.rng.normal(size=(4, 3))
        cases = (
            2.0,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0]),
            np.ones((4, 3)),
        )
        with physics_backend(False):
            expected = [kinetic_energy(velocities, masses) for masses in cases]
        with physics_backend(True):
            actual = [kinetic_energy(velocities, masses) for masses in cases]
        for got, want, masses in zip(actual, expected, cases):
            with self.subTest(masses=np.asarray(masses).shape):
                self.assertAlmostEqual(got, want, places=12)

    def test_lorentz_force_matches_cross_product(self) -> None:
        electric = self.rng.normal(size=(5, 3))
        magnetic = self.rng.normal(size=(5, 3))
        velocity = self.rng.normal(size=(5, 3))
        expected = -2.0 * (electric + np.cross(velocity, magnetic))
        with physics_backend(False):
            fallback = kernels.lorentz_force(-2.0, electric, magnetic, velocity)
        with physics_backend(True):
            compiled = kernels.lorentz_force(-2.0, electric, magnetic, velocity)
        self.assertTrue(np.allclose(fallback, expected))
        self.assertTrue(np.allclose(compiled, expected))

    def test_jacobian_reduction(self) -> None:
        n, dim, out_dim, h = 7, 3, 2, 1e-3
        values = self.rng.normal(size=(2 * dim, n, out_dim))
        with physics_backend(False):
            fallback = kernels.jacobian_from_offsets(values, n, dim, out_dim, h)
        with physics_backend(True):
            compiled = kernels.jacobian_from_offsets(values, n, dim, out_dim, h)
        expected = ((values[0::2] - values[1::2]) / (2.0 * h)).transpose(1, 2, 0)
        self.assertEqual(fallback.shape, (n, out_dim, dim))
        self.assertTrue(np.allclose(fallback, expected))
        self.assertTrue(np.allclose(compiled, expected))

    def test_verlet_half_steps_and_combined_update(self) -> None:
        x = self.rng.normal(size=(4, 2))
        v = self.rng.normal(size=(4, 2))
        a = self.rng.normal(size=(4, 2))
        next_a = self.rng.normal(size=(4, 2))
        dt = 0.01
        with physics_backend(False):
            pos_fallback = kernels.verlet_position(x, v, a, dt)
            vel_fallback = kernels.verlet_velocity(v, a, next_a, dt)
            both_fallback = kernels.verlet_update(x, v, a, next_a, dt)
        with physics_backend(True):
            pos_compiled = kernels.verlet_position(x, v, a, dt)
            vel_compiled = kernels.verlet_velocity(v, a, next_a, dt)
            both_compiled = kernels.verlet_update(x, v, a, next_a, dt)

        expected_pos = x + v * dt + 0.5 * a * dt * dt
        expected_vel = v + 0.5 * (a + next_a) * dt
        for actual in (pos_fallback, pos_compiled, both_fallback[0], both_compiled[0]):
            self.assertTrue(np.allclose(actual, expected_pos))
        for actual in (vel_fallback, vel_compiled, both_fallback[1], both_compiled[1]):
            self.assertTrue(np.allclose(actual, expected_vel))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kernels.jacobian_from_offsets(np.zeros((2, 1, 1)), 1, 1, 1, 0.0)
        with self.assertRaises(ValueError):
            kernels.verlet_position(np.zeros(3), np.zeros(3), np.zeros(3), 0.0)
        with self.assertRaises(ValueError):
            kernels.verlet_velocity(np.zeros(3), np.zeros(2), np.zeros(3), 0.1)
        with physics_backend(True):
            with self.assertRaises(ValueError):
                kernels.jacobian_from_offsets(np.zeros(10), 1, 1, 1, 1e-3)


class VectorFieldTests(unittest.TestCase):
    def _check_rotation_field(self) -> None:
        field = VectorField(lambda p: np.array([-p[1], p[0]]))
        points = np.array([[1.0, 2.0], [3.0, -1.0]])
        jacobian = field.jacobian(points)
        expected = np.tile(np.array([[0.0, -1.0], [1.0, 0.0]]), (2, 1, 1))
        self.assertTrue(np.allclose(jacobian, expected, atol=1e-6))
        self.assertTrue(np.allclose(field.divergence(points), 0.0, atol=1e-6))
        self.assertTrue(np.allclose(field.curl(points), 2.0, atol=1e-6))
        hessian = field.hessian(points)
        self.assertEqual(hessian.shape, (2, 2, 2, 2))
        self.assertTrue(np.allclose(hessian, 0.0, atol=1e-4))
        self.assertTrue(np.allclose(field.laplacian(points), 0.0, atol=1e-4))

    def test_derivatives_on_both_paths(self) -> None:
        for enabled in (False, True):
            with self.subTest(rust=enabled), physics_backend(enabled):
                self._check_rotation_field()

    def test_scalar_derivatives(self) -> None:
        def scalar(point):
            return point[0] ** 2 + point[1] ** 3

        for enabled in (False, True):
            with self.subTest(rust=enabled), physics_backend(enabled):
                self.assertTrue(np.allclose(gradient(scalar, [2.0, 3.0]), [4.0, 27.0], atol=1e-5))
                matrix = hessian(scalar, [2.0, 3.0])
                self.assertTrue(np.allclose(matrix, [[2.0, 0.0], [0.0, 18.0]], atol=1e-3))
                self.assertAlmostEqual(float(laplacian(scalar, [2.0, 3.0])), 20.0, places=2)

    def test_batched_field_is_evaluated_as_one_batch(self) -> None:
        calls: list[int] = []

        def batched(points):
            calls.append(len(points))
            return np.column_stack((-points[:, 1], points[:, 0]))

        field = VectorField(batched, dim=2, batched=True)
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        jacobian = field.jacobian(points)
        self.assertEqual(jacobian.shape, (2, 2, 2))
        self.assertEqual(calls, [8])  # 2 axes x (plus, minus) x 2 points
        self.assertTrue(np.allclose(jacobian[0], [[0.0, -1.0], [1.0, 0.0]], atol=1e-6))


class DynamicsTests(unittest.TestCase):
    def test_momentum_and_angular_momentum(self) -> None:
        self.assertTrue(np.array_equal(momentum([3.0, 4.0], 2.0), [6.0, 8.0]))
        self.assertTrue(
            np.allclose(
                angular_momentum([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], 3.0), [0.0, 0.0, 6.0]
            )
        )
        self.assertTrue(
            np.allclose(
                angular_momentum([3.0, 0.0, 0.0], [0.0, 1.0, 0.0], origin=[1.0, 0.0, 0.0]),
                [0.0, 0.0, 2.0],
            )
        )
        with self.assertRaises(ValueError):
            angular_momentum([1.0, 0.0], [0.0, 1.0])

    def test_velocity_verlet_conserves_energy(self) -> None:
        for enabled in (False, True):
            with self.subTest(rust=enabled), physics_backend(enabled):
                trajectory = velocity_verlet(
                    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.01, 628, lambda point: -point
                )
                self.assertIsInstance(trajectory, Trajectory)
                self.assertEqual(len(trajectory), 629)
                self.assertTrue(np.allclose(trajectory.speeds(), 1.0, atol=5e-3))
                self.assertTrue(np.allclose(trajectory.kinetic_energies(), 0.5, atol=5e-3))
                self.assertTrue(np.allclose(trajectory.momenta(), trajectory.velocities))
                energies = trajectory.energy(lambda point: 0.5 * np.sum(point * point))
                self.assertTrue(np.allclose(energies, 1.0, atol=5e-3))
                self.assertEqual(trajectory.angular_momentum().shape, (629, 3))

    def test_trajectory_slices(self) -> None:
        trajectory = Trajectory([[1.0]], [[2.0]], [0.0])
        time, position, velocity = trajectory[0]
        self.assertEqual((time, position.tolist(), velocity.tolist()), (0.0, [1.0], [2.0]))


class DualAndMetricTests(unittest.TestCase):
    def test_value_and_gradient(self) -> None:
        value, gradient_values = Dual.value_and_gradient(
            lambda xs: xs[0] ** 2 + xs[1] ** 3, [2.0, 3.0]
        )
        self.assertAlmostEqual(value, 31.0)
        self.assertTrue(np.allclose(gradient_values, [4.0, 27.0]))

    def test_jacobian(self) -> None:
        result = Dual.jacobian(
            lambda xs: [xs[0] ** 2, xs[0] * xs[1]], [2.0, 3.0]
        )
        self.assertTrue(np.allclose(result, [[4.0, 0.0], [3.0, 2.0]]))

    def test_metric_scale_factors_and_volume(self) -> None:
        self.assertEqual(SpaceMetric().scale_factors([1.0, 2.0]), [1.0, 1.0])
        spherical = SpaceMetric("spherical")
        self.assertTrue(
            np.allclose(spherical.scale_factors([2.0, np.pi / 2, 0.0]), [1.0, 2.0, 2.0])
        )
        self.assertAlmostEqual(spherical.volume_element([2.0, np.pi / 2, 0.0]), 4.0)
        cylindrical = SpaceMetric("cylindrical")
        self.assertEqual(cylindrical.scale_factors([3.0, 0.0, 5.0]), [1.0, 3.0, 1.0])
        self.assertEqual(cylindrical.transform_gradient([1.0, 2.0, 3.0], [3.0, 0.0, 5.0]), [1.0, 2 / 3, 3.0])
        with self.assertRaises(ValueError):
            SpaceMetric("toroidal")
        with self.assertRaises(ValueError):
            spherical.scale_factors([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
