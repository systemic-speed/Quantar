"""Physics kernels: Rust when available, NumPy otherwise.

The compiled kernels live in ``quantum.quantar_kernels`` alongside the quantum
backend. Importing this module does **not** import or build the extension; the
loader runs on the first kernel call. Set ``QUANTAR_PHYSICS_KERNELS=0`` to stay
on the NumPy path, or call :func:`use_rust_physics` at runtime.

Each public kernel has a ``_numpy`` twin, so behaviour is identical whether the
extension is installed or not. The Rust passes release the GIL and use rayon,
which matters for large batches of points.
"""

from __future__ import annotations

import importlib
import os

import numpy as np

_ENV_PHYSICS_KERNELS = "QUANTAR_PHYSICS_KERNELS"

_REQUIRED_NAMES = (
    "kinetic_energy",
    "lorentz_force",
    "jacobian_from_offsets",
    "verlet_position",
    "verlet_update",
    "verlet_velocity",
)


def _enabled_by_environment() -> bool:
    value = os.environ.get(_ENV_PHYSICS_KERNELS, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


_enabled = _enabled_by_environment()
_module: object | None = None
_checked = False


def _rust_module():
    """Return the validated extension module, or ``None`` for the fallback."""
    global _checked, _module
    if not _enabled:
        return None
    if not _checked:
        _checked = True
        try:
            candidate = importlib.import_module("quantum.quantar_kernels")
        except (ImportError, OSError):
            return None
        if all(hasattr(candidate, name) for name in _REQUIRED_NAMES):
            _module = candidate
    return _module


def rust_available() -> bool:
    """Return whether the compiled physics kernels can be used."""
    return _rust_module() is not None


def use_rust_physics(enabled: bool | None = None) -> bool:
    """Enable or disable the compiled kernels and return the active setting.

    With no argument this only reports the current setting. Passing a boolean
    changes it for subsequent kernel calls; the NumPy implementations stay
    available either way.
    """
    global _enabled, _checked, _module
    if enabled is not None:
        _enabled = bool(enabled)
        _checked = False
        _module = None
    return _enabled and rust_available()


# ── NumPy fallbacks ──────────────────────────────────────────────────────────


def _masses_numpy(masses, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast scalar/per-particle/per-component masses to ``shape``."""
    values = np.asarray(masses, dtype=float)
    if values.shape == shape:
        return values
    if values.ndim == 0:
        return np.full(shape, float(values))
    if values.shape == shape[:-1]:
        return np.broadcast_to(values[..., None], shape)
    return np.broadcast_to(values, shape)


def kinetic_energy_numpy(velocities, masses=1.0) -> float:
    """NumPy twin of the Rust kinetic-energy sum."""
    values = np.asarray(velocities, dtype=float)
    mass = _masses_numpy(masses, values.shape)
    return float(0.5 * np.sum(mass * values * values))


def lorentz_force_numpy(charge, electric, magnetic, velocity) -> np.ndarray:
    """NumPy twin of the Rust Lorentz-force kernel."""
    electric = np.asarray(electric, dtype=float)
    magnetic = np.asarray(magnetic, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    electric, magnetic, velocity = np.broadcast_arrays(electric, magnetic, velocity)
    if electric.shape[-1] != 3:
        raise ValueError("lorentz_force expects 3D vectors.")
    return charge * (electric + np.cross(velocity, magnetic))


def jacobian_from_offsets_numpy(values, n: int, dim: int, out_dim: int, h: float) -> np.ndarray:
    """NumPy twin of :func:`jacobian_from_offsets`."""
    if h <= 0:
        raise ValueError("h must be positive.")
    if n < 1 or dim < 1 or out_dim < 1:
        raise ValueError("n, dim and out_dim must all be positive.")
    source = np.asarray(values, dtype=float).reshape(2 * dim, n, out_dim)
    difference = source[0::2] - source[1::2]  # (dim, n, out_dim)
    return (difference / (2.0 * h)).transpose(1, 2, 0)


def verlet_position_numpy(positions, velocities, acceleration, dt: float) -> np.ndarray:
    """NumPy twin of :func:`verlet_position`."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    x = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    return x + v * dt + 0.5 * a * dt * dt


def verlet_velocity_numpy(velocities, acceleration, next_acceleration, dt: float) -> np.ndarray:
    """NumPy twin of :func:`verlet_velocity`."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    v = np.asarray(velocities, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    next_a = np.asarray(next_acceleration, dtype=float)
    return v + 0.5 * (a + next_a) * dt


def verlet_update_numpy(positions, velocities, acceleration, next_acceleration, dt: float):
    """NumPy twin of :func:`verlet_update`."""
    return (
        verlet_position_numpy(positions, velocities, acceleration, dt),
        verlet_velocity_numpy(velocities, acceleration, next_acceleration, dt),
    )


# ── Dispatched kernels ───────────────────────────────────────────────────────


def kinetic_energy(velocities, masses=1.0) -> float:
    """``0.5 * sum(m * v**2)`` through Rust, with a NumPy fallback."""
    module = _rust_module()
    if module is not None:
        try:
            return float(module.kinetic_energy(velocities, masses))
        except (TypeError, ValueError):
            pass
    return kinetic_energy_numpy(velocities, masses)


def lorentz_force(charge, electric, magnetic, velocity) -> np.ndarray:
    """``charge * (E + v x B)`` through Rust, with a NumPy fallback."""
    electric = np.asarray(electric, dtype=float)
    magnetic = np.asarray(magnetic, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    electric, magnetic, velocity = np.broadcast_arrays(electric, magnetic, velocity)
    module = _rust_module()
    if module is not None and electric.shape[-1] == 3:
        try:
            result = module.lorentz_force(charge, electric, magnetic, velocity)
            return np.asarray(result, dtype=float).reshape(electric.shape)
        except (TypeError, ValueError):
            pass
    return lorentz_force_numpy(charge, electric, magnetic, velocity)


def jacobian_from_offsets(values, n: int, dim: int, out_dim: int, h: float) -> np.ndarray:
    """Reduce stacked plus/minus evaluations to ``(n, out_dim, dim)``.

    ``values`` is laid out as ``(2 * dim, n, out_dim)``: for each coordinate
    axis, all plus evaluations come before all minus evaluations.
    """
    module = _rust_module()
    if module is not None:
        try:
            return np.asarray(
                module.jacobian_from_offsets(values, n, dim, out_dim, h), dtype=float
            )
        except (TypeError, ValueError):
            pass
    return jacobian_from_offsets_numpy(values, n, dim, out_dim, h)


def verlet_position(positions, velocities, acceleration, dt: float) -> np.ndarray:
    """First Verlet half-step: new positions from the current acceleration."""
    module = _rust_module()
    if module is not None:
        try:
            return np.asarray(
                module.verlet_position(positions, velocities, acceleration, dt), dtype=float
            )
        except (TypeError, ValueError):
            pass
    return verlet_position_numpy(positions, velocities, acceleration, dt)


def verlet_velocity(velocities, acceleration, next_acceleration, dt: float) -> np.ndarray:
    """Second Verlet half-step: new velocities from both accelerations."""
    module = _rust_module()
    if module is not None:
        try:
            return np.asarray(
                module.verlet_velocity(velocities, acceleration, next_acceleration, dt),
                dtype=float,
            )
        except (TypeError, ValueError):
            pass
    return verlet_velocity_numpy(velocities, acceleration, next_acceleration, dt)


def verlet_update(positions, velocities, acceleration, next_acceleration, dt: float):
    """One velocity-Verlet update, returning new positions and velocities.

    For integrators that evaluate acceleration at the new position, use
    :func:`verlet_position` and :func:`verlet_velocity` separately instead.
    """
    module = _rust_module()
    if module is not None:
        try:
            new_x, new_v = module.verlet_update(
                positions, velocities, acceleration, next_acceleration, dt
            )
            return np.asarray(new_x, dtype=float), np.asarray(new_v, dtype=float)
        except (TypeError, ValueError):
            pass
    return verlet_update_numpy(positions, velocities, acceleration, next_acceleration, dt)


__all__ = [
    "jacobian_from_offsets",
    "jacobian_from_offsets_numpy",
    "kinetic_energy",
    "kinetic_energy_numpy",
    "lorentz_force",
    "lorentz_force_numpy",
    "rust_available",
    "use_rust_physics",
    "verlet_position",
    "verlet_position_numpy",
    "verlet_update",
    "verlet_update_numpy",
    "verlet_velocity",
    "verlet_velocity_numpy",
]
