"""Vector fields and numerical differential operators.

A :class:`VectorField` wraps any callable that maps a point to a vector.  It
works with plain Python functions, NumPy code, or numba-jitted functions, and
adds the operations you normally want on a field: evaluation, magnitude,
normalisation, Jacobian, divergence, curl, line integrals, and streamlines.

Points follow one convention throughout:

* a single point is ``(dim,)``;
* a batch of points is ``(n, dim)``.

The wrapped function is point-wise by default -- ``f(point) -> vector``.  Pass
``batched=True`` if it instead accepts an ``(n, dim)`` array.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from . import kernels as _kernels


def _as_points(points) -> tuple[np.ndarray, bool]:
    """Return ``(points_2d, single)`` for a point or a batch of points."""
    array = np.asarray(points, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1), True
    if array.ndim == 1:
        return array.reshape(1, -1), True
    if array.ndim == 2:
        return array, False
    raise ValueError(
        f"Expected a point (dim,) or points (n, dim), got shape {array.shape}."
    )


def _step_along(axis: int, dim: int, h: float) -> np.ndarray:
    """Return a zero vector with ``h`` in ``axis``."""
    step = np.zeros(dim)
    step[axis] = h
    return step


class VectorField:
    """A vector-valued function of position.

    Parameters
    ----------
    function:
        ``f(point) -> vector`` by default, or ``f(points) -> vectors`` when
        ``batched=True``.
    dim:
        Coordinate dimension.  Inferred from the first call when omitted.
    batched:
        Whether ``function`` accepts an ``(n, dim)`` array instead of one point.

    Example
    -------
    >>> field = VectorField(lambda p: np.array([-p[1], p[0]]))
    >>> field([0.0, 1.0])
    array([-1.,  0.])
    >>> float(np.round(field.divergence([1.0, 2.0]), 6))
    0.0
    >>> float(np.round(field.curl([1.0, 2.0]), 6))
    2.0
    """

    def __init__(
        self,
        function: Callable,
        dim: int | None = None,
        *,
        batched: bool = False,
    ):
        if not callable(function):
            raise TypeError("VectorField needs a callable.")
        if dim is not None and dim < 1:
            raise ValueError("dim must be positive.")
        self.function = function
        self.dim = dim
        self.batched = batched

    # ─── Evaluation ───

    def _vectors(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the raw function on an ``(n, dim)`` array."""
        if self.batched:
            values = np.asarray(self.function(points), dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            elif values.shape != points.shape and values.shape == points.shape[::-1]:
                values = values.T
        else:
            values = np.asarray(
                [self.function(point) for point in points],
                dtype=float,
            )
            if values.ndim == 1:
                values = values[:, None]

        if values.ndim != 2 or values.shape[0] != points.shape[0]:
            raise ValueError(
                f"Vector field returned shape {values.shape} for {points.shape[0]} points; "
                "expected (n, dim)."
            )
        if self.dim is None:
            self.dim = values.shape[1] #type: ignore
        elif values.shape[1] != self.dim: #type: ignore
            raise ValueError(
                f"Vector field returned {values.shape[1]} components, expected {self.dim}." #type: ignore
            )
        return values

    def __call__(self, points):
        """Evaluate the field at one point or at a batch of points."""
        array, single = _as_points(points)
        values = self._vectors(array)
        return values[0] if single else values

    def magnitude(self, points):
        """Length of the field vector(s)."""
        values = np.asarray(self(points), dtype=float)
        return np.linalg.norm(values, axis=-1)

    def normalize(self, points):
        """Unit field vector(s); zero vectors stay zero."""
        values = np.asarray(self(points), dtype=float)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)

    # ─── Differential operators ───

    def _central(self, points, h: float):
        """Return the central-difference Jacobian and single-point flag.

        All offset points are evaluated in one batch, then reduced by the
        compiled Jacobian kernel (or its NumPy twin). This keeps the callback
        count low for batched fields and makes the reduction a single pass.
        """
        if h <= 0:
            raise ValueError("h must be positive.")
        array, single = _as_points(points)
        n, dim = array.shape
        offsets = np.concatenate(
            [
                pair
                for axis in range(dim)
                for pair in (
                    array + _step_along(axis, dim, h),
                    array - _step_along(axis, dim, h),
                )
            ],
            axis=0,
        )
        values = self._vectors(offsets)
        out_dim = values.shape[1]
        jacobian = _kernels.jacobian_from_offsets(
            values.reshape(-1), n, dim, out_dim, h
        )
        return jacobian, single

    def jacobian(self, points, h: float = 1e-6):
        """``dF_i / dx_j`` at each point: ``(dim, dim)`` or ``(n, dim, dim)``."""
        jacobian, single = self._central(points, h)
        return jacobian[0] if single else jacobian

    def divergence(self, points, h: float = 1e-6):
        """Trace of the Jacobian: ``sum_i dF_i / dx_i``."""
        jacobian, single = self._central(points, h)
        if jacobian.shape[1] != jacobian.shape[2]:
            raise ValueError(
                "Divergence needs matching input/output dimensions, "
                f"got {jacobian.shape[2]} -> {jacobian.shape[1]}."
            )
        values = np.trace(jacobian, axis1=1, axis2=2)
        return float(values[0]) if single else values

    def curl(self, points, h: float = 1e-6):
        """Curl of the field.

        Returns a scalar in 2D (the out-of-plane component), a 3-vector in 3D,
        and ``0`` in 1D.
        """
        jacobian, single = self._central(points, h)
        out_dim, in_dim = jacobian.shape[1], jacobian.shape[2]
        if out_dim != in_dim:
            raise ValueError(
                f"Curl needs matching input/output dimensions, got {in_dim} -> {out_dim}."
            )
        if in_dim == 1:
            values = np.zeros(jacobian.shape[0])
        elif in_dim == 2:
            values = jacobian[:, 1, 0] - jacobian[:, 0, 1]
        elif in_dim == 3:
            values = np.stack(
                [
                    jacobian[:, 2, 1] - jacobian[:, 1, 2],
                    jacobian[:, 0, 2] - jacobian[:, 2, 0],
                    jacobian[:, 1, 0] - jacobian[:, 0, 1],
                ],
                axis=-1,
            )
        else:
            raise ValueError("Curl is only implemented for 1D, 2D and 3D fields.")
        if in_dim in (1, 2):
            return float(values[0]) if single else values
        return values[0] if single else values

    def hessian(self, points, h: float = 1e-5):
        """Second derivatives ``d²F_i / dx_a dx_j`` at each point.

        Returns ``(out_dim, dim, dim)`` for one point or
        ``(n, out_dim, dim, dim)`` for a batch. The result is built from two
        passes of the central-difference Jacobian, so it is substantially more
        sensitive to roundoff than the first derivatives: keep ``h`` moderate.
        """
        if h <= 0:
            raise ValueError("h must be positive.")
        array, single = _as_points(points)
        n, dim = array.shape
        offsets = np.concatenate(
            [
                pair
                for axis in range(dim)
                for pair in (
                    array + _step_along(axis, dim, h),
                    array - _step_along(axis, dim, h),
                )
            ],
            axis=0,
        )
        jacobians = self.jacobian(offsets, h).reshape(2 * dim, n, -1, dim)
        out_dim = jacobians.shape[2]
        hessian = _kernels.jacobian_from_offsets(
            jacobians.reshape(-1), n, dim, out_dim * dim, h
        ).reshape(n, out_dim, dim, dim)
        return hessian[0] if single else hessian

    def laplacian(self, points, h: float = 1e-5):
        """Vector Laplacian: ``sum_j d²F_i / dx_j²`` at each point."""
        hessian = self.hessian(points, h)
        single = np.ndim(hessian) == 3
        values = np.trace(hessian, axis1=-2, axis2=-1)
        return values[0] if single else values

    # ─── Integrals and trajectories ───

    def line_integral(self, path) -> float:
        """Work ``∫ F · dl`` along a polyline given as ``(m, dim)`` points."""
        path = np.asarray(path, dtype=float)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        if path.ndim != 2 or path.shape[0] < 2:
            return 0.0
        segments = np.diff(path, axis=0)
        midpoints = 0.5 * (path[:-1] + path[1:])
        values = np.asarray(self(midpoints), dtype=float)
        return float(np.sum(values * segments))

    def streamline(self, start, dt: float, steps: int) -> np.ndarray:
        """Integrate ``dx/dt = F(x)`` with classical RK4.

        Returns an ``(steps + 1, dim)`` array of positions.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        point = np.asarray(start, dtype=float).copy()
        history = [point.copy()]
        for _ in range(steps):
            k1 = np.asarray(self(point), dtype=float)
            k2 = np.asarray(self(point + 0.5 * dt * k1), dtype=float)
            k3 = np.asarray(self(point + 0.5 * dt * k2), dtype=float)
            k4 = np.asarray(self(point + dt * k3), dtype=float)
            point = point + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            history.append(point.copy())
        return np.asarray(history)

    def __repr__(self) -> str:
        name = getattr(self.function, "__name__", type(self.function).__name__)
        return f"VectorField({name}, dim={self.dim}, batched={self.batched})"


def gradient(scalar_field: Callable, points, h: float = 1e-6):
    """Numerical gradient of a point-wise scalar field.

    ``scalar_field(point) -> float``; returns ``(dim,)`` for one point or
    ``(n, dim)`` for a batch.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    array, single = _as_points(points)
    n, dim = array.shape
    offsets = np.concatenate(
        [
            pair
            for axis in range(dim)
            for pair in (
                array + _step_along(axis, dim, h),
                array - _step_along(axis, dim, h),
            )
        ],
        axis=0,
    )
    values = np.asarray([scalar_field(point) for point in offsets], dtype=float)
    result = _kernels.jacobian_from_offsets(
        values.reshape(-1, 1), n, dim, 1, h
    )[:, 0, :]
    return result[0] if single else result


def hessian(scalar_field: Callable, points, h: float = 1e-5):
    """Numerical Hessian ``d²f / dx_i dx_j`` of a point-wise scalar field.

    Returns ``(dim, dim)`` for one point or ``(n, dim, dim)`` for a batch.
    Like :func:`gradient`, the field stays a plain Python callable; only the
    reductions are handed to the compiled kernel.
    """
    if h <= 0:
        raise ValueError("h must be positive.")
    array, single = _as_points(points)
    n, dim = array.shape
    offsets = np.concatenate(
        [
            pair
            for axis in range(dim)
            for pair in (
                array + _step_along(axis, dim, h),
                array - _step_along(axis, dim, h),
            )
        ],
        axis=0,
    )
    gradients = gradient(scalar_field, offsets, h).reshape(2 * dim, n, dim)
    values = _kernels.jacobian_from_offsets(
        gradients.reshape(-1), n, dim, dim, h
    )
    return values[0] if single else values


def laplacian(scalar_field: Callable, points, h: float = 1e-5):
    """Scalar Laplacian ``sum_i d²f / dx_i²`` at each point."""
    values = np.trace(hessian(scalar_field, points, h), axis1=-2, axis2=-1)
    array, single = _as_points(points)
    return float(values) if single else values


def lorentz_force(charge, electric, magnetic, velocity):
    """Lorentz force ``q (E + v × B)`` for 3D vectors (broadcasting supported)."""
    return _kernels.lorentz_force(charge, electric, magnetic, velocity)


__all__ = [
    "VectorField",
    "gradient",
    "hessian",
    "laplacian",
    "lorentz_force",
]
