"""Shared deterministic seeding for both simulator backends.

``quantum.set_seed(seed)`` is the public entry point. The seed lives here so
the Python simulator, the Rust simulator, measurement, and sampling all draw
from the same source of truth. On first use a seed is generated from OS entropy
and remembered, so ``quantum.get_seed()`` always returns the value that can
reproduce the current run.

The PCG64 setup mirrors Rust's :func:`quantar_kernels.make_pcg64` helper: the
seed is hashed with SHA-256 into a 128-bit state and an odd increment.
"""

from __future__ import annotations

import hashlib
import operator
import threading
from os import urandom

from numpy import random

__all__ = ["get_rng", "get_seed", "make_pcg64", "set_seed"]

_MAX_SEED = (1 << 64) - 1
_SEED: int | None = None
_rng: random.Generator | None = None
_lock = threading.RLock()


def _coerce_seed(seed: int) -> int:
    """Normalize a seed to an unsigned 64-bit Python int."""
    try:
        value = operator.index(seed)
    except TypeError as error:
        raise TypeError(
            f"seed must be an integer in [0, {_MAX_SEED}], got {type(seed).__name__}"
        ) from error
    if not 0 <= value <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {value}")
    return value


def make_pcg64(seed: int) -> random.Generator:
    """Build a NumPy PCG64 generator from the same seed Rust uses."""
    value = _coerce_seed(seed)
    digest = hashlib.sha256(value.to_bytes(8, "little", signed=False)).digest()
    state = int.from_bytes(digest[:16], "little")
    inc = int.from_bytes(digest[16:], "little") | 1

    bitgen = random.PCG64()
    bitgen.state = {
        "bit_generator": "PCG64",
        "state": {"state": state, "inc": inc},
        "has_uint32": 0,
        "uinteger": 0,
    }
    return random.Generator(bitgen)


def _ensure_locked() -> tuple[int, random.Generator]:
    global _SEED, _rng
    if _SEED is None or _rng is None:
        _SEED = int.from_bytes(urandom(8), "little")
        _rng = make_pcg64(_SEED)
    return _SEED, _rng


def get_seed() -> int:
    """Return the active seed, generating and remembering one on first use."""
    with _lock:
        seed, _ = _ensure_locked()
        return seed


def set_seed(seed: int) -> int:
    """Reset every future sample and measurement to a reproducible stream.

    Existing ``QuantumComputer`` objects keep their own generator; the new seed
    applies to computers and samples created after this call.
    """
    global _SEED, _rng
    value = _coerce_seed(seed)
    with _lock:
        _SEED = value
        _rng = make_pcg64(value)
        return value


def get_rng() -> random.Generator:
    """Return the active NumPy generator, seeding it from entropy if needed."""
    with _lock:
        _, generator = _ensure_locked()
        return generator
