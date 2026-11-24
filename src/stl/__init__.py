"""Quantar's standard library of algorithms.

The two subpackages are :mod:`stl.quantum_algs` and :mod:`stl.physics_algs`.
Common entry points are re-exported here so both of these work::

    from stl import quantum_algs
    from stl import shors_alg, grovers_alg, Quantity

Subpackages and names are loaded lazily, so importing ``stl`` stays cheap.
"""

from importlib import import_module

_SUBPACKAGES = ("quantum_algs", "physics_algs")

_EXPORTS = {
    # Quantum algorithms
    "shors_alg": ("stl.quantum_algs", "shors_alg"),
    "grovers_alg": ("stl.quantum_algs", "grovers_alg"),
    "qft_program": ("stl.quantum_algs", "qft_program"),
    "iqft_program": ("stl.quantum_algs", "iqft_program"),
    "qft_gate": ("stl.quantum_algs", "qft_gate"),
    "iqft_gate": ("stl.quantum_algs", "iqft_gate"),
    "factor": ("stl.quantum_algs", "factor"),
    "Ansatz": ("stl.quantum_algs", "Ansatz"),
    "Hamiltonian": ("stl.quantum_algs", "Hamiltonian"),
    "QuantumOptimizer": ("stl.quantum_algs", "QuantumOptimizer"),
    # Physics helpers
    "Quantity": ("stl.physics_algs", "Quantity"),
    "Unit": ("stl.physics_algs", "Unit"),
    "Q": ("stl.physics_algs", "Q"),
    "SI": ("stl.physics_algs", "SI"),
    "Dual": ("stl.physics_algs", "Dual"),
    "VectorField": ("stl.physics_algs", "VectorField"),
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        value = import_module(f"{__name__}.{name}")
    else:
        target = _EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_name, attribute = target
        value = getattr(import_module(module_name), attribute)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [*_SUBPACKAGES, *_EXPORTS]
