"""Quantar umbrella package.

Import ``quantar`` for the most useful names without caring which subpackage
owns them. Everything is resolved lazily, so the import itself stays light::

    import quantar

    qc = quantar.QuantumComputer(2)
    quantar.use_backend("rust")
    quantar.set_seed(1234)
    counts = quantar.grovers_alg(qc, oracle)

The implementation lives in :mod:`lang_systems`, :mod:`quantum`, and :mod:`stl`;
those packages can always be imported directly as well.
"""

from importlib import import_module

from lang_systems import __version__

_SUBPACKAGES = (
    "lang_systems",
    "quantum",
    "stl",
    "stl.quantum_algs",
    "stl.physics_algs",
)

_EXPORTS = {
    # Language frontend
    "parse": ("lang_systems", "parse"),
    "parse_ast": ("lang_systems", "parse_ast"),
    "build_ast": ("lang_systems", "build_ast"),
    "lower_ast": ("lang_systems", "lower_ast"),
    "compile_ast": ("lang_systems", "compile_ast"),
    "QuantumOp": ("lang_systems", "QuantumOp"),
    "load_operators": ("lang_systems", "load_operators"),
    "list_known_operators": ("lang_systems", "list_known_operators"),
    "install_import_hook": ("lang_systems", "install_import_hook"),
    "preload_imports": ("lang_systems", "preload_imports"),
    # Simulator and backend control
    "QuantumComputer": ("quantum", "QuantumComputer"),
    "Qubit": ("quantum", "Qubit"),
    "Gate": ("quantum", "Gate"),
    "backend": ("quantum", "backend"),
    "get_backend": ("quantum", "get_backend"),
    "set_backend": ("quantum", "set_backend"),
    "use_backend": ("quantum", "use_backend"),
    "available_backends": ("quantum", "available_backends"),
    "build_rust_backend": ("quantum", "build_rust_backend"),
    "find_cargo": ("quantum", "find_cargo"),
    "find_rust_source_dir": ("quantum", "find_rust_source_dir"),
    "get_backend_module": ("quantum", "get_backend_module"),
    "is_free_threaded": ("quantum", "is_free_threaded"),
    "load_rust_backend": ("quantum", "load_rust_backend"),
    "reset_backend": ("quantum", "reset_backend"),
    "rust_build_available": ("quantum", "rust_build_available"),
    "set_seed": ("quantum", "set_seed"),
    "get_seed": ("quantum", "get_seed"),
    "make_pcg64": ("quantum", "make_pcg64"),
    "CompositeGate": ("quantum", "CompositeGate"),
    "CircuitGate": ("quantum", "CircuitGate"),
    "named_gate": ("quantum", "named_gate"),
    "to_openqasm3": ("quantum", "to_openqasm3"),
    "crk": ("quantum", "crk"),
    # Algorithms
    "quantum_algs": ("stl", "quantum_algs"),
    "physics_algs": ("stl", "physics_algs"),
    "shors_alg": ("stl.quantum_algs", "shors_alg"),
    "grovers_alg": ("stl.quantum_algs", "grovers_alg"),
    "qft_gate": ("stl.quantum_algs", "qft_gate"),
    "iqft_gate": ("stl.quantum_algs", "iqft_gate"),
    "factor": ("stl.quantum_algs", "factor"),
    "Ansatz": ("stl.quantum_algs", "Ansatz"),
    "Hamiltonian": ("stl.quantum_algs", "Hamiltonian"),
    "QuantumOptimizer": ("stl.quantum_algs", "QuantumOptimizer"),
    # Physics
    "Quantity": ("stl.physics_algs", "Quantity"),
    "Unit": ("stl.physics_algs", "Unit"),
    "Q": ("stl.physics_algs", "Q"),
    "SI": ("stl.physics_algs", "SI"),
    "Dual": ("stl.physics_algs", "Dual"),
    "VectorField": ("stl.physics_algs", "VectorField"),
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        value = import_module(name)
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


__all__ = ["__version__", *_SUBPACKAGES, *_EXPORTS]
