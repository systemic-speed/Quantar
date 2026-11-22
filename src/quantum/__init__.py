"""quantum -- a small, readable state-vector quantum simulator.

Public API::

    from quantum import QuantumComputer, Qubit, Gate, CircuitGate
    from quantum import h_MAT, px_MAT, crk, to_openqasm3

``QuantumComputer``, ``Qubit`` and ``Gate`` come from the active backend, which
is selected with ``use_backend("python")`` or ``use_backend("rust")``.

Internals are grouped by concern (eight files, not a pile of fragments):

    backend       backend selection, validation and local cargo auto-build
    qubit_abs     Python QuantumComputer and Qubit -- the simulator itself
    gate_abs      Gate, CompositeGate, CircuitGate -- how operations are recorded
    gate_matrix   gate matrices and their OpenQASM 3 names
    kernels       numba kernels + numpy fallbacks (the hot loops)
    basic_funcs   bit helpers and the apply entry points wiring kernels to gates
    composites    crk (controlled phase shifts)
    qasm          OpenQASM 3 export
    parse_qasm    OpenQASM 3 import (parse_qasm / load)

QFT/IQFT circuit builders (`qft_program`, `qft_gate`, ...) live in
`stl.quantum_algs.shors_alg`, next to the algorithm that uses them.
"""

from quantum import backend
from quantum.backend import (
    available_backends,
    build_rust_backend,
    find_cargo,
    find_rust_source_dir,
    get_backend,
    get_backend_module,
    load_rust_backend,
    reset_backend,
    rust_build_available,
    set_backend,
    use_backend,
)
from quantum.gate_abs import CompositeGate, CircuitGate
from quantum import gate_matrix
from quantum import kernels
from quantum.gate_matrix import (
    h_MAT, px_MAT, py_MAT, pz_MAT, i_MAT,
    s_MAT, sdg_MAT, t_MAT, tdg_MAT,
    swap_MAT, iswap_MAT, swap_sqrt_MAT,
    rx_MAT, ry_MAT, rz_MAT, p_MAT, cp_MAT, rk_MAT,
    qft_MAT, iqft_MAT,
    named_gate, name_is_controlled,
)
from quantum.kernels import (
    apply_gate_kernel, fused_diagonal_kernel, swap_run_kernel, sample_counts,
    HAVE_NUMBA,
)
from quantum.composites import crk
from quantum.qasm import to_openqasm3

_BACKEND_EXPORTS = frozenset({"QuantumComputer", "Qubit", "Gate"})


def __getattr__(name):
    """Resolve backend-dependent classes on every access so switches take effect."""
    if name in _BACKEND_EXPORTS:
        return backend.backend_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_BACKEND_EXPORTS) | set(__all__))


__all__ = [
    "QuantumComputer",
    "Qubit",
    "Gate",
    "CompositeGate",
    "CircuitGate",
    "backend",
    "available_backends",
    "build_rust_backend",
    "find_cargo",
    "find_rust_source_dir",
    "get_backend",
    "get_backend_module",
    "load_rust_backend",
    "reset_backend",
    "rust_build_available",
    "set_backend",
    "use_backend",
    "gate_matrix",
    "kernels",
    "h_MAT", "px_MAT", "py_MAT", "pz_MAT", "i_MAT",
    "s_MAT", "sdg_MAT", "t_MAT", "tdg_MAT",
    "swap_MAT", "iswap_MAT", "swap_sqrt_MAT",
    "rx_MAT", "ry_MAT", "rz_MAT", "p_MAT", "cp_MAT", "rk_MAT",
    "qft_MAT", "iqft_MAT",
    "named_gate", "name_is_controlled",
    "apply_gate_kernel", "fused_diagonal_kernel", "swap_run_kernel",
    "sample_counts", "HAVE_NUMBA",
    "crk",
    "to_openqasm3",
]
