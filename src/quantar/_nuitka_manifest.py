"""Static import manifest for frozen builds.

The public packages resolve most names through module ``__getattr__`` so that
plain ``import quantar`` stays cheap. Freezers do static import analysis and
cannot see those lazy lookups, so a compiled executable needs this manifest::

    python -m PyInstaller --hidden-import=quantar._nuitka_manifest program.py

Every name is imported from its defining module with a normal ``from ... import
...`` statement, which is exactly what a freezer follows. Importing this module
normally works too; it just eagerly loads the public API.

``quantar_kernels`` is imported in a ``try`` block so source checkouts without
a built extension still work. The build wrapper also passes an explicit hidden
import so the extension is bundled when it exists.
"""

from __future__ import annotations

# Language frontend ----------------------------------------------------------
from lang_systems import __version__
from lang_systems.ast_builder import (
    QuantumOp,
    LowerQuantumOps,
    build_ast,
    compile_ast,
    lower_ast,
    parse_ast,
)
from lang_systems.imports import (
    QuantarFinder,
    QuantarLoader,
    import_hook_installed,
    install_import_hook,
    preload_imports,
    uninstall_import_hook,
)
from lang_systems.operators import list_known_operators, load_operators
from lang_systems.quantar import parse
from lang_systems.runtime import apply_quantar_operator
from lang_systems.visitors import OperatorCollector, QuantarNodeVisitor

# Simulator, backend control, and gate matrices ------------------------------
from quantum import backend
from quantum.backend import (
    available_backends,
    build_rust_backend,
    find_cargo,
    find_rust_source_dir,
    get_backend,
    get_backend_module,
    is_free_threaded,
    load_rust_backend,
    reset_backend,
    rust_build_available,
    set_backend,
    use_backend,
)
from quantum.gate_abs import CircuitGate, CompositeGate, Gate
from quantum.gate_matrix import (
    cp_MAT,
    h_MAT,
    i_MAT,
    iqft_MAT,
    iswap_MAT,
    named_gate,
    name_is_controlled,
    p_MAT,
    px_MAT,
    py_MAT,
    pz_MAT,
    qft_MAT,
    rk_MAT,
    rx_MAT,
    ry_MAT,
    rz_MAT,
    s_MAT,
    sdg_MAT,
    swap_MAT,
    swap_sqrt_MAT,
    t_MAT,
    tdg_MAT,
)
from quantum.kernels import (
    HAVE_NUMBA,
    apply_gate_kernel,
    fused_diagonal_kernel,
    sample_counts,
    swap_run_kernel,
)
from quantum.qasm import to_openqasm3
from quantum.seed_set import get_seed, make_pcg64, set_seed
from quantum.composites import crk
from quantum.qubit_abs import Qubit, QuantumComputer

# Concrete Python backend classes (the umbrella package resolves these
# dynamically so backend switches take effect; Nuitka needs the static path).
_PythonQuantumComputer = QuantumComputer
_PythonQubit = Qubit
_PythonGate = Gate

try:  # Nuitka follows this import even when the extension is absent locally.
    from quantum import quantar_kernels
except ImportError:  # pragma: no cover - depends on the build environment
    quantar_kernels = None

# Algorithms -----------------------------------------------------------------
from stl.quantum_algs import (
    Ansatz,
    Hamiltonian,
    QuantumOptimizer,
    factor,
    grovers_alg,
    iqft_gate,
    iqft_program,
    qft_gate,
    qft_program,
    shors_alg,
)
from stl.physics_algs import (
    Q,
    SI,
    Dual,
    Quantity,
    SpaceMetric,
    Trajectory,
    Unit,
    VectorField,
    angular_momentum,
    gradient,
    hessian,
    kinetic_energy,
    laplacian,
    momentum,
    rust_available,
    use_rust_physics,
    velocity_verlet,
)
from stl.physics_algs.kernels import (
    jacobian_from_offsets_numpy,
    kinetic_energy_numpy,
    lorentz_force_numpy,
    verlet_position_numpy,
    verlet_update_numpy,
    verlet_velocity_numpy,
)

__all__ = [
    "__version__",
    "parse",
    "parse_ast",
    "build_ast",
    "lower_ast",
    "compile_ast",
    "QuantumOp",
    "LowerQuantumOps",
    "load_operators",
    "list_known_operators",
    "install_import_hook",
    "uninstall_import_hook",
    "import_hook_installed",
    "preload_imports",
    "QuantarFinder",
    "QuantarLoader",
    "apply_quantar_operator",
    "QuantarNodeVisitor",
    "OperatorCollector",
    "QuantumComputer",
    "Qubit",
    "Gate",
    "_PythonQuantumComputer",
    "_PythonQubit",
    "_PythonGate",
    "CompositeGate",
    "CircuitGate",
    "backend",
    "available_backends",
    "build_rust_backend",
    "find_cargo",
    "find_rust_source_dir",
    "get_backend",
    "get_backend_module",
    "is_free_threaded",
    "load_rust_backend",
    "reset_backend",
    "rust_build_available",
    "set_backend",
    "use_backend",
    "get_seed",
    "set_seed",
    "make_pcg64",
    "h_MAT", "px_MAT", "py_MAT", "pz_MAT", "i_MAT",
    "s_MAT", "sdg_MAT", "t_MAT", "tdg_MAT",
    "swap_MAT", "iswap_MAT", "swap_sqrt_MAT",
    "rx_MAT", "ry_MAT", "rz_MAT", "p_MAT", "cp_MAT", "rk_MAT",
    "qft_MAT", "iqft_MAT", "named_gate", "name_is_controlled",
    "apply_gate_kernel", "fused_diagonal_kernel", "swap_run_kernel",
    "sample_counts", "HAVE_NUMBA", "crk", "to_openqasm3",
    "quantar_kernels",
    "shors_alg", "grovers_alg", "qft_program", "iqft_program",
    "qft_gate", "iqft_gate", "factor", "Ansatz", "Hamiltonian",
    "QuantumOptimizer",
    "Quantity", "Unit", "Q", "SI", "Dual", "VectorField", "SpaceMetric",
    "Trajectory", "gradient", "hessian", "laplacian", "kinetic_energy",
    "momentum", "angular_momentum", "velocity_verlet", "rust_available",
    "use_rust_physics",
    "kinetic_energy_numpy", "lorentz_force_numpy",
    "jacobian_from_offsets_numpy", "verlet_position_numpy",
    "verlet_velocity_numpy", "verlet_update_numpy",
]
