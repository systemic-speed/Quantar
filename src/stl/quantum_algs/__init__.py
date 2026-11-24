"""Quantar's quantum algorithms.

Everything is importable from the package root, so callers can write::

    from stl.quantum_algs import shors_alg, grovers_alg
    from stl.quantum_algs import QuantumOptimizer, Ansatz, Hamiltonian
    from stl.quantum_algs import qft_gate, iqft_gate, factor

Modules:
    shors_alg_lib    Shor's factoring (incl. QFT/IQFT programs and gates)
    grovers_alg_lib  Grover's search (oracle + diffuser)
    quantum_optim    Ansatz, Hamiltonian, Pauli expectations
    quantum_optim_hl QuantumOptimizer and penalty helpers
    fast_factor      classical factorization utilities

The old ``shors_alg`` / ``grovers_alg`` module names were replaced by the
``*_lib`` modules in 1.2.1; the function names above are unchanged.
"""

from stl.quantum_algs.shors_alg_lib import (
    shors_alg,
    shors_alg_optim,
    quantum_subroutine,
    mod_exp,
    qft_program,
    iqft_program,
    qft_circuit,
    iqft_circuit,
    qft_gate,
    iqft_gate,
    is_prime_fast,
    is_perfect_power,
)
from stl.quantum_algs.grovers_alg_lib import (
    grovers_alg,
    grover_oracle,
    grover_diffuser,
    optimal_iterations,
)
from stl.quantum_algs.quantum_optim import (
    Ansatz,
    Hamiltonian,
    hardware_efficient_ansatz,
    measure_pauli_expectation,
    pauli_string_to_matrices,
    pauli_string_to_tensor,
)
from stl.quantum_algs.quantum_optim_hl import (
    QuantumOptimizer,
    OptimizeResult,
    Penalty,
    penalized_cost,
    minimize_cost,
)
from stl.quantum_algs.grovers_alg_lib import demo as grovers_demo
from stl.quantum_algs.fast_factor import (
    factor,
    factor_pair,
    is_prime,
    is_probable_prime,
    trial_division,
    pollards_rho,
    pollards_p1,
    factorize,
)

__all__ = [
    # Shor's
    "shors_alg", "shors_alg_optim", "quantum_subroutine", "mod_exp",
    "qft_program", "iqft_program", "qft_circuit", "iqft_circuit",
    "qft_gate", "iqft_gate", "is_prime_fast", "is_perfect_power",
    # Grover's
    "grovers_alg", "grover_oracle", "grover_diffuser", "optimal_iterations",
    "grovers_demo",
    # Optimization
    "Ansatz", "Hamiltonian", "hardware_efficient_ansatz",
    "measure_pauli_expectation", "pauli_string_to_matrices",
    "pauli_string_to_tensor",
    "QuantumOptimizer", "OptimizeResult", "Penalty",
    "penalized_cost", "minimize_cost",
    # Classical helpers
    "factor", "factor_pair", "is_prime", "is_probable_prime",
    "trial_division", "pollards_rho", "pollards_p1", "factorize",
    # Modules (useful for direct access)
    "shors_alg_lib", "grovers_alg_lib", "quantum_optim",
    "quantum_optim_hl", "fast_factor",
]
