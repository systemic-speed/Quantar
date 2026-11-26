# Quantar API guide

Version 1.2.3 (current release). This page is the
quick map of "where do I import X from?" and documents the public behavior that
is easy to miss.

## Import styles

All three styles are supported. Pick whichever reads best in your project:

```python
# Umbrella package: the common names, loaded lazily
import quantar

qc = quantar.QuantumComputer(2)
quantar.set_seed(1234)
counts = quantar.grovers_alg(qc, oracle)

# Direct subpackages: the implementation modules
import quantum
from stl.quantum_algs import shors_alg, qft_gate
from stl.physics_algs import Quantity, Unit, SI

# Explicit modules: when you want the source of truth
from quantum.seed_set import get_rng
from stl.quantum_algs.shors_alg_lib import qft_program
from lang_systems import parse_ast
```

`import quantar` does not import the Rust extension or compile the parser. Names
are resolved on first access.

## Public exports

### `quantar`

| Area | Names |
| --- | --- |
| Language | `parse`, `parse_ast`, `build_ast`, `lower_ast`, `compile_ast`, `QuantumOp`, `load_operators`, `install_import_hook`, `preload_imports` |
| Simulator | `QuantumComputer`, `Qubit`, `Gate`, `CompositeGate`, `CircuitGate`, `named_gate`, `crk`, `to_openqasm3` |
| Backend | `use_backend`, `set_backend`, `get_backend`, `available_backends`, `load_rust_backend`, `build_rust_backend`, `is_free_threaded` |
| Seeding | `set_seed`, `get_seed`, `make_pcg64` |
| Algorithms | `grovers_alg`, `shors_alg`, `qft_gate`, `iqft_gate`, `factor`, `Ansatz`, `Hamiltonian`, `QuantumOptimizer` |
| Physics | `Quantity`, `Unit`, `Q`, `SI`, `Dual`, `VectorField`, `SpaceMetric`, `Trajectory`, `gradient`, `hessian`, `laplacian`, `momentum`, `angular_momentum`, `rust_available`, `use_rust_physics` |
| Modules | `lang_systems`, `quantum`, `stl`, `quantum_algs`, `physics_algs` |

The full list is always available at runtime:

```python
import quantar
print(quantar.__all__)
```

### `quantum`

```python
import quantum

quantum.QuantumComputer, quantum.Qubit, quantum.Gate  # active backend
quantum.use_backend("python")  # pure NumPy/Numba simulator
quantum.use_backend("rust")    # compiled quantar_kernels extension

quantum.set_seed(7)
quantum.get_seed()             # 7
quantum.seed_set               # shared seed/RNG module
```

Additional gate matrices (`h_MAT`, `px_MAT`, `rz_MAT`, `qft_MAT`, ...), kernels
(`apply_gate_kernel`, `sample_counts`, ...), and `is_free_threaded()` are also
exported here. See `quantum.__all__`.

### `stl` and its subpackages

```python
from stl import shors_alg, grovers_alg, Quantity
from stl import quantum_algs, physics_algs

from stl.quantum_algs import (
    shors_alg, shors_alg_optim, qft_program, iqft_gate,
    grovers_alg, grover_oracle, grover_diffuser,
    factor, is_probable_prime, Ansatz, Hamiltonian, QuantumOptimizer,
)
from stl.physics_algs import (
    Quantity, Unit, SI, Q, Dual, VectorField, SpaceMetric, Trajectory,
    gradient, hessian, laplacian, momentum, angular_momentum,
    rust_available, use_rust_physics,
)
```

The Shor and Grover functions moved into `shors_alg_lib.py` and
`grovers_alg_lib.py` in 1.2.1. Import the functions from `stl.quantum_algs`
instead of the old module paths; the function names did not change.

### Physics helpers

The physics side is meant to remove bookkeeping: units carry their dimensions,
`Dual` propagates derivatives through ordinary arithmetic, and `VectorField`
evaluates and differentiates fields on points. Everything below runs on plain
Python callables; the numeric reductions use Rust when the extension is present
and NumPy otherwise.

```python
import numpy as np
from stl.physics_algs import (
    Q, Dual, VectorField, gradient, hessian, laplacian,
    momentum, angular_momentum, velocity_verlet, SpaceMetric,
)

# Units check dimensions and convert on request.
distance = Q(12, "km").to("m")       # 12000 m
speed = Q(36, "km/h").to("m/s")      # 10 m/s
travel_time = distance / speed       # 1200 s

# Forward-mode automatic differentiation, no symbolic algebra needed.
x = Dual(3.0, [1.0])                 # value 3, dx/dx = 1
y = x * x + 2 * x
print(y.real, y.dual.tolist())       # 15.0 [8.0]
print(Dual.value_and_gradient(lambda xs: xs[0] ** 2, [3.0]))
# (9.0, array([6.0]))
print(Dual.jacobian(lambda xs: [xs[0] ** 2, xs[0] * xs[1]], [2.0, 3.0]))
# [[4. 0.]
#  [3. 2.]]

# Fields work on one point or a batch of points.
field = VectorField(lambda p: [-p[1], p[0]])
print(field([1.0, 2.0]))                         # [-2.  1.]
print(round(float(field.divergence([1.0, 2.0])), 6))  # 0.0
print(round(float(field.curl([1.0, 2.0])), 6))   # 2.0
print(field.jacobian([[1.0, 2.0], [3.0, 4.0]]).shape)  # (2, 2, 2)
print(np.round(field.hessian([1.0, 2.0]), 4))    # two zero 2x2 blocks
print(field.laplacian([1.0, 2.0]))               # [0. 0.]

# Scalar derivatives use the same finite-difference machinery.
scalar = lambda p: p[0] ** 2 + p[1] ** 3
print(gradient(scalar, [2.0, 3.0]))              # [ 4. 27.]
print(np.round(hessian(scalar, [2.0, 3.0]), 3))  # [[2. 0.] [0. 18.]]
print(round(float(laplacian(scalar, [2.0, 3.0])), 3))  # 20.0

# Dynamics: trajectory summaries and momentum helpers.
orbit = velocity_verlet([1.0, 0.0], [0.0, 1.0], 0.01, 628, lambda point: -point)
print(len(orbit), round(float(orbit.speeds()[-1]), 4))       # 629 1.0
print(round(float(orbit.kinetic_energies()[-1]), 4))         # 0.5
print(momentum([3.0, 4.0], 2.0))                            # [6. 8.]
print(angular_momentum([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))   # [0. 0. 1.]
print(round(SpaceMetric("spherical").volume_element([2.0, np.pi / 2, 0.0]), 6))
# 4.0
```

`Q(value, "unit")` is the quick constructor; `Quantity(value, unit_object)` is
the explicit form. `velocity_verlet` returns a `Trajectory` with positions,
velocities and times, plus `speeds()`, `momenta()`, `angular_momentum()` and
`energy(potential)` for analysis. `SpaceMetric` supplies scale factors for
cartesian, spherical and cylindrical coordinates, including `volume_element()`
for integration weights.

#### Rust-backed physics kernels

Four hot paths use compiled kernels in `quantum.quantar_kernels`: kinetic-energy
sums, Lorentz forces, central-difference reductions (Jacobians, gradients,
Hessians), and the two velocity-Verlet half-steps. They release the GIL and
parallelise with rayon for large batches. Your own field/acceleration functions
remain ordinary Python callables; only the numeric work is moved.

```python
from stl.physics_algs import rust_available, use_rust_physics

if rust_available():
    print("compiled physics kernels active")
use_rust_physics(False)   # force the NumPy implementations
print(use_rust_physics()) # False (or True after re-enabling)
```

Setting `QUANTAR_PHYSICS_KERNELS=0` before import does the same thing without
code changes. The loader checks the installed extension for the required
functions and falls back silently if it is missing or older than 1.2.3; it does
not trigger a Rust build for physics-only programs. Direct access to the
kernels, including their NumPy twins, is in `stl.physics_algs.kernels`.

### `lang_systems`

```python
from lang_systems import (
    parse, parse_ast, QuantumOp, OperatorCollector,
    load_operators, install_import_hook, preload_imports,
)
```

`qtr program.qtr` runs a Quantar program, and `qtr program.qtr --compile out.py`
compiles it to ordinary Python. During development use
`python -m lang_systems.runner program.qtr`.

## Deterministic sampling

`set_seed` is shared by both backends. NumPy sampling, qubit measurement, Pauli
expectations, and Rust per-computer RNGs all derive from the same value:

```python
import quantum

quantum.set_seed(1234)
qc = quantum.QuantumComputer(3)
first = qc.sample_parallel(circuit, 1000)

quantum.set_seed(1234)
qc = quantum.QuantumComputer(3)
second = qc.sample_parallel(circuit, 1000)

assert first == second
```

Details:

- `quantum.get_seed()` generates and remembers a seed from OS entropy on first
  use, so a run can always be reproduced afterward.
- The new seed applies to computers and samples created after `set_seed`;
  existing `QuantumComputer` objects keep their own generator.
- The Rust extension exposes the same functions as
  `quantum.quantar_kernels.set_seed()` / `get_seed()`; they update the same
  Python-side state, so switching backends does not require reseeding twice.

## Backends and free-threaded Python

| Interpreter | Source build | Wheel tag |
| --- | --- | --- |
| CPython 3.12–3.14, GIL enabled | `abi3` (`abi3-py312`) | `cp312-abi3-*` |
| CPython 3.14t | version-specific `cp314t` | `cp314t-*` |
| CPython 3.15/3.15t+ | `abi3t` (`abi3t-py315`) | `abi3t-*` |

```powershell
uv python install 3.14t
uv venv --python 3.14t .venv-3.14t
.\.venv-3.14t\Scripts\python.exe -m pip install .
.\.venv-3.14t\Scripts\python.exe -c "import quantum; print(quantum.is_free_threaded()); quantum.use_backend('rust')"
```

A GIL-enabled `cp312-abi3` wheel is rejected by pip on free-threaded CPython.
Install the matching free-threaded wheel or install from source. The Rust module
declares `gil_used = false` and uses NumPy borrow guards plus `py.detach`, so
importing it does not re-enable the GIL.

If the extension is missing or stale and `cargo` plus `rust/` are available,
Quantar rebuilds it automatically. Overrides:

- `QUANTAR_BACKEND=rust` selects the backend at startup.
- `QUANTAR_RUST_AUTO_BUILD=0` disables automatic builds.
- `QUANTAR_RUST_SOURCE_DIR` points at the repo root or `rust/`.
- `QUANTAR_CARGO` selects a specific cargo executable.
- `QUANTAR_RUST_BACKEND_PATH` loads a specific `.pyd`/`.so`.
- `QUANTAR_RUST_BUILD_CACHE` changes the fallback output directory.

## Build output

Normal `pip install .`, `python -m build`, and `uv build` runs are quiet. Rust
dependency deprecation warnings (for example the `core::f32::NAN` warnings from
`num-traits 0.2.19`) are suppressed because they come from dependencies.

```powershell
$env:QUANTAR_VERBOSE_BUILD = "1"   # equivalent to HATCH_MATURIN_ARGS="-vv"
uv build
Remove-Item Env:QUANTAR_VERBOSE_BUILD

$env:QUANTAR_STRICT_RUST = "1"     # keep dependency warnings visible
uv build
```

For release artifacts, build all wheels plus the source archive:

```powershell
python -m build --outdir dist
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip wheel . --no-deps --wheel-dir dist  # with a 3.15t interpreter
python -m twine check dist/*
python -m twine upload dist/*
```

PyPI/pip selects the correct wheel; uploading only `tar.gz` forces every user
to compile the Rust extension from source.

## Where things live

| Path | Contents |
| --- | --- |
| `src/lang_systems/` | parser, AST, operators, import hook, LSP, `qtr` runner |
| `src/quantum/` | simulators, gate matrices/kernels, QASM, backends, seeding |
| `src/stl/quantum_algs/` | Shor, Grover, variational optimization, factorization |
| `src/stl/physics_algs/` | units, dual numbers, vector fields, trajectories |
| `rust/` | PyO3/rayon kernels and backend bindings |
| `docs/API.md` | this guide |
| `roadmap.md` | shipped and planned work |

See [roadmap.md](../roadmap.md) for planned work and known limitations.
