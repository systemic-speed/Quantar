# Quantar

![Quantar Demo]()

Quantar is a strict superset of Python for quantum computing and physics
simulations. It keeps Python syntax and adds custom infix operators such as
`a <-> b`, plus a library of quantum gates, circuits, algorithms, and physics
helpers.

## Requirements

- Python 3.12 or newer
- Rust (to install it, click [here](https://rust-lang.org/tools/install/) and follow the commands)
- NumPy, Numba, and `openqasm3[parser]` (installed automatically)

## Installation

```bash
pip install quantar

# or with uv
uv pip install quantar
```

Optional extras:

```bash
# alternative quantum backends
pip install quantar[backends]

# SciPy-powered COBYLA optimization
pip install quantar[optimization]

# LSP support
pip install quantar[lsp]
```

also install the VS code extension [here](https://marketplace.visualstudio.com/items?itemName=Gauthamrgb.quantar-lang) for better coloring and errors or just use the `ext` command as follows:
```bash
ext install Gauthamrgb.quantar-lang
```
## Running Quantar

```bash
qtr path/to/program.qtr
```

Compile a file to ordinary Python instead of executing it:

```bash
qtr path/to/program.qtr --compile path/to/program.py
```

During development, the same runner is available as a module:

```bash
python -m lang_systems.runner path/to/program.qtr
```

Small registers use vectorized NumPy so short programs skip Numba's startup and
compilation cost. Registers with at least 1,048,576 amplitudes use the JIT. Set
`QUANTAR_NUMBA_MIN_AMPLITUDES` to change that cutoff, or `QUANTAR_NUMBA=0` to
disable Numba entirely.

## VS Code extension

The extension lives in `extension/`. It provides Python-based syntax
highlighting for `.qtr` and `.quantar` files, parser diagnostics, and semantic
highlighting for registered Quantar operators. Operators use one consistent
cyan accent; the rest of the syntax follows the active VS Code theme.

```bash
python -m pip install -e ".[lsp]"
cd extension
npm install
```

Open the project root in VS Code and press F5 to launch the `Run Quantar
Extension` development configuration. The extension uses the Python
interpreter selected by the VS Code Python extension. Install the LSP extra in
that interpreter; for a local checkout, run the install command above from the
project root.

## Parsing Quantar

`lang_systems.parse()` returns pegen's raw parse tree. This is the low-level
form used by tooling and tests:

```python
from lang_systems import parse

tree = parse("a <-> b\nx = 1 + 2\n")
```

`lang_systems.parse_ast()` returns a normal Python AST with custom `QuantumOp`
nodes:

```python
import ast

from lang_systems import OperatorCollector, parse_ast

module = parse_ast("x = a <-> b\n")
print(ast.dump(module, indent=2))

collector = OperatorCollector()
collector.visit(module)
print(collector.operators)  # [("<->", "binary")]
```

Custom operators are registered with `lang_systems.operators.operator`:

```python
from lang_systems.operators import operator

@operator.binary("><")
def reverse(left, right):
    return right, left
```

If the operators live in another module, import that module before parsing:

```python
from lang_systems import load_operators, parse

load_operators("my_package.quantar_ops")
tree = parse("a >< b\n")
```

## Quantar as a library

`quantar` is the umbrella package. It loads the language, quantum, and physics
packages lazily, so importing it stays lightweight:

```python
import quantar

q = quantar.QuantumComputer(2)
tree = quantar.parse_ast("a <-> b\n")
print(quantar.__version__)

quantar.lang_systems
quantar.quantum
quantar.stl
```

Common algorithms and physics helpers are re-exported from `quantar` and `stl`
too:

```python
import quantar
from stl import shors_alg, grovers_alg, Quantity

# The implementation modules are still available directly:
from stl.quantum_algs.shors_alg_lib import shors_alg as shors_alg_impl
from stl.physics_algs.units import Quantity as QuantityImpl
```

Physics helpers share the same shortcut: `quantar.VectorField`,
`quantar.momentum`, and `quantar.angular_momentum` resolve lazily. Numeric hot
paths in `stl.physics_algs` (kinetic energy, Lorentz forces, finite differences,
and Verlet steps) use compiled Rust kernels when available and identical NumPy
fallbacks otherwise. Check with `stl.physics_algs.rust_available()`.

See [`docs/API.md`](docs/API.md) for the full import cheat sheet, public API,
and backend/seed details.

### Selecting a simulator backend

`QuantumComputer`, `Qubit`, and `Gate` follow the active backend. The default is
the pure-Python simulator; switch to the compiled Rust backend at runtime when
the extension is available:

```python
import quantum

print(quantum.get_backend())        # "python"
quantum.use_backend("rust")         # or "python" to switch back
qc = quantum.QuantumComputer(2)     # quantum.quantar_kernels.QuantumComputer

quantum.load_rust_backend()         # import/validate the extension directly
quantum.available_backends()        # e.g. ("python", "rust")
```

The backend can also be chosen before startup with `QUANTAR_BACKEND=rust`. The
bundled extension is built with PyO3's `abi3-py312` feature, so the same binary
works on CPython 3.12+ and wheels are tagged `cp312-abi3-*`. In the source tree
it is named `src/quantum/quantar_kernels.pyd`, with a copy kept in the root
`kernels/` directory.

If that extension is missing or stale, Quantar automatically builds it locally
when `cargo` and the `rust/` source tree are available:

```python
import quantum

quantum.rust_build_available()   # True when cargo + rust/Cargo.toml are found
quantum.build_rust_backend()     # force a fresh release build and load it
quantum.use_backend("rust")      # auto-builds if needed
```

Environment overrides:

- `QUANTAR_RUST_AUTO_BUILD=0` disables automatic builds.
- `QUANTAR_RUST_SOURCE_DIR` points at the repo root or its `rust/` directory.
- `QUANTAR_CARGO` selects a specific cargo executable.
- `QUANTAR_RUST_BACKEND_PATH` loads a specific `.pyd`/`.so` instead.
- `QUANTAR_RUST_BUILD_CACHE` changes the fallback output directory when the
  installed package directory is not writable.

### Reproducible sampling

Both backends share one seed. `set_seed` affects measurement, `sample`,
`sample_parallel`, `measure_pauli`, and the Rust per-computer RNG:

```python
import quantum

quantum.set_seed(1234)
print(quantum.get_seed())          # 1234
qc = quantum.QuantumComputer(3)

# Recreate the same stream later:
quantum.set_seed(1234)
qc = quantum.QuantumComputer(3)
```

`quantum.get_seed()` generates and remembers a seed from OS entropy on first
use, so any run can be reproduced afterward. The new seed applies to computers
and samples created after the call; existing computers keep their generator.
The compiled extension exposes `quantum.quantar_kernels.set_seed()` /
`get_seed()` as aliases of the same shared state.

### Build output

Normal `uv build`, `python -m build`, and `pip install .` runs are quiet.
Dependency deprecation warnings (for example `num-traits 0.2.19`'s deprecated
`core::f32::NAN` constants) are suppressed because they come from dependencies:

```powershell
$env:QUANTAR_VERBOSE_BUILD = "1"   # detailed maturin/cargo output
uv build
Remove-Item Env:QUANTAR_VERBOSE_BUILD

$env:QUANTAR_STRICT_RUST = "1"     # keep dependency warnings visible
uv build
```

### Building the Rust backend manually

```powershell
$env:PYO3_PYTHON = "C:\path\to\your\python.exe"
cargo build --release --manifest-path rust/Cargo.toml

Copy-Item rust\target\release\quantar_kernels.dll `
  src\quantum\quantar_kernels.pyd -Force
Copy-Item rust\target\release\quantar_kernels.dll `
  kernels\quantar_kernels.pyd -Force
```

For distribution, prefer wheels over copying `.pyd` files by hand. The
GIL-enabled wheel is tagged `cp312-abi3-*`; free-threaded builds need their own
wheels as described below.

### Free-threaded Python

Quantar can build and load the Rust backend on free-threaded CPython. The ABI
matrix is:

| Interpreter | Extension built from source | Prebuilt wheel to publish |
| --- | --- | --- |
| CPython 3.12–3.14, GIL enabled | `abi3` (`abi3-py312`) | `cp312-abi3-*` |
| CPython 3.14t, free-threaded | version-specific `cp314t` | `cp314t-*` |
| CPython 3.15/3.15t and newer | `abi3t` (`abi3t-py315`) | `abi3t-*` |

Important details:

- The existing `cp312-abi3-*` wheel **cannot** be installed on a free-threaded
  interpreter; pip rejects it as unsupported. Install a free-threaded wheel or
  install from source with the free-threaded interpreter.
- `python3.14t -m pip install .` compiles a `cp314t` extension automatically.
- Python 3.15+ uses PEP 803's `abi3t` stable ABI, so one binary works with both
  GIL-enabled and free-threaded interpreters.
- PyO3 0.30 no longer supports CPython 3.13t; use 3.14t or newer.
- The runtime loader detects `Py_GIL_DISABLED`, so local auto-builds never write
  a GIL-only `abi3` extension for a free-threaded interpreter.

```powershell
uv python install 3.14t
uv venv --python 3.14t .venv-3.14t
.\.venv-3.14t\Scripts\python.exe -m pip install . --no-deps
.\.venv-3.14t\Scripts\python.exe -c "import quantum; print(quantum.is_free_threaded()); quantum.use_backend('rust')"
```

The extension declares `gil_used = false`: NumPy access uses borrow guards and
the heavy kernels run after `py.detach`, so importing it does not force the
interpreter to re-enable the GIL.

### Imports in Quantar programs

Regular Python imports work inside `.qtr` files, and Quantar can import other
`.qtr` modules too:

```python
# gates.qtr
from lang_systems.operators import operator

@operator.binary("<~>")
def reverse(left, right):
    return right, left

VALUE = 42
```

```python
# main.qtr
from gates import VALUE
import gates

a, b = 1, 2
print(a <~> b)   # operator is registered before parsing
print(VALUE)
```

Run it with `qtr main.qtr`. Directory packages with `__init__.qtr` support
relative imports such as `from .gates import VALUE`. The runner adds the source
file's directory to `sys.path`, so sibling `.py` and `.qtr` modules are found
without extra setup. Compiled `.py` output installs the same import hook.

See the modules under `src/quantum/` and `src/stl/` for the available gates,
circuits, algorithms, and physics helpers.

## Development

The Python packages live under `src/`. Install the project in editable mode
before running the CLI or LSP from a checkout:

```bash
pip install -e ".[lsp]"
```

Regenerate the parser after changing `src/lang_systems/quantar.gram`:

```bash
python -m lang_systems.pegen_gram_gen
```

Build the wheel and source distribution:

```bash
uv build
```

The generated parser is `src/lang_systems/quantar_parser.py`; the build hook in
`hatch_build.py` regenerates it automatically. The parser is generated from
CPython's `Grammar/python.gram` plus the operator overlay in
`src/lang_systems/quantar.gram`.

## Project maturity

Quantar is usable today as a Python superset for quantum programs, but it is
still a maturing project. The core language frontend, simulator, Rust backend,
and CLI are in place; tooling, platform coverage, and the compiled execution
pipeline are still being expanded. See [roadmap.md](roadmap.md) for the planned
work.

### What is stable enough to use

- **Language frontend:** the parser and AST builder handle Python syntax plus
  Quantar's custom infix operators.
- **Imports:** `.py` imports and `.qtr` module imports work, including operator
  preloading, sibling modules, and relative imports inside packages.
- **Simulation:** the pure-Python/NumPy backend is the default and supports the
  main state-vector operations, circuits, QFT helpers, measurement, sampling,
  and QASM export.
- **Rust backend:** the optional compiled backend mirrors the Python API. It
  uses `abi3-py312` on GIL-enabled CPython and can build for free-threaded
  interpreters (`cp314t` or `abi3t-py315`). If the extension is missing or
  stale, Quantar can build it locally when `cargo` and the Rust source tree are
  available.
- **CLI:** `qtr` can run or compile `.qtr` files.
- **LSP basics:** the VS Code extension provides syntax highlighting, operator
  highlighting, and parser diagnostics.

### What is still evolving

- The Rust backend API is additive and tested for parity, but its surface is
  still growing.
- Distribution currently produces a Windows `cp312-abi3` wheel from this
  checkout; Linux/macOS wheels and prebuilt free-threaded wheels are planned.
  Free-threaded users can install from source today.
- The `.qtr` import hook works for common cases, but namespace packages,
  reloads, and circular-import diagnostics are still being refined.
- Error messages and source locations are being improved across the parser,
  runner, and LSP.
- Performance work is ongoing, especially for multi-controlled gates, common
  1-qubit/2-qubit kernels, and Python-to-Rust callback reduction.

### What is not implemented yet

- A dedicated Quantar bytecode interpreter.
- A debugger with breakpoints, stepping, and state inspection.
- GPU, distributed, or tensor-network execution backends.
- Full LSP support for completions, hover documentation, and cross-file
  go-to-definition.

In short: Quantar is ready for experimentation, teaching, prototyping, and
library use on supported platforms. It is not yet a finished production
toolchain, and the roadmap tracks the work needed to get there.

## Feedback

Feedback and bug reports are welcome through GitHub issues or by contacting the
maintainer. The main contributor is
[Gautham_rgb](https://github.com/systemic-speed/).

## Notes on AI

Most of the code was written by people. Some AI assistance was used for
boilerplate and for explaining the mathematics behind topics such as Kronecker
products and gate matrices and for understanding Rust.
