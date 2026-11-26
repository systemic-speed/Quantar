# Quantar roadmap

This is the working plan for Quantar after the `1.2.1` release. It is intentionally
practical: each item describes what should change, why it matters, and roughly how
large it is.

## Status key

- **Shipped** — available in `1.2.1`.
- **Next (`1.3.0`)** — highest-priority work for the next feature release.
- **Later (`1.4.0+`)** — planned, but not yet scheduled.
- **Ideas** — interesting, but needs design before committing.

---

## Shipped in 1.2.0–1.2.1

### Python + Rust backends

- **Shipped**
- `quantum.use_backend("python" | "rust")` selects the simulator at runtime.
- `QuantumComputer`, `Qubit`, and `Gate` resolve dynamically through
  `quantum.__getattr__`.
- The Rust backend mirrors the Python API: `apply`, `gate`, `op_hist`,
  checkpointing, `uncompute`, QFT/IQFT, Pauli measurement, sampling, and QASM
  export.

### Stable ABI extensions

- **Shipped**
- PyO3 uses `abi3-py312` for GIL-enabled CPython 3.12+.
- GIL-enabled wheels are tagged `cp312-abi3-*`.
- PyO3 also enables `abi3t-py315`, so Python 3.15+ can produce a PEP 803 stable
  ABI extension usable from both GIL-enabled and free-threaded interpreters.
- Free-threaded CPython 3.14 builds a version-specific `cp314t` extension;
  PyO3 0.30 no longer supports 3.13t.
- Version-specific `cp315` binaries were removed to avoid shadowing the stable
  ABI build.

### Local auto-build

- **Shipped**
- If the extension is missing or stale, and `cargo` plus `rust/Cargo.toml` are
  available, Quantar builds it locally in release mode and loads it.
- The loader detects `Py_GIL_DISABLED` and avoids GIL-only `abi3` files under
  free-threaded interpreters.
- Controlled by `QUANTAR_RUST_AUTO_BUILD`, `QUANTAR_RUST_SOURCE_DIR`,
  `QUANTAR_CARGO`, and `QUANTAR_RUST_BACKEND_PATH`.

### Quiet builds

- **Shipped (`1.2.1`)**
- Wheel builds are quiet by default; `QUANTAR_VERBOSE_BUILD=1` (or
  `HATCH_MATURIN_ARGS="-vv"`) restores detailed cargo/maturin output.
- Dependency deprecation spam, such as `num-traits 0.2.19`'s deprecated float
  constants, is suppressed unless `QUANTAR_STRICT_RUST=1` is set.
- The same suppression is applied to runtime `cargo` auto-builds.

### Public imports and docs

- **Shipped (`1.2.1`)**
- `quantar` re-exports common language, simulator, algorithm, physics, backend,
  and seeding names lazily.
- `stl` re-exports its two subpackages and common algorithms/helpers.
- `docs/API.md` is the import cheat sheet and public API guide.
- The Shor/Grover functions now live in `*_lib` modules and remain available
  from `stl.quantum_algs`.

### Import support

- **Shipped**
- `.qtr` modules are importable through a custom meta-path finder.
- Operators are preloaded before parsing.
- Runner sets `__file__`/`__package__`, adds the source directory to `sys.path`,
  and injects runtime imports after `__future__` imports.

### Performance work

- **Shipped**
- Rust-side unitarity checks for small matrices.
- Contiguous matrix normalization.
- Direct `GateView` for Rust gates.
- One-pass gate classification in `apply(program)`.
- Fused diagonal and disjoint-SWAP kernels.
- Reproducible benchmark at `test/bench_backend.py`.

---

## Post-1.2.1 (on `main`, not yet released)

### Physics kernels and helpers

- Rust kernels in `quantar_kernels` for kinetic energy, Lorentz forces,
  central-difference Jacobians/gradients/Hessians, and velocity-Verlet steps.
- NumPy twins live next to them in `stl.physics_algs.kernels`; choosing the
  compiled path can be done with `use_rust_physics()` or
  `QUANTAR_PHYSICS_KERNELS=0`. Missing or stale extensions fall back silently.
- New helpers: `VectorField.hessian()`/`laplacian()`, scalar `hessian()`/
  `laplacian()`, `momentum()`, `angular_momentum()`, `Dual.jacobian()`/
  `value_and_gradient()`, `SpaceMetric.volume_element()`, and `Trajectory`
  series methods.
- Version in the source tree is `1.2.3`, tagged `v1.2.3`.

---

## Next: 1.3.0

Theme: **make it faster, safer, reproducible, and available everywhere**.

### P0 — Cross-platform and free-threaded wheels

- **Next (`1.3.0`)**
- Build wheels in CI for:
  - Windows x64
  - Linux x86_64
  - macOS x86_64
  - macOS ARM64
- For each platform, publish:
  - `cp312-abi3-*` for GIL-enabled CPython 3.12–3.14
  - `cp314t-*` for free-threaded CPython 3.14
  - `abi3t-*` for CPython 3.15+ in both interpreter modes
- Install each built wheel in a clean environment and run the import/Rust API
  tests, including a `python3.14t -Xgil=0` job.
- Add `twine check` and wheel metadata validation to CI.
- Why: the stable ABIs solve Python-version compatibility, but not OS,
  architecture, or GIL-mode compatibility. Without platform wheels, users may
  need a Rust toolchain for source installs.

### P0 — Sparse multi-controlled gate kernel

- **Next (`1.3.0`)**
- Current behavior: the general kernel scans all `2^n` amplitudes and branches
  on `control_mask`.
- Proposed behavior:
  1. copy `state` into `scratch` once;
  2. enumerate only basis indices where all control bits are set;
  3. apply the matrix only to those branches.
- Add a threshold so low-control-count gates keep the current path.
- Why: multi-controlled gates do useful work on only a small fraction of the
  state vector.

### P0 — Specialized 1-qubit and 2-qubit kernels

- **Shipped (`1.2.1`)** for 1 target; **Next (`1.3.0`)** for 2 targets
- The Rust backend has an in-place 2x2 path for one target, with correct
  control-bit checks even when controls sit below the target.
- Still planned:
  - 2 targets, no controls
  - 2 targets, controls
  - fewer row-index rebuilds in the general kernel
- Why: H/X/Z/P/RZ and CP/SWAP-style gates dominate many real circuits.

### P1 — Seeded RNG and reproducible sampling

- **Shipped (`1.2.1`)**
- `quantum.set_seed(1234)` / `quantum.get_seed()` are public and shared by
  both backends.
- The Rust backend uses a seeded PCG64 per `QuantumComputer` for measurement,
  sampling, and Pauli expectations.
- The Python backend uses the same seed for measurement, `sample`, and
  `measure_pauli`.
- See `docs/API.md` for semantics and examples.

### P1 — Randomized differential tests

- **Next (`1.3.0`)**
- Generate random circuits and compare Rust against the Python backend:
  - random 1–6 qubit circuits
  - controls
  - diagonal gates
  - SWAP runs
  - QFT/IQFT
  - `uncompute`
  - QASM output
  - state vectors after each gate
- Add benchmark regression checks with generous tolerances.
- Why: fusion and bit-order bugs are subtle; randomized parity testing catches
  cases hand-written tests miss.

### P1 — Native QFT and QASM helpers

- **Next (`1.3.0`)**
- Move QFT/IQFT gate construction into Rust.
- Remove or reduce Python callbacks from `qft_impl`.
- Consider a native OpenQASM 3 exporter.
- Keep Python builders as references for parity tests.
- Why: the Rust QFT path still imports Python modules and builds Python gate
  objects.

### P2 — CLI backend and seed flags

- **Next (`1.3.0`)**
- Add flags such as:

  ```powershell
  qtr program.qtr --backend rust
  qtr program.qtr --backend python
  qtr program.qtr --seed 1234
  ```

- Why: backend selection is currently library-focused; CLI users should be able
  to try the Rust backend directly.

### P2 — Import hook polish

- **Next (`1.3.0`)**
- Harden `.qtr` imports:
  - namespace package support
  - clearer circular-import errors
  - cached/compiled `.qtr` modules
  - better `reload()` behavior
  - more precise source locations in errors
- Why: larger Quantar projects will stress the import system.

### P2 — Packaging and docs polish

- **Next (`1.3.0`)**
- Add `CHANGELOG.md`.
- Add PyPI classifiers, keywords, and project URLs.
- Decide whether to keep or exclude `quantar_kernels.pdb` from wheels.
- Include tests in the sdist if source-build verification should be possible.
- Expand API reference documentation for both backends.
- Why: release quality is more than passing tests.

---

## Later: 1.4.0+

Theme: **language and tooling maturity**.

### Bytecode interpreter

- **Later (`1.4.0+`)**
- Define a Quantar bytecode format for compiled `.qtr` programs.
- Separate frontend parsing from execution.
- Add a stable loader/runtime boundary.
- Why: enables faster startup, better tooling, and distribution of compiled
  Quantar programs.

### Debugger

- **Later (`1.4.0+`)**
- Breakpoints, stepping, variable inspection, and state-vector inspection.
- Integrate with the LSP and VS Code extension.
- Depends on the bytecode/runtime work.
- Why: debugging is one of the largest missing developer-experience features.

### LSP improvements

- **Later (`1.4.0+`)**
- Completions for gates, operators, and imported `.qtr` symbols.
- Hover documentation for operators and gates.
- Go-to-definition across `.qtr` and `.py` modules.
- Diagnostics for invalid gate arity/control combinations.
- Why: the current LSP is a good base, but still basic.

### Alternative execution backends

- **Later (`1.4.0+`)**
- Unified interface for PennyLane, Qiskit, qbraid, and Qiskit Aer.
- Backend capability negotiation.
- Consistent sampling/measurement semantics across backends.
- Why: Quantar already has optional backend dependencies; the next step is a
  coherent execution layer.

---

## Ideas / larger research items

These are not scheduled. They need a design proposal before entering the
roadmap.

- GPU state-vector backend.
- Distributed state vectors.
- Tensor-network backend.
- Noise models and density-matrix simulation.
- Automatic differentiation through circuits.
- Circuit optimization passes.
- QASM 3 import/export conformance suite.
- Persistent compiled `.qtr` cache format.

---

## Definition of done

For any roadmap item, the work is done when:

1. Python and Rust backends still pass parity tests.
2. New behavior has regression tests.
3. Benchmarks are measured before and after performance changes.
4. Public API changes are documented in `README.md` and `CHANGELOG.md`.
5. Wheels/sdists build cleanly.
6. The feature is verified from an installed wheel, not only from the source
   tree.
7. Old behavior either still works or has a clear migration note.
