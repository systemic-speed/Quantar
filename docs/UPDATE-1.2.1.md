# Quantar 1.2.1 — what's new

Released 24 November 2026. [Install it from PyPI](https://pypi.org/project/Quantar/1.2.1/) or read the [API guide](API.md).

Quantar is a Python superset for quantum computing and physics simulation. Version 1.2.1 is a polish release: results are easier to reproduce, imports are simpler, installs are quieter, and the optional compiled accelerator works on more Python versions.

## Highlights

### Reproducible runs

One shared seed now controls measurements, sampling, Pauli expectations, and both simulators. If you repeat the same circuit with the same seed, you get the same counts.

```python
import quantum

quantum.set_seed(1234)
qc = quantum.QuantumComputer(3)
counts = qc.sample_parallel(circuit, 1000)

# Later, repeat the same run:
quantum.set_seed(1234)
qc = quantum.QuantumComputer(3)
assert counts == qc.sample_parallel(circuit, 1000)
```

`quantum.get_seed()` returns the seed currently in use. If you have not chosen one, Quantar picks one from your system and remembers it, so a past run can still be reproduced.

### Easier imports

You can now import common names from the top-level package without knowing which internal module owns them:

```python
import quantar
from stl import shors_alg, grovers_alg, Quantity

qc = quantar.QuantumComputer(2)
quantar.set_seed(7)
```

The Shor and Grover functions kept their names. Only their implementation files were renamed, so existing `from stl.quantum_algs import grovers_alg` code still works.

Physics helpers take the same shortcut: `quantar.Q(9.81, "m/s^2")` builds a unit-checked value without importing an internal module. The [API guide](API.md) now shows short examples for units, automatic derivatives, and vector fields.

### Quieter installs

Normal `pip install`, `uv build`, and source builds no longer print long Rust dependency warnings. If you want the full detail, set `QUANTAR_VERBOSE_BUILD=1`; to keep dependency warnings, set `QUANTAR_STRICT_RUST=1`.

### More Python versions supported

| Your Python | What pip installs |
| --- | --- |
| 3.12, 3.13, 3.14 (standard) | `cp312-abi3` wheel |
| 3.14 free-threaded | `cp314-cp314t` wheel |
| 3.15 or 3.15 free-threaded | `cp315-abi3t` wheel |
| Any supported Python | `tar.gz` source archive |

Two quick definitions:

- **abi3** means one compiled file works across several Python versions.
- **free-threaded** means a Python build that runs without the GIL, used for some parallel workloads.

The source archive is there as a fallback. Uploading it alone would make everyone compile from source, so all four files were published.

## Fixes worth knowing about

- Fixed a controlled-gate edge case where a control qubit sat below the target.
- Fixed the QFT/IQFT helpers after the algorithm files were reorganised.
- Fixed building from source on Python 3.12 and 3.13.
- Added regression tests for seeding, imports, build settings, and the gate fix.

## Verification

The release passed Rust formatting, linting, and unit tests; 20 Python tests; 19 Rust-backend API tests; and `twine check`. Installed wheels were smoke-tested on Python 3.12, 3.14t, and 3.15t, along with a fresh source install on 3.12.

## Upgrade

```bash
python -m pip install -U quantar
```

Release tag: `v1.2.1` (`b62c045`). See [roadmap.md](../roadmap.md) for what is planned next.
