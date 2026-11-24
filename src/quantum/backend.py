"""Selectable state-vector backend for :mod:`quantum`.

By default Quantar keeps using the pure-Python/NumPy simulator in
``quantum.qubit_abs``. The compiled Rust extension can be selected at runtime
with :func:`use_backend` or the ``QUANTAR_BACKEND`` environment variable:

    import quantum

    quantum.use_backend("rust")       # or "python" to switch back
    qc = quantum.QuantumComputer(2)   # now backed by quantar_kernels

If the extension is missing or out of date, and ``cargo`` plus the Quantar Rust
source tree are available, it is built automatically in release mode and copied
next to the ``quantum`` package. Set ``QUANTAR_RUST_AUTO_BUILD=0`` to disable
that behavior. Useful overrides:

    QUANTAR_RUST_SOURCE_DIR   directory containing ``rust/Cargo.toml``
    QUANTAR_CARGO             cargo executable to use
    QUANTAR_RUST_BACKEND_PATH load a specific ``.pyd``/``.so`` instead
    QUANTAR_STRICT_RUST       keep dependency deprecation warnings visible

Dependency deprecation warnings from crates such as num-traits are suppressed
during automatic builds by default. Set ``QUANTAR_STRICT_RUST=1`` to see them.

The loader also detects free-threaded CPython (``python3.14t`` and newer) and
selects a matching extension suffix. A `cp312-abi3` wheel is GIL-only; source
installs and local auto-builds under a free-threaded interpreter create either
a version-specific ``cp314t`` extension or, on Python 3.15+, an ``abi3t``
extension that both interpreter modes can load.

The Rust module is validated on load, so an old/incomplete ``quantar_kernels``
extension produces a clear error instead of failing later with a missing
attribute.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import tomllib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

PYTHON_BACKEND = "python"
RUST_BACKEND = "rust"

_ENV_BACKEND = "QUANTAR_BACKEND"
_ENV_RUST_PATH = "QUANTAR_RUST_BACKEND_PATH"
_ENV_RUST_SOURCE = "QUANTAR_RUST_SOURCE_DIR"
_ENV_CARGO = "QUANTAR_CARGO"
_ENV_AUTO_BUILD = "QUANTAR_RUST_AUTO_BUILD"
_ENV_BUILD_CACHE = "QUANTAR_RUST_BUILD_CACHE"

_BACKEND_ALIASES = {
    "python": PYTHON_BACKEND,
    "qubit_abs": PYTHON_BACKEND,
    "numpy": PYTHON_BACKEND,
    "rust": RUST_BACKEND,
    "quantar_kernels": RUST_BACKEND,
    "native": RUST_BACKEND,
}

_REQUIRED_RUST_NAMES = ("QuantumComputer", "Qubit", "Gate")
_RUST_MODULE_NAME = "quantum.quantar_kernels"
_BUILD_LOCK = threading.RLock()

_current: str = _BACKEND_ALIASES.get(
    os.environ.get(_ENV_BACKEND, PYTHON_BACKEND).strip().lower(), PYTHON_BACKEND
)
_python_module: ModuleType | None = None
_rust_module: ModuleType | None = None


def _load_python_backend() -> ModuleType:
    global _python_module
    if _python_module is None:
        _python_module = importlib.import_module("quantum.qubit_abs")
    return _python_module


def _validate_rust_backend(module: ModuleType) -> ModuleType:
    missing = [name for name in _REQUIRED_RUST_NAMES if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "quantum.quantar_kernels is missing the current Rust backend API "
            f"({', '.join(missing)}). Rebuild/reinstall the extension; the "
            "installed build is out of date."
        )
    computer = getattr(module, "QuantumComputer")
    if not hasattr(computer, "apply"):
        raise RuntimeError(
            "quantum.quantar_kernels is an older build without QuantumComputer.apply(). "
            "Rebuild/reinstall the extension."
        )
    return module


def _load_from_path(path: str | os.PathLike[str]) -> ModuleType:
    """Load the extension from an explicit file, replacing any cached module."""
    spec = spec_from_file_location(_RUST_MODULE_NAME, os.fspath(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Rust backend from {path!r}")
    module = module_from_spec(spec)
    sys.modules[_RUST_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_RUST_MODULE_NAME, None)
        raise
    return module


def _import_backend(path: str | os.PathLike[str] | None) -> ModuleType:
    if path is None:
        return importlib.import_module(_RUST_MODULE_NAME)
    return _load_from_path(path)


def _auto_build_enabled() -> bool:
    value = os.environ.get(_ENV_AUTO_BUILD, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def find_cargo() -> str | None:
    """Return the cargo executable to use, or ``None`` if it is unavailable."""
    configured = os.environ.get(_ENV_CARGO) or os.environ.get("CARGO")
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        return None
    return shutil.which("cargo")


def _is_quantar_crate(directory: Path) -> bool:
    manifest = directory / "Cargo.toml"
    if not manifest.is_file():
        return False
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    return data.get("package", {}).get("name") == "quantar-kernels"


def find_rust_source_dir() -> Path | None:
    """Find the ``rust/`` crate directory for an editable/source checkout.

    Installed wheels do not ship the Rust source tree, so this may return
    ``None``. The current working directory is searched too, which makes
    ``python -m unittest`` from the repository root work even when an older
    wheel is installed.
    """
    configured = os.environ.get(_ENV_RUST_SOURCE)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        for root in (candidate, candidate / "rust"):
            if _is_quantar_crate(root):
                return root
        return None

    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        for parent in (start, *start.parents):
            for candidate in (parent / "rust", parent):
                candidate = candidate.resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if _is_quantar_crate(candidate):
                    return candidate
    return None


def rust_build_available() -> bool:
    """Return whether local automatic Rust builds are currently possible."""
    return (
        _auto_build_enabled()
        and find_cargo() is not None
        and find_rust_source_dir() is not None
    )


def is_free_threaded() -> bool:
    """Return whether this interpreter uses the free-threaded CPython ABI."""
    disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if disabled:
        return True
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return callable(is_enabled) and not is_enabled()


def _extension_suffix() -> str:
    """Pick an extension suffix compatible with this interpreter's ABI.

    `abi3` wheels cannot be imported by free-threaded CPython, so never choose
    that suffix in a `python3.14t` process. Python 3.15+ can use the newer
    `abi3t` stable ABI from both interpreter modes.
    """
    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    if not suffixes:  # pragma: no cover - every supported CPython has one
        return ".pyd" if os.name == "nt" else ".so"

    free_threaded_suffix = re.compile(r"\d+t[-_.]")
    abi3t = [suffix for suffix in suffixes if "abi3t" in suffix]
    versioned_ft = [suffix for suffix in suffixes if free_threaded_suffix.search(suffix)]

    if is_free_threaded():
        if abi3t:
            return abi3t[0]
        if versioned_ft:
            return versioned_ft[0]
    elif sys.version_info >= (3, 15) and abi3t:
        return abi3t[0]

    abi3 = [suffix for suffix in suffixes if "abi3" in suffix]
    if abi3:
        return abi3[0]

    excluded = set(abi3t) | set(versioned_ft)
    generic = [
        suffix
        for suffix in suffixes
        if suffix not in excluded and sys.implementation.cache_tag not in suffix
    ]
    return generic[-1] if generic else suffixes[-1]


def _default_extension_path() -> Path:
    package_dir = Path(__file__).resolve().parent
    return package_dir / f"quantar_kernels{_extension_suffix()}"


def _is_writable(directory: Path) -> bool:
    probe = directory / f".quantar-write-test-{os.getpid()}"
    try:
        probe.touch(exist_ok=False)
        probe.unlink()
        return True
    except OSError:
        return False


def _build_output_path() -> Path:
    destination = _default_extension_path()
    if _is_writable(destination.parent):
        return destination

    configured = os.environ.get(_ENV_BUILD_CACHE)
    cache_root = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "quantar-rust-backend"
    )
    cache = (
        cache_root
        / f"py{sys.version_info.major}{sys.version_info.minor}"
        / f"{sys.platform}-{getattr(sys.implementation, 'cache_tag', 'python')}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    return cache / destination.name


def _release_dir(source_dir: Path) -> Path:
    configured = os.environ.get("CARGO_TARGET_DIR")
    if not configured:
        return source_dir / "target" / "release"
    target = Path(configured).expanduser()
    if not target.is_absolute():
        target = source_dir / target
    return target / "release"


def _find_built_library(source_dir: Path) -> Path | None:
    directories = [source_dir / "target" / "release"]
    configured = _release_dir(source_dir)
    if configured not in directories:
        directories.append(configured)

    names = (
        "quantar_kernels.dll",
        "libquantar_kernels.so",
        "libquantar_kernels.dylib",
    )
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        for pattern in ("quantar_kernels*.so", "quantar_kernels*.dylib"):
            candidates = sorted(directory.glob(pattern))
            if candidates:
                return candidates[0]
    return None


def _tail(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else f"...{text[-limit:]}"


def _build_environment() -> dict[str, str]:
    """Return the subprocess environment, without dependency warning spam.

    The same suppression is applied by ``hatch_build.py`` for wheel builds so
    local auto-builds and packaged builds behave consistently.
    """
    env = os.environ.copy()
    strict = env.get("QUANTAR_STRICT_RUST", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not strict:
        flags = env.get("RUSTFLAGS", "").strip()
        if "-A deprecated" not in flags:
            env["RUSTFLAGS"] = f"{flags} -A deprecated".strip()
    return env


def _build_rust_backend_unlocked(source_dir: Path | None = None) -> ModuleType:
    global _rust_module

    cargo = find_cargo()
    if cargo is None:
        raise RuntimeError(
            "cargo was not found on PATH. Install Rust from https://rustup.rs or "
            f"set {_ENV_CARGO} to the cargo executable."
        )

    if source_dir is None:
        source_dir = find_rust_source_dir()
    else:
        source_dir = source_dir.expanduser().resolve()
        if not _is_quantar_crate(source_dir):
            candidate = source_dir / "rust"
            if _is_quantar_crate(candidate):
                source_dir = candidate
            else:
                raise RuntimeError(
                    f"{source_dir} is not Quantar's rust crate directory."
                )

    if source_dir is None:
        raise RuntimeError(
            "Could not find Quantar's rust/Cargo.toml. Run from the source "
            f"checkout or set {_ENV_RUST_SOURCE}."
        )

    env = _build_environment()
    env.setdefault("PYO3_PYTHON", sys.executable)
    command = [
        cargo,
        "build",
        "--release",
        "--manifest-path",
        str(source_dir / "Cargo.toml"),
    ]
    result = subprocess.run(
        command,
        cwd=str(source_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = _tail(result.stderr or result.stdout or "no build output")
        raise RuntimeError(
            "Automatic Rust backend build with cargo failed "
            f"(exit code {result.returncode}):\n{details}"
        )

    built = _find_built_library(source_dir)
    if built is None:
        raise RuntimeError(
            "cargo build succeeded but no compiled quantar_kernels library was "
            f"found under {source_dir / 'target' / 'release'}."
        )

    destination = _build_output_path()
    shutil.copy2(built, destination)
    importlib.invalidate_caches()
    sys.modules.pop(_RUST_MODULE_NAME, None)
    module = _load_from_path(destination)
    _rust_module = _validate_rust_backend(module)
    return _rust_module


def build_rust_backend(source_dir: str | os.PathLike[str] | None = None) -> ModuleType:
    """Build and load the Rust backend locally using cargo.

    This is normally called automatically by :func:`load_rust_backend`. Call it
    explicitly to force a fresh release build.
    """
    resolved = (
        Path(source_dir).expanduser().resolve() if source_dir is not None else None
    )
    with _BUILD_LOCK:
        return _build_rust_backend_unlocked(resolved)


def load_rust_backend(path: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import and validate the compiled Rust backend.

    ``path`` loads a specific extension file. When omitted,
    ``QUANTAR_RUST_BACKEND_PATH`` is honoured first, then the normal import. If
    the import fails or the module is stale, a local cargo build is attempted
    when Rust and the source tree are available.
    """
    global _rust_module

    if path is None:
        path = os.environ.get(_ENV_RUST_PATH)

    with _BUILD_LOCK:
        try:
            module = _import_backend(path)
            validated = _validate_rust_backend(module)
        except (ImportError, OSError, RuntimeError) as original:
            if path is not None or not _auto_build_enabled():
                raise
            try:
                return _build_rust_backend_unlocked()
            except Exception as build_error:
                raise RuntimeError(
                    f"Rust backend unavailable ({original}). Automatic cargo "
                    f"build also failed: {build_error}"
                ) from build_error

        _rust_module = validated
        return validated


def get_backend() -> str:
    """Return the active backend name (``"python"`` or ``"rust"``)."""
    return _current


def set_backend(name: str) -> str:
    """Select ``"python"`` or ``"rust"`` and return the active backend name."""
    global _current

    key = name.strip().lower()
    backend = _BACKEND_ALIASES.get(key)
    if backend is None:
        raise ValueError(
            f"Unknown quantum backend {name!r}. Choose one of: "
            f"{', '.join(sorted(_BACKEND_ALIASES))}"
        )

    if backend == RUST_BACKEND:
        load_rust_backend()  # fail early if unavailable/stale/unbuildable
    else:
        _load_python_backend()

    _current = backend
    return _current


def use_backend(name: str) -> str:
    """Alias for :func:`set_backend`, for call-site readability."""
    return set_backend(name)


def reset_backend() -> str:
    """Return to the backend named by ``QUANTAR_BACKEND`` (or Python)."""
    global _current
    _current = _BACKEND_ALIASES.get(
        os.environ.get(_ENV_BACKEND, PYTHON_BACKEND).strip().lower(), PYTHON_BACKEND
    )
    return _current


def get_backend_module() -> ModuleType:
    """Return the module implementing the active backend."""
    if _current == RUST_BACKEND:
        if _rust_module is None:
            return load_rust_backend()
        return _rust_module
    return _load_python_backend()


def available_backends() -> tuple[str, ...]:
    """Return the backends that can currently be used."""
    if _current != RUST_BACKEND:
        try:
            load_rust_backend()
        except (ImportError, RuntimeError, OSError):
            return (PYTHON_BACKEND,)
    return (PYTHON_BACKEND, RUST_BACKEND)


def backend_export(name: str) -> Any:
    """Resolve a backend-dependent public name such as ``QuantumComputer``."""
    return getattr(get_backend_module(), name)


def __getattr__(name: str) -> Any:
    if name in ("QuantumComputer", "Qubit", "Gate"):
        return backend_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PYTHON_BACKEND",
    "RUST_BACKEND",
    "find_cargo",
    "find_rust_source_dir",
    "is_free_threaded",
    "rust_build_available",
    "build_rust_backend",
    "load_rust_backend",
    "get_backend",
    "set_backend",
    "use_backend",
    "reset_backend",
    "get_backend_module",
    "available_backends",
    "backend_export",
]
