"""Build an executable from a Quantar program with PyInstaller.

The entry point may be a ``.qtr`` source file or an already lowered ``.py``
file. Local ``.qtr`` imports are lowered to Python first, so PyInstaller sees
plain ``import`` statements instead of the runtime meta-path finder.

PyInstaller is run as ``python -m PyInstaller`` through :mod:`subprocess`; this
module does not import PyInstaller itself.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Sequence

from . import imports

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SRC = Path(__file__).resolve().parents[1]
_MANIFEST_MODULE = "quantar._nuitka_manifest"
_KERNEL_MODULE = "quantum.quantar_kernels"
_HIDDEN_IMPORTS = (_MANIFEST_MODULE, _KERNEL_MODULE)
_MODES = ("onefile", "standalone")
_QUANTAR_ROOTS = ("quantar", "quantum", "stl")

# Inserted into lowered entry points when ``--rust`` is requested. It must run
# before the program imports ``quantum`` so the backend is selected at import
# time, including inside the built executable.
RUST_PRELUDE = """import os as _quantar_os
_quantar_os.environ["QUANTAR_BACKEND"] = "rust"
try:
    from quantum import backend as _quantar_backend
    _quantar_backend.set_backend("rust")
    from stl.physics_algs import use_rust_physics as _quantar_use_rust_physics
except ImportError:
    pass
else:
    _quantar_use_rust_physics(True)
"""


def rust_extension_status(*, prepare: bool = False) -> tuple[bool, str]:
    """Return whether the Rust extension imports under this interpreter.

    With ``prepare=True``, use Quantar's loader, which auto-builds a matching
    extension from the Rust source when the installed one is missing/stale.
    """
    try:
        if prepare:
            from quantum import backend

            module = backend.load_rust_backend()
        else:
            module = importlib.import_module(_KERNEL_MODULE)
    except Exception as error:  # ImportError / OSError / RuntimeError
        return False, f"{type(error).__name__}: {error}"
    location = getattr(module, "__file__", None)
    return True, str(location) if location else _KERNEL_MODULE


def _inject_prelude(entry: Path, prelude: str) -> None:
    """Insert ``prelude`` after any docstring and ``__future__`` imports."""
    source = entry.read_text(encoding="utf-8")
    if "_quantar_os" in source:
        return
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    index = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        index = 1
    for position, statement in enumerate(tree.body):
        if (
            isinstance(statement, ast.ImportFrom)
            and statement.module == "__future__"
        ):
            index = max(index, position + 1)
    insert_at = tree.body[index].lineno - 1 if index < len(tree.body) else len(lines)
    lines.insert(insert_at, prelude.rstrip() + "\n")
    entry.write_text("".join(lines), encoding="utf-8")


def _source_tree_needs_quantar(source: Path, search_paths: Sequence[str | Path]) -> bool:
    """Return whether the original program imports a Quantar package."""
    roots = [Path(root).resolve() for root in search_paths]
    pending = [source.resolve()]
    seen: set[Path] = set()
    while pending:
        current = pending.pop(0)
        if current in seen or current.suffix not in {".qtr", ".py"}:
            continue
        seen.add(current)
        candidates: list[str] = []
        for statement in imports.iter_import_statements(current.read_text(encoding="utf-8")):
            names = [alias.name for alias in statement.names] if isinstance(statement, ast.Import) else [statement.module or ""]
            if any(name.split(".", 1)[0] in _QUANTAR_ROOTS for name in names):
                return True
            candidates.extend(names)
        for candidate in candidates:
            found = imports.find_qtr_module(candidate, roots)
            if found is not None and found not in seen:
                pending.append(found)
    return False


def _output_target(compile_destination: str | Path, entry_name: str) -> tuple[Path, str]:
    """Split the destination into an output directory and executable name."""
    destination = Path(compile_destination).resolve()
    if destination.suffix.lower() in {".exe", ".bin", ".app"}:
        return destination.parent, destination.stem
    return destination, entry_name or "program"


def pyinstaller_command(entry: str | Path, output_dir: str | Path, output_name: str, *, mode: Literal["onefile", "standalone"] = "standalone", work_dir: str | Path | None = None, paths: Sequence[str | Path] = (), hidden_imports: Sequence[str] = _HIDDEN_IMPORTS, extra_args: Sequence[str] = ()) -> list[str]:
    """Return the PyInstaller command used by :func:`make_bin_for_file`."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    work = Path(work_dir).resolve() if work_dir is not None else Path(output_dir).resolve() / "pyinstaller-build"
    mode_flag = "--onefile" if mode == "onefile" else "--onedir"
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--log-level=WARN",
        mode_flag,
        f"--name={output_name}",
        f"--distpath={output_dir}",
        f"--workpath={work / 'build'}",
        f"--specpath={work / 'spec'}",
        *(f"--paths={Path(root).resolve()}" for root in paths),
        *(f"--hidden-import={name}" for name in hidden_imports),
        *extra_args,
        str(entry),
    ]


def pyinstaller_env(*source_roots: str | Path) -> dict[str, str]:
    """Environment with ``source_roots`` ahead of the existing ``PYTHONPATH``."""
    env = os.environ.copy()
    additions = [str(Path(root).resolve()) for root in source_roots]
    existing = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    env["PYTHONPATH"] = os.pathsep.join([*additions, *existing])
    return env


def _find_artifact(output_dir: Path, output_name: str, mode: str) -> Path:
    """Locate the executable or distribution PyInstaller wrote in ``output_dir``."""
    if mode == "standalone":
        bundle = output_dir / output_name
        for candidate in (bundle / f"{output_name}.exe", bundle / output_name, bundle / f"{output_name}.bin"):
            if candidate.is_file():
                return candidate
        if bundle.is_dir():
            return bundle
        raise FileNotFoundError(f"PyInstaller wrote no {output_name} directory in {output_dir}")

    for candidate in (output_dir / f"{output_name}.exe", output_dir / output_name, output_dir / f"{output_name}.bin"):
        if candidate.is_file():
            return candidate
    executables = [path for path in output_dir.iterdir() if path.is_file() and os.access(path, os.X_OK) and path.suffix not in {".py", ".pyi", ".spec"}]
    if not executables:
        raise FileNotFoundError(f"PyInstaller wrote no executable in {output_dir}")
    return max(executables, key=lambda path: path.stat().st_mtime)


def make_bin_for_file(filename: str | Path, compile_destination: str | Path = "./quantar_compile/bins", mode: Literal["onefile", "standalone"] = "standalone", jobs: int = 2, *, output_name: str | None = None, extra_args: Sequence[str] = (), build_dir: str | Path | None = None, rust: bool = False) -> Path:
    """Compile ``filename`` (a ``.qtr`` or lowered ``.py`` file) to an executable.

    ``compile_destination`` is normally a directory; the executable name is
    derived from the source file. A path ending in ``.exe``/``.bin``/``.app``
    is treated as the full destination instead. Returns the artifact path.

    The lowering step writes a temporary Python tree containing the entry point
    and every local ``.qtr`` module it imports. Programs that import Quantar
    packages also get the source tree on the search path and the static import
    manifest so lazy package exports are bundled; plain Python programs skip
    both to keep analysis fast.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    source = Path(filename).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Quantar source file not found: {source}")

    output_dir, derived_name = _output_target(compile_destination, source.stem)
    name = output_name or derived_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="quantar-pyinstaller-") as temporary:
        work = Path(build_dir).resolve() if build_dir is not None else Path(temporary)
        work.mkdir(parents=True, exist_ok=True)

        lowered_root = work / "lowered"
        search_paths = [source.parent, _REPO_SRC, _REPO_ROOT]
        entry, _ = imports.lower_qtr_tree(source, lowered_root, search_paths=search_paths)
        build_args = list(extra_args)
        needs_quantar = rust or _source_tree_needs_quantar(source, search_paths)
        hidden_imports = list(_HIDDEN_IMPORTS) if needs_quantar else []

        if rust:
            available, detail = rust_extension_status(prepare=True)
            if not available:
                raise RuntimeError(f"--rust requires the Rust extension, but it could not be loaded: {detail}")
            print(f"[RUST] Bundling {detail}", flush=True)
            if Path(detail).is_file():
                build_args.append(f"--add-binary={detail}{os.pathsep}quantum")
            _inject_prelude(entry, RUST_PRELUDE)

        build_paths = [lowered_root, _REPO_SRC, source.parent] if needs_quantar else [lowered_root, source.parent]
        command = pyinstaller_command(entry, output_dir, name, mode=mode, work_dir=work, paths=build_paths, hidden_imports=hidden_imports, extra_args=build_args)
        env = pyinstaller_env(*build_paths)
        subprocess.run(command, check=True, cwd=work, env=env)

    return _find_artifact(output_dir, name, mode)


__all__ = [
    "make_bin_for_file",
    "pyinstaller_command",
    "pyinstaller_env",
    "rust_extension_status",
]
