"""CLI and command implementations for Quantar programs."""

from ast import unparse
from pathlib import Path
from typing import Literal, Sequence
import ast
import linecache
import os
import sys
import traceback

import typer

from . import ast_builder
from .checker import QuantarCheckError, check_project
from .config import QuantarConfig, find_config
from .imports import find_qtr_module, install_import_hook, iter_import_statements, preload_imports, resolve_import_name
from .project import init_project, new_project
from .runtime import runtime_globals
from .to_bin import RUST_PRELUDE, make_bin_for_file

INTERNAL_RUNNER_FILE = Path(__file__).resolve()
INTERNAL_PACKAGE_DIR = INTERNAL_RUNNER_FILE.parent.parent.resolve()

_SOURCE_SUFFIXES = {".qtr", ".py"}
_PYTHON_TARGETS = {"py", "python"}
_BINARY_TARGETS = {"bin", "binary"}
_COMPILE_TARGETS = _PYTHON_TARGETS | _BINARY_TARGETS
_PYINSTALLER_MODES = ("onefile", "standalone")
_DEFAULT_BINARY_DIR = "./quantar_compile/bins"
_DEFAULT_PY_DIR = "./quantar_compile/py"

_COMMAND_NAMES = frozenset({"run", "r", "compile", "c", "comp", "check", "new", "init", "lower"})
_GLOBAL_OPTIONS = frozenset({"-h", "--help", "--install-completion", "--show-completion"})
_RESERVED_ARGV = _COMMAND_NAMES | _GLOBAL_OPTIONS

_APPLY_IMPORT = "from lang_systems.runtime import apply_quantar_operator as __quantar_apply__"
_IMPORT_HOOK_IMPORT = "from lang_systems.imports import install_import_hook as __quantar_install_import_hook__"
_IMPORT_HOOK_CALL = "__quantar_install_import_hook__()"

Source = str | Path
OptionalSource = Source | None
OptionalText = str | None
OptionalFlag = bool | None
PyInstallerMode = Literal["onefile", "standalone"]

_FILENAME = typer.Argument(None, help="The .qtr or .py file to process; defaults to the project entry")
_NAME = typer.Argument(..., help="Name of the new project")
_TARGET_DIR = typer.Argument(None, help="Directory to initialize; defaults to the current directory")
_OUTPUT = typer.Argument(None, help="Where to write the .py file or binary; defaults to the project output")
_COMPILE_TO = typer.Option(None, "--compile-to", "-c", help="Compile target: py, python, bin, or binary")
_MODE = typer.Option(None, "--mode", help="PyInstaller output mode: onefile or standalone")
_JOBS = typer.Option(None, "--jobs", help="Kept for compatibility; ignored by PyInstaller")
_RUST = typer.Option(None, "--rust/--no-rust", help="Override [tool.quantar].rust")
_TRACEBACK = typer.Option(False, "--full-traceback", help="Show the complete traceback, including internal frames")

app = typer.Typer(name="qtr", help="CLI for working with Quantar (.qtr) programs", epilog="A bare filename is shorthand for 'qtr run'; commands read [tool.quantar] defaults.", no_args_is_help=True)
_FULL_TRACEBACK = False


# ---------------------------------------------------------------------------
# File and AST helpers
# ---------------------------------------------------------------------------


def _validate_source(path: Path) -> None:
    """Raise a helpful error when ``path`` is not a Quantar source file."""
    if not path.is_file():
        raise FileNotFoundError(f"Quantar source file not found: {path}")
    if path.suffix not in _SOURCE_SUFFIXES:
        raise ValueError(f"Expected a .qtr or .py file, got {path.suffix!r}")


def _package_for(path: Path) -> str:
    """Best-effort ``__package__`` for a file run as the main module."""
    directory = path.parent.resolve()
    is_package = (
        path.name == "__init__.qtr"
        or (directory / "__init__.qtr").is_file()
        or (directory / "__init__.py").is_file()
    )
    if not is_package:
        return ""
    for entry in (*sys.path, str(INTERNAL_PACKAGE_DIR)):
        if not entry:
            continue
        try:
            parts = directory.relative_to(Path(entry).resolve()).parts
        except ValueError:
            continue
        return ".".join(parts)
    return ""


def _add_source_dir(path: Path) -> None:
    """Match Python's script behaviour by putting the source directory on sys.path."""
    directory = str(path.parent.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)


def _cache_source(source_name: str, source_code: str) -> None:
    """Keep source lines available to tracebacks after lowering or exec."""
    source_lines = source_code.splitlines(keepends=True)
    linecache.cache[source_name] = (len(source_code), None, source_lines, source_name)


def _uses_quantar_apply(tree: ast.AST) -> bool:
    """Return whether lowering produced a ``__quantar_apply__`` call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__quantar_apply__":
            return True
    return False


def _needs_import_hook(source_code: str, path: Path, package: str) -> bool:
    """Return whether any imported name resolves to a local ``.qtr`` file."""
    root = path.parent
    if package:
        for _ in package.split("."):
            root = root.parent

    for statement in iter_import_statements(source_code):
        if isinstance(statement, ast.Import):
            names = [alias.name for alias in statement.names]
        else:
            base = resolve_import_name(statement, package or None)
            names = [base] if base else []
            for alias in statement.names:
                if alias.name != "*":
                    names.append(f"{base}.{alias.name}")
        if any(find_qtr_module(name, [root]) is not None for name in names):
            return True
    return False


def _is_docstring(statement: ast.stmt) -> bool:
    """Return whether ``statement`` is a module docstring."""
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)


def _helper_insertion_index(module: ast.Module) -> int:
    """Return the first statement index after docstring and future imports."""
    index = 0
    if module.body and _is_docstring(module.body[0]):
        index = 1
    for position, statement in enumerate(module.body):
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            index = max(index, position + 1)
    return index


def _inject_runtime_imports(tree: ast.AST, rust: bool = False, source_code: OptionalText = None, source_path: OptionalSource = None, package: OptionalText = None) -> ast.Module:
    """Add only the helpers the lowered program actually needs."""
    lowered = ast_builder.lower_ast(tree)
    assert isinstance(lowered, ast.Module)

    helper_lines: list[str] = []
    if source_code is None or _uses_quantar_apply(lowered):
        helper_lines.append(_APPLY_IMPORT)
    if source_code is None or source_path is None or _needs_import_hook(source_code, Path(source_path), package or ""):
        helper_lines.extend((_IMPORT_HOOK_IMPORT, _IMPORT_HOOK_CALL))
    if rust:
        helper_lines.append(RUST_PRELUDE.rstrip())
    if not helper_lines:
        return lowered

    helpers = ast.parse("\n".join(helper_lines) + "\n").body
    index = _helper_insertion_index(lowered)
    lowered.body[index:index] = helpers
    ast.fix_missing_locations(lowered)
    return lowered


# ---------------------------------------------------------------------------
# Public command implementations
# ---------------------------------------------------------------------------


def qtr_check(filename: Source) -> Path:
    """Check a program and its local imports without executing any of them."""
    path = Path(filename).resolve()
    _validate_source(path)

    report = check_project(path)
    if report.issues:
        raise QuantarCheckError(report.issues)
    print(f"[CHECK] OK: {len(report.checked)} source file(s) checked")
    return path


def qtr_new(name: str, parent: OptionalSource = None, *, rust: bool = False) -> Path:
    """Create a new Quantar project directory."""
    return new_project(name, parent, rust=rust)


def qtr_init(target_dir: OptionalSource = None, *, name: OptionalText = None, rust: bool = False) -> Path:
    """Initialize a Quantar project in an existing directory."""
    return init_project(target_dir, name, rust=rust)


def qtr_compile(filename: Source, output_path: OptionalSource = None, mode: PyInstallerMode = "standalone", jobs: int = 2, *, rust: bool = False) -> Path:
    """Compile a Quantar (or lowered Python) file to a PyInstaller executable.

    ``rust`` loads/builds the Rust extension and selects it inside the generated
    binary. A missing or incompatible extension is an error, not a silent
    fallback to NumPy.
    """
    path = Path(filename).resolve()
    _validate_source(path)
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")

    destination = output_path if output_path is not None else _DEFAULT_BINARY_DIR
    return make_bin_for_file(path, destination, mode=mode, jobs=jobs, rust=rust)


qtr_compiler = qtr_compile


def _enable_rust_physics() -> None:
    """Turn on the Rust physics path; the NumPy twins remain available."""
    try:
        from stl.physics_algs import use_rust_physics
    except ImportError:
        return
    use_rust_physics(True)


def _enable_rust_backend() -> None:
    """Load/build the Rust extension, select it, and enable Rust physics."""
    from quantum import backend

    os.environ["QUANTAR_BACKEND"] = "rust"
    backend.set_backend("rust")
    _enable_rust_physics()


def qtr_runner(filename: Source, output_path: OptionalSource = None, *, rust: bool = False) -> Path:
    """Run a Quantar file, or lower it to Python when ``output_path`` is given."""
    path = Path(filename)
    _validate_source(path)

    if rust:
        # Must happen before preload_imports() imports ``quantum``.
        _enable_rust_backend()

    install_import_hook()
    package = _package_for(path)
    _add_source_dir(path)

    source_code = path.read_text(encoding="utf-8")
    source_name = str(path)
    _cache_source(source_name, source_code)
    preload_imports(source_code, source_name, package)

    tree = ast_builder.parse_ast(source_code, source_name)
    if output_path is not None:
        lowered = _inject_runtime_imports(tree, rust, source_code, path, package)
        out_path = Path(output_path)
        out_path.write_text(unparse(lowered) + "\n", "utf-8")
        print(f"[COMPILE] Compiled to {out_path}")
        return out_path

    python_ast = ast_builder.compile_ast(tree, source_name)
    globals_dict = {
        "__name__": "__main__",
        "__file__": source_name,
        "__package__": package,
        **runtime_globals(),
    }
    exec(python_ast, globals_dict)
    return path


def qtr_lower(filename: Source, output_path: Source = _DEFAULT_PY_DIR, *, rust: bool = False) -> Path:
    """Lower a Quantar file to Python without executing it."""
    return qtr_runner(filename, output_path, rust=rust)


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


def _remember_traceback(full_traceback: bool) -> None:
    """Remember whether command errors should include internal frames."""
    global _FULL_TRACEBACK
    _FULL_TRACEBACK = full_traceback


def _parse_compile_options(compile_to: str, mode: str) -> tuple[str, str]:
    """Normalize and validate ``--compile-to`` and ``--mode``."""
    target = compile_to.lower()
    if target not in _COMPILE_TARGETS:
        raise typer.BadParameter("compile target must be py, python, bin, or binary")
    selected_mode = mode.lower()
    if selected_mode not in _PYINSTALLER_MODES:
        raise typer.BadParameter("mode must be onefile or standalone")
    return target, selected_mode


def _resolve_source(filename: OptionalText, config: QuantarConfig | None) -> Source:
    """Use an explicit filename, or the configured project entry point."""
    if filename is not None and filename.strip():
        return filename
    if config is not None:
        return config.entry_path
    raise typer.BadParameter("provide a source file or add [tool.quantar].entry to pyproject.toml")


def _rust_value(rust: OptionalFlag, config: QuantarConfig | None) -> bool:
    """Resolve an explicit Rust flag against the project setting."""
    if rust is not None:
        return rust
    return config.rust if config is not None else False


def _jobs_value(jobs: int | None, config: QuantarConfig | None) -> int:
    """Resolve an explicit job count against the project setting."""
    if jobs is not None:
        return jobs
    return config.jobs if config is not None else 2


def _compile_settings(compile_to: OptionalText, mode: OptionalText, config: QuantarConfig | None) -> tuple[str, str]:
    """Resolve compile target and mode from CLI flags or project settings."""
    target = compile_to if compile_to is not None else (config.compile_to if config is not None else "py")
    selected_mode = mode if mode is not None else (config.mode if config is not None else "standalone")
    return _parse_compile_options(str(target), str(selected_mode))


def _compile_destination(output_path: OptionalSource, filename: Source, config: QuantarConfig | None, target: str) -> Source:
    """Resolve where a compile command should write its output."""
    if output_path is not None and str(output_path).strip():
        return output_path
    if config is not None and config.output_path is not None:
        return config.output_path
    if target in _PYTHON_TARGETS:
        return Path(filename).with_suffix(".py")
    return _DEFAULT_BINARY_DIR


@app.command("run")
def cmd_run(filename: str | None = _FILENAME, rust: OptionalFlag = _RUST, full_traceback: bool = _TRACEBACK) -> None:
    """Run a Quantar file or the configured project entry point."""
    _remember_traceback(full_traceback)
    config = find_config(filename)
    qtr_runner(_resolve_source(filename, config), rust=_rust_value(rust, config))


@app.command("compile")
def cmd_compile(filename: str | None = _FILENAME, output_path: str | None = _OUTPUT, compile_to: str | None = _COMPILE_TO, mode: str | None = _MODE, jobs: int | None = _JOBS, rust: OptionalFlag = _RUST, full_traceback: bool = _TRACEBACK) -> None:
    """Lower a Quantar file to Python or build a binary."""
    _remember_traceback(full_traceback)
    config = find_config(filename)
    source = _resolve_source(filename, config)
    target, selected_mode = _compile_settings(compile_to, mode, config)
    destination = _compile_destination(output_path, source, config, target)
    use_rust = _rust_value(rust, config)

    if target in _PYTHON_TARGETS:
        qtr_runner(source, destination, rust=use_rust)
        return
    artifact = qtr_compile(source, destination, mode=selected_mode, jobs=_jobs_value(jobs, config), rust=use_rust) #type: ignore[arg-type]
    print(f"[BINARY] Compiled to {artifact}")


@app.command("lower")
def cmd_lower(filename: str | None = _FILENAME, output_path: str | None = _OUTPUT, rust: OptionalFlag = _RUST, full_traceback: bool = _TRACEBACK) -> None:
    """Lower a Quantar file to Python without executing it."""
    _remember_traceback(full_traceback)
    config = find_config(filename)
    source = _resolve_source(filename, config)
    destination = _compile_destination(output_path, source, config, "py")
    qtr_lower(source, destination, rust=_rust_value(rust, config))


@app.command("new")
def cmd_new(name: str = _NAME, rust: OptionalFlag = _RUST, full_traceback: bool = _TRACEBACK) -> None:
    """Create a new Quantar project."""
    _remember_traceback(full_traceback)
    root = qtr_new(name, rust=_rust_value(rust, None))
    print(f"[NEW] Created project {root}")


@app.command("init")
def cmd_init(target_dir: str | None = _TARGET_DIR, rust: OptionalFlag = _RUST, full_traceback: bool = _TRACEBACK) -> None:
    """Initialize a Quantar project in a directory."""
    _remember_traceback(full_traceback)
    root = qtr_init(target_dir, rust=_rust_value(rust, None))
    print(f"[INIT] Initialized project {root}")


@app.command("check")
def cmd_check(filename: str | None = _FILENAME, full_traceback: bool = _TRACEBACK) -> None:
    """Check a program and its local imports without running any of them."""
    _remember_traceback(full_traceback)
    config = find_config(filename)
    source = _resolve_source(filename, config)
    try:
        qtr_check(source)
    except QuantarCheckError as error:
        for issue in error.issues:
            sys.stderr.write(f"[CHECK] ERROR {issue.format()}\n")
        sys.stderr.write(f"[CHECK] FAILED: {len(error.issues)} error(s)\n")
        raise SystemExit(1) from None


run = cmd_run
compile_cmd = cmd_compile
lower = cmd_lower
new = cmd_new
init = cmd_init
check = cmd_check

app.command("r", hidden=True)(cmd_run)
app.command("c", hidden=True)(cmd_compile)
app.command("comp", hidden=True)(cmd_compile)


# ---------------------------------------------------------------------------
# Entry point and error reporting
# ---------------------------------------------------------------------------


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Treat a bare filename as shorthand for ``qtr run``."""
    if argv and argv[0] not in _RESERVED_ARGV and not argv[0].startswith("-"):
        return ["run", *argv]
    return list(argv)


def _source_filename(args: Sequence[str]) -> str:
    """Return the source filename named in an argument list."""
    if not args:
        return ""
    if args[0] in _COMMAND_NAMES:
        return args[1] if len(args) > 1 else ""
    return args[0]


def _first_user_traceback(error: BaseException, full_traceback: bool):
    """Return the first traceback frame outside Quantar internals."""
    selected_tb = error.__traceback__
    if full_traceback or selected_tb is None:
        return selected_tb

    current = selected_tb
    while current is not None:
        raw_filename = current.tb_frame.f_code.co_filename
        is_virtual = raw_filename.startswith("<") and raw_filename.endswith(">")
        frame_path = Path(raw_filename).resolve()
        is_internal = frame_path == INTERNAL_RUNNER_FILE or INTERNAL_PACKAGE_DIR in frame_path.parents
        if is_virtual or not is_internal:
            return current
        current = current.tb_next
    return selected_tb


def _exception_text(error: BaseException, full_traceback: bool) -> list[str]:
    """Format an error for the user, hiding Quantar internal frames by default."""
    selected_tb = _first_user_traceback(error, full_traceback)
    formatted = traceback.TracebackException(type(error), error, selected_tb)
    return list(formatted.format(colorize=True))  # type: ignore[arg-type]


def _show_cli_error(error: BaseException) -> None:
    """Print a Typer usage error and exit with its status code."""
    show_error = getattr(error, "show", None)
    if show_error is not None:
        show_error()
    sys.exit(getattr(error, "exit_code", 1))


def main() -> None:
    """Console-script entry point for ``qtr``."""
    args = _normalize_argv(sys.argv[1:])
    try:
        app(args=args, standalone_mode=False)
    except KeyboardInterrupt:
        print(f"[PROCESS TERMINATED] Process from {_source_filename(args)} has been terminated")
    except typer.Exit:
        return
    except typer.TyperException as error:
        _show_cli_error(error)
    except Exception as error:
        for line in _exception_text(error, _FULL_TRACEBACK):
            sys.stderr.write(line)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
