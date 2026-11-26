from . import ast_builder
from .imports import find_qtr_module, install_import_hook, iter_import_statements, preload_imports, resolve_import_name
from .runtime import runtime_globals
from .to_bin import RUST_PRELUDE, make_bin_for_file
from pathlib import Path
import argparse
import ast
import os
from ast import unparse
import sys
from typing import Literal, Sequence

INTERNAL_RUNNER_FILE = Path(__file__).resolve()
INTERNAL_PACKAGE_DIR = INTERNAL_RUNNER_FILE.parent.parent.resolve()


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


def _uses_quantar_apply(tree: ast.AST) -> bool:
    """Return whether lowering produced a ``__quantar_apply__`` call."""
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__quantar_apply__" for node in ast.walk(tree))


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
            names.extend(f"{base}.{alias.name}" for alias in statement.names if alias.name != "*")
        if any(find_qtr_module(name, [root]) is not None for name in names):
            return True
    return False


def _inject_runtime_imports(tree: ast.AST, rust: bool = False, source_code: str | None = None, source_path: str | Path | None = None, package: str | None = None) -> ast.Module:
    """Add only the helpers the lowered program actually needs."""
    lowered = ast_builder.lower_ast(tree)
    assert isinstance(lowered, ast.Module)

    helper_lines: list[str] = []
    if source_code is None or _uses_quantar_apply(lowered):
        helper_lines.append("from lang_systems.runtime import apply_quantar_operator as __quantar_apply__")
    if source_code is None or source_path is None or _needs_import_hook(source_code, Path(source_path), package or ""):
        helper_lines.extend(("from lang_systems.imports import install_import_hook as __quantar_install_import_hook__", "__quantar_install_import_hook__()"))
    if rust:
        helper_lines.append(RUST_PRELUDE.rstrip())
    if not helper_lines:
        return lowered

    index = 0
    if (
        lowered.body
        and isinstance(lowered.body[0], ast.Expr)
        and isinstance(lowered.body[0].value, ast.Constant)
        and isinstance(lowered.body[0].value.value, str)
    ):
        index = 1
    for position, statement in enumerate(lowered.body):
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            index = max(index, position + 1)

    helpers = ast.parse("\n".join(helper_lines) + "\n").body
    lowered.body[index:index] = helpers
    ast.fix_missing_locations(lowered)
    return lowered

def qtr_compile(filename: str | Path, output_path: str | Path | None = None, mode: Literal["onefile", "standalone"] = "standalone", jobs: int = 2, *, rust: bool = False) -> Path:
    """Compile a Quantar (or lowered Python) file to a PyInstaller executable.

    ``rust`` loads/builds the Rust extension and selects it inside the generated
    binary. A missing or incompatible extension is an error, not a silent
    fallback to NumPy.
    """
    path = Path(filename).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Quantar source file not found: {path}")
    if path.suffix not in {".qtr", ".py"}:
        raise ValueError(f"Expected a .qtr or .py file, got {path.suffix!r}")
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")

    return make_bin_for_file(path, output_path if output_path is not None else "./quantar_compile/bins", mode=mode, jobs=jobs, rust=rust)


# Backwards-compatible name for callers already using the old spelling.
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


def qtr_runner(filename: str | Path, output_path: str | Path | None = None, *, rust: bool = False):
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Quantar source file not found: {path}")
    if path.suffix not in {".qtr", ".py"}:
        raise ValueError(f"Expected a .qtr or .py file, got {path.suffix!r}")

    if rust:
        # Must happen before preload_imports() imports ``quantum``.
        _enable_rust_backend()

    install_import_hook()
    package = _package_for(path)
    _add_source_dir(path)

    source_code = path.read_text(encoding="utf-8")
    preload_imports(source_code, str(path), package)

    tree = ast_builder.parse_ast(source_code, str(path))

    if output_path:
        lowered = _inject_runtime_imports(tree, rust, source_code, path, package)
        final_code = unparse(lowered) + "\n"

        out_path = Path(output_path)
        out_path.write_text(final_code, "utf-8")
        print(f"[COMPILE] Compiled to {out_path}")

    else:
        python_ast = ast_builder.compile_ast(tree, str(path))
        globals_dict = {
            "__name__": "__main__",
            "__file__": str(path),
            "__package__": package,
            **runtime_globals(),
        }
        exec(python_ast, globals_dict)

def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Treat a bare filename as shorthand for ``qtr run``."""
    if argv and argv[0] not in {"run", "r", "compile", "c", "comp", "-h", "--help"}:
        return ["run", *argv]
    return list(argv)


def main():
    parser = argparse.ArgumentParser(prog="qtr", description="Run or compile Quantar (.qtr) programs", epilog="A bare filename is shorthand for 'qtr run'.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", aliases=["r"], help="Run a Quantar file")
    compile = subparsers.add_parser("compile", aliases=["c", "comp"], help="Lower to .py or build a binary")

    for command in (run, compile):
        command.add_argument("filename", help="The .qtr or .py file to process")
        command.add_argument("--rust", action="store_true", help="Require and use the Rust quantum backend and Rust physics")
        command.add_argument("--full-traceback", action="store_true", help="Show the complete traceback, including internal frames")

    compile.add_argument("output_path", help="Where to write the .py file or binary")
    compile.add_argument("--compile-to", "-c", choices=("py", "python", "bin", "binary"), default="py", help="Compile target (default: py)")
    compile.add_argument("--mode", choices=("onefile", "standalone"), default="standalone", help="PyInstaller output mode (default: standalone)")
    compile.add_argument("--jobs", type=int, default=2, help="Kept for compatibility; ignored by PyInstaller")
    args = parser.parse_args(_normalize_argv(sys.argv[1:]))

    try:
        if args.command in {"run", "r"}:
            qtr_runner(args.filename, rust=args.rust)
        elif args.compile_to in {"py", "python"}:
            qtr_runner(args.filename, args.output_path, rust=args.rust)
        else:
            artifact = qtr_compile(args.filename, args.output_path, mode=args.mode, jobs=args.jobs, rust=args.rust)
            print(f"[BINARY] Compiled to {artifact}")
    except KeyboardInterrupt:
        print(f"[PROCESS TERMINATED] Process from {args.filename} has been terminated")
    except Exception as e:
        import traceback

        selected_tb = e.__traceback__

        if not getattr(args, "full_traceback", False) and selected_tb is not None:
            current = selected_tb
            while current is not None:
                raw_filename = current.tb_frame.f_code.co_filename
                
                # Instantly catch virtual frames generated via exec() or eval()
                if raw_filename.startswith("<") and raw_filename.endswith(">"):
                    selected_tb = current
                    break

                # Evaluate actual file system boundaries explicitly
                frame_path = Path(raw_filename).resolve()
                is_internal = (frame_path == INTERNAL_RUNNER_FILE or INTERNAL_PACKAGE_DIR in frame_path.parents)

                if not is_internal:
                    selected_tb = current
                    break

                current = current.tb_next

        # Instantiated with the exact slice found, avoiding global state mutation
        te = traceback.TracebackException(type(e), e, selected_tb)
        for line in te.format(colorize=True): #type: ignore
            sys.stderr.write(line)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()