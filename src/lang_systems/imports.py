"""Import support for Quantar (``.qtr``) modules.

Python modules already import normally from ``.qtr`` programs.  This module
adds the other direction: a meta-path finder lets Python import ``foo.qtr`` as
``foo``, including ``__init__.qtr`` packages and relative imports.

Custom Quantar operators are registered while a module is *parsed*, not while
it is executed.  A program that writes ``a <~> b`` with ``<~>`` defined in an
imported module therefore needs those imports run before parsing.  Scanning
import statements out of the raw source and importing them in
:func:`preload_imports` makes that work without losing Quantar syntax.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import io
import sys
import tokenize
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Iterator

from lang_systems import ast_builder, runtime


def _line_starts(source: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    for line in source.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    return starts


def _offset(position: tuple[int, int], starts: list[int], source: str) -> int:
    row, column = position
    if row < 1 or row > len(starts):
        return len(source)
    return starts[row - 1] + column


def _iter_import_chunks(source: str) -> Iterator[str]:
    """Yield complete import statements from raw (possibly Quantar) source."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        return

    starts = _line_starts(source)
    chunks: list[str] = []
    first: tokenize.TokenInfo | None = None
    last: tokenize.TokenInfo | None = None
    depth = 0

    def emit() -> None:
        nonlocal first, last
        if first is not None and last is not None:
            chunks.append(source[_offset(first.start, starts, source): _offset(last.end, starts, source)])
        first = None
        last = None

    for item in tokens:
        if item.type in (tokenize.ENCODING, tokenize.COMMENT, tokenize.NL,
                         tokenize.INDENT, tokenize.DEDENT):
            continue
        if item.type == tokenize.ENDMARKER:
            break
        if item.type == tokenize.NEWLINE:
            emit()
            continue
        if item.type == tokenize.OP:
            if item.string in "([{":
                depth += 1
            elif item.string in ")]}":
                depth = max(0, depth - 1)
            elif item.string == ";" and depth == 0:
                emit()
                continue

        if first is None:
            first = item
        last = item

    emit()
    return iter(chunks)


def _iter_import_statements(source: str) -> Iterator[ast.Import | ast.ImportFrom]:
    for chunk in _iter_import_chunks(source):
        try:
            statement = ast.parse(chunk).body[0]
        except (IndexError, SyntaxError):
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            yield statement


def resolve_import_name(statement: ast.ImportFrom, package: str | None = None) -> str:
    """Turn an ``ImportFrom`` node into the absolute module name it means."""
    if not statement.level:
        return statement.module or ""

    parts = package.split(".") if package else []
    if statement.level > 1:
        drop = statement.level - 1
        if len(parts) < drop:
            raise ImportError(
                "attempted relative import beyond top-level package "
                f"(package={package!r}, level={statement.level})"
            )
        parts = parts[: len(parts) - drop]
    if statement.module:
        parts.extend(statement.module.split("."))
    return ".".join(parts)


def _import_submodule(module_name: str, child: str) -> None:
    """Import ``module.child`` when it exists, ignoring ordinary attributes."""
    candidate = f"{module_name}.{child}"
    try:
        import_module(candidate)
    except ModuleNotFoundError as error:
        if error.name != candidate:
            raise


def preload_imports(
    source: str,
    filename: str = "<quantar>",
    package: str | None = None,
) -> tuple[ModuleType, ...]:
    """Import modules named by ``source`` before Quantar parses it.

    This runs the imports for their side effects -- especially custom operator
    registration.  Python caches the modules, so the later ``exec`` of the
    program sees the same module objects.
    """
    install_import_hook()
    loaded: list[ModuleType] = []

    for statement in _iter_import_statements(source):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                loaded.append(import_module(alias.name))
            continue

        module_name = resolve_import_name(statement, package)
        if not module_name:
            continue
        loaded.append(import_module(module_name))
        for alias in statement.names:
            if alias.name != "*":
                _import_submodule(module_name, alias.name)

    return tuple(loaded)


class QuantarLoader(importlib.abc.Loader):
    """Loader that lowers a ``.qtr`` source file to executable Python."""

    def __init__(self, path: str | Path, package: str | None = None):
        self.path = Path(path).resolve()
        self.package = package

    def create_module(self, spec):
        return None

    def get_source(self, fullname: str) -> str:
        return self.path.read_text(encoding="utf-8")

    def get_code(self, fullname: str):
        source = self.get_source(fullname)
        preload_imports(source, str(self.path), self.package)
        tree = ast_builder.parse_ast(source, str(self.path))
        return ast_builder.compile_ast(tree, str(self.path))

    def exec_module(self, module: ModuleType) -> None:
        source = self.get_source(module.__name__)
        package = module.__package__ or None
        preload_imports(source, str(self.path), package)
        tree = ast_builder.parse_ast(source, str(self.path))
        code = ast_builder.compile_ast(tree, str(self.path))
        module.__dict__.update(runtime.runtime_globals())
        exec(code, module.__dict__)

    def is_package(self, fullname: str) -> bool:
        return self.path.name == "__init__.qtr"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.path)!r})"


class QuantarFinder(importlib.abc.MetaPathFinder):
    """Find ``foo.qtr`` modules and ``foo/__init__.qtr`` packages on ``sys.path``."""

    def find_spec(self, fullname, path=None, target=None):
        leaf = fullname.rpartition(".")[2]
        if not leaf:
            return None

        search_paths = path if path is not None else sys.path
        for entry in search_paths:
            base = Path(entry) if entry else Path.cwd()
            package_init = base / leaf / "__init__.qtr"
            if package_init.is_file():
                loader = QuantarLoader(package_init, fullname)
                return importlib.util.spec_from_file_location(
                    fullname,
                    package_init,
                    loader=loader,
                    submodule_search_locations=[str(package_init.parent)],
                )

            module_path = base / f"{leaf}.qtr"
            if module_path.is_file():
                loader = QuantarLoader(module_path, fullname.rpartition(".")[0])
                return importlib.util.spec_from_file_location(
                    fullname, module_path, loader=loader
                )

        return None


def install_import_hook() -> QuantarFinder:
    """Install the ``.qtr`` finder once and return it."""
    for finder in sys.meta_path:
        if isinstance(finder, QuantarFinder):
            return finder
    finder = QuantarFinder()
    sys.meta_path.append(finder)  # after PathFinder, so .py/.pyd still win
    return finder


def uninstall_import_hook() -> None:
    """Remove all installed :class:`QuantarFinder` instances."""
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, QuantarFinder)]


def import_hook_installed() -> bool:
    return any(isinstance(finder, QuantarFinder) for finder in sys.meta_path)


__all__ = [
    "QuantarFinder",
    "QuantarLoader",
    "install_import_hook",
    "uninstall_import_hook",
    "import_hook_installed",
    "preload_imports",
    "resolve_import_name",
]
