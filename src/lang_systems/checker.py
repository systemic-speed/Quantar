"""Static project checking for Quantar programs.

This module powers ``qtr check``. It walks the local module graph, parses and
compiles every reachable source file, and reports errors without executing any
of the checked code. The CLI stays thin; editor extensions can also call
:func:`check_project` directly and format :class:`CheckIssue` objects however
they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import ast
import importlib.util
import linecache
import tokenize

from . import ast_builder
from .imports import iter_import_statements, resolve_import_name
from .operators import operator as operator_decorator

_SOURCE_SUFFIXES = {".qtr", ".py"}


@dataclass(frozen=True)
class CheckIssue:
    """A single problem found while checking a project."""

    path: Path
    message: str
    line: int | None = None

    def format(self) -> str:
        """Return a cargo-style ``file:line: message`` description."""
        location = str(self.path)
        if self.line is not None:
            location = f"{location}:{self.line}"
        return f"{location}: {self.message}"


class QuantarCheckError(Exception):
    """Raised when a static check finds one or more errors."""

    def __init__(self, issues: Sequence[CheckIssue]):
        self.issues = tuple(issues)
        super().__init__(f"{len(self.issues)} check error(s)")


@dataclass(frozen=True)
class CheckReport:
    """Summary of a completed static check."""

    checked: tuple[Path, ...]
    issues: tuple[CheckIssue, ...]


# ---------------------------------------------------------------------------
# Module resolution
# ---------------------------------------------------------------------------


def _check_roots(path: Path) -> list[Path]:
    """Return search roots for resolving a source file's local imports."""
    directory = path.parent.resolve()
    roots = [directory]
    while (directory / "__init__.qtr").is_file() or (directory / "__init__.py").is_file():
        directory = directory.parent
        if directory not in roots:
            roots.append(directory)

    cwd = Path.cwd().resolve()
    if cwd not in roots:
        try:
            path.resolve().relative_to(cwd)
        except ValueError:
            pass
        else:
            roots.append(cwd)
    return roots


def _module_name_for(path: Path, roots: Sequence[Path]) -> str:
    """Derive a dotted module name by using the longest matching root."""
    resolved = path.resolve()
    best: list[str] = []
    for root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts and len(parts) > len(best):
            best = parts
    return ".".join(best) if best else resolved.stem


def _local_module_path(module_name: str, roots: Sequence[Path]) -> Path | None:
    """Return the local ``.qtr`` or ``.py`` file for ``module_name``."""
    if not module_name:
        return None
    parts = module_name.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        candidates = (base.with_suffix(".qtr"), base.with_suffix(".py"), base / "__init__.qtr", base / "__init__.py")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _package_for_source(path: Path, roots: Sequence[Path]) -> str:
    """Return the package name to use for a source file's relative imports."""
    module_name = _module_name_for(path, roots)
    if path.name.startswith("__init__."):
        return module_name
    return module_name.rpartition(".")[0]


def _parent_modules(module_name: str) -> list[str]:
    """Return the parent packages named by a dotted module path."""
    parts = module_name.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts))]


def _import_line(source: str, needle: str) -> int | None:
    """Find the source line that mentions ``needle`` for error locations."""
    for number, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return number
    return None


def _module_outside_local_exists(module_name: str) -> bool:
    """Return whether the top-level name is importable outside local sources."""
    top_level = module_name.split(".")[0]
    if not top_level:
        return False
    try:
        return importlib.util.find_spec(top_level) is not None
    except (ImportError, ValueError):
        return False


def _resolve_required_module(module_name: str, path: Path, roots: Sequence[Path], issues: list[CheckIssue], line: int | None = None) -> Path | None:
    """Resolve a required module, recording an issue when it cannot be found."""
    local_path = _local_module_path(module_name, roots)
    if local_path is not None:
        return local_path

    top_level_local = _local_module_path(module_name.split(".")[0], roots)
    if top_level_local is not None or not _module_outside_local_exists(module_name):
        issues.append(CheckIssue(path, f"cannot resolve import {module_name!r}", line))
    return None


# ---------------------------------------------------------------------------
# Import and export analysis
# ---------------------------------------------------------------------------


def _assigned_names(target: ast.expr) -> set[str]:
    """Return the names bound by an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assigned_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    return set()


def _exported_names(path: Path) -> set[str] | None:
    """Return top-level exported names, or ``None`` when exports are dynamic."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast_builder.parse_ast(source, str(path))
    except Exception:
        return None

    names: set[str] = set()
    dynamic = False

    def collect(statements: Sequence[ast.stmt]) -> None:
        nonlocal dynamic
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(statement.name)
                if statement.name == "__getattr__":
                    dynamic = True
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    names.update(_assigned_names(target))
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                names.update(_assigned_names(statement.target))
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in statement.names):
                    dynamic = True
                for alias in statement.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(statement, ast.If):
                collect(statement.body)
                collect(statement.orelse)
            elif isinstance(statement, ast.Try):
                collect(statement.body)
                collect(statement.orelse)
                collect(statement.finalbody)
                for handler in statement.handlers:
                    collect(handler.body)
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                collect(statement.body)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                called = statement.value.func
                if isinstance(called, ast.Name) and called.id in {"setattr", "globals", "locals"}:
                    dynamic = True

    collect(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__all__":
            dynamic = True

    if dynamic:
        return None
    return names


def _analyze_imports(source: str, path: Path, roots: Sequence[Path], package: str, issues: list[CheckIssue]) -> list[Path]:
    """Resolve imports and return local dependencies that still need checking."""
    dependencies: list[Path] = []

    for statement in iter_import_statements(source):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                found = _resolve_required_module(alias.name, path, roots, issues, _import_line(source, alias.name))
                if found is not None:
                    dependencies.append(found)
                for parent in _parent_modules(alias.name):
                    parent_path = _local_module_path(parent, roots)
                    if parent_path is not None:
                        dependencies.append(parent_path)
            continue

        try:
            base = resolve_import_name(statement, package or None)
        except ImportError as error:
            issues.append(CheckIssue(path, str(error), _import_line(source, "import")))
            continue
        if not base:
            continue

        base_path = _resolve_required_module(base, path, roots, issues, _import_line(source, base))
        if base_path is not None:
            dependencies.append(base_path)
        for parent in _parent_modules(base):
            parent_path = _local_module_path(parent, roots)
            if parent_path is not None:
                dependencies.append(parent_path)

        for alias in statement.names:
            if alias.name == "*":
                continue
            child_path = _local_module_path(f"{base}.{alias.name}", roots)
            if child_path is not None:
                dependencies.append(child_path)
                continue
            if base_path is None:
                continue
            exported = _exported_names(base_path)
            if exported is not None and alias.name not in exported:
                issues.append(CheckIssue(path, f"module {base!r} has no exported name {alias.name!r}", _import_line(source, alias.name)))

    return dependencies


# ---------------------------------------------------------------------------
# Syntax checking
# ---------------------------------------------------------------------------


def _cache_source(source_name: str, source_code: str) -> None:
    """Keep source lines available to reported errors."""
    source_lines = source_code.splitlines(keepends=True)
    linecache.cache[source_name] = (len(source_code), None, source_lines, source_name)


def _operator_declaration(decorator: ast.expr) -> tuple[str | None, str]:
    """Return the operator name and kind from an ``@operator`` decorator."""
    if not isinstance(decorator, ast.Call):
        return None, "callable"

    name: str | None = None
    kind = "callable"
    function = decorator.func

    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "operator":
        kind = function.attr
    elif isinstance(function, ast.Name) and function.id == "operator":
        for keyword in decorator.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                name = keyword.value.value
            elif keyword.arg == "kind" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                kind = keyword.value.value

    if name is None and decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str):
        name = decorator.args[0].value
    return name, kind


def _register_placeholder_operator(name: str, kind: str) -> None:
    """Register a parse-only stand-in for an operator declared in local source."""
    if kind in {"matrix", "gate"}:
        def placeholder(): #type: ignore
            import numpy as np
            return np.zeros((2, 2), dtype=complex)
    else:
        def placeholder(*args, **kwargs):
            return None
    operator_decorator(name, kind)(placeholder) #type: ignore


def _register_declared_operators(source: str) -> None:
    """Register operators declared in source without executing the module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        for decorator in getattr(node, "decorator_list", ()):
            name, kind = _operator_declaration(decorator)
            if name:
                _register_placeholder_operator(name, kind)


def _discover_sources(entry: Path, roots: Sequence[Path], issues: list[CheckIssue]) -> list[Path]:
    """Walk reachable local sources without importing or executing them."""
    pending = [entry.resolve()]
    seen: set[Path] = set()
    sources: list[Path] = []

    while pending:
        path = pending.pop(0)
        if path in seen or path.suffix not in _SOURCE_SUFFIXES:
            continue
        seen.add(path)
        sources.append(path)

        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            issues.append(CheckIssue(path, f"cannot read source: {error}"))
            continue

        package = _package_for_source(path, roots)
        for dependency in _analyze_imports(source, path, roots, package, issues):
            if dependency not in seen:
                pending.append(dependency)

    return sources


def _check_source(path: Path) -> None:
    """Parse, lower, and compile one source file without executing it."""
    source = path.read_text(encoding="utf-8")
    source_name = str(path)
    _cache_source(source_name, source)
    tree = ast_builder.parse_ast(source, source_name)
    ast_builder.compile_ast(tree, source_name)


def check_project(entry: str | Path) -> CheckReport:
    """Check an entry point and every reachable local source file."""
    path = Path(entry).resolve()
    roots = _check_roots(path)
    issues: list[CheckIssue] = []
    sources = _discover_sources(path, roots, issues)

    # Operator decorators have to be known before Quantar parses custom syntax.
    for source_path in sources:
        try:
            _register_declared_operators(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue

    for source_path in sources:
        try:
            _check_source(source_path)
        except SyntaxError as error:
            issues.append(CheckIssue(source_path, error.msg or str(error), error.lineno))
        except tokenize.TokenError as error:
            message, position = error.args if len(error.args) == 2 else (str(error), None)
            line = position[0] if isinstance(position, tuple) and position else None
            issues.append(CheckIssue(source_path, f"invalid syntax: {message}", line))
        except Exception as error:
            issues.append(CheckIssue(source_path, f"{type(error).__name__}: {error}"))

    return CheckReport(checked=tuple(sources), issues=tuple(issues))


__all__ = [
    "CheckIssue",
    "CheckReport",
    "QuantarCheckError",
    "check_project",
]
