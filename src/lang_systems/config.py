"""Read Quantar settings from ``pyproject.toml``.

A project created by ``qtr new`` or ``qtr init`` contains a
``[tool.quantar]`` table. The CLI reads it so commands such as ``qtr run`` and
``qtr compile`` can work without repeating the entry path and common options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

Source = str | Path

_COMPILE_TARGETS = {"py", "python", "bin", "binary"}
_PYINSTALLER_MODES = {"onefile", "standalone"}


@dataclass(frozen=True)
class QuantarConfig:
    """Resolved Quantar project settings."""

    root: Path
    name: str | None
    entry_path: Path
    output_path: Path | None
    rust: bool
    compile_to: str
    mode: str
    jobs: int


def _setting(table: dict[str, Any], *names: str) -> Any:
    """Return the first present setting from a TOML table."""
    for name in names:
        if name in table:
            return table[name]
    return None


def _resolve_path(root: Path, value: Any, label: str) -> Path:
    """Resolve a config path relative to the project root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[tool.quantar].{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: Source) -> QuantarConfig | None:
    """Load a project config from one ``pyproject.toml`` file."""
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError:
        return None
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML in {config_path}: {error}") from error

    tool = document.get("tool")
    if not isinstance(tool, dict) or not isinstance(tool.get("quantar"), dict):
        return None
    table = tool["quantar"]

    entry_value = _setting(table, "entry", "entrypoint", "main")
    if entry_value is None:
        raise ValueError(f"[tool.quantar] in {config_path} is missing 'entry'")
    entry_path = _resolve_path(config_path.parent, entry_value, "entry")
    if entry_path.suffix not in {".qtr", ".py"}:
        raise ValueError(f"[tool.quantar].entry must be a .qtr or .py file, got {entry_path.suffix!r}")
    if not entry_path.is_file():
        raise FileNotFoundError(f"Quantar entry file not found: {entry_path}")

    output_value = _setting(table, "output", "out")
    output_path = _resolve_path(config_path.parent, output_value, "output") if output_value is not None else None

    compile_to = str(_setting(table, "compile-to", "compile_to") or "py").lower()
    if compile_to not in _COMPILE_TARGETS:
        raise ValueError(f"[tool.quantar].compile-to must be py, python, bin, or binary, got {compile_to!r}")

    mode = str(_setting(table, "mode") or "standalone").lower()
    if mode not in _PYINSTALLER_MODES:
        raise ValueError(f"[tool.quantar].mode must be onefile or standalone, got {mode!r}")

    rust = _setting(table, "rust", False) #type: ignore
    if not isinstance(rust, bool):
        raise ValueError("[tool.quantar].rust must be a boolean")

    jobs = _setting(table, "jobs", 2) #type: ignore
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("[tool.quantar].jobs must be a positive integer")

    project = document.get("project")
    name = project.get("name") if isinstance(project, dict) else None
    return QuantarConfig(
        root=config_path.parent,
        name=name if isinstance(name, str) else None,
        entry_path=entry_path,
        output_path=output_path,
        rust=rust,
        compile_to=compile_to,
        mode=mode,
        jobs=jobs,
    )


def find_config(start: Source | None = None) -> QuantarConfig | None:
    """Search upward from ``start`` for a ``pyproject.toml`` with Quantar settings."""
    directory = Path.cwd().resolve() if start is None else Path(start).resolve()
    if directory.is_file():
        directory = directory.parent
    for candidate in (directory, *directory.parents):
        config = load_config(candidate / "pyproject.toml")
        if config is not None:
            return config
    return None


__all__ = ["QuantarConfig", "find_config", "load_config"]
