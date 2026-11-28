"""Keep the Python and Rust package versions in sync.

``src/lang_systems/__init__.py`` is the single source of truth. The Hatch build
hook calls :func:`sync_cargo_version` before every build, so ``uv build`` writes
the same version into ``rust/Cargo.toml`` and ``rust/Cargo.lock``.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PYTHON_VERSION_FILE = ROOT / "src" / "lang_systems" / "__init__.py"
CARGO_MANIFEST = ROOT / "rust" / "Cargo.toml"
CARGO_LOCK = ROOT / "rust" / "Cargo.lock"

_PYTHON_VERSION_RE = re.compile(r"^__version__\s*=\s*[\"'](?P<version>[^\"']+)[\"']", re.MULTILINE)
_CARGO_MANIFEST_VERSION_RE = re.compile(r"(?ms)^(?P<prefix>\[package\][\s\S]*?^version\s*=\s*)\"(?P<version>[^\"]+)\"")
_CARGO_LOCK_VERSION_RE = re.compile(r"(?ms)^(?P<prefix>\[\[package\]\]\s*\nname\s*=\s*\"quantar-kernels\"\s*\nversion\s*=\s*)\"(?P<version>[^\"]+)\"")


def _read(path: Path) -> str:
    """Read UTF-8 text without changing the file's line endings."""
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    """Write UTF-8 text without changing the file's line endings."""
    path.write_bytes(text.encode("utf-8"))


def _match_version(path: Path, pattern: re.Pattern[str], label: str) -> str:
    """Return the matched version or raise a helpful error."""
    match = pattern.search(_read(path))
    if match is None:
        raise RuntimeError(f"Could not find {label} in {path}")
    return match.group("version")


def source_version(root: Path = ROOT) -> str:
    """Return the canonical version from ``lang_systems/__init__.py``."""
    path = Path(root) / "src" / "lang_systems" / "__init__.py"
    return _match_version(path, _PYTHON_VERSION_RE, "__version__")


def cargo_toml_version(root: Path = ROOT) -> str:
    """Return the version currently in ``rust/Cargo.toml``."""
    path = Path(root) / "rust" / "Cargo.toml"
    return _match_version(path, _CARGO_MANIFEST_VERSION_RE, "[package].version")


def cargo_lock_version(root: Path = ROOT) -> str:
    """Return the version currently in ``rust/Cargo.lock``."""
    path = Path(root) / "rust" / "Cargo.lock"
    return _match_version(path, _CARGO_LOCK_VERSION_RE, "quantar-kernels version")


def sync_cargo_version(version: str | None = None, root: Path = ROOT) -> list[Path]:
    """Write the canonical version into Cargo metadata and return changed files."""
    target = source_version(root) if version is None else version
    changed: list[Path] = []

    manifest = Path(root) / "rust" / "Cargo.toml"
    if cargo_toml_version(root) != target:
        text = _read(manifest)
        updated, count = _CARGO_MANIFEST_VERSION_RE.subn(lambda match: f'{match.group("prefix")}"{target}"', text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not update [package].version in {manifest}")
        _write(manifest, updated)
        changed.append(manifest)

    lock = Path(root) / "rust" / "Cargo.lock"
    if cargo_lock_version(root) != target:
        text = _read(lock)
        updated, count = _CARGO_LOCK_VERSION_RE.subn(lambda match: f'{match.group("prefix")}"{target}"', text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not update quantar-kernels version in {lock}")
        _write(lock, updated)
        changed.append(lock)

    return changed


def assert_versions_match(root: Path = ROOT) -> None:
    """Raise when Cargo metadata no longer matches the Python version."""
    expected = source_version(root)
    actual = {"Cargo.toml": cargo_toml_version(root), "Cargo.lock": cargo_lock_version(root)}
    mismatches = [f"{name} has {version}, expected {expected}" for name, version in actual.items() if version != expected]
    if mismatches:
        raise RuntimeError("Version mismatch: " + "; ".join(mismatches))


def main() -> None:
    """Sync Cargo versions, or check them when called with ``--check``."""
    if "--check" in sys.argv[1:]:
        assert_versions_match()
        print(f"Versions match at {source_version()}")
        return

    changed = sync_cargo_version()
    if changed:
        print("Synced: " + ", ".join(str(path) for path in changed))
    else:
        print(f"Versions already match at {source_version()}")


if __name__ == "__main__":
    main()


__all__ = ["assert_versions_match", "cargo_lock_version", "cargo_toml_version", "source_version", "sync_cargo_version"]
