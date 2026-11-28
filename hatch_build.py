"""Hatch build hook.

Quantar's parser is generated from the vendored CPython grammar
(``Grammar/python.gram``) plus the operator overlay
(``src/lang_systems/quantar.gram``); see ``src/lang_systems/pegen_gram_gen.py``
for the details and the pegen-0.3.0 workarounds. This hook also syncs the Cargo
version from ``src/lang_systems/__init__.py`` so there is only one version to
edit before each build.
"""

import os
import sys
import urllib.request
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from version_sync import sync_cargo_version

_TRUTHY = {"1", "true", "yes", "on"}


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return _TRUTHY.__contains__(value) if value else default


class PegenBuildHook(BuildHookInterface):
    """Generate the Quantar parser before building."""

    PLUGIN_NAME = "pegen"

    def initialize(self, version, build_data):
        sync_cargo_version()
        self._configure_rust_output()
        self._fetch_python_grammar()
        self._generate_parser()

    def _debug(self, message: str) -> None:
        """Print build detail only when the caller asks for it."""
        if _enabled("QUANTAR_VERBOSE_BUILD"):
            print(message)

    def _configure_rust_output(self) -> None:
        """Keep normal builds quiet; make detail opt-in.

        `hatch-maturin-build` streams cargo output only when its own args say so,
        so Hatch's global verbosity is left alone. ``QUANTAR_VERBOSE_BUILD=1``
        is a friendlier alias for ``HATCH_MATURIN_ARGS="-vv"``.
        """
        if _enabled("QUANTAR_VERBOSE_BUILD") and "HATCH_MATURIN_ARGS" not in os.environ:
            os.environ["HATCH_MATURIN_ARGS"] = "-vv"

        # num-traits 0.2.19 uses deprecated `core::f32::NAN`-style constants.
        # They come from a dependency and are noisy on every build; allow them
        # unless the caller explicitly asks for strict warnings.
        if _enabled("QUANTAR_STRICT_RUST"):
            return

        flags = os.environ.get("RUSTFLAGS", "").strip()
        if "-A deprecated" not in flags:
            os.environ["RUSTFLAGS"] = f"{flags} -A deprecated".strip()

    def _fetch_python_grammar(self):
        """Download the CPython grammar matching the installer's Python version."""
        grammar_dir = ROOT / "Grammar"
        grammar_dir.mkdir(exist_ok=True)
        target = grammar_dir / "python.gram"

        if target.exists() and target.stat().st_size > 0:
            self._debug(f"Using existing {target}")
            return

        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        base = "https://raw.githubusercontent.com/python/cpython"
        for ref in (version, "main"):
            url = f"{base}/{ref}/Grammar/python.gram"
            temporary = target.with_suffix(".gram.tmp")
            self._debug(f"Fetching CPython grammar from {url}")
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    temporary.write_bytes(response.read())
                temporary.replace(target)  # atomic: never leave a partial file
                self._debug(f"Downloaded {target} ({target.stat().st_size} bytes)")
                return
            except Exception as error:  # noqa: BLE001
                temporary.unlink(missing_ok=True)
                self._debug(f"Could not fetch {url}: {error}")

        raise RuntimeError(
            "Could not download Grammar/python.gram and no vendored copy exists."
        )

    def _generate_parser(self):
        # The build hook runs in an isolated environment where the source tree
        # is not necessarily on sys.path, so make our own modules importable.
        source = str(ROOT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)

        from lang_systems.pegen_gram_gen import build

        build()
