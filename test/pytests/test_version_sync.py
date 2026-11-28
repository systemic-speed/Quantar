"""Tests for single-source version synchronization."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import version_sync


class VersionSyncTests(unittest.TestCase):
    def test_repository_versions_match(self) -> None:
        version = version_sync.source_version()
        self.assertRegex(version, r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
        self.assertEqual(version_sync.cargo_toml_version(), version)
        self.assertEqual(version_sync.cargo_lock_version(), version)
        version_sync.assert_versions_match()

    def test_sync_updates_only_cargo_package_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "lang_systems").mkdir(parents=True)
            (root / "rust").mkdir()
            (root / "src" / "lang_systems" / "__init__.py").write_text('__version__ = "9.8.7"\n', encoding="utf-8")
            manifest = root / "rust" / "Cargo.toml"
            manifest.write_text(
                '[package]\nname = "quantar-kernels"\nversion = "1.0.0"\n\n[dependencies]\npyo3 = { version = "0.30" }\n',
                encoding="utf-8",
            )
            lock = root / "rust" / "Cargo.lock"
            lock.write_text('[[package]]\nname = "quantar-kernels"\nversion = "1.0.0"\n', encoding="utf-8")

            changed = version_sync.sync_cargo_version(root=root)

            self.assertEqual(changed, [manifest, lock])
            self.assertEqual(version_sync.cargo_toml_version(root), "9.8.7")
            self.assertEqual(version_sync.cargo_lock_version(root), "9.8.7")
            self.assertIn('pyo3 = { version = "0.30" }', manifest.read_text(encoding="utf-8"))
            version_sync.assert_versions_match(root)


if __name__ == "__main__":
    unittest.main()
