"""Tests for ``qtr new``, ``qtr init``, and project config defaults."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lang_systems import runner


class ProjectScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._directory.name)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._directory.cleanup()

    def test_new_creates_pyproject_scaffold(self) -> None:
        root = runner.qtr_new("demo")

        self.assertEqual(root.name, "demo")
        manifest = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "demo"', manifest)
        self.assertIn('dependencies = ["quantar"]', manifest)
        self.assertIn("[tool.quantar]", manifest)
        self.assertIn('entry = "src/main.qtr"', manifest)
        self.assertTrue((root / "src" / "main.qtr").is_file())
        self.assertTrue((root / "README.md").is_file())

        with contextlib.redirect_stdout(io.StringIO()):
            runner.qtr_check(root / "src" / "main.qtr")

    def test_new_rejects_non_empty_directory(self) -> None:
        root = Path("taken")
        root.mkdir()
        (root / "existing.txt").write_text("keep me", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            runner.qtr_new("taken")
        self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "keep me")

    def test_init_does_not_overwrite_existing_files(self) -> None:
        entry = Path("src") / "main.qtr"
        entry.parent.mkdir()
        entry.write_text("print('mine')\n", encoding="utf-8")

        root = runner.qtr_init()

        self.assertEqual(root, Path.cwd())
        self.assertEqual(entry.read_text(encoding="utf-8"), "print('mine')\n")
        self.assertTrue((root / "pyproject.toml").is_file())

    def test_init_extends_existing_pyproject(self) -> None:
        manifest = Path("pyproject.toml")
        original = '[project]\nname = "mine"\nversion = "1.2.3"\n'
        manifest.write_text(original, encoding="utf-8")

        runner.qtr_init()

        text = manifest.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(original))
        self.assertIn("[tool.quantar]", text)
        self.assertIn('entry = "src/main.qtr"', text)

    def test_invalid_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.qtr_new("../escape")


class ProjectConfigCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._directory.name)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._directory.cleanup()

    def make_project(self) -> Path:
        root = runner.qtr_new("demo")
        os.chdir(root)
        return root

    def replace_setting(self, name: str, value: str) -> None:
        manifest = Path("pyproject.toml")
        lines = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name} ="):
                line = f"{name} = {value}"
            lines.append(line)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_plain_run_uses_configured_entry(self) -> None:
        root = self.make_project()

        with patch.object(runner.sys, "argv", ["qtr", "run"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()

        run.assert_called_once_with(root / "src" / "main.qtr", rust=False)

    def test_plain_compile_uses_configured_entry_and_output(self) -> None:
        root = self.make_project()

        with patch.object(runner.sys, "argv", ["qtr", "compile"]):
            with patch.object(runner, "qtr_runner") as run:
                with patch.object(runner, "qtr_compile") as compile_bin:
                    runner.main()

        run.assert_called_once_with(root / "src" / "main.qtr", root / "src" / "main.py", rust=False)
        compile_bin.assert_not_called()

    def test_plain_check_uses_configured_entry(self) -> None:
        root = self.make_project()

        with patch.object(runner.sys, "argv", ["qtr", "check"]):
            with patch.object(runner, "qtr_check") as check:
                runner.main()

        check.assert_called_once_with(root / "src" / "main.qtr")

    def test_configured_rust_and_flag_override(self) -> None:
        self.make_project()
        self.replace_setting("rust", "true")

        with patch.object(runner.sys, "argv", ["qtr", "run"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        self.assertTrue(run.call_args.kwargs["rust"])

        with patch.object(runner.sys, "argv", ["qtr", "run", "--no-rust"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        self.assertFalse(run.call_args.kwargs["rust"])

    def test_plain_binary_compile_uses_all_config_settings(self) -> None:
        root = self.make_project()
        self.replace_setting("compile-to", '"bin"')
        self.replace_setting("output", '"dist/app"')
        self.replace_setting("mode", '"onefile"')
        self.replace_setting("jobs", "5")
        self.replace_setting("rust", "true")

        with patch.object(runner.sys, "argv", ["qtr", "compile"]):
            with patch.object(runner, "qtr_compile", return_value=Path("app.exe")) as compile_bin:
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.main()

        compile_bin.assert_called_once_with(root / "src" / "main.qtr", root / "dist" / "app", mode="onefile", jobs=5, rust=True)


class ProjectCliTests(unittest.TestCase):
    def test_command_aliases(self) -> None:
        self.assertIs(runner.cmd_new, runner.new)
        self.assertIs(runner.cmd_init, runner.init)

    def test_new_command(self) -> None:
        output = io.StringIO()
        with patch.object(runner.sys, "argv", ["qtr", "new", "demo"]):
            with patch.object(runner, "qtr_new", return_value=Path("demo")) as new_project:
                with contextlib.redirect_stdout(output):
                    runner.main()
        new_project.assert_called_once_with("demo", rust=False)
        self.assertIn("[NEW] Created project demo", output.getvalue())

    def test_init_command(self) -> None:
        output = io.StringIO()
        with patch.object(runner.sys, "argv", ["qtr", "init"]):
            with patch.object(runner, "qtr_init", return_value=Path(".")) as init_project:
                with contextlib.redirect_stdout(output):
                    runner.main()
        init_project.assert_called_once_with(None, rust=False)
        self.assertIn("[INIT] Initialized project .", output.getvalue())


if __name__ == "__main__":
    unittest.main()
