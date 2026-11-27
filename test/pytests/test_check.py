"""Tests for cargo-style ``qtr check`` behavior."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lang_systems import runner
from lang_systems.checker import CheckIssue, QuantarCheckError


class CheckProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_checks_reachable_sources_without_executing_them(self) -> None:
        self.write("helper.qtr", "VALUE = 42\n")
        main = self.write("main.qtr", 'from helper import VALUE\nprint("SIDE EFFECT")\n')

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.qtr_check(main)

        self.assertEqual(result, main)
        self.assertIn("[CHECK] OK: 2 source file(s) checked", output.getvalue())
        self.assertNotIn("SIDE EFFECT", output.getvalue())

    def test_reports_syntax_error_in_nested_source(self) -> None:
        helper = self.write("helper.qtr", "def broken(:\n    pass\n")
        main = self.write("main.qtr", "from helper import VALUE\nprint('SIDE EFFECT')\n")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(QuantarCheckError) as caught:
                runner.qtr_check(main)

        self.assertTrue(any(str(helper) in issue.format() for issue in caught.exception.issues))
        self.assertNotIn("SIDE EFFECT", output.getvalue())

    def test_reports_missing_import(self) -> None:
        main = self.write("main.qtr", "import definitely_missing_qtr_module_xyz\n")

        with self.assertRaises(QuantarCheckError) as caught:
            runner.qtr_check(main)

        self.assertTrue(any("cannot resolve import" in issue.message for issue in caught.exception.issues))

    def test_reports_missing_exported_name(self) -> None:
        self.write("helper.py", "VALUE = 1\n")
        main = self.write("main.qtr", "from helper import MISSING_VALUE\n")

        with self.assertRaises(QuantarCheckError) as caught:
            runner.qtr_check(main)

        self.assertTrue(any("no exported name" in issue.message for issue in caught.exception.issues))

    def test_registers_local_operators_without_running_their_module(self) -> None:
        self.write(
            "ops.qtr",
            "from lang_systems.operators import operator\n"
            "@operator.binary('<~>')\n"
            "def reverse(left, right):\n"
            "    return right, left\n"
            'print("OPERATOR SIDE EFFECT")\n'
            "VALUE = 42\n",
        )
        main = self.write("main.qtr", "from ops import VALUE\na, b = 1, 2\nx = a <~> b\n")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.qtr_check(main)

        self.assertNotIn("OPERATOR SIDE EFFECT", output.getvalue())


class CheckCliTests(unittest.TestCase):
    def test_check_command_reports_cargo_style_errors(self) -> None:
        issue = CheckIssue(Path("main.qtr"), "cannot resolve import 'missing'", 3)
        errors = io.StringIO()

        argv = ["qtr", "check", "main.qtr"]
        with patch.object(runner.sys, "argv", argv):
            with patch.object(runner, "qtr_check", side_effect=QuantarCheckError([issue])):
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main()

        self.assertEqual(caught.exception.code, 1)
        self.assertIn("[CHECK] ERROR main.qtr:3: cannot resolve import 'missing'", errors.getvalue())
        self.assertIn("[CHECK] FAILED: 1 error(s)", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
