"""Regression tests for bugs found during the broad cleanup pass."""

from __future__ import annotations

import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from lang_systems import OperatorRegistry, ast_builder, operators, runner, runtime, to_bin
from lang_systems.checker import QuantarCheckError, check_project
from lang_systems.config import QuantarConfig
from lang_systems.imports import _module_name_for, iter_import_statements
from lang_systems.to_bin import RUST_PRELUDE_SENTINEL


class ImportRegressionTests(unittest.TestCase):
    def test_scanner_ignores_indented_imports(self) -> None:
        source = (
            "import always\n"
            "try:\n"
            "    import optional_missing_xyz\n"
            "except ImportError:\n"
            "    pass\n"
        )
        statements = list(iter_import_statements(source))
        self.assertEqual(len(statements), 1)
        self.assertEqual(statements[0].names[0].name, "always")

    def test_guarded_missing_import_is_not_preloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main = root / "main.qtr"
            main.write_text(
                "try:\n"
                "    import definitely_missing_regression_module\n"
                "except ImportError:\n"
                "    HAS = False\n"
                "else:\n"
                "    HAS = True\n"
                "print(HAS)\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner.qtr_runner(main)
            self.assertEqual(output.getvalue().strip(), "False")

    def test_module_name_prefers_specific_root(self) -> None:
        path = Path("C:/repo/src/main.qtr")
        self.assertEqual(_module_name_for(path, [Path("C:/repo/src"), Path("C:/repo")]), "main")
        self.assertEqual(_module_name_for(Path("C:/repo/pkg/main.qtr"), [Path("C:/repo")]), "pkg.main")


class CheckRegressionTests(unittest.TestCase):
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

    def test_guarded_missing_import_is_not_reported(self) -> None:
        main = self.write(
            "main.qtr",
            "try:\n"
            "    import definitely_missing_regression_module\n"
            "except ImportError:\n"
            "    HAS = False\n"
            "print(HAS)\n",
        )
        report = check_project(main)
        self.assertEqual(report.issues, ())

    def test_operator_declaration_in_quantar_source_is_registered(self) -> None:
        self.write(
            "ops.qtr",
            "from lang_systems.operators import operator\n"
            "@operator.binary('<~regress~>')\n"
            "def reverse(left, right):\n"
            "    return right, left\n"
            "x = 1 <-> 2\n",
        )
        main = self.write("main.qtr", "from ops import reverse\nx = 1 <~regress~> 2\n")
        report = check_project(main)
        self.assertEqual(report.issues, ())

    def test_invalid_operator_kind_becomes_check_issue(self) -> None:
        main = self.write("main.qtr", "from lang_systems.operators import operator\n@operator.bogus_kind('x')\ndef f():\n    return 1\n")
        report = check_project(main)
        self.assertTrue(any("Unknown operator kind" in issue.message for issue in report.issues))

    def test_missing_submodule_of_local_package_is_reported(self) -> None:
        self.write("pkg/__init__.qtr", "")
        main = self.write("main.qtr", "import pkg.missing_regression_module\n")
        with self.assertRaises(QuantarCheckError) as caught:
            runner.qtr_check(main)
        self.assertTrue(any("pkg.missing_regression_module" in issue.message for issue in caught.exception.issues))


class OperatorRegressionTests(unittest.TestCase):
    def test_python_keywords_and_operators_cannot_be_registered(self) -> None:
        for name in ("+", "as", "and"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved Python syntax"):
                    operators.operator.binary(name)(lambda left, right: (left, right))

    def test_matrix_factory_type_error_is_not_swallowed(self) -> None:
        def broken():
            raise TypeError("factory bug")

        with self.assertRaisesRegex(TypeError, "factory bug"):
            operators.operator.matrix("broken_matrix_regression")(broken)

    def test_clear_operators_can_restore_builtin_gates(self) -> None:
        import quantum.gate_matrix  # noqa: F401

        operators.clear_operators()
        runtime._load_quantum_operators()
        self.assertIsNotNone(operators.get_global_operator("<->")[0])


class OperatorRegistryTests(unittest.TestCase):
    def test_registrations_are_isolated_from_global_registry(self) -> None:
        name = "isolated_registry_op"
        ops = OperatorRegistry()

        @ops.binary(name)
        def reverse(left, right):
            return right, left

        self.assertEqual(ops[name](1, 2), (2, 1))
        self.assertEqual(ops.apply(name, 1, 2), (2, 1))
        self.assertEqual(ops.kind(name), "binary")
        self.assertEqual(ops.names("binary"), [name])
        self.assertIsNone(operators.get_global_operator(name)[0])

    def test_capture_keeps_imported_decorators_out_of_global_registry(self) -> None:
        name = "captured_registry_op"
        ops = OperatorRegistry()
        namespace = {}

        with ops.capture():
            exec(
                "from lang_systems.operators import operator\n"
                f"@operator.binary({name!r})\n"
                "def swap(left, right):\n"
                "    return right, left\n",
                namespace,
            )

        self.assertEqual(ops.apply(name, "a", "b"), ("b", "a"))
        self.assertIsNone(operators.get_global_operator(name)[0])

    def test_import_global_copies_only_requested_operators(self) -> None:
        first = "registry_import_first"
        second = "registry_import_second"
        for name in (first, second):
            operators.operator.binary(name)(lambda left, right: left + right)
            self.addCleanup(operators._global_operators.pop, name, None)
            self.addCleanup(operators._operator_metadata.pop, name, None)

        ops = OperatorRegistry()
        imported = ops.import_global(first)

        self.assertEqual(len(imported), 1)
        self.assertIs(imported[0], operators.get_global_operator(first)[0])
        self.assertEqual(ops.apply(first, 2, 3), 5)
        self.assertNotIn(second, ops)
        self.assertEqual(ops.names(), [first])

    def test_import_module_copies_named_operators(self) -> None:
        name = "module_registry_op"
        module = ModuleType("registry_op_fixture")
        staging = OperatorRegistry()

        with staging.capture():
            exec(
                "from lang_systems.operators import operator\n"
                f"@operator.binary({name!r})\n"
                "def swap(left, right):\n"
                "    return right, left\n",
                vars(module),
            )

        ops = OperatorRegistry()
        imported = ops.import_module(module, name)

        self.assertEqual(len(imported), 1)
        self.assertIs(imported[0], module.swap)
        self.assertEqual(ops.apply(name, 1, 2), (2, 1))
        self.assertIsNone(operators.get_global_operator(name)[0])

    def test_bracket_registration_is_isolated(self) -> None:
        pattern = "|{}>"
        ops = OperatorRegistry()

        @ops.bracket(pattern)
        def ket(value):
            return ("ket", value)

        self.assertTrue(ops.is_bracket(pattern))
        self.assertEqual(ops.bracket_parts(pattern), ("|", ">"))
        self.assertEqual(ops.apply(pattern, 7), ("ket", 7))
        self.assertIsNone(operators.get_global_operator(pattern)[0])

    def test_top_level_export(self) -> None:
        import lang_systems

        self.assertIs(lang_systems.OperatorRegistry, OperatorRegistry)


class BracketOperatorTests(unittest.TestCase):
    def _register(self, pattern: str, func) -> None:
        operators.operator.bracket(pattern)(func)
        self.addCleanup(operators._global_operators.pop, pattern, None)
        self.addCleanup(operators._operator_metadata.pop, pattern, None)

    def test_bracket_patterns_parse_with_operand_slots(self) -> None:
        patterns = ("|{}>", "<{}|", "<{}|{}>", "|{}><{}|")
        for pattern in patterns:
            self._register(pattern, lambda *operands: operands)

        tree = ast_builder.parse_ast(
            "a = |x>\n"
            "b = <x|\n"
            "c = <x|y>\n"
            "d = |x><y|\n"
        )
        found = [node for node in ast.walk(tree) if isinstance(node, ast_builder.QuantumOp)]

        self.assertEqual([node.operator for node in found], list(patterns))
        self.assertEqual([len(node.operands) for node in found], [1, 1, 2, 2])
        self.assertEqual([node.kind for node in found], ["bracket"] * len(patterns))
        for node in found:
            self.assertEqual(
                [operand.id for operand in node.operands],
                [["x"], ["x"], ["x", "y"], ["x", "y"]][found.index(node)],
            )

    def test_bracket_operators_execute_through_runtime(self) -> None:
        self._register("<{}|{}>", lambda bra, ket: (bra, ket))
        code = ast_builder.compile_ast(ast_builder.parse_ast("value = <1|2>\n"))
        scope = runtime.runtime_globals()

        exec(code, scope)

        self.assertEqual(scope["value"], (1, 2))

    def test_python_bitwise_and_comparison_syntax_is_preserved(self) -> None:
        self._register("|{}>", lambda value: value)
        self._register("<{}|", lambda value: value)

        tree = ast_builder.parse_ast("x = a|b>c\n")
        expression = tree.body[0].value  # type: ignore[attr-defined]

        self.assertIsInstance(expression, ast.Compare)
        self.assertIsInstance(expression.left, ast.BinOp)  # type: ignore[attr-defined]
        self.assertIsInstance(expression.left.op, ast.BitOr)  # type: ignore[attr-defined]

    def test_literal_infix_bracket_tokens_still_work(self) -> None:
        operators.operator.binary("|>")(lambda left, right: (left, right))
        self.addCleanup(operators._global_operators.pop, "|>", None)
        self.addCleanup(operators._operator_metadata.pop, "|>", None)

        tree = ast_builder.parse_ast("x = a |> b\n")
        found = [node for node in ast.walk(tree) if isinstance(node, ast_builder.QuantumOp)]

        self.assertEqual([node.operator for node in found], ["|>"])
        self.assertEqual([node.kind for node in found], ["binary"])

    def test_python_compatible_source_replaces_bracket_expressions(self) -> None:
        self._register("|{}>", lambda value: value)
        rewritten = ast_builder.python_compatible_source("x = |value>\n")

        compile(rewritten, "<quantar-python>", "exec")

        self.assertNotIn("|", rewritten)
        self.assertNotIn(">", rewritten)

    def test_runner_registers_same_file_bracket_declaration(self) -> None:
        pattern = "|{}>"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "main.qtr"
            source.write_text(
                "from lang_systems.operators import operator\n"
                f"@operator.bracket({pattern!r})\n"
                "def ket(value):\n"
                "    return ('ket', value)\n"
                "print(|9>)\n",
                encoding="utf-8",
            )
            self.addCleanup(operators._global_operators.pop, pattern, None)
            self.addCleanup(operators._operator_metadata.pop, pattern, None)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner.qtr_runner(source)

        self.assertIn("('ket', 9)", output.getvalue())

    def test_checker_accepts_bracket_declarations(self) -> None:
        pattern = "<{}|checker>"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "main.qtr"
            source.write_text(
                "from lang_systems.operators import operator\n"
                f"@operator.bracket({pattern!r})\n"
                "def combine(a, b):\n"
                "    return a, b\n"
                "value = <a|checker>\n",
                encoding="utf-8",
            )
            self.addCleanup(operators._global_operators.pop, pattern, None)
            self.addCleanup(operators._operator_metadata.pop, pattern, None)

            runner.qtr_check(source)


class RunnerRegressionTests(unittest.TestCase):
    def test_binary_compile_ignores_configured_python_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "main.qtr"
            source.parent.mkdir()
            source.write_text("print('hi')\n", encoding="utf-8")
            config = QuantarConfig(
                root=root,
                name="demo",
                entry_path=source,
                output_path=root / "src" / "main.py",
                rust=False,
                compile_to="py",
                mode="standalone",
                jobs=2,
            )
            self.assertEqual(runner._compile_destination(None, source, config, "bin"), runner._DEFAULT_BINARY_DIR)

    def test_options_before_bare_filename_use_run(self) -> None:
        self.assertEqual(
            runner._normalize_argv(["--rust", "main.qtr"]),
            ["run", "--rust", "main.qtr"],
        )

    def test_jobs_zero_is_a_usage_error(self) -> None:
        errors = io.StringIO()
        with patch.object(runner.sys, "argv", ["qtr", "compile", "main.qtr", "--jobs", "0"]):
            with contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
        self.assertEqual(caught.exception.code, 2)

    def test_keyboard_interrupt_exits_130(self) -> None:
        output = io.StringIO()
        with patch.object(runner, "app", side_effect=KeyboardInterrupt):
            with patch.object(runner.sys, "argv", ["qtr", "main.qtr"]):
                with contextlib.redirect_stdout(output):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main()
        self.assertEqual(caught.exception.code, 130)
        self.assertIn("[PROCESS TERMINATED] Process from main.qtr", output.getvalue())


class BuildRegressionTests(unittest.TestCase):
    def test_rust_prelude_is_fatal_on_missing_backend(self) -> None:
        self.assertIn("_quantar_backend.get_backend()", to_bin.RUST_PRELUDE)
        self.assertIn("raise RuntimeError", to_bin.RUST_PRELUDE)
        self.assertNotIn("except ImportError", to_bin.RUST_PRELUDE)

    def test_prelude_guard_uses_sentinel_not_arbitrary_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry = Path(temporary) / "main.py"
            entry.write_text('x = "_quantar_os"\n', encoding="utf-8")
            to_bin._inject_prelude(entry, to_bin.RUST_PRELUDE)
            self.assertIn(RUST_PRELUDE_SENTINEL, entry.read_text(encoding="utf-8"))


class AstBuilderRegressionTests(unittest.TestCase):
    def test_custom_operator_operands_keep_original_positions(self) -> None:
        source = "x = 1/0 <-> 2\n"
        operation = ast_builder.parse_ast(source).body[0].value #type: ignore[attr-defined]
        self.assertIsInstance(operation, ast_builder.QuantumOp)
        left = operation.operands[0] #type: ignore[attr-defined]
        self.assertEqual((left.lineno, left.col_offset, left.end_lineno, left.end_col_offset), (1, 4, 1, 7))
        self.assertEqual(ast.get_source_segment(source, left), "1/0")

    def test_multiline_rewrite_keeps_following_line_numbers(self) -> None:
        source = "x = (a\n     <-> b)\ny = 1\n"
        tree = ast_builder.parse_ast(source)
        self.assertEqual(tree.body[1].lineno, 3)

    def test_python_mirror_preserves_operator_precedence(self) -> None:
        mirrored = ast_builder.python_compatible_source("value = a <-> b | c and d\n")
        expression = ast.parse(mirrored).body[0].value #type: ignore[attr-defined]
        self.assertIsInstance(expression, ast.BoolOp)
        comparison = expression.values[0] #type: ignore[attr-defined]
        self.assertIsInstance(comparison, ast.Compare)
        self.assertIsInstance(comparison.comparators[0], ast.BinOp)


class LoweringRegressionTests(unittest.TestCase):
    def test_lowering_does_not_execute_local_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed.marker"
            (root / "helper.qtr").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            main = root / "main.qtr"
            main.write_text("import helper\nprint('ok')\n", encoding="utf-8")
            output = root / "main.py"

            runner.qtr_lower(main, output)

            self.assertFalse(marker.exists())
            self.assertTrue(output.is_file())

    def test_lowering_registers_operators_without_running_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed.marker"
            (root / "ops.qtr").write_text(
                "from pathlib import Path\n"
                "from lang_systems.operators import operator\n"
                "@operator.binary('<~noexec~>')\n"
                "def reverse(left, right):\n"
                "    return right, left\n"
                f"Path({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            main = root / "main.qtr"
            main.write_text("from ops import reverse\nvalue = 1 <~noexec~> 2\n", encoding="utf-8")
            output = root / "main.py"

            runner.qtr_lower(main, output)

            self.assertFalse(marker.exists())
            self.assertIn("__quantar_apply__", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
