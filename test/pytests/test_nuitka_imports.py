"""Tests for the freezer-friendly import surface and .qtr lowering."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quantar
import quantar._nuitka_manifest as manifest
from quantum import backend as quantum_backend
from lang_systems import runner, to_bin
from lang_systems.imports import (
    find_qtr_module,
    iter_import_statements,
    lower_qtr_tree,
)

_SUBPACKAGES = {
    "lang_systems",
    "quantum",
    "stl",
    "stl.quantum_algs",
    "stl.physics_algs",
    "quantum_algs",
    "physics_algs",
}


class StaticManifestTests(unittest.TestCase):
    def test_manifest_covers_eager_quarantine_of_lazy_exports(self) -> None:
        missing = [
            name
            for name in quantar.__all__
            if name not in _SUBPACKAGES and not hasattr(manifest, name)
        ]
        self.assertEqual(missing, [])

    def test_extension_import_is_optional(self) -> None:
        # A source checkout without a built .pyd must still import the manifest.
        self.assertTrue(manifest.quantar_kernels is None or hasattr(manifest.quantar_kernels, "set_seed"))

    def test_known_static_sources(self) -> None:
        self.assertEqual(manifest.parse.__module__, "lang_systems.quantar")
        self.assertEqual(manifest.Quantity.__module__, "stl.physics_algs.units")


class ImportScannerTests(unittest.TestCase):
    def test_iter_import_statements(self) -> None:
        source = (
            "import quantum\n"
            "from stl.physics_algs import Q, Dual\n"
            "print('not an import')\n"
        )
        statements = list(iter_import_statements(source))
        self.assertEqual(len(statements), 2)
        self.assertEqual([type(s).__name__ for s in statements], ["Import", "ImportFrom"])

    def test_find_qtr_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.qtr").write_text("", encoding="utf-8")
            (root / "pkg" / "helper.qtr").write_text("", encoding="utf-8")
            self.assertEqual(find_qtr_module("pkg", [root]), root / "pkg" / "__init__.qtr")
            self.assertEqual(find_qtr_module("pkg.helper", [root]), root / "pkg" / "helper.qtr")
            self.assertIsNone(find_qtr_module("missing", [root]))


class LoweringTests(unittest.TestCase):
    def test_lowers_entry_and_local_qtr_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.qtr").write_text("value = 1\n", encoding="utf-8")
            (root / "pkg" / "helper.qtr").write_text("answer = 42\n", encoding="utf-8")
            (root / "main.qtr").write_text("import pkg.helper\nprint(pkg.helper.answer)\n", encoding="utf-8")

            entry, files = lower_qtr_tree(root / "main.qtr", root / "out", search_paths=[root])

            self.assertEqual(entry, root / "out" / "main.py")
            self.assertIn(root / "out" / "pkg" / "__init__.py", files)
            self.assertIn(root / "out" / "pkg" / "helper.py", files)
            main_source = entry.read_text(encoding="utf-8")
            self.assertIn("import pkg.helper", main_source)
            self.assertIn("install_import_hook", main_source)
            self.assertNotIn("<->", main_source)

    def test_lowers_relative_imports_from_package_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.qtr").write_text("", encoding="utf-8")
            (root / "pkg" / "helper.qtr").write_text("answer = 42\n", encoding="utf-8")
            (root / "pkg" / "main.qtr").write_text("from .helper import answer\nprint(answer)\n", encoding="utf-8")

            entry, files = lower_qtr_tree(root / "pkg" / "main.qtr", root / "out", search_paths=[root])

            self.assertEqual(entry, root / "out" / "pkg" / "main.py")
            self.assertIn(root / "out" / "pkg" / "__init__.py", files)
            self.assertIn(root / "out" / "pkg" / "helper.py", files)


class PyInstallerBuildTests(unittest.TestCase):
    def test_command_includes_manifest_and_output_dir(self) -> None:
        command = to_bin.pyinstaller_command("main.py", "build/out", "app", mode="onefile", work_dir="build/work", paths=["lowered", "src"], extra_args=["--quiet"])
        self.assertEqual(command[:3], [to_bin.sys.executable, "-m", "PyInstaller"])
        self.assertIn("--onefile", command)
        self.assertIn("--noconfirm", command)
        self.assertIn("--clean", command)
        self.assertIn("--hidden-import=quantar._nuitka_manifest", command)
        self.assertIn("--hidden-import=quantum.quantar_kernels", command)
        self.assertIn("--distpath=build/out", command)
        self.assertIn("--name=app", command)
        self.assertIn(f"--paths={to_bin.Path('lowered').resolve()}", command)
        self.assertIn("--quiet", command)
        self.assertEqual(command[-1], "main.py")

    def test_standalone_mode_uses_onedir(self) -> None:
        command = to_bin.pyinstaller_command("main.py", "out", "app", mode="standalone")
        self.assertIn("--onedir", command)
        self.assertNotIn("--onefile", command)

    def test_env_puts_lowered_tree_first(self) -> None:
        env = to_bin.pyinstaller_env("lowered", "src")
        expected = [str(to_bin.Path(name).resolve()) for name in ("lowered", "src")]
        self.assertEqual(env["PYTHONPATH"].split(to_bin.os.pathsep)[:2], expected)

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            to_bin.make_bin_for_file("main.py", mode="flatpak")

    def test_make_bin_runs_pyinstaller_and_returns_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            destination = root / "dist"

            def fake_run(command, **kwargs):
                options = dict(part.split("=", 1) for part in command if part.startswith(("--distpath=", "--name=")))
                output_dir = Path(options["--distpath"])
                name = options["--name"]
                (output_dir / f"{name}.exe").write_bytes(b"fake binary")

            with patch.object(to_bin.subprocess, "run", side_effect=fake_run) as run:
                result = to_bin.make_bin_for_file(source, destination, mode="onefile", build_dir=root / "build")

            self.assertEqual(result, destination / "main.exe")
            self.assertEqual(result.read_bytes(), b"fake binary")
            called_command = run.call_args.args[0]
            entry = Path(called_command[-1])
            self.assertTrue(entry.is_file())
            self.assertIn("lowered", entry.parts)
            self.assertNotIn("--hidden-import=quantar._nuitka_manifest", called_command)
            self.assertIn(f"--paths={to_bin._REPO_SRC}", called_command)
            env_path = run.call_args.kwargs["env"]["PYTHONPATH"]
            self.assertTrue(env_path.startswith(str(entry.parent)))

    def test_quantar_program_keeps_hidden_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.qtr"
            source.write_text("import quantum\nprint(quantum.QuantumComputer(1).dimension)\n", encoding="utf-8")

            def fake_run(command, **kwargs):
                options = dict(part.split("=", 1) for part in command if part.startswith(("--distpath=", "--name=")))
                output_dir = Path(options["--distpath"])
                (output_dir / f"{options['--name']}.exe").write_bytes(b"fake binary")

            with patch.object(to_bin.subprocess, "run", side_effect=fake_run) as run:
                to_bin.make_bin_for_file(source, root / "dist", mode="onefile", build_dir=root / "build")

            called_command = run.call_args.args[0]
            self.assertIn("--hidden-import=quantar._nuitka_manifest", called_command)
            self.assertIn("--hidden-import=quantum.quantar_kernels", called_command)

    def test_standalone_returns_dist_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.qtr"
            source.write_text("print('hello')\n", encoding="utf-8")
            destination = root / "dist"

            def fake_run(command, **kwargs):
                options = dict(part.split("=", 1) for part in command if part.startswith(("--distpath=", "--name=")))
                bundle = Path(options["--distpath"]) / options["--name"]
                bundle.mkdir(parents=True, exist_ok=True)
                (bundle / f"{bundle.name}.exe").write_bytes(b"fake binary")

            with patch.object(to_bin.subprocess, "run", side_effect=fake_run):
                result = to_bin.make_bin_for_file(source, destination, mode="standalone", build_dir=root / "build")

            self.assertEqual(result, destination / "main" / "main.exe")
            self.assertTrue(result.is_file())

    def test_rust_prelude_is_injected_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            captured = {}

            def fake_run(command, **kwargs):
                options = dict(part.split("=", 1) for part in command if part.startswith(("--distpath=", "--name=")))
                captured["entry"] = Path(command[-1]).read_text(encoding="utf-8")
                (Path(options["--distpath"]) / f"{options['--name']}.exe").write_bytes(b"fake binary")

            with patch.object(to_bin, "rust_extension_status", return_value=(True, "fake.pyd")) as status:
                with patch.object(to_bin.subprocess, "run", side_effect=fake_run):
                    to_bin.make_bin_for_file(source, root / "dist", mode="onefile", build_dir=root / "build", rust=True)

            status.assert_called_once_with(prepare=True)
            self.assertIn('QUANTAR_BACKEND"] = "rust"', captured["entry"])
            self.assertIn('_quantar_backend.set_backend("rust")', captured["entry"])
            self.assertIn("use_rust_physics(True)", captured["entry"])

    def test_missing_rust_extension_aborts_binary_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            with patch.object(to_bin, "rust_extension_status", return_value=(False, "bad .pyd")) as status:
                with patch.object(to_bin.subprocess, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, "--rust requires the Rust extension"):
                        to_bin.make_bin_for_file(source, root / "dist", build_dir=root / "build", rust=True)
            status.assert_called_once_with(prepare=True)
            run.assert_not_called()


class QtrCompileTests(unittest.TestCase):
    def test_passthrough_to_make_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "main.qtr"
            source.write_text("print('hello')\n", encoding="utf-8")
            destination = root / "out"
            expected = destination / "main.exe"

            with patch.object(runner, "make_bin_for_file", return_value=expected) as make:
                result = runner.qtr_compile(source, destination, mode="standalone", jobs=3)

            self.assertEqual(result, expected)
            make.assert_called_once_with(source.resolve(), destination, mode="standalone", jobs=3, rust=False)

    def test_default_destination_and_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "main.py"
            source.write_text("print('hello')\n", encoding="utf-8")

            with patch.object(runner, "make_bin_for_file") as make:
                runner.qtr_compile(source)

            make.assert_called_once_with(source.resolve(), "./quantar_compile/bins", mode="standalone", jobs=2, rust=False)
            self.assertIs(runner.qtr_compiler, runner.qtr_compile)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(FileNotFoundError):
            runner.qtr_compile("does-not-exist.qtr")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "notes.txt"
            bad.write_text("nope\n", encoding="utf-8")
            good = root / "main.qtr"
            good.write_text("print('hi')\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                runner.qtr_compile(bad)
            with self.assertRaises(ValueError):
                runner.qtr_compile(good, jobs=0)

    def test_rust_flag_is_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "main.qtr"
            source.write_text("print('hi')\n", encoding="utf-8")
            with patch.object(runner, "make_bin_for_file") as make:
                runner.qtr_compile(source, rust=True)
            make.assert_called_once_with(source.resolve(), "./quantar_compile/bins", mode="standalone", jobs=2, rust=True)

    def test_lowered_py_gets_rust_prelude(self) -> None:
        lowered = runner.unparse(runner._inject_runtime_imports(ast.parse("from __future__ import annotations\nx = 1\n"), True))
        self.assertIn("environ['QUANTAR_BACKEND'] = 'rust'", lowered)
        self.assertIn("backend.set_backend('rust')", lowered)
        self.assertIn("use_rust_physics(True)", lowered)

    def test_pure_lowered_py_gets_only_import_hook(self) -> None:
        source = "x = 1\nprint(x)\n"
        lowered = runner.unparse(runner._inject_runtime_imports(ast.parse(source), source_code=source, source_path=Path("main.qtr")))
        self.assertIn("from lang_systems.imports import install_import_hook", lowered)
        self.assertNotIn("__quantar_apply__", lowered)

    def test_lowered_quantar_operator_keeps_runtime_helper(self) -> None:
        source = "__quantar_apply__('test', 1)\n"
        lowered = runner.unparse(runner._inject_runtime_imports(ast.parse(source), source_code=source, source_path=Path("main.qtr")))
        self.assertIn("from lang_systems.runtime import apply_quantar_operator as __quantar_apply__", lowered)


class RustRuntimeTests(unittest.TestCase):
    def test_rust_flag_selects_backend_and_physics(self) -> None:
        previous = os.environ.get("QUANTAR_BACKEND")
        try:
            with patch.object(quantum_backend, "set_backend") as set_backend, patch.object(runner, "_enable_rust_physics") as enable_physics:
                runner._enable_rust_backend()
            set_backend.assert_called_once_with("rust")
            enable_physics.assert_called_once_with()
            self.assertEqual(os.environ["QUANTAR_BACKEND"], "rust")
        finally:
            if previous is None:
                os.environ.pop("QUANTAR_BACKEND", None)
            else:
                os.environ["QUANTAR_BACKEND"] = previous

    def test_qtr_runner_rust_prepares_backend_before_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "main.qtr"
            source.write_text("print('hi')\n", encoding="utf-8")
            with patch.object(runner, "_enable_rust_backend") as enable_backend, patch.object(runner, "preload_imports") as preload:
                runner.qtr_runner(source, rust=True)
            enable_backend.assert_called_once_with()
            preload.assert_called_once()


class CliWiringTests(unittest.TestCase):
    def test_run_subcommand(self) -> None:
        with patch.object(runner.sys, "argv", ["qtr", "run", "main.qtr"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        run.assert_called_once_with("main.qtr", rust=False)

    def test_run_rust(self) -> None:
        with patch.object(runner.sys, "argv", ["qtr", "run", "main.qtr", "--rust"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        run.assert_called_once_with("main.qtr", rust=True)

    def test_compile_py(self) -> None:
        with patch.object(runner.sys, "argv", ["qtr", "compile", "main.qtr", "out.py", "--rust"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        run.assert_called_once_with("main.qtr", "out.py", rust=True)

    def test_compile_bin(self) -> None:
        artifact = Path("dist") / "main.exe"
        output = io.StringIO()
        argv = ["qtr", "compile", "main.qtr", "dist", "--compile-to", "bin", "--mode", "standalone", "--jobs", "4", "--rust"]
        with patch.object(runner.sys, "argv", argv):
            with patch.object(runner, "qtr_compile", return_value=artifact) as compile_bin:
                with contextlib.redirect_stdout(output):
                    runner.main()
        compile_bin.assert_called_once_with("main.qtr", "dist", mode="standalone", jobs=4, rust=True)
        self.assertIn(f"[BINARY] Compiled to {artifact}", output.getvalue())

    def test_pure_filename_shorthand(self) -> None:
        with patch.object(runner.sys, "argv", ["qtr", "main.qtr"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        run.assert_called_once_with("main.qtr", rust=False)

    def test_pure_filename_shorthand_rust(self) -> None:
        with patch.object(runner.sys, "argv", ["qtr", "main.qtr", "--rust"]):
            with patch.object(runner, "qtr_runner") as run:
                runner.main()
        run.assert_called_once_with("main.qtr", rust=True)


if __name__ == "__main__":
    unittest.main()
