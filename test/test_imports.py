"""Tests for Python/Quantar import paths and backend selection."""

from __future__ import annotations

import contextlib
import importlib.machinery
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import quantar
import quantum
import quantum.backend as backend
import stl
from lang_systems.imports import install_import_hook, uninstall_import_hook
from lang_systems.operators import is_known_operator
from lang_systems.runner import qtr_runner

_OPS_MODULE = "qtr_import_ops_fixture"
_PKG_MODULE = "qtr_import_pkg_fixture"


class ImportHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._sys_path = sys.path[:]
        self._modules = set(sys.modules)
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        sys.path.insert(0, str(self.root))
        install_import_hook()

    def tearDown(self) -> None:
        uninstall_import_hook()
        sys.path[:] = self._sys_path
        for name in set(sys.modules) - self._modules:
            sys.modules.pop(name, None)
        self._directory.cleanup()

    def test_qtr_module_import_and_preparse_operator(self) -> None:
        (self.root / f"{_OPS_MODULE}.qtr").write_text(
            "from lang_systems.operators import operator\n"
            "\n"
            "@operator.binary('<~>')\n"
            "def reverse(left, right):\n"
            "    return right, left\n"
            "\n"
            "VALUE = 42\n",
            encoding="utf-8",
        )
        main = self.root / "main.qtr"
        main.write_text(
            f"from {_OPS_MODULE} import VALUE\n"
            "a, b = 1, 2\n"
            "print(a <~> b)\n"
            "print(VALUE)\n",
            encoding="utf-8",
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qtr_runner(main)

        self.assertEqual(output.getvalue().splitlines(), ["(2, 1)", "42"])
        self.assertTrue(is_known_operator("<~>"))

        module = sys.modules[_OPS_MODULE]
        self.assertEqual(module.VALUE, 42)

    def test_sibling_python_import(self) -> None:
        (self.root / "sibling_helper.py").write_text("VALUE = 11\n", encoding="utf-8")
        main = self.root / "sibling_main.qtr"
        main.write_text("from sibling_helper import VALUE\nprint(VALUE)\n", encoding="utf-8")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qtr_runner(main)

        self.assertEqual(output.getvalue().strip(), "11")

    def test_relative_import_from_qtr_package(self) -> None:
        package = self.root / _PKG_MODULE
        package.mkdir()
        (package / "__init__.qtr").write_text("", encoding="utf-8")
        (package / "child.qtr").write_text("CHILD_VALUE = 7\n", encoding="utf-8")
        main = package / "main.qtr"
        main.write_text(
            "from .child import CHILD_VALUE\nprint(CHILD_VALUE)\n",
            encoding="utf-8",
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qtr_runner(main)

        self.assertEqual(output.getvalue().strip(), "7")

    def test_compile_mode_keeps_future_imports_first(self) -> None:
        source = self.root / "future.qtr"
        source.write_text(
            "from __future__ import annotations\n\nVALUE: int = 3\nprint(VALUE)\n",
            encoding="utf-8",
        )
        compiled = self.root / "future.py"

        qtr_runner(source, compiled)
        text = compiled.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]

        self.assertIn("from __future__ import annotations", lines[0])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(text, str(compiled), "exec"), {"__name__": "__main__"})
        self.assertEqual(output.getvalue().strip(), "3")


class BackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backend = os.environ.get("QUANTAR_BACKEND")
        self._env_path = os.environ.get("QUANTAR_RUST_BACKEND_PATH")
        backend.reset_backend()

    def tearDown(self) -> None:
        backend.reset_backend()
        if self._env_backend is None:
            os.environ.pop("QUANTAR_BACKEND", None)
        else:
            os.environ["QUANTAR_BACKEND"] = self._env_backend
        if self._env_path is None:
            os.environ.pop("QUANTAR_RUST_BACKEND_PATH", None)
        else:
            os.environ["QUANTAR_RUST_BACKEND_PATH"] = self._env_path

    def test_python_backend_is_default_and_switchable(self) -> None:
        from quantum.qubit_abs import QuantumComputer as PythonQC

        self.assertEqual(backend.get_backend(), "python")
        self.assertIs(quantum.QuantumComputer, PythonQC)

        backend.use_backend("python")
        self.assertIs(quantum.QuantumComputer, PythonQC)
        self.assertIsInstance(quantum.QuantumComputer(1), PythonQC)

    def test_rust_backend_switch(self) -> None:
        # With no explicit path this validates the installed extension.
        backend.use_backend("rust")
        self.assertEqual(backend.get_backend(), "rust")
        rust_module = backend.get_backend_module()
        self.assertIs(quantum.QuantumComputer, rust_module.QuantumComputer)

        qc = quantum.QuantumComputer(2)
        self.assertEqual(qc.dimension, 4)
        self.assertTrue(hasattr(qc, "apply"))

        import quantum.gate_matrix as gm

        qc.apply_gate(gm.h_MAT(), [0])
        probabilities = np.abs(qc.state_vector[:, 0]) ** 2
        self.assertTrue(np.allclose(probabilities, [0.5, 0.0, 0.5, 0.0]))

        backend.use_backend("python")
        from quantum.qubit_abs import QuantumComputer as PythonQC

        self.assertIs(quantum.QuantumComputer, PythonQC)


class PublicApiTests(unittest.TestCase):
    def test_umbrella_exports_common_names(self) -> None:
        for name in (
            "QuantumComputer", "Qubit", "Gate", "use_backend", "set_seed",
            "get_seed", "grovers_alg", "shors_alg", "Quantity",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(quantar, name))
        self.assertIn("set_seed", quantar.__all__)

    def test_stl_exports_subpackages_and_algorithms(self) -> None:
        self.assertIs(stl.quantum_algs.grovers_alg, stl.grovers_alg)
        self.assertIs(stl.physics_algs.Quantity, stl.Quantity)
        self.assertTrue(hasattr(stl.quantum_algs, "grovers_alg_lib"))
        self.assertTrue(hasattr(stl.quantum_algs, "shors_alg_lib"))

    def test_quantum_exports_seed_helpers(self) -> None:
        self.assertTrue(all(
            hasattr(quantum, name) for name in ("set_seed", "get_seed", "make_pcg64")
        ))


class SeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_seed = quantum.get_seed()

    def tearDown(self) -> None:
        quantum.set_seed(self._previous_seed)

    def test_set_seed_updates_shared_state_and_generator(self) -> None:
        quantum.set_seed(1234)
        self.assertEqual(quantum.get_seed(), 1234)
        self.assertEqual(quantum.seed_set.get_seed(), 1234)

        first = quantum.seed_set.get_rng().random(4)
        quantum.set_seed(1234)
        self.assertTrue(np.array_equal(quantum.seed_set.get_rng().random(4), first))

    def test_sample_counts_is_reproducible(self) -> None:
        from quantum.kernels import sample_counts

        probabilities = np.array([0.2, 0.3, 0.5])
        quantum.set_seed(99)
        first = sample_counts(probabilities, 100, 3)
        quantum.set_seed(99)
        second = sample_counts(probabilities, 100, 3)
        self.assertTrue(np.array_equal(first, second))

    def test_seed_validation(self) -> None:
        with self.assertRaises(TypeError):
            quantum.set_seed("1234")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            quantum.set_seed(-1)
        with self.assertRaises(ValueError):
            quantum.set_seed(1 << 64)


class RustAutoBuildTests(unittest.TestCase):
    def test_extension_suffix_is_supported(self) -> None:
        suffix = backend._extension_suffix()
        self.assertIn(suffix, importlib.machinery.EXTENSION_SUFFIXES)

    def test_free_threaded_suffix_avoids_gil_only_abi3(self) -> None:
        suffixes = [".cpython-314t-win_amd64.pyd", ".abi3.pyd", ".pyd"]
        with (
            patch.object(backend.sysconfig, "get_config_var", return_value=1),
            patch.object(backend.importlib.machinery, "EXTENSION_SUFFIXES", suffixes),
        ):
            self.assertEqual(backend._extension_suffix(), suffixes[0])

    def test_abi3t_suffix_is_preferred_on_python315(self) -> None:
        suffixes = [
            ".cpython-315-win_amd64.pyd",
            ".abi3t.pyd",
            ".abi3.pyd",
            ".pyd",
        ]
        with (
            patch.object(backend.sysconfig, "get_config_var", return_value=0),
            patch.object(backend.sys, "version_info", (3, 15, 0)),
            patch.object(backend.importlib.machinery, "EXTENSION_SUFFIXES", suffixes),
        ):
            self.assertEqual(backend._extension_suffix(), ".abi3t.pyd")

    def test_cargo_discovery_is_safe(self) -> None:
        cargo = backend.find_cargo()
        self.assertTrue(cargo is None or Path(cargo).name)

    def test_rust_source_discovery_is_safe(self) -> None:
        source = backend.find_rust_source_dir()
        if source is not None:
            self.assertTrue((source / "Cargo.toml").is_file())
            self.assertTrue((source / "src" / "lib.rs").is_file())

    def test_auto_build_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {backend._ENV_AUTO_BUILD: "0"}):
            self.assertFalse(backend._auto_build_enabled())
        with patch.dict(os.environ, {backend._ENV_AUTO_BUILD: "1"}):
            self.assertTrue(backend._auto_build_enabled())

    def test_build_environment_suppresses_dependency_warnings(self) -> None:
        with patch.dict(
            os.environ,
            {"QUANTAR_STRICT_RUST": "0", "RUSTFLAGS": "-C opt-level=3"},
        ):
            env = backend._build_environment()
            self.assertIn("-A deprecated", env["RUSTFLAGS"])
            self.assertEqual(os.environ["RUSTFLAGS"], "-C opt-level=3")

    def test_strict_rust_keeps_warnings(self) -> None:
        with patch.dict(
            os.environ,
            {"QUANTAR_STRICT_RUST": "1", "RUSTFLAGS": "-C opt-level=3"},
        ):
            env = backend._build_environment()
            self.assertEqual(env["RUSTFLAGS"], "-C opt-level=3")


if __name__ == "__main__":
    unittest.main()
