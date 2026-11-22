from collections.abc import Callable
from typing import Any, overload

import numpy as np
import numpy.typing as npt

class Qubit:
    index: int
    name: str
    computer: QuantumComputer
    is_measured: bool
    _true_val: int | None

    def __init__(self, computer: QuantumComputer, index: int, name: str = "qubit") -> None: ...
    def is_alive(self) -> bool: ...
    def alpha(self) -> float: ...
    def measure(self) -> int: ...
    def apply_gate(self, gate_matrix: npt.ArrayLike) -> None: ...

class Gate:
    matrix: npt.NDArray[np.complex128]
    controls: list[int]
    targets: list[int]
    name: str | None
    params: list[float]
    inverted: bool

    def __init__(
        self,
        matrix: npt.ArrayLike,
        controls: list[int],
        targets: list[int],
        name: str | None = None,
        params: list[float] | None = None,
        inverted: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def inverse(self) -> Gate: ...
    def remap(self, mapping: list[int], extra_controls: list[int] | None = None) -> Gate: ...

class QuantumComputer:
    num_qubits: int
    name: str
    qubits: list[Qubit]
    dimension: int
    state_vector: npt.NDArray[np.complex128]
    op_hist: list[Gate]
    _recording: bool
    _custom_operators: dict[str, Any]
    _scratch: npt.NDArray[np.complex128]

    def __init__(
        self, num_qubits: int, name: str = "QC1", qubit_names: list[str] | None = None
    ) -> None: ...
    def __repr__(self) -> str: ...
    def bit_position(self, qubit_index: int) -> int: ...
    @overload
    def __getitem__(self, key: Qubit | int | str) -> Qubit: ...
    @overload
    def __getitem__(self, key: slice) -> list[Qubit]: ...
    def index_of(self, qubit: Qubit | int | str) -> int: ...
    def apply_gate(
        self,
        gate_matrix: npt.ArrayLike,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
        qasm_name: str | None = None,
        qasm_params: list[float] | None = None,
        log_history: bool | None = None,
    ) -> None: ...
    def gate(
        self,
        gate_matrix: npt.ArrayLike,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
        qasm_name: str | None = None,
        qasm_params: list[float] | None = None,
    ) -> Gate: ...
    def apply(
        self,
        operation: Any,
        target_qubits: list[Qubit | int | str] | None = None,
        control_qubits: list[Qubit | int | str] | None = None,
    ) -> None: ...
    def _apply_program(self, program: list[Gate]) -> None: ...
    def suppress_history(self) -> Any: ...
    def checkpoint(self) -> int: ...
    def uncompute(self, target_qubits: list[Qubit | int | str], checkpoint: int = 0) -> None: ...
    def qft(
        self,
        qubits: list[Qubit | int | str],
        inverse: bool = False,
        record: bool | None = None,
    ) -> QuantumComputer: ...
    def iqft(
        self, qubits: list[Qubit | int | str], record: bool | None = None
    ) -> QuantumComputer: ...
    def measure_pauli(
        self, pauli: str, qubits: list[Qubit | int | str], shots: int = 1000
    ) -> float: ...
    def sample(self, circuit_func: Callable[[QuantumComputer], Any], num_samples: int = 1000) -> dict[str, int]: ...
    def sample_parallel(
        self, circuit_func: Callable[..., Any], num_samples: int = 1000, **kwargs: Any
    ) -> dict[str, int]: ...
    def to_openqasm3(self) -> str: ...
    def register_operator(self, name: str, matrix: npt.ArrayLike | Callable[..., Any]) -> None: ...
    def get_operator(self, name: str) -> Any | None: ...
    def list_operators(self) -> list[str]: ...
