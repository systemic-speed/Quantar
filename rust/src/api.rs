//! The rest of the `QuantumComputer` surface: gate records, the `apply`
//! dispatcher, history control, Fourier transforms, sampling and measurement.
//!
//! Program-level operations call back into Python (the QFT builder, operator
//! registry and user circuits are Python objects), so these methods take a
//! `PyRefMut` receiver, turn it into an owned `Py` handle immediately, and keep
//! pyclass borrows short. That way a callback can freely call back into `qc`.

use crate::gate::Gate;
use crate::gates::contiguous_slice;
use crate::{
    gate_matrix_module, gate_view, new_amplitude_array, numpy_module, GateView, QuantumComputer,
    Qubit, QubitState,
};
use num_complex::Complex64;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::RngExt;
use std::collections::HashSet;

/// Resolve handles/indices/names without the liveness check: `gate()` only
/// records, so a locked qubit is rejected when the gate is actually applied.
fn resolve<'py>(
    py: Python<'py>,
    computer: &QuantumComputer,
    items: &[Bound<'py, PyAny>],
) -> PyResult<Vec<usize>> {
    let mut indices = Vec::with_capacity(items.len());
    for item in items {
        let resolved = computer.__getitem__(py, item)?;
        let qubit = resolved
            .bind(py)
            .cast::<Qubit>()
            .map_err(|_| PyTypeError::new_err("Expected a single qubit, not a list of qubits."))?;
        indices.push(qubit.borrow().index);
    }
    Ok(indices)
}

fn non_empty(items: &Option<Vec<Bound<'_, PyAny>>>) -> bool {
    items.as_ref().is_some_and(|list| !list.is_empty())
}

fn is_diagonal(
    computer: &QuantumComputer,
    view: &GateView<'_>,
) -> Option<(usize, usize, Complex64, Complex64)> {
    if view.targets.len() != 1 || view.matrix.shape() != [2, 2] {
        return None;
    }
    let array = view
        .matrix
        .clone()
        .cast_into::<PyArray2<Complex64>>()
        .ok()?;
    let matrix = array.readonly();
    let matrix = matrix.as_array();
    let zero = Complex64::new(0.0, 0.0);
    if matrix[[0, 1]] != zero || matrix[[1, 0]] != zero {
        return None;
    }
    let control_mask = view.controls.iter().fold(0usize, |mask, &index| {
        mask | (1usize << computer.bit_position_unchecked(index))
    });
    Some((
        control_mask,
        computer.bit_position_unchecked(view.targets[0]),
        matrix[[0, 0]],
        matrix[[1, 1]],
    ))
}

fn is_swap(computer: &QuantumComputer, view: &GateView<'_>) -> Option<(usize, usize)> {
    if view.targets.len() != 2 || !view.controls.is_empty() || view.matrix.shape() != [4, 4] {
        return None;
    }
    let array = view
        .matrix
        .clone()
        .cast_into::<PyArray2<Complex64>>()
        .ok()?;
    let matrix = array.readonly();
    let matrix = matrix.as_array();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let expected = [
        [one, zero, zero, zero],
        [zero, zero, one, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
    ];
    for row in 0..4 {
        for column in 0..4 {
            if matrix[[row, column]] != expected[row][column] {
                return None;
            }
        }
    }
    Some((
        computer.bit_position_unchecked(view.targets[0]),
        computer.bit_position_unchecked(view.targets[1]),
    ))
}

#[derive(Clone, Copy)]
enum GateClass {
    Diagonal((usize, usize, Complex64, Complex64)),
    Swap((usize, usize)),
    Other,
}

impl QuantumComputer {
    /// Run a program the way `quantum.qubit_abs._apply_program` does: validate
    /// everything first, then fuse diagonal and disjoint-SWAP runs. Individual
    /// gates still reach `op_hist`, so `uncompute` and QASM export see them.
    pub(crate) fn apply_program(
        &mut self,
        py: Python<'_>,
        gates: &[Bound<'_, PyAny>],
    ) -> PyResult<()> {
        let views = gates
            .iter()
            .map(|gate| gate_view(py, gate))
            .collect::<PyResult<Vec<_>>>()?;
        for view in &views {
            self.validate_view(py, view)?;
        }

        // Classifying each gate once avoids repeating the matrix inspection on
        // every step of a fused run (the old code checked neighbours twice).
        let classes: Vec<GateClass> = views
            .iter()
            .map(|view| {
                if let Some(spec) = is_diagonal(self, view) {
                    GateClass::Diagonal(spec)
                } else if let Some(pair) = is_swap(self, view) {
                    GateClass::Swap(pair)
                } else {
                    GateClass::Other
                }
            })
            .collect();

        let record = self.recording;
        let history = self.op_hist.clone_ref(py);
        let mut index = 0;
        while index < views.len() {
            let mut end = index + 1;

            match classes[index] {
                GateClass::Diagonal(_) => {
                    while end < views.len() && matches!(classes[end], GateClass::Diagonal(_)) {
                        end += 1;
                    }
                    if end - index == 1 {
                        self.run_gate_view_validated(py, &views[index])?;
                    } else {
                        let specs: Vec<_> = classes[index..end]
                            .iter()
                            .filter_map(|class| match class {
                                GateClass::Diagonal(spec) => Some(*spec),
                                _ => None,
                            })
                            .collect();
                        self.exec_diagonal_run(py, &specs)?;
                    }
                }
                GateClass::Swap((low, high)) => {
                    let mut touched = (1usize << low) | (1usize << high);
                    while end < views.len() {
                        let GateClass::Swap((other_low, other_high)) = classes[end] else {
                            break;
                        };
                        let bits = (1usize << other_low) | (1usize << other_high);
                        if touched & bits != 0 {
                            break;
                        }
                        touched |= bits;
                        end += 1;
                    }
                    if end - index == 1 {
                        self.run_gate_view_validated(py, &views[index])?;
                    } else {
                        let pairs: Vec<_> = classes[index..end]
                            .iter()
                            .filter_map(|class| match class {
                                GateClass::Swap(pair) => Some(*pair),
                                _ => None,
                            })
                            .collect();
                        self.exec_swap_run(py, &pairs)?;
                    }
                }
                GateClass::Other => {
                    self.run_gate_view_validated(py, &views[index])?;
                }
            }

            if record {
                for gate in &gates[index..end] {
                    history.bind(py).append(gate)?;
                }
            }
            index = end;
        }
        Ok(())
    }

    fn sample_counts(&mut self, py: Python<'_>, shots: usize) -> PyResult<Vec<usize>> {
        let guard = self.state_vector.bind(py).try_readonly()?;
        let state = contiguous_slice(&guard, "state_vector")?;
        let mut cumulative = Vec::with_capacity(state.len());
        let mut total = 0.0;
        for amplitude in state {
            total += amplitude.norm_sqr();
            cumulative.push(total);
        }
        if total > 0.0 {
            for entry in &mut cumulative {
                *entry /= total;
            }
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        let dimension = state.len();
        let mut counts = vec![0usize; dimension];
        let rng = &mut self.rng;
        for _ in 0..shots {
            let draw: f64 = rng.random();
            let index = cumulative.partition_point(|value| *value < draw);
            counts[index.min(dimension - 1)] += 1;
        }
        Ok(counts)
    }
}

#[pymethods]
impl QuantumComputer {
    fn index_of(&self, py: Python<'_>, qubit: &Bound<'_, PyAny>) -> PyResult<usize> {
        Ok(resolve(py, self, std::slice::from_ref(qubit))?[0])
    }

    /// Python `CircuitGate.__call__` replays a saved program through this
    /// method, so expose the same name for cross-backend use.
    #[pyo3(name = "_apply_program")]
    fn py_apply_program(&mut self, py: Python<'_>, program: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.apply_program(py, &program)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (gate_matrix, target_qubits, control_qubits = None, qasm_name = None, qasm_params = None))]
    fn gate<'py>(
        &self,
        py: Python<'py>,
        gate_matrix: &Bound<'py, PyAny>,
        target_qubits: Vec<Bound<'py, PyAny>>,
        control_qubits: Option<Vec<Bound<'py, PyAny>>>,
        qasm_name: Option<String>,
        qasm_params: Option<Vec<f64>>,
    ) -> PyResult<Py<Gate>> {
        let targets = resolve(py, self, &target_qubits)?;
        let controls = match &control_qubits {
            Some(items) => resolve(py, self, items)?,
            None => Vec::new(),
        };
        Py::new(
            py,
            Gate::new(
                py,
                gate_matrix,
                controls,
                targets,
                qasm_name,
                qasm_params,
                false,
            )?,
        )
    }

    #[pyo3(signature = (operation, target_qubits = None, control_qubits = None))]
    fn apply(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        operation: &Bound<'_, PyAny>,
        target_qubits: Option<Vec<Bound<'_, PyAny>>>,
        control_qubits: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<()> {
        let me: Py<Self> = slf.into();
        apply_operation(py, &me, operation, target_qubits, control_qubits)
    }

    fn suppress_history(slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Py<SuppressHistory>> {
        let me: Py<Self> = slf.into();
        Py::new(
            py,
            SuppressHistory {
                computer: me,
                was: true,
            },
        )
    }

    fn checkpoint(&self, py: Python<'_>) -> usize {
        self.op_hist.bind(py).len()
    }

    #[pyo3(signature = (target_qubits, checkpoint = 0))]
    fn uncompute(
        &mut self,
        py: Python<'_>,
        target_qubits: Vec<Bound<'_, PyAny>>,
        checkpoint: isize,
    ) -> PyResult<()> {
        let history: Py<PyList> = self.op_hist.clone_ref(py);
        if history.bind(py).is_empty() {
            return Err(PyValueError::new_err("No operations to uncompute."));
        }

        let wanted: HashSet<usize> = resolve(py, self, &target_qubits)?.into_iter().collect();
        let was_recording = self.recording;
        self.recording = true;
        let result = (|| {
            let list = history.bind(py);
            let length = list.len() as isize;
            let start = if checkpoint < 0 {
                (length + checkpoint).max(0)
            } else {
                checkpoint.min(length)
            } as usize;

            for position in (start..list.len()).rev() {
                let original = list.get_item(position)?;
                let view = gate_view(py, &original)?;
                let touched: Vec<usize> = view
                    .controls
                    .iter()
                    .chain(view.targets.iter())
                    .copied()
                    .collect();
                // op_hist is a real Python list and callers may append their
                // own records, so never trust an index from it.
                for &index in &touched {
                    if index >= self.num_qubits {
                        return Err(PyIndexError::new_err(format!(
                            "Recorded gate references qubit {index}, but this computer has {} qubit(s).",
                            self.num_qubits
                        )));
                    }
                }
                if !touched.iter().any(|index| wanted.contains(index)) {
                    continue;
                }
                for &index in &touched {
                    if self.qubits[index].bind(py).borrow().is_measured() {
                        return Err(PyValueError::new_err(
                            "Cannot uncompute after measurement of involved qubits.",
                        ));
                    }
                }
                let inverse = original.call_method0("inverse")?;
                let inverse_view = gate_view(py, &inverse)?;
                self.run_gate_view(py, &inverse_view)?;
                list.append(inverse)?;
            }
            Ok(())
        })();
        self.recording = was_recording;
        result
    }

    #[pyo3(signature = (qubits, inverse = false, record = None))]
    fn qft(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        qubits: Vec<Bound<'_, PyAny>>,
        inverse: bool,
        record: Option<bool>,
    ) -> PyResult<Py<Self>> {
        qft_impl(py, slf, qubits, inverse, record)
    }

    #[pyo3(signature = (qubits, record = None))]
    fn iqft(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        qubits: Vec<Bound<'_, PyAny>>,
        record: Option<bool>,
    ) -> PyResult<Py<Self>> {
        qft_impl(py, slf, qubits, true, record)
    }

    #[pyo3(signature = (pauli, qubits, shots = 1000))]
    fn measure_pauli(
        &mut self,
        py: Python<'_>,
        pauli: String,
        qubits: Vec<Bound<'_, PyAny>>,
        shots: isize,
    ) -> PyResult<f64> {
        let targets = resolve(py, self, &qubits)?;
        if targets.len() != pauli.len() {
            return Err(PyValueError::new_err(format!(
                "Pauli string length ({}) must match number of qubits ({})",
                pauli.len(),
                targets.len()
            )));
        }
        if shots <= 0 {
            return Err(PyValueError::new_err(
                "Number of shots must be a positive integer.",
            ));
        }
        let operators: Vec<char> = pauli.chars().collect();
        for &operator in &operators {
            if !matches!(operator, 'X' | 'Y' | 'Z' | 'I') {
                return Err(PyValueError::new_err(format!(
                    "Unknown Pauli operator: {operator}"
                )));
            }
        }

        let gate_matrix = gate_matrix_module(py)?.bind(py);
        let matrix = |name: &str| -> PyResult<Bound<'_, PyAny>> { gate_matrix.call_method0(name) };
        let was_recording = self.recording;
        self.recording = false;
        let result = (|| {
            for (&index, &operator) in targets.iter().zip(&operators) {
                match operator {
                    'X' => self.apply_gate_recorded(
                        py,
                        &matrix("h_MAT")?,
                        &[index],
                        &[],
                        None,
                        None,
                        None,
                    )?,
                    'Y' => {
                        self.apply_gate_recorded(
                            py,
                            &matrix("sdg_MAT")?,
                            &[index],
                            &[],
                            None,
                            None,
                            None,
                        )?;
                        self.apply_gate_recorded(
                            py,
                            &matrix("h_MAT")?,
                            &[index],
                            &[],
                            None,
                            None,
                            None,
                        )?;
                    }
                    _ => {}
                }
            }

            let mut probabilities = Vec::with_capacity(self.dimension());
            {
                let guard = self.state_vector.bind(py).try_readonly()?;
                let state = contiguous_slice(&guard, "state_vector")?;
                for amplitude in state {
                    probabilities.push(amplitude.norm_sqr());
                }
            }
            let total: f64 = probabilities.iter().sum();
            if total > 0.0 {
                for probability in &mut probabilities {
                    *probability /= total;
                }
            }
            let mut cumulative = Vec::with_capacity(probabilities.len());
            let mut running = 0.0;
            for probability in probabilities {
                running += probability;
                cumulative.push(running);
            }
            if let Some(last) = cumulative.last_mut() {
                *last = 1.0;
            }

            let measured_bits: Vec<usize> = targets
                .iter()
                .map(|&index| self.bit_position_unchecked(index))
                .collect();
            let draws: Vec<f64> = {
                let rng = &mut self.rng;
                (0..shots).map(|_| rng.random()).collect()
            };
            let mut expectation = 0.0;
            for draw in draws {
                let outcome = cumulative.partition_point(|value| *value < draw);
                let mut parity = 1.0;
                for (&bit, &operator) in measured_bits.iter().zip(&operators) {
                    if operator != 'I' && (outcome >> bit) & 1 != 0 {
                        parity = -parity;
                    }
                }
                expectation += parity;
            }

            for (&index, &operator) in targets.iter().zip(&operators) {
                match operator {
                    'X' => self.apply_gate_recorded(
                        py,
                        &matrix("h_MAT")?,
                        &[index],
                        &[],
                        None,
                        None,
                        None,
                    )?,
                    'Y' => {
                        self.apply_gate_recorded(
                            py,
                            &matrix("h_MAT")?,
                            &[index],
                            &[],
                            None,
                            None,
                            None,
                        )?;
                        self.apply_gate_recorded(
                            py,
                            &matrix("s_MAT")?,
                            &[index],
                            &[],
                            None,
                            None,
                            None,
                        )?;
                    }
                    _ => {}
                }
            }
            Ok(expectation / shots as f64)
        })();
        self.recording = was_recording;
        result
    }

    #[pyo3(signature = (circuit_func, num_samples = 1000))]
    fn sample(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        circuit_func: Bound<'_, PyAny>,
        num_samples: isize,
    ) -> PyResult<Py<PyDict>> {
        sample_impl(py, slf, &circuit_func, num_samples, None)
    }

    #[pyo3(signature = (circuit_func, num_samples = 1000, **kwargs))]
    fn sample_parallel(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        circuit_func: Bound<'_, PyAny>,
        num_samples: isize,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        sample_impl(py, slf, &circuit_func, num_samples, kwargs)
    }

    fn to_openqasm3(&self, py: Python<'_>) -> PyResult<String> {
        py.import("quantum.qasm")?
            .call_method1(
                "to_openqasm3",
                (self.num_qubits, &self.name, self.op_hist.clone_ref(py)),
            )?
            .extract()
    }

    fn register_operator(
        &mut self,
        py: Python<'_>,
        name: String,
        matrix: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyValueError::new_err(
                "Operator name must be a non-empty string",
            ));
        }
        let operators = self.custom_operators.bind(py);
        if operators.contains(&name)? {
            return Err(PyValueError::new_err(format!(
                "Operator '{name}' already registered"
            )));
        }
        operators.set_item(name, matrix)
    }

    fn get_operator(&self, py: Python<'_>, name: String) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .custom_operators
            .bind(py)
            .get_item(name)?
            .map(|value| value.unbind()))
    }

    fn list_operators(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.custom_operators
            .bind(py)
            .keys()
            .iter()
            .map(|key| key.extract())
            .collect::<PyResult<Vec<String>>>()
    }
}

/// `with qc.suppress_history(): ...`
#[pyclass(name = "_SuppressHistory")]
pub struct SuppressHistory {
    computer: Py<QuantumComputer>,
    was: bool,
}

#[pymethods]
impl SuppressHistory {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.computer)?;
        Ok(())
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Py<QuantumComputer>> {
        slf.was = slf.computer.bind(py).borrow().recording;
        slf.computer.bind(py).borrow_mut().recording = false;
        Ok(slf.computer.clone_ref(py))
    }

    #[pyo3(signature = (_exc_type, _exc_value, _traceback))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.computer.bind(py).borrow_mut().recording = self.was;
        Ok(false)
    }
}

fn qft_impl(
    py: Python<'_>,
    slf: PyRefMut<'_, QuantumComputer>,
    qubits: Vec<Bound<'_, PyAny>>,
    inverse: bool,
    record: Option<bool>,
) -> PyResult<Py<QuantumComputer>> {
    let me: Py<QuantumComputer> = slf.into();
    let targets = resolve(py, &me.bind(py).borrow(), &qubits)?;
    if targets.is_empty() {
        return Ok(me);
    }

    let module = py.import("stl.quantum_algs.shors_alg_lib")?;
    let builder = if inverse {
        "iqft_program"
    } else {
        "qft_program"
    };
    let program = module.call_method1(builder, (me.bind(py), targets))?;

    if let Some(record) = record {
        let was = me.bind(py).borrow().recording;
        me.bind(py).borrow_mut().recording = record;
        let applied = me.bind(py).call_method1("apply", (program,));
        me.bind(py).borrow_mut().recording = was;
        applied?;
    } else {
        me.bind(py).call_method1("apply", (program,))?;
    }
    Ok(me)
}

fn sample_impl(
    py: Python<'_>,
    slf: PyRefMut<'_, QuantumComputer>,
    circuit_func: &Bound<'_, PyAny>,
    num_samples: isize,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDict>> {
    if num_samples <= 0 {
        return Err(PyValueError::new_err(
            "Number of samples must be a positive integer.",
        ));
    }
    let me: Py<QuantumComputer> = slf.into();
    {
        let mut computer = me.bind(py).borrow_mut();
        let dimension = computer.dimension();
        computer.state_vector = new_amplitude_array(py, dimension)?;
        computer.op_hist = PyList::empty(py).unbind();
        for qubit in &computer.qubits {
            qubit.bind(py).borrow_mut().state = QubitState::Alive;
        }
    }

    match kwargs {
        Some(kwargs) => {
            circuit_func.call((me.bind(py),), Some(kwargs))?;
        }
        None => {
            circuit_func.call1((me.bind(py),))?;
        }
    }

    let counts = me
        .bind(py)
        .borrow_mut()
        .sample_counts(py, num_samples as usize)?;
    let result = PyDict::new(py);
    let width = me.bind(py).borrow().num_qubits;
    for (index, count) in counts.into_iter().enumerate() {
        if count != 0 {
            result.set_item(format!("{index:0width$b}"), count)?;
        }
    }
    Ok(result.unbind())
}

/// The string form of `apply`: custom operators plus the global registry.
fn apply_operation(
    py: Python<'_>,
    me: &Py<QuantumComputer>,
    operation: &Bound<'_, PyAny>,
    target_qubits: Option<Vec<Bound<'_, PyAny>>>,
    control_qubits: Option<Vec<Bound<'_, PyAny>>>,
) -> PyResult<()> {
    if let Ok(name) = operation.cast::<pyo3::types::PyString>() {
        return apply_named_operator(py, me, name.to_str()?, target_qubits, control_qubits);
    }

    if operation.cast::<PyList>().is_ok() || operation.cast::<PyTuple>().is_ok() {
        if non_empty(&target_qubits) {
            return Err(PyValueError::new_err(
                "A program names its own qubits; drop the target operand.",
            ));
        }
        let mut gates = operation.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if non_empty(&control_qubits) {
            let extra = resolve(py, &me.bind(py).borrow(), control_qubits.as_ref().unwrap())?;
            let mapping: Vec<usize> = (0..me.bind(py).borrow().num_qubits).collect();
            let mut remapped = Vec::with_capacity(gates.len());
            for gate in &gates {
                remapped.push(gate.call_method1("remap", (mapping.clone(), extra.clone()))?);
            }
            gates = remapped;
        }
        return me.bind(py).borrow_mut().apply_program(py, &gates);
    }

    // Both the Rust `Gate` class and Python's `quantum.gate_abs.Gate` carry
    // these three attributes, so duck-type them the way `_apply_program` does.
    let is_gate_record = operation.cast::<Gate>().is_ok()
        || (operation.hasattr("matrix")?
            && operation.hasattr("controls")?
            && operation.hasattr("targets")?);
    if is_gate_record {
        return me
            .bind(py)
            .borrow_mut()
            .apply_program(py, std::slice::from_ref(operation));
    }

    let Some(targets) = target_qubits else {
        return Err(PyValueError::new_err("This operation needs target qubits."));
    };
    let controls = control_qubits.unwrap_or_default();
    let target_indices = resolve(py, &me.bind(py).borrow(), &targets)?;
    let control_indices = resolve(py, &me.bind(py).borrow(), &controls)?;

    if operation.is_callable() {
        operation.call1((me.bind(py), target_indices, control_indices))?;
        return Ok(());
    }
    me.bind(py).borrow_mut().apply_gate_recorded(
        py,
        operation,
        &target_indices,
        &control_indices,
        None,
        None,
        None,
    )
}

fn apply_named_operator(
    py: Python<'_>,
    me: &Py<QuantumComputer>,
    name: &str,
    target_qubits: Option<Vec<Bound<'_, PyAny>>>,
    control_qubits: Option<Vec<Bound<'_, PyAny>>>,
) -> PyResult<()> {
    let operators = py.import("lang_systems.operators")?;
    let (mut op, mut kind): (Option<Py<PyAny>>, Option<String>) = operators
        .call_method1("get_global_operator", (name,))?
        .extract()?;
    if op.is_none() {
        op = me.bind(py).borrow().get_operator(py, name.to_string())?;
    }
    let op = op.ok_or_else(|| PyValueError::new_err(format!("Unknown operator: '{name}'")))?;
    let bound = op.bind(py);
    let requires_instance: bool = bound
        .getattr("_requires_instance")
        .and_then(|value| value.extract())
        .unwrap_or(false);
    if kind.is_none() {
        kind = Some(
            bound
                .getattr("_kind")
                .and_then(|value| value.extract())
                .unwrap_or_else(|_| "matrix".to_string()),
        );
    }
    let kind: String = operators
        .call_method1("normalize_kind", (kind.unwrap(),))?
        .extract()?;
    let call = |args: Vec<Py<PyAny>>| -> PyResult<Py<PyAny>> {
        let tuple = PyTuple::new(py, args)?;
        if requires_instance {
            let mut full = vec![me.clone_ref(py).into_any()];
            full.extend(tuple.iter().map(|item| item.unbind()));
            Ok(bound.call(PyTuple::new(py, full)?, None)?.unbind())
        } else {
            Ok(bound.call(tuple, None)?.unbind())
        }
    };
    let index_arg =
        |value: usize| -> PyResult<Py<PyAny>> { Ok(value.into_pyobject(py)?.into_any().unbind()) };

    match kind.as_str() {
        "nullary" => {
            if bound.is_callable() {
                let _ = call(Vec::new())?;
            }
            Ok(())
        }
        "matrix" => {
            let matrix = if bound.is_callable() {
                call(Vec::new())?
            } else {
                op.clone_ref(py)
            };
            let numpy_array = numpy_module(py)?.bind(py).getattr("ndarray")?;
            if !matrix.bind(py).is_instance(&numpy_array)? {
                return Err(PyValueError::new_err(format!(
                    "Operator '{name}' must return a numpy array"
                )));
            }
            let Some(targets) = target_qubits.filter(|items| !items.is_empty()) else {
                return Err(PyValueError::new_err(
                    "Custom operator requires target qubits",
                ));
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("qasm_name", name)?;
            me.bind(py).call_method(
                "apply_gate",
                (matrix, targets, control_qubits.unwrap_or_default()),
                Some(&kwargs),
            )?;
            Ok(())
        }
        "unary_pre" | "unary_post" => {
            let Some(targets) = target_qubits else {
                return Err(PyValueError::new_err(
                    "Unary operator requires exactly one target qubit",
                ));
            };
            if targets.len() != 1 {
                return Err(PyValueError::new_err(
                    "Unary operator requires exactly one target qubit",
                ));
            }
            let index = resolve(py, &me.bind(py).borrow(), &targets)?[0];
            let _ = call(vec![index_arg(index)?])?;
            Ok(())
        }
        "binary" => {
            let Some(targets) = target_qubits else {
                return Err(PyValueError::new_err(
                    "Binary operator requires exactly two target qubits (control, target)",
                ));
            };
            if targets.len() != 2 {
                return Err(PyValueError::new_err(
                    "Binary operator requires exactly two target qubits (control, target)",
                ));
            }
            let indices = resolve(py, &me.bind(py).borrow(), &targets)?;
            let args = indices
                .into_iter()
                .map(index_arg)
                .collect::<PyResult<Vec<_>>>()?;
            let _ = call(args)?;
            Ok(())
        }
        "ternary" => {
            let Some(targets) = target_qubits else {
                return Err(PyValueError::new_err(
                    "Ternary operator requires exactly three target qubits",
                ));
            };
            if targets.len() != 3 {
                return Err(PyValueError::new_err(
                    "Ternary operator requires exactly three target qubits",
                ));
            }
            let indices = resolve(py, &me.bind(py).borrow(), &targets)?;
            let args = indices
                .into_iter()
                .map(index_arg)
                .collect::<PyResult<Vec<_>>>()?;
            let _ = call(args)?;
            Ok(())
        }
        _ => {
            let Some(targets) = target_qubits else {
                return Err(PyValueError::new_err(
                    "Callable operator requires target qubits",
                ));
            };
            let target_indices = resolve(py, &me.bind(py).borrow(), &targets)?;
            let control_indices = resolve(
                py,
                &me.bind(py).borrow(),
                &control_qubits.unwrap_or_default(),
            )?;
            let _ = call(vec![
                target_indices.into_pyobject(py)?.into_any().unbind(),
                control_indices.into_pyobject(py)?.into_any().unbind(),
            ])?;
            Ok(())
        }
    }
}
