//! Python bindings for the Quantar state-vector simulator.
//!
//! The two classes here mirror `quantum.qubit_abs`: `Qubit` is a cheap handle
//! onto one wire, and `QuantumComputer` owns the amplitude array. The amplitude
//! storage is a real NumPy array, not a Rust `Vec`, so `state_vector` behaves
//! like the Python attribute (it can be read in place and assigned back).
//!
//! This extension is built with PyO3's `abi3-py312` feature: it links only to
//! CPython's stable ABI, so the same binary loads on Python 3.12 and newer.
//! NumPy is deliberately a runtime dependency; the `numpy` crate resolves its
//! C API when the module is imported, which keeps the wheel independent of a
//! particular CPython minor version.

mod api;
mod gate;
mod gates;

use gate::Gate;
use num_complex::Complex64;
use numpy::{PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyList, PyModule, PySlice, PyString};
use rand::RngExt;
use rayon::prelude::*;
use std::sync::OnceLock;

// Importing a module already loaded in sys.modules is cheap, but these helpers
// are called on every gate construction, so keep the handles around.
static NUMPY: OnceLock<Py<PyModule>> = OnceLock::new();
static GATE_MATRIX: OnceLock<Py<PyModule>> = OnceLock::new();

pub(crate) fn numpy_module(py: Python<'_>) -> PyResult<&Py<PyModule>> {
    if let Some(module) = NUMPY.get() {
        return Ok(module);
    }
    let module = py.import("numpy")?.unbind();
    Ok(NUMPY.get_or_init(|| module))
}

pub(crate) fn gate_matrix_module(py: Python<'_>) -> PyResult<&Py<PyModule>> {
    if let Some(module) = GATE_MATRIX.get() {
        return Ok(module);
    }
    let module = py.import("quantum.gate_matrix")?.unbind();
    Ok(GATE_MATRIX.get_or_init(|| module))
}

/// Convert any array-like to an owned-reference complex128 NumPy array.
pub(crate) fn as_complex_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
    let numpy = numpy_module(py)?.bind(py);
    let array = numpy.call_method1("asarray", (value, "complex128"))?;
    // Kernels read raw slices and the new Rust-side unitarity check reads raw
    // bytes, so make one contiguous copy up front (a no-op for normal gates).
    let array = numpy.call_method1("ascontiguousarray", (&array,))?;
    array
        .cast_into::<PyArrayDyn<Complex64>>()
        .map_err(|_| PyTypeError::new_err("Expected an array of complex numbers."))
}

/// The attribute view Python gate objects expose; Rust `Gate` implements the
/// same names, so both work in `op_hist`, `apply`, `uncompute` and QASM export.
pub(crate) struct GateView<'py> {
    #[allow(dead_code)]
    pub(crate) obj: Bound<'py, PyAny>,
    pub(crate) matrix: Bound<'py, PyArrayDyn<Complex64>>,
    pub(crate) controls: Vec<usize>,
    pub(crate) targets: Vec<usize>,
}

pub(crate) fn gate_view<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<GateView<'py>> {
    // Rust gates already hold a converted, contiguous NumPy array. Skipping
    // the Python attribute round-trip matters for large programs.
    if let Ok(gate) = obj.cast::<Gate>() {
        return Ok(gate.borrow().view(py));
    }

    let matrix = as_complex_array(py, &obj.getattr("matrix")?)?;
    let controls = obj.getattr("controls")?.extract()?;
    let targets = obj.getattr("targets")?.extract()?;
    Ok(GateView {
        obj: obj.clone(),
        matrix,
        controls,
        targets,
    })
}

#[derive(Debug, Clone, Copy)]
enum QubitState {
    Alive,
    Measured { outcome: u8 },
}

/// A handle to one physical qubit. Holds an index, never any amplitudes.
#[pyclass(name = "Qubit")]
pub struct Qubit {
    // `Option` so the garbage collector can break the
    // QuantumComputer -> Qubit -> QuantumComputer reference cycle.
    computer: Option<Py<QuantumComputer>>,

    #[pyo3(get)]
    index: usize,

    name: String,

    state: QubitState,
}

#[pymethods]
impl Qubit {
    #[new]
    #[pyo3(signature = (computer, index, name = "qubit".to_string()))]
    fn new(computer: Py<QuantumComputer>, index: usize, name: String) -> Self {
        Self {
            computer: Some(computer),
            index,
            name,
            state: QubitState::Alive,
        }
    }

    fn __repr__(&self) -> String {
        format!("{}({})", self.name, self.index)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(computer) = &self.computer {
            visit.call(computer)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.computer = None;
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[getter]
    fn computer(&self, py: Python<'_>) -> PyResult<Py<QuantumComputer>> {
        Ok(self.computer_ref()?.clone_ref(py))
    }

    #[getter]
    fn is_measured(&self) -> bool {
        matches!(self.state, QubitState::Measured { .. })
    }

    #[getter]
    fn _true_val(&self) -> Option<u8> {
        match self.state {
            QubitState::Measured { outcome } => Some(outcome),
            QubitState::Alive => None,
        }
    }

    fn is_alive(&self) -> bool {
        matches!(self.state, QubitState::Alive)
    }

    fn _assert_alive(&self) -> PyResult<()> {
        if !self.is_alive() {
            return Err(PyValueError::new_err(format!(
                "Qubit {} is locked after measurement (no gates can be applied).",
                self.name
            )));
        }
        Ok(())
    }

    fn alpha(&self, py: Python<'_>) -> PyResult<f64> {
        self.computer_ref()?
            .bind(py)
            .borrow()
            .alpha_at(py, self.index)
    }

    fn measure(&mut self, py: Python<'_>) -> PyResult<u8> {
        self._assert_alive()?;
        let computer = self.computer_ref()?.clone_ref(py);
        let outcome = computer.bind(py).borrow_mut().collapse_at(py, self.index)?;
        self.state = QubitState::Measured { outcome };
        Ok(outcome)
    }

    /// Apply a single-qubit gate to this qubit.
    fn apply_gate(&self, py: Python<'_>, gate_matrix: &Bound<'_, PyAny>) -> PyResult<()> {
        self._assert_alive()?;
        let computer = self.computer_ref()?.clone_ref(py);
        computer.bind(py).borrow_mut().apply_gate_recorded(
            py,
            gate_matrix,
            &[self.index],
            &[],
            None,
            None,
            None,
        )
    }
}

impl Qubit {
    fn computer_ref(&self) -> PyResult<&Py<QuantumComputer>> {
        self.computer.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("This qubit is no longer attached to a QuantumComputer.")
        })
    }
}

#[pyclass(name = "QuantumComputer")]
pub struct QuantumComputer {
    #[pyo3(get)]
    num_qubits: usize,

    name: String,

    #[pyo3(get)]
    qubits: Vec<Py<Qubit>>,

    // Stored as NumPy arrays so `state_vector` is a real, in-place-mutable
    // Python attribute rather than a copy produced on every access.
    state_vector: Py<PyArray2<Complex64>>,
    scratch: Py<PyArray2<Complex64>>,

    // Real Python containers so `qc.op_hist.append(...)` and
    // `qc.register_operator(...)` behave exactly like the Python backend.
    op_hist: Py<PyList>,
    recording: bool,
    custom_operators: Py<PyDict>,
}

#[pymethods]
impl QuantumComputer {
    #[new]
    #[pyo3(signature = (num_qubits, name = "QC1".to_string(), qubit_names = None))]
    fn new(
        py: Python<'_>,
        num_qubits: usize,
        name: String,
        qubit_names: Option<Vec<String>>,
    ) -> PyResult<Py<Self>> {
        if num_qubits >= usize::BITS as usize {
            return Err(PyValueError::new_err(format!(
                "num_qubits must be smaller than {} (got {num_qubits}).",
                usize::BITS
            )));
        }

        let names = match qubit_names {
            Some(list) => {
                if list.len() != num_qubits {
                    return Err(PyValueError::new_err(format!(
                        "The number of qubit names ({}) must match the number of qubits ({}).",
                        list.len(),
                        num_qubits
                    )));
                }
                list
            }
            None => (0..num_qubits).map(|i| format!("q{i}")).collect(),
        };

        let dimension = 1usize << num_qubits;
        let state_vector = new_amplitude_array(py, dimension)?;
        let scratch = new_amplitude_array(py, dimension)?;

        let computer = Py::new(
            py,
            Self {
                num_qubits,
                name,
                qubits: Vec::with_capacity(num_qubits),
                state_vector,
                scratch,
                op_hist: PyList::empty(py).unbind(),
                recording: true,
                custom_operators: PyDict::new(py).unbind(),
            },
        )?;

        let mut handles = Vec::with_capacity(num_qubits);
        for (index, qubit_name) in names.into_iter().enumerate() {
            handles.push(Py::new(
                py,
                Qubit::new(computer.clone_ref(py), index, qubit_name),
            )?);
        }
        computer.bind(py).borrow_mut().qubits = handles;

        Ok(computer)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for qubit in &self.qubits {
            visit.call(qubit)?;
        }
        visit.call(&self.state_vector)?;
        visit.call(&self.scratch)?;
        visit.call(&self.op_hist)?;
        visit.call(&self.custom_operators)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // The amplitudes are not part of the cycle; dropping the handles is
        // enough for the collector to reclaim the whole register.
        self.qubits.clear();
    }

    #[getter]
    fn dimension(&self) -> usize {
        1usize << self.num_qubits
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[getter]
    fn op_hist(&self, py: Python<'_>) -> Py<PyList> {
        self.op_hist.clone_ref(py)
    }

    #[setter]
    fn set_op_hist(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let list = PyList::new(py, value.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        self.op_hist = list.unbind();
        Ok(())
    }

    #[getter]
    fn _recording(&self) -> bool {
        self.recording
    }

    #[setter(_recording)]
    fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
    }

    #[getter]
    fn _custom_operators(&self, py: Python<'_>) -> Py<PyDict> {
        self.custom_operators.clone_ref(py)
    }

    #[getter]
    fn _scratch(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.scratch.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let listed: Vec<String> = self
            .qubits
            .iter()
            .map(|qubit| qubit.bind(py).borrow().__repr__())
            .collect();
        format!(
            "System: {} ({} Qubits)\n  +- {}",
            self.name,
            self.num_qubits,
            listed.join("\n  +- ")
        )
    }

    /// Turn a qubit index into a physical bit position (q0 is the MSB).
    fn bit_position(&self, qubit_index: isize) -> PyResult<usize> {
        if qubit_index < 0 || qubit_index as usize >= self.num_qubits {
            return Err(PyIndexError::new_err(format!(
                "Qubit index {qubit_index} is out of range for {} qubits.",
                self.num_qubits
            )));
        }
        Ok(self.bit_position_unchecked(qubit_index as usize))
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if key.cast::<Qubit>().is_ok() {
            return Ok(key.clone().unbind());
        }

        if let Ok(q) = key.cast::<Qubit>() {
            let idx = q.borrow().index;
            if idx < self.num_qubits && self.qubits[idx].is(q) {
                return Ok(key.clone().unbind());
            }
            return Err(PyValueError::new_err(
                "This Qubit does not belong to this QuantumComputer.",
            ));
        }

        if let Ok(raw_index) = key.extract::<isize>() {
            let length = self.num_qubits as isize;
            let index = if raw_index < 0 {
                raw_index + length
            } else {
                raw_index
            };
            if index < 0 || index >= length {
                return Err(PyIndexError::new_err(format!(
                    "Qubit index {raw_index} is out of range for {} qubits.",
                    self.num_qubits
                )));
            }
            return Ok(self.qubits[index as usize].clone_ref(py).into_any());
        }

        if let Ok(name) = key.cast::<PyString>() {
            let target = name.to_str()?;
            for qubit in &self.qubits {
                if qubit.bind(py).borrow().name == target {
                    return Ok(qubit.clone_ref(py).into_any());
                }
            }
            return Err(PyKeyError::new_err(format!(
                "No qubit found with name '{target}'."
            )));
        }

        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.num_qubits as isize)?;
            let mut result = Vec::with_capacity(indices.slicelength);
            let mut current = indices.start;
            for _ in 0..indices.slicelength {
                result.push(self.qubits[current as usize].clone_ref(py));
                current += indices.step;
            }
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyTypeError::new_err(
            "Key must be a Qubit, integer index, string name, or slice.",
        ))
    }

    #[getter]
    fn state_vector(&self, py: Python<'_>) -> Py<PyArray2<Complex64>> {
        self.state_vector.clone_ref(py)
    }

    #[setter]
    fn set_state_vector(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.state_vector = state_array_from_value(py, value, self.dimension(), self.num_qubits)?;
        Ok(())
    }

    /// Apply a gate to `target_qubits`, optionally controlled.
    ///
    /// The extra QASM/history arguments mirror `qubit_abs.QuantumComputer`
    /// so callers can switch backends without changing their code; this
    /// kernel-only implementation does not record history.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (gate_matrix, target_qubits, control_qubits = None, qasm_name = None, qasm_params = None, log_history = None))]
    fn apply_gate<'py>(
        &mut self,
        py: Python<'py>,
        gate_matrix: &Bound<'py, PyAny>,
        target_qubits: Vec<Bound<'py, PyAny>>,
        control_qubits: Option<Vec<Bound<'py, PyAny>>>,
        qasm_name: Option<String>,
        qasm_params: Option<Vec<f64>>,
        log_history: Option<bool>,
    ) -> PyResult<()> {
        let targets = self.resolve_qubit_indices(py, &target_qubits)?;
        let controls = match &control_qubits {
            Some(items) => self.resolve_qubit_indices(py, items)?,
            None => Vec::new(),
        };

        self.apply_gate_recorded(
            py,
            gate_matrix,
            &targets,
            &controls,
            qasm_name,
            qasm_params,
            log_history,
        )
    }
}

impl QuantumComputer {
    fn bit_position_unchecked(&self, qubit_index: usize) -> usize {
        self.num_qubits - 1 - qubit_index
    }

    /// Resolve handles, indices and names to plain qubit indices, checking
    /// that each qubit is still alive (measurement locks it).
    fn resolve_qubit_indices(
        &self,
        py: Python<'_>,
        items: &[Bound<'_, PyAny>],
    ) -> PyResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(items.len());
        for item in items {
            let resolved = self.__getitem__(py, item)?;
            let resolved = resolved.bind(py);
            let qubit = resolved.cast::<Qubit>().map_err(|_| {
                PyTypeError::new_err("Expected a single qubit, not a list of qubits.")
            })?;
            let qubit = qubit.borrow();
            qubit._assert_alive()?;
            if qubit.index >= self.num_qubits {
                return Err(PyIndexError::new_err(format!(
                    "Qubit index {} is out of range for {} qubits.",
                    qubit.index, self.num_qubits
                )));
            }
            indices.push(qubit.index);
        }
        Ok(indices)
    }

    /// Apply a gate and append its record to `op_hist` when history is on.
    /// This is the path shared by `QuantumComputer.apply_gate`, `Qubit.apply_gate`
    /// and Python's composite/circuit helpers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_gate_recorded(
        &mut self,
        py: Python<'_>,
        gate_matrix: &Bound<'_, PyAny>,
        targets: &[usize],
        controls: &[usize],
        qasm_name: Option<String>,
        qasm_params: Option<Vec<f64>>,
        log_history: Option<bool>,
    ) -> PyResult<()> {
        let record = log_history.unwrap_or(true);
        let gate = Py::new(
            py,
            Gate::new(
                py,
                gate_matrix,
                controls.to_vec(),
                targets.to_vec(),
                qasm_name,
                qasm_params,
                false,
            )?,
        )?;
        let view = gate.bind(py).borrow().view(py);
        self.run_gate_view(py, &view)?;
        if record && self.recording {
            self.op_hist.bind(py).append(gate.bind(py).as_any())?;
        }
        Ok(())
    }

    fn alpha_at(&self, py: Python<'_>, qubit_index: usize) -> PyResult<f64> {
        if qubit_index >= self.num_qubits {
            return Err(PyIndexError::new_err(format!(
                "Qubit index {qubit_index} is out of range for {} qubits.",
                self.num_qubits
            )));
        }
        let bit = self.bit_position_unchecked(qubit_index);
        let mask = 1usize << bit;

        let state_guard = self.state_vector.bind(py).try_readonly()?;
        let state = gates::contiguous_slice(&state_guard, "state_vector")?;

        Ok(py.detach(move || {
            let compute = |(index, amplitude): (usize, &Complex64)| {
                if index & mask != 0 {
                    amplitude.norm_sqr()
                } else {
                    0.0
                }
            };
            if state.len() <= 1 << 12 {
                state.iter().enumerate().map(compute).sum()
            } else {
                state.par_iter().enumerate().map(compute).sum()
            }
        }))
    }

    fn collapse_at(&mut self, py: Python<'_>, qubit_index: usize) -> PyResult<u8> {
        let probability = self.alpha_at(py, qubit_index)?.clamp(0.0, 1.0);
        let mut rng = rand::rng();
        let outcome = u8::from(rng.random_bool(probability));

        let bit = self.bit_position_unchecked(qubit_index);
        let mask = 1usize << bit;

        {
            let state_guard = self.state_vector.bind(py).try_readonly()?;
            let state = gates::contiguous_slice(&state_guard, "state_vector")?;
            let mut scratch_guard = self.scratch.bind(py).try_readwrite()?;
            let scratch = gates::contiguous_slice_mut(&mut scratch_guard, "scratch buffer")?;

            py.detach(move || {
                // Renormalise on the norm of the surviving branch, like the
                // Python implementation, so an imperfectly normalised state
                // stays sane.
                let survives = |index: usize| ((index & mask) != 0) as u8 == outcome;
                let norm_squared: f64 = if state.len() <= 1 << 12 {
                    state
                        .iter()
                        .enumerate()
                        .map(|(index, amplitude)| {
                            if survives(index) {
                                amplitude.norm_sqr()
                            } else {
                                0.0
                            }
                        })
                        .sum()
                } else {
                    state
                        .par_iter()
                        .enumerate()
                        .map(|(index, amplitude)| {
                            if survives(index) {
                                amplitude.norm_sqr()
                            } else {
                                0.0
                            }
                        })
                        .sum()
                };
                let norm_factor = if norm_squared > 0.0 {
                    norm_squared.sqrt().recip()
                } else {
                    0.0
                };

                let write = |(index, amplitude): (usize, &mut Complex64)| {
                    if survives(index) {
                        *amplitude = state[index] * norm_factor;
                    } else {
                        *amplitude = Complex64::new(0.0, 0.0);
                    }
                };
                if state.len() <= 1 << 12 {
                    scratch.iter_mut().enumerate().for_each(write);
                } else {
                    scratch.par_iter_mut().enumerate().for_each(write);
                }
            });
        }

        std::mem::swap(&mut self.state_vector, &mut self.scratch);
        Ok(outcome)
    }
}

fn new_amplitude_array(py: Python<'_>, dimension: usize) -> PyResult<Py<PyArray2<Complex64>>> {
    let array = PyArray2::<Complex64>::zeros(py, [dimension, 1], false);
    {
        let mut writable = array.readwrite();
        let amplitudes = writable
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("Failed to allocate the state vector."))?;
        amplitudes[0] = Complex64::new(1.0, 0.0);
    }
    Ok(array.unbind())
}

/// Convert an assigned `state_vector` value to a C-contiguous `(dimension, 1)`
/// complex array, rejecting values with the wrong number of amplitudes.
fn state_array_from_value(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dimension: usize,
    num_qubits: usize,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let numpy = numpy_module(py)?.bind(py);
    let array = numpy.call_method1("asarray", (value, "complex128"))?;
    let size: usize = array.getattr("size")?.extract()?;
    if size != dimension {
        return Err(PyValueError::new_err(format!(
            "state_vector for {num_qubits} qubits must hold {dimension} amplitudes, got {size}."
        )));
    }

    let array = numpy.call_method1("ascontiguousarray", (&array,))?;
    let array = array.call_method1("reshape", (dimension, 1usize))?;
    let array = array.cast_into::<PyArray2<Complex64>>()?;
    Ok(array.unbind())
}

#[pymodule]
fn quantar_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Qubit>()?;
    m.add_class::<QuantumComputer>()?;
    m.add_class::<Gate>()?;
    m.add_class::<api::SuppressHistory>()?;
    Ok(())
}
