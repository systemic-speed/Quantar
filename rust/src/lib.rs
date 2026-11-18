#![allow(dead_code)]
mod gates;

use num_complex::Complex64;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyString};
use rand::RngExt;
use rayon::prelude::*;
use numpy::PyReadonlyArray2;

#[derive(Debug, Clone, Copy)]
enum QubitState {
    Alive,
    Measured { outcome: u8 },
}

#[pyclass(name = "Qubit")]
pub struct Qubit {
    computer: Py<QuantumComputer>,

    #[pyo3(get)]
    index: usize,

    #[pyo3(get)]
    name: String,

    state: QubitState,
}

#[pymethods]
impl Qubit {
    #[new]
    #[pyo3(signature = (computer, index, name = "qubit".to_string()))]
    fn new(computer: Py<QuantumComputer>, index: usize, name: String) -> Self {
        Self {
            computer,
            index,
            name,
            state: QubitState::Alive,
        }
    }

    fn __repr__(&self) -> String {
        format!("{}({})", self.name, self.index)
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

    fn alpha(&self, py: Python<'_>) -> f64 {
        self.computer.bind(py).borrow().alpha_at(self.index)
    }

    fn measure(&mut self, py: Python<'_>) -> PyResult<u8> {
        self._assert_alive()?;
        let outcome = self.computer.bind(py).borrow_mut().collapse_at(self.index);
        self.state = QubitState::Measured { outcome };
        Ok(outcome)
    }
}

#[pyclass(name = "QuantumComputer")]
pub struct QuantumComputer {
    #[pyo3(get)]
    num_qubits: usize,

    #[pyo3(get)]
    name: String,

    #[pyo3(get)]
    qubits: Vec<Py<Qubit>>,

    state_vector: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl QuantumComputer {
    fn alpha_at(&self, qubit_index: usize) -> f64 {
        let bit = self.bit_position(qubit_index);
        let mask = 1usize << bit;

        self.state_vector.par_iter()
                         .enumerate()
                         .filter_map(|(idx, amp)| ((idx & mask) != 0).then(|| amp.norm_sqr()))
                         .sum()
    }

    fn collapse_at(&mut self, qubit_index: usize) -> u8 {
        let prob = self.alpha_at(qubit_index).clamp(0.0, 1.0);
        let mut rng = rand::rng();
        let outcome: u8 = if rng.random_bool(prob) {1} else {0};

        let bit = self.bit_position(qubit_index);
        let mask = 1usize << bit;

        let outcome_prob = if outcome == 1 {prob} else {1.0 - prob};
        let norm_factor = if outcome_prob > 0.0 {1./outcome_prob.sqrt()} else {0.};

        self.state_vector.par_iter_mut()
                         .enumerate()
                         .for_each(|(idx, amp)|{
                            let bit_val = ((idx & mask) != 0) as u8;
                            if bit_val == outcome {
                                *amp *= norm_factor;
                            }
                            else {
                                *amp = Complex64::new(0., 0.);
                            }
                         });
        outcome
    }

    fn index_of(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<usize> {
        let q_obj = self.__getitem__(py, &key)?;
        let q_ref = q_obj.bind(py).cast::<Qubit>()?.borrow();
        Ok(q_ref.index)
    }

    fn _resolve_qubit_bits(&mut self, py: Python<'_>, items: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
        let mut bits = Vec::with_capacity(items.len());
        for item in items {
            let q_obj = self.__getitem__(py, &item)?;
            let q_ref = q_obj.bind(py).cast::<Qubit>()?.borrow();
            q_ref._assert_alive()?;
            bits.push(self.bit_position(q_ref.index));
        }
        Ok(bits)
    }

    fn _indices_to_bits(&self, py: Python<'_>, indices: &[usize]) -> PyResult<Vec<usize>> {
        let mut bits = Vec::with_capacity(indices.len());
        for &idx in indices {
            let q = self.qubits.get(idx).ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "Qubit index {idx} is out of range for {} qubits.",
                    self.num_qubits
                ))
            })?;
            q.bind(py).borrow()._assert_alive()?;
            bits.push(self.bit_position(idx));
        }
        Ok(bits)
    }

    fn execute_gate(&mut self, py: Python<'_>, control_qubits: &[usize], target_qubits: &[usize], matrix: PyReadonlyArray2<'_, Complex64>) -> PyResult<()> {
        let ctrl_bits = self._indices_to_bits(py, control_qubits)?;
        let targ_bits = self._indices_to_bits(py, target_qubits)?;

        self.exec_gate(&ctrl_bits, &targ_bits, matrix)
    }
}

#[pymethods]
impl QuantumComputer {
    #[new]
    #[pyo3(signature = (num_qubits, name = "QC1".to_string(), qubit_names = None))]
    fn new(py: Python<'_>, num_qubits: usize, name: String, qubit_names: Option<Vec<String>>) -> PyResult<Py<Self>> {

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

        let dim = 1usize << num_qubits;
        let mut state_vector = vec![Complex64::new(0.0, 0.0); dim];
        if dim > 0 {
            state_vector[0] = Complex64::new(1.0, 0.0);
        }

        let qc = Py::new(
            py,
            Self {
                num_qubits,
                name,
                qubits: Vec::with_capacity(num_qubits),
                state_vector,
                scratch: vec![Complex64::new(0.0, 0.0); dim],
            },
        )?;

        let mut qubit_handles = Vec::with_capacity(num_qubits);
        for (i, q_name) in names.into_iter().enumerate() {
            let q = Py::new(py, Qubit::new(qc.clone_ref(py), i, q_name))?;
            qubit_handles.push(q);
        }

        qc.bind(py).borrow_mut().qubits = qubit_handles;
        Ok(qc)
    }

    #[getter]
    fn dimension(&self) -> usize {
        1 << self.num_qubits
    }

    fn bit_position(&self, qubit_index: usize) -> usize {
        self.num_qubits - 1 - qubit_index
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let listed: Vec<String> = self
            .qubits
            .iter()
            .map(|q| q.bind(py).borrow().__repr__())
            .collect();
        format!(
            "System: {} ({} Qubits)\n  +- {}",
            self.name,
            self.num_qubits,
            listed.join("\n  +- ")
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if key.is_instance_of::<Qubit>() {
            return Ok(key.clone().unbind());
        }
        if let Ok(raw_idx) = key.extract::<isize>() {
            let len = self.num_qubits as isize;
            let idx = if raw_idx < 0 { raw_idx + len } else { raw_idx };
            if idx < 0 || idx >= len {
                return Err(PyIndexError::new_err(format!(
                    "Qubit index {raw_idx} is out of range for {} qubits.",
                    self.num_qubits
                )));
            }
            return Ok(self.qubits[idx as usize].clone_ref(py).into_any());
        }
        if let Ok(name) = key.cast::<PyString>() {
            let target = name.to_str()?;
            for q in &self.qubits {
                if q.bind(py).borrow().name == target {
                    return Ok(q.clone_ref(py).into_any());
                }
            }
            return Err(PyKeyError::new_err(format!(
                "No qubit found with name '{target}'."
            )));
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(self.num_qubits as isize)?;
            let mut result = Vec::with_capacity(indices.slicelength);
            let mut curr = indices.start;
            for _ in 0..indices.slicelength {
                result.push(self.qubits[curr as usize].clone_ref(py));
                curr += indices.step;
            }
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }
        Err(PyTypeError::new_err(
            "Key must be a Qubit, integer index, string name, or slice.",
        ))
    }
}
