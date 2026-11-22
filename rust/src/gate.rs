//! The `Gate` record, mirroring `quantum.gate_abs.Gate`.
//!
//! A gate is data, not an operation: `QuantumComputer.gate(...)` builds one,
//! `apply(...)` runs it, and every applied gate lands in `op_hist`. Keeping
//! the same attributes as the Python class also lets `quantum.qasm` and the
//! composite/circuit gate helpers duck-type Rust gates without a bridge.

use crate::{as_complex_array, numpy_module, GateView};
use num_complex::Complex64;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyDict;

/// Gate matrices up to this side are validated directly in Rust. Bigger ones
/// still use NumPy/BLAS; for 2x2 and 4x4 gates the Python calls cost far more
/// than the actual unitarity check.
const SMALL_UNITARY_SIDE: usize = 16;

/// Return the largest entrywise deviation from I, and whether `allclose`
/// (atol=1e-9, rtol=1e-5, as in `quantum.gate_abs.Gate`) would accept it.
fn unitary_deviation(matrix: &[Complex64], side: usize) -> (f64, bool) {
    let mut largest: f64 = 0.0;
    let mut accepted = true;
    for row in 0..side {
        for column in 0..side {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..side {
                // conj(T)[row, k] * matrix[k, column]
                sum += matrix[k * side + row].conj() * matrix[k * side + column];
            }
            let expected = if row == column { 1.0 } else { 0.0 };
            let deviation = (sum - Complex64::new(expected, 0.0)).norm();
            largest = largest.max(deviation);
            if deviation > 1e-9 + 1e-5 * expected.abs() {
                accepted = false;
            }
        }
    }
    (largest, accepted)
}

#[pyclass(name = "Gate")]
pub struct Gate {
    matrix: Py<PyArrayDyn<Complex64>>,
    controls: Vec<usize>,
    targets: Vec<usize>,
    name: Option<String>,
    params: Vec<f64>,
    inverted: bool,
}

#[pymethods]
impl Gate {
    #[new]
    #[pyo3(signature = (matrix, controls, targets, name = None, params = None, inverted = false))]
    pub(crate) fn new(
        py: Python<'_>,
        matrix: &Bound<'_, PyAny>,
        controls: Vec<usize>,
        targets: Vec<usize>,
        name: Option<String>,
        params: Option<Vec<f64>>,
        inverted: bool,
    ) -> PyResult<Self> {
        let matrix = as_complex_array(py, matrix)?;
        let shape = matrix.shape();
        let side = shape.first().copied().unwrap_or(0);

        let unitary = if shape.len() == 2 && shape[1] == side && side <= SMALL_UNITARY_SIDE {
            // as_complex_array() just produced a C-contiguous complex128 array.
            let flat = unsafe { matrix.as_slice() }
                .map_err(|_| PyValueError::new_err("Gate matrix must be contiguous."))?;
            let (deviation, accepted) = unitary_deviation(flat, side);
            if !accepted {
                return Err(PyValueError::new_err(format!(
                    "Gate matrix is not unitary (max deviation = {deviation})"
                )));
            }
            true
        } else {
            // Same unitarity check (and same wording) as quantum.gate_abs.Gate.
            let numpy = numpy_module(py)?.bind(py);
            let conjugate_t = matrix.call_method0("conj")?.getattr("T")?;
            let product = conjugate_t.call_method1("__matmul__", (&matrix,))?;
            let identity = numpy.call_method1("eye", (side,))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("atol", 1e-9)?;
            numpy
                .call_method("allclose", (&product, identity), Some(&kwargs))?
                .extract::<bool>()?
        };
        if !unitary {
            return Err(PyValueError::new_err("Gate matrix is not unitary"));
        }

        Ok(Self {
            matrix: matrix.unbind(),
            controls,
            targets,
            name,
            params: params.unwrap_or_default(),
            inverted,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.matrix)?;
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let shape = self.matrix.bind(py).shape();
        let label = self.name.clone().unwrap_or_else(|| {
            format!(
                "{}x{} matrix",
                shape.first().copied().unwrap_or(0),
                shape.get(1).copied().unwrap_or(0)
            )
        });
        let direction = if self.inverted { "inv @ " } else { "" };
        let qubits: Vec<String> = self
            .controls
            .iter()
            .chain(self.targets.iter())
            .map(ToString::to_string)
            .collect();
        format!("<{direction}{label} on [{}]>", qubits.join(", "))
    }

    #[getter]
    fn matrix(&self, py: Python<'_>) -> Py<PyArrayDyn<Complex64>> {
        self.matrix.clone_ref(py)
    }

    #[setter]
    fn set_matrix(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Python assigns this attribute directly, without re-checking unitarity.
        self.matrix = as_complex_array(py, value)?.unbind();
        Ok(())
    }

    #[getter]
    fn controls(&self) -> Vec<usize> {
        self.controls.clone()
    }

    #[setter]
    fn set_controls(&mut self, controls: Vec<usize>) {
        self.controls = controls;
    }

    #[getter]
    fn targets(&self) -> Vec<usize> {
        self.targets.clone()
    }

    #[setter]
    fn set_targets(&mut self, targets: Vec<usize>) {
        self.targets = targets;
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    #[setter]
    fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    #[getter]
    fn params(&self) -> Vec<f64> {
        self.params.clone()
    }

    #[setter]
    fn set_params(&mut self, params: Vec<f64>) {
        self.params = params;
    }

    #[getter]
    fn inverted(&self) -> bool {
        self.inverted
    }

    #[setter]
    fn set_inverted(&mut self, inverted: bool) {
        self.inverted = inverted;
    }

    /// The same operation run backwards (the adjoint of a unitary).
    fn inverse(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        let conjugate_t = self.matrix.bind(py).call_method0("conj")?.getattr("T")?;
        Py::new(
            py,
            Gate::new(
                py,
                &conjugate_t,
                self.controls.clone(),
                self.targets.clone(),
                self.name.clone(),
                Some(self.params.clone()),
                !self.inverted,
            )?,
        )
    }

    /// The same gate with its qubits rewritten through `mapping`; extra
    /// controls are appended (how an outer control conditions a program).
    #[pyo3(signature = (mapping, extra_controls = None))]
    fn remap(
        &self,
        py: Python<'_>,
        mapping: Vec<usize>,
        extra_controls: Option<Vec<usize>>,
    ) -> PyResult<Py<Self>> {
        let remap = |index: usize, label: &str| -> PyResult<usize> {
            mapping.get(index).copied().ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "{label} index {index} is out of range for a mapping of {} qubit(s).",
                    mapping.len()
                ))
            })
        };

        let mut controls =
            Vec::with_capacity(self.controls.len() + extra_controls.as_ref().map_or(0, Vec::len));
        for &index in &self.controls {
            controls.push(remap(index, "Control")?);
        }
        if let Some(extra) = extra_controls {
            controls.extend(extra);
        }

        let mut targets = Vec::with_capacity(self.targets.len());
        for &index in &self.targets {
            targets.push(remap(index, "Target")?);
        }

        Py::new(
            py,
            Gate::new(
                py,
                self.matrix.bind(py).as_any(),
                controls,
                targets,
                self.name.clone(),
                Some(self.params.clone()),
                self.inverted,
            )?,
        )
    }
}

impl Gate {
    /// Borrow the already-converted matrix without going back through NumPy.
    pub(crate) fn view_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArrayDyn<Complex64>>, Vec<usize>, Vec<usize>) {
        (
            self.matrix.bind(py).clone(),
            self.controls.clone(),
            self.targets.clone(),
        )
    }

    pub(crate) fn view<'py>(&self, py: Python<'py>) -> GateView<'py> {
        let (matrix, controls, targets) = self.view_data(py);
        GateView {
            obj: matrix.as_any().clone(),
            matrix,
            controls,
            targets,
        }
    }
}
