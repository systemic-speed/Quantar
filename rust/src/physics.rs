//! Small numeric kernels used by `stl.physics_algs`.
//!
//! These are pure-array passes: no Python callbacks happen inside a kernel, so
//! the GIL can stay released while rayon walks the data. Python wrappers keep
//! identical NumPy fallbacks for installs without the extension.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;

/// Below this many values rayon's scheduling costs more than the pass itself.
const SEQUENTIAL_THRESHOLD: usize = 1 << 12;

type FloatArray = Py<PyArrayDyn<f64>>;

/// Convert an array-like to a contiguous float64 array with a clear error.
fn contiguous_f64<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let numpy = crate::numpy_module(py)?.bind(py);
    let array = numpy.call_method1("asarray", (value, "float64"))?;
    let array = numpy.call_method1("ascontiguousarray", (&array,))?;
    array
        .cast_into::<PyArrayDyn<f64>>()
        .map_err(|_| PyTypeError::new_err(format!("{label} must contain real numbers.")))
}

/// Allocate an empty C-contiguous float64 array of the given shape.
fn empty_f64<'py>(py: Python<'py>, shape: &[usize]) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let numpy = crate::numpy_module(py)?.bind(py);
    let array = numpy.call_method1("empty", (shape.to_vec(), "float64"))?;
    array
        .cast_into::<PyArrayDyn<f64>>()
        .map_err(|_| PyTypeError::new_err("Failed to allocate a float64 array."))
}

/// Number of values in an array (the product of its dimensions).
fn array_size(array: &Bound<'_, PyArrayDyn<f64>>) -> usize {
    array.shape().iter().product()
}

/// A uniform mass avoids allocating a per-entry mass vector.
enum Masses {
    Uniform(f64),
    Values(Vec<f64>),
}

/// Materialise masses broadcast to `shape`, matching `stl` semantics.
fn mass_values(
    py: Python<'_>,
    masses: Option<&Bound<'_, PyAny>>,
    shape: &[usize],
) -> PyResult<Masses> {
    let Some(value) = masses else {
        return Ok(Masses::Uniform(1.0));
    };
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(Masses::Uniform(scalar));
    }

    let array = contiguous_f64(py, value, "masses")?;
    let numpy = crate::numpy_module(py)?.bind(py);
    // Per-particle masses `(n,)` map to `(n, 1)` for a `(n, dim)` state, just
    // as `_broadcast_mass` does in Python.
    let broadcast = if !shape.is_empty() && array.shape() == &shape[..shape.len() - 1] {
        let mut reshaped = shape.to_vec();
        *reshaped.last_mut().expect("shape is not empty") = 1;
        let column = numpy.call_method1("reshape", (&array, reshaped))?;
        numpy.call_method1("broadcast_to", (&column, shape.to_vec()))?
    } else {
        numpy.call_method1("broadcast_to", (&array, shape.to_vec()))?
    };
    let contiguous = numpy.call_method1("ascontiguousarray", (&broadcast,))?;
    let casted = contiguous
        .cast_into::<PyArrayDyn<f64>>()
        .map_err(|_| PyTypeError::new_err("masses must contain real numbers."))?;
    let guard = casted.readonly();
    let values = guard
        .as_slice()
        .map_err(|_| PyValueError::new_err("masses must be C-contiguous."))?;
    Ok(Masses::Values(values.to_vec()))
}

/// `0.5 * sum(m * v**2)` over every entry of `velocities`.
#[pyfunction]
#[pyo3(signature = (velocities, masses = None))]
pub fn kinetic_energy(
    py: Python<'_>,
    velocities: &Bound<'_, PyAny>,
    masses: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let array = contiguous_f64(py, velocities, "velocities")?;
    let count = array_size(&array);
    let masses = mass_values(py, masses, array.shape())?;
    let guard = array.readonly();
    let values = guard
        .as_slice()
        .map_err(|_| PyValueError::new_err("velocities must be C-contiguous."))?;

    match masses {
        Masses::Uniform(mass) => Ok(py.detach(move || {
            let term = |velocity: &f64| 0.5 * mass * velocity * velocity;
            if count <= SEQUENTIAL_THRESHOLD {
                values.iter().map(term).sum()
            } else {
                values.par_iter().map(term).sum()
            }
        })),
        Masses::Values(masses) => Ok(py.detach(move || {
            let term = |(index, velocity): (usize, &f64)| 0.5 * masses[index] * velocity * velocity;
            if count <= SEQUENTIAL_THRESHOLD {
                values.iter().enumerate().map(term).sum()
            } else {
                values.par_iter().enumerate().map(term).sum()
            }
        })),
    }
}

/// Lorentz force `q (E + v x B)` for arrays whose last axis has length 3.
#[pyfunction]
pub fn lorentz_force(
    py: Python<'_>,
    charge: f64,
    electric: &Bound<'_, PyAny>,
    magnetic: &Bound<'_, PyAny>,
    velocity: &Bound<'_, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let e_array = contiguous_f64(py, electric, "electric")?;
    let b_array = contiguous_f64(py, magnetic, "magnetic")?;
    let v_array = contiguous_f64(py, velocity, "velocity")?;
    let shape = e_array.shape().to_vec();

    if shape.last() != Some(&3) {
        return Err(PyValueError::new_err(
            "lorentz_force expects vectors with three components.",
        ));
    }
    if b_array.shape() != shape || v_array.shape() != shape {
        return Err(PyValueError::new_err(
            "electric, magnetic and velocity must have the same shape after broadcasting.",
        ));
    }
    let count = array_size(&e_array) / 3;

    let output = empty_f64(py, &shape)?;
    {
        let e_guard = e_array.readonly();
        let b_guard = b_array.readonly();
        let v_guard = v_array.readonly();
        let electric_values = e_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("electric must be C-contiguous."))?;
        let magnetic_values = b_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("magnetic must be C-contiguous."))?;
        let velocity_values = v_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("velocity must be C-contiguous."))?;
        let mut output_guard = output.readwrite();
        let result = output_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the Lorentz force array."))?;

        py.detach(move || {
            let compute = |(vector, chunk): (usize, &mut [f64; 3])| {
                let base = vector * 3;
                let (ex, ey, ez) = (
                    electric_values[base],
                    electric_values[base + 1],
                    electric_values[base + 2],
                );
                let (bx, by, bz) = (
                    magnetic_values[base],
                    magnetic_values[base + 1],
                    magnetic_values[base + 2],
                );
                let (vx, vy, vz) = (
                    velocity_values[base],
                    velocity_values[base + 1],
                    velocity_values[base + 2],
                );
                chunk[0] = charge * (ex + vy * bz - vz * by);
                chunk[1] = charge * (ey + vz * bx - vx * bz);
                chunk[2] = charge * (ez + vx * by - vy * bx);
            };
            let chunks = result.as_chunks_mut::<3>().0;
            if count <= SEQUENTIAL_THRESHOLD {
                chunks.iter_mut().enumerate().for_each(compute);
            } else {
                chunks.par_iter_mut().enumerate().for_each(compute);
            }
        });
    }
    Ok(output.unbind())
}

/// Reduce stacked central-difference evaluations to a Jacobian.
///
/// `values` has shape `(2 * dim * n, out_dim)`, laid out as, for every axis,
/// all `n` plus evaluations followed by all `n` minus evaluations. The result
/// has shape `(n, out_dim, dim)` with `result[p, i, a] = dF_i / dx_a` at point
/// `p`.
#[pyfunction]
#[pyo3(signature = (values, n, dim, out_dim, h))]
pub fn jacobian_from_offsets(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    n: usize,
    dim: usize,
    out_dim: usize,
    h: f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    if h <= 0.0 {
        return Err(PyValueError::new_err("h must be positive."));
    }
    if n == 0 || dim == 0 || out_dim == 0 {
        return Err(PyValueError::new_err(
            "n, dim and out_dim must all be positive.",
        ));
    }
    let array = contiguous_f64(py, values, "offset values")?;
    let guard = array.readonly();
    let source = guard
        .as_slice()
        .map_err(|_| PyValueError::new_err("offset values must be C-contiguous."))?;
    let expected = 2 * dim * n * out_dim;
    if source.len() != expected {
        return Err(PyValueError::new_err(format!(
            "offset values must hold {expected} entries, got {}.",
            source.len()
        )));
    }

    let output = empty_f64(py, &[n, out_dim, dim])?;
    {
        let mut output_guard = output.readwrite();
        let target = output_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the Jacobian array."))?;
        let denominator = 2.0 * h;

        py.detach(move || {
            let fill_point = |(point, chunk): (usize, &mut [f64])| {
                for axis in 0..dim {
                    let plus_base = (axis * 2 * n + point) * out_dim;
                    let minus_base = (axis * 2 * n + n + point) * out_dim;
                    for component in 0..out_dim {
                        chunk[component * dim + axis] = (source[plus_base + component]
                            - source[minus_base + component])
                            / denominator;
                    }
                }
            };
            let per_point = dim * out_dim;
            if n <= SEQUENTIAL_THRESHOLD {
                target
                    .chunks_exact_mut(per_point)
                    .enumerate()
                    .for_each(fill_point);
            } else {
                target
                    .par_chunks_exact_mut(per_point)
                    .enumerate()
                    .for_each(fill_point);
            }
        });
    }
    Ok(output.unbind())
}

/// Advance positions by `x + v dt + 0.5 a dt²` (the first Verlet half-step).
#[pyfunction]
pub fn verlet_position(
    py: Python<'_>,
    positions: &Bound<'_, PyAny>,
    velocities: &Bound<'_, PyAny>,
    acceleration: &Bound<'_, PyAny>,
    dt: f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    if dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be positive."));
    }
    let position_array = contiguous_f64(py, positions, "positions")?;
    let shape = position_array.shape().to_vec();
    let velocity_array = contiguous_f64(py, velocities, "velocities")?;
    let acceleration_array = contiguous_f64(py, acceleration, "acceleration")?;
    for (label, array) in [
        ("velocities", &velocity_array),
        ("acceleration", &acceleration_array),
    ] {
        if array.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "{label} has shape {:?}, expected {:?}.",
                array.shape(),
                shape
            )));
        }
    }

    let output = empty_f64(py, &shape)?;
    {
        let x_guard = position_array.readonly();
        let v_guard = velocity_array.readonly();
        let a_guard = acceleration_array.readonly();
        let x = x_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("positions must be C-contiguous."))?;
        let v = v_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("velocities must be C-contiguous."))?;
        let a = a_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("acceleration must be C-contiguous."))?;
        let mut output_guard = output.readwrite();
        let result = output_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the position buffer."))?;
        let half = 0.5 * dt * dt;

        py.detach(move || {
            let advance = |(index, new_x): (usize, &mut f64)| {
                *new_x = x[index] + v[index] * dt + a[index] * half;
            };
            if result.len() <= SEQUENTIAL_THRESHOLD {
                result.iter_mut().enumerate().for_each(advance);
            } else {
                result.par_iter_mut().enumerate().for_each(advance);
            }
        });
    }
    Ok(output.unbind())
}

/// Advance velocities by `v + 0.5 (a + a_next) dt` (the second Verlet half-step).
#[pyfunction]
pub fn verlet_velocity(
    py: Python<'_>,
    velocities: &Bound<'_, PyAny>,
    acceleration: &Bound<'_, PyAny>,
    next_acceleration: &Bound<'_, PyAny>,
    dt: f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    if dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be positive."));
    }
    let velocity_array = contiguous_f64(py, velocities, "velocities")?;
    let shape = velocity_array.shape().to_vec();
    let acceleration_array = contiguous_f64(py, acceleration, "acceleration")?;
    let next_acceleration_array = contiguous_f64(py, next_acceleration, "next_acceleration")?;
    for (label, array) in [
        ("acceleration", &acceleration_array),
        ("next_acceleration", &next_acceleration_array),
    ] {
        if array.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "{label} has shape {:?}, expected {:?}.",
                array.shape(),
                shape
            )));
        }
    }

    let output = empty_f64(py, &shape)?;
    {
        let v_guard = velocity_array.readonly();
        let a_guard = acceleration_array.readonly();
        let next_a_guard = next_acceleration_array.readonly();
        let v = v_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("velocities must be C-contiguous."))?;
        let a = a_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("acceleration must be C-contiguous."))?;
        let next_a = next_a_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("next_acceleration must be C-contiguous."))?;
        let mut output_guard = output.readwrite();
        let result = output_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the velocity buffer."))?;
        let half = 0.5 * dt;

        py.detach(move || {
            let advance = |(index, new_v): (usize, &mut f64)| {
                *new_v = v[index] + (a[index] + next_a[index]) * half;
            };
            if result.len() <= SEQUENTIAL_THRESHOLD {
                result.iter_mut().enumerate().for_each(advance);
            } else {
                result.par_iter_mut().enumerate().for_each(advance);
            }
        });
    }
    Ok(output.unbind())
}

/// One velocity-Verlet update, returning the new positions and velocities.
#[pyfunction]
pub fn verlet_update(
    py: Python<'_>,
    positions: &Bound<'_, PyAny>,
    velocities: &Bound<'_, PyAny>,
    acceleration: &Bound<'_, PyAny>,
    next_acceleration: &Bound<'_, PyAny>,
    dt: f64,
) -> PyResult<(FloatArray, FloatArray)> {
    if dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be positive."));
    }
    let position_array = contiguous_f64(py, positions, "positions")?;
    let shape = position_array.shape().to_vec();
    let velocity_array = contiguous_f64(py, velocities, "velocities")?;
    let acceleration_array = contiguous_f64(py, acceleration, "acceleration")?;
    let next_acceleration_array = contiguous_f64(py, next_acceleration, "next_acceleration")?;
    for (label, array) in [
        ("velocities", &velocity_array),
        ("acceleration", &acceleration_array),
        ("next_acceleration", &next_acceleration_array),
    ] {
        if array.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "{label} has shape {:?}, expected {:?}.",
                array.shape(),
                shape
            )));
        }
    }
    let count = array_size(&position_array);

    let next_positions = empty_f64(py, &shape)?;
    let next_velocities = empty_f64(py, &shape)?;
    {
        let x_guard = position_array.readonly();
        let v_guard = velocity_array.readonly();
        let a_guard = acceleration_array.readonly();
        let next_a_guard = next_acceleration_array.readonly();
        let x = x_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("positions must be C-contiguous."))?;
        let v = v_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("velocities must be C-contiguous."))?;
        let a = a_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("acceleration must be C-contiguous."))?;
        let next_a = next_a_guard
            .as_slice()
            .map_err(|_| PyValueError::new_err("next_acceleration must be C-contiguous."))?;
        let mut x_out_guard = next_positions.readwrite();
        let mut v_out_guard = next_velocities.readwrite();
        let x_out = x_out_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the position buffer."))?;
        let v_out = v_out_guard
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("Failed to allocate the velocity buffer."))?;

        py.detach(move || {
            let update = |(index, (new_x, new_v)): (usize, (&mut f64, &mut f64))| {
                *new_x = x[index] + v[index] * dt + 0.5 * a[index] * dt * dt;
                *new_v = v[index] + 0.5 * (a[index] + next_a[index]) * dt;
            };
            if count <= SEQUENTIAL_THRESHOLD {
                x_out
                    .iter_mut()
                    .zip(v_out.iter_mut())
                    .enumerate()
                    .for_each(update);
            } else {
                x_out
                    .par_iter_mut()
                    .zip(v_out.par_iter_mut())
                    .enumerate()
                    .for_each(update);
            }
        });
    }
    Ok((next_positions.unbind(), next_velocities.unbind()))
}

/// Add the physics kernels to the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kinetic_energy, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_force, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_from_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(verlet_position, m)?)?;
    m.add_function(wrap_pyfunction!(verlet_update, m)?)?;
    m.add_function(wrap_pyfunction!(verlet_velocity, m)?)?;
    Ok(())
}
