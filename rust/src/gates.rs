use crate::{GateView, QuantumComputer};
use num_complex::Complex64;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Below this many amplitudes, rayon's scheduling costs more than the pass
/// itself. The Python backend has the same small-register fast path.
const SEQUENTIAL_THRESHOLD: usize = 1 << 12;

/// Apply a gate whose bits are already expressed as physical bit positions.
///
/// `target_bits[0]` is the most significant bit of the gate matrix (the same
/// convention as `quantum.basic_funcs.apply_local_gate`), and a basis state
/// only takes part in the gate when every control bit is set.
///
/// The GIL is released for the amplitude pass: every closure capture is a
/// plain Rust slice (or a `Vec` of integers), so nothing touches Python while
/// other threads are running. The NumPy borrow guards stay alive in this
/// frame for the whole detached section.
impl QuantumComputer {
    /// Check a gate record against the register before any amplitude is
    /// touched, mirroring `QuantumComputer._validate_gate`.
    pub(crate) fn validate_view(&self, py: Python<'_>, view: &GateView<'_>) -> PyResult<()> {
        for &index in view.controls.iter().chain(view.targets.iter()) {
            if index >= self.num_qubits {
                return Err(PyIndexError::new_err(format!(
                    "Qubit index {index} is out of range for {} qubits.",
                    self.num_qubits
                )));
            }
            let qubit = self.qubits[index].bind(py).borrow();
            qubit._assert_alive()?;
        }

        if bit_mask("Control", &view.controls, self.num_qubits)?
            & bit_mask("Target", &view.targets, self.num_qubits)?
            != 0
        {
            return Err(PyValueError::new_err(
                "Control and target qubits must be disjoint.",
            ));
        }

        let side = 1usize
            .checked_shl(view.targets.len() as u32)
            .ok_or_else(|| PyValueError::new_err("Too many target qubits for a single gate."))?;
        if view.matrix.shape() != [side, side] {
            return Err(PyValueError::new_err(format!(
                "A gate on {} target(s) needs a {side}x{side} matrix, got {:?}.",
                view.targets.len(),
                view.matrix.shape()
            )));
        }
        Ok(())
    }

    /// Validate and run one gate record (Rust `Gate` or Python duck-typed gate).
    pub(crate) fn run_gate_view(&mut self, py: Python<'_>, view: &GateView<'_>) -> PyResult<()> {
        self.validate_view(py, view)?;
        self.run_gate_view_validated(py, view)
    }

    /// Run a record whose qubit indices and liveness were already checked by
    /// `apply_program` (shape/mask checks still happen in `exec_gate`).
    pub(crate) fn run_gate_view_validated(
        &mut self,
        py: Python<'_>,
        view: &GateView<'_>,
    ) -> PyResult<()> {
        let control_bits: Vec<usize> = view
            .controls
            .iter()
            .map(|&index| self.bit_position_unchecked(index))
            .collect();
        let target_bits: Vec<usize> = view
            .targets
            .iter()
            .map(|&index| self.bit_position_unchecked(index))
            .collect();
        let matrix = view
            .matrix
            .clone()
            .cast_into::<PyArray2<Complex64>>()
            .map_err(|_| PyValueError::new_err("Gate matrix must be two-dimensional."))?;
        self.exec_gate(py, &control_bits, &target_bits, matrix.readonly())
    }

    pub(crate) fn exec_gate<'py>(
        &mut self,
        py: Python<'py>,
        control_bits: &[usize],
        target_bits: &[usize],
        matrix: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<()> {
        let num_targets = target_bits.len();
        let expected_dim = 1usize
            .checked_shl(num_targets as u32)
            .ok_or_else(|| PyValueError::new_err("Too many target qubits for a single gate."))?;

        let array = matrix.as_array();
        if array.shape() != [expected_dim, expected_dim] {
            return Err(PyValueError::new_err(format!(
                "A gate on {num_targets} target(s) needs a {expected_dim}x{expected_dim} matrix, got {:?}.",
                array.shape()
            )));
        }

        // NumPy slices are only valid for contiguous data; a Fortran-ordered
        // gate is copied once into standard layout instead of being rejected.
        let matrix_view = array.as_standard_layout();
        let matrix_slice = matrix_view
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Gate matrix must be a contiguous array."))?;

        let control_mask = bit_mask("Control", control_bits, self.num_qubits)?;
        let target_mask = bit_mask("Target", target_bits, self.num_qubits)?;
        if control_mask & target_mask != 0 {
            return Err(PyValueError::new_err(
                "Control and target qubits must be disjoint.",
            ));
        }

        // offset[column] is the physical basis offset the gate's `column`
        // value corresponds to. Precomputing it keeps the hot loop to one
        // array lookup per matrix column.
        let mut offsets = vec![0usize; expected_dim];
        for (column, offset) in offsets.iter_mut().enumerate() {
            let mut index = 0usize;
            for (gate_bit, &bit) in target_bits.iter().enumerate() {
                if (column >> (num_targets - 1 - gate_bit)) & 1 != 0 {
                    index |= 1usize << bit;
                }
            }
            *offset = index;
        }

        {
            let state_guard = self.state_vector.bind(py).try_readonly()?;
            let state = contiguous_slice(&state_guard, "state_vector")?;
            if state.len() != self.dimension() {
                return Err(PyValueError::new_err(format!(
                    "state_vector holds {} amplitudes, expected {}.",
                    state.len(),
                    self.dimension()
                )));
            }

            let mut scratch_guard = self.scratch.bind(py).try_readwrite()?;
            let scratch = contiguous_slice_mut(&mut scratch_guard, "scratch buffer")?;

            let target_bits = target_bits.to_vec();
            py.detach(move || {
                apply_gate_kernel(
                    state,
                    scratch,
                    matrix_slice,
                    control_mask,
                    target_mask,
                    &target_bits,
                    &offsets,
                    expected_dim,
                );
            });
        }

        std::mem::swap(&mut self.state_vector, &mut self.scratch);
        Ok(())
    }

    /// Apply a run of diagonal single-target gates in one pass over the state.
    /// Each entry is (control mask, target bit, diag[0], diag[1]).
    pub(crate) fn exec_diagonal_run(
        &mut self,
        py: Python<'_>,
        specs: &[(usize, usize, Complex64, Complex64)],
    ) -> PyResult<()> {
        if specs.is_empty() {
            return Ok(());
        }
        {
            let state_guard = self.state_vector.bind(py).try_readonly()?;
            let state = contiguous_slice(&state_guard, "state_vector")?;
            let mut scratch_guard = self.scratch.bind(py).try_readwrite()?;
            let scratch = contiguous_slice_mut(&mut scratch_guard, "scratch buffer")?;
            let specs = specs.to_vec();

            py.detach(move || {
                let compute = |index: usize, out: &mut Complex64| {
                    let mut value = Complex64::new(1.0, 0.0);
                    for &(control_mask, bit, diag0, diag1) in &specs {
                        if index & control_mask == control_mask {
                            value *= if (index >> bit) & 1 != 0 {
                                diag1
                            } else {
                                diag0
                            };
                        }
                    }
                    *out = state[index] * value;
                };

                if state.len() <= SEQUENTIAL_THRESHOLD {
                    for (index, out) in scratch.iter_mut().enumerate() {
                        compute(index, out);
                    }
                } else {
                    scratch
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(index, out)| compute(index, out));
                }
            });
        }
        std::mem::swap(&mut self.state_vector, &mut self.scratch);
        Ok(())
    }

    /// Apply a run of disjoint SWAPs as one permutation pass. Each pair holds
    /// the two physical bit positions to exchange.
    pub(crate) fn exec_swap_run(
        &mut self,
        py: Python<'_>,
        pairs: &[(usize, usize)],
    ) -> PyResult<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        {
            let state_guard = self.state_vector.bind(py).try_readonly()?;
            let state = contiguous_slice(&state_guard, "state_vector")?;
            let mut scratch_guard = self.scratch.bind(py).try_readwrite()?;
            let scratch = contiguous_slice_mut(&mut scratch_guard, "scratch buffer")?;
            let pairs = pairs.to_vec();

            py.detach(move || {
                let compute = |index: usize, out: &mut Complex64| {
                    let mut swapped = index;
                    for &(low, high) in &pairs {
                        let bit_low = (swapped >> low) & 1;
                        let bit_high = (swapped >> high) & 1;
                        swapped = (swapped & !(1usize << low) & !(1usize << high))
                            | (bit_low << high)
                            | (bit_high << low);
                    }
                    *out = state[swapped];
                };

                if state.len() <= SEQUENTIAL_THRESHOLD {
                    for (index, out) in scratch.iter_mut().enumerate() {
                        compute(index, out);
                    }
                } else {
                    scratch
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(index, out)| compute(index, out));
                }
            });
        }
        std::mem::swap(&mut self.state_vector, &mut self.scratch);
        Ok(())
    }
}

/// Validate physical bit positions and fold them into a single mask, rejecting
/// duplicates (which would otherwise silently collapse two qubits into one).
pub(crate) fn bit_mask(label: &str, bits: &[usize], num_qubits: usize) -> PyResult<usize> {
    let mut mask = 0usize;
    for &bit in bits {
        if bit >= num_qubits {
            return Err(PyIndexError::new_err(format!(
                "Bit position {bit} is out of range for {num_qubits} qubits."
            )));
        }
        let bit_mask = 1usize << bit;
        if mask & bit_mask != 0 {
            return Err(PyValueError::new_err(format!(
                "{label} qubits must be distinct."
            )));
        }
        mask |= bit_mask;
    }
    Ok(mask)
}

/// Get the amplitude slice of a NumPy array, with a consistent error message.
pub(crate) fn contiguous_slice<'a>(
    array: &'a PyReadonlyArray2<'_, Complex64>,
    label: &str,
) -> PyResult<&'a [Complex64]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{label} must be C-contiguous.")))
}

/// Mutable twin of [`contiguous_slice`].
pub(crate) fn contiguous_slice_mut<'a>(
    array: &'a mut PyReadwriteArray2<'_, Complex64>,
    label: &str,
) -> PyResult<&'a mut [Complex64]> {
    array
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("{label} must be C-contiguous.")))
}

/// The amplitude pass itself. Every input is plain data, so the caller can
/// safely run it from `py.detach`.
#[allow(clippy::too_many_arguments)]
fn apply_gate_kernel(
    state: &[Complex64],
    scratch: &mut [Complex64],
    matrix: &[Complex64],
    control_mask: usize,
    target_mask: usize,
    target_bits: &[usize],
    offsets: &[usize],
    expected_dim: usize,
) {
    let num_targets = target_bits.len();
    let compute = |index: usize, out: &mut Complex64| {
        // A failed control simply carries the amplitude through.
        if index & control_mask != control_mask {
            *out = state[index];
            return;
        }

        // Gather the target bits into the gate's own row index.
        let mut row = 0usize;
        for (gate_bit, &bit) in target_bits.iter().enumerate() {
            if (index >> bit) & 1 != 0 {
                row |= 1usize << (num_targets - 1 - gate_bit);
            }
        }

        let base = index & !target_mask;
        let row_offset = row * expected_dim;
        let mut sum = Complex64::new(0.0, 0.0);
        for (column, &offset) in offsets.iter().enumerate() {
            sum += matrix[row_offset + column] * state[base | offset];
        }
        *out = sum;
    };

    if state.len() <= SEQUENTIAL_THRESHOLD {
        for (index, out) in scratch.iter_mut().enumerate() {
            compute(index, out);
        }
    } else {
        scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, out)| compute(index, out));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }

    fn one() -> Complex64 {
        Complex64::new(1.0, 0.0)
    }

    #[test]
    fn bit_mask_validates_positions_and_duplicates() {
        assert_eq!(bit_mask("Qubit", &[0, 2, 3], 4).unwrap(), 0b1101);
        assert!(bit_mask("Qubit", &[4], 4).is_err());
        assert!(bit_mask("Qubit", &[1, 1], 4).is_err());
    }

    #[test]
    fn single_qubit_kernel_matches_x_and_h() {
        // Qubit 1 is the MSB of a two-qubit register; start from |10>.
        let state = [z(), z(), one(), z()];
        let mut scratch = [z(); 4];
        let x = [z(), one(), one(), z()];
        apply_gate_kernel(&state, &mut scratch, &x, 0, 0b10, &[1], &[0, 2], 2);
        assert_eq!(scratch, [one(), z(), z(), z()]);

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ];
        apply_gate_kernel(
            &one_state(4, 0),
            &mut scratch,
            &h,
            0,
            0b10,
            &[1],
            &[0, 2],
            2,
        );
        assert!((scratch[0].re - inv_sqrt2).abs() < 1e-12);
        assert!((scratch[2].re - inv_sqrt2).abs() < 1e-12);
        assert_eq!(scratch[1], z());
        assert_eq!(scratch[3], z());
    }

    #[test]
    fn controlled_kernel_leaves_failing_branch_untouched() {
        // Control bit 1, target bit 0; only |10> and |11> are transformed.
        let x = [z(), one(), one(), z()];
        let mut scratch = [z(); 4];

        // |01> has the control bit clear, so it passes through unchanged.
        apply_gate_kernel(
            &one_state(4, 1),
            &mut scratch,
            &x,
            0b10,
            1,
            &[0],
            &[0, 1],
            2,
        );
        assert_eq!(one_state(4, 1), scratch);

        // |10> has the control bit set, so X flips the target to give |11>.
        apply_gate_kernel(
            &one_state(4, 2),
            &mut scratch,
            &x,
            0b10,
            1,
            &[0],
            &[0, 1],
            2,
        );
        assert_eq!(one_state(4, 3), scratch);
    }

    fn one_state(dimension: usize, index: usize) -> Vec<Complex64> {
        let mut state = vec![z(); dimension];
        state[index] = one();
        state
    }
}
