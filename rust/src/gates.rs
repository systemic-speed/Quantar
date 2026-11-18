use crate::QuantumComputer;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rayon::prelude::*;

impl QuantumComputer {
    pub(crate) fn exec_gate(&mut self, control_qubits: &[usize], target_qubits: &[usize], 
        matrix: PyReadonlyArray2<'_, Complex64>) -> PyResult<()> {
            let arr = matrix.as_array();
            let expected_dim = 1usize << target_qubits.len();

            if arr.shape() != [expected_dim, expected_dim] {
                return Err(PyValueError::new_err(format!(
                "Expected a {}x{} gate matrix, got {:?}.",
                expected_dim,
                expected_dim,
                arr.shape()
            )));
            }

            let mat_cow = arr.as_standard_layout();
            let mat_slice = mat_cow.as_slice().unwrap();

            let mut ctrl_mask = 0usize;
            for &b in control_qubits {
                ctrl_mask |= 1usize << b;
            }

            let mut targ_mask = 0usize;
            for &b in target_qubits {
                targ_mask |= 1usize << b;
            }

            let state = &self.state_vector;
            let scratch = &mut self.scratch;
            let k = target_qubits.len();

            scratch.par_iter_mut().enumerate().for_each(|(idx, out)| {
            // Fast skip: if any control qubit is 0, state amplitude is unchanged
            if (idx & ctrl_mask) != ctrl_mask {
                *out = state[idx];
                return;
            }

            // Extract row index from target bits
            let mut row = 0usize;
            for (i, &bit_pos) in target_qubits.iter().enumerate() {
                if ((idx >> bit_pos) & 1) != 0 {
                    row |= 1usize << (k - 1 - i);
                }
            }

            let base_idx = idx & !targ_mask;
            let row_offset = row * expected_dim;
            let mut sum = Complex64::new(0.0, 0.0);

            for col in 0..expected_dim {
                let mut src_idx = base_idx;
                for (i, &bit_pos) in target_qubits.iter().enumerate() {
                    if ((col >> (k - 1 - i)) & 1) != 0 {
                        src_idx |= 1usize << bit_pos;
                    }
                }
                sum += mat_slice[row_offset + col] * state[src_idx];
            }

            *out = sum;
        });

        std::mem::swap(&mut self.state_vector, &mut self.scratch);
        Ok(())
            }
    }
            
