//! Shared seeding helpers.
//!
//! Python's ``quantum.seed_set`` is the source of truth so both backends and
//! the public ``quantum.set_seed()`` API agree. This module just adapts that
//! state to Rust's PCG64 generator and exposes the same functions on the
//! compiled extension for users who prefer calling the backend directly.

use pyo3::prelude::*;
use rand::RngExt;
use rand_pcg::Pcg64;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

static SEED_SET: OnceLock<Py<PyModule>> = OnceLock::new();

/// Build a PCG64 generator from a 64-bit seed, mirroring `seed_set.make_pcg64`.
pub fn make_pcg64(seed: u64) -> Pcg64 {
    let digest = Sha256::digest(seed.to_le_bytes());
    let state = u128::from_le_bytes(digest[..16].try_into().expect("SHA-256 is 32 bytes"));
    let inc = u128::from_le_bytes(digest[16..].try_into().expect("SHA-256 is 32 bytes")) | 1;
    Pcg64::new(state, inc)
}

/// Generate a seed from OS-backed entropy when the user has not set one.
pub fn random_seed() -> u64 {
    rand::rng().random::<u64>()
}

fn seed_set_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    if let Some(module) = SEED_SET.get() {
        return Ok(module.bind(py).clone());
    }
    let module = py.import("quantum.seed_set")?;
    SEED_SET.get_or_init(|| module.clone().unbind());
    Ok(module)
}

/// Read the shared Python seed. ``None`` means "generate a random one".
pub(crate) fn get_seed(py: Python<'_>) -> PyResult<Option<u64>> {
    seed_set_module(py)?
        .getattr("get_seed")?
        .call0()?
        .extract::<Option<u64>>()
}

/// Set the shared seed through Python so both backends stay in sync.
#[pyfunction]
pub fn set_seed(py: Python<'_>, seed: u64) -> PyResult<u64> {
    let value = seed_set_module(py)?
        .getattr("set_seed")?
        .call1((seed,))?
        .extract::<u64>()?;
    Ok(value)
}

/// Python-visible alias for the shared seed getter.
#[pyfunction(name = "get_seed")]
pub fn get_seed_py(py: Python<'_>) -> PyResult<Option<u64>> {
    get_seed(py)
}

/// Register the seed helpers on the compiled extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_seed, m)?)?;
    m.add_function(wrap_pyfunction!(get_seed_py, m)?)?;
    Ok(())
}
