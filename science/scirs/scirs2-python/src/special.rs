//! Python bindings for scirs2-special
//!
//! Provides special mathematical functions similar to scipy.special

use pyo3::prelude::*;
use scirs2_core::python::numpy_compat::{scirs_to_numpy_array1, Array1};
use scirs2_numpy::{PyArray1, PyReadonlyArray1};

// Import special functions from scirs2-special
use scirs2_special::gamma::polygamma;
use scirs2_special::{bessel, erf as erf_mod, gamma as gamma_fn};

// =============================================================================
// Gamma Functions
// =============================================================================

/// Gamma function
#[pyfunction]
fn gamma_py(x: f64) -> PyResult<f64> {
    Ok(gamma_fn(x))
}

/// Log-gamma function
#[pyfunction]
fn lgamma_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::gammaln(x))
}

/// Digamma (psi) function
#[pyfunction]
fn digamma_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::digamma(x))
}

/// Beta function
#[pyfunction]
fn beta_py(a: f64, b: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::beta(a, b))
}

// =============================================================================
// Bessel Functions
// =============================================================================

/// Bessel function of the first kind, J0
#[pyfunction]
fn j0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::j0(x))
}

/// Bessel function of the first kind, J1
#[pyfunction]
fn j1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::j1(x))
}

/// Bessel function of the first kind, Jn
#[pyfunction]
fn jn_py(n: i32, x: f64) -> PyResult<f64> {
    Ok(bessel::jn(n, x))
}

/// Bessel function of the second kind, Y0
#[pyfunction]
fn y0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::y0(x))
}

/// Bessel function of the second kind, Y1
#[pyfunction]
fn y1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::y1(x))
}

/// Bessel function of the second kind, Yn
#[pyfunction]
fn yn_py(n: i32, x: f64) -> PyResult<f64> {
    Ok(bessel::yn(n, x))
}

/// Modified Bessel function of the first kind, I0
#[pyfunction]
fn i0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::i0(x))
}

/// Modified Bessel function of the first kind, I1
#[pyfunction]
fn i1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::i1(x))
}

/// Modified Bessel function of the second kind, K0
#[pyfunction]
fn k0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::k0(x))
}

/// Modified Bessel function of the second kind, K1
#[pyfunction]
fn k1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::k1(x))
}

// =============================================================================
// Error Functions
// =============================================================================

/// Error function
#[pyfunction]
fn erf_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erf(x))
}

/// Complementary error function
#[pyfunction]
fn erfc_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfc(x))
}

/// Inverse error function
#[pyfunction]
fn erfinv_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfinv(x))
}

/// Inverse complementary error function
#[pyfunction]
fn erfcinv_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfcinv(x))
}

/// Scaled complementary error function
#[pyfunction]
fn erfcx_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfcx(x))
}

/// Imaginary error function
#[pyfunction]
fn erfi_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfi(x))
}

/// Dawson's integral
#[pyfunction]
fn dawsn_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::dawsn(x))
}

// =============================================================================
// Combinatorial Functions
// =============================================================================

/// Factorial function
#[pyfunction]
fn factorial_py(n: u32) -> PyResult<f64> {
    scirs2_special::factorial(n)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Binomial coefficient (n choose k)
#[pyfunction]
fn comb_py(n: u32, k: u32) -> PyResult<f64> {
    scirs2_special::comb(n, k)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Permutations (n permute k)
#[pyfunction]
fn perm_py(n: u32, k: u32) -> PyResult<f64> {
    scirs2_special::perm(n, k)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Elliptic Integrals
// =============================================================================

/// Complete elliptic integral of the first kind
#[pyfunction]
fn ellipk_py(m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_k(m))
}

/// Complete elliptic integral of the second kind
#[pyfunction]
fn ellipe_py(m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_e(m))
}

/// Incomplete elliptic integral of the first kind
#[pyfunction]
fn ellipkinc_py(phi: f64, m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_f(phi, m))
}

/// Incomplete elliptic integral of the second kind
#[pyfunction]
fn ellipeinc_py(phi: f64, m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_e_inc(phi, m))
}

// =============================================================================
// Polygamma Function
// =============================================================================

/// Polygamma function: the n-th derivative of the digamma function
#[pyfunction]
fn polygamma_py(n: u32, x: f64) -> PyResult<f64> {
    Ok(polygamma(n, x))
}

// =============================================================================
// Zeta Functions
// =============================================================================

/// Riemann zeta function
#[pyfunction]
fn zeta_py(s: f64) -> PyResult<f64> {
    scirs2_special::zeta(s).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Hurwitz zeta function
#[pyfunction]
fn hurwitz_zeta_py(s: f64, q: f64) -> PyResult<f64> {
    scirs2_special::hurwitz_zeta(s, q)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Riemann zeta minus 1: zeta(s) - 1
#[pyfunction]
fn zetac_py(s: f64) -> PyResult<f64> {
    scirs2_special::zetac(s).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Hypergeometric Functions
// =============================================================================

/// Confluent hypergeometric function 0F1
#[pyfunction]
fn hyp0f1_py(v: f64, z: f64) -> PyResult<f64> {
    scirs2_special::hyp0f1(v, z)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Confluent hypergeometric function 1F1 (Kummer's function)
#[pyfunction]
fn hyp1f1_py(a: f64, b: f64, z: f64) -> PyResult<f64> {
    scirs2_special::hyp1f1(a, b, z)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Gauss hypergeometric function 2F1
#[pyfunction]
fn hyp2f1_py(a: f64, b: f64, c: f64, z: f64) -> PyResult<f64> {
    scirs2_special::hyp2f1(a, b, c, z)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Tricomi's confluent hypergeometric function U(a, b, x)
#[pyfunction]
fn hyperu_py(a: f64, b: f64, x: f64) -> PyResult<f64> {
    scirs2_special::hyperu(a, b, x)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Airy Functions
// =============================================================================

/// Airy function Ai(x)
#[pyfunction]
fn airy_ai_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::ai(x))
}

/// Derivative of Airy function Ai'(x)
#[pyfunction]
fn airy_aip_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::aip(x))
}

/// Airy function Bi(x)
#[pyfunction]
fn airy_bi_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::bi(x))
}

/// Derivative of Airy function Bi'(x)
#[pyfunction]
fn airy_bip_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::bip(x))
}

// =============================================================================
// Trigonometric and Exponential Integrals
// =============================================================================

/// Sine integral Si(x)
#[pyfunction]
fn sici_si_py(x: f64) -> PyResult<f64> {
    scirs2_special::si(x).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Cosine integral Ci(x)
#[pyfunction]
fn sici_ci_py(x: f64) -> PyResult<f64> {
    scirs2_special::ci(x).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Hyperbolic sine integral Shi(x)
#[pyfunction]
fn shichi_shi_py(x: f64) -> PyResult<f64> {
    scirs2_special::shi(x).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Hyperbolic cosine integral Chi(x)
#[pyfunction]
fn shichi_chi_py(x: f64) -> PyResult<f64> {
    scirs2_special::chi(x).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Incomplete Beta Function
// =============================================================================

/// Regularized incomplete beta function I_x(a, b)
#[pyfunction]
fn betainc_py(a: f64, b: f64, x: f64) -> PyResult<f64> {
    scirs2_special::gamma::betainc_regularized(a, b, x)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Inverse of regularized incomplete beta function
#[pyfunction]
fn betaincinv_py(a: f64, b: f64, p: f64) -> PyResult<f64> {
    scirs2_special::gamma::betaincinv(a, b, p)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Vectorized versions (for arrays)
// =============================================================================

/// Vectorized gamma function
#[pyfunction]
fn gamma_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| gamma_fn(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized error function
#[pyfunction]
fn erf_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| erf_mod::erf(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized J0 Bessel function
#[pyfunction]
fn j0_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| bessel::j0(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized lgamma function
#[pyfunction]
fn lgamma_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input
        .iter()
        .map(|&v| scirs2_special::gamma::gammaln(v))
        .collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized erfc function
#[pyfunction]
fn erfc_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| erf_mod::erfc(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized digamma function
#[pyfunction]
fn digamma_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input
        .iter()
        .map(|&v| scirs2_special::gamma::digamma(v))
        .collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized J1 Bessel function
#[pyfunction]
fn j1_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| bessel::j1(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Gamma functions
    m.add_function(wrap_pyfunction!(gamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(lgamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(digamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(beta_py, m)?)?;

    // Bessel functions
    m.add_function(wrap_pyfunction!(j0_py, m)?)?;
    m.add_function(wrap_pyfunction!(j1_py, m)?)?;
    m.add_function(wrap_pyfunction!(jn_py, m)?)?;
    m.add_function(wrap_pyfunction!(y0_py, m)?)?;
    m.add_function(wrap_pyfunction!(y1_py, m)?)?;
    m.add_function(wrap_pyfunction!(yn_py, m)?)?;
    m.add_function(wrap_pyfunction!(i0_py, m)?)?;
    m.add_function(wrap_pyfunction!(i1_py, m)?)?;
    m.add_function(wrap_pyfunction!(k0_py, m)?)?;
    m.add_function(wrap_pyfunction!(k1_py, m)?)?;

    // Error functions
    m.add_function(wrap_pyfunction!(erf_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfc_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfinv_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfcinv_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfcx_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfi_py, m)?)?;
    m.add_function(wrap_pyfunction!(dawsn_py, m)?)?;

    // Combinatorial functions
    m.add_function(wrap_pyfunction!(factorial_py, m)?)?;
    m.add_function(wrap_pyfunction!(comb_py, m)?)?;
    m.add_function(wrap_pyfunction!(perm_py, m)?)?;

    // Elliptic integrals
    m.add_function(wrap_pyfunction!(ellipk_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipe_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipkinc_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipeinc_py, m)?)?;

    // Polygamma
    m.add_function(wrap_pyfunction!(polygamma_py, m)?)?;

    // Zeta functions
    m.add_function(wrap_pyfunction!(zeta_py, m)?)?;
    m.add_function(wrap_pyfunction!(hurwitz_zeta_py, m)?)?;
    m.add_function(wrap_pyfunction!(zetac_py, m)?)?;

    // Hypergeometric functions
    m.add_function(wrap_pyfunction!(hyp0f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(hyp1f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(hyp2f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(hyperu_py, m)?)?;

    // Airy functions
    m.add_function(wrap_pyfunction!(airy_ai_py, m)?)?;
    m.add_function(wrap_pyfunction!(airy_aip_py, m)?)?;
    m.add_function(wrap_pyfunction!(airy_bi_py, m)?)?;
    m.add_function(wrap_pyfunction!(airy_bip_py, m)?)?;

    // Trig/exponential integrals
    m.add_function(wrap_pyfunction!(sici_si_py, m)?)?;
    m.add_function(wrap_pyfunction!(sici_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(shichi_shi_py, m)?)?;
    m.add_function(wrap_pyfunction!(shichi_chi_py, m)?)?;

    // Incomplete beta
    m.add_function(wrap_pyfunction!(betainc_py, m)?)?;
    m.add_function(wrap_pyfunction!(betaincinv_py, m)?)?;

    // Vectorized versions
    m.add_function(wrap_pyfunction!(gamma_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(lgamma_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(erf_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfc_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(digamma_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(j0_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(j1_array_py, m)?)?;

    Ok(())
}
