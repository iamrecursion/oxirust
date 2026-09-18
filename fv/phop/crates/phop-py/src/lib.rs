//! # phop — Python bindings for differentiable symbolic discovery
//!
//! This crate exposes the [`phop-core`](phop_core) discovery engine to Python via
//! [PyO3](https://pyo3.rs) and [rust-numpy](https://docs.rs/numpy). It adds no engine
//! logic of its own; it merely marshals NumPy arrays into the core's
//! [`scirs2_core::ndarray`] arrays, runs discovery with the GIL released, and wraps the
//! resulting Pareto front in lightweight Python classes.
//!
//! ## Example
//!
//! ```python
//! import numpy as np
//! import phop
//!
//! x = np.array([[1.0], [2.0], [3.0]])
//! y = np.array([2.0, 4.0, 6.0])
//! disco = phop.Discoverer(population=64, max_epochs=50, seed=0)
//! result = disco.fit(x, y)
//! print(result.top_latex(5))
//! ```

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};

use phop_core::{Config, DataSet, Discoverer as CoreDiscoverer, ParetoFront, PhopError, Solution};

/// Convert a [`PhopError`] into a Python [`ValueError`].
fn to_py_err(e: PhopError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A single discovered closed-form expression and its quality metrics.
///
/// Instances are produced by [`DiscoveryResult`] accessors and are read-only.
#[pyclass(name = "Solution", skip_from_py_object)]
#[derive(Clone)]
pub struct PySolution {
    inner: Solution,
}

#[pymethods]
impl PySolution {
    /// LaTeX rendering of the expression.
    #[getter]
    fn latex(&self) -> String {
        self.inner.latex()
    }

    /// Human-readable infix rendering of the expression.
    #[getter]
    fn pretty(&self) -> String {
        self.inner.pretty()
    }

    /// Rust source code that evaluates the expression.
    #[getter]
    fn rust_code(&self) -> String {
        self.inner.rust_code()
    }

    /// NumPy/Python source code that evaluates the expression.
    #[getter]
    fn numpy_code(&self) -> String {
        self.inner.numpy_code()
    }

    /// SymPy source code that constructs the expression.
    #[getter]
    fn sympy_code(&self) -> String {
        self.inner.sympy_code()
    }

    /// Mean squared error of the expression on the training data.
    #[getter]
    fn mse(&self) -> f64 {
        self.inner.mse
    }

    /// Structural complexity (node count) of the expression.
    #[getter]
    fn complexity(&self) -> usize {
        self.inner.complexity
    }

    /// String representation showing the pretty form and metrics.
    fn __repr__(&self) -> String {
        format!(
            "Solution(pretty={:?}, mse={}, complexity={})",
            self.inner.pretty(),
            self.inner.mse,
            self.inner.complexity
        )
    }
}

/// The Pareto front returned by `Discoverer::fit`.
///
/// Wraps a [`phop_core::ParetoFront`] and exposes convenient accessors for the
/// top-`k` solutions.
#[pyclass(name = "DiscoveryResult")]
pub struct DiscoveryResult {
    front: ParetoFront,
}

#[pymethods]
impl DiscoveryResult {
    /// Return the LaTeX strings of the top-`k` Pareto-optimal solutions.
    fn top_latex(&self, k: usize) -> Vec<String> {
        self.front
            .pareto_top(k)
            .into_iter()
            .map(|s| s.latex())
            .collect()
    }

    /// Return the SymPy source strings of the top-`k` Pareto-optimal solutions.
    fn top_sympy(&self, k: usize) -> Vec<String> {
        self.front
            .pareto_top(k)
            .into_iter()
            .map(|s| s.sympy_code())
            .collect()
    }

    /// Return the top-`k` Pareto-optimal solutions as [`PySolution`] objects.
    fn top(&self, k: usize) -> Vec<PySolution> {
        self.front
            .pareto_top(k)
            .into_iter()
            .map(|s| PySolution { inner: s.clone() })
            .collect()
    }

    /// Return the single best solution, or `None` if the front is empty.
    fn best(&self) -> Option<PySolution> {
        self.front.best().map(|s| PySolution { inner: s.clone() })
    }

    /// Number of solutions in the Pareto front.
    fn __len__(&self) -> usize {
        self.front.len()
    }
}

/// Differentiable symbolic discovery driver.
///
/// Constructed with hyperparameters; call `fit` with NumPy arrays to run
/// discovery. The expensive compute releases the GIL.
#[pyclass(name = "Discoverer")]
pub struct Discoverer {
    config: Config,
}

#[pymethods]
impl Discoverer {
    /// Create a new discoverer.
    ///
    /// All arguments are optional keyword arguments with the defaults shown.
    /// The `lambda_*` penalties default to the core [`Config`] defaults.
    #[new]
    #[pyo3(signature = (
        population = 256,
        max_depth = 3,
        max_epochs = 1000,
        learning_rate = 0.05,
        seed = 0,
        top_k = 5,
        lambda_complexity = Config::default().lambda_complexity,
        lambda_sparsity = Config::default().lambda_sparsity,
        lambda_parsimony = Config::default().lambda_parsimony,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        population: usize,
        max_depth: usize,
        max_epochs: usize,
        learning_rate: f64,
        seed: u64,
        top_k: usize,
        lambda_complexity: f64,
        lambda_sparsity: f64,
        lambda_parsimony: f64,
    ) -> Self {
        let mut config = Config::default()
            .population(population)
            .max_depth(max_depth)
            .max_epochs(max_epochs)
            .learning_rate(learning_rate)
            .seed(seed)
            .top_k(top_k);
        config.lambda_complexity = lambda_complexity;
        config.lambda_sparsity = lambda_sparsity;
        config.lambda_parsimony = lambda_parsimony;
        Self { config }
    }

    /// Run discovery on `x` (2D, rows × n_vars) and `y` (1D) NumPy float64 arrays.
    ///
    /// Returns a [`DiscoveryResult`]. Raises `ValueError` on shape/numeric errors.
    fn fit(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<DiscoveryResult> {
        // Extract the raw data while we still hold the GIL (numpy borrow is tied to it).
        let x_view = x.as_array();
        let (rows, cols) = (x_view.nrows(), x_view.ncols());
        // Materialize into owned, contiguous row-major data so we can rebuild the
        // core's ndarray type without sharing the (foreign) numpy buffer.
        let x_data: Vec<f64> = x_view.iter().copied().collect();
        let y_data: Vec<f64> = y.as_array().iter().copied().collect();

        let config = self.config.clone();

        // Release the GIL around the actual compute (`detach` is pyo3 0.29's
        // renamed `allow_threads`).
        let front = py.detach(move || -> phop_core::Result<ParetoFront> {
            let x_arr = Array2::from_shape_vec((rows, cols), x_data)
                .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
            let y_arr = Array1::from(y_data);
            let ds = DataSet::from_arrays(x_arr, y_arr)?;
            CoreDiscoverer::new(config).fit(&ds)
        });

        front
            .map(|front| DiscoveryResult { front })
            .map_err(to_py_err)
    }
}

/// The `phop` Python extension module.
#[pymodule]
fn phop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Discoverer>()?;
    m.add_class::<DiscoveryResult>()?;
    m.add_class::<PySolution>()?;
    Ok(())
}
