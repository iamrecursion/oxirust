// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Linear state-space control systems of the constraint API.
//!
//! [`PyControlSystem`] is the linear time-invariant system
//!
//! ```text
//! dx/dt = A x + B u
//!     y = C x + D u
//! ```
//!
//! with `n` states, `m` inputs and `p` outputs. The matrices are stored flat in
//! row-major order and their sizes are tied together (`A` is `n x n`, `B` is
//! `n x m`, `C` is `p x n`, `D` is `p x m`, the state holds `n` values). The
//! Python class only exposes the matrices read-only; they change together
//! through [`PyControlSystem::set_matrices`], which checks the sizes, so a
//! Python caller cannot build a system whose methods would index out of bounds.
//! The methods check the sizes again before they compute, because the Rust
//! fields are public.

use nalgebra::Complex;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Largest number of response samples reserved up front by `step_response`.
const PREALLOCATION_LIMIT: usize = 1 << 16;

/// A pivot below this multiple of the largest matrix entry makes the matrix
/// `j w I - A` count as singular in `frequency_gain`.
const SINGULAR_TOLERANCE: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Input rejected by [`PyControlSystem`].
///
/// An index outside the inputs or outputs surfaces in Python as an
/// `IndexError`, every other error as a `ValueError`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ControlSystemError {
    /// `a_matrix` does not hold a square number of values.
    #[error("`a_matrix` holds {got} values, which is not n*n for a system of order n")]
    NotSquare {
        /// Number of values given.
        got: usize,
    },
    /// A matrix does not hold `rows * cols` values.
    #[error("`{name}` must be a {rows}x{cols} matrix of {} row-major values, got {got}", .rows * .cols)]
    MatrixSize {
        /// Name of the matrix.
        name: &'static str,
        /// Number of rows it must have.
        rows: usize,
        /// Number of columns it must have.
        cols: usize,
        /// Number of values given.
        got: usize,
    },
    /// The state vector does not have one entry per state.
    #[error("the state holds {got} values, but the system has order {expected}")]
    StateSize {
        /// The order of the system.
        expected: usize,
        /// Number of values given.
        got: usize,
    },
    /// The input vector does not have one entry per input.
    #[error("the input holds {got} values, but the system has {expected} input(s)")]
    InputSize {
        /// Number of inputs of the system.
        expected: usize,
        /// Number of values given.
        got: usize,
    },
    /// The system has no output to observe.
    #[error("the system has no outputs")]
    NoOutputs,
    /// An input or output index is not smaller than the number of inputs or outputs.
    #[error("{what} index {index} is out of range for a system with {count}")]
    IndexOutOfRange {
        /// `"input"` or `"output"`.
        what: &'static str,
        /// The offending index.
        index: usize,
        /// How many of them the system has.
        count: usize,
    },
    /// A scalar argument is NaN or infinite.
    #[error("`{name}` must be finite, got {value}")]
    NotFinite {
        /// Name of the argument.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// An entry of a matrix or of the state is NaN or infinite.
    #[error("`{name}[{index}]` is {value}, but the entries must be finite")]
    NotFiniteEntry {
        /// Name of the matrix or vector.
        name: &'static str,
        /// Index of the offending entry.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The time constant of a first-order lag is zero, NaN or infinite.
    #[error("the time constant must be finite and non-zero, got {0}")]
    InvalidTimeConstant(f64),
    /// The frequency response is only defined here for one input and one output.
    #[error(
        "the frequency gain needs a single-input single-output system, got {inputs} input(s) and {outputs} output(s)"
    )]
    NotSiso {
        /// Number of inputs of the system.
        inputs: usize,
        /// Number of outputs of the system.
        outputs: usize,
    },
    /// `j w I - A` is singular at the requested frequency.
    #[error(
        "j*w*I - A is singular at {frequency} Hz: the response is unbounded there (a pole on the imaginary axis)"
    )]
    Singular {
        /// The frequency in Hz.
        frequency: f64,
    },
}

impl From<ControlSystemError> for PyErr {
    fn from(err: ControlSystemError) -> PyErr {
        let message = err.to_string();
        match err {
            ControlSystemError::IndexOutOfRange { .. } => {
                pyo3::exceptions::PyIndexError::new_err(message)
            }
            _ => pyo3::exceptions::PyValueError::new_err(message),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `Ok` if `got == rows * cols`.
fn check_matrix_size(
    name: &'static str,
    rows: usize,
    cols: usize,
    got: usize,
) -> Result<(), ControlSystemError> {
    if rows.checked_mul(cols) == Some(got) {
        Ok(())
    } else {
        Err(ControlSystemError::MatrixSize {
            name,
            rows,
            cols,
            got,
        })
    }
}

/// `Ok` if `value` is finite.
fn check_finite(name: &'static str, value: f64) -> Result<(), ControlSystemError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ControlSystemError::NotFinite { name, value })
    }
}

/// `Ok` if every entry is finite.
fn check_finite_entries(name: &'static str, values: &[f64]) -> Result<(), ControlSystemError> {
    match values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        Some((index, &value)) => Err(ControlSystemError::NotFiniteEntry { name, index, value }),
        None => Ok(()),
    }
}

/// Row `i` of the row-major matrix with `cols` columns (empty if it is absent).
fn matrix_row(matrix: &[f64], cols: usize, i: usize) -> &[f64] {
    i.checked_mul(cols)
        .and_then(|start| matrix.get(start..start.checked_add(cols)?))
        .unwrap_or(&[])
}

/// Dot product of the common prefix of `a` and `b`.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// PyControlSystem
// ---------------------------------------------------------------------------

/// A simple state-space system: dx/dt = A x + B u, y = C x + D u.
///
/// The attributes are read-only; the matrices change together through
/// `set_matrices` and the state through `set_state`, which check the sizes
/// (`ValueError`).
#[pyclass(get_all, from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyControlSystem {
    /// System order (n).
    pub order: usize,
    /// State matrix A (n×n, row-major flat).
    pub a_matrix: Vec<f64>,
    /// Input matrix B (n×m, row-major flat).
    pub b_matrix: Vec<f64>,
    /// Output matrix C (p×n, row-major flat).
    pub c_matrix: Vec<f64>,
    /// Feedthrough matrix D (p×m, row-major flat).
    pub d_matrix: Vec<f64>,
    /// Number of inputs (m).
    pub num_inputs: usize,
    /// Number of outputs (p).
    pub num_outputs: usize,
    /// Current state vector (n values).
    pub state: Vec<f64>,
}

#[pymethods]
impl PyControlSystem {
    /// Create a system from its matrices (row-major flat lists).
    ///
    /// The order `n` follows from `a_matrix`, which must hold `n*n` values;
    /// `b_matrix` must hold `n*num_inputs`, `c_matrix` `num_outputs*n` and
    /// `d_matrix` `num_outputs*num_inputs` values. The initial `state` defaults
    /// to zeros.
    ///
    /// Raises `ValueError` for a wrong size or a NaN or infinite value.
    #[new]
    #[pyo3(signature = (a_matrix, b_matrix, c_matrix, d_matrix, num_inputs, num_outputs, state = None))]
    pub fn new(
        a_matrix: Vec<f64>,
        b_matrix: Vec<f64>,
        c_matrix: Vec<f64>,
        d_matrix: Vec<f64>,
        num_inputs: usize,
        num_outputs: usize,
        state: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self::try_new(
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            num_inputs,
            num_outputs,
            state,
        )?)
    }

    /// Create a first-order lag system: dx/dt = -x/tau + u/tau.
    ///
    /// Raises `ValueError` if `tau` is zero, NaN or infinite.
    #[staticmethod]
    pub fn first_order_lag(tau: f64) -> PyResult<Self> {
        Ok(Self::try_first_order_lag(tau)?)
    }

    /// Replace the matrices and the input and output counts together.
    ///
    /// The order of the system does not change: `a_matrix` must hold `n*n`
    /// values for the current order `n`, and the other matrices must fit
    /// `num_inputs` and `num_outputs` (see the constructor). The state is kept.
    /// On an error the system is left unchanged.
    ///
    /// Raises `ValueError` for a wrong size or a NaN or infinite value.
    pub fn set_matrices(
        &mut self,
        a_matrix: Vec<f64>,
        b_matrix: Vec<f64>,
        c_matrix: Vec<f64>,
        d_matrix: Vec<f64>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> PyResult<()> {
        Ok(self.try_set_matrices(
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            num_inputs,
            num_outputs,
        )?)
    }

    /// Set the state vector, which must hold `order` finite values
    /// (`ValueError` otherwise).
    pub fn set_state(&mut self, state: Vec<f64>) -> PyResult<()> {
        Ok(self.try_set_state(state)?)
    }

    /// Step the system using Euler integration and return the output `y`.
    ///
    /// The state advances by `dt * (A x + B u)` and the output is
    /// `C x + D u` for the new state.
    ///
    /// Raises `ValueError` if `u` does not hold one value per input or `dt` is
    /// not finite.
    pub fn step_euler(&mut self, u: Vec<f64>, dt: f64) -> PyResult<Vec<f64>> {
        Ok(self.try_step_euler(&u, dt)?)
    }

    /// Compute the step response over `n_steps` time steps.
    ///
    /// The state is reset to zero, input number `input` is held at 1 (the other
    /// inputs at 0) and the output number `output` is recorded after each
    /// Euler step; both default to the first.
    ///
    /// Raises `ValueError` if the system has no outputs or `dt` is not finite,
    /// and `IndexError` if `input` or `output` does not exist.
    #[pyo3(signature = (dt, n_steps, input = 0, output = 0))]
    pub fn step_response(
        &mut self,
        dt: f64,
        n_steps: usize,
        input: usize,
        output: usize,
    ) -> PyResult<Vec<f64>> {
        Ok(self.try_step_response(dt, n_steps, input, output)?)
    }

    /// Evaluate the frequency response gain `|G(j 2 pi f)|` at frequency `f` (Hz),
    /// with `G(s) = C (s I - A)^-1 B + D`.
    ///
    /// Raises `ValueError` if the system is not single-input single-output, `f`
    /// is not finite, the sizes of the matrices are inconsistent, or `j w I - A`
    /// is singular (a pole on the imaginary axis at that frequency, for example
    /// an integrator at `f = 0`).
    pub fn frequency_gain(&self, f: f64) -> PyResult<f64> {
        Ok(self.try_frequency_gain(f)?)
    }
}

impl PyControlSystem {
    /// Rust-level core of the constructor with a typed error.
    pub fn try_new(
        a_matrix: Vec<f64>,
        b_matrix: Vec<f64>,
        c_matrix: Vec<f64>,
        d_matrix: Vec<f64>,
        num_inputs: usize,
        num_outputs: usize,
        state: Option<Vec<f64>>,
    ) -> Result<Self, ControlSystemError> {
        let order = a_matrix.len().isqrt();
        if order * order != a_matrix.len() {
            return Err(ControlSystemError::NotSquare {
                got: a_matrix.len(),
            });
        }
        let system = Self {
            order,
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            num_inputs,
            num_outputs,
            state: state.unwrap_or_else(|| vec![0.0; order]),
        };
        system.check_dimensions()?;
        system.check_entries()?;
        Ok(system)
    }

    /// Rust-level core of [`Self::first_order_lag`] with a typed error.
    pub fn try_first_order_lag(tau: f64) -> Result<Self, ControlSystemError> {
        if !tau.is_finite() || tau == 0.0 {
            return Err(ControlSystemError::InvalidTimeConstant(tau));
        }
        Self::try_new(
            vec![-1.0 / tau],
            vec![1.0 / tau],
            vec![1.0],
            vec![0.0],
            1,
            1,
            None,
        )
    }

    /// Rust-level core of [`Self::set_matrices`] with a typed error.
    pub fn try_set_matrices(
        &mut self,
        a_matrix: Vec<f64>,
        b_matrix: Vec<f64>,
        c_matrix: Vec<f64>,
        d_matrix: Vec<f64>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<(), ControlSystemError> {
        let candidate = Self {
            order: self.order,
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            num_inputs,
            num_outputs,
            state: self.state.clone(),
        };
        candidate.check_dimensions()?;
        candidate.check_entries()?;
        *self = candidate;
        Ok(())
    }

    /// Rust-level core of [`Self::set_state`] with a typed error.
    pub fn try_set_state(&mut self, state: Vec<f64>) -> Result<(), ControlSystemError> {
        if state.len() != self.order {
            return Err(ControlSystemError::StateSize {
                expected: self.order,
                got: state.len(),
            });
        }
        check_finite_entries("state", &state)?;
        self.state = state;
        Ok(())
    }

    /// Check that the sizes of the matrices and of the state fit the dimensions.
    fn check_dimensions(&self) -> Result<(), ControlSystemError> {
        let (n, m, p) = (self.order, self.num_inputs, self.num_outputs);
        check_matrix_size("a_matrix", n, n, self.a_matrix.len())?;
        check_matrix_size("b_matrix", n, m, self.b_matrix.len())?;
        check_matrix_size("c_matrix", p, n, self.c_matrix.len())?;
        check_matrix_size("d_matrix", p, m, self.d_matrix.len())?;
        if self.state.len() != n {
            return Err(ControlSystemError::StateSize {
                expected: n,
                got: self.state.len(),
            });
        }
        Ok(())
    }

    /// Check that no matrix entry and no state value is NaN or infinite.
    fn check_entries(&self) -> Result<(), ControlSystemError> {
        check_finite_entries("a_matrix", &self.a_matrix)?;
        check_finite_entries("b_matrix", &self.b_matrix)?;
        check_finite_entries("c_matrix", &self.c_matrix)?;
        check_finite_entries("d_matrix", &self.d_matrix)?;
        check_finite_entries("state", &self.state)
    }

    /// The output `y = C x + D u` for the current state and the input `u`.
    fn output(&self, u: &[f64]) -> Vec<f64> {
        (0..self.num_outputs)
            .map(|i| {
                dot(matrix_row(&self.c_matrix, self.order, i), &self.state)
                    + dot(matrix_row(&self.d_matrix, self.num_inputs, i), u)
            })
            .collect()
    }

    /// Rust-level core of [`Self::step_euler`] with a typed error.
    pub fn try_step_euler(&mut self, u: &[f64], dt: f64) -> Result<Vec<f64>, ControlSystemError> {
        self.check_dimensions()?;
        check_finite("dt", dt)?;
        if u.len() != self.num_inputs {
            return Err(ControlSystemError::InputSize {
                expected: self.num_inputs,
                got: u.len(),
            });
        }
        // dx = A x + B u
        let dx: Vec<f64> = (0..self.order)
            .map(|i| {
                dot(matrix_row(&self.a_matrix, self.order, i), &self.state)
                    + dot(matrix_row(&self.b_matrix, self.num_inputs, i), u)
            })
            .collect();
        for (s, dx_i) in self.state.iter_mut().zip(&dx) {
            *s += dx_i * dt;
        }
        Ok(self.output(u))
    }

    /// Rust-level core of [`Self::step_response`] with a typed error.
    pub fn try_step_response(
        &mut self,
        dt: f64,
        n_steps: usize,
        input: usize,
        output: usize,
    ) -> Result<Vec<f64>, ControlSystemError> {
        self.check_dimensions()?;
        if self.num_outputs == 0 {
            return Err(ControlSystemError::NoOutputs);
        }
        let mut u = vec![0.0; self.num_inputs];
        match u.get_mut(input) {
            Some(held) => *held = 1.0,
            None => {
                return Err(ControlSystemError::IndexOutOfRange {
                    what: "input",
                    index: input,
                    count: self.num_inputs,
                });
            }
        }
        if output >= self.num_outputs {
            return Err(ControlSystemError::IndexOutOfRange {
                what: "output",
                index: output,
                count: self.num_outputs,
            });
        }
        self.state = vec![0.0; self.order];
        let mut response = Vec::with_capacity(n_steps.min(PREALLOCATION_LIMIT));
        for _ in 0..n_steps {
            let y = self.try_step_euler(&u, dt)?;
            response.push(y.get(output).copied().unwrap_or(0.0));
        }
        Ok(response)
    }

    /// Rust-level core of [`Self::frequency_gain`] with a typed error.
    ///
    /// Solves `(j w I - A) x = B` by Gaussian elimination with partial pivoting
    /// in complex arithmetic and returns `|C x + D|`.
    pub fn try_frequency_gain(&self, f: f64) -> Result<f64, ControlSystemError> {
        self.check_dimensions()?;
        if self.num_inputs != 1 || self.num_outputs != 1 {
            return Err(ControlSystemError::NotSiso {
                inputs: self.num_inputs,
                outputs: self.num_outputs,
            });
        }
        check_finite("f", f)?;
        let omega = 2.0 * std::f64::consts::PI * f;
        let n = self.order;
        // m = j w I - A, x = B (one column).
        let mut m: Vec<Complex<f64>> = self
            .a_matrix
            .iter()
            .enumerate()
            .map(|(k, &a)| Complex::new(-a, if k / n == k % n { omega } else { 0.0 }))
            .collect();
        let mut x: Vec<Complex<f64>> = self
            .b_matrix
            .iter()
            .map(|&b| Complex::new(b, 0.0))
            .collect();
        let scale = m.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let singular = ControlSystemError::Singular { frequency: f };
        for col in 0..n {
            let (pivot_row, pivot_norm) = (col..n)
                .map(|r| (r, m[r * n + col].norm()))
                .max_by(|p, q| p.1.total_cmp(&q.1))
                .ok_or_else(|| singular.clone())?;
            if pivot_norm <= SINGULAR_TOLERANCE * scale {
                return Err(singular);
            }
            for c in 0..n {
                m.swap(col * n + c, pivot_row * n + c);
            }
            x.swap(col, pivot_row);
            let pivot = m[col * n + col];
            for r in col + 1..n {
                let factor = m[r * n + col] / pivot;
                for c in col..n {
                    let upper = m[col * n + c];
                    m[r * n + c] -= factor * upper;
                }
                let upper = x[col];
                x[r] -= factor * upper;
            }
        }
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in i + 1..n {
                sum -= m[i * n + j] * x[j];
            }
            x[i] = sum / m[i * n + i];
        }
        let feedthrough = self.d_matrix.first().copied().unwrap_or(0.0);
        let gain = self
            .c_matrix
            .iter()
            .zip(&x)
            .fold(Complex::new(feedthrough, 0.0), |acc, (&c, &xj)| {
                acc + xj * c
            });
        Ok(gain.norm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyAttributeError, PyIndexError, PyValueError};

    /// Run `f` with the interpreter of the test process attached.
    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    /// `G(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2)` in controllable canonical form.
    fn second_order(wn: f64, zeta: f64) -> PyControlSystem {
        PyControlSystem::try_new(
            vec![0.0, 1.0, -wn * wn, -2.0 * zeta * wn],
            vec![0.0, wn * wn],
            vec![1.0, 0.0],
            vec![0.0],
            1,
            1,
            None,
        )
        .expect("valid second-order system")
    }

    fn lag(tau: f64) -> PyControlSystem {
        PyControlSystem::try_first_order_lag(tau).expect("valid time constant")
    }

    /// Two decoupled lags `x1' = -x1 + u1`, `x2' = -2 x2 + u2`, both observed.
    fn two_lags() -> PyControlSystem {
        PyControlSystem::try_new(
            vec![-1.0, 0.0, 0.0, -2.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0; 4],
            2,
            2,
            None,
        )
        .expect("valid two-input two-output system")
    }

    // ---- construction ------------------------------------------------------

    #[test]
    fn test_first_order_lag_matrices() {
        let sys = lag(0.25);
        assert_eq!(sys.order, 1);
        assert_eq!(sys.a_matrix, [-4.0]);
        assert_eq!(sys.b_matrix, [4.0]);
        assert_eq!(sys.c_matrix, [1.0]);
        assert_eq!(sys.d_matrix, [0.0]);
        assert_eq!((sys.num_inputs, sys.num_outputs), (1, 1));
        assert_eq!(sys.state, [0.0]);
    }

    #[test]
    fn test_first_order_lag_rejects_an_unusable_time_constant() {
        for tau in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = PyControlSystem::try_first_order_lag(tau).expect_err("bad tau");
            assert!(
                matches!(err, ControlSystemError::InvalidTimeConstant(_)),
                "{err}"
            );
        }
        // A negative time constant is an unstable system, which is allowed.
        assert_eq!(lag(-0.5).a_matrix, [2.0]);
    }

    #[test]
    fn test_try_new_derives_the_order_and_defaults_the_state() {
        let sys = second_order(1.0, 0.5);
        assert_eq!(sys.order, 2);
        assert_eq!(sys.state, [0.0, 0.0]);
        let with_state = PyControlSystem::try_new(
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0],
            1,
            1,
            Some(vec![1.0, 2.0]),
        )
        .expect("valid system");
        assert_eq!(with_state.state, [1.0, 2.0]);
        // A system without states is a pure feedthrough.
        let gain = PyControlSystem::try_new(vec![], vec![], vec![], vec![3.0], 1, 1, None)
            .expect("order 0");
        assert_eq!(gain.order, 0);
    }

    #[test]
    fn test_try_new_rejects_inconsistent_dimensions() {
        let new = |a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, d: Vec<f64>, state: Option<Vec<f64>>| {
            PyControlSystem::try_new(a, b, c, d, 1, 1, state)
        };
        let err = new(vec![0.0; 3], vec![0.0], vec![0.0], vec![0.0], None);
        assert_eq!(
            err.expect_err("3 is not a square"),
            ControlSystemError::NotSquare { got: 3 }
        );
        let err = new(vec![0.0; 4], vec![0.0; 3], vec![0.0; 2], vec![0.0], None);
        assert_eq!(
            err.expect_err("b is n x m"),
            ControlSystemError::MatrixSize {
                name: "b_matrix",
                rows: 2,
                cols: 1,
                got: 3
            }
        );
        let err = new(vec![0.0; 4], vec![0.0; 2], vec![0.0; 3], vec![0.0], None);
        assert_eq!(
            err.expect_err("c is p x n"),
            ControlSystemError::MatrixSize {
                name: "c_matrix",
                rows: 1,
                cols: 2,
                got: 3
            }
        );
        let err = new(vec![0.0; 4], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], None);
        assert_eq!(
            err.expect_err("d is p x m"),
            ControlSystemError::MatrixSize {
                name: "d_matrix",
                rows: 1,
                cols: 1,
                got: 2
            }
        );
        let err = new(
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0],
            Some(vec![0.0]),
        );
        assert_eq!(
            err.expect_err("state is n"),
            ControlSystemError::StateSize {
                expected: 2,
                got: 1
            }
        );
        let err = new(vec![0.0], vec![f64::NAN], vec![0.0], vec![0.0], None);
        assert!(matches!(
            err.expect_err("NaN entry"),
            ControlSystemError::NotFiniteEntry {
                name: "b_matrix",
                index: 0,
                ..
            }
        ));
        let err = new(vec![f64::INFINITY], vec![0.0], vec![0.0], vec![0.0], None);
        assert!(matches!(
            err.expect_err("infinite entry"),
            ControlSystemError::NotFiniteEntry {
                name: "a_matrix",
                index: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_set_matrices_checks_the_sizes_and_changes_nothing_on_error() {
        let mut sys = lag(0.1);
        sys.try_set_state(vec![0.7]).expect("one state");
        let before = sys.clone();
        // A second-order matrix does not fit a first-order system.
        let err = sys
            .try_set_matrices(vec![0.0; 4], vec![0.0; 2], vec![0.0; 2], vec![0.0], 1, 1)
            .expect_err("order mismatch");
        assert!(
            matches!(
                err,
                ControlSystemError::MatrixSize {
                    name: "a_matrix",
                    ..
                }
            ),
            "{err}"
        );
        // b does not fit two inputs.
        let err = sys
            .try_set_matrices(vec![-1.0], vec![1.0], vec![1.0], vec![0.0, 0.0], 2, 1)
            .expect_err("b needs two columns");
        assert!(
            matches!(
                err,
                ControlSystemError::MatrixSize {
                    name: "b_matrix",
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(
            format!("{sys:?}"),
            format!("{before:?}"),
            "an error leaves the system as it was"
        );
        // A consistent replacement keeps the state and the order.
        sys.try_set_matrices(vec![-2.0], vec![3.0, 4.0], vec![5.0], vec![0.0, 0.0], 2, 1)
            .expect("one state, two inputs, one output");
        assert_eq!(sys.order, 1);
        assert_eq!(sys.state, [0.7]);
        assert_eq!(sys.num_inputs, 2);
        assert_eq!(sys.b_matrix, [3.0, 4.0]);
    }

    #[test]
    fn test_set_state_requires_one_finite_value_per_state() {
        let mut sys = second_order(1.0, 0.1);
        assert_eq!(
            sys.try_set_state(vec![1.0]),
            Err(ControlSystemError::StateSize {
                expected: 2,
                got: 1
            })
        );
        assert!(matches!(
            sys.try_set_state(vec![1.0, f64::NAN]),
            Err(ControlSystemError::NotFiniteEntry {
                name: "state",
                index: 1,
                ..
            })
        ));
        assert_eq!(sys.state, [0.0, 0.0], "a rejected state is not stored");
        sys.try_set_state(vec![1.0, -1.0]).expect("two states");
        assert_eq!(sys.state, [1.0, -1.0]);
    }

    // ---- time stepping -----------------------------------------------------

    #[test]
    fn test_control_system_step_response() {
        let mut sys = lag(0.1);
        let resp = sys.try_step_response(0.01, 50, 0, 0).expect("SISO system");
        assert_eq!(resp.len(), 50);
        // Forward Euler on dx = (u - x) / tau gives x_k = 1 - (1 - dt / tau)^k.
        for (k, y) in resp.iter().enumerate() {
            let expected = 1.0 - 0.9f64.powi(k as i32 + 1);
            assert!((y - expected).abs() < 1e-12, "step {k}: {y} vs {expected}");
        }
    }

    #[test]
    fn test_step_response_restarts_from_zero_state() {
        let mut sys = lag(0.1);
        sys.try_set_state(vec![5.0]).expect("one state");
        let resp = sys.try_step_response(0.01, 3, 0, 0).expect("SISO system");
        assert!((resp[0] - 0.1).abs() < 1e-12, "first sample {}", resp[0]);
    }

    #[test]
    fn test_step_response_selects_input_and_output() {
        let mut sys = two_lags();
        let first = sys
            .try_step_response(0.1, 4, 0, 0)
            .expect("input 0 to output 0");
        let cross = sys
            .try_step_response(0.1, 4, 0, 1)
            .expect("input 0 to output 1");
        let second = sys
            .try_step_response(0.1, 4, 1, 1)
            .expect("input 1 to output 1");
        assert_eq!(cross, [0.0; 4], "the lags are decoupled");
        for k in 0..4 {
            // x1_k = 1 - 0.9^k and x2_k = (1 - 0.8^k) / 2 for forward Euler, dt = 0.1.
            let n = k as i32 + 1;
            assert!((first[k] - (1.0 - 0.9f64.powi(n))).abs() < 1e-12);
            assert!((second[k] - 0.5 * (1.0 - 0.8f64.powi(n))).abs() < 1e-12);
        }
        assert_eq!(
            sys.try_step_response(0.1, 4, 2, 0),
            Err(ControlSystemError::IndexOutOfRange {
                what: "input",
                index: 2,
                count: 2
            })
        );
        assert_eq!(
            sys.try_step_response(0.1, 4, 0, 2),
            Err(ControlSystemError::IndexOutOfRange {
                what: "output",
                index: 2,
                count: 2
            })
        );
    }

    #[test]
    fn test_step_response_needs_an_output() {
        let mut sys = PyControlSystem::try_new(vec![-1.0], vec![1.0], vec![], vec![], 1, 0, None)
            .expect("a system without outputs is valid");
        assert_eq!(
            sys.try_step_response(0.1, 3, 0, 0),
            Err(ControlSystemError::NoOutputs)
        );
        assert_eq!(sys.try_step_euler(&[1.0], 0.1), Ok(vec![]));
    }

    #[test]
    fn test_step_euler_requires_one_input_value_per_input() {
        let mut sys = lag(1.0);
        assert_eq!(
            sys.try_step_euler(&[1.0, 2.0], 0.1),
            Err(ControlSystemError::InputSize {
                expected: 1,
                got: 2
            })
        );
        assert_eq!(
            sys.try_step_euler(&[], 0.1),
            Err(ControlSystemError::InputSize {
                expected: 1,
                got: 0
            })
        );
        assert_eq!(
            sys.state,
            [0.0],
            "a rejected step does not advance the state"
        );
        let y = sys.try_step_euler(&[1.0], 0.5).expect("one input");
        assert_eq!(y, [0.5]);
    }

    #[test]
    fn test_step_euler_rejects_a_non_finite_time_step() {
        let mut sys = lag(1.0);
        for dt in [f64::NAN, f64::INFINITY] {
            let err = sys.try_step_euler(&[1.0], dt).expect_err("bad dt");
            assert!(
                matches!(err, ControlSystemError::NotFinite { name: "dt", .. }),
                "{err}"
            );
            let err = sys
                .try_step_response(dt, usize::MAX, 0, 0)
                .expect_err("bad dt");
            assert!(
                matches!(err, ControlSystemError::NotFinite { name: "dt", .. }),
                "{err}"
            );
        }
        assert_eq!(sys.state, [0.0]);
    }

    #[test]
    fn test_step_euler_includes_the_feedthrough() {
        // y = C x + D u with C = 2, D = 3 and a held input: x' = u, dt = 1.
        let mut sys =
            PyControlSystem::try_new(vec![0.0], vec![1.0], vec![2.0], vec![3.0], 1, 1, None)
                .expect("integrator with feedthrough");
        let y = sys.try_step_euler(&[1.0], 1.0).expect("one input");
        assert_eq!(y, [2.0 * 1.0 + 3.0 * 1.0]);
    }

    #[test]
    fn test_a_system_resized_through_the_public_fields_reports_an_error_instead_of_panicking() {
        // The reproduction of the audit finding: first_order_lag, then order = 2.
        let mut sys = lag(1.0);
        sys.order = 2;
        let err = sys.try_step_euler(&[1.0], 0.1).expect_err("sizes disagree");
        assert!(
            matches!(
                err,
                ControlSystemError::MatrixSize {
                    name: "a_matrix",
                    ..
                }
            ),
            "{err}"
        );
        assert!(sys.try_step_response(0.1, 3, 0, 0).is_err());
        assert!(sys.try_frequency_gain(1.0).is_err());
        // Missing outputs behind the system's back are an error as well.
        let mut sys = lag(1.0);
        sys.c_matrix.clear();
        assert!(sys.try_step_response(0.1, 3, 0, 0).is_err());
        let mut sys = lag(1.0);
        sys.state.clear();
        assert!(sys.try_step_euler(&[1.0], 0.1).is_err());
    }

    // ---- frequency response ------------------------------------------------

    #[test]
    fn test_first_order_gain_matches_the_closed_form() {
        let tau = 0.1;
        let sys = lag(tau);
        for f in [0.0, 0.5, 1.0, 1.59, 10.0, 100.0, -3.0] {
            let w = 2.0 * std::f64::consts::PI * f;
            let expected = 1.0 / (1.0 + (w * tau).powi(2)).sqrt();
            let gain = sys.try_frequency_gain(f).expect("regular point");
            assert!(
                (gain - expected).abs() < 1e-12,
                "f = {f}: {gain} vs {expected}"
            );
        }
        assert!((sys.try_frequency_gain(0.0).expect("DC") - 1.0).abs() < 1e-12);
        assert!(
            sys.try_frequency_gain(100.0).expect("fast") < sys.try_frequency_gain(0.0).expect("DC")
        );
    }

    #[test]
    fn test_second_order_gain_peaks_at_the_resonance() {
        // wn = 1, zeta = 0.1 at w = wn: |G| = 1 / (2 zeta) = 5.
        let sys = second_order(1.0, 0.1);
        let f = 1.0 / (2.0 * std::f64::consts::PI);
        let gain = sys.try_frequency_gain(f).expect("resonance");
        assert!((gain - 5.0).abs() < 1e-9, "gain {gain}");
        // Away from the resonance: |G| = wn^2 / sqrt((wn^2 - w^2)^2 + (2 zeta wn w)^2).
        for w in [0.0f64, 0.5, 2.0, 7.0] {
            let expected = 1.0 / ((1.0 - w * w).powi(2) + (0.2 * w).powi(2)).sqrt();
            let gain = sys
                .try_frequency_gain(w / (2.0 * std::f64::consts::PI))
                .expect("regular point");
            assert!(
                (gain - expected).abs() < 1e-9,
                "w = {w}: {gain} vs {expected}"
            );
        }
    }

    #[test]
    fn test_gain_uses_the_output_and_feedthrough_matrices() {
        // G(s) = 2 / (tau s + 1) + 0.5 with tau = 0.1 at w = 10 (w tau = 1):
        // 2 / (1 + j) + 0.5 = 1.5 - j, magnitude sqrt(3.25).
        let sys =
            PyControlSystem::try_new(vec![-10.0], vec![10.0], vec![2.0], vec![0.5], 1, 1, None)
                .expect("lag with scaled output and feedthrough");
        let gain = sys
            .try_frequency_gain(10.0 / (2.0 * std::f64::consts::PI))
            .expect("regular point");
        assert!((gain - 3.25f64.sqrt()).abs() < 1e-12, "gain {gain}");
        // A pure feedthrough has the gain |D| at every frequency.
        let direct = PyControlSystem::try_new(vec![], vec![], vec![], vec![-3.0], 1, 1, None)
            .expect("order 0");
        assert!((direct.try_frequency_gain(12.3).expect("static gain") - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_gain_of_an_unstable_system_is_not_the_stable_formula() {
        // x' = 10 x - 10 u gives G(s) = -10 / (s - 10), |G(j10)| = 10 / sqrt(200).
        let sys = lag(-0.1);
        let gain = sys
            .try_frequency_gain(10.0 / (2.0 * std::f64::consts::PI))
            .expect("regular point");
        assert!((gain - 0.5f64.sqrt()).abs() < 1e-12, "gain {gain}");
    }

    #[test]
    fn test_gain_is_a_siso_quantity() {
        let sys = two_lags();
        assert_eq!(
            sys.try_frequency_gain(1.0),
            Err(ControlSystemError::NotSiso {
                inputs: 2,
                outputs: 2
            })
        );
        let no_output = PyControlSystem::try_new(vec![-1.0], vec![1.0], vec![], vec![], 1, 0, None)
            .expect("no outputs");
        assert!(matches!(
            no_output.try_frequency_gain(1.0),
            Err(ControlSystemError::NotSiso {
                inputs: 1,
                outputs: 0
            })
        ));
    }

    #[test]
    fn test_gain_reports_a_pole_on_the_imaginary_axis() {
        // The undamped oscillator is singular at its natural frequency.
        let oscillator = second_order(1.0, 0.0);
        let f = 1.0 / (2.0 * std::f64::consts::PI);
        assert!(matches!(
            oscillator.try_frequency_gain(f),
            Err(ControlSystemError::Singular { .. })
        ));
        // ... and regular elsewhere.
        assert!(oscillator.try_frequency_gain(2.0 * f).is_ok());
        // An integrator is singular at DC.
        let integrator =
            PyControlSystem::try_new(vec![0.0], vec![1.0], vec![1.0], vec![0.0], 1, 1, None)
                .expect("integrator");
        assert!(matches!(
            integrator.try_frequency_gain(0.0),
            Err(ControlSystemError::Singular { .. })
        ));
        let at_one_hz = integrator.try_frequency_gain(1.0).expect("regular point");
        assert!((at_one_hz - 1.0 / (2.0 * std::f64::consts::PI)).abs() < 1e-12);
    }

    #[test]
    fn test_gain_rejects_a_non_finite_frequency() {
        let sys = lag(1.0);
        for f in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                sys.try_frequency_gain(f),
                Err(ControlSystemError::NotFinite { name: "f", .. })
            ));
        }
    }

    // ---- Python behaviour --------------------------------------------------

    #[test]
    fn test_python_attributes_are_read_only() {
        with_python(|py| {
            let sys = Bound::new(py, lag(1.0)).expect("a Python object");
            assert_eq!(
                sys.getattr("order")
                    .expect("getter")
                    .extract::<usize>()
                    .expect("int"),
                1
            );
            for attribute in ["order", "a_matrix", "num_inputs", "state"] {
                let err = sys
                    .setattr(attribute, 2)
                    .expect_err("the attribute must not be writable");
                assert!(
                    err.is_instance_of::<PyAttributeError>(py),
                    "{attribute}: {err}"
                );
            }
        });
    }

    #[test]
    fn test_python_calls_raise_value_error_not_panic_exception() {
        with_python(|py| {
            let sys = Bound::new(py, lag(1.0)).expect("a Python object");
            // The reproduction of the audit finding, minus the order assignment.
            let err = sys
                .call_method1("step_euler", (vec![1.0, 2.0], 0.1))
                .expect_err("a wrong input size");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = sys
                .call_method1(
                    "set_matrices",
                    (vec![0.0; 4], vec![0.0; 2], vec![0.0; 2], vec![0.0], 1, 1),
                )
                .expect_err("a wrong matrix size");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = sys
                .call_method1("set_state", (vec![1.0, 2.0],))
                .expect_err("a wrong state size");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = sys
                .call_method1("step_response", (0.1, 3, 1, 0))
                .expect_err("a missing input");
            assert!(err.is_instance_of::<PyIndexError>(py), "{err}");
            let err = sys
                .call_method1("frequency_gain", (f64::NAN,))
                .expect_err("a NaN frequency");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            // The regular calls work and use the documented defaults.
            let response: Vec<f64> = sys
                .call_method1("step_response", (0.5, 2))
                .expect("defaults for input and output")
                .extract()
                .expect("list of floats");
            assert_eq!(response, [0.5, 0.75]);
            let gain: f64 = sys
                .call_method1("frequency_gain", (0.0,))
                .expect("DC gain")
                .extract()
                .expect("float");
            assert!((gain - 1.0).abs() < 1e-12);
        });
    }

    #[test]
    fn test_python_constructor_validates_the_matrices() {
        with_python(|py| {
            let class = py.get_type::<PyControlSystem>();
            let system = class
                .call1((vec![-1.0], vec![1.0], vec![1.0], vec![0.0], 1, 1))
                .expect("a valid system");
            assert_eq!(
                system
                    .getattr("a_matrix")
                    .expect("getter")
                    .extract::<Vec<f64>>()
                    .expect("list"),
                [-1.0]
            );
            let err = class
                .call1((vec![-1.0, 0.0, 0.0], vec![1.0], vec![1.0], vec![0.0], 1, 1))
                .expect_err("three values are not a square matrix");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = class
                .call1((vec![-1.0], vec![1.0, 1.0], vec![1.0], vec![0.0], 1, 1))
                .expect_err("b does not fit one state and one input");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = py
                .get_type::<PyControlSystem>()
                .call_method1("first_order_lag", (0.0,))
                .expect_err("a zero time constant");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
        });
    }
}
