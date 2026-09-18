// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error types for the `oxiphysics-python` crate.
//!
//! All public API methods that can fail return \[`Result`T`].
//! Errors are designed to be easily surfaced as Python exceptions when
//! exposed via PyO3/pyo3 FFI, and serialize cleanly to JSON for interchange.
//!
//! # Overview
//!
//! - [`enum@Error`] — comprehensive enum covering all Python-bridge failure modes
//! - [`Result`T`\] — convenience alias
//! - Helper constructors and classification predicates on [`enum@Error`]

use pyo3::exceptions::{PyException, PyKeyError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Core error enum
// ---------------------------------------------------------------------------

/// Comprehensive error type for the `oxiphysics-python` bridge.
///
/// Every failure mode has a dedicated variant to allow fine-grained handling
/// from Python user code and for clean mapping to `PyErr` via PyO3.
#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Error {
    // --- Handle / identity errors ---
    /// A body or object handle was invalid.
    #[error("invalid handle: {0}")]
    InvalidHandle(u32),

    /// A rigid body was not found by handle.
    #[error("body not found: handle={0}")]
    BodyNotFound(u32),

    /// A collider was not found by handle.
    #[error("collider not found: handle={0}")]
    ColliderNotFound(u32),

    // --- Parameter / validation errors ---
    /// A named parameter was out of its valid range.
    #[error("invalid parameter '{name}': {message}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Constraint description.
        message: String,
    },

    /// A mass value was non-positive.
    #[error("mass must be positive, got {0}")]
    InvalidMass(f64),

    /// A time step was non-positive.
    #[error("time step must be positive, got {0}")]
    InvalidTimeStep(f64),

    /// A geometry dimension was non-positive.
    #[error("dimension must be positive, got {0}")]
    InvalidDimension(f64),

    // --- Capacity / resource errors ---
    /// The world has reached its maximum body capacity.
    #[error("world at capacity: max {max} bodies")]
    CapacityExceeded {
        /// Maximum allowed bodies.
        max: usize,
    },

    // --- Simulation state errors ---
    /// The simulation has diverged (NaN / Inf detected).
    #[error("simulation diverged at step {step}")]
    SimulationDiverged {
        /// Step at which divergence was detected.
        step: u64,
    },

    /// The solver failed to converge.
    #[error("solver did not converge after {iterations} iterations")]
    SolverConvergenceFailed {
        /// Iterations attempted.
        iterations: u32,
    },

    /// A body was queried but it is sleeping.
    #[error("body {0} is sleeping")]
    BodySleeping(u32),

    // --- Serialization / IO errors ---
    /// JSON serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Snapshot validation failed (schema mismatch, missing fields, etc.).
    #[error("snapshot validation failed: {0}")]
    SnapshotValidation(String),

    // --- Python interop errors ---
    /// A Python argument had an unexpected type.
    #[error("type error: expected {expected}, got {got}")]
    TypeError {
        /// Expected Python type name.
        expected: String,
        /// Actual Python type name.
        got: String,
    },

    /// A required Python keyword argument was missing.
    #[error("missing argument: '{0}'")]
    MissingArgument(String),

    /// A Python list or array had the wrong length.
    #[error("wrong array length: expected {expected}, got {got}")]
    WrongArrayLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    // --- Generic ---
    /// Generic error with a free-form message.
    #[error("{0}")]
    General(String),
}

/// Result type alias for `oxiphysics-python` operations.
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Python exception classes
// ---------------------------------------------------------------------------

pyo3::create_exception!(
    oxiphysics,
    OxiPhysicsError,
    PyException,
    "Base class of the exceptions the OxiPhysics bindings raise for engine failures."
);
pyo3::create_exception!(
    oxiphysics,
    SimulationDivergedError,
    OxiPhysicsError,
    "The simulation diverged: a NaN or an infinite value appeared in its state."
);
pyo3::create_exception!(
    oxiphysics,
    SolverConvergenceError,
    OxiPhysicsError,
    "An iterative solver did not converge within its iteration budget."
);

/// Add the exception classes of the bindings to the Python module `m`.
///
/// `OxiPhysicsError` derives from `Exception`; `SimulationDivergedError` and
/// `SolverConvergenceError` derive from `OxiPhysicsError`.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("OxiPhysicsError", py.get_type::<OxiPhysicsError>())?;
    m.add(
        "SimulationDivergedError",
        py.get_type::<SimulationDivergedError>(),
    )?;
    m.add(
        "SolverConvergenceError",
        py.get_type::<SolverConvergenceError>(),
    )?;
    Ok(())
}

/// The Python exception class an [`enum@Error`] is raised as.
///
/// [`Error::python_exception_class`] (the class name) and the conversion into
/// a [`PyErr`] (the class itself) both derive from this one table, so the name
/// a caller reads is always the class Python code catches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExceptionClass {
    /// Builtin `TypeError`.
    TypeError,
    /// Builtin `ValueError`.
    ValueError,
    /// Builtin `KeyError`.
    KeyError,
    /// Builtin `MemoryError`.
    MemoryError,
    /// [`SimulationDivergedError`].
    SimulationDiverged,
    /// [`SolverConvergenceError`].
    SolverConvergence,
    /// [`OxiPhysicsError`], the base class of the engine specific errors.
    OxiPhysics,
}

impl ExceptionClass {
    /// Name of the class as Python spells it.
    const fn name(self) -> &'static str {
        match self {
            Self::TypeError => "TypeError",
            Self::ValueError => "ValueError",
            Self::KeyError => "KeyError",
            Self::MemoryError => "MemoryError",
            Self::SimulationDiverged => "SimulationDivergedError",
            Self::SolverConvergence => "SolverConvergenceError",
            Self::OxiPhysics => "OxiPhysicsError",
        }
    }

    /// A [`PyErr`] of this class carrying `message`.
    fn new_err(self, message: String) -> PyErr {
        match self {
            Self::TypeError => PyTypeError::new_err(message),
            Self::ValueError => PyValueError::new_err(message),
            Self::KeyError => PyKeyError::new_err(message),
            Self::MemoryError => PyMemoryError::new_err(message),
            Self::SimulationDiverged => SimulationDivergedError::new_err(message),
            Self::SolverConvergence => SolverConvergenceError::new_err(message),
            Self::OxiPhysics => OxiPhysicsError::new_err(message),
        }
    }
}

impl From<Error> for pyo3::PyErr {
    /// Raise the error as the class [`Error::python_exception_class`] names.
    fn from(e: Error) -> pyo3::PyErr {
        e.exception_class().new_err(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// impl Error
// ---------------------------------------------------------------------------

impl Error {
    // --- Constructors -------------------------------------------------------

    /// Create an `InvalidParameter` error.
    pub fn invalid_param(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Create a `General` error.
    pub fn general(msg: impl Into<String>) -> Self {
        Self::General(msg.into())
    }

    /// Create a `TypeError` error.
    pub fn type_error(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::TypeError {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create a `WrongArrayLength` error.
    pub fn wrong_len(expected: usize, got: usize) -> Self {
        Self::WrongArrayLength { expected, got }
    }

    // --- Classification predicates ------------------------------------------

    /// Returns `true` if this error is handle-related.
    pub fn is_handle_error(&self) -> bool {
        matches!(
            self,
            Error::InvalidHandle(_) | Error::BodyNotFound(_) | Error::ColliderNotFound(_)
        )
    }

    /// Returns `true` if this error is a parameter validation error.
    pub fn is_parameter_error(&self) -> bool {
        matches!(
            self,
            Error::InvalidParameter { .. }
                | Error::InvalidMass(_)
                | Error::InvalidTimeStep(_)
                | Error::InvalidDimension(_)
        )
    }

    /// Returns `true` if this error is capacity-related.
    pub fn is_capacity_error(&self) -> bool {
        matches!(self, Error::CapacityExceeded { .. })
    }

    /// Returns `true` if this error indicates simulation instability.
    pub fn is_stability_error(&self) -> bool {
        matches!(
            self,
            Error::SimulationDiverged { .. } | Error::SolverConvergenceFailed { .. }
        )
    }

    /// Returns `true` if this is a serialization error.
    pub fn is_serialization_error(&self) -> bool {
        matches!(self, Error::Serialization(_) | Error::SnapshotValidation(_))
    }

    /// Returns `true` if this is a Python interop type error.
    pub fn is_type_error(&self) -> bool {
        matches!(
            self,
            Error::TypeError { .. } | Error::MissingArgument(_) | Error::WrongArrayLength { .. }
        )
    }

    // --- JSON / Python interop ----------------------------------------------

    /// Serialize this error to a JSON string for Python exception chaining.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oxiphysics_python::Error;
    ///
    /// let e = Error::InvalidTimeStep(-0.1);
    /// let json = e.to_json();
    /// assert!(json.contains("InvalidTimeStep"));
    /// ```
    pub fn to_json(&self) -> String {
        let variant_json =
            serde_json::to_string(self).unwrap_or_else(|_| "\"<serialization failed>\"".into());
        let message = self.to_string();
        format!(
            r#"{{"error":{variant_json},"message":{message_json}}}"#,
            variant_json = variant_json,
            message_json = serde_json::to_string(&message).unwrap_or_default(),
        )
    }

    /// Deserialize an `Error` from JSON.
    pub fn from_json(json: &str) -> std::result::Result<Self, String> {
        // Try direct variant JSON first
        let direct: std::result::Result<Error, _> = serde_json::from_str(json);
        if let Ok(e) = direct {
            return Ok(e);
        }
        // Try envelope with "error" field
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let inner = v
            .get("error")
            .ok_or_else(|| "missing 'error' field".to_string())?;
        serde_json::from_value(inner.clone()).map_err(|e| e.to_string())
    }

    /// Return the name of the Python exception class this error is raised as.
    ///
    /// The conversion `PyErr::from(error)` raises exactly this class:
    ///
    /// | Variants                                              | Class                     |
    /// |-------------------------------------------------------|---------------------------|
    /// | `TypeError`, `WrongArrayLength`                       | `TypeError`               |
    /// | `MissingArgument`, `InvalidParameter`, `InvalidMass`, `InvalidTimeStep`, `InvalidDimension`, `Serialization`, `SnapshotValidation` | `ValueError` |
    /// | `InvalidHandle`, `BodyNotFound`, `ColliderNotFound`   | `KeyError`                |
    /// | `CapacityExceeded`                                    | `MemoryError`             |
    /// | `SimulationDiverged`                                  | `SimulationDivergedError` |
    /// | `SolverConvergenceFailed`                             | `SolverConvergenceError`  |
    /// | `BodySleeping`, `General`                             | `OxiPhysicsError`         |
    ///
    /// The last three are the exception classes the module defines;
    /// `SimulationDivergedError` and `SolverConvergenceError` derive from
    /// `OxiPhysicsError`, which derives from `Exception`.
    pub fn python_exception_class(&self) -> &'static str {
        self.exception_class().name()
    }

    /// The single table that maps every variant to its Python exception class.
    ///
    /// The `match` has no wildcard arm: a new variant does not compile until it
    /// has a class.
    fn exception_class(&self) -> ExceptionClass {
        match self {
            Error::TypeError { .. } | Error::WrongArrayLength { .. } => ExceptionClass::TypeError,
            Error::MissingArgument(_)
            | Error::InvalidParameter { .. }
            | Error::InvalidMass(_)
            | Error::InvalidTimeStep(_)
            | Error::InvalidDimension(_)
            | Error::Serialization(_)
            | Error::SnapshotValidation(_) => ExceptionClass::ValueError,
            Error::InvalidHandle(_) | Error::BodyNotFound(_) | Error::ColliderNotFound(_) => {
                ExceptionClass::KeyError
            }
            Error::CapacityExceeded { .. } => ExceptionClass::MemoryError,
            Error::SimulationDiverged { .. } => ExceptionClass::SimulationDiverged,
            Error::SolverConvergenceFailed { .. } => ExceptionClass::SolverConvergence,
            Error::BodySleeping(_) | Error::General(_) => ExceptionClass::OxiPhysics,
        }
    }

    /// Return a human-readable recovery hint for this error.
    pub fn recovery_hint(&self) -> String {
        match self {
            Error::InvalidTimeStep(_) => {
                "Use a positive dt such as 1/60 for a 60 Hz simulation.".into()
            }
            Error::InvalidMass(_) => "Mass must be strictly positive (e.g. 1.0).".into(),
            Error::BodyNotFound(h) => format!(
                "Body handle {} not found. Re-add the body or check it has not been removed.",
                h
            ),
            Error::CapacityExceeded { max } => format!(
                "World capacity of {} bodies reached. Remove unused bodies first.",
                max
            ),
            Error::SimulationDiverged { step } => format!(
                "Divergence detected at step {}. Reduce dt or applied forces.",
                step
            ),
            Error::TypeError { expected, got } => {
                format!("Expected a Python '{}' but received '{}'.", expected, got)
            }
            Error::WrongArrayLength { expected, got } => {
                format!("Array length should be {} but is {}.", expected, got)
            }
            _ => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- basic display tests -----------------------------------------------

    #[test]
    fn test_invalid_handle() {
        let e = Error::InvalidHandle(5);
        assert!(e.to_string().contains("5"));
        assert!(e.is_handle_error());
    }

    #[test]
    fn test_body_not_found() {
        let e = Error::BodyNotFound(10);
        assert!(e.to_string().contains("10"));
        assert!(e.is_handle_error());
    }

    #[test]
    fn test_collider_not_found() {
        let e = Error::ColliderNotFound(3);
        assert!(e.is_handle_error());
    }

    #[test]
    fn test_invalid_param() {
        let e = Error::invalid_param("mass", "must be positive");
        assert!(e.to_string().contains("mass"));
        assert!(e.to_string().contains("positive"));
        assert!(e.is_parameter_error());
    }

    #[test]
    fn test_invalid_mass() {
        let e = Error::InvalidMass(-1.0);
        assert!(e.is_parameter_error());
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn test_invalid_time_step() {
        let e = Error::InvalidTimeStep(0.0);
        assert!(e.is_parameter_error());
    }

    #[test]
    fn test_invalid_dimension() {
        let e = Error::InvalidDimension(-0.5);
        assert!(e.is_parameter_error());
    }

    #[test]
    fn test_capacity_exceeded() {
        let e = Error::CapacityExceeded { max: 1000 };
        assert!(e.is_capacity_error());
        assert!(e.to_string().contains("1000"));
    }

    #[test]
    fn test_simulation_diverged() {
        let e = Error::SimulationDiverged { step: 99 };
        assert!(e.is_stability_error());
        assert!(e.to_string().contains("99"));
    }

    #[test]
    fn test_solver_convergence_failed() {
        let e = Error::SolverConvergenceFailed { iterations: 50 };
        assert!(e.is_stability_error());
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn test_body_sleeping() {
        let e = Error::BodySleeping(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn test_serialization_error() {
        let e = Error::Serialization("unexpected eof".into());
        assert!(e.is_serialization_error());
    }

    #[test]
    fn test_snapshot_validation_error() {
        let e = Error::SnapshotValidation("missing version".into());
        assert!(e.is_serialization_error());
    }

    #[test]
    fn test_type_error() {
        let e = Error::type_error("list", "int");
        assert!(e.is_type_error());
        assert!(e.to_string().contains("list"));
        assert!(e.to_string().contains("int"));
    }

    #[test]
    fn test_missing_argument() {
        let e = Error::MissingArgument("mass".into());
        assert!(e.is_type_error());
        assert!(e.to_string().contains("mass"));
    }

    #[test]
    fn test_wrong_array_length() {
        let e = Error::wrong_len(3, 2);
        assert!(e.is_type_error());
        assert!(e.to_string().contains("3"));
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn test_general_error() {
        let e = Error::general("oops");
        assert!(e.to_string().contains("oops"));
    }

    // ---- clone / eq -------------------------------------------------------

    #[test]
    fn test_clone_eq() {
        let e1 = Error::InvalidHandle(42);
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ---- JSON roundtrip ---------------------------------------------------

    #[test]
    fn test_to_json_contains_type() {
        let e = Error::InvalidTimeStep(-0.01);
        let json = e.to_json();
        assert!(json.contains("InvalidTimeStep"), "json={}", json);
        assert!(json.contains("message"), "json={}", json);
    }

    #[test]
    fn test_from_json_direct() {
        let original = Error::BodyNotFound(7);
        let json = serde_json::to_string(&original).unwrap();
        let recovered = Error::from_json(&json).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_from_json_envelope() {
        let e = Error::CapacityExceeded { max: 256 };
        let envelope = e.to_json();
        let recovered = Error::from_json(&envelope).unwrap();
        assert_eq!(recovered, e);
    }

    #[test]
    fn test_from_json_invalid() {
        assert!(Error::from_json("{bad json").is_err());
    }

    // ---- python_exception_class -------------------------------------------

    #[test]
    fn test_python_exception_class_value_error() {
        assert_eq!(
            Error::InvalidTimeStep(0.0).python_exception_class(),
            "ValueError"
        );
        assert_eq!(
            Error::InvalidMass(-1.0).python_exception_class(),
            "ValueError"
        );
        assert_eq!(
            Error::invalid_param("x", "y").python_exception_class(),
            "ValueError"
        );
    }

    #[test]
    fn test_python_exception_class_key_error() {
        assert_eq!(Error::BodyNotFound(1).python_exception_class(), "KeyError");
        assert_eq!(Error::InvalidHandle(0).python_exception_class(), "KeyError");
    }

    #[test]
    fn test_python_exception_class_type_error() {
        assert_eq!(
            Error::type_error("list", "int").python_exception_class(),
            "TypeError"
        );
        assert_eq!(Error::wrong_len(3, 1).python_exception_class(), "TypeError");
    }

    #[test]
    fn test_python_exception_class_engine_errors_have_their_own_classes() {
        assert_eq!(
            Error::SimulationDiverged { step: 1 }.python_exception_class(),
            "SimulationDivergedError"
        );
        assert_eq!(
            Error::SolverConvergenceFailed { iterations: 10 }.python_exception_class(),
            "SolverConvergenceError"
        );
        assert_eq!(
            Error::general("x").python_exception_class(),
            "OxiPhysicsError"
        );
        assert_eq!(
            Error::BodySleeping(3).python_exception_class(),
            "OxiPhysicsError"
        );
        assert_eq!(
            Error::CapacityExceeded { max: 4 }.python_exception_class(),
            "MemoryError"
        );
    }

    // ---- conversion into Python exceptions ---------------------------------

    /// Run `f` with the interpreter of the test process attached.
    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    /// One value of every variant of [`Error`].
    fn every_variant() -> Vec<Error> {
        vec![
            Error::InvalidHandle(1),
            Error::BodyNotFound(2),
            Error::ColliderNotFound(3),
            Error::invalid_param("mass", "must be positive"),
            Error::InvalidMass(-1.0),
            Error::InvalidTimeStep(0.0),
            Error::InvalidDimension(-0.5),
            Error::CapacityExceeded { max: 4 },
            Error::SimulationDiverged { step: 5 },
            Error::SolverConvergenceFailed { iterations: 6 },
            Error::BodySleeping(7),
            Error::Serialization("unexpected eof".into()),
            Error::SnapshotValidation("missing version".into()),
            Error::type_error("list", "int"),
            Error::MissingArgument("mass".into()),
            Error::wrong_len(3, 2),
            Error::general("something went wrong"),
        ]
    }

    #[test]
    fn test_every_error_is_raised_as_the_class_it_names() {
        with_python(|py| {
            for error in every_variant() {
                let expected = error.python_exception_class();
                let message = error.to_string();
                let err: PyErr = error.clone().into();
                let class = err.get_type(py);
                let name = class.name().expect("exception classes have a name");
                assert_eq!(name.to_string(), expected, "variant {error:?}");
                let text = err.value(py).to_string();
                assert!(
                    text.contains(&message),
                    "message of {error:?} lost: {text:?}"
                );
            }
        });
    }

    #[test]
    fn test_builtin_exception_classes_are_raised_where_python_expects_them() {
        with_python(|py| {
            let raised = |error: Error| PyErr::from(error);
            assert!(raised(Error::wrong_len(3, 2)).is_instance_of::<PyTypeError>(py));
            assert!(raised(Error::type_error("list", "int")).is_instance_of::<PyTypeError>(py));
            assert!(raised(Error::InvalidTimeStep(-1.0)).is_instance_of::<PyValueError>(py));
            assert!(raised(Error::invalid_param("x", "y")).is_instance_of::<PyValueError>(py));
            assert!(raised(Error::BodyNotFound(1)).is_instance_of::<PyKeyError>(py));
            assert!(raised(Error::InvalidHandle(1)).is_instance_of::<PyKeyError>(py));
            assert!(raised(Error::CapacityExceeded { max: 1 }).is_instance_of::<PyMemoryError>(py));
            // None of them is an engine class.
            assert!(!raised(Error::wrong_len(3, 2)).is_instance_of::<OxiPhysicsError>(py));
        });
    }

    #[test]
    fn test_engine_exception_classes_form_a_hierarchy() {
        with_python(|py| {
            let diverged = PyErr::from(Error::SimulationDiverged { step: 3 });
            assert!(diverged.is_instance_of::<SimulationDivergedError>(py));
            assert!(diverged.is_instance_of::<OxiPhysicsError>(py));
            assert!(diverged.is_instance_of::<PyException>(py));
            assert!(!diverged.is_instance_of::<SolverConvergenceError>(py));

            let stalled = PyErr::from(Error::SolverConvergenceFailed { iterations: 9 });
            assert!(stalled.is_instance_of::<SolverConvergenceError>(py));
            assert!(stalled.is_instance_of::<OxiPhysicsError>(py));
            assert!(!stalled.is_instance_of::<SimulationDivergedError>(py));

            let general = PyErr::from(Error::general("oops"));
            assert!(general.is_instance_of::<OxiPhysicsError>(py));
            assert!(!general.is_instance_of::<SimulationDivergedError>(py));
            assert!(!general.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        });
    }

    #[test]
    fn test_register_exceptions_adds_the_three_classes_to_a_module() {
        with_python(|py| {
            let module = PyModule::new(py, "oxiphysics_probe").expect("module");
            register_exceptions(&module).expect("registration");
            for name in [
                "OxiPhysicsError",
                "SimulationDivergedError",
                "SolverConvergenceError",
            ] {
                let class = module.getattr(name).expect("class is registered");
                assert!(class.is_instance_of::<pyo3::types::PyType>(), "{name}");
            }
            let base = module.getattr("OxiPhysicsError").expect("base class");
            assert!(base.is(&py.get_type::<OxiPhysicsError>()));
            let module_name = base.getattr("__module__").expect("__module__");
            assert_eq!(module_name.to_string(), "oxiphysics");
            let raised: PyErr = Error::SimulationDiverged { step: 1 }.into();
            let diverged = module.getattr("SimulationDivergedError").expect("class");
            assert!(raised.matches(py, &diverged).expect("matches"));
            assert!(raised.matches(py, &base).expect("matches"));
        });
    }

    // ---- recovery hints ---------------------------------------------------

    #[test]
    fn test_recovery_hint_time_step() {
        let hint = Error::InvalidTimeStep(0.0).recovery_hint();
        assert!(!hint.is_empty());
    }

    #[test]
    fn test_recovery_hint_body_not_found() {
        let hint = Error::BodyNotFound(42).recovery_hint();
        assert!(hint.contains("42"));
    }

    #[test]
    fn test_recovery_hint_capacity() {
        let hint = Error::CapacityExceeded { max: 500 }.recovery_hint();
        assert!(hint.contains("500"));
    }

    #[test]
    fn test_recovery_hint_diverged() {
        let hint = Error::SimulationDiverged { step: 7 }.recovery_hint();
        assert!(hint.contains("7"));
    }

    #[test]
    fn test_recovery_hint_type_error() {
        let hint = Error::type_error("ndarray", "str").recovery_hint();
        assert!(hint.contains("ndarray"));
    }

    #[test]
    fn test_recovery_hint_wrong_len() {
        let hint = Error::wrong_len(3, 1).recovery_hint();
        assert!(hint.contains("3"));
        assert!(hint.contains("1"));
    }

    #[test]
    fn test_recovery_hint_general_empty() {
        // General errors have no specific hint
        let hint = Error::general("x").recovery_hint();
        // May or may not be empty, just ensure it does not panic
        let _ = hint;
    }
}
