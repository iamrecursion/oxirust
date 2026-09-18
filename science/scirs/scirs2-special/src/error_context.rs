//! Enhanced error handling with context tracking for special functions
//!
//! This module provides consistent error handling patterns across all special functions
//! with detailed context tracking for better debugging and error recovery.

use crate::error::{SpecialError, SpecialResult};
use std::fmt;

/// Context information for error tracking
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The function where the error occurred
    pub function_name: String,
    /// The specific operation that failed
    pub operation: String,
    /// Input parameters that caused the error
    pub parameters: Vec<(String, String)>,
    /// Additional context information
    pub additional_info: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(function_name: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            operation: operation.into(),
            parameters: Vec::new(),
            additional_info: None,
        }
    }

    /// Add a parameter to the context
    pub fn with_param(mut self, name: impl Into<String>, value: impl fmt::Display) -> Self {
        self.parameters.push((name.into(), value.to_string()));
        self
    }

    /// Add additional information
    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.additional_info = Some(info.into());
        self
    }

    /// Convert to a formatted error message
    pub fn to_error_message(&self) -> String {
        let mut msg = format!("Error in {} during {}", self.function_name, self.operation);

        if !self.parameters.is_empty() {
            msg.push_str(" with parameters: ");
            let params: Vec<String> = self
                .parameters
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            msg.push_str(&params.join(", "));
        }

        if let Some(ref info) = self.additional_info {
            msg.push_str(&format!(". {info}"));
        }

        msg
    }
}

/// Enhanced error type with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: SpecialError,
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context.to_error_message(), self.error)
    }
}

impl std::error::Error for ContextualError {}

/// Extension trait for adding context to errors
pub trait ErrorContextExt<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> SpecialResult<T>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T> ErrorContextExt<T> for SpecialResult<T> {
    fn with_context<F>(self, f: F) -> SpecialResult<T>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| {
            let ctx = f();
            SpecialError::ComputationError(format!("{}: {e}", ctx.to_error_message()))
        })
    }
}

/// Standard validation trait for special functions
pub trait ValidatedFunction<Input, Output> {
    /// Validate inputs before computation
    fn validateinputs(&self, input: &Input) -> SpecialResult<()>;

    /// Compute the function with validated inputs
    fn compute_validated(&self, input: Input) -> SpecialResult<Output>;

    /// Main entry point that combines validation and computation
    fn evaluate(&self, input: Input) -> SpecialResult<Output> {
        self.validateinputs(&input)?;
        self.compute_validated(input)
    }
}

/// Error recovery strategies
#[derive(Debug, Clone, Copy)]
pub enum RecoveryStrategy {
    /// Return a default value
    ReturnDefault,
    /// Clamp to valid range
    ClampToRange,
    /// Use approximation
    UseApproximation,
    /// Propagate error
    PropagateError,
}

/// Error recovery trait
pub trait ErrorRecovery<T> {
    /// Attempt to recover from an error
    fn recover(&self, error: &SpecialError, strategy: RecoveryStrategy) -> Option<T>;
}

/// Macro for consistent error creation with context
#[macro_export]
macro_rules! special_error {
    (domain: $func:expr, $op:expr, $($param:expr => $value:expr),* $(,)?) => {{
        let mut ctx = $crate::error_context::ErrorContext::new($func, $op);
        $(ctx = ctx.with_param($param, $value);)*
        $crate::error::SpecialError::DomainError(ctx.to_error_message())
    }};

    (convergence: $func:expr, $op:expr, $($param:expr => $value:expr),* $(,)?) => {{
        let mut ctx = $crate::error_context::ErrorContext::new($func, $op);
        $(ctx = ctx.with_param($param, $value);)*
        $crate::error::SpecialError::ConvergenceError(ctx.to_error_message())
    }};

    (computation: $func:expr, $op:expr, $($param:expr => $value:expr),* $(,)?) => {{
        let mut ctx = $crate::error_context::ErrorContext::new($func, $op);
        $(ctx = ctx.with_param($param, $value);)*
        $crate::error::SpecialError::ComputationError(ctx.to_error_message())
    }};
}

/// Macro for consistent validation with error context
#[macro_export]
macro_rules! validate_with_context {
    ($condition:expr, $error_type:ident, $func:expr, $msg:expr $(, $param:expr => $value:expr)*) => {
        if !($condition) {
            let mut ctx = $crate::error_context::ErrorContext::new($func, "validation");
            $(ctx = ctx.with_param($param, $value);)*
            ctx = ctx.with_info($msg);
            return Err($crate::error::SpecialError::$error_type(ctx.to_error_message()));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("gamma", "computation")
            .with_param("x", -1.0)
            .with_info("Gamma function is undefined at negative integers");

        let msg = ctx.to_error_message();
        assert!(msg.contains("gamma"));
        assert!(msg.contains("x=-1"));
        assert!(msg.contains("undefined"));
    }

    #[test]
    fn test_error_context_macro() {
        let err = special_error!(
            domain: "bessel_j", "evaluation",
            "n" => 5,
            "x" => -10.0
        );

        match err {
            SpecialError::DomainError(msg) => {
                assert!(msg.contains("bessel_j"));
                assert!(msg.contains("n=5"));
                assert!(msg.contains("x=-10"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}
