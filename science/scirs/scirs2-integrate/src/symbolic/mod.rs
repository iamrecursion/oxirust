//! Symbolic integration support for enhanced numerical methods
//!
//! This module provides symbolic manipulation capabilities that enhance
//! the numerical integration methods, including:
//! - Automatic Jacobian generation using symbolic differentiation
//! - Higher-order ODE to first-order system conversion
//! - Conservation law detection and enforcement
//! - Symbolic simplification for performance optimization

pub mod conservation;
pub mod conversion;
pub mod expression;
pub mod jacobian;

// Re-export main types and functions
pub use conservation::{detect_conservation_laws, ConservationEnforcer, ConservationLaw};
pub use conversion::{higher_order_to_first_order, FirstOrderSystem, HigherOrderODE};
pub use expression::{simplify, SymbolicExpression, Variable};
pub use jacobian::{generate_jacobian, SymbolicJacobian};
