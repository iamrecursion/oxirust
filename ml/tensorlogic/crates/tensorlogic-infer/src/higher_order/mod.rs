//! Higher-order differentiation: Jacobian and Hessian via finite differences.

pub mod hessian;
pub mod jacobian;

pub use hessian::{HessianComputer, HessianStats};
pub use jacobian::{FiniteDiffMethod, JacobianComputer, JacobianConfig};
