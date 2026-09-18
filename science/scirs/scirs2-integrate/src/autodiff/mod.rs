//! Enhanced automatic differentiation for numerical integration
//!
//! This module provides advanced automatic differentiation capabilities including:
//! - Forward mode AD for efficient gradient computation
//! - Reverse mode AD for efficient Jacobian computation
//! - Sparse Jacobian optimization
//! - Sensitivity analysis tools

pub mod dual;
pub mod forward;
pub mod reverse;
pub mod sensitivity;
pub mod sparse;

// Re-export main types and functions
pub use dual::{Dual, DualVector};
pub use forward::{
    forward_gradient, forward_jacobian, ForwardAD, ForwardODEJacobian, VectorizedForwardAD,
};
pub use reverse::{
    reverse_gradient, reverse_jacobian, CheckpointStrategy, ReverseAD, Tape, TapeNode,
};
pub use sensitivity::{
    compute_sensitivities, MorrisScreening, ParameterSensitivity, SensitivityAnalysis,
    SobolSensitivity, EFAST,
};
pub use sparse::{
    colored_jacobian, compress_jacobian, detect_sparsity, detect_sparsity_adaptive, BlockPattern,
    CSCJacobian, CSRJacobian, ColGrouping, HybridJacobian, SparseJacobian, SparseJacobianUpdater,
    SparsePattern,
};
