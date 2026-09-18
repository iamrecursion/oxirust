//! Parallel execution strategies
//!
//! This module contains various strategies for parallel execution including
//! work-stealing, data parallelism, task parallelism, and pipeline parallelism.

pub mod work_stealing;
pub mod data_parallel;

// Re-export main types
pub use work_stealing::WorkStealingScheduler;

// Re-export data parallel functions
pub use data_parallel::{
    parallel_matvec, parallel_power_iteration, parallel_gemm, 
    parallel_conjugate_gradient, parallel_jacobi,
    vector_ops,
};