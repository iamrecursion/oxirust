//! Tensor Cores Advanced Optimization Modules
//!
//! This module contains advanced optimization components for tensor cores,
//! including quantum-inspired optimization, AI-driven strategies, and
//! performance analytics.

pub mod quantum_optimization;

/// Re-export quantum optimization components
pub use quantum_optimization::{
    ConvergenceMetrics, EntanglementPattern, EntanglementType, OptimizationStep,
    QuantumInspiredOptimizer, QuantumStateApproximation,
};
