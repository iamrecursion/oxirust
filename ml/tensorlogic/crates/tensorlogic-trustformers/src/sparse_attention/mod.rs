//! Sparse attention patterns and Longformer-style numerical attention.
//!
//! This module provides two layers:
//!
//! 1. **Graph-building** (`graph`) — the original einsum-graph sparse-attention
//!    types (`SparseAttentionGraph`, `SparseAttentionGraphConfig`, `LocalAttention`,
//!    `SparsePatternType`).
//!
//! 2. **Numerical Longformer attention** (`attention`, `config`, `mask`, `error`)
//!    — a research-preview implementation of sliding-window + global-token
//!    attention (Beltagy et al., 2020) that performs actual tensor computation
//!    rather than building an einsum graph.

pub mod attention;
pub mod config;
pub mod error;
pub mod graph;
pub mod mask;

#[cfg(test)]
mod tests;

// ---- Numerical sparse attention types (Longformer-style) ----
pub use attention::SparseAttention;
pub use config::SparseAttentionConfig;
pub use error::{SparseAttentionError, SparseAttentionResult};
pub use mask::{build_mask, AttentionMask};

// ---- Graph-building types (original) ----
pub use graph::{
    LocalAttention, SparseAttentionGraph, SparseAttentionGraphConfig, SparsePatternType,
};
