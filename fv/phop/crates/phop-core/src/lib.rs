//! # phop-core — differentiable symbolic discovery on the EML operator
//!
//! phop discovers closed-form expressions from data by learning both the *topology*
//! and the *numeric parameters* of homogeneous binary trees built from a single
//! primitive, the EML operator
//!
//! ```text
//! eml(x, y) = exp(x) - ln(y)
//! ```
//!
//! together with the constant `1`. The grammar is the trivial production
//! `S -> 1 | eml(S, S)`, which (Odrzywołek 2026) is functionally complete for the
//! elementary functions. Homogeneity lets an entire population of candidate trees be
//! evaluated inside a single automatic-differentiation graph and optimized end-to-end.
//!
//! This crate is layered:
//! - **Layer A** ([`forest`]) — tensorized EML forest forward evaluation.
//! - Layer B (topology), C (loss/Pareto), D (distillation) build on top (see the crate TODO).
//!
//! The AST, canonicalization, and LaTeX rendering are delegated to the [`oxieml`] crate;
//! automatic differentiation and optimization to `scirs2-autograd`.

pub mod accel;
pub mod affine;
pub mod analyze;
pub mod any_solution;
pub mod codegen;
pub mod config;
pub mod dataset;
pub mod dimension;
pub mod discoverer;
pub mod distill;
#[cfg(feature = "egraph")]
pub mod egraph;
pub mod error;
pub mod fit;
pub mod forest;
pub mod gated;
#[cfg(feature = "gpu-cuda")]
pub mod gpu;
pub mod gumbel;
#[cfg(feature = "lean")]
pub mod lean;
pub mod loss;
#[cfg(feature = "gpu-metal")]
pub mod metal;
pub mod ode;
pub mod optimize;
pub mod pareto;
pub mod polish;
pub(crate) mod rng;
pub mod silence;
pub mod solution;
#[cfg(feature = "smt")]
pub mod verify;
#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_forward;

pub use accel::{gpu_backend, GpuBackend};
pub use affine::{discover_affine, discover_affine_pareto, AffineSolution};
pub use analyze::{analyze, certified_range, certified_root, Analysis};
pub use any_solution::{merge_pareto, AnySolution};
pub use config::{Backend, Config, TempSchedule};
pub use dataset::{DataSet, Standardizer};
pub use dimension::{pi_groups, Dimension};
pub use discoverer::{discover_auto, discover_auto_all, Discoverer};
pub use distill::{distill, Distilled};
#[cfg(feature = "egraph")]
pub use egraph::canonical_latex_egraph;
pub use error::{PhopError, Result};
pub use fit::{fit_constants, mse, n_constants};
pub use forest::{eml_guarded, eval_tree};
pub use gated::{discover_gated, discover_gated_warm};
#[cfg(feature = "gpu-cuda")]
pub use gpu::{cuda_available, discover_gumbel_cuda, eval_tree_cuda, CudaEmlEngine};
pub use gumbel::discover_gumbel;
#[cfg(feature = "lean")]
pub use lean::{kernel_self_check, prove_eml_one_lowering, LeanProof};
pub use loss::RobustLoss;
#[cfg(feature = "gpu-metal")]
pub use metal::{discover_gumbel_metal, eval_tree_metal, metal_available, MetalEmlEngine};
pub use ode::discover_ode;
pub use optimize::{polish_constants_scirs, ScirsPolish};
pub use pareto::ParetoFront;
pub use polish::{polish_constants, polish_constants_robust, snap_constants};
pub use solution::Solution;
#[cfg(feature = "smt")]
pub use verify::{prove_equivalent, prove_lower_bound, prove_no_root, prove_upper_bound, Verdict};
#[cfg(feature = "gpu-wgpu")]
pub use wgpu_forward::{eval_tree_wgpu, wgpu_available};

// Re-export the EML AST + certified-analysis types so downstream crates have a single import surface.
pub use oxieml::{EmlNode, EmlTree, RootCertificate, RootStatus};
