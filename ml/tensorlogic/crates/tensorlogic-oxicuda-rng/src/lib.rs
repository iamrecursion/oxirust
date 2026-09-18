//! # tensorlogic-oxicuda-rng
//!
//! GPU-accelerated random number generation for TensorLogic, with a pure-Rust
//! CPU fallback.
//!
//! ## Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `cpu` (default) | PCG-XSH-RR generator with Box-Muller transform, zero external dependencies |
//! | `gpu`           | `oxicuda-rand` GPU backend (requires an NVIDIA GPU + CUDA driver at runtime) |
//!
//! ## Quick start
//!
//! ```rust
//! use tensorlogic_oxicuda_rng::{RngEngine, RngEngineKind};
//!
//! let mut rng = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
//!
//! // f32 variants
//! let mut out = vec![0f32; 1024];
//! rng.uniform_f32(&mut out).unwrap();
//!
//! let mut normal_out = vec![0f32; 1024];
//! rng.normal_f32(&mut normal_out, 0.0, 1.0).unwrap();
//!
//! // f64 variants (52-bit mantissa precision)
//! let mut out64 = vec![0f64; 1024];
//! rng.uniform_f64(&mut out64).unwrap();
//!
//! let mut normal_out64 = vec![0f64; 1024];
//! rng.normal_f64(&mut normal_out64, 0.0, 1.0).unwrap();
//!
//! // Streaming API — large buffers without single allocation
//! rng.fill_uniform_chunked(1_000_000, 4096, &mut |chunk: &[f32]| {
//!     let _ = chunk; // process chunk
//! }).unwrap();
//!
//! let mut bernoulli_out = vec![0u8; 1024];
//! rng.bernoulli(&mut bernoulli_out, 0.3).unwrap();
//! ```
//!
//! ## Thread safety
//!
//! On the **CPU path** (`default`), [`RngEngine`] is `Send + Sync`.
//! On the **GPU path** (`feature = "gpu"`), [`RngEngine`] is `Send` but not `Sync`.
//!
//! ## Policy notes
//!
//! * No `rand`, `rand_distr`, or `ndarray` imports — the PCG generator and
//!   Box-Muller transform are implemented from scratch in [`engine`].
//! * `scirs2-core` is listed as an optional dependency under the `cpu` feature
//!   to satisfy workspace policy; the actual random primitives live in
//!   `engine::CpuRngState` for zero-dependency builds.

#![deny(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod engine;
pub mod error;

pub use engine::{RngEngine, RngEngineKind};
pub use error::RngError;
