//! # phop
//!
//! Façade crate for the **phop** differentiable symbolic-discovery engine, built on the
//! single binary operator `eml(x, y) = exp(x) − ln(y)`.
//!
//! This crate currently re-exports [`phop_core`] (the engine library). It reserves the bare
//! `phop` name for a forthcoming higher-level façade API.
//!
//! - **Library:** this crate, or [`phop-core`](https://crates.io/crates/phop-core).
//! - **Command-line tool:** install [`phop-cli`](https://crates.io/crates/phop-cli), which
//!   provides the `phop` binary (e.g. `phop discover data.csv`).
pub use phop_core::*;
