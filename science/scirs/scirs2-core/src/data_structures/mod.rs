//! Advanced data structures for SciRS2.
//!
//! This module provides specialised data structures not found in the Rust
//! standard library.
//!
//! # Submodules
//!
//! - [`rrb_tree`]: Persistent generic vector backed by a Relaxed Radix-Balanced
//!   tree with structural sharing via `Arc`.

pub mod rrb_tree;

pub use rrb_tree::{PersistentVec, PersistentVecIter};
