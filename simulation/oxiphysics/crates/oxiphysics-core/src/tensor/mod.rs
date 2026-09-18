// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tensor algebra operations for continuum mechanics.
//!
//! Provides second-order tensors (`Tensor2`), fourth-order tensors (`Tensor4`),
//! and a Voigt notation helper for symmetric second-order tensors.

mod cp_init;
#[cfg(test)]
mod cp_init_tests;
mod decomposition;
mod operations;
#[cfg(test)]
mod regression_tests;
mod types;

// Re-export all public items
pub use cp_init::CpInit;
pub(crate) use cp_init::initial_cp_factors;
pub use decomposition::*;
pub use operations::*;
pub use types::*;
