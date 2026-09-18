//! API freeze module for scirs2-core 1.0
//!
//! This module manages the frozen API surface and compatibility checking.

mod api_freeze_impl;
mod compatibility;

pub use api_freeze_impl::{generate_frozen_api_report, initialize_api_freeze, is_api_frozen};
pub use compatibility::{
    check_apis_available, current_libraryversion, is_api_available, is_version_compatible,
    ApiCompatibilityChecker,
};
