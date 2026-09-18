//! ICC profile generation, parsing, and application.
//!
//! This module provides tools for working with ICC color profiles.

pub mod apply;
pub mod generate;
pub mod parse;

pub use apply::IccProfileApplicator;
pub use generate::IccProfileGenerator;
pub use parse::{
    parse_iccmax, IccMaxProfile, IccMaxTag, IccProfile, IccProfileVersion, SpectralData,
};
