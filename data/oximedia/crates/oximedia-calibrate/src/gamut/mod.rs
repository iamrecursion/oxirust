//! Gamut mapping and compression.
//!
//! This module provides tools for mapping colors between different color gamuts.

pub mod compress;
pub mod map;

pub use compress::{GamutCompression, GamutCompressionMethod};
pub use map::{GamutMapper, GamutMappingStrategy};
