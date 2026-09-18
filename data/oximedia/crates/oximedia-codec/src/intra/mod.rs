//! Shared intra prediction module for AV1 and VP9 decoders.
//!
//! This module provides common intra prediction functionality that can be used
//! by both AV1 and VP9 video decoders. Intra prediction generates prediction
//! samples using only samples from the current frame.
//!
//! # Prediction Modes
//!
//! The module supports several prediction mode categories:
//!
//! - **DC Prediction** - Average of neighboring samples
//! - **Directional Prediction** - Project samples along angles (0-270 degrees)
//! - **Smooth Prediction** - Weighted interpolation (AV1)
//! - **Paeth Prediction** - Adaptive selection (AV1)
//! - **Palette Mode** - Index-based colors (AV1)
//!
//! # Architecture
//!
//! The prediction context ([`IntraPredContext`]) manages neighbor sample
//! availability and provides the interface for prediction operations.
//! Each prediction mode implements the [`IntraPredictor`] trait.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::intra::{IntraPredContext, IntraMode, DcPredictor};
//!
//! let ctx = IntraPredContext::new(width, height, bit_depth);
//! ctx.reconstruct_neighbors(&frame, block_x, block_y);
//!
//! let predictor = DcPredictor::new();
//! predictor.predict(&ctx, &mut block);
//! ```

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]

mod context;
mod dc;
mod directional;
mod filter;
mod modes;
mod paeth;
mod palette;
mod smooth;

// Re-export primary types
pub use context::{IntraPredContext, LeftSamples, NeighborAvailability, TopSamples};
pub use dc::{DcMode, DcPredictor};
pub use directional::{
    D113Predictor, D117Predictor, D135Predictor, D153Predictor, D157Predictor, D203Predictor,
    D207Predictor, D45Predictor, D63Predictor, D67Predictor, DirectionalPredictor,
    HorizontalPredictor, VerticalPredictor,
};
pub use filter::{apply_intra_filter, FilterStrength, IntraEdgeFilter};
pub use modes::{AngleDelta, DirectionalMode, IntraMode, IntraPredictor, NonDirectionalMode};
pub use paeth::{paeth_predictor, PaethPredictor};
pub use palette::{ColorCache, PaletteInfo, PalettePredictor};
pub use smooth::{SmoothHPredictor, SmoothPredictor, SmoothVPredictor};

/// Maximum block size supported (128x128 for AV1).
pub const MAX_BLOCK_SIZE: usize = 128;

/// Maximum number of samples for neighbor arrays (2 * MAX_BLOCK_SIZE + 1).
pub const MAX_NEIGHBOR_SAMPLES: usize = 2 * MAX_BLOCK_SIZE + 1;

/// Supported bit depths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BitDepth {
    /// 8-bit samples.
    #[default]
    Bits8 = 8,
    /// 10-bit samples (AV1 HDR).
    Bits10 = 10,
    /// 12-bit samples (AV1 HDR).
    Bits12 = 12,
}

impl BitDepth {
    /// Get the maximum sample value for this bit depth.
    #[must_use]
    pub const fn max_value(self) -> u16 {
        match self {
            Self::Bits8 => 255,
            Self::Bits10 => 1023,
            Self::Bits12 => 4095,
        }
    }

    /// Get the midpoint value (used for DC prediction with no neighbors).
    #[must_use]
    pub const fn midpoint(self) -> u16 {
        match self {
            Self::Bits8 => 128,
            Self::Bits10 => 512,
            Self::Bits12 => 2048,
        }
    }

    /// Get the number of bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Bits8 => 8,
            Self::Bits10 => 10,
            Self::Bits12 => 12,
        }
    }
}

/// Block dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockDimensions {
    /// Block width in samples.
    pub width: usize,
    /// Block height in samples.
    pub height: usize,
}

impl BlockDimensions {
    /// Create new block dimensions.
    #[must_use]
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Get the total number of samples.
    #[must_use]
    pub const fn num_samples(self) -> usize {
        self.width * self.height
    }

    /// Check if dimensions are square.
    #[must_use]
    pub const fn is_square(self) -> bool {
        self.width == self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_depth_values() {
        assert_eq!(BitDepth::Bits8.max_value(), 255);
        assert_eq!(BitDepth::Bits10.max_value(), 1023);
        assert_eq!(BitDepth::Bits12.max_value(), 4095);

        assert_eq!(BitDepth::Bits8.midpoint(), 128);
        assert_eq!(BitDepth::Bits10.midpoint(), 512);
        assert_eq!(BitDepth::Bits12.midpoint(), 2048);

        assert_eq!(BitDepth::Bits8.bits(), 8);
        assert_eq!(BitDepth::Bits10.bits(), 10);
        assert_eq!(BitDepth::Bits12.bits(), 12);
    }

    #[test]
    fn test_block_dimensions() {
        let dim = BlockDimensions::new(8, 8);
        assert_eq!(dim.num_samples(), 64);
        assert!(dim.is_square());

        let rect = BlockDimensions::new(16, 8);
        assert_eq!(rect.num_samples(), 128);
        assert!(!rect.is_square());
    }
}
