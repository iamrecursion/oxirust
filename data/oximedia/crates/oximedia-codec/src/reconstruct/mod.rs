//! Video frame reconstruction pipeline.
//!
//! This module provides the complete reconstruction pipeline for decoded video frames,
//! coordinating all stages from entropy decoding through final output formatting.
//!
//! # Pipeline Stages
//!
//! 1. **Parsing** - OBU/bitstream parsing
//! 2. **Entropy** - Entropy decoding of coefficients
//! 3. **Prediction** - Intra/inter prediction
//! 4. **Transform** - Inverse transform of residuals
//! 5. **Loop Filter** - Deblocking and edge filtering
//! 6. **CDEF** - Constrained Directional Enhancement Filter
//! 7. **Super-res** - AV1 super-resolution upscaling
//! 8. **Film Grain** - Film grain synthesis
//! 9. **Output** - Final format conversion
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::reconstruct::{DecoderPipeline, PipelineConfig};
//!
//! let config = PipelineConfig::default();
//! let mut pipeline = DecoderPipeline::new(config)?;
//!
//! // Process a frame through all stages
//! let output = pipeline.process_frame(&encoded_data)?;
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]

mod buffer;
mod cdef_apply;
mod deblock;
mod film_grain;
mod loop_filter;
mod output;
mod pipeline;
mod residual;
mod super_res;

// Public exports
pub use buffer::{BufferPool, FrameBuffer, PlaneBuffer, ReferenceFrameManager};
pub use cdef_apply::{CdefApplicator, CdefBlockConfig, CdefFilterResult};
pub use deblock::{DeblockFilter, DeblockParams, FilterStrength};
pub use film_grain::{FilmGrainParams, FilmGrainSynthesizer, GrainBlock};
pub use loop_filter::{EdgeFilter, FilterDirection, LoopFilterPipeline};
pub use output::{OutputConfig, OutputFormat, OutputFormatter};
pub use pipeline::{DecoderPipeline, FrameContext, PipelineConfig, PipelineStage, StageResult};
pub use residual::{ResidualBuffer, ResidualPlane};
pub use super_res::{SuperResConfig, SuperResUpscaler, UpscaleMethod};

use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during frame reconstruction.
#[derive(Debug, Error)]
pub enum ReconstructionError {
    /// Invalid input data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Buffer allocation failed.
    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    /// Reference frame not available.
    #[error("Reference frame not available: index {0}")]
    ReferenceNotAvailable(usize),

    /// Pipeline stage error.
    #[error("Pipeline stage '{stage}' failed: {message}")]
    StageError {
        /// The stage that failed.
        stage: String,
        /// Error message.
        message: String,
    },

    /// Invalid dimensions.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },

    /// Unsupported bit depth.
    #[error("Unsupported bit depth: {0}")]
    UnsupportedBitDepth(u8),

    /// Coefficient overflow.
    #[error("Coefficient overflow at ({x}, {y})")]
    CoefficientOverflow {
        /// X coordinate.
        x: usize,
        /// Y coordinate.
        y: usize,
    },

    /// Filter parameter out of range.
    #[error("Filter parameter out of range: {name} = {value}")]
    FilterParameterOutOfRange {
        /// Parameter name.
        name: String,
        /// Parameter value.
        value: i32,
    },

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for reconstruction operations.
pub type ReconstructResult<T> = Result<T, ReconstructionError>;

// =============================================================================
// Common Constants
// =============================================================================

/// Maximum supported bit depth.
pub const MAX_BIT_DEPTH: u8 = 12;

/// Minimum supported bit depth.
pub const MIN_BIT_DEPTH: u8 = 8;

/// Maximum frame width.
pub const MAX_FRAME_WIDTH: u32 = 16384;

/// Maximum frame height.
pub const MAX_FRAME_HEIGHT: u32 = 16384;

/// Number of reference frame slots.
pub const NUM_REF_FRAMES: usize = 8;

/// Maximum superblock size.
pub const MAX_SB_SIZE: usize = 128;

/// Minimum superblock size.
pub const MIN_SB_SIZE: usize = 64;

// =============================================================================
// Common Types
// =============================================================================

/// Plane identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneType {
    /// Luma plane.
    Y,
    /// Chroma U plane.
    U,
    /// Chroma V plane.
    V,
}

impl PlaneType {
    /// Get plane index (0, 1, or 2).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Y => 0,
            Self::U => 1,
            Self::V => 2,
        }
    }

    /// Check if this is a chroma plane.
    #[must_use]
    pub const fn is_chroma(self) -> bool {
        !matches!(self, Self::Y)
    }

    /// Get all plane types.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Y, Self::U, Self::V]
    }
}

/// Chroma subsampling format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaSubsampling {
    /// 4:4:4 - No subsampling.
    Cs444,
    /// 4:2:2 - Horizontal subsampling.
    Cs422,
    /// 4:2:0 - Horizontal and vertical subsampling.
    #[default]
    Cs420,
    /// Monochrome.
    Mono,
}

impl ChromaSubsampling {
    /// Get subsampling ratios (horizontal, vertical).
    #[must_use]
    pub const fn ratios(self) -> (u32, u32) {
        match self {
            Self::Cs444 => (1, 1),
            Self::Cs422 => (2, 1),
            Self::Cs420 => (2, 2),
            Self::Mono => (1, 1),
        }
    }

    /// Get number of planes.
    #[must_use]
    pub const fn num_planes(self) -> usize {
        match self {
            Self::Mono => 1,
            _ => 3,
        }
    }

    /// Calculate chroma dimensions from luma dimensions.
    #[must_use]
    pub fn chroma_size(self, luma_width: u32, luma_height: u32) -> (u32, u32) {
        let (h_ratio, v_ratio) = self.ratios();
        (luma_width.div_ceil(h_ratio), luma_height.div_ceil(v_ratio))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_type() {
        assert_eq!(PlaneType::Y.index(), 0);
        assert_eq!(PlaneType::U.index(), 1);
        assert_eq!(PlaneType::V.index(), 2);

        assert!(!PlaneType::Y.is_chroma());
        assert!(PlaneType::U.is_chroma());
        assert!(PlaneType::V.is_chroma());
    }

    #[test]
    fn test_chroma_subsampling() {
        assert_eq!(ChromaSubsampling::Cs444.ratios(), (1, 1));
        assert_eq!(ChromaSubsampling::Cs422.ratios(), (2, 1));
        assert_eq!(ChromaSubsampling::Cs420.ratios(), (2, 2));

        assert_eq!(ChromaSubsampling::Cs420.num_planes(), 3);
        assert_eq!(ChromaSubsampling::Mono.num_planes(), 1);
    }

    #[test]
    fn test_chroma_size_calculation() {
        let cs = ChromaSubsampling::Cs420;
        assert_eq!(cs.chroma_size(1920, 1080), (960, 540));
        assert_eq!(cs.chroma_size(1921, 1081), (961, 541));
    }

    #[test]
    fn test_reconstruction_error_display() {
        let err = ReconstructionError::InvalidInput("test".to_string());
        assert_eq!(format!("{err}"), "Invalid input: test");

        let err = ReconstructionError::ReferenceNotAvailable(3);
        assert_eq!(format!("{err}"), "Reference frame not available: index 3");

        let err = ReconstructionError::InvalidDimensions {
            width: 0,
            height: 100,
        };
        assert_eq!(format!("{err}"), "Invalid dimensions: 0x100");
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_BIT_DEPTH, 12);
        assert_eq!(MIN_BIT_DEPTH, 8);
        assert_eq!(NUM_REF_FRAMES, 8);
        assert_eq!(MAX_SB_SIZE, 128);
        assert_eq!(MIN_SB_SIZE, 64);
    }
}
