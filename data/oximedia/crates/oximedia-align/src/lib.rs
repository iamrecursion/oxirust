//! Video alignment and registration tools for multi-camera synchronization in `OxiMedia`.
//!
//! This crate provides comprehensive tools for aligning and registering video from multiple cameras:
//!
//! # Temporal Alignment
//!
//! The [`temporal`] module provides time-based synchronization:
//!
//! - **Audio Cross-Correlation** - Sync cameras using audio tracks
//! - **Timecode Synchronization** - LTC/VITC-based alignment
//! - **Visual Markers** - Clapper detection and flash-based sync
//! - **Sub-frame Accuracy** - Precise timing down to microseconds
//!
//! # Spatial Registration
//!
//! The [`spatial`] module provides geometric alignment:
//!
//! - **Homography Estimation** - Planar perspective transformation
//! - **Perspective Correction** - Remove keystone distortion
//! - **Feature Matching** - Correspond points between views
//! - **RANSAC** - Robust outlier rejection
//!
//! # Feature Detection
//!
//! The [`features`] module provides patent-free feature detection and matching:
//!
//! - **FAST Corners** - High-speed corner detection
//! - **BRIEF Descriptors** - Binary robust independent elementary features
//! - **ORB Features** - Oriented FAST and Rotated BRIEF
//! - **Brute-Force Matching** - Hamming distance matching
//!
//! # Lens Distortion
//!
//! The [`distortion`] module corrects lens aberrations:
//!
//! - **Brown-Conrady Model** - Radial and tangential distortion
//! - **Fisheye Model** - Wide-angle lens correction
//! - **Calibration** - Camera intrinsic parameter estimation
//! - **Undistortion** - Real-time image correction
//!
//! # Color Matching
//!
//! The [`color`] module matches color across cameras:
//!
//! - **Color Transfer** - Match color distributions
//! - **Histogram Matching** - Equalize color histograms
//! - **White Balance** - Illuminant estimation
//! - **Color Calibration** - ColorChecker-based calibration
//!
//! # Sync Markers
//!
//! The [`markers`] module detects synchronization markers:
//!
//! - **Clapper Detection** - Automatic slate detection
//! - **Flash Detection** - Bright flash sync
//! - **LED Markers** - Coded light patterns
//! - **Audio Spike** - Sharp transient detection
//!
//! # Rolling Shutter
//!
//! The [`rolling_shutter`] module corrects rolling shutter artifacts:
//!
//! - **Motion Estimation** - Per-scanline motion vectors
//! - **Correction** - Remove wobble and skew
//! - **Global Shutter Simulation** - Temporal interpolation
//!
//! # Example: Audio-Based Sync
//!
//! ```
//! use oximedia_align::temporal::{AudioSync, SyncConfig};
//! use oximedia_align::AlignResult;
//!
//! # fn example() -> AlignResult<()> {
//! // Configure audio synchronization
//! let config = SyncConfig {
//!     sample_rate: 48000,
//!     window_size: 480000, // 10 seconds
//!     max_offset: 240000,  // ±5 seconds
//! };
//!
//! // Create audio sync analyzer
//! let sync = AudioSync::new(config);
//!
//! // Find offset between two audio tracks
//! // let offset = sync.find_offset(&audio1, &audio2)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Homography Estimation
//!
//! ```
//! use oximedia_align::spatial::{HomographyEstimator, RansacConfig};
//! use oximedia_align::features::{FeatureMatcher, MatchPair};
//!
//! # fn example() -> oximedia_align::AlignResult<()> {
//! // Configure RANSAC for robust estimation
//! let config = RansacConfig {
//!     threshold: 3.0,
//!     max_iterations: 1000,
//!     min_inliers: 8,
//! };
//!
//! let estimator = HomographyEstimator::new(config);
//!
//! // Estimate homography from matched points
//! // let (homography, inliers) = estimator.estimate(&matches)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod affine;
pub mod align_report;
pub mod audio_align;
pub mod beat_align;
pub mod bundle_adjust;
pub mod color;
pub mod confidence_map;
pub mod distortion;
pub mod drift_correct;
pub mod drift_correction;
pub mod elastic_align;
pub mod farneback_flow;
pub mod feature_cache;
pub mod features;
pub mod frame_matcher;
pub mod frequency_align;
pub mod gradient_flow;
pub mod icp;
pub mod illumination_invariant;
pub mod image_stitch;
pub mod integral_image;
pub mod klt_tracker;
pub mod lens_calibration;
pub mod lip_sync;
pub mod markers;
pub mod motion_compensate;
pub mod motion_interp;
pub mod multi_stream;
pub mod multicam_sync;
pub mod multitrack_align;
pub mod optical_flow;
pub mod parallel_ransac;
pub mod phase_correlate;
pub mod projective_warp;
pub mod prosac;
pub mod rigid_transform;
pub mod rolling_shutter;
pub mod scene_align;
pub mod simd_hamming;
pub mod spatial;
pub mod stabilize;
pub mod stereo_depth;
pub mod stereo_rectify;
pub mod subframe_interp;
pub mod sync_score;
pub mod tempo_align;
pub mod temporal;
pub mod temporal_align;
pub mod transform;
pub mod warp;

pub use feature_cache::DescriptorCache;

use thiserror::Error;

/// Result type for alignment operations
pub type AlignResult<T> = Result<T, AlignError>;

/// Errors that can occur during alignment operations
#[derive(Debug, Error)]
pub enum AlignError {
    /// Insufficient data for alignment
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No solution found
    #[error("No solution found: {0}")]
    NoSolution(String),

    /// Numerical instability
    #[error("Numerical instability: {0}")]
    NumericalError(String),

    /// Feature detection failed
    #[error("Feature detection failed: {0}")]
    FeatureError(String),

    /// Matching failed
    #[error("Matching failed: {0}")]
    MatchingError(String),

    /// Estimation failed
    #[error("Estimation failed: {0}")]
    EstimationError(String),

    /// Synchronization failed
    #[error("Synchronization failed: {0}")]
    SyncError(String),

    /// Color correction failed
    #[error("Color correction failed: {0}")]
    ColorError(String),

    /// Distortion correction failed
    #[error("Distortion correction failed: {0}")]
    DistortionError(String),

    /// Rolling shutter correction failed
    #[error("Rolling shutter correction failed: {0}")]
    RollingShutterError(String),

    /// Generic error from core
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::error::OxiError),
}

/// 2D point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Compute Euclidean distance to another point
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute squared distance (faster than distance)
    #[must_use]
    pub fn distance_squared(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

/// Time offset between two streams
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeOffset {
    /// Offset in samples (for audio) or frames (for video)
    pub samples: i64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Cross-correlation peak value
    pub correlation: f64,
}

impl TimeOffset {
    /// Create a new time offset
    #[must_use]
    pub fn new(samples: i64, confidence: f64, correlation: f64) -> Self {
        Self {
            samples,
            confidence,
            correlation,
        }
    }

    /// Convert offset to seconds
    #[must_use]
    pub fn to_seconds(&self, sample_rate: u32) -> f64 {
        self.samples as f64 / f64::from(sample_rate)
    }

    /// Convert offset to milliseconds
    #[must_use]
    pub fn to_milliseconds(&self, sample_rate: u32) -> f64 {
        self.to_seconds(sample_rate) * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < f64::EPSILON);
        assert!((p1.distance_squared(&p2) - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_offset_conversion() {
        let offset = TimeOffset::new(48000, 0.95, 0.85);
        assert!((offset.to_seconds(48000) - 1.0).abs() < f64::EPSILON);
        assert!((offset.to_milliseconds(48000) - 1000.0).abs() < f64::EPSILON);
    }
}
