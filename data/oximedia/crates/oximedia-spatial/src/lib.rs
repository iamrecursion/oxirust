//! Spatial audio processing for OxiMedia.
//!
//! This crate provides a suite of spatial audio tools:
//!
//! - **Ambisonics** — Higher-Order Ambisonics (HOA) encoding and decoding up to 5th order,
//!   with SIMD-accelerated spherical harmonic computation
//! - **Binaural** — HRTF-based binaural rendering for headphone spatialization
//! - **Binauralizer** — Multi-channel to binaural downmix via virtual speaker rendering
//! - **DBAP** — Distance-Based Amplitude Panning for arbitrary loudspeaker arrays
//! - **HOA Decoder** — AllRAD decoding for arbitrary speaker arrays
//! - **Room Simulation** — Image-source room acoustics with early reflections and late reverberation
//! - **Reverb** — Schroeder/Moorer algorithmic reverberation
//! - **Spatial Audio Format** — ADM BWF and MPEG-H 3D Audio metadata I/O
//! - **VBAP** — Vector Base Amplitude Panning for 2D and 3D loudspeaker arrays
//! - **Head Tracking** — IMU sensor fusion (gyro + accel + magnetometer) with complementary filter
//! - **Wave Field Synthesis** — Per-speaker delay/gain computation for WFS arrays
//! - **Object Audio** — ADM objects with distance attenuation, Doppler effect, and Dolby Atmos beds
//! - **Zone Control** — Multi-zone spatial audio rendering for installations
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_spatial::ambisonics::{AmbisonicsEncoder, AmbisonicsOrder, SoundSource};
//!
//! let encoder = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
//! let source = SoundSource::new(45.0, 0.0);
//! let mono = vec![0.5_f32; 256];
//! let channels = encoder.encode_mono(&mono, &source);
//! assert_eq!(channels.len(), 4); // W, Y, Z, X
//! ```

use thiserror::Error;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by `oximedia-spatial` operations.
#[derive(Debug, Error)]
pub enum SpatialError {
    /// An invalid configuration was supplied (e.g. too few speakers for VBAP).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Failed to parse structured data (e.g. ADM XML attributes).
    #[error("parse error: {0}")]
    ParseError(String),

    /// A required computation failed (e.g. matrix inversion on a degenerate input).
    #[error("computation error: {0}")]
    ComputationError(String),
}

// ─── Modules ──────────────────────────────────────────────────────────────────

pub mod acoustic_raytracer;
pub mod adm_bwf;
pub mod ambisonics;
pub mod ambisonics_rotation;
pub mod binaural;
pub mod binauralizer;
pub mod cross_talk_cancellation;
pub mod dbap;
pub mod distance_attenuation;
pub mod doppler;
pub mod head_tracking;
pub mod hoa_decoder;
pub mod imu_fusion;
pub mod loudness_map;
pub mod near_field;
pub mod near_field_compensation;
pub mod object_audio;
pub mod occlusion;
pub mod partitioned_convolution;
pub mod reverb;
pub mod reverb_tail_analyzer;
pub mod room_simulation;
pub mod spatial_audio_format;
pub mod spatial_capture;
pub mod speaker_placement;
pub mod vbap;
pub mod wave_field;
pub mod zone_control;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use head_tracking::{cached_rotation_matrix, compute_rotation_matrix};
