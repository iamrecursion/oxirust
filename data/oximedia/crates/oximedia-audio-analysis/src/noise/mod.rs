//! Noise analysis and profiling module.

pub mod classify;
pub mod profile;
pub mod snr;

pub use classify::{
    classify_noise, classify_noise_detailed, classify_noise_from_samples, NoiseClassification,
    NoiseScores, NoiseType,
};
pub use profile::{NoiseProfile, NoiseProfiler};
pub use snr::{compute_snr_db, signal_to_noise_ratio};
