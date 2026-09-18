//! Utility modules for audio restoration.

pub mod interpolation;
pub mod spectral;

pub use interpolation::{interpolate, InterpolationMethod};
pub use spectral::{
    apply_window, find_peaks, spectral_centroid, spectral_flatness, spectral_rolloff, FftProcessor,
    WindowFunction,
};
