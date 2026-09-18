//! Spectral analysis module for frequency-domain audio analysis.

pub mod analyze;
pub mod bandwidth;
pub mod centroid;
pub mod chroma;
pub mod crest;
pub mod fft_frame;
pub mod flatness;
pub mod flux;
pub mod overlap_save;
pub mod rolloff;
pub mod zcr;

pub use analyze::{SpectralAnalyzer, SpectralFeatures};
pub use bandwidth::spectral_bandwidth;
pub use centroid::spectral_centroid;
pub use chroma::{
    chroma_track, chroma_vector, estimate_key, mean_chroma, ChromaConfig, ChromaVector,
    NUM_PITCH_CLASSES, PITCH_CLASS_NAMES,
};
pub use crest::spectral_crest;
pub use fft_frame::{FftSize, FftSpectralAnalyzer, SpectralConfig, SpectralFrame, WindowFunction};
pub use flatness::spectral_flatness;
pub use flux::{
    detect_onsets_from_flux, spectral_flux, spectral_flux_hwr, spectral_flux_hwr_track,
    spectral_flux_normalised, spectral_flux_track,
};
pub use overlap_save::{
    overlap_save_power_envelope, overlap_save_spectra, OlaBlock, OverlapSaveAnalyzer,
};
pub use rolloff::{
    spectral_rolloff, spectral_rolloff_85, spectral_rolloff_95, spectral_rolloff_track,
};
pub use zcr::{
    mean_zcr, voiced_unvoiced_frames, zcr_statistics, zero_crossing_count, zero_crossing_rate,
    zero_crossing_rate_framed, ZcrStats,
};
