//! Signal Transform Module for v0.2.0
//!
//! Provides comprehensive signal transformation capabilities including:
//! - Discrete Wavelet Transforms (DWT, DWT2D, DWTN)
//! - Continuous Wavelet Transforms (CWT)
//! - Wavelet Packet Decomposition (WPT)
//! - Short-Time Fourier Transform (STFT) and Spectrograms
//! - Mel-Frequency Cepstral Coefficients (MFCC)
//! - Constant-Q Transform (CQT) and Chromagram

pub mod cqt;
pub mod cwt;
pub mod dwt;
pub mod mfcc;
pub mod stft;
pub mod wpt;

// Re-export key types and functions
pub use cqt::{CQTConfig, Chromagram, CQT};
pub use cwt::{ContinuousWavelet, MexicanHatWavelet, MorletWavelet, CWT};
pub use dwt::{BoundaryMode, WaveletType, DWT, DWT2D, DWTN};
pub use mfcc::{MFCCConfig, MelFilterbank, MFCC};
pub use stft::{STFTConfig, Spectrogram, WindowType, STFT};
pub use wpt::{BestBasisCriterion, WaveletPacketNode, WPT};
