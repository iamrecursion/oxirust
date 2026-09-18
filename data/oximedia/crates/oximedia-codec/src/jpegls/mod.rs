//! JPEG-LS lossless/near-lossless image codec (ISO 14495-1).
//! Implements the LOCO-I predictor + Golomb-Rice entropy coding.
//!
//! JPEG-LS uses LOCO-I (Low COmplexity LOssless COmpression for Images):
//! context-modeled Golomb-Rice entropy coding driven by a simple
//! edge-detecting predictor. Patents US6195465 and US6094511 (HP) expired
//! 2017-2019 — this implementation is entirely patent-free.
//!
//! # Feature flag
//!
//! ```toml
//! oximedia-codec = { version = "0.1.7", features = ["jpegls"] }
//! ```

pub mod context;
pub mod decoder;
pub mod encoder;
pub mod golomb;
pub mod golomb_write;
pub mod marker_write;
pub mod markers;
pub mod predictor;
pub mod run_mode;

pub use decoder::{DecodedImage, JpegLsDecoder};
pub use encoder::{JpegLsEncoder, JpegLsEncoderConfig};

use std::fmt;

/// JPEG-LS decode error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JlsError {
    /// Truncated input — stream ended before all data was consumed.
    Truncated {
        /// Location in the decode pipeline where truncation was detected.
        context: &'static str,
    },
    /// Invalid or unrecognized marker.
    InvalidMarker(u16),
    /// Unsupported JPEG-LS feature.
    Unsupported(String),
    /// Data does not begin with JPEG SOI + SOF55 markers.
    NotJpegLs,
}

impl fmt::Display for JlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { context } => write!(f, "truncated JPEG-LS stream in {context}"),
            Self::InvalidMarker(m) => write!(f, "invalid JPEG-LS marker: 0x{m:04X}"),
            Self::Unsupported(s) => write!(f, "unsupported JPEG-LS feature: {s}"),
            Self::NotJpegLs => write!(f, "data is not a JPEG-LS stream"),
        }
    }
}

impl std::error::Error for JlsError {}

/// Convenience alias for `Result<T, JlsError>`.
pub type JlsResult<T> = Result<T, JlsError>;
