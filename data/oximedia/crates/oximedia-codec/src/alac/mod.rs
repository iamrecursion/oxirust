//! ALAC (Apple Lossless Audio Codec) encoder and decoder.
//!
//! ALAC is Apple's royalty-free lossless audio codec. Apple released the
//! reference implementation under the Apache License 2.0 in October 2011, so
//! ALAC is patent-free and freely implementable.
//!
//! This module operates on **raw ALAC frames** plus the `ALACSpecificConfig`
//! "magic cookie" (it does not parse MP4/CAF containers — that is the job of a
//! container crate). It is the Apple-ecosystem complement to the [`crate::flac`]
//! lossless codec.
//!
//! # Algorithm
//!
//! ALAC compresses each audio frame with three stages, mirroring Apple's
//! reference (`ALACDecoder`/`ALACEncoder`, `ag_dec`/`ag_enc`, `dp_dec`/`dp_enc`,
//! `matrix_dec`/`matrix_enc`):
//!
//! 1. **Inter-channel decorrelation** ([`mix`]) — for stereo pairs, the two
//!    channels are recombined with a mid/side-style integer predictor
//!    (`mixBits`/`mixRes`) that has an exact integer inverse.
//! 2. **Dynamic predictor** ([`lpc`]) — an order-N adaptive FIR predictor with
//!    sign-LMS coefficient adaptation. The encoder produces residuals with the
//!    identical adaptation, so it is the exact inverse of the decoder.
//! 3. **Adaptive Golomb / modified Rice** ([`rice`]) — residuals are entropy
//!    coded with a parameter derived from a running mean controlled by the
//!    `pb`/`mb`/`kb` tuning values, including an escape-to-fixed-bits path for
//!    outliers.
//!
//! # Lossless guarantee
//!
//! Encode → decode is **byte-exact** for 16/20/24-bit audio (32-bit is
//! best-effort; the rare extended predictor mode is not used by this encoder).
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::alac::{AlacEncoder, AlacEncoderConfig, AlacDecoder};
//!
//! let cfg = AlacEncoderConfig {
//!     frame_length: 4096,
//!     sample_rate: 44_100,
//!     channels: 1,
//!     bit_depth: 16,
//! };
//! let mut encoder = AlacEncoder::new(cfg).expect("encoder");
//! let cookie = encoder.magic_cookie();
//!
//! let pcm: Vec<i32> = (0..4096).map(|i| ((i as f64 * 0.1).sin() * 8000.0) as i32).collect();
//! let frame = encoder.encode_frame(&pcm).expect("encode");
//!
//! let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
//! let decoded = decoder.decode_packet(&frame).expect("decode");
//! assert_eq!(decoded, pcm);
//! ```

pub mod bitstream;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod lpc;
pub mod mix;
pub mod rice;

pub use config::AlacSpecificConfig;
pub use decoder::AlacDecoder;
pub use encoder::{AlacEncoder, AlacEncoderConfig};

use thiserror::Error;

/// Errors produced by the ALAC encoder and decoder.
#[derive(Debug, Error)]
pub enum AlacError {
    /// The magic cookie (`ALACSpecificConfig`) was malformed or too short.
    #[error("invalid ALAC magic cookie: {0}")]
    InvalidCookie(String),

    /// A configuration value was out of range or unsupported.
    #[error("invalid ALAC configuration: {0}")]
    InvalidConfig(String),

    /// The compressed frame ran out of bits before decoding finished.
    #[error("truncated ALAC frame: {0}")]
    Truncated(String),

    /// The bitstream contained a value that violates the ALAC format.
    #[error("invalid ALAC bitstream: {0}")]
    InvalidBitstream(String),

    /// The supplied PCM did not match the configured frame geometry.
    #[error("invalid ALAC input: {0}")]
    InvalidInput(String),

    /// A feature required by the bitstream is not implemented (e.g. the rare
    /// 32-bit extended predictor).
    #[error("unsupported ALAC feature: {0}")]
    Unsupported(String),
}

/// Result type for ALAC operations.
pub type AlacResult<T> = Result<T, AlacError>;
