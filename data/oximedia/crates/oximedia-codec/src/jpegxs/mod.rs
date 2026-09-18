// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS (ISO/IEC 21122-1:2019) decoder.
//!
//! Implements the JPEG XS low-latency intra-frame codec for broadcast
//! and SMPTE ST 2110 IP production workflows.
//!
//! # Overview
//!
//! JPEG XS is a visually-lossless, intra-frame video compression format
//! standardised in ISO/IEC 21122-1:2019. It is designed for minimal latency
//! (sub-millisecond), low complexity, and is patent-free. It is used in:
//!
//! - SMPTE ST 2110-22 (IP video transport for broadcast production)
//! - HDMI 2.1 Display Stream Compression (DSC variant)
//! - VSF TR-08 (video over IP)
//!
//! # Codec structure
//!
//! - **Wavelet**: LeGall 5/3 integer reversible wavelet (same filter as JPEG 2000 lossless)
//! - **Entropy coding**: VLC-based coefficient coding (NOT arithmetic coding)
//! - **Codestream**: framed by JPEG-style markers (FF xx)
//!
//! # Feature flag
//!
//! ```toml
//! oximedia-codec = { version = "0.1.7", features = ["jpegxs"] }
//! ```

use thiserror::Error;

/// Error type for JPEG XS decode operations.
#[derive(Debug, Error, Clone)]
pub enum JxsError {
    /// Codestream was shorter than required.
    #[error("truncated codestream: need {need} bytes, have {have}")]
    TruncatedStream {
        /// Number of bytes required.
        need: usize,
        /// Number of bytes available.
        have: usize,
    },
    /// An unexpected marker value was encountered.
    #[error("invalid marker: expected {expected:#06x}, got {got:#06x}")]
    InvalidMarker {
        /// Expected marker value.
        expected: u16,
        /// Actual marker value found.
        got: u16,
    },
    /// A codec feature or profile is not yet supported.
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    /// The codestream header contains invalid values.
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    /// VLC entropy decoding error.
    #[error("VLC decode error: {0}")]
    VlcError(String),
}

/// Result alias for JPEG XS operations.
pub type JxsResult<T> = Result<T, JxsError>;

pub mod bitreader;
pub mod bitwriter;
pub mod decoder;
pub mod encoder;
pub mod entropy;
pub mod marker_write;
pub mod markers;
pub mod nlt;
pub mod vlc;
pub mod vlc_encode;
pub mod wavelet;

pub use decoder::{DecodedImage, JpegXsDecoder};
pub use encoder::{JpegXsEncoder, JpegXsEncoderConfig};
