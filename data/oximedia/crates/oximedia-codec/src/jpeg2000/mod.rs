//! JPEG 2000 codec (ISO/IEC 15444-1, patents expired 2010).
//!
//! Decoder supports:
//! - Arbitrary-tile, single-layer, 5-3 reversible (lossless) and 9-7
//!   irreversible (lossy) wavelet decode
//! - 8-bit and 16-bit unsigned components
//! - Raw J2K codestreams and JP2 (ISOBMFF) container files
//!
//! Encoder ([`Jpeg2000Encoder`]) supports both the lossless 5-3 reversible
//! pair the decoder reconstructs byte-exact (forward 5-3 DWT, MQ arithmetic
//! encoding (Annex C), forward EBCOT Tier-1 (SPP/MRP/CUP), forward Tier-2
//! single-layer packets, and J2K marker writers) and — since Wave 10 Slice 2 —
//! the lossy 9-7 irreversible path (forward CDF 9/7 DWT, per-subband mid-tread
//! quantisation, lossy QCD marker writer). Multi-layer and progression-order
//! encode remain out of scope.
//!
//! Feature gate: `jpeg2000` (opt-in, not default).

pub mod bitreader;
pub mod box_parser;
pub mod decoder;
pub mod encoder;
pub mod marker_write;
pub mod markers;
pub mod mq_coder;
pub mod mq_encoder;
pub mod quantize_fwd;
pub mod tier1;
pub mod tier1_encode;
pub mod tier2;
pub mod tier2_encode;
pub mod wavelet;

pub use decoder::{DecodedImage, Jpeg2000Decoder};
pub use encoder::{Jpeg2000Encoder, Jpeg2000EncoderConfig};

use thiserror::Error;

/// Errors produced by the JPEG 2000 decoder.
#[derive(Debug, Error)]
pub enum Jp2Error {
    /// The input data was truncated or too short.
    #[error("truncated JPEG 2000 data: expected {needed} bytes at {context}, had {available}")]
    Truncated {
        /// Context description for where truncation was detected.
        context: &'static str,
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes available.
        available: usize,
    },
    /// An invalid or unrecognised marker was encountered.
    #[error("invalid JPEG 2000 marker 0x{marker:04X} at offset {offset}")]
    InvalidMarker {
        /// The bad marker code.
        marker: u16,
        /// Byte offset in the codestream.
        offset: usize,
    },
    /// A JP2 box had an unrecognised or unexpected type.
    #[error("unexpected JP2 box type '{box_type}' at offset {offset}")]
    UnexpectedBox {
        /// Four-character box type string.
        box_type: String,
        /// Byte offset in the file.
        offset: usize,
    },
    /// The JP2 signature magic bytes were wrong.
    #[error("invalid JP2 file signature")]
    InvalidSignature,
    /// The MQ coder ran out of input data.
    #[error("MQ coder out of input")]
    MqOutOfInput,
    /// A coding parameter in a marker segment is unsupported.
    #[error("unsupported JPEG 2000 parameter: {0}")]
    Unsupported(String),
    /// The codestream uses multi-layer quality progression, which this decoder
    /// does not support (single quality layer only; multi-tile grids are supported).
    #[error("multi-layer codestreams are not supported (single layer only)")]
    MultiTileOrLayer,
    /// Arithmetic overflow or other internal computation error.
    #[error("internal computation error: {0}")]
    InternalError(String),
    /// Bit reader ran out of input.
    #[error("bit reader out of input")]
    BitReaderOutOfInput,
}

/// Convenience `Result` alias for JPEG 2000 operations.
pub type Jp2Result<T> = Result<T, Jp2Error>;
