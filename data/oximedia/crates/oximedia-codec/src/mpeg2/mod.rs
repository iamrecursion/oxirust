//! MPEG-2 video **I-frame** decoder (ISO/IEC 13818-2 / ITU-T H.262).
//!
//! # Legal posture
//!
//! The core MPEG-2 video patents expired in **February 2023** (the MPEG-LA
//! MPEG-2 patent portfolio licence ended; the last US patents lapsed). MPEG-2
//! video is therefore now patent-free and admissible in OxiMedia under the same
//! posture as Theora / JPEG 2000 / DNxHD.
//!
//! # Scope (Wave 8 Slice 2)
//!
//! This module implements an **intra-only** (I-frame) decoder:
//!
//! - `picture_coding_type == 1` (I pictures, intra macroblocks only).
//! - **4:2:0** chroma subsampling (chroma_format == 1).
//! - Frame (progressive) picture structure.
//! - Decode to **YUV 4:2:0 planar** (Y full-res, Cb/Cr half-res each axis).
//!
//! It parses the sequence header, sequence extension, GOP header, picture
//! header, picture coding extension and slice headers, then decodes intra
//! macroblocks: DC via Table B-12/B-13, AC via Table B-14 (or B-15 when
//! `intra_vlc_format == 1`), intra inverse quantisation (§7.4) with saturation
//! and mismatch control (§7.4.4), inverse scan (progressive Fig 7-2 or
//! alternate Fig 7-3), and an IEEE-1180-tolerant 8×8 inverse DCT.
//!
//! # Out of scope (documented follow-ups)
//!
//! - **P and B pictures** (motion compensation, motion vectors, forward/backward
//!   prediction). Only `picture_coding_type == 1` is accepted; others return an
//!   `Err`.
//! - **4:2:2 and 4:4:4** chroma formats. Only 4:2:0 is decoded.
//! - **Field pictures / interlaced** reconstruction (only frame pictures).
//! - **An encoder.** This module decodes only.
//! - **`d_picture` (DC intra) and scalable / SNR / spatial extensions.**
//!
//! # Self-containment
//!
//! The module is intentionally self-contained: it pattern-copies the bit
//! reader, IDCT and zig-zag building blocks (which also exist behind the
//! `dnxhd` feature) with a local [`Mpeg2Error`] so that it compiles with only
//! `--features mpeg2` (it does NOT depend on the `dnxhd` feature).
//!
//! # Feature gate
//!
//! ```toml
//! oximedia-codec = { version = "0.1.7", features = ["mpeg2"] }
//! ```

pub mod bitreader;
pub mod bitwriter;
pub mod decode;
pub mod dequant;
pub mod encoder;
pub mod entropy;
pub mod fdct;
pub mod headers;
pub mod idct;
pub mod marker_write;
pub mod quantize_fwd;
pub mod vlc_encode;
pub mod vlc_tables;
pub mod zigzag;

pub use decode::{Mpeg2Decoder, Mpeg2Frame};
pub use encoder::{Mpeg2Encoder, Mpeg2EncoderConfig};
pub use headers::{
    PictureCodingExtension, PictureHeader, SequenceExtension, SequenceHeader, SliceHeader,
};

use std::fmt;

/// Error type for MPEG-2 I-frame decoding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mpeg2Error {
    /// A required start code (`00 00 01 xx`) was not found in the bitstream.
    StartCodeNotFound(u8),
    /// The bitstream ended before the decoder had consumed all required bits.
    UnexpectedEof {
        /// Number of bits required by the operation.
        need: usize,
        /// Number of bits actually available.
        have: usize,
    },
    /// A header field held a value the decoder cannot handle (e.g. P/B picture,
    /// 4:2:2/4:4:4 chroma, reserved code).
    Unsupported(String),
    /// The bitstream is structurally invalid (bad marker bit, impossible size).
    InvalidData(String),
    /// A variable-length code could not be matched against the relevant table.
    VlcDecode(String),
    /// The encoder configuration is invalid (bad dimensions, qscale, precision).
    InvalidConfig(String),
    /// Encoding failed (plane too small, VLC/header write error).
    Encode(String),
}

impl fmt::Display for Mpeg2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartCodeNotFound(code) => {
                write!(f, "MPEG-2 start code 0x{code:02X} not found")
            }
            Self::UnexpectedEof { need, have } => {
                write!(
                    f,
                    "MPEG-2 unexpected end of stream: need {need} bits, have {have}"
                )
            }
            Self::Unsupported(msg) => write!(f, "MPEG-2 unsupported: {msg}"),
            Self::InvalidData(msg) => write!(f, "MPEG-2 invalid data: {msg}"),
            Self::VlcDecode(msg) => write!(f, "MPEG-2 VLC decode error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "MPEG-2 invalid encoder config: {msg}"),
            Self::Encode(msg) => write!(f, "MPEG-2 encode error: {msg}"),
        }
    }
}

impl std::error::Error for Mpeg2Error {}

/// Convenient result alias for MPEG-2 decoding.
pub type Mpeg2Result<T> = Result<T, Mpeg2Error>;
