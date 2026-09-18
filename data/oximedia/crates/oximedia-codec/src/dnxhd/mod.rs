//! DNxHD / VC-3 decoder (SMPTE ST 2019-1).
//!
//! Supports profiles:
//! - DNxHD 145 (CID 1237, 1440×1080, 8-bit 4:2:2)
//! - DNxHD 220 (CID 1238, 1920×1080, 8-bit 4:2:2)
//! - DNxHD 220x (CID 1235, 1920×1080, 10-bit 4:2:2)
//! - DNxHD 145x (CID 1241, 1440×1080, 10-bit 4:2:2)
//! - DNxHD 100 (CID 1242, 1280×720, 8-bit 4:2:2)
//! - DNxHD 60 (CID 1243, 1280×720, 8-bit 4:2:2)
//!
//! # Legal posture
//!
//! SMPTE ST 2019-1 (VC-3) is a publicly documented standard. The decoder
//! is patent-unencumbered — the same posture as ProRes (SMPTE public spec,
//! only encoding requires an Avid licence). No encoder is provided.
//!
//! # Feature gate
//!
//! This module is only compiled when the `dnxhd` Cargo feature is enabled:
//!
//! ```toml
//! oximedia-codec = { version = "0.1.7", features = ["dnxhd"] }
//! ```

pub mod bitreader;
pub mod decode;
pub mod entropy;
pub mod frame_header;
pub mod idct;
pub mod vlc_tables;
pub mod zigzag;

pub use decode::{DecodedFrame, DnxhdDecoder};
pub use frame_header::{parse_frame_header, DnxhdProfile, FrameHeader};

/// Error type for DNxHD decoding operations.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Frame did not start with the expected 4-byte magic `0x00 0x00 0x02 0x80`.
    #[error("invalid frame magic")]
    InvalidMagic,

    /// The CID (Compression ID) field is not one of the known DNxHD CIDs.
    #[error("unknown compression ID {0}")]
    UnknownCid(u32),

    /// The profile identified by the CID is not supported by this decoder.
    #[error("unsupported profile {0:?}")]
    UnsupportedProfile(DnxhdProfile),

    /// A buffer provided to the decoder was too small.
    #[error("buffer too small: need {need}, have {have}")]
    BufferTooSmall {
        /// Number of bytes / bits required.
        need: usize,
        /// Number of bytes / bits available.
        have: usize,
    },

    /// The entropy decoder encountered invalid or malformed data.
    #[error("entropy decode: {0}")]
    Entropy(String),

    /// The bitstream contained logically invalid data.
    #[error("invalid data: {0}")]
    InvalidData(String),
}
