//! `ALACSpecificConfig` "magic cookie" parsing and serialization.
//!
//! The magic cookie is the codec-private configuration ALAC stores in the
//! container (the `alac` box payload in MP4, or the `kuki` chunk in CAF). It is
//! a fixed 24-byte big-endian structure (Apple's `ALACSpecificConfig`):
//!
//! | Field             | Type | Bytes | Description                              |
//! |-------------------|------|-------|------------------------------------------|
//! | `frameLength`     | u32  | 4     | Samples per channel per frame            |
//! | `compatibleVersion`| u8  | 1     | Format version (0)                       |
//! | `bitDepth`        | u8   | 1     | Source PCM bit depth                     |
//! | `pb`              | u8   | 1     | Rice "history multiplier" tuning         |
//! | `mb`              | u8   | 1     | Rice "initial history" tuning            |
//! | `kb`              | u8   | 1     | Rice "k modifier" tuning                 |
//! | `numChannels`     | u8   | 1     | Channel count                            |
//! | `maxRun`          | u16  | 2     | Maximum run of equal samples             |
//! | `maxFrameBytes`   | u32  | 4     | Largest compressed frame in bytes        |
//! | `avgBitRate`      | u32  | 4     | Average bit rate (0 if unknown)          |
//! | `sampleRate`      | u32  | 4     | Sample rate in Hz                        |
//!
//! Some real-world cookies prefix the structure with a 12-byte ISOBMFF box
//! header (`size` + `'alac'` + version/flags). [`AlacSpecificConfig::parse`]
//! tolerates that prefix.

use super::{AlacError, AlacResult};

/// Default Rice "history multiplier" (`pb`) from Apple's reference encoder.
pub const DEFAULT_PB: u8 = 40;
/// Default Rice "initial history" (`mb`) from Apple's reference encoder.
pub const DEFAULT_MB: u8 = 10;
/// Default Rice "k modifier" (`kb`) from Apple's reference encoder.
pub const DEFAULT_KB: u8 = 14;

/// The size in bytes of the bare `ALACSpecificConfig` structure.
pub const COOKIE_SIZE: usize = 24;

/// Parsed `ALACSpecificConfig` magic cookie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlacSpecificConfig {
    /// Samples per channel in a full frame.
    pub frame_length: u32,
    /// Compatible format version (always 0 for the released format).
    pub compatible_version: u8,
    /// Source PCM bit depth (16, 20, 24, or 32).
    pub bit_depth: u8,
    /// Rice parameter "history multiplier" (Apple default 40).
    pub pb: u8,
    /// Rice parameter "initial history" (Apple default 10).
    pub mb: u8,
    /// Rice parameter "k modifier" (Apple default 14).
    pub kb: u8,
    /// Number of audio channels.
    pub num_channels: u8,
    /// Maximum run length (informational; encoder writes 255).
    pub max_run: u16,
    /// Maximum compressed frame size in bytes (0 if unknown).
    pub max_frame_bytes: u32,
    /// Average bit rate in bits per second (0 if unknown).
    pub avg_bit_rate: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AlacSpecificConfig {
    /// Build a config from the essential fields, filling tuning + bookkeeping
    /// fields with Apple's defaults.
    #[must_use]
    pub fn new(frame_length: u32, sample_rate: u32, num_channels: u8, bit_depth: u8) -> Self {
        Self {
            frame_length,
            compatible_version: 0,
            bit_depth,
            pb: DEFAULT_PB,
            mb: DEFAULT_MB,
            kb: DEFAULT_KB,
            num_channels,
            max_run: 255,
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            sample_rate,
        }
    }

    /// Validate the structural invariants the codec relies on.
    pub fn validate(&self) -> AlacResult<()> {
        if self.frame_length == 0 {
            return Err(AlacError::InvalidConfig("frame_length is zero".into()));
        }
        if self.num_channels == 0 {
            return Err(AlacError::InvalidConfig("num_channels is zero".into()));
        }
        match self.bit_depth {
            16 | 20 | 24 | 32 => {}
            other => {
                return Err(AlacError::InvalidConfig(format!(
                    "unsupported bit_depth {other} (expected 16/20/24/32)"
                )));
            }
        }
        if self.sample_rate == 0 {
            return Err(AlacError::InvalidConfig("sample_rate is zero".into()));
        }
        Ok(())
    }

    /// Parse a magic cookie from raw bytes (big-endian).
    ///
    /// Accepts either the bare 24-byte structure or one prefixed with a
    /// 12-byte ISOBMFF `'alac'` full-box header.
    pub fn parse(bytes: &[u8]) -> AlacResult<Self> {
        let body = locate_config(bytes)?;
        if body.len() < COOKIE_SIZE {
            return Err(AlacError::InvalidCookie(format!(
                "need {COOKIE_SIZE} bytes, have {}",
                body.len()
            )));
        }

        let frame_length = read_u32(body, 0);
        let compatible_version = body[4];
        let bit_depth = body[5];
        let pb = body[6];
        let mb = body[7];
        let kb = body[8];
        let num_channels = body[9];
        let max_run = read_u16(body, 10);
        let max_frame_bytes = read_u32(body, 12);
        let avg_bit_rate = read_u32(body, 16);
        let sample_rate = read_u32(body, 20);

        let config = Self {
            frame_length,
            compatible_version,
            bit_depth,
            pb,
            mb,
            kb,
            num_channels,
            max_run,
            max_frame_bytes,
            avg_bit_rate,
            sample_rate,
        };
        config.validate()?;
        Ok(config)
    }

    /// Serialize this config to the bare 24-byte big-endian magic cookie.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(COOKIE_SIZE);
        out.extend_from_slice(&self.frame_length.to_be_bytes());
        out.push(self.compatible_version);
        out.push(self.bit_depth);
        out.push(self.pb);
        out.push(self.mb);
        out.push(self.kb);
        out.push(self.num_channels);
        out.extend_from_slice(&self.max_run.to_be_bytes());
        out.extend_from_slice(&self.max_frame_bytes.to_be_bytes());
        out.extend_from_slice(&self.avg_bit_rate.to_be_bytes());
        out.extend_from_slice(&self.sample_rate.to_be_bytes());
        out
    }
}

/// Locate the start of the `ALACSpecificConfig` body, skipping an optional
/// ISOBMFF full-box header (`size:u32` + `'alac'` + `version/flags:u32`).
fn locate_config(bytes: &[u8]) -> AlacResult<&[u8]> {
    if bytes.len() >= 12 && &bytes[4..8] == b"alac" {
        Ok(&bytes[12..])
    } else if bytes.is_empty() {
        Err(AlacError::InvalidCookie("empty cookie".into()))
    } else {
        Ok(bytes)
    }
}

#[inline]
fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

#[inline]
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_serialize_parse() {
        let cfg = AlacSpecificConfig::new(4096, 44_100, 2, 16);
        let bytes = cfg.serialize();
        assert_eq!(bytes.len(), COOKIE_SIZE);
        let parsed = AlacSpecificConfig::parse(&bytes).expect("parse");
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn test_defaults() {
        let cfg = AlacSpecificConfig::new(4096, 48_000, 1, 24);
        assert_eq!(cfg.pb, DEFAULT_PB);
        assert_eq!(cfg.mb, DEFAULT_MB);
        assert_eq!(cfg.kb, DEFAULT_KB);
        assert_eq!(cfg.compatible_version, 0);
    }

    #[test]
    fn test_parse_with_box_header() {
        let cfg = AlacSpecificConfig::new(4096, 44_100, 2, 16);
        let mut boxed = Vec::new();
        boxed.extend_from_slice(&(36u32).to_be_bytes());
        boxed.extend_from_slice(b"alac");
        boxed.extend_from_slice(&0u32.to_be_bytes()); // version + flags
        boxed.extend_from_slice(&cfg.serialize());
        let parsed = AlacSpecificConfig::parse(&boxed).expect("parse boxed");
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn test_parse_too_short() {
        assert!(AlacSpecificConfig::parse(&[0u8; 10]).is_err());
        assert!(AlacSpecificConfig::parse(&[]).is_err());
    }

    #[test]
    fn test_validate_rejects_bad_depth() {
        let mut cfg = AlacSpecificConfig::new(4096, 44_100, 2, 16);
        cfg.bit_depth = 17;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_channels() {
        let mut cfg = AlacSpecificConfig::new(4096, 44_100, 2, 16);
        cfg.num_channels = 0;
        assert!(cfg.validate().is_err());
    }
}
