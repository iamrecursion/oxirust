//! MXF (Material Exchange Format) container probing.
//!
//! Provides a lightweight parser that inspects the leading bytes of a buffer
//! to identify MXF containers, determine the operational pattern, and enumerate
//! the essence tracks present.
//!
//! ## MXF structure overview
//!
//! An MXF file starts with a **Header Partition Pack** whose KLV key begins
//! with the 12-byte prefix `06 0E 2B 34 02 05 01 01 0D 01 02 01`.  The
//! partition kind is encoded in byte 13 (`0x01` = header, `0x03` = body,
//! `0x04` = footer).
//!
//! The Operational Pattern is identified by searching the buffer for the
//! SMPTE UL prefix `06 0E 2B 34 04 01 01` followed by a disambiguating byte.

/// The kind of essence carried by a track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MxfTrackType {
    /// Picture / video essence.
    Video,
    /// Sound / audio essence.
    Audio,
    /// Auxiliary / data essence (timecode, ANC data, etc.).
    Data,
}

impl std::fmt::Display for MxfTrackType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MxfTrackType::Video => write!(f, "Video"),
            MxfTrackType::Audio => write!(f, "Audio"),
            MxfTrackType::Data => write!(f, "Data"),
        }
    }
}

/// One essence track discovered inside an MXF container.
#[derive(Debug, Clone)]
pub struct MxfEssenceTrack {
    /// Broad category of this track's essence.
    pub track_type: MxfTrackType,
    /// 16-byte SMPTE UL label identifying the codec/essence container.
    pub codec_label: [u8; 16],
}

/// Information extracted from a parsed MXF container.
#[derive(Debug, Clone)]
pub struct MxfInfo {
    /// SMPTE operational pattern string (e.g. `"OP1a"`, `"OPAtom"`).
    pub operational_pattern: String,
    /// Essence tracks found in the file.
    pub essence_tracks: Vec<MxfEssenceTrack>,
    /// Estimated duration in milliseconds, when available.
    pub duration_ms: Option<u64>,
}

/// Errors that can occur while probing an MXF buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MxfProbeError {
    /// The buffer does not start with the MXF partition pack key.
    NotMxf,
    /// The buffer is too short to contain a valid partition pack.
    TruncatedData,
    /// A structural error was detected during parsing.
    ParseError(String),
}

impl std::fmt::Display for MxfProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MxfProbeError::NotMxf => write!(f, "not an MXF file"),
            MxfProbeError::TruncatedData => write!(f, "truncated MXF data"),
            MxfProbeError::ParseError(msg) => write!(f, "MXF parse error: {msg}"),
        }
    }
}

impl std::error::Error for MxfProbeError {}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// 12-byte KLV key prefix that identifies any MXF partition pack.
const MXF_PARTITION_KEY_PREFIX: [u8; 12] = [
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01,
];

/// Minimum number of bytes required before any meaningful parse is possible.
/// Partition pack = 16-byte key + at least 1-byte BER length = 17 bytes.
const MIN_MXF_SIZE: usize = 17;

/// 7-byte Operational Pattern UL registry prefix.
const OP_KEY_PREFIX: [u8; 7] = [0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01];

/// 7-byte essence descriptor UL prefix (SMPTE 377M registry designator).
const ESSENCE_KEY_PREFIX: [u8; 7] = [0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01];

/// Maximum number of essence tracks to extract per file.
const MAX_TRACKS: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Prober
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight MXF container prober.
pub struct MxfProber;

impl MxfProber {
    /// Probe `data` and return [`MxfInfo`] on success.
    ///
    /// # Errors
    ///
    /// - [`MxfProbeError::NotMxf`] if the buffer does not begin with the MXF
    ///   partition-pack KLV key.
    /// - [`MxfProbeError::TruncatedData`] if the buffer is shorter than the
    ///   minimum required for a valid partition pack.
    /// - [`MxfProbeError::ParseError`] on any structural inconsistency.
    pub fn probe(data: &[u8]) -> Result<MxfInfo, MxfProbeError> {
        if data.len() < MIN_MXF_SIZE {
            return Err(MxfProbeError::TruncatedData);
        }
        if !Self::is_mxf_header(data) {
            return Err(MxfProbeError::NotMxf);
        }

        let operational_pattern = Self::parse_operational_pattern(data);
        let essence_tracks = Self::extract_essence_tracks(data);
        let duration_ms = Self::extract_duration_ms(data);

        Ok(MxfInfo {
            operational_pattern,
            essence_tracks,
            duration_ms,
        })
    }

    /// Return `true` when the buffer starts with a valid MXF partition-pack key.
    ///
    /// Checks the 12-byte KLV key prefix and verifies that the partition kind
    /// byte (offset 13 within the 16-byte key) is a known value.
    fn is_mxf_header(data: &[u8]) -> bool {
        if data.len() < 16 {
            return false;
        }
        // Check the 12-byte MXF partition pack prefix.
        if data[..12] != MXF_PARTITION_KEY_PREFIX {
            return false;
        }
        // Byte 13 (0-indexed: [12]) is the version/item designator.
        // Byte 14 ([13]) is the partition kind: 0x01=header, 0x03=body, 0x04=footer.
        let kind = data[13];
        matches!(kind, 0x01 | 0x03 | 0x04)
    }

    /// Scan the buffer for a SMPTE Operational Pattern UL and return its name.
    ///
    /// Returns `"Unknown"` when no recognisable OP label is found.
    fn parse_operational_pattern(data: &[u8]) -> String {
        // Search for the 7-byte OP label prefix anywhere in the buffer.
        let prefix = &OP_KEY_PREFIX;
        let search_end = data.len().saturating_sub(prefix.len() + 2);

        for i in 0..search_end {
            if &data[i..i + 7] == prefix.as_slice() {
                // The 8th byte (data[i+7]) discriminates the OP variant.
                let discriminator = data[i + 7];
                let op = match discriminator {
                    0x01 => "OP1a",
                    0x02 => "OP1b",
                    0x03 => "OP1c",
                    0x04 => "OP2a",
                    0x05 => "OP2b",
                    0x06 => "OP2c",
                    0x07 => "OP3a",
                    0x08 => "OP3b",
                    0x09 => "OP3c",
                    0x10 => "OPAtom",
                    _ => continue, // not an OP label; keep searching
                };
                return op.to_owned();
            }
        }

        "Unknown".to_owned()
    }

    /// Scan the buffer for essence-descriptor ULs and build a track list.
    ///
    /// Looks for the 7-byte ESSENCE_KEY_PREFIX and uses the surrounding bytes
    /// to classify the track as Video, Audio, or Data.  At most [`MAX_TRACKS`]
    /// tracks are returned.
    fn extract_essence_tracks(data: &[u8]) -> Vec<MxfEssenceTrack> {
        let mut tracks = Vec::new();
        let prefix = &ESSENCE_KEY_PREFIX;

        // We need at least prefix.len() + 9 bytes for a 16-byte label.
        let search_end = data.len().saturating_sub(16);

        let mut i = 0usize;
        while i < search_end && tracks.len() < MAX_TRACKS {
            if &data[i..i + 7] != prefix.as_slice() {
                i += 1;
                continue;
            }

            // Found a potential UL.  Extract the full 16-byte label.
            if i + 16 > data.len() {
                break;
            }
            let mut codec_label = [0u8; 16];
            codec_label.copy_from_slice(&data[i..i + 16]);

            // Classify by bytes 12–13 of the UL (item type designators in
            // SMPTE 377M / 378M / 379M).
            //
            // Byte 12 (codec_label[12]) carries the essence type:
            //   0x01 = Picture (video)
            //   0x02 = Sound (audio)
            //   0x03..0x05 = Data / timecode
            let essence_type_byte = codec_label[12];
            let track_type = match essence_type_byte {
                0x01 => MxfTrackType::Video,
                0x02 => MxfTrackType::Audio,
                _ => MxfTrackType::Data,
            };

            // Deduplicate by codec_label to avoid duplicate entries for the
            // same essence container.
            let already_seen = tracks
                .iter()
                .any(|t: &MxfEssenceTrack| t.codec_label == codec_label);
            if !already_seen {
                tracks.push(MxfEssenceTrack {
                    track_type,
                    codec_label,
                });
            }

            i += 16; // skip past this label
        }

        tracks
    }

    /// Attempt to extract a duration from the partition pack.
    ///
    /// MXF partition packs store a `BodySID` and other fixed fields after the
    /// BER-encoded length.  The heuristic here looks for a plausible 8-byte
    /// big-endian frame count near the start of the buffer and converts it
    /// using an assumed 25 fps edit rate.  Returns `None` when nothing
    /// plausible is found.
    fn extract_duration_ms(data: &[u8]) -> Option<u64> {
        // MXF partition pack key is 16 bytes; then a BER length; then the
        // value fields.  We read a very simple heuristic: look at bytes
        // [72..80] (typical offset of Duration in a header partition pack
        // with a BER-1 or BER-4 length field).
        //
        // If the 8-byte value is in a plausible range (1–864000 frames at 25
        // fps = up to 10 hours), treat it as a frame count.
        let offsets: &[usize] = &[72, 80, 88];
        for &offset in offsets {
            if offset + 8 > data.len() {
                continue;
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[offset..offset + 8]);
            let frame_count = u64::from_be_bytes(buf);
            // Plausible range: 25 fps × 10 hours = 900_000 frames.
            if frame_count > 0 && frame_count <= 900_000 {
                // Assume 25 fps: ms = frames * 1000 / 25 = frames * 40.
                return Some(frame_count * 40);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid MXF header buffer (partition pack key + BER length byte).
    fn minimal_mxf_header(partition_kind: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[..12].copy_from_slice(&MXF_PARTITION_KEY_PREFIX);
        buf[12] = 0x01; // version/item byte
        buf[13] = partition_kind;
        buf[14] = 0x01; // byte 15 of 16-byte key
        buf[15] = 0x01; // byte 16 of 16-byte key
        buf[16] = 0x04; // BER length: 4 bytes follow
        buf
    }

    // ── is_mxf_header ────────────────────────────────────────────────────────

    #[test]
    fn test_is_mxf_header_valid_header_partition() {
        let buf = minimal_mxf_header(0x01);
        assert!(MxfProber::is_mxf_header(&buf));
    }

    #[test]
    fn test_is_mxf_header_valid_body_partition() {
        let buf = minimal_mxf_header(0x03);
        assert!(MxfProber::is_mxf_header(&buf));
    }

    #[test]
    fn test_is_mxf_header_valid_footer_partition() {
        let buf = minimal_mxf_header(0x04);
        assert!(MxfProber::is_mxf_header(&buf));
    }

    #[test]
    fn test_is_mxf_header_invalid_partition_kind() {
        let mut buf = minimal_mxf_header(0xFF);
        buf[13] = 0xFF; // unknown kind
        assert!(!MxfProber::is_mxf_header(&buf));
    }

    #[test]
    fn test_is_mxf_header_wrong_magic() {
        let mut buf = vec![0u8; 64];
        buf[0] = 0xDE;
        buf[1] = 0xAD;
        buf[2] = 0xBE;
        buf[3] = 0xEF;
        assert!(!MxfProber::is_mxf_header(&buf));
    }

    #[test]
    fn test_is_mxf_header_too_short() {
        let buf = [0x06u8, 0x0E, 0x2B, 0x34];
        assert!(!MxfProber::is_mxf_header(&buf));
    }

    // ── probe errors ─────────────────────────────────────────────────────────

    #[test]
    fn test_probe_empty_returns_truncated() {
        let result = MxfProber::probe(&[]);
        assert!(matches!(result, Err(MxfProbeError::TruncatedData)));
    }

    #[test]
    fn test_probe_too_short_returns_truncated() {
        let buf = vec![0u8; MIN_MXF_SIZE - 1];
        let result = MxfProber::probe(&buf);
        assert!(matches!(result, Err(MxfProbeError::TruncatedData)));
    }

    #[test]
    fn test_probe_not_mxf_returns_not_mxf() {
        let buf = vec![0xFFu8; 64];
        let result = MxfProber::probe(&buf);
        assert!(matches!(result, Err(MxfProbeError::NotMxf)));
    }

    #[test]
    fn test_probe_jpeg_magic_returns_not_mxf() {
        let mut buf = vec![0u8; 64];
        buf[0] = 0xFF;
        buf[1] = 0xD8;
        let result = MxfProber::probe(&buf);
        assert!(matches!(result, Err(MxfProbeError::NotMxf)));
    }

    // ── probe success ─────────────────────────────────────────────────────────

    #[test]
    fn test_probe_valid_header_partition_succeeds() {
        let buf = minimal_mxf_header(0x01);
        let result = MxfProber::probe(&buf);
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }

    #[test]
    fn test_probe_returns_unknown_op_when_no_op_label() {
        let buf = minimal_mxf_header(0x01);
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        assert_eq!(info.operational_pattern, "Unknown");
    }

    #[test]
    fn test_probe_detects_op1a() {
        let mut buf = minimal_mxf_header(0x01);
        buf.resize(128, 0);
        // Embed OP1a label at offset 32.
        buf[32..39].copy_from_slice(&OP_KEY_PREFIX);
        buf[39] = 0x01; // OP1a discriminator
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        assert_eq!(info.operational_pattern, "OP1a");
    }

    #[test]
    fn test_probe_detects_op3c() {
        let mut buf = minimal_mxf_header(0x01);
        buf.resize(128, 0);
        buf[40..47].copy_from_slice(&OP_KEY_PREFIX);
        buf[47] = 0x09; // OP3c
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        assert_eq!(info.operational_pattern, "OP3c");
    }

    #[test]
    fn test_probe_detects_essence_video_track() {
        let mut buf = minimal_mxf_header(0x01);
        buf.resize(200, 0);
        // Embed essence key prefix at offset 80 with Video essence type.
        buf[80..87].copy_from_slice(&ESSENCE_KEY_PREFIX);
        buf[80 + 12] = 0x01; // Video essence type byte
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        let has_video = info
            .essence_tracks
            .iter()
            .any(|t| t.track_type == MxfTrackType::Video);
        assert!(
            has_video,
            "expected a Video track, got {:?}",
            info.essence_tracks
        );
    }

    #[test]
    fn test_probe_detects_essence_audio_track() {
        let mut buf = minimal_mxf_header(0x01);
        buf.resize(200, 0);
        buf[80..87].copy_from_slice(&ESSENCE_KEY_PREFIX);
        buf[80 + 12] = 0x02; // Audio essence type byte
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        let has_audio = info
            .essence_tracks
            .iter()
            .any(|t| t.track_type == MxfTrackType::Audio);
        assert!(
            has_audio,
            "expected an Audio track, got {:?}",
            info.essence_tracks
        );
    }

    #[test]
    fn test_probe_no_essence_tracks_when_none_present() {
        let buf = minimal_mxf_header(0x01);
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        // The minimal buffer has no essence key prefix, so no tracks.
        assert!(
            info.essence_tracks.is_empty(),
            "expected no tracks, got {:?}",
            info.essence_tracks
        );
    }

    #[test]
    fn test_probe_duration_none_for_minimal_buffer() {
        let buf = minimal_mxf_header(0x01);
        let info = MxfProber::probe(&buf).expect("probe should succeed");
        // Minimal 64-byte buffer has no plausible duration field.
        assert!(info.duration_ms.is_none());
    }

    // ── display impls ─────────────────────────────────────────────────────────

    #[test]
    fn test_mxf_track_type_display() {
        assert_eq!(MxfTrackType::Video.to_string(), "Video");
        assert_eq!(MxfTrackType::Audio.to_string(), "Audio");
        assert_eq!(MxfTrackType::Data.to_string(), "Data");
    }

    #[test]
    fn test_mxf_probe_error_display() {
        assert!(MxfProbeError::NotMxf.to_string().contains("MXF"));
        assert!(MxfProbeError::TruncatedData
            .to_string()
            .contains("truncated"));
        let pe = MxfProbeError::ParseError("bad field".to_owned());
        assert!(pe.to_string().contains("bad field"));
    }
}
