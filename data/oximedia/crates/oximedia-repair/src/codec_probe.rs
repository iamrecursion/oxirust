//! Codec parameter identification from raw bitstreams.
//!
//! When file headers are damaged or missing, this module can scan raw byte
//! data for codec-specific patterns to identify the codec and extract
//! basic parameters (resolution, profile, level, etc.).
//!
//! Supported bitstream signatures:
//! - AV1 OBU (Open Bitstream Unit) headers
//! - VP9 superframe index / frame headers
//! - H.264 NAL unit patterns (SPS/PPS)
//! - H.265/HEVC NAL unit patterns (VPS/SPS/PPS)
//! - Opus identification header
//! - FLAC stream info block

use crate::Result;

/// Identified codec from a raw bitstream probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeCodec {
    /// AV1 video codec.
    Av1,
    /// VP9 video codec.
    Vp9,
    /// VP8 video codec.
    Vp8,
    /// H.264 / AVC video codec.
    H264,
    /// H.265 / HEVC video codec.
    H265,
    /// Opus audio codec.
    Opus,
    /// FLAC audio codec.
    Flac,
    /// Unknown codec.
    Unknown,
}

impl std::fmt::Display for ProbeCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Av1 => write!(f, "AV1"),
            Self::Vp9 => write!(f, "VP9"),
            Self::Vp8 => write!(f, "VP8"),
            Self::H264 => write!(f, "H.264/AVC"),
            Self::H265 => write!(f, "H.265/HEVC"),
            Self::Opus => write!(f, "Opus"),
            Self::Flac => write!(f, "FLAC"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Video parameters extracted from a bitstream probe.
#[derive(Debug, Clone, Default)]
pub struct VideoParams {
    /// Width in pixels (0 if unknown).
    pub width: u32,
    /// Height in pixels (0 if unknown).
    pub height: u32,
    /// Profile index (codec-specific).
    pub profile: Option<u8>,
    /// Level index (codec-specific).
    pub level: Option<u8>,
    /// Bit depth (8, 10, 12, etc.).
    pub bit_depth: Option<u8>,
    /// Chroma subsampling string (e.g. "4:2:0").
    pub chroma_subsampling: Option<String>,
}

/// Audio parameters extracted from a bitstream probe.
#[derive(Debug, Clone, Default)]
pub struct AudioParams {
    /// Sample rate in Hz (0 if unknown).
    pub sample_rate: u32,
    /// Number of channels (0 if unknown).
    pub channels: u8,
    /// Bits per sample (0 if unknown).
    pub bits_per_sample: Option<u16>,
}

/// Result of probing a raw bitstream for codec identity and parameters.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Identified codec.
    pub codec: ProbeCodec,
    /// Confidence score 0.0 .. 1.0.
    pub confidence: f64,
    /// Video parameters (if video codec detected).
    pub video: Option<VideoParams>,
    /// Audio parameters (if audio codec detected).
    pub audio: Option<AudioParams>,
    /// Byte offset of the first recognized structure.
    pub first_match_offset: u64,
    /// Number of codec-specific structures found.
    pub match_count: usize,
}

impl ProbeResult {
    fn unknown() -> Self {
        Self {
            codec: ProbeCodec::Unknown,
            confidence: 0.0,
            video: None,
            audio: None,
            first_match_offset: 0,
            match_count: 0,
        }
    }
}

/// Probe raw data for codec identification.
///
/// Tries each codec scanner in sequence and returns the one with the
/// highest confidence score. Returns `ProbeCodec::Unknown` with
/// `confidence = 0.0` if nothing was recognised.
pub fn probe_codec(data: &[u8]) -> Result<ProbeResult> {
    if data.is_empty() {
        return Ok(ProbeResult::unknown());
    }

    let candidates = [
        probe_av1(data),
        probe_vp9(data),
        probe_vp8(data),
        probe_h264(data),
        probe_h265(data),
        probe_opus(data),
        probe_flac(data),
    ];

    let best = candidates.into_iter().max_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(best.unwrap_or_else(ProbeResult::unknown))
}

// ---------------------------------------------------------------------------
// AV1 OBU probe
// ---------------------------------------------------------------------------

/// AV1 OBU type IDs.
const AV1_OBU_SEQUENCE_HEADER: u8 = 1;
const AV1_OBU_TEMPORAL_DELIMITER: u8 = 2;
const AV1_OBU_FRAME_HEADER: u8 = 3;
const AV1_OBU_FRAME: u8 = 6;

/// Scan for AV1 OBU (Open Bitstream Unit) patterns.
fn probe_av1(data: &[u8]) -> ProbeResult {
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut video = VideoParams::default();
    let mut i = 0;

    while i < data.len() {
        if let Some(obu) = parse_av1_obu_header(&data[i..]) {
            if matches!(
                obu.obu_type,
                AV1_OBU_SEQUENCE_HEADER
                    | AV1_OBU_TEMPORAL_DELIMITER
                    | AV1_OBU_FRAME_HEADER
                    | AV1_OBU_FRAME
            ) {
                if match_count == 0 {
                    first_offset = i as u64;
                }
                match_count += 1;

                // Try to parse sequence header for parameters
                if obu.obu_type == AV1_OBU_SEQUENCE_HEADER
                    && obu.payload_offset + 4 <= data.len() - i
                {
                    parse_av1_sequence_header(&data[i + obu.payload_offset..], &mut video);
                }

                // Skip past this OBU
                if obu.has_size && obu.size > 0 {
                    i += obu.payload_offset + obu.size;
                    continue;
                }
            }
        }
        i += 1;
    }

    let confidence = match match_count {
        0 => 0.0,
        1 => 0.3,
        2..=4 => 0.6,
        _ => 0.9,
    };

    ProbeResult {
        codec: ProbeCodec::Av1,
        confidence,
        video: if match_count > 0 { Some(video) } else { None },
        audio: None,
        first_match_offset: first_offset,
        match_count,
    }
}

/// Parsed AV1 OBU header fields.
struct Av1ObuHeader {
    obu_type: u8,
    has_size: bool,
    size: usize,
    payload_offset: usize,
}

/// Parse a single AV1 OBU header from the start of `data`.
fn parse_av1_obu_header(data: &[u8]) -> Option<Av1ObuHeader> {
    if data.is_empty() {
        return None;
    }

    let byte0 = data[0];
    // obu_forbidden_bit must be 0
    if byte0 & 0x80 != 0 {
        return None;
    }

    let obu_type = (byte0 >> 3) & 0x0F;
    let has_extension = (byte0 >> 2) & 0x01 != 0;
    let has_size = (byte0 >> 1) & 0x01 != 0;

    // Validate OBU type range (0 is reserved, 1-8 are defined, 9-14 reserved, 15 is padding)
    if obu_type == 0 || (obu_type > 8 && obu_type < 15) {
        return None;
    }

    let mut offset = 1;
    if has_extension {
        if offset >= data.len() {
            return None;
        }
        offset += 1; // skip extension byte
    }

    let size = if has_size {
        let (val, bytes_read) = read_leb128(&data[offset..])?;
        offset += bytes_read;
        val
    } else {
        0
    };

    Some(Av1ObuHeader {
        obu_type,
        has_size,
        size,
        payload_offset: offset,
    })
}

/// Read a LEB128 encoded unsigned integer. Returns (value, bytes_consumed).
fn read_leb128(data: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0usize;
    let mut shift = 0u32;

    for (i, &byte) in data.iter().enumerate().take(8) {
        value |= ((byte & 0x7F) as usize) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Parse AV1 sequence header for video parameters (best-effort).
fn parse_av1_sequence_header(data: &[u8], params: &mut VideoParams) {
    if data.len() < 4 {
        return;
    }

    // The sequence header is bit-packed. We do a simplified extraction.
    // seq_profile is bits 0-2 of the first byte after the OBU header.
    let seq_profile = (data[0] >> 5) & 0x07;
    params.profile = Some(seq_profile);

    // Bit depth inference from profile
    params.bit_depth = Some(match seq_profile {
        0 => 8,
        1 => 8, // can also be 10-bit but default assumption
        2 => 10,
        _ => 8,
    });

    params.chroma_subsampling = Some(match seq_profile {
        0 => "4:2:0".to_string(),
        1 => "4:4:4".to_string(),
        2 => "4:2:2".to_string(),
        _ => "4:2:0".to_string(),
    });
}

// ---------------------------------------------------------------------------
// VP9 superframe / frame header probe
// ---------------------------------------------------------------------------

/// Scan for VP9 frame headers and superframe indices.
fn probe_vp9(data: &[u8]) -> ProbeResult {
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut video = VideoParams::default();
    let mut i = 0;

    while i < data.len() {
        if let Some(frame) = parse_vp9_frame_header(&data[i..]) {
            if match_count == 0 {
                first_offset = i as u64;
                video.profile = Some(frame.profile);
                video.bit_depth = Some(frame.bit_depth);
                if frame.width > 0 && frame.height > 0 {
                    video.width = frame.width;
                    video.height = frame.height;
                }
            }
            match_count += 1;
            // VP9 frames don't carry their own length in the unframed format;
            // we move forward by 1 to avoid re-detecting the same header.
            i += 1;
        } else {
            i += 1;
        }
    }

    // Also check for superframe index at end of data
    if data.len() >= 3 {
        let marker = data[data.len() - 1];
        // VP9 superframe marker: upper 3 bits = 0b110
        if marker & 0xE0 == 0xC0 {
            match_count += 1;
        }
    }

    let confidence = match match_count {
        0 => 0.0,
        1 => 0.25,
        2..=3 => 0.55,
        _ => 0.85,
    };

    ProbeResult {
        codec: ProbeCodec::Vp9,
        confidence,
        video: if match_count > 0 { Some(video) } else { None },
        audio: None,
        first_match_offset: first_offset,
        match_count,
    }
}

struct Vp9FrameHeader {
    profile: u8,
    bit_depth: u8,
    width: u32,
    height: u32,
}

/// Parse VP9 uncompressed frame header (first 2-3 bytes).
fn parse_vp9_frame_header(data: &[u8]) -> Option<Vp9FrameHeader> {
    if data.len() < 3 {
        return None;
    }

    // VP9 frame header starts with a 2-bit frame marker (0b10)
    let byte0 = data[0];
    let frame_marker = (byte0 >> 6) & 0x03;
    if frame_marker != 0x02 {
        return None;
    }

    let profile_low = (byte0 >> 5) & 0x01;
    let profile_high = (byte0 >> 4) & 0x01;
    let profile = (profile_high << 1) | profile_low;

    // Show-existing-frame bit
    let show_existing = if profile == 3 {
        (byte0 >> 3) & 0x01
    } else {
        (byte0 >> 4) & 0x01
    };

    if show_existing != 0 {
        // Show-existing-frame: short header, still a valid VP9 frame
        return Some(Vp9FrameHeader {
            profile,
            bit_depth: if profile >= 2 { 10 } else { 8 },
            width: 0,
            height: 0,
        });
    }

    let bit_depth = if profile >= 2 { 10 } else { 8 };

    Some(Vp9FrameHeader {
        profile,
        bit_depth,
        width: 0,
        height: 0,
    })
}

// ---------------------------------------------------------------------------
// VP8 frame header probe
// ---------------------------------------------------------------------------

/// Scan for VP8 frame headers.
fn probe_vp8(data: &[u8]) -> ProbeResult {
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut video = VideoParams::default();
    let mut i = 0;

    while i + 10 <= data.len() {
        if let Some(frame) = parse_vp8_frame_header(&data[i..]) {
            if match_count == 0 {
                first_offset = i as u64;
                video.width = frame.width;
                video.height = frame.height;
                video.bit_depth = Some(8);
                video.profile = Some(frame.version);
            }
            match_count += 1;
            i += 3; // skip at least the tag bytes
        } else {
            i += 1;
        }
    }

    let confidence = match match_count {
        0 => 0.0,
        1 => 0.2,
        2..=3 => 0.5,
        _ => 0.8,
    };

    ProbeResult {
        codec: ProbeCodec::Vp8,
        confidence,
        video: if match_count > 0 { Some(video) } else { None },
        audio: None,
        first_match_offset: first_offset,
        match_count,
    }
}

struct Vp8FrameHeader {
    version: u8,
    width: u32,
    height: u32,
}

/// Parse VP8 frame header. Key frames carry a 3-byte tag plus 7-byte start code.
fn parse_vp8_frame_header(data: &[u8]) -> Option<Vp8FrameHeader> {
    if data.len() < 10 {
        return None;
    }

    // 3-byte frame tag (little-endian 24-bit)
    let tag = u32::from_le_bytes([data[0], data[1], data[2], 0]);
    let is_keyframe = (tag & 0x01) == 0;
    let version = ((tag >> 1) & 0x07) as u8;

    if !is_keyframe {
        return None; // only detect keyframes for reliable probing
    }

    // VP8 keyframe start code: 0x9D 0x01 0x2A
    if data[3] != 0x9D || data[4] != 0x01 || data[5] != 0x2A {
        return None;
    }

    let width = u16::from_le_bytes([data[6], data[7]]) as u32 & 0x3FFF;
    let height = u16::from_le_bytes([data[8], data[9]]) as u32 & 0x3FFF;

    // Sanity-check: reasonable resolutions
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return None;
    }

    Some(Vp8FrameHeader {
        version,
        width,
        height,
    })
}

// ---------------------------------------------------------------------------
// H.264 NAL probe
// ---------------------------------------------------------------------------

/// Scan for H.264 NAL start codes and extract SPS parameters.
fn probe_h264(data: &[u8]) -> ProbeResult {
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut video = VideoParams::default();
    let mut i = 0;

    while i + 4 <= data.len() {
        let is_4byte = i + 4 <= data.len()
            && data[i] == 0x00
            && data[i + 1] == 0x00
            && data[i + 2] == 0x00
            && data[i + 3] == 0x01;
        let is_3byte = data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01;

        if is_4byte || is_3byte {
            let prefix_len = if is_4byte { 4 } else { 3 };
            let nal_offset = i + prefix_len;
            if nal_offset < data.len() {
                let nal_type = data[nal_offset] & 0x1F;
                // Valid H.264 NAL types: 1-12, 19, 20, 24-31
                if nal_type >= 1 && nal_type <= 12 || nal_type == 19 || nal_type == 20 {
                    if match_count == 0 {
                        first_offset = i as u64;
                    }
                    match_count += 1;

                    // SPS (type 7) carries profile, level, resolution
                    if nal_type == 7 && nal_offset + 4 <= data.len() {
                        parse_h264_sps(&data[nal_offset + 1..], &mut video);
                    }
                }
            }
            i += prefix_len;
        } else {
            i += 1;
        }
    }

    let confidence = match match_count {
        0 => 0.0,
        1 => 0.3,
        2..=5 => 0.6,
        _ => 0.9,
    };

    ProbeResult {
        codec: ProbeCodec::H264,
        confidence,
        video: if match_count > 0 { Some(video) } else { None },
        audio: None,
        first_match_offset: first_offset,
        match_count,
    }
}

/// Parse H.264 SPS for profile/level (simplified, byte-aligned fields only).
fn parse_h264_sps(data: &[u8], params: &mut VideoParams) {
    if data.len() < 3 {
        return;
    }
    params.profile = Some(data[0]); // profile_idc
    params.level = Some(data[2]); // level_idc
    params.bit_depth = Some(8);
    params.chroma_subsampling = Some("4:2:0".to_string());
}

// ---------------------------------------------------------------------------
// H.265 NAL probe
// ---------------------------------------------------------------------------

/// Scan for H.265/HEVC NAL start codes.
fn probe_h265(data: &[u8]) -> ProbeResult {
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut video = VideoParams::default();
    let mut i = 0;

    while i + 5 <= data.len() {
        let is_4byte =
            data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x00 && data[i + 3] == 0x01;
        let is_3byte = data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01;

        if is_4byte || is_3byte {
            let prefix_len = if is_4byte { 4 } else { 3 };
            let nal_offset = i + prefix_len;
            if nal_offset + 1 < data.len() {
                let nal_type = (data[nal_offset] >> 1) & 0x3F;
                // HEVC NAL types: 0-47 are valid
                if nal_type <= 47 {
                    // Distinguish from H.264: HEVC NAL header is 2 bytes
                    // and the forbidden_zero_bit (bit 7 of byte 0) must be 0
                    let forbidden = data[nal_offset] & 0x80;
                    if forbidden == 0 {
                        if match_count == 0 {
                            first_offset = i as u64;
                        }
                        match_count += 1;

                        // VPS (32), SPS (33), PPS (34) carry parameters
                        if nal_type == 33 && nal_offset + 4 < data.len() {
                            // SPS: extract profile from general_profile_idc
                            params_from_hevc_sps(&data[nal_offset + 2..], &mut video);
                        }
                    }
                }
            }
            i += prefix_len;
        } else {
            i += 1;
        }
    }

    // H.265 and H.264 both use Annex-B start codes; if we also have H.264
    // matches we need to disambiguate. H.265 NAL header is 2 bytes (with
    // nuh_layer_id and nuh_temporal_id_plus1). A crude heuristic: if the
    // second byte after the start code has bit 0 set (temporal_id_plus1 >= 1)
    // and bit 6 of byte 0 is 0, it looks more like HEVC.
    let confidence = match match_count {
        0 => 0.0,
        1 => 0.25,
        2..=5 => 0.55,
        _ => 0.85,
    };

    ProbeResult {
        codec: ProbeCodec::H265,
        confidence,
        video: if match_count > 0 { Some(video) } else { None },
        audio: None,
        first_match_offset: first_offset,
        match_count,
    }
}

/// Extract basic HEVC SPS parameters.
fn params_from_hevc_sps(data: &[u8], params: &mut VideoParams) {
    if data.len() < 2 {
        return;
    }
    // Very simplified: general_profile_idc is in the profile tier level structure
    // which follows the SPS NAL header. Real parsing requires bit-level reading.
    params.profile = Some(data[0] & 0x1F);
    params.bit_depth = Some(8);
    params.chroma_subsampling = Some("4:2:0".to_string());
}

// ---------------------------------------------------------------------------
// Opus probe
// ---------------------------------------------------------------------------

/// Scan for Opus identification header ("OpusHead").
fn probe_opus(data: &[u8]) -> ProbeResult {
    let magic = b"OpusHead";
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut audio = AudioParams::default();

    for i in 0..data.len().saturating_sub(magic.len()) {
        if &data[i..i + magic.len()] == magic {
            if match_count == 0 {
                first_offset = i as u64;
            }
            match_count += 1;

            // Parse Opus ID header (19 bytes minimum)
            if i + 19 <= data.len() {
                let version = data[i + 8];
                if version <= 15 {
                    audio.channels = data[i + 9];
                    audio.sample_rate = u32::from_le_bytes([
                        data[i + 12],
                        data[i + 13],
                        data[i + 14],
                        data[i + 15],
                    ]);
                }
            }
        }
    }

    let confidence = if match_count > 0 { 0.95 } else { 0.0 };

    ProbeResult {
        codec: ProbeCodec::Opus,
        confidence,
        video: None,
        audio: if match_count > 0 { Some(audio) } else { None },
        first_match_offset: first_offset,
        match_count,
    }
}

// ---------------------------------------------------------------------------
// FLAC probe
// ---------------------------------------------------------------------------

/// Scan for FLAC stream marker ("fLaC") and STREAMINFO block.
fn probe_flac(data: &[u8]) -> ProbeResult {
    let magic = b"fLaC";
    let mut match_count = 0usize;
    let mut first_offset = 0u64;
    let mut audio = AudioParams::default();

    for i in 0..data.len().saturating_sub(magic.len()) {
        if &data[i..i + magic.len()] == magic {
            if match_count == 0 {
                first_offset = i as u64;
            }
            match_count += 1;

            // STREAMINFO metadata block follows the magic (4 bytes header + 34 bytes data)
            if i + 4 + 4 + 18 <= data.len() {
                let block_header = data[i + 4];
                let block_type = block_header & 0x7F;
                if block_type == 0 {
                    // STREAMINFO
                    let info = &data[i + 8..];
                    if info.len() >= 18 {
                        // Sample rate: 20 bits at offset 10
                        let sr = ((info[10] as u32) << 12)
                            | ((info[11] as u32) << 4)
                            | ((info[12] as u32) >> 4);
                        audio.sample_rate = sr;

                        // Channels: 3 bits at offset 12 bits 3-1
                        let channels = ((info[12] >> 1) & 0x07) + 1;
                        audio.channels = channels;

                        // Bits per sample: 5 bits spanning bytes 12-13
                        let bps = ((info[12] & 0x01) as u16) << 4 | ((info[13] >> 4) as u16);
                        audio.bits_per_sample = Some(bps + 1);
                    }
                }
            }
        }
    }

    let confidence = if match_count > 0 { 0.95 } else { 0.0 };

    ProbeResult {
        codec: ProbeCodec::Flac,
        confidence,
        video: None,
        audio: if match_count > 0 { Some(audio) } else { None },
        first_match_offset: first_offset,
        match_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_empty_data() {
        let result = probe_codec(&[]).expect("probe should succeed");
        assert_eq!(result.codec, ProbeCodec::Unknown);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_probe_random_data() {
        // Use 0x70 which has OBU type 14 (reserved, rejected by AV1 probe),
        // no start codes, no MPEG-TS sync, no ADTS sync, no codec signatures.
        let data = vec![0x70; 256];
        let result = probe_codec(&data).expect("probe should succeed");
        // Random data should have very low confidence for any codec
        assert!(
            result.confidence < 0.5,
            "codec={:?} confidence={} matches={}",
            result.codec,
            result.confidence,
            result.match_count
        );
    }

    // -- AV1 tests --

    #[test]
    fn test_av1_obu_temporal_delimiter() {
        // OBU type 2 (temporal delimiter), has_size=1, size=0
        // Build 6 consecutive OBUs so we get 6 matches (>= 5 for 0.9 confidence)
        let data = vec![
            0x12, 0x00, // OBU header: type=2, has_size=1; leb128 size=0
            0x12, 0x00, 0x12, 0x00, 0x12, 0x00, 0x12, 0x00, 0x12, 0x00,
        ];
        let result = probe_av1(&data);
        assert!(result.match_count >= 5);
        assert!(result.confidence >= 0.9);
    }

    #[test]
    fn test_av1_obu_forbidden_bit_set() {
        // Forbidden bit set => should not match
        let data = vec![0x92, 0x00];
        let result = probe_av1(&data);
        assert_eq!(result.match_count, 0);
    }

    #[test]
    fn test_av1_sequence_header_profile() {
        // OBU type 1 (sequence header), has_size=1, size=4
        // Then 4 bytes of payload with profile=1 (bits 5-7 of first byte)
        let data = vec![
            0x0A, 0x04, // OBU header: type=1, has_size=1; leb128 size=4
            0x20, 0x00, 0x00, 0x00, // seq header payload, profile=1 (0x20 >> 5 = 1)
        ];
        let result = probe_av1(&data);
        assert_eq!(result.match_count, 1);
        if let Some(ref v) = result.video {
            assert_eq!(v.profile, Some(1));
        }
    }

    // -- VP9 tests --

    #[test]
    fn test_vp9_frame_marker() {
        // VP9 frame marker is 0b10 in bits 7-6, i.e. byte starts with 0x80-0xBF
        // Profile 0, not show-existing: 0x80
        let data = vec![0x80, 0x00, 0x00, 0x00, 0x00];
        let result = probe_vp9(&data);
        assert!(result.match_count >= 1);
    }

    #[test]
    fn test_vp9_superframe_marker() {
        // VP9 superframe marker: last byte has 0b110xxxxx pattern
        let mut data = vec![0x00; 100];
        data[99] = 0xC0; // superframe marker
        let result = probe_vp9(&data);
        assert!(result.match_count >= 1);
    }

    // -- VP8 tests --

    #[test]
    fn test_vp8_keyframe_detection() {
        // VP8 keyframe: tag bit 0 = 0, then start code 0x9D 0x01 0x2A
        // width=320 (0x0140), height=240 (0x00F0)
        let mut data = vec![0u8; 10];
        data[0] = 0x00; // keyframe tag (bit 0 = 0)
        data[1] = 0x00;
        data[2] = 0x00;
        data[3] = 0x9D;
        data[4] = 0x01;
        data[5] = 0x2A;
        data[6] = 0x40; // width low byte (320)
        data[7] = 0x01; // width high byte
        data[8] = 0xF0; // height low byte (240)
        data[9] = 0x00; // height high byte
        let result = probe_vp8(&data);
        assert!(result.match_count >= 1);
        if let Some(ref v) = result.video {
            assert_eq!(v.width, 320);
            assert_eq!(v.height, 240);
        }
    }

    #[test]
    fn test_vp8_invalid_start_code() {
        let data = vec![0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC, 0x40, 0x01, 0xF0, 0x00];
        let result = probe_vp8(&data);
        assert_eq!(result.match_count, 0);
    }

    // -- H.264 tests --

    #[test]
    fn test_h264_nal_detection() {
        // Two H.264 NAL units: IDR slice (type 5) and SPS (type 7)
        let data = vec![
            0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, // IDR NAL
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, // SPS NAL (profile=66, level=30)
        ];
        let result = probe_h264(&data);
        assert!(result.match_count >= 2);
        assert!(result.confidence >= 0.6);
        if let Some(ref v) = result.video {
            assert_eq!(v.profile, Some(0x42)); // Baseline profile
            assert_eq!(v.level, Some(0x1E)); // Level 3.0
        }
    }

    #[test]
    fn test_h264_3byte_start_code() {
        let data = vec![
            0x00, 0x00, 0x01, 0x65, 0xAA, // IDR NAL with 3-byte start code
            0x00, 0x00, 0x01, 0x41, 0xBB, // Non-IDR slice
        ];
        let result = probe_h264(&data);
        assert!(result.match_count >= 2);
    }

    // -- H.265 tests --

    #[test]
    fn test_h265_nal_detection() {
        // HEVC NAL: 2-byte header, type in bits 1-6 of byte 0
        // SPS (type 33 = 0x21): byte0 = (33 << 1) = 0x42, byte1 = 0x01
        let data = vec![
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x10, 0x00, // SPS NAL
            0x00, 0x00, 0x00, 0x01, 0x02, 0x01, 0x00, 0x00, // IDR_W_RADL (type 1)
        ];
        let result = probe_h265(&data);
        assert!(result.match_count >= 2);
    }

    // -- Opus tests --

    #[test]
    fn test_opus_identification_header() {
        let mut data = vec![0u8; 32];
        data[0..8].copy_from_slice(b"OpusHead");
        data[8] = 1; // version
        data[9] = 2; // channels
                     // sample rate 48000 = 0x0000BB80 LE
        data[12] = 0x80;
        data[13] = 0xBB;
        data[14] = 0x00;
        data[15] = 0x00;

        let result = probe_opus(&data);
        assert_eq!(result.codec, ProbeCodec::Opus);
        assert!(result.confidence >= 0.9);
        if let Some(ref a) = result.audio {
            assert_eq!(a.channels, 2);
            assert_eq!(a.sample_rate, 48000);
        }
    }

    #[test]
    fn test_opus_no_match() {
        let data = b"RandomDataHere";
        let result = probe_opus(data);
        assert_eq!(result.match_count, 0);
    }

    // -- FLAC tests --

    #[test]
    fn test_flac_stream_marker() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"fLaC");
        data[4] = 0x00; // block type 0 = STREAMINFO, not last
        data[5] = 0x00;
        data[6] = 0x00;
        data[7] = 0x22; // block length = 34

        // STREAMINFO at offset 8, 34 bytes
        // info[10..12] carry sample rate as 20 bits:
        //   sr = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
        // For 44100 = 0xAC44:
        //   info[10] = 0x0A, info[11] = 0xC4, info[12] upper nibble = 0x4
        // info[12] also carries channel bits: lower nibble = (ch-1)<<1 | (bps-1)>>4
        //   channels=2 => ch-1=1=0b001, bps=16 => bps-1=15=0b01111, bit4=0
        //   info[12] = 0x40 | (1 << 1) | 0 = 0x42
        // info[13] = (bps-1 & 0xF) << 4 = 0xF0
        data[18] = 0x0A; // info[10]
        data[19] = 0xC4; // info[11]
        data[20] = 0x42; // info[12]: sr low nibble + channels + bps high bit
        data[21] = 0xF0; // info[13]: bps low 4 bits << 4

        let result = probe_flac(&data);
        assert_eq!(result.codec, ProbeCodec::Flac);
        assert!(result.confidence >= 0.9);
        if let Some(ref a) = result.audio {
            assert_eq!(a.sample_rate, 44100);
            assert_eq!(a.channels, 2);
        }
    }

    #[test]
    fn test_flac_no_match() {
        let data = b"NotFLAC";
        let result = probe_flac(data);
        assert_eq!(result.match_count, 0);
    }

    // -- LEB128 tests --

    #[test]
    fn test_read_leb128_single_byte() {
        let data = [0x05];
        let (value, bytes) = read_leb128(&data).expect("should parse");
        assert_eq!(value, 5);
        assert_eq!(bytes, 1);
    }

    #[test]
    fn test_read_leb128_two_bytes() {
        // 300 = 0b100101100 => 0xAC 0x02
        let data = [0xAC, 0x02];
        let (value, bytes) = read_leb128(&data).expect("should parse");
        assert_eq!(value, 300);
        assert_eq!(bytes, 2);
    }

    #[test]
    fn test_read_leb128_empty() {
        let data: [u8; 0] = [];
        assert!(read_leb128(&data).is_none());
    }

    // -- ProbeCodec display --

    #[test]
    fn test_probe_codec_display() {
        assert_eq!(format!("{}", ProbeCodec::Av1), "AV1");
        assert_eq!(format!("{}", ProbeCodec::Vp9), "VP9");
        assert_eq!(format!("{}", ProbeCodec::H264), "H.264/AVC");
        assert_eq!(format!("{}", ProbeCodec::Opus), "Opus");
        assert_eq!(format!("{}", ProbeCodec::Flac), "FLAC");
        assert_eq!(format!("{}", ProbeCodec::Unknown), "Unknown");
    }

    // -- Integration-style probe tests --

    #[test]
    fn test_probe_picks_opus_over_noise() {
        let mut data = vec![0xAB; 256];
        data[50..58].copy_from_slice(b"OpusHead");
        data[58] = 1; // version
        data[59] = 2; // channels
        data[62] = 0x80;
        data[63] = 0xBB;

        let result = probe_codec(&data).expect("probe should succeed");
        assert_eq!(result.codec, ProbeCodec::Opus);
    }

    #[test]
    fn test_probe_picks_flac_over_noise() {
        let mut data = vec![0x00; 128];
        data[10..14].copy_from_slice(b"fLaC");
        data[14] = 0x00; // STREAMINFO block type

        let result = probe_codec(&data).expect("probe should succeed");
        assert_eq!(result.codec, ProbeCodec::Flac);
    }
}
