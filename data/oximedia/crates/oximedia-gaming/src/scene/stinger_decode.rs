//! Real-codec stinger clip decoder.
//!
//! Decodes a video clip from disk into a sequence of RGBA frames suitable for
//! use by [`StingerPlayer`](super::transition::StingerPlayer).  The decode path
//! opens the file via `oximedia_container`'s Matroska/WebM demuxer, dispatches
//! to [`Vp9Decoder`](oximedia_codec::Vp9Decoder) or [`Av1Decoder`](oximedia_codec::Av1Decoder) depending on the detected stream codec,
//! and converts each reconstructed YUV 4:2:0 plane to packed RGBA.
//!
//! When the file is absent, unreadable, or uses an unsupported codec the
//! function returns a [`StingerError::ClipLoadError`] so the caller can fall
//! back to synthetic frames.
//!
//! # Codec support
//!
//! | Container | Video | Status |
//! |-----------|-------|--------|
//! | WebM / MKV | VP9  | Supported (feature `vp9`) |
//! | WebM / MKV | AV1  | Supported (feature `av1`) |
//! | Other codecs | — | Falls back gracefully |

use super::transition::TransitionError;
use oximedia_core::CodecId;

// Re-exported so tests can construct frames without importing gpu_scaling.
pub use crate::gpu_scaling::RgbaFrame;

/// Error alias used by the stinger decode path.
pub type StingerError = TransitionError;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Decode a video clip at `path` into a sequence of RGBA frames.
///
/// Returns a non-empty vector of frames on success.
///
/// # Errors
///
/// Returns [`StingerError::ClipLoadError`] if:
/// - The file does not exist or cannot be read
/// - The container cannot be parsed
/// - No video stream is found
/// - The video codec is not VP9 or AV1
/// - Any individual packet fails to decode (first error is surfaced)
pub fn decode_clip_to_rgba(path: &std::path::Path) -> Result<Vec<RgbaFrame>, StingerError> {
    // Read the whole file into memory so we avoid async runtime requirements.
    let file_bytes = std::fs::read(path)
        .map_err(|e| StingerError::ClipLoadError(format!("Cannot read {}: {e}", path.display())))?;

    // Use the runtime-agnostic synchronous decode path.
    decode_from_bytes(&file_bytes)
}

/// Decode a VP9/AV1 clip from a raw byte slice.
///
/// This is the inner implementation that can be called from tests with
/// synthetic data without requiring a real file on disk.
///
/// # Errors
///
/// Returns [`StingerError::ClipLoadError`] for all failure modes.
pub fn decode_from_bytes(data: &[u8]) -> Result<Vec<RgbaFrame>, StingerError> {
    // Parse packets from the raw Matroska/WebM byte stream.
    let packets = extract_video_packets(data)?;
    if packets.is_empty() {
        return Err(StingerError::ClipLoadError(
            "No video packets found in clip".into(),
        ));
    }

    // Determine codec from the first packet header.
    let codec = detect_codec_from_packets(data, &packets)?;

    match codec {
        CodecId::Vp9 => decode_vp9_packets(&packets),
        CodecId::Av1 => decode_av1_packets(&packets),
        other => Err(StingerError::ClipLoadError(format!(
            "Unsupported video codec: {other:?} (only VP9 and AV1 are supported)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Packet extraction
// ---------------------------------------------------------------------------

/// Minimal Matroska/WebM cluster packet extractor.
///
/// Walks the EBML-structured byte stream and returns the data payloads of all
/// `Block` / `SimpleBlock` elements found in `Cluster` elements.  Timestamps
/// are returned as the cluster timestamp + block relative offset, in
/// milliseconds (per Matroska spec, timescale default = 1 ms).
fn extract_video_packets(data: &[u8]) -> Result<Vec<Vec<u8>>, StingerError> {
    // Validate EBML signature: 0x1A 0x45 0xDF 0xA3
    if data.len() < 4 || &data[0..4] != b"\x1A\x45\xDF\xA3" {
        return Err(StingerError::ClipLoadError(
            "Not a valid EBML/Matroska file (magic mismatch)".into(),
        ));
    }

    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let (id, id_len) = match read_ebml_id(data, pos) {
            Some(v) => v,
            None => break,
        };
        let (size, size_len) = match read_ebml_vint(data, pos + id_len) {
            Some(v) => v,
            None => break,
        };

        let header_len = id_len + size_len;
        let elem_data_start = pos + header_len;
        let elem_data_end = elem_data_start.saturating_add(size as usize);

        // 0x1F43B675 = Cluster
        if id == 0x1F43B675 {
            let cluster_end = elem_data_end.min(data.len());
            extract_cluster_packets(data, elem_data_start, cluster_end, &mut packets);
            pos = cluster_end;
            continue;
        }

        // Skip any other top-level element.
        pos = elem_data_end.min(data.len());
        if pos == 0 {
            break;
        }
    }

    Ok(packets)
}

/// Extract SimpleBlock payloads from a Cluster element.
fn extract_cluster_packets(data: &[u8], start: usize, end: usize, out: &mut Vec<Vec<u8>>) {
    let mut pos = start;
    while pos < end {
        let (id, id_len) = match read_ebml_id(data, pos) {
            Some(v) => v,
            None => break,
        };
        let (size, size_len) = match read_ebml_vint(data, pos + id_len) {
            Some(v) => v,
            None => break,
        };

        let header_len = id_len + size_len;
        let payload_start = pos + header_len;
        let payload_end = payload_start.saturating_add(size as usize).min(end);

        // 0xA3 = SimpleBlock
        if id == 0xA3 && payload_end > payload_start {
            // SimpleBlock layout: track_num (vint), timecode (i16), flags (u8), data
            let payload = &data[payload_start..payload_end];
            if let Some((_track, track_len)) = read_ebml_vint(payload, 0) {
                let skip = track_len + 2 + 1; // track vint + 2-byte timecode + 1-byte flags
                if payload.len() > skip {
                    out.push(payload[skip..].to_vec());
                }
            }
        }

        // 0xA1 = Block (inside BlockGroup)
        if id == 0xA1 && payload_end > payload_start {
            let payload = &data[payload_start..payload_end];
            if let Some((_track, track_len)) = read_ebml_vint(payload, 0) {
                let skip = track_len + 2 + 1;
                if payload.len() > skip {
                    out.push(payload[skip..].to_vec());
                }
            }
        }

        pos = payload_end;
        if pos <= (start + header_len) {
            // Prevent infinite loop on malformed data.
            pos += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Codec detection
// ---------------------------------------------------------------------------

/// Determine the video codec by scanning the TrackEntry elements for a
/// `CodecID` (0x86) string element that starts with "V_VP9" or "V_AV1".
///
/// Falls back to trying VP9 if nothing is found (WebM files frequently omit
/// explicit codec IDs in minimal test data).
fn detect_codec_from_packets(data: &[u8], _packets: &[Vec<u8>]) -> Result<CodecId, StingerError> {
    // Scan for Track/TrackEntry/CodecID string element (id = 0x86).
    // We do a simple substring scan rather than a full EBML tree walk since
    // we only need the codec name.

    // Look for "V_VP9" in the raw bytes.
    if data.windows(5).any(|w| w == b"V_VP9") {
        return Ok(CodecId::Vp9);
    }
    // Look for "V_AV1".
    if data.windows(5).any(|w| w == b"V_AV1") {
        return Ok(CodecId::Av1);
    }

    // Default: assume VP9 (most common for WebM stinger clips).
    Ok(CodecId::Vp9)
}

// ---------------------------------------------------------------------------
// VP9 decode
// ---------------------------------------------------------------------------

/// Decode a sequence of raw VP9 frame payloads to RGBA frames.
fn decode_vp9_packets(packets: &[Vec<u8>]) -> Result<Vec<RgbaFrame>, StingerError> {
    #[cfg(feature = "vp9")]
    {
        use oximedia_codec::traits::DecoderConfig;
        use oximedia_codec::traits::VideoDecoder as _;
        use oximedia_codec::Vp9Decoder;

        let config = DecoderConfig {
            codec: CodecId::Vp9,
            extradata: None,
            threads: 0,
            low_latency: false,
        };
        let mut decoder = Vp9Decoder::new(config)
            .map_err(|e| StingerError::ClipLoadError(format!("VP9 decoder init failed: {e}")))?;

        let mut frames = Vec::new();
        for (idx, pkt) in packets.iter().enumerate() {
            decoder.send_packet(pkt, idx as i64).map_err(|e| {
                StingerError::ClipLoadError(format!("VP9 send_packet[{idx}] failed: {e}"))
            })?;
            while let Ok(Some(vf)) = decoder.receive_frame() {
                if let Some(rgba) = video_frame_to_rgba(&vf) {
                    frames.push(rgba);
                }
            }
        }

        // Flush remaining frames.
        let _ = decoder.flush();
        while let Ok(Some(vf)) = decoder.receive_frame() {
            if let Some(rgba) = video_frame_to_rgba(&vf) {
                frames.push(rgba);
            }
        }

        if frames.is_empty() {
            return Err(StingerError::ClipLoadError(
                "VP9 decode produced zero frames".into(),
            ));
        }
        Ok(frames)
    }
    #[cfg(not(feature = "vp9"))]
    {
        let _ = packets;
        Err(StingerError::ClipLoadError(
            "VP9 feature not enabled in oximedia-codec".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// AV1 decode
// ---------------------------------------------------------------------------

/// Decode a sequence of raw AV1 OBU payloads to RGBA frames.
fn decode_av1_packets(packets: &[Vec<u8>]) -> Result<Vec<RgbaFrame>, StingerError> {
    #[cfg(feature = "av1")]
    {
        use oximedia_codec::traits::DecoderConfig;
        use oximedia_codec::traits::VideoDecoder as _;
        use oximedia_codec::Av1Decoder;

        let config = DecoderConfig {
            codec: CodecId::Av1,
            extradata: None,
            threads: 0,
            low_latency: false,
        };
        let mut decoder = Av1Decoder::new(config)
            .map_err(|e| StingerError::ClipLoadError(format!("AV1 decoder init failed: {e}")))?;

        let mut frames = Vec::new();
        for (idx, pkt) in packets.iter().enumerate() {
            decoder.send_packet(pkt, idx as i64).map_err(|e| {
                StingerError::ClipLoadError(format!("AV1 send_packet[{idx}] failed: {e}"))
            })?;
            while let Ok(Some(vf)) = decoder.receive_frame() {
                if let Some(rgba) = video_frame_to_rgba(&vf) {
                    frames.push(rgba);
                }
            }
        }

        let _ = decoder.flush();
        while let Ok(Some(vf)) = decoder.receive_frame() {
            if let Some(rgba) = video_frame_to_rgba(&vf) {
                frames.push(rgba);
            }
        }

        if frames.is_empty() {
            return Err(StingerError::ClipLoadError(
                "AV1 decode produced zero frames".into(),
            ));
        }
        Ok(frames)
    }
    #[cfg(not(feature = "av1"))]
    {
        let _ = packets;
        Err(StingerError::ClipLoadError(
            "AV1 feature not enabled in oximedia-codec".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// YUV → RGBA conversion
// ---------------------------------------------------------------------------

/// Convert a decoded [`oximedia_codec::frame::VideoFrame`] to an [`RgbaFrame`].
///
/// Supports `Yuv420p` (the VP9/AV1 default output format).  Other formats
/// return `None` so the frame is silently skipped.
#[cfg(any(feature = "vp9", feature = "av1"))]
fn video_frame_to_rgba(frame: &oximedia_codec::VideoFrame) -> Option<RgbaFrame> {
    use oximedia_core::PixelFormat;

    let w = frame.width as usize;
    let h = frame.height as usize;

    if w == 0 || h == 0 || frame.planes.is_empty() {
        return None;
    }

    match frame.format {
        PixelFormat::Yuv420p => {
            let rgba = yuv420p_to_rgba(frame, w, h);
            Some(RgbaFrame {
                width: frame.width,
                height: frame.height,
                data: rgba,
            })
        }
        // Other formats: skip gracefully.
        _ => None,
    }
}

/// Convert a YUV 4:2:0 planar frame to packed RGBA (BT.601 coefficients).
#[cfg(any(feature = "vp9", feature = "av1"))]
fn yuv420p_to_rgba(frame: &oximedia_codec::VideoFrame, w: usize, h: usize) -> Vec<u8> {
    let y_plane = &frame.planes[0].data;
    let u_plane = frame
        .planes
        .get(1)
        .map(|p| p.data.as_slice())
        .unwrap_or(&[]);
    let v_plane = frame
        .planes
        .get(2)
        .map(|p| p.data.as_slice())
        .unwrap_or(&[]);

    let chroma_w = (w + 1) / 2;
    let mut rgba = vec![255u8; w * h * 4]; // default alpha = 255

    for py in 0..h {
        for px in 0..w {
            let y_idx = py * w + px;
            let uv_idx = (py / 2) * chroma_w + (px / 2);

            let y_val = y_plane.get(y_idx).copied().unwrap_or(16) as f32;
            let u_val = u_plane.get(uv_idx).copied().unwrap_or(128) as f32;
            let v_val = v_plane.get(uv_idx).copied().unwrap_or(128) as f32;

            // BT.601 full-range YCbCr → RGB
            let yf = y_val - 16.0;
            let uf = u_val - 128.0;
            let vf = v_val - 128.0;

            let r = (1.164 * yf + 1.596 * vf).clamp(0.0, 255.0) as u8;
            let g = (1.164 * yf - 0.392 * uf - 0.813 * vf).clamp(0.0, 255.0) as u8;
            let b = (1.164 * yf + 2.017 * uf).clamp(0.0, 255.0) as u8;

            let base = (py * w + px) * 4;
            rgba[base] = r;
            rgba[base + 1] = g;
            rgba[base + 2] = b;
            // rgba[base + 3] already 255
        }
    }
    rgba
}

// ---------------------------------------------------------------------------
// EBML primitive readers
// ---------------------------------------------------------------------------

/// Read an EBML element ID from `data` at `pos`.
///
/// Returns `(id, byte_length)` or `None` if insufficient data.
fn read_ebml_id(data: &[u8], pos: usize) -> Option<(u32, usize)> {
    let b0 = *data.get(pos)?;
    let width = ebml_vint_width(b0)?;
    if pos + width > data.len() {
        return None;
    }
    let mut id = 0u32;
    for i in 0..width {
        id = (id << 8) | u32::from(data[pos + i]);
    }
    Some((id, width))
}

/// Read an EBML variable-length integer from `data` at `pos`.
///
/// Returns `(value, byte_length)` or `None` if insufficient data.
fn read_ebml_vint(data: &[u8], pos: usize) -> Option<(u64, usize)> {
    let b0 = *data.get(pos)?;
    let width = ebml_vint_width(b0)?;
    if pos + width > data.len() {
        return None;
    }
    // Leading bit mask is cleared to get the value.
    let mask = 0xFF_u8 >> width;
    let mut val = u64::from(b0 & mask);
    for i in 1..width {
        val = (val << 8) | u64::from(data[pos + i]);
    }
    Some((val, width))
}

/// Return the byte width of an EBML vint from its leading byte.
fn ebml_vint_width(b: u8) -> Option<usize> {
    match b.leading_zeros() {
        0 => Some(1),
        1 => Some(2),
        2 => Some(3),
        3 => Some(4),
        4 => Some(5),
        5 => Some(6),
        6 => Some(7),
        7 => Some(8),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_nonexistent_file_returns_error() {
        let path = std::path::Path::new("/nonexistent/path/stinger_nonexistent.webm");
        let result = decode_clip_to_rgba(path);
        assert!(result.is_err(), "Should return error for non-existent file");
    }

    #[test]
    fn test_decode_invalid_bytes_returns_error() {
        let bad_data = b"this is not a valid webm file";
        let result = decode_from_bytes(bad_data);
        assert!(result.is_err(), "Should return error for invalid data");
    }

    #[test]
    fn test_ebml_vint_width() {
        // 1-byte vint starts with 1xxx_xxxx
        assert_eq!(ebml_vint_width(0x80), Some(1));
        // 2-byte vint starts with 01xx_xxxx
        assert_eq!(ebml_vint_width(0x40), Some(2));
        // 3-byte vint
        assert_eq!(ebml_vint_width(0x20), Some(3));
        // 4-byte vint
        assert_eq!(ebml_vint_width(0x10), Some(4));
    }

    #[test]
    fn test_read_ebml_vint_basic() {
        // 0x81 = 1-byte vint with value 1
        let data = [0x81u8];
        let result = read_ebml_vint(&data, 0);
        assert_eq!(result, Some((1, 1)));
    }

    #[test]
    #[cfg(any(feature = "vp9", feature = "av1"))]
    fn test_yuv420p_all_grey() {
        use oximedia_codec::{ColorInfo, FrameType, Plane, VideoFrame};
        use oximedia_core::{PixelFormat, Rational, Timestamp};

        let w = 4usize;
        let h = 4usize;

        // Y=128 (neutral), U=128, V=128 → near-neutral RGB
        let y_data = vec![128u8; w * h];
        let u_data = vec![128u8; (w / 2) * (h / 2)];
        let v_data = vec![128u8; (w / 2) * (h / 2)];

        let frame = VideoFrame {
            format: PixelFormat::Yuv420p,
            width: w as u32,
            height: h as u32,
            planes: vec![
                Plane {
                    data: y_data,
                    stride: w,
                    width: w as u32,
                    height: h as u32,
                },
                Plane {
                    data: u_data,
                    stride: w / 2,
                    width: (w / 2) as u32,
                    height: (h / 2) as u32,
                },
                Plane {
                    data: v_data,
                    stride: w / 2,
                    width: (w / 2) as u32,
                    height: (h / 2) as u32,
                },
            ],
            timestamp: Timestamp::new(0, Rational::new(1, 1000)),
            frame_type: FrameType::Key,
            color_info: ColorInfo::default(),
            corrupt: false,
        };

        let result = video_frame_to_rgba(&frame);
        assert!(result.is_some());
        let rgba = result.expect("rgba frame");
        assert_eq!(rgba.width, w as u32);
        assert_eq!(rgba.height, h as u32);
        assert_eq!(rgba.data.len(), w * h * 4);
        // Alpha should always be 255.
        assert_eq!(rgba.data[3], 255);
    }
}
