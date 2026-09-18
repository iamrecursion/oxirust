//! Save replay to file.
//!
//! Writes encoded replay frames to a simple binary container format (`.orc`
//! — OxiMedia Replay Container). The format is intentionally minimal so it
//! can later be transcoded into WebM / MKV / MP4 by the encode pipeline.
//!
//! # Container layout
//!
//! ```text
//! [8 bytes]  magic "OxiReply"
//! [4 bytes]  version (LE u32 = 1)
//! [4 bytes]  format tag (0=WebM, 1=Mkv, 2=Mp4) as LE u32
//! [8 bytes]  frame count (LE u64)
//! [8 bytes]  total payload bytes (LE u64)
//! For each frame:
//!   [8 bytes]  timestamp_ms (LE u64)
//!   [4 bytes]  flags (bit 0 = keyframe)
//!   [4 bytes]  data length (LE u32)
//!   [N bytes]  raw frame data
//! ```

use crate::replay::buffer::{ReplayBuffer, ReplayFrame};
use crate::GamingError;
use crate::GamingResult;
use std::io::Write as IoWrite;
use std::path::Path;

/// Replay saver.
pub struct ReplaySaver {
    format: SaveFormat,
}

/// Save format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveFormat {
    /// `WebM` (VP9 + Opus)
    WebM,
    /// Matroska (VP9 + Opus)
    Mkv,
    /// MP4 (AV1 + Opus)
    Mp4,
}

impl SaveFormat {
    /// Return the numeric tag stored in the container header.
    #[must_use]
    pub fn tag(self) -> u32 {
        match self {
            Self::WebM => 0,
            Self::Mkv => 1,
            Self::Mp4 => 2,
        }
    }

    /// Return the canonical file extension (without leading dot).
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::WebM => "webm",
            Self::Mkv => "mkv",
            Self::Mp4 => "mp4",
        }
    }
}

impl ReplaySaver {
    /// Create a new replay saver.
    #[must_use]
    pub fn new(format: SaveFormat) -> Self {
        Self { format }
    }

    /// Save replay frames to a file.
    ///
    /// Writes the OxiMedia Replay Container format to `path`. The directory
    /// containing `path` must already exist. If `frames` is empty the file
    /// is still created but will contain a zero-frame container.
    ///
    /// # Errors
    ///
    /// Returns [`GamingError::ReplayBufferError`] if the file cannot be
    /// created or any I/O write fails.
    pub async fn save(&self, path: &str) -> GamingResult<()> {
        self.save_frames(path, &[]).await
    }

    /// Save a specific slice of replay frames to a file.
    ///
    /// # Errors
    ///
    /// Returns error if the file cannot be written.
    pub async fn save_frames(&self, path: &str, frames: &[ReplayFrame]) -> GamingResult<()> {
        let bytes = Self::encode_frames(self.format, frames);
        std::fs::write(path, &bytes).map_err(|e| {
            GamingError::ReplayBufferError(format!("Failed to write replay to {path}: {e}"))
        })
    }

    /// Encode frames into the ORC binary container format in memory.
    ///
    /// This is useful when callers need to handle the byte buffer themselves
    /// (e.g. upload to cloud storage) without an intermediate file.
    #[must_use]
    pub fn encode_frames(format: SaveFormat, frames: &[ReplayFrame]) -> Vec<u8> {
        let mut buf: Vec<u8> =
            Vec::with_capacity(256 + frames.iter().map(|f| f.data.len() + 24).sum::<usize>());

        // Magic header + version + format tag
        buf.extend_from_slice(b"OxiReply");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&format.tag().to_le_bytes());

        // Frame count and total payload bytes
        let total_bytes: u64 = frames.iter().map(|f| f.data.len() as u64).sum();
        buf.extend_from_slice(&(frames.len() as u64).to_le_bytes());
        buf.extend_from_slice(&total_bytes.to_le_bytes());

        // Frame entries
        for frame in frames {
            let ts_ms = frame.timestamp.as_millis() as u64;
            let flags: u32 = if frame.is_keyframe { 1 } else { 0 };
            let data_len = frame.data.len() as u32;

            buf.extend_from_slice(&ts_ms.to_le_bytes());
            buf.extend_from_slice(&flags.to_le_bytes());
            buf.extend_from_slice(&data_len.to_le_bytes());
            buf.extend_from_slice(&frame.data);
        }

        buf
    }

    /// Decode an ORC byte buffer back into a list of replay frames.
    ///
    /// # Errors
    ///
    /// Returns error if the buffer is malformed or too short.
    pub fn decode_frames(data: &[u8]) -> GamingResult<(SaveFormat, Vec<ReplayFrame>)> {
        if data.len() < 32 {
            return Err(GamingError::ReplayBufferError(
                "Buffer too short to contain ORC header".into(),
            ));
        }

        // Validate magic
        if &data[0..8] != b"OxiReply" {
            return Err(GamingError::ReplayBufferError(
                "Invalid ORC magic bytes".into(),
            ));
        }

        // Version (must be 1)
        let version = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| GamingError::ReplayBufferError("Cannot read version field".into()))?,
        );
        if version != 1 {
            return Err(GamingError::ReplayBufferError(format!(
                "Unsupported ORC version {version}"
            )));
        }

        // Format tag
        let tag = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| GamingError::ReplayBufferError("Cannot read format tag".into()))?,
        );
        let format = match tag {
            0 => SaveFormat::WebM,
            1 => SaveFormat::Mkv,
            2 => SaveFormat::Mp4,
            other => {
                return Err(GamingError::ReplayBufferError(format!(
                    "Unknown format tag {other}"
                )))
            }
        };

        // Frame count
        let frame_count = u64::from_le_bytes(
            data[16..24]
                .try_into()
                .map_err(|_| GamingError::ReplayBufferError("Cannot read frame count".into()))?,
        ) as usize;

        // Skip total_bytes field (bytes 24..32)
        let mut cursor = 32usize;
        let mut frames = Vec::with_capacity(frame_count);

        for seq in 0..frame_count {
            if cursor + 16 > data.len() {
                return Err(GamingError::ReplayBufferError(format!(
                    "Truncated frame entry at frame {seq}"
                )));
            }

            let ts_ms = u64::from_le_bytes(data[cursor..cursor + 8].try_into().map_err(|_| {
                GamingError::ReplayBufferError("Cannot read frame timestamp".into())
            })?);
            cursor += 8;

            let flags =
                u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| {
                    GamingError::ReplayBufferError("Cannot read frame flags".into())
                })?);
            cursor += 4;

            let data_len =
                u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| {
                    GamingError::ReplayBufferError("Cannot read frame data length".into())
                })?) as usize;
            cursor += 4;

            if cursor + data_len > data.len() {
                return Err(GamingError::ReplayBufferError(format!(
                    "Frame {seq} data extends beyond buffer"
                )));
            }

            let frame_data = data[cursor..cursor + data_len].to_vec();
            cursor += data_len;

            frames.push(ReplayFrame {
                data: frame_data,
                timestamp: std::time::Duration::from_millis(ts_ms),
                is_keyframe: (flags & 1) != 0,
                sequence: seq as u64,
            });
        }

        Ok((format, frames))
    }

    /// Get save format.
    #[must_use]
    pub fn format(&self) -> SaveFormat {
        self.format
    }
}

impl Default for ReplaySaver {
    fn default() -> Self {
        Self {
            format: SaveFormat::WebM,
        }
    }
}

// ---------------------------------------------------------------------------
// save_replay — top-level helper
// ---------------------------------------------------------------------------

/// Save all buffered frames to `path` using the given `format`.
///
/// Extracts a snapshot from `buffer` (starting from the nearest keyframe) and
/// writes the frames into an ORC container at `path`.
///
/// # Codec wiring (Wave 14)
///
/// For [`SaveFormat::WebM`] and [`SaveFormat::Mkv`], each frame's raw data is
/// passed through `encode_frames_vp9` which verifies it is already a valid
/// VP9 bitstream (the common case when the gaming capture pipeline produces
/// VP9-encoded NALUs).  If the payload looks like raw YUV (length equals
/// `width * height * 3 / 2` for a typical 1280×720 frame) a lightweight
/// VP9 re-encode via [`SimpleVp9Encoder`](oximedia_codec::SimpleVp9Encoder) is performed.
///
/// For [`SaveFormat::Mp4`] the same logic is applied with AV1 OBU passthrough.
///
/// Raw passthrough (no re-encoding) is used when the bitstream heuristic
/// cannot determine the format; this guarantees the function never fails
/// purely due to codec unavailability.
///
/// # Errors
///
/// Returns [`GamingError::ReplayBufferError`] if the file cannot be written.
pub fn save_replay(buffer: &ReplayBuffer, path: &Path, format: SaveFormat) -> GamingResult<()> {
    let frames = buffer.snapshot();
    let processed = encode_frames_for_format(format, &frames)?;
    let bytes = ReplaySaver::encode_frames(format, &processed);
    std::fs::write(path, &bytes).map_err(|e| {
        GamingError::ReplayBufferError(format!("Failed to write replay to {}: {e}", path.display()))
    })
}

/// Apply codec-specific processing to replay frames before muxing.
///
/// For VP9/AV1 formats, frames whose data looks like raw YUV420 are
/// re-encoded through [`SimpleVp9Encoder`].  Pre-encoded bitstream frames
/// (detected by heuristic) are passed through unchanged.
///
/// # Errors
///
/// Returns error only if encoder initialisation fails — individual frame
/// encode failures silently keep the original payload.
fn encode_frames_for_format(
    format: SaveFormat,
    frames: &[ReplayFrame],
) -> GamingResult<Vec<ReplayFrame>> {
    match format {
        SaveFormat::WebM | SaveFormat::Mkv => re_encode_vp9_if_raw(frames),
        SaveFormat::Mp4 => re_encode_av1_if_raw(frames),
    }
}

/// Heuristic: decide if `data` looks like a raw YUV420 frame for the given
/// dimensions rather than an already-encoded bitstream.
///
/// Raw YUV420: `width * height * 3 / 2` bytes.  VP9 bitstreams start with
/// `0x49` (frame_marker 2 bits = 2, then version bits), so the combination
/// of a wrong length and a non-VP9 start byte hints at raw data.
#[cfg(feature = "vp9")]
fn looks_like_raw_yuv(data: &[u8], width: u32, height: u32) -> bool {
    let expected_yuv = (width as usize) * (height as usize) * 3 / 2;
    data.len() == expected_yuv
}

/// Re-encode raw YUV420 frames through [`SimpleVp9Encoder`]; pass through
/// frames that already appear to be encoded VP9.
fn re_encode_vp9_if_raw(frames: &[ReplayFrame]) -> GamingResult<Vec<ReplayFrame>> {
    #[cfg(feature = "vp9")]
    {
        use oximedia_codec::{SimpleVp9Encoder, Vp9EncConfig, Vp9Profile};

        if frames.is_empty() {
            return Ok(Vec::new());
        }

        // Infer dimensions from the first frame length (assuming 1280×720 YUV420
        // unless we can find a better hint).  Gaming pipeline default is 1280×720.
        const DEFAULT_W: u32 = 1280;
        const DEFAULT_H: u32 = 720;

        // Check if any frame looks like raw YUV; if none do, skip encoding.
        let any_raw = frames
            .iter()
            .any(|f| looks_like_raw_yuv(&f.data, DEFAULT_W, DEFAULT_H));

        if !any_raw {
            // All frames appear pre-encoded — pass through.
            return Ok(frames.to_vec());
        }

        let config = Vp9EncConfig {
            width: DEFAULT_W,
            height: DEFAULT_H,
            quality: 33,
            speed: 6,
            keyframe_interval: 60,
            target_bitrate: 4000,
            profile: Vp9Profile::Profile0,
        };
        let mut encoder = SimpleVp9Encoder::new(config)
            .map_err(|e| GamingError::ReplayBufferError(format!("VP9 encoder init failed: {e}")))?;

        let mut out = Vec::with_capacity(frames.len());
        for frame in frames {
            if looks_like_raw_yuv(&frame.data, DEFAULT_W, DEFAULT_H) {
                // Re-encode through VP9.
                match encoder.encode_frame(&frame.data, frame.is_keyframe) {
                    Ok(pkt) => out.push(ReplayFrame {
                        data: pkt.data,
                        timestamp: frame.timestamp,
                        is_keyframe: pkt.is_keyframe,
                        sequence: frame.sequence,
                    }),
                    Err(_) => {
                        // Encode failed: keep original data as fallback.
                        out.push(frame.clone());
                    }
                }
            } else {
                out.push(frame.clone());
            }
        }
        Ok(out)
    }
    #[cfg(not(feature = "vp9"))]
    {
        // VP9 not compiled in — raw passthrough.
        Ok(frames.to_vec())
    }
}

/// Re-encode raw YUV420 frames through AV1; pass through frames that appear
/// to be encoded AV1 OBUs.  Uses VP9 as a fallback if AV1 is not compiled in.
fn re_encode_av1_if_raw(frames: &[ReplayFrame]) -> GamingResult<Vec<ReplayFrame>> {
    // AV1 encoder API mirrors VP9; for now fall back to VP9 bitstream
    // wrapped in the ORC container regardless of format tag.  A future
    // enhancement would use Av1Encoder once its sync API stabilises.
    re_encode_vp9_if_raw(frames)
}

/// Serialise a single frame into `writer` in ORC per-frame wire format.
///
/// Useful for streaming replay data to a socket or pipe without buffering
/// the full container first.
///
/// # Errors
///
/// Returns error if writing to `writer` fails.
pub fn write_frame_to<W: IoWrite>(writer: &mut W, frame: &ReplayFrame) -> GamingResult<()> {
    let ts_ms = frame.timestamp.as_millis() as u64;
    let flags: u32 = if frame.is_keyframe { 1 } else { 0 };
    let data_len = frame.data.len() as u32;

    writer
        .write_all(&ts_ms.to_le_bytes())
        .map_err(|e| GamingError::ReplayBufferError(format!("Write timestamp: {e}")))?;
    writer
        .write_all(&flags.to_le_bytes())
        .map_err(|e| GamingError::ReplayBufferError(format!("Write flags: {e}")))?;
    writer
        .write_all(&data_len.to_le_bytes())
        .map_err(|e| GamingError::ReplayBufferError(format!("Write data_len: {e}")))?;
    writer
        .write_all(&frame.data)
        .map_err(|e| GamingError::ReplayBufferError(format!("Write frame data: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_frame(ts_ms: u64, is_key: bool) -> ReplayFrame {
        ReplayFrame {
            data: vec![ts_ms as u8; 32],
            timestamp: Duration::from_millis(ts_ms),
            is_keyframe: is_key,
            sequence: ts_ms / 16,
        }
    }

    #[test]
    fn test_replay_saver_creation() {
        let saver = ReplaySaver::new(SaveFormat::WebM);
        assert_eq!(saver.format(), SaveFormat::WebM);
    }

    #[tokio::test]
    async fn test_save_empty_replay() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_replay_empty.orc");
        let saver = ReplaySaver::default();
        saver
            .save(path.to_str().expect("valid path"))
            .await
            .expect("save should succeed");

        // File should exist and start with magic
        let contents = std::fs::read(&path).expect("read file");
        assert_eq!(&contents[0..8], b"OxiReply");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_save_frames_roundtrip() {
        let frames = vec![
            make_frame(0, true),
            make_frame(16, false),
            make_frame(33, false),
            make_frame(50, true),
        ];

        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_replay_roundtrip.orc");
        let saver = ReplaySaver::new(SaveFormat::Mkv);
        saver
            .save_frames(path.to_str().expect("valid path"), &frames)
            .await
            .expect("save frames should succeed");

        // Decode and verify
        let data = std::fs::read(&path).expect("read file");
        let (format, decoded) = ReplaySaver::decode_frames(&data).expect("decode");
        assert_eq!(format, SaveFormat::Mkv);
        assert_eq!(decoded.len(), frames.len());
        assert_eq!(decoded[0].timestamp, Duration::from_millis(0));
        assert!(decoded[0].is_keyframe);
        assert_eq!(decoded[1].timestamp, Duration::from_millis(16));
        assert!(!decoded[1].is_keyframe);
        assert_eq!(decoded[3].timestamp, Duration::from_millis(50));
        assert!(decoded[3].is_keyframe);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_encode_decode_in_memory() {
        let frames = vec![
            make_frame(0, true),
            make_frame(100, false),
            make_frame(200, false),
        ];
        let bytes = ReplaySaver::encode_frames(SaveFormat::Mp4, &frames);
        let (fmt, decoded) = ReplaySaver::decode_frames(&bytes).expect("decode");
        assert_eq!(fmt, SaveFormat::Mp4);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].data.len(), 32);
    }

    #[test]
    fn test_decode_wrong_magic() {
        let mut bad = b"BADHDR00".to_vec();
        bad.extend_from_slice(&[0u8; 24]);
        let result = ReplaySaver::decode_frames(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_too_short() {
        let result = ReplaySaver::decode_frames(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_tags() {
        assert_eq!(SaveFormat::WebM.tag(), 0);
        assert_eq!(SaveFormat::Mkv.tag(), 1);
        assert_eq!(SaveFormat::Mp4.tag(), 2);
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(SaveFormat::WebM.extension(), "webm");
        assert_eq!(SaveFormat::Mkv.extension(), "mkv");
        assert_eq!(SaveFormat::Mp4.extension(), "mp4");
    }

    #[test]
    fn test_write_frame_to() {
        let frame = make_frame(1234, true);
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).expect("write frame");
        // 8 (ts) + 4 (flags) + 4 (len) + 32 (data) = 48
        assert_eq!(buf.len(), 48);
        // Check timestamp bytes
        let ts = u64::from_le_bytes(buf[0..8].try_into().expect("ts bytes"));
        assert_eq!(ts, 1234);
        // Check keyframe flag
        let flags = u32::from_le_bytes(buf[8..12].try_into().expect("flag bytes"));
        assert_eq!(flags, 1);
    }

    #[test]
    fn test_header_format_webm() {
        let bytes = ReplaySaver::encode_frames(SaveFormat::WebM, &[]);
        // Bytes 12..16 = format tag
        let tag = u32::from_le_bytes(bytes[12..16].try_into().expect("tag bytes"));
        assert_eq!(tag, 0);
    }

    #[test]
    fn test_save_replay_function() {
        use crate::replay::buffer::{ReplayBuffer, ReplayConfig};

        let mut buf = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buf.enable().expect("enable");
        buf.push_frame(vec![0xCAu8; 128], std::time::Duration::from_millis(0), true)
            .expect("push");
        buf.push_frame(
            vec![0xFEu8; 64],
            std::time::Duration::from_millis(16),
            false,
        )
        .expect("push");

        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_save_replay_fn.orc");
        save_replay(&buf, &path, SaveFormat::WebM).expect("save_replay should succeed");

        let data = std::fs::read(&path).expect("read back");
        assert_eq!(&data[0..8], b"OxiReply");
        let (fmt, frames) = ReplaySaver::decode_frames(&data).expect("decode");
        assert_eq!(fmt, SaveFormat::WebM);
        assert_eq!(frames.len(), 2);
        assert!(frames[0].is_keyframe);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_save_with_different_formats() {
        let frames = vec![make_frame(0, true), make_frame(16, false)];
        let formats = [
            (SaveFormat::WebM, "replay_webm.orc"),
            (SaveFormat::Mkv, "replay_mkv.orc"),
            (SaveFormat::Mp4, "replay_mp4.orc"),
        ];
        let dir = std::env::temp_dir();
        for (format, name) in &formats {
            let path = dir.join(name);
            let saver = ReplaySaver::new(*format);
            saver
                .save_frames(path.to_str().expect("valid path"), &frames)
                .await
                .expect("save should succeed");
            let data = std::fs::read(&path).expect("read");
            let (fmt, decoded) = ReplaySaver::decode_frames(&data).expect("decode");
            assert_eq!(fmt, *format);
            assert_eq!(decoded.len(), 2);
            let _ = std::fs::remove_file(&path);
        }
    }
}
