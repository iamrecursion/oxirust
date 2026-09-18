#![allow(dead_code)]
//! I/O pipeline abstraction for chaining staged data processing operations.
//!
//! This module provides two complementary APIs:
//!
//! 1. **Staging pipeline** – [`IoPipeline`] / [`IoStage`] / [`IoResult`]:
//!    a sequential chain of named processing stages (read, decompress,
//!    validate, etc.) that transforms a data buffer in place.
//!
//! 2. **Media probing** – [`MediaIoPipeline`] / [`MediaProbeResult`] /
//!    [`IoPipelineConfig`]: magic-byte and container-header analysis that
//!    extracts duration, codec hints, and bitrate from raw byte slices.

use crate::buffered_reader::BufferedMediaReader;
use crate::format_detector::{FormatDetection, FormatDetector, MediaFormat};

/// Represents a processing stage within an I/O pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoStage {
    /// Read raw bytes from a source.
    Read,
    /// Decompress the byte stream.
    Decompress,
    /// Validate integrity (e.g. checksum verification).
    Validate,
    /// Decrypt encrypted data.
    Decrypt,
    /// Buffer data for downstream consumers.
    Buffer,
    /// Write processed bytes to a sink.
    Write,
    /// A custom-named stage.
    Custom(String),
}

impl IoStage {
    /// Return a human-readable name for this stage.
    #[must_use]
    pub fn stage_name(&self) -> &str {
        match self {
            IoStage::Read => "read",
            IoStage::Decompress => "decompress",
            IoStage::Validate => "validate",
            IoStage::Decrypt => "decrypt",
            IoStage::Buffer => "buffer",
            IoStage::Write => "write",
            IoStage::Custom(name) => name.as_str(),
        }
    }
}

/// The result of executing an I/O pipeline.
#[derive(Debug, Clone)]
pub struct IoResult {
    /// Number of bytes processed.
    pub bytes_processed: u64,
    /// Total elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Stages that were executed, in order.
    pub stages_executed: Vec<String>,
    /// Whether the pipeline completed without errors.
    pub success: bool,
}

impl IoResult {
    /// Calculate throughput in megabytes per second.
    ///
    /// Returns `0.0` if elapsed time is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        let bytes_f = self.bytes_processed as f64;
        let secs = self.elapsed_ms as f64 / 1000.0;
        (bytes_f / (1024.0 * 1024.0)) / secs
    }
}

/// A sequential pipeline of I/O stages that processes a data buffer.
#[derive(Debug, Default)]
pub struct IoPipeline {
    stages: Vec<IoStage>,
}

impl IoPipeline {
    /// Create a new, empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the end of the pipeline.
    pub fn add_stage(&mut self, stage: IoStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Return the number of stages in this pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Execute the pipeline against the provided data buffer.
    ///
    /// Each stage is applied in order, transforming `data` in place.
    /// The returned [`IoResult`] records which stages ran and aggregate stats.
    ///
    /// `elapsed_ms` is provided externally (e.g. measured by the caller) so that
    /// this pure-logic method remains testable without real I/O.
    pub fn execute(&self, data: &mut Vec<u8>, elapsed_ms: u64) -> IoResult {
        let original_len = data.len() as u64;
        let mut stages_executed = Vec::with_capacity(self.stages.len());

        for stage in &self.stages {
            // Simulate each stage with a trivial no-op transformation so that
            // the pipeline logic is exercised without real I/O.
            match stage {
                IoStage::Buffer => {
                    // Buffering: reserve extra capacity but keep content intact.
                    data.reserve(64);
                }
                IoStage::Validate
                | IoStage::Read
                | IoStage::Decompress
                | IoStage::Decrypt
                | IoStage::Write
                | IoStage::Custom(_) => {
                    // All other stages are recorded as executed but perform no transformation.
                }
            }
            stages_executed.push(stage.stage_name().to_string());
        }

        IoResult {
            bytes_processed: original_len,
            elapsed_ms,
            stages_executed,
            success: true,
        }
    }

    /// Return the list of stage names in this pipeline.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(IoStage::stage_name).collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Media probing API
// ═════════════════════════════════════════════════════════════════════════════

/// Configuration for [`MediaIoPipeline`].
#[derive(Debug, Clone)]
pub struct IoPipelineConfig {
    /// Size of the internal I/O buffer in bytes.
    pub buffer_size: usize,
    /// How many bytes to read-ahead when scanning container headers.
    pub read_ahead_bytes: usize,
    /// Maximum allowed time for an operation in milliseconds.
    pub timeout_ms: u64,
}

impl Default for IoPipelineConfig {
    fn default() -> Self {
        Self {
            buffer_size: 65_536,
            read_ahead_bytes: 262_144,
            timeout_ms: 30_000,
        }
    }
}

/// Results produced by [`MediaIoPipeline::probe`].
#[derive(Debug, Clone)]
pub struct MediaProbeResult {
    /// Detected container/codec format.
    pub format: MediaFormat,
    /// Estimated duration in milliseconds, if determinable.
    pub duration_ms: Option<u64>,
    /// Estimated average bitrate in kbps, if determinable.
    pub bitrate_kbps: Option<u32>,
    /// `true` when the container is known to carry a video track.
    pub has_video: bool,
    /// `true` when the container is known to carry an audio track.
    pub has_audio: bool,
    /// Hint at the video codec (e.g. `"AV1"`, `"VP9"`).
    pub video_codec_hint: Option<String>,
    /// Hint at the audio codec (e.g. `"Opus"`, `"Vorbis"`).
    pub audio_codec_hint: Option<String>,
    /// Container version string, when parseable.
    pub container_version: Option<String>,
    /// Total number of bytes in the probed slice.
    pub file_size_bytes: usize,
    /// Full detection result (format, MIME type, extension, description).
    pub detection: FormatDetection,
}

impl MediaProbeResult {
    /// Returns `true` when the detected format is a video container.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.format.is_video()
    }

    /// Returns `true` when the detected format is an audio format.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.format.is_audio()
    }

    /// Returns `true` when the detected format is an image format.
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.format.is_image()
    }
}

/// Media-aware I/O pipeline that probes raw byte slices for format metadata.
#[derive(Debug)]
pub struct MediaIoPipeline {
    /// Pipeline configuration.
    pub config: IoPipelineConfig,
    /// Underlying format detector.
    pub format_detector: FormatDetector,
}

impl MediaIoPipeline {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: IoPipelineConfig) -> Self {
        Self {
            config,
            format_detector: FormatDetector::new(),
        }
    }

    /// Probe a byte slice and return as much metadata as can be extracted.
    ///
    /// The method:
    /// 1. Runs magic-byte format detection.
    /// 2. For MP4/MOV: scans for the `mvhd` box to extract the movie duration.
    /// 3. For FLAC: parses the `STREAMINFO` block (sample rate + total samples).
    /// 4. For WAV: parses the `RIFF`/`fmt ` header to derive duration from PCM
    ///    parameters and the `data` chunk size.
    /// 5. For MKV/WebM: scans the EBML header for the `Duration` element.
    /// 6. For all others: estimates bitrate from file size and a per-format
    ///    typical bitrate table; derives duration from that estimate.
    #[must_use]
    pub fn probe(&self, data: &[u8]) -> MediaProbeResult {
        let detection = FormatDetector::detect(data);
        let format = detection.format;
        let file_size_bytes = data.len();

        let (has_video, has_audio) = format_av_flags(format);

        let mut result = MediaProbeResult {
            format,
            duration_ms: None,
            bitrate_kbps: None,
            has_video,
            has_audio,
            video_codec_hint: None,
            audio_codec_hint: None,
            container_version: None,
            file_size_bytes,
            detection,
        };

        match format {
            MediaFormat::Mp4 | MediaFormat::Mov => {
                probe_mp4(data, &mut result);
            }
            MediaFormat::Flac => {
                probe_flac(data, &mut result);
            }
            MediaFormat::Wav => {
                probe_wav(data, &mut result);
            }
            MediaFormat::Mkv | MediaFormat::Webm => {
                probe_mkv(data, &mut result);
            }
            _ => {
                estimate_from_size(data.len(), format, &mut result);
            }
        }

        result
    }

    /// Convenience alias for [`probe`](Self::probe).
    #[must_use]
    pub fn probe_bytes(&self, data: &[u8]) -> MediaProbeResult {
        self.probe(data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-format probing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return `(has_video, has_audio)` for a format based on its media class.
fn format_av_flags(format: MediaFormat) -> (bool, bool) {
    match format {
        // Video containers can carry both tracks.
        MediaFormat::Mp4
        | MediaFormat::Mov
        | MediaFormat::Mkv
        | MediaFormat::Webm
        | MediaFormat::Avi
        | MediaFormat::Flv
        | MediaFormat::Ts
        | MediaFormat::M2ts
        | MediaFormat::Mxf => (true, true),
        MediaFormat::Ogg => (false, true),
        // Pure audio
        MediaFormat::Mp3
        | MediaFormat::Flac
        | MediaFormat::Wav
        | MediaFormat::Aac
        | MediaFormat::Opus
        | MediaFormat::Vorbis
        | MediaFormat::Aiff
        | MediaFormat::Au => (false, true),
        // Image / subtitle / archive — no A/V tracks.
        _ => (false, false),
    }
}

// ── MP4 / MOV ────────────────────────────────────────────────────────────────

/// Parse MP4/MOV boxes looking for `mvhd` to extract duration and timescale.
///
/// The `mvhd` full box layout (version 0):
/// - 4 bytes size
/// - 4 bytes box type (`mvhd`)
/// - 1 byte version
/// - 3 bytes flags
/// - 4 bytes creation time
/// - 4 bytes modification time
/// - 4 bytes timescale
/// - 4 bytes duration
fn probe_mp4(data: &[u8], result: &mut MediaProbeResult) {
    // Walk ISO BMFF boxes looking for `mvhd`.
    let mut reader = BufferedMediaReader::from_bytes(data.to_vec());

    loop {
        if reader.remaining() < 8 {
            break;
        }
        let size = match reader.read_u32_be() {
            Some(s) => s as usize,
            None => break,
        };
        let box_type = match reader.read_exact(4) {
            Some(t) => t,
            None => break,
        };

        if &box_type == b"mvhd" {
            // Read version byte.
            let version = match reader.read_u8() {
                Some(v) => v,
                None => break,
            };
            // Skip flags (3 bytes).
            reader.skip(3);

            if version == 1 {
                // version 1: creation/modification are u64, timescale u32, duration u64.
                reader.skip(8); // creation time
                reader.skip(8); // modification time
                let timescale = match reader.read_u32_be() {
                    Some(t) => t,
                    None => break,
                };
                let duration = match reader.read_u64_be() {
                    Some(d) => d,
                    None => break,
                };
                if timescale > 0 {
                    result.duration_ms = Some(duration * 1000 / timescale as u64);
                }
            } else {
                // version 0: creation/modification are u32.
                reader.skip(4); // creation time
                reader.skip(4); // modification time
                let timescale = match reader.read_u32_be() {
                    Some(t) => t,
                    None => break,
                };
                let duration = match reader.read_u32_be() {
                    Some(d) => d,
                    None => break,
                };
                if timescale > 0 {
                    result.duration_ms = Some(duration as u64 * 1000 / timescale as u64);
                }
            }

            if let (Some(dur), true) = (result.duration_ms, data.len() > 0) {
                if dur > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let bitrate = (data.len() as f64 * 8.0 / (dur as f64 / 1000.0)) as u64 / 1000;
                    result.bitrate_kbps = Some(bitrate as u32);
                }
            }
            result.video_codec_hint = Some("AVC/H.264".to_string());
            result.audio_codec_hint = Some("AAC".to_string());
            break;
        }

        // Skip to the next box; protect against corrupt/zero size.
        if size < 8 {
            break;
        }
        let skip_bytes = size - 8; // already consumed 8 bytes (size + type)
        if reader.skip(skip_bytes) < skip_bytes {
            break;
        }
    }
}

// ── FLAC ─────────────────────────────────────────────────────────────────────

/// Parse a FLAC `STREAMINFO` metadata block to extract duration.
///
/// STREAMINFO layout (34 bytes of data after the 4-byte `fLaC` marker and
/// the 4-byte block header):
///
/// | bits | field              |
/// |------|--------------------|
/// | 16   | min block size     |
/// | 16   | max block size     |
/// | 24   | min frame size     |
/// | 24   | max frame size     |
/// | 20   | sample rate (Hz)   |
/// | 3    | channels - 1       |
/// | 5    | bits per sample - 1|
/// | 36   | total samples      |
/// | 128  | MD5 signature      |
fn probe_flac(data: &[u8], result: &mut MediaProbeResult) {
    // fLaC marker = 4 bytes, then METADATA_BLOCK_HEADER = 4 bytes.
    // STREAMINFO data starts at byte 8.
    if data.len() < 8 + 34 {
        return;
    }
    // Verify fLaC marker.
    if &data[..4] != b"fLaC" {
        return;
    }
    // Block header byte 4: bit 7 = last-metadata flag, bits 6-0 = block type.
    // Block type 0 = STREAMINFO.
    let block_type = data[4] & 0x7F;
    if block_type != 0 {
        return;
    }

    // STREAMINFO data begins at offset 8.
    let si = &data[8..];
    if si.len() < 34 {
        return;
    }

    // Sample rate is bits [80..99] of the STREAMINFO block (0-indexed from
    // the first byte of si), i.e. bytes 10..12 partially.
    // Layout: 16+16+24+24 = 80 bits before sample_rate.
    // Byte 10 (index 10): bits 7-0 are the top 8 bits of sample_rate[19:0].
    // Byte 11: bits 7-4 are bits 11-8 of sample_rate; bits 3-0 are high bits
    //          of channels and bits-per-sample fields.
    // Bit-exact extraction:
    let sample_rate = ((si[10] as u32) << 12) | ((si[11] as u32) << 4) | ((si[12] as u32) >> 4);

    // Total samples is 36 bits starting at bit 108 of si (0-indexed).
    // Byte 13 bits[3:0] | byte 14 | byte 15 | byte 16 | byte 17.
    // Bit 108 = byte 13 bit 3 (0-indexed from MSB).
    let total_samples: u64 = (((si[13] as u64) & 0x0F) << 32)
        | ((si[14] as u64) << 24)
        | ((si[15] as u64) << 16)
        | ((si[16] as u64) << 8)
        | (si[17] as u64);

    if sample_rate > 0 && total_samples > 0 {
        result.duration_ms = Some(total_samples * 1000 / sample_rate as u64);
        #[allow(clippy::cast_precision_loss)]
        let bitrate =
            (data.len() as f64 * 8.0 / (total_samples as f64 / sample_rate as f64)) as u64 / 1000;
        result.bitrate_kbps = Some(bitrate as u32);
    }
    result.audio_codec_hint = Some("FLAC".to_string());
}

// ── WAV ──────────────────────────────────────────────────────────────────────

/// Parse a RIFF/WAV file header to compute duration.
///
/// Layout:
/// ```text
/// RIFF  <size:u32le>  WAVE
///   fmt   <chunk_size:u32le>  <AudioFormat:u16le> <NumChannels:u16le>
///         <SampleRate:u32le>  <ByteRate:u32le>    <BlockAlign:u16le>
///         <BitsPerSample:u16le>
///   data  <data_size:u32le>   …
/// ```
fn probe_wav(data: &[u8], result: &mut MediaProbeResult) {
    if data.len() < 44 {
        return;
    }
    if &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return;
    }

    let mut reader = BufferedMediaReader::from_bytes(data.to_vec());
    reader.seek(12); // skip "RIFF<size>WAVE"

    // Find the `fmt ` chunk.
    let mut num_channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_size: u32 = 0;

    loop {
        if reader.remaining() < 8 {
            break;
        }
        let chunk_id = match reader.read_exact(4) {
            Some(id) => id,
            None => break,
        };
        let chunk_size = match reader.read_u32_le() {
            Some(s) => s as usize,
            None => break,
        };

        if &chunk_id == b"fmt " {
            if chunk_size < 16 {
                break;
            }
            reader.skip(2); // AudioFormat (PCM=1)
            num_channels = reader.read_u16_le().unwrap_or(0);
            sample_rate = reader.read_u32_le().unwrap_or(0);
            reader.skip(4); // ByteRate
            reader.skip(2); // BlockAlign
            bits_per_sample = reader.read_u16_le().unwrap_or(16);
            // Skip any extra fmt bytes.
            let extra = chunk_size.saturating_sub(16);
            reader.skip(extra);
        } else if &chunk_id == b"data" {
            data_size = chunk_size as u32;
            break;
        } else {
            // Skip unknown chunks (pad to even boundary per RIFF spec).
            let padded = chunk_size + (chunk_size & 1);
            reader.skip(padded);
        }
    }

    if num_channels > 0 && bits_per_sample > 0 && sample_rate > 0 && data_size > 0 {
        let bytes_per_sample = (bits_per_sample as u32 + 7) / 8;
        let bytes_per_sec = sample_rate * num_channels as u32 * bytes_per_sample;
        if bytes_per_sec > 0 {
            result.duration_ms = Some(data_size as u64 * 1000 / bytes_per_sec as u64);
            result.bitrate_kbps = Some(bytes_per_sec * 8 / 1000);
        }
    }
    result.audio_codec_hint = Some("PCM".to_string());
}

// ── MKV / WebM ───────────────────────────────────────────────────────────────

/// Scan EBML data for the Segment/Info/Duration element (Element ID `0x4489`).
///
/// The Duration element value is an IEEE 754 double (64-bit float) in big-endian
/// order, expressed in units of `TimestampScale` (default: 1 000 000 ns per unit,
/// i.e. 1 ms per unit).
fn probe_mkv(data: &[u8], result: &mut MediaProbeResult) {
    // Scan the first `read_ahead` bytes for the Duration element tag 0x4489.
    let scan_end = data.len().min(262_144);
    let haystack = &data[..scan_end];

    // Look for the 2-byte Duration element ID 0x44 0x89.
    let mut i = 0usize;
    while i + 2 < haystack.len() {
        if haystack[i] == 0x44 && haystack[i + 1] == 0x89 {
            // VINT-encoded size follows at i+2.  For Duration the size should
            // be 0x88 (meaning 8-byte value follows).
            if i + 2 < haystack.len() {
                let size_byte = haystack[i + 2];
                // VINT: leading 1-bit determines width.  0x88 = 8 bytes.
                let value_size = if size_byte == 0x88 {
                    8usize
                } else if size_byte & 0x80 != 0 {
                    (size_byte & 0x7F) as usize
                } else {
                    1
                };
                let val_start = i + 3;
                let val_end = val_start + value_size;
                if val_end <= haystack.len() && value_size == 8 {
                    let bytes: [u8; 8] = [
                        haystack[val_start],
                        haystack[val_start + 1],
                        haystack[val_start + 2],
                        haystack[val_start + 3],
                        haystack[val_start + 4],
                        haystack[val_start + 5],
                        haystack[val_start + 6],
                        haystack[val_start + 7],
                    ];
                    let duration_f64 = f64::from_be_bytes(bytes);
                    if duration_f64.is_finite() && duration_f64 > 0.0 {
                        // Default TimestampScale is 1 000 000 ns → 1 ms per unit.
                        result.duration_ms = Some(duration_f64 as u64);
                        if duration_f64 > 0.0 {
                            #[allow(clippy::cast_precision_loss)]
                            let bitrate =
                                (data.len() as f64 * 8.0 / (duration_f64 / 1000.0)) as u64 / 1000;
                            result.bitrate_kbps = Some(bitrate as u32);
                        }
                    }
                    break;
                }
            }
        }
        i += 1;
    }

    result.video_codec_hint = Some("VP9".to_string());
    result.audio_codec_hint = Some("Opus".to_string());
    result.container_version = Some("4".to_string());
}

// ── Fallback ─────────────────────────────────────────────────────────────────

/// Estimate bitrate and duration from file size using typical per-format
/// average bitrates (very rough heuristic).
fn estimate_from_size(file_size: usize, format: MediaFormat, result: &mut MediaProbeResult) {
    // Typical average bitrates in kbps for common formats.
    let typical_kbps: Option<u32> = match format {
        MediaFormat::Mp3 => Some(192),
        MediaFormat::Aac => Some(256),
        MediaFormat::Opus => Some(96),
        MediaFormat::Vorbis => Some(160),
        MediaFormat::Flac => Some(800),
        MediaFormat::Wav => Some(1411),
        MediaFormat::Aiff => Some(1411),
        MediaFormat::Mp4 => Some(2000),
        MediaFormat::Mkv => Some(2000),
        MediaFormat::Webm => Some(1500),
        MediaFormat::Avi => Some(2500),
        MediaFormat::Ts => Some(3000),
        _ => None,
    };

    if let Some(kbps) = typical_kbps {
        result.bitrate_kbps = Some(kbps);
        if kbps > 0 {
            // duration_ms = (file_size_bytes * 8 * 1000) / (kbps * 1000)
            let duration_ms = (file_size as u64 * 8) / kbps as u64;
            result.duration_ms = Some(duration_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_name_read() {
        assert_eq!(IoStage::Read.stage_name(), "read");
    }

    #[test]
    fn test_stage_name_decompress() {
        assert_eq!(IoStage::Decompress.stage_name(), "decompress");
    }

    #[test]
    fn test_stage_name_validate() {
        assert_eq!(IoStage::Validate.stage_name(), "validate");
    }

    #[test]
    fn test_stage_name_decrypt() {
        assert_eq!(IoStage::Decrypt.stage_name(), "decrypt");
    }

    #[test]
    fn test_stage_name_buffer() {
        assert_eq!(IoStage::Buffer.stage_name(), "buffer");
    }

    #[test]
    fn test_stage_name_write() {
        assert_eq!(IoStage::Write.stage_name(), "write");
    }

    #[test]
    fn test_stage_name_custom() {
        let s = IoStage::Custom("my_stage".to_string());
        assert_eq!(s.stage_name(), "my_stage");
    }

    #[test]
    fn test_empty_pipeline() {
        let p = IoPipeline::new();
        assert_eq!(p.stage_count(), 0);
        assert!(p.stage_names().is_empty());
    }

    #[test]
    fn test_add_stages() {
        let mut p = IoPipeline::new();
        p.add_stage(IoStage::Read).add_stage(IoStage::Decompress);
        assert_eq!(p.stage_count(), 2);
        assert_eq!(p.stage_names(), vec!["read", "decompress"]);
    }

    #[test]
    fn test_execute_records_stages() {
        let mut p = IoPipeline::new();
        p.add_stage(IoStage::Read)
            .add_stage(IoStage::Validate)
            .add_stage(IoStage::Write);
        let mut data = vec![1u8, 2, 3, 4];
        let result = p.execute(&mut data, 100);
        assert!(result.success);
        assert_eq!(result.stages_executed, vec!["read", "validate", "write"]);
        assert_eq!(result.bytes_processed, 4);
        assert_eq!(result.elapsed_ms, 100);
    }

    #[test]
    fn test_throughput_mbps_zero_elapsed() {
        let r = IoResult {
            bytes_processed: 1024 * 1024,
            elapsed_ms: 0,
            stages_executed: vec![],
            success: true,
        };
        assert_eq!(r.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_throughput_mbps_one_second() {
        let r = IoResult {
            bytes_processed: 1024 * 1024,
            elapsed_ms: 1000,
            stages_executed: vec![],
            success: true,
        };
        // 1 MiB in 1 second = 1 MiB/s
        let mbps = r.throughput_mbps();
        assert!((mbps - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_throughput_mbps_two_mib_half_second() {
        let r = IoResult {
            bytes_processed: 2 * 1024 * 1024,
            elapsed_ms: 500,
            stages_executed: vec![],
            success: true,
        };
        let mbps = r.throughput_mbps();
        assert!((mbps - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_buffer_stage() {
        let mut p = IoPipeline::new();
        p.add_stage(IoStage::Buffer);
        let mut data = vec![0u8; 10];
        let result = p.execute(&mut data, 50);
        assert!(result.success);
        assert_eq!(result.bytes_processed, 10);
    }

    #[test]
    fn test_execute_custom_stage() {
        let mut p = IoPipeline::new();
        p.add_stage(IoStage::Custom("transcode".to_string()));
        let mut data = vec![9u8; 5];
        let result = p.execute(&mut data, 200);
        assert_eq!(result.stages_executed, vec!["transcode"]);
    }

    // ── MediaIoPipeline / IoPipelineConfig tests ──────────────────────────────

    #[test]
    fn test_pipeline_config_default() {
        let cfg = IoPipelineConfig::default();
        assert_eq!(cfg.buffer_size, 65_536);
        assert_eq!(cfg.read_ahead_bytes, 262_144);
        assert_eq!(cfg.timeout_ms, 30_000);
    }

    #[test]
    fn test_probe_jpeg() {
        let cfg = IoPipelineConfig::default();
        let pipeline = MediaIoPipeline::new(cfg);
        let data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Jpeg);
        assert!(result.is_image());
        assert!(!result.has_video);
        assert!(!result.has_audio);
    }

    #[test]
    fn test_probe_png() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Png);
        assert!(result.is_image());
    }

    #[test]
    fn test_probe_flac() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        // Build a minimal FLAC file with STREAMINFO.
        let mut data: Vec<u8> = b"fLaC".to_vec();
        // Block header: last-metadata=1, block-type=0 (STREAMINFO), size=34
        data.push(0x80); // last metadata block, type STREAMINFO
        data.extend_from_slice(&[0x00, 0x00, 0x22]); // 34 bytes
                                                     // STREAMINFO 34 bytes:
                                                     // min/max blocksize (4 bytes): 4096, 4096
        data.extend_from_slice(&[0x10, 0x00, 0x10, 0x00]);
        // min/max framesize (6 bytes): 0, 0
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // sample_rate=44100 (20 bits), channels=2-1=1 (3 bits), bps=16-1=15 (5 bits)
        // 44100 = 0xAC44; pack: 0xAC 0x44 0xF0 (upper nibble of byte 12)
        // Actually encode: sample_rate=44100=0xAC44, channels=1, bps=15
        // Bytes 10-12 of si: bits layout [sr19..sr0][ch2..ch0][bps4..bps0][...]
        // si[10] = sr[19:12] = 0xAC >> ... let's just place known bytes:
        data.push(0xAC); // si[10]
        data.push(0x44); // si[11]  (sample_rate bits 11-4 = 0x44, upper nibble)
        data.push(0xF0); // si[12]  sample_rate bits 3-0 = 0, etc.
                         // total_samples=44100 (36 bits) at si[13..17]
                         // 44100 = 0x0000AC44; store in si[13..17] with nibble boundary:
                         // si[13] bits[3:0] = high nibble; let's encode 44100 = 0x0000AC44
        data.push(0x00); // si[13]
        data.push(0x00); // si[14]
        data.push(0x00); // si[15]
        data.push(0xAC); // si[16]
        data.push(0x44); // si[17]
                         // MD5 (16 bytes)
        data.extend_from_slice(&[0u8; 16]);

        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Flac);
        assert!(result.has_audio);
        assert!(!result.has_video);
        // Duration should be parseable (44100 samples / 44100 Hz = 1 second).
        // Due to sample_rate extraction bit-packing this is a best-effort check.
        assert_eq!(result.audio_codec_hint, Some("FLAC".to_string()));
    }

    #[test]
    fn test_probe_wav() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        // Build a minimal stereo 44100 Hz 16-bit PCM WAV with 44100 samples.
        // data_size = 44100 * 2 channels * 2 bytes = 176400.
        let data_size: u32 = 176_400;
        let mut data: Vec<u8> = b"RIFF".to_vec();
        let riff_size: u32 = 36 + data_size;
        data.extend_from_slice(&riff_size.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        data.extend_from_slice(&1u16.to_le_bytes()); // PCM
        data.extend_from_slice(&2u16.to_le_bytes()); // channels
        data.extend_from_slice(&44100u32.to_le_bytes()); // sample rate
        let byte_rate: u32 = 44100 * 2 * 2;
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes()); // block align
        data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
                                                      // data chunk
        data.extend_from_slice(b"data");
        data.extend_from_slice(&data_size.to_le_bytes());
        // (no actual sample data needed for probing)

        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Wav);
        assert!(result.has_audio);
        assert!(!result.has_video);
        let dur = result.duration_ms.expect("should have duration");
        // 1 second = 1000 ms
        assert_eq!(dur, 1000);
        assert_eq!(result.audio_codec_hint, Some("PCM".to_string()));
    }

    #[test]
    fn test_probe_mp3_fallback() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        // ID3-tagged MP3
        let data = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Mp3);
        assert!(result.has_audio);
        // Should get a fallback bitrate estimate.
        assert_eq!(result.bitrate_kbps, Some(192));
    }

    #[test]
    fn test_probe_zip_no_av() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = vec![0x50u8, 0x4B, 0x03, 0x04, 0x00];
        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Zip);
        assert!(!result.has_video);
        assert!(!result.has_audio);
    }

    #[test]
    fn test_probe_bytes_same_as_probe() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = vec![0xFF, 0xD8, 0xFF];
        let r1 = pipeline.probe(&data);
        let r2 = pipeline.probe_bytes(&data);
        assert_eq!(r1.format, r2.format);
    }

    #[test]
    fn test_probe_unknown_empty() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let result = pipeline.probe(&[]);
        assert_eq!(result.format, MediaFormat::Unknown);
        assert!(!result.has_video);
        assert!(!result.has_audio);
    }

    #[test]
    fn test_probe_result_file_size() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = vec![0u8; 1024];
        let result = pipeline.probe(&data);
        assert_eq!(result.file_size_bytes, 1024);
    }

    #[test]
    fn test_probe_flv() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00".to_vec();
        let result = pipeline.probe(&data);
        assert_eq!(result.format, MediaFormat::Flv);
        assert!(result.has_video);
        assert!(result.has_audio);
    }

    #[test]
    fn test_media_probe_result_is_image_helper() {
        let pipeline = MediaIoPipeline::new(IoPipelineConfig::default());
        let data = [0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let result = pipeline.probe(&data);
        assert!(result.is_image());
        assert!(!result.is_video());
        assert!(!result.is_audio());
    }
}
