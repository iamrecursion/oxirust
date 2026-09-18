//! Frame-level transcode context: decoder/filter-graph/encoder pipeline.
//!
//! This module provides the `TranscodeContext` that wires together a
//! `FrameDecoder`, a `FilterGraph`, and a `FrameEncoder` into a single
//! execute-loop.  The design is codec-agnostic: callers supply concrete
//! implementations of the `FrameDecoder` and `FrameEncoder` traits.
//!
//! # Example
//!
//! ```rust,ignore
//! use oximedia_transcode::pipeline_context::{
//!     TranscodeContext, FilterGraph, Frame,
//! };
//!
//! // Supply your own decoder/encoder implementations.
//! let ctx = TranscodeContext::new(decoder, FilterGraph::new(), encoder);
//! let stats = ctx.execute()?;
//! println!("{} frames in, {} frames out", stats.pass.input_frames, stats.pass.output_frames);
//! ```

#![allow(clippy::module_name_repetitions)]

use std::time::Instant;

use crate::hdr_passthrough::{
    ColourPrimaries, HdrMetadata, HdrPassthroughMode, HdrProcessor, TransferFunction,
};
use crate::{Result, TranscodeError};

// ─── Frame ────────────────────────────────────────────────────────────────────

/// A raw decoded frame flowing through the transcode pipeline.
///
/// Frames carry either video (planar YUV pixel data) or audio (interleaved
/// i16 / f32 PCM) depending on the `is_audio` flag.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw pixel (YUV) or sample (PCM) data.
    pub data: Vec<u8>,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: i64,
    /// `true` for audio frames, `false` for video frames.
    pub is_audio: bool,
    /// Video frame width in pixels (0 for audio frames).
    pub width: u32,
    /// Video frame height in pixels (0 for audio frames).
    pub height: u32,
    /// HDR metadata attached to this video frame, if any.
    pub hdr_meta: Option<HdrMetadata>,
}

impl Frame {
    /// Creates a new video frame.
    #[must_use]
    pub fn video(data: Vec<u8>, pts_ms: i64, width: u32, height: u32) -> Self {
        Self {
            data,
            pts_ms,
            is_audio: false,
            width,
            height,
            hdr_meta: None,
        }
    }

    /// Creates a new audio frame.
    #[must_use]
    pub fn audio(data: Vec<u8>, pts_ms: i64) -> Self {
        Self {
            data,
            pts_ms,
            is_audio: true,
            width: 0,
            height: 0,
            hdr_meta: None,
        }
    }

    /// Attaches HDR metadata to a video frame (builder-style).
    #[must_use]
    pub fn with_hdr(mut self, meta: HdrMetadata) -> Self {
        self.hdr_meta = Some(meta);
        self
    }
}

// ─── Traits ───────────────────────────────────────────────────────────────────

/// A frame-level decoder that produces raw `Frame` values.
///
/// Implementations wrap a container demuxer + codec decoder.
/// When the input is exhausted `decode_next` returns `None`
/// and `eof` returns `true`.
pub trait FrameDecoder: Send {
    /// Decode the next frame from the input stream.
    ///
    /// Returns `None` when no more frames are available (end-of-stream).
    fn decode_next(&mut self) -> Option<Frame>;

    /// Returns `true` when the input stream is fully consumed.
    fn eof(&self) -> bool;
}

/// A frame-level encoder that converts raw `Frame` values to encoded bytes.
///
/// Implementations wrap a codec encoder and optional container muxer.
pub trait FrameEncoder: Send {
    /// Encode a single decoded frame.
    ///
    /// Returns the encoded byte payload (may be empty for encoders that
    /// buffer internally).
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails (e.g. invalid frame dimensions,
    /// codec state error).
    fn encode_frame(&mut self, frame: &Frame) -> Result<Vec<u8>>;

    /// Flush any internally buffered frames.
    ///
    /// Must be called at end-of-stream.  Returns any remaining encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush operation fails.
    fn flush(&mut self) -> Result<Vec<u8>>;
}

// ─── FilterOp (private) ───────────────────────────────────────────────────────

/// A single filter operation applied to a frame in the `FilterGraph`.
#[derive(Debug, Clone)]
enum FilterOp {
    /// Scale a video frame to the given resolution.
    VideoScale { width: u32, height: u32 },
    /// Apply a constant linear gain (in dB) to audio samples.
    AudioGainDb(f64),
    /// Apply HDR metadata passthrough / conversion.
    HdrPassthrough(HdrPassthroughMode),
}

// ─── FilterGraph ──────────────────────────────────────────────────────────────

/// A composable filter graph applied to frames between decode and encode.
///
/// Operations are applied in the order they were added.  Video ops are
/// skipped for audio frames, and vice-versa.
///
/// The baseline implementation is a pass-through: an empty `FilterGraph`
/// forwards every frame unchanged.
#[derive(Debug, Clone, Default)]
pub struct FilterGraph {
    ops: Vec<FilterOp>,
}

impl FilterGraph {
    /// Creates a new, empty (pass-through) filter graph.
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Adds a video scale operation (nearest-neighbour).
    #[must_use]
    pub fn add_video_scale(mut self, width: u32, height: u32) -> Self {
        self.ops.push(FilterOp::VideoScale { width, height });
        self
    }

    /// Adds an audio gain operation (dB).
    #[must_use]
    pub fn add_audio_gain_db(mut self, db: f64) -> Self {
        self.ops.push(FilterOp::AudioGainDb(db));
        self
    }

    /// Adds an HDR passthrough / conversion operation.
    #[must_use]
    pub fn add_hdr_passthrough(mut self, mode: HdrPassthroughMode) -> Self {
        self.ops.push(FilterOp::HdrPassthrough(mode));
        self
    }

    /// Apply all filter operations to `frame`.
    ///
    /// Returns `Ok(Some(frame))` to pass the frame on, or `Ok(None)` to
    /// drop it (future use — currently never drops).
    ///
    /// # Errors
    ///
    /// Returns an error if an HDR conversion is unsupported or a filter
    /// operation encounters invalid data.
    pub fn apply(&self, mut frame: Frame) -> Result<Option<Frame>> {
        for op in &self.ops {
            match op {
                FilterOp::VideoScale { width, height } => {
                    if !frame.is_audio {
                        apply_video_scale(&mut frame, *width, *height);
                    }
                }
                FilterOp::AudioGainDb(db) => {
                    if frame.is_audio {
                        apply_audio_gain_db(&mut frame, *db);
                    }
                }
                FilterOp::HdrPassthrough(mode) => {
                    if !frame.is_audio {
                        let processor = HdrProcessor::new(mode.clone());
                        let resolved = processor.process(frame.hdr_meta.as_ref()).map_err(|e| {
                            TranscodeError::CodecError(format!("HDR filter failed: {e}"))
                        })?;
                        frame.hdr_meta = resolved;
                    }
                }
            }
        }
        Ok(Some(frame))
    }
}

// ─── Internal filter helpers ──────────────────────────────────────────────────

/// Nearest-neighbour video scale on RGBA/planar data.
///
/// For YUV420 frames the data layout is: Y plane (W×H bytes) followed by
/// U plane (W/2 × H/2 bytes) and V plane (W/2 × H/2 bytes).  For other
/// layouts the function treats the data as a flat RGBA buffer (4 bytes/pixel).
fn apply_video_scale(frame: &mut Frame, dst_w: u32, dst_h: u32) {
    if dst_w == 0 || dst_h == 0 || (dst_w == frame.width && dst_h == frame.height) {
        return;
    }
    let src_w = frame.width;
    let src_h = frame.height;
    if src_w == 0 || src_h == 0 {
        return;
    }

    let y_size = (src_w * src_h) as usize;
    let uv_size = y_size / 4;
    let expected_yuv = y_size + uv_size * 2;

    if frame.data.len() == expected_yuv {
        // YUV420 planar
        let dst_y_size = (dst_w * dst_h) as usize;
        let dst_uv_size = dst_y_size / 4;
        let mut out = vec![0u8; dst_y_size + dst_uv_size * 2];

        // Scale Y plane
        scale_plane(
            &frame.data[..y_size],
            src_w,
            src_h,
            &mut out[..dst_y_size],
            dst_w,
            dst_h,
        );
        // Scale U plane
        let uv_src_w = src_w / 2;
        let uv_src_h = src_h / 2;
        let dst_uv_w = dst_w / 2;
        let dst_uv_h = dst_h / 2;
        scale_plane(
            &frame.data[y_size..y_size + uv_size],
            uv_src_w,
            uv_src_h,
            &mut out[dst_y_size..dst_y_size + dst_uv_size],
            dst_uv_w,
            dst_uv_h,
        );
        // Scale V plane
        scale_plane(
            &frame.data[y_size + uv_size..],
            uv_src_w,
            uv_src_h,
            &mut out[dst_y_size + dst_uv_size..],
            dst_uv_w,
            dst_uv_h,
        );

        frame.data = out;
        frame.width = dst_w;
        frame.height = dst_h;
    } else {
        // Assume RGBA (4 bytes/pixel) or generic planar — scale the whole buffer.
        let bytes_per_pixel = if frame.data.len() == (src_w * src_h * 4) as usize {
            4usize
        } else {
            1usize
        };

        let dst_size = (dst_w * dst_h) as usize * bytes_per_pixel;
        let mut out = vec![0u8; dst_size];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (f64::from(dx) * f64::from(src_w) / f64::from(dst_w)) as u32;
                let sy = (f64::from(dy) * f64::from(src_h) / f64::from(dst_h)) as u32;
                let src_off = ((sy * src_w + sx) as usize) * bytes_per_pixel;
                let dst_off = ((dy * dst_w + dx) as usize) * bytes_per_pixel;
                for b in 0..bytes_per_pixel {
                    if src_off + b < frame.data.len() && dst_off + b < out.len() {
                        out[dst_off + b] = frame.data[src_off + b];
                    }
                }
            }
        }

        frame.data = out;
        frame.width = dst_w;
        frame.height = dst_h;
    }
}

/// Nearest-neighbour scale of a single planar luma/chroma plane.
fn scale_plane(src: &[u8], src_w: u32, src_h: u32, dst: &mut [u8], dst_w: u32, dst_h: u32) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (f64::from(dx) * f64::from(src_w) / f64::from(dst_w)) as u32;
            let sy = (f64::from(dy) * f64::from(src_h) / f64::from(dst_h)) as u32;
            let src_idx = (sy * src_w + sx) as usize;
            let dst_idx = (dy * dst_w + dx) as usize;
            if src_idx < src.len() && dst_idx < dst.len() {
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

/// Apply a dB gain to interleaved i16 PCM LE audio data in-place.
fn apply_audio_gain_db(frame: &mut Frame, db: f64) {
    if db.abs() < 0.001 {
        return;
    }
    let linear = 10f64.powf(db / 20.0) as f32;
    let n_samples = frame.data.len() / 2;
    for i in 0..n_samples {
        let lo = frame.data[i * 2];
        let hi = frame.data[i * 2 + 1];
        let sample = i16::from_le_bytes([lo, hi]) as f32;
        let gained = (sample * linear).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        let bytes = gained.to_le_bytes();
        frame.data[i * 2] = bytes[0];
        frame.data[i * 2 + 1] = bytes[1];
    }
}

// ─── PassStats ────────────────────────────────────────────────────────────────

/// Per-pass statistics from a `TranscodeContext::execute` call.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Total frames that entered the decode→filter→encode loop.
    pub input_frames: u64,
    /// Total frames that were written by the encoder (excluding frames
    /// dropped by the filter graph).
    pub output_frames: u64,
    /// Total raw bytes consumed from decoded frames.
    pub input_bytes: u64,
    /// Total encoded bytes produced by the encoder.
    pub output_bytes: u64,
    /// Number of video frames processed.
    pub video_frames: u64,
    /// Number of audio frames processed.
    pub audio_frames: u64,
}

// ─── TranscodeStats ───────────────────────────────────────────────────────────

/// Statistics returned by `TranscodeContext::execute`.
#[derive(Debug, Clone, Default)]
pub struct TranscodeStats {
    /// Per-frame counts and byte totals.
    pub pass: PassStats,
    /// Wall-clock duration of the execute call in seconds.
    pub wall_time_secs: f64,
}

impl TranscodeStats {
    /// Compute a speed factor (input_frames / wall_time_secs).
    ///
    /// Returns 0.0 when timing data is unavailable.
    #[must_use]
    pub fn speed_factor(&self) -> f64 {
        if self.wall_time_secs > 0.0 && self.pass.input_frames > 0 {
            self.pass.input_frames as f64 / self.wall_time_secs
        } else {
            0.0
        }
    }
}

// ─── HdrPassthroughConfig ─────────────────────────────────────────────────────

/// High-level configuration for HDR metadata passthrough in the
/// transcode pipeline.
///
/// This is a simplified overlay on top of `HdrPassthroughMode`: the three
/// boolean fields map to common production requirements without requiring
/// callers to construct the full enum.
#[derive(Debug, Clone, Default)]
pub struct HdrPassthroughConfig {
    /// Enable HDR metadata passthrough.  When `false`, all HDR metadata
    /// is stripped from the output.
    pub enabled: bool,
    /// When `true` and `enabled`, convert HDR10 (PQ/ST-2084) input to
    /// HLG (ITU-R BT.2100).  The pixel-level tone-map must be handled
    /// separately by a filter op; this flag only updates the stream-level
    /// transfer-function descriptor.
    pub convert_hdr10_to_hlg: bool,
    /// When `true` and `enabled`, inject SMPTE ST 2086 mastering-display
    /// and CTA-861.3 content-light-level SEI payloads into every output
    /// packet.
    pub inject_sei: bool,
}

impl HdrPassthroughConfig {
    /// Create a simple pass-through config (no conversion, no SEI injection).
    #[must_use]
    pub fn passthrough() -> Self {
        Self {
            enabled: true,
            convert_hdr10_to_hlg: false,
            inject_sei: false,
        }
    }

    /// Create a strip config (remove all HDR metadata).
    #[must_use]
    pub fn strip() -> Self {
        Self {
            enabled: false,
            convert_hdr10_to_hlg: false,
            inject_sei: false,
        }
    }

    /// Resolve this config into an `HdrPassthroughMode` for use with
    /// `HdrProcessor`.
    #[must_use]
    pub fn to_mode(&self) -> HdrPassthroughMode {
        if !self.enabled {
            return HdrPassthroughMode::Strip;
        }
        if self.convert_hdr10_to_hlg {
            return HdrPassthroughMode::Convert {
                target_tf: TransferFunction::Hlg,
                target_primaries: ColourPrimaries::Bt2020,
            };
        }
        HdrPassthroughMode::Passthrough
    }
}

// ─── HdrSeiInjector ───────────────────────────────────────────────────────────

/// Stores SMPTE ST 2086 and CTA-861.3 SEI payloads extracted from the input
/// stream and optionally prepends them to output packet data.
pub struct HdrSeiInjector {
    config: HdrPassthroughConfig,
    /// 24-byte mastering display SEI payload when present.
    mastering_display_sei: Option<[u8; 24]>,
    /// 4-byte CLL SEI payload when present.
    cll_sei: Option<[u8; 4]>,
}

impl HdrSeiInjector {
    /// Creates a new injector with the given configuration.
    #[must_use]
    pub fn new(config: HdrPassthroughConfig) -> Self {
        Self {
            config,
            mastering_display_sei: None,
            cll_sei: None,
        }
    }

    /// Store HDR metadata from the input stream for later injection into
    /// output packets.
    pub fn store_from_metadata(&mut self, meta: &HdrMetadata) {
        if let Some(md) = &meta.mastering_display {
            self.mastering_display_sei =
                Some(crate::hdr_passthrough::encode_mastering_display_sei(md));
        }
        if let Some(cll) = &meta.content_light_level {
            self.cll_sei = Some(crate::hdr_passthrough::encode_cll_sei(cll));
        }
    }

    /// Inject stored SEI bytes prepended to `data`.
    ///
    /// When `inject_sei` is `false` or no SEI data has been stored, the
    /// original data is returned unchanged.
    #[must_use]
    pub fn inject_into_packet(&self, data: &[u8]) -> Vec<u8> {
        if !self.config.inject_sei
            || (self.mastering_display_sei.is_none() && self.cll_sei.is_none())
        {
            return data.to_vec();
        }
        let mut out = Vec::with_capacity(
            self.mastering_display_sei.as_ref().map_or(0, |s| s.len())
                + self.cll_sei.as_ref().map_or(0, |c| c.len())
                + data.len(),
        );
        if let Some(sei) = &self.mastering_display_sei {
            out.extend_from_slice(sei.as_slice());
        }
        if let Some(cll) = &self.cll_sei {
            out.extend_from_slice(cll.as_slice());
        }
        out.extend_from_slice(data);
        out
    }

    /// Resolve the output `HdrMetadata` from the input using the configured
    /// passthrough mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the HDR conversion is unsupported.
    pub fn resolve_output_metadata(
        &self,
        input: Option<&HdrMetadata>,
    ) -> Result<Option<HdrMetadata>> {
        let mode = self.config.to_mode();
        let processor = HdrProcessor::new(mode);
        processor
            .process(input)
            .map_err(|e| TranscodeError::CodecError(format!("HDR SEI resolve failed: {e}")))
    }

    /// Returns `true` when SEI injection is enabled and at least one payload
    /// has been stored.
    #[must_use]
    pub fn has_sei_data(&self) -> bool {
        self.config.inject_sei && (self.mastering_display_sei.is_some() || self.cll_sei.is_some())
    }
}

// ─── TranscodeContext ─────────────────────────────────────────────────────────

/// Wires a `FrameDecoder`, `FilterGraph`, and `FrameEncoder` together into
/// a single execute loop.
///
/// # Execute loop
///
/// ```text
/// while !decoder.eof() {
///     frame = decoder.decode_next()
///     filtered = filter_graph.apply(frame)
///     encoded  = encoder.encode_frame(filtered)
///     accumulate stats
/// }
/// encoder.flush()
/// ```
pub struct TranscodeContext {
    /// The frame decoder (source).
    pub decoder: Box<dyn FrameDecoder>,
    /// The filter graph applied between decode and encode.
    pub filter_graph: FilterGraph,
    /// The frame encoder (sink).
    pub encoder: Box<dyn FrameEncoder>,
}

impl TranscodeContext {
    /// Creates a new context.
    #[must_use]
    pub fn new(
        decoder: Box<dyn FrameDecoder>,
        filter_graph: FilterGraph,
        encoder: Box<dyn FrameEncoder>,
    ) -> Self {
        Self {
            decoder,
            filter_graph,
            encoder,
        }
    }

    /// Execute the full decode → filter → encode pipeline loop.
    ///
    /// Returns `TranscodeStats` containing per-frame counts, byte totals,
    /// and wall-clock timing.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding or filter operations fail.
    pub fn execute(&mut self) -> Result<TranscodeStats> {
        let start = Instant::now();
        let mut stats = PassStats::default();

        while !self.decoder.eof() {
            match self.decoder.decode_next() {
                Some(frame) => {
                    stats.input_bytes += frame.data.len() as u64;
                    stats.input_frames += 1;
                    if frame.is_audio {
                        stats.audio_frames += 1;
                    } else {
                        stats.video_frames += 1;
                    }

                    match self.filter_graph.apply(frame)? {
                        Some(filtered) => {
                            let encoded = self.encoder.encode_frame(&filtered)?;
                            stats.output_bytes += encoded.len() as u64;
                            stats.output_frames += 1;
                        }
                        None => {
                            // Frame was dropped by the filter graph — do not increment output_frames.
                        }
                    }
                }
                None => {
                    // Decoder returned None before reporting eof; treat as eof.
                    break;
                }
            }
        }

        // Flush any buffered frames from the encoder.
        let flushed = self.encoder.flush()?;
        stats.output_bytes += flushed.len() as u64;

        Ok(TranscodeStats {
            pass: stats,
            wall_time_secs: start.elapsed().as_secs_f64(),
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdr_passthrough::{ContentLightLevel, MasteringDisplay, TransferFunction};

    // ── Frame constructors ────────────────────────────────────────────────────

    #[test]
    fn test_frame_video_defaults() {
        let f = Frame::video(vec![0u8; 12], 42, 4, 3);
        assert!(!f.is_audio);
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 3);
        assert_eq!(f.pts_ms, 42);
        assert!(f.hdr_meta.is_none());
    }

    #[test]
    fn test_frame_audio_defaults() {
        let f = Frame::audio(vec![0u8; 16], 100);
        assert!(f.is_audio);
        assert_eq!(f.width, 0);
        assert_eq!(f.height, 0);
        assert_eq!(f.pts_ms, 100);
    }

    #[test]
    fn test_frame_with_hdr() {
        let meta = HdrMetadata::hlg();
        let f = Frame::video(vec![0u8; 4], 0, 2, 2).with_hdr(meta.clone());
        assert!(f.hdr_meta.is_some());
        assert_eq!(
            f.hdr_meta.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    // ── FilterGraph – pass-through ────────────────────────────────────────────

    #[test]
    fn test_filter_graph_empty_passthrough_video() {
        let fg = FilterGraph::new();
        let frame = Frame::video(vec![1u8, 2, 3, 4], 0, 2, 1);
        let data_before = frame.data.clone();
        let result = fg.apply(frame).expect("apply should succeed");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|f| &f.data), Some(&data_before));
    }

    #[test]
    fn test_filter_graph_empty_passthrough_audio() {
        let fg = FilterGraph::new();
        let frame = Frame::audio(vec![0x10u8, 0x00, 0x20, 0x00], 0);
        let data_before = frame.data.clone();
        let result = fg.apply(frame).expect("apply should succeed");
        assert!(result.is_some());
        assert_eq!(result.as_ref().map(|f| &f.data), Some(&data_before));
    }

    // ── FilterGraph – video scale ─────────────────────────────────────────────

    #[test]
    fn test_filter_graph_video_scale_rgba() {
        // 4×4 RGBA frame → scale to 2×2.
        let src_w = 4u32;
        let src_h = 4u32;
        let data = vec![0u8; (src_w * src_h * 4) as usize];
        let fg = FilterGraph::new().add_video_scale(2, 2);
        let frame = Frame::video(data, 0, src_w, src_h);
        let result = fg.apply(frame).expect("scale should succeed");
        let out = result.expect("should produce a frame");
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
        assert_eq!(out.data.len(), 2 * 2 * 4);
    }

    #[test]
    fn test_filter_graph_video_scale_yuv420() {
        // 4×4 YUV420 frame → scale to 2×2.
        let w = 4u32;
        let h = 4u32;
        let y_size = (w * h) as usize;
        let uv_size = y_size / 4;
        let data = vec![200u8; y_size + uv_size * 2]; // bright Y, neutral UV
        let fg = FilterGraph::new().add_video_scale(2, 2);
        let frame = Frame::video(data, 0, w, h);
        let result = fg.apply(frame).expect("yuv420 scale should succeed");
        let out = result.expect("should produce a frame");
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
        let expected_size = (2 * 2 + 2 * (1 * 1)) as usize; // 4 + 2 = 6
        assert_eq!(out.data.len(), expected_size);
    }

    #[test]
    fn test_filter_graph_video_scale_noop_same_dims() {
        let data = vec![42u8; 16 * 16 * 4];
        let fg = FilterGraph::new().add_video_scale(16, 16);
        let frame = Frame::video(data.clone(), 0, 16, 16);
        let out = fg.apply(frame).expect("noop scale").expect("frame");
        assert_eq!(out.data, data);
        assert_eq!(out.width, 16);
        assert_eq!(out.height, 16);
    }

    // ── FilterGraph – audio gain ──────────────────────────────────────────────

    #[test]
    fn test_filter_graph_audio_gain_double() {
        // +6.02 dB ≈ ×2; 1000 → ~2000.
        let sample: i16 = 1000;
        let mut data = sample.to_le_bytes().to_vec();
        data.extend_from_slice(&sample.to_le_bytes());
        let fg = FilterGraph::new().add_audio_gain_db(6.0206);
        let frame = Frame::audio(data, 0);
        let out = fg.apply(frame).expect("gain apply").expect("frame");
        let s0 = i16::from_le_bytes([out.data[0], out.data[1]]);
        assert!((s0 as i32 - 2000).abs() < 10, "expected ~2000, got {s0}");
    }

    #[test]
    fn test_filter_graph_audio_gain_zero_db_noop() {
        let sample: i16 = 5000;
        let data = sample.to_le_bytes().to_vec();
        let fg = FilterGraph::new().add_audio_gain_db(0.0);
        let frame = Frame::audio(data.clone(), 0);
        let out = fg.apply(frame).expect("0dB gain").expect("frame");
        assert_eq!(out.data, data);
    }

    #[test]
    fn test_filter_graph_audio_gain_skips_video() {
        // An audio gain op must not modify video frames.
        let data = vec![0xFFu8; 16];
        let fg = FilterGraph::new().add_audio_gain_db(20.0);
        let frame = Frame::video(data.clone(), 0, 4, 1);
        let out = fg.apply(frame).expect("skip video").expect("frame");
        assert_eq!(out.data, data);
    }

    // ── FilterGraph – HDR passthrough ─────────────────────────────────────────

    #[test]
    fn test_filter_graph_hdr_strip() {
        let meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let fg = FilterGraph::new().add_hdr_passthrough(HdrPassthroughMode::Strip);
        let frame = Frame::video(vec![0u8; 4], 0, 2, 1).with_hdr(meta);
        let out = fg.apply(frame).expect("strip hdr").expect("frame");
        assert!(out.hdr_meta.is_none(), "HDR should be stripped");
    }

    #[test]
    fn test_filter_graph_hdr_passthrough() {
        let meta = HdrMetadata::hlg();
        let fg = FilterGraph::new().add_hdr_passthrough(HdrPassthroughMode::Passthrough);
        let frame = Frame::video(vec![0u8; 4], 0, 2, 1).with_hdr(meta);
        let out = fg.apply(frame).expect("passthrough hdr").expect("frame");
        assert!(out.hdr_meta.is_some(), "HDR should be preserved");
        assert_eq!(
            out.hdr_meta.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    // ── PassStats / TranscodeStats ────────────────────────────────────────────

    #[test]
    fn test_pass_stats_default_zeroed() {
        let s = PassStats::default();
        assert_eq!(s.input_frames, 0);
        assert_eq!(s.output_frames, 0);
        assert_eq!(s.input_bytes, 0);
        assert_eq!(s.output_bytes, 0);
        assert_eq!(s.video_frames, 0);
        assert_eq!(s.audio_frames, 0);
    }

    #[test]
    fn test_transcode_stats_speed_factor_zero_when_no_time() {
        let stats = TranscodeStats {
            pass: PassStats {
                input_frames: 100,
                ..PassStats::default()
            },
            wall_time_secs: 0.0,
        };
        assert_eq!(stats.speed_factor(), 0.0);
    }

    #[test]
    fn test_transcode_stats_speed_factor_computed() {
        let stats = TranscodeStats {
            pass: PassStats {
                input_frames: 100,
                ..PassStats::default()
            },
            wall_time_secs: 2.0,
        };
        assert!((stats.speed_factor() - 50.0).abs() < 0.001);
    }

    // ── HdrPassthroughConfig ──────────────────────────────────────────────────

    #[test]
    fn test_hdr_passthrough_config_default() {
        let cfg = HdrPassthroughConfig::default();
        assert!(!cfg.enabled);
        assert!(!cfg.convert_hdr10_to_hlg);
        assert!(!cfg.inject_sei);
    }

    #[test]
    fn test_hdr_passthrough_config_strip_mode() {
        let cfg = HdrPassthroughConfig::strip();
        assert!(matches!(cfg.to_mode(), HdrPassthroughMode::Strip));
    }

    #[test]
    fn test_hdr_passthrough_config_passthrough_mode() {
        let cfg = HdrPassthroughConfig::passthrough();
        assert!(matches!(cfg.to_mode(), HdrPassthroughMode::Passthrough));
    }

    #[test]
    fn test_hdr_passthrough_config_convert_hdr10_to_hlg() {
        let cfg = HdrPassthroughConfig {
            enabled: true,
            convert_hdr10_to_hlg: true,
            inject_sei: false,
        };
        let mode = cfg.to_mode();
        match mode {
            HdrPassthroughMode::Convert { target_tf, .. } => {
                assert_eq!(target_tf, TransferFunction::Hlg);
            }
            _ => panic!("Expected Convert mode"),
        }
    }

    // ── HdrSeiInjector ────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_sei_injector_no_sei_inject_disabled() {
        let cfg = HdrPassthroughConfig {
            enabled: true,
            inject_sei: false,
            convert_hdr10_to_hlg: false,
        };
        let injector = HdrSeiInjector::new(cfg);
        let data = vec![0xAAu8, 0xBB, 0xCC];
        let result = injector.inject_into_packet(&data);
        assert_eq!(result, data);
    }

    #[test]
    fn test_hdr_sei_injector_no_sei_when_no_metadata_stored() {
        let cfg = HdrPassthroughConfig {
            enabled: true,
            inject_sei: true,
            convert_hdr10_to_hlg: false,
        };
        let injector = HdrSeiInjector::new(cfg);
        let data = vec![0x01u8, 0x02, 0x03];
        let result = injector.inject_into_packet(&data);
        // No SEI stored → returns original data.
        assert_eq!(result, data);
        assert!(!injector.has_sei_data());
    }

    #[test]
    fn test_hdr_sei_injector_stores_metadata_and_injects() {
        let cfg = HdrPassthroughConfig {
            enabled: true,
            inject_sei: true,
            convert_hdr10_to_hlg: false,
        };
        let mut injector = HdrSeiInjector::new(cfg);
        let meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        injector.store_from_metadata(&meta);
        assert!(injector.has_sei_data());

        let payload = vec![0xDEu8, 0xAD];
        let result = injector.inject_into_packet(&payload);
        // Should prepend 24 (mastering display) + 4 (CLL) = 28 bytes.
        assert_eq!(result.len(), 28 + 2);
        // Tail must be the original payload.
        assert_eq!(&result[28..], &payload[..]);
    }

    #[test]
    fn test_hdr_sei_injector_resolve_passthrough() {
        let cfg = HdrPassthroughConfig::passthrough();
        let injector = HdrSeiInjector::new(cfg);
        let meta = HdrMetadata::hlg();
        let resolved = injector
            .resolve_output_metadata(Some(&meta))
            .expect("resolve should succeed");
        assert!(resolved.is_some());
        assert_eq!(
            resolved.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }

    #[test]
    fn test_hdr_sei_injector_resolve_strip() {
        let cfg = HdrPassthroughConfig::strip();
        let injector = HdrSeiInjector::new(cfg);
        let meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let resolved = injector
            .resolve_output_metadata(Some(&meta))
            .expect("resolve should succeed");
        assert!(resolved.is_none(), "strip should produce None");
    }

    #[test]
    fn test_hdr_sei_injector_resolve_convert_hdr10_to_hlg() {
        let cfg = HdrPassthroughConfig {
            enabled: true,
            convert_hdr10_to_hlg: true,
            inject_sei: false,
        };
        let injector = HdrSeiInjector::new(cfg);
        let meta = HdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        );
        let resolved = injector
            .resolve_output_metadata(Some(&meta))
            .expect("conversion should succeed");
        assert_eq!(
            resolved.as_ref().and_then(|m| m.transfer_function),
            Some(TransferFunction::Hlg)
        );
    }
}
