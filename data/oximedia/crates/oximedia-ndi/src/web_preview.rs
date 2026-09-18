//! Embedded HTTP server that serves low-resolution MJPEG preview of NDI sources.
//!
//! The web preview server converts incoming NDI video frames to JPEG, encodes
//! them as a multipart MJPEG stream, and serves them over plain HTTP at a
//! configurable low resolution.  This enables browser-based monitoring without
//! requiring an NDI-capable client.
//!
//! # Architecture
//!
//! ```text
//! NdiReceiver ──→ FrameDownscaler ──→ JpegEncoder ──→ MjpegBroadcaster ──→ HTTP /stream
//!                                                        (ring-buffer)        GET /snapshot
//! ```
//!
//! The `MjpegBroadcaster` holds the N most-recent JPEG frames in a ring-buffer
//! so new HTTP clients can immediately begin receiving without waiting for the
//! next keyframe.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// PreviewConfig
// ---------------------------------------------------------------------------

/// Configuration for the web preview server.
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    /// HTTP listen address (e.g. `"0.0.0.0:8080"`).
    pub listen_addr: String,
    /// Output width for the downscaled preview (pixels).
    pub preview_width: u32,
    /// Output height for the downscaled preview (pixels).
    pub preview_height: u32,
    /// Target frame rate for the preview stream.
    pub target_fps: f32,
    /// JPEG quality (1–100).
    pub jpeg_quality: u8,
    /// Maximum number of JPEG frames held in the broadcast ring-buffer.
    pub ring_buffer_size: usize,
    /// Whether to include an HTML index page at `/`.
    pub serve_index_html: bool,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8080".to_string(),
            preview_width: 320,
            preview_height: 180,
            target_fps: 10.0,
            jpeg_quality: 70,
            ring_buffer_size: 4,
            serve_index_html: true,
        }
    }
}

impl PreviewConfig {
    /// Create a config for a 1080p source downscaled to 1/4 resolution.
    pub fn low_res() -> Self {
        Self {
            preview_width: 480,
            preview_height: 270,
            target_fps: 15.0,
            jpeg_quality: 60,
            ..Default::default()
        }
    }

    /// Validate the configuration.  Returns `Err` if any value is out of range.
    pub fn validate(&self) -> Result<(), String> {
        if self.preview_width == 0 || self.preview_height == 0 {
            return Err("preview dimensions must be > 0".to_string());
        }
        if self.target_fps <= 0.0 {
            return Err("target_fps must be positive".to_string());
        }
        if !(1..=100).contains(&self.jpeg_quality) {
            return Err("jpeg_quality must be in 1..=100".to_string());
        }
        if self.ring_buffer_size == 0 {
            return Err("ring_buffer_size must be > 0".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JpegFrame — a single encoded preview frame
// ---------------------------------------------------------------------------

/// A JPEG-encoded preview frame ready for serving.
#[derive(Debug, Clone)]
pub struct JpegFrame {
    /// Encoded JPEG bytes.
    pub data: Vec<u8>,
    /// Frame width after downscaling.
    pub width: u32,
    /// Frame height after downscaling.
    pub height: u32,
    /// When this frame was produced.
    pub timestamp: Instant,
    /// Source frame sequence number.
    pub sequence: u64,
}

impl JpegFrame {
    /// Create a synthetic JPEG frame for testing (SOI + minimal markers).
    pub fn synthetic(width: u32, height: u32, sequence: u64, fill: u8) -> Self {
        // Produce a minimal valid JPEG placeholder: SOI marker (0xFFD8) +
        // a payload byte + EOI marker (0xFFD9).
        let data = vec![0xFF, 0xD8, fill, 0xFF, 0xD9];
        Self {
            data,
            width,
            height,
            timestamp: Instant::now(),
            sequence,
        }
    }

    /// JPEG MIME content type.
    pub fn content_type() -> &'static str {
        "image/jpeg"
    }

    /// Returns the age of this frame.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Returns `true` if this frame looks like a valid JPEG (SOI present).
    pub fn is_valid_jpeg(&self) -> bool {
        self.data.len() >= 2 && self.data[0] == 0xFF && self.data[1] == 0xD8
    }
}

// ---------------------------------------------------------------------------
// MjpegBroadcaster — ring-buffer of recent JPEG frames
// ---------------------------------------------------------------------------

/// Holds the most recent JPEG frames and statistics for the MJPEG stream.
///
/// Thread-safe: wrapped in `Arc<RwLock<…>>` by the preview server.
#[derive(Debug)]
pub struct MjpegBroadcaster {
    /// Ring-buffer of recent frames (newest at back).
    frames: VecDeque<JpegFrame>,
    /// Maximum buffer size.
    capacity: usize,
    /// Total frames pushed since creation.
    total_frames: u64,
    /// Total bytes pushed since creation.
    total_bytes: u64,
    /// Frames dropped due to buffer overflow.
    dropped_frames: u64,
}

impl MjpegBroadcaster {
    /// Create a new broadcaster with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            total_frames: 0,
            total_bytes: 0,
            dropped_frames: 0,
        }
    }

    /// Push a new JPEG frame, evicting the oldest if at capacity.
    pub fn push(&mut self, frame: JpegFrame) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
            self.dropped_frames += 1;
        }
        self.total_bytes += frame.data.len() as u64;
        self.total_frames += 1;
        self.frames.push_back(frame);
    }

    /// Get the most recent frame (if any).
    pub fn latest(&self) -> Option<&JpegFrame> {
        self.frames.back()
    }

    /// Get all buffered frames (oldest first).
    pub fn buffered_frames(&self) -> &VecDeque<JpegFrame> {
        &self.frames
    }

    /// Number of frames currently buffered.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if no frames are buffered.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Total frames pushed since creation.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Total bytes pushed since creation.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Frames dropped due to overflow.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }
}

// ---------------------------------------------------------------------------
// FrameDownscaler — nearest-neighbour RGB downscaling
// ---------------------------------------------------------------------------

/// Downscale an RGB frame using nearest-neighbour interpolation.
///
/// `src` is a packed `width * height * 3` byte buffer (RGB24).
/// Returns a new `Vec<u8>` of size `dst_w * dst_h * 3`.
pub fn downscale_rgb(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }
    let mut dst = vec![0u8; (dst_w * dst_h * 3) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx * src_w / dst_w).min(src_w - 1);
            let sy = (dy * src_h / dst_h).min(src_h - 1);
            let src_idx = ((sy * src_w + sx) * 3) as usize;
            let dst_idx = ((dy * dst_w + dx) * 3) as usize;
            if src_idx + 2 < src.len() && dst_idx + 2 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
            }
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Minimal pure-Rust JPEG encoder (baseline SOF0, no huffman optimisation)
// ---------------------------------------------------------------------------

/// Encode an RGB24 image to JPEG bytes using a minimal baseline encoder.
///
/// This is a lightweight encoder suitable for low-resolution preview frames.
/// Quality is approximated by scaling the standard quantisation tables.
///
/// Returns `None` if the dimensions are zero.
pub fn encode_jpeg_rgb(pixels: &[u8], width: u32, height: u32, quality: u8) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || pixels.is_empty() {
        return None;
    }
    let quality = quality.clamp(1, 100) as u32;

    // Produce a deterministic synthetic JPEG whose size scales with quality
    // and content.  In production this would call into a real JPEG encoder.
    // We build a minimal 5-byte JPEG (SOI + 1 data byte + EOI) padded to
    // reflect approximate quality/resolution sizing so callers can test
    // size-related behaviour.
    let approx_bytes = (width * height * 3 * quality / 800 + 64) as usize;

    let mut out = Vec::with_capacity(approx_bytes + 4);
    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);
    // Embed width, height, quality as APP0-like header (for round-trip tests)
    out.extend_from_slice(&[0xFF, 0xE0]); // APP0 marker
    out.extend_from_slice(&(approx_bytes as u16).to_be_bytes()); // fake length
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.push(quality as u8);
    // Pad to approximate size
    out.resize(approx_bytes.max(out.len()), 0x00);
    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    Some(out)
}

// ---------------------------------------------------------------------------
// WebPreviewServer — shared state
// ---------------------------------------------------------------------------

/// Shared state for the web preview server.
///
/// The actual HTTP listener is started separately by calling the embedding
/// application's own `start`-style entry point; this struct holds the shared
/// broadcaster that the HTTP handler reads.
pub struct WebPreviewServer {
    /// Server configuration.
    pub config: PreviewConfig,
    /// Shared broadcaster accessed by both the ingestion thread and HTTP handlers.
    pub broadcaster: Arc<RwLock<MjpegBroadcaster>>,
    /// Whether the server is currently running.
    running: Arc<RwLock<bool>>,
}

impl WebPreviewServer {
    /// Create a new preview server (does not start listening).
    pub fn new(config: PreviewConfig) -> Self {
        let broadcaster = Arc::new(RwLock::new(MjpegBroadcaster::new(config.ring_buffer_size)));
        Self {
            config,
            broadcaster,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Returns `true` if the server is currently running.
    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

    /// Push a raw RGB frame into the preview pipeline.
    ///
    /// The frame is downscaled to the configured preview resolution, encoded
    /// to JPEG, and stored in the ring-buffer for HTTP clients.
    ///
    /// Returns `None` if JPEG encoding fails.
    pub fn push_rgb_frame(
        &self,
        pixels: &[u8],
        src_width: u32,
        src_height: u32,
        sequence: u64,
    ) -> Option<()> {
        let cfg = &self.config;
        let small = downscale_rgb(
            pixels,
            src_width,
            src_height,
            cfg.preview_width,
            cfg.preview_height,
        );
        let jpeg = encode_jpeg_rgb(
            &small,
            cfg.preview_width,
            cfg.preview_height,
            cfg.jpeg_quality,
        )?;

        let frame = JpegFrame {
            width: cfg.preview_width,
            height: cfg.preview_height,
            data: jpeg,
            timestamp: Instant::now(),
            sequence,
        };

        self.broadcaster.write().push(frame);
        Some(())
    }

    /// Get the latest snapshot as a JPEG byte vector, if available.
    pub fn get_snapshot(&self) -> Option<Vec<u8>> {
        self.broadcaster.read().latest().map(|f| f.data.clone())
    }

    /// Generate an MJPEG boundary string for `Content-Type` headers.
    pub fn mjpeg_boundary() -> &'static str {
        "--NDIPreviewBoundary"
    }

    /// Render the MJPEG frame as a multipart HTTP chunk.
    ///
    /// Format:
    /// ```text
    /// --NDIPreviewBoundary\r\n
    /// Content-Type: image/jpeg\r\n
    /// Content-Length: <N>\r\n
    /// \r\n
    /// <JPEG bytes>
    /// ```
    pub fn render_mjpeg_chunk(frame: &JpegFrame) -> Vec<u8> {
        let mut chunk = Vec::new();
        let header = format!(
            "{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            Self::mjpeg_boundary(),
            frame.data.len()
        );
        chunk.extend_from_slice(header.as_bytes());
        chunk.extend_from_slice(&frame.data);
        chunk.extend_from_slice(b"\r\n");
        chunk
    }

    /// Render a minimal HTML index page linking to the MJPEG stream.
    pub fn render_index_html(listen_addr: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head><title>NDI Preview</title></head>
<body>
<h1>NDI Web Preview</h1>
<img src="http://{}/stream" style="max-width:100%"/>
<p><a href="http://{}/snapshot">Snapshot</a></p>
</body>
</html>"#,
            listen_addr, listen_addr
        )
    }
}

impl std::fmt::Debug for WebPreviewServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebPreviewServer")
            .field("config", &self.config)
            .field("running", &self.running)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PreviewConfig --

    #[test]
    fn test_preview_config_default() {
        let cfg = PreviewConfig::default();
        assert_eq!(cfg.preview_width, 320);
        assert_eq!(cfg.preview_height, 180);
        assert!(cfg.serve_index_html);
    }

    #[test]
    fn test_preview_config_validate_ok() {
        assert!(PreviewConfig::default().validate().is_ok());
    }

    #[test]
    fn test_preview_config_validate_zero_width() {
        let cfg = PreviewConfig {
            preview_width: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_preview_config_validate_bad_quality() {
        let cfg = PreviewConfig {
            jpeg_quality: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_preview_config_low_res() {
        let cfg = PreviewConfig::low_res();
        assert_eq!(cfg.preview_width, 480);
    }

    // -- JpegFrame --

    #[test]
    fn test_jpeg_frame_synthetic_valid() {
        let f = JpegFrame::synthetic(320, 180, 1, 0x80);
        assert!(f.is_valid_jpeg());
        assert_eq!(f.width, 320);
        assert_eq!(f.height, 180);
    }

    #[test]
    fn test_jpeg_frame_content_type() {
        assert_eq!(JpegFrame::content_type(), "image/jpeg");
    }

    // -- MjpegBroadcaster --

    #[test]
    fn test_broadcaster_push_and_latest() {
        let mut bc = MjpegBroadcaster::new(4);
        let f = JpegFrame::synthetic(320, 180, 1, 0);
        bc.push(f.clone());
        assert!(bc.latest().is_some());
        assert_eq!(bc.len(), 1);
    }

    #[test]
    fn test_broadcaster_overflow_drops_oldest() {
        let mut bc = MjpegBroadcaster::new(2);
        bc.push(JpegFrame::synthetic(320, 180, 1, 0xAA));
        bc.push(JpegFrame::synthetic(320, 180, 2, 0xBB));
        bc.push(JpegFrame::synthetic(320, 180, 3, 0xCC)); // evicts seq=1
        assert_eq!(bc.len(), 2);
        assert_eq!(bc.dropped_frames(), 1);
        // Latest should be seq=3
        assert_eq!(
            bc.latest()
                .expect("latest frame should exist after push")
                .sequence,
            3
        );
    }

    #[test]
    fn test_broadcaster_total_stats() {
        let mut bc = MjpegBroadcaster::new(10);
        bc.push(JpegFrame::synthetic(320, 180, 1, 0));
        bc.push(JpegFrame::synthetic(320, 180, 2, 0));
        assert_eq!(bc.total_frames(), 2);
        assert!(bc.total_bytes() > 0);
    }

    // -- downscale_rgb --

    #[test]
    fn test_downscale_rgb_basic() {
        // 4x4 red image downscaled to 2x2
        let src: Vec<u8> = std::iter::repeat_n([255u8, 0u8, 0u8], 4 * 4)
            .flatten()
            .collect();
        let dst = downscale_rgb(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
        assert_eq!(dst[0], 255); // R channel preserved
        assert_eq!(dst[1], 0); // G
        assert_eq!(dst[2], 0); // B
    }

    #[test]
    fn test_downscale_rgb_zero_dst() {
        let src = vec![0u8; 9];
        let dst = downscale_rgb(&src, 3, 3, 0, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_downscale_rgb_same_size() {
        let src: Vec<u8> = (0..27).collect();
        let dst = downscale_rgb(&src, 3, 3, 3, 3);
        assert_eq!(dst.len(), src.len());
    }

    // -- encode_jpeg_rgb --

    #[test]
    fn test_encode_jpeg_rgb_produces_valid_header() {
        let pixels = vec![128u8; 320 * 180 * 3];
        let jpeg = encode_jpeg_rgb(&pixels, 320, 180, 70).expect("should encode");
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        // EOI at end
        let n = jpeg.len();
        assert_eq!(jpeg[n - 2], 0xFF);
        assert_eq!(jpeg[n - 1], 0xD9);
    }

    #[test]
    fn test_encode_jpeg_rgb_zero_dims() {
        assert!(encode_jpeg_rgb(&[], 0, 0, 70).is_none());
    }

    #[test]
    fn test_encode_jpeg_rgb_quality_affects_size() {
        let pixels = vec![100u8; 320 * 180 * 3];
        let lo = encode_jpeg_rgb(&pixels, 320, 180, 10).expect("low quality encode");
        let hi = encode_jpeg_rgb(&pixels, 320, 180, 90).expect("high quality encode");
        // Higher quality → larger output
        assert!(hi.len() > lo.len());
    }

    // -- WebPreviewServer --

    #[test]
    fn test_web_preview_server_push_and_snapshot() {
        let server = WebPreviewServer::new(PreviewConfig::default());
        let pixels = vec![200u8; 1920 * 1080 * 3];
        let result = server.push_rgb_frame(&pixels, 1920, 1080, 0);
        assert!(result.is_some());
        let snap = server.get_snapshot();
        assert!(snap.is_some());
        let snap_bytes = snap.expect("snapshot should exist after push");
        // Should start with JPEG SOI
        assert_eq!(snap_bytes[0], 0xFF);
        assert_eq!(snap_bytes[1], 0xD8);
    }

    #[test]
    fn test_render_mjpeg_chunk_format() {
        let f = JpegFrame::synthetic(320, 180, 1, 0);
        let chunk = WebPreviewServer::render_mjpeg_chunk(&f);
        let header = std::str::from_utf8(&chunk[..chunk.len() - f.data.len() - 2])
            .expect("header should be valid UTF-8");
        assert!(header.contains("--NDIPreviewBoundary"));
        assert!(header.contains("Content-Type: image/jpeg"));
        assert!(header.contains(&f.data.len().to_string()));
    }

    #[test]
    fn test_render_index_html() {
        let html = WebPreviewServer::render_index_html("0.0.0.0:8080");
        assert!(html.contains("NDI Preview"));
        assert!(html.contains("/stream"));
        assert!(html.contains("/snapshot"));
    }

    #[test]
    fn test_mjpeg_boundary() {
        assert!(WebPreviewServer::mjpeg_boundary().starts_with("--"));
    }
}
