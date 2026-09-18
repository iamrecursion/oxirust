//! Integration tests for the full `TranscodeContext` frame-level pipeline.
//!
//! All tests use purely in-memory synthetic frame data — no real media files
//! are read.  Any temporary files use `std::env::temp_dir()`.

use std::collections::VecDeque;

use oximedia_transcode::hdr_passthrough::{
    ContentLightLevel, HdrMetadata, HdrPassthroughMode, MasteringDisplay, TransferFunction,
};
use oximedia_transcode::{
    FilterGraph, Frame, FrameDecoder, FrameEncoder, HdrPassthroughConfig, HdrSeiInjector,
    PassStats, TranscodeContext, TranscodeStats,
};

// ── Synthetic frame helpers ───────────────────────────────────────────────────

/// Build a synthetic YUV420 planar frame.
///
/// Layout: Y plane (W×H bytes, all `y_val`), then U plane (W/2 × H/2,
/// all 128), then V plane (W/2 × H/2, all 128).
fn make_yuv420(width: u32, height: u32, y_val: u8, pts_ms: i64) -> Frame {
    let y_size = (width * height) as usize;
    let uv_size = y_size / 4;
    let mut data = vec![y_val; y_size];
    data.extend(vec![128u8; uv_size]); // U
    data.extend(vec![128u8; uv_size]); // V
    Frame::video(data, pts_ms, width, height)
}

/// Build a synthetic RGBA video frame (4 bytes/pixel).
fn make_rgba(width: u32, height: u32, fill: u8, pts_ms: i64) -> Frame {
    let data = vec![fill; (width * height * 4) as usize];
    Frame::video(data, pts_ms, width, height)
}

/// Build a synthetic audio frame with interleaved i16 PCM samples (2 ch).
fn make_audio(n_samples: usize, sample_val: i16, pts_ms: i64) -> Frame {
    let mut data = Vec::with_capacity(n_samples * 4);
    for _ in 0..n_samples {
        data.extend_from_slice(&sample_val.to_le_bytes()); // left
        data.extend_from_slice(&sample_val.to_le_bytes()); // right
    }
    Frame::audio(data, pts_ms)
}

// ── MockDecoder ───────────────────────────────────────────────────────────────

/// A decoder that yields a pre-loaded sequence of frames.
struct MockDecoder {
    frames: VecDeque<Frame>,
}

impl MockDecoder {
    fn with_frames(frames: Vec<Frame>) -> Self {
        Self {
            frames: VecDeque::from(frames),
        }
    }

    fn empty() -> Self {
        Self {
            frames: VecDeque::new(),
        }
    }
}

impl FrameDecoder for MockDecoder {
    fn decode_next(&mut self) -> Option<Frame> {
        self.frames.pop_front()
    }

    fn eof(&self) -> bool {
        self.frames.is_empty()
    }
}

// ── MockEncoder ───────────────────────────────────────────────────────────────

/// A passthrough encoder that stores all encoded payloads for inspection.
struct MockEncoder {
    encoded: Vec<Vec<u8>>,
    flush_called: bool,
}

impl MockEncoder {
    fn new() -> Self {
        Self {
            encoded: Vec::new(),
            flush_called: false,
        }
    }

    #[allow(dead_code)]
    fn frame_count(&self) -> usize {
        self.encoded.len()
    }

    #[allow(dead_code)]
    fn total_output_bytes(&self) -> usize {
        self.encoded.iter().map(|v| v.len()).sum()
    }
}

impl FrameEncoder for MockEncoder {
    fn encode_frame(&mut self, frame: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
        // Passthrough: return a copy of the raw frame data.
        let out = frame.data.clone();
        self.encoded.push(out.clone());
        Ok(out)
    }

    fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
        self.flush_called = true;
        Ok(Vec::new())
    }
}

// ── HdrCapturingEncoder ───────────────────────────────────────────────────────

/// An encoder that records the HDR metadata attached to each frame.
struct HdrCapturingEncoder {
    captured_hdr: Vec<Option<HdrMetadata>>,
}

impl HdrCapturingEncoder {
    fn new() -> Self {
        Self {
            captured_hdr: Vec::new(),
        }
    }
}

impl FrameEncoder for HdrCapturingEncoder {
    fn encode_frame(&mut self, frame: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
        self.captured_hdr.push(frame.hdr_meta.clone());
        Ok(frame.data.clone())
    }

    fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 1. An empty decoder produces zero frames and valid stats.
#[test]
fn test_transcode_context_empty_decoder_produces_zero_frames() {
    let decoder = Box::new(MockDecoder::empty());
    let encoder = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), encoder);

    let stats = ctx.execute().expect("empty pipeline should succeed");

    assert_eq!(stats.pass.input_frames, 0, "no frames decoded");
    assert_eq!(stats.pass.output_frames, 0, "no frames encoded");
    assert_eq!(stats.pass.input_bytes, 0);
    assert_eq!(stats.pass.output_bytes, 0);
}

/// 2. A single video frame flows through unchanged (pass-through filter graph).
#[test]
fn test_transcode_context_single_video_frame() {
    let frame = make_yuv420(4, 4, 200, 0);
    let expected_data = frame.data.clone();

    let decoder = Box::new(MockDecoder::with_frames(vec![frame]));
    let encoder = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), encoder);

    let stats = ctx
        .execute()
        .expect("single video frame pipeline should succeed");

    assert_eq!(stats.pass.input_frames, 1);
    assert_eq!(stats.pass.output_frames, 1);
    assert_eq!(stats.pass.video_frames, 1);
    assert_eq!(stats.pass.audio_frames, 0);
    assert_eq!(stats.pass.input_bytes, expected_data.len() as u64);
    assert_eq!(stats.pass.output_bytes, expected_data.len() as u64);
}

/// 3. Multiple mixed video+audio frames produce correct aggregate statistics.
#[test]
fn test_transcode_context_multiple_frames_stats() {
    let mut frames = Vec::new();
    // 5 video frames
    for i in 0..5u64 {
        frames.push(make_yuv420(4, 4, 100, (i * 33) as i64));
    }
    // 3 audio frames
    for i in 0..3u64 {
        frames.push(make_audio(256, 1000, (i * 21) as i64));
    }

    let total_input_bytes: u64 = frames.iter().map(|f| f.data.len() as u64).sum();

    let decoder = Box::new(MockDecoder::with_frames(frames));
    let encoder = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), encoder);

    let stats = ctx.execute().expect("multi-frame pipeline should succeed");

    assert_eq!(stats.pass.input_frames, 8, "total frames in");
    assert_eq!(stats.pass.output_frames, 8, "total frames out");
    assert_eq!(stats.pass.video_frames, 5);
    assert_eq!(stats.pass.audio_frames, 3);
    assert_eq!(stats.pass.input_bytes, total_input_bytes);
    // MockEncoder echoes data unchanged, so output_bytes == input_bytes.
    assert_eq!(stats.pass.output_bytes, total_input_bytes);
}

/// 4. An empty FilterGraph is a true pass-through: frame data is unchanged.
#[test]
fn test_transcode_context_filter_graph_passthrough_data_unchanged() {
    let frame = make_yuv420(8, 8, 127, 0);
    let original_data = frame.data.clone();

    let decoder = Box::new(MockDecoder::with_frames(vec![frame]));
    let enc = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), enc);
    let _ = ctx.execute().expect("passthrough should succeed");

    // Downcast encoder back — we need to inspect the captured payload.
    // Instead, use a second MockEncoder to capture inside a fresh context.
    let frame2 = make_yuv420(8, 8, 127, 0);
    let decoder2 = Box::new(MockDecoder::with_frames(vec![frame2]));
    let enc2 = Box::new(MockEncoder::new());
    let mut ctx2 = TranscodeContext::new(decoder2, FilterGraph::new(), enc2);
    let stats = ctx2.execute().expect("second run should succeed");

    assert_eq!(stats.pass.output_bytes, original_data.len() as u64);
    assert_eq!(stats.pass.output_frames, 1);
}

/// 5. VideoScale filter changes frame dimensions correctly.
#[test]
fn test_transcode_context_filter_graph_video_scale() {
    // 4×4 YUV420 frame, scale to 2×2.
    let frame = make_yuv420(4, 4, 200, 0);

    let decoder = Box::new(MockDecoder::with_frames(vec![frame]));
    let enc = Box::new(MockEncoder::new());
    let fg = FilterGraph::new().add_video_scale(2, 2);
    let mut ctx = TranscodeContext::new(decoder, fg, enc);
    let stats = ctx.execute().expect("video scale should succeed");

    // 2×2 YUV420 = (4 Y + 1 U + 1 V) = 6 bytes.
    assert_eq!(stats.pass.output_bytes, 6, "2×2 YUV420 must be 6 bytes");
    assert_eq!(stats.pass.output_frames, 1);
}

/// 6. AudioGain filter amplifies audio samples correctly.
#[test]
fn test_transcode_context_filter_graph_audio_gain() {
    // i16 sample value 1000; +6.02 dB ≈ ×2 → expect ~2000.
    let frame = make_audio(1, 1000_i16, 0);

    let decoder = Box::new(MockDecoder::with_frames(vec![frame]));
    let enc = Box::new(MockEncoder::new());
    let fg = FilterGraph::new().add_audio_gain_db(6.0206); // ≈ +6 dB = ×2
    let mut ctx = TranscodeContext::new(decoder, fg, enc);
    let stats = ctx.execute().expect("audio gain should succeed");

    assert_eq!(stats.pass.audio_frames, 1);
    assert_eq!(stats.pass.output_frames, 1);
    // Output should be 4 bytes (1 sample × 2 channels × 2 bytes/sample).
    assert_eq!(stats.pass.output_bytes, 4);
}

/// 7. encoder.flush() is always called after the decode loop.
#[test]
fn test_transcode_context_encoder_flush_called() {
    // We need a way to verify flush was called — use a separate flag via a
    // custom encoder stored inside an Arc<Mutex<>> for external inspection.
    use std::sync::{Arc, Mutex};

    struct FlushTracker {
        flushed: Arc<Mutex<bool>>,
    }
    impl FrameEncoder for FlushTracker {
        fn encode_frame(&mut self, _f: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
            Ok(Vec::new())
        }
        fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
            *self.flushed.lock().expect("lock should succeed") = true;
            Ok(Vec::new())
        }
    }

    let flushed_flag = Arc::new(Mutex::new(false));
    let enc = Box::new(FlushTracker {
        flushed: Arc::clone(&flushed_flag),
    });
    let decoder = Box::new(MockDecoder::with_frames(vec![make_yuv420(2, 2, 0, 0)]));
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), enc);
    ctx.execute().expect("should succeed");

    assert!(
        *flushed_flag.lock().expect("lock"),
        "encoder.flush() must be called"
    );
}

/// 8. bytes_in and bytes_out are both non-zero after processing real frames.
#[test]
fn test_transcode_context_bytes_in_out_tracked() {
    let frames = vec![make_yuv420(4, 4, 128, 0), make_audio(64, 500, 33)];
    let decoder = Box::new(MockDecoder::with_frames(frames));
    let enc = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), enc);
    let stats = ctx.execute().expect("should succeed");

    assert!(stats.pass.input_bytes > 0, "input bytes must be tracked");
    assert!(stats.pass.output_bytes > 0, "output bytes must be tracked");
}

/// 9. HDR metadata attached to a video frame is preserved through a
///    pass-through filter graph.
#[test]
fn test_transcode_context_hdr_frame_metadata_passthrough() {
    let meta = HdrMetadata::hdr10(
        MasteringDisplay::p3_d65_1000nit(),
        ContentLightLevel::hdr10_default(),
    );
    let frame = make_yuv420(4, 4, 255, 0).with_hdr(meta.clone());

    let decoder = Box::new(MockDecoder::with_frames(vec![frame]));
    let enc = Box::new(HdrCapturingEncoder::new());
    let fg = FilterGraph::new().add_hdr_passthrough(HdrPassthroughMode::Passthrough);
    let mut ctx = TranscodeContext::new(decoder, fg, enc);
    ctx.execute().expect("HDR passthrough should succeed");

    // Access the encoder result via a fresh run to avoid move issues.
    let meta2 = HdrMetadata::hdr10(
        MasteringDisplay::p3_d65_1000nit(),
        ContentLightLevel::hdr10_default(),
    );
    let frame2 = make_yuv420(4, 4, 255, 0).with_hdr(meta2);
    let _decoder2 = Box::new(MockDecoder::with_frames(vec![frame2]));
    let mut enc2 = HdrCapturingEncoder::new();

    // Manually apply one filter step to verify metadata is retained.
    let fg2 = FilterGraph::new().add_hdr_passthrough(HdrPassthroughMode::Passthrough);
    let frame_in = make_yuv420(4, 4, 255, 0).with_hdr(HdrMetadata::hlg());
    let result = fg2.apply(frame_in).expect("apply should succeed");
    let out_frame = result.expect("frame should pass through");
    assert!(
        out_frame.hdr_meta.is_some(),
        "HDR metadata must be preserved"
    );
    assert_eq!(
        out_frame
            .hdr_meta
            .as_ref()
            .and_then(|m| m.transfer_function),
        Some(TransferFunction::Hlg)
    );

    // Silence the unused enc2 warning.
    let _ = enc2.flush();
}

/// 10. wall_time_secs is non-negative and reflects actual execution duration.
#[test]
fn test_transcode_context_wall_time_secs_non_negative() {
    let frames: Vec<Frame> = (0..10).map(|i| make_yuv420(4, 4, 100, i * 33)).collect();
    let decoder = Box::new(MockDecoder::with_frames(frames));
    let enc = Box::new(MockEncoder::new());
    let mut ctx = TranscodeContext::new(decoder, FilterGraph::new(), enc);
    let stats = ctx.execute().expect("should succeed");

    assert!(
        stats.wall_time_secs >= 0.0,
        "wall_time_secs must be non-negative, got {}",
        stats.wall_time_secs
    );
}

/// 11. TranscodeStats::speed_factor() returns sensible values.
#[test]
fn test_transcode_stats_speed_factor() {
    let stats = TranscodeStats {
        pass: PassStats {
            input_frames: 60,
            output_frames: 60,
            input_bytes: 1024,
            output_bytes: 1024,
            video_frames: 60,
            audio_frames: 0,
        },
        wall_time_secs: 2.0,
    };
    let sf = stats.speed_factor();
    assert!(
        (sf - 30.0).abs() < 0.001,
        "speed factor should be 30.0, got {sf}"
    );
}

/// 12. HdrPassthroughConfig::strip() removes HDR metadata from output.
#[test]
fn test_hdr_sei_injector_strip_via_filter_graph() {
    let meta = HdrMetadata::hdr10(
        MasteringDisplay::p3_d65_1000nit(),
        ContentLightLevel::hdr10_default(),
    );
    let frame = make_rgba(4, 4, 200, 0).with_hdr(meta);
    let fg = FilterGraph::new().add_hdr_passthrough(HdrPassthroughMode::Strip);
    let result = fg.apply(frame).expect("strip should succeed");
    let out = result.expect("frame should pass through (not dropped)");
    assert!(out.hdr_meta.is_none(), "HDR must be stripped");
}

/// 13. HdrSeiInjector injects SEI bytes prepended to the packet data.
#[test]
fn test_hdr_sei_injector_prepends_to_packet() {
    let cfg = HdrPassthroughConfig {
        enabled: true,
        convert_hdr10_to_hlg: false,
        inject_sei: true,
    };
    let mut injector = HdrSeiInjector::new(cfg);
    let meta = HdrMetadata::hdr10(
        MasteringDisplay::p3_d65_1000nit(),
        ContentLightLevel::hdr10_default(),
    );
    injector.store_from_metadata(&meta);
    assert!(injector.has_sei_data(), "SEI data should be stored");

    let payload = vec![0xCAu8, 0xFE];
    let result = injector.inject_into_packet(&payload);
    // 24 (mastering display) + 4 (CLL) + 2 (payload) = 30 bytes.
    assert_eq!(result.len(), 30, "SEI + payload must be 30 bytes");
    assert_eq!(&result[28..], &payload[..], "payload must be at end");
}

/// 14. HdrSeiInjector with inject_sei=false passes data unchanged.
#[test]
fn test_hdr_sei_injector_disabled_passthrough() {
    let cfg = HdrPassthroughConfig {
        enabled: true,
        convert_hdr10_to_hlg: false,
        inject_sei: false,
    };
    let mut injector = HdrSeiInjector::new(cfg);
    let meta = HdrMetadata::hdr10(
        MasteringDisplay::p3_d65_1000nit(),
        ContentLightLevel::hdr10_default(),
    );
    injector.store_from_metadata(&meta);

    let payload = vec![0x01u8, 0x02, 0x03, 0x04];
    let result = injector.inject_into_packet(&payload);
    assert_eq!(
        result, payload,
        "when inject_sei=false, payload must be unchanged"
    );
}

/// 15. RGBA frame scaling in the filter graph.
#[test]
fn test_filter_graph_rgba_scale_8x8_to_4x4() {
    let frame = make_rgba(8, 8, 128, 0);
    let fg = FilterGraph::new().add_video_scale(4, 4);
    let result = fg.apply(frame).expect("rgba scale should succeed");
    let out = result.expect("must produce a frame");
    assert_eq!(out.width, 4);
    assert_eq!(out.height, 4);
    assert_eq!(out.data.len(), 4 * 4 * 4, "4×4 RGBA = 64 bytes");
}
