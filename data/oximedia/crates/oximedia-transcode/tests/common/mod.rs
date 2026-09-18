//! Shared synthetic-frame helpers and mock decoder/encoder implementations
//! for the `oximedia-transcode` integration test suite.
//!
//! All helpers operate purely on in-memory data; no real media files are read.
//! Any temporary files in dependent tests use `std::env::temp_dir()`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use oximedia_transcode::{Frame, FrameDecoder, FrameEncoder};

// ── Synthetic frame helpers ───────────────────────────────────────────────────

/// Builds a synthetic YUV420 planar frame.
///
/// Layout: Y plane (W×H bytes, all `y_val`), then U plane (W/2 × H/2,
/// all 128), then V plane (W/2 × H/2, all 128).
#[must_use]
pub fn make_yuv420(width: u32, height: u32, y_val: u8, pts_ms: i64) -> Frame {
    let y_size = (width * height) as usize;
    let uv_size = y_size / 4;
    let mut data = vec![y_val; y_size];
    data.extend(vec![128u8; uv_size]); // U
    data.extend(vec![128u8; uv_size]); // V
    Frame::video(data, pts_ms, width, height)
}

// ── MockDecoder ───────────────────────────────────────────────────────────────

/// A decoder that yields a pre-loaded sequence of frames.
pub struct MockDecoder {
    frames: VecDeque<Frame>,
}

impl MockDecoder {
    /// Creates a decoder that will emit `frames` in order then report EOF.
    #[must_use]
    pub fn with_frames(frames: Vec<Frame>) -> Self {
        Self {
            frames: VecDeque::from(frames),
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

// ── ChecksumEncoder ───────────────────────────────────────────────────────────

/// A pass-through encoder that captures every encoded payload into a shared
/// `Arc<Mutex<Vec<Vec<u8>>>>` so tests can inspect exactly what was written.
///
/// `encode_frame` returns a clone of the frame's data (true pass-through),
/// pushing the same bytes into the shared capture buffer.
pub struct ChecksumEncoder {
    captured: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl ChecksumEncoder {
    /// Creates an encoder writing into the given shared capture buffer.
    #[must_use]
    pub fn new(captured: Arc<Mutex<Vec<Vec<u8>>>>) -> Self {
        Self { captured }
    }
}

impl FrameEncoder for ChecksumEncoder {
    fn encode_frame(&mut self, frame: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
        let out = frame.data.clone();
        match self.captured.lock() {
            Ok(mut guard) => guard.push(out.clone()),
            Err(poisoned) => poisoned.into_inner().push(out.clone()),
        }
        Ok(out)
    }

    fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

/// Convenience: locks the shared capture buffer and returns a clone of its
/// contents, recovering from poisoning.
#[must_use]
pub fn captured_payloads(captured: &Arc<Mutex<Vec<Vec<u8>>>>) -> Vec<Vec<u8>> {
    match captured.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}
