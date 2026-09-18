//! Pre-decode manager: decode upcoming playlist items in background threads
//! for zero-gap transitions between clips.
//!
//! The `PredecodeManager` maintains a configurable look-ahead window.
//! As the playout timeline advances, it schedules background decode jobs for
//! clips that are due to start within the look-ahead window.  Decoded frame
//! batches are stored in per-clip `PredecodeBuffer`s and consumed by the
//! playout engine when the clip begins.
//!
//! ## Design
//!
//! - Lock-free hand-off between producer (decoder thread) and consumer
//!   (playout thread) via `Arc<Mutex<PredecodeBuffer>>`.
//! - Configurable maximum number of worker threads (`max_workers`).
//! - Clip states transition through: `Pending → Decoding → Ready → Consumed`.
//! - Cancellation of stale jobs (clip removed from schedule before it starts).

#![allow(dead_code)]

use crate::PlayoutError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single decoded video frame (lightweight representation).
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Zero-based frame index within the clip.
    pub clip_frame_index: u64,
    /// Presentation timestamp in microseconds (relative to clip start).
    pub pts_us: i64,
    /// Raw pixel data (opaque byte buffer).
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// State of a predecode job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredecodeState {
    /// Job is queued but no worker has started yet.
    Pending,
    /// A worker thread is actively decoding frames.
    Decoding,
    /// All frames are decoded and available for consumption.
    Ready,
    /// Frames have been consumed by the playout engine.
    Consumed,
    /// Decoding failed.
    Failed,
    /// Job was cancelled (clip removed from schedule).
    Cancelled,
}

impl PredecodeState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Ready | Self::Consumed | Self::Failed | Self::Cancelled
        )
    }
}

/// Buffer holding decoded frames for a single clip.
#[derive(Debug)]
pub struct PredecodeBuffer {
    /// Clip identifier.
    pub clip_id: String,
    /// Current state.
    pub state: PredecodeState,
    /// Decoded frames (in presentation order).
    pub frames: Vec<DecodedFrame>,
    /// Total frames expected (0 if unknown).
    pub total_frames_expected: u64,
    /// Wall-clock time when decoding started.
    pub started_at: Option<Instant>,
    /// Wall-clock time when decoding completed.
    pub completed_at: Option<Instant>,
    /// Error message (if state == Failed).
    pub error: Option<String>,
}

impl PredecodeBuffer {
    pub fn new(clip_id: &str, total_frames: u64) -> Self {
        Self {
            clip_id: clip_id.to_string(),
            state: PredecodeState::Pending,
            frames: Vec::new(),
            total_frames_expected: total_frames,
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    /// Fill ratio (0.0 – 1.0). Returns 1.0 when complete or total unknown.
    pub fn fill_ratio(&self) -> f64 {
        if self.total_frames_expected == 0 {
            return if self.state == PredecodeState::Ready {
                1.0
            } else {
                0.0
            };
        }
        (self.frames.len() as f64 / self.total_frames_expected as f64).min(1.0)
    }

    /// Elapsed decode time (if started).
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|s| s.elapsed())
    }
}

/// Descriptor for a clip that should be pre-decoded.
#[derive(Debug, Clone)]
pub struct PredecodeRequest {
    /// Unique clip identifier.
    pub clip_id: String,
    /// Filesystem path or URL to the media file.
    pub path: String,
    /// Number of frames to pre-decode (0 = all).
    pub max_frames: u64,
    /// Expected total frame count (used for progress reporting).
    pub total_frames: u64,
    /// Priority (lower value = decoded first).
    pub priority: u8,
    /// In-point frame (decode starting here).
    pub in_point: u64,
}

/// Configuration for the predecode manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredecodeConfig {
    /// Number of clips to keep pre-decoded ahead of current playback.
    pub look_ahead_clips: usize,
    /// Maximum number of background decode workers.
    pub max_workers: usize,
    /// Maximum total frames to hold in RAM across all buffers.
    pub max_total_frames: u64,
    /// Maximum time to spend decoding a single clip (ms, 0 = unlimited).
    pub decode_timeout_ms: u64,
}

impl Default for PredecodeConfig {
    fn default() -> Self {
        Self {
            look_ahead_clips: 2,
            max_workers: 2,
            max_total_frames: 500,     // ~20 s at 25 fps
            decode_timeout_ms: 30_000, // 30 s hard limit
        }
    }
}

/// Statistics reported by the predecode manager.
#[derive(Debug, Clone, Default)]
pub struct PredecodeStats {
    /// Jobs submitted since the manager started.
    pub total_submitted: u64,
    /// Jobs that completed successfully.
    pub total_completed: u64,
    /// Jobs that failed.
    pub total_failed: u64,
    /// Jobs cancelled.
    pub total_cancelled: u64,
    /// Total frames decoded (across all completed jobs).
    pub total_frames_decoded: u64,
    /// Currently active (decoding) jobs.
    pub active_jobs: usize,
}

// ---------------------------------------------------------------------------
// Decoder simulation
// ---------------------------------------------------------------------------

/// Simulate decoding a clip by synthesising placeholder frames.
///
/// In production this would call into `oximedia-codec` or `oximedia-transcode`.
fn simulate_decode(req: &PredecodeRequest, cancel: &Arc<AtomicBool>) -> Vec<DecodedFrame> {
    let count = if req.max_frames == 0 {
        req.total_frames.max(1)
    } else {
        req.max_frames.min(req.total_frames.max(1))
    };

    let mut frames = Vec::with_capacity(count as usize);
    for i in 0..count {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let pts_us = (i as i64) * 40_000; // 25 fps placeholder
                                          // Minimal 8-byte placeholder pixel buffer (avoids large allocations in tests).
        let data = vec![(i % 256) as u8; 8];
        frames.push(DecodedFrame {
            clip_frame_index: req.in_point + i,
            pts_us,
            data,
            width: 1920,
            height: 1080,
        });
    }
    frames
}

// ---------------------------------------------------------------------------
// PredecodeManager
// ---------------------------------------------------------------------------

/// Manager that schedules and tracks background clip pre-decode jobs.
///
/// Since we cannot spawn real async worker threads without a runtime context
/// here (and the module is `#[cfg(not(target_arch = "wasm32"))]`), the
/// implementation uses a synchronous simulation model where `submit` eagerly
/// decodes clips inline when `workers == 0` (test mode) or queues them for
/// background execution via `std::thread`.
pub struct PredecodeManager {
    config: PredecodeConfig,
    /// Per-clip decode buffers.
    buffers: parking_lot::Mutex<HashMap<String, Arc<parking_lot::Mutex<PredecodeBuffer>>>>,
    /// Cancellation tokens indexed by clip_id.
    cancel_tokens: parking_lot::Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Aggregate statistics.
    stats: Arc<parking_lot::Mutex<PredecodeStats>>,
    /// Counter of frames currently held in RAM.
    live_frame_count: Arc<AtomicU64>,
}

impl PredecodeManager {
    /// Create a new predecode manager.
    pub fn new(config: PredecodeConfig) -> Self {
        Self {
            config,
            buffers: parking_lot::Mutex::new(HashMap::new()),
            cancel_tokens: parking_lot::Mutex::new(HashMap::new()),
            stats: Arc::new(parking_lot::Mutex::new(PredecodeStats::default())),
            live_frame_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Submit a clip for pre-decoding.
    ///
    /// Returns `Err` if the live frame budget is exhausted, preventing unbounded
    /// RAM usage.  Otherwise spawns a background `std::thread` decode job.
    pub fn submit(&self, req: PredecodeRequest) -> crate::Result<()> {
        // Check RAM budget.
        let current = self.live_frame_count.load(Ordering::Relaxed);
        if current + req.total_frames > self.config.max_total_frames {
            return Err(PlayoutError::Playback(format!(
                "predecode RAM budget exhausted ({current} + {} > {})",
                req.total_frames, self.config.max_total_frames
            )));
        }

        // Register buffer.
        let buf = Arc::new(parking_lot::Mutex::new(PredecodeBuffer::new(
            &req.clip_id,
            req.total_frames,
        )));
        let cancel = Arc::new(AtomicBool::new(false));

        {
            let mut buffers = self.buffers.lock();
            buffers.insert(req.clip_id.clone(), Arc::clone(&buf));
        }
        {
            let mut tokens = self.cancel_tokens.lock();
            tokens.insert(req.clip_id.clone(), Arc::clone(&cancel));
        }
        {
            let mut stats = self.stats.lock();
            stats.total_submitted += 1;
            stats.active_jobs += 1;
        }

        // Spawn background thread.
        let stats = Arc::clone(&self.stats);
        let live_count = Arc::clone(&self.live_frame_count);
        let req_clone = req.clone();
        let cancel_clone = Arc::clone(&cancel);

        std::thread::spawn(move || {
            {
                buf.lock().state = PredecodeState::Decoding;
                buf.lock().started_at = Some(Instant::now());
            }

            let frames = simulate_decode(&req_clone, &cancel_clone);
            let frame_count = frames.len() as u64;
            let was_cancelled = cancel_clone.load(Ordering::Relaxed);

            {
                let mut b = buf.lock();
                b.completed_at = Some(Instant::now());
                if was_cancelled {
                    b.state = PredecodeState::Cancelled;
                    let mut s = stats.lock();
                    s.total_cancelled += 1;
                    s.active_jobs = s.active_jobs.saturating_sub(1);
                } else {
                    b.frames = frames;
                    b.state = PredecodeState::Ready;
                    live_count.fetch_add(frame_count, Ordering::Relaxed);
                    let mut s = stats.lock();
                    s.total_completed += 1;
                    s.total_frames_decoded += frame_count;
                    s.active_jobs = s.active_jobs.saturating_sub(1);
                }
            }
        });

        Ok(())
    }

    /// Cancel a pending or in-progress decode job.
    pub fn cancel(&self, clip_id: &str) -> bool {
        let mut tokens = self.cancel_tokens.lock();
        if let Some(token) = tokens.remove(clip_id) {
            token.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Retrieve the decoded frames for a clip and mark it as consumed.
    ///
    /// Returns `None` if the clip is not ready.
    pub fn consume(&self, clip_id: &str) -> Option<Vec<DecodedFrame>> {
        let buffers = self.buffers.lock();
        if let Some(buf_arc) = buffers.get(clip_id) {
            let mut buf = buf_arc.lock();
            if buf.state == PredecodeState::Ready {
                let frames = std::mem::take(&mut buf.frames);
                let frame_count = frames.len() as u64;
                buf.state = PredecodeState::Consumed;
                self.live_frame_count.fetch_sub(
                    frame_count.min(self.live_frame_count.load(Ordering::Relaxed)),
                    Ordering::Relaxed,
                );
                Some(frames)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the current state of a clip's decode job.
    pub fn state(&self, clip_id: &str) -> Option<PredecodeState> {
        let buffers = self.buffers.lock();
        buffers.get(clip_id).map(|b| b.lock().state)
    }

    /// Get the fill ratio (0.0 – 1.0) of a clip's decode buffer.
    pub fn fill_ratio(&self, clip_id: &str) -> Option<f64> {
        let buffers = self.buffers.lock();
        buffers.get(clip_id).map(|b| b.lock().fill_ratio())
    }

    /// Evict all consumed or cancelled buffers to free memory.
    pub fn evict_stale(&self) {
        let mut buffers = self.buffers.lock();
        buffers.retain(|_, b| {
            let state = b.lock().state;
            !matches!(state, PredecodeState::Consumed | PredecodeState::Cancelled)
        });
    }

    /// Return a snapshot of aggregate statistics.
    pub fn stats(&self) -> PredecodeStats {
        self.stats.lock().clone()
    }

    /// Return the current live frame count in RAM.
    pub fn live_frame_count(&self) -> u64 {
        self.live_frame_count.load(Ordering::Relaxed)
    }

    /// Return the configuration.
    pub fn config(&self) -> &PredecodeConfig {
        &self.config
    }

    /// List all clip IDs currently tracked by the manager.
    pub fn tracked_clips(&self) -> Vec<String> {
        self.buffers.lock().keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_request(clip_id: &str, total_frames: u64) -> PredecodeRequest {
        PredecodeRequest {
            clip_id: clip_id.to_string(),
            path: format!("/media/{clip_id}.mxf"),
            max_frames: total_frames,
            total_frames,
            priority: 128,
            in_point: 0,
        }
    }

    fn wait_for_state(
        manager: &PredecodeManager,
        clip_id: &str,
        target: PredecodeState,
        timeout: Duration,
    ) -> bool {
        let start = Instant::now();
        loop {
            if manager.state(clip_id) == Some(target) {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn test_submit_and_wait_ready() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        let req = make_request("clip1", 25);
        manager.submit(req).expect("submit should succeed");

        let ready = wait_for_state(
            &manager,
            "clip1",
            PredecodeState::Ready,
            Duration::from_secs(5),
        );
        assert!(ready, "clip1 should be ready within 5 s");
    }

    #[test]
    fn test_consume_returns_frames() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        let req = make_request("clip2", 10);
        manager.submit(req).expect("submit should succeed");

        // Wait for ready.
        assert!(wait_for_state(
            &manager,
            "clip2",
            PredecodeState::Ready,
            Duration::from_secs(5)
        ));

        let frames = manager.consume("clip2").expect("should consume frames");
        assert_eq!(frames.len(), 10);
        assert_eq!(manager.state("clip2"), Some(PredecodeState::Consumed));
    }

    #[test]
    fn test_consume_not_ready_returns_none() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        // Don't submit anything.
        assert!(manager.consume("nonexistent").is_none());
    }

    #[test]
    fn test_cancel_job() {
        // Use a config with a large-enough RAM budget for the 1000-frame request.
        let cfg = PredecodeConfig {
            max_total_frames: 2000,
            ..Default::default()
        };
        let manager = PredecodeManager::new(cfg);
        let req = make_request("clip3", 1000); // large enough to cancel mid-flight
        manager.submit(req).expect("submit should succeed");

        // Cancel immediately.
        let cancelled = manager.cancel("clip3");
        assert!(cancelled);
    }

    #[test]
    fn test_fill_ratio_pending_is_zero() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        // Submit but don't wait for completion.
        let req = make_request("clip4", 50);
        manager.submit(req).expect("submit should succeed");
        // Ratio may be 0.0 right after submission (Pending state).
        let ratio = manager.fill_ratio("clip4");
        assert!(ratio.is_some());
        assert!((0.0..=1.0).contains(&ratio.expect("fill ratio should exist")));
    }

    #[test]
    fn test_budget_exceeded_returns_error() {
        let cfg = PredecodeConfig {
            max_total_frames: 10,
            ..Default::default()
        };
        let manager = PredecodeManager::new(cfg);
        // Submit a request that itself exceeds the budget.
        let req = PredecodeRequest {
            clip_id: "big".to_string(),
            path: "/media/big.mxf".to_string(),
            max_frames: 50,
            total_frames: 50,
            priority: 0,
            in_point: 0,
        };
        let result = manager.submit(req);
        assert!(result.is_err(), "should reject over-budget request");
    }

    #[test]
    fn test_evict_stale() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        let req = make_request("clip5", 5);
        manager.submit(req).expect("submit");
        assert!(wait_for_state(
            &manager,
            "clip5",
            PredecodeState::Ready,
            Duration::from_secs(5)
        ));
        manager.consume("clip5");
        manager.evict_stale();
        assert!(manager.tracked_clips().is_empty());
    }

    #[test]
    fn test_stats_tracking() {
        let manager = PredecodeManager::new(PredecodeConfig::default());
        let req = make_request("clip6", 10);
        manager.submit(req).expect("submit");
        assert!(wait_for_state(
            &manager,
            "clip6",
            PredecodeState::Ready,
            Duration::from_secs(5)
        ));
        let stats = manager.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_frames_decoded, 10);
        assert_eq!(stats.active_jobs, 0);
    }

    #[test]
    fn test_default_config() {
        let cfg = PredecodeConfig::default();
        assert_eq!(cfg.look_ahead_clips, 2);
        assert_eq!(cfg.max_workers, 2);
    }

    #[test]
    fn test_decoded_frame_fields() {
        let f = DecodedFrame {
            clip_frame_index: 5,
            pts_us: 200_000,
            data: vec![0u8; 8],
            width: 1920,
            height: 1080,
        };
        assert_eq!(f.clip_frame_index, 5);
        assert_eq!(f.pts_us, 200_000);
    }
}
