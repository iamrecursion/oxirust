//! Proxy generation: profiles, task management, and queue.
//!
//! Provides lightweight, pure-Rust types for queuing and tracking proxy
//! generation tasks without requiring an actual encoder.

#![allow(dead_code)]
#![allow(missing_docs)]

// ---------------------------------------------------------------------------
// ProxyProfile
// ---------------------------------------------------------------------------

/// Configuration that describes how a proxy should be encoded.
#[derive(Debug, Clone, PartialEq)]
pub struct ProxyProfile {
    /// Human-readable name for this profile.
    pub name: String,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Codec identifier (e.g. "h264", "vp9").
    pub codec: String,
    /// Target frame rate.
    pub frame_rate: f32,
}

impl ProxyProfile {
    /// Create a new proxy profile.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        bitrate_kbps: u32,
        codec: impl Into<String>,
        frame_rate: f32,
    ) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            bitrate_kbps,
            codec: codec.into(),
            frame_rate,
        }
    }

    /// Standard offline-edit proxy: 1920×1080 H.264 @ 8 000 kbps / 25 fps.
    #[must_use]
    pub fn offline_edit() -> Self {
        Self::new("offline_edit", 1920, 1080, 8_000, "h264", 25.0)
    }

    /// Web preview proxy: 1280×720 H.264 @ 2 000 kbps / 25 fps.
    #[must_use]
    pub fn web_preview() -> Self {
        Self::new("web_preview", 1280, 720, 2_000, "h264", 25.0)
    }

    /// Mobile proxy: 854×480 H.264 @ 800 kbps / 25 fps.
    #[must_use]
    pub fn mobile() -> Self {
        Self::new("mobile", 854, 480, 800, "h264", 25.0)
    }
}

// ---------------------------------------------------------------------------
// ProxyStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a single proxy generation task.
#[derive(Debug, Clone, PartialEq)]
pub enum ProxyStatus {
    /// Waiting to be picked up by a worker.
    Queued,
    /// Currently being encoded; inner value is percentage complete (0–100).
    Processing(u8),
    /// Successfully finished.
    Done,
    /// Failed with an error message.
    Failed(String),
}

impl ProxyStatus {
    /// Returns `true` when the task has reached a terminal state (done or failed).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Done | Self::Failed(_))
    }

    /// Returns the progress percentage.
    ///
    /// * `Queued`      → 0
    /// * `Processing(p)` → p
    /// * `Done`        → 100
    /// * `Failed`      → 0
    #[must_use]
    pub fn progress_pct(&self) -> u8 {
        match self {
            Self::Queued => 0,
            Self::Processing(p) => *p,
            Self::Done => 100,
            Self::Failed(_) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ProxyTask
// ---------------------------------------------------------------------------

/// A single proxy generation task.
#[derive(Debug, Clone)]
pub struct ProxyTask {
    /// Path to the high-resolution source file.
    pub source_path: String,
    /// Destination path for the generated proxy.
    pub output_path: String,
    /// Encoding profile to use.
    pub profile: ProxyProfile,
    /// Current status of this task.
    pub status: ProxyStatus,
}

impl ProxyTask {
    /// Create a new task with [`ProxyStatus::Queued`] status.
    #[must_use]
    pub fn new(
        source_path: impl Into<String>,
        output_path: impl Into<String>,
        profile: ProxyProfile,
    ) -> Self {
        Self {
            source_path: source_path.into(),
            output_path: output_path.into(),
            profile,
            status: ProxyStatus::Queued,
        }
    }
}

// ---------------------------------------------------------------------------
// ProxyGenerator
// ---------------------------------------------------------------------------

/// Queue-based proxy generator that tracks tasks.
#[derive(Debug, Default)]
pub struct ProxyGenerator {
    /// All registered tasks.
    pub tasks: Vec<ProxyTask>,
}

impl ProxyGenerator {
    /// Create an empty proxy generator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a new proxy generation task.
    pub fn queue(&mut self, source: &str, output: &str, profile: ProxyProfile) {
        self.tasks.push(ProxyTask::new(source, output, profile));
    }

    /// Number of tasks that have not yet completed (Queued or Processing).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| !t.status.is_complete())
            .count()
    }

    /// Number of tasks that finished successfully.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, ProxyStatus::Done))
            .count()
    }

    /// References to all tasks that have failed.
    #[must_use]
    pub fn failed_tasks(&self) -> Vec<&ProxyTask> {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, ProxyStatus::Failed(_)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_profile_offline_edit() {
        let p = ProxyProfile::offline_edit();
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert_eq!(p.codec, "h264");
    }

    #[test]
    fn test_proxy_profile_web_preview() {
        let p = ProxyProfile::web_preview();
        assert_eq!(p.width, 1280);
        assert_eq!(p.height, 720);
    }

    #[test]
    fn test_proxy_profile_mobile() {
        let p = ProxyProfile::mobile();
        assert_eq!(p.width, 854);
        assert_eq!(p.height, 480);
        assert!(p.bitrate_kbps < 1_000);
    }

    #[test]
    fn test_proxy_status_is_complete_queued() {
        assert!(!ProxyStatus::Queued.is_complete());
    }

    #[test]
    fn test_proxy_status_is_complete_processing() {
        assert!(!ProxyStatus::Processing(50).is_complete());
    }

    #[test]
    fn test_proxy_status_is_complete_done() {
        assert!(ProxyStatus::Done.is_complete());
    }

    #[test]
    fn test_proxy_status_is_complete_failed() {
        assert!(ProxyStatus::Failed("err".to_string()).is_complete());
    }

    #[test]
    fn test_proxy_status_progress_queued() {
        assert_eq!(ProxyStatus::Queued.progress_pct(), 0);
    }

    #[test]
    fn test_proxy_status_progress_processing() {
        assert_eq!(ProxyStatus::Processing(75).progress_pct(), 75);
    }

    #[test]
    fn test_proxy_status_progress_done() {
        assert_eq!(ProxyStatus::Done.progress_pct(), 100);
    }

    #[test]
    fn test_proxy_generator_queue_pending() {
        let mut gen = ProxyGenerator::new();
        gen.queue("src.mov", "out.mp4", ProxyProfile::offline_edit());
        gen.queue("src2.mov", "out2.mp4", ProxyProfile::mobile());
        assert_eq!(gen.pending_count(), 2);
        assert_eq!(gen.complete_count(), 0);
    }

    #[test]
    fn test_proxy_generator_complete_count() {
        let mut gen = ProxyGenerator::new();
        gen.queue("src.mov", "out.mp4", ProxyProfile::offline_edit());
        gen.tasks[0].status = ProxyStatus::Done;
        assert_eq!(gen.complete_count(), 1);
        assert_eq!(gen.pending_count(), 0);
    }

    #[test]
    fn test_proxy_generator_failed_tasks() {
        let mut gen = ProxyGenerator::new();
        gen.queue("a.mov", "a.mp4", ProxyProfile::mobile());
        gen.queue("b.mov", "b.mp4", ProxyProfile::mobile());
        gen.tasks[0].status = ProxyStatus::Failed("codec error".to_string());
        let failed = gen.failed_tasks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].source_path, "a.mov");
    }
}
