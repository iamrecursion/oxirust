//! Multi-format export from a single timeline.
//!
//! Allows exporting the same timeline to multiple resolutions, codecs,
//! and formats in a single batch operation. Commonly used for
//! delivering to different platforms (e.g., YouTube 4K, Twitter 720p,
//! Instagram 1080x1080 square).

#![allow(dead_code)]

use std::collections::HashMap;

/// Unique identifier for an export profile.
pub type ProfileId = u64;

/// A preset defining resolution, codec, and format for one export target.
#[derive(Debug, Clone)]
pub struct ExportProfile {
    /// Unique ID.
    pub id: ProfileId,
    /// Human-readable name (e.g. "YouTube 4K", "Twitter 720p").
    pub name: String,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Frame rate.
    pub fps: f64,
    /// Video codec preset label (e.g. "av1-crf28").
    pub video_codec: String,
    /// Audio codec preset label (e.g. "opus-128k").
    pub audio_codec: String,
    /// Container format (e.g. "webm", "mkv", "mp4").
    pub container: String,
    /// Whether to include video.
    pub include_video: bool,
    /// Whether to include audio.
    pub include_audio: bool,
    /// Output file suffix (appended before extension).
    pub file_suffix: String,
    /// Custom metadata fields.
    pub metadata: HashMap<String, String>,
}

impl ExportProfile {
    /// Create a new export profile.
    #[must_use]
    pub fn new(id: ProfileId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            width: 1920,
            height: 1080,
            fps: 30.0,
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            include_video: true,
            include_audio: true,
            file_suffix: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = fps.max(1.0);
        self
    }

    /// Set video codec.
    #[must_use]
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = codec.into();
        self
    }

    /// Set audio codec.
    #[must_use]
    pub fn with_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codec = codec.into();
        self
    }

    /// Set container format.
    #[must_use]
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = container.into();
        self
    }

    /// Set file suffix.
    #[must_use]
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.file_suffix = suffix.into();
        self
    }

    /// Returns the aspect ratio as a float.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f64 / self.height as f64
    }

    /// Returns the pixel count (width * height).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Standard export profile presets.
pub struct ProfilePresets;

impl ProfilePresets {
    /// YouTube 4K preset.
    #[must_use]
    pub fn youtube_4k(id: ProfileId) -> ExportProfile {
        ExportProfile::new(id, "YouTube 4K")
            .with_resolution(3840, 2160)
            .with_fps(60.0)
            .with_video_codec("av1-crf28")
            .with_audio_codec("opus-256k")
            .with_suffix("_yt4k")
    }

    /// YouTube 1080p preset.
    #[must_use]
    pub fn youtube_1080p(id: ProfileId) -> ExportProfile {
        ExportProfile::new(id, "YouTube 1080p")
            .with_resolution(1920, 1080)
            .with_fps(30.0)
            .with_video_codec("av1-crf32")
            .with_audio_codec("opus-128k")
            .with_suffix("_yt1080")
    }

    /// Twitter/X 720p preset.
    #[must_use]
    pub fn twitter_720p(id: ProfileId) -> ExportProfile {
        ExportProfile::new(id, "Twitter 720p")
            .with_resolution(1280, 720)
            .with_fps(30.0)
            .with_video_codec("vp9-crf36")
            .with_audio_codec("opus-96k")
            .with_suffix("_tw720")
    }

    /// Instagram square (1080x1080) preset.
    #[must_use]
    pub fn instagram_square(id: ProfileId) -> ExportProfile {
        ExportProfile::new(id, "Instagram Square")
            .with_resolution(1080, 1080)
            .with_fps(30.0)
            .with_video_codec("av1-crf32")
            .with_audio_codec("opus-128k")
            .with_suffix("_ig_sq")
    }

    /// Audio-only preset.
    #[must_use]
    pub fn audio_only(id: ProfileId) -> ExportProfile {
        let mut profile = ExportProfile::new(id, "Audio Only")
            .with_audio_codec("opus-256k")
            .with_suffix("_audio");
        profile.include_video = false;
        profile.container = "ogg".to_string();
        profile
    }

    /// Archive quality preset.
    #[must_use]
    pub fn archive(id: ProfileId) -> ExportProfile {
        ExportProfile::new(id, "Archive")
            .with_resolution(3840, 2160)
            .with_fps(60.0)
            .with_video_codec("ffv1")
            .with_audio_codec("flac")
            .with_container("mkv")
            .with_suffix("_archive")
    }
}

/// Status of one export in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportStatus {
    /// Waiting to start.
    Pending,
    /// Currently exporting.
    InProgress,
    /// Successfully completed.
    Completed,
    /// Export failed.
    Failed,
    /// Export was cancelled.
    Cancelled,
}

impl ExportStatus {
    /// Returns `true` if the export is in a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// A single export job in a multi-export batch.
#[derive(Debug, Clone)]
pub struct ExportJob {
    /// Export profile.
    pub profile: ExportProfile,
    /// Output file path.
    pub output_path: String,
    /// Status.
    pub status: ExportStatus,
    /// Progress (0.0 to 1.0).
    pub progress: f64,
    /// Error message (if failed).
    pub error: Option<String>,
}

impl ExportJob {
    /// Create a new export job.
    #[must_use]
    pub fn new(profile: ExportProfile, output_path: String) -> Self {
        Self {
            profile,
            output_path,
            status: ExportStatus::Pending,
            progress: 0.0,
            error: None,
        }
    }

    /// Set the progress (clamped to 0.0-1.0).
    pub fn set_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
    }
}

/// Batch multi-format export manager.
#[derive(Debug, Default)]
pub struct MultiExportManager {
    /// Export jobs.
    jobs: Vec<ExportJob>,
    /// Available profiles.
    profiles: Vec<ExportProfile>,
    /// Next profile ID.
    next_profile_id: ProfileId,
}

impl MultiExportManager {
    /// Create a new multi-export manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            profiles: Vec::new(),
            next_profile_id: 1,
        }
    }

    /// Create a manager with standard presets.
    #[must_use]
    pub fn with_standard_presets() -> Self {
        let mut mgr = Self::new();
        mgr.add_profile(ProfilePresets::youtube_4k(mgr.next_profile_id));
        mgr.add_profile(ProfilePresets::youtube_1080p(mgr.next_profile_id));
        mgr.add_profile(ProfilePresets::twitter_720p(mgr.next_profile_id));
        mgr.add_profile(ProfilePresets::instagram_square(mgr.next_profile_id));
        mgr.add_profile(ProfilePresets::audio_only(mgr.next_profile_id));
        mgr.add_profile(ProfilePresets::archive(mgr.next_profile_id));
        mgr
    }

    /// Add an export profile.
    pub fn add_profile(&mut self, mut profile: ExportProfile) {
        profile.id = self.next_profile_id;
        self.next_profile_id += 1;
        self.profiles.push(profile);
    }

    /// Get all profiles.
    #[must_use]
    pub fn profiles(&self) -> &[ExportProfile] {
        &self.profiles
    }

    /// Queue an export job for a given profile.
    pub fn queue_export(&mut self, profile_id: ProfileId, base_output: &str) -> Option<usize> {
        let profile = self.profiles.iter().find(|p| p.id == profile_id)?.clone();
        let output_path = format!(
            "{}{}.{}",
            base_output, profile.file_suffix, profile.container
        );
        let job = ExportJob::new(profile, output_path);
        let index = self.jobs.len();
        self.jobs.push(job);
        Some(index)
    }

    /// Queue exports for all profiles.
    pub fn queue_all(&mut self, base_output: &str) -> Vec<usize> {
        let profile_ids: Vec<ProfileId> = self.profiles.iter().map(|p| p.id).collect();
        let mut indices = Vec::new();
        for id in profile_ids {
            if let Some(idx) = self.queue_export(id, base_output) {
                indices.push(idx);
            }
        }
        indices
    }

    /// Get all queued jobs.
    #[must_use]
    pub fn jobs(&self) -> &[ExportJob] {
        &self.jobs
    }

    /// Get a mutable job by index.
    pub fn get_job_mut(&mut self, index: usize) -> Option<&mut ExportJob> {
        self.jobs.get_mut(index)
    }

    /// Get the overall progress across all jobs.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_progress(&self) -> f64 {
        if self.jobs.is_empty() {
            return 0.0;
        }
        let total: f64 = self.jobs.iter().map(|j| j.progress).sum();
        total / self.jobs.len() as f64
    }

    /// Get count of completed jobs.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == ExportStatus::Completed)
            .count()
    }

    /// Check if all jobs are done (terminal state).
    #[must_use]
    pub fn all_done(&self) -> bool {
        !self.jobs.is_empty() && self.jobs.iter().all(|j| j.status.is_terminal())
    }

    /// Clear all jobs.
    pub fn clear_jobs(&mut self) {
        self.jobs.clear();
    }

    /// Get total job count.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_profile_defaults() {
        let p = ExportProfile::new(1, "Test");
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert!((p.fps - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_export_profile_aspect_ratio() {
        let p = ExportProfile::new(1, "HD").with_resolution(1920, 1080);
        let ar = p.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_export_profile_pixel_count() {
        let p = ExportProfile::new(1, "4K").with_resolution(3840, 2160);
        assert_eq!(p.pixel_count(), 3840 * 2160);
    }

    #[test]
    fn test_export_profile_zero_height() {
        let p = ExportProfile::new(1, "Bad").with_resolution(100, 0);
        assert!((p.aspect_ratio()).abs() < 1e-9);
    }

    #[test]
    fn test_standard_presets() {
        let yt4k = ProfilePresets::youtube_4k(1);
        assert_eq!(yt4k.width, 3840);
        assert_eq!(yt4k.height, 2160);

        let tw = ProfilePresets::twitter_720p(2);
        assert_eq!(tw.width, 1280);
        assert_eq!(tw.height, 720);

        let ig = ProfilePresets::instagram_square(3);
        assert_eq!(ig.width, 1080);
        assert_eq!(ig.height, 1080);

        let audio = ProfilePresets::audio_only(4);
        assert!(!audio.include_video);

        let archive = ProfilePresets::archive(5);
        assert_eq!(archive.container, "mkv");
    }

    #[test]
    fn test_export_status() {
        assert!(ExportStatus::Completed.is_terminal());
        assert!(ExportStatus::Failed.is_terminal());
        assert!(ExportStatus::Cancelled.is_terminal());
        assert!(!ExportStatus::Pending.is_terminal());
        assert!(!ExportStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_export_job_progress() {
        let profile = ExportProfile::new(1, "Test");
        let mut job = ExportJob::new(profile, "/out/test.webm".to_string());
        assert!((job.progress).abs() < 1e-9);
        job.set_progress(0.5);
        assert!((job.progress - 0.5).abs() < 1e-9);
        job.set_progress(1.5);
        assert!((job.progress - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_multi_export_manager() {
        let mut mgr = MultiExportManager::new();
        mgr.add_profile(ProfilePresets::youtube_1080p(0));
        mgr.add_profile(ProfilePresets::twitter_720p(0));
        assert_eq!(mgr.profiles().len(), 2);
    }

    #[test]
    fn test_queue_export() {
        let mut mgr = MultiExportManager::with_standard_presets();
        let profile_id = mgr.profiles()[0].id;
        let idx = mgr.queue_export(profile_id, "/out/video");
        assert!(idx.is_some());
        assert_eq!(mgr.job_count(), 1);

        // Non-existent profile
        assert!(mgr.queue_export(999, "/out/video").is_none());
    }

    #[test]
    fn test_queue_all() {
        let mut mgr = MultiExportManager::with_standard_presets();
        let indices = mgr.queue_all("/out/project");
        assert_eq!(indices.len(), mgr.profiles().len());
        assert_eq!(mgr.job_count(), mgr.profiles().len());
    }

    #[test]
    fn test_overall_progress() {
        let mut mgr = MultiExportManager::new();
        assert!((mgr.overall_progress()).abs() < 1e-9);

        mgr.add_profile(ExportProfile::new(0, "A"));
        mgr.add_profile(ExportProfile::new(0, "B"));
        mgr.queue_all("/out/test");

        mgr.get_job_mut(0).expect("job exists").set_progress(0.5);
        mgr.get_job_mut(1).expect("job exists").set_progress(1.0);
        assert!((mgr.overall_progress() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_all_done() {
        let mut mgr = MultiExportManager::new();
        mgr.add_profile(ExportProfile::new(0, "A"));
        mgr.queue_all("/out/test");
        assert!(!mgr.all_done());

        mgr.get_job_mut(0).expect("job exists").status = ExportStatus::Completed;
        assert!(mgr.all_done());
    }

    #[test]
    fn test_completed_count() {
        let mut mgr = MultiExportManager::new();
        mgr.add_profile(ExportProfile::new(0, "A"));
        mgr.add_profile(ExportProfile::new(0, "B"));
        mgr.queue_all("/out/test");

        assert_eq!(mgr.completed_count(), 0);
        mgr.get_job_mut(0).expect("job exists").status = ExportStatus::Completed;
        assert_eq!(mgr.completed_count(), 1);
    }

    #[test]
    fn test_clear_jobs() {
        let mut mgr = MultiExportManager::new();
        mgr.add_profile(ExportProfile::new(0, "A"));
        mgr.queue_all("/out/test");
        mgr.clear_jobs();
        assert_eq!(mgr.job_count(), 0);
    }
}
