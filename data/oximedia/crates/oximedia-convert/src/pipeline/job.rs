// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion job management.

use super::{AudioSettings, VideoSettings};
use crate::filters::FilterChain;
use crate::formats::ContainerFormat;
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// A conversion job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionJob {
    /// Job ID
    pub id: String,
    /// Input file path
    pub input: PathBuf,
    /// Output file path
    pub output: PathBuf,
    /// Output container format
    pub container: ContainerFormat,
    /// Video conversion settings
    pub video: Option<VideoSettings>,
    /// Audio conversion settings
    pub audio: Option<AudioSettings>,
    /// Filter chain to apply
    pub filters: Option<FilterChain>,
    /// Metadata to preserve/add
    pub metadata: HashMap<String, String>,
    /// Priority level
    pub priority: JobPriority,
    /// Job status
    pub status: JobStatus,
    /// Creation time
    pub created_at: SystemTime,
    /// Start time
    pub started_at: Option<SystemTime>,
    /// Completion time
    pub completed_at: Option<SystemTime>,
    /// Error message if failed
    pub error: Option<String>,
    /// Progress percentage (0-100)
    pub progress: f64,
    /// Total number of source frames, when known from a media probe.
    ///
    /// `None` until probed. Used for accurate checkpoint/resume bookkeeping so
    /// `frames_processed` reflects the real frame count rather than a guess.
    /// `#[serde(default)]` keeps queues persisted before this field was added
    /// loadable (missing → `None`).
    #[serde(default)]
    pub total_frames: Option<u64>,
}

/// Job priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JobPriority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is queued
    Queued,
    /// Job is currently processing
    Processing,
    /// Job completed successfully
    Completed,
    /// Job failed with error
    Failed,
    /// Job was cancelled
    Cancelled,
}

impl ConversionJob {
    /// Create a new conversion job.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        input: PathBuf,
        output: PathBuf,
        container: ContainerFormat,
        video: Option<VideoSettings>,
        audio: Option<AudioSettings>,
        filters: Option<FilterChain>,
        metadata: HashMap<String, String>,
        priority: JobPriority,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            input,
            output,
            container,
            video,
            audio,
            filters,
            metadata,
            priority,
            status: JobStatus::Queued,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            error: None,
            progress: 0.0,
            total_frames: None,
        }
    }

    /// Record the total number of source frames (typically from a media probe).
    ///
    /// This feeds accurate `frames_processed` / `total_frames` values into
    /// checkpoints so an interrupted conversion can resume precisely instead of
    /// relying on a placeholder frame count.
    pub fn set_total_frames(&mut self, total_frames: u64) {
        self.total_frames = Some(total_frames);
    }

    /// Builder-style variant of [`set_total_frames`](Self::set_total_frames).
    #[must_use]
    pub fn with_total_frames(mut self, total_frames: u64) -> Self {
        self.total_frames = Some(total_frames);
        self
    }

    /// Mark job as started.
    pub fn start(&mut self) {
        self.status = JobStatus::Processing;
        self.started_at = Some(SystemTime::now());
    }

    /// Mark job as completed.
    pub fn complete(&mut self) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(SystemTime::now());
        self.progress = 100.0;
    }

    /// Mark job as failed.
    pub fn fail(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.completed_at = Some(SystemTime::now());
        self.error = Some(error);
    }

    /// Update job progress.
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 100.0);
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed_time(&self) -> Option<Duration> {
        self.started_at.and_then(|start| {
            let end = self.completed_at.unwrap_or_else(SystemTime::now);
            end.duration_since(start).ok()
        })
    }

    /// Persist a checkpoint for this job to `{dir}/oximedia-checkpoints/{job_id}.json`.
    ///
    /// The checkpoint captures the current progress snapshot so a future run
    /// can resume from where the conversion was interrupted.
    pub fn save_checkpoint(&self, dir: &Path) -> Result<()> {
        let checkpoint_dir = dir.join("oximedia-checkpoints");
        std::fs::create_dir_all(&checkpoint_dir).map_err(ConversionError::Io)?;

        // Derive the frame counts from the probed total when available. A
        // `total_frames` of 0 honestly signals "unknown" rather than inventing
        // a placeholder count.
        let total_frames = self.total_frames.unwrap_or(0);
        let frames_processed = if total_frames > 0 {
            (self.progress / 100.0 * total_frames as f64).round() as u64
        } else {
            0
        };
        let checkpoint = ConversionCheckpoint {
            job_id: self.id.clone(),
            input_path: self.input.clone(),
            output_path: self.output.clone(),
            frames_processed,
            total_frames,
            byte_offset: 0,
            created_at: SystemTime::now(),
        };

        let path = checkpoint_dir.join(format!("{}.json", self.id));
        let json = serde_json::to_string_pretty(&checkpoint)
            .map_err(|e| ConversionError::InvalidOutput(format!("checkpoint serialise: {e}")))?;
        std::fs::write(&path, json.as_bytes()).map_err(ConversionError::Io)?;
        Ok(())
    }

    /// Load a checkpoint for `job_id` from `{dir}/oximedia-checkpoints/{job_id}.json`.
    ///
    /// Returns `Ok(None)` when no checkpoint file is found (fresh start).
    pub fn load_checkpoint(dir: &Path, job_id: &str) -> Result<Option<ConversionCheckpoint>> {
        let path = dir
            .join("oximedia-checkpoints")
            .join(format!("{job_id}.json"));

        if !path.exists() {
            return Ok(None);
        }

        let data = std::fs::read(&path).map_err(ConversionError::Io)?;
        let checkpoint: ConversionCheckpoint = serde_json::from_slice(&data)
            .map_err(|e| ConversionError::InvalidInput(format!("checkpoint deserialise: {e}")))?;
        Ok(Some(checkpoint))
    }

    /// Construct a [`ConversionJob`] that resumes from the given checkpoint.
    ///
    /// The returned job starts in `Queued` state with `progress` initialised
    /// proportionally from the checkpoint's frame counters.
    #[must_use]
    pub fn resume_from_checkpoint(checkpoint: ConversionCheckpoint) -> Self {
        let progress = if checkpoint.total_frames > 0 {
            (checkpoint.frames_processed as f64 / checkpoint.total_frames as f64 * 100.0)
                .clamp(0.0, 99.0)
        } else {
            0.0
        };

        Self {
            id: checkpoint.job_id.clone(),
            input: checkpoint.input_path,
            output: checkpoint.output_path,
            container: ContainerFormat::Matroska, // default; caller should override
            video: None,
            audio: None,
            filters: None,
            metadata: HashMap::new(),
            priority: JobPriority::Normal,
            status: JobStatus::Queued,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            error: None,
            progress,
            // Restore the probed frame total when the checkpoint recorded one;
            // 0 honestly means it was unknown at checkpoint time.
            total_frames: (checkpoint.total_frames > 0).then_some(checkpoint.total_frames),
        }
    }
}

// ── ConversionCheckpoint ──────────────────────────────────────────────────────

/// A snapshot of conversion progress, serialised to disk so an interrupted
/// job can be resumed later without re-encoding already-processed frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionCheckpoint {
    /// Unique identifier of the associated [`ConversionJob`].
    pub job_id: String,
    /// Absolute path of the input file.
    pub input_path: PathBuf,
    /// Absolute path of the (partial) output file.
    pub output_path: PathBuf,
    /// Number of frames (or audio blocks) already encoded.
    pub frames_processed: u64,
    /// Estimated total frames in the source.
    pub total_frames: u64,
    /// Byte offset into the output file at the time of checkpointing.
    pub byte_offset: u64,
    /// Wall-clock time when the checkpoint was written.
    #[serde(with = "system_time_serde")]
    pub created_at: SystemTime,
}

/// `serde` helpers for [`SystemTime`] (serialize as Unix timestamp seconds).
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> std::result::Result<S::Ok, S::Error> {
        let secs = t
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        s.serialize_u64(secs)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_job_creation() {
        let job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.priority, JobPriority::Normal);
        assert_eq!(job.progress, 0.0);
        assert!(job.started_at.is_none());
    }

    #[test]
    fn test_job_lifecycle() {
        let mut job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        job.start();
        assert_eq!(job.status, JobStatus::Processing);
        assert!(job.started_at.is_some());

        job.update_progress(50.0);
        assert_eq!(job.progress, 50.0);

        job.complete();
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(job.progress, 100.0);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_job_failure() {
        let mut job = ConversionJob::new(
            PathBuf::from("input.mp4"),
            PathBuf::from("output.webm"),
            ContainerFormat::Webm,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );

        job.start();
        job.fail("Test error".to_string());

        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Critical > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    // ── Checkpoint tests ──────────────────────────────────────────────────────

    #[test]
    fn test_save_and_load_checkpoint() {
        let dir = std::env::temp_dir().join("oximedia_checkpoint_test_save_load");
        let _ = std::fs::create_dir_all(&dir);

        let mut job = ConversionJob::new(
            PathBuf::from("/tmp/input.mkv"),
            PathBuf::from("/tmp/output.mkv"),
            ContainerFormat::Matroska,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        );
        job.update_progress(50.0);

        job.save_checkpoint(&dir)
            .expect("save_checkpoint should succeed");

        let loaded =
            ConversionJob::load_checkpoint(&dir, &job.id).expect("load_checkpoint should succeed");
        assert!(loaded.is_some(), "checkpoint should be found");

        let checkpoint = loaded.expect("checked above");
        assert_eq!(checkpoint.job_id, job.id);
        assert_eq!(checkpoint.input_path, PathBuf::from("/tmp/input.mkv"));
        assert_eq!(checkpoint.output_path, PathBuf::from("/tmp/output.mkv"));
        // No probe was performed → total_frames is honestly reported as unknown
        // (0), and frames_processed is 0 rather than a fabricated fraction.
        assert_eq!(checkpoint.total_frames, 0);
        assert_eq!(checkpoint.frames_processed, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_uses_probed_total_frames() {
        let dir = std::env::temp_dir().join("oximedia_checkpoint_test_total_frames");
        let _ = std::fs::create_dir_all(&dir);

        let mut job = ConversionJob::new(
            PathBuf::from("/tmp/input.mkv"),
            PathBuf::from("/tmp/output.mkv"),
            ContainerFormat::Matroska,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        )
        .with_total_frames(2_500);
        job.update_progress(40.0);

        job.save_checkpoint(&dir)
            .expect("save_checkpoint should succeed");
        let checkpoint = ConversionJob::load_checkpoint(&dir, &job.id)
            .expect("load_checkpoint should succeed")
            .expect("checkpoint should be found");

        // total_frames reflects the probed value; frames_processed is derived
        // from real progress (40% of 2500 = 1000), not a placeholder.
        assert_eq!(checkpoint.total_frames, 2_500);
        assert_eq!(checkpoint.frames_processed, 1_000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_checkpoint_missing_returns_none() {
        let dir = std::env::temp_dir().join("oximedia_checkpoint_test_missing");
        let _ = std::fs::create_dir_all(&dir);

        let result = ConversionJob::load_checkpoint(&dir, "nonexistent-job-id-xyz")
            .expect("should succeed with Ok(None)");
        assert!(result.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resume_from_checkpoint() {
        use std::time::UNIX_EPOCH;

        let checkpoint = ConversionCheckpoint {
            job_id: "resume-test-job".to_string(),
            input_path: PathBuf::from("/tmp/in.mkv"),
            output_path: PathBuf::from("/tmp/out.mkv"),
            frames_processed: 500,
            total_frames: 1_000,
            byte_offset: 256_000,
            created_at: UNIX_EPOCH,
        };

        let job = ConversionJob::resume_from_checkpoint(checkpoint);
        assert_eq!(job.id, "resume-test-job");
        assert_eq!(job.status, JobStatus::Queued);
        // 500/1000 = 50%
        assert!(
            (job.progress - 50.0).abs() < 1.0,
            "progress = {}",
            job.progress
        );
    }

    #[test]
    fn test_checkpoint_serialization_roundtrip() {
        let dir = std::env::temp_dir().join("oximedia_checkpoint_test_roundtrip");
        let _ = std::fs::create_dir_all(&dir);

        let job = ConversionJob::new(
            PathBuf::from("/tmp/media/source.webm"),
            PathBuf::from("/tmp/media/dest.mkv"),
            ContainerFormat::Matroska,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::High,
        );

        job.save_checkpoint(&dir).expect("save should succeed");

        let loaded = ConversionJob::load_checkpoint(&dir, &job.id)
            .expect("load should succeed")
            .expect("checkpoint should exist");

        // Verify the job reconstructed from checkpoint has the right identity
        let resumed = ConversionJob::resume_from_checkpoint(loaded);
        assert_eq!(resumed.id, job.id);
        assert_eq!(resumed.input, PathBuf::from("/tmp/media/source.webm"));
        assert_eq!(resumed.output, PathBuf::from("/tmp/media/dest.mkv"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
