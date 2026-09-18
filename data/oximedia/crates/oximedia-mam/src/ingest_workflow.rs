//! MAM ingest workflow
//!
//! Provides a lightweight, in-memory ingest pipeline for MAM assets:
//! - Staged processing model (Receiving → Indexing → Approved/Rejected)
//! - Job queue with per-job progress tracking
//! - Pending and complete job counts

#![allow(dead_code)]

/// The current processing stage of an ingest job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestStage {
    /// Binary data is being received from the source
    Receiving,
    /// Asset is being transcoded to a proxy/mezzanine format
    Transcoding,
    /// Quality-control check is in progress
    QcCheck,
    /// Technical and descriptive metadata are being extracted
    MetadataExtract,
    /// Asset is being indexed for search
    Indexing,
    /// Ingest completed successfully; asset is approved
    Approved,
    /// Ingest failed or the asset was manually rejected
    Rejected,
}

impl IngestStage {
    /// Returns `true` if the job has reached a final state and will not advance further.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, IngestStage::Approved | IngestStage::Rejected)
    }

    /// Returns `true` if the job is still being actively processed.
    #[must_use]
    pub fn is_processing(&self) -> bool {
        matches!(
            self,
            IngestStage::Receiving
                | IngestStage::Transcoding
                | IngestStage::QcCheck
                | IngestStage::MetadataExtract
                | IngestStage::Indexing
        )
    }
}

/// A single ingest job tracking one source file through the pipeline.
#[derive(Debug, Clone)]
pub struct IngestJob {
    /// Unique job identifier
    pub id: u64,
    /// Absolute path to the source media file
    pub source_path: String,
    /// Asset ID assigned after initial registration, or `None` if not yet registered
    pub asset_id: Option<u64>,
    /// Current stage in the ingest pipeline
    pub stage: IngestStage,
    /// Progress within the current stage, 0.0–100.0
    pub progress_pct: f32,
    /// Unix epoch timestamp (seconds) when the job was created
    pub started_epoch: u64,
}

impl IngestJob {
    /// Advance the job to the next stage in the pipeline.
    ///
    /// Terminal stages (`Approved`, `Rejected`) are not advanced further.
    /// Progress is reset to 0.0 on each transition.
    pub fn advance_stage(&mut self) {
        self.stage = match self.stage {
            IngestStage::Receiving => IngestStage::Transcoding,
            IngestStage::Transcoding => IngestStage::QcCheck,
            IngestStage::QcCheck => IngestStage::MetadataExtract,
            IngestStage::MetadataExtract => IngestStage::Indexing,
            IngestStage::Indexing => IngestStage::Approved,
            // Terminal — no change
            IngestStage::Approved | IngestStage::Rejected => return,
        };
        self.progress_pct = 0.0;
    }

    /// Returns `true` if the job has successfully completed (`Approved`).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stage == IngestStage::Approved
    }

    /// Returns `true` if the job has failed (`Rejected`).
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.stage == IngestStage::Rejected
    }
}

/// An in-memory queue of ingest jobs.
#[derive(Debug, Default)]
pub struct IngestQueue {
    /// All submitted ingest jobs
    pub jobs: Vec<IngestJob>,
    /// Counter used to assign unique job IDs
    pub next_id: u64,
}

impl IngestQueue {
    /// Create a new, empty ingest queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new ingest job for the given source path and return its ID.
    ///
    /// The job starts in the `Receiving` stage.
    pub fn submit(&mut self, path: impl Into<String>, epoch: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.jobs.push(IngestJob {
            id,
            source_path: path.into(),
            asset_id: None,
            stage: IngestStage::Receiving,
            progress_pct: 0.0,
            started_epoch: epoch,
        });
        id
    }

    /// Look up a job by its ID.
    #[must_use]
    pub fn find_job(&self, id: u64) -> Option<&IngestJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Advance the stage of the job with the given ID.
    ///
    /// Does nothing if the ID is not found.
    pub fn advance_job(&mut self, id: u64) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            job.advance_stage();
        }
    }

    /// Returns the number of jobs that have not yet reached a terminal stage.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.stage.is_processing()).count()
    }

    /// Returns the number of jobs in the `Approved` stage.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.is_complete()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_stage_is_terminal_approved() {
        assert!(IngestStage::Approved.is_terminal());
    }

    #[test]
    fn test_ingest_stage_is_terminal_rejected() {
        assert!(IngestStage::Rejected.is_terminal());
    }

    #[test]
    fn test_ingest_stage_not_terminal_receiving() {
        assert!(!IngestStage::Receiving.is_terminal());
    }

    #[test]
    fn test_ingest_stage_is_processing_receiving() {
        assert!(IngestStage::Receiving.is_processing());
    }

    #[test]
    fn test_ingest_stage_not_processing_approved() {
        assert!(!IngestStage::Approved.is_processing());
    }

    #[test]
    fn test_submit_returns_id() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/mnt/media/clip.mxf", 1_000_000);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_find_job_found() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/clip.mxf", 100);
        assert!(queue.find_job(id).is_some());
    }

    #[test]
    fn test_find_job_not_found() {
        let queue = IngestQueue::new();
        assert!(queue.find_job(0).is_none());
    }

    #[test]
    fn test_advance_job_receiving_to_transcoding() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/clip.mxf", 100);
        queue.advance_job(id);
        let job = queue.find_job(id).expect("should succeed in test");
        assert_eq!(job.stage, IngestStage::Transcoding);
    }

    #[test]
    fn test_advance_job_full_pipeline() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/clip.mxf", 100);
        for _ in 0..5 {
            queue.advance_job(id);
        }
        let job = queue.find_job(id).expect("should succeed in test");
        assert!(job.is_complete());
        assert_eq!(job.stage, IngestStage::Approved);
    }

    #[test]
    fn test_advance_approved_does_not_change() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/clip.mxf", 100);
        for _ in 0..6 {
            queue.advance_job(id);
        }
        let job = queue.find_job(id).expect("should succeed in test");
        assert_eq!(job.stage, IngestStage::Approved);
    }

    #[test]
    fn test_pending_count() {
        let mut queue = IngestQueue::new();
        queue.submit("/a.mxf", 1);
        queue.submit("/b.mxf", 2);
        assert_eq!(queue.pending_count(), 2);
    }

    #[test]
    fn test_complete_count() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/a.mxf", 1);
        for _ in 0..5 {
            queue.advance_job(id);
        }
        assert_eq!(queue.complete_count(), 1);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_is_failed_rejected() {
        let mut job = IngestJob {
            id: 0,
            source_path: "/x.mxf".into(),
            asset_id: None,
            stage: IngestStage::Rejected,
            progress_pct: 0.0,
            started_epoch: 0,
        };
        assert!(job.is_failed());
        // advance should be no-op
        job.advance_stage();
        assert_eq!(job.stage, IngestStage::Rejected);
    }

    #[test]
    fn test_advance_job_missing_id_no_panic() {
        let mut queue = IngestQueue::new();
        // Should not panic
        queue.advance_job(999);
    }

    #[test]
    fn test_progress_reset_on_advance() {
        let mut queue = IngestQueue::new();
        let id = queue.submit("/a.mxf", 0);
        {
            let job = queue
                .jobs
                .iter_mut()
                .find(|j| j.id == id)
                .expect("should succeed in test");
            job.progress_pct = 75.0;
        }
        queue.advance_job(id);
        let job = queue.find_job(id).expect("should succeed in test");
        assert_eq!(job.progress_pct, 0.0);
    }
}
