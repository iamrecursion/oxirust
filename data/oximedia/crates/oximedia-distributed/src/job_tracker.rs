//! Distributed job tracking.
//!
//! Provides a lifecycle-aware store for distributed encoding jobs with
//! progress percentage tracking and ETA estimation.

/// State machine for a distributed job.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum JobState {
    Queued,
    Assigned { worker_id: u64 },
    Running { progress_pct: f32 },
    Completed { output: String },
    Failed { error: String },
    Cancelled,
}

impl JobState {
    fn is_completed(&self) -> bool {
        matches!(self, JobState::Completed { .. })
    }

    fn is_failed(&self) -> bool {
        matches!(self, JobState::Failed { .. })
    }

    fn is_queued(&self) -> bool {
        matches!(self, JobState::Queued)
    }

    fn is_running(&self) -> bool {
        matches!(self, JobState::Running { .. })
    }
}

/// Progress snapshot used for ETA estimation.
#[derive(Debug, Clone)]
struct ProgressSample {
    /// Progress percentage at the time of sampling (0.0–100.0).
    progress_pct: f32,
    /// Timestamp of the sample (milliseconds since epoch).
    timestamp_ms: u64,
}

/// A single distributed encoding job with progress and ETA tracking.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DistributedJob {
    pub id: u64,
    pub name: String,
    pub state: JobState,
    pub created_at: u64,
    pub updated_at: u64,
    pub priority: i32,
    /// Time the job was first started (ms since epoch), set on first progress update.
    started_at_ms: Option<u64>,
    /// Ring buffer of recent progress samples for ETA rolling average.
    progress_samples: Vec<ProgressSample>,
}

impl DistributedJob {
    /// Create a new job in the `Queued` state.
    #[must_use]
    pub fn new(id: u64, name: &str, priority: i32, now: u64) -> Self {
        Self {
            id,
            name: name.to_string(),
            state: JobState::Queued,
            created_at: now,
            updated_at: now,
            priority,
            started_at_ms: None,
            progress_samples: Vec::new(),
        }
    }

    /// Transition to `Assigned`.
    pub fn assign(&mut self, worker_id: u64, now: u64) {
        self.state = JobState::Assigned { worker_id };
        self.updated_at = now;
    }

    /// Transition to `Running` with the given progress percentage.
    ///
    /// Stores a progress sample for ETA computation and sets `started_at_ms`
    /// on the first call.
    pub fn update_progress(&mut self, pct: f32, now: u64) {
        let clamped = pct.clamp(0.0, 100.0);
        // Record start time on first progress update
        if self.started_at_ms.is_none() {
            self.started_at_ms = Some(now);
        }
        // Keep up to the last 8 samples for a rolling average
        const MAX_SAMPLES: usize = 8;
        self.progress_samples.push(ProgressSample {
            progress_pct: clamped,
            timestamp_ms: now,
        });
        if self.progress_samples.len() > MAX_SAMPLES {
            self.progress_samples.remove(0);
        }
        self.state = JobState::Running {
            progress_pct: clamped,
        };
        self.updated_at = now;
    }

    /// Current progress as a percentage (0.0–100.0).
    ///
    /// Returns `0.0` for jobs not yet in the `Running` state.
    #[must_use]
    pub fn progress_pct(&self) -> f32 {
        match self.state {
            JobState::Running { progress_pct } => progress_pct,
            JobState::Completed { .. } => 100.0,
            _ => 0.0,
        }
    }

    /// Estimated time to completion in milliseconds.
    ///
    /// Uses the most recent two progress samples to compute the current
    /// encoding rate, then extrapolates to 100 %.  Returns `None` when
    /// there are fewer than two samples, the progress is already at 100 %,
    /// or the elapsed time is zero.
    #[must_use]
    pub fn eta_ms(&self) -> Option<u64> {
        if self.progress_samples.len() < 2 {
            return None;
        }
        let oldest = &self.progress_samples[0];
        let newest = self.progress_samples.last()?;
        let pct_delta = newest.progress_pct - oldest.progress_pct;
        if pct_delta <= 0.0 {
            return None;
        }
        let ms_delta = newest.timestamp_ms.saturating_sub(oldest.timestamp_ms);
        if ms_delta == 0 {
            return None;
        }
        let remaining_pct = 100.0_f32 - newest.progress_pct;
        if remaining_pct <= 0.0 {
            return Some(0);
        }
        // rate = pct_delta / ms_delta  →  eta_ms = remaining_pct / rate
        let eta = (remaining_pct / pct_delta) * ms_delta as f32;
        Some(eta.round() as u64)
    }

    /// Transition to `Completed`.
    pub fn complete(&mut self, output: &str, now: u64) {
        self.state = JobState::Completed {
            output: output.to_string(),
        };
        self.updated_at = now;
    }

    /// Transition to `Failed`.
    pub fn fail(&mut self, error: &str, now: u64) {
        self.state = JobState::Failed {
            error: error.to_string(),
        };
        self.updated_at = now;
    }

    /// Transition to `Cancelled`.
    pub fn cancel(&mut self, now: u64) {
        self.state = JobState::Cancelled;
        self.updated_at = now;
    }
}

/// Stores and queries distributed jobs.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct JobTracker {
    jobs: Vec<DistributedJob>,
}

impl JobTracker {
    /// Create an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Add a job to the tracker.
    pub fn submit(&mut self, job: DistributedJob) {
        self.jobs.push(job);
    }

    /// Look up a job by ID (immutable).
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&DistributedJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Look up a job by ID (mutable).
    pub fn get_mut(&mut self, id: u64) -> Option<&mut DistributedJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Return all jobs currently in the `Queued` state.
    #[must_use]
    pub fn queued_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_queued()).collect()
    }

    /// Return all jobs currently in the `Running` state.
    #[must_use]
    pub fn running_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_running()).collect()
    }

    /// Return all jobs currently in the `Failed` state.
    #[must_use]
    pub fn failed_jobs(&self) -> Vec<&DistributedJob> {
        self.jobs.iter().filter(|j| j.state.is_failed()).collect()
    }

    /// Fraction of jobs that completed successfully.
    ///
    /// Returns `0.0` when no jobs have been submitted.
    #[must_use]
    pub fn completion_rate(&self) -> f64 {
        if self.jobs.is_empty() {
            return 0.0;
        }
        let completed = self.jobs.iter().filter(|j| j.state.is_completed()).count();
        completed as f64 / self.jobs.len() as f64
    }

    /// Return the total number of tracked jobs.
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: u64) -> DistributedJob {
        DistributedJob::new(id, &format!("job-{}", id), 0, 1000)
    }

    #[test]
    fn test_new_job_queued() {
        let j = job(1);
        assert!(j.state.is_queued());
        assert_eq!(j.id, 1);
        assert_eq!(j.name, "job-1");
    }

    #[test]
    fn test_assign_job() {
        let mut j = job(1);
        j.assign(42, 2000);
        assert!(matches!(j.state, JobState::Assigned { worker_id: 42 }));
        assert_eq!(j.updated_at, 2000);
    }

    #[test]
    fn test_update_progress() {
        let mut j = job(1);
        j.assign(1, 1001);
        j.update_progress(55.5, 1002);
        assert!(matches!(
            j.state,
            JobState::Running { progress_pct } if (progress_pct - 55.5).abs() < 1e-4
        ));
    }

    #[test]
    fn test_progress_clamps() {
        let mut j = job(1);
        j.update_progress(200.0, 1001);
        assert!(matches!(
            j.state,
            JobState::Running { progress_pct } if (progress_pct - 100.0).abs() < 1e-4
        ));
    }

    #[test]
    fn test_complete_job() {
        let mut j = job(1);
        j.complete("s3://bucket/output.mp4", 3000);
        assert!(j.state.is_completed());
        assert!(matches!(j.state, JobState::Completed { ref output } if output.contains("mp4")));
    }

    #[test]
    fn test_fail_job() {
        let mut j = job(1);
        j.fail("out of memory", 4000);
        assert!(j.state.is_failed());
    }

    #[test]
    fn test_cancel_job() {
        let mut j = job(1);
        j.cancel(5000);
        assert!(matches!(j.state, JobState::Cancelled));
    }

    #[test]
    fn test_tracker_submit_and_get() {
        let mut t = JobTracker::new();
        t.submit(job(10));
        let j = t.get(10).expect("get should return a value");
        assert_eq!(j.id, 10);
    }

    #[test]
    fn test_tracker_get_mut() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .assign(99, 2000);
        assert!(matches!(
            t.get(1).expect("get should return a value").state,
            JobState::Assigned { .. }
        ));
    }

    #[test]
    fn test_tracker_queued_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .assign(7, 1001);
        assert_eq!(t.queued_jobs().len(), 1);
        assert_eq!(t.queued_jobs()[0].id, 2);
    }

    #[test]
    fn test_tracker_running_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .update_progress(50.0, 1001);
        assert_eq!(t.running_jobs().len(), 1);
    }

    #[test]
    fn test_tracker_failed_jobs() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(2)
            .expect("get_mut should return a value")
            .fail("error", 1001);
        assert_eq!(t.failed_jobs().len(), 1);
        assert_eq!(t.failed_jobs()[0].id, 2);
    }

    #[test]
    fn test_completion_rate_empty() {
        let t = JobTracker::new();
        assert_eq!(t.completion_rate(), 0.0);
    }

    #[test]
    fn test_completion_rate_all_complete() {
        let mut t = JobTracker::new();
        for i in 1..=3 {
            t.submit(job(i));
            t.get_mut(i)
                .expect("get_mut should return a value")
                .complete("out", 2000);
        }
        assert!((t.completion_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_completion_rate_partial() {
        let mut t = JobTracker::new();
        t.submit(job(1));
        t.submit(job(2));
        t.get_mut(1)
            .expect("get_mut should return a value")
            .complete("out", 2000);
        assert!((t.completion_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_total_jobs() {
        let mut t = JobTracker::new();
        assert_eq!(t.total_jobs(), 0);
        t.submit(job(1));
        t.submit(job(2));
        assert_eq!(t.total_jobs(), 2);
    }

    // ── Progress and ETA ────────────────────────────────────────────────

    #[test]
    fn test_progress_pct_queued_is_zero() {
        let j = job(1);
        assert!((j.progress_pct() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_progress_pct_running() {
        let mut j = job(1);
        j.update_progress(42.5, 1000);
        assert!((j.progress_pct() - 42.5).abs() < 1e-5);
    }

    #[test]
    fn test_progress_pct_completed_is_100() {
        let mut j = job(1);
        j.complete("out", 2000);
        assert!((j.progress_pct() - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_progress_clamped_above_100() {
        let mut j = job(1);
        j.update_progress(150.0, 1000);
        assert!((j.progress_pct() - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_eta_single_sample_returns_none() {
        let mut j = job(1);
        j.update_progress(10.0, 1000);
        assert!(j.eta_ms().is_none());
    }

    #[test]
    fn test_eta_two_samples_estimates() {
        let mut j = job(1);
        // 0→25 % in 1000 ms  →  remaining 75 % at same rate = 3000 ms
        j.update_progress(0.0, 0);
        j.update_progress(25.0, 1000);
        let eta = j.eta_ms().expect("eta should be estimated");
        assert!((eta as i64 - 3000).abs() < 100, "eta={eta}");
    }

    #[test]
    fn test_eta_at_100_pct_is_zero() {
        let mut j = job(1);
        j.update_progress(50.0, 0);
        j.update_progress(100.0, 1000);
        let eta = j.eta_ms().expect("eta should be Some");
        assert_eq!(eta, 0);
    }

    #[test]
    fn test_eta_zero_delta_returns_none() {
        let mut j = job(1);
        j.update_progress(50.0, 1000);
        j.update_progress(50.0, 2000); // no progress
        assert!(j.eta_ms().is_none());
    }

    #[test]
    fn test_started_at_set_on_first_progress() {
        let mut j = job(1);
        assert!(j.started_at_ms.is_none());
        j.update_progress(5.0, 500);
        assert_eq!(j.started_at_ms, Some(500));
        // Second call should NOT update started_at
        j.update_progress(10.0, 1000);
        assert_eq!(j.started_at_ms, Some(500));
    }
}
