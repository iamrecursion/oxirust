//! Archive restore planning.
//!
//! Generates ordered restore plans from a catalog, with priority-based
//! scheduling and estimated completion times.

#![allow(dead_code)]

// ── Priority ──────────────────────────────────────────────────────────────────

/// Priority level for a restore request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestorePriority {
    /// Immediate restoration required.
    Critical,
    /// High-priority; should be served before normal work.
    High,
    /// Standard restoration request.
    Normal,
    /// Best-effort; may be deferred.
    Low,
}

impl RestorePriority {
    /// Numeric sort key: lower value = higher priority.
    #[must_use]
    pub fn sort_key(&self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 1,
            Self::Normal => 2,
            Self::Low => 3,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
        }
    }
}

// ── Request ───────────────────────────────────────────────────────────────────

/// A request to restore one or more archive assets.
#[derive(Debug, Clone)]
pub struct RestoreRequest {
    /// Unique request identifier.
    pub id: u64,
    /// IDs of the assets to restore.
    pub asset_ids: Vec<u64>,
    /// How urgently the assets are needed.
    pub priority: RestorePriority,
    /// Unix timestamp when the request was created.
    pub requested_at: u64,
    /// Optional deadline (Unix timestamp).
    pub deadline_at: Option<u64>,
}

impl RestoreRequest {
    /// Creates a new restore request.
    #[must_use]
    pub fn new(id: u64, asset_ids: Vec<u64>, priority: RestorePriority, requested_at: u64) -> Self {
        Self {
            id,
            asset_ids,
            priority,
            requested_at,
            deadline_at: None,
        }
    }

    /// Creates a request with a hard deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline: u64) -> Self {
        self.deadline_at = Some(deadline);
        self
    }

    /// Returns `true` if a deadline has been set and the current time has passed it.
    #[must_use]
    pub fn is_overdue(&self, now: u64) -> bool {
        self.deadline_at.is_some_and(|d| now > d)
    }
}

// ── Plan ─────────────────────────────────────────────────────────────────────

/// A single step in a restore plan.
#[derive(Debug, Clone)]
pub struct RestoreStep {
    /// Step sequence number (1-based).
    pub step_id: u32,
    /// Source media identifier (e.g. tape barcode or disk path).
    pub source_media: String,
    /// Asset being restored in this step.
    pub asset_id: u64,
    /// Size of the asset in bytes.
    pub size_bytes: u64,
    /// Execution order (same as `step_id` in the basic planner).
    pub order: u32,
}

/// A complete restore plan for a single request.
#[derive(Debug, Clone)]
pub struct RestorePlan {
    /// The request this plan fulfils.
    pub request_id: u64,
    /// Ordered list of restore steps.
    pub steps: Vec<RestoreStep>,
    /// Estimated total restore time in seconds.
    pub estimated_duration_s: u64,
}

impl RestorePlan {
    /// Returns the total bytes that will be restored.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.steps.iter().map(|s| s.size_bytes).sum()
    }

    /// Returns the number of steps in the plan.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

// ── Functions ─────────────────────────────────────────────────────────────────

/// Estimates restore time given a total byte count and a read speed.
///
/// `speed_mbps` is in MB/s (megabytes per second, 1 MB = 1 048 576 bytes).
/// Returns the ceiling in whole seconds; at least 1 second for any non-zero load.
#[must_use]
pub fn estimate_restore_time_s(total_bytes: u64, speed_mbps: f64) -> u64 {
    if total_bytes == 0 || speed_mbps <= 0.0 {
        return 0;
    }
    let speed_bps = speed_mbps * 1_048_576.0;
    let seconds = total_bytes as f64 / speed_bps;
    seconds.ceil() as u64
}

/// Builds a restore plan from a request and a catalog snapshot.
///
/// `catalog` is a slice of `(asset_id, source_media, size_bytes)` tuples.
/// Assets not present in the catalog are silently skipped.
/// Steps are ordered so that assets from the same source are grouped together
/// (reducing tape mounts / disk seeks).
///
/// The estimated duration uses a default tape read speed of 400 MB/s.
#[must_use]
pub fn plan_restore(request: &RestoreRequest, catalog: &[(u64, String, u64)]) -> RestorePlan {
    const DEFAULT_SPEED_MBPS: f64 = 400.0;

    // Collect steps for assets that appear in the catalog
    let mut tuples: Vec<(u64, String, u64)> = request
        .asset_ids
        .iter()
        .filter_map(|&aid| {
            catalog
                .iter()
                .find(|(cid, _, _)| *cid == aid)
                .map(|(_, src, sz)| (aid, src.clone(), *sz))
        })
        .collect();

    // Group by source media to minimise mount operations
    tuples.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    let steps: Vec<RestoreStep> = tuples
        .into_iter()
        .enumerate()
        .map(|(i, (asset_id, source_media, size_bytes))| {
            let order = (i + 1) as u32;
            RestoreStep {
                step_id: order,
                source_media,
                asset_id,
                size_bytes,
                order,
            }
        })
        .collect();

    let total_bytes: u64 = steps.iter().map(|s| s.size_bytes).sum();
    let estimated_duration_s = estimate_restore_time_s(total_bytes, DEFAULT_SPEED_MBPS);

    RestorePlan {
        request_id: request.id,
        steps,
        estimated_duration_s,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> Vec<(u64, String, u64)> {
        vec![
            (1, "TAPE001".to_string(), 5_000_000_000),
            (2, "TAPE001".to_string(), 3_000_000_000),
            (3, "TAPE002".to_string(), 8_000_000_000),
            (4, "DISK001".to_string(), 1_000_000_000),
        ]
    }

    #[test]
    fn test_priority_sort_key_ordering() {
        assert!(RestorePriority::Critical.sort_key() < RestorePriority::High.sort_key());
        assert!(RestorePriority::High.sort_key() < RestorePriority::Normal.sort_key());
        assert!(RestorePriority::Normal.sort_key() < RestorePriority::Low.sort_key());
    }

    #[test]
    fn test_priority_labels() {
        assert_eq!(RestorePriority::Critical.label(), "critical");
        assert_eq!(RestorePriority::Low.label(), "low");
    }

    #[test]
    fn test_restore_request_new() {
        let req = RestoreRequest::new(42, vec![1, 2, 3], RestorePriority::High, 1_000_000);
        assert_eq!(req.id, 42);
        assert_eq!(req.asset_ids.len(), 3);
        assert!(req.deadline_at.is_none());
    }

    #[test]
    fn test_restore_request_with_deadline() {
        let req = RestoreRequest::new(1, vec![1], RestorePriority::Normal, 0).with_deadline(9999);
        assert_eq!(req.deadline_at, Some(9999));
    }

    #[test]
    fn test_restore_request_is_overdue() {
        let req = RestoreRequest::new(1, vec![], RestorePriority::Normal, 0).with_deadline(500);
        assert!(!req.is_overdue(499));
        assert!(req.is_overdue(501));
    }

    #[test]
    fn test_estimate_restore_time_basic() {
        // 400 MB/s → 400 * 1_048_576 bytes/s ≈ 419_430_400 bytes/s
        // 419_430_400 bytes at 400 MB/s → should be ≈ 1 second
        let t = estimate_restore_time_s(419_430_400, 400.0);
        assert_eq!(t, 1);
    }

    #[test]
    fn test_estimate_restore_time_zero_bytes() {
        assert_eq!(estimate_restore_time_s(0, 400.0), 0);
    }

    #[test]
    fn test_estimate_restore_time_zero_speed() {
        assert_eq!(estimate_restore_time_s(1_000_000, 0.0), 0);
    }

    #[test]
    fn test_plan_restore_step_count() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![1, 2, 3], RestorePriority::Normal, 0);
        let plan = plan_restore(&req, &catalog);
        assert_eq!(plan.step_count(), 3);
    }

    #[test]
    fn test_plan_restore_skips_missing_assets() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![1, 99], RestorePriority::Normal, 0);
        let plan = plan_restore(&req, &catalog);
        assert_eq!(plan.step_count(), 1);
    }

    #[test]
    fn test_plan_restore_grouped_by_source() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![3, 1, 2], RestorePriority::Normal, 0);
        let plan = plan_restore(&req, &catalog);
        // TAPE001 assets (1, 2) should come before TAPE002 asset (3)
        let sources: Vec<&str> = plan.steps.iter().map(|s| s.source_media.as_str()).collect();
        let tape001_count = sources.iter().filter(|&&s| s == "TAPE001").count();
        let tape002_count = sources.iter().filter(|&&s| s == "TAPE002").count();
        assert_eq!(tape001_count, 2);
        assert_eq!(tape002_count, 1);
        // First two steps should be TAPE001
        assert_eq!(plan.steps[0].source_media, "TAPE001");
        assert_eq!(plan.steps[1].source_media, "TAPE001");
    }

    #[test]
    fn test_plan_restore_total_bytes() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![1, 2], RestorePriority::Normal, 0);
        let plan = plan_restore(&req, &catalog);
        assert_eq!(plan.total_bytes(), 8_000_000_000);
    }

    #[test]
    fn test_plan_restore_step_ids_sequential() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![1, 3, 4], RestorePriority::High, 0);
        let plan = plan_restore(&req, &catalog);
        for (i, step) in plan.steps.iter().enumerate() {
            assert_eq!(step.step_id, (i + 1) as u32);
        }
    }

    #[test]
    fn test_plan_restore_empty_request() {
        let catalog = sample_catalog();
        let req = RestoreRequest::new(1, vec![], RestorePriority::Normal, 0);
        let plan = plan_restore(&req, &catalog);
        assert_eq!(plan.step_count(), 0);
        assert_eq!(plan.total_bytes(), 0);
        assert_eq!(plan.estimated_duration_s, 0);
    }
}
