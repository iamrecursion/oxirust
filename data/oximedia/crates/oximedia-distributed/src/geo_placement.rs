//! Cross-region geo-aware worker placement.
//!
//! Selects the best worker for a job based on geographic region preferences
//! and network latency estimates.  The selection algorithm:
//!
//! 1. Prefers workers whose region matches `preferred_region`.
//! 2. Falls back to workers in `fallback_regions` (in order).
//! 3. Accepts any worker within `max_latency_ms` if no preferred/fallback
//!    worker is available.
//!
//! # Example
//!
//! ```rust
//! use oximedia_distributed::geo_placement::{
//!     GeoPlacementPolicy, Region, WorkerRegion, select_worker_by_region,
//! };
//! use uuid::Uuid;
//!
//! let workers = vec![
//!     WorkerRegion { worker_id: Uuid::new_v4(), region: Region::new("us-east-1"), latency_estimate_ms: 5 },
//!     WorkerRegion { worker_id: Uuid::new_v4(), region: Region::new("eu-west-1"), latency_estimate_ms: 80 },
//! ];
//! let policy = GeoPlacementPolicy {
//!     preferred_region: Region::new("us-east-1"),
//!     fallback_regions: vec![Region::new("eu-west-1")],
//!     max_latency_ms: 200,
//! };
//! let selected = select_worker_by_region(&workers, &policy);
//! assert!(selected.is_some());
//! ```

use uuid::Uuid;

/// A geographic region identifier (e.g. `"us-east-1"`, `"eu-west-1"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Region(pub String);

impl Region {
    /// Create a new region from any string-like value.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the region name as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Region {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Region {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A worker annotated with its geographic region and estimated latency.
#[derive(Debug, Clone)]
pub struct WorkerRegion {
    /// Unique identifier for the worker.
    pub worker_id: Uuid,
    /// Region where the worker is hosted.
    pub region: Region,
    /// Estimated round-trip latency to this worker in milliseconds.
    pub latency_estimate_ms: u64,
}

impl WorkerRegion {
    /// Create a new `WorkerRegion`.
    #[must_use]
    pub fn new(worker_id: Uuid, region: Region, latency_estimate_ms: u64) -> Self {
        Self {
            worker_id,
            region,
            latency_estimate_ms,
        }
    }
}

/// Policy for selecting workers across geographic regions.
#[derive(Debug, Clone)]
pub struct GeoPlacementPolicy {
    /// Preferred region — workers here are tried first.
    pub preferred_region: Region,
    /// Ordered list of fallback regions if the preferred region has no
    /// suitable worker.
    pub fallback_regions: Vec<Region>,
    /// Maximum acceptable latency in milliseconds.  Workers whose
    /// `latency_estimate_ms` exceeds this threshold are excluded unless
    /// no other option exists.
    pub max_latency_ms: u64,
}

impl GeoPlacementPolicy {
    /// Create a simple policy with only a preferred region.
    #[must_use]
    pub fn preferred_only(region: Region, max_latency_ms: u64) -> Self {
        Self {
            preferred_region: region,
            fallback_regions: Vec::new(),
            max_latency_ms,
        }
    }
}

/// Select the best worker from `workers` according to `policy`.
///
/// Returns the `worker_id` of the selected worker, or `None` if the slice is
/// empty.
///
/// # Algorithm
///
/// 1. Among workers in `preferred_region` whose latency ≤ `max_latency_ms`,
///    pick the one with the lowest latency.
/// 2. If none, iterate `fallback_regions` in order; for each fallback region
///    pick the lowest-latency worker within `max_latency_ms`.
/// 3. If still none, accept any worker within `max_latency_ms`, picking the
///    lowest-latency one.
/// 4. If still none (all exceed latency limit), return the overall lowest-
///    latency worker as a last resort.
#[must_use]
pub fn select_worker_by_region(
    workers: &[WorkerRegion],
    policy: &GeoPlacementPolicy,
) -> Option<Uuid> {
    if workers.is_empty() {
        return None;
    }

    // Helper: find lowest-latency worker in a given region within the latency cap.
    let best_in_region = |region: &Region| -> Option<&WorkerRegion> {
        workers
            .iter()
            .filter(|w| &w.region == region && w.latency_estimate_ms <= policy.max_latency_ms)
            .min_by_key(|w| w.latency_estimate_ms)
    };

    // 1. Preferred region
    if let Some(w) = best_in_region(&policy.preferred_region) {
        return Some(w.worker_id);
    }

    // 2. Fallback regions (in declared order)
    for fallback in &policy.fallback_regions {
        if let Some(w) = best_in_region(fallback) {
            return Some(w.worker_id);
        }
    }

    // 3. Any worker within latency cap
    if let Some(w) = workers
        .iter()
        .filter(|w| w.latency_estimate_ms <= policy.max_latency_ms)
        .min_by_key(|w| w.latency_estimate_ms)
    {
        return Some(w.worker_id);
    }

    // 4. Last resort: lowest latency regardless of cap
    workers
        .iter()
        .min_by_key(|w| w.latency_estimate_ms)
        .map(|w| w.worker_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workers() -> Vec<WorkerRegion> {
        vec![
            WorkerRegion::new(
                Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                Region::new("us-east-1"),
                10,
            ),
            WorkerRegion::new(
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                Region::new("eu-west-1"),
                80,
            ),
            WorkerRegion::new(
                Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                Region::new("ap-southeast-1"),
                150,
            ),
        ]
    }

    #[test]
    fn test_geo_placement_prefers_primary_region() {
        let workers = make_workers();
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("us-east-1"),
            fallback_regions: vec![Region::new("eu-west-1")],
            max_latency_ms: 200,
        };
        let selected = select_worker_by_region(&workers, &policy).expect("should select");
        assert_eq!(
            selected,
            Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
        );
    }

    #[test]
    fn test_geo_placement_falls_back_correctly() {
        let workers = make_workers();
        // Preferred region has no workers in list (us-west-2)
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("us-west-2"),
            fallback_regions: vec![Region::new("eu-west-1")],
            max_latency_ms: 200,
        };
        let selected = select_worker_by_region(&workers, &policy).expect("should select");
        assert_eq!(
            selected,
            Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap()
        );
    }

    #[test]
    fn test_geo_placement_any_within_latency_cap() {
        let workers = make_workers();
        // Preferred and fallback regions are unknown; should pick lowest-latency within cap
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("unknown"),
            fallback_regions: vec![],
            max_latency_ms: 100,
        };
        let selected = select_worker_by_region(&workers, &policy).expect("should select");
        // us-east-1 has 10ms (< 100ms) — should be picked as lowest within cap
        assert_eq!(
            selected,
            Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
        );
    }

    #[test]
    fn test_geo_placement_last_resort_when_all_over_cap() {
        let workers = vec![WorkerRegion::new(
            Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
            Region::new("far-away"),
            999,
        )];
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("us-east-1"),
            fallback_regions: vec![],
            max_latency_ms: 50, // All workers exceed cap
        };
        // Last resort: return the only worker
        let selected = select_worker_by_region(&workers, &policy).expect("should select");
        assert_eq!(
            selected,
            Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap()
        );
    }

    #[test]
    fn test_geo_placement_empty_workers_returns_none() {
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("us-east-1"),
            fallback_regions: vec![],
            max_latency_ms: 200,
        };
        assert!(select_worker_by_region(&[], &policy).is_none());
    }

    #[test]
    fn test_geo_placement_picks_lowest_latency_in_preferred() {
        let id_slow = Uuid::parse_str("00000000-0000-0000-0000-000000000010").unwrap();
        let id_fast = Uuid::parse_str("00000000-0000-0000-0000-000000000011").unwrap();
        let workers = vec![
            WorkerRegion::new(id_slow, Region::new("us-east-1"), 90),
            WorkerRegion::new(id_fast, Region::new("us-east-1"), 20),
        ];
        let policy = GeoPlacementPolicy {
            preferred_region: Region::new("us-east-1"),
            fallback_regions: vec![],
            max_latency_ms: 200,
        };
        let selected = select_worker_by_region(&workers, &policy).expect("should select");
        assert_eq!(selected, id_fast);
    }

    #[test]
    fn test_region_display() {
        let r = Region::new("eu-central-1");
        assert_eq!(r.to_string(), "eu-central-1");
        assert_eq!(r.as_str(), "eu-central-1");
    }

    #[test]
    fn test_region_from_str() {
        let r: Region = "us-west-2".into();
        assert_eq!(r, Region::new("us-west-2"));
    }
}
