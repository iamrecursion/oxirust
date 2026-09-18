// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Multi-site render farm federation.
//!
//! Provides routing logic for distributing jobs across geographically dispersed
//! render farm sites, including cost-optimal, latency-optimal, and data-locality
//! routing strategies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FarmSite
// ---------------------------------------------------------------------------

/// Describes a single physical or virtual render farm site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmSite {
    /// Unique identifier for this site (e.g. `"us-west-2"`).
    pub site_id: String,
    /// Human-readable geographic location (e.g. `"AWS US-West-2, Oregon"`).
    pub location: String,
    /// Number of workers currently available to accept new jobs.
    pub available_workers: u32,
    /// Number of jobs currently waiting in this site's queue.
    pub queue_depth: u32,
    /// Observed average network latency to this site in milliseconds.
    pub avg_latency_ms: u32,
    /// Available inbound network bandwidth in Mbit/s.
    pub network_bandwidth_mbps: u32,
    /// Relative cost multiplier (1.0 = base price, >1 = more expensive).
    pub cost_multiplier: f32,
}

impl FarmSite {
    /// Effective cost score (lower is cheaper).
    ///
    /// Incorporates the current queue depth so that busy sites are penalised
    /// slightly even if they have the same base cost.
    ///
    /// `score = cost_multiplier * (1 + queue_depth / 100.0)`
    #[must_use]
    pub fn cost_score(&self) -> f32 {
        self.cost_multiplier * (1.0 + self.queue_depth as f32 / 100.0)
    }

    /// Whether the site has at least one available worker.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.available_workers > 0
    }
}

// ---------------------------------------------------------------------------
// JobRouting
// ---------------------------------------------------------------------------

/// Strategy that determines how a job should be routed to a site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobRouting {
    /// Prefer a specific named site; fall back to load balance if unavailable.
    Affinity(String),
    /// Distribute across sites based on available worker count.
    LoadBalance,
    /// Send to the site with the lowest effective cost score.
    CostOptimal,
    /// Send to the site with the lowest measured average latency.
    LatencyOptimal,
    /// Prefer the site whose `site_id` matches `data_site`; otherwise load balance.
    DataLocality(String),
}

// ---------------------------------------------------------------------------
// MultiSiteRouter
// ---------------------------------------------------------------------------

/// Routes jobs across a federation of `FarmSite` instances.
pub struct MultiSiteRouter {
    /// All registered sites.
    pub sites: Vec<FarmSite>,
}

impl MultiSiteRouter {
    /// Create a router with an initial list of sites.
    #[must_use]
    pub fn new(sites: Vec<FarmSite>) -> Self {
        Self { sites }
    }

    /// Add a site to the federation.
    pub fn add_site(&mut self, site: FarmSite) {
        self.sites.push(site);
    }

    /// Route a job of `job_size_mb` megabytes according to the given `routing`
    /// strategy.
    ///
    /// Returns `None` when no site has available capacity.
    #[must_use]
    pub fn route_job(&self, _job_size_mb: u32, routing: &JobRouting) -> Option<&FarmSite> {
        let candidates: Vec<&FarmSite> = self.sites.iter().filter(|s| s.has_capacity()).collect();
        if candidates.is_empty() {
            return None;
        }

        match routing {
            JobRouting::Affinity(preferred_id) => {
                // Try to find the preferred site first.
                candidates
                    .iter()
                    .find(|s| &s.site_id == preferred_id)
                    .copied()
                    .or_else(|| self.load_balance_site(&candidates))
            }

            JobRouting::LoadBalance => self.load_balance_site(&candidates),

            JobRouting::CostOptimal => candidates
                .iter()
                .min_by(|a, b| {
                    a.cost_score()
                        .partial_cmp(&b.cost_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied(),

            JobRouting::LatencyOptimal => {
                candidates.iter().min_by_key(|s| s.avg_latency_ms).copied()
            }

            JobRouting::DataLocality(data_site_id) => {
                // Prefer site with matching id and maximum bandwidth; fall back to load balance.
                let affinity_match: Vec<&FarmSite> = candidates
                    .iter()
                    .filter(|s| &s.site_id == data_site_id)
                    .copied()
                    .collect();

                if affinity_match.is_empty() {
                    self.load_balance_site(&candidates)
                } else {
                    affinity_match
                        .iter()
                        .max_by_key(|s| s.network_bandwidth_mbps)
                        .copied()
                }
            }
        }
    }

    /// Select the site with the most available workers (load balance).
    fn load_balance_site<'a>(&self, candidates: &[&'a FarmSite]) -> Option<&'a FarmSite> {
        candidates
            .iter()
            .max_by_key(|s| s.available_workers)
            .copied()
    }
}

// ---------------------------------------------------------------------------
// SiteCapacity
// ---------------------------------------------------------------------------

/// Capacity summary for a single site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteCapacity {
    /// Total worker slots (reserved + available).
    pub total_workers: u32,
    /// Workers currently occupied by running jobs.
    pub reserved: u32,
    /// Workers free to accept new jobs.
    pub available: u32,
    /// Utilisation as a percentage (0.0–100.0).
    pub utilization_pct: f32,
}

impl SiteCapacity {
    /// Derive capacity statistics from a `FarmSite`.
    #[must_use]
    pub fn from_site(site: &FarmSite) -> Self {
        let total = site.available_workers + site.queue_depth;
        let reserved = site.queue_depth.min(total);
        let available = site.available_workers;
        let utilization_pct = if total == 0 {
            0.0
        } else {
            reserved as f32 / total as f32 * 100.0
        };
        Self {
            total_workers: total,
            reserved,
            available,
            utilization_pct,
        }
    }
}

// ---------------------------------------------------------------------------
// FederatedJobTracker
// ---------------------------------------------------------------------------

/// Tracks which site is running each active job and when the job started.
#[derive(Debug, Default)]
pub struct FederatedJobTracker {
    /// `job_id` → `(site_id, started_ms)`.
    pub active_jobs: HashMap<String, (String, i64)>,
}

impl FederatedJobTracker {
    /// Create an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `job_id` started on `site_id` at `started_ms`.
    pub fn record_job_start(
        &mut self,
        job_id: impl Into<String>,
        site_id: impl Into<String>,
        started_ms: i64,
    ) {
        self.active_jobs
            .insert(job_id.into(), (site_id.into(), started_ms));
    }

    /// Remove a job (called when it finishes or is cancelled).
    pub fn remove_job(&mut self, job_id: &str) {
        self.active_jobs.remove(job_id);
    }

    /// Fraction of tracked jobs that are on `site_id` (0.0 when no jobs).
    #[must_use]
    pub fn site_utilization(&self, site_id: &str) -> f32 {
        let total = self.active_jobs.len();
        if total == 0 {
            return 0.0;
        }
        let on_site = self
            .active_jobs
            .values()
            .filter(|(sid, _)| sid == site_id)
            .count();
        on_site as f32 / total as f32
    }

    /// Group active job IDs by site.
    #[must_use]
    pub fn jobs_by_site(&self) -> HashMap<String, Vec<String>> {
        let mut result: HashMap<String, Vec<String>> = HashMap::new();
        for (job_id, (site_id, _)) in &self.active_jobs {
            result
                .entry(site_id.clone())
                .or_default()
                .push(job_id.clone());
        }
        result
    }
}

// ---------------------------------------------------------------------------
// DataTransferCost
// ---------------------------------------------------------------------------

/// Estimates the monetary cost of transferring data between two sites.
pub struct DataTransferCost;

impl DataTransferCost {
    /// Estimate the egress cost (USD) of moving `size_mb` megabytes from
    /// `from_site` to `to_site`.
    ///
    /// Uses a hypothetical $0.09/GB base rate, scaled by `to_site.cost_multiplier`.
    #[must_use]
    pub fn estimate(size_mb: u64, _from_site: &FarmSite, to_site: &FarmSite) -> f32 {
        let size_gb = size_mb as f32 / 1024.0;
        size_gb * 0.09 * to_site.cost_multiplier
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_site(
        id: &str,
        workers: u32,
        queue: u32,
        latency: u32,
        bandwidth: u32,
        cost_mult: f32,
    ) -> FarmSite {
        FarmSite {
            site_id: id.to_string(),
            location: format!("Location-{id}"),
            available_workers: workers,
            queue_depth: queue,
            avg_latency_ms: latency,
            network_bandwidth_mbps: bandwidth,
            cost_multiplier: cost_mult,
        }
    }

    // --- FarmSite ---

    #[test]
    fn test_farm_site_cost_score_no_queue() {
        let site = make_site("a", 10, 0, 10, 1000, 1.0);
        assert!((site.cost_score() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_farm_site_cost_score_with_queue() {
        let site = make_site("a", 10, 100, 10, 1000, 1.0);
        // 1.0 * (1 + 100/100) = 2.0
        assert!((site.cost_score() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_farm_site_has_capacity() {
        let site_yes = make_site("a", 5, 0, 0, 0, 1.0);
        let site_no = make_site("b", 0, 10, 0, 0, 1.0);
        assert!(site_yes.has_capacity());
        assert!(!site_no.has_capacity());
    }

    // --- MultiSiteRouter ---

    #[test]
    fn test_route_no_capacity_returns_none() {
        let sites = vec![make_site("a", 0, 0, 5, 100, 1.0)];
        let router = MultiSiteRouter::new(sites);
        assert!(router.route_job(10, &JobRouting::LoadBalance).is_none());
    }

    #[test]
    fn test_route_load_balance_picks_most_workers() {
        let sites = vec![
            make_site("small", 2, 0, 5, 100, 1.0),
            make_site("large", 8, 0, 5, 100, 1.0),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(10, &JobRouting::LoadBalance);
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("large"));
    }

    #[test]
    fn test_route_cost_optimal() {
        let sites = vec![
            make_site("cheap", 5, 0, 20, 100, 0.8),
            make_site("expensive", 5, 0, 5, 1000, 2.0),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(100, &JobRouting::CostOptimal);
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("cheap"));
    }

    #[test]
    fn test_route_latency_optimal() {
        let sites = vec![
            make_site("far", 5, 0, 200, 100, 1.0),
            make_site("near", 5, 0, 5, 100, 1.0),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(10, &JobRouting::LatencyOptimal);
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("near"));
    }

    #[test]
    fn test_route_affinity_found() {
        let sites = vec![
            make_site("alpha", 3, 0, 10, 100, 1.0),
            make_site("beta", 5, 0, 10, 100, 1.0),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(10, &JobRouting::Affinity("alpha".to_string()));
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("alpha"));
    }

    #[test]
    fn test_route_affinity_fallback_load_balance() {
        let sites = vec![
            make_site("alpha", 3, 0, 10, 100, 1.0),
            make_site("beta", 7, 0, 10, 100, 1.0),
        ];
        let router = MultiSiteRouter::new(sites);
        // "gamma" doesn't exist → load balance picks "beta"
        let chosen = router.route_job(10, &JobRouting::Affinity("gamma".to_string()));
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("beta"));
    }

    #[test]
    fn test_route_data_locality_match() {
        let sites = vec![
            make_site("data-site", 5, 0, 15, 5000, 1.2),
            make_site("other", 5, 0, 5, 100, 0.9),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(512, &JobRouting::DataLocality("data-site".to_string()));
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("data-site"));
    }

    #[test]
    fn test_route_data_locality_no_match_falls_back() {
        let sites = vec![
            make_site("a", 2, 0, 10, 100, 1.0),
            make_site("b", 9, 0, 10, 100, 1.0),
        ];
        let router = MultiSiteRouter::new(sites);
        let chosen = router.route_job(100, &JobRouting::DataLocality("missing".to_string()));
        // Falls back to load balance → "b"
        assert_eq!(chosen.map(|s| s.site_id.as_str()), Some("b"));
    }

    // --- SiteCapacity ---

    #[test]
    fn test_site_capacity_utilization() {
        let site = make_site("s", 6, 4, 10, 100, 1.0);
        let cap = SiteCapacity::from_site(&site);
        assert_eq!(cap.total_workers, 10);
        assert_eq!(cap.available, 6);
        assert!((cap.utilization_pct - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_site_capacity_zero_workers() {
        let site = make_site("empty", 0, 0, 5, 100, 1.0);
        let cap = SiteCapacity::from_site(&site);
        assert_eq!(cap.utilization_pct, 0.0);
    }

    // --- FederatedJobTracker ---

    #[test]
    fn test_tracker_empty_utilization() {
        let tracker = FederatedJobTracker::new();
        assert_eq!(tracker.site_utilization("any"), 0.0);
    }

    #[test]
    fn test_tracker_record_and_utilization() {
        let mut tracker = FederatedJobTracker::new();
        tracker.record_job_start("j1", "site-a", 1000);
        tracker.record_job_start("j2", "site-a", 1001);
        tracker.record_job_start("j3", "site-b", 1002);

        let util_a = tracker.site_utilization("site-a");
        assert!((util_a - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_tracker_jobs_by_site() {
        let mut tracker = FederatedJobTracker::new();
        tracker.record_job_start("j1", "site-x", 0);
        tracker.record_job_start("j2", "site-y", 0);
        tracker.record_job_start("j3", "site-x", 0);

        let by_site = tracker.jobs_by_site();
        assert_eq!(by_site["site-x"].len(), 2);
        assert_eq!(by_site["site-y"].len(), 1);
    }

    #[test]
    fn test_tracker_remove_job() {
        let mut tracker = FederatedJobTracker::new();
        tracker.record_job_start("j1", "site-a", 0);
        tracker.remove_job("j1");
        assert!(tracker.active_jobs.is_empty());
    }

    // --- DataTransferCost ---

    #[test]
    fn test_data_transfer_cost_basic() {
        let from = make_site("src", 1, 0, 5, 1000, 1.0);
        let to = make_site("dst", 1, 0, 5, 1000, 1.0);
        // 1024 MB = 1 GB → $0.09 * 1.0 = $0.09
        let cost = DataTransferCost::estimate(1024, &from, &to);
        assert!((cost - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_data_transfer_cost_multiplier() {
        let from = make_site("src", 1, 0, 5, 1000, 1.0);
        let to = make_site("dst-premium", 1, 0, 5, 1000, 2.0);
        let cost = DataTransferCost::estimate(1024, &from, &to);
        assert!((cost - 0.18).abs() < 0.001);
    }

    #[test]
    fn test_data_transfer_cost_zero_size() {
        let from = make_site("src", 1, 0, 5, 1000, 1.0);
        let to = make_site("dst", 1, 0, 5, 1000, 1.5);
        let cost = DataTransferCost::estimate(0, &from, &to);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_cost_score_multiplier_effect() {
        let cheap = make_site("cheap", 5, 50, 10, 500, 0.5);
        let pricey = make_site("pricey", 5, 50, 10, 500, 2.0);
        assert!(cheap.cost_score() < pricey.cost_score());
    }
}
