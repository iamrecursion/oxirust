//! Multi-strategy load balancer for render farm workers.
//!
//! Provides six distinct load-balancing strategies:
//! - `RoundRobin`: cycle through workers ignoring load
//! - `LeastConnections`: prefer workers with fewest active jobs
//! - `WeightedRoundRobin`: distribute proportional to configured weight
//! - `ResourceAware`: score by available CPU + memory headroom
//! - `LatencyBased`: prefer workers with lowest observed round-trip latency
//! - `Sticky`: affinity to a worker that handles the same job type

use std::collections::HashMap;

/// Load-balancing strategy selector.
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Cycle through all available workers, skipping saturated ones.
    RoundRobin,
    /// Always pick the worker with the fewest active jobs that has capacity.
    LeastConnections,
    /// Distribute proportional to the `weight` field of each `WorkerCapacity`.
    WeightedRoundRobin,
    /// Score workers by available CPU and memory headroom and pick the highest.
    ResourceAware,
    /// Among workers that have capacity, pick the one with the lowest `avg_latency_ms`.
    LatencyBased,
    /// Prefer a worker whose `job_type_affinity` contains the given hint; fall back
    /// to `LeastConnections` when no affinity match is found.
    Sticky(String),
}

/// Per-worker capacity descriptor used by the load balancer.
#[derive(Debug, Clone)]
pub struct WorkerCapacity {
    /// Unique identifier for this worker.
    pub worker_id: String,
    /// Relative weight for `WeightedRoundRobin` (default 1).
    pub weight: u32,
    /// Available CPU cores as a fractional count.
    pub cpu_cores_available: f32,
    /// Available RAM in mebibytes.
    pub memory_mb_available: u64,
    /// Number of jobs currently executing on this worker.
    pub active_jobs: u32,
    /// Upper bound on concurrent jobs this worker accepts.
    pub max_jobs: u32,
    /// Exponential-moving-average latency observed for this worker (milliseconds).
    pub avg_latency_ms: f64,
    /// Job-type strings this worker is optimised for (used by `Sticky`).
    pub job_type_affinity: Vec<String>,
    /// Number of GPU devices available (0 = no GPU).
    pub gpu_count: u32,
    /// Available GPU memory in mebibytes.
    pub gpu_memory_mb_available: u64,
    /// Network bandwidth available in Mbps.
    pub network_bandwidth_mbps: f64,
    /// Disk I/O throughput available in MB/s.
    pub disk_io_mbps: f64,
}

impl WorkerCapacity {
    /// Returns `true` when the worker can accept at least one more job.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.active_jobs < self.max_jobs
    }
}

/// Configurable weights for the composite resource scoring model.
///
/// Each weight controls how much a specific resource dimension contributes
/// to the overall worker score.  All weights default to sensible values
/// that prioritize CPU and memory while still rewarding GPU availability,
/// network bandwidth, disk throughput, and low latency.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for CPU availability (default 0.30).
    pub cpu_weight: f64,
    /// Weight for memory availability (default 0.25).
    pub memory_weight: f64,
    /// Weight for GPU availability and VRAM (default 0.20).
    pub gpu_weight: f64,
    /// Weight for network bandwidth (default 0.10).
    pub network_weight: f64,
    /// Weight for disk I/O throughput (default 0.10).
    pub disk_weight: f64,
    /// Weight for latency (inversely proportional; default 0.05).
    pub latency_weight: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            cpu_weight: 0.30,
            memory_weight: 0.25,
            gpu_weight: 0.20,
            network_weight: 0.10,
            disk_weight: 0.10,
            latency_weight: 0.05,
        }
    }
}

impl ScoringWeights {
    /// Create weights optimized for CPU-bound workloads.
    #[must_use]
    pub fn cpu_heavy() -> Self {
        Self {
            cpu_weight: 0.50,
            memory_weight: 0.20,
            gpu_weight: 0.05,
            network_weight: 0.10,
            disk_weight: 0.10,
            latency_weight: 0.05,
        }
    }

    /// Create weights optimized for GPU-bound workloads.
    #[must_use]
    pub fn gpu_heavy() -> Self {
        Self {
            cpu_weight: 0.10,
            memory_weight: 0.15,
            gpu_weight: 0.50,
            network_weight: 0.10,
            disk_weight: 0.10,
            latency_weight: 0.05,
        }
    }

    /// Create weights optimized for I/O-bound workloads.
    #[must_use]
    pub fn io_heavy() -> Self {
        Self {
            cpu_weight: 0.10,
            memory_weight: 0.15,
            gpu_weight: 0.05,
            network_weight: 0.30,
            disk_weight: 0.30,
            latency_weight: 0.10,
        }
    }
}

/// Multi-strategy load balancer that maintains a live view of worker capacity.
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    /// Live worker capacity records.  Exposed as `pub(crate)` so that
    /// [`crate::job_distribution::JobDistributor`] can update `active_jobs`
    /// between placements within a single batch without an extra round-trip.
    pub(crate) workers: Vec<WorkerCapacity>,
    /// Index into `workers` used by `RoundRobin` and `WeightedRoundRobin`.
    round_robin_index: usize,
    /// Per-worker counter used internally by `WeightedRoundRobin`.
    weighted_counters: HashMap<String, u32>,
}

impl LoadBalancer {
    /// Create a new load balancer using the given strategy.
    #[must_use]
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            workers: Vec::new(),
            round_robin_index: 0,
            weighted_counters: HashMap::new(),
        }
    }

    /// Register a new worker with the load balancer.
    pub fn add_worker(&mut self, worker: WorkerCapacity) {
        self.weighted_counters
            .entry(worker.worker_id.clone())
            .or_insert(0);
        self.workers.push(worker);
    }

    /// Remove a worker by ID.  Returns `true` if the worker was present.
    pub fn remove_worker(&mut self, id: &str) -> bool {
        let before = self.workers.len();
        self.workers.retain(|w| w.worker_id != id);
        self.weighted_counters.remove(id);
        // Keep round_robin_index in bounds after shrink.
        if !self.workers.is_empty() {
            self.round_robin_index %= self.workers.len();
        } else {
            self.round_robin_index = 0;
        }
        self.workers.len() < before
    }

    /// Update real-time metrics for a worker after a job state change.
    pub fn update_worker(&mut self, id: &str, active_jobs: u32, latency_ms: f64) {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == id) {
            w.active_jobs = active_jobs;
            w.avg_latency_ms = latency_ms;
        }
    }

    /// Return references to all workers that currently have room for more jobs.
    #[must_use]
    pub fn available_workers(&self) -> Vec<&WorkerCapacity> {
        self.workers.iter().filter(|w| w.has_capacity()).collect()
    }

    /// Return `true` when every registered worker is at or above `max_jobs`.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        if self.workers.is_empty() {
            return true;
        }
        self.workers.iter().all(|w| !w.has_capacity())
    }

    /// Select a worker for the given job, applying the configured strategy.
    ///
    /// Returns the `worker_id` of the selected worker, or `None` when no
    /// worker has capacity or the worker list is empty.
    pub fn select(
        &mut self,
        job_type: &str,
        required_cpu: f32,
        required_mem_mb: u64,
    ) -> Option<String> {
        // We must clone the strategy to avoid a double-borrow on `self`.
        let strategy = self.strategy.clone();
        match strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(),
            LoadBalancingStrategy::LeastConnections => self.select_least_connections(),
            LoadBalancingStrategy::WeightedRoundRobin => self.select_weighted_round_robin(),
            LoadBalancingStrategy::ResourceAware => {
                self.select_resource_aware(required_cpu, required_mem_mb)
            }
            LoadBalancingStrategy::LatencyBased => self.select_latency_based(),
            LoadBalancingStrategy::Sticky(ref hint) => {
                let hint = hint.clone();
                self.select_sticky(&hint, job_type)
            }
        }
    }

    // ── Strategy implementations ──────────────────────────────────────────────

    /// Round-robin: advance the cursor through workers that have capacity.
    fn select_round_robin(&mut self) -> Option<String> {
        let n = self.workers.len();
        if n == 0 {
            return None;
        }
        // Try at most n times to find a worker with capacity.
        for _ in 0..n {
            let idx = self.round_robin_index % n;
            self.round_robin_index = (self.round_robin_index + 1) % n;
            if self.workers[idx].has_capacity() {
                return Some(self.workers[idx].worker_id.clone());
            }
        }
        None
    }

    /// Least connections: return the worker with the smallest `active_jobs`
    /// count that still has room for at least one more job.
    fn select_least_connections(&self) -> Option<String> {
        self.workers
            .iter()
            .filter(|w| w.has_capacity())
            .min_by_key(|w| w.active_jobs)
            .map(|w| w.worker_id.clone())
    }

    /// Weighted round-robin: each worker gets a share of tokens equal to its
    /// weight.  We increment the token counter for the selected worker and
    /// pick the one whose `counter / weight` ratio is smallest (i.e. the
    /// most "underserved" worker relative to its configured share).
    fn select_weighted_round_robin(&mut self) -> Option<String> {
        // Collect eligible workers with their current counter values.
        let eligible: Vec<(String, u32, u32)> = self
            .workers
            .iter()
            .filter(|w| w.has_capacity() && w.weight > 0)
            .map(|w| {
                let cnt = self
                    .weighted_counters
                    .get(&w.worker_id)
                    .copied()
                    .unwrap_or(0);
                (w.worker_id.clone(), w.weight, cnt)
            })
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Pick the worker with the smallest (counter / weight) ratio.
        let selected = eligible
            .into_iter()
            .min_by(|(_, wa, ca), (_, wb, cb)| {
                // Compare ca/wa vs cb/wb using cross-multiplication to avoid floats.
                let lhs = (*ca as u64) * (*wb as u64);
                let rhs = (*cb as u64) * (*wa as u64);
                lhs.cmp(&rhs)
            })
            .map(|(id, _, _)| id)?;

        *self.weighted_counters.entry(selected.clone()).or_insert(0) += 1;
        Some(selected)
    }

    /// Resource-aware: weighted composite score across CPU, memory, GPU, network, and disk.
    ///
    /// The scoring formula uses configurable weights (defaulting to the values in
    /// `ScoringWeights::default()`) to combine normalized resource availability
    /// into a single scalar.  Workers with GPUs receive a bonus when GPU resources
    /// are requested.
    fn select_resource_aware(&self, required_cpu: f32, required_mem_mb: u64) -> Option<String> {
        self.select_resource_aware_weighted(
            required_cpu,
            required_mem_mb,
            false,
            &ScoringWeights::default(),
        )
    }

    /// Extended resource-aware selection with explicit GPU requirement and custom weights.
    fn select_resource_aware_weighted(
        &self,
        required_cpu: f32,
        required_mem_mb: u64,
        requires_gpu: bool,
        weights: &ScoringWeights,
    ) -> Option<String> {
        let cpu_denom = if required_cpu > 0.0 {
            required_cpu
        } else {
            1.0
        };
        let mem_denom = if required_mem_mb > 0 {
            required_mem_mb as f64
        } else {
            1.0
        };

        self.workers
            .iter()
            .filter(|w| w.has_capacity())
            .filter(|w| !requires_gpu || w.gpu_count > 0)
            .map(|w| {
                let cpu_score = (w.cpu_cores_available / cpu_denom) as f64 * weights.cpu_weight;
                let mem_score = (w.memory_mb_available as f64 / mem_denom) * weights.memory_weight;
                let gpu_score = if requires_gpu && w.gpu_count > 0 {
                    (f64::from(w.gpu_count) + w.gpu_memory_mb_available as f64 / 8192.0)
                        * weights.gpu_weight
                } else {
                    0.0
                };
                let net_score = (w.network_bandwidth_mbps / 1000.0) * weights.network_weight;
                let disk_score = (w.disk_io_mbps / 500.0) * weights.disk_weight;
                // Penalize latency: lower latency = higher score
                let latency_penalty = if w.avg_latency_ms > 0.0 {
                    (100.0 / w.avg_latency_ms) * weights.latency_weight
                } else {
                    weights.latency_weight
                };

                let total =
                    cpu_score + mem_score + gpu_score + net_score + disk_score + latency_penalty;
                (w.worker_id.clone(), total)
            })
            .max_by(|(_, sa), (_, sb)| sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Select the best worker using the composite weighted scoring model.
    ///
    /// This provides a richer selection than `select()` when GPU, network, and
    /// disk metrics are relevant.
    pub fn select_weighted(
        &mut self,
        job_type: &str,
        required_cpu: f32,
        required_mem_mb: u64,
        requires_gpu: bool,
        weights: &ScoringWeights,
    ) -> Option<String> {
        let _ = job_type;
        self.select_resource_aware_weighted(required_cpu, required_mem_mb, requires_gpu, weights)
    }

    /// Latency-based: among workers with capacity, prefer the one with the
    /// lowest observed average latency.
    fn select_latency_based(&self) -> Option<String> {
        self.workers
            .iter()
            .filter(|w| w.has_capacity())
            .min_by(|a, b| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.worker_id.clone())
    }

    /// Sticky: if any worker whose `job_type_affinity` contains `hint` has
    /// capacity, prefer the one with the fewest active jobs.  Otherwise fall
    /// back to `LeastConnections`.
    fn select_sticky(&self, hint: &str, _job_type: &str) -> Option<String> {
        let affinity_pick = self
            .workers
            .iter()
            .filter(|w| w.has_capacity() && w.job_type_affinity.iter().any(|a| a == hint))
            .min_by_key(|w| w.active_jobs)
            .map(|w| w.worker_id.clone());

        affinity_pick.or_else(|| self.select_least_connections())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(
        id: &str,
        weight: u32,
        cpu: f32,
        mem: u64,
        active: u32,
        max: u32,
    ) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight,
            cpu_cores_available: cpu,
            memory_mb_available: mem,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: 10.0,
            job_type_affinity: vec![],
            gpu_count: 0,
            gpu_memory_mb_available: 0,
            network_bandwidth_mbps: 1000.0,
            disk_io_mbps: 500.0,
        }
    }

    fn make_worker_with_latency(id: &str, latency: f64, active: u32, max: u32) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight: 1,
            cpu_cores_available: 4.0,
            memory_mb_available: 8192,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: latency,
            job_type_affinity: vec![],
            gpu_count: 0,
            gpu_memory_mb_available: 0,
            network_bandwidth_mbps: 1000.0,
            disk_io_mbps: 500.0,
        }
    }

    fn make_worker_with_affinity(
        id: &str,
        affinities: Vec<&str>,
        active: u32,
        max: u32,
    ) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight: 1,
            cpu_cores_available: 4.0,
            memory_mb_available: 8192,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: 5.0,
            job_type_affinity: affinities.into_iter().map(|s| s.to_string()).collect(),
            gpu_count: 0,
            gpu_memory_mb_available: 0,
            network_bandwidth_mbps: 1000.0,
            disk_io_mbps: 500.0,
        }
    }

    fn make_worker_with_gpu(
        id: &str,
        cpu: f32,
        mem: u64,
        gpu_count: u32,
        gpu_mem: u64,
        active: u32,
        max: u32,
    ) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight: 1,
            cpu_cores_available: cpu,
            memory_mb_available: mem,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: 10.0,
            job_type_affinity: vec![],
            gpu_count,
            gpu_memory_mb_available: gpu_mem,
            network_bandwidth_mbps: 1000.0,
            disk_io_mbps: 500.0,
        }
    }

    fn make_worker_full(
        id: &str,
        cpu: f32,
        mem: u64,
        gpu_count: u32,
        gpu_mem: u64,
        net_mbps: f64,
        disk_mbps: f64,
        latency: f64,
        active: u32,
        max: u32,
    ) -> WorkerCapacity {
        WorkerCapacity {
            worker_id: id.to_string(),
            weight: 1,
            cpu_cores_available: cpu,
            memory_mb_available: mem,
            active_jobs: active,
            max_jobs: max,
            avg_latency_ms: latency,
            job_type_affinity: vec![],
            gpu_count,
            gpu_memory_mb_available: gpu_mem,
            network_bandwidth_mbps: net_mbps,
            disk_io_mbps: disk_mbps,
        }
    }

    // ── RoundRobin ────────────────────────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles_through_workers() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 0, 4));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 4));

        let s1 = lb.select("video", 1.0, 512).expect("should select");
        let s2 = lb.select("video", 1.0, 512).expect("should select");
        // Must alternate between the two workers.
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_round_robin_skips_saturated_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4)); // full
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 4));

        let selected = lb.select("video", 1.0, 512).expect("should select w2");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_round_robin_returns_none_when_all_full() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 4, 4));

        assert!(lb.select("video", 1.0, 512).is_none());
    }

    // ── LeastConnections ─────────────────────────────────────────────────────

    #[test]
    fn test_least_connections_picks_lowest_active() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 3, 8));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 1, 8));
        lb.add_worker(make_worker("w3", 1, 4.0, 8192, 2, 8));

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_least_connections_ignores_full_workers() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4)); // full
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 2, 8));

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    // ── WeightedRoundRobin ────────────────────────────────────────────────────

    #[test]
    fn test_weighted_round_robin_distributes_by_weight() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::WeightedRoundRobin);
        lb.add_worker(make_worker("w1", 3, 4.0, 8192, 0, 100)); // weight 3
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 100)); // weight 1

        let mut counts: HashMap<String, u32> = HashMap::new();
        for _ in 0..40 {
            let id = lb.select("video", 1.0, 512).expect("should select");
            *counts.entry(id).or_insert(0) += 1;
        }
        // w1 should receive roughly 3x more calls than w2.
        let c1 = *counts.get("w1").unwrap_or(&0);
        let c2 = *counts.get("w2").unwrap_or(&0);
        assert!(
            c1 > c2,
            "w1 ({c1}) should have more selections than w2 ({c2})"
        );
    }

    #[test]
    fn test_weighted_round_robin_skips_full_workers() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::WeightedRoundRobin);
        lb.add_worker(make_worker("w1", 5, 4.0, 8192, 4, 4)); // full
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 4));

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    // ── ResourceAware ─────────────────────────────────────────────────────────

    #[test]
    fn test_resource_aware_picks_most_headroom() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker("w1", 1, 2.0, 1024, 0, 8)); // low headroom
        lb.add_worker(make_worker("w2", 1, 8.0, 16384, 0, 8)); // high headroom

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_resource_aware_cpu_vs_memory_weighting() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        // w1 has lots of CPU but low memory; w2 is balanced.
        lb.add_worker(make_worker("w1", 1, 16.0, 512, 0, 8));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 8));
        // required: 2 cpu, 4096 MB
        // w1 score: (16/2)*0.6 + (512/4096)*0.4 = 4.8 + 0.05 = 4.85
        // w2 score: (4/2)*0.6 + (8192/4096)*0.4 = 1.2 + 0.8 = 2.0
        let selected = lb.select("video", 2.0, 4096).expect("should select");
        assert_eq!(selected, "w1");
    }

    #[test]
    fn test_resource_aware_returns_none_when_all_full() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4));

        assert!(lb.select("video", 1.0, 512).is_none());
    }

    // ── LatencyBased ──────────────────────────────────────────────────────────

    #[test]
    fn test_latency_based_picks_lowest_latency() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LatencyBased);
        lb.add_worker(make_worker_with_latency("w1", 120.0, 0, 8));
        lb.add_worker(make_worker_with_latency("w2", 30.0, 0, 8));
        lb.add_worker(make_worker_with_latency("w3", 75.0, 0, 8));

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_latency_based_skips_full_workers() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LatencyBased);
        lb.add_worker(make_worker_with_latency("w1", 5.0, 4, 4)); // full but fast
        lb.add_worker(make_worker_with_latency("w2", 200.0, 0, 8)); // slow but available

        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    // ── Sticky ────────────────────────────────────────────────────────────────

    #[test]
    fn test_sticky_prefers_affinity_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::Sticky("transcode".to_string()));
        lb.add_worker(make_worker_with_affinity("w1", vec!["transcode"], 1, 8));
        lb.add_worker(make_worker_with_affinity("w2", vec!["qc"], 0, 8));

        let selected = lb.select("transcode", 1.0, 512).expect("should select");
        assert_eq!(selected, "w1");
    }

    #[test]
    fn test_sticky_falls_back_to_least_connections() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::Sticky("unknown".to_string()));
        lb.add_worker(make_worker_with_affinity("w1", vec!["transcode"], 3, 8));
        lb.add_worker(make_worker_with_affinity("w2", vec!["qc"], 1, 8));

        // No affinity match → falls back to least connections → w2.
        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_sticky_picks_least_loaded_affinity_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::Sticky("transcode".to_string()));
        lb.add_worker(make_worker_with_affinity("w1", vec!["transcode"], 5, 8));
        lb.add_worker(make_worker_with_affinity("w2", vec!["transcode"], 1, 8));

        let selected = lb.select("transcode", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    // ── Lifecycle helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_add_and_remove_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 0, 4));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 0, 4));

        assert!(lb.remove_worker("w1"));
        assert!(!lb.remove_worker("w999")); // not present
        assert_eq!(lb.available_workers().len(), 1);
    }

    #[test]
    fn test_update_worker_active_jobs_and_latency() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LatencyBased);
        lb.add_worker(make_worker_with_latency("w1", 100.0, 0, 8));
        lb.add_worker(make_worker_with_latency("w2", 200.0, 0, 8));

        // After updating w2 to have very low latency, it should be picked.
        lb.update_worker("w2", 0, 1.0);
        let selected = lb.select("video", 1.0, 512).expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_available_workers_filters_correctly() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4)); // full
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 2, 4)); // available
        lb.add_worker(make_worker("w3", 1, 4.0, 8192, 0, 4)); // available

        let avail = lb.available_workers();
        assert_eq!(avail.len(), 2);
    }

    #[test]
    fn test_is_overloaded_false_when_capacity_exists() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4)); // full
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 1, 4)); // has capacity

        assert!(!lb.is_overloaded());
    }

    #[test]
    fn test_is_overloaded_true_when_all_full() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        lb.add_worker(make_worker("w1", 1, 4.0, 8192, 4, 4));
        lb.add_worker(make_worker("w2", 1, 4.0, 8192, 4, 4));

        assert!(lb.is_overloaded());
    }

    #[test]
    fn test_is_overloaded_true_when_empty() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        assert!(lb.is_overloaded());
    }

    // ── Weighted scoring tests ──────────────────────────────────────────────

    #[test]
    fn test_weighted_scoring_prefers_gpu_worker_for_gpu_job() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker_with_gpu("cpu-only", 8.0, 16384, 0, 0, 0, 8));
        lb.add_worker(make_worker_with_gpu("gpu-node", 4.0, 8192, 2, 16384, 0, 8));

        let selected = lb
            .select_weighted("transcode", 2.0, 4096, true, &ScoringWeights::default())
            .expect("should select gpu-node");
        assert_eq!(selected, "gpu-node");
    }

    #[test]
    fn test_weighted_scoring_gpu_heavy_weights() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker_with_gpu("w1", 16.0, 32768, 1, 4096, 0, 8));
        lb.add_worker(make_worker_with_gpu("w2", 4.0, 8192, 4, 32768, 0, 8));

        // With GPU-heavy weights, w2 should win despite less CPU/RAM
        let selected = lb
            .select_weighted(
                "ml-transcode",
                2.0,
                4096,
                true,
                &ScoringWeights::gpu_heavy(),
            )
            .expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_weighted_scoring_cpu_heavy_weights() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker_with_gpu("w1", 32.0, 8192, 0, 0, 0, 8));
        lb.add_worker(make_worker_with_gpu("w2", 4.0, 32768, 0, 0, 0, 8));

        let selected = lb
            .select_weighted("encode", 4.0, 2048, false, &ScoringWeights::cpu_heavy())
            .expect("should select");
        assert_eq!(selected, "w1");
    }

    #[test]
    fn test_weighted_scoring_io_heavy_weights() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker_full(
            "w1", 8.0, 8192, 0, 0, 100.0, 100.0, 10.0, 0, 8,
        ));
        lb.add_worker(make_worker_full(
            "w2", 4.0, 4096, 0, 0, 10000.0, 2000.0, 5.0, 0, 8,
        ));

        let selected = lb
            .select_weighted("ingest", 1.0, 1024, false, &ScoringWeights::io_heavy())
            .expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_weighted_scoring_filters_non_gpu_when_required() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        lb.add_worker(make_worker("no-gpu", 1, 16.0, 32768, 0, 8));

        let selected = lb.select_weighted("gpu-job", 1.0, 1024, true, &ScoringWeights::default());
        assert!(selected.is_none());
    }

    #[test]
    fn test_weighted_scoring_latency_penalty() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::ResourceAware);
        // Same resources, but w1 has much higher latency
        lb.add_worker(make_worker_full(
            "w1", 8.0, 8192, 0, 0, 1000.0, 500.0, 500.0, 0, 8,
        ));
        lb.add_worker(make_worker_full(
            "w2", 8.0, 8192, 0, 0, 1000.0, 500.0, 5.0, 0, 8,
        ));

        let selected = lb
            .select_weighted("encode", 2.0, 2048, false, &ScoringWeights::default())
            .expect("should select");
        assert_eq!(selected, "w2");
    }

    #[test]
    fn test_scoring_weights_default() {
        let w = ScoringWeights::default();
        let total = w.cpu_weight
            + w.memory_weight
            + w.gpu_weight
            + w.network_weight
            + w.disk_weight
            + w.latency_weight;
        assert!((total - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scoring_weights_presets_sum_to_one() {
        for weights in [
            ScoringWeights::cpu_heavy(),
            ScoringWeights::gpu_heavy(),
            ScoringWeights::io_heavy(),
        ] {
            let total = weights.cpu_weight
                + weights.memory_weight
                + weights.gpu_weight
                + weights.network_weight
                + weights.disk_weight
                + weights.latency_weight;
            assert!(
                (total - 1.0).abs() < f64::EPSILON,
                "weights should sum to 1.0, got {total}"
            );
        }
    }
}
