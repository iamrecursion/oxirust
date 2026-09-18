//! Prefetch scheduler — proactively push popular content to edge nodes before
//! demand spikes based on access-pattern signals.
//!
//! # Overview
//!
//! `PrefetchScheduler` analyses a rolling access log to identify **hot
//! content** (ranked by a configurable popularity score), computes a list of
//! edge PoPs that are missing that content, and generates `PrefetchTask`s
//! that an operator's delivery pipeline can execute.
//!
//! Popularity scoring uses a combination of:
//!
//! - **Recency** — requests closer to `now` contribute more (exponential decay).
//! - **Frequency** — raw request count within the window.
//! - **Cache-miss rate** — content that is being missed often is a better
//!   prefetch target than content that is already cached.
//!
//! The scheduler is intentionally pure-logic — no network I/O is performed.

use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors from the prefetch scheduler.
#[derive(Debug, Error)]
pub enum PrefetchError {
    /// A content item with the given key was not found.
    #[error("content item '{0}' not found")]
    NotFound(String),
    /// The edge-node list is empty.
    #[error("no edge nodes registered")]
    NoEdgeNodes,
}

// ─── AccessRecord ─────────────────────────────────────────────────────────────

/// A single observed access to a content item.
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// Content key (URL path, asset ID, etc.).
    pub key: String,
    /// Edge PoP that served (or missed) the request.
    pub edge_id: String,
    /// Whether the content was served from cache (`true`) or from origin.
    pub cache_hit: bool,
    /// When the access was observed.
    pub observed_at: Instant,
    /// Estimated content size in bytes (used for prioritising smaller content).
    pub size_bytes: u64,
}

impl AccessRecord {
    /// Create a new access record with the current timestamp.
    pub fn new(key: &str, edge_id: &str, cache_hit: bool, size_bytes: u64) -> Self {
        Self {
            key: key.to_string(),
            edge_id: edge_id.to_string(),
            cache_hit,
            observed_at: Instant::now(),
            size_bytes,
        }
    }
}

// ─── ContentStats ─────────────────────────────────────────────────────────────

/// Aggregated per-content statistics over the rolling window.
#[derive(Debug, Clone)]
pub struct ContentStats {
    /// Content key.
    pub key: String,
    /// Total accesses in the window.
    pub access_count: u64,
    /// Cache-hit count.
    pub hit_count: u64,
    /// Cache-miss count.
    pub miss_count: u64,
    /// Popularity score (higher = more prefetch-worthy).
    pub popularity_score: f64,
    /// Set of edge PoPs that have seen at least one hit for this content
    /// (i.e., content is already cached there).
    pub cached_at_edges: std::collections::HashSet<String>,
    /// Estimated content size in bytes.
    pub size_bytes: u64,
}

impl ContentStats {
    fn new(key: &str) -> Self {
        Self {
            key: key.to_string(),
            access_count: 0,
            hit_count: 0,
            miss_count: 0,
            popularity_score: 0.0,
            cached_at_edges: std::collections::HashSet::new(),
            size_bytes: 0,
        }
    }

    /// Cache-miss rate in [0, 1].
    pub fn miss_rate(&self) -> f64 {
        if self.access_count == 0 {
            return 0.0;
        }
        self.miss_count as f64 / self.access_count as f64
    }
}

// ─── PrefetchTask ─────────────────────────────────────────────────────────────

/// A prefetch instruction: push `content_key` to `target_edge`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchTask {
    /// Content item to prefetch.
    pub content_key: String,
    /// Target edge PoP.
    pub target_edge: String,
    /// Popularity score at the time the task was generated.
    pub priority: u32,
    /// Estimated content size in bytes.
    pub size_bytes: u64,
}

impl PrefetchTask {
    fn new(content_key: &str, target_edge: &str, priority: u32, size_bytes: u64) -> Self {
        Self {
            content_key: content_key.to_string(),
            target_edge: target_edge.to_string(),
            priority,
            size_bytes,
        }
    }
}

/// Ordering wrapper so tasks can be stored in a [`BinaryHeap`] (max-heap on
/// priority).
#[derive(Debug, Clone, Eq, PartialEq)]
struct TaskHeapEntry {
    priority: u32,
    task: PrefetchTask,
}

impl Ord for TaskHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for TaskHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─── PrefetchConfig ───────────────────────────────────────────────────────────

/// Configuration for the prefetch scheduler.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Rolling window for access-record collection.
    pub window: Duration,
    /// Number of top content items to consider for prefetching.
    pub top_n_content: usize,
    /// Maximum size (bytes) of a content item eligible for prefetching.
    /// Items larger than this are skipped (too expensive to push proactively).
    pub max_item_size_bytes: u64,
    /// Minimum popularity score threshold for a task to be generated.
    pub min_popularity_score: f64,
    /// Recency decay half-life in seconds: older accesses decay exponentially.
    pub decay_half_life_secs: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(3600),
            top_n_content: 100,
            max_item_size_bytes: 100 * 1024 * 1024, // 100 MiB
            min_popularity_score: 1.0,
            decay_half_life_secs: 300.0, // 5-minute half-life
        }
    }
}

impl PrefetchConfig {
    /// Set the rolling window.
    pub fn with_window(mut self, window: Duration) -> Self {
        self.window = window;
        self
    }

    /// Set the top-N limit.
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n_content = n;
        self
    }

    /// Set the minimum popularity score.
    pub fn with_min_score(mut self, score: f64) -> Self {
        self.min_popularity_score = score;
        self
    }

    /// Set the recency decay half-life.
    pub fn with_decay_half_life_secs(mut self, secs: f64) -> Self {
        self.decay_half_life_secs = secs;
        self
    }
}

// ─── PrefetchScheduler ───────────────────────────────────────────────────────

/// Tracks content access patterns and generates prefetch tasks for edge nodes.
pub struct PrefetchScheduler {
    config: PrefetchConfig,
    /// Registered edge PoP IDs.
    edges: Vec<String>,
    /// Raw access records in arrival order.
    records: Vec<AccessRecord>,
    /// Pending prefetch tasks in priority order (max-heap).
    task_queue: BinaryHeap<TaskHeapEntry>,
}

impl PrefetchScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config,
            edges: Vec::new(),
            records: Vec::new(),
            task_queue: BinaryHeap::new(),
        }
    }

    /// Register an edge PoP.  Only registered edges will receive prefetch tasks.
    pub fn add_edge(&mut self, edge_id: &str) {
        if !self.edges.contains(&edge_id.to_string()) {
            self.edges.push(edge_id.to_string());
        }
    }

    /// Ingest an access record.
    pub fn record_access(&mut self, record: AccessRecord) {
        self.records.push(record);
        self.evict_stale();
    }

    /// Recompute content statistics and regenerate the prefetch task queue.
    ///
    /// Call this periodically (e.g. once per minute) to refresh the queue.
    /// Returns the number of new tasks generated.
    pub fn refresh(&mut self) -> usize {
        self.evict_stale();
        let stats = self.compute_stats();
        let mut count = 0usize;

        // Sort content by score descending; take top-N.
        let mut ranked: Vec<&ContentStats> = stats.values().collect();
        ranked.sort_by(|a, b| {
            b.popularity_score
                .partial_cmp(&a.popularity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(self.config.top_n_content);

        self.task_queue.clear();

        for stat in ranked {
            if stat.popularity_score < self.config.min_popularity_score {
                continue;
            }
            if stat.size_bytes > self.config.max_item_size_bytes {
                continue;
            }
            // Generate a task for every edge that does NOT already have the content.
            for edge in &self.edges {
                if !stat.cached_at_edges.contains(edge) {
                    let priority = stat.popularity_score.clamp(0.0, u32::MAX as f64) as u32;
                    let task = PrefetchTask::new(&stat.key, edge, priority, stat.size_bytes);
                    self.task_queue.push(TaskHeapEntry { priority, task });
                    count += 1;
                }
            }
        }
        count
    }

    /// Pop the highest-priority prefetch task from the queue.
    ///
    /// Returns `None` when the queue is empty.
    pub fn next_task(&mut self) -> Option<PrefetchTask> {
        self.task_queue.pop().map(|e| e.task)
    }

    /// Drain up to `limit` tasks from the queue in priority order.
    pub fn drain_tasks(&mut self, limit: usize) -> Vec<PrefetchTask> {
        let mut tasks = Vec::with_capacity(limit);
        for _ in 0..limit {
            match self.task_queue.pop() {
                Some(entry) => tasks.push(entry.task),
                None => break,
            }
        }
        tasks
    }

    /// Number of pending prefetch tasks.
    pub fn pending_task_count(&self) -> usize {
        self.task_queue.len()
    }

    /// Compute content statistics over the current access window.
    pub fn compute_stats(&self) -> HashMap<String, ContentStats> {
        let now = Instant::now();
        let half_life = self.config.decay_half_life_secs;

        let mut stats: HashMap<String, ContentStats> = HashMap::new();
        for record in &self.records {
            let age_secs = now
                .checked_duration_since(record.observed_at)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            // Exponential decay: weight = 2^(-age / half_life)
            let decay = 2.0_f64.powf(-age_secs / half_life);

            let stat = stats
                .entry(record.key.clone())
                .or_insert_with(|| ContentStats::new(&record.key));
            stat.access_count += 1;
            stat.size_bytes = record.size_bytes;

            if record.cache_hit {
                stat.hit_count += 1;
                stat.cached_at_edges.insert(record.edge_id.clone());
            } else {
                stat.miss_count += 1;
            }

            // Popularity: weighted by recency + miss penalty (misses are
            // more valuable to warm up than already-hot cached items).
            let miss_bonus = if record.cache_hit { 1.0 } else { 2.0 };
            stat.popularity_score += decay * miss_bonus;
        }
        stats
    }

    /// Top-N content keys by popularity score.
    pub fn top_content(&self, n: usize) -> Vec<(String, f64)> {
        let stats = self.compute_stats();
        let mut ranked: Vec<(String, f64)> = stats
            .iter()
            .map(|(k, s)| (k.clone(), s.popularity_score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(n);
        ranked
    }

    /// Number of access records currently in the rolling window.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// All registered edge IDs.
    pub fn edges(&self) -> &[String] {
        &self.edges
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn evict_stale(&mut self) {
        let cutoff = Instant::now()
            .checked_sub(self.config.window)
            .unwrap_or_else(Instant::now);
        self.records.retain(|r| r.observed_at >= cutoff);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_record(key: &str, edge: &str, hit: bool, size: u64) -> AccessRecord {
        AccessRecord::new(key, edge, hit, size)
    }

    // 1. add_edge registers edges
    #[test]
    fn test_add_edge() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-iad");
        sched.add_edge("pop-lon");
        assert_eq!(sched.edges().len(), 2);
    }

    // 2. add_edge is idempotent
    #[test]
    fn test_add_edge_idempotent() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-iad");
        sched.add_edge("pop-iad");
        assert_eq!(sched.edges().len(), 1);
    }

    // 3. record_access adds to window
    #[test]
    fn test_record_access_adds() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.record_access(make_record("/a", "pop-iad", true, 1000));
        assert_eq!(sched.record_count(), 1);
    }

    // 4. compute_stats accumulates hits and misses
    #[test]
    fn test_compute_stats_hits_misses() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.record_access(make_record("/video.mp4", "pop-iad", true, 5000));
        sched.record_access(make_record("/video.mp4", "pop-lon", false, 5000));
        sched.record_access(make_record("/video.mp4", "pop-iad", false, 5000));
        let stats = sched.compute_stats();
        let s = stats.get("/video.mp4").expect("stats");
        assert_eq!(s.access_count, 3);
        assert_eq!(s.hit_count, 1);
        assert_eq!(s.miss_count, 2);
    }

    // 5. cached_at_edges tracks which edges have the content
    #[test]
    fn test_cached_at_edges() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.record_access(make_record("/a", "pop-iad", true, 100));
        sched.record_access(make_record("/a", "pop-lon", false, 100));
        let stats = sched.compute_stats();
        let s = stats.get("/a").expect("stats");
        assert!(s.cached_at_edges.contains("pop-iad"));
        assert!(!s.cached_at_edges.contains("pop-lon"));
    }

    // 6. refresh generates tasks for edges missing content
    #[test]
    fn test_refresh_generates_tasks() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-iad");
        sched.add_edge("pop-lon");
        // Content hit at pop-iad but not at pop-lon.
        for _ in 0..5 {
            sched.record_access(make_record("/popular.mp4", "pop-iad", true, 1000));
        }
        let generated = sched.refresh();
        assert!(generated > 0, "should generate at least one task");
        let task = sched.next_task().expect("task");
        assert_eq!(task.content_key, "/popular.mp4");
        assert_eq!(task.target_edge, "pop-lon");
    }

    // 7. refresh skips content already cached at edge
    #[test]
    fn test_refresh_skips_already_cached() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-iad");
        for _ in 0..5 {
            sched.record_access(make_record("/video.mp4", "pop-iad", true, 1000));
        }
        sched.refresh();
        // No tasks — pop-iad already has the content.
        assert_eq!(sched.pending_task_count(), 0);
    }

    // 8. refresh skips oversized content
    #[test]
    fn test_refresh_skips_oversized() {
        let config = PrefetchConfig::default().with_top_n(10).with_min_score(0.0);
        let mut sched = PrefetchScheduler::new(config);
        sched.add_edge("pop-lon");
        // Content larger than max_item_size_bytes (100 MiB).
        let huge = 200 * 1024 * 1024;
        sched.record_access(make_record("/huge.iso", "pop-iad", true, huge));
        sched.refresh();
        assert_eq!(sched.pending_task_count(), 0);
    }

    // 9. drain_tasks returns tasks in order
    #[test]
    fn test_drain_tasks() {
        let mut sched =
            PrefetchScheduler::new(PrefetchConfig::default().with_min_score(0.0).with_top_n(20));
        sched.add_edge("pop-lon");
        sched.add_edge("pop-tok");
        for _ in 0..10 {
            sched.record_access(make_record("/hot.mp4", "pop-iad", true, 500));
        }
        for _ in 0..3 {
            sched.record_access(make_record("/cold.mp4", "pop-iad", true, 200));
        }
        sched.refresh();
        let tasks = sched.drain_tasks(2);
        // All returned tasks should have the highest priority (/hot.mp4).
        for t in &tasks {
            assert_eq!(t.content_key, "/hot.mp4");
        }
    }

    // 10. top_content returns ranked list
    #[test]
    fn test_top_content() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        for _ in 0..10 {
            sched.record_access(make_record("/popular", "e", false, 100));
        }
        for _ in 0..3 {
            sched.record_access(make_record("/rare", "e", false, 100));
        }
        let top = sched.top_content(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "/popular");
    }

    // 11. ContentStats miss_rate
    #[test]
    fn test_content_stats_miss_rate() {
        let mut s = ContentStats::new("/x");
        s.access_count = 10;
        s.miss_count = 3;
        assert!((s.miss_rate() - 0.3).abs() < 1e-10);
        let empty = ContentStats::new("/y");
        assert!((empty.miss_rate() - 0.0).abs() < 1e-10);
    }

    // 12. PrefetchConfig builder
    #[test]
    fn test_prefetch_config_builder() {
        let cfg = PrefetchConfig::default()
            .with_window(Duration::from_secs(600))
            .with_top_n(50)
            .with_min_score(5.0)
            .with_decay_half_life_secs(120.0);
        assert_eq!(cfg.window, Duration::from_secs(600));
        assert_eq!(cfg.top_n_content, 50);
        assert!((cfg.min_popularity_score - 5.0).abs() < 1e-10);
        assert!((cfg.decay_half_life_secs - 120.0).abs() < 1e-10);
    }

    // 13. Recency decay: newer records get higher scores
    #[test]
    fn test_recency_decay() {
        let config = PrefetchConfig::default().with_decay_half_life_secs(1.0); // 1-second half-life
        let mut sched = PrefetchScheduler::new(config);
        // Old access (sleep to age it).
        sched.record_access(make_record("/old", "e", false, 100));
        thread::sleep(Duration::from_millis(200));
        // Fresh access.
        sched.record_access(make_record("/new", "e", false, 100));

        let stats = sched.compute_stats();
        let old_score = stats.get("/old").map(|s| s.popularity_score).unwrap_or(0.0);
        let new_score = stats.get("/new").map(|s| s.popularity_score).unwrap_or(0.0);
        assert!(
            new_score > old_score,
            "newer content should score higher: new={new_score} old={old_score}"
        );
    }

    // 14. Misses score double compared to hits
    #[test]
    fn test_miss_bonus_scoring() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.record_access(make_record("/hit", "e", true, 100));
        sched.record_access(make_record("/miss", "e", false, 100));
        let stats = sched.compute_stats();
        let hit_score = stats.get("/hit").map(|s| s.popularity_score).unwrap_or(0.0);
        let miss_score = stats
            .get("/miss")
            .map(|s| s.popularity_score)
            .unwrap_or(0.0);
        // Both recorded at approximately the same time so decay is equal.
        assert!(
            miss_score > hit_score,
            "miss should score higher: miss={miss_score} hit={hit_score}"
        );
    }

    // 15. window eviction removes old records
    #[test]
    fn test_window_eviction() {
        let config = PrefetchConfig::default().with_window(Duration::from_millis(10));
        let mut sched = PrefetchScheduler::new(config);
        sched.record_access(make_record("/old", "e", true, 100));
        thread::sleep(Duration::from_millis(20));
        // Ingest a new record which triggers eviction.
        sched.record_access(make_record("/new", "e", true, 100));
        assert_eq!(sched.record_count(), 1, "old record should be evicted");
    }

    // 16. Multiple edges get individual tasks
    #[test]
    fn test_multiple_edges_get_tasks() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-a");
        sched.add_edge("pop-b");
        sched.add_edge("pop-c");
        for _ in 0..3 {
            sched.record_access(make_record("/video.ts", "pop-src", false, 500));
        }
        sched.refresh();
        // 3 edges × 1 content item = 3 tasks.
        assert_eq!(sched.pending_task_count(), 3);
    }

    // 17. refresh clears and rebuilds queue
    #[test]
    fn test_refresh_rebuilds_queue() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-a");
        sched.record_access(make_record("/x", "pop-src", false, 100));
        sched.refresh();
        let count1 = sched.pending_task_count();
        sched.refresh(); // second refresh rebuilds from scratch
        let count2 = sched.pending_task_count();
        assert_eq!(count1, count2, "refresh should be idempotent");
    }

    // 18. pending_task_count decreases as tasks are consumed
    #[test]
    fn test_pending_task_count_decreases() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        sched.add_edge("pop-a");
        sched.add_edge("pop-b");
        sched.record_access(make_record("/v", "pop-src", false, 100));
        sched.refresh();
        let before = sched.pending_task_count();
        sched.next_task();
        assert_eq!(sched.pending_task_count(), before - 1);
    }

    // 19. PrefetchTask fields
    #[test]
    fn test_prefetch_task_fields() {
        let t = PrefetchTask::new("/segment.ts", "pop-lon", 42, 65536);
        assert_eq!(t.content_key, "/segment.ts");
        assert_eq!(t.target_edge, "pop-lon");
        assert_eq!(t.priority, 42);
        assert_eq!(t.size_bytes, 65536);
    }

    // 20. No tasks when no edges registered
    #[test]
    fn test_no_tasks_without_edges() {
        let mut sched = PrefetchScheduler::new(PrefetchConfig::default());
        for _ in 0..5 {
            sched.record_access(make_record("/popular", "pop-src", false, 100));
        }
        sched.refresh();
        assert_eq!(sched.pending_task_count(), 0);
    }
}
