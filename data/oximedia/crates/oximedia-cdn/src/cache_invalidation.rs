//! CDN cache invalidation — scoped purge requests, priority queuing,
//! per-node rate limiting, and glob-pattern matching.
//!
//! # Overview
//!
//! [`InvalidationQueue`] accepts [`InvalidationRequest`]s keyed by an
//! [`InvalidationScope`] and drains them in descending-priority order while
//! respecting a configurable per-node rate limit (default: 100 requests per
//! minute, enforced via a sliding 60-second window).
//!
//! Glob matching is implemented in pure Rust without any external crate:
//! - `*`  — any sequence of characters within a single path segment (no `/`).
//! - `?`  — exactly one character (no `/`).
//! - `**` — any sequence of characters including `/` (cross-segment, greedy).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise during cache invalidation.
#[derive(Debug, Error)]
pub enum InvalidationError {
    /// The queue has reached its maximum capacity.
    #[error("invalidation queue is full (capacity {0})")]
    QueueFull(usize),
    /// The per-node rate limit has been reached.
    #[error("rate limit exceeded: max {max_per_min} invalidations/min for node '{node}'")]
    RateLimitExceeded {
        /// Maximum allowed per minute.
        max_per_min: usize,
        /// The node identifier that hit the limit.
        node: String,
    },
    /// No matching invalidation was found.
    #[error("invalidation '{0}' not found")]
    NotFound(String),
}

// ─── InvalidationScope ────────────────────────────────────────────────────────

/// Defines the set of cached objects to be purged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidationScope {
    /// Purge exactly one URL.
    Url(String),
    /// Purge all URLs whose path starts with this prefix.
    PathPrefix(String),
    /// Purge all URLs matching this glob pattern (`*`, `?`, `**` supported).
    Glob(String),
    /// Purge all objects tagged with at least one of these cache tags.
    Tag(Vec<String>),
    /// Purge the entire cache.
    All,
}

impl InvalidationScope {
    /// Returns `true` if `url` is covered by this scope.
    pub fn matches(&self, url: &str) -> bool {
        match self {
            Self::Url(u) => u == url,
            Self::PathPrefix(prefix) => url.starts_with(prefix.as_str()),
            Self::Glob(pattern) => glob_match(pattern, url),
            Self::Tag(_) => {
                // Tag matching requires external metadata; always returns true
                // so callers can filter using their own tag index.
                true
            }
            Self::All => true,
        }
    }

    /// Human-readable label for the scope type (useful in logs/metrics).
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Url(_) => "url",
            Self::PathPrefix(_) => "path_prefix",
            Self::Glob(_) => "glob",
            Self::Tag(_) => "tag",
            Self::All => "all",
        }
    }
}

// ─── Glob matching (pure Rust, no external crate) ─────────────────────────────

/// Match `text` against `pattern` using `*`, `?`, and `**` wildcards.
///
/// Rules:
/// - `**` matches any sequence of characters including `/` (cross-segment).
/// - `*`  matches any characters **except** `/` (within one path segment).
/// - `?`  matches any single character **except** `/`.
/// - All other characters match literally (case-sensitive).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    glob_inner(pattern.as_bytes(), text.as_bytes())
}

/// Recursive glob matcher.  Uses a divide-and-conquer strategy for `**`:
/// try matching zero, one, or more characters (including `/`) at the current
/// text position, recursing on the rest of the pattern.
fn glob_inner(pat: &[u8], txt: &[u8]) -> bool {
    let mut pi = 0usize;
    let mut ti = 0usize;

    loop {
        // ── Consume all leading `*` or `**` wildcards ────────────────────────
        if pi < pat.len() && pat[pi] == b'*' {
            // Detect `**`
            let dstar = pi + 1 < pat.len() && pat[pi + 1] == b'*';
            if dstar {
                let after = {
                    let mut a = pi + 2;
                    // Skip optional `/` separator after `**`
                    if a < pat.len() && pat[a] == b'/' {
                        a += 1;
                    }
                    a
                };
                // Try matching the rest of the pattern at every text position
                // (including current) — `**` can match zero or more characters.
                for split in ti..=txt.len() {
                    if glob_inner(&pat[after..], &txt[split..]) {
                        return true;
                    }
                }
                return false;
            }

            // Single `*`: match within a segment (no `/`).
            // Advance `pi` past `*` and then try to match at each text position
            // that does not cross a `/`.
            pi += 1;
            loop {
                if glob_inner(&pat[pi..], &txt[ti..]) {
                    return true;
                }
                if ti >= txt.len() || txt[ti] == b'/' {
                    return false;
                }
                ti += 1;
            }
        }

        // ── No wildcard at `pi` — match character by character ───────────────
        if pi == pat.len() {
            return ti == txt.len();
        }

        if ti >= txt.len() {
            return false;
        }

        let literal_match = pat[pi] == txt[ti];
        let question_match = pat[pi] == b'?' && txt[ti] != b'/';
        if literal_match || question_match {
            pi += 1;
            ti += 1;
        } else {
            return false;
        }
    }
}

// ─── InvalidationRequest ──────────────────────────────────────────────────────

/// A single cache-purge request.
#[derive(Debug, Clone)]
pub struct InvalidationRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// What to purge.
    pub scope: InvalidationScope,
    /// Higher values are processed first (0–255). Ties broken by `submitted_at` (FIFO).
    pub priority: u8,
    /// Wall-clock time when this request was submitted.
    pub submitted_at: Instant,
}

impl InvalidationRequest {
    /// Create a new request with an auto-generated UUID-based ID.
    pub fn new(scope: InvalidationScope, priority: u8) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            scope,
            priority,
            submitted_at: Instant::now(),
        }
    }

    /// Create a request with an explicit ID (useful in tests).
    pub fn with_id(id: impl Into<String>, scope: InvalidationScope, priority: u8) -> Self {
        Self {
            id: id.into(),
            scope,
            priority,
            submitted_at: Instant::now(),
        }
    }
}

// ─── InvalidationResult ───────────────────────────────────────────────────────

/// The outcome of processing an [`InvalidationRequest`].
#[derive(Debug, Clone)]
pub struct InvalidationResult {
    /// The ID of the original request.
    pub id: String,
    /// The scope that was purged.
    pub scope: InvalidationScope,
    /// Number of edge nodes that acknowledged the purge.
    pub nodes_purged: usize,
    /// Wall-clock milliseconds from dispatch start to completion.
    pub duration_ms: u64,
    /// `true` if every targeted node confirmed the purge.
    pub success: bool,
}

// ─── Per-node sliding-window rate limiter ─────────────────────────────────────

/// Tracks request counts within a rolling 60-second window for one node.
#[derive(Debug)]
struct NodeRateLimiter {
    max_per_min: usize,
    /// Timestamps of successfully acquired slots, oldest first.
    timestamps: VecDeque<Instant>,
}

impl NodeRateLimiter {
    fn new(max_per_min: usize) -> Self {
        Self {
            max_per_min,
            timestamps: VecDeque::new(),
        }
    }

    /// Attempt to consume one slot.
    /// Returns `true` on success, `false` when the limit is already reached.
    fn try_acquire(&mut self, now: Instant) -> bool {
        self.evict_old(now);
        if self.timestamps.len() >= self.max_per_min {
            return false;
        }
        self.timestamps.push_back(now);
        true
    }

    /// Number of slots currently consumed within the active window.
    fn current_count(&self) -> usize {
        let cutoff = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .unwrap_or_else(Instant::now);
        self.timestamps.iter().filter(|&&t| t > cutoff).count()
    }

    fn evict_old(&mut self, now: Instant) {
        let cutoff = now.checked_sub(Duration::from_secs(60)).unwrap_or(now);
        while self
            .timestamps
            .front()
            .map(|&t| t <= cutoff)
            .unwrap_or(false)
        {
            self.timestamps.pop_front();
        }
    }
}

// ─── InvalidationQueue ────────────────────────────────────────────────────────

/// Priority queue of [`InvalidationRequest`]s with per-node rate limiting.
///
/// Requests are ordered by descending `priority`; equal-priority requests are
/// served FIFO (by `submitted_at`).
pub struct InvalidationQueue {
    /// Pending requests ordered: highest priority first, then oldest first.
    queue: Vec<InvalidationRequest>,
    /// Maximum number of requests that may be held at once.
    capacity: usize,
    /// Maximum invalidations dispatched to a single node within 60 seconds.
    max_per_node_per_min: usize,
    /// Per-node sliding-window rate limiters.
    node_limiters: HashMap<String, NodeRateLimiter>,
}

impl InvalidationQueue {
    /// Create a new queue with explicit limits.
    pub fn new(capacity: usize, max_per_node_per_min: usize) -> Self {
        Self {
            queue: Vec::new(),
            capacity,
            max_per_node_per_min,
            node_limiters: HashMap::new(),
        }
    }

    /// Create a queue with defaults (capacity 10 000, 100 invalidations/min/node).
    pub fn with_defaults() -> Self {
        Self::new(10_000, 100)
    }

    /// Submit a new invalidation request.
    ///
    /// Inserts in sorted order so that higher-priority requests appear first.
    /// Within the same priority, earlier `submitted_at` appears first (FIFO).
    pub fn submit(&mut self, request: InvalidationRequest) -> Result<(), InvalidationError> {
        if self.queue.len() >= self.capacity {
            return Err(InvalidationError::QueueFull(self.capacity));
        }
        let pos = self.queue.partition_point(|existing| {
            existing.priority > request.priority
                || (existing.priority == request.priority
                    && existing.submitted_at <= request.submitted_at)
        });
        self.queue.insert(pos, request);
        Ok(())
    }

    /// Process up to `batch_size` requests against the supplied `node_ids`,
    /// subject to the per-node rate limit.
    ///
    /// Requests that cannot be dispatched to **any** node (all at limit) are
    /// requeued.  Partially-dispatched requests are consumed with
    /// `success = false`.
    pub fn process_batch(
        &mut self,
        node_ids: &[&str],
        batch_size: usize,
    ) -> Vec<InvalidationResult> {
        let mut results: Vec<InvalidationResult> = Vec::new();
        let mut requeue: Vec<InvalidationRequest> = Vec::new();
        let now = Instant::now();

        for request in self.queue.drain(..) {
            if results.len() >= batch_size {
                requeue.push(request);
                continue;
            }

            let start = Instant::now();
            let mut nodes_purged = 0usize;
            let mut all_ok = true;

            for &node_id in node_ids {
                let limiter = self
                    .node_limiters
                    .entry(node_id.to_string())
                    .or_insert_with(|| NodeRateLimiter::new(self.max_per_node_per_min));

                if limiter.try_acquire(now) {
                    nodes_purged += 1;
                } else {
                    all_ok = false;
                }
            }

            if nodes_purged == 0 {
                requeue.push(request);
            } else {
                let duration_ms = start.elapsed().as_millis() as u64;
                results.push(InvalidationResult {
                    id: request.id,
                    scope: request.scope,
                    nodes_purged,
                    duration_ms,
                    success: all_ok,
                });
            }
        }

        requeue.append(&mut self.queue);
        self.queue = requeue;
        results
    }

    /// Drain and return all pending requests without dispatching them.
    pub fn drain(&mut self) -> Vec<InvalidationRequest> {
        std::mem::take(&mut self.queue)
    }

    /// Number of requests currently waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Current rate-limit consumption for `node_id` (requests in last 60 s).
    pub fn node_rate_usage(&self, node_id: &str) -> usize {
        self.node_limiters
            .get(node_id)
            .map(|l| l.current_count())
            .unwrap_or(0)
    }

    /// Reset the rate-limit counters for all nodes (useful for testing).
    pub fn reset_rate_limits(&mut self) {
        self.node_limiters.clear();
    }
}

// ─── Tag-based invalidation index ─────────────────────────────────────────────

/// An index that maps content tags to the URLs they are associated with.
///
/// This enables efficient tag-based cache invalidation: instead of scanning
/// every cached URL, callers register URL→tag associations and then query
/// which URLs need to be purged for a given set of tags.
pub struct TagIndex {
    /// tag → set of URLs carrying that tag.
    tag_to_urls: HashMap<String, Vec<String>>,
    /// url → set of tags.
    url_to_tags: HashMap<String, Vec<String>>,
}

impl TagIndex {
    /// Create an empty tag index.
    pub fn new() -> Self {
        Self {
            tag_to_urls: HashMap::new(),
            url_to_tags: HashMap::new(),
        }
    }

    /// Associate `url` with the given `tags`.
    ///
    /// Duplicate associations are avoided. Calling this multiple times with
    /// the same URL merges the tag sets.
    pub fn associate(&mut self, url: &str, tags: &[&str]) {
        for tag in tags {
            let urls = self
                .tag_to_urls
                .entry((*tag).to_string())
                .or_insert_with(Vec::new);
            if !urls.contains(&url.to_string()) {
                urls.push(url.to_string());
            }
            let url_tags = self
                .url_to_tags
                .entry(url.to_string())
                .or_insert_with(Vec::new);
            if !url_tags.contains(&(*tag).to_string()) {
                url_tags.push((*tag).to_string());
            }
        }
    }

    /// Return all URLs associated with **any** of the given `tags`.
    pub fn urls_for_tags(&self, tags: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        for tag in tags {
            if let Some(urls) = self.tag_to_urls.get(tag) {
                for url in urls {
                    if !result.contains(url) {
                        result.push(url.clone());
                    }
                }
            }
        }
        result
    }

    /// Return all tags associated with `url`.
    pub fn tags_for_url(&self, url: &str) -> Vec<String> {
        self.url_to_tags.get(url).cloned().unwrap_or_default()
    }

    /// Remove a URL from the index entirely (all tag associations).
    pub fn remove_url(&mut self, url: &str) {
        if let Some(tags) = self.url_to_tags.remove(url) {
            for tag in &tags {
                if let Some(urls) = self.tag_to_urls.get_mut(tag) {
                    urls.retain(|u| u != url);
                }
            }
        }
    }

    /// Remove a tag entirely (all URL associations for that tag).
    pub fn remove_tag(&mut self, tag: &str) {
        if let Some(urls) = self.tag_to_urls.remove(tag) {
            for url in &urls {
                if let Some(url_tags) = self.url_to_tags.get_mut(url) {
                    url_tags.retain(|t| t != tag);
                }
            }
        }
    }

    /// Total number of unique URLs in the index.
    pub fn url_count(&self) -> usize {
        self.url_to_tags.len()
    }

    /// Total number of unique tags in the index.
    pub fn tag_count(&self) -> usize {
        self.tag_to_urls.len()
    }
}

impl Default for TagIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Soft-purge / stale-while-revalidate ──────────────────────────────────────

/// The freshness state of a cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// The entry is fresh and can be served directly.
    Fresh,
    /// The entry is stale but can be served while revalidation is in progress.
    Stale,
    /// The entry has been hard-purged and must not be served.
    Purged,
}

/// A cache entry with soft-purge support.
///
/// Instead of immediately removing content from the cache on invalidation,
/// a soft-purge marks the entry as *stale*. The CDN continues serving the
/// stale content while asynchronously revalidating from the origin.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached URL.
    pub url: String,
    /// Content tags associated with this entry.
    pub tags: Vec<String>,
    /// Current freshness state.
    pub state: CacheState,
    /// Timestamp when the entry was last validated (seconds since UNIX epoch).
    pub last_validated: u64,
    /// Maximum age in seconds before the entry becomes stale.
    pub max_age_secs: u64,
    /// Grace period in seconds during which a stale entry can still be served.
    pub stale_while_revalidate_secs: u64,
}

impl CacheEntry {
    /// Create a new fresh cache entry.
    pub fn new(url: impl Into<String>, max_age_secs: u64) -> Self {
        Self {
            url: url.into(),
            tags: Vec::new(),
            state: CacheState::Fresh,
            last_validated: current_epoch_secs(),
            max_age_secs,
            stale_while_revalidate_secs: 0,
        }
    }

    /// Set the stale-while-revalidate grace period.
    pub fn with_stale_while_revalidate(mut self, secs: u64) -> Self {
        self.stale_while_revalidate_secs = secs;
        self
    }

    /// Add content tags to this entry.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Compute the current state based on the provided `now` (epoch seconds).
    pub fn effective_state(&self, now: u64) -> CacheState {
        if self.state == CacheState::Purged {
            return CacheState::Purged;
        }
        let age = now.saturating_sub(self.last_validated);
        if age <= self.max_age_secs {
            CacheState::Fresh
        } else if age <= self.max_age_secs + self.stale_while_revalidate_secs {
            CacheState::Stale
        } else {
            CacheState::Purged
        }
    }

    /// Soft-purge: mark as stale so it can be served during revalidation.
    pub fn soft_purge(&mut self) {
        if self.state != CacheState::Purged {
            self.state = CacheState::Stale;
        }
    }

    /// Hard-purge: mark as purged, must not be served.
    pub fn hard_purge(&mut self) {
        self.state = CacheState::Purged;
    }

    /// Revalidate: reset to fresh state with a new validation timestamp.
    pub fn revalidate(&mut self, now: u64) {
        self.state = CacheState::Fresh;
        self.last_validated = now;
    }

    /// Returns `true` if this entry can be served (fresh or stale within grace).
    pub fn is_servable(&self, now: u64) -> bool {
        matches!(
            self.effective_state(now),
            CacheState::Fresh | CacheState::Stale
        )
    }
}

/// Policy for soft-purge behaviour.
#[derive(Debug, Clone)]
pub struct SoftPurgePolicy {
    /// Default stale-while-revalidate window in seconds.
    pub default_grace_secs: u64,
    /// Maximum allowed grace period in seconds.
    pub max_grace_secs: u64,
    /// Whether to log stale-serve events.
    pub log_stale_serves: bool,
}

impl Default for SoftPurgePolicy {
    fn default() -> Self {
        Self {
            default_grace_secs: 60,
            max_grace_secs: 3600,
            log_stale_serves: false,
        }
    }
}

impl SoftPurgePolicy {
    /// Clamp a requested grace period to the maximum.
    pub fn clamp_grace(&self, requested: u64) -> u64 {
        requested.min(self.max_grace_secs)
    }

    /// Create a [`CacheEntry`] with this policy's default grace period.
    pub fn new_entry(&self, url: impl Into<String>, max_age_secs: u64) -> CacheEntry {
        CacheEntry::new(url, max_age_secs).with_stale_while_revalidate(self.default_grace_secs)
    }
}

/// Helper: current epoch seconds (best-effort, no unwrap).
fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Legacy InvalidationManager (kept for backward compatibility) ─────────────

/// Priority class for use with [`InvalidationManager`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum InvalidationPriority {
    /// Must be processed before all others.
    Immediate,
    /// High priority.
    High,
    /// Default priority.
    Normal,
    /// Background / best-effort.
    Low,
}

/// A pending cache-invalidation request for use with [`InvalidationManager`].
#[derive(Debug, Clone)]
pub struct ManagedInvalidationRequest {
    /// Unique request identifier.
    pub id: String,
    /// What to invalidate.
    pub scope: InvalidationScope,
    /// Edge-node IDs to target; empty means all nodes.
    pub nodes: Vec<String>,
    /// When the request was submitted.
    pub submitted_at: SystemTime,
    /// Processing priority.
    pub priority: InvalidationPriority,
}

/// The outcome of processing one managed invalidation request against one node.
#[derive(Debug, Clone)]
pub struct ManagedInvalidationResult {
    /// ID of the originating request.
    pub request_id: String,
    /// Edge-node ID that was targeted.
    pub node_id: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Number of URL entries purged (simulated).
    pub urls_purged: u64,
    /// Elapsed time in milliseconds (simulated).
    pub duration_ms: u64,
}

static MGR_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// High-level manager that queues, prioritises, and simulates cache purges.
pub struct InvalidationManager {
    pending: VecDeque<ManagedInvalidationRequest>,
    history: Vec<ManagedInvalidationResult>,
    max_history: usize,
    rate_limit_per_min: u32,
    requests_this_minute: u32,
    /// Tag-based invalidation index embedded in the manager.
    tag_store: TagInvalidationStore,
    /// Active stale-while-revalidate entries (key → entry).
    stale_entries: HashMap<String, StaleCacheEntry>,
}

// ─── Tag-based invalidation store ─────────────────────────────────────────────

/// A path → tag mapping store enabling tag-based bulk invalidation.
///
/// Paths are registered with zero or more content tags. Calling
/// [`TagInvalidationStore::invalidate_by_tag`] purges every path associated
/// with the given tag and returns the list of affected paths.
#[derive(Debug, Default)]
pub struct TagInvalidationStore {
    /// path → tags.
    path_to_tags: HashMap<String, Vec<String>>,
    /// tag → paths.
    tag_to_paths: HashMap<String, Vec<String>>,
}

impl TagInvalidationStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Associate `path` with the given `tags`.
    ///
    /// Duplicate associations are silently ignored. Multiple calls for the
    /// same path merge the tag sets.
    pub fn tag_entry(&mut self, path: &str, tags: Vec<String>) {
        for tag in &tags {
            let paths = self
                .tag_to_paths
                .entry(tag.clone())
                .or_insert_with(Vec::new);
            if !paths.contains(&path.to_string()) {
                paths.push(path.to_string());
            }
            let path_tags = self
                .path_to_tags
                .entry(path.to_string())
                .or_insert_with(Vec::new);
            if !path_tags.contains(tag) {
                path_tags.push(tag.clone());
            }
        }
    }

    /// Purge all paths whose tags include `tag`.
    ///
    /// Removes every matching path from the store entirely (including its
    /// associations with other tags) and returns the list of purged paths.
    /// Returns an empty `Vec` if no paths carry the given tag.
    pub fn invalidate_by_tag(&mut self, tag: &str) -> Vec<String> {
        let paths = match self.tag_to_paths.remove(tag) {
            None => return Vec::new(),
            Some(p) => p,
        };
        for path in &paths {
            // Remove path → tags entry and clean back-references.
            if let Some(path_tags) = self.path_to_tags.remove(path) {
                for other_tag in &path_tags {
                    if other_tag == tag {
                        continue;
                    }
                    if let Some(tag_paths) = self.tag_to_paths.get_mut(other_tag) {
                        tag_paths.retain(|p| p != path);
                    }
                }
            }
        }
        paths
    }

    /// Return the tags currently associated with `path`.
    pub fn tags_for_path(&self, path: &str) -> Vec<String> {
        self.path_to_tags.get(path).cloned().unwrap_or_default()
    }

    /// Return the paths currently associated with `tag`.
    pub fn paths_for_tag(&self, tag: &str) -> Vec<String> {
        self.tag_to_paths.get(tag).cloned().unwrap_or_default()
    }

    /// Total number of tracked paths.
    pub fn path_count(&self) -> usize {
        self.path_to_tags.len()
    }

    /// Total number of tracked tags.
    pub fn tag_count(&self) -> usize {
        self.tag_to_paths.len()
    }
}

impl InvalidationManager {
    /// Create a new manager with the given per-minute rate limit.
    pub fn new(rate_limit_per_min: u32) -> Self {
        Self {
            pending: VecDeque::new(),
            history: Vec::new(),
            max_history: 1000,
            rate_limit_per_min,
            requests_this_minute: 0,
            tag_store: TagInvalidationStore::new(),
            stale_entries: HashMap::new(),
        }
    }

    /// Associate `path` with the given content `tags`.
    ///
    /// Paths registered here can later be bulk-purged by tag using
    /// [`Self::invalidate_by_tag`].
    pub fn tag_entry(&mut self, path: &str, tags: Vec<String>) {
        self.tag_store.tag_entry(path, tags);
    }

    /// Purge all paths carrying `tag` and return the list of invalidated paths.
    ///
    /// The affected paths are removed from the internal tag store. If no paths
    /// carry the tag an empty `Vec` is returned.
    pub fn invalidate_by_tag(&mut self, tag: &str) -> Vec<String> {
        self.tag_store.invalidate_by_tag(tag)
    }

    /// Submit an invalidation request, returning its generated ID.
    ///
    /// `Immediate` and `High` priority requests go to the front of the queue.
    pub fn submit(
        &mut self,
        scope: InvalidationScope,
        nodes: Vec<String>,
        priority: InvalidationPriority,
    ) -> String {
        let id = format!(
            "inv-{}",
            MGR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let req = ManagedInvalidationRequest {
            id: id.clone(),
            scope,
            nodes,
            submitted_at: SystemTime::now(),
            priority: priority.clone(),
        };
        match priority {
            InvalidationPriority::Immediate | InvalidationPriority::High => {
                self.pending.push_front(req);
            }
            InvalidationPriority::Normal | InvalidationPriority::Low => {
                self.pending.push_back(req);
            }
        }
        id
    }

    /// Dequeue the next request if the rate limit allows.
    pub fn process_next(&mut self) -> Option<&ManagedInvalidationResult> {
        if self.requests_this_minute >= self.rate_limit_per_min {
            return None;
        }
        let req = self.pending.pop_front()?;
        self.requests_this_minute += 1;
        self.history.push(ManagedInvalidationResult {
            request_id: req.id.clone(),
            node_id: "__pending__".to_string(),
            success: false,
            urls_purged: 0,
            duration_ms: 0,
        });
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }
        self.history.last()
    }

    /// Reset the per-minute request counter.
    pub fn reset_rate_counter(&mut self) {
        self.requests_this_minute = 0;
    }

    /// Simulate a purge for `request`, returning one result per targeted node.
    pub fn simulate_purge(
        &mut self,
        request: &ManagedInvalidationRequest,
    ) -> Vec<ManagedInvalidationResult> {
        let targets: Vec<String> = if request.nodes.is_empty() {
            vec!["__all__".to_string()]
        } else {
            request.nodes.clone()
        };

        let (urls_purged, duration_ms) = match &request.scope {
            InvalidationScope::Url(_) => (1u64, 5u64),
            InvalidationScope::PathPrefix(_) => (50, 20),
            InvalidationScope::Tag(_) => (25, 15),
            InvalidationScope::All => (10_000, 500),
            InvalidationScope::Glob(_) => (100, 30),
        };

        let results: Vec<ManagedInvalidationResult> = targets
            .iter()
            .map(|node_id| ManagedInvalidationResult {
                request_id: request.id.clone(),
                node_id: node_id.clone(),
                success: true,
                urls_purged,
                duration_ms,
            })
            .collect();

        for r in &results {
            self.history.push(r.clone());
            while self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
        results
    }

    /// Number of requests still pending.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Full history slice (oldest first).
    pub fn history(&self) -> &[ManagedInvalidationResult] {
        &self.history
    }

    /// Fraction of historical results that were successful.
    /// Returns `1.0` when the history is empty.
    pub fn success_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let successes = self.history.iter().filter(|r| r.success).count();
        successes as f64 / self.history.len() as f64
    }
}

// ─── SoftPurge / StaleCacheEntry on InvalidationManager ──────────────────────

/// A soft-purge directive: marks an entry stale rather than removing it
/// immediately, allowing continued serving during background revalidation.
#[derive(Debug, Clone)]
pub struct SoftPurge {
    /// Cache key (URL path) to soft-purge.
    pub key: String,
    /// How long (seconds) the entry may continue to be served while stale.
    pub stale_while_revalidate_secs: u64,
}

/// A stale cache entry tracked by [`InvalidationManager`].
#[derive(Debug, Clone)]
pub struct StaleCacheEntry {
    /// The cache key (URL path).
    pub key: String,
    /// When the entry became stale.
    pub stale_at: Instant,
    /// How long the entry may be served while stale.
    pub ttl: Duration,
}

impl StaleCacheEntry {
    /// Returns `true` if the stale-while-revalidate grace period has not yet
    /// expired (i.e., the entry may still be served).
    pub fn is_still_servable(&self) -> bool {
        self.stale_at.elapsed() < self.ttl
    }
}

impl InvalidationManager {
    /// Soft-purge `key`: register it as stale for `ttl` duration.
    ///
    /// Returns `true` if the entry was newly registered (or refreshed);
    /// `false` if `ttl` is zero (no-op).
    pub fn soft_purge(&mut self, key: &str, ttl: Duration) -> bool {
        if ttl.is_zero() {
            return false;
        }
        self.stale_entries.insert(
            key.to_string(),
            StaleCacheEntry {
                key: key.to_string(),
                stale_at: Instant::now(),
                ttl,
            },
        );
        true
    }

    /// Returns `true` if `key` is currently tracked as stale (and within its
    /// TTL window).
    pub fn is_stale(&self, key: &str) -> bool {
        self.stale_entries
            .get(key)
            .map(|e| e.is_still_servable())
            .unwrap_or(false)
    }

    /// Return a reference to the [`StaleCacheEntry`] for `key` if it exists
    /// **and** is still within its grace period, otherwise `None`.
    pub fn serve_while_stale(&self, key: &str) -> Option<&StaleCacheEntry> {
        self.stale_entries
            .get(key)
            .filter(|e| e.is_still_servable())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── glob_match ──────────────────────────────────────────────────────────

    #[test]
    fn test_glob_exact_match() {
        assert!(glob_match("/foo/bar.jpg", "/foo/bar.jpg"));
        assert!(!glob_match("/foo/bar.jpg", "/foo/baz.jpg"));
    }

    #[test]
    fn test_glob_single_star_within_segment() {
        assert!(glob_match("/images/*.jpg", "/images/photo.jpg"));
        assert!(!glob_match("/images/*.jpg", "/images/sub/photo.jpg"));
    }

    #[test]
    fn test_glob_single_star_prefix() {
        assert!(glob_match("/api/*", "/api/v1"));
        assert!(!glob_match("/api/*", "/api/v1/users"));
    }

    #[test]
    fn test_glob_question_mark() {
        assert!(glob_match("/v?deo", "/video"));
        assert!(glob_match("/v?deo", "/vXdeo"));
        assert!(!glob_match("/v?deo", "/video/extra"));
    }

    #[test]
    fn test_glob_double_star_cross_segment() {
        assert!(glob_match("/assets/**", "/assets/css/main.css"));
        assert!(glob_match("/assets/**", "/assets/js/lib/bundle.js"));
    }

    #[test]
    fn test_glob_double_star_mid_pattern() {
        assert!(glob_match("/cdn/**/thumb.jpg", "/cdn/images/thumb.jpg"));
        assert!(glob_match("/cdn/**/thumb.jpg", "/cdn/a/b/c/thumb.jpg"));
        assert!(!glob_match("/cdn/**/thumb.jpg", "/cdn/a/nope.jpg"));
    }

    #[test]
    fn test_glob_no_wildcards() {
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "noexact"));
    }

    #[test]
    fn test_glob_empty_strings() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
        assert!(!glob_match("x", ""));
    }

    #[test]
    fn test_glob_trailing_star() {
        assert!(glob_match("abc*", "abcdef"));
        assert!(glob_match("abc*", "abc"));
    }

    // ── InvalidationScope::matches ──────────────────────────────────────────

    #[test]
    fn test_scope_url_matches() {
        let s = InvalidationScope::Url("/foo".to_string());
        assert!(s.matches("/foo"));
        assert!(!s.matches("/bar"));
    }

    #[test]
    fn test_scope_path_prefix_matches() {
        let s = InvalidationScope::PathPrefix("/static/".to_string());
        assert!(s.matches("/static/img.png"));
        assert!(!s.matches("/dynamic/data.json"));
    }

    #[test]
    fn test_scope_glob_matches() {
        let s = InvalidationScope::Glob("/cdn/**/*.css".to_string());
        assert!(s.matches("/cdn/v1/main.css"));
        assert!(s.matches("/cdn/v1/sub/theme.css"));
    }

    #[test]
    fn test_scope_all_matches_everything() {
        let s = InvalidationScope::All;
        assert!(s.matches("/anything"));
        assert!(s.matches(""));
    }

    #[test]
    fn test_scope_tag_matches_always() {
        let s = InvalidationScope::Tag(vec!["product".to_string(), "v2".to_string()]);
        assert!(s.matches("/product/123"));
        assert!(s.matches(""));
    }

    #[test]
    fn test_scope_kind_str() {
        assert_eq!(InvalidationScope::All.kind_str(), "all");
        assert_eq!(InvalidationScope::Url("/x".to_string()).kind_str(), "url");
        assert_eq!(
            InvalidationScope::PathPrefix("/".to_string()).kind_str(),
            "path_prefix"
        );
        assert_eq!(InvalidationScope::Glob("*".to_string()).kind_str(), "glob");
        assert_eq!(InvalidationScope::Tag(vec![]).kind_str(), "tag");
    }

    // ── InvalidationRequest ─────────────────────────────────────────────────

    #[test]
    fn test_request_has_unique_ids() {
        let r1 = InvalidationRequest::new(InvalidationScope::All, 5);
        let r2 = InvalidationRequest::new(InvalidationScope::All, 5);
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn test_request_priority_stored() {
        let r = InvalidationRequest::new(InvalidationScope::All, 200);
        assert_eq!(r.priority, 200);
    }

    #[test]
    fn test_request_with_id() {
        let r = InvalidationRequest::with_id("my-id", InvalidationScope::All, 10);
        assert_eq!(r.id, "my-id");
    }

    // ── InvalidationQueue ───────────────────────────────────────────────────

    #[test]
    fn test_queue_submit_and_pending_count() {
        let mut q = InvalidationQueue::with_defaults();
        assert_eq!(q.pending_count(), 0);
        q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .expect("submit ok");
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_queue_capacity_exceeded() {
        let mut q = InvalidationQueue::new(2, 100);
        q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .expect("first ok");
        q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .expect("second ok");
        let err = q
            .submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .unwrap_err();
        assert!(matches!(err, InvalidationError::QueueFull(2)));
    }

    #[test]
    fn test_queue_priority_ordering() {
        let mut q = InvalidationQueue::new(100, 1000);
        q.submit(InvalidationRequest::new(InvalidationScope::All, 10))
            .expect("submit priority-10 request");
        q.submit(InvalidationRequest::new(InvalidationScope::All, 50))
            .expect("submit priority-50 request");
        q.submit(InvalidationRequest::new(InvalidationScope::All, 30))
            .expect("submit priority-30 request");
        let drained = q.drain();
        assert_eq!(drained[0].priority, 50);
        assert_eq!(drained[1].priority, 30);
        assert_eq!(drained[2].priority, 10);
    }

    #[test]
    fn test_queue_process_batch() {
        let mut q = InvalidationQueue::new(100, 1000);
        for _ in 0..5 {
            q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("submit batch request");
        }
        let results = q.process_batch(&["node-1", "node-2"], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(q.pending_count(), 2);
        for r in &results {
            assert_eq!(r.nodes_purged, 2);
            assert!(r.success);
        }
    }

    #[test]
    fn test_queue_drain_empties_queue() {
        let mut q = InvalidationQueue::with_defaults();
        q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .expect("submit first drain request");
        q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
            .expect("submit second drain request");
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_queue_rate_limit_per_node() {
        let mut q = InvalidationQueue::new(1000, 2);
        for _ in 0..5 {
            q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("submit rate-limit test request");
        }
        let results = q.process_batch(&["node-x"], 5);
        assert_eq!(results.len(), 2);
        assert_eq!(q.pending_count(), 3);
    }

    #[test]
    fn test_queue_process_batch_respects_batch_size() {
        let mut q = InvalidationQueue::new(100, 1000);
        for _ in 0..10 {
            q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("submit batch-size test request");
        }
        let results = q.process_batch(&["n1"], 4);
        assert_eq!(results.len(), 4);
        assert_eq!(q.pending_count(), 6);
    }

    #[test]
    fn test_queue_fifo_within_same_priority() {
        let mut q = InvalidationQueue::new(100, 1000);
        let mut r1 = InvalidationRequest::new(InvalidationScope::Url("/a".to_string()), 5);
        r1.submitted_at = Instant::now();
        std::thread::sleep(Duration::from_millis(2));
        let mut r2 = InvalidationRequest::new(InvalidationScope::Url("/b".to_string()), 5);
        r2.submitted_at = Instant::now();
        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        q.submit(r1).expect("submit FIFO request r1");
        q.submit(r2).expect("submit FIFO request r2");
        let drained = q.drain();
        assert_eq!(drained[0].id, id1);
        assert_eq!(drained[1].id, id2);
    }

    #[test]
    fn test_node_rate_limiter_basic() {
        let mut limiter = NodeRateLimiter::new(3);
        let now = Instant::now();
        assert!(limiter.try_acquire(now));
        assert!(limiter.try_acquire(now));
        assert!(limiter.try_acquire(now));
        assert!(!limiter.try_acquire(now));
    }

    #[test]
    fn test_node_rate_limiter_reset_after_window() {
        let mut limiter = NodeRateLimiter::new(2);
        let now = Instant::now();
        assert!(limiter.try_acquire(now));
        assert!(limiter.try_acquire(now));
        assert!(!limiter.try_acquire(now));
        limiter.timestamps.clear();
        assert!(limiter.try_acquire(Instant::now()));
    }

    #[test]
    fn test_queue_node_rate_usage() {
        let mut q = InvalidationQueue::new(100, 5);
        for _ in 0..3 {
            q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("submit rate-usage test request");
        }
        q.process_batch(&["edge-1"], 3);
        assert_eq!(q.node_rate_usage("edge-1"), 3);
        assert_eq!(q.node_rate_usage("unknown-node"), 0);
    }

    #[test]
    fn test_queue_reset_rate_limits() {
        let mut q = InvalidationQueue::new(100, 2);
        for _ in 0..3 {
            q.submit(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("submit reset-rate-limits test request");
        }
        q.process_batch(&["n1"], 3);
        assert_eq!(q.pending_count(), 1);
        q.reset_rate_limits();
        q.process_batch(&["n1"], 1);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_invalidation_result_fields() {
        let result = InvalidationResult {
            id: "test-id".to_string(),
            scope: InvalidationScope::All,
            nodes_purged: 5,
            duration_ms: 42,
            success: true,
        };
        assert_eq!(result.nodes_purged, 5);
        assert!(result.success);
    }

    // ── InvalidationManager (legacy API) ────────────────────────────────────

    #[test]
    fn test_manager_submit_unique_ids() {
        let mut mgr = InvalidationManager::new(100);
        let id1 = mgr.submit(InvalidationScope::All, vec![], InvalidationPriority::Normal);
        let id2 = mgr.submit(InvalidationScope::All, vec![], InvalidationPriority::Normal);
        assert_ne!(id1, id2);
        assert_eq!(mgr.pending_count(), 2);
    }

    #[test]
    fn test_manager_immediate_priority_front() {
        let mut mgr = InvalidationManager::new(100);
        mgr.submit(
            InvalidationScope::Url("/a".to_string()),
            vec![],
            InvalidationPriority::Normal,
        );
        let urgent_id = mgr.submit(
            InvalidationScope::Url("/b".to_string()),
            vec![],
            InvalidationPriority::Immediate,
        );
        let next = mgr.process_next().expect("should have item");
        assert_eq!(next.request_id, urgent_id);
    }

    #[test]
    fn test_manager_rate_limiting() {
        let mut mgr = InvalidationManager::new(2);
        for _ in 0..5 {
            mgr.submit(InvalidationScope::All, vec![], InvalidationPriority::Normal);
        }
        assert!(mgr.process_next().is_some());
        assert!(mgr.process_next().is_some());
        assert!(mgr.process_next().is_none());
        mgr.reset_rate_counter();
        assert!(mgr.process_next().is_some());
    }

    #[test]
    fn test_manager_simulate_purge_per_node() {
        let mut mgr = InvalidationManager::new(100);
        let req = ManagedInvalidationRequest {
            id: "test-req".to_string(),
            scope: InvalidationScope::Url("/foo".to_string()),
            nodes: vec!["node-a".to_string(), "node-b".to_string()],
            submitted_at: SystemTime::now(),
            priority: InvalidationPriority::Normal,
        };
        let results = mgr.simulate_purge(&req);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
        assert_eq!(results[0].urls_purged, 1);
    }

    #[test]
    fn test_manager_simulate_all_nodes() {
        let mut mgr = InvalidationManager::new(100);
        let req = ManagedInvalidationRequest {
            id: "test-all".to_string(),
            scope: InvalidationScope::All,
            nodes: vec![],
            submitted_at: SystemTime::now(),
            priority: InvalidationPriority::Normal,
        };
        let results = mgr.simulate_purge(&req);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "__all__");
        assert_eq!(results[0].urls_purged, 10_000);
    }

    #[test]
    fn test_manager_success_rate() {
        let mut mgr = InvalidationManager::new(100);
        assert!((mgr.success_rate() - 1.0).abs() < 1e-10);
        let req = ManagedInvalidationRequest {
            id: "r1".to_string(),
            scope: InvalidationScope::PathPrefix("/media/".to_string()),
            nodes: vec!["n1".to_string()],
            submitted_at: SystemTime::now(),
            priority: InvalidationPriority::Normal,
        };
        mgr.simulate_purge(&req);
        assert!((mgr.success_rate() - 1.0).abs() < 1e-10);
        mgr.history.push(ManagedInvalidationResult {
            request_id: "r2".to_string(),
            node_id: "n2".to_string(),
            success: false,
            urls_purged: 0,
            duration_ms: 0,
        });
        assert!((mgr.success_rate() - 0.5).abs() < 1e-10);
    }

    // ── TagIndex ───────────────────────────────────────────────────────────

    #[test]
    fn test_tag_index_associate_and_query() {
        let mut idx = TagIndex::new();
        idx.associate("/img/hero.jpg", &["product", "homepage"]);
        idx.associate("/img/logo.png", &["homepage", "branding"]);

        let urls = idx.urls_for_tags(&["homepage".to_string()]);
        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"/img/hero.jpg".to_string()));
        assert!(urls.contains(&"/img/logo.png".to_string()));
    }

    #[test]
    fn test_tag_index_urls_for_multiple_tags() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["t1"]);
        idx.associate("/b", &["t2"]);
        idx.associate("/c", &["t1", "t2"]);

        let urls = idx.urls_for_tags(&["t1".to_string(), "t2".to_string()]);
        assert_eq!(urls.len(), 3);
    }

    #[test]
    fn test_tag_index_no_duplicates() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["tag"]);
        idx.associate("/a", &["tag"]); // duplicate
        let urls = idx.urls_for_tags(&["tag".to_string()]);
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn test_tag_index_tags_for_url() {
        let mut idx = TagIndex::new();
        idx.associate("/x", &["alpha", "beta"]);
        let tags = idx.tags_for_url("/x");
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"alpha".to_string()));
        assert!(tags.contains(&"beta".to_string()));
    }

    #[test]
    fn test_tag_index_remove_url() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["t1"]);
        idx.associate("/b", &["t1"]);
        idx.remove_url("/a");
        let urls = idx.urls_for_tags(&["t1".to_string()]);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "/b");
        assert_eq!(idx.url_count(), 1);
    }

    #[test]
    fn test_tag_index_remove_tag() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["t1", "t2"]);
        idx.remove_tag("t1");
        assert_eq!(idx.tag_count(), 1);
        let tags = idx.tags_for_url("/a");
        assert_eq!(tags, vec!["t2".to_string()]);
    }

    #[test]
    fn test_tag_index_empty_query() {
        let idx = TagIndex::new();
        let urls = idx.urls_for_tags(&["nonexistent".to_string()]);
        assert!(urls.is_empty());
    }

    #[test]
    fn test_tag_index_counts() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["t1", "t2"]);
        idx.associate("/b", &["t2", "t3"]);
        assert_eq!(idx.url_count(), 2);
        assert_eq!(idx.tag_count(), 3);
    }

    #[test]
    fn test_tag_index_merge_tags() {
        let mut idx = TagIndex::new();
        idx.associate("/a", &["t1"]);
        idx.associate("/a", &["t2"]);
        let tags = idx.tags_for_url("/a");
        assert_eq!(tags.len(), 2);
    }

    // ── CacheEntry / Soft-purge ────────────────────────────────────────────

    #[test]
    fn test_cache_entry_new_is_fresh() {
        let entry = CacheEntry::new("/resource", 300);
        assert_eq!(entry.state, CacheState::Fresh);
        assert_eq!(entry.max_age_secs, 300);
    }

    #[test]
    fn test_cache_entry_soft_purge() {
        let mut entry = CacheEntry::new("/res", 300);
        entry.soft_purge();
        assert_eq!(entry.state, CacheState::Stale);
    }

    #[test]
    fn test_cache_entry_hard_purge() {
        let mut entry = CacheEntry::new("/res", 300);
        entry.hard_purge();
        assert_eq!(entry.state, CacheState::Purged);
    }

    #[test]
    fn test_cache_entry_soft_purge_after_hard_purge_stays_purged() {
        let mut entry = CacheEntry::new("/res", 300);
        entry.hard_purge();
        entry.soft_purge();
        assert_eq!(entry.state, CacheState::Purged);
    }

    #[test]
    fn test_cache_entry_revalidate() {
        let mut entry = CacheEntry::new("/res", 300);
        entry.soft_purge();
        let now = current_epoch_secs();
        entry.revalidate(now);
        assert_eq!(entry.state, CacheState::Fresh);
        assert_eq!(entry.last_validated, now);
    }

    #[test]
    fn test_cache_entry_effective_state_fresh() {
        let entry = CacheEntry::new("/res", 300);
        let now = entry.last_validated;
        assert_eq!(entry.effective_state(now), CacheState::Fresh);
        assert_eq!(entry.effective_state(now + 299), CacheState::Fresh);
    }

    #[test]
    fn test_cache_entry_effective_state_stale() {
        let entry = CacheEntry::new("/res", 300).with_stale_while_revalidate(60);
        let now = entry.last_validated;
        // Just past max_age → stale
        assert_eq!(entry.effective_state(now + 301), CacheState::Stale);
        // Within grace period
        assert_eq!(entry.effective_state(now + 359), CacheState::Stale);
    }

    #[test]
    fn test_cache_entry_effective_state_purged_after_grace() {
        let entry = CacheEntry::new("/res", 300).with_stale_while_revalidate(60);
        let now = entry.last_validated;
        // Past grace period
        assert_eq!(entry.effective_state(now + 361), CacheState::Purged);
    }

    #[test]
    fn test_cache_entry_effective_state_hard_purge_overrides() {
        let mut entry = CacheEntry::new("/res", 300).with_stale_while_revalidate(60);
        entry.hard_purge();
        let now = entry.last_validated;
        assert_eq!(entry.effective_state(now), CacheState::Purged);
    }

    #[test]
    fn test_cache_entry_is_servable() {
        let entry = CacheEntry::new("/res", 300).with_stale_while_revalidate(60);
        let now = entry.last_validated;
        assert!(entry.is_servable(now)); // fresh
        assert!(entry.is_servable(now + 301)); // stale but in grace
        assert!(!entry.is_servable(now + 400)); // past grace
    }

    #[test]
    fn test_cache_entry_with_tags() {
        let entry =
            CacheEntry::new("/res", 300).with_tags(vec!["product".to_string(), "v2".to_string()]);
        assert_eq!(entry.tags.len(), 2);
    }

    // ── SoftPurgePolicy ────────────────────────────────────────────────────

    #[test]
    fn test_soft_purge_policy_defaults() {
        let policy = SoftPurgePolicy::default();
        assert_eq!(policy.default_grace_secs, 60);
        assert_eq!(policy.max_grace_secs, 3600);
    }

    #[test]
    fn test_soft_purge_policy_clamp_grace() {
        let policy = SoftPurgePolicy {
            max_grace_secs: 120,
            ..SoftPurgePolicy::default()
        };
        assert_eq!(policy.clamp_grace(60), 60);
        assert_eq!(policy.clamp_grace(200), 120);
    }

    #[test]
    fn test_soft_purge_policy_new_entry() {
        let policy = SoftPurgePolicy::default();
        let entry = policy.new_entry("/test", 600);
        assert_eq!(entry.stale_while_revalidate_secs, 60);
        assert_eq!(entry.max_age_secs, 600);
        assert_eq!(entry.state, CacheState::Fresh);
    }

    // ── Tag-based invalidation through InvalidationScope ──────────────────

    #[test]
    fn test_tag_invalidation_with_index() {
        let mut idx = TagIndex::new();
        idx.associate("/product/1", &["product", "featured"]);
        idx.associate("/product/2", &["product"]);
        idx.associate("/blog/1", &["blog"]);

        // Simulate tag-based invalidation
        let scope = InvalidationScope::Tag(vec!["product".to_string()]);
        assert_eq!(scope.kind_str(), "tag");

        // Use the tag index to resolve affected URLs
        if let InvalidationScope::Tag(ref tags) = scope {
            let urls = idx.urls_for_tags(tags);
            assert_eq!(urls.len(), 2);
            assert!(urls.contains(&"/product/1".to_string()));
            assert!(urls.contains(&"/product/2".to_string()));
        }
    }

    #[test]
    fn test_tag_invalidation_scope_matches_always() {
        // Tag scope matches() returns true (tag filtering is metadata-based)
        let scope = InvalidationScope::Tag(vec!["x".to_string()]);
        assert!(scope.matches("/anything"));
    }

    // ── SoftPurge / StaleCacheEntry on InvalidationManager ───────────────────

    #[test]
    fn test_soft_purge_marks_entry_stale() {
        let mut mgr = InvalidationManager::new(100);
        let ttl = Duration::from_secs(60);
        let inserted = mgr.soft_purge("/video/clip.mp4", ttl);
        assert!(inserted, "soft_purge should return true for non-zero TTL");
        assert!(mgr.is_stale("/video/clip.mp4"), "entry should be stale");
    }

    #[test]
    fn test_soft_purge_zero_ttl_is_noop() {
        let mut mgr = InvalidationManager::new(100);
        let inserted = mgr.soft_purge("/image/logo.png", Duration::ZERO);
        assert!(!inserted, "zero TTL should return false");
        assert!(
            !mgr.is_stale("/image/logo.png"),
            "no stale entry for zero TTL"
        );
    }

    #[test]
    fn test_serve_while_stale_returns_entry() {
        let mut mgr = InvalidationManager::new(100);
        let ttl = Duration::from_secs(300);
        mgr.soft_purge("/media/hls.m3u8", ttl);
        let entry = mgr.serve_while_stale("/media/hls.m3u8");
        assert!(entry.is_some(), "should serve stale entry within TTL");
        let e = entry.expect("entry present");
        assert_eq!(e.key, "/media/hls.m3u8");
        assert_eq!(e.ttl, ttl);
    }

    #[test]
    fn test_serve_while_stale_none_for_unknown_key() {
        let mgr = InvalidationManager::new(100);
        assert!(mgr.serve_while_stale("/nonexistent").is_none());
    }

    #[test]
    fn test_soft_purge_refresh_extends_window() {
        let mut mgr = InvalidationManager::new(100);
        mgr.soft_purge("/key", Duration::from_secs(10));
        // Refresh with longer TTL
        mgr.soft_purge("/key", Duration::from_secs(300));
        let entry = mgr.serve_while_stale("/key").expect("entry");
        assert_eq!(entry.ttl, Duration::from_secs(300));
    }

    #[test]
    fn test_stale_cache_entry_is_still_servable() {
        let entry = StaleCacheEntry {
            key: "/test".to_string(),
            stale_at: Instant::now(),
            ttl: Duration::from_secs(60),
        };
        assert!(
            entry.is_still_servable(),
            "freshly inserted entry should be servable"
        );
    }
}
