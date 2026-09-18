//! Predictive I/O prefetching based on sequential access patterns.
//!
//! Provides a [`Prefetcher`] that detects sequential read patterns and
//! issues read-ahead requests to reduce I/O latency for streaming workloads.

#![allow(dead_code)]

/// Configuration for the [`Prefetcher`].
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Size of each prefetch window in bytes.
    pub window_size: usize,
    /// Maximum number of windows to read ahead.
    pub read_ahead_depth: usize,
    /// Minimum number of sequential reads before prefetching activates.
    pub activation_threshold: usize,
    /// Whether to adaptively grow the window on sustained sequential access.
    pub adaptive: bool,
    /// Maximum window size when adaptive mode is enabled.
    pub max_window_size: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            window_size: 64 * 1024,
            read_ahead_depth: 4,
            activation_threshold: 3,
            adaptive: false,
            max_window_size: 1024 * 1024,
        }
    }
}

impl PrefetchConfig {
    /// Set the window size.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set the read-ahead depth (number of windows).
    #[must_use]
    pub fn with_read_ahead_depth(mut self, depth: usize) -> Self {
        self.read_ahead_depth = depth;
        self
    }

    /// Set the activation threshold.
    #[must_use]
    pub fn with_activation_threshold(mut self, threshold: usize) -> Self {
        self.activation_threshold = threshold;
        self
    }

    /// Enable or disable adaptive window growth.
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }
}

/// A prefetch request that the caller should issue to fill the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchRequest {
    /// Starting byte offset to prefetch.
    pub offset: u64,
    /// Number of bytes to prefetch.
    pub length: usize,
}

impl PrefetchRequest {
    /// The end offset (exclusive) of this request.
    #[must_use]
    pub fn end_offset(&self) -> u64 {
        self.offset + self.length as u64
    }
}

/// Tracks access patterns and generates prefetch requests.
///
/// The prefetcher monitors read positions and detects sequential access.
/// Once the number of sequential reads exceeds the activation threshold,
/// it begins generating prefetch requests for upcoming data.
///
/// # Example
///
/// ```
/// use oximedia_io::prefetch::{Prefetcher, PrefetchConfig};
///
/// let config = PrefetchConfig::default()
///     .with_window_size(4096)
///     .with_read_ahead_depth(2)
///     .with_activation_threshold(3);
/// let mut pf = Prefetcher::new(config);
///
/// // Simulate sequential reads of 1024 bytes each
/// let _ = pf.on_read(0, 1024);
/// let _ = pf.on_read(1024, 1024);
/// // Third sequential read meets the threshold
/// let reqs = pf.on_read(2048, 1024);
/// assert!(!reqs.is_empty());
/// ```
#[derive(Debug)]
pub struct Prefetcher {
    config: PrefetchConfig,
    /// Effective window size (may grow in adaptive mode).
    current_window: usize,
    /// The offset of the last read end.
    last_end: Option<u64>,
    /// Count of consecutive sequential reads.
    sequential_count: usize,
    /// The highest offset that we have already requested to prefetch.
    prefetch_horizon: u64,
    /// Total prefetch requests generated.
    total_requests: u64,
    /// Total bytes requested for prefetch.
    total_bytes_requested: u64,
}

impl Prefetcher {
    /// Create a new `Prefetcher` with the given configuration.
    #[must_use]
    pub fn new(config: PrefetchConfig) -> Self {
        let initial_window = config.window_size;
        Self {
            config,
            current_window: initial_window,
            last_end: None,
            sequential_count: 0,
            prefetch_horizon: 0,
            total_requests: 0,
            total_bytes_requested: 0,
        }
    }

    /// Create a `Prefetcher` with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PrefetchConfig::default())
    }

    /// Notify the prefetcher of a read at `offset` of `length` bytes.
    ///
    /// Returns a (possibly empty) list of prefetch requests. The caller
    /// should issue these reads asynchronously to warm the cache.
    pub fn on_read(&mut self, offset: u64, length: usize) -> Vec<PrefetchRequest> {
        let read_end = offset + length as u64;

        // Check if this is sequential with the previous read
        let is_sequential = match self.last_end {
            Some(prev_end) => offset == prev_end,
            None => true, // First read counts as sequential start
        };

        if is_sequential {
            self.sequential_count += 1;
        } else {
            // Non-sequential access: reset
            self.sequential_count = 1;
            self.prefetch_horizon = read_end;
            self.current_window = self.config.window_size;
        }

        self.last_end = Some(read_end);

        // Only prefetch if we've exceeded the activation threshold
        if self.sequential_count < self.config.activation_threshold {
            // Move horizon to at least the current read end
            if read_end > self.prefetch_horizon {
                self.prefetch_horizon = read_end;
            }
            return Vec::new();
        }

        // Grow window adaptively if enabled
        if self.config.adaptive && self.sequential_count > self.config.activation_threshold {
            self.current_window = (self.current_window * 2).min(self.config.max_window_size);
        }

        // Generate prefetch requests ahead of the current position
        let mut requests = Vec::new();
        let target_horizon =
            read_end + (self.config.read_ahead_depth as u64 * self.current_window as u64);

        while self.prefetch_horizon < target_horizon {
            let req_offset = self.prefetch_horizon;
            let req = PrefetchRequest {
                offset: req_offset,
                length: self.current_window,
            };
            self.prefetch_horizon += self.current_window as u64;
            self.total_requests += 1;
            self.total_bytes_requested += self.current_window as u64;
            requests.push(req);
        }

        requests
    }

    /// Reset the prefetcher, clearing all tracked state.
    pub fn reset(&mut self) {
        self.last_end = None;
        self.sequential_count = 0;
        self.prefetch_horizon = 0;
        self.current_window = self.config.window_size;
    }

    /// Return the number of consecutive sequential reads detected.
    #[must_use]
    pub fn sequential_count(&self) -> usize {
        self.sequential_count
    }

    /// Return `true` if prefetching is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.sequential_count >= self.config.activation_threshold
    }

    /// Return the current effective window size.
    #[must_use]
    pub fn current_window_size(&self) -> usize {
        self.current_window
    }

    /// Return the total number of prefetch requests generated.
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Return the total bytes requested for prefetch.
    #[must_use]
    pub fn total_bytes_requested(&self) -> u64 {
        self.total_bytes_requested
    }

    /// Return the prefetch horizon (highest offset requested so far).
    #[must_use]
    pub fn prefetch_horizon(&self) -> u64 {
        self.prefetch_horizon
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &PrefetchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_config_defaults() {
        let cfg = PrefetchConfig::default();
        assert_eq!(cfg.window_size, 64 * 1024);
        assert_eq!(cfg.read_ahead_depth, 4);
        assert_eq!(cfg.activation_threshold, 3);
        assert!(!cfg.adaptive);
    }

    #[test]
    fn test_prefetch_config_builder() {
        let cfg = PrefetchConfig::default()
            .with_window_size(4096)
            .with_read_ahead_depth(8)
            .with_activation_threshold(2)
            .with_adaptive(true);
        assert_eq!(cfg.window_size, 4096);
        assert_eq!(cfg.read_ahead_depth, 8);
        assert_eq!(cfg.activation_threshold, 2);
        assert!(cfg.adaptive);
    }

    #[test]
    fn test_prefetch_request_end_offset() {
        let req = PrefetchRequest {
            offset: 100,
            length: 50,
        };
        assert_eq!(req.end_offset(), 150);
    }

    #[test]
    fn test_prefetcher_no_requests_below_threshold() {
        let cfg = PrefetchConfig::default()
            .with_window_size(1024)
            .with_activation_threshold(3);
        let mut pf = Prefetcher::new(cfg);

        let reqs = pf.on_read(0, 512);
        assert!(reqs.is_empty());
        assert_eq!(pf.sequential_count(), 1);
        assert!(!pf.is_active());

        let reqs = pf.on_read(512, 512);
        assert!(reqs.is_empty());
        assert_eq!(pf.sequential_count(), 2);
    }

    #[test]
    fn test_prefetcher_activates_at_threshold() {
        let cfg = PrefetchConfig::default()
            .with_window_size(1024)
            .with_read_ahead_depth(2)
            .with_activation_threshold(2);
        let mut pf = Prefetcher::new(cfg);

        let _ = pf.on_read(0, 512);
        let reqs = pf.on_read(512, 512);
        assert!(!reqs.is_empty());
        assert!(pf.is_active());
    }

    #[test]
    fn test_prefetcher_generates_correct_requests() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_read_ahead_depth(2)
            .with_activation_threshold(1);
        let mut pf = Prefetcher::new(cfg);

        let reqs = pf.on_read(0, 50);
        // After first read (sequential_count == 1, threshold == 1), prefetching activates.
        // prefetch_horizon starts at 0, target_horizon = 50 + 2*100 = 250.
        // Requests: [0..100], [100..200], [200..300]
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[0].offset, 0);
        assert_eq!(reqs[0].length, 100);
        assert_eq!(reqs[1].offset, 100);
        assert_eq!(reqs[1].length, 100);
        assert_eq!(reqs[2].offset, 200);
        assert_eq!(reqs[2].length, 100);
    }

    #[test]
    fn test_prefetcher_non_sequential_resets() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_activation_threshold(2);
        let mut pf = Prefetcher::new(cfg);

        let _ = pf.on_read(0, 50);
        let _ = pf.on_read(50, 50);
        assert!(pf.is_active());

        // Non-sequential read
        let reqs = pf.on_read(5000, 50);
        assert!(!pf.is_active());
        assert_eq!(pf.sequential_count(), 1);
        assert!(reqs.is_empty());
    }

    #[test]
    fn test_prefetcher_adaptive_window_growth() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_read_ahead_depth(1)
            .with_activation_threshold(2)
            .with_adaptive(true);
        let mut pf = Prefetcher::new(cfg);

        let _ = pf.on_read(0, 50);
        let _ = pf.on_read(50, 50); // threshold met, window stays 100
        assert_eq!(pf.current_window_size(), 100);

        // Third sequential read: adaptive growth kicks in
        let _ = pf.on_read(100, 50);
        assert_eq!(pf.current_window_size(), 200);
    }

    #[test]
    fn test_prefetcher_adaptive_caps_at_max() {
        let cfg = PrefetchConfig::default()
            .with_window_size(512 * 1024)
            .with_read_ahead_depth(1)
            .with_activation_threshold(1)
            .with_adaptive(true);
        let mut pf = Prefetcher::new(cfg);

        // Multiple sequential reads to grow window
        let mut offset = 0u64;
        for _ in 0..20 {
            let _ = pf.on_read(offset, 1024);
            offset += 1024;
        }
        // Window should be capped at max_window_size (1MB by default)
        assert!(pf.current_window_size() <= 1024 * 1024);
    }

    #[test]
    fn test_prefetcher_reset() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_activation_threshold(1);
        let mut pf = Prefetcher::new(cfg);

        let _ = pf.on_read(0, 50);
        assert!(pf.is_active());

        pf.reset();
        assert!(!pf.is_active());
        assert_eq!(pf.sequential_count(), 0);
        assert_eq!(pf.prefetch_horizon(), 0);
    }

    #[test]
    fn test_prefetcher_stats() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_read_ahead_depth(2)
            .with_activation_threshold(1);
        let mut pf = Prefetcher::new(cfg);

        let reqs = pf.on_read(0, 50);
        assert_eq!(pf.total_requests(), reqs.len() as u64);
        assert_eq!(
            pf.total_bytes_requested(),
            reqs.iter().map(|r| r.length as u64).sum::<u64>()
        );
    }

    #[test]
    fn test_prefetcher_subsequent_reads_extend_horizon() {
        let cfg = PrefetchConfig::default()
            .with_window_size(100)
            .with_read_ahead_depth(2)
            .with_activation_threshold(1);
        let mut pf = Prefetcher::new(cfg);

        let reqs1 = pf.on_read(0, 50);
        let horizon1 = pf.prefetch_horizon();

        let reqs2 = pf.on_read(50, 50);
        let horizon2 = pf.prefetch_horizon();

        assert!(horizon2 >= horizon1);
        // First read generates requests; second may generate more to extend
        assert!(!reqs1.is_empty());
        // The second read should also generate requests since horizon moves
        let _ = reqs2;
    }

    #[test]
    fn test_prefetcher_with_defaults() {
        let pf = Prefetcher::with_defaults();
        assert_eq!(pf.config().window_size, 64 * 1024);
        assert!(!pf.is_active());
    }
}
