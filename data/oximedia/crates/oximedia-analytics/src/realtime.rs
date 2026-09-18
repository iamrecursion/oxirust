//! Real-time analytics aggregation with sliding window metrics.
//!
//! Provides a time-bucketed sliding window that tracks concurrent viewers,
//! bitrate statistics, and buffer event counts over a configurable window
//! horizon and bucket granularity.
//!
//! All timestamps are Unix epoch milliseconds.

use std::collections::VecDeque;

use crate::error::AnalyticsError;

// ─── Viewer event ─────────────────────────────────────────────────────────────

/// A real-time analytics event representing one viewer action.
#[derive(Debug, Clone)]
pub enum RealtimeEvent {
    /// A viewer started (or resumed) watching.
    ViewerJoin {
        viewer_id: String,
        timestamp_ms: i64,
    },
    /// A viewer left (or paused / closed the player).
    ViewerLeave {
        viewer_id: String,
        timestamp_ms: i64,
    },
    /// A bitrate sample from the player (bits per second).
    BitrateReport {
        viewer_id: String,
        timestamp_ms: i64,
        bitrate_bps: u64,
    },
    /// A buffering event with duration.
    BufferEvent {
        viewer_id: String,
        timestamp_ms: i64,
        duration_ms: u32,
    },
}

impl RealtimeEvent {
    fn timestamp_ms(&self) -> i64 {
        match self {
            RealtimeEvent::ViewerJoin { timestamp_ms, .. } => *timestamp_ms,
            RealtimeEvent::ViewerLeave { timestamp_ms, .. } => *timestamp_ms,
            RealtimeEvent::BitrateReport { timestamp_ms, .. } => *timestamp_ms,
            RealtimeEvent::BufferEvent { timestamp_ms, .. } => *timestamp_ms,
        }
    }
}

// ─── Per-bucket aggregation ───────────────────────────────────────────────────

/// Aggregated metrics for one time bucket.
#[derive(Debug, Clone, Default)]
pub struct BucketMetrics {
    /// Start of this bucket (epoch ms).
    pub bucket_start_ms: i64,
    /// Peak concurrent-viewer count observed within this bucket.
    pub peak_concurrent_viewers: u32,
    /// Average bitrate across all samples in this bucket (bps, 0 if none).
    pub avg_bitrate_bps: f64,
    /// Minimum bitrate sample in this bucket (0 if no samples).
    pub min_bitrate_bps: u64,
    /// Maximum bitrate sample in this bucket (0 if no samples).
    pub max_bitrate_bps: u64,
    /// Total number of buffer events in this bucket.
    pub buffer_event_count: u32,
    /// Total buffer stall time in this bucket (ms).
    pub buffer_stall_ms: u64,
    /// Number of bitrate samples contributing to avg/min/max.
    pub bitrate_sample_count: u32,
}

// ─── Sliding window aggregator ────────────────────────────────────────────────

/// A sliding window analytics aggregator for real-time media metrics.
///
/// Events are ingested in order via [`SlidingWindowAggregator::ingest`].  The
/// aggregator maintains a rolling window of `window_duration_ms` worth of
/// time buckets, each `bucket_ms` wide.  Old buckets outside the window are
/// automatically evicted.
#[derive(Debug)]
pub struct SlidingWindowAggregator {
    /// Window duration in milliseconds.
    window_duration_ms: i64,
    /// Bucket width in milliseconds.
    bucket_ms: i64,
    /// Ordered queue of active buckets (front = oldest).
    buckets: VecDeque<BucketMetrics>,
    /// Current concurrent viewer count (active joins minus leaves).
    concurrent_viewers: i64,
    /// Watermark: the latest timestamp seen.
    latest_ms: i64,
}

impl SlidingWindowAggregator {
    /// Create a new aggregator.
    ///
    /// Returns an error if `window_duration_ms < bucket_ms` or either is zero.
    pub fn new(window_duration_ms: i64, bucket_ms: i64) -> Result<Self, AnalyticsError> {
        if bucket_ms <= 0 || window_duration_ms <= 0 {
            return Err(AnalyticsError::ConfigError(
                "window and bucket duration must be positive".to_string(),
            ));
        }
        if window_duration_ms < bucket_ms {
            return Err(AnalyticsError::ConfigError(
                "window_duration_ms must be >= bucket_ms".to_string(),
            ));
        }
        Ok(Self {
            window_duration_ms,
            bucket_ms,
            buckets: VecDeque::new(),
            concurrent_viewers: 0,
            latest_ms: i64::MIN,
        })
    }

    /// Ingest a real-time event.
    ///
    /// Events should be delivered roughly in time order; out-of-order events
    /// within the current bucket are merged correctly, but events older than
    /// the window start are silently dropped.
    pub fn ingest(&mut self, event: RealtimeEvent) {
        let ts = event.timestamp_ms();
        if self.latest_ms == i64::MIN {
            self.latest_ms = ts;
        } else {
            self.latest_ms = self.latest_ms.max(ts);
        }

        // Evict expired buckets.
        let window_start = self.latest_ms - self.window_duration_ms;
        while self
            .buckets
            .front()
            .map(|b| b.bucket_start_ms + self.bucket_ms <= window_start)
            .unwrap_or(false)
        {
            self.buckets.pop_front();
        }

        // Find or create the bucket for this timestamp.
        let bucket_start = ts - ts.rem_euclid(self.bucket_ms);
        if bucket_start < window_start {
            // Event is too old; drop it.
            return;
        }

        let _bucket = self.get_or_create_bucket(bucket_start);

        // Update concurrent_viewers for join/leave before borrowing bucket.
        let new_concurrent = match &event {
            RealtimeEvent::ViewerJoin { .. } => {
                self.concurrent_viewers += 1;
                Some(self.concurrent_viewers.max(0) as u32)
            }
            RealtimeEvent::ViewerLeave { .. } => {
                self.concurrent_viewers = (self.concurrent_viewers - 1).max(0);
                None
            }
            _ => None,
        };

        let bucket = self.get_or_create_bucket(bucket_start);

        match &event {
            RealtimeEvent::ViewerJoin { .. } => {
                if let Some(c) = new_concurrent {
                    if c > bucket.peak_concurrent_viewers {
                        bucket.peak_concurrent_viewers = c;
                    }
                }
            }
            RealtimeEvent::ViewerLeave { .. } => {}
            RealtimeEvent::BitrateReport { bitrate_bps, .. } => {
                let bps = *bitrate_bps;
                bucket.bitrate_sample_count += 1;
                let n = bucket.bitrate_sample_count as f64;
                bucket.avg_bitrate_bps += (bps as f64 - bucket.avg_bitrate_bps) / n;
                if bucket.min_bitrate_bps == 0 || bps < bucket.min_bitrate_bps {
                    bucket.min_bitrate_bps = bps;
                }
                if bps > bucket.max_bitrate_bps {
                    bucket.max_bitrate_bps = bps;
                }
            }
            RealtimeEvent::BufferEvent { duration_ms, .. } => {
                bucket.buffer_event_count += 1;
                bucket.buffer_stall_ms += u64::from(*duration_ms);
            }
        }
    }

    /// Return a snapshot of all active buckets (oldest first).
    pub fn buckets(&self) -> &VecDeque<BucketMetrics> {
        &self.buckets
    }

    /// Current instantaneous concurrent viewer count.
    pub fn concurrent_viewers(&self) -> u32 {
        self.concurrent_viewers.max(0) as u32
    }

    /// Aggregate bitrate statistics across all active buckets.
    ///
    /// Returns `(avg_bps, min_bps, max_bps)`.  Returns `(0.0, 0, 0)` if no
    /// bitrate samples exist in the window.
    pub fn window_bitrate_stats(&self) -> (f64, u64, u64) {
        let mut total_weight = 0u64;
        let mut weighted_sum = 0.0f64;
        let mut min_bps = u64::MAX;
        let mut max_bps = 0u64;

        for bucket in &self.buckets {
            if bucket.bitrate_sample_count > 0 {
                let w = bucket.bitrate_sample_count as u64;
                total_weight += w;
                weighted_sum += bucket.avg_bitrate_bps * w as f64;
                if bucket.min_bitrate_bps < min_bps {
                    min_bps = bucket.min_bitrate_bps;
                }
                if bucket.max_bitrate_bps > max_bps {
                    max_bps = bucket.max_bitrate_bps;
                }
            }
        }

        if total_weight == 0 {
            return (0.0, 0, 0);
        }
        (weighted_sum / total_weight as f64, min_bps, max_bps)
    }

    /// Total buffer events in the current window.
    pub fn window_buffer_events(&self) -> u32 {
        self.buckets.iter().map(|b| b.buffer_event_count).sum()
    }

    /// Peak concurrent viewers across all active buckets.
    pub fn window_peak_concurrent(&self) -> u32 {
        self.buckets
            .iter()
            .map(|b| b.peak_concurrent_viewers)
            .max()
            .unwrap_or(0)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn get_or_create_bucket(&mut self, bucket_start: i64) -> &mut BucketMetrics {
        // Check if the newest bucket matches.
        if self
            .buckets
            .back()
            .map(|b| b.bucket_start_ms == bucket_start)
            .unwrap_or(false)
        {
            return self.buckets.back_mut().unwrap_or_else(|| {
                // Unreachable but needed for type safety.
                unreachable!("back_mut after back returned Some")
            });
        }

        // Find existing bucket or insert new one at the right position.
        let pos = self
            .buckets
            .iter()
            .position(|b| b.bucket_start_ms == bucket_start);

        if pos.is_none() {
            // Insert in sorted order.
            let insert_pos = self
                .buckets
                .iter()
                .position(|b| b.bucket_start_ms > bucket_start)
                .unwrap_or(self.buckets.len());
            self.buckets.insert(
                insert_pos,
                BucketMetrics {
                    bucket_start_ms: bucket_start,
                    ..Default::default()
                },
            );
        }

        // Now find and return the mutable reference.
        // Safety: we just inserted or confirmed the bucket exists above,
        // so this position lookup will always succeed. Use saturating
        // fallback to last element if somehow not found.
        let idx = self
            .buckets
            .iter()
            .position(|b| b.bucket_start_ms == bucket_start)
            .unwrap_or(self.buckets.len().saturating_sub(1));
        &mut self.buckets[idx]
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn aggregator() -> SlidingWindowAggregator {
        SlidingWindowAggregator::new(60_000, 10_000).expect("new should succeed")
    }

    // ── constructor ──────────────────────────────────────────────────────────

    #[test]
    fn aggregator_new_invalid_params() {
        assert!(SlidingWindowAggregator::new(0, 1000).is_err());
        assert!(SlidingWindowAggregator::new(1000, 0).is_err());
        assert!(SlidingWindowAggregator::new(500, 1000).is_err());
    }

    #[test]
    fn aggregator_new_valid() {
        assert!(SlidingWindowAggregator::new(60_000, 10_000).is_ok());
    }

    // ── concurrent viewers ───────────────────────────────────────────────────

    #[test]
    fn concurrent_viewers_join_leave() {
        let mut agg = aggregator();
        agg.ingest(RealtimeEvent::ViewerJoin {
            viewer_id: "a".to_string(),
            timestamp_ms: 1_000,
        });
        agg.ingest(RealtimeEvent::ViewerJoin {
            viewer_id: "b".to_string(),
            timestamp_ms: 2_000,
        });
        assert_eq!(agg.concurrent_viewers(), 2);
        agg.ingest(RealtimeEvent::ViewerLeave {
            viewer_id: "a".to_string(),
            timestamp_ms: 3_000,
        });
        assert_eq!(agg.concurrent_viewers(), 1);
    }

    #[test]
    fn concurrent_viewers_does_not_go_negative() {
        let mut agg = aggregator();
        agg.ingest(RealtimeEvent::ViewerLeave {
            viewer_id: "ghost".to_string(),
            timestamp_ms: 1_000,
        });
        assert_eq!(agg.concurrent_viewers(), 0);
    }

    // ── bitrate stats ────────────────────────────────────────────────────────

    #[test]
    fn bitrate_stats_basic() {
        let mut agg = aggregator();
        for bps in [1_000_000u64, 2_000_000, 3_000_000] {
            agg.ingest(RealtimeEvent::BitrateReport {
                viewer_id: "v".to_string(),
                timestamp_ms: 5_000,
                bitrate_bps: bps,
            });
        }
        let (avg, min, max) = agg.window_bitrate_stats();
        assert!((avg - 2_000_000.0).abs() < 1.0, "avg={avg}");
        assert_eq!(min, 1_000_000);
        assert_eq!(max, 3_000_000);
    }

    #[test]
    fn bitrate_stats_empty_window() {
        let agg = aggregator();
        assert_eq!(agg.window_bitrate_stats(), (0.0, 0, 0));
    }

    // ── buffer events ────────────────────────────────────────────────────────

    #[test]
    fn buffer_events_counted() {
        let mut agg = aggregator();
        for i in 0..5 {
            agg.ingest(RealtimeEvent::BufferEvent {
                viewer_id: "v".to_string(),
                timestamp_ms: i * 1_000 + 1_000,
                duration_ms: 200,
            });
        }
        assert_eq!(agg.window_buffer_events(), 5);
    }

    // ── window eviction ──────────────────────────────────────────────────────

    #[test]
    fn window_evicts_old_buckets() {
        let mut agg = SlidingWindowAggregator::new(20_000, 10_000).expect("new should succeed");
        // Bucket at t=0–10s.
        agg.ingest(RealtimeEvent::BitrateReport {
            viewer_id: "v".to_string(),
            timestamp_ms: 5_000,
            bitrate_bps: 1_000_000,
        });
        // Advance time beyond window: t=30s → bucket at t=0 should evict.
        agg.ingest(RealtimeEvent::BitrateReport {
            viewer_id: "v".to_string(),
            timestamp_ms: 35_000,
            bitrate_bps: 2_000_000,
        });
        // Only the recent bucket should remain.
        let (avg, _, _) = agg.window_bitrate_stats();
        // avg should be 2_000_000 (old bucket evicted).
        assert!((avg - 2_000_000.0).abs() < 1.0, "avg after eviction={avg}");
    }

    // ── peak concurrent ──────────────────────────────────────────────────────

    #[test]
    fn peak_concurrent_tracked_per_bucket() {
        let mut agg = aggregator();
        for i in 0..5 {
            agg.ingest(RealtimeEvent::ViewerJoin {
                viewer_id: format!("v{i}"),
                timestamp_ms: 5_000,
            });
        }
        assert_eq!(agg.window_peak_concurrent(), 5);
    }
}
