//! Scope recording — capturing scope data over time as a time-series.
//!
//! `ScopeRecorder` accumulates statistical snapshots from a scope analysis
//! pipeline into a time-ordered ring buffer.  Each snapshot stores the key
//! scalars (min, max, mean, std-dev) extracted from a video frame's scope
//! output.  The recorder can then be queried for trends, exported as CSV, or
//! used to drive a graph overlay.
//!
//! # Features
//!
//! - Configurable ring-buffer capacity (number of frames retained)
//! - Per-channel or per-component snapshot storage
//! - Running statistics (min/max/mean across all retained frames)
//! - Broadcast-legal level alerting
//! - CSV export for offline analysis

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// The scope component a snapshot tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeChannel {
    /// Luminance (Y) channel.
    Luma,
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Cb chroma component.
    Cb,
    /// Cr chroma component.
    Cr,
    /// User-defined channel index.
    Custom(u8),
}

impl ScopeChannel {
    /// Short ASCII label for CSV headers.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Luma => "Y",
            Self::Red => "R",
            Self::Green => "G",
            Self::Blue => "B",
            Self::Cb => "Cb",
            Self::Cr => "Cr",
            Self::Custom(_) => "X",
        }
    }
}

/// A single per-frame statistical snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeSnapshot {
    /// Monotonic frame index (counter).
    pub frame_index: u64,
    /// Timestamp in milliseconds (wall-clock or stream time).
    pub timestamp_ms: f64,
    /// Minimum signal value in the frame (0.0–1.0 normalised).
    pub min: f32,
    /// Maximum signal value in the frame (0.0–1.0 normalised).
    pub max: f32,
    /// Mean signal value (0.0–1.0 normalised).
    pub mean: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Percentage of pixels below the legal lower limit (e.g. < 16/255).
    pub black_clip_pct: f32,
    /// Percentage of pixels above the legal upper limit (e.g. > 235/255).
    pub white_clip_pct: f32,
    /// Which scope channel this snapshot represents.
    pub channel: ScopeChannel,
}

impl ScopeSnapshot {
    /// Creates a new snapshot with all-zero values for the given channel.
    #[must_use]
    pub fn zero(channel: ScopeChannel) -> Self {
        Self {
            frame_index: 0,
            timestamp_ms: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            black_clip_pct: 0.0,
            white_clip_pct: 0.0,
            channel,
        }
    }

    /// Returns `true` if the signal is entirely within legal broadcast range.
    #[must_use]
    pub fn is_legal(&self) -> bool {
        self.black_clip_pct == 0.0 && self.white_clip_pct == 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recorder configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for `ScopeRecorder`.
#[derive(Debug, Clone)]
pub struct ScopeRecorderConfig {
    /// Maximum number of snapshots retained (ring buffer capacity).
    pub capacity: usize,
    /// Which channels to record.
    pub channels: Vec<ScopeChannel>,
    /// Lower legal limit for alerting (normalised 0–1, default 16/255 ≈ 0.0627).
    pub legal_min: f32,
    /// Upper legal limit for alerting (normalised 0–1, default 235/255 ≈ 0.9216).
    pub legal_max: f32,
}

impl Default for ScopeRecorderConfig {
    fn default() -> Self {
        Self {
            capacity: 1800, // ~1 minute at 30 fps
            channels: vec![ScopeChannel::Luma],
            legal_min: 16.0 / 255.0,
            legal_max: 235.0 / 255.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recorder
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates scope snapshots over time.
///
/// Internal storage is a per-channel ring buffer.  When `capacity` is reached
/// the oldest frame is overwritten.
#[derive(Debug)]
pub struct ScopeRecorder {
    config: ScopeRecorderConfig,
    /// One ring buffer per channel: `ring_buffers[channel_idx][slot]`.
    ring_buffers: Vec<Vec<ScopeSnapshot>>,
    /// Write head for each channel.
    write_heads: Vec<usize>,
    /// Total frames pushed to each channel.
    frame_counts: Vec<u64>,
}

impl ScopeRecorder {
    /// Creates a new recorder with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `capacity` is zero or `channels` is empty.
    pub fn new(config: ScopeRecorderConfig) -> OxiResult<Self> {
        if config.capacity == 0 {
            return Err(OxiError::InvalidData("capacity must be > 0".into()));
        }
        if config.channels.is_empty() {
            return Err(OxiError::InvalidData(
                "at least one channel required".into(),
            ));
        }
        let n = config.channels.len();
        let cap = config.capacity;
        let ring_buffers = (0..n)
            .map(|i| vec![ScopeSnapshot::zero(config.channels[i]); cap])
            .collect();
        Ok(Self {
            config,
            ring_buffers,
            write_heads: vec![0; n],
            frame_counts: vec![0u64; n],
        })
    }

    /// Push a snapshot for the channel at `channel_idx`.
    ///
    /// # Errors
    ///
    /// Returns an error if `channel_idx` is out of range.
    pub fn push(&mut self, channel_idx: usize, mut snapshot: ScopeSnapshot) -> OxiResult<()> {
        if channel_idx >= self.config.channels.len() {
            return Err(OxiError::InvalidData(format!(
                "channel_idx {channel_idx} >= number of channels {}",
                self.config.channels.len()
            )));
        }
        snapshot.channel = self.config.channels[channel_idx];
        snapshot.frame_index = self.frame_counts[channel_idx];
        let head = self.write_heads[channel_idx];
        self.ring_buffers[channel_idx][head % self.config.capacity] = snapshot;
        self.write_heads[channel_idx] = head.wrapping_add(1);
        self.frame_counts[channel_idx] += 1;
        Ok(())
    }

    /// Returns the number of snapshots currently stored for a channel.
    #[must_use]
    pub fn len(&self, channel_idx: usize) -> usize {
        if channel_idx >= self.frame_counts.len() {
            return 0;
        }
        (self.frame_counts[channel_idx] as usize).min(self.config.capacity)
    }

    /// Returns `true` if no snapshots have been recorded for a channel.
    #[must_use]
    pub fn is_empty(&self, channel_idx: usize) -> bool {
        self.len(channel_idx) == 0
    }

    /// Returns an iterator over the most-recently stored snapshots for a channel,
    /// in chronological order (oldest first).
    ///
    /// Returns an empty slice if the channel index is out of range.
    #[must_use]
    pub fn snapshots(&self, channel_idx: usize) -> Vec<ScopeSnapshot> {
        if channel_idx >= self.ring_buffers.len() {
            return Vec::new();
        }
        let stored = self.len(channel_idx);
        let cap = self.config.capacity;
        let head = self.write_heads[channel_idx];
        let buf = &self.ring_buffers[channel_idx];

        // Determine the read start (oldest entry in the ring)
        let start = if stored < cap { 0 } else { head % cap };

        (0..stored).map(|i| buf[(start + i) % cap]).collect()
    }

    /// Compute running statistics (min/max/mean) over all retained snapshots
    /// for a channel.
    ///
    /// Returns `None` if no snapshots have been recorded.
    #[must_use]
    pub fn running_stats(&self, channel_idx: usize) -> Option<RunningStats> {
        let snaps = self.snapshots(channel_idx);
        if snaps.is_empty() {
            return None;
        }
        let n = snaps.len() as f32;
        let mut min_mean = f32::MAX;
        let mut max_mean = f32::MIN;
        let mut sum_mean = 0.0f32;
        let mut clip_count = 0u32;

        for s in &snaps {
            min_mean = min_mean.min(s.mean);
            max_mean = max_mean.max(s.mean);
            sum_mean += s.mean;
            if !s.is_legal() {
                clip_count += 1;
            }
        }

        Some(RunningStats {
            frame_count: snaps.len(),
            min_mean,
            max_mean,
            avg_mean: sum_mean / n,
            clip_frame_count: clip_count,
        })
    }

    /// Export all snapshots for a channel as a CSV string.
    ///
    /// Format: `frame,timestamp_ms,min,max,mean,std_dev,black_clip_pct,white_clip_pct`
    ///
    /// Returns an empty string if the channel is out of range.
    #[must_use]
    pub fn export_csv(&self, channel_idx: usize) -> String {
        let snaps = self.snapshots(channel_idx);
        let ch_label = if channel_idx < self.config.channels.len() {
            self.config.channels[channel_idx].label()
        } else {
            "?"
        };
        let mut out = format!(
            "channel,frame,timestamp_ms,min,max,mean,std_dev,black_clip_pct,white_clip_pct\n"
        );
        for s in &snaps {
            out.push_str(&format!(
                "{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}\n",
                ch_label,
                s.frame_index,
                s.timestamp_ms,
                s.min,
                s.max,
                s.mean,
                s.std_dev,
                s.black_clip_pct,
                s.white_clip_pct
            ));
        }
        out
    }

    /// Clears all stored snapshots for all channels.
    pub fn clear(&mut self) {
        for ch_idx in 0..self.config.channels.len() {
            let chan = self.config.channels[ch_idx];
            for snap in &mut self.ring_buffers[ch_idx] {
                *snap = ScopeSnapshot::zero(chan);
            }
            self.write_heads[ch_idx] = 0;
            self.frame_counts[ch_idx] = 0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Running statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics across all retained frames.
#[derive(Debug, Clone, Copy)]
pub struct RunningStats {
    /// Number of frames included.
    pub frame_count: usize,
    /// Minimum mean level observed.
    pub min_mean: f32,
    /// Maximum mean level observed.
    pub max_mean: f32,
    /// Average of per-frame means.
    pub avg_mean: f32,
    /// Number of frames with out-of-legal-range pixels.
    pub clip_frame_count: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(mean: f32, frame: u64) -> ScopeSnapshot {
        ScopeSnapshot {
            frame_index: frame,
            timestamp_ms: frame as f64 * (1000.0 / 30.0),
            min: mean - 0.05,
            max: mean + 0.05,
            mean,
            std_dev: 0.02,
            black_clip_pct: 0.0,
            white_clip_pct: 0.0,
            channel: ScopeChannel::Luma,
        }
    }

    fn make_recorder(cap: usize) -> ScopeRecorder {
        let cfg = ScopeRecorderConfig {
            capacity: cap,
            channels: vec![ScopeChannel::Luma],
            ..Default::default()
        };
        ScopeRecorder::new(cfg).expect("valid config")
    }

    #[test]
    fn test_recorder_new_valid() {
        let r = make_recorder(100);
        assert!(r.is_empty(0));
        assert_eq!(r.len(0), 0);
    }

    #[test]
    fn test_recorder_zero_capacity_error() {
        let cfg = ScopeRecorderConfig {
            capacity: 0,
            channels: vec![ScopeChannel::Luma],
            ..Default::default()
        };
        assert!(ScopeRecorder::new(cfg).is_err());
    }

    #[test]
    fn test_recorder_empty_channels_error() {
        let cfg = ScopeRecorderConfig {
            capacity: 10,
            channels: vec![],
            ..Default::default()
        };
        assert!(ScopeRecorder::new(cfg).is_err());
    }

    #[test]
    fn test_push_and_len() {
        let mut r = make_recorder(10);
        for i in 0..5u64 {
            r.push(0, make_snap(0.5, i)).expect("valid push");
        }
        assert_eq!(r.len(0), 5);
        assert!(!r.is_empty(0));
    }

    #[test]
    fn test_push_out_of_range_channel() {
        let mut r = make_recorder(10);
        let result = r.push(1, make_snap(0.5, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_ring_buffer_wraps() {
        let mut r = make_recorder(5);
        for i in 0..10u64 {
            r.push(0, make_snap(i as f32 / 10.0, i)).expect("push");
        }
        // After wrapping, len should be capped at capacity
        assert_eq!(r.len(0), 5);
        // Snapshots should be the last 5 (frames 5–9)
        let snaps = r.snapshots(0);
        assert_eq!(snaps.len(), 5);
        // The mean of the newest snapshot should be > 0.5
        assert!(snaps.last().expect("non-empty").mean > 0.5);
    }

    #[test]
    fn test_running_stats_empty() {
        let r = make_recorder(10);
        assert!(r.running_stats(0).is_none());
    }

    #[test]
    fn test_running_stats() {
        let mut r = make_recorder(100);
        for i in 0..10u64 {
            r.push(0, make_snap(0.1 * (i + 1) as f32, i)).expect("push");
        }
        let stats = r.running_stats(0).expect("should have stats");
        assert_eq!(stats.frame_count, 10);
        assert!(stats.min_mean < stats.max_mean);
        assert!(stats.avg_mean > 0.0);
        assert_eq!(stats.clip_frame_count, 0);
    }

    #[test]
    fn test_running_stats_clip_detection() {
        let mut r = make_recorder(10);
        let mut snap = make_snap(0.5, 0);
        snap.white_clip_pct = 5.0; // illegal
        r.push(0, snap).expect("push");
        let stats = r.running_stats(0).expect("stats");
        assert_eq!(stats.clip_frame_count, 1);
    }

    #[test]
    fn test_export_csv_headers() {
        let r = make_recorder(5);
        let csv = r.export_csv(0);
        assert!(csv.starts_with("channel,frame,"));
    }

    #[test]
    fn test_export_csv_row_count() {
        let mut r = make_recorder(10);
        for i in 0..3u64 {
            r.push(0, make_snap(0.5, i)).expect("push");
        }
        let csv = r.export_csv(0);
        // 1 header + 3 data rows
        assert_eq!(csv.lines().count(), 4);
    }

    #[test]
    fn test_clear_resets_all() {
        let mut r = make_recorder(10);
        for i in 0..5u64 {
            r.push(0, make_snap(0.5, i)).expect("push");
        }
        r.clear();
        assert_eq!(r.len(0), 0);
        assert!(r.is_empty(0));
        assert!(r.running_stats(0).is_none());
    }

    #[test]
    fn test_snapshot_zero() {
        let s = ScopeSnapshot::zero(ScopeChannel::Red);
        assert_eq!(s.mean, 0.0);
        assert!(s.is_legal());
    }

    #[test]
    fn test_scope_channel_labels() {
        assert_eq!(ScopeChannel::Luma.label(), "Y");
        assert_eq!(ScopeChannel::Red.label(), "R");
        assert_eq!(ScopeChannel::Cb.label(), "Cb");
    }

    #[test]
    fn test_recorder_multiple_channels() {
        let cfg = ScopeRecorderConfig {
            capacity: 10,
            channels: vec![ScopeChannel::Red, ScopeChannel::Green, ScopeChannel::Blue],
            ..Default::default()
        };
        let mut r = ScopeRecorder::new(cfg).expect("valid");
        r.push(0, make_snap(0.8, 0)).expect("R push");
        r.push(1, make_snap(0.5, 0)).expect("G push");
        r.push(2, make_snap(0.2, 0)).expect("B push");
        assert_eq!(r.len(0), 1);
        assert_eq!(r.len(1), 1);
        assert_eq!(r.len(2), 1);
    }

    #[test]
    fn test_snapshots_out_of_range_channel() {
        let r = make_recorder(5);
        let s = r.snapshots(99);
        assert!(s.is_empty());
    }
}
