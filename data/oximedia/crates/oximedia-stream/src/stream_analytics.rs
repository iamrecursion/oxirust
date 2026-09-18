//! Stream analytics — collects and aggregates viewer-side playback metrics.
//!
//! Tracks buffer health, quality switches, segment load times, and errors to
//! produce a [`PlaybackStats`] summary suitable for QoE dashboards.

use std::time::Instant;

// ─── Playback Events ──────────────────────────────────────────────────────────

/// A discrete playback event emitted by the player.
#[derive(Debug, Clone)]
pub enum PlaybackEvent {
    /// Playback stalled because the buffer ran empty.
    BufferStart,
    /// Playback resumed after a buffer stall (paired with `BufferStart`).
    BufferEnd,
    /// Player switched from one quality level to another.
    ///
    /// Tuple: `(from_bitrate_kbps, to_bitrate_kbps)`.
    QualitySwitch(u32, u32),
    /// A recoverable or fatal playback error occurred.
    Error(String),
    /// A media segment was successfully loaded.
    ///
    /// `(segment_index, load_time_ms)`.
    SegmentLoad(u32, u64),
    /// Playback started (first frame rendered).
    PlaybackStart,
    /// Session ended (user navigated away or stopped playback).
    PlaybackEnd,
}

// ─── Timed Event ─────────────────────────────────────────────────────────────

/// An event paired with a wall-clock timestamp.
#[derive(Debug)]
pub struct TimedEvent {
    event: PlaybackEvent,
    /// Milliseconds since the analytics session was created.
    session_offset_ms: u64,
}

// ─── Playback Stats ───────────────────────────────────────────────────────────

/// Aggregated playback quality-of-experience statistics.
#[derive(Debug, Clone)]
pub struct PlaybackStats {
    /// Weighted average bitrate observed across all quality switches (kbps).
    pub avg_bitrate_kbps: f64,
    /// Ratio of time spent buffering to total session time (0.0–1.0).
    pub buffer_ratio: f64,
    /// Number of quality-level switches that occurred.
    pub quality_switch_count: u32,
    /// Number of playback errors recorded.
    pub error_count: u32,
    /// Total number of segments loaded.
    pub segment_count: u32,
    /// Average segment load time (ms), or 0.0 if no segments loaded.
    pub avg_segment_load_ms: f64,
    /// Total session duration in milliseconds.
    pub session_duration_ms: u64,
    /// Total buffering duration in milliseconds.
    pub total_buffer_ms: u64,
}

impl Default for PlaybackStats {
    fn default() -> Self {
        Self {
            avg_bitrate_kbps: 0.0,
            buffer_ratio: 0.0,
            quality_switch_count: 0,
            error_count: 0,
            segment_count: 0,
            avg_segment_load_ms: 0.0,
            session_duration_ms: 0,
            total_buffer_ms: 0,
        }
    }
}

// ─── Stream Analytics ─────────────────────────────────────────────────────────

/// Collects playback events and computes QoE statistics.
#[derive(Debug)]
pub struct StreamAnalytics {
    /// All recorded events with session-relative timestamps.
    events: Vec<TimedEvent>,
    /// Wall-clock time when this analytics session was created.
    session_start: Instant,
    /// Optional override for session start time (milliseconds since epoch),
    /// used in tests where `Instant` is not controllable.
    session_start_override_ms: Option<u64>,
}

impl StreamAnalytics {
    /// Create a new analytics session starting now.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            session_start: Instant::now(),
            session_start_override_ms: None,
        }
    }

    /// Create a new analytics session with an explicit start time (ms).
    ///
    /// Useful for tests and replay scenarios.
    pub fn with_start_ms(start_ms: u64) -> Self {
        let mut s = Self::new();
        s.session_start_override_ms = Some(start_ms);
        s
    }

    /// Record a playback event at the current wall-clock time.
    pub fn record_event(&mut self, event: PlaybackEvent) {
        let offset = self.session_start.elapsed().as_millis() as u64;
        self.events.push(TimedEvent {
            event,
            session_offset_ms: offset,
        });
    }

    /// Record a playback event at a specific session offset in milliseconds.
    ///
    /// Useful for injecting events at known times in tests or replay.
    pub fn record_event_at(&mut self, event: PlaybackEvent, offset_ms: u64) {
        self.events.push(TimedEvent {
            event,
            session_offset_ms: offset_ms,
        });
    }

    /// Return a reference to all recorded events.
    pub fn events(&self) -> &[TimedEvent] {
        &self.events
    }

    /// Compute and return aggregated [`PlaybackStats`].
    pub fn compute_stats(&self) -> PlaybackStats {
        let mut stats = PlaybackStats::default();

        // Determine session duration
        let last_offset = self
            .events
            .iter()
            .map(|e| e.session_offset_ms)
            .max()
            .unwrap_or(0);
        stats.session_duration_ms = last_offset;

        // Accumulate bitrates for weighted average (weighted by segment count)
        let mut bitrate_sum: f64 = 0.0;
        let mut bitrate_samples: u32 = 0;

        // Buffer tracking
        let mut buffer_start_ms: Option<u64> = None;
        let mut total_buffer_ms: u64 = 0;

        // Segment load times
        let mut load_time_sum: u64 = 0;

        for te in &self.events {
            match &te.event {
                PlaybackEvent::BufferStart => {
                    buffer_start_ms = Some(te.session_offset_ms);
                }
                PlaybackEvent::BufferEnd => {
                    if let Some(start) = buffer_start_ms.take() {
                        total_buffer_ms += te.session_offset_ms.saturating_sub(start);
                    }
                }
                PlaybackEvent::QualitySwitch(from, to) => {
                    stats.quality_switch_count += 1;
                    bitrate_sum += *from as f64 + *to as f64;
                    bitrate_samples += 2;
                }
                PlaybackEvent::Error(_) => {
                    stats.error_count += 1;
                }
                PlaybackEvent::SegmentLoad(_, load_ms) => {
                    stats.segment_count += 1;
                    load_time_sum += load_ms;
                }
                PlaybackEvent::PlaybackStart | PlaybackEvent::PlaybackEnd => {}
            }
        }

        // Close any still-open buffer stall at the end of the session.
        if let Some(start) = buffer_start_ms {
            total_buffer_ms += last_offset.saturating_sub(start);
        }

        stats.total_buffer_ms = total_buffer_ms;

        // Buffer ratio
        if stats.session_duration_ms > 0 {
            stats.buffer_ratio = total_buffer_ms as f64 / stats.session_duration_ms as f64;
        }

        // Average bitrate
        if bitrate_samples > 0 {
            stats.avg_bitrate_kbps = bitrate_sum / bitrate_samples as f64;
        }

        // Average segment load time
        if stats.segment_count > 0 {
            stats.avg_segment_load_ms = load_time_sum as f64 / stats.segment_count as f64;
        }

        stats
    }

    /// Return the total number of events recorded.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Clear all recorded events (resets the session).
    pub fn reset(&mut self) {
        self.events.clear();
        self.session_start = Instant::now();
        self.session_start_override_ms = None;
    }

    /// Return the last recorded error message, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.events.iter().rev().find_map(|te| {
            if let PlaybackEvent::Error(msg) = &te.event {
                Some(msg.as_str())
            } else {
                None
            }
        })
    }

    /// Return the number of buffer stall events.
    pub fn buffer_stall_count(&self) -> u32 {
        self.events
            .iter()
            .filter(|te| matches!(te.event, PlaybackEvent::BufferStart))
            .count() as u32
    }

    /// Compute the mean opinion score (MOS) proxy on a 1–5 scale.
    ///
    /// Uses a heuristic:
    /// - Start at 5.0
    /// - Subtract 0.5 per buffering stall
    /// - Subtract 0.1 per quality switch
    /// - Subtract 0.2 per error
    /// - Clamp to [1.0, 5.0]
    pub fn mean_opinion_score(&self) -> f64 {
        let stats = self.compute_stats();
        let mos = 5.0_f64
            - self.buffer_stall_count() as f64 * 0.5
            - stats.quality_switch_count as f64 * 0.1
            - stats.error_count as f64 * 0.2;
        mos.max(1.0).min(5.0)
    }
}

impl Default for StreamAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_analytics_is_empty() {
        let a = StreamAnalytics::new();
        assert_eq!(a.event_count(), 0);
    }

    #[test]
    fn test_record_event_increments_count() {
        let mut a = StreamAnalytics::new();
        a.record_event(PlaybackEvent::PlaybackStart);
        a.record_event(PlaybackEvent::BufferStart);
        assert_eq!(a.event_count(), 2);
    }

    #[test]
    fn test_error_count() {
        let mut a = StreamAnalytics::new();
        a.record_event_at(PlaybackEvent::Error("timeout".into()), 100);
        a.record_event_at(PlaybackEvent::Error("decode".into()), 200);
        let stats = a.compute_stats();
        assert_eq!(stats.error_count, 2);
    }

    #[test]
    fn test_quality_switch_count() {
        let mut a = StreamAnalytics::new();
        a.record_event_at(PlaybackEvent::QualitySwitch(2000, 4000), 500);
        a.record_event_at(PlaybackEvent::QualitySwitch(4000, 1000), 1000);
        let stats = a.compute_stats();
        assert_eq!(stats.quality_switch_count, 2);
    }

    #[test]
    fn test_segment_count_and_avg_load() {
        let mut a = StreamAnalytics::new();
        a.record_event_at(PlaybackEvent::SegmentLoad(0, 100), 0);
        a.record_event_at(PlaybackEvent::SegmentLoad(1, 200), 500);
        let stats = a.compute_stats();
        assert_eq!(stats.segment_count, 2);
        assert!((stats.avg_segment_load_ms - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_buffer_ratio_calculation() {
        let mut a = StreamAnalytics::new();
        // Session: 0–1000 ms; buffer stall: 0–200 ms
        a.record_event_at(PlaybackEvent::BufferStart, 0);
        a.record_event_at(PlaybackEvent::BufferEnd, 200);
        a.record_event_at(PlaybackEvent::PlaybackEnd, 1000);
        let stats = a.compute_stats();
        assert!((stats.buffer_ratio - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_avg_bitrate_from_quality_switches() {
        let mut a = StreamAnalytics::new();
        // Switch from 2000 kbps → 4000 kbps: sum=6000, samples=2 → avg=3000
        a.record_event_at(PlaybackEvent::QualitySwitch(2000, 4000), 0);
        let stats = a.compute_stats();
        assert!((stats.avg_bitrate_kbps - 3000.0).abs() < 1e-9);
    }

    #[test]
    fn test_buffer_stall_count() {
        let mut a = StreamAnalytics::new();
        a.record_event_at(PlaybackEvent::BufferStart, 0);
        a.record_event_at(PlaybackEvent::BufferEnd, 500);
        a.record_event_at(PlaybackEvent::BufferStart, 800);
        a.record_event_at(PlaybackEvent::BufferEnd, 900);
        assert_eq!(a.buffer_stall_count(), 2);
    }

    #[test]
    fn test_last_error() {
        let mut a = StreamAnalytics::new();
        a.record_event(PlaybackEvent::Error("first".into()));
        a.record_event(PlaybackEvent::Error("second".into()));
        assert_eq!(a.last_error(), Some("second"));
    }

    #[test]
    fn test_mos_perfect_session() {
        let a = StreamAnalytics::new();
        let mos = a.mean_opinion_score();
        assert!((mos - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_mos_degraded_session() {
        let mut a = StreamAnalytics::new();
        a.record_event(PlaybackEvent::BufferStart);
        a.record_event(PlaybackEvent::BufferEnd);
        a.record_event(PlaybackEvent::Error("oops".into()));
        let mos = a.mean_opinion_score();
        // 5.0 - 0.5 (stall) - 0.2 (error) = 4.3
        assert!((mos - 4.3).abs() < 1e-9);
    }

    #[test]
    fn test_reset_clears_events() {
        let mut a = StreamAnalytics::new();
        a.record_event(PlaybackEvent::PlaybackStart);
        a.reset();
        assert_eq!(a.event_count(), 0);
    }
}
