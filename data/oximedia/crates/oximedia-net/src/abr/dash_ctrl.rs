//! DASH (Dynamic Adaptive Streaming over HTTP) ABR controller implementation.

use super::{
    AbrConfig, AbrDecision, AdaptiveBitrateController, BandwidthEstimator, QualityLevel,
    QualitySelector,
};
use crate::abr::history::SegmentDownloadHistory;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// DASH-specific ABR Controller
// ─────────────────────────────────────────────────────────────────────────────

/// DASH MPD segment availability state.
#[derive(Debug, Clone)]
pub struct DashSegmentAvailability {
    /// Number of segments currently available in the manifest.
    pub available_segments: usize,
    /// Segment duration in seconds (typically 2–6 seconds).
    pub segment_duration: Duration,
    /// Manifest update interval (for live/DVR).
    pub update_interval: Option<Duration>,
    /// Whether this is a live stream (vs. on-demand VOD).
    pub is_live: bool,
    /// Time shift buffer depth for DVR (live only).
    pub time_shift_buffer: Option<Duration>,
    /// Minimum update period from MPD.
    pub min_update_period: Option<Duration>,
    /// Availability start time offset (seconds from now for live).
    pub availability_start_offset: f64,
}

impl Default for DashSegmentAvailability {
    fn default() -> Self {
        Self {
            available_segments: 10,
            segment_duration: Duration::from_secs(4),
            update_interval: None,
            is_live: false,
            time_shift_buffer: None,
            min_update_period: None,
            availability_start_offset: 0.0,
        }
    }
}

/// DASH-specific ABR controller that uses MPD segment availability information.
///
/// This controller goes beyond pure throughput estimation by incorporating:
/// - Segment availability windows from the DASH MPD
/// - Segment duration awareness for buffer projection
/// - Live vs. VOD adaptation strategies
/// - Availability start offset for low-latency DASH
#[derive(Debug)]
pub struct DashAbrController {
    /// Base ABR configuration.
    config: AbrConfig,
    /// Bandwidth estimator.
    bandwidth_estimator: BandwidthEstimator,
    /// Current buffer level.
    buffer_level: Duration,
    /// Quality selector for hysteresis.
    quality_selector: QualitySelector,
    /// Segment download history.
    download_history: SegmentDownloadHistory,
    /// Current DASH segment availability state.
    availability: DashSegmentAvailability,
    /// Number of segments downloaded since last quality switch.
    segments_since_switch: usize,
    /// Whether we are in startup phase.
    in_startup: bool,
}

impl DashAbrController {
    /// Creates a new DASH ABR controller.
    #[must_use]
    pub fn new(config: AbrConfig) -> Self {
        let alpha = config.mode.ema_alpha();
        let bandwidth_estimator =
            BandwidthEstimator::new(config.estimation_window, config.sample_ttl, alpha);
        Self {
            config,
            bandwidth_estimator,
            buffer_level: Duration::ZERO,
            quality_selector: QualitySelector::new(),
            download_history: SegmentDownloadHistory::new(50),
            availability: DashSegmentAvailability::default(),
            segments_since_switch: 0,
            in_startup: true,
        }
    }

    /// Updates the DASH segment availability information from the MPD.
    pub fn update_availability(&mut self, availability: DashSegmentAvailability) {
        self.availability = availability;
    }

    /// Reports a segment download with full DASH metadata.
    pub fn report_dash_segment(
        &mut self,
        quality_index: usize,
        bytes: usize,
        download_duration: Duration,
    ) {
        let seg_dur = self.availability.segment_duration;
        self.download_history
            .add(quality_index, bytes, download_duration, seg_dur);
        self.bandwidth_estimator
            .add_sample(bytes, download_duration);
        self.segments_since_switch += 1;

        if self.in_startup && self.segments_since_switch >= 3 {
            self.in_startup = false;
        }
    }

    /// Estimates buffer duration after downloading the next segment at the given quality.
    fn project_buffer_after_download(&self, level: &QualityLevel) -> Duration {
        let seg_dur = self.availability.segment_duration.as_secs_f64();
        let estimated_bps = self.bandwidth_estimator.estimate_ema() * 8.0;
        if estimated_bps <= 0.0 {
            return self.buffer_level;
        }
        let download_time = level.effective_bandwidth() as f64 * seg_dur / estimated_bps;
        let delta = seg_dur - download_time;
        let new_buf = self.buffer_level.as_secs_f64() + delta;
        Duration::from_secs_f64(new_buf.max(0.0))
    }

    /// Returns the minimum safe buffer level based on availability.
    fn min_safe_buffer(&self) -> Duration {
        if self.availability.is_live {
            // For live, keep enough buffer to handle manifest updates
            let update_interval = self
                .availability
                .update_interval
                .unwrap_or(self.availability.segment_duration);
            update_interval * 3
        } else {
            // VOD: static minimum
            Duration::from_secs(5)
        }
    }

    /// Selects quality considering DASH-specific constraints.
    fn dash_select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // Emergency downswitch if buffer is critically low
        let critical = self.config.mode.critical_buffer();
        if self.buffer_level < critical && current_index > 0 {
            return AbrDecision::SwitchTo(self.config.min_quality.unwrap_or(0));
        }

        let estimated_bps = self.bandwidth_estimator.estimate_balanced() * 8.0;
        if estimated_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        let safety = self.config.mode.safety_factor();
        let available_bw = estimated_bps * safety;

        // For live streams, also consider availability window
        let effective_bw = if self.availability.is_live {
            // In live mode, be more conservative based on available segment count
            let avail_factor = if self.availability.available_segments < 3 {
                0.7 // Very short buffer — be extra conservative
            } else if self.availability.available_segments < 6 {
                0.85
            } else {
                1.0
            };
            available_bw * avail_factor
        } else {
            available_bw
        };

        // Find highest quality level whose projected download rate is sustainable
        let mut target = 0usize;
        for (idx, level) in levels.iter().enumerate().rev() {
            let level_bw = level.effective_bandwidth() as f64;
            if level_bw <= effective_bw {
                // Also verify buffer won't drop below safe minimum
                let projected = self.project_buffer_after_download(level);
                if projected >= self.min_safe_buffer() {
                    target = idx;
                    break;
                }
            }
        }

        // Apply config constraints
        let min_q = self.config.min_quality.unwrap_or(0);
        let max_q = self
            .config
            .max_quality
            .unwrap_or(levels.len().saturating_sub(1));
        let target = target.clamp(min_q, max_q.min(levels.len().saturating_sub(1)));

        // Hysteresis: don't switch up too quickly after a downswitch
        if !self
            .quality_selector
            .can_switch(self.config.mode.min_switch_interval())
        {
            return AbrDecision::Maintain;
        }

        // Prevent upswitch if buffer is too low
        if target > current_index && self.buffer_level < self.config.mode.min_buffer_for_upswitch()
        {
            return AbrDecision::Maintain;
        }

        if target != current_index {
            AbrDecision::SwitchTo(target)
        } else {
            AbrDecision::Maintain
        }
    }

    /// Returns a reference to the download history.
    #[must_use]
    pub fn download_history(&self) -> &SegmentDownloadHistory {
        &self.download_history
    }
}

impl AdaptiveBitrateController for DashAbrController {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if self.bandwidth_estimator.sample_count() == 0 {
            let initial = self.config.initial_quality.unwrap_or(0);
            let initial = initial.min(levels.len().saturating_sub(1));
            return AbrDecision::SwitchTo(initial);
        }
        if !self.bandwidth_estimator.is_reliable() {
            return AbrDecision::Maintain;
        }
        self.dash_select_quality(levels, current_index)
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.report_dash_segment(0, bytes, duration);
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.bandwidth_estimator.estimate_balanced() * 8.0
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.bandwidth_estimator.reset();
        self.buffer_level = Duration::ZERO;
        self.quality_selector.reset();
        self.download_history.reset();
        self.segments_since_switch = 0;
        self.in_startup = true;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}
