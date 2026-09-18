//! BOLA (Buffer Occupancy based Lyapunov Algorithm) adaptive bitrate controller.

use super::{
    AbrConfig, AbrDecision, AdaptiveBitrateController, BandwidthEstimator, QualityLevel,
    QualitySelector,
};
use crate::abr::history::SegmentDownloadHistory;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// BOLA – Buffer Occupancy based Lyapunov Algorithm
// ─────────────────────────────────────────────────────────────────────────────
//
// BOLA (Spiteri et al., IEEE INFOCOM 2016) formulates ABR as an online
// optimisation problem solved via Lyapunov optimisation.  The controller
// maximises a utility function that trades off quality against buffer
// occupancy.  For each quality level q_i with bitrate r_i, the utility is:
//
//   v(q_i) = log(r_i / r_0)           (log-scale quality utility)
//
// At each decision epoch the controller chooses the quality that maximises:
//
//   V * v(q_i) + Q(t) / segment_duration - r_i / throughput
//
// where:
//   V              = Lyapunov parameter (trade-off weight, > 0)
//   Q(t)           = current virtual queue (buffer level in seconds)
//   segment_duration = playback duration of one segment (seconds)
//   throughput     = estimated network throughput (bits/sec)

/// BOLA (Buffer Occupancy based Lyapunov Algorithm) ABR controller.
///
/// Maximises log-quality utility while maintaining buffer stability via
/// Lyapunov optimisation.  Pure buffer-based, no throughput model needed
/// for quality selection — throughput is only used to avoid stall.
#[derive(Debug)]
pub struct BolaBbrController {
    /// Base configuration.
    config: AbrConfig,
    /// Bandwidth estimator (used only for stall avoidance).
    bandwidth_estimator: BandwidthEstimator,
    /// Current buffer occupancy.
    buffer_level: Duration,
    /// Lyapunov trade-off parameter V.  Higher V → higher quality preference.
    lyapunov_v: f64,
    /// Playback duration of one segment (seconds).
    segment_duration: f64,
    /// Quality selector for switch hysteresis.
    quality_selector: QualitySelector,
    /// Download history.
    download_history: SegmentDownloadHistory,
    /// Minimum bitrate across all quality levels (set on first `select_quality` call).
    min_bitrate: f64,
    /// Startup phase flag.
    in_startup: bool,
}

impl BolaBbrController {
    /// Creates a new BOLA controller.
    ///
    /// * `lyapunov_v` — trade-off weight (recommended: 5.0 for most streams).
    /// * `segment_duration` — segment playback duration in seconds (e.g. 4.0).
    #[must_use]
    pub fn new(config: AbrConfig, lyapunov_v: f64, segment_duration: f64) -> Self {
        let alpha = config.mode.ema_alpha();
        let bandwidth_estimator =
            BandwidthEstimator::new(config.estimation_window, config.sample_ttl, alpha);
        Self {
            config,
            bandwidth_estimator,
            buffer_level: Duration::ZERO,
            lyapunov_v: lyapunov_v.max(0.1),
            segment_duration: segment_duration.max(1.0),
            quality_selector: QualitySelector::new(),
            download_history: SegmentDownloadHistory::new(50),
            min_bitrate: 0.0,
            in_startup: true,
        }
    }

    /// Creates a BOLA controller with default parameters.
    #[must_use]
    pub fn default_params(config: AbrConfig) -> Self {
        Self::new(config, 5.0, 4.0)
    }

    /// Sets the segment duration.
    pub fn set_segment_duration(&mut self, duration: f64) {
        self.segment_duration = duration.max(1.0);
    }

    /// Computes log utility for a given bitrate relative to `min_bitrate`.
    fn utility(&self, bitrate: f64) -> f64 {
        if self.min_bitrate <= 0.0 || bitrate <= 0.0 {
            return 0.0;
        }
        (bitrate / self.min_bitrate).ln().max(0.0)
    }

    /// Computes the BOLA objective for a quality level.
    ///
    /// Maximise:  V * utility(r_i) + Q(t) / T_s  -  r_i / throughput
    ///
    /// The last term penalises choosing a quality whose download time would
    /// significantly drain the buffer.
    fn bola_objective(&self, bitrate: f64, throughput_bps: f64) -> f64 {
        let q = self.buffer_level.as_secs_f64();
        let utility = self.utility(bitrate);
        let quality_term = self.lyapunov_v * utility;
        let buffer_term = q / self.segment_duration;
        // Download-time penalty: if throughput is unknown, use a generous upper bound
        let penalty = if throughput_bps > 0.0 {
            bitrate / throughput_bps
        } else {
            0.0
        };
        quality_term + buffer_term - penalty
    }

    /// Selects quality using the BOLA objective.
    fn bola_select_quality(
        &mut self,
        levels: &[QualityLevel],
        current_index: usize,
    ) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // Update min_bitrate cache
        if self.min_bitrate <= 0.0 {
            self.min_bitrate = levels
                .iter()
                .map(|l| l.effective_bandwidth() as f64)
                .fold(f64::INFINITY, f64::min);
            if self.min_bitrate <= 0.0 {
                return AbrDecision::Maintain;
            }
        }

        let throughput_bps = self.bandwidth_estimator.estimate_ema() * 8.0;

        // Emergency downswitch on critically low buffer
        if self.buffer_level < self.config.mode.critical_buffer() && current_index > 0 {
            return AbrDecision::SwitchTo(self.config.min_quality.unwrap_or(0));
        }

        // Stall-avoidance: never pick a quality that would drain the buffer to empty.
        // We skip quality levels whose download would take longer than buffer.
        let buffer_secs = self.buffer_level.as_secs_f64();

        let min_q = self.config.min_quality.unwrap_or(0);
        let max_q = self
            .config
            .max_quality
            .unwrap_or(levels.len().saturating_sub(1))
            .min(levels.len().saturating_sub(1));

        let mut best_idx = min_q;
        let mut best_obj = f64::NEG_INFINITY;

        for idx in min_q..=max_q {
            let bitrate = levels[idx].effective_bandwidth() as f64;
            // Check stall risk: download time = bitrate * seg_dur / throughput
            if throughput_bps > 0.0 {
                let download_time = bitrate * self.segment_duration / throughput_bps;
                if download_time > buffer_secs + self.segment_duration {
                    // Would stall — skip unless we're at minimum
                    if idx > min_q {
                        continue;
                    }
                }
            }
            let obj = self.bola_objective(bitrate, throughput_bps);
            if obj > best_obj {
                best_obj = obj;
                best_idx = idx;
            }
        }

        // Apply switch hysteresis
        if !self
            .quality_selector
            .can_switch(self.config.mode.min_switch_interval())
        {
            return AbrDecision::Maintain;
        }

        if best_idx != current_index {
            AbrDecision::SwitchTo(best_idx)
        } else {
            AbrDecision::Maintain
        }
    }

    /// Returns reference to the download history.
    #[must_use]
    pub fn download_history(&self) -> &SegmentDownloadHistory {
        &self.download_history
    }

    /// Returns the current Lyapunov V parameter.
    #[must_use]
    pub fn lyapunov_v(&self) -> f64 {
        self.lyapunov_v
    }
}

impl AdaptiveBitrateController for BolaBbrController {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        // Need mutable self for bola_select_quality due to min_bitrate cache.
        // Work around with a local copy of min_bitrate.
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }
        if self.in_startup && self.bandwidth_estimator.sample_count() == 0 {
            return AbrDecision::SwitchTo(self.config.min_quality.unwrap_or(0));
        }

        let min_bitrate = if self.min_bitrate > 0.0 {
            self.min_bitrate
        } else {
            levels
                .iter()
                .map(|l| l.effective_bandwidth() as f64)
                .fold(f64::INFINITY, f64::min)
                .max(1.0)
        };

        let throughput_bps = self.bandwidth_estimator.estimate_ema() * 8.0;

        if self.buffer_level < self.config.mode.critical_buffer() && current_index > 0 {
            return AbrDecision::SwitchTo(self.config.min_quality.unwrap_or(0));
        }

        let buffer_secs = self.buffer_level.as_secs_f64();
        let min_q = self.config.min_quality.unwrap_or(0);
        let max_q = self
            .config
            .max_quality
            .unwrap_or(levels.len().saturating_sub(1))
            .min(levels.len().saturating_sub(1));

        let utility_fn = |bitrate: f64| -> f64 {
            if min_bitrate <= 0.0 || bitrate <= 0.0 {
                return 0.0;
            }
            (bitrate / min_bitrate).ln().max(0.0)
        };

        let objective = |bitrate: f64| -> f64 {
            let q = buffer_secs;
            let u = utility_fn(bitrate);
            let quality_term = self.lyapunov_v * u;
            let buffer_term = q / self.segment_duration;
            let penalty = if throughput_bps > 0.0 {
                bitrate / throughput_bps
            } else {
                0.0
            };
            quality_term + buffer_term - penalty
        };

        let mut best_idx = min_q;
        let mut best_obj = f64::NEG_INFINITY;

        for idx in min_q..=max_q {
            let bitrate = levels[idx].effective_bandwidth() as f64;
            if throughput_bps > 0.0 {
                let download_time = bitrate * self.segment_duration / throughput_bps;
                if download_time > buffer_secs + self.segment_duration && idx > min_q {
                    continue;
                }
            }
            let obj = objective(bitrate);
            if obj > best_obj {
                best_obj = obj;
                best_idx = idx;
            }
        }

        if !self
            .quality_selector
            .can_switch(self.config.mode.min_switch_interval())
        {
            return AbrDecision::Maintain;
        }

        if best_idx != current_index {
            AbrDecision::SwitchTo(best_idx)
        } else {
            AbrDecision::Maintain
        }
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.bandwidth_estimator.add_sample(bytes, duration);
        let seg_dur = Duration::from_secs_f64(self.segment_duration);
        self.download_history.add(0, bytes, duration, seg_dur);
        if self.in_startup && self.download_history.len() >= 3 {
            self.in_startup = false;
        }
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.bandwidth_estimator.estimate_ema() * 8.0
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.bandwidth_estimator.reset();
        self.buffer_level = Duration::ZERO;
        self.quality_selector.reset();
        self.download_history.reset();
        self.min_bitrate = 0.0;
        self.in_startup = true;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}
