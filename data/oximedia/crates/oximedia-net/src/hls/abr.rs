//! Adaptive bitrate (ABR) control for HLS streaming.
//!
//! This module provides traits and implementations for adaptive bitrate
//! selection in HLS streaming.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Quality level representing a variant stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityLevel {
    /// Unique index for this level.
    pub index: usize,
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Resolution (width, height) if available.
    pub resolution: Option<(u32, u32)>,
    /// Codec string.
    pub codecs: Option<String>,
}

impl QualityLevel {
    /// Creates a new quality level.
    #[must_use]
    pub const fn new(index: usize, bandwidth: u64) -> Self {
        Self {
            index,
            bandwidth,
            resolution: None,
            codecs: None,
        }
    }

    /// Sets the resolution.
    #[must_use]
    pub const fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Sets the codecs string.
    #[must_use]
    pub fn with_codecs(mut self, codecs: impl Into<String>) -> Self {
        self.codecs = Some(codecs.into());
        self
    }

    /// Returns the height if resolution is available.
    #[must_use]
    pub fn height(&self) -> Option<u32> {
        self.resolution.map(|(_, h)| h)
    }

    /// Returns the width if resolution is available.
    #[must_use]
    pub fn width(&self) -> Option<u32> {
        self.resolution.map(|(w, _)| w)
    }
}

/// ABR decision indicating which quality level to switch to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbrDecision {
    /// Keep current quality level.
    Maintain,
    /// Switch to a higher quality level.
    SwitchUp(usize),
    /// Switch to a lower quality level.
    SwitchDown(usize),
}

impl AbrDecision {
    /// Returns true if this is a switch decision.
    #[must_use]
    pub const fn is_switch(&self) -> bool {
        !matches!(self, Self::Maintain)
    }

    /// Returns the target level index if switching.
    #[must_use]
    pub const fn target_level(&self) -> Option<usize> {
        match self {
            Self::Maintain => None,
            Self::SwitchUp(idx) | Self::SwitchDown(idx) => Some(*idx),
        }
    }
}

/// Trait for adaptive bitrate controllers.
pub trait AbrController: Send + Sync {
    /// Selects the best quality level based on current conditions.
    fn select_quality(&self, levels: &[QualityLevel], current_level: usize) -> AbrDecision;

    /// Reports a completed segment download.
    fn report_download(&mut self, bytes: usize, duration: Duration);

    /// Reports current buffer level.
    fn report_buffer(&mut self, buffer_duration: Duration);

    /// Returns estimated throughput in bits per second.
    fn estimated_throughput(&self) -> f64;

    /// Resets the controller state.
    fn reset(&mut self);
}

/// Throughput sample for estimation.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ThroughputSample {
    /// Bytes downloaded.
    bytes: usize,
    /// Time taken.
    duration: Duration,
    /// When the sample was taken.
    timestamp: Instant,
}

/// Throughput estimator using exponential moving average.
#[derive(Debug)]
pub struct ThroughputEstimator {
    /// Recent throughput samples.
    samples: VecDeque<ThroughputSample>,
    /// Maximum number of samples to keep.
    max_samples: usize,
    /// Sample TTL.
    sample_ttl: Duration,
    /// EMA alpha factor.
    alpha: f64,
    /// Current EMA estimate in bytes per second.
    ema_estimate: f64,
    /// Safety factor (0.0 to 1.0).
    safety_factor: f64,
}

impl ThroughputEstimator {
    /// Creates a new throughput estimator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: 20,
            sample_ttl: Duration::from_secs(60),
            alpha: 0.7,
            ema_estimate: 0.0,
            safety_factor: 0.9,
        }
    }

    /// Sets the EMA alpha factor (0.0 to 1.0).
    /// Higher values give more weight to recent samples.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Sets the safety factor (0.0 to 1.0).
    /// The estimate is multiplied by this factor to be conservative.
    #[must_use]
    pub fn with_safety_factor(mut self, factor: f64) -> Self {
        self.safety_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Adds a throughput sample.
    pub fn add_sample(&mut self, bytes: usize, duration: Duration) {
        let now = Instant::now();

        // Remove expired samples
        while let Some(front) = self.samples.front() {
            if now.duration_since(front.timestamp) > self.sample_ttl {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        // Remove oldest if at capacity
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }

        // Calculate throughput for this sample
        let throughput = if duration.as_secs_f64() > 0.0 {
            bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Update EMA
        if self.ema_estimate <= 0.0 {
            self.ema_estimate = throughput;
        } else {
            self.ema_estimate = self.alpha * throughput + (1.0 - self.alpha) * self.ema_estimate;
        }

        self.samples.push_back(ThroughputSample {
            bytes,
            duration,
            timestamp: now,
        });
    }

    /// Returns the estimated throughput in bytes per second.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        self.ema_estimate * self.safety_factor
    }

    /// Returns the estimated throughput in bits per second.
    #[must_use]
    pub fn estimate_bps(&self) -> f64 {
        self.estimate() * 8.0
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Resets the estimator.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.ema_estimate = 0.0;
    }

    /// Returns the raw (unsafe) estimate without safety factor.
    #[must_use]
    pub fn raw_estimate(&self) -> f64 {
        self.ema_estimate
    }
}

impl Default for ThroughputEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Throughput-based ABR controller.
///
/// Selects quality based on measured network throughput.
#[derive(Debug)]
pub struct ThroughputBasedAbr {
    /// Throughput estimator.
    estimator: ThroughputEstimator,
    /// Minimum switch interval.
    min_switch_interval: Duration,
    /// Last switch time.
    last_switch: Option<Instant>,
    /// Bandwidth headroom factor.
    headroom: f64,
}

impl ThroughputBasedAbr {
    /// Creates a new throughput-based ABR controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            estimator: ThroughputEstimator::new(),
            min_switch_interval: Duration::from_secs(10),
            last_switch: None,
            headroom: 0.8, // Only use 80% of measured bandwidth
        }
    }

    /// Sets the minimum interval between quality switches.
    #[must_use]
    pub const fn with_min_switch_interval(mut self, interval: Duration) -> Self {
        self.min_switch_interval = interval;
        self
    }

    /// Sets the bandwidth headroom factor.
    #[must_use]
    pub fn with_headroom(mut self, headroom: f64) -> Self {
        self.headroom = headroom.clamp(0.1, 1.0);
        self
    }

    fn can_switch(&self) -> bool {
        match self.last_switch {
            Some(t) => t.elapsed() >= self.min_switch_interval,
            None => true,
        }
    }

    fn find_best_level(&self, levels: &[QualityLevel], available_bandwidth: f64) -> usize {
        let mut best_idx = 0;
        let mut best_bandwidth = 0u64;

        for (idx, level) in levels.iter().enumerate() {
            if (level.bandwidth as f64) <= available_bandwidth && level.bandwidth > best_bandwidth {
                best_idx = idx;
                best_bandwidth = level.bandwidth;
            }
        }

        best_idx
    }
}

impl Default for ThroughputBasedAbr {
    fn default() -> Self {
        Self::new()
    }
}

impl AbrController for ThroughputBasedAbr {
    fn select_quality(&self, levels: &[QualityLevel], current_level: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        let estimated_bps = self.estimator.estimate_bps() * self.headroom;
        if estimated_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        let target_level = self.find_best_level(levels, estimated_bps);

        if target_level == current_level {
            AbrDecision::Maintain
        } else if !self.can_switch() {
            AbrDecision::Maintain
        } else if target_level > current_level {
            AbrDecision::SwitchUp(target_level)
        } else {
            AbrDecision::SwitchDown(target_level)
        }
    }

    fn report_download(&mut self, bytes: usize, duration: Duration) {
        self.estimator.add_sample(bytes, duration);
    }

    fn report_buffer(&mut self, _buffer_duration: Duration) {
        // Throughput-based ABR doesn't use buffer level
    }

    fn estimated_throughput(&self) -> f64 {
        self.estimator.estimate_bps()
    }

    fn reset(&mut self) {
        self.estimator.reset();
        self.last_switch = None;
    }
}

/// Buffer-based ABR controller.
///
/// Selects quality based on buffer level in addition to throughput.
#[derive(Debug)]
pub struct BufferBasedAbr {
    /// Throughput estimator.
    estimator: ThroughputEstimator,
    /// Current buffer level.
    buffer_level: Duration,
    /// Minimum buffer for switching up.
    buffer_min: Duration,
    /// Maximum buffer target.
    buffer_max: Duration,
    /// Critical buffer level (panic mode).
    buffer_critical: Duration,
    /// Last switch time.
    last_switch: Option<Instant>,
}

impl BufferBasedAbr {
    /// Creates a new buffer-based ABR controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            estimator: ThroughputEstimator::new(),
            buffer_level: Duration::ZERO,
            buffer_min: Duration::from_secs(10),
            buffer_max: Duration::from_secs(30),
            buffer_critical: Duration::from_secs(5),
            last_switch: None,
        }
    }

    /// Sets the minimum buffer threshold for switching up.
    #[must_use]
    pub const fn with_buffer_min(mut self, min: Duration) -> Self {
        self.buffer_min = min;
        self
    }

    /// Sets the maximum buffer target.
    #[must_use]
    pub const fn with_buffer_max(mut self, max: Duration) -> Self {
        self.buffer_max = max;
        self
    }

    /// Sets the critical buffer threshold.
    #[must_use]
    pub const fn with_buffer_critical(mut self, critical: Duration) -> Self {
        self.buffer_critical = critical;
        self
    }

    fn buffer_ratio(&self) -> f64 {
        if self.buffer_max.as_secs_f64() > 0.0 {
            self.buffer_level.as_secs_f64() / self.buffer_max.as_secs_f64()
        } else {
            0.0
        }
    }
}

impl Default for BufferBasedAbr {
    fn default() -> Self {
        Self::new()
    }
}

impl AbrController for BufferBasedAbr {
    fn select_quality(&self, levels: &[QualityLevel], current_level: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        // In critical buffer situation, switch to lowest quality
        if self.buffer_level < self.buffer_critical && current_level > 0 {
            return AbrDecision::SwitchDown(0);
        }

        let buffer_ratio = self.buffer_ratio();
        let estimated_bps = self.estimator.estimate_bps();

        // Calculate effective bandwidth based on buffer level
        // Higher buffer = more aggressive quality selection
        let effective_bandwidth = estimated_bps * (0.5 + 0.5 * buffer_ratio);

        // Find best fitting level
        let mut target_level = 0;
        for (idx, level) in levels.iter().enumerate() {
            if (level.bandwidth as f64) <= effective_bandwidth {
                target_level = idx;
            }
        }

        // Only switch up if buffer is above minimum
        if target_level > current_level && self.buffer_level < self.buffer_min {
            return AbrDecision::Maintain;
        }

        if target_level == current_level {
            AbrDecision::Maintain
        } else if target_level > current_level {
            AbrDecision::SwitchUp(target_level)
        } else {
            AbrDecision::SwitchDown(target_level)
        }
    }

    fn report_download(&mut self, bytes: usize, duration: Duration) {
        self.estimator.add_sample(bytes, duration);
    }

    fn report_buffer(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.estimator.estimate_bps()
    }

    fn reset(&mut self) {
        self.estimator.reset();
        self.buffer_level = Duration::ZERO;
        self.last_switch = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_levels() -> Vec<QualityLevel> {
        vec![
            QualityLevel::new(0, 500_000).with_resolution(640, 360),
            QualityLevel::new(1, 1_500_000).with_resolution(1280, 720),
            QualityLevel::new(2, 3_000_000).with_resolution(1920, 1080),
            QualityLevel::new(3, 6_000_000).with_resolution(2560, 1440),
        ]
    }

    #[test]
    fn test_quality_level() {
        let level = QualityLevel::new(0, 1_000_000)
            .with_resolution(1920, 1080)
            .with_codecs("avc1.4d401f,mp4a.40.2");

        assert_eq!(level.index, 0);
        assert_eq!(level.bandwidth, 1_000_000);
        assert_eq!(level.resolution, Some((1920, 1080)));
        assert_eq!(level.width(), Some(1920));
        assert_eq!(level.height(), Some(1080));
    }

    #[test]
    fn test_abr_decision() {
        assert!(!AbrDecision::Maintain.is_switch());
        assert!(AbrDecision::SwitchUp(1).is_switch());
        assert!(AbrDecision::SwitchDown(0).is_switch());

        assert_eq!(AbrDecision::Maintain.target_level(), None);
        assert_eq!(AbrDecision::SwitchUp(2).target_level(), Some(2));
        assert_eq!(AbrDecision::SwitchDown(0).target_level(), Some(0));
    }

    #[test]
    fn test_throughput_estimator() {
        let mut estimator = ThroughputEstimator::new()
            .with_alpha(0.5)
            .with_safety_factor(1.0);

        // Add sample: 1MB in 1 second = 1MB/s
        estimator.add_sample(1_000_000, Duration::from_secs(1));
        assert!((estimator.estimate() - 1_000_000.0).abs() < 1.0);

        // Add another sample: 2MB in 1 second
        estimator.add_sample(2_000_000, Duration::from_secs(1));
        // EMA: 0.5 * 2M + 0.5 * 1M = 1.5M
        assert!((estimator.estimate() - 1_500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput_estimator_safety_factor() {
        let mut estimator = ThroughputEstimator::new().with_safety_factor(0.8);

        estimator.add_sample(1_000_000, Duration::from_secs(1));
        assert!((estimator.estimate() - 800_000.0).abs() < 1.0);
        assert!((estimator.raw_estimate() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput_based_abr() {
        let levels = make_levels();
        let mut abr = ThroughputBasedAbr::new().with_headroom(1.0);

        // Report high bandwidth
        abr.report_download(5_000_000, Duration::from_secs(1)); // 5MB/s = 40Mbps

        let decision = abr.select_quality(&levels, 0);
        assert!(matches!(decision, AbrDecision::SwitchUp(_)));

        // Report low bandwidth
        abr.reset();
        abr.report_download(50_000, Duration::from_secs(1)); // 50KB/s = 400kbps

        let decision = abr.select_quality(&levels, 2);
        assert!(matches!(decision, AbrDecision::SwitchDown(_)));
    }

    #[test]
    fn test_buffer_based_abr() {
        let levels = make_levels();
        let mut abr = BufferBasedAbr::new()
            .with_buffer_critical(Duration::from_secs(5))
            .with_buffer_min(Duration::from_secs(10));

        // Good throughput but critical buffer - should drop to lowest
        abr.report_download(5_000_000, Duration::from_secs(1));
        abr.report_buffer(Duration::from_secs(3));

        let decision = abr.select_quality(&levels, 2);
        assert_eq!(decision, AbrDecision::SwitchDown(0));
    }

    #[test]
    fn test_buffer_based_abr_maintain() {
        let levels = make_levels();
        let mut abr = BufferBasedAbr::new();

        // Low buffer - should not switch up even with good throughput
        abr.report_download(5_000_000, Duration::from_secs(1));
        abr.report_buffer(Duration::from_secs(8)); // Below buffer_min

        let decision = abr.select_quality(&levels, 0);
        // Should maintain since buffer is too low to switch up
        assert_eq!(decision, AbrDecision::Maintain);
    }

    #[test]
    fn test_abr_empty_levels() {
        let abr = ThroughputBasedAbr::new();
        let decision = abr.select_quality(&[], 0);
        assert_eq!(decision, AbrDecision::Maintain);
    }

    #[test]
    fn test_throughput_estimator_reset() {
        let mut estimator = ThroughputEstimator::new();

        estimator.add_sample(1_000_000, Duration::from_secs(1));
        assert!(estimator.sample_count() > 0);

        estimator.reset();
        assert_eq!(estimator.sample_count(), 0);
        assert!((estimator.estimate()).abs() < 0.001);
    }
}
