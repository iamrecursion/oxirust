//! Filtering of clock offset samples to reduce noise and reject outliers.
#![allow(dead_code)]

use std::collections::VecDeque;

/// The type of smoothing algorithm applied to offset samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterType {
    /// No filtering — raw samples are returned as-is.
    None,
    /// First-order low-pass (exponential moving average).
    LoPass,
    /// Sliding-window median filter.
    Median,
    /// Simple Kalman filter (scalar, constant-velocity model).
    Kalman,
}

impl FilterType {
    /// Returns the nominal smoothing factor in `[0, 1]` for display purposes.
    ///
    /// A value of `1.0` means maximum smoothing (slow response).
    #[allow(clippy::cast_precision_loss)]
    pub fn smoothing_factor(self) -> f64 {
        match self {
            FilterType::None => 0.0,
            FilterType::LoPass => 0.125,
            FilterType::Median => 0.5,
            FilterType::Kalman => 0.75,
        }
    }

    /// Short name suitable for logging.
    pub fn name(self) -> &'static str {
        match self {
            FilterType::None => "none",
            FilterType::LoPass => "lopass",
            FilterType::Median => "median",
            FilterType::Kalman => "kalman",
        }
    }
}

/// A single measured clock offset sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetSample {
    /// Offset from the reference clock in microseconds (positive = local is ahead).
    pub offset_us: i64,
    /// Round-trip delay at sample time in microseconds.
    pub delay_us: u64,
    /// Monotonic timestamp of the sample in milliseconds since an arbitrary epoch.
    pub timestamp_ms: u64,
}

impl OffsetSample {
    /// Create a new sample.
    pub fn new(offset_us: i64, delay_us: u64, timestamp_ms: u64) -> Self {
        Self {
            offset_us,
            delay_us,
            timestamp_ms,
        }
    }

    /// Returns `true` if the sample is likely an outlier.
    ///
    /// A sample is considered an outlier when its round-trip delay exceeds
    /// `max_delay_us`.
    pub fn is_outlier(&self, max_delay_us: u64) -> bool {
        self.delay_us > max_delay_us
    }
}

/// Kalman filter state (scalar).
#[derive(Debug, Clone, Copy)]
struct KalmanState {
    /// Estimated offset in µs.
    x: f64,
    /// Estimate error covariance.
    p: f64,
    /// Process noise covariance.
    q: f64,
    /// Measurement noise covariance.
    r: f64,
}

impl KalmanState {
    fn new(initial: f64) -> Self {
        Self {
            x: initial,
            p: 1.0,
            q: 1e-4,
            r: 1.0,
        }
    }

    fn update(&mut self, measurement: f64) -> f64 {
        // Predict
        self.p += self.q;
        // Update
        let k = self.p / (self.p + self.r);
        self.x += k * (measurement - self.x);
        self.p *= 1.0 - k;
        self.x
    }
}

/// A filter that accumulates `OffsetSample` values and produces a smoothed offset.
#[derive(Debug)]
pub struct OffsetSampleFilter {
    filter_type: FilterType,
    window: VecDeque<OffsetSample>,
    max_window: usize,
    /// Maximum acceptable delay before a sample is treated as an outlier.
    max_delay_us: u64,
    /// Current smoothed output in µs (for `LoPass` and `Kalman`).
    smoothed: f64,
    /// Whether the filter has been seeded with at least one sample.
    initialized: bool,
    /// Kalman state (only used when `filter_type == Kalman`).
    kalman: Option<KalmanState>,
    /// EMA alpha (1 – `smoothing_factor`).
    ema_alpha: f64,
}

impl OffsetSampleFilter {
    /// Create a new filter.
    ///
    /// `window_size` is the maximum number of samples retained in the sliding window.
    /// `max_delay_us` is the delay threshold above which samples are treated as outliers.
    pub fn new(filter_type: FilterType, window_size: usize, max_delay_us: u64) -> Self {
        let kalman = if filter_type == FilterType::Kalman {
            Some(KalmanState::new(0.0))
        } else {
            None
        };
        Self {
            filter_type,
            window: VecDeque::with_capacity(window_size),
            max_window: window_size,
            max_delay_us,
            smoothed: 0.0,
            initialized: false,
            kalman,
            ema_alpha: 1.0 - filter_type.smoothing_factor(),
        }
    }

    /// Add a new offset sample. Outliers (high delay) are silently dropped.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_sample(&mut self, sample: OffsetSample) {
        if sample.is_outlier(self.max_delay_us) {
            return;
        }
        if self.window.len() >= self.max_window {
            self.window.pop_front();
        }
        self.window.push_back(sample);

        let raw = sample.offset_us as f64;

        match self.filter_type {
            FilterType::None => {
                self.smoothed = raw;
            }
            FilterType::LoPass => {
                if self.initialized {
                    self.smoothed = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self.smoothed;
                } else {
                    self.smoothed = raw;
                    self.initialized = true;
                }
            }
            FilterType::Median => {
                // Recomputed in `filtered_offset_us`, nothing to update here.
                self.smoothed = raw;
            }
            FilterType::Kalman => {
                if let Some(ref mut ks) = self.kalman {
                    if !self.initialized {
                        ks.x = raw;
                        self.initialized = true;
                    }
                    self.smoothed = ks.update(raw);
                }
            }
        }
        self.initialized = true;
    }

    /// Returns the current filtered offset in microseconds.
    ///
    /// Returns `None` if no valid samples have been added.
    #[allow(clippy::cast_possible_truncation)]
    pub fn filtered_offset_us(&self) -> Option<i64> {
        if self.window.is_empty() {
            return None;
        }
        match self.filter_type {
            FilterType::Median => {
                let mut values: Vec<i64> = self.window.iter().map(|s| s.offset_us).collect();
                values.sort_unstable();
                let mid = values.len() / 2;
                if values.len() % 2 == 0 {
                    Some((values[mid - 1] + values[mid]) / 2)
                } else {
                    Some(values[mid])
                }
            }
            _ => Some(self.smoothed as i64),
        }
    }

    /// Number of samples currently held in the window.
    pub fn sample_count(&self) -> usize {
        self.window.len()
    }

    /// Returns the filter type this instance was configured with.
    pub fn filter_type(&self) -> FilterType {
        self.filter_type
    }

    /// Clear all accumulated samples and reset the filter state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.smoothed = 0.0;
        self.initialized = false;
        if let Some(ref mut ks) = self.kalman {
            ks.x = 0.0;
            ks.p = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(offset_us: i64, delay_us: u64) -> OffsetSample {
        OffsetSample::new(offset_us, delay_us, 0)
    }

    #[test]
    fn test_filter_type_smoothing_factor_none() {
        assert!((FilterType::None.smoothing_factor() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filter_type_smoothing_factor_kalman() {
        assert!(FilterType::Kalman.smoothing_factor() > 0.0);
    }

    #[test]
    fn test_filter_type_names() {
        assert_eq!(FilterType::Median.name(), "median");
        assert_eq!(FilterType::LoPass.name(), "lopass");
    }

    #[test]
    fn test_offset_sample_is_outlier_true() {
        let s = sample(100, 5000);
        assert!(s.is_outlier(1000));
    }

    #[test]
    fn test_offset_sample_is_outlier_false() {
        let s = sample(100, 500);
        assert!(!s.is_outlier(1000));
    }

    #[test]
    fn test_filter_none_returns_last_sample() {
        let mut f = OffsetSampleFilter::new(FilterType::None, 8, 10_000);
        f.add_sample(sample(200, 100));
        assert_eq!(f.filtered_offset_us(), Some(200));
    }

    #[test]
    fn test_filter_none_empty_returns_none() {
        let f = OffsetSampleFilter::new(FilterType::None, 8, 10_000);
        assert_eq!(f.filtered_offset_us(), None);
    }

    #[test]
    fn test_filter_median_odd_window() {
        let mut f = OffsetSampleFilter::new(FilterType::Median, 8, 10_000);
        for v in [10_i64, 30, 20] {
            f.add_sample(sample(v, 100));
        }
        // sorted: [10, 20, 30] => median = 20
        assert_eq!(f.filtered_offset_us(), Some(20));
    }

    #[test]
    fn test_filter_median_even_window() {
        let mut f = OffsetSampleFilter::new(FilterType::Median, 8, 10_000);
        for v in [10_i64, 40, 20, 30] {
            f.add_sample(sample(v, 100));
        }
        // sorted: [10, 20, 30, 40] => median = (20+30)/2 = 25
        assert_eq!(f.filtered_offset_us(), Some(25));
    }

    #[test]
    fn test_filter_lopass_smooths() {
        let mut f = OffsetSampleFilter::new(FilterType::LoPass, 8, 10_000);
        f.add_sample(sample(0, 100));
        f.add_sample(sample(1000, 100));
        let out = f.filtered_offset_us().expect("should succeed in test");
        // EMA with alpha ~0.875 → should be well below 1000
        assert!(out < 1000);
        assert!(out > 0);
    }

    #[test]
    fn test_filter_kalman_converges() {
        let mut f = OffsetSampleFilter::new(FilterType::Kalman, 16, 10_000);
        for _ in 0..50 {
            f.add_sample(sample(500, 200));
        }
        let out = f.filtered_offset_us().expect("should succeed in test");
        // Should converge close to 500
        assert!((out - 500).abs() < 10);
    }

    #[test]
    fn test_outlier_dropped() {
        let mut f = OffsetSampleFilter::new(FilterType::None, 8, 1000);
        f.add_sample(sample(100, 500)); // ok
        f.add_sample(sample(999, 9999)); // outlier — dropped
        assert_eq!(f.sample_count(), 1);
    }

    #[test]
    fn test_window_size_limit() {
        let mut f = OffsetSampleFilter::new(FilterType::None, 4, 10_000);
        for i in 0..10 {
            f.add_sample(sample(i, 100));
        }
        assert_eq!(f.sample_count(), 4);
    }

    #[test]
    fn test_reset_clears_samples() {
        let mut f = OffsetSampleFilter::new(FilterType::Median, 8, 10_000);
        f.add_sample(sample(100, 50));
        f.reset();
        assert_eq!(f.sample_count(), 0);
        assert_eq!(f.filtered_offset_us(), None);
    }

    #[test]
    fn test_filter_type_accessor() {
        let f = OffsetSampleFilter::new(FilterType::Kalman, 8, 1000);
        assert_eq!(f.filter_type(), FilterType::Kalman);
    }
}
