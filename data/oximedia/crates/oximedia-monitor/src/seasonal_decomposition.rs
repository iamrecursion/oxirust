//! Seasonal decomposition for time-series anomaly detection.
//!
//! Implements classical additive seasonal decomposition (STL-like) to detect
//! hourly and daily patterns in encoding throughput and other media pipeline
//! metrics.  The decomposition separates a time series into three components:
//!
//! - **Trend**: The long-term direction (computed via a centered moving average).
//! - **Seasonal**: Repeating periodic pattern (estimated from de-trended averages).
//! - **Residual**: What remains after removing trend and seasonality.
//!
//! Anomalies are detected when the residual component exceeds a configurable
//! number of standard deviations from the residual mean.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Period hint for seasonal decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalPeriod {
    /// Hourly pattern (period = 60 samples at 1-minute granularity).
    Hourly,
    /// Daily pattern (period = 1440 samples at 1-minute granularity).
    Daily,
    /// Custom period in number of samples.
    Custom(usize),
}

impl SeasonalPeriod {
    /// Return the period length in number of samples.
    #[must_use]
    pub fn len(self) -> usize {
        match self {
            Self::Hourly => 60,
            Self::Daily => 1440,
            Self::Custom(n) => n.max(2),
        }
    }

    /// Returns whether the period length is zero (always false for valid periods).
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Configuration for the seasonal decomposition detector.
#[derive(Debug, Clone)]
pub struct SeasonalConfig {
    /// The expected seasonal period.
    pub period: SeasonalPeriod,
    /// Number of standard deviations for anomaly threshold.
    pub anomaly_sigma: f64,
    /// Maximum number of observations retained.
    pub max_observations: usize,
    /// Minimum observations required before decomposition is attempted.
    pub min_observations: usize,
}

impl Default for SeasonalConfig {
    fn default() -> Self {
        Self {
            period: SeasonalPeriod::Hourly,
            anomaly_sigma: 3.0,
            max_observations: 10_000,
            min_observations: 120, // 2 full hourly cycles
        }
    }
}

impl SeasonalConfig {
    /// Create a new config for the given period.
    #[must_use]
    pub fn new(period: SeasonalPeriod) -> Self {
        let min_obs = period.len() * 2;
        Self {
            period,
            anomaly_sigma: 3.0,
            max_observations: 10_000,
            min_observations: min_obs,
        }
    }

    /// Set the anomaly sigma threshold.
    #[must_use]
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.anomaly_sigma = sigma.max(0.5);
        self
    }

    /// Set the maximum observation buffer size.
    #[must_use]
    pub fn with_max_observations(mut self, n: usize) -> Self {
        self.max_observations = n.max(self.min_observations);
        self
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// A timestamped observation for seasonal analysis.
#[derive(Debug, Clone, Copy)]
pub struct SeasonalObservation {
    /// Timestamp.
    pub timestamp: SystemTime,
    /// Observed value.
    pub value: f64,
}

impl SeasonalObservation {
    /// Create an observation at the current time.
    #[must_use]
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
        }
    }

    /// Create an observation with an explicit timestamp.
    #[must_use]
    pub fn at(timestamp: SystemTime, value: f64) -> Self {
        Self { timestamp, value }
    }
}

// ---------------------------------------------------------------------------
// Decomposition result
// ---------------------------------------------------------------------------

/// The three-component decomposition of a time series.
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// Trend component (same length as input).
    pub trend: Vec<f64>,
    /// Seasonal component (same length as input).
    pub seasonal: Vec<f64>,
    /// Residual component (same length as input).
    pub residual: Vec<f64>,
}

/// A detected seasonal anomaly.
#[derive(Debug, Clone)]
pub struct SeasonalAnomaly {
    /// Index in the observation buffer where the anomaly was detected.
    pub index: usize,
    /// Timestamp of the anomalous observation.
    pub timestamp: SystemTime,
    /// Observed value.
    pub observed: f64,
    /// Expected value (trend + seasonal).
    pub expected: f64,
    /// Residual value (observed - expected).
    pub residual: f64,
    /// Standard deviations away from mean residual.
    pub sigma_distance: f64,
    /// Human-readable description.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Seasonal decomposition detector for time-series anomaly detection.
#[derive(Debug)]
pub struct SeasonalDecomposer {
    config: SeasonalConfig,
    observations: VecDeque<SeasonalObservation>,
}

impl SeasonalDecomposer {
    /// Create a new decomposer with the given configuration.
    #[must_use]
    pub fn new(config: SeasonalConfig) -> Self {
        Self {
            config,
            observations: VecDeque::new(),
        }
    }

    /// Create a decomposer with default hourly configuration.
    #[must_use]
    pub fn hourly() -> Self {
        Self::new(SeasonalConfig::new(SeasonalPeriod::Hourly))
    }

    /// Create a decomposer with daily configuration.
    #[must_use]
    pub fn daily() -> Self {
        Self::new(SeasonalConfig::new(SeasonalPeriod::Daily))
    }

    /// Add an observation.
    pub fn observe(&mut self, obs: SeasonalObservation) {
        self.observations.push_back(obs);
        while self.observations.len() > self.config.max_observations {
            self.observations.pop_front();
        }
    }

    /// Add a value at the current time.
    pub fn observe_now(&mut self, value: f64) {
        self.observe(SeasonalObservation::now(value));
    }

    /// Number of stored observations.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Returns `true` if enough data has been collected for decomposition.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.observations.len() >= self.config.min_observations
    }

    /// Extract the raw values from the observation buffer.
    fn values(&self) -> Vec<f64> {
        self.observations.iter().map(|o| o.value).collect()
    }

    /// Compute a centered moving average as the trend component.
    ///
    /// For an even window size `w`, each trend value at index `i` is the
    /// average of `values[i - w/2 .. i + w/2]` (with boundary padding using
    /// the nearest available value).
    fn compute_trend(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }
        let half = window / 2;
        let mut trend = Vec::with_capacity(n);
        for i in 0..n {
            let lo = if i >= half { i - half } else { 0 };
            let hi = (i + half + 1).min(n);
            let sum: f64 = values[lo..hi].iter().sum();
            let count = (hi - lo) as f64;
            trend.push(sum / count);
        }
        trend
    }

    /// Estimate the seasonal component from the de-trended series.
    ///
    /// For each position within the period, the seasonal value is the average
    /// of the de-trended values at all corresponding positions across cycles.
    fn compute_seasonal(detrended: &[f64], period: usize) -> Vec<f64> {
        let n = detrended.len();
        if n == 0 || period == 0 {
            return Vec::new();
        }

        // Average de-trended values at each position mod period.
        let mut sums = vec![0.0_f64; period];
        let mut counts = vec![0usize; period];
        for (i, &v) in detrended.iter().enumerate() {
            let pos = i % period;
            sums[pos] += v;
            counts[pos] += 1;
        }

        let mut pattern = vec![0.0_f64; period];
        for i in 0..period {
            if counts[i] > 0 {
                pattern[i] = sums[i] / counts[i] as f64;
            }
        }

        // Center the seasonal component (subtract the mean so it sums to ~0).
        let pattern_mean: f64 = pattern.iter().sum::<f64>() / period as f64;
        for v in &mut pattern {
            *v -= pattern_mean;
        }

        // Tile the pattern to cover the full series.
        (0..n).map(|i| pattern[i % period]).collect()
    }

    /// Perform additive seasonal decomposition.
    ///
    /// Returns `None` if insufficient data is available.
    #[must_use]
    pub fn decompose(&self) -> Option<Decomposition> {
        if !self.is_ready() {
            return None;
        }

        let values = self.values();
        let period = self.config.period.len();

        // Step 1: compute trend via centered moving average.
        let trend = Self::compute_trend(&values, period);

        // Step 2: de-trend.
        let detrended: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .map(|(v, t)| v - t)
            .collect();

        // Step 3: estimate seasonal component.
        let seasonal = Self::compute_seasonal(&detrended, period);

        // Step 4: compute residual.
        let residual: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((v, t), s)| v - t - s)
            .collect();

        Some(Decomposition {
            trend,
            seasonal,
            residual,
        })
    }

    /// Detect anomalies in the current observation window.
    ///
    /// Returns a list of observations whose residual component exceeds
    /// `anomaly_sigma` standard deviations from the residual mean.
    #[must_use]
    pub fn detect_anomalies(&self) -> Vec<SeasonalAnomaly> {
        let decomp = match self.decompose() {
            Some(d) => d,
            None => return Vec::new(),
        };

        let n = decomp.residual.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute mean and std dev of residuals.
        let mean = decomp.residual.iter().sum::<f64>() / n as f64;
        let variance = decomp
            .residual
            .iter()
            .map(|r| {
                let d = r - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-12 {
            return Vec::new(); // all values identical
        }

        let threshold = self.config.anomaly_sigma;
        let obs_slice: Vec<&SeasonalObservation> = self.observations.iter().collect();

        let mut anomalies = Vec::new();
        for i in 0..n {
            let sigma_dist = (decomp.residual[i] - mean).abs() / std_dev;
            if sigma_dist >= threshold {
                let expected = decomp.trend[i] + decomp.seasonal[i];
                let obs = obs_slice[i];
                let direction = if decomp.residual[i] > mean {
                    "above"
                } else {
                    "below"
                };
                anomalies.push(SeasonalAnomaly {
                    index: i,
                    timestamp: obs.timestamp,
                    observed: obs.value,
                    expected,
                    residual: decomp.residual[i],
                    sigma_distance: sigma_dist,
                    description: format!(
                        "Value {:.2} is {sigma_dist:.1} sigma {direction} expected {expected:.2}",
                        obs.value
                    ),
                });
            }
        }

        anomalies
    }

    /// Return the seasonal pattern (one full period of the estimated seasonal component).
    ///
    /// Returns `None` if decomposition is not yet possible.
    #[must_use]
    pub fn seasonal_pattern(&self) -> Option<Vec<f64>> {
        let decomp = self.decompose()?;
        let period = self.config.period.len();
        if decomp.seasonal.len() < period {
            return None;
        }
        Some(decomp.seasonal[..period].to_vec())
    }

    /// Return the current trend slope (change per sample) using OLS on the
    /// trend component.
    #[must_use]
    pub fn trend_slope(&self) -> Option<f64> {
        let decomp = self.decompose()?;
        let n = decomp.trend.len();
        if n < 2 {
            return None;
        }

        // Simple OLS: x = 0,1,...,n-1, y = trend values.
        let n_f = n as f64;
        let sum_x = n_f * (n_f - 1.0) / 2.0;
        let sum_x2 = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
        let sum_y: f64 = decomp.trend.iter().sum();
        let sum_xy: f64 = decomp
            .trend
            .iter()
            .enumerate()
            .map(|(i, &v)| i as f64 * v)
            .sum();

        let denom = n_f * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return Some(0.0);
        }

        Some((n_f * sum_xy - sum_x * sum_y) / denom)
    }

    /// Predict the value at `steps_ahead` samples into the future.
    ///
    /// Uses the latest trend value extended by the trend slope, plus the
    /// seasonal component at the corresponding phase.
    #[must_use]
    pub fn predict(&self, steps_ahead: usize) -> Option<f64> {
        let decomp = self.decompose()?;
        let slope = self.trend_slope()?;
        let n = decomp.trend.len();
        if n == 0 {
            return None;
        }

        let last_trend = decomp.trend[n - 1];
        let projected_trend = last_trend + slope * steps_ahead as f64;

        let period = self.config.period.len();
        let phase = (n + steps_ahead) % period;
        let seasonal = if phase < decomp.seasonal.len() {
            decomp.seasonal[phase]
        } else {
            0.0
        };

        Some(projected_trend + seasonal)
    }

    /// Compute the strength of seasonality as a ratio (0.0 = none, 1.0 = perfect).
    ///
    /// Defined as `1 - Var(residual) / Var(detrended)`.
    #[must_use]
    pub fn seasonal_strength(&self) -> Option<f64> {
        let decomp = self.decompose()?;
        let n = decomp.residual.len();
        if n < 2 {
            return None;
        }

        let values = self.values();
        let detrended: Vec<f64> = values
            .iter()
            .zip(decomp.trend.iter())
            .map(|(v, t)| v - t)
            .collect();

        let var_detrended = variance_of(&detrended);
        let var_residual = variance_of(&decomp.residual);

        if var_detrended < 1e-15 {
            return Some(0.0);
        }

        Some((1.0 - var_residual / var_detrended).max(0.0))
    }

    /// Clear all stored observations.
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &SeasonalConfig {
        &self.config
    }

    /// Extract the epoch seconds for an observation timestamp.
    fn epoch_secs(ts: SystemTime) -> f64 {
        ts.duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// Encoding Throughput Monitor (Task 1 — seasonal anomaly on encoding metrics)
// ---------------------------------------------------------------------------

/// Health classification for encoding throughput based on seasonal analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThroughputHealth {
    /// Throughput is within expected seasonal norms.
    Normal,
    /// Throughput is slightly outside seasonal norms (1.5–3 σ).
    Warning,
    /// Throughput is significantly outside seasonal norms (> 3 σ).
    Critical,
    /// Insufficient data for seasonal analysis.
    Unknown,
}

impl ThroughputHealth {
    /// Returns `true` if action is required.
    #[must_use]
    pub fn requires_action(self) -> bool {
        matches!(self, Self::Warning | Self::Critical)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Unknown => "unknown",
        }
    }
}

/// Encoding throughput observation with frame count and wall-clock time.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputSample {
    /// Frames-per-second observed at this sample.
    pub fps: f64,
    /// Wall-clock timestamp.
    pub timestamp: SystemTime,
}

impl ThroughputSample {
    /// Create a new throughput sample captured at the current instant.
    #[must_use]
    pub fn now(fps: f64) -> Self {
        Self {
            fps,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a sample with an explicit timestamp.
    #[must_use]
    pub fn at(fps: f64, timestamp: SystemTime) -> Self {
        Self { fps, timestamp }
    }
}

/// Result of an encoding throughput health check.
#[derive(Debug, Clone)]
pub struct ThroughputCheckResult {
    /// Latest FPS sample.
    pub fps: f64,
    /// Overall health classification.
    pub health: ThroughputHealth,
    /// Expected FPS from the seasonal model (if available).
    pub expected_fps: Option<f64>,
    /// How many standard deviations the observation is from the model.
    pub sigma_distance: Option<f64>,
    /// Predicted FPS at the next sample period.
    pub next_prediction: Option<f64>,
    /// Seasonal strength (0 = no seasonality, 1 = perfect seasonality).
    pub seasonal_strength: Option<f64>,
    /// Any anomaly detected by the underlying decomposer.
    pub anomaly: Option<SeasonalAnomaly>,
}

/// Monitors encoding throughput (frames-per-second) using seasonal decomposition
/// to distinguish genuine throughput anomalies from expected hourly/daily patterns.
///
/// For example, night-time encoding jobs naturally run at lower throughput when
/// fewer tasks compete for GPU resources; a naive threshold would fire spuriously.
/// This monitor only fires when the residual component exceeds the configured
/// sigma threshold, ensuring real anomalies are surfaced regardless of the
/// time-of-day baseline.
#[derive(Debug)]
pub struct EncodingThroughputMonitor {
    decomposer: SeasonalDecomposer,
    /// Warning threshold in standard deviations (default 1.5).
    pub warn_sigma: f64,
    /// Critical threshold in standard deviations (default 3.0).
    pub crit_sigma: f64,
}

impl EncodingThroughputMonitor {
    /// Create a monitor that detects hourly patterns in encoding throughput.
    #[must_use]
    pub fn hourly() -> Self {
        Self::new(SeasonalConfig::new(SeasonalPeriod::Hourly))
    }

    /// Create a monitor that detects daily patterns in encoding throughput.
    #[must_use]
    pub fn daily() -> Self {
        Self::new(SeasonalConfig::new(SeasonalPeriod::Daily))
    }

    /// Create a monitor with custom seasonal configuration.
    #[must_use]
    pub fn new(config: SeasonalConfig) -> Self {
        let crit = config.anomaly_sigma;
        let warn = (crit * 0.5).max(1.0);
        Self {
            decomposer: SeasonalDecomposer::new(config),
            warn_sigma: warn,
            crit_sigma: crit,
        }
    }

    /// Override the warning sigma threshold.
    #[must_use]
    pub fn with_warn_sigma(mut self, sigma: f64) -> Self {
        self.warn_sigma = sigma.max(0.5);
        self
    }

    /// Override the critical sigma threshold.
    #[must_use]
    pub fn with_crit_sigma(mut self, sigma: f64) -> Self {
        self.crit_sigma = sigma.max(self.warn_sigma);
        self
    }

    /// Record an encoding throughput sample.
    pub fn record(&mut self, sample: ThroughputSample) {
        self.decomposer.observe(SeasonalObservation {
            timestamp: sample.timestamp,
            value: sample.fps,
        });
    }

    /// Record FPS at the current time.
    pub fn record_fps(&mut self, fps: f64) {
        self.decomposer.observe_now(fps);
    }

    /// Check the current throughput health against the seasonal model.
    ///
    /// Returns a [`ThroughputCheckResult`] containing the health classification,
    /// expected value, sigma distance, and next-period prediction.
    ///
    /// If there are insufficient observations for seasonal decomposition, the
    /// result will have `health = ThroughputHealth::Unknown`.
    #[must_use]
    pub fn check(&self) -> ThroughputCheckResult {
        // Last observed FPS.
        let latest_fps = self
            .decomposer
            .observations
            .back()
            .map(|o| o.value)
            .unwrap_or(0.0);

        if !self.decomposer.is_ready() {
            return ThroughputCheckResult {
                fps: latest_fps,
                health: ThroughputHealth::Unknown,
                expected_fps: None,
                sigma_distance: None,
                next_prediction: None,
                seasonal_strength: None,
                anomaly: None,
            };
        }

        let decomp = match self.decomposer.decompose() {
            Some(d) => d,
            None => {
                return ThroughputCheckResult {
                    fps: latest_fps,
                    health: ThroughputHealth::Unknown,
                    expected_fps: None,
                    sigma_distance: None,
                    next_prediction: None,
                    seasonal_strength: None,
                    anomaly: None,
                }
            }
        };

        let n = decomp.residual.len();
        let mean_res = decomp.residual.iter().sum::<f64>() / n as f64;
        let variance_res = decomp
            .residual
            .iter()
            .map(|r| {
                let d = r - mean_res;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance_res.sqrt();

        // Expected = trend + seasonal for the last index.
        let last_idx = n.saturating_sub(1);
        let expected_fps = decomp.trend[last_idx] + decomp.seasonal[last_idx];
        let last_residual = decomp.residual[last_idx];

        let sigma_distance = if std_dev > 1e-12 {
            (last_residual - mean_res).abs() / std_dev
        } else {
            0.0
        };

        let health = if sigma_distance >= self.crit_sigma {
            ThroughputHealth::Critical
        } else if sigma_distance >= self.warn_sigma {
            ThroughputHealth::Warning
        } else {
            ThroughputHealth::Normal
        };

        // Find the anomaly record for the last index if it was flagged.
        let all_anomalies = self.decomposer.detect_anomalies();
        let anomaly = all_anomalies.into_iter().find(|a| a.index == last_idx);

        ThroughputCheckResult {
            fps: latest_fps,
            health,
            expected_fps: Some(expected_fps),
            sigma_distance: Some(sigma_distance),
            next_prediction: self.decomposer.predict(1),
            seasonal_strength: self.decomposer.seasonal_strength(),
            anomaly,
        }
    }

    /// Number of recorded samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.decomposer.observation_count()
    }

    /// Returns `true` if enough data has been collected for seasonal analysis.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.decomposer.is_ready()
    }

    /// Access the underlying decomposer.
    #[must_use]
    pub fn decomposer(&self) -> &SeasonalDecomposer {
        &self.decomposer
    }

    /// Clear all recorded samples (reset monitor).
    pub fn clear(&mut self) {
        self.decomposer.clear();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute sample variance of a slice.
fn variance_of(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn base_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    /// Generate a synthetic series: linear trend + sinusoidal seasonality + noise.
    fn synthetic_series(n: usize, period: usize) -> Vec<SeasonalObservation> {
        let base = base_time();
        (0..n)
            .map(|i| {
                let trend = 100.0 + 0.01 * i as f64; // slow upward trend
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                let value = trend + seasonal;
                SeasonalObservation::at(base + Duration::from_secs(i as u64 * 60), value)
            })
            .collect()
    }

    /// Generate a series with an anomalous spike at a specific index.
    fn series_with_anomaly(
        n: usize,
        period: usize,
        anomaly_idx: usize,
    ) -> Vec<SeasonalObservation> {
        let base = base_time();
        (0..n)
            .map(|i| {
                let trend = 100.0 + 0.01 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                let spike = if i == anomaly_idx { 80.0 } else { 0.0 };
                let value = trend + seasonal + spike;
                SeasonalObservation::at(base + Duration::from_secs(i as u64 * 60), value)
            })
            .collect()
    }

    // -- SeasonalPeriod --

    #[test]
    fn test_period_hourly_len() {
        assert_eq!(SeasonalPeriod::Hourly.len(), 60);
    }

    #[test]
    fn test_period_daily_len() {
        assert_eq!(SeasonalPeriod::Daily.len(), 1440);
    }

    #[test]
    fn test_period_custom_len() {
        assert_eq!(SeasonalPeriod::Custom(30).len(), 30);
    }

    #[test]
    fn test_period_custom_min_2() {
        assert_eq!(SeasonalPeriod::Custom(0).len(), 2);
        assert_eq!(SeasonalPeriod::Custom(1).len(), 2);
    }

    // -- SeasonalConfig --

    #[test]
    fn test_config_default() {
        let cfg = SeasonalConfig::default();
        assert_eq!(cfg.period, SeasonalPeriod::Hourly);
        assert!((cfg.anomaly_sigma - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_with_sigma() {
        let cfg = SeasonalConfig::default().with_sigma(2.0);
        assert!((cfg.anomaly_sigma - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_sigma_min() {
        let cfg = SeasonalConfig::default().with_sigma(0.1);
        assert!((cfg.anomaly_sigma - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_config_new_sets_min_observations() {
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(10));
        assert_eq!(cfg.min_observations, 20);
    }

    // -- SeasonalObservation --

    #[test]
    fn test_observation_now() {
        let obs = SeasonalObservation::now(42.0);
        assert!((obs.value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_observation_at() {
        let t = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let obs = SeasonalObservation::at(t, 55.0);
        assert_eq!(obs.timestamp, t);
        assert!((obs.value - 55.0).abs() < 1e-9);
    }

    // -- SeasonalDecomposer basics --

    #[test]
    fn test_decomposer_new() {
        let d = SeasonalDecomposer::hourly();
        assert_eq!(d.observation_count(), 0);
        assert!(!d.is_ready());
    }

    #[test]
    fn test_decomposer_observe() {
        let mut d = SeasonalDecomposer::hourly();
        d.observe_now(1.0);
        d.observe_now(2.0);
        assert_eq!(d.observation_count(), 2);
    }

    #[test]
    fn test_decomposer_max_observations() {
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(5)).with_max_observations(20);
        let mut d = SeasonalDecomposer::new(cfg);
        for i in 0..30 {
            d.observe_now(i as f64);
        }
        assert!(d.observation_count() <= 20);
    }

    #[test]
    fn test_decompose_insufficient_data() {
        let mut d = SeasonalDecomposer::new(SeasonalConfig::new(SeasonalPeriod::Custom(5)));
        d.observe_now(1.0);
        assert!(d.decompose().is_none());
    }

    // -- Decomposition --

    #[test]
    fn test_decompose_synthetic_series() {
        let period = 20;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(100, period) {
            d.observe(obs);
        }
        assert!(d.is_ready());
        let decomp = d.decompose().expect("decompose should succeed");
        assert_eq!(decomp.trend.len(), 100);
        assert_eq!(decomp.seasonal.len(), 100);
        assert_eq!(decomp.residual.len(), 100);
    }

    #[test]
    fn test_decompose_seasonal_component_is_periodic() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(60, period) {
            d.observe(obs);
        }
        let decomp = d.decompose().expect("decompose should succeed");
        // The seasonal component should repeat every `period` samples.
        for i in 0..period {
            let s0 = decomp.seasonal[i];
            let s1 = decomp.seasonal[i + period];
            assert!(
                (s0 - s1).abs() < 1e-9,
                "seasonal[{i}]={s0} != seasonal[{}]={s1}",
                i + period
            );
        }
    }

    #[test]
    fn test_decompose_residual_small_for_clean_data() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(100, period) {
            d.observe(obs);
        }
        let decomp = d.decompose().expect("decompose should succeed");
        let max_residual = decomp
            .residual
            .iter()
            .map(|r| r.abs())
            .fold(0.0_f64, f64::max);
        // For clean synthetic data, residuals should be small.
        assert!(
            max_residual < 5.0,
            "max residual {max_residual} is too large for clean data"
        );
    }

    // -- Anomaly detection --

    #[test]
    fn test_detect_anomalies_clean_data() {
        let period = 10;
        // Use a higher sigma threshold to avoid false positives from
        // boundary effects in the centered moving-average trend estimator.
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period)).with_sigma(5.0);
        let mut d = SeasonalDecomposer::new(cfg);
        // Use many data points (500 = 50 full cycles) to minimize edge effects.
        for obs in synthetic_series(500, period) {
            d.observe(obs);
        }
        let anomalies = d.detect_anomalies();
        // Only boundary samples (first/last ~half-period) may appear as anomalies.
        assert!(
            anomalies.len() <= period,
            "clean data should have very few anomalies (boundary effects only), got {}",
            anomalies.len()
        );
    }

    #[test]
    fn test_detect_anomalies_spike_detected() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period)).with_sigma(3.0);
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in series_with_anomaly(100, period, 50) {
            d.observe(obs);
        }
        let anomalies = d.detect_anomalies();
        assert!(
            !anomalies.is_empty(),
            "spike at index 50 should be detected"
        );
        // The anomaly should be near index 50.
        let has_spike = anomalies.iter().any(|a| a.index == 50);
        assert!(has_spike, "anomaly at index 50 expected");
    }

    #[test]
    fn test_detect_anomalies_returns_metadata() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period)).with_sigma(2.0);
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in series_with_anomaly(80, period, 40) {
            d.observe(obs);
        }
        let anomalies = d.detect_anomalies();
        if let Some(a) = anomalies.iter().find(|a| a.index == 40) {
            assert!(a.sigma_distance >= 2.0);
            assert!(!a.description.is_empty());
            assert!(a.observed > a.expected);
        }
    }

    #[test]
    fn test_detect_anomalies_not_ready() {
        let d = SeasonalDecomposer::new(SeasonalConfig::new(SeasonalPeriod::Custom(10)));
        assert!(d.detect_anomalies().is_empty());
    }

    // -- Seasonal pattern --

    #[test]
    fn test_seasonal_pattern_length() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(60, period) {
            d.observe(obs);
        }
        let pattern = d.seasonal_pattern().expect("pattern should exist");
        assert_eq!(pattern.len(), period);
    }

    #[test]
    fn test_seasonal_pattern_centered() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(60, period) {
            d.observe(obs);
        }
        let pattern = d.seasonal_pattern().expect("pattern should exist");
        let sum: f64 = pattern.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "seasonal pattern should be centered (sum={sum})"
        );
    }

    // -- Trend slope --

    #[test]
    fn test_trend_slope_positive_for_increasing_data() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        // Use enough data (300 samples = 30 full cycles) so boundary effects
        // in the centered moving average do not dominate the gentle 0.01/sample trend.
        for obs in synthetic_series(300, period) {
            d.observe(obs);
        }
        let slope = d.trend_slope().expect("slope should exist");
        assert!(slope > 0.0, "slope should be positive, got {slope}");
    }

    #[test]
    fn test_trend_slope_near_zero_for_flat_data() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        let base = base_time();
        for i in 0..60 {
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            d.observe(SeasonalObservation::at(
                base + Duration::from_secs(i * 60),
                50.0 + seasonal,
            ));
        }
        let slope = d.trend_slope().expect("slope should exist");
        assert!(slope.abs() < 0.1, "slope should be near zero, got {slope}");
    }

    // -- Prediction --

    #[test]
    fn test_predict_follows_trend() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        for obs in synthetic_series(60, period) {
            d.observe(obs);
        }
        let pred_1 = d.predict(1).expect("predict should succeed");
        let pred_10 = d.predict(10).expect("predict should succeed");
        // With positive trend, further predictions should be higher on average.
        // (seasonal variation may cause exceptions but over a full period the trend dominates.)
        let pred_period = d.predict(period).expect("predict should succeed");
        let current = d.predict(0).expect("predict should succeed");
        assert!(
            pred_period > current - 1.0,
            "prediction at +{period} ({pred_period}) should be >= current ({current}) minus tolerance"
        );
        // Simple sanity: predictions should be finite.
        assert!(pred_1.is_finite());
        assert!(pred_10.is_finite());
    }

    #[test]
    fn test_predict_not_ready() {
        let d = SeasonalDecomposer::new(SeasonalConfig::new(SeasonalPeriod::Custom(10)));
        assert!(d.predict(5).is_none());
    }

    // -- Seasonal strength --

    #[test]
    fn test_seasonal_strength_strong_for_periodic_data() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        let base = base_time();
        for i in 0..100 {
            let seasonal = 20.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            d.observe(SeasonalObservation::at(
                base + Duration::from_secs(i * 60),
                100.0 + seasonal, // pure seasonal, no noise
            ));
        }
        let strength = d.seasonal_strength().expect("strength should exist");
        assert!(
            strength > 0.5,
            "seasonal strength should be high for pure periodic data, got {strength}"
        );
    }

    #[test]
    fn test_seasonal_strength_low_for_random_data() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut d = SeasonalDecomposer::new(cfg);
        let base = base_time();
        // Pseudo-random via simple LCG.
        let mut rng_state: u64 = 12345;
        for i in 0..100 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pseudo_random = (rng_state >> 33) as f64 / (u32::MAX as f64) * 100.0;
            d.observe(SeasonalObservation::at(
                base + Duration::from_secs(i * 60),
                pseudo_random,
            ));
        }
        let strength = d.seasonal_strength().expect("strength should exist");
        assert!(
            strength < 0.8,
            "seasonal strength should be low for random data, got {strength}"
        );
    }

    // -- Clear --

    #[test]
    fn test_clear() {
        let mut d = SeasonalDecomposer::hourly();
        d.observe_now(1.0);
        d.observe_now(2.0);
        d.clear();
        assert_eq!(d.observation_count(), 0);
        assert!(!d.is_ready());
    }

    // -- variance_of helper --

    #[test]
    fn test_variance_of_constant() {
        assert!((variance_of(&[5.0, 5.0, 5.0]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_variance_of_simple() {
        // [1,2,3,4,5] mean=3, var = (4+1+0+1+4)/5 = 2.0
        let v = variance_of(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((v - 2.0).abs() < 1e-9, "variance={v}");
    }

    #[test]
    fn test_variance_of_empty() {
        assert!((variance_of(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_variance_of_single() {
        assert!((variance_of(&[42.0]) - 0.0).abs() < 1e-12);
    }

    // -- EncodingThroughputMonitor --

    /// Build a throughput monitor with enough synthetic FPS data to be ready.
    fn make_ready_throughput_monitor(period: usize, base_fps: f64) -> EncodingThroughputMonitor {
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period));
        let mut monitor = EncodingThroughputMonitor::new(cfg);
        let base = base_time();
        for i in 0..(period * 3) {
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            let fps = base_fps + seasonal;
            monitor.record(ThroughputSample::at(
                fps,
                base + Duration::from_secs(i as u64 * 60),
            ));
        }
        monitor
    }

    #[test]
    fn test_throughput_monitor_hourly_ctor() {
        let m = EncodingThroughputMonitor::hourly();
        assert!(!m.is_ready());
        assert_eq!(m.sample_count(), 0);
    }

    #[test]
    fn test_throughput_monitor_daily_ctor() {
        let m = EncodingThroughputMonitor::daily();
        assert!(!m.is_ready());
    }

    #[test]
    fn test_throughput_monitor_unknown_when_not_ready() {
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(10));
        let mut monitor = EncodingThroughputMonitor::new(cfg);
        monitor.record_fps(30.0);
        let result = monitor.check();
        assert_eq!(result.health, ThroughputHealth::Unknown);
        assert!(result.expected_fps.is_none());
        assert!(result.sigma_distance.is_none());
    }

    #[test]
    fn test_throughput_monitor_normal_for_expected_fps() {
        let monitor = make_ready_throughput_monitor(10, 30.0);
        assert!(monitor.is_ready());
        let result = monitor.check();
        // Steady FPS within seasonal pattern → Normal or at most Warning.
        assert!(
            result.health != ThroughputHealth::Critical,
            "steady FPS should not be Critical, got {:?}",
            result.health
        );
        assert!(result.expected_fps.is_some());
        assert!(result.sigma_distance.is_some());
        assert!(result.seasonal_strength.is_some());
    }

    #[test]
    fn test_throughput_monitor_detects_fps_drop() {
        let period = 10;
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(period)).with_sigma(2.0);
        let mut monitor = EncodingThroughputMonitor::new(cfg).with_warn_sigma(1.5);
        let base = base_time();
        // Feed 40 samples of steady FPS = 30.
        for i in 0..40usize {
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            monitor.record(ThroughputSample::at(
                30.0 + seasonal,
                base + Duration::from_secs(i as u64 * 60),
            ));
        }
        // Inject a severe FPS drop (30 → 1) — residual should exceed thresholds.
        monitor.record(ThroughputSample::at(
            1.0, // massive drop
            base + Duration::from_secs(40 * 60),
        ));
        let result = monitor.check();
        // The FPS drop should push the result to Warning or Critical.
        assert!(
            result.health.requires_action(),
            "severe FPS drop should trigger Warning/Critical, got {:?} sigma={:?}",
            result.health,
            result.sigma_distance,
        );
    }

    #[test]
    fn test_throughput_monitor_clear_resets() {
        let mut monitor = make_ready_throughput_monitor(10, 30.0);
        assert!(monitor.is_ready());
        monitor.clear();
        assert!(!monitor.is_ready());
        assert_eq!(monitor.sample_count(), 0);
    }

    #[test]
    fn test_throughput_monitor_next_prediction_exists_when_ready() {
        let monitor = make_ready_throughput_monitor(10, 25.0);
        let result = monitor.check();
        assert!(
            result.next_prediction.is_some(),
            "next_prediction should be Some when ready"
        );
        assert!(result
            .next_prediction
            .expect("prediction should be Some")
            .is_finite());
    }

    #[test]
    fn test_throughput_health_label() {
        assert_eq!(ThroughputHealth::Normal.label(), "normal");
        assert_eq!(ThroughputHealth::Warning.label(), "warning");
        assert_eq!(ThroughputHealth::Critical.label(), "critical");
        assert_eq!(ThroughputHealth::Unknown.label(), "unknown");
    }

    #[test]
    fn test_throughput_health_requires_action() {
        assert!(!ThroughputHealth::Normal.requires_action());
        assert!(!ThroughputHealth::Unknown.requires_action());
        assert!(ThroughputHealth::Warning.requires_action());
        assert!(ThroughputHealth::Critical.requires_action());
    }

    #[test]
    fn test_throughput_sample_at() {
        let ts = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let sample = ThroughputSample::at(60.0, ts);
        assert!((sample.fps - 60.0).abs() < 1e-9);
        assert_eq!(sample.timestamp, ts);
    }

    #[test]
    fn test_throughput_decomposer_access() {
        let monitor = make_ready_throughput_monitor(10, 30.0);
        let decomposer = monitor.decomposer();
        assert!(decomposer.observation_count() > 0);
        assert!(decomposer.is_ready());
    }

    #[test]
    fn test_throughput_monitor_with_sigma_override() {
        let cfg = SeasonalConfig::new(SeasonalPeriod::Custom(10));
        let monitor = EncodingThroughputMonitor::new(cfg)
            .with_warn_sigma(2.0)
            .with_crit_sigma(4.0);
        assert!((monitor.warn_sigma - 2.0).abs() < 1e-9);
        assert!((monitor.crit_sigma - 4.0).abs() < 1e-9);
    }
}
