//! Uncertainty estimation for probabilistic predictions.
//!
//! Provides Monte Carlo sampling, calibration metrics, confidence intervals,
//! and prediction intervals for regression and classification tasks.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during uncertainty estimation.
#[derive(Debug, Clone)]
pub enum UncertaintyError {
    EmptyPredictions,
    InvalidNumSamples(usize),
    InvalidConfidenceLevel(f64),
    ShapeMismatch { expected: usize, got: usize },
    InvalidBins(usize),
    SamplingError(String),
}

impl fmt::Display for UncertaintyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UncertaintyError::EmptyPredictions => write!(f, "predictions slice is empty"),
            UncertaintyError::InvalidNumSamples(n) => {
                write!(f, "num_samples must be >= 1, got {n}")
            }
            UncertaintyError::InvalidConfidenceLevel(l) => {
                write!(f, "confidence_level must be in (0, 1), got {l}")
            }
            UncertaintyError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            UncertaintyError::InvalidBins(b) => {
                write!(f, "num_bins must be >= 1, got {b}")
            }
            UncertaintyError::SamplingError(msg) => write!(f, "sampling error: {msg}"),
        }
    }
}

impl std::error::Error for UncertaintyError {}

// ---------------------------------------------------------------------------
// Simple LCG-based RNG (no external rand crate)
// ---------------------------------------------------------------------------

struct SimpleUncertaintyRng {
    state: u64,
}

impl SimpleUncertaintyRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    /// Returns a uniformly distributed f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Xorshift64 for better quality than a plain LCG
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // Map to [0, 1)
        (self.state as f64) / (u64::MAX as f64 + 1.0)
    }

    /// Box-Muller transform — returns a standard normal sample N(0,1).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// ConfidenceInterval
// ---------------------------------------------------------------------------

/// Method used to construct a confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalMethod {
    /// Empirical percentiles derived from sample quantiles.
    Percentile,
    /// Gaussian approximation: mean ± z * std.
    Normal,
}

/// A confidence interval [lower, upper] at a given level (e.g. 0.95).
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
    pub method: IntervalMethod,
}

impl ConfidenceInterval {
    /// Construct a percentile-based CI by sorting `samples` and taking quantiles.
    pub fn percentile(samples: &[f64], level: f64) -> Result<Self, UncertaintyError> {
        if samples.is_empty() {
            return Err(UncertaintyError::EmptyPredictions);
        }
        if level <= 0.0 || level >= 1.0 {
            return Err(UncertaintyError::InvalidConfidenceLevel(level));
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let alpha = (1.0 - level) / 2.0;
        let lower = quantile_sorted(&sorted, alpha);
        let upper = quantile_sorted(&sorted, 1.0 - alpha);
        Ok(Self {
            lower,
            upper,
            level,
            method: IntervalMethod::Percentile,
        })
    }

    /// Construct a Normal CI using mean ± z * std.
    pub fn normal(mean: f64, std: f64, level: f64) -> Self {
        let z = z_score(level);
        Self {
            lower: mean - z * std,
            upper: mean + z * std,
            level,
            method: IntervalMethod::Normal,
        }
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Whether `value` lies within [lower, upper].
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

// ---------------------------------------------------------------------------
// UncertaintyEstimate
// ---------------------------------------------------------------------------

/// A single uncertainty estimate for a prediction.
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub confidence_interval: ConfidenceInterval,
    pub entropy: f64,
    /// Uncertainty due to the model (epistemic).
    pub epistemic_uncertainty: f64,
    /// Uncertainty inherent in the data (aleatoric).
    pub aleatoric_uncertainty: f64,
}

impl UncertaintyEstimate {
    /// Compute an estimate from a slice of scalar samples.
    pub fn from_samples(samples: &[f64], confidence_level: f64) -> Result<Self, UncertaintyError> {
        if samples.is_empty() {
            return Err(UncertaintyError::EmptyPredictions);
        }
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(UncertaintyError::InvalidConfidenceLevel(confidence_level));
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let confidence_interval = ConfidenceInterval::percentile(samples, confidence_level)?;
        let entropy = histogram_entropy(samples, 10);

        // With a single vector of samples we treat the full variance as epistemic.
        let epistemic_uncertainty = variance;
        let aleatoric_uncertainty = 0.0;

        Ok(Self {
            mean,
            variance,
            std_dev,
            confidence_interval,
            entropy,
            epistemic_uncertainty,
            aleatoric_uncertainty,
        })
    }

    /// Returns `true` when the standard deviation is below `threshold`.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.std_dev < threshold
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "UncertaintyEstimate {{ mean: {:.4}, std: {:.4}, CI [{:.4}, {:.4}] @{:.0}%, \
             entropy: {:.4}, epistemic: {:.4}, aleatoric: {:.4} }}",
            self.mean,
            self.std_dev,
            self.confidence_interval.lower,
            self.confidence_interval.upper,
            self.confidence_interval.level * 100.0,
            self.entropy,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
        )
    }
}

// ---------------------------------------------------------------------------
// MonteCarloEstimator
// ---------------------------------------------------------------------------

/// Monte Carlo estimator: run a function N times with injected Gaussian noise.
pub struct MonteCarloEstimator {
    pub num_samples: usize,
    pub confidence_level: f64,
    rng: SimpleUncertaintyRng,
}

impl MonteCarloEstimator {
    /// Create a new estimator.
    ///
    /// * `num_samples` – must be >= 1.
    /// * `confidence_level` – must be in (0, 1).
    /// * `seed` – RNG seed for reproducibility.
    pub fn new(
        num_samples: usize,
        confidence_level: f64,
        seed: u64,
    ) -> Result<Self, UncertaintyError> {
        if num_samples < 1 {
            return Err(UncertaintyError::InvalidNumSamples(num_samples));
        }
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(UncertaintyError::InvalidConfidenceLevel(confidence_level));
        }
        Ok(Self {
            num_samples,
            confidence_level,
            rng: SimpleUncertaintyRng::new(seed),
        })
    }

    /// Convenient constructor with defaults: 100 samples, 0.95 CI, seed=42.
    pub fn with_defaults() -> Self {
        // Safe: values are valid.
        Self {
            num_samples: 100,
            confidence_level: 0.95,
            rng: SimpleUncertaintyRng::new(42),
        }
    }

    /// Run `f(noise)` `num_samples` times, injecting N(0,1) noise each call.
    pub fn estimate<F>(&mut self, f: F) -> Result<UncertaintyEstimate, UncertaintyError>
    where
        F: Fn(f64) -> f64,
    {
        let samples: Vec<f64> = (0..self.num_samples)
            .map(|_| {
                let noise = self.rng.next_normal();
                f(noise)
            })
            .collect();
        UncertaintyEstimate::from_samples(&samples, self.confidence_level)
    }

    /// Run `f(noise)` `num_samples` times, average per element.
    ///
    /// `dim` – expected length of the vector returned by `f`.
    pub fn estimate_vector<F>(
        &mut self,
        dim: usize,
        f: F,
    ) -> Result<Vec<UncertaintyEstimate>, UncertaintyError>
    where
        F: Fn(f64) -> Vec<f64>,
    {
        if dim == 0 {
            return Err(UncertaintyError::ShapeMismatch {
                expected: 1,
                got: 0,
            });
        }
        // Collect num_samples × dim matrix
        let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            let noise = self.rng.next_normal();
            let row = f(noise);
            if row.len() != dim {
                return Err(UncertaintyError::ShapeMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
            matrix.push(row);
        }

        // Transpose: for each dimension, collect samples across runs
        let mut estimates = Vec::with_capacity(dim);
        for col in 0..dim {
            let col_samples: Vec<f64> = matrix.iter().map(|row| row[col]).collect();
            let est = UncertaintyEstimate::from_samples(&col_samples, self.confidence_level)?;
            estimates.push(est);
        }
        Ok(estimates)
    }
}

// ---------------------------------------------------------------------------
// CalibrationMetrics
// ---------------------------------------------------------------------------

/// Statistics for a single calibration bin.
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub count: usize,
    pub avg_confidence: f64,
    pub accuracy: f64,
    /// avg_confidence − accuracy (positive → overconfident).
    pub gap: f64,
}

/// Calibration metrics for probabilistic classifiers.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error.
    pub ece: f64,
    /// Maximum Calibration Error.
    pub mce: f64,
    /// Average gap in overconfident bins.
    pub overconfidence: f64,
    /// Average gap in underconfident bins.
    pub underconfidence: f64,
    pub num_bins: usize,
    pub bin_stats: Vec<CalibrationBin>,
}

impl CalibrationMetrics {
    /// Compute calibration metrics from predicted probabilities and true binary labels.
    ///
    /// * `predicted_probs` – probability of positive class, in [0, 1].
    /// * `true_labels` – 0 or 1.
    /// * `num_bins` – number of equal-width bins.
    pub fn compute(
        predicted_probs: &[f64],
        true_labels: &[u8],
        num_bins: usize,
    ) -> Result<Self, UncertaintyError> {
        if predicted_probs.is_empty() {
            return Err(UncertaintyError::EmptyPredictions);
        }
        if num_bins < 1 {
            return Err(UncertaintyError::InvalidBins(num_bins));
        }
        if predicted_probs.len() != true_labels.len() {
            return Err(UncertaintyError::ShapeMismatch {
                expected: predicted_probs.len(),
                got: true_labels.len(),
            });
        }

        let total = predicted_probs.len() as f64;
        let bin_width = 1.0 / num_bins as f64;

        // Accumulate per-bin sums
        let mut bin_conf_sum = vec![0.0_f64; num_bins];
        let mut bin_acc_sum = vec![0.0_f64; num_bins];
        let mut bin_count = vec![0usize; num_bins];

        for (p, y) in predicted_probs.iter().zip(true_labels.iter()) {
            let p = p.clamp(0.0, 1.0);
            let bin_idx = ((p / bin_width).floor() as usize).min(num_bins - 1);
            bin_conf_sum[bin_idx] += p;
            bin_acc_sum[bin_idx] += *y as f64;
            bin_count[bin_idx] += 1;
        }

        let mut bin_stats = Vec::with_capacity(num_bins);
        let mut ece = 0.0_f64;
        let mut mce = 0.0_f64;
        let mut over_gaps = Vec::new();
        let mut under_gaps = Vec::new();

        for i in 0..num_bins {
            let count = bin_count[i];
            let conf_lower = i as f64 * bin_width;
            let conf_upper = conf_lower + bin_width;
            let (avg_confidence, accuracy, gap) = if count == 0 {
                (0.0, 0.0, 0.0)
            } else {
                let avg_conf = bin_conf_sum[i] / count as f64;
                let acc = bin_acc_sum[i] / count as f64;
                (avg_conf, acc, avg_conf - acc)
            };

            if count > 0 {
                let weight = count as f64 / total;
                ece += weight * gap.abs();
                mce = mce.max(gap.abs());
                if gap > 0.0 {
                    over_gaps.push(gap);
                } else if gap < 0.0 {
                    under_gaps.push(-gap);
                }
            }

            bin_stats.push(CalibrationBin {
                confidence_lower: conf_lower,
                confidence_upper: conf_upper,
                count,
                avg_confidence,
                accuracy,
                gap,
            });
        }

        let overconfidence = if over_gaps.is_empty() {
            0.0
        } else {
            over_gaps.iter().sum::<f64>() / over_gaps.len() as f64
        };
        let underconfidence = if under_gaps.is_empty() {
            0.0
        } else {
            under_gaps.iter().sum::<f64>() / under_gaps.len() as f64
        };

        Ok(Self {
            ece,
            mce,
            overconfidence,
            underconfidence,
            num_bins,
            bin_stats,
        })
    }

    /// True when ECE is below `ece_threshold`.
    pub fn is_well_calibrated(&self, ece_threshold: f64) -> bool {
        self.ece < ece_threshold
    }

    /// ASCII reliability diagram (confidence vs accuracy per bin).
    pub fn format_reliability_diagram(&self) -> String {
        let mut lines = vec!["Reliability Diagram (conf → accuracy):".to_string()];
        for bin in &self.bin_stats {
            if bin.count == 0 {
                continue;
            }
            let bar_len = (bin.accuracy * 20.0).round() as usize;
            let bar = "#".repeat(bar_len);
            lines.push(format!(
                "[{:.2},{:.2}] n={:4}  acc={:.3}  conf={:.3}  gap={:+.3}  |{}|",
                bin.confidence_lower,
                bin.confidence_upper,
                bin.count,
                bin.accuracy,
                bin.avg_confidence,
                bin.gap,
                bar,
            ));
        }
        lines.join("\n")
    }

    /// Short summary string.
    pub fn summary(&self) -> String {
        format!(
            "CalibrationMetrics {{ ECE: {:.4}, MCE: {:.4}, over: {:.4}, under: {:.4}, bins: {} }}",
            self.ece, self.mce, self.overconfidence, self.underconfidence, self.num_bins
        )
    }
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------

/// Apply temperature scaling to raw logits: softmax(logit / T).
pub fn temperature_scale(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
    softmax_vec(&scaled)
}

/// Find the temperature that minimises the negative log-likelihood on the
/// provided logits and true labels.
///
/// `temperatures` – a grid of candidate temperatures to evaluate.
pub fn find_optimal_temperature(
    logits: &[f64],
    true_labels: &[u8],
    temperatures: &[f64],
) -> Result<f64, UncertaintyError> {
    if logits.is_empty() {
        return Err(UncertaintyError::EmptyPredictions);
    }
    if logits.len() != true_labels.len() {
        return Err(UncertaintyError::ShapeMismatch {
            expected: logits.len(),
            got: true_labels.len(),
        });
    }
    if temperatures.is_empty() {
        return Err(UncertaintyError::SamplingError(
            "temperatures slice is empty".to_string(),
        ));
    }

    let mut best_temp = temperatures[0];
    let mut best_nll = f64::INFINITY;

    for &t in temperatures {
        if t <= 0.0 {
            continue;
        }
        let nll = compute_nll(logits, true_labels, t);
        if nll < best_nll {
            best_nll = nll;
            best_temp = t;
        }
    }
    Ok(best_temp)
}

// ---------------------------------------------------------------------------
// PredictionInterval
// ---------------------------------------------------------------------------

/// Prediction intervals for regression tasks.
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    pub predictions: Vec<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    /// Empirical coverage (fraction of actuals inside the interval), if actuals provided.
    pub coverage: f64,
    /// Average interval width.
    pub avg_width: f64,
}

impl PredictionInterval {
    /// Construct from quantile predictions.
    ///
    /// If `actuals` is `Some`, compute empirical coverage.
    pub fn from_quantile_predictions(
        lower_preds: Vec<f64>,
        upper_preds: Vec<f64>,
        actuals: Option<&[f64]>,
    ) -> Result<Self, UncertaintyError> {
        if lower_preds.is_empty() {
            return Err(UncertaintyError::EmptyPredictions);
        }
        if lower_preds.len() != upper_preds.len() {
            return Err(UncertaintyError::ShapeMismatch {
                expected: lower_preds.len(),
                got: upper_preds.len(),
            });
        }

        // Use midpoint of [lower, upper] as point prediction
        let predictions: Vec<f64> = lower_preds
            .iter()
            .zip(upper_preds.iter())
            .map(|(lo, hi)| (lo + hi) / 2.0)
            .collect();

        let avg_width = lower_preds
            .iter()
            .zip(upper_preds.iter())
            .map(|(lo, hi)| (hi - lo).abs())
            .sum::<f64>()
            / lower_preds.len() as f64;

        let coverage = match actuals {
            None => 0.0,
            Some(act) => {
                if act.len() != lower_preds.len() {
                    return Err(UncertaintyError::ShapeMismatch {
                        expected: lower_preds.len(),
                        got: act.len(),
                    });
                }
                let covered = lower_preds
                    .iter()
                    .zip(upper_preds.iter())
                    .zip(act.iter())
                    .filter(|((lo, hi), y)| *y >= *lo && *y <= *hi)
                    .count();
                covered as f64 / act.len() as f64
            }
        };

        Ok(Self {
            predictions,
            lower_bounds: lower_preds,
            upper_bounds: upper_preds,
            coverage,
            avg_width,
        })
    }

    /// Short summary string.
    pub fn summary(&self) -> String {
        format!(
            "PredictionInterval {{ n: {}, avg_width: {:.4}, coverage: {:.4} }}",
            self.predictions.len(),
            self.avg_width,
            self.coverage,
        )
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Linear interpolation quantile on a sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Return a z-score for a given confidence level (two-tailed).
fn z_score(level: f64) -> f64 {
    if (level - 0.99).abs() < 1e-9 {
        2.576
    } else if (level - 0.90).abs() < 1e-9 {
        1.645
    } else {
        // Default to 1.96 (≈ 0.95)
        1.96
    }
}

/// Entropy from an empirical histogram of `samples` with `num_bins` bins.
fn histogram_entropy(samples: &[f64], num_bins: usize) -> f64 {
    if samples.is_empty() || num_bins == 0 {
        return 0.0;
    }
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return 0.0;
    }
    let width = (max - min) / num_bins as f64;
    let mut counts = vec![0usize; num_bins];
    for &x in samples {
        let idx = (((x - min) / width).floor() as usize).min(num_bins - 1);
        counts[idx] += 1;
    }
    let n = samples.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum()
}

/// Numerically stable softmax over a vector.
fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Compute NLL for binary classification with temperature scaling.
/// Logits are treated as log-odds (sigmoid output).
fn compute_nll(logits: &[f64], true_labels: &[u8], temperature: f64) -> f64 {
    let mut nll = 0.0_f64;
    for (&l, &y) in logits.iter().zip(true_labels.iter()) {
        let scaled = l / temperature;
        // Sigmoid probability
        let p = sigmoid(scaled);
        let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
        if y == 1 {
            nll -= p_clamped.ln();
        } else {
            nll -= (1.0 - p_clamped).ln();
        }
    }
    nll / logits.len() as f64
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- UncertaintyEstimate -----------------------------------------------

    #[test]
    fn test_uncertainty_estimate_from_samples_basic() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let est = UncertaintyEstimate::from_samples(&samples, 0.95).unwrap();
        // Mean of 0..99 = 49.5
        assert!((est.mean - 49.5).abs() < 0.01, "mean={}", est.mean);
        assert!(est.variance > 0.0);
        assert!(est.std_dev > 0.0);
    }

    #[test]
    fn test_uncertainty_estimate_confident() {
        // Near-constant samples → very low std_dev
        let samples = vec![1.0_f64; 50];
        let est = UncertaintyEstimate::from_samples(&samples, 0.95).unwrap();
        assert!(est.is_confident(0.1), "std_dev should be ~0");
    }

    #[test]
    fn test_uncertainty_estimate_not_confident() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 * 10.0).collect();
        let est = UncertaintyEstimate::from_samples(&samples, 0.95).unwrap();
        assert!(!est.is_confident(1.0), "high variance → not confident");
    }

    #[test]
    fn test_uncertainty_estimate_summary_nonempty() {
        let samples: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let est = UncertaintyEstimate::from_samples(&samples, 0.95).unwrap();
        let s = est.summary();
        assert!(!s.is_empty());
        assert!(s.contains("mean"));
    }

    // ---- ConfidenceInterval ------------------------------------------------

    #[test]
    fn test_confidence_interval_percentile() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let ci = ConfidenceInterval::percentile(&samples, 0.95).unwrap();
        assert!(ci.lower < ci.upper, "lower={} upper={}", ci.lower, ci.upper);
        assert_eq!(ci.method, IntervalMethod::Percentile);
    }

    #[test]
    fn test_confidence_interval_normal_width() {
        // 95% CI: width = 2 * 1.96 * std
        let mean = 0.0;
        let std = 1.0;
        let ci = ConfidenceInterval::normal(mean, std, 0.95);
        let expected_width = 2.0 * 1.96 * std;
        assert!(
            (ci.width() - expected_width).abs() < 1e-9,
            "width={}",
            ci.width()
        );
    }

    #[test]
    fn test_confidence_interval_contains() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 / 10.0).collect();
        let ci = ConfidenceInterval::percentile(&samples, 0.95).unwrap();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(ci.contains(mean), "mean should be inside CI");
    }

    #[test]
    fn test_confidence_interval_width_positive() {
        let ci = ConfidenceInterval::normal(5.0, 2.0, 0.95);
        assert!(ci.width() > 0.0);
    }

    // ---- MonteCarloEstimator -----------------------------------------------

    #[test]
    fn test_mc_estimator_with_defaults() {
        let est = MonteCarloEstimator::with_defaults();
        assert_eq!(est.num_samples, 100);
        assert!((est.confidence_level - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_mc_estimator_estimate_constant_fn() {
        let mut mc = MonteCarloEstimator::new(200, 0.95, 1).unwrap();
        // f(noise) = 5.0 regardless of noise
        let est = mc.estimate(|_noise| 5.0).unwrap();
        assert!((est.mean - 5.0).abs() < 1e-9, "mean={}", est.mean);
        assert!(est.std_dev < 1e-9, "std_dev={}", est.std_dev);
    }

    #[test]
    fn test_mc_estimator_estimate_linear_fn() {
        let mut mc = MonteCarloEstimator::new(2000, 0.95, 7).unwrap();
        // f(noise) = noise ~ N(0,1) → mean ≈ 0, std ≈ 1
        let est = mc.estimate(|noise| noise).unwrap();
        assert!(est.mean.abs() < 0.15, "mean should be ~0, got {}", est.mean);
        assert!(
            (est.std_dev - 1.0).abs() < 0.15,
            "std_dev should be ~1, got {}",
            est.std_dev
        );
    }

    #[test]
    fn test_mc_estimator_estimate_vector() {
        let mut mc = MonteCarloEstimator::new(50, 0.95, 99).unwrap();
        let dim = 4;
        let estimates = mc.estimate_vector(dim, |noise| vec![noise; dim]).unwrap();
        assert_eq!(estimates.len(), dim);
    }

    // ---- CalibrationMetrics ------------------------------------------------

    #[test]
    fn test_calibration_metrics_compute_perfect() {
        // When predicted probability equals true label (perfect, deterministic)
        // ECE should be 0 (or very close).
        let predicted: Vec<f64> = vec![1.0; 100];
        let labels: Vec<u8> = vec![1u8; 100];
        let metrics = CalibrationMetrics::compute(&predicted, &labels, 10).unwrap();
        assert!(
            metrics.ece < 1e-9,
            "ECE should be 0 for perfect preds, got {}",
            metrics.ece
        );
    }

    #[test]
    fn test_calibration_metrics_compute_uniform() {
        // Uniformly random predictions → ECE > 0
        let mut rng = SimpleUncertaintyRng::new(42);
        let n = 200;
        let predicted: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let labels: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let metrics = CalibrationMetrics::compute(&predicted, &labels, 10).unwrap();
        // ECE might be small by chance but the test just checks it is non-negative and computable
        assert!(metrics.ece >= 0.0);
        assert!(metrics.num_bins == 10);
    }

    #[test]
    fn test_calibration_metrics_bins() {
        let predicted = vec![0.1, 0.5, 0.9];
        let labels = vec![0u8, 1, 1];
        let metrics = CalibrationMetrics::compute(&predicted, &labels, 5).unwrap();
        assert_eq!(metrics.num_bins, 5);
        assert_eq!(metrics.bin_stats.len(), 5);
    }

    #[test]
    fn test_calibration_is_well_calibrated() {
        // Perfect calibration
        let predicted = vec![1.0_f64; 50];
        let labels = vec![1u8; 50];
        let metrics = CalibrationMetrics::compute(&predicted, &labels, 5).unwrap();
        assert!(metrics.is_well_calibrated(0.01));
    }

    // ---- Temperature scaling -----------------------------------------------

    #[test]
    fn test_temperature_scale_identity() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = temperature_scale(&logits, 1.0);
        let direct = {
            let exps: Vec<f64> = logits.iter().map(|l| l.exp()).collect();
            let s: f64 = exps.iter().sum();
            exps.iter().map(|e| e / s).collect::<Vec<_>>()
        };
        for (a, b) in scaled.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-9, "a={a} b={b}");
        }
    }

    #[test]
    fn test_temperature_scale_high_temp() {
        // High temperature should push towards uniform distribution
        let logits = vec![10.0, 0.0, 0.0];
        let high_t = temperature_scale(&logits, 100.0);
        // Each should be close to 1/3
        for p in &high_t {
            assert!((p - 1.0 / 3.0).abs() < 0.1, "p={p}");
        }
    }

    #[test]
    fn test_find_optimal_temperature() {
        let logits: Vec<f64> = vec![2.0, -1.0, 0.5, -2.0, 1.0];
        let labels: Vec<u8> = vec![1, 0, 1, 0, 1];
        let temps: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0];
        let opt_t = find_optimal_temperature(&logits, &labels, &temps).unwrap();
        assert!(temps.contains(&opt_t), "optimal temp not in candidates");
    }

    // ---- PredictionInterval ------------------------------------------------

    #[test]
    fn test_prediction_interval_basic() {
        let lower = vec![0.0, 1.0, 2.0];
        let upper = vec![1.0, 2.0, 3.0];
        let pi = PredictionInterval::from_quantile_predictions(lower, upper, None).unwrap();
        assert_eq!(pi.predictions.len(), 3);
        assert!((pi.avg_width - 1.0).abs() < 1e-9);
        let s = pi.summary();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_prediction_interval_coverage() {
        let lower = vec![0.0, 1.0, 2.0, 3.0];
        let upper = vec![1.0, 2.0, 3.0, 4.0];
        // All actuals inside the interval
        let actuals = vec![0.5, 1.5, 2.5, 3.5];
        let pi =
            PredictionInterval::from_quantile_predictions(lower, upper, Some(&actuals)).unwrap();
        assert!((pi.coverage - 1.0).abs() < 1e-9, "coverage={}", pi.coverage);
    }

    #[test]
    fn test_prediction_interval_partial_coverage() {
        let lower = vec![0.0, 0.0, 0.0, 0.0];
        let upper = vec![1.0, 1.0, 1.0, 1.0];
        // Half inside, half outside
        let actuals = vec![0.5, 0.5, 2.0, 2.0];
        let pi =
            PredictionInterval::from_quantile_predictions(lower, upper, Some(&actuals)).unwrap();
        assert!((pi.coverage - 0.5).abs() < 1e-9, "coverage={}", pi.coverage);
    }
}
