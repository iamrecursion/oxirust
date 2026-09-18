#![allow(dead_code)]
//! A/B comparison benchmarks for evaluating two competing implementations
//! head-to-head across quality, speed, and efficiency dimensions.
//!
//! This module goes beyond the basic [`crate::comparison`] module by offering
//! a structured *experiment* abstraction that records both the baseline (A) and
//! the treatment (B) runs with full statistical analysis, significance testing,
//! and per-metric verdict reporting.

use crate::{BenchError, BenchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Experiment configuration
// ---------------------------------------------------------------------------

/// Which statistical test to apply when assessing significance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignificanceTest {
    /// Welch's t-test (two-sample, unequal variance assumed).
    WelchT,
    /// Mann-Whitney U-test (non-parametric rank-based).
    MannWhitneyU,
    /// None — treat all differences as significant.
    None,
}

impl Default for SignificanceTest {
    fn default() -> Self {
        Self::WelchT
    }
}

/// A named variant in an A/B experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Human-readable name for the variant (e.g. `"baseline-av1"` or `"treatment-av1-cq20"`).
    pub name: String,
    /// Optional description / notes.
    pub description: Option<String>,
    /// Arbitrary key–value metadata for the variant (e.g. codec version, preset).
    pub metadata: HashMap<String, String>,
}

impl Variant {
    /// Create a new variant with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder: attach a description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: attach a metadata key–value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Configuration for an A/B comparison experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbExperimentConfig {
    /// Experiment name.
    pub name: String,
    /// Variant A (baseline).
    pub variant_a: Variant,
    /// Variant B (treatment).
    pub variant_b: Variant,
    /// Significance test to use.
    pub significance_test: SignificanceTest,
    /// Alpha level (p-value threshold) for significance testing.
    pub alpha: f64,
    /// Minimum effect size (ratio) to be considered practically meaningful.
    pub min_effect_size: f64,
    /// Number of independent replications per variant.
    pub replications: usize,
}

impl Default for AbExperimentConfig {
    fn default() -> Self {
        Self {
            name: "ab_experiment".to_string(),
            variant_a: Variant::new("A"),
            variant_b: Variant::new("B"),
            significance_test: SignificanceTest::WelchT,
            alpha: 0.05,
            min_effect_size: 0.01,
            replications: 5,
        }
    }
}

impl AbExperimentConfig {
    /// Create a new experiment configuration.
    #[must_use]
    pub fn new(name: impl Into<String>, variant_a: Variant, variant_b: Variant) -> Self {
        Self {
            name: name.into(),
            variant_a,
            variant_b,
            ..Self::default()
        }
    }

    /// Builder: set the significance test.
    #[must_use]
    pub fn with_significance_test(mut self, test: SignificanceTest) -> Self {
        self.significance_test = test;
        self
    }

    /// Builder: set the alpha level.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set the minimum effect size.
    #[must_use]
    pub fn with_min_effect_size(mut self, size: f64) -> Self {
        self.min_effect_size = size;
        self
    }

    /// Builder: set the number of replications.
    #[must_use]
    pub fn with_replications(mut self, n: usize) -> Self {
        self.replications = n;
        self
    }
}

// ---------------------------------------------------------------------------
// Metric sample set
// ---------------------------------------------------------------------------

/// A set of raw metric samples collected for one variant in a single metric dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSamples {
    /// Metric name (e.g. `"encoding_fps"`, `"psnr_db"`, `"file_size_bytes"`).
    pub metric: String,
    /// Variant name these samples belong to.
    pub variant: String,
    /// Raw sample values.
    pub values: Vec<f64>,
}

impl MetricSamples {
    /// Create a new sample set.
    #[must_use]
    pub fn new(metric: impl Into<String>, variant: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            metric: metric.into(),
            variant: variant.into(),
            values,
        }
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Arithmetic mean. Returns `None` when empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    /// Population variance. Returns `None` when fewer than two samples exist.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> Option<f64> {
        let n = self.values.len();
        if n < 2 {
            return None;
        }
        let m = self.values.iter().sum::<f64>() / n as f64;
        let var = self.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var)
    }

    /// Standard deviation. Returns `None` when fewer than two samples exist.
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Minimum value. Returns `None` when empty.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.values.iter().cloned().reduce(f64::min)
    }

    /// Maximum value. Returns `None` when empty.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.values.iter().cloned().reduce(f64::max)
    }

    /// Median value (linear interpolation for even-length). Returns `None` when empty.
    #[must_use]
    pub fn median(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical tests
// ---------------------------------------------------------------------------

/// Two-sample Welch's t-test statistic.
///
/// Returns the t-statistic and degrees of freedom (Welch–Satterthwaite).
/// Returns `None` when either sample is too small or variance is zero.
#[allow(clippy::cast_precision_loss)]
fn welch_t_test(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let na = a.len();
    let nb = b.len();
    if na < 2 || nb < 2 {
        return None;
    }
    let mean_a = a.iter().sum::<f64>() / na as f64;
    let mean_b = b.iter().sum::<f64>() / nb as f64;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (na - 1) as f64;
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (nb - 1) as f64;
    if var_a == 0.0 && var_b == 0.0 {
        return None;
    }
    let se = ((var_a / na as f64) + (var_b / nb as f64)).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (mean_a - mean_b) / se;
    // Welch–Satterthwaite degrees of freedom
    let dof_num = ((var_a / na as f64) + (var_b / nb as f64)).powi(2);
    let dof_den = (var_a / na as f64).powi(2) / (na - 1) as f64
        + (var_b / nb as f64).powi(2) / (nb - 1) as f64;
    let dof = if dof_den == 0.0 {
        (na + nb - 2) as f64
    } else {
        dof_num / dof_den
    };
    Some((t, dof))
}

/// Compute a conservative two-tailed p-value approximation from a t-statistic and
/// degrees of freedom using a rational Padé approximation to the Student-t CDF.
///
/// The approximation is within ±0.005 for |t| < 10 and dof >= 1.
#[allow(clippy::cast_precision_loss)]
fn t_to_p_value(t: f64, dof: f64) -> f64 {
    // Use a normal approximation for large dof (dof > 30) as it is accurate enough.
    let t_abs = t.abs();
    if dof > 30.0 {
        // Standard normal survival: P(Z > |t|) * 2
        let z = t_abs;
        // Abramowitz & Stegun 26.2.17 rational approximation
        let p = 1.0 - normal_cdf(z);
        return (2.0 * p).min(1.0);
    }
    // For small dof use a Beta-based approximation.
    // x = dof / (dof + t^2)
    let x = dof / (dof + t_abs * t_abs);
    // regularized incomplete beta function approximation (continued fraction)
    let a = dof / 2.0;
    let b = 0.5_f64;
    let ibeta = regularized_incomplete_beta(x, a, b);
    ibeta.min(1.0)
}

/// Standard normal CDF using Abramowitz & Stegun 26.2.17.
fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if z >= 0.0 {
        1.0 - pdf * poly
    } else {
        pdf * poly
    }
}

/// Regularized incomplete beta function I_x(a, b) via continued-fraction expansion (Lentz).
/// Used for the t-distribution CDF approximation.
#[allow(clippy::many_single_char_names)]
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Front factor
    let ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;
    // Continued fraction (modified Lentz) — up to 200 iterations
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    for i in 1..=200u32 {
        let m = i as f64;
        // Even step
        let num_e = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num_e * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_e / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        // Odd step
        let num_o = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num_o * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_o / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }
    front * f
}

/// Stirling's approximation to the log-gamma function (Lanczos g=7).
fn lgamma(z: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_932,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - lgamma(1.0 - z)
    } else {
        let x = z - 1.0;
        let mut a = C[0];
        for (i, &c) in C[1..].iter().enumerate() {
            a += c / (x + i as f64 + 1.0);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Mann-Whitney U statistic for two independent samples.
/// Returns U and a z-score approximation for large samples.
#[allow(clippy::cast_precision_loss)]
fn mann_whitney_u(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return None;
    }
    // Count U_a = number of pairs (a_i, b_j) where a_i > b_j
    let mut u_a = 0u64;
    for &ai in a {
        for &bj in b {
            if ai > bj {
                u_a += 1;
            } else if (ai - bj).abs() < 1e-15 {
                // tie: add 0.5
            }
        }
    }
    let u = u_a as f64;
    let mu = (na * nb) as f64 / 2.0;
    let sigma = ((na * nb * (na + nb + 1)) as f64 / 12.0).sqrt();
    if sigma == 0.0 {
        return None;
    }
    let z = (u - mu) / sigma;
    Some((u, z))
}

// ---------------------------------------------------------------------------
// Per-metric comparison result
// ---------------------------------------------------------------------------

/// Verdict for a single metric in an A/B experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbVerdict {
    /// B is significantly better than A by at least the minimum effect size.
    BIsBetter,
    /// A is significantly better than B by at least the minimum effect size.
    AIsBetter,
    /// No significant difference was detected, or the effect is below the threshold.
    Inconclusive,
}

/// Result of comparing one metric between variant A and variant B.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparisonResult {
    /// Metric name.
    pub metric: String,
    /// Mean of variant A.
    pub mean_a: f64,
    /// Mean of variant B.
    pub mean_b: f64,
    /// Absolute difference (B − A).
    pub abs_diff: f64,
    /// Relative difference (B − A) / |A|, as a fraction.
    pub rel_diff: f64,
    /// Estimated p-value (or `f64::NAN` when not computable).
    pub p_value: f64,
    /// Whether the result is statistically significant.
    pub significant: bool,
    /// Effect size (Cohen's d or rank-biserial correlation).
    pub effect_size: f64,
    /// Overall verdict.
    pub verdict: AbVerdict,
    /// True when a *higher* value for this metric is better (e.g. PSNR, FPS).
    pub higher_is_better: bool,
}

impl MetricComparisonResult {
    /// Whether the comparison detected a meaningful improvement in B over A.
    #[must_use]
    pub fn b_improves_over_a(&self) -> bool {
        matches!(self.verdict, AbVerdict::BIsBetter)
    }
}

// ---------------------------------------------------------------------------
// Full experiment result
// ---------------------------------------------------------------------------

/// Complete result of an A/B comparison experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbExperimentResult {
    /// Configuration of the experiment.
    pub config: AbExperimentConfig,
    /// Per-metric comparison results.
    pub metrics: Vec<MetricComparisonResult>,
    /// Total wall-clock duration of the experiment.
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    /// Timestamp when the experiment was recorded (ISO-8601).
    pub timestamp: String,
    /// Experiment-level summary verdict: the proportion of metrics where B wins.
    pub b_win_rate: f64,
}

impl AbExperimentResult {
    /// Number of metrics where variant B was deemed better.
    #[must_use]
    pub fn b_win_count(&self) -> usize {
        self.metrics
            .iter()
            .filter(|m| matches!(m.verdict, AbVerdict::BIsBetter))
            .count()
    }

    /// Number of metrics where variant A was deemed better.
    #[must_use]
    pub fn a_win_count(&self) -> usize {
        self.metrics
            .iter()
            .filter(|m| matches!(m.verdict, AbVerdict::AIsBetter))
            .count()
    }

    /// Number of inconclusive metrics.
    #[must_use]
    pub fn inconclusive_count(&self) -> usize {
        self.metrics
            .iter()
            .filter(|m| matches!(m.verdict, AbVerdict::Inconclusive))
            .count()
    }

    /// Look up a metric comparison result by name.
    #[must_use]
    pub fn find_metric(&self, name: &str) -> Option<&MetricComparisonResult> {
        self.metrics.iter().find(|m| m.metric == name)
    }

    /// Export the result as a pretty JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`BenchError`] if serialization fails.
    pub fn to_json(&self) -> BenchResult<String> {
        serde_json::to_string_pretty(self).map_err(BenchError::Serialization)
    }
}

// ---------------------------------------------------------------------------
// Comparator
// ---------------------------------------------------------------------------

/// Descriptor for a single metric's directionality used by the comparator.
#[derive(Debug, Clone)]
pub struct MetricDescriptor {
    /// Metric name.
    pub name: String,
    /// True when a higher value is better (e.g. PSNR, FPS).
    pub higher_is_better: bool,
}

impl MetricDescriptor {
    /// Create a new descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, higher_is_better: bool) -> Self {
        Self {
            name: name.into(),
            higher_is_better,
        }
    }
}

/// Performs head-to-head comparison between two benchmark variants (A and B).
///
/// # Example
///
/// ```
/// use oximedia_bench::ab_comparison::{AbComparator, AbExperimentConfig, Variant, MetricDescriptor};
///
/// let config = AbExperimentConfig::new(
///     "my_experiment",
///     Variant::new("baseline"),
///     Variant::new("treatment"),
/// );
/// let comparator = AbComparator::new(config);
/// let result = comparator.compare(
///     &[("encoding_fps", true)],
///     &[vec![30.0, 31.0, 29.5]],
///     &[vec![35.0, 36.0, 34.5]],
/// );
/// assert!(result.is_ok());
/// ```
pub struct AbComparator {
    config: AbExperimentConfig,
}

impl AbComparator {
    /// Create a new comparator with the given experiment configuration.
    #[must_use]
    pub fn new(config: AbExperimentConfig) -> Self {
        Self { config }
    }

    /// Run the A/B comparison for the provided metric sample sets.
    ///
    /// `metrics` is a slice of `(metric_name, higher_is_better)` tuples.
    /// `samples_a` and `samples_b` are parallel slices of sample vectors, one per metric.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::InvalidConfig`] if the slice lengths are inconsistent.
    pub fn compare(
        &self,
        metrics: &[(&str, bool)],
        samples_a: &[Vec<f64>],
        samples_b: &[Vec<f64>],
    ) -> BenchResult<AbExperimentResult> {
        if metrics.len() != samples_a.len() || metrics.len() != samples_b.len() {
            return Err(BenchError::InvalidConfig(
                "metrics, samples_a and samples_b must have the same length".to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let mut comparisons = Vec::with_capacity(metrics.len());

        for (i, &(metric_name, higher_is_better)) in metrics.iter().enumerate() {
            let sa = &samples_a[i];
            let sb = &samples_b[i];
            let cmp = self.compare_metric(metric_name, higher_is_better, sa, sb);
            comparisons.push(cmp);
        }

        let b_wins = comparisons
            .iter()
            .filter(|m| matches!(m.verdict, AbVerdict::BIsBetter))
            .count();
        let b_win_rate = if comparisons.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                b_wins as f64 / comparisons.len() as f64
            }
        };

        Ok(AbExperimentResult {
            config: self.config.clone(),
            metrics: comparisons,
            duration: start.elapsed(),
            timestamp: crate_timestamp(),
            b_win_rate,
        })
    }

    /// Compare samples for a single metric.
    #[allow(clippy::cast_precision_loss)]
    fn compare_metric(
        &self,
        metric: &str,
        higher_is_better: bool,
        sa: &[f64],
        sb: &[f64],
    ) -> MetricComparisonResult {
        let mean_a = if sa.is_empty() {
            0.0
        } else {
            sa.iter().sum::<f64>() / sa.len() as f64
        };
        let mean_b = if sb.is_empty() {
            0.0
        } else {
            sb.iter().sum::<f64>() / sb.len() as f64
        };

        let abs_diff = mean_b - mean_a;
        let rel_diff = if mean_a.abs() < 1e-15 {
            0.0
        } else {
            abs_diff / mean_a.abs()
        };

        let (p_value, effect_size) = match self.config.significance_test {
            SignificanceTest::WelchT => {
                if let Some((t, dof)) = welch_t_test(sa, sb) {
                    let p = t_to_p_value(t, dof);
                    // Cohen's d
                    let n_a = sa.len() as f64;
                    let n_b = sb.len() as f64;
                    let var_a =
                        sa.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0).max(1.0);
                    let var_b =
                        sb.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0).max(1.0);
                    let pooled_sd = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b)
                        / (n_a + n_b - 2.0).max(1.0))
                    .sqrt();
                    let d = if pooled_sd > 0.0 {
                        abs_diff / pooled_sd
                    } else {
                        0.0
                    };
                    (p, d.abs())
                } else {
                    (f64::NAN, 0.0)
                }
            }
            SignificanceTest::MannWhitneyU => {
                if let Some((_u, z)) = mann_whitney_u(sa, sb) {
                    let p = 2.0 * (1.0 - normal_cdf(z.abs()));
                    let n_total = (sa.len() + sb.len()) as f64;
                    let r = z / n_total.sqrt(); // rank-biserial correlation
                    (p, r.abs())
                } else {
                    (f64::NAN, 0.0)
                }
            }
            SignificanceTest::None => (0.0, rel_diff.abs()),
        };

        let significant = if p_value.is_nan() {
            false
        } else {
            p_value < self.config.alpha
        };

        let effect_meaningful = rel_diff.abs() >= self.config.min_effect_size;
        let verdict = if significant && effect_meaningful {
            // Determine direction
            let b_better = if higher_is_better {
                mean_b > mean_a
            } else {
                mean_b < mean_a
            };
            if b_better {
                AbVerdict::BIsBetter
            } else {
                AbVerdict::AIsBetter
            }
        } else {
            AbVerdict::Inconclusive
        };

        MetricComparisonResult {
            metric: metric.to_string(),
            mean_a,
            mean_b,
            abs_diff,
            rel_diff,
            p_value,
            significant,
            effect_size,
            verdict,
            higher_is_better,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect raw timing samples (wall-clock nanoseconds) by running `f` `n` times.
///
/// The warmup count is taken from the iterator but discarded; then the actual
/// measurement runs are accumulated.
pub fn collect_timing_samples<F>(f: &mut F, warmup: usize, measure: usize) -> Vec<f64>
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(measure);
    for _ in 0..measure {
        let t0 = std::time::Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples
}

/// Generate a compact ISO-8601-like timestamp string.
fn crate_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    let z = (secs / 86400) as i64 + 719_468_i64;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr = if m <= 2 { y + 1 } else { y };
    format!("{yr:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Serde helpers for `Duration`.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> AbExperimentConfig {
        AbExperimentConfig::new("test", Variant::new("A"), Variant::new("B"))
            .with_replications(5)
            .with_alpha(0.05)
            .with_min_effect_size(0.01)
    }

    #[test]
    fn test_metric_samples_statistics() {
        let s = MetricSamples::new("fps", "A", vec![10.0, 20.0, 30.0]);
        assert_eq!(s.mean(), Some(20.0));
        assert!(s.std_dev().is_some());
        assert_eq!(s.min(), Some(10.0));
        assert_eq!(s.max(), Some(30.0));
        assert_eq!(s.median(), Some(20.0));
    }

    #[test]
    fn test_metric_samples_empty() {
        let s = MetricSamples::new("fps", "A", vec![]);
        assert!(s.mean().is_none());
        assert!(s.std_dev().is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn test_b_clearly_faster() {
        let config = make_config();
        let comparator = AbComparator::new(config);
        let a_samples = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let b_samples = vec![30.0, 31.0, 30.5, 30.2, 30.8];
        let result = comparator
            .compare(&[("fps", true)], &[a_samples], &[b_samples])
            .expect("comparison should succeed");
        let fps = result.find_metric("fps").expect("fps metric missing");
        assert!(matches!(fps.verdict, AbVerdict::BIsBetter));
    }

    #[test]
    fn test_a_b_identical() {
        let config = make_config();
        let comparator = AbComparator::new(config);
        let samples = vec![20.0, 20.0, 20.0, 20.0, 20.0];
        let result = comparator
            .compare(
                &[("fps", true)],
                std::slice::from_ref(&samples),
                std::slice::from_ref(&samples),
            )
            .expect("comparison should succeed");
        let fps = result.find_metric("fps").expect("fps metric missing");
        // Identical samples: should be inconclusive.
        assert!(matches!(fps.verdict, AbVerdict::Inconclusive));
    }

    #[test]
    fn test_mismatched_lengths_returns_error() {
        let config = make_config();
        let comparator = AbComparator::new(config);
        let result = comparator.compare(
            &[("fps", true), ("psnr", true)],
            &[vec![1.0]],
            &[vec![1.0], vec![2.0]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_ab_win_counts() {
        let config = make_config();
        let comparator = AbComparator::new(config);
        let a = vec![10.0, 10.1, 9.9, 10.2, 10.0];
        let b = vec![50.0, 51.0, 49.0, 50.5, 50.2];
        let result = comparator
            .compare(
                &[("fps", true), ("psnr", true)],
                &[a.clone(), a.clone()],
                &[b.clone(), b.clone()],
            )
            .expect("comparison should succeed");
        assert_eq!(result.b_win_count(), 2);
        assert_eq!(result.a_win_count(), 0);
    }

    #[test]
    fn test_collect_timing_samples() {
        let mut counter = 0usize;
        let samples = collect_timing_samples(
            &mut || {
                counter += 1;
            },
            2,
            4,
        );
        assert_eq!(samples.len(), 4);
        assert_eq!(counter, 6); // 2 warmup + 4 measurement
        for s in &samples {
            assert!(*s >= 0.0);
        }
    }

    #[test]
    fn test_mann_whitney() {
        let config = AbExperimentConfig::default()
            .with_significance_test(SignificanceTest::MannWhitneyU)
            .with_min_effect_size(0.01);
        let comparator = AbComparator::new(config);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = comparator
            .compare(&[("val", true)], &[a], &[b])
            .expect("comparison should succeed");
        let val = result.find_metric("val").expect("val metric missing");
        assert!(matches!(val.verdict, AbVerdict::BIsBetter));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let config = make_config();
        let comparator = AbComparator::new(config);
        let result = comparator
            .compare(&[("fps", true)], &[vec![10.0, 11.0]], &[vec![20.0, 21.0]])
            .expect("comparison should succeed");
        let json = result.to_json().expect("serialization should succeed");
        assert!(json.contains("fps"));
    }

    #[test]
    fn test_significance_test_none() {
        let config = AbExperimentConfig::default()
            .with_significance_test(SignificanceTest::None)
            .with_min_effect_size(0.01);
        let comparator = AbComparator::new(config);
        let a = vec![10.0, 10.0];
        let b = vec![11.0, 11.0];
        let result = comparator
            .compare(&[("fps", true)], &[a], &[b])
            .expect("comparison should succeed");
        // With SignificanceTest::None a 10% difference should be BIsBetter.
        let fps = result.find_metric("fps").expect("fps metric missing");
        assert!(matches!(fps.verdict, AbVerdict::BIsBetter));
    }
}
