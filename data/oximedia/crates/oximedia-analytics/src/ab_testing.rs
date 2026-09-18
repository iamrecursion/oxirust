//! A/B testing framework for media content experiments.
//!
//! Provides variant assignment (deterministic FNV-1a or random), per-variant
//! metric collection, statistical significance testing, and winner selection.

use std::collections::HashMap;

use crate::error::AnalyticsError;

// ─── Experiment model ─────────────────────────────────────────────────────────

/// One treatment arm in an experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub id: String,
    pub name: String,
    /// Relative weight used during assignment (will be normalised to sum=1.0).
    pub allocation_weight: f32,
}

/// A configured A/B experiment.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub variants: Vec<Variant>,
    /// Wall-clock start of the experiment (Unix epoch ms).
    pub start_ms: i64,
    /// Optional end time; `None` means the experiment is still running.
    pub end_ms: Option<i64>,
    /// Minimum sample size per variant before results are considered reliable.
    pub min_sample_size: u32,
}

impl Experiment {
    /// Sum of all variant allocation weights (for normalisation).
    fn weight_sum(&self) -> f32 {
        self.variants.iter().map(|v| v.allocation_weight).sum()
    }
}

/// How to assign a user to a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentMethod {
    /// Hash the user ID deterministically — the same user always gets the
    /// same variant.
    Deterministic,
    /// Not truly random in this context; provided for API completeness.  Uses
    /// the same FNV-1a path because we have no PRNG state here.
    Random,
}

/// FNV-1a 32-bit hash of a byte slice.
pub(crate) fn fnv1a_32(data: &[u8]) -> u32 {
    const FNV_OFFSET: u32 = 2_166_136_261;
    const FNV_PRIME: u32 = 16_777_619;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Assign a user to a variant in the given experiment.
///
/// For `Deterministic` (and `Random` — see note on `AssignmentMethod`), the
/// assignment is derived from the FNV-1a hash of the `user_id`, which means
/// the same user always lands in the same bucket.  Variants with higher
/// `allocation_weight` receive proportionally more users.
///
/// Returns an error if the experiment has no variants or all weights are zero.
pub fn assign_variant<'e>(
    experiment: &'e Experiment,
    user_id: &str,
    _method: AssignmentMethod,
) -> Result<&'e Variant, AnalyticsError> {
    if experiment.variants.is_empty() {
        return Err(AnalyticsError::NoVariants(experiment.id.clone()));
    }

    let weight_sum = experiment.weight_sum();
    if weight_sum <= 0.0 {
        return Err(AnalyticsError::InvalidWeights(experiment.id.clone()));
    }

    let hash = fnv1a_32(user_id.as_bytes());
    // Map hash to [0, weight_sum).
    // Use f64 to retain precision for large weight sums.
    let pos = (hash as f64 / u32::MAX as f64) * weight_sum as f64;

    let mut cumulative = 0.0f64;
    for variant in &experiment.variants {
        cumulative += variant.allocation_weight as f64;
        if pos < cumulative {
            return Ok(variant);
        }
    }

    // Fallback: return the last variant (handles floating-point edge cases).
    // Safety: this line is only reachable when `experiment.variants` is non-empty
    // (checked at the beginning of the function).  `last()` therefore always
    // returns `Some`.
    experiment
        .variants
        .last()
        .ok_or_else(|| AnalyticsError::NoVariants(experiment.id.clone()))
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Collected metrics for a single experiment variant.
#[derive(Debug, Clone, Default)]
pub struct VariantMetrics {
    pub variant_id: String,
    pub impressions: u32,
    pub clicks: u32,
    pub conversions: u32,
    /// Total watch time (ms) summed across all impressions.
    pub watch_duration_sum_ms: u64,
    /// Number of sessions that completed the content.
    pub completion_count: u32,
}

impl VariantMetrics {
    pub fn new(variant_id: impl Into<String>) -> Self {
        Self {
            variant_id: variant_id.into(),
            ..Default::default()
        }
    }
}

/// Aggregate experiment results keyed by variant ID.
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    pub experiment: Experiment,
    pub variant_metrics: HashMap<String, VariantMetrics>,
}

impl ExperimentResults {
    pub fn new(experiment: Experiment) -> Self {
        let mut variant_metrics = HashMap::new();
        for variant in &experiment.variants {
            variant_metrics.insert(variant.id.clone(), VariantMetrics::new(variant.id.clone()));
        }
        Self {
            experiment,
            variant_metrics,
        }
    }

    /// Record an impression for a variant.
    pub fn record_impression(&mut self, variant_id: &str) {
        if let Some(m) = self.variant_metrics.get_mut(variant_id) {
            m.impressions += 1;
        }
    }

    /// Record a click for a variant.
    pub fn record_click(&mut self, variant_id: &str) {
        if let Some(m) = self.variant_metrics.get_mut(variant_id) {
            m.clicks += 1;
        }
    }

    /// Record a conversion for a variant.
    pub fn record_conversion(&mut self, variant_id: &str) {
        if let Some(m) = self.variant_metrics.get_mut(variant_id) {
            m.conversions += 1;
        }
    }

    /// Record a completed view for a variant.
    pub fn record_completion(&mut self, variant_id: &str, watch_duration_ms: u64) {
        if let Some(m) = self.variant_metrics.get_mut(variant_id) {
            m.completion_count += 1;
            m.watch_duration_sum_ms += watch_duration_ms;
        }
    }

    /// Record a watch session (non-completing) for a variant.
    pub fn record_watch(&mut self, variant_id: &str, watch_duration_ms: u64) {
        if let Some(m) = self.variant_metrics.get_mut(variant_id) {
            m.watch_duration_sum_ms += watch_duration_ms;
        }
    }
}

// ─── Rate helpers ─────────────────────────────────────────────────────────────

/// Click-through rate: `clicks / impressions` (0.0 if no impressions).
pub fn click_through_rate(metrics: &VariantMetrics) -> f32 {
    if metrics.impressions == 0 {
        return 0.0;
    }
    metrics.clicks as f32 / metrics.impressions as f32
}

/// Conversion rate: `conversions / impressions` (0.0 if no impressions).
pub fn conversion_rate(metrics: &VariantMetrics) -> f32 {
    if metrics.impressions == 0 {
        return 0.0;
    }
    metrics.conversions as f32 / metrics.impressions as f32
}

/// Average watch duration per impression in milliseconds.
pub fn average_watch_duration(metrics: &VariantMetrics) -> f32 {
    if metrics.impressions == 0 {
        return 0.0;
    }
    metrics.watch_duration_sum_ms as f32 / metrics.impressions as f32
}

/// Completion rate: `completion_count / impressions`.
pub fn completion_rate(metrics: &VariantMetrics) -> f32 {
    if metrics.impressions == 0 {
        return 0.0;
    }
    metrics.completion_count as f32 / metrics.impressions as f32
}

// ─── Statistical significance ─────────────────────────────────────────────────

/// Compute the two-proportion z-score for rates `p1` (from `n1` observations)
/// and `p2` (from `n2` observations).
///
/// `p1`, `p2` should be in [0, 1].  Returns `0.0` when either sample is empty
/// or when the pooled proportion is degenerate (0 or 1).
pub fn z_test(p1: f32, n1: u32, p2: f32, n2: u32) -> f32 {
    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    // Reconstruct integer counts for pooled proportion.
    let x1 = (p1 * n1 as f32).round() as u32;
    let x2 = (p2 * n2 as f32).round() as u32;

    let p_pool = (x1 + x2) as f32 / (n1 + n2) as f32;

    // If the pooled proportion is at the boundary the variance is 0 → undefined.
    if p_pool <= 0.0 || p_pool >= 1.0 {
        return 0.0;
    }

    let variance = p_pool * (1.0 - p_pool) * (1.0 / n1 as f32 + 1.0 / n2 as f32);
    if variance <= 0.0 {
        return 0.0;
    }

    (p1 - p2) / variance.sqrt()
}

/// Determine whether a z-score is statistically significant at the given
/// significance level `alpha`.
///
/// Supported `alpha` values: 0.05 (critical z = 1.96) and 0.01 (z = 2.576).
/// For all other values the function uses 1.96 as the critical value.
pub fn is_significant(z_score: f32, alpha: f32) -> bool {
    let critical_z = if (alpha - 0.01).abs() < 1e-6 {
        2.576
    } else {
        1.96
    };
    z_score.abs() >= critical_z
}

// ─── Winner selection ─────────────────────────────────────────────────────────

/// Supported optimisation metrics for winner selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimisationMetric {
    Ctr,
    Conversion,
    Completion,
    WatchDuration,
}

/// Return the variant ID that wins on the given metric, or `None` if there are
/// no variants with data.
///
/// `metric` is a string: `"ctr"`, `"conversion"`, `"completion"`, or
/// `"watch_duration"`.  Unrecognised strings fall back to `"ctr"`.
///
/// Uses a hardcoded significance level of α = 0.05 (critical z = 1.96).
/// Use [`winning_variant_with_alpha`] for a configurable significance level.
pub fn winning_variant<'r>(results: &'r ExperimentResults, metric: &str) -> Option<&'r str> {
    winning_variant_with_alpha(results, metric, 0.05)
}

/// Return the variant ID that wins on the given metric at the specified
/// significance level `alpha`, or `None` if no variant has data or if the
/// best-scoring variant does not beat the runner-up with statistical
/// significance at `alpha`.
///
/// The winner is the variant with the highest score on `metric`.  A winner
/// is only declared when the two-proportion z-test comparing the best variant
/// against every other variant with data yields a z-score that clears the
/// critical value `alpha_to_critical_z(alpha)`.  If the best variant does not
/// beat all others significantly, `None` is returned (no significant winner).
///
/// For experiments with a single variant that has data that variant is always
/// returned (no comparison is possible).
///
/// Supported `alpha` values: 0.10, 0.05, 0.01, 0.001.  Any other value
/// maps to the closest standard level via [`alpha_to_critical_z`].
pub fn winning_variant_with_alpha<'r>(
    results: &'r ExperimentResults,
    metric: &str,
    alpha: f32,
) -> Option<&'r str> {
    let opt_metric = match metric {
        "conversion" => OptimisationMetric::Conversion,
        "completion" => OptimisationMetric::Completion,
        "watch_duration" => OptimisationMetric::WatchDuration,
        _ => OptimisationMetric::Ctr,
    };

    let candidates: Vec<&VariantMetrics> = results
        .variant_metrics
        .values()
        .filter(|m| m.impressions > 0)
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Single-variant shortcut — no comparison possible.
    if candidates.len() == 1 {
        return Some(candidates[0].variant_id.as_str());
    }

    // Find the highest-scoring variant.
    let best = candidates.iter().copied().max_by(|a, b| {
        let score_a = variant_score(a, opt_metric);
        let score_b = variant_score(b, opt_metric);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    // Check that the best variant is significantly better than all others.
    let critical_z = alpha_to_critical_z(alpha);
    let best_score = variant_score(best, opt_metric);
    let best_n = best.impressions;

    for other in candidates.iter().copied() {
        if other.variant_id == best.variant_id {
            continue;
        }
        let other_score = variant_score(other, opt_metric);
        let other_n = other.impressions;

        // z > 0 means `best` outperforms `other`; we require |z| >= critical_z
        // and z in the correct direction (best_score >= other_score is already
        // guaranteed by max_by, so we just need z >= critical_z).
        let z = z_test(best_score, best_n, other_score, other_n);
        if z < critical_z {
            return None;
        }
    }

    Some(best.variant_id.as_str())
}

/// Convert an alpha significance level to a two-tailed critical z-value.
///
/// Supported levels: 0.10 → 1.645, 0.05 → 1.96, 0.01 → 2.576, 0.001 → 3.291.
/// Values outside these levels are clamped to the nearest supported level.
pub fn alpha_to_critical_z(alpha: f32) -> f32 {
    if alpha <= 0.001 {
        3.291
    } else if alpha <= 0.01 {
        2.576
    } else if alpha <= 0.05 {
        1.96
    } else {
        1.645
    }
}

fn variant_score(metrics: &VariantMetrics, opt: OptimisationMetric) -> f32 {
    match opt {
        OptimisationMetric::Ctr => click_through_rate(metrics),
        OptimisationMetric::Conversion => conversion_rate(metrics),
        OptimisationMetric::Completion => completion_rate(metrics),
        OptimisationMetric::WatchDuration => average_watch_duration(metrics),
    }
}

// ─── Bayesian A/B testing ─────────────────────────────────────────────────────

/// Result of a Bayesian A/B test comparison between two variants.
///
/// Uses Beta-Binomial conjugate updates to estimate the probability that
/// variant B outperforms variant A on the selected metric.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianAbResult {
    /// ID of the "control" variant (variant A).
    pub variant_a_id: String,
    /// ID of the "treatment" variant (variant B).
    pub variant_b_id: String,
    /// Estimated probability that variant B has a higher true rate than A.
    /// In [0.0, 1.0]; values > 0.95 are conventionally considered "significant".
    pub prob_b_beats_a: f64,
    /// Expected uplift: E\[`rate_B`\] − E\[`rate_A`\] using posterior means.
    pub expected_uplift: f64,
    /// Posterior mean of variant A's rate.
    pub posterior_mean_a: f64,
    /// Posterior mean of variant B's rate.
    pub posterior_mean_b: f64,
}

/// Compute a Bayesian A/B test for two variants.
///
/// Uses Beta-Binomial conjugacy with a non-informative Jeffreys prior
/// Beta(0.5, 0.5) for both variants.  The probability that B beats A is
/// approximated via Monte Carlo sampling with the given `rng_seed` and
/// `num_samples`.
///
/// `metric` selects which count to treat as "successes":
/// * `"ctr"` or `"click"` — clicks / impressions
/// * `"conversion"` — conversions / impressions
/// * `"completion"` — completion_count / impressions
///
/// Returns an error if either variant has zero impressions.
pub fn bayesian_winner(
    results: &ExperimentResults,
    variant_a_id: &str,
    variant_b_id: &str,
    metric: &str,
    num_samples: usize,
    rng_seed: u64,
) -> Result<BayesianAbResult, crate::error::AnalyticsError> {
    let ma = results.variant_metrics.get(variant_a_id).ok_or_else(|| {
        crate::error::AnalyticsError::ConfigError(format!("variant '{variant_a_id}' not found"))
    })?;
    let mb = results.variant_metrics.get(variant_b_id).ok_or_else(|| {
        crate::error::AnalyticsError::ConfigError(format!("variant '{variant_b_id}' not found"))
    })?;

    if ma.impressions == 0 || mb.impressions == 0 {
        return Err(crate::error::AnalyticsError::InsufficientData(
            "both variants require at least one impression for Bayesian test".to_string(),
        ));
    }

    let (successes_a, n_a) = extract_metric_counts(ma, metric);
    let (successes_b, n_b) = extract_metric_counts(mb, metric);

    // Jeffreys prior: Beta(0.5, 0.5).
    let alpha_a = 0.5 + successes_a as f64;
    let beta_a = 0.5 + (n_a - successes_a) as f64;
    let alpha_b = 0.5 + successes_b as f64;
    let beta_b = 0.5 + (n_b - successes_b) as f64;

    let posterior_mean_a = alpha_a / (alpha_a + beta_a);
    let posterior_mean_b = alpha_b / (alpha_b + beta_b);

    // Monte Carlo estimate: P(rate_B > rate_A).
    let prob_b_beats_a =
        monte_carlo_prob_b_beats_a(alpha_a, beta_a, alpha_b, beta_b, num_samples, rng_seed);

    Ok(BayesianAbResult {
        variant_a_id: variant_a_id.to_string(),
        variant_b_id: variant_b_id.to_string(),
        prob_b_beats_a,
        expected_uplift: posterior_mean_b - posterior_mean_a,
        posterior_mean_a,
        posterior_mean_b,
    })
}

/// Extract (successes, total) counts from a `VariantMetrics` for the named metric.
fn extract_metric_counts(m: &VariantMetrics, metric: &str) -> (u32, u32) {
    match metric {
        "conversion" => (m.conversions, m.impressions),
        "completion" => (m.completion_count, m.impressions),
        _ => (m.clicks, m.impressions), // "ctr" / "click" / default
    }
}

/// Monte Carlo estimate of P(Beta(α_B, β_B) > Beta(α_A, β_A)).
///
/// Uses an xoshiro256** PRNG seeded from `rng_seed` and draws `num_samples`
/// pairs of Beta samples.
fn monte_carlo_prob_b_beats_a(
    alpha_a: f64,
    beta_a: f64,
    alpha_b: f64,
    beta_b: f64,
    num_samples: usize,
    seed: u64,
) -> f64 {
    if num_samples == 0 {
        return 0.5;
    }
    let mut rng = Xoshiro256 {
        state: [
            seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
            seed ^ 0xdead_beef_cafe_babe,
            seed.rotate_left(17).wrapping_add(0x0123_4567_89ab_cdef),
        ],
    };

    let mut b_wins = 0u64;
    for _ in 0..num_samples {
        let sa = sample_beta(&mut rng, alpha_a, beta_a);
        let sb = sample_beta(&mut rng, alpha_b, beta_b);
        if sb > sa {
            b_wins += 1;
        }
    }
    b_wins as f64 / num_samples as f64
}

/// Minimal xoshiro256** state for Bayesian sampling.
struct Xoshiro256 {
    state: [u64; 4],
}

impl Xoshiro256 {
    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.state;
        let result = s1.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s1 << 17;
        self.state[2] ^= s0;
        self.state[3] ^= s1;
        self.state[1] ^= s2;
        self.state[0] ^= s3;
        self.state[2] ^= t;
        self.state[3] = s3.rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sample one value from N(0,1) using the Box-Muller transform.
///
/// Consumes two uniform samples from `rng` and returns one standard-normal variate.
fn sample_normal(rng: &mut Xoshiro256) -> f64 {
    // Box-Muller: use a small epsilon to avoid log(0).
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Sample one value from Gamma(shape, 1) using the Marsaglia-Tsang (2000) method.
///
/// Valid for all `shape` > 0.  For `shape` < 1 the boost trick
/// `Gamma(shape+1, 1) * U^(1/shape)` is applied.
fn sample_gamma(rng: &mut Xoshiro256, shape: f64) -> f64 {
    debug_assert!(shape > 0.0, "Gamma shape must be positive");

    if shape < 1.0 {
        // Boost: Gamma(shape) = Gamma(shape+1) * U^(1/shape).
        let g = sample_gamma(rng, shape + 1.0);
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        return g * u.powf(1.0 / shape);
    }

    // Marsaglia-Tsang squeeze-and-accept for shape >= 1.
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let vc = 1.0 + c * x;
        if vc <= 0.0 {
            continue;
        }
        let v = vc * vc * vc;
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        // Squeeze test (avoids log most of the time).
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        // Full acceptance test.
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample one value from Beta(alpha, beta) using the ratio-of-Gamma-samples method.
///
/// `Beta(α, β) = X / (X + Y)` where `X ~ Gamma(α, 1)` and `Y ~ Gamma(β, 1)`.
/// This is numerically stable and correct for all positive α, β.
fn sample_beta(rng: &mut Xoshiro256, alpha: f64, beta: f64) -> f64 {
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    let s = x + y;
    if s <= 0.0 {
        // Degenerate: fall back to posterior mean.
        return alpha / (alpha + beta);
    }
    x / s
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_variant_experiment() -> Experiment {
        Experiment {
            id: "exp1".to_string(),
            name: "Thumbnail Test".to_string(),
            variants: vec![
                Variant {
                    id: "A".to_string(),
                    name: "Control".to_string(),
                    allocation_weight: 1.0,
                },
                Variant {
                    id: "B".to_string(),
                    name: "Treatment".to_string(),
                    allocation_weight: 1.0,
                },
            ],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 100,
        }
    }

    // ── assign_variant ───────────────────────────────────────────────────────

    #[test]
    fn assign_variant_deterministic_same_user() {
        let exp = two_variant_experiment();
        let v1 = assign_variant(&exp, "user_42", AssignmentMethod::Deterministic)
            .expect("assign variant should succeed");
        let v2 = assign_variant(&exp, "user_42", AssignmentMethod::Deterministic)
            .expect("assign variant should succeed");
        assert_eq!(v1.id, v2.id);
    }

    #[test]
    fn assign_variant_different_users_may_differ() {
        let exp = two_variant_experiment();
        let ids: Vec<_> = (0..100)
            .map(|i| {
                assign_variant(&exp, &format!("user_{i}"), AssignmentMethod::Deterministic)
                    .expect("value should be present should succeed")
                    .id
                    .clone()
            })
            .collect();
        let has_a = ids.iter().any(|id| id == "A");
        let has_b = ids.iter().any(|id| id == "B");
        assert!(has_a, "expected some users in variant A");
        assert!(has_b, "expected some users in variant B");
    }

    #[test]
    fn assign_variant_no_variants_returns_error() {
        let exp = Experiment {
            id: "empty".to_string(),
            name: "Empty".to_string(),
            variants: vec![],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 10,
        };
        let result = assign_variant(&exp, "u1", AssignmentMethod::Deterministic);
        assert!(result.is_err());
    }

    #[test]
    fn assign_variant_zero_weight_returns_error() {
        let exp = Experiment {
            id: "zero".to_string(),
            name: "Zero".to_string(),
            variants: vec![Variant {
                id: "A".to_string(),
                name: "A".to_string(),
                allocation_weight: 0.0,
            }],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 10,
        };
        let result = assign_variant(&exp, "u1", AssignmentMethod::Deterministic);
        assert!(result.is_err());
    }

    #[test]
    fn assign_variant_single_variant() {
        let exp = Experiment {
            id: "single".to_string(),
            name: "Single".to_string(),
            variants: vec![Variant {
                id: "only".to_string(),
                name: "Only".to_string(),
                allocation_weight: 1.0,
            }],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 10,
        };
        let v = assign_variant(&exp, "u1", AssignmentMethod::Deterministic)
            .expect("assign variant should succeed");
        assert_eq!(v.id, "only");
    }

    #[test]
    fn assign_variant_weighted_distribution() {
        // 90 % to A, 10 % to B — over 1000 users B should get ~100 (±50 for tolerance).
        let exp = Experiment {
            id: "weighted".to_string(),
            name: "Weighted".to_string(),
            variants: vec![
                Variant {
                    id: "A".to_string(),
                    name: "A".to_string(),
                    allocation_weight: 9.0,
                },
                Variant {
                    id: "B".to_string(),
                    name: "B".to_string(),
                    allocation_weight: 1.0,
                },
            ],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 100,
        };
        let b_count = (0..1000)
            .filter(|i| {
                assign_variant(&exp, &format!("u{i}"), AssignmentMethod::Deterministic)
                    .expect("value should be present should succeed")
                    .id
                    == "B"
            })
            .count();
        assert!(b_count < 250, "too many in B: {b_count}");
    }

    // ── Rate calculations ────────────────────────────────────────────────────

    #[test]
    fn click_through_rate_basic() {
        let m = VariantMetrics {
            variant_id: "A".to_string(),
            impressions: 100,
            clicks: 5,
            ..Default::default()
        };
        assert!((click_through_rate(&m) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn click_through_rate_zero_impressions() {
        let m = VariantMetrics::new("A");
        assert_eq!(click_through_rate(&m), 0.0);
    }

    #[test]
    fn conversion_rate_basic() {
        let m = VariantMetrics {
            variant_id: "B".to_string(),
            impressions: 200,
            conversions: 10,
            ..Default::default()
        };
        assert!((conversion_rate(&m) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn average_watch_duration_basic() {
        let m = VariantMetrics {
            variant_id: "A".to_string(),
            impressions: 4,
            watch_duration_sum_ms: 40_000,
            ..Default::default()
        };
        assert!((average_watch_duration(&m) - 10_000.0).abs() < 1e-3);
    }

    #[test]
    fn completion_rate_basic() {
        let m = VariantMetrics {
            variant_id: "A".to_string(),
            impressions: 10,
            completion_count: 3,
            ..Default::default()
        };
        assert!((completion_rate(&m) - 0.3).abs() < 1e-6);
    }

    // ── z_test ───────────────────────────────────────────────────────────────

    #[test]
    fn z_test_no_difference() {
        let z = z_test(0.05, 1000, 0.05, 1000);
        assert!(z.abs() < 1e-3, "z={z}");
    }

    #[test]
    fn z_test_large_difference_significant() {
        // p1=0.10, p2=0.05 with n=5000 each should be very significant.
        let z = z_test(0.10, 5000, 0.05, 5000);
        assert!(z > 1.96, "z={z}");
    }

    #[test]
    fn z_test_zero_sample_returns_zero() {
        assert_eq!(z_test(0.05, 0, 0.05, 100), 0.0);
        assert_eq!(z_test(0.05, 100, 0.05, 0), 0.0);
    }

    #[test]
    fn is_significant_alpha_05() {
        assert!(is_significant(2.0, 0.05));
        assert!(!is_significant(1.5, 0.05));
    }

    #[test]
    fn is_significant_alpha_01() {
        assert!(is_significant(2.6, 0.01));
        assert!(!is_significant(2.0, 0.01));
    }

    // ── winning_variant ──────────────────────────────────────────────────────

    #[test]
    fn winning_variant_by_ctr() {
        // Use n=500 each so z≈3.0 (significant at α=0.05 critical_z=1.96).
        // B: 50/500 = 10% CTR, A: 25/500 = 5% CTR.
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 500,
                clicks: 25,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 500,
                clicks: 50,
                ..Default::default()
            },
        );
        let winner = winning_variant(&results, "ctr");
        assert_eq!(winner, Some("B"));
    }

    #[test]
    fn winning_variant_no_impressions_returns_none() {
        let exp = two_variant_experiment();
        let results = ExperimentResults::new(exp);
        let winner = winning_variant(&results, "ctr");
        assert!(winner.is_none());
    }

    #[test]
    fn winning_variant_by_completion() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 100,
                completion_count: 30,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 100,
                completion_count: 50,
                ..Default::default()
            },
        );
        let winner = winning_variant(&results, "completion");
        assert_eq!(winner, Some("B"));
    }

    #[test]
    fn experiment_results_record_methods() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.record_impression("A");
        results.record_impression("A");
        results.record_click("A");
        results.record_conversion("A");
        results.record_completion("A", 5000);
        let m = &results.variant_metrics["A"];
        assert_eq!(m.impressions, 2);
        assert_eq!(m.clicks, 1);
        assert_eq!(m.conversions, 1);
        assert_eq!(m.completion_count, 1);
        assert_eq!(m.watch_duration_sum_ms, 5000);
    }

    // ── winning_variant_with_alpha ────────────────────────────────────────────

    #[test]
    fn winning_variant_with_alpha_same_result_as_default() {
        // Use n=500 so z≈3.0 (significant at α=0.05).  Both `winning_variant`
        // (which uses α=0.05) and `winning_variant_with_alpha(..., 0.05)` must
        // agree and declare B the winner.
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 500,
                clicks: 25,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 500,
                clicks: 50,
                ..Default::default()
            },
        );
        let w_default = winning_variant(&results, "ctr");
        let w_alpha = winning_variant_with_alpha(&results, "ctr", 0.05);
        assert_eq!(w_default, w_alpha);
        assert_eq!(w_default, Some("B"));
    }

    #[test]
    fn winning_variant_with_alpha_01() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 1000,
                conversions: 50,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 1000,
                conversions: 80,
                ..Default::default()
            },
        );
        // B should win at both common alpha levels.
        assert_eq!(
            winning_variant_with_alpha(&results, "conversion", 0.05),
            Some("B")
        );
        assert_eq!(
            winning_variant_with_alpha(&results, "conversion", 0.01),
            Some("B")
        );
    }

    #[test]
    fn winning_variant_with_alpha_no_impressions() {
        let exp = two_variant_experiment();
        let results = ExperimentResults::new(exp);
        assert!(winning_variant_with_alpha(&results, "ctr", 0.05).is_none());
    }

    // ── alpha_to_critical_z ──────────────────────────────────────────────────

    #[test]
    fn alpha_to_critical_z_values() {
        assert!((alpha_to_critical_z(0.05) - 1.96).abs() < 0.01);
        assert!((alpha_to_critical_z(0.01) - 2.576).abs() < 0.01);
        assert!((alpha_to_critical_z(0.10) - 1.645).abs() < 0.01);
        assert!((alpha_to_critical_z(0.001) - 3.291).abs() < 0.01);
    }

    // ── bayesian_winner ──────────────────────────────────────────────────────

    #[test]
    fn bayesian_winner_b_clearly_beats_a() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        // A: 5 clicks / 100, B: 30 clicks / 100 — B should dominate.
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 100,
                clicks: 5,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 100,
                clicks: 30,
                ..Default::default()
            },
        );
        let res = bayesian_winner(&results, "A", "B", "ctr", 10_000, 42)
            .expect("bayesian winner should succeed");
        assert!(
            res.prob_b_beats_a > 0.95,
            "expected high prob that B beats A, got {}",
            res.prob_b_beats_a
        );
        assert!(res.expected_uplift > 0.0, "uplift should be positive");
    }

    #[test]
    fn bayesian_winner_equal_variants_around_50pct() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        // Identical click rates — prob should be near 0.5.
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 1000,
                clicks: 100,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 1000,
                clicks: 100,
                ..Default::default()
            },
        );
        let res = bayesian_winner(&results, "A", "B", "ctr", 20_000, 99)
            .expect("bayesian winner should succeed");
        assert!(
            (res.prob_b_beats_a - 0.5).abs() < 0.05,
            "equal variants should give ~50% prob, got {}",
            res.prob_b_beats_a
        );
    }

    #[test]
    fn bayesian_winner_missing_variant_returns_error() {
        let exp = two_variant_experiment();
        let results = ExperimentResults::new(exp);
        let err = bayesian_winner(&results, "A", "nonexistent", "ctr", 100, 0);
        assert!(err.is_err());
    }

    #[test]
    fn bayesian_winner_zero_impressions_returns_error() {
        let exp = two_variant_experiment();
        let results = ExperimentResults::new(exp);
        // A and B are initialised with 0 impressions.
        let err = bayesian_winner(&results, "A", "B", "ctr", 100, 0);
        assert!(err.is_err());
    }

    #[test]
    fn bayesian_winner_conversion_metric() {
        let exp = two_variant_experiment();
        let mut results = ExperimentResults::new(exp);
        results.variant_metrics.insert(
            "A".to_string(),
            VariantMetrics {
                variant_id: "A".to_string(),
                impressions: 200,
                conversions: 10,
                ..Default::default()
            },
        );
        results.variant_metrics.insert(
            "B".to_string(),
            VariantMetrics {
                variant_id: "B".to_string(),
                impressions: 200,
                conversions: 50,
                ..Default::default()
            },
        );
        let res = bayesian_winner(&results, "A", "B", "conversion", 10_000, 7)
            .expect("bayesian winner should succeed");
        assert!(res.prob_b_beats_a > 0.95);
        assert!((res.posterior_mean_b - 0.25).abs() < 0.05);
    }

    // ── FNV-1a hash pin ──────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_exact_hash_pin() {
        // Pin the exact FNV-1a 32-bit hash of b"user_42" by computing it in
        // the test using the same constants as the production function.
        const FNV_OFFSET: u32 = 2_166_136_261;
        const FNV_PRIME: u32 = 16_777_619;
        let input = b"user_42";
        let mut expected = FNV_OFFSET;
        for &b in input.iter() {
            expected ^= u32::from(b);
            expected = expected.wrapping_mul(FNV_PRIME);
        }
        // Call the pub(crate) function directly and compare.
        let actual = fnv1a_32(b"user_42");
        assert_eq!(
            actual, expected,
            "FNV-1a hash of \"user_42\" must be deterministic: got {actual}, want {expected}"
        );
    }

    #[test]
    fn test_assign_variant_exact_bucket() {
        // Three equal-weight variants.  The same user must always land in the
        // same variant regardless of how many times assign_variant is called.
        let exp = Experiment {
            id: "bucket_pin".to_string(),
            name: "Bucket Pin Test".to_string(),
            variants: vec![
                Variant {
                    id: "alpha".to_string(),
                    name: "Alpha".to_string(),
                    allocation_weight: 1.0,
                },
                Variant {
                    id: "beta".to_string(),
                    name: "Beta".to_string(),
                    allocation_weight: 1.0,
                },
                Variant {
                    id: "gamma".to_string(),
                    name: "Gamma".to_string(),
                    allocation_weight: 1.0,
                },
            ],
            start_ms: 0,
            end_ms: None,
            min_sample_size: 10,
        };
        let user = "test_determinism_user";
        let v1 = assign_variant(&exp, user, AssignmentMethod::Deterministic)
            .expect("first assignment should succeed");
        let v2 = assign_variant(&exp, user, AssignmentMethod::Deterministic)
            .expect("second assignment should succeed");
        let v3 = assign_variant(&exp, user, AssignmentMethod::Deterministic)
            .expect("third assignment should succeed");
        assert_eq!(v1.id, v2.id, "assignment must be deterministic (1 vs 2)");
        assert_eq!(v2.id, v3.id, "assignment must be deterministic (2 vs 3)");
    }
}
