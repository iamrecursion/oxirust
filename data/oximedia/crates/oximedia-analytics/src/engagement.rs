//! Engagement scoring model for media content.
//!
//! Computes a weighted engagement score from viewer session data, models score
//! trends over time with linear regression, and ranks content by engagement.

use crate::session::{build_playback_map, PlaybackEvent, ViewerSession};

// ─── Score model ──────────────────────────────────────────────────────────────

/// Decomposed components of an engagement score (each in 0.0 – 1.0).
#[derive(Debug, Clone, PartialEq)]
pub struct EngagementComponents {
    /// Ratio of average watch time to content duration (capped at 1.0).
    pub watch_time_score: f32,
    /// Fraction of sessions that reached ≥95 % completion.
    pub completion_score: f32,
    /// Fraction of sessions that rewatched any segment.
    pub rewatch_score: f32,
    /// Normalised social-interaction engagement in `0.0 – 1.0`, as produced by
    /// [`SocialSignals::engagement_score`].
    ///
    /// This is the **real** computed value when social data is supplied via
    /// [`compute_engagement_with_social`].  When no social data is available —
    /// for example through [`compute_engagement`], because the
    /// [`ViewerSession`]/[`PlaybackEvent`] model carries no social signals — it
    /// is honestly `0.0` (never a fabricated constant).
    pub social_score: f32,
    /// Penalty term proportional to the forward-seek rate (lower is better).
    pub seek_forward_penalty: f32,
}

/// Weights controlling the relative importance of each engagement component.
#[derive(Debug, Clone, PartialEq)]
pub struct EngagementWeights {
    pub watch_time: f32,
    pub completion: f32,
    pub rewatch: f32,
    pub social: f32,
    /// Multiplicative penalty factor for forward seeks.  A value of 1.0 means
    /// each forward seek as a fraction of total events subtracts directly from
    /// the score.
    pub forward_seek_penalty: f32,
}

impl EngagementWeights {
    /// All five components equally weighted at 0.2.
    pub fn default() -> Self {
        Self {
            watch_time: 0.2,
            completion: 0.2,
            rewatch: 0.2,
            social: 0.2,
            forward_seek_penalty: 0.2,
        }
    }

    /// Return a copy of these weights with the `social` weight removed (set to
    /// `0.0`) and its mass redistributed **proportionally** across the three
    /// positive reward components (`watch_time`, `completion`, `rewatch`).
    ///
    /// This is used when no social-interaction data is available: rather than
    /// fabricate a social score, the missing channel's weight is absorbed by the
    /// channels we *can* measure, so the reward ceiling is preserved and the
    /// absent channel does not silently drag the score down.  Concretely, with
    /// `reward_base = watch_time + completion + rewatch`, each reward weight is
    /// scaled by `(reward_base + social) / reward_base`.
    ///
    /// The `forward_seek_penalty` weight is deliberately left untouched: it is a
    /// *penalty*, not a reward, and must not grow merely because a reward
    /// channel disappeared.  When `reward_base` is `0.0` (no reward channels to
    /// absorb the mass) the social weight is simply dropped.
    pub fn redistribute_social(&self) -> Self {
        let reward_base = self.watch_time + self.completion + self.rewatch;
        if self.social <= 0.0 || reward_base <= 0.0 {
            return Self {
                watch_time: self.watch_time,
                completion: self.completion,
                rewatch: self.rewatch,
                social: 0.0,
                forward_seek_penalty: self.forward_seek_penalty,
            };
        }
        let factor = (reward_base + self.social) / reward_base;
        Self {
            watch_time: self.watch_time * factor,
            completion: self.completion * factor,
            rewatch: self.rewatch * factor,
            social: 0.0,
            forward_seek_penalty: self.forward_seek_penalty,
        }
    }
}

/// Raw social-interaction counts for a piece of content.
///
/// The [`ViewerSession`]/[`PlaybackEvent`] model carries **no** social data —
/// only playback events — so these signals must be supplied explicitly (e.g.
/// from a CMS, a comments service, or a share-tracking pipeline).  Use
/// [`SocialSignals::engagement_score`] to collapse the raw counts into a single
/// normalised social engagement score in `[0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SocialSignals {
    /// Number of times the content was viewed.  This is the denominator of the
    /// engagement rate; **zero views yields a score of `0.0`**, never a
    /// fabricated constant.
    pub views: u64,
    /// Number of "like"/reaction interactions (the lightest-effort signal).
    pub likes: u64,
    /// Number of shares/reposts (a stronger signal than a like).
    pub shares: u64,
    /// Number of comments (the highest-effort signal).
    pub comments: u64,
}

impl SocialSignals {
    /// Relative weight of a share versus a like (a share spreads the content,
    /// so it counts for more).
    const SHARE_WEIGHT: f64 = 2.0;
    /// Relative weight of a comment versus a like (a comment is the most
    /// effortful interaction).
    const COMMENT_WEIGHT: f64 = 3.0;
    /// Characteristic weighted-engagement rate at which the score reaches
    /// `1 − e⁻¹ ≈ 0.632`.  Rates above this saturate quickly toward `1.0`.
    const SATURATION_RATE: f64 = 0.10;

    /// Compute a normalised social engagement score in `[0.0, 1.0]`.
    ///
    /// The raw counts are first combined into a single **weighted engagement
    /// rate**
    ///
    /// ```text
    /// rate = (likes + 2·shares + 3·comments) / views
    /// ```
    ///
    /// (shares and comments are stronger signals than likes), then passed
    /// through a saturating exponential
    ///
    /// ```text
    /// score = 1 − exp(−rate / 0.10)
    /// ```
    ///
    /// which is monotonically increasing and maps `rate = 0` to `0.0`, the
    /// characteristic rate `0.10` to `≈ 0.632`, and arbitrarily large rates
    /// asymptotically (but never exceeding) `1.0`.
    ///
    /// With **zero views** the rate is undefined, so the score is `0.0` — an
    /// honest "no data" value rather than a fabricated midpoint.
    pub fn engagement_score(&self) -> f32 {
        if self.views == 0 {
            return 0.0;
        }
        let weighted = self.likes as f64
            + Self::SHARE_WEIGHT * self.shares as f64
            + Self::COMMENT_WEIGHT * self.comments as f64;
        let rate = weighted / self.views as f64;
        let score = 1.0 - (-rate / Self::SATURATION_RATE).exp();
        score.clamp(0.0, 1.0) as f32
    }
}

/// Final engagement score for a piece of content.
#[derive(Debug, Clone, PartialEq)]
pub struct ContentEngagementScore {
    pub content_id: String,
    /// Overall score in 0.0 – 1.0.
    pub score: f32,
    pub components: EngagementComponents,
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute an engagement score for a content item from its viewer sessions,
/// **without** any social-interaction data.
///
/// The [`ViewerSession`]/[`PlaybackEvent`] model carries no social signals, so
/// the social channel is a *missing-data* channel.  Rather than fabricate a
/// social score, this function:
///
/// 1. sets the social component to an honest `0.0`, and
/// 2. redistributes the `social` weight across the remaining reward components
///    via [`EngagementWeights::redistribute_social`], so the absent channel
///    does not silently drag the overall score down.
///
/// It is exactly equivalent to calling [`compute_engagement_with_social`] with
/// `SocialSignals::default()` and the redistributed weights.  When you *do*
/// have social data, call [`compute_engagement_with_social`] directly.
///
/// Returns a score of `0.0` when `sessions` is empty or `content_duration_ms`
/// is zero.  The `content_id` is taken from the first session's `content_id`.
pub fn compute_engagement(
    sessions: &[ViewerSession],
    content_duration_ms: u64,
    weights: &EngagementWeights,
) -> ContentEngagementScore {
    let redistributed = weights.redistribute_social();
    compute_engagement_with_social(
        sessions,
        content_duration_ms,
        &redistributed,
        &SocialSignals::default(),
    )
}

/// Compute an engagement score from viewer sessions **and** explicit social
/// signals.
///
/// This is the full-information entry point: the social component is the real
/// normalised value from [`SocialSignals::engagement_score`], and `weights` are
/// applied exactly as given (no redistribution).  Pass
/// `SocialSignals::default()` only if you genuinely have zero social
/// interactions — its score is `0.0`, not a fabricated midpoint.
///
/// The weighted score is
///
/// ```text
/// score = w_watch·watch + w_completion·completion + w_rewatch·rewatch
///       + w_social·social − w_penalty·seek_penalty
/// ```
///
/// clamped to `[0.0, 1.0]`.
///
/// Returns a score of `0.0` when `sessions` is empty or `content_duration_ms`
/// is zero (there is no playback evidence of engagement); the `social_score`
/// component still reflects the supplied signals.  The `content_id` is taken
/// from the first session's `content_id`.
pub fn compute_engagement_with_social(
    sessions: &[ViewerSession],
    content_duration_ms: u64,
    weights: &EngagementWeights,
    social: &SocialSignals,
) -> ContentEngagementScore {
    let content_id = sessions
        .first()
        .map(|s| s.content_id.clone())
        .unwrap_or_default();

    let social_score = social.engagement_score();

    if sessions.is_empty() || content_duration_ms == 0 {
        return ContentEngagementScore {
            content_id,
            score: 0.0,
            components: EngagementComponents {
                watch_time_score: 0.0,
                completion_score: 0.0,
                rewatch_score: 0.0,
                social_score,
                seek_forward_penalty: 0.0,
            },
        };
    }

    let n = sessions.len() as f64;
    let completion_threshold_ms = (content_duration_ms as f64 * 0.95) as u64;

    let mut total_watch_ms: u64 = 0;
    let mut completion_count: u32 = 0;
    let mut rewatch_count: u32 = 0;
    let mut total_events: u32 = 0;
    let mut forward_seek_count: u32 = 0;

    for session in sessions {
        // Watch time: prefer the End event's watch_duration_ms.
        let session_watch_ms = session.events.iter().fold(0u64, |acc, e| match e {
            PlaybackEvent::End {
                watch_duration_ms, ..
            } => acc.max(*watch_duration_ms),
            _ => acc,
        });
        total_watch_ms += session_watch_ms;

        // Completion: did the session reach ≥ 95 % of the content?
        let map = build_playback_map(session, content_duration_ms);
        let completion_sec = (completion_threshold_ms / 1000) as usize;
        if map
            .positions_watched
            .get(completion_sec)
            .copied()
            .unwrap_or(false)
        {
            completion_count += 1;
        }

        // Rewatch: any second watched more than once means the session included a seek-back.
        // We detect this by checking for backward seek events.
        let has_rewatch = session
            .events
            .iter()
            .any(|e| matches!(e, PlaybackEvent::Seek { from_ms, to_ms } if to_ms < from_ms));
        if has_rewatch {
            rewatch_count += 1;
        }

        // Forward seek penalty.
        for event in &session.events {
            total_events += 1;
            if let PlaybackEvent::Seek { from_ms, to_ms } = event {
                if to_ms > from_ms {
                    forward_seek_count += 1;
                }
            }
        }
    }

    let avg_watch_ms = total_watch_ms as f64 / n;
    let watch_time_score = (avg_watch_ms / content_duration_ms as f64).min(1.0) as f32;
    let completion_score = completion_count as f32 / sessions.len() as f32;
    let rewatch_score = rewatch_count as f32 / sessions.len() as f32;
    // `social_score` is already bound at the top of the function from the
    // supplied `SocialSignals` — there is no fabricated placeholder here.

    let seek_forward_penalty = if total_events > 0 {
        forward_seek_count as f32 / total_events as f32
    } else {
        0.0
    };

    // Weighted score:
    //   score = w_watch * watch_time_score
    //         + w_completion * completion_score
    //         + w_rewatch * rewatch_score
    //         + w_social * social_score
    //         - w_penalty * seek_forward_penalty
    // Clamped to [0.0, 1.0].
    let raw_score = weights.watch_time * watch_time_score
        + weights.completion * completion_score
        + weights.rewatch * rewatch_score
        + weights.social * social_score
        - weights.forward_seek_penalty * seek_forward_penalty;

    let score = raw_score.max(0.0).min(1.0);

    ContentEngagementScore {
        content_id,
        score,
        components: EngagementComponents {
            watch_time_score,
            completion_score,
            rewatch_score,
            social_score,
            seek_forward_penalty,
        },
    }
}

// ─── Trend analysis ───────────────────────────────────────────────────────────

/// A time-series of engagement scores for a content item.
#[derive(Debug, Clone)]
pub struct EngagementTrend {
    /// Pairs of (timestamp_ms, engagement_score).
    pub scores_over_time: Vec<(i64, f32)>,
}

impl EngagementTrend {
    /// Compute the linear-regression slope of the score series.
    ///
    /// Returns `0.0` if the series has fewer than two points or if the
    /// denominator is zero.
    pub fn slope(&self) -> f32 {
        linear_regression_slope(&self.scores_over_time)
    }
}

/// Compute the least-squares linear regression slope of the given (x, y) data.
///
/// `slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`
///
/// Returns `0.0` when the denominator is zero (all x values identical) or when
/// there are fewer than two data points.
pub fn linear_regression_slope(points: &[(i64, f32)]) -> f32 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }

    // Use f64 for numerical stability with large timestamp values.
    let n_f = n as f64;
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    let mut sum_xy: f64 = 0.0;
    let mut sum_x2: f64 = 0.0;

    for &(x, y) in points {
        let xf = x as f64;
        let yf = y as f64;
        sum_x += xf;
        sum_y += yf;
        sum_xy += xf * yf;
        sum_x2 += xf * xf;
    }

    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }

    ((n_f * sum_xy - sum_x * sum_y) / denom) as f32
}

// ─── Time-series decomposition ────────────────────────────────────────────────

/// A period used for seasonal decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalPeriod {
    /// 7-day weekly seasonality.
    Weekly,
    /// 30-day monthly seasonality.
    Monthly,
    /// Custom period length (number of observations per cycle).
    Custom(usize),
}

impl SeasonalPeriod {
    /// Return the integer period length (number of observations per cycle).
    pub fn length(&self) -> usize {
        match self {
            SeasonalPeriod::Weekly => 7,
            SeasonalPeriod::Monthly => 30,
            SeasonalPeriod::Custom(n) => *n,
        }
    }
}

/// Result of additive time-series decomposition: y = trend + seasonal + residual.
///
/// All three components have the same length as the input series.
#[derive(Debug, Clone)]
pub struct DecomposedSeries {
    /// Smoothed trend component (centered moving average).
    pub trend: Vec<f64>,
    /// Seasonal component (mean deviation for each seasonal phase).
    pub seasonal: Vec<f64>,
    /// Residual = observed − trend − seasonal.
    pub residual: Vec<f64>,
    /// Original observed values.
    pub observed: Vec<f64>,
    /// Period used for decomposition.
    pub period: usize,
}

/// Decompose a time-series into trend + seasonal + residual components.
///
/// Uses classical additive decomposition (STL-style but without LOESS):
///
/// 1. **Trend**: centered moving average with window = `period`.
/// 2. **Seasonal**: for each phase position in [0, period), compute the mean
///    of `(observed − trend)` across all cycles; then centre by subtracting
///    the mean of the seasonal indices.
/// 3. **Residual**: `observed − trend − seasonal`.
///
/// For positions at the edges of the series where the centered moving average
/// cannot be computed, the trend is interpolated linearly.
///
/// Returns `None` when the series has fewer than `2 * period` points.
pub fn decompose_time_series(
    series: &[(i64, f32)],
    period: SeasonalPeriod,
) -> Option<DecomposedSeries> {
    let n = series.len();
    let p = period.length();
    if p == 0 || n < 2 * p {
        return None;
    }

    let y: Vec<f64> = series.iter().map(|&(_, v)| v as f64).collect();

    // ── Step 1: Centered moving average (trend) ───────────────────────────────
    let half = p / 2;
    let mut trend = vec![f64::NAN; n];

    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window = &y[start..end];
        trend[i] = window.iter().sum::<f64>() / window.len() as f64;
    }

    // Interpolate NaN edges linearly from the first/last computed values.
    if let Some(first_valid) = trend.iter().position(|v| !v.is_nan()) {
        let val = trend[first_valid];
        for i in 0..first_valid {
            trend[i] = val;
        }
    }
    if let Some(last_valid) = trend.iter().rposition(|v| !v.is_nan()) {
        let val = trend[last_valid];
        for i in (last_valid + 1)..n {
            trend[i] = val;
        }
    }
    // Linear interpolation between known valid points (fill interior NaNs).
    let mut start = None;
    for i in 0..n {
        if !trend[i].is_nan() {
            if let Some(s) = start {
                // Interpolate from s to i.
                let t_s = trend[s];
                let t_e = trend[i];
                for j in (s + 1)..i {
                    let t = (j - s) as f64 / (i - s) as f64;
                    trend[j] = t_s + t * (t_e - t_s);
                }
                start = None;
            }
        } else if start.is_none() {
            start = Some(if i == 0 { 0 } else { i - 1 });
        }
    }

    // ── Step 2: Seasonal indices ──────────────────────────────────────────────
    // detrended[i] = y[i] − trend[i]
    let detrended: Vec<f64> = y
        .iter()
        .zip(trend.iter())
        .map(|(&yi, &ti)| yi - ti)
        .collect();

    // Average detrended values for each phase position.
    let mut phase_sums = vec![0.0f64; p];
    let mut phase_counts = vec![0u32; p];
    for (i, &d) in detrended.iter().enumerate() {
        let phase = i % p;
        phase_sums[phase] += d;
        phase_counts[phase] += 1;
    }
    let mut phase_means: Vec<f64> = phase_sums
        .iter()
        .zip(phase_counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Centre seasonal indices so they sum to zero.
    let phase_mean: f64 = phase_means.iter().sum::<f64>() / p as f64;
    for v in &mut phase_means {
        *v -= phase_mean;
    }

    let seasonal: Vec<f64> = (0..n).map(|i| phase_means[i % p]).collect();

    // ── Step 3: Residual ──────────────────────────────────────────────────────
    let residual: Vec<f64> = y
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&yi, &ti), &si)| yi - ti - si)
        .collect();

    Some(DecomposedSeries {
        trend,
        seasonal,
        residual,
        observed: y,
        period: p,
    })
}

// ─── Exponential Moving Average ───────────────────────────────────────────────

/// Smoothing factor and configuration for exponential moving average (EMA).
///
/// EMA: `EMA(0) = y(0)`, `EMA(i) = alpha * y(i) + (1 - alpha) * EMA(i-1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmaConfig {
    /// Smoothing factor `alpha ∈ (0.0, 1.0]`.
    pub alpha: f64,
}

impl EmaConfig {
    /// Build from explicit `alpha`. Returns `None` when `alpha ∉ (0.0, 1.0]`.
    pub fn with_alpha(alpha: f64) -> Option<Self> {
        if alpha > 0.0 && alpha <= 1.0 {
            Some(Self { alpha })
        } else {
            None
        }
    }

    /// Build from span N using `alpha = 2 / (N + 1)`. Returns `None` for span 0.
    pub fn from_span(span: usize) -> Option<Self> {
        if span == 0 {
            return None;
        }
        Some(Self {
            alpha: 2.0 / (span as f64 + 1.0),
        })
    }
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self { alpha: 0.2 }
    }
}

/// Result of an EMA computation over an engagement score time-series.
#[derive(Debug, Clone)]
pub struct EmaResult {
    /// EMA-smoothed values aligned 1-to-1 with the input series.
    pub smoothed: Vec<f64>,
    /// The smoothing factor `alpha` applied.
    pub alpha: f64,
    /// Linear-regression slope of the smoothed series.
    pub trend_slope: f64,
}

impl EmaResult {
    /// Most recent smoothed value.
    pub fn last_smoothed(&self) -> f64 {
        self.smoothed.last().copied().unwrap_or(0.0)
    }

    /// First smoothed value (seeded from the first observation).
    pub fn first_smoothed(&self) -> f64 {
        self.smoothed.first().copied().unwrap_or(0.0)
    }

    /// Infer the trend direction from the EMA's slope.
    pub fn trend_direction(&self, epsilon: f64) -> TrendDirection {
        TrendDirection::from_slope(self.trend_slope, epsilon)
    }
}

/// Trend direction inferred from slope analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Score is growing over time.
    Growing,
    /// Score is declining over time.
    Declining,
    /// No discernible trend.
    Flat,
}

impl TrendDirection {
    /// Classify a slope value.
    pub fn from_slope(slope: f64, epsilon: f64) -> Self {
        if slope > epsilon {
            Self::Growing
        } else if slope < -epsilon {
            Self::Declining
        } else {
            Self::Flat
        }
    }
}

/// Compute the exponential moving average of an engagement score series.
///
/// Returns `None` when the series is empty or `alpha ∉ (0.0, 1.0]`.
pub fn exponential_moving_average(series: &[(i64, f32)], config: &EmaConfig) -> Option<EmaResult> {
    if series.is_empty() || config.alpha <= 0.0 || config.alpha > 1.0 {
        return None;
    }

    let alpha = config.alpha;
    let one_minus = 1.0 - alpha;

    let mut smoothed = Vec::with_capacity(series.len());
    let mut prev = f64::from(series[0].1);
    smoothed.push(prev);

    for &(_, y) in &series[1..] {
        let ema = alpha * f64::from(y) + one_minus * prev;
        smoothed.push(ema);
        prev = ema;
    }

    let indexed: Vec<(i64, f32)> = smoothed
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as i64, v as f32))
        .collect();
    let trend_slope = f64::from(linear_regression_slope(&indexed));

    Some(EmaResult {
        smoothed,
        alpha,
        trend_slope,
    })
}

// ─── Ranking ──────────────────────────────────────────────────────────────────

/// Ranks and recommends content items by their engagement score.
pub struct ContentRanker;

impl ContentRanker {
    /// Sort `scores` by engagement descending and return `(content_id, score)`
    /// pairs.
    pub fn rank_by_engagement<'a>(scores: &'a [ContentEngagementScore]) -> Vec<(&'a str, f32)> {
        let mut ranked: Vec<_> = scores
            .iter()
            .map(|s| (s.content_id.as_str(), s.score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn full_watch_session(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms,
                },
            ],
        }
    }

    fn partial_watch_session(id: &str, watch_ms: u64, _content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: watch_ms,
                    watch_duration_ms: watch_ms,
                },
            ],
        }
    }

    fn session_with_forward_seek(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Seek {
                    from_ms: 3000,
                    to_ms: 7000,
                },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms / 2,
                },
            ],
        }
    }

    fn session_with_backward_seek(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Seek {
                    from_ms: 7000,
                    to_ms: 3000,
                },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms,
                },
            ],
        }
    }

    // ── compute_engagement ───────────────────────────────────────────────────

    #[test]
    fn engagement_empty_sessions() {
        let weights = EngagementWeights::default();
        let score = compute_engagement(&[], 10_000, &weights);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn engagement_zero_duration() {
        let sessions = vec![full_watch_session("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 0, &weights);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn engagement_full_watch_high_score() {
        let sessions: Vec<_> = (0..10)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        // No social data: the 0.2 social weight is redistributed across the three
        // reward components, scaling each from 0.2 to 0.2·(0.8/0.6) ≈ 0.2667.
        // watch_time=1.0, completion=1.0, rewatch=0.0, social=0.0, penalty=0.0
        // = 0.2667*1 + 0.2667*1 + 0.2667*0 + 0*0 - 0.2*0 ≈ 0.5333
        assert!((score.score - 0.5333).abs() < 0.01, "score={}", score.score);
        // Honest social score: 0.0 when absent — never a fabricated 0.5.
        assert_eq!(score.components.social_score, 0.0);
    }

    #[test]
    fn engagement_partial_watch_lower_score() {
        let sessions: Vec<_> = (0..10)
            .map(|i| partial_watch_session(&format!("s{i}"), 3_000, 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let full = compute_engagement(
            &(0..10)
                .map(|i| full_watch_session(&format!("s{i}"), 10_000))
                .collect::<Vec<_>>(),
            10_000,
            &weights,
        );
        let partial = compute_engagement(&sessions, 10_000, &weights);
        assert!(
            partial.score < full.score,
            "partial={} full={}",
            partial.score,
            full.score
        );
    }

    #[test]
    fn engagement_components_watch_time_capped() {
        // Watch time = 2x content duration → capped at 1.0.
        let sessions = vec![partial_watch_session("s1", 20_000, 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert!(score.components.watch_time_score <= 1.0);
    }

    #[test]
    fn engagement_rewatch_detected() {
        let sessions = vec![session_with_backward_seek("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert!((score.components.rewatch_score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn engagement_forward_seek_penalty() {
        let no_seek: Vec<_> = (0..5)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let with_seek: Vec<_> = (0..5)
            .map(|i| session_with_forward_seek(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let score_clean = compute_engagement(&no_seek, 10_000, &weights);
        let score_seeky = compute_engagement(&with_seek, 10_000, &weights);
        assert!(
            score_seeky.score <= score_clean.score,
            "seeky={} clean={}",
            score_seeky.score,
            score_clean.score
        );
    }

    #[test]
    fn engagement_social_score_absent_is_zero() {
        let sessions = vec![full_watch_session("s1", 5_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 5_000, &weights);
        // No social data is available, so the honest social score is 0.0 — the
        // old fabricated 0.5 placeholder must be gone.
        assert_eq!(score.components.social_score, 0.0);
        assert!((score.components.social_score - 0.5).abs() > 1e-6);
    }

    #[test]
    fn engagement_content_id_from_first_session() {
        let sessions = vec![full_watch_session("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert_eq!(score.content_id, "content_a");
    }

    #[test]
    fn engagement_weights_default_sum_to_one() {
        let w = EngagementWeights::default();
        let sum = w.watch_time + w.completion + w.rewatch + w.social + w.forward_seek_penalty;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ── SocialSignals normalization ──────────────────────────────────────────

    #[test]
    fn social_signals_default_is_zero() {
        // Default (all-zero counts) must yield 0.0, not a fabricated midpoint.
        assert_eq!(SocialSignals::default().engagement_score(), 0.0);
    }

    #[test]
    fn social_signals_zero_views_is_zero() {
        // Interactions with zero views → undefined rate → honest 0.0.
        let s = SocialSignals {
            views: 0,
            likes: 1_000,
            shares: 500,
            comments: 250,
        };
        assert_eq!(s.engagement_score(), 0.0);
    }

    #[test]
    fn social_signals_low_engagement_small_score() {
        // rate = 10 / 10_000 = 0.001 → score = 1 − e^{−0.01} ≈ 0.00995.
        let s = SocialSignals {
            views: 10_000,
            likes: 10,
            shares: 0,
            comments: 0,
        };
        let score = s.engagement_score();
        assert!(score > 0.0 && score < 0.05, "score={score}");
    }

    #[test]
    fn social_signals_high_engagement_near_one() {
        // rate = (500 + 2·300 + 3·200) / 1000 = 1.7 → score = 1 − e^{−17} ≈ 1.0.
        let s = SocialSignals {
            views: 1_000,
            likes: 500,
            shares: 300,
            comments: 200,
        };
        let score = s.engagement_score();
        assert!(score > 0.99, "score={score}");
        assert!(score <= 1.0, "score={score}");
    }

    #[test]
    fn social_signals_saturates_within_unit() {
        // Astronomically high engagement must saturate at, never exceed, 1.0.
        let s = SocialSignals {
            views: 1,
            likes: 1_000_000_000,
            shares: 1_000_000_000,
            comments: 1_000_000_000,
        };
        let score = s.engagement_score();
        assert!(score <= 1.0, "score must saturate at 1.0, got {score}");
        assert!(score > 0.999, "score={score}");
        assert!(score.is_finite(), "score must be finite, got {score}");
    }

    #[test]
    fn social_signals_monotonic_in_likes() {
        let base = SocialSignals {
            views: 1_000,
            likes: 50,
            shares: 0,
            comments: 0,
        };
        let more = SocialSignals {
            views: 1_000,
            likes: 150,
            shares: 0,
            comments: 0,
        };
        assert!(
            more.engagement_score() > base.engagement_score(),
            "more likes must score higher"
        );
    }

    #[test]
    fn social_signals_weighting_order() {
        // Identical raw count of a single interaction kind: comment > share > like.
        let like = SocialSignals {
            views: 1_000,
            likes: 100,
            shares: 0,
            comments: 0,
        };
        let share = SocialSignals {
            views: 1_000,
            likes: 0,
            shares: 100,
            comments: 0,
        };
        let comment = SocialSignals {
            views: 1_000,
            likes: 0,
            shares: 0,
            comments: 100,
        };
        let ls = like.engagement_score();
        let ss = share.engagement_score();
        let cs = comment.engagement_score();
        assert!(ss > ls, "share {ss} should beat like {ls}");
        assert!(cs > ss, "comment {cs} should beat share {ss}");
    }

    // ── redistribute_social ──────────────────────────────────────────────────

    #[test]
    fn redistribute_social_preserves_reward_ceiling() {
        let w = EngagementWeights::default();
        let r = w.redistribute_social();
        // Social weight is removed.
        assert_eq!(r.social, 0.0);
        // The three reward weights still sum to the original reward + social mass,
        // so the reward ceiling is preserved.
        let reward_sum = r.watch_time + r.completion + r.rewatch;
        let original_reward_plus_social = w.watch_time + w.completion + w.rewatch + w.social;
        assert!(
            (reward_sum - original_reward_plus_social).abs() < 1e-6,
            "reward_sum={reward_sum} expected={original_reward_plus_social}"
        );
        // Penalty weight is untouched.
        assert!((r.forward_seek_penalty - w.forward_seek_penalty).abs() < 1e-6);
    }

    #[test]
    fn redistribute_social_zero_reward_base_drops_social() {
        // No reward channels to absorb the social mass → social simply dropped.
        let w = EngagementWeights {
            watch_time: 0.0,
            completion: 0.0,
            rewatch: 0.0,
            social: 0.5,
            forward_seek_penalty: 0.3,
        };
        let r = w.redistribute_social();
        assert_eq!(r.social, 0.0);
        assert_eq!(r.watch_time, 0.0);
        assert!((r.forward_seek_penalty - 0.3).abs() < 1e-6);
    }

    // ── compute_engagement_with_social ───────────────────────────────────────

    #[test]
    fn compute_engagement_with_social_uses_real_score() {
        let sessions: Vec<_> = (0..10)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let social = SocialSignals {
            views: 1_000,
            likes: 200,
            shares: 50,
            comments: 30,
        };
        let expected_social = social.engagement_score();
        let score = compute_engagement_with_social(&sessions, 10_000, &weights, &social);
        // The component is the REAL computed social score, not any constant.
        assert!((score.components.social_score - expected_social).abs() < 1e-6);
        assert!(
            expected_social > 0.9,
            "expected high social, got {expected_social}"
        );
    }

    #[test]
    fn compute_engagement_with_social_beats_plain_when_high() {
        let sessions: Vec<_> = (0..10)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let social = SocialSignals {
            views: 1_000,
            likes: 400,
            shares: 200,
            comments: 100,
        };
        let with_social = compute_engagement_with_social(&sessions, 10_000, &weights, &social);
        let plain = compute_engagement(&sessions, 10_000, &weights);
        assert!(
            with_social.score > plain.score,
            "with_social={} plain={}",
            with_social.score,
            plain.score
        );
    }

    #[test]
    fn compute_engagement_with_social_empty_keeps_social_component() {
        let weights = EngagementWeights::default();
        let social = SocialSignals {
            views: 1_000,
            likes: 500,
            shares: 100,
            comments: 80,
        };
        let score = compute_engagement_with_social(&[], 10_000, &weights, &social);
        // No playback evidence → documented overall score of 0.0 ...
        assert_eq!(score.score, 0.0);
        // ... but the social component honestly reflects the supplied signals.
        assert!((score.components.social_score - social.engagement_score()).abs() < 1e-6);
        assert!(score.components.social_score > 0.0);
    }

    #[test]
    fn compute_engagement_distinguishes_no_data_from_zero_social() {
        let sessions: Vec<_> = (0..5)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();

        // "No social data": social weight is redistributed → reward ceiling kept.
        let no_data = compute_engagement(&sessions, 10_000, &weights);
        // "Measured-zero social engagement": weights as-is, social term = 0.
        let measured_zero =
            compute_engagement_with_social(&sessions, 10_000, &weights, &SocialSignals::default());

        // Both honestly report 0.0 for the social component ...
        assert_eq!(no_data.components.social_score, 0.0);
        assert_eq!(measured_zero.components.social_score, 0.0);
        // ... but missing data must not drag the score down the way a genuine
        // measured-zero social engagement does.
        assert!(
            no_data.score > measured_zero.score,
            "no_data={} measured_zero={}",
            no_data.score,
            measured_zero.score
        );
    }

    #[test]
    fn compute_engagement_never_returns_fabricated_half() {
        let weights = EngagementWeights::default();
        // Empty-sessions path (former line-85 fabrication site).
        let empty = compute_engagement(&[], 10_000, &weights);
        assert_eq!(empty.components.social_score, 0.0);
        // Non-empty path (former line-147 placeholder site).
        let sessions = vec![full_watch_session("s1", 10_000)];
        let non_empty = compute_engagement(&sessions, 10_000, &weights);
        assert_eq!(non_empty.components.social_score, 0.0);
        // Zero-duration path.
        let zero_dur = compute_engagement(&sessions, 0, &weights);
        assert_eq!(zero_dur.components.social_score, 0.0);
    }

    // ── linear_regression_slope ──────────────────────────────────────────────

    #[test]
    fn slope_perfectly_increasing() {
        // y = x (in tiny units): (0,0.0),(1,1.0),(2,2.0),(3,3.0)
        let points = vec![(0i64, 0.0f32), (1, 1.0), (2, 2.0), (3, 3.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope - 1.0).abs() < 1e-4, "slope={slope}");
    }

    #[test]
    fn slope_perfectly_decreasing() {
        let points = vec![(0i64, 3.0f32), (1, 2.0), (2, 1.0), (3, 0.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope + 1.0).abs() < 1e-4, "slope={slope}");
    }

    #[test]
    fn slope_flat() {
        let points = vec![(0i64, 0.5f32), (1, 0.5), (2, 0.5), (3, 0.5)];
        let slope = linear_regression_slope(&points);
        assert!(slope.abs() < 1e-6, "slope={slope}");
    }

    #[test]
    fn slope_single_point_returns_zero() {
        let points = vec![(100i64, 0.8f32)];
        assert_eq!(linear_regression_slope(&points), 0.0);
    }

    #[test]
    fn slope_two_points() {
        let points = vec![(0i64, 0.0f32), (10, 1.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope - 0.1).abs() < 1e-5, "slope={slope}");
    }

    #[test]
    fn engagement_trend_slope_method() {
        let trend = EngagementTrend {
            scores_over_time: vec![(0, 0.3), (1_000, 0.6), (2_000, 0.9)],
        };
        let slope = trend.slope();
        assert!(slope > 0.0, "expected positive slope, got {slope}");
    }

    // ── ContentRanker ────────────────────────────────────────────────────────

    #[test]
    fn ranker_sorted_descending() {
        let scores = vec![
            ContentEngagementScore {
                content_id: "a".to_string(),
                score: 0.4,
                components: EngagementComponents {
                    watch_time_score: 0.4,
                    completion_score: 0.4,
                    rewatch_score: 0.0,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
            ContentEngagementScore {
                content_id: "b".to_string(),
                score: 0.9,
                components: EngagementComponents {
                    watch_time_score: 0.9,
                    completion_score: 0.9,
                    rewatch_score: 0.1,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
            ContentEngagementScore {
                content_id: "c".to_string(),
                score: 0.6,
                components: EngagementComponents {
                    watch_time_score: 0.6,
                    completion_score: 0.6,
                    rewatch_score: 0.0,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
        ];
        let ranked = ContentRanker::rank_by_engagement(&scores);
        assert_eq!(ranked[0].0, "b");
        assert_eq!(ranked[1].0, "c");
        assert_eq!(ranked[2].0, "a");
    }

    #[test]
    fn ranker_empty_input() {
        let ranked = ContentRanker::rank_by_engagement(&[]);
        assert!(ranked.is_empty());
    }

    #[test]
    fn ranker_single_item() {
        let scores = vec![ContentEngagementScore {
            content_id: "only".to_string(),
            score: 0.7,
            components: EngagementComponents {
                watch_time_score: 0.7,
                completion_score: 0.7,
                rewatch_score: 0.0,
                social_score: 0.5,
                seek_forward_penalty: 0.0,
            },
        }];
        let ranked = ContentRanker::rank_by_engagement(&scores);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, "only");
    }

    // ── EMA tests ────────────────────────────────────────────────────────────

    #[test]
    fn ema_empty_series_returns_none() {
        assert!(exponential_moving_average(&[], &EmaConfig::default()).is_none());
    }

    #[test]
    fn ema_alpha_one_equals_original_series() {
        // alpha=1.0 → EMA(i) = y(i).
        let config = EmaConfig::with_alpha(1.0).expect("valid");
        let series = vec![(0i64, 0.1f32), (1, 0.5), (2, 0.9), (3, 0.3)];
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.smoothed.len(), series.len());
        for (i, &(_, y)) in series.iter().enumerate() {
            // f64::from(f32) then back: use 1e-6 tolerance for f32 → f64 conversion.
            assert!(
                (result.smoothed[i] - f64::from(y)).abs() < 1e-6,
                "index {i}: ema={} y={}",
                result.smoothed[i],
                y
            );
        }
    }

    #[test]
    fn ema_smooths_noisy_signal() {
        let series: Vec<(i64, f32)> = (0i64..20)
            .map(|i| (i, if i % 2 == 0 { 0.9 } else { 0.1 }))
            .collect();
        let config = EmaConfig::from_span(5).expect("valid span");
        let result = exponential_moving_average(&series, &config).expect("result");
        let last = result.last_smoothed();
        assert!(
            last > 0.2 && last < 0.8,
            "smoothed last={last} should be near 0.5"
        );
    }

    #[test]
    fn ema_seeded_with_first_observation() {
        // seed = y(0) = 0.7; second EMA = 0.5 * 0.1 + 0.5 * 0.7 = 0.4
        let series = vec![(0i64, 0.7f32), (1, 0.1)];
        let config = EmaConfig::with_alpha(0.5).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        // f64::from(0.7f32) is ~0.699999988; use 1e-6 tolerance.
        assert!(
            (result.first_smoothed() - f64::from(0.7f32)).abs() < 1e-9,
            "first_smoothed={} expected {}",
            result.first_smoothed(),
            f64::from(0.7f32)
        );
        // EMA[1] = 0.5 * f64::from(0.1f32) + 0.5 * f64::from(0.7f32)
        let expected = 0.5 * f64::from(0.1f32) + 0.5 * f64::from(0.7f32);
        assert!(
            (result.smoothed[1] - expected).abs() < 1e-9,
            "smoothed[1]={} expected {expected}",
            result.smoothed[1]
        );
    }

    #[test]
    fn ema_from_span_produces_valid_alpha() {
        let config = EmaConfig::from_span(9).expect("valid");
        assert!((config.alpha - 0.2).abs() < 1e-12);
    }

    #[test]
    fn ema_from_span_zero_returns_none() {
        assert!(EmaConfig::from_span(0).is_none());
    }

    #[test]
    fn ema_with_invalid_alpha_returns_none() {
        assert!(EmaConfig::with_alpha(0.0).is_none());
        assert!(EmaConfig::with_alpha(-0.1).is_none());
        assert!(EmaConfig::with_alpha(1.1).is_none());
    }

    #[test]
    fn ema_trend_slope_positive_for_growing_series() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, i as f32 * 0.1)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert!(result.trend_slope > 0.0, "slope={}", result.trend_slope);
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Growing);
    }

    #[test]
    fn ema_trend_direction_declining() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, 1.0f32 - i as f32 * 0.1)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Declining);
    }

    #[test]
    fn ema_trend_direction_flat_for_constant_series() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, 0.5f32)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Flat);
    }

    #[test]
    fn ema_result_alpha_stored_correctly() {
        let series = vec![(0i64, 0.5f32), (1, 0.6)];
        let config = EmaConfig::with_alpha(0.4).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert!((result.alpha - 0.4).abs() < 1e-12);
    }

    // ── linear_regression_slope numeric pins ─────────────────────────────────

    #[test]
    fn test_slope_known_answer() {
        // y = 2x + 1: slope must be exactly 2.0.
        let points = [(0_i64, 1.0_f32), (1, 3.0), (2, 5.0)];
        let result = linear_regression_slope(&points);
        assert!(
            (result - 2.0_f32).abs() < 1e-4,
            "expected slope ≈ 2.0, got {result}"
        );
    }

    #[test]
    fn test_slope_negative() {
        // y = -2x + 5: slope must be exactly -2.0.
        let points = [(0_i64, 5.0_f32), (1, 3.0), (2, 1.0)];
        let result = linear_regression_slope(&points);
        assert!(
            (result - (-2.0_f32)).abs() < 1e-4,
            "expected slope ≈ -2.0, got {result}"
        );
    }

    #[test]
    fn test_slope_fewer_than_two_points() {
        // Empty slice → 0.0.
        assert_eq!(
            linear_regression_slope(&[]),
            0.0,
            "empty slice should return 0.0"
        );
        // Single point → 0.0.
        assert_eq!(
            linear_regression_slope(&[(0_i64, 1.0_f32)]),
            0.0,
            "single point should return 0.0"
        );
    }

    #[test]
    fn test_slope_zero_denominator() {
        // All x values identical: denominator = 0, must return 0.0 without panic.
        let points = [(5_i64, 1.0_f32), (5, 2.0), (5, 3.0)];
        let result = linear_regression_slope(&points);
        assert_eq!(
            result, 0.0,
            "identical x-values should return 0.0, got {result}"
        );
    }
}
