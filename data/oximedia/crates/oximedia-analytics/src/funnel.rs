//! Funnel analysis, churn prediction, and viewer loyalty scoring.
//!
//! ## Funnel analysis
//!
//! Tracks viewer progression through ordered content milestones (e.g. intro →
//! first chapter → halfway → completion).  For each step the funnel measures
//! how many viewers reached it, the conversion rate from the previous step, and
//! the drop-off.
//!
//! ## Churn prediction
//!
//! Analyses a time-series of engagement scores for a viewer and predicts
//! whether they are at risk of churning based on a sustained decline pattern
//! detected via linear-regression slope thresholding.
//!
//! ## Viewer loyalty scoring
//!
//! Combines recency, frequency, and duration into a single composite score
//! in [0.0, 1.0] using configurable weights.

use crate::engagement::linear_regression_slope;
use crate::error::AnalyticsError;
use crate::session::{build_playback_map, ViewerSession};

// ─── Funnel analysis ──────────────────────────────────────────────────────────

/// A single milestone in a viewer progression funnel.
///
/// A milestone is reached when the viewer's playback map contains a `true`
/// entry at the given `position_ms`.
#[derive(Debug, Clone)]
pub struct FunnelMilestone {
    pub name: String,
    /// The content position (ms) that must be reached to pass this milestone.
    pub position_ms: u64,
}

/// One step in the computed funnel.
#[derive(Debug, Clone)]
pub struct FunnelStep {
    pub milestone_name: String,
    pub position_ms: u64,
    /// Number of viewers who reached this milestone.
    pub viewers_reached: u32,
    /// Conversion rate from the *previous* step (1.0 for the first step).
    pub conversion_from_prev: f32,
    /// Fraction of all session starters who reached this step.
    pub overall_rate: f32,
}

/// The result of a funnel analysis.
#[derive(Debug, Clone)]
pub struct FunnelResult {
    pub steps: Vec<FunnelStep>,
    pub total_starters: u32,
}

impl FunnelResult {
    /// Overall funnel completion rate: fraction reaching the last milestone.
    pub fn completion_rate(&self) -> f32 {
        self.steps.last().map(|s| s.overall_rate).unwrap_or(0.0)
    }

    /// Index of the step with the largest absolute drop-off (viewers lost).
    pub fn biggest_drop_step(&self) -> Option<usize> {
        if self.steps.len() < 2 {
            return None;
        }
        let mut max_drop = 0u32;
        let mut max_idx = 1usize;
        for i in 1..self.steps.len() {
            let drop = self.steps[i - 1]
                .viewers_reached
                .saturating_sub(self.steps[i].viewers_reached);
            if drop > max_drop {
                max_drop = drop;
                max_idx = i;
            }
        }
        Some(max_idx)
    }
}

/// Compute a viewer funnel from a slice of sessions.
///
/// `milestones` must be provided in ascending `position_ms` order.  Each
/// milestone is independent — a viewer can reach a later milestone without
/// having reached an earlier one (this models skip behaviour).
///
/// Returns an error if `milestones` is empty or `sessions` is empty.
pub fn compute_funnel(
    sessions: &[ViewerSession],
    milestones: &[FunnelMilestone],
    content_duration_ms: u64,
) -> Result<FunnelResult, AnalyticsError> {
    if sessions.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "funnel requires at least one session".to_string(),
        ));
    }
    if milestones.is_empty() {
        return Err(AnalyticsError::ConfigError(
            "funnel requires at least one milestone".to_string(),
        ));
    }

    // Pre-build playback maps.
    let maps: Vec<_> = sessions
        .iter()
        .map(|s| build_playback_map(s, content_duration_ms))
        .collect();

    let total_starters = sessions.len() as u32;

    let mut steps = Vec::with_capacity(milestones.len());
    let mut prev_viewers = total_starters;

    for milestone in milestones {
        let pos_sec = (milestone.position_ms / 1000) as usize;
        let viewers_reached = maps
            .iter()
            .filter(|m| m.positions_watched.get(pos_sec).copied().unwrap_or(false))
            .count() as u32;

        let conversion_from_prev = if prev_viewers == 0 {
            0.0
        } else {
            viewers_reached as f32 / prev_viewers as f32
        };
        let overall_rate = viewers_reached as f32 / total_starters as f32;

        steps.push(FunnelStep {
            milestone_name: milestone.name.clone(),
            position_ms: milestone.position_ms,
            viewers_reached,
            conversion_from_prev,
            overall_rate,
        });
        prev_viewers = viewers_reached;
    }

    Ok(FunnelResult {
        steps,
        total_starters,
    })
}

// ─── Churn prediction ─────────────────────────────────────────────────────────

/// Configuration for the churn prediction model.
#[derive(Debug, Clone)]
pub struct ChurnConfig {
    /// Minimum number of engagement data points required.
    pub min_data_points: usize,
    /// Slope threshold below which a viewer is classified as churning.
    /// Expressed in engagement-score-units per millisecond.
    pub decline_slope_threshold: f32,
    /// Minimum absolute engagement score below which a viewer is always at risk.
    pub low_engagement_threshold: f32,
}

impl Default for ChurnConfig {
    fn default() -> Self {
        Self {
            min_data_points: 3,
            decline_slope_threshold: -1e-9, // negative slope in score/ms
            low_engagement_threshold: 0.2,
        }
    }
}

/// Churn risk classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChurnRisk {
    /// Viewer shows stable or growing engagement.
    Low,
    /// Viewer shows modest decline; monitor closely.
    Medium,
    /// Viewer shows strong decline or very low engagement; likely to churn.
    High,
}

/// Result of a churn risk assessment for a single viewer.
#[derive(Debug, Clone)]
pub struct ChurnAssessment {
    pub viewer_id: String,
    pub risk: ChurnRisk,
    /// Computed slope of the engagement score time-series (score/ms).
    pub engagement_slope: f32,
    /// Most recent engagement score.
    pub latest_score: f32,
}

/// Predict churn risk for a viewer from their engagement score time-series.
///
/// `scores_over_time` is a series of `(unix_epoch_ms, engagement_score)` pairs.
/// Scores should be in [0.0, 1.0].
///
/// Returns an error if there are fewer than `config.min_data_points` data points.
pub fn predict_churn(
    viewer_id: &str,
    scores_over_time: &[(i64, f32)],
    config: &ChurnConfig,
) -> Result<ChurnAssessment, AnalyticsError> {
    if scores_over_time.len() < config.min_data_points {
        return Err(AnalyticsError::InsufficientData(format!(
            "churn prediction requires at least {} data points, got {}",
            config.min_data_points,
            scores_over_time.len()
        )));
    }

    let slope = linear_regression_slope(scores_over_time);
    let latest_score = scores_over_time
        .iter()
        .max_by_key(|(t, _)| *t)
        .map(|(_, s)| *s)
        .unwrap_or(0.0);

    let risk = if latest_score < config.low_engagement_threshold {
        ChurnRisk::High
    } else if slope < config.decline_slope_threshold * 2.0 {
        ChurnRisk::High
    } else if slope < config.decline_slope_threshold {
        ChurnRisk::Medium
    } else {
        ChurnRisk::Low
    };

    Ok(ChurnAssessment {
        viewer_id: viewer_id.to_string(),
        risk,
        engagement_slope: slope,
        latest_score,
    })
}

// ─── Viewer loyalty scoring ───────────────────────────────────────────────────

/// Weights for the recency-frequency-duration loyalty model.
#[derive(Debug, Clone)]
pub struct LoyaltyWeights {
    /// Weight for recency component (how recently did they watch).
    pub recency: f32,
    /// Weight for frequency component (how often do they watch).
    pub frequency: f32,
    /// Weight for duration component (how long do they watch per session).
    pub duration: f32,
}

impl Default for LoyaltyWeights {
    fn default() -> Self {
        Self {
            recency: 0.35,
            frequency: 0.35,
            duration: 0.30,
        }
    }
}

/// Decomposed loyalty score components.
#[derive(Debug, Clone)]
pub struct LoyaltyComponents {
    /// Recency score: 1.0 if viewed within `recency_window_ms`, decaying to 0.
    pub recency_score: f32,
    /// Frequency score: normalised session count (capped at 1.0).
    pub frequency_score: f32,
    /// Duration score: avg watch duration relative to `max_duration_ms`.
    pub duration_score: f32,
}

/// Final loyalty assessment for a viewer.
#[derive(Debug, Clone)]
pub struct LoyaltyScore {
    pub viewer_id: String,
    /// Composite loyalty score in [0.0, 1.0].
    pub score: f32,
    pub components: LoyaltyComponents,
}

/// Compute a loyalty score for a viewer from their session history.
///
/// # Parameters
///
/// * `viewer_id`         — identifier for the viewer.
/// * `session_starts_ms` — Unix epoch ms timestamps of all their sessions.
/// * `watch_durations_ms`— Watch duration in ms for each session (parallel to
///   `session_starts_ms`).
/// * `now_ms`            — Current wall-clock time (epoch ms), used for recency.
/// * `recency_window_ms` — Viewing within this window scores full recency.
/// * `freq_cap`          — Session count at which frequency score is capped at 1.0.
/// * `max_duration_ms`   — Watch duration at which duration score is capped at 1.0.
/// * `weights`           — Component weights (should sum to 1.0).
///
/// Returns an error if `session_starts_ms` and `watch_durations_ms` have
/// different lengths.
pub fn compute_loyalty(
    viewer_id: &str,
    session_starts_ms: &[i64],
    watch_durations_ms: &[u64],
    now_ms: i64,
    recency_window_ms: i64,
    freq_cap: usize,
    max_duration_ms: u64,
    weights: &LoyaltyWeights,
) -> Result<LoyaltyScore, AnalyticsError> {
    if session_starts_ms.len() != watch_durations_ms.len() {
        return Err(AnalyticsError::ConfigError(
            "session_starts_ms and watch_durations_ms must have equal length".to_string(),
        ));
    }

    // Recency: time since last session, normalised against recency_window_ms.
    let recency_score = if session_starts_ms.is_empty() {
        0.0f32
    } else {
        let last_ms = session_starts_ms.iter().copied().max().unwrap_or(0);
        let age_ms = (now_ms - last_ms).max(0) as f64;
        let window = recency_window_ms.max(1) as f64;
        (1.0 - (age_ms / window).min(1.0)) as f32
    };

    // Frequency: number of sessions normalised to freq_cap.
    let frequency_score = if freq_cap == 0 {
        0.0f32
    } else {
        (session_starts_ms.len() as f32 / freq_cap as f32).min(1.0)
    };

    // Duration: average watch time normalised to max_duration_ms.
    let duration_score = if watch_durations_ms.is_empty() || max_duration_ms == 0 {
        0.0f32
    } else {
        let avg_dur: f64 =
            watch_durations_ms.iter().sum::<u64>() as f64 / watch_durations_ms.len() as f64;
        (avg_dur / max_duration_ms as f64).min(1.0) as f32
    };

    let score = (weights.recency * recency_score
        + weights.frequency * frequency_score
        + weights.duration * duration_score)
        .min(1.0)
        .max(0.0);

    Ok(LoyaltyScore {
        viewer_id: viewer_id.to_string(),
        score,
        components: LoyaltyComponents {
            recency_score,
            frequency_score,
            duration_score,
        },
    })
}

// ─── Event-driven funnel analysis ────────────────────────────────────────────

/// A raw session event used for event-driven funnel analysis.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    /// Unique user identifier.
    pub user_id: String,
    /// Event type name (e.g. `"page_view"`, `"add_to_cart"`, `"purchase"`).
    pub event_type: String,
    /// Unix epoch milliseconds when the event occurred.
    pub timestamp_ms: u64,
}

/// One step in a funnel definition.
#[derive(Debug, Clone)]
pub struct FunnelStepDef {
    /// Human-readable name for this step.
    pub name: String,
    /// The `event_type` that constitutes completion of this step.
    pub event_type: String,
}

/// Defines an ordered sequence of steps that constitute a conversion funnel.
#[derive(Debug, Clone)]
pub struct FunnelDefinition {
    /// Ordered steps; users must complete them in order.
    pub steps: Vec<FunnelStepDef>,
    /// Maximum time allowed between consecutive steps (ms).
    /// If a user takes longer than this between any two steps, the funnel
    /// sequence resets from the beginning.
    pub max_time_between_steps_ms: u64,
}

/// Output of a funnel analysis.
#[derive(Debug, Clone)]
pub struct FunnelReport {
    /// Number of users who completed each step (`step_completions[i]` for step i).
    /// Length equals the number of steps in the definition.
    pub step_completions: Vec<u64>,
    /// Conversion rate from the previous step to this step (1.0 for step 0).
    /// `conversion_rates[i] = step_completions[i] / step_completions[i-1]`.
    pub conversion_rates: Vec<f64>,
    /// Drop-off rate at each step (`1.0 - conversion_rates[i]`; 0.0 for step 0).
    pub drop_offs: Vec<f64>,
}

impl FunnelReport {
    /// Overall completion rate: fraction of users who reached the final step
    /// relative to those who reached step 0.
    pub fn overall_completion_rate(&self) -> f64 {
        let first = self.step_completions.first().copied().unwrap_or(0);
        let last = self.step_completions.last().copied().unwrap_or(0);
        if first == 0 {
            0.0
        } else {
            last as f64 / first as f64
        }
    }
}

/// Analyses event-driven funnels from raw `SessionEvent` streams.
pub struct FunnelAnalyzer;

impl FunnelAnalyzer {
    /// Analyse `sessions` against `definition` and return a [`FunnelReport`].
    ///
    /// For each user, the algorithm attempts to walk through the funnel steps
    /// in order.  A step is completed when the user fires an event of the
    /// required `event_type` after completing the previous step AND within
    /// `max_time_between_steps_ms`.
    ///
    /// Returns an empty report (all zeros) if `definition.steps` is empty.
    pub fn analyze(sessions: &[SessionEvent], definition: &FunnelDefinition) -> FunnelReport {
        let n_steps = definition.steps.len();
        if n_steps == 0 {
            return FunnelReport {
                step_completions: Vec::new(),
                conversion_rates: Vec::new(),
                drop_offs: Vec::new(),
            };
        }

        let mut step_completions = vec![0u64; n_steps];

        // Group events by user_id, sorted by timestamp.
        let mut by_user: std::collections::HashMap<&str, Vec<&SessionEvent>> =
            std::collections::HashMap::new();
        for ev in sessions {
            by_user.entry(ev.user_id.as_str()).or_default().push(ev);
        }
        for events in by_user.values_mut() {
            events.sort_by_key(|e| e.timestamp_ms);
        }

        for events in by_user.values() {
            // Walk through the funnel steps for this user.
            let mut step_idx = 0usize;
            let mut last_step_ts: Option<u64> = None;

            for ev in events.iter() {
                if step_idx >= n_steps {
                    break;
                }
                let required = &definition.steps[step_idx].event_type;
                if ev.event_type != *required {
                    continue;
                }
                // Check time window constraint (applies from step 1 onwards).
                if let Some(prev_ts) = last_step_ts {
                    if ev.timestamp_ms.saturating_sub(prev_ts)
                        > definition.max_time_between_steps_ms
                    {
                        // Timed out; restart from step 0.
                        step_idx = 0;
                        last_step_ts = None;
                        // Check if this event matches step 0.
                        if ev.event_type == definition.steps[0].event_type {
                            step_completions[0] += 1;
                            step_idx = 1;
                            last_step_ts = Some(ev.timestamp_ms);
                        }
                        continue;
                    }
                }
                step_completions[step_idx] += 1;
                step_idx += 1;
                last_step_ts = Some(ev.timestamp_ms);
            }
        }

        // Compute conversion rates and drop-offs.
        let mut conversion_rates = vec![0f64; n_steps];
        let mut drop_offs = vec![0f64; n_steps];
        conversion_rates[0] = 1.0;
        drop_offs[0] = 0.0;
        for i in 1..n_steps {
            let prev = step_completions[i - 1];
            conversion_rates[i] = if prev == 0 {
                0.0
            } else {
                step_completions[i] as f64 / prev as f64
            };
            drop_offs[i] = 1.0 - conversion_rates[i];
        }

        FunnelReport {
            step_completions,
            conversion_rates,
            drop_offs,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn watch_session(id: &str, end_ms: u64, duration_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "c1".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: end_ms,
                    watch_duration_ms: duration_ms,
                },
            ],
        }
    }

    // ── compute_funnel ───────────────────────────────────────────────────────

    #[test]
    fn funnel_all_viewers_reach_all_milestones() {
        let sessions = vec![
            watch_session("s1", 10_000, 10_000),
            watch_session("s2", 10_000, 10_000),
        ];
        let milestones = vec![
            FunnelMilestone {
                name: "start".to_string(),
                position_ms: 0,
            },
            FunnelMilestone {
                name: "mid".to_string(),
                position_ms: 5_000,
            },
            FunnelMilestone {
                name: "end".to_string(),
                position_ms: 9_000,
            },
        ];
        let result =
            compute_funnel(&sessions, &milestones, 10_000).expect("compute funnel should succeed");
        assert_eq!(result.steps.len(), 3);
        assert_eq!(result.total_starters, 2);
        // All 2 viewers should reach every milestone.
        assert_eq!(result.steps[2].viewers_reached, 2);
        assert!((result.completion_rate() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn funnel_dropout_midway() {
        // s1 watches 0-5s; s2 and s3 watch 0-10s.
        let sessions = vec![
            watch_session("s1", 5_000, 5_000),
            watch_session("s2", 10_000, 10_000),
            watch_session("s3", 10_000, 10_000),
        ];
        let milestones = vec![
            FunnelMilestone {
                name: "intro".to_string(),
                position_ms: 1_000,
            },
            FunnelMilestone {
                name: "end".to_string(),
                position_ms: 9_000,
            },
        ];
        let result =
            compute_funnel(&sessions, &milestones, 10_000).expect("compute funnel should succeed");
        assert_eq!(result.steps[0].viewers_reached, 3);
        assert_eq!(result.steps[1].viewers_reached, 2);
        let biggest = result
            .biggest_drop_step()
            .expect("biggest drop step should succeed");
        assert_eq!(biggest, 1);
    }

    #[test]
    fn funnel_empty_sessions_returns_error() {
        let milestones = vec![FunnelMilestone {
            name: "start".to_string(),
            position_ms: 0,
        }];
        assert!(compute_funnel(&[], &milestones, 10_000).is_err());
    }

    #[test]
    fn funnel_empty_milestones_returns_error() {
        let sessions = vec![watch_session("s1", 10_000, 10_000)];
        assert!(compute_funnel(&sessions, &[], 10_000).is_err());
    }

    // ── predict_churn ────────────────────────────────────────────────────────

    #[test]
    fn churn_high_risk_strong_decline() {
        // Strongly declining scores.
        let scores: Vec<(i64, f32)> = (0..10)
            .map(|i| (i as i64 * 7 * 86_400_000, 1.0 - i as f32 * 0.09))
            .collect();
        let config = ChurnConfig::default();
        let result =
            predict_churn("viewer1", &scores, &config).expect("predict churn should succeed");
        // Slope is very negative → at least medium risk.
        assert_ne!(result.risk, ChurnRisk::Low);
    }

    #[test]
    fn churn_low_risk_growing_engagement() {
        let scores: Vec<(i64, f32)> = (0..8)
            .map(|i| (i as i64 * 86_400_000, 0.3 + i as f32 * 0.05))
            .collect();
        let config = ChurnConfig::default();
        let result =
            predict_churn("viewer2", &scores, &config).expect("predict churn should succeed");
        assert_eq!(result.risk, ChurnRisk::Low);
    }

    #[test]
    fn churn_insufficient_data_returns_error() {
        let scores = vec![(0i64, 0.5f32), (1, 0.4)]; // only 2 points
        let config = ChurnConfig {
            min_data_points: 3,
            ..Default::default()
        };
        assert!(predict_churn("v", &scores, &config).is_err());
    }

    #[test]
    fn churn_low_engagement_always_high_risk() {
        let scores: Vec<(i64, f32)> = (0..5)
            .map(|i| (i as i64 * 86_400_000, 0.05)) // very low but flat
            .collect();
        let config = ChurnConfig::default();
        let result =
            predict_churn("v_low", &scores, &config).expect("predict churn should succeed");
        assert_eq!(result.risk, ChurnRisk::High);
    }

    // ── compute_loyalty ──────────────────────────────────────────────────────

    #[test]
    fn loyalty_perfect_viewer() {
        let now_ms = 10 * 86_400_000i64; // 10 days in
                                         // Watched 10 times within the last day; each session 30 min.
        let starts: Vec<i64> = (0..10).map(|i| now_ms - i * 3_600_000).collect();
        let durations = vec![1_800_000u64; 10]; // 30 min
        let weights = LoyaltyWeights::default();
        let score = compute_loyalty(
            "v1",
            &starts,
            &durations,
            now_ms,
            7 * 86_400_000,
            10,
            3_600_000,
            &weights,
        )
        .expect("value should be present should succeed");
        assert!(
            score.score > 0.8,
            "expected high loyalty, got {}",
            score.score
        );
        assert!(score.components.recency_score > 0.95);
    }

    #[test]
    fn loyalty_churned_viewer() {
        let now_ms = 100 * 86_400_000i64;
        // Last watched 60 days ago, only 1 session.
        let starts = vec![now_ms - 60 * 86_400_000];
        let durations = vec![60_000u64]; // 1 minute
        let weights = LoyaltyWeights::default();
        let score = compute_loyalty(
            "v2",
            &starts,
            &durations,
            now_ms,
            7 * 86_400_000,
            20,
            3_600_000,
            &weights,
        )
        .expect("value should be present should succeed");
        assert!(
            score.score < 0.3,
            "expected low loyalty, got {}",
            score.score
        );
        assert_eq!(score.components.recency_score, 0.0);
    }

    #[test]
    fn loyalty_mismatched_lengths_error() {
        let result = compute_loyalty(
            "v",
            &[0i64, 1],
            &[1000u64],
            1000,
            86_400_000,
            10,
            3_600_000,
            &LoyaltyWeights::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn loyalty_empty_sessions() {
        let score = compute_loyalty(
            "v_new",
            &[],
            &[],
            0,
            86_400_000,
            10,
            3_600_000,
            &LoyaltyWeights::default(),
        )
        .expect("value should be present should succeed");
        assert_eq!(score.score, 0.0);
    }

    // ── FunnelAnalyzer ────────────────────────────────────────────────────────

    fn make_def(steps: &[(&str, &str)], max_gap_ms: u64) -> FunnelDefinition {
        FunnelDefinition {
            steps: steps
                .iter()
                .map(|(name, ev)| FunnelStepDef {
                    name: name.to_string(),
                    event_type: ev.to_string(),
                })
                .collect(),
            max_time_between_steps_ms: max_gap_ms,
        }
    }

    fn ev(user: &str, event_type: &str, ts: u64) -> SessionEvent {
        SessionEvent {
            user_id: user.to_string(),
            event_type: event_type.to_string(),
            timestamp_ms: ts,
        }
    }

    #[test]
    fn funnel_analyzer_empty_sessions() {
        let def = make_def(&[("view", "view")], 60_000);
        let report = FunnelAnalyzer::analyze(&[], &def);
        assert_eq!(report.step_completions, vec![0]);
        assert_eq!(report.conversion_rates, vec![1.0]);
        assert_eq!(report.drop_offs, vec![0.0]);
    }

    #[test]
    fn funnel_analyzer_empty_steps_returns_empty_report() {
        let def = FunnelDefinition {
            steps: vec![],
            max_time_between_steps_ms: 60_000,
        };
        let report = FunnelAnalyzer::analyze(&[ev("u1", "view", 0)], &def);
        assert!(report.step_completions.is_empty());
        assert!(report.conversion_rates.is_empty());
        assert!(report.drop_offs.is_empty());
    }

    #[test]
    fn funnel_analyzer_single_step_single_user() {
        let def = make_def(&[("view", "view")], 60_000);
        let events = vec![ev("u1", "view", 1000)];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 1);
        assert_eq!(report.conversion_rates[0], 1.0);
        assert_eq!(report.drop_offs[0], 0.0);
    }

    #[test]
    fn funnel_analyzer_full_conversion_two_steps() {
        let def = make_def(&[("view", "view"), ("purchase", "purchase")], 300_000);
        let events = vec![
            ev("u1", "view", 0),
            ev("u1", "purchase", 10_000),
            ev("u2", "view", 0),
            ev("u2", "purchase", 20_000),
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 2);
        assert_eq!(report.step_completions[1], 2);
        assert!((report.conversion_rates[1] - 1.0).abs() < 1e-9);
        assert!(report.drop_offs[1].abs() < 1e-9);
    }

    #[test]
    fn funnel_analyzer_partial_conversion() {
        let def = make_def(&[("view", "view"), ("purchase", "purchase")], 300_000);
        let events = vec![
            ev("u1", "view", 0),
            ev("u1", "purchase", 5_000),
            ev("u2", "view", 0),
            // u2 does not purchase
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 2);
        assert_eq!(report.step_completions[1], 1);
        assert!((report.conversion_rates[1] - 0.5).abs() < 1e-9);
        assert!((report.drop_offs[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn funnel_analyzer_time_window_exceeded_resets() {
        let def = make_def(
            &[("view", "view"), ("purchase", "purchase")],
            5_000, // only 5 seconds allowed
        );
        let events = vec![
            ev("u1", "view", 0),
            ev("u1", "purchase", 100_000), // 100s later → reset
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        // u1 completed step 0 but not step 1 (time exceeded).
        assert_eq!(report.step_completions[0], 1);
        assert_eq!(report.step_completions[1], 0);
    }

    #[test]
    fn funnel_analyzer_three_step_funnel() {
        let def = make_def(
            &[
                ("view", "view"),
                ("cart", "add_to_cart"),
                ("purchase", "purchase"),
            ],
            600_000,
        );
        let events = vec![
            ev("u1", "view", 0),
            ev("u1", "add_to_cart", 5_000),
            ev("u1", "purchase", 10_000),
            ev("u2", "view", 0),
            ev("u2", "add_to_cart", 5_000),
            // u2 stops here
            ev("u3", "view", 0),
            // u3 stops at view
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 3);
        assert_eq!(report.step_completions[1], 2);
        assert_eq!(report.step_completions[2], 1);
    }

    #[test]
    fn funnel_analyzer_conversion_rates_sum_correctly() {
        let def = make_def(&[("a", "a"), ("b", "b"), ("c", "c")], 60_000);
        let events = vec![
            ev("u1", "a", 0),
            ev("u1", "b", 1_000),
            ev("u1", "c", 2_000),
            ev("u2", "a", 0),
            ev("u2", "b", 1_000),
            ev("u3", "a", 0),
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        // step 0: 3 users; step 1: 2; step 2: 1
        assert_eq!(report.step_completions[0], 3);
        assert!((report.conversion_rates[1] - 2.0 / 3.0).abs() < 1e-9);
        assert!((report.conversion_rates[2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn funnel_analyzer_overall_completion_rate() {
        let def = make_def(&[("a", "a"), ("b", "b")], 60_000);
        let events = vec![ev("u1", "a", 0), ev("u1", "b", 1_000), ev("u2", "a", 0)];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert!((report.overall_completion_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn funnel_analyzer_irrelevant_events_ignored() {
        let def = make_def(&[("view", "view"), ("buy", "purchase")], 60_000);
        let events = vec![
            ev("u1", "view", 0),
            ev("u1", "click", 1_000),  // irrelevant
            ev("u1", "scroll", 2_000), // irrelevant
            ev("u1", "purchase", 3_000),
        ];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 1);
        assert_eq!(report.step_completions[1], 1);
    }

    #[test]
    fn funnel_analyzer_multiple_users_independent() {
        let def = make_def(&[("start", "start"), ("end", "end")], 120_000);
        let mut events = Vec::new();
        for i in 0..10u64 {
            events.push(ev(&format!("u{i}"), "start", i * 1000));
            if i % 2 == 0 {
                events.push(ev(&format!("u{i}"), "end", i * 1000 + 500));
            }
        }
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.step_completions[0], 10);
        assert_eq!(report.step_completions[1], 5);
    }

    #[test]
    fn funnel_analyzer_step0_first_conversion_rate_always_one() {
        let def = make_def(&[("x", "x"), ("y", "y")], 60_000);
        let events = vec![ev("u1", "x", 0)];
        let report = FunnelAnalyzer::analyze(&events, &def);
        assert_eq!(report.conversion_rates[0], 1.0);
        assert_eq!(report.drop_offs[0], 0.0);
    }

    #[test]
    fn funnel_analyzer_no_step0_users_conversion_rate_is_zero() {
        let def = make_def(&[("x", "x"), ("y", "y")], 60_000);
        // No events at all for step 0.
        let events = vec![ev("u1", "y", 0)];
        let report = FunnelAnalyzer::analyze(&events, &def);
        // step 0 completions = 0; step 1 completions = 0.
        assert_eq!(report.step_completions[0], 0);
        assert_eq!(report.conversion_rates[1], 0.0);
    }
}
