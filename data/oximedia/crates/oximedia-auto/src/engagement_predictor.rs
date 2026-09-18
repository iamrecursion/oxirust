//! Engagement prediction using interest curve analysis for audience retention.
//!
//! Predicts how engaging a video sequence will be by modelling the **interest
//! curve** — the probability that a viewer is still watching at each moment.
//! The module draws on:
//!
//! - **Retention curve modelling**: exponential decay baseline with
//!   interest-driven boosts and attention-valley penalties.
//! - **Drop-off risk scoring**: identifies timestamps where retention is
//!   predicted to fall sharply.
//! - **Hook scoring**: rates the first few seconds for immediate engagement.
//! - **Re-engagement detection**: finds climactic moments that recapture
//!   lost attention.
//! - **Pacing recommendations**: suggests where to trim or speed up to
//!   maintain engagement.
//!
//! All computations are purely mathematical — no network or ML model is needed.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::engagement_predictor::{EngagementPredictor, PredictorConfig};
//!
//! let config = PredictorConfig::default();
//! let predictor = EngagementPredictor::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::{InterestCurve, ScoredScene};
use oximedia_core::Timestamp;

// ─── Data types ───────────────────────────────────────────────────────────────

/// A single point on the predicted retention curve.
#[derive(Debug, Clone, Copy)]
pub struct RetentionPoint {
    /// Timestamp in the video (ms timebase).
    pub timestamp: Timestamp,
    /// Predicted fraction of the audience still watching (0.0 – 1.0).
    pub retention: f64,
    /// Local engagement score at this moment (0.0 – 1.0).
    pub engagement: f64,
}

/// A predicted drop-off risk zone.
#[derive(Debug, Clone)]
pub struct DropOffZone {
    /// Start of the risk window.
    pub start: Timestamp,
    /// End of the risk window.
    pub end: Timestamp,
    /// Severity of the predicted drop (0.0 – 1.0; higher = worse).
    pub severity: f64,
    /// Human-readable reason.
    pub reason: String,
}

impl DropOffZone {
    /// Duration of this zone in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }
}

/// A moment where the model predicts re-engagement (a "hook" moment).
#[derive(Debug, Clone)]
pub struct ReEngagementPoint {
    /// Timestamp of the peak.
    pub timestamp: Timestamp,
    /// Estimated engagement boost (0.0 – 1.0).
    pub boost: f64,
    /// Nature of the re-engagement signal.
    pub signal_type: ReEngagementSignal,
}

/// Kind of signal that drives predicted re-engagement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReEngagementSignal {
    /// Sudden increase in motion / action.
    ActionSpike,
    /// Audio peak (applause, music drop, etc.).
    AudioPeak,
    /// High-scoring composite scene.
    HighImportanceScene,
    /// Narrative climax detected from position in content arc.
    NarrativeClimax,
}

impl ReEngagementSignal {
    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ActionSpike => "Action spike",
            Self::AudioPeak => "Audio peak",
            Self::HighImportanceScene => "High-importance scene",
            Self::NarrativeClimax => "Narrative climax",
        }
    }
}

/// Full engagement prediction result for a video.
#[derive(Debug, Clone)]
pub struct EngagementPrediction {
    /// Predicted retention curve (one point per scene boundary).
    pub retention_curve: Vec<RetentionPoint>,
    /// Zones where retention is expected to drop sharply.
    pub drop_off_zones: Vec<DropOffZone>,
    /// Moments predicted to recapture viewer attention.
    pub re_engagement_points: Vec<ReEngagementPoint>,
    /// Predicted average audience retention (0.0 – 1.0) across the full clip.
    pub average_retention: f64,
    /// Hook score for the first `hook_window_ms` milliseconds (0.0 – 1.0).
    pub hook_score: f64,
    /// Predicted overall engagement score (0.0 – 1.0).
    pub overall_engagement: f64,
}

impl EngagementPrediction {
    /// Return the worst drop-off zone by severity, if any.
    #[must_use]
    pub fn worst_drop_off(&self) -> Option<&DropOffZone> {
        self.drop_off_zones.iter().max_by(|a, b| {
            a.severity
                .partial_cmp(&b.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return all drop-off zones above the given severity threshold.
    #[must_use]
    pub fn critical_drop_offs(&self, severity_threshold: f64) -> Vec<&DropOffZone> {
        self.drop_off_zones
            .iter()
            .filter(|z| z.severity >= severity_threshold)
            .collect()
    }

    /// Retention at the last scene (i.e. fraction of viewers who watched to the end).
    #[must_use]
    pub fn completion_rate(&self) -> f64 {
        self.retention_curve.last().map_or(0.0, |p| p.retention)
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the engagement predictor.
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    /// Base exponential decay constant per second of video (λ in e^{−λt}).
    /// Typical YouTube retention: ≈ 0.015 / s → 50 % at ~46 s.
    pub base_decay_per_second: f64,
    /// Weight of the local scene importance score in counteracting decay.
    pub engagement_boost_weight: f64,
    /// Threshold below which a retention drop is classified as a "drop-off zone".
    pub drop_off_threshold: f64,
    /// Minimum severity for a drop-off zone to be reported.
    pub min_drop_off_severity: f64,
    /// Length of the "hook" window at the start of the video (ms).
    pub hook_window_ms: i64,
    /// Minimum importance score for a scene to trigger a re-engagement point.
    pub re_engagement_min_score: f64,
    /// Smoothing window (number of scenes) for the interest curve.
    pub smoothing_window: usize,
    /// Penalty factor for a scene with importance below this threshold.
    pub low_interest_penalty: f64,
    /// Low interest threshold below which a scene increases drop-off risk.
    pub low_interest_threshold: f64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            base_decay_per_second: 0.015,
            engagement_boost_weight: 0.60,
            drop_off_threshold: 0.30,
            min_drop_off_severity: 0.20,
            hook_window_ms: 15_000, // first 15 seconds
            re_engagement_min_score: 0.70,
            smoothing_window: 3,
            low_interest_penalty: 0.03,
            low_interest_threshold: 0.35,
        }
    }
}

impl PredictorConfig {
    /// Create a config tuned for short-form content (Shorts / Reels / TikTok).
    #[must_use]
    pub fn short_form() -> Self {
        Self {
            base_decay_per_second: 0.025, // faster decay on short content
            hook_window_ms: 3_000,
            re_engagement_min_score: 0.60,
            ..Self::default()
        }
    }

    /// Create a config tuned for long-form (movies / documentaries).
    #[must_use]
    pub fn long_form() -> Self {
        Self {
            base_decay_per_second: 0.008, // slower decay for long form
            hook_window_ms: 60_000,       // 1-minute hook
            re_engagement_min_score: 0.75,
            ..Self::default()
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> AutoResult<()> {
        if self.base_decay_per_second < 0.0 {
            return Err(AutoError::InvalidParameter {
                name: "base_decay_per_second".into(),
                value: "must be non-negative".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.engagement_boost_weight) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.engagement_boost_weight,
                min: 0.0,
                max: 1.0,
            });
        }
        if !(0.0..=1.0).contains(&self.drop_off_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.drop_off_threshold,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.hook_window_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.hook_window_ms,
            });
        }
        if self.smoothing_window == 0 {
            return Err(AutoError::InvalidParameter {
                name: "smoothing_window".into(),
                value: "must be at least 1".into(),
            });
        }
        Ok(())
    }
}

// ─── Predictor ────────────────────────────────────────────────────────────────

/// Engagement and retention predictor.
///
/// Given a sequence of scored scenes, models the audience retention curve
/// and identifies opportunities to improve engagement.
pub struct EngagementPredictor {
    config: PredictorConfig,
}

impl EngagementPredictor {
    /// Create a new predictor with the given config.
    #[must_use]
    pub fn new(config: PredictorConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_predictor() -> Self {
        Self::new(PredictorConfig::default())
    }

    /// Predict engagement for a sequence of scored scenes.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid or insufficient data is given.
    pub fn predict(&self, scenes: &[ScoredScene]) -> AutoResult<EngagementPrediction> {
        self.config.validate()?;

        if scenes.is_empty() {
            return Err(AutoError::insufficient_data(
                "No scenes provided for engagement prediction",
            ));
        }

        let smoothed_scores = self.smooth_scores(scenes);
        let retention_curve = self.build_retention_curve(scenes, &smoothed_scores);
        let drop_off_zones = self.detect_drop_off_zones(scenes, &retention_curve);
        let re_engagement_points = self.detect_re_engagement(scenes, &smoothed_scores);
        let hook_score = self.compute_hook_score(scenes, &smoothed_scores);

        let average_retention = if retention_curve.is_empty() {
            0.0
        } else {
            retention_curve.iter().map(|p| p.retention).sum::<f64>() / retention_curve.len() as f64
        };

        // Overall engagement: weighted blend of hook, average retention, and
        // re-engagement density.
        let re_eng_density = if scenes.is_empty() {
            0.0
        } else {
            (re_engagement_points.len() as f64 / scenes.len() as f64).min(1.0)
        };
        let overall_engagement =
            (hook_score * 0.30 + average_retention * 0.50 + re_eng_density * 0.20).clamp(0.0, 1.0);

        Ok(EngagementPrediction {
            retention_curve,
            drop_off_zones,
            re_engagement_points,
            average_retention,
            hook_score,
            overall_engagement,
        })
    }

    /// Build a predicted retention curve using exponential decay modulated by
    /// per-scene engagement.
    fn build_retention_curve(
        &self,
        scenes: &[ScoredScene],
        smoothed: &[f64],
    ) -> Vec<RetentionPoint> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut curve = Vec::with_capacity(scenes.len());
        let mut retention = 1.0f64;

        for (i, scene) in scenes.iter().enumerate() {
            let duration_s = scene.duration() as f64 / 1000.0;
            let score = smoothed.get(i).copied().unwrap_or(scene.adjusted_score());

            // Engagement modulates the decay: high-interest scenes slow drop-off
            let effective_decay = if score >= self.config.low_interest_threshold {
                self.config.base_decay_per_second
                    * (1.0 - score * self.config.engagement_boost_weight)
            } else {
                // Low-interest scenes add a penalty on top of base decay
                self.config.base_decay_per_second
                    + self.config.low_interest_penalty
                        * (self.config.low_interest_threshold - score)
            };

            let effective_decay = effective_decay.max(0.0);
            retention *= (-effective_decay * duration_s).exp();
            retention = retention.clamp(0.0, 1.0);

            curve.push(RetentionPoint {
                timestamp: Timestamp::new(scene.start.pts, timebase),
                retention,
                engagement: score,
            });
        }

        curve
    }

    /// Detect zones where the retention curve is expected to drop sharply.
    fn detect_drop_off_zones(
        &self,
        scenes: &[ScoredScene],
        curve: &[RetentionPoint],
    ) -> Vec<DropOffZone> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut zones = Vec::new();

        if curve.len() < 2 {
            return zones;
        }

        // Find windows where:
        //   (a) current retention is already below threshold, OR
        //   (b) the retention drop over the window is severe
        for i in 1..curve.len() {
            let prev = &curve[i - 1];
            let curr = &curve[i];
            let drop = (prev.retention - curr.retention).max(0.0);

            // Normalise the drop relative to the duration of the scene
            let duration_s = scenes
                .get(i)
                .map_or(1.0, |s| (s.duration() as f64 / 1000.0).max(0.001));
            let drop_rate = drop / duration_s;

            let severity = (drop_rate * 10.0).clamp(0.0, 1.0); // normalise: 0.1 drop/s → severity 1.0

            if curr.retention < self.config.drop_off_threshold
                || severity >= self.config.min_drop_off_severity
            {
                zones.push(DropOffZone {
                    start: Timestamp::new(prev.timestamp.pts, timebase),
                    end: Timestamp::new(curr.timestamp.pts, timebase),
                    severity,
                    reason: format!(
                        "Retention dropped {:.1}% (rate {:.2}/s)",
                        drop * 100.0,
                        drop_rate
                    ),
                });
            }
        }

        zones
    }

    /// Identify moments that are likely to recapture viewer attention.
    fn detect_re_engagement(
        &self,
        scenes: &[ScoredScene],
        smoothed: &[f64],
    ) -> Vec<ReEngagementPoint> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut points = Vec::new();

        for (i, scene) in scenes.iter().enumerate() {
            let score = smoothed.get(i).copied().unwrap_or(scene.adjusted_score());

            if score < self.config.re_engagement_min_score {
                continue;
            }

            // Determine signal type
            let signal_type = if scene.features.motion_intensity > 0.70 {
                ReEngagementSignal::ActionSpike
            } else if scene.features.audio_peak > 0.70 {
                ReEngagementSignal::AudioPeak
            } else if i > scenes.len() * 3 / 4 {
                // Near the end → narrative climax territory
                ReEngagementSignal::NarrativeClimax
            } else {
                ReEngagementSignal::HighImportanceScene
            };

            points.push(ReEngagementPoint {
                timestamp: Timestamp::new(scene.start.pts, timebase),
                boost: score,
                signal_type,
            });
        }

        points
    }

    /// Compute a hook score for the opening segment of the video.
    fn compute_hook_score(&self, scenes: &[ScoredScene], smoothed: &[f64]) -> f64 {
        let mut total_weight = 0.0f64;
        let mut weighted_score = 0.0f64;

        for (i, scene) in scenes.iter().enumerate() {
            if scene.start.pts > self.config.hook_window_ms {
                break;
            }

            let duration_ms = scene
                .duration()
                .min(self.config.hook_window_ms - scene.start.pts)
                as f64;
            let score = smoothed.get(i).copied().unwrap_or(scene.adjusted_score());

            weighted_score += score * duration_ms;
            total_weight += duration_ms;
        }

        if total_weight > 0.0 {
            (weighted_score / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Smooth the scene scores using a simple moving average.
    fn smooth_scores(&self, scenes: &[ScoredScene]) -> Vec<f64> {
        let n = scenes.len();
        let half = self.config.smoothing_window / 2;
        let mut smoothed = Vec::with_capacity(n);

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let slice = &scenes[start..end];
            let avg = slice.iter().map(|s| s.adjusted_score()).sum::<f64>() / slice.len() as f64;
            smoothed.push(avg);
        }

        smoothed
    }

    /// Build an [`InterestCurve`] from the engagement prediction (for downstream use).
    ///
    /// # Errors
    ///
    /// Returns an error if the prediction itself fails.
    pub fn build_interest_curve(&self, scenes: &[ScoredScene]) -> AutoResult<InterestCurve> {
        let prediction = self.predict(scenes)?;
        let mut curve = InterestCurve::new(self.config.smoothing_window);
        for point in &prediction.retention_curve {
            curve.add_point(point.timestamp, point.engagement);
        }
        Ok(curve)
    }

    /// Suggest pacing improvements based on drop-off risk.
    ///
    /// Returns a list of `(timestamp_ms, suggestion)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the prediction fails.
    pub fn suggest_pacing_improvements(
        &self,
        scenes: &[ScoredScene],
    ) -> AutoResult<Vec<(i64, String)>> {
        let prediction = self.predict(scenes)?;
        let mut suggestions = Vec::new();

        for zone in &prediction.drop_off_zones {
            if zone.severity > 0.50 {
                suggestions.push((
                    zone.start.pts,
                    format!(
                        "Consider trimming or replacing content at {}ms: {}",
                        zone.start.pts, zone.reason
                    ),
                ));
            }
        }

        for point in &prediction.re_engagement_points {
            if prediction.hook_score < 0.40 && point.timestamp.pts < 10_000 {
                suggestions.push((
                    0,
                    format!(
                        "Move strong scene at {}ms to the opening to improve hook",
                        point.timestamp.pts
                    ),
                ));
            }
        }

        if prediction.completion_rate() < 0.50 {
            suggestions.push((
                0,
                "Overall completion rate is low — consider shortening the video".to_string(),
            ));
        }

        Ok(suggestions)
    }
}

impl Default for EngagementPredictor {
    fn default() -> Self {
        Self::default_predictor()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::{ContentType, SceneFeatures, ScoredScene, Sentiment};
    use oximedia_core::{Rational, Timestamp};

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_scene(start_ms: i64, end_ms: i64, score: f64) -> ScoredScene {
        ScoredScene::new(
            ts(start_ms),
            ts(end_ms),
            score,
            ContentType::Unknown,
            Sentiment::Neutral,
        )
    }

    fn make_scene_with_features(
        start_ms: i64,
        end_ms: i64,
        score: f64,
        motion: f64,
        audio_peak: f64,
    ) -> ScoredScene {
        let mut s = make_scene(start_ms, end_ms, score);
        s.features = SceneFeatures {
            motion_intensity: motion,
            audio_peak,
            ..SceneFeatures::default()
        };
        s
    }

    #[test]
    fn test_default_config_valid() {
        assert!(PredictorConfig::default().validate().is_ok());
    }

    #[test]
    fn test_short_form_config_valid() {
        assert!(PredictorConfig::short_form().validate().is_ok());
    }

    #[test]
    fn test_long_form_config_valid() {
        assert!(PredictorConfig::long_form().validate().is_ok());
    }

    #[test]
    fn test_invalid_decay() {
        let mut cfg = PredictorConfig::default();
        cfg.base_decay_per_second = -0.01;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_hook_window() {
        let mut cfg = PredictorConfig::default();
        cfg.hook_window_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_empty_scenes_error() {
        let predictor = EngagementPredictor::default();
        assert!(predictor.predict(&[]).is_err());
    }

    #[test]
    fn test_single_scene_prediction() {
        let predictor = EngagementPredictor::default();
        let scenes = vec![make_scene(0, 10_000, 0.80)];
        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert_eq!(pred.retention_curve.len(), 1);
        assert!(pred.retention_curve[0].retention <= 1.0);
        assert!(pred.retention_curve[0].retention >= 0.0);
    }

    #[test]
    fn test_retention_monotonically_decreasing_for_constant_low_interest() {
        let predictor = EngagementPredictor::default();
        // All scenes have very low scores → retention should decline
        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| make_scene(i * 5000, (i + 1) * 5000, 0.10))
            .collect();
        let pred = predictor.predict(&scenes).expect("predict should succeed");
        for window in pred.retention_curve.windows(2) {
            assert!(
                window[1].retention <= window[0].retention + 1e-9,
                "Retention should not increase in low-interest content: {:?} > {:?}",
                window[1].retention,
                window[0].retention
            );
        }
    }

    #[test]
    fn test_high_interest_slows_decay() {
        let predictor = EngagementPredictor::default();

        // High-interest scenes (score = 0.95)
        let high_scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 5000, (i + 1) * 5000, 0.95))
            .collect();
        // Low-interest scenes (score = 0.10)
        let low_scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 5000, (i + 1) * 5000, 0.10))
            .collect();

        let high_pred = predictor
            .predict(&high_scenes)
            .expect("predict should succeed");
        let low_pred = predictor
            .predict(&low_scenes)
            .expect("predict should succeed");

        // High-interest videos should retain more viewers
        assert!(
            high_pred.average_retention > low_pred.average_retention,
            "High-interest content should have higher retention"
        );
    }

    #[test]
    fn test_hook_score_uses_first_window() {
        // Disable smoothing (window=1) so the high score of the first scene is not
        // averaged down by the low-score tail scenes.
        let predictor = EngagementPredictor::new(PredictorConfig {
            hook_window_ms: 10_000,
            smoothing_window: 1,
            ..PredictorConfig::default()
        });

        // First 10 s: high score; rest: low
        let mut scenes = vec![make_scene(0, 10_000, 0.90)];
        scenes.extend((1..5).map(|i| make_scene(i * 10_000, (i + 1) * 10_000, 0.10)));

        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert!(
            pred.hook_score > 0.5,
            "Hook score should reflect high-interest opening"
        );
    }

    #[test]
    fn test_drop_off_zones_detected() {
        let predictor = EngagementPredictor::new(PredictorConfig {
            base_decay_per_second: 0.20, // very fast decay to guarantee drop-offs
            min_drop_off_severity: 0.01,
            ..PredictorConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| make_scene(i * 10_000, (i + 1) * 10_000, 0.10))
            .collect();

        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert!(
            !pred.drop_off_zones.is_empty(),
            "Expected drop-off zones in fast-decaying low-interest video"
        );
    }

    #[test]
    fn test_re_engagement_detected_for_high_score_scenes() {
        // Use smoothing_window=1 so the high-scoring scene is not averaged down.
        let predictor = EngagementPredictor::new(PredictorConfig {
            re_engagement_min_score: 0.70,
            smoothing_window: 1,
            ..PredictorConfig::default()
        });

        let scenes = vec![
            make_scene(0, 5000, 0.20),
            make_scene_with_features(5000, 10_000, 0.90, 0.80, 0.80),
        ];

        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert!(
            !pred.re_engagement_points.is_empty(),
            "Expected re-engagement point for high-importance scene"
        );
    }

    #[test]
    fn test_action_spike_signal_type() {
        let predictor = EngagementPredictor::new(PredictorConfig {
            re_engagement_min_score: 0.60,
            ..PredictorConfig::default()
        });

        let scenes = vec![make_scene_with_features(0, 5000, 0.85, 0.80, 0.10)];
        let pred = predictor.predict(&scenes).expect("predict should succeed");
        let action_points: Vec<_> = pred
            .re_engagement_points
            .iter()
            .filter(|p| p.signal_type == ReEngagementSignal::ActionSpike)
            .collect();
        assert!(!action_points.is_empty());
    }

    #[test]
    fn test_completion_rate() {
        let predictor = EngagementPredictor::new(PredictorConfig {
            base_decay_per_second: 0.001, // very slow decay
            ..PredictorConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 5000, (i + 1) * 5000, 0.70))
            .collect();
        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert!(
            pred.completion_rate() > 0.80,
            "Slow decay should yield high completion"
        );
    }

    #[test]
    fn test_worst_drop_off_returns_highest_severity() {
        let predictor = EngagementPredictor::new(PredictorConfig {
            base_decay_per_second: 0.15,
            min_drop_off_severity: 0.0,
            ..PredictorConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 10_000, (i + 1) * 10_000, 0.05))
            .collect();
        let pred = predictor.predict(&scenes).expect("predict should succeed");

        if let Some(worst) = pred.worst_drop_off() {
            for zone in &pred.drop_off_zones {
                assert!(zone.severity <= worst.severity + 1e-9);
            }
        }
    }

    #[test]
    fn test_build_interest_curve() {
        let predictor = EngagementPredictor::default();
        let scenes: Vec<ScoredScene> = (0..4)
            .map(|i| make_scene(i * 5000, (i + 1) * 5000, 0.60))
            .collect();
        let curve = predictor
            .build_interest_curve(&scenes)
            .expect("build interest curve should succeed");
        assert_eq!(curve.points.len(), scenes.len());
    }

    #[test]
    fn test_pacing_suggestions_low_completion() {
        let predictor = EngagementPredictor::new(PredictorConfig {
            base_decay_per_second: 0.50, // extreme decay → low completion
            min_drop_off_severity: 0.0,
            ..PredictorConfig::default()
        });

        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| make_scene(i * 10_000, (i + 1) * 10_000, 0.05))
            .collect();
        let suggestions = predictor
            .suggest_pacing_improvements(&scenes)
            .expect("suggest pacing improvements should succeed");
        assert!(
            !suggestions.is_empty(),
            "Expected pacing suggestions for low-retention video"
        );
    }

    #[test]
    fn test_smooth_scores_returns_correct_length() {
        let predictor = EngagementPredictor::default();
        let scenes: Vec<ScoredScene> = (0..7)
            .map(|i| make_scene(i * 2000, (i + 1) * 2000, i as f64 / 7.0))
            .collect();
        let smoothed = predictor.smooth_scores(&scenes);
        assert_eq!(smoothed.len(), scenes.len());
    }

    #[test]
    fn test_overall_engagement_bounded() {
        let predictor = EngagementPredictor::default();
        let scenes: Vec<ScoredScene> = (0..6)
            .map(|i| make_scene(i * 3000, (i + 1) * 3000, 0.60))
            .collect();
        let pred = predictor.predict(&scenes).expect("predict should succeed");
        assert!((0.0..=1.0).contains(&pred.overall_engagement));
    }
}
