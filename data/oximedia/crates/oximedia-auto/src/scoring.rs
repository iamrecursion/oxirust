//! Scene scoring and importance analysis.
//!
//! This module provides algorithms for scoring video scenes based on various
//! features and generating interest curves for content analysis.
//!
//! # Features
//!
//! - **Scene importance scoring**: Combine multiple metrics
//! - **Interest curve generation**: Temporal engagement analysis
//! - **Content classification**: Categorize scene types
//! - **Sentiment analysis**: Visual emotion detection
//! - **Auto-titling suggestions**: Generate descriptive titles
//!
//! # Example
//!
//! ```
//! use oximedia_auto::scoring::{SceneScorer, ScoringConfig};
//!
//! let config = ScoringConfig::default();
//! let scorer = SceneScorer::new(config);
//! ```

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// Stable identifier for a scene used as a cache key.
///
/// Scenes are identified by their start and end PTS values, which are
/// immutable after construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneId {
    /// Start PTS in milliseconds.
    pub start_pts: i64,
    /// End PTS in milliseconds.
    pub end_pts: i64,
}

impl SceneId {
    /// Construct a `SceneId` from timestamps.
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp) -> Self {
        Self {
            start_pts: start.pts,
            end_pts: end.pts,
        }
    }
}

/// Per-scene feature component scores (before weighting).
///
/// These are the raw, weight-independent scores derived from [`SceneFeatures`].
/// Caching this struct allows a weight change to skip raw feature computation
/// and only re-apply the weight vector.
#[derive(Debug, Clone)]
pub struct SceneComponentScores {
    /// Raw motion intensity score.
    pub motion: f64,
    /// Raw face coverage score.
    pub face: f64,
    /// Raw audio peak score.
    pub audio_peak: f64,
    /// Raw audio energy score.
    pub audio_energy: f64,
    /// Raw color diversity score.
    pub color: f64,
    /// Raw edge density score.
    pub edge: f64,
    /// Raw contrast score.
    pub contrast: f64,
    /// Raw sharpness score.
    pub sharpness: f64,
    /// Raw object diversity score.
    pub object: f64,
    /// Cached content type (classification is deterministic from features).
    pub content_type: ContentType,
    /// Cached sentiment (deterministic from features).
    pub sentiment: Sentiment,
    /// Cached face count (needed for title generation).
    pub face_count: usize,
}

/// Scene importance score (0.0 to 1.0).
pub type ImportanceScore = f64;

/// Type of content detected in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// Action scene with high motion.
    Action,
    /// Dialogue or conversation.
    Dialogue,
    /// Static or slow-moving scene.
    Static,
    /// Establishing shot or landscape.
    Establishing,
    /// Close-up shot.
    CloseUp,
    /// Group shot with multiple subjects.
    Group,
    /// Transition or filler content.
    Transition,
    /// Unknown or unclassified content.
    Unknown,
}

impl ContentType {
    /// Get the base importance weight for this content type.
    #[must_use]
    pub const fn base_importance(&self) -> f64 {
        match self {
            Self::Action => 0.85,
            Self::Dialogue => 0.75,
            Self::CloseUp => 0.70,
            Self::Group => 0.65,
            Self::Establishing => 0.55,
            Self::Static => 0.40,
            Self::Transition => 0.20,
            Self::Unknown => 0.50,
        }
    }
}

/// Visual sentiment detected in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentiment {
    /// Positive or uplifting.
    Positive,
    /// Neutral.
    Neutral,
    /// Negative or somber.
    Negative,
    /// Tense or suspenseful.
    Tense,
    /// Calm or peaceful.
    Calm,
}

impl Sentiment {
    /// Get the emotional intensity multiplier.
    #[must_use]
    pub const fn intensity_multiplier(&self) -> f64 {
        match self {
            Self::Tense => 1.3,
            Self::Positive => 1.2,
            Self::Negative => 1.1,
            Self::Neutral => 1.0,
            Self::Calm => 0.9,
        }
    }
}

/// Scene feature metrics used for scoring.
#[derive(Debug, Clone, Default)]
pub struct SceneFeatures {
    /// Motion intensity (0.0 to 1.0).
    pub motion_intensity: f64,
    /// Face count in the scene.
    pub face_count: usize,
    /// Face coverage ratio (0.0 to 1.0).
    pub face_coverage: f64,
    /// Audio peak level (0.0 to 1.0).
    pub audio_peak: f64,
    /// Audio energy (0.0 to 1.0).
    pub audio_energy: f64,
    /// Color diversity score (0.0 to 1.0).
    pub color_diversity: f64,
    /// Edge density (0.0 to 1.0).
    pub edge_density: f64,
    /// Brightness mean (0.0 to 1.0).
    pub brightness_mean: f64,
    /// Contrast level (0.0 to 1.0).
    pub contrast: f64,
    /// Sharpness metric (0.0 to 1.0).
    pub sharpness: f64,
    /// Object count.
    pub object_count: usize,
    /// Object diversity score (0.0 to 1.0).
    pub object_diversity: f64,
    /// Temporal stability (0.0 to 1.0).
    pub temporal_stability: f64,
}

impl SceneFeatures {
    /// Create new scene features with all values set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute a composite feature score.
    #[must_use]
    pub fn composite_score(&self, weights: &FeatureWeights) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        score += self.motion_intensity * weights.motion;
        total_weight += weights.motion;

        score += self.face_coverage * weights.face;
        total_weight += weights.face;

        score += self.audio_peak * weights.audio_peak;
        total_weight += weights.audio_peak;

        score += self.audio_energy * weights.audio_energy;
        total_weight += weights.audio_energy;

        score += self.color_diversity * weights.color;
        total_weight += weights.color;

        score += self.edge_density * weights.edge;
        total_weight += weights.edge;

        score += self.contrast * weights.contrast;
        total_weight += weights.contrast;

        score += self.sharpness * weights.sharpness;
        total_weight += weights.sharpness;

        score += self.object_diversity * weights.object;
        total_weight += weights.object;

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }
}

/// Weights for different feature components in scene scoring.
///
/// Each weight scales the contribution of a single feature to the final scene
/// importance score.  All weights are non-negative; a value of `0.0` disables
/// the corresponding feature entirely.  The scorer normalises by the sum of
/// all weights so absolute magnitudes do not matter — only the *ratios* between
/// weights influence the result.
///
/// # Tuning Guidelines
///
/// The default weights (`motion=1.5`, `face=1.2`, `audio_peak=1.3`, …) are
/// balanced for general highlight detection.  Below are starting-point presets
/// for common content types; tune from there with real content:
///
/// ## Sports / Action
/// Prioritise motion and audio intensity; de-emphasise face and sharpness.
/// ```ignore
/// FeatureWeights { motion: 2.5, face: 0.8, audio_peak: 2.0, audio_energy: 1.5,
///                  color: 0.5, edge: 0.6, contrast: 0.4, sharpness: 0.6, object: 0.6 }
/// ```
///
/// ## Interview / Documentary
/// Prioritise face coverage and spoken-word audio; motion matters less.
/// ```ignore
/// FeatureWeights { motion: 0.5, face: 2.5, audio_peak: 1.5, audio_energy: 0.8,
///                  color: 0.6, edge: 0.5, contrast: 0.5, sharpness: 1.0, object: 0.8 }
/// ```
///
/// ## Music Video
/// Balance motion with strong audio cues and rich colour.
/// ```ignore
/// FeatureWeights { motion: 1.8, face: 1.0, audio_peak: 2.2, audio_energy: 1.8,
///                  color: 1.2, edge: 0.8, contrast: 0.8, sharpness: 0.6, object: 0.5 }
/// ```
///
/// ## Nature / B-roll
/// Visual quality (colour, sharpness, contrast) drives importance; audio is
/// secondary and motion should be low to avoid wind-shake false positives.
/// ```ignore
/// FeatureWeights { motion: 0.6, face: 0.3, audio_peak: 0.5, audio_energy: 0.4,
///                  color: 2.0, edge: 1.2, contrast: 1.5, sharpness: 1.8, object: 1.0 }
/// ```
#[derive(Debug, Clone)]
pub struct FeatureWeights {
    /// Motion importance weight.
    pub motion: f64,
    /// Face detection weight.
    pub face: f64,
    /// Audio peak weight.
    pub audio_peak: f64,
    /// Audio energy weight.
    pub audio_energy: f64,
    /// Color diversity weight.
    pub color: f64,
    /// Edge density weight.
    pub edge: f64,
    /// Contrast weight.
    pub contrast: f64,
    /// Sharpness weight.
    pub sharpness: f64,
    /// Object detection weight.
    pub object: f64,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            motion: 1.5,
            face: 1.2,
            audio_peak: 1.3,
            audio_energy: 1.0,
            color: 0.8,
            edge: 0.7,
            contrast: 0.6,
            sharpness: 0.5,
            object: 1.1,
        }
    }
}

/// A scored scene segment.
#[derive(Debug, Clone)]
pub struct ScoredScene {
    /// Scene start timestamp.
    pub start: Timestamp,
    /// Scene end timestamp.
    pub end: Timestamp,
    /// Importance score (0.0 to 1.0).
    pub score: ImportanceScore,
    /// Content type classification.
    pub content_type: ContentType,
    /// Detected sentiment.
    pub sentiment: Sentiment,
    /// Scene features used for scoring.
    pub features: SceneFeatures,
    /// Suggested title or description.
    pub suggested_title: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl ScoredScene {
    /// Create a new scored scene.
    #[must_use]
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        score: ImportanceScore,
        content_type: ContentType,
        sentiment: Sentiment,
    ) -> Self {
        Self {
            start,
            end,
            score: score.clamp(0.0, 1.0),
            content_type,
            sentiment,
            features: SceneFeatures::default(),
            suggested_title: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the duration of this scene.
    #[must_use]
    pub fn duration(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if this scene meets a minimum score threshold.
    #[must_use]
    pub fn meets_threshold(&self, threshold: ImportanceScore) -> bool {
        self.score >= threshold
    }

    /// Compute the final adjusted score with sentiment multiplier.
    #[must_use]
    pub fn adjusted_score(&self) -> ImportanceScore {
        (self.score * self.sentiment.intensity_multiplier()).clamp(0.0, 1.0)
    }
}

/// Interest curve for temporal engagement analysis.
#[derive(Debug, Clone)]
pub struct InterestCurve {
    /// Timestamp and score pairs.
    pub points: Vec<(Timestamp, ImportanceScore)>,
    /// Smoothing window size.
    pub window_size: usize,
}

impl InterestCurve {
    /// Create a new interest curve.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            points: Vec::new(),
            window_size,
        }
    }

    /// Add a point to the curve.
    pub fn add_point(&mut self, timestamp: Timestamp, score: ImportanceScore) {
        self.points.push((timestamp, score.clamp(0.0, 1.0)));
    }

    /// Get the score at a specific timestamp (interpolated).
    #[must_use]
    pub fn score_at(&self, timestamp: Timestamp) -> ImportanceScore {
        if self.points.is_empty() {
            return 0.0;
        }

        // Find surrounding points
        let mut before = None;
        let mut after = None;

        for (i, (ts, score)) in self.points.iter().enumerate() {
            if *ts <= timestamp {
                before = Some((i, *ts, *score));
            }
            if *ts >= timestamp && after.is_none() {
                after = Some((i, *ts, *score));
                break;
            }
        }

        match (before, after) {
            (Some((_, _, score)), None) => score,
            (None, Some((_, _, score))) => score,
            (Some((_, t1, s1)), Some((_, t2, s2))) => {
                if t1 == t2 {
                    s1
                } else {
                    let ratio = (timestamp.pts - t1.pts) as f64 / (t2.pts - t1.pts) as f64;
                    s1 + (s2 - s1) * ratio
                }
            }
            (None, None) => 0.0,
        }
    }

    /// Get smoothed curve using moving average.
    #[must_use]
    pub fn smoothed(&self) -> Self {
        if self.points.len() < self.window_size {
            return self.clone();
        }

        let mut smoothed = Self::new(self.window_size);

        for i in 0..self.points.len() {
            let start = i.saturating_sub(self.window_size / 2);
            let end = (i + self.window_size / 2 + 1).min(self.points.len());

            let avg_score: f64 =
                self.points[start..end].iter().map(|(_, s)| s).sum::<f64>() / (end - start) as f64;

            smoothed.add_point(self.points[i].0, avg_score);
        }

        smoothed
    }

    /// Find peaks in the interest curve.
    #[must_use]
    pub fn find_peaks(&self, threshold: ImportanceScore) -> Vec<(Timestamp, ImportanceScore)> {
        let smoothed = self.smoothed();
        let mut peaks = Vec::new();

        for i in 1..smoothed.points.len().saturating_sub(1) {
            let (ts, score) = smoothed.points[i];
            let prev_score = smoothed.points[i - 1].1;
            let next_score = smoothed.points[i + 1].1;

            if score >= threshold && score >= prev_score && score >= next_score {
                peaks.push((ts, score));
            }
        }

        peaks
    }

    /// Find valleys in the interest curve.
    #[must_use]
    pub fn find_valleys(&self, threshold: ImportanceScore) -> Vec<(Timestamp, ImportanceScore)> {
        let smoothed = self.smoothed();
        let mut valleys = Vec::new();

        for i in 1..smoothed.points.len().saturating_sub(1) {
            let (ts, score) = smoothed.points[i];
            let prev_score = smoothed.points[i - 1].1;
            let next_score = smoothed.points[i + 1].1;

            if score <= threshold && score <= prev_score && score <= next_score {
                valleys.push((ts, score));
            }
        }

        valleys
    }
}

/// Configuration for scene scoring.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Feature weights.
    pub feature_weights: FeatureWeights,
    /// Minimum scene duration in milliseconds.
    pub min_scene_duration_ms: i64,
    /// Interest curve smoothing window size.
    pub curve_window_size: usize,
    /// Enable content classification.
    pub enable_classification: bool,
    /// Enable sentiment analysis.
    pub enable_sentiment: bool,
    /// Enable auto-titling.
    pub enable_auto_titling: bool,
    /// Peak detection threshold.
    pub peak_threshold: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            feature_weights: FeatureWeights::default(),
            min_scene_duration_ms: 500,
            curve_window_size: 5,
            enable_classification: true,
            enable_sentiment: true,
            enable_auto_titling: true,
            peak_threshold: 0.65,
        }
    }
}

impl ScoringConfig {
    /// Create a new scoring configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the feature weights.
    #[must_use]
    pub fn with_feature_weights(mut self, weights: FeatureWeights) -> Self {
        self.feature_weights = weights;
        self
    }

    /// Set the minimum scene duration.
    #[must_use]
    pub const fn with_min_scene_duration_ms(mut self, duration_ms: i64) -> Self {
        self.min_scene_duration_ms = duration_ms;
        self
    }

    /// Set the curve window size.
    #[must_use]
    pub const fn with_curve_window_size(mut self, window_size: usize) -> Self {
        self.curve_window_size = window_size;
        self
    }

    /// Enable or disable content classification.
    #[must_use]
    pub const fn with_classification(mut self, enable: bool) -> Self {
        self.enable_classification = enable;
        self
    }

    /// Enable or disable sentiment analysis.
    #[must_use]
    pub const fn with_sentiment(mut self, enable: bool) -> Self {
        self.enable_sentiment = enable;
        self
    }

    /// Enable or disable auto-titling.
    #[must_use]
    pub const fn with_auto_titling(mut self, enable: bool) -> Self {
        self.enable_auto_titling = enable;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.min_scene_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.min_scene_duration_ms,
            });
        }

        if self.curve_window_size == 0 {
            return Err(AutoError::invalid_parameter(
                "curve_window_size",
                "must be greater than 0",
            ));
        }

        if !(0.0..=1.0).contains(&self.peak_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.peak_threshold,
                min: 0.0,
                max: 1.0,
            });
        }

        Ok(())
    }
}

/// Configuration for temporal context scoring.
#[derive(Debug, Clone)]
pub struct TemporalContextConfig {
    /// Enable temporal context adjustment.
    pub enabled: bool,
    /// Number of neighboring scenes to consider on each side.
    pub neighbor_radius: usize,
    /// Weight of the neighbor average score in the final adjustment (0.0-1.0).
    pub neighbor_weight: f64,
    /// Bonus multiplier for scenes that are significantly above their neighbors.
    pub relative_boost: f64,
}

impl Default for TemporalContextConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            neighbor_radius: 2,
            neighbor_weight: 0.15,
            relative_boost: 1.2,
        }
    }
}

/// Scene scorer for importance analysis.
#[derive(Clone)]
pub struct SceneScorer {
    /// Configuration.
    config: ScoringConfig,
    /// Temporal context configuration.
    pub temporal_context: TemporalContextConfig,
    /// Cache of per-scene component scores keyed by [`SceneId`].
    ///
    /// Stores the raw, weight-independent feature sub-scores so that a config
    /// change only needs to re-apply the weight vector rather than re-derive
    /// components from raw [`SceneFeatures`].
    feature_cache: HashMap<SceneId, SceneComponentScores>,
}

impl SceneScorer {
    /// Create a new scene scorer.
    #[must_use]
    pub fn new(config: ScoringConfig) -> Self {
        Self {
            config,
            temporal_context: TemporalContextConfig::default(),
            feature_cache: HashMap::new(),
        }
    }

    /// Create a scene scorer with a custom temporal context configuration.
    #[must_use]
    pub fn with_temporal_context(mut self, ctx: TemporalContextConfig) -> Self {
        self.temporal_context = ctx;
        self
    }

    /// Create a scene scorer with default configuration.
    #[must_use]
    pub fn default_scorer() -> Self {
        Self::new(ScoringConfig::default())
    }

    /// Invalidate the entire feature cache.
    ///
    /// Call this when the underlying scene data has changed and the cached
    /// component scores may no longer be valid.
    pub fn invalidate_cache(&mut self) {
        self.feature_cache.clear();
    }

    /// Remove a single scene from the feature cache.
    ///
    /// Useful when only one scene's raw data has been mutated.
    pub fn clear_scene(&mut self, id: SceneId) {
        self.feature_cache.remove(&id);
    }

    /// Extract raw component scores from [`SceneFeatures`] without applying weights.
    ///
    /// These values are cached; the weighted combination is applied separately
    /// in [`Self::score_scene`] so that a weight change only needs to re-apply
    /// the weight vector rather than recompute from raw feature data.
    fn extract_components(&self, features: &SceneFeatures) -> SceneComponentScores {
        let content_type = if self.config.enable_classification {
            self.classify_content(features)
        } else {
            ContentType::Unknown
        };
        let sentiment = if self.config.enable_sentiment {
            self.detect_sentiment(features)
        } else {
            Sentiment::Neutral
        };
        SceneComponentScores {
            motion: features.motion_intensity,
            face: features.face_coverage,
            audio_peak: features.audio_peak,
            audio_energy: features.audio_energy,
            color: features.color_diversity,
            edge: features.edge_density,
            contrast: features.contrast,
            sharpness: features.sharpness,
            object: features.object_diversity,
            content_type,
            sentiment,
            face_count: features.face_count,
        }
    }

    /// Apply a weight vector to cached component scores to produce a weighted sum.
    fn apply_weights(components: &SceneComponentScores, weights: &FeatureWeights) -> f64 {
        let mut score = 0.0_f64;
        let mut total_weight = 0.0_f64;

        score += components.motion * weights.motion;
        total_weight += weights.motion;

        score += components.face * weights.face;
        total_weight += weights.face;

        score += components.audio_peak * weights.audio_peak;
        total_weight += weights.audio_peak;

        score += components.audio_energy * weights.audio_energy;
        total_weight += weights.audio_energy;

        score += components.color * weights.color;
        total_weight += weights.color;

        score += components.edge * weights.edge;
        total_weight += weights.edge;

        score += components.contrast * weights.contrast;
        total_weight += weights.contrast;

        score += components.sharpness * weights.sharpness;
        total_weight += weights.sharpness;

        score += components.object * weights.object;
        total_weight += weights.object;

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }

    /// Score a scene based on its features, using a cache to avoid redundant
    /// component extraction when only the weight configuration has changed.
    ///
    /// **Cache semantics:**
    /// - First call for a `(start, end)` pair: extract components from `features`,
    ///   populate the cache, then apply the weight vector.
    /// - Subsequent calls with the same `(start, end)`: re-apply the current
    ///   weight vector to the cached components (no raw feature work).
    ///
    /// The cache key is [`SceneId`] derived from `(start, end)` timestamps.
    /// Call [`Self::invalidate_cache`] or [`Self::clear_scene`] if the
    /// underlying scene data changes.
    pub fn score_scene(
        &mut self,
        start: Timestamp,
        end: Timestamp,
        features: SceneFeatures,
    ) -> AutoResult<ScoredScene> {
        self.config.validate()?;

        let id = SceneId::new(start, end);

        // Check cache — if hit, re-apply weight vector to cached components.
        // If miss, extract components, populate cache, then apply weights.
        let components = if let Some(cached) = self.feature_cache.get(&id) {
            cached.clone()
        } else {
            let c = self.extract_components(&features);
            self.feature_cache.insert(id, c.clone());
            c
        };

        // Fast path: re-apply current weight vector to cached component scores.
        let base_score = Self::apply_weights(&components, &self.config.feature_weights);

        // Apply content type base importance.
        let type_adjusted_score =
            base_score * 0.7 + components.content_type.base_importance() * 0.3;

        // Generate title suggestion (uses cached components, cheap string work).
        let suggested_title = if self.config.enable_auto_titling {
            // Reconstruct a minimal feature view for title generation.
            let face_count = components.face_count;
            Some(self.generate_title(components.content_type, components.sentiment, face_count))
        } else {
            None
        };

        let mut scene = ScoredScene::new(
            start,
            end,
            type_adjusted_score,
            components.content_type,
            components.sentiment,
        );
        scene.features = features;
        scene.suggested_title = suggested_title;

        Ok(scene)
    }

    /// Score a scene with temporal context — the score is adjusted relative to
    /// its neighbors so that scenes that stand out from their surroundings
    /// receive a boost while unremarkable scenes in a uniformly high-scoring
    /// region are penalised slightly.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying scene scoring fails.
    pub fn score_scene_with_context(
        &mut self,
        start: Timestamp,
        end: Timestamp,
        features: SceneFeatures,
        neighbor_scores: &[f64],
    ) -> AutoResult<ScoredScene> {
        let mut scene = self.score_scene(start, end, features)?;

        if !self.temporal_context.enabled || neighbor_scores.is_empty() {
            return Ok(scene);
        }

        let neighbor_avg = neighbor_scores.iter().sum::<f64>() / neighbor_scores.len() as f64;
        let relative = scene.score - neighbor_avg;

        // Boost scenes that are notably above their neighbors
        let context_adjustment = if relative > 0.1 {
            relative * self.temporal_context.relative_boost
        } else {
            relative * self.temporal_context.neighbor_weight
        };

        scene.score = (scene.score + context_adjustment).clamp(0.0, 1.0);
        Ok(scene)
    }

    /// Classify content type based on features.
    #[must_use]
    fn classify_content(&self, features: &SceneFeatures) -> ContentType {
        // High motion indicates action
        if features.motion_intensity > 0.7 {
            return ContentType::Action;
        }

        // Multiple faces with low motion suggests dialogue
        if features.face_count >= 2 && features.motion_intensity < 0.3 {
            return ContentType::Dialogue;
        }

        // Single face with high coverage is close-up
        if features.face_count == 1 && features.face_coverage > 0.4 {
            return ContentType::CloseUp;
        }

        // Multiple faces is group shot
        if features.face_count > 2 {
            return ContentType::Group;
        }

        // Low motion and high temporal stability is static
        if features.motion_intensity < 0.2 && features.temporal_stability > 0.7 {
            return ContentType::Static;
        }

        // High edge density with low faces suggests establishing shot
        if features.edge_density > 0.6 && features.face_count == 0 {
            return ContentType::Establishing;
        }

        ContentType::Unknown
    }

    /// Detect visual sentiment based on features.
    #[must_use]
    fn detect_sentiment(&self, features: &SceneFeatures) -> Sentiment {
        // High motion and energy suggests tense or action-oriented
        if features.motion_intensity > 0.6 && features.audio_energy > 0.6 {
            return Sentiment::Tense;
        }

        // Bright with high color diversity suggests positive
        if features.brightness_mean > 0.6 && features.color_diversity > 0.5 {
            return Sentiment::Positive;
        }

        // Low brightness and low color suggests negative
        if features.brightness_mean < 0.4 && features.color_diversity < 0.4 {
            return Sentiment::Negative;
        }

        // Low motion and high stability suggests calm
        if features.motion_intensity < 0.3 && features.temporal_stability > 0.7 {
            return Sentiment::Calm;
        }

        Sentiment::Neutral
    }

    /// Generate a suggested title for the scene.
    ///
    /// Takes `face_count` as a plain integer so the method can be called with
    /// the cached value from [`SceneComponentScores`] without borrowing raw
    /// [`SceneFeatures`].
    #[must_use]
    fn generate_title(
        &self,
        content_type: ContentType,
        sentiment: Sentiment,
        face_count: usize,
    ) -> String {
        let type_str = match content_type {
            ContentType::Action => "Action Sequence",
            ContentType::Dialogue => "Conversation",
            ContentType::CloseUp => "Close-up Shot",
            ContentType::Group => "Group Scene",
            ContentType::Establishing => "Establishing Shot",
            ContentType::Static => "Static Scene",
            ContentType::Transition => "Transition",
            ContentType::Unknown => "Scene",
        };

        let sentiment_modifier = match sentiment {
            Sentiment::Positive => " (Uplifting)",
            Sentiment::Negative => " (Somber)",
            Sentiment::Tense => " (Intense)",
            Sentiment::Calm => " (Peaceful)",
            Sentiment::Neutral => "",
        };

        // Add face count if relevant
        let face_note = if face_count > 0 {
            format!(
                " - {} {}",
                face_count,
                if face_count == 1 { "person" } else { "people" }
            )
        } else {
            String::new()
        };

        format!("{type_str}{sentiment_modifier}{face_note}")
    }

    /// Generate an interest curve from scored scenes.
    #[must_use]
    pub fn generate_interest_curve(&self, scenes: &[ScoredScene]) -> InterestCurve {
        let mut curve = InterestCurve::new(self.config.curve_window_size);

        for scene in scenes {
            // Add points at start and end of each scene
            curve.add_point(scene.start, scene.adjusted_score());
            curve.add_point(scene.end, scene.adjusted_score());
        }

        curve
    }

    /// Find highlight moments using interest curve peaks.
    #[must_use]
    pub fn find_highlights(&self, curve: &InterestCurve) -> Vec<(Timestamp, ImportanceScore)> {
        curve.find_peaks(self.config.peak_threshold)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ScoringConfig {
        &self.config
    }
}

impl Default for SceneScorer {
    fn default() -> Self {
        Self::default_scorer()
    }
}

/// Batch score multiple scenes.
///
/// When `scorer.temporal_context.enabled` is true, each scene's final score
/// is adjusted relative to its `neighbor_radius` neighbors.
#[allow(dead_code)]
pub fn batch_score_scenes(
    scorer: &mut SceneScorer,
    scene_data: &[(Timestamp, Timestamp, SceneFeatures)],
) -> AutoResult<Vec<ScoredScene>> {
    // First pass: compute raw scores (also populates the feature cache).
    let mut raw: Vec<ScoredScene> = Vec::with_capacity(scene_data.len());
    for (start, end, features) in scene_data {
        raw.push(scorer.score_scene(*start, *end, features.clone())?);
    }

    if !scorer.temporal_context.enabled {
        return Ok(raw);
    }

    let radius = scorer.temporal_context.neighbor_radius.max(1);
    let raw_scores: Vec<f64> = raw.iter().map(|s| s.score).collect();

    // Second pass: apply temporal context
    let mut adjusted = Vec::with_capacity(raw.len());
    for (i, (scene, (start, end, features))) in raw.iter().zip(scene_data.iter()).enumerate() {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(raw_scores.len());
        let neighbors: Vec<f64> = raw_scores[lo..hi]
            .iter()
            .enumerate()
            .filter(|(j, _)| lo + j != i)
            .map(|(_, &s)| s)
            .collect();

        let mut adj_scene =
            scorer.score_scene_with_context(*start, *end, features.clone(), &neighbors)?;
        // Preserve the suggested title from the raw scene
        adj_scene.suggested_title = scene.suggested_title.clone();
        adjusted.push(adj_scene);
    }

    Ok(adjusted)
}

/// Compute normalized importance scores across all scenes.
#[allow(dead_code)]
pub fn normalize_scores(scenes: &mut [ScoredScene]) {
    if scenes.is_empty() {
        return;
    }

    let max_score = scenes
        .iter()
        .map(ScoredScene::adjusted_score)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    if max_score > 0.0 {
        for scene in scenes {
            scene.score = (scene.adjusted_score() / max_score).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn make_ts(pts: i64) -> Timestamp {
        Timestamp::new(pts, Rational::new(1, 1000))
    }

    fn make_features(motion: f64, face_count: usize) -> SceneFeatures {
        SceneFeatures {
            motion_intensity: motion,
            face_count,
            face_coverage: if face_count > 0 { 0.3 } else { 0.0 },
            audio_peak: 0.5,
            audio_energy: 0.4,
            color_diversity: 0.6,
            edge_density: 0.5,
            brightness_mean: 0.5,
            contrast: 0.5,
            sharpness: 0.7,
            object_count: 2,
            object_diversity: 0.4,
            temporal_stability: 0.6,
        }
    }

    /// Score 5 scenes, then score the same 5 scenes again and assert identical results.
    #[test]
    fn test_scorer_cache_same_as_fresh() {
        let config = ScoringConfig::default();
        let mut scorer = SceneScorer::new(config);

        let scenes: Vec<(Timestamp, Timestamp, SceneFeatures)> = (0..5)
            .map(|i| {
                let start = make_ts(i * 2000);
                let end = make_ts(i * 2000 + 1500);
                let features = make_features(0.1 * i as f64 + 0.1, i as usize);
                (start, end, features)
            })
            .collect();

        // First pass: populates cache.
        let first_pass: Vec<f64> = scenes
            .iter()
            .map(|(s, e, f)| {
                scorer
                    .score_scene(*s, *e, f.clone())
                    .expect("score_scene should succeed")
                    .score
            })
            .collect();

        // Second pass: cache hit for every scene.
        let second_pass: Vec<f64> = scenes
            .iter()
            .map(|(s, e, f)| {
                scorer
                    .score_scene(*s, *e, f.clone())
                    .expect("score_scene should succeed")
                    .score
            })
            .collect();

        assert_eq!(
            first_pass, second_pass,
            "cached scores must match fresh scores"
        );
    }

    /// Score a scene, change the weight config, score again — score changes but
    /// components must not be recomputed (cache entry count stays the same).
    #[test]
    fn test_scorer_cache_weight_change() {
        let config = ScoringConfig::default();
        let mut scorer = SceneScorer::new(config);

        let start = make_ts(0);
        let end = make_ts(2000);
        let features = make_features(0.8, 2);

        let score_before = scorer
            .score_scene(start, end, features.clone())
            .expect("score_scene should succeed")
            .score;

        // Cache now has 1 entry.
        let cache_size_before = scorer.feature_cache.len();
        assert_eq!(cache_size_before, 1, "cache should have exactly 1 entry");

        // Modify weights: significantly boost motion.
        scorer.config.feature_weights.motion = 10.0;
        scorer.config.feature_weights.face = 0.1;

        let score_after = scorer
            .score_scene(start, end, features)
            .expect("score_scene should succeed")
            .score;

        // Score changed because weights changed.
        assert_ne!(
            score_before, score_after,
            "score should change when weights change"
        );

        // Cache still has 1 entry — components were not recomputed.
        assert_eq!(
            scorer.feature_cache.len(),
            1,
            "cache size should stay at 1 after weight-only change"
        );
    }

    /// Invalidate after scoring, assert the next call recomputes (scores
    /// unchanged on same data, but cache entry is re-populated).
    #[test]
    fn test_scorer_cache_invalidation() {
        let config = ScoringConfig::default();
        let mut scorer = SceneScorer::new(config);

        let start = make_ts(0);
        let end = make_ts(3000);
        let features = make_features(0.5, 1);

        let score_first = scorer
            .score_scene(start, end, features.clone())
            .expect("score_scene should succeed")
            .score;

        assert_eq!(scorer.feature_cache.len(), 1);

        // Invalidate entire cache.
        scorer.invalidate_cache();
        assert_eq!(
            scorer.feature_cache.len(),
            0,
            "cache should be empty after invalidation"
        );

        // Next call must recompute and re-populate the cache.
        let score_after_invalidate = scorer
            .score_scene(start, end, features)
            .expect("score_scene should succeed")
            .score;

        assert!(
            (score_first - score_after_invalidate).abs() < 1e-12,
            "score must be identical after invalidation with same data"
        );
        assert_eq!(
            scorer.feature_cache.len(),
            1,
            "cache must have 1 entry after re-scoring"
        );
    }

    /// `clear_scene` removes only the targeted entry.
    #[test]
    fn test_scorer_clear_scene() {
        let config = ScoringConfig::default();
        let mut scorer = SceneScorer::new(config);

        let s0 = (make_ts(0), make_ts(1000), make_features(0.3, 0));
        let s1 = (make_ts(1000), make_ts(2000), make_features(0.6, 1));

        scorer.score_scene(s0.0, s0.1, s0.2.clone()).unwrap();
        scorer.score_scene(s1.0, s1.1, s1.2.clone()).unwrap();
        assert_eq!(scorer.feature_cache.len(), 2);

        scorer.clear_scene(SceneId::new(s0.0, s0.1));
        assert_eq!(scorer.feature_cache.len(), 1);
        assert!(scorer.feature_cache.contains_key(&SceneId::new(s1.0, s1.1)));
    }
}
