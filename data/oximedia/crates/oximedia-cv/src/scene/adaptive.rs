//! Adaptive scene detection.
//!
//! This module provides adaptive scene detection that combines multiple
//! detection methods and adjusts thresholds based on content characteristics.

use crate::error::CvResult;
use oximedia_codec::VideoFrame;

use super::{edge, histogram, motion, ChangeType, SceneChange, SceneConfig, SceneMetadata};

/// Configuration for adaptive detection.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Weight for histogram-based detection.
    pub histogram_weight: f64,
    /// Weight for edge-based detection.
    pub edge_weight: f64,
    /// Weight for motion-based detection.
    pub motion_weight: f64,
    /// Enable dynamic threshold adjustment.
    pub dynamic_threshold: bool,
    /// Minimum threshold (for dark/static scenes).
    pub min_threshold: f64,
    /// Maximum threshold (for bright/active scenes).
    pub max_threshold: f64,
    /// Content analysis window size.
    pub analysis_window: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            histogram_weight: 0.4,
            edge_weight: 0.3,
            motion_weight: 0.3,
            dynamic_threshold: true,
            min_threshold: 0.2,
            max_threshold: 0.5,
            analysis_window: 30,
        }
    }
}

impl AdaptiveConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> bool {
        let total_weight = self.histogram_weight + self.edge_weight + self.motion_weight;
        (total_weight - 1.0).abs() < 0.01
            && self.min_threshold >= 0.0
            && self.max_threshold <= 1.0
            && self.min_threshold <= self.max_threshold
    }
}

/// Content characteristics for a frame or sequence.
#[derive(Debug, Clone, Default)]
pub struct ContentCharacteristics {
    /// Average brightness (0-255).
    pub brightness: f64,
    /// Contrast (standard deviation of pixel values).
    pub contrast: f64,
    /// Edge density (0-1).
    pub edge_density: f64,
    /// Motion activity (0-1).
    pub motion_activity: f64,
    /// Color saturation (0-1).
    pub saturation: f64,
    /// Temporal variance over window.
    pub temporal_variance: f64,
}

impl ContentCharacteristics {
    /// Analyze content characteristics of a single frame.
    pub fn analyze_frame(frame: &VideoFrame) -> CvResult<Self> {
        let brightness = histogram::compute_average_brightness(frame)?;

        let edge_config = edge::EdgeConfig::default();
        let edge_density = edge::compute_edge_density(frame, &edge_config)?;

        Ok(Self {
            brightness,
            contrast: 0.0, // Will be computed from histogram if needed
            edge_density,
            motion_activity: 0.0, // Requires two frames
            saturation: 0.5,      // Default
            temporal_variance: 0.0,
        })
    }

    /// Analyze content characteristics over a sequence of frames.
    pub fn analyze_sequence(frames: &[VideoFrame], window: usize) -> CvResult<Vec<Self>> {
        let mut characteristics = Vec::new();

        // Compute motion activity
        let motion_scores = if frames.len() >= 2 {
            motion::analyze_motion_pattern(frames, window)?
        } else {
            vec![0.0; frames.len()]
        };

        for (i, frame) in frames.iter().enumerate() {
            let mut char = Self::analyze_frame(frame)?;

            // Add motion activity
            if i > 0 && i - 1 < motion_scores.len() {
                char.motion_activity = motion_scores[i - 1];
            }

            // Compute temporal variance
            if i >= window && i < frames.len() {
                char.temporal_variance =
                    compute_temporal_variance(frames, i.saturating_sub(window), i)?;
            }

            characteristics.push(char);
        }

        Ok(characteristics)
    }

    /// Compute adaptive threshold based on content.
    #[must_use]
    pub fn compute_adaptive_threshold(&self, config: &AdaptiveConfig) -> f64 {
        if !config.dynamic_threshold {
            return (config.min_threshold + config.max_threshold) / 2.0;
        }

        // Low brightness or low contrast -> lower threshold
        let brightness_factor = (self.brightness / 255.0).clamp(0.0, 1.0);
        let edge_factor = self.edge_density.clamp(0.0, 1.0);

        // High motion -> higher threshold to avoid false positives
        let motion_factor = 1.0 - self.motion_activity.clamp(0.0, 1.0);

        // Combine factors
        let factor = (brightness_factor + edge_factor + motion_factor) / 3.0;

        // Map to threshold range
        config.min_threshold + factor * (config.max_threshold - config.min_threshold)
    }

    /// Check if content is suitable for scene detection.
    #[must_use]
    pub fn is_suitable_for_detection(&self) -> bool {
        // Too dark, too bright, or too little detail -> less suitable
        self.brightness > 10.0 && self.brightness < 245.0 && self.edge_density > 0.01
    }
}

/// Compute temporal variance over a window.
fn compute_temporal_variance(frames: &[VideoFrame], start: usize, end: usize) -> CvResult<f64> {
    if start >= end || end > frames.len() {
        return Ok(0.0);
    }

    let mut brightness_values = Vec::new();

    for i in start..end {
        let brightness = histogram::compute_average_brightness(&frames[i])?;
        brightness_values.push(brightness);
    }

    if brightness_values.is_empty() {
        return Ok(0.0);
    }

    let mean: f64 = brightness_values.iter().sum::<f64>() / brightness_values.len() as f64;
    let variance: f64 = brightness_values
        .iter()
        .map(|b| {
            let diff = b - mean;
            diff * diff
        })
        .sum::<f64>()
        / brightness_values.len() as f64;

    Ok(variance.sqrt())
}

/// Detect adaptive scene changes.
pub fn detect_adaptive_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    if frames.len() < 2 {
        return Ok(Vec::new());
    }

    // Analyze content characteristics
    let window = config.adaptive_config.analysis_window;
    let characteristics = ContentCharacteristics::analyze_sequence(frames, window)?;

    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let frame1 = &frames[i - 1];
        let frame2 = &frames[i];

        // Check if content is suitable for detection
        if !characteristics[i].is_suitable_for_detection() {
            continue;
        }

        // Compute adaptive threshold
        let adaptive_threshold =
            characteristics[i].compute_adaptive_threshold(&config.adaptive_config);

        // Compute multiple scores
        let hist_similarity = histogram::compute_frame_similarity(frame1, frame2)?;
        let edge_similarity = edge::compute_edge_similarity(frame1, frame2, &config.edge_config)?;
        let motion_similarity =
            motion::compute_motion_score(frame1, frame2, &config.motion_config)?;

        let hist_diff = 1.0 - hist_similarity;
        let edge_diff = 1.0 - edge_similarity;
        let motion_diff = 1.0 - motion_similarity;

        // Weighted combination
        let combined_diff = hist_diff * config.adaptive_config.histogram_weight
            + edge_diff * config.adaptive_config.edge_weight
            + motion_diff * config.adaptive_config.motion_weight;

        // Check against adaptive threshold
        if combined_diff > adaptive_threshold {
            // Determine change type based on pattern
            let change_type =
                classify_change_type(hist_diff, edge_diff, motion_diff, &characteristics[i]);

            changes.push(SceneChange {
                frame_number: i,
                timestamp: frame2.timestamp,
                confidence: combined_diff,
                change_type,
                metadata: SceneMetadata {
                    histogram_diff: Some(hist_diff),
                    edge_change_ratio: Some(edge_diff),
                    motion_score: Some(motion_diff),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}

/// Classify the type of change based on different metrics.
fn classify_change_type(
    hist_diff: f64,
    edge_diff: f64,
    motion_diff: f64,
    _characteristics: &ContentCharacteristics,
) -> ChangeType {
    // All high -> hard cut
    if hist_diff > 0.5 && edge_diff > 0.5 && motion_diff > 0.5 {
        return ChangeType::Cut;
    }

    // High histogram but low edge/motion -> fade
    if hist_diff > 0.4 && (edge_diff < 0.3 || motion_diff < 0.3) {
        return ChangeType::Fade;
    }

    // Moderate in all -> dissolve
    if hist_diff > 0.3 && hist_diff < 0.6 && edge_diff > 0.2 && motion_diff > 0.2 {
        return ChangeType::Dissolve;
    }

    // Default to cut if we're not sure
    ChangeType::Cut
}

/// Multi-scale adaptive detection.
pub fn detect_multiscale_adaptive(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    if frames.len() < 2 {
        return Ok(Vec::new());
    }

    // Detect at multiple scales (different thresholds)
    let scales = vec![
        config.threshold * 0.8, // More sensitive
        config.threshold,       // Normal
        config.threshold * 1.2, // Less sensitive
    ];

    let mut all_changes = Vec::new();

    for scale_threshold in scales {
        let mut scale_config = config.clone();
        scale_config.threshold = scale_threshold;

        let changes = detect_adaptive_changes(frames, &scale_config)?;

        for change in changes {
            // Only add if not already detected nearby
            let exists = all_changes.iter().any(|c: &SceneChange| {
                (c.frame_number as i32 - change.frame_number as i32).abs() <= 2
            });

            if !exists {
                all_changes.push(change);
            }
        }
    }

    // Sort by frame number
    all_changes.sort_by_key(|c| c.frame_number);

    Ok(all_changes)
}

/// Compute confidence boost based on content characteristics.
#[must_use]
pub fn compute_confidence_boost(characteristics: &ContentCharacteristics) -> f64 {
    let mut boost: f64 = 1.0;

    // Boost confidence for high-detail scenes
    if characteristics.edge_density > 0.3 {
        boost += 0.1;
    }

    // Boost confidence for normal brightness
    if characteristics.brightness > 50.0 && characteristics.brightness < 200.0 {
        boost += 0.1;
    }

    // Reduce confidence for very low or high motion
    if characteristics.motion_activity < 0.1 || characteristics.motion_activity > 0.9 {
        boost -= 0.1;
    }

    boost.clamp(0.8, 1.3)
}

/// Adaptive temporal filtering.
pub fn adaptive_temporal_filter(
    changes: Vec<SceneChange>,
    characteristics: &[ContentCharacteristics],
    min_distance: usize,
) -> Vec<SceneChange> {
    if changes.is_empty() {
        return changes;
    }

    let mut filtered = Vec::new();
    let mut last_frame = 0;

    for change in changes {
        // Compute adaptive minimum distance based on content
        let adaptive_distance = if change.frame_number < characteristics.len() {
            let char = &characteristics[change.frame_number];

            // High motion -> longer minimum distance
            let motion_factor = 1.0 + char.motion_activity * 0.5;
            (min_distance as f64 * motion_factor) as usize
        } else {
            min_distance
        };

        if change.frame_number - last_frame >= adaptive_distance {
            filtered.push(change.clone());
            last_frame = change.frame_number;
        }
    }

    filtered
}

/// Compute scene stability score (how stable is the content).
#[must_use]
pub fn compute_scene_stability(frames: &[VideoFrame], start: usize, end: usize) -> f64 {
    if start >= end || end > frames.len() || start + 1 >= end {
        return 1.0; // Assume stable if we can't compute
    }

    let mut motion_scores = Vec::new();
    let config = motion::MotionConfig::default();

    for i in (start + 1)..end {
        if let Ok(score) = motion::compute_motion_score(&frames[i - 1], &frames[i], &config) {
            motion_scores.push(1.0 - score);
        }
    }

    if motion_scores.is_empty() {
        return 1.0;
    }

    // Compute variance of motion scores
    let mean: f64 = motion_scores.iter().sum::<f64>() / motion_scores.len() as f64;
    let variance: f64 = motion_scores
        .iter()
        .map(|s| {
            let diff = s - mean;
            diff * diff
        })
        .sum::<f64>()
        / motion_scores.len() as f64;

    // Low variance -> stable, high variance -> unstable
    let stability = 1.0 / (1.0 + variance);
    stability.clamp(0.0, 1.0)
}

/// Refine scene changes using content-aware post-processing.
pub fn refine_scene_changes(
    changes: Vec<SceneChange>,
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    let mut refined = Vec::new();

    let characteristics =
        ContentCharacteristics::analyze_sequence(frames, config.adaptive_config.analysis_window)?;

    for change in changes {
        let mut refined_change = change.clone();

        // Boost confidence based on content
        if refined_change.frame_number < characteristics.len() {
            let boost = compute_confidence_boost(&characteristics[refined_change.frame_number]);
            refined_change.confidence *= boost;
            refined_change.confidence = refined_change.confidence.clamp(0.0, 1.0);
        }

        // Verify the scene change by looking at neighboring frames
        let is_valid = verify_scene_change(frames, refined_change.frame_number, config)?;

        if is_valid {
            refined.push(refined_change);
        }
    }

    Ok(refined)
}

/// Verify a scene change by checking neighboring frames.
fn verify_scene_change(
    frames: &[VideoFrame],
    frame_number: usize,
    config: &SceneConfig,
) -> CvResult<bool> {
    // Check stability before and after the change
    let before_start = frame_number.saturating_sub(5);
    let before_end = frame_number;
    let after_start = frame_number;
    let after_end = (frame_number + 5).min(frames.len());

    if before_end <= before_start || after_end <= after_start {
        return Ok(true); // Can't verify, assume valid
    }

    let stability_before = compute_scene_stability(frames, before_start, before_end);
    let stability_after = compute_scene_stability(frames, after_start, after_end);

    // Both sides should be relatively stable for a valid scene change
    Ok(stability_before > 0.3 && stability_after > 0.3)
}
