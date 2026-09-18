#![allow(dead_code)]
//! Shot-to-shot matching and similarity analysis.
//!
//! This module provides algorithms for comparing shots to find similar shots
//! based on visual appearance, duration, camera parameters, and composition.
//! Useful for finding matching shots across takes, detecting repeated setups,
//! and organizing shot libraries.

use std::collections::HashMap;

/// Criteria used when comparing shots for similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchCriterion {
    /// Compare shot type (close-up, medium, wide, etc.).
    ShotType,
    /// Compare shot duration.
    Duration,
    /// Compare camera angle.
    CameraAngle,
    /// Compare dominant motion direction.
    MotionDirection,
    /// Compare average luminance.
    Luminance,
    /// Compare color histogram similarity.
    ColorHistogram,
    /// Compare composition features.
    Composition,
}

/// Weight configuration for match criteria.
#[derive(Debug, Clone)]
pub struct MatchWeights {
    /// Per-criterion weights.
    weights: HashMap<MatchCriterion, f64>,
}

impl MatchWeights {
    /// Create default weights with all criteria at 1.0.
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert(MatchCriterion::ShotType, 1.0);
        weights.insert(MatchCriterion::Duration, 0.8);
        weights.insert(MatchCriterion::CameraAngle, 0.9);
        weights.insert(MatchCriterion::MotionDirection, 0.5);
        weights.insert(MatchCriterion::Luminance, 0.6);
        weights.insert(MatchCriterion::ColorHistogram, 0.7);
        weights.insert(MatchCriterion::Composition, 0.8);
        Self { weights }
    }

    /// Set the weight for a specific criterion.
    pub fn set(&mut self, criterion: MatchCriterion, weight: f64) {
        self.weights.insert(criterion, weight);
    }

    /// Get the weight for a criterion.
    pub fn get(&self, criterion: &MatchCriterion) -> f64 {
        self.weights.get(criterion).copied().unwrap_or(0.0)
    }

    /// Return total number of criteria.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Default for MatchWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature descriptor for a single shot used in matching.
#[derive(Debug, Clone)]
pub struct ShotDescriptor {
    /// Unique shot identifier.
    pub shot_id: u64,
    /// Shot type label (e.g. "CU", "MS", "LS").
    pub shot_type: String,
    /// Duration in frames.
    pub duration_frames: u64,
    /// Camera angle label.
    pub camera_angle: String,
    /// Dominant motion vector (dx, dy) normalized.
    pub motion: (f64, f64),
    /// Average luminance (0.0..1.0).
    pub avg_luminance: f64,
    /// Simplified color histogram bins (8 bins, normalized).
    pub color_histogram: [f64; 8],
    /// Composition score (0.0..1.0).
    pub composition_score: f64,
}

impl ShotDescriptor {
    /// Create a new shot descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shot_id: u64,
        shot_type: &str,
        duration_frames: u64,
        camera_angle: &str,
        motion: (f64, f64),
        avg_luminance: f64,
        color_histogram: [f64; 8],
        composition_score: f64,
    ) -> Self {
        Self {
            shot_id,
            shot_type: shot_type.to_string(),
            duration_frames,
            camera_angle: camera_angle.to_string(),
            motion,
            avg_luminance,
            color_histogram,
            composition_score,
        }
    }
}

/// A match result pairing two shots with a similarity score.
#[derive(Debug, Clone)]
pub struct ShotMatch {
    /// First shot ID.
    pub shot_a: u64,
    /// Second shot ID.
    pub shot_b: u64,
    /// Overall similarity score (0.0..1.0).
    pub similarity: f64,
    /// Per-criterion scores.
    pub criterion_scores: HashMap<MatchCriterion, f64>,
}

/// Shot matching engine that compares shots pairwise.
#[derive(Debug, Clone)]
pub struct ShotMatcher {
    /// Criteria weights.
    weights: MatchWeights,
    /// Minimum similarity threshold to report a match.
    threshold: f64,
    /// Maximum duration difference ratio for duration matching.
    max_duration_ratio: f64,
}

impl ShotMatcher {
    /// Create a new shot matcher with default settings.
    pub fn new() -> Self {
        Self {
            weights: MatchWeights::new(),
            threshold: 0.5,
            max_duration_ratio: 2.0,
        }
    }

    /// Set custom weights.
    pub fn with_weights(mut self, weights: MatchWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the minimum similarity threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum duration ratio.
    pub fn with_max_duration_ratio(mut self, ratio: f64) -> Self {
        self.max_duration_ratio = ratio;
        self
    }

    /// Compare two shots and return per-criterion and aggregate similarity.
    #[allow(clippy::cast_precision_loss)]
    pub fn compare(&self, a: &ShotDescriptor, b: &ShotDescriptor) -> ShotMatch {
        let mut criterion_scores = HashMap::new();
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        // Shot type similarity
        let shot_type_sim = if a.shot_type == b.shot_type { 1.0 } else { 0.0 };
        criterion_scores.insert(MatchCriterion::ShotType, shot_type_sim);
        let w = self.weights.get(&MatchCriterion::ShotType);
        weighted_sum += shot_type_sim * w;
        weight_total += w;

        // Duration similarity
        let dur_sim = self.duration_similarity(a.duration_frames, b.duration_frames);
        criterion_scores.insert(MatchCriterion::Duration, dur_sim);
        let w = self.weights.get(&MatchCriterion::Duration);
        weighted_sum += dur_sim * w;
        weight_total += w;

        // Camera angle
        let angle_sim = if a.camera_angle == b.camera_angle {
            1.0
        } else {
            0.0
        };
        criterion_scores.insert(MatchCriterion::CameraAngle, angle_sim);
        let w = self.weights.get(&MatchCriterion::CameraAngle);
        weighted_sum += angle_sim * w;
        weight_total += w;

        // Motion direction similarity (cosine similarity of 2D vectors)
        let motion_sim = self.motion_similarity(a.motion, b.motion);
        criterion_scores.insert(MatchCriterion::MotionDirection, motion_sim);
        let w = self.weights.get(&MatchCriterion::MotionDirection);
        weighted_sum += motion_sim * w;
        weight_total += w;

        // Luminance similarity
        let lum_sim = 1.0 - (a.avg_luminance - b.avg_luminance).abs();
        criterion_scores.insert(MatchCriterion::Luminance, lum_sim.max(0.0));
        let w = self.weights.get(&MatchCriterion::Luminance);
        weighted_sum += lum_sim.max(0.0) * w;
        weight_total += w;

        // Color histogram similarity (histogram intersection)
        let color_sim = self.histogram_similarity(&a.color_histogram, &b.color_histogram);
        criterion_scores.insert(MatchCriterion::ColorHistogram, color_sim);
        let w = self.weights.get(&MatchCriterion::ColorHistogram);
        weighted_sum += color_sim * w;
        weight_total += w;

        // Composition score similarity
        let comp_sim = 1.0 - (a.composition_score - b.composition_score).abs();
        criterion_scores.insert(MatchCriterion::Composition, comp_sim.max(0.0));
        let w = self.weights.get(&MatchCriterion::Composition);
        weighted_sum += comp_sim.max(0.0) * w;
        weight_total += w;

        let similarity = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        };

        ShotMatch {
            shot_a: a.shot_id,
            shot_b: b.shot_id,
            similarity,
            criterion_scores,
        }
    }

    /// Find all matches above threshold in a collection of shots.
    pub fn find_matches(&self, shots: &[ShotDescriptor]) -> Vec<ShotMatch> {
        let mut matches = Vec::new();
        for i in 0..shots.len() {
            for j in (i + 1)..shots.len() {
                let m = self.compare(&shots[i], &shots[j]);
                if m.similarity >= self.threshold {
                    matches.push(m);
                }
            }
        }
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Find the best match for a query shot within a collection.
    pub fn find_best_match<'a>(
        &self,
        query: &ShotDescriptor,
        candidates: &'a [ShotDescriptor],
    ) -> Option<(usize, ShotMatch)> {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.shot_id != query.shot_id)
            .map(|(idx, c)| (idx, self.compare(query, c)))
            .max_by(|(_, a), (_, b)| {
                a.similarity
                    .partial_cmp(&b.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Duration similarity based on ratio (1.0 = identical, 0.0 = very different).
    #[allow(clippy::cast_precision_loss)]
    fn duration_similarity(&self, a: u64, b: u64) -> f64 {
        if a == 0 && b == 0 {
            return 1.0;
        }
        let (shorter, longer) = if a <= b { (a, b) } else { (b, a) };
        if shorter == 0 {
            return 0.0;
        }
        let ratio = longer as f64 / shorter as f64;
        if ratio > self.max_duration_ratio {
            0.0
        } else {
            1.0 - (ratio - 1.0) / (self.max_duration_ratio - 1.0)
        }
    }

    /// Cosine similarity of 2D motion vectors.
    fn motion_similarity(&self, a: (f64, f64), b: (f64, f64)) -> f64 {
        let dot = a.0 * b.0 + a.1 * b.1;
        let mag_a = (a.0 * a.0 + a.1 * a.1).sqrt();
        let mag_b = (b.0 * b.0 + b.1 * b.1).sqrt();
        if mag_a < f64::EPSILON || mag_b < f64::EPSILON {
            return if mag_a < f64::EPSILON && mag_b < f64::EPSILON {
                1.0
            } else {
                0.0
            };
        }
        let cos = dot / (mag_a * mag_b);
        (cos + 1.0) / 2.0 // Normalize from [-1,1] to [0,1]
    }

    /// Histogram intersection similarity.
    fn histogram_similarity(&self, a: &[f64; 8], b: &[f64; 8]) -> f64 {
        let mut intersection = 0.0_f64;
        let mut sum_a = 0.0_f64;
        for i in 0..8 {
            intersection += a[i].min(b[i]);
            sum_a += a[i];
        }
        if sum_a > f64::EPSILON {
            intersection / sum_a
        } else {
            0.0
        }
    }
}

impl Default for ShotMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_descriptor(
        id: u64,
        shot_type: &str,
        dur: u64,
        angle: &str,
        luminance: f64,
    ) -> ShotDescriptor {
        ShotDescriptor::new(
            id,
            shot_type,
            dur,
            angle,
            (0.0, 0.0),
            luminance,
            [0.125; 8],
            0.5,
        )
    }

    #[test]
    fn test_identical_shots_perfect_match() {
        let matcher = ShotMatcher::new();
        let a = make_descriptor(1, "CU", 30, "eye", 0.5);
        let b = make_descriptor(2, "CU", 30, "eye", 0.5);
        let m = matcher.compare(&a, &b);
        assert!((m.similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_different_shot_types_lower_score() {
        let matcher = ShotMatcher::new();
        let a = make_descriptor(1, "CU", 30, "eye", 0.5);
        let b = make_descriptor(2, "LS", 30, "eye", 0.5);
        let m = matcher.compare(&a, &b);
        assert!(m.similarity < 1.0);
        assert_eq!(
            *m.criterion_scores
                .get(&MatchCriterion::ShotType)
                .expect("should succeed in test") as i32,
            0
        );
    }

    #[test]
    fn test_duration_similarity_same() {
        let matcher = ShotMatcher::new();
        let sim = matcher.duration_similarity(60, 60);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_duration_similarity_double() {
        let matcher = ShotMatcher::new().with_max_duration_ratio(2.0);
        let sim = matcher.duration_similarity(30, 60);
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn test_duration_similarity_zero() {
        let matcher = ShotMatcher::new();
        let sim = matcher.duration_similarity(0, 0);
        assert!((sim - 1.0).abs() < f64::EPSILON);
        let sim2 = matcher.duration_similarity(0, 30);
        assert!(sim2.abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_similarity_same_direction() {
        let matcher = ShotMatcher::new();
        let sim = matcher.motion_similarity((1.0, 0.0), (2.0, 0.0));
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_motion_similarity_opposite() {
        let matcher = ShotMatcher::new();
        let sim = matcher.motion_similarity((1.0, 0.0), (-1.0, 0.0));
        assert!(sim < 0.01);
    }

    #[test]
    fn test_motion_similarity_both_zero() {
        let matcher = ShotMatcher::new();
        let sim = matcher.motion_similarity((0.0, 0.0), (0.0, 0.0));
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_similarity_identical() {
        let matcher = ShotMatcher::new();
        let h = [0.125; 8];
        let sim = matcher.histogram_similarity(&h, &h);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_find_matches_threshold() {
        let matcher = ShotMatcher::new().with_threshold(0.9);
        let shots = vec![
            make_descriptor(1, "CU", 30, "eye", 0.5),
            make_descriptor(2, "CU", 30, "eye", 0.5),
            make_descriptor(3, "LS", 120, "high", 0.2),
        ];
        let matches = matcher.find_matches(&shots);
        // Shots 1 and 2 are nearly identical, 3 is very different
        assert!(!matches.is_empty());
        assert_eq!(matches[0].shot_a, 1);
        assert_eq!(matches[0].shot_b, 2);
    }

    #[test]
    fn test_find_best_match() {
        let matcher = ShotMatcher::new();
        let query = make_descriptor(1, "CU", 30, "eye", 0.5);
        let candidates = vec![
            make_descriptor(2, "LS", 120, "high", 0.2),
            make_descriptor(3, "CU", 30, "eye", 0.5),
        ];
        let best = matcher.find_best_match(&query, &candidates);
        assert!(best.is_some());
        let (idx, _) = best.expect("should succeed in test");
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_match_weights() {
        let mut w = MatchWeights::new();
        assert!(!w.is_empty());
        w.set(MatchCriterion::ShotType, 2.0);
        assert!((w.get(&MatchCriterion::ShotType) - 2.0).abs() < f64::EPSILON);
        assert!(w.len() >= 7);
    }
}
