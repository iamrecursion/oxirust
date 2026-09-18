//! Camera angle scoring for automated selection.
//!
//! Provides `ScoringMetric`, `AngleScore`, and `AngleScorer` to evaluate and
//! rank camera angles based on multiple perceptual criteria.

/// A single criterion used to evaluate a camera angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoringMetric {
    /// Sharpness / focus quality.
    Focus,
    /// Correct exposure (not blown or crushed).
    Exposure,
    /// Amount and quality of on-screen motion.
    Motion,
    /// Rule-of-thirds and framing quality.
    Composition,
}

impl ScoringMetric {
    /// Default weight for this metric in a composite score (0.0–1.0).
    #[must_use]
    pub fn weight(&self) -> f32 {
        match self {
            Self::Focus => 0.35,
            Self::Exposure => 0.25,
            Self::Motion => 0.20,
            Self::Composition => 0.20,
        }
    }
}

/// Per-metric score for a single camera angle.
#[derive(Debug, Clone)]
pub struct AngleScore {
    /// The angle index this score belongs to.
    pub angle_index: usize,
    /// Focus score in \[0.0, 1.0\].
    pub focus: f32,
    /// Exposure score in \[0.0, 1.0\].
    pub exposure: f32,
    /// Motion score in \[0.0, 1.0\].
    pub motion: f32,
    /// Composition score in \[0.0, 1.0\].
    pub composition: f32,
}

impl AngleScore {
    /// Create a new `AngleScore` with all metrics set to zero.
    #[must_use]
    pub fn new(angle_index: usize) -> Self {
        Self {
            angle_index,
            focus: 0.0,
            exposure: 0.0,
            motion: 0.0,
            composition: 0.0,
        }
    }

    /// Compute a weighted total score across all metrics.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_score(&self) -> f32 {
        self.focus * ScoringMetric::Focus.weight()
            + self.exposure * ScoringMetric::Exposure.weight()
            + self.motion * ScoringMetric::Motion.weight()
            + self.composition * ScoringMetric::Composition.weight()
    }
}

// ── Rule-of-thirds composition scoring ───────────────────────────────────────

/// Score a frame's composition quality using the rule-of-thirds principle.
///
/// The rule of thirds divides the frame into a 3×3 grid.  Interesting content
/// placed at any of the four intersection points (1/3 and 2/3 of width/height)
/// produces a more compelling composition than content centred on the frame.
///
/// # Algorithm
///
/// The function computes a simple edge-density metric (Sobel-like gradient
/// magnitude) sampled in small patches centred on each of the four
/// rule-of-thirds intersections and normalises it against the global frame
/// edge density.
///
/// A frame whose interesting edges are concentrated at those four points will
/// score close to `1.0`.  A uniform or featureless frame will score close to
/// `0.5`.
///
/// # Arguments
///
/// * `frame` – Raw pixel data in **grayscale** format (one byte per pixel) or
///   any interleaved format; only luminance is used.  For RGBA data pass the
///   raw bytes and set `bytes_per_pixel = 4`; for grayscale use
///   `bytes_per_pixel = 1`.
/// * `width`  – Frame width in pixels.
/// * `height` – Frame height in pixels.
///
/// # Returns
///
/// A normalised score in `[0.0, 1.0]`.  Returns `0.5` when the frame is
/// empty or contains insufficient data.
///
/// # Note on input format
///
/// This implementation treats every `bytes_per_pixel`-th byte as a luminance
/// sample.  Pass a grayscale frame (1 byte/pixel) for the most accurate
/// result; RGBA is supported by extracting every 4th byte as the red channel.
/// Strictly speaking the function signature takes raw `u8` bytes and assumes
/// 1 byte per pixel.  Callers with multi-channel data should convert to
/// grayscale first or pass a pre-extracted luma plane.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn score_rule_of_thirds(frame: &[u8], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;

    if frame.is_empty() || w < 6 || h < 6 || frame.len() < w * h {
        return 0.5;
    }

    // Patch half-side in pixels.  Use ~3% of the smaller dimension, min 2.
    let patch_r = ((w.min(h) as f32 * 0.03) as usize).max(2);

    // Rule-of-thirds intersection points (column, row) in pixel coordinates.
    let thirds_x = [w / 3, 2 * w / 3];
    let thirds_y = [h / 3, 2 * h / 3];

    // Helper: compute mean absolute gradient magnitude in a patch centred at (cx, cy).
    let patch_gradient = |cx: usize, cy: usize| -> f32 {
        let x0 = cx.saturating_sub(patch_r);
        let x1 = (cx + patch_r).min(w.saturating_sub(2));
        let y0 = cy.saturating_sub(patch_r);
        let y1 = (cy + patch_r).min(h.saturating_sub(2));

        if x0 >= x1 || y0 >= y1 {
            return 0.0;
        }

        let mut sum = 0.0f64;
        let mut count = 0usize;

        for row in y0..y1 {
            for col in x0..x1 {
                // Horizontal gradient (central difference, clamped)
                let left = frame[row * w + col.saturating_sub(1)] as f64;
                let right = frame[row * w + (col + 1).min(w - 1)] as f64;
                let gx = (right - left).abs();

                // Vertical gradient
                let up = frame[row.saturating_sub(1) * w + col] as f64;
                let down = frame[(row + 1).min(h - 1) * w + col] as f64;
                let gy = (down - up).abs();

                sum += (gx * gx + gy * gy).sqrt();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        }
    };

    // Sum gradient energy at the four rule-of-thirds intersection points.
    let mut thirds_energy = 0.0f32;
    for &cx in &thirds_x {
        for &cy in &thirds_y {
            thirds_energy += patch_gradient(cx, cy);
        }
    }
    let mean_thirds = thirds_energy / 4.0;

    // Global mean gradient (sampled on a coarse 8×8 grid to avoid O(w*h) cost).
    let step_x = (w / 8).max(1);
    let step_y = (h / 8).max(1);
    let mut global_sum = 0.0f64;
    let mut global_count = 0usize;

    let mut row = 1usize;
    while row < h.saturating_sub(1) {
        let mut col = 1usize;
        while col < w.saturating_sub(1) {
            let left = frame[row * w + col - 1] as f64;
            let right = frame[row * w + col + 1] as f64;
            let gx = (right - left).abs();
            let up = frame[(row - 1) * w + col] as f64;
            let down = frame[(row + 1) * w + col] as f64;
            let gy = (down - up).abs();
            global_sum += (gx * gx + gy * gy).sqrt();
            global_count += 1;
            col += step_x;
        }
        row += step_y;
    }

    let mean_global = if global_count == 0 {
        1.0f32
    } else {
        (global_sum / global_count as f64) as f32
    };

    if mean_global < f32::EPSILON {
        // Featureless frame – no edges anywhere; return neutral score.
        return 0.5;
    }

    // Ratio of thirds-intersection energy to global energy, normalised to [0,1].
    // A ratio of 1.0 means thirds-energy equals the global average → score 0.5.
    // A ratio > 1.0 (interesting content at intersections) → score > 0.5.
    // A ratio of 0.0 → score 0.0.
    let ratio = mean_thirds / mean_global;
    // Map ratio through a sigmoid-like ramp so that the neutral point (1.0)
    // maps to ~0.5 and values grow/shrink smoothly.
    let score = ratio / (1.0 + ratio);
    score.clamp(0.0, 1.0)
}

// ── AngleScorer ───────────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Accumulates per-metric data and produces `AngleScore` results.
///
/// The scorer maintains an internal memoization cache keyed by `(angle_index,
/// frame_number)` so that repeated queries for the same angle/frame pair
/// return immediately without re-running any expensive detection pipeline.
/// Call [`invalidate_cache`] whenever the underlying video data or scorer
/// configuration changes.
///
/// [`invalidate_cache`]: AngleScorer::invalidate_cache
#[derive(Debug, Default)]
pub struct AngleScorer {
    scores: Vec<AngleScore>,
    /// Memoization cache: (angle_index, frame_number) → AngleScore
    score_cache: HashMap<(usize, u64), AngleScore>,
}

impl AngleScorer {
    /// Create a new, empty scorer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pre-built `AngleScore`.
    pub fn score_angle(&mut self, score: AngleScore) {
        self.scores.push(score);
    }

    /// Retrieve (or insert) the cached score for a specific `(angle_index,
    /// frame_number)` pair.
    ///
    /// On the first call for a given key the supplied `compute` closure is
    /// invoked to produce the score; the result is stored and returned.
    /// Subsequent calls for the **same** key return the cached value without
    /// calling `compute` again.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_multicam::angle_score::{AngleScore, AngleScorer};
    ///
    /// let mut scorer = AngleScorer::new();
    /// let score = scorer.score_for_frame(0, 42, |angle_idx, _frame| {
    ///     // Heavy detection pipeline would run here in production.
    ///     let mut s = AngleScore::new(angle_idx);
    ///     s.focus = 0.9;
    ///     s
    /// });
    /// assert!((score.focus - 0.9).abs() < 1e-6);
    /// ```
    pub fn score_for_frame(
        &mut self,
        angle_index: usize,
        frame_number: u64,
        compute: impl Fn(usize, u64) -> AngleScore,
    ) -> AngleScore {
        let key = (angle_index, frame_number);
        if let Some(cached) = self.score_cache.get(&key) {
            return cached.clone();
        }
        let score = compute(angle_index, frame_number);
        self.score_cache.insert(key, score.clone());
        score
    }

    /// Discard all cached per-frame scores.
    ///
    /// Call this whenever the underlying video data or scorer configuration
    /// changes to ensure stale values are not returned.
    pub fn invalidate_cache(&mut self) {
        self.score_cache.clear();
    }

    /// Return the number of entries currently held in the score cache.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.score_cache.len()
    }

    /// Return the index of the angle with the highest total score, or `None`
    /// if no angles have been added.
    #[must_use]
    pub fn best_angle(&self) -> Option<usize> {
        self.scores
            .iter()
            .max_by(|a, b| {
                a.total_score()
                    .partial_cmp(&b.total_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.angle_index)
    }

    /// Return all stored scores.
    #[must_use]
    pub fn scores(&self) -> &[AngleScore] {
        &self.scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_weights_sum_to_one() {
        let sum = ScoringMetric::Focus.weight()
            + ScoringMetric::Exposure.weight()
            + ScoringMetric::Motion.weight()
            + ScoringMetric::Composition.weight();
        assert!(
            (sum - 1.0_f32).abs() < 1e-6,
            "weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_focus_weight() {
        assert!((ScoringMetric::Focus.weight() - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_angle_score_zero_total() {
        let s = AngleScore::new(0);
        assert!((s.total_score()).abs() < 1e-6);
    }

    #[test]
    fn test_angle_score_perfect() {
        let s = AngleScore {
            angle_index: 0,
            focus: 1.0,
            exposure: 1.0,
            motion: 1.0,
            composition: 1.0,
        };
        assert!((s.total_score() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_angle_score_partial() {
        let s = AngleScore {
            angle_index: 1,
            focus: 0.8,
            exposure: 0.6,
            motion: 0.5,
            composition: 0.7,
        };
        let expected = 0.8 * 0.35 + 0.6 * 0.25 + 0.5 * 0.20 + 0.7 * 0.20;
        assert!((s.total_score() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_scorer_empty_best_angle() {
        let scorer = AngleScorer::new();
        assert!(scorer.best_angle().is_none());
    }

    #[test]
    fn test_scorer_single_angle() {
        let mut scorer = AngleScorer::new();
        scorer.score_angle(AngleScore {
            angle_index: 2,
            focus: 0.9,
            exposure: 0.9,
            motion: 0.9,
            composition: 0.9,
        });
        assert_eq!(scorer.best_angle(), Some(2));
    }

    #[test]
    fn test_scorer_best_of_two() {
        let mut scorer = AngleScorer::new();
        scorer.score_angle(AngleScore {
            angle_index: 0,
            focus: 0.5,
            exposure: 0.5,
            motion: 0.5,
            composition: 0.5,
        });
        scorer.score_angle(AngleScore {
            angle_index: 1,
            focus: 0.9,
            exposure: 0.9,
            motion: 0.9,
            composition: 0.9,
        });
        assert_eq!(scorer.best_angle(), Some(1));
    }

    #[test]
    fn test_scorer_best_of_three() {
        let mut scorer = AngleScorer::new();
        scorer.score_angle(AngleScore {
            angle_index: 0,
            focus: 0.3,
            exposure: 0.3,
            motion: 0.3,
            composition: 0.3,
        });
        scorer.score_angle(AngleScore {
            angle_index: 1,
            focus: 1.0,
            exposure: 1.0,
            motion: 1.0,
            composition: 1.0,
        });
        scorer.score_angle(AngleScore {
            angle_index: 2,
            focus: 0.7,
            exposure: 0.7,
            motion: 0.7,
            composition: 0.7,
        });
        assert_eq!(scorer.best_angle(), Some(1));
    }

    #[test]
    fn test_scorer_scores_accessor() {
        let mut scorer = AngleScorer::new();
        scorer.score_angle(AngleScore::new(0));
        scorer.score_angle(AngleScore::new(1));
        assert_eq!(scorer.scores().len(), 2);
    }

    #[test]
    fn test_angle_index_preserved() {
        let s = AngleScore::new(7);
        assert_eq!(s.angle_index, 7);
    }

    #[test]
    fn test_composition_metric_weight() {
        assert!((ScoringMetric::Composition.weight() - 0.20).abs() < 1e-6);
    }

    // ── Cache tests ──────────────────────────────────────────────────────────

    /// Score the same (angle, frame) twice; the results must be identical and
    /// the compute closure must only be called once.
    #[test]
    fn test_angle_score_cache_hit() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let mut scorer = AngleScorer::new();

        let score_first = {
            let cc = Arc::clone(&call_count);
            scorer.score_for_frame(0, 10, move |idx, _frame| {
                cc.fetch_add(1, Ordering::SeqCst);
                AngleScore {
                    angle_index: idx,
                    focus: 0.8,
                    exposure: 0.7,
                    motion: 0.6,
                    composition: 0.5,
                }
            })
        };

        let score_second = {
            let cc = Arc::clone(&call_count);
            scorer.score_for_frame(0, 10, move |idx, _frame| {
                cc.fetch_add(1, Ordering::SeqCst);
                AngleScore {
                    angle_index: idx,
                    focus: 0.8,
                    exposure: 0.7,
                    motion: 0.6,
                    composition: 0.5,
                }
            })
        };

        // Compute closure must only have run once (cache hit on second call).
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "compute closure should be called exactly once"
        );

        // Both calls must return the same score values.
        assert!((score_first.focus - score_second.focus).abs() < 1e-6);
        assert!((score_first.exposure - score_second.exposure).abs() < 1e-6);
        assert!((score_first.motion - score_second.motion).abs() < 1e-6);
        assert!((score_first.composition - score_second.composition).abs() < 1e-6);
        assert_eq!(scorer.cache_len(), 1);
    }

    /// After invalidation the compute closure must be called again.
    #[test]
    fn test_angle_score_cache_invalidation() {
        let mut scorer = AngleScorer::new();

        let make_score = |idx: usize, val: f32| AngleScore {
            angle_index: idx,
            focus: val,
            exposure: val,
            motion: val,
            composition: val,
        };

        // Populate the cache.
        let before = scorer.score_for_frame(1, 5, |idx, _| make_score(idx, 0.4));
        assert_eq!(scorer.cache_len(), 1);

        // Invalidate and re-score — results should still be consistent because
        // the closure is deterministic, but the cache must have been cleared.
        scorer.invalidate_cache();
        assert_eq!(
            scorer.cache_len(),
            0,
            "cache should be empty after invalidation"
        );

        let after = scorer.score_for_frame(1, 5, |idx, _| make_score(idx, 0.4));

        // Both results originate from the same computation logic.
        assert!(
            (before.focus - after.focus).abs() < 1e-6,
            "scores before and after invalidation should match when computation is deterministic"
        );
        assert_eq!(
            scorer.cache_len(),
            1,
            "cache should hold one entry after re-scoring"
        );
    }
}
