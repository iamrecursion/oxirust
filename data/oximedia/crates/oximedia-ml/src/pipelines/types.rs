//! Shared value types for ML pipelines.
//!
//! These types are compiled regardless of which pipeline features are
//! enabled so that generic helpers (for example: re-exports at the
//! crate root, documentation examples, integration tests that validate
//! contract shapes) always resolve.
//!
//! Geometric primitives like [`BoundingBox`] live in
//! [`crate::postprocess`] because they are consumed by the always-on
//! IoU / NMS helpers.
//!
//! ## Types at a glance
//!
//! | Type                | Produced by                             | Notes                              |
//! |---------------------|-----------------------------------------|------------------------------------|
//! | [`Detection`]       | `crate::pipelines::ObjectDetector`      | `class_id` + corner bbox + score   |
//! | [`FaceEmbedding`]   | `crate::pipelines::FaceEmbedder`        | L2-normalised; use `cosine_similarity` |
//! | [`AestheticScore`]  | `crate::pipelines::AestheticScorer`     | NIMA weighted mean over 10 bins    |
//!
//! Value types are deliberately thin wrappers: they hold data and a few
//! convenience methods, no I/O. They are `Clone + Debug` so callers can
//! freely pass them through channels or collect them into containers.

use crate::postprocess::BoundingBox;

/// Single object-detection result.
///
/// `class_id` indexes into the detector's class vocabulary (COCO has
/// 80 classes, indices `0..=79`). `score` is a sigmoid-normalised
/// confidence in `0.0..=1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Bounding box in corner form.
    pub bbox: BoundingBox,
    /// Class index produced by the detector.
    pub class_id: u32,
    /// Post-sigmoid confidence in `[0, 1]`.
    pub score: f32,
}

impl Detection {
    /// Convenience constructor.
    #[must_use]
    pub fn new(bbox: BoundingBox, class_id: u32, score: f32) -> Self {
        Self {
            bbox,
            class_id,
            score,
        }
    }
}

/// L2-normalised face embedding.
///
/// The embedding is stored as a unit-norm `Vec<f32>` — callers can
/// compare embeddings directly via [`FaceEmbedding::cosine_similarity`].
///
/// # Examples
///
/// ```
/// use oximedia_ml::FaceEmbedding;
///
/// let a = FaceEmbedding::from_raw(vec![1.0, 0.0, 0.0]);
/// let b = FaceEmbedding::from_raw(vec![0.0, 1.0, 0.0]);
/// assert!(a.cosine_similarity(&b).abs() < 1e-5);
/// assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-5);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FaceEmbedding(Vec<f32>);

impl FaceEmbedding {
    /// Wrap an already-normalised vector without re-checking the norm.
    ///
    /// Prefer [`FaceEmbedding::from_raw`] unless you've just called
    /// [`crate::postprocess::l2_normalize`] yourself.
    #[must_use]
    pub fn from_unit(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Wrap a raw vector and L2-normalise it in place.
    #[must_use]
    pub fn from_raw(mut values: Vec<f32>) -> Self {
        crate::postprocess::l2_normalize(&mut values);
        Self(values)
    }

    /// Dimensionality of the embedding.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` if the embedding is empty (should never happen in practice).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Read-only view of the embedding values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Consume the wrapper and return the underlying storage.
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Cosine similarity with another embedding.
    ///
    /// Because both vectors are L2-normalised, this is equivalent to a
    /// dot product and is clamped to `[-1, 1]`. Returns `0.0` when
    /// dimensions mismatch.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        crate::postprocess::cosine_similarity(self.as_slice(), other.as_slice())
    }
}

/// Aesthetic score in the NIMA sense: weighted mean of a 10-bin
/// quality distribution.
///
/// Produced by `crate::pipelines::AestheticScorer`. `score()` returns
/// a value in the `[1.0, 10.0]` range for a well-formed distribution;
/// `distribution()` exposes the underlying 10-bin probabilities for
/// callers that want to inspect shape (variance, skew, ...).
///
/// # Examples
///
/// ```
/// use oximedia_ml::AestheticScore;
///
/// // Peaked at bin 10 → score of exactly 10.0.
/// let mut dist = [0.0_f32; 10];
/// dist[9] = 1.0;
/// let s = AestheticScore::from_distribution(dist);
/// assert!((s.score() - 10.0).abs() < 1e-5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AestheticScore {
    score: f32,
    distribution: [f32; 10],
}

impl AestheticScore {
    /// Build an [`AestheticScore`] from a 10-bin probability distribution.
    ///
    /// The distribution is expected to already be softmax-normalised;
    /// the function does not re-normalise but will still compute the
    /// weighted mean with whatever weights it is given.
    #[must_use]
    pub fn from_distribution(distribution: [f32; 10]) -> Self {
        let score = weighted_mean_score(&distribution);
        Self {
            score,
            distribution,
        }
    }

    /// Weighted-mean aesthetic score in `[1.0, 10.0]` (for a valid distribution).
    #[must_use]
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Full 10-bin probability distribution as produced by the model.
    #[must_use]
    pub fn distribution(&self) -> &[f32; 10] {
        &self.distribution
    }
}

/// NIMA-style weighted mean: `sum(i * p_{i-1})` for i=1..=10.
pub(crate) fn weighted_mean_score(distribution: &[f32; 10]) -> f32 {
    let mut acc = 0.0_f32;
    for (i, &p) in distribution.iter().enumerate() {
        acc += ((i + 1) as f32) * p;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_constructor_populates_fields() {
        let d = Detection::new(BoundingBox::new(0.0, 0.0, 1.0, 1.0), 3, 0.87);
        assert_eq!(d.class_id, 3);
        assert!((d.score - 0.87).abs() < 1e-6);
    }

    #[test]
    fn face_embedding_from_raw_is_unit_norm() {
        let emb = FaceEmbedding::from_raw(vec![3.0, 4.0]);
        let norm: f32 = emb.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn face_embedding_cosine_self_is_one() {
        let emb = FaceEmbedding::from_raw(vec![0.1, 0.2, 0.3, 0.4]);
        let sim = emb.cosine_similarity(&emb);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn face_embedding_orthogonal_is_zero() {
        let a = FaceEmbedding::from_raw(vec![1.0, 0.0, 0.0, 0.0]);
        let b = FaceEmbedding::from_raw(vec![0.0, 1.0, 0.0, 0.0]);
        assert!(a.cosine_similarity(&b).abs() < 1e-5);
    }

    #[test]
    fn aesthetic_score_uniform_distribution_is_5_5() {
        let dist = [0.1_f32; 10];
        let s = AestheticScore::from_distribution(dist);
        // 0.1 * (1+2+...+10) = 0.1 * 55 = 5.5
        assert!((s.score() - 5.5).abs() < 1e-5);
    }

    #[test]
    fn aesthetic_score_peaked_at_ten_is_ten() {
        let mut dist = [0.0_f32; 10];
        dist[9] = 1.0;
        let s = AestheticScore::from_distribution(dist);
        assert!((s.score() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn aesthetic_score_peaked_at_one_is_one() {
        let mut dist = [0.0_f32; 10];
        dist[0] = 1.0;
        let s = AestheticScore::from_distribution(dist);
        assert!((s.score() - 1.0).abs() < 1e-5);
    }
}
