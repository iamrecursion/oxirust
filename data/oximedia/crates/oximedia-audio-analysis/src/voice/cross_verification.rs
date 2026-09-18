//! Cross-recording speaker verification across different sessions.
//!
//! Compares speaker embeddings to verify same-speaker identity across recording sessions.

/// Result of a single speaker embedding comparison.
#[derive(Debug, Clone)]
pub struct SpeakerVerificationResult {
    /// Cosine similarity in \[-1.0, 1.0\]; higher = more similar.
    pub cosine_similarity: f64,
    /// Euclidean distance between embeddings; lower = more similar.
    pub euclidean_distance: f64,
    /// Whether the embeddings are judged to be the same speaker.
    pub is_same_speaker: bool,
}

/// Result of cross-session speaker comparison (multiple embeddings per session).
#[derive(Debug, Clone)]
pub struct CrossSessionResult {
    /// Cosine similarity of session centroids.
    pub centroid_cosine_similarity: f64,
    /// Euclidean distance between session centroids.
    pub centroid_euclidean_distance: f64,
    /// Minimum cosine similarity in the pairwise matrix.
    pub min_cosine_similarity: f64,
    /// Maximum cosine similarity in the pairwise matrix.
    pub max_cosine_similarity: f64,
    /// Mean cosine similarity across all pairwise comparisons.
    pub mean_cosine_similarity: f64,
    /// Whether the two sessions are judged to be the same speaker.
    pub is_same_speaker: bool,
}

/// Cross-recording verifier for speaker identity.
///
/// Compares speaker embeddings using cosine similarity and Euclidean distance.
/// The `same_speaker_threshold` controls when two embeddings are declared same-speaker
/// (default 0.75 cosine similarity).
pub struct CrossRecordingVerifier {
    /// Cosine similarity threshold above which speakers are considered the same.
    pub same_speaker_threshold: f64,
}

impl Default for CrossRecordingVerifier {
    fn default() -> Self {
        Self {
            same_speaker_threshold: 0.75,
        }
    }
}

impl CrossRecordingVerifier {
    /// Create a new verifier with the default threshold (0.75).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a verifier with a custom similarity threshold.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            same_speaker_threshold: threshold,
        }
    }

    /// Compare two speaker embeddings.
    ///
    /// Returns a [`SpeakerVerificationResult`] with cosine similarity, Euclidean distance,
    /// and a same-speaker decision.
    ///
    /// If either embedding is empty or all-zeros, cosine similarity is 0 and the
    /// speakers are not considered the same.
    #[must_use]
    pub fn compare_speakers(
        &self,
        embedding_a: &[f32],
        embedding_b: &[f32],
    ) -> SpeakerVerificationResult {
        let cosine_similarity = cosine_similarity_f32(embedding_a, embedding_b);
        let euclidean_distance = euclidean_distance_f32(embedding_a, embedding_b);
        let is_same_speaker = cosine_similarity >= self.same_speaker_threshold;
        SpeakerVerificationResult {
            cosine_similarity,
            euclidean_distance,
            is_same_speaker,
        }
    }

    /// Compare two recording sessions, each described by a collection of embeddings.
    ///
    /// Sessions are compared via:
    /// 1. Centroid comparison (mean embedding per session).
    /// 2. Full pairwise similarity matrix (min / max / mean).
    ///
    /// The `is_same_speaker` flag is set when the centroid cosine similarity
    /// meets the threshold.
    ///
    /// Returns `None` if either session slice is empty.
    #[must_use]
    pub fn verify_across_sessions(
        &self,
        session_a: &[Vec<f32>],
        session_b: &[Vec<f32>],
    ) -> Option<CrossSessionResult> {
        if session_a.is_empty() || session_b.is_empty() {
            return None;
        }

        let centroid_a = compute_centroid(session_a)?;
        let centroid_b = compute_centroid(session_b)?;

        let centroid_cosine_similarity = cosine_similarity_f32(&centroid_a, &centroid_b);
        let centroid_euclidean_distance = euclidean_distance_f32(&centroid_a, &centroid_b);

        // Full pairwise matrix
        let mut all_similarities: Vec<f64> = Vec::with_capacity(session_a.len() * session_b.len());
        for emb_a in session_a {
            for emb_b in session_b {
                all_similarities.push(cosine_similarity_f32(emb_a, emb_b));
            }
        }

        let min_cosine_similarity = all_similarities
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_cosine_similarity = all_similarities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_cosine_similarity = if all_similarities.is_empty() {
            0.0
        } else {
            all_similarities.iter().sum::<f64>() / all_similarities.len() as f64
        };

        let is_same_speaker = centroid_cosine_similarity >= self.same_speaker_threshold;

        Some(CrossSessionResult {
            centroid_cosine_similarity,
            centroid_euclidean_distance,
            min_cosine_similarity,
            max_cosine_similarity,
            mean_cosine_similarity,
            is_same_speaker,
        })
    }
}

// --- helpers ---------------------------------------------------------------

/// Cosine similarity between two f32 slices.
/// Returns 0.0 when either vector is zero-length or has zero norm.
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for i in 0..len {
        let ai = f64::from(a[i]);
        let bi = f64::from(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Euclidean distance between two f32 slices (shortest prefix).
fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let sq_sum: f64 = (0..len)
        .map(|i| {
            let d = f64::from(a[i]) - f64::from(b[i]);
            d * d
        })
        .sum();
    sq_sum.sqrt()
}

/// Compute the mean embedding (centroid) over a session.
/// Returns `None` when embeddings have length 0 or inconsistent lengths.
fn compute_centroid(session: &[Vec<f32>]) -> Option<Vec<f32>> {
    if session.is_empty() {
        return None;
    }
    let dim = session[0].len();
    if dim == 0 {
        return None;
    }
    let mut centroid = vec![0.0_f32; dim];
    for embedding in session {
        let d = dim.min(embedding.len());
        for i in 0..d {
            centroid[i] += embedding[i];
        }
    }
    let n = session.len() as f32;
    for v in &mut centroid {
        *v /= n;
    }
    Some(centroid)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    fn orthogonal_embedding(dim: usize) -> Vec<f32> {
        // Alternating signs — very different direction from a constant vector.
        (0..dim)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect()
    }

    #[test]
    fn test_same_speaker_high_similarity() {
        let verifier = CrossRecordingVerifier::new();

        // Identical embeddings → cosine similarity = 1.0
        let emb_a = make_embedding(64, 0.5);
        let emb_b = make_embedding(64, 0.5);

        let result = verifier.compare_speakers(&emb_a, &emb_b);

        assert!(
            result.cosine_similarity > 0.99,
            "Expected cosine ~1.0, got {}",
            result.cosine_similarity
        );
        assert!(
            result.euclidean_distance < 1e-6,
            "Expected distance ~0, got {}",
            result.euclidean_distance
        );
        assert!(
            result.is_same_speaker,
            "Identical embeddings should be same speaker"
        );
    }

    #[test]
    fn test_different_speaker_low_similarity() {
        let verifier = CrossRecordingVerifier::new();

        // Near-orthogonal embeddings → cosine similarity close to 0 (or negative)
        let emb_a = make_embedding(64, 1.0);
        let emb_b = orthogonal_embedding(64);

        let result = verifier.compare_speakers(&emb_a, &emb_b);

        assert!(
            result.cosine_similarity < 0.5,
            "Expected low cosine similarity for different speakers, got {}",
            result.cosine_similarity
        );
        assert!(
            !result.is_same_speaker,
            "Orthogonal embeddings should not be same speaker"
        );
    }

    #[test]
    fn test_cross_session_centroid() {
        let verifier = CrossRecordingVerifier::new();

        // Session A: three very similar embeddings around (1.0, 1.0, …)
        let session_a: Vec<Vec<f32>> = vec![
            make_embedding(32, 1.0),
            make_embedding(32, 1.05),
            make_embedding(32, 0.95),
        ];

        // Session B: three very similar embeddings around (1.0, 1.0, …) — same speaker
        let session_b: Vec<Vec<f32>> = vec![
            make_embedding(32, 1.0),
            make_embedding(32, 0.98),
            make_embedding(32, 1.02),
        ];

        let result = verifier
            .verify_across_sessions(&session_a, &session_b)
            .expect("should return result for non-empty sessions");

        assert!(
            result.centroid_cosine_similarity > 0.99,
            "Expected high centroid similarity, got {}",
            result.centroid_cosine_similarity
        );
        assert!(
            result.is_same_speaker,
            "Similar sessions should be flagged same speaker"
        );
        assert!(
            result.min_cosine_similarity <= result.mean_cosine_similarity,
            "min <= mean must hold"
        );
        assert!(
            result.mean_cosine_similarity <= result.max_cosine_similarity,
            "mean <= max must hold"
        );

        // Now compare against a very different session
        let session_c: Vec<Vec<f32>> = vec![orthogonal_embedding(32), orthogonal_embedding(32)];

        let result2 = verifier
            .verify_across_sessions(&session_a, &session_c)
            .expect("should return result");

        assert!(
            result2.centroid_cosine_similarity < 0.5,
            "Different sessions should have low centroid similarity, got {}",
            result2.centroid_cosine_similarity
        );
        assert!(
            !result2.is_same_speaker,
            "Different sessions should not be same speaker"
        );
    }
}
