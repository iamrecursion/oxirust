#![allow(dead_code)]
//! Chorus and refrain detection in musical audio.
//!
//! Identifies repeating musical sections such as choruses and refrains by
//! analyzing self-similarity matrices, chroma feature recurrence, and energy
//! profiles. Useful for music summarization, thumbnail generation, and
//! skip-to-chorus features.

use std::collections::HashMap;

/// A detected chorus or repeating section.
#[derive(Debug, Clone, PartialEq)]
pub struct ChorusSection {
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
    /// Which repetition group this belongs to (sections with same group repeat).
    pub group_id: u32,
    /// Occurrence index within the group (0 = first, 1 = second, ...).
    pub occurrence: u32,
    /// Confidence that this is indeed a chorus/refrain (0.0 - 1.0).
    pub confidence: f32,
    /// Mean energy level of this section (0.0 - 1.0).
    pub energy: f32,
}

impl ChorusSection {
    /// Duration of this section in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.end_secs - self.start_secs
    }

    /// Whether a given time falls within this section.
    #[must_use]
    pub fn contains_time(&self, t: f64) -> bool {
        t >= self.start_secs && t < self.end_secs
    }
}

/// Full result of chorus detection.
#[derive(Debug, Clone, PartialEq)]
pub struct ChorusDetectResult {
    /// All detected chorus/refrain sections.
    pub sections: Vec<ChorusSection>,
    /// The single "best" chorus section for summarization.
    pub best_chorus: Option<ChorusSection>,
    /// Total duration of the analyzed audio in seconds.
    pub total_duration_secs: f64,
}

impl ChorusDetectResult {
    /// Count distinct repetition groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        let groups: std::collections::HashSet<u32> =
            self.sections.iter().map(|s| s.group_id).collect();
        groups.len()
    }

    /// Get all sections belonging to a specific group.
    #[must_use]
    pub fn sections_in_group(&self, group_id: u32) -> Vec<&ChorusSection> {
        self.sections
            .iter()
            .filter(|s| s.group_id == group_id)
            .collect()
    }

    /// Fraction of the track covered by chorus sections.
    #[must_use]
    pub fn chorus_fraction(&self) -> f64 {
        if self.total_duration_secs <= 0.0 {
            return 0.0;
        }
        let chorus_dur: f64 = self.sections.iter().map(ChorusSection::duration_secs).sum();
        (chorus_dur / self.total_duration_secs).min(1.0)
    }
}

/// Configuration for chorus detection.
#[derive(Debug, Clone)]
pub struct ChorusDetectConfig {
    /// Minimum section length to consider as a chorus (seconds).
    pub min_section_secs: f64,
    /// Maximum section length (seconds).
    pub max_section_secs: f64,
    /// Minimum similarity to count as repetition (0.0 - 1.0).
    pub similarity_threshold: f32,
    /// Minimum number of occurrences to qualify as "chorus".
    pub min_occurrences: u32,
    /// Hop size for feature extraction (seconds).
    pub hop_secs: f64,
}

impl Default for ChorusDetectConfig {
    fn default() -> Self {
        Self {
            min_section_secs: 5.0,
            max_section_secs: 60.0,
            similarity_threshold: 0.75,
            min_occurrences: 2,
            hop_secs: 0.5,
        }
    }
}

/// Computes cosine similarity between two feature vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Build a self-similarity matrix from feature frames.
///
/// Returns a flat vec of size `n * n` representing the similarity between
/// every pair of frames.
#[must_use]
pub fn self_similarity_matrix(frames: &[Vec<f32>]) -> Vec<f32> {
    let n = frames.len();
    let mut matrix = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            let sim = cosine_similarity(&frames[i], &frames[j]);
            matrix[i * n + j] = sim;
            matrix[j * n + i] = sim;
        }
    }
    matrix
}

/// Find diagonal stripes in a self-similarity matrix that indicate repeating
/// sections. Returns (offset, length, `avg_similarity`) tuples.
#[must_use]
pub fn find_diagonals(
    matrix: &[f32],
    n: usize,
    threshold: f32,
    min_len: usize,
) -> Vec<(usize, usize, f32)> {
    if n == 0 || matrix.len() != n * n {
        return Vec::new();
    }
    let mut diagonals = Vec::new();

    // Check diagonals offset from the main diagonal
    for offset in 1..n {
        let max_len = n - offset;
        let mut run_start = None;
        let mut run_sum = 0.0_f32;

        for k in 0..max_len {
            let sim = matrix[k * n + (k + offset)];
            if sim >= threshold {
                if run_start.is_none() {
                    run_start = Some(k);
                    run_sum = 0.0;
                }
                run_sum += sim;
            } else if let Some(start) = run_start {
                let length = k - start;
                if length >= min_len {
                    #[allow(clippy::cast_precision_loss)]
                    let avg = run_sum / length as f32;
                    diagonals.push((offset, length, avg));
                }
                run_start = None;
            }
        }
        // Handle run ending at boundary
        if let Some(start) = run_start {
            let length = max_len - start;
            if length >= min_len {
                #[allow(clippy::cast_precision_loss)]
                let avg = run_sum / length as f32;
                diagonals.push((offset, length, avg));
            }
        }
    }

    diagonals
}

/// Compute RMS energy for a window of audio samples.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Simple heuristic: find the section with highest combined energy + repetition.
#[must_use]
pub fn find_best_chorus(sections: &[ChorusSection]) -> Option<ChorusSection> {
    if sections.is_empty() {
        return None;
    }

    // Group by group_id, prefer groups with more occurrences
    let mut group_counts: HashMap<u32, u32> = HashMap::new();
    for s in sections {
        *group_counts.entry(s.group_id).or_insert(0) += 1;
    }

    sections
        .iter()
        .filter(|s| s.occurrence == 0) // First occurrence of each group
        .max_by(|a, b| {
            let a_count = group_counts.get(&a.group_id).copied().unwrap_or(0);
            let b_count = group_counts.get(&b.group_id).copied().unwrap_or(0);
            let a_score = a.confidence * a.energy + (a_count as f32 * 0.1);
            let b_score = b.confidence * b.energy + (b_count as f32 * 0.1);
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert!(cosine_similarity(&[], &[]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn test_self_similarity_matrix_identity() {
        let frames = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let mat = self_similarity_matrix(&frames);
        assert_eq!(mat.len(), 9);
        // Diagonal should be 1.0
        assert!((mat[0] - 1.0).abs() < 0.01);
        assert!((mat[4] - 1.0).abs() < 0.01);
        assert!((mat[8] - 1.0).abs() < 0.01);
        // Frame 0 and frame 2 are identical
        assert!((mat[2] - 1.0).abs() < 0.01);
        // Frame 0 and frame 1 are orthogonal
        assert!(mat[1].abs() < 0.01);
    }

    #[test]
    fn test_find_diagonals_simple() {
        // 4x4 matrix with a repeated section at offset 2
        let n = 4;
        let mut mat = vec![0.0_f32; n * n];
        // Main diagonal = 1.0
        for i in 0..n {
            mat[i * n + i] = 1.0;
        }
        // Offset 2 diagonal has high similarity
        mat[0 * n + 2] = 0.9;
        mat[2 * n + 0] = 0.9;
        mat[1 * n + 3] = 0.85;
        mat[3 * n + 1] = 0.85;

        let diags = find_diagonals(&mat, n, 0.8, 2);
        assert!(
            !diags.is_empty(),
            "Should find at least one diagonal stripe"
        );
    }

    #[test]
    fn test_find_diagonals_empty() {
        let diags = find_diagonals(&[], 0, 0.5, 1);
        assert!(diags.is_empty());
    }

    #[test]
    fn test_rms_energy() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = rms_energy(&samples);
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_energy_silence() {
        let samples = vec![0.0; 100];
        assert!(rms_energy(&samples).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rms_energy_empty() {
        assert!(rms_energy(&[]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chorus_section_duration() {
        let sec = ChorusSection {
            start_secs: 30.0,
            end_secs: 60.0,
            group_id: 0,
            occurrence: 0,
            confidence: 0.9,
            energy: 0.8,
        };
        assert!((sec.duration_secs() - 30.0).abs() < f64::EPSILON);
        assert!(sec.contains_time(45.0));
        assert!(!sec.contains_time(60.0));
    }

    #[test]
    fn test_chorus_detect_result_groups() {
        let result = ChorusDetectResult {
            sections: vec![
                ChorusSection {
                    start_secs: 0.0,
                    end_secs: 20.0,
                    group_id: 0,
                    occurrence: 0,
                    confidence: 0.9,
                    energy: 0.8,
                },
                ChorusSection {
                    start_secs: 60.0,
                    end_secs: 80.0,
                    group_id: 0,
                    occurrence: 1,
                    confidence: 0.85,
                    energy: 0.75,
                },
                ChorusSection {
                    start_secs: 30.0,
                    end_secs: 50.0,
                    group_id: 1,
                    occurrence: 0,
                    confidence: 0.7,
                    energy: 0.6,
                },
            ],
            best_chorus: None,
            total_duration_secs: 120.0,
        };
        assert_eq!(result.group_count(), 2);
        assert_eq!(result.sections_in_group(0).len(), 2);
        assert!((result.chorus_fraction() - (60.0 / 120.0)).abs() < 0.01);
    }

    #[test]
    fn test_find_best_chorus() {
        let sections = vec![
            ChorusSection {
                start_secs: 0.0,
                end_secs: 20.0,
                group_id: 0,
                occurrence: 0,
                confidence: 0.9,
                energy: 0.8,
            },
            ChorusSection {
                start_secs: 60.0,
                end_secs: 80.0,
                group_id: 0,
                occurrence: 1,
                confidence: 0.85,
                energy: 0.75,
            },
            ChorusSection {
                start_secs: 30.0,
                end_secs: 50.0,
                group_id: 1,
                occurrence: 0,
                confidence: 0.5,
                energy: 0.4,
            },
        ];
        let best = find_best_chorus(&sections);
        assert!(best.is_some());
        // Group 0 has 2 occurrences and higher scores, so group 0 occurrence 0 should win
        assert_eq!(best.expect("should succeed in test").group_id, 0);
    }

    #[test]
    fn test_find_best_chorus_empty() {
        assert!(find_best_chorus(&[]).is_none());
    }

    #[test]
    fn test_chorus_detect_config_default() {
        let cfg = ChorusDetectConfig::default();
        assert!((cfg.min_section_secs - 5.0).abs() < f64::EPSILON);
        assert_eq!(cfg.min_occurrences, 2);
    }
}
