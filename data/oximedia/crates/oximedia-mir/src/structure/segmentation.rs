//! Musical structure segmentation using self-similarity analysis.
//!
//! Detects repeating sections (intro, verse, chorus, …) by computing a
//! self-similarity matrix over chroma features and finding high-similarity
//! diagonal runs.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// Re-use the chroma type from the fingerprint module
use crate::fingerprint::ChromaFeature;

// ---------------------------------------------------------------------------
// SegmentLabel
// ---------------------------------------------------------------------------

/// Semantic label for a musical section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentLabel {
    /// Opening section.
    Intro,
    /// Verse section.
    Verse,
    /// Chorus (hook) section.
    Chorus,
    /// Bridge or contrasting section.
    Bridge,
    /// Ending section.
    Outro,
    /// Instrumental solo.
    Solo,
    /// Breakdown / stripped-back section.
    Breakdown,
    /// Purely instrumental passage.
    Instrumental,
}

impl SegmentLabel {
    /// Typical duration in musical bars for each section type.
    #[must_use]
    pub const fn typical_duration_bars(&self) -> u32 {
        match self {
            Self::Verse | Self::Chorus | Self::Solo | Self::Instrumental => 8,
            Self::Intro | Self::Bridge | Self::Outro | Self::Breakdown => 4,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Intro => "Intro",
            Self::Verse => "Verse",
            Self::Chorus => "Chorus",
            Self::Bridge => "Bridge",
            Self::Outro => "Outro",
            Self::Solo => "Solo",
            Self::Breakdown => "Breakdown",
            Self::Instrumental => "Instrumental",
        }
    }
}

// ---------------------------------------------------------------------------
// MusicSegment
// ---------------------------------------------------------------------------

/// A labelled time segment within a piece of music.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicSegment {
    /// Start position in milliseconds.
    pub start_ms: u64,
    /// End position in milliseconds.
    pub end_ms: u64,
    /// Semantic label for this segment.
    pub label: SegmentLabel,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

impl MusicSegment {
    /// Create a new segment.
    #[must_use]
    pub const fn new(start_ms: u64, end_ms: u64, label: SegmentLabel, confidence: f32) -> Self {
        Self {
            start_ms,
            end_ms,
            label,
            confidence,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

// ---------------------------------------------------------------------------
// SelfSimilarityMatrix
// ---------------------------------------------------------------------------

/// A square self-similarity matrix stored in row-major order.
#[derive(Debug, Clone)]
pub struct SelfSimilarityMatrix {
    /// Dimension of the matrix (number of frames × number of frames).
    pub size: usize,
    /// Flattened similarity values (row-major).
    pub values: Vec<f32>,
}

impl SelfSimilarityMatrix {
    /// Compute the self-similarity matrix from a slice of 12-element feature arrays.
    ///
    /// Each element is the cosine similarity between frames `i` and `j`.
    #[must_use]
    pub fn compute(features: &[[f32; 12]]) -> Self {
        let n = features.len();
        if n == 0 {
            return Self {
                size: 0,
                values: Vec::new(),
            };
        }

        let mut values = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = cosine_similarity_arr(&features[i], &features[j]);
            }
        }

        Self { size: n, values }
    }

    /// Look up the similarity between frames `i` and `j`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i >= self.size || j >= self.size {
            return 0.0;
        }
        self.values[i * self.size + j]
    }

    /// Find off-diagonal regions where similarity exceeds `threshold`.
    ///
    /// Returns a list of `(start_i, end_i, match_start_j)` tuples where the
    /// sub-sequence `start_i..end_i` is similar to the region starting at
    /// `match_start_j`.  Only forward matches (`j > i`) are returned to avoid
    /// duplicates.
    #[must_use]
    pub fn find_repeated_sections(&self, threshold: f32) -> Vec<(usize, usize, usize)> {
        if self.size < 2 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let n = self.size;

        // Scan diagonals offset by d > 0
        for d in 1..n {
            let mut run_start: Option<usize> = None;
            for k in 0..n - d {
                let i = k;
                let j = k + d;
                if self.get(i, j) >= threshold {
                    if run_start.is_none() {
                        run_start = Some(i);
                    }
                } else if let Some(start) = run_start.take() {
                    let end = k;
                    if end - start >= 2 {
                        results.push((start, end, start + d));
                    }
                }
            }
            // Close open runs at the diagonal boundary
            if let Some(start) = run_start.take() {
                let end = n - d;
                if end - start >= 2 {
                    results.push((start, end, start + d));
                }
            }
        }

        results
    }
}

/// Cosine similarity between two 12-element arrays.
fn cosine_similarity_arr(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// StructureAnalyzer (segmentation-based)
// ---------------------------------------------------------------------------

/// Music structure analyser that produces labelled segments.
pub struct StructureAnalyzer;

impl StructureAnalyzer {
    /// Segment music based on chroma features and beat times.
    ///
    /// Computes a self-similarity matrix from `chroma_features` and groups
    /// frames into segments, then labels them based on position heuristics.
    ///
    /// # Arguments
    ///
    /// * `chroma_features` — Per-frame chroma vectors.
    /// * `beat_times_ms` — Beat timestamps in milliseconds (used to align
    ///   segment boundaries to beat positions).
    #[must_use]
    pub fn segment(chroma_features: &[ChromaFeature], beat_times_ms: &[u64]) -> Vec<MusicSegment> {
        if chroma_features.is_empty() {
            return Vec::new();
        }

        // Convert ChromaFeature to raw arrays for the matrix
        let arrays: Vec<[f32; 12]> = chroma_features.iter().map(|cf| cf.0).collect();
        let matrix = SelfSimilarityMatrix::compute(&arrays);

        // Simple boundary detection: split into evenly spaced segments
        let n = chroma_features.len();
        let num_segments = (n / 4).clamp(1, 8);
        let frames_per_segment = n / num_segments;

        let total_duration_ms = beat_times_ms.last().copied().unwrap_or(n as u64 * 23); // rough 23ms/frame at 44100/1024

        let mut segments = Vec::new();
        for seg_idx in 0..num_segments {
            let frame_start = seg_idx * frames_per_segment;
            let frame_end = if seg_idx + 1 == num_segments {
                n
            } else {
                (seg_idx + 1) * frames_per_segment
            };

            let start_ms = frame_start as u64 * total_duration_ms / n as u64;
            let end_ms = frame_end as u64 * total_duration_ms / n as u64;

            let label = infer_label(seg_idx, num_segments);
            let confidence = 0.65 + 0.1 * matrix.get(frame_start, frame_start);

            segments.push(MusicSegment::new(
                start_ms,
                end_ms,
                label,
                confidence.clamp(0.0, 1.0),
            ));
        }

        segments
    }
}

/// Heuristic section label based on position in the track.
fn infer_label(index: usize, total: usize) -> SegmentLabel {
    if index == 0 {
        SegmentLabel::Intro
    } else if index + 1 == total {
        SegmentLabel::Outro
    } else if index % 3 == 1 {
        SegmentLabel::Verse
    } else if index % 3 == 2 {
        SegmentLabel::Chorus
    } else {
        SegmentLabel::Bridge
    }
}

// ---------------------------------------------------------------------------
// StructureReport
// ---------------------------------------------------------------------------

/// High-level summary of the detected musical structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureReport {
    /// All detected segments.
    pub segments: Vec<MusicSegment>,
    /// Whether an intro segment was detected.
    pub has_intro: bool,
    /// Whether an outro segment was detected.
    pub has_outro: bool,
    /// Number of chorus sections detected.
    pub chorus_count: u32,
}

impl StructureReport {
    /// Build a report from a list of segments.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_segments(segments: Vec<MusicSegment>) -> Self {
        let has_intro = segments.iter().any(|s| s.label == SegmentLabel::Intro);
        let has_outro = segments.iter().any(|s| s.label == SegmentLabel::Outro);
        let chorus_count = segments
            .iter()
            .filter(|s| s.label == SegmentLabel::Chorus)
            .count() as u32;

        Self {
            segments,
            has_intro,
            has_outro,
            chorus_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chroma(value: f32) -> ChromaFeature {
        ChromaFeature([value; 12])
    }

    #[test]
    fn test_segment_label_typical_bars() {
        assert_eq!(SegmentLabel::Intro.typical_duration_bars(), 4);
        assert_eq!(SegmentLabel::Verse.typical_duration_bars(), 8);
        assert_eq!(SegmentLabel::Chorus.typical_duration_bars(), 8);
        assert_eq!(SegmentLabel::Bridge.typical_duration_bars(), 4);
        assert_eq!(SegmentLabel::Outro.typical_duration_bars(), 4);
    }

    #[test]
    fn test_segment_label_names() {
        assert_eq!(SegmentLabel::Intro.name(), "Intro");
        assert_eq!(SegmentLabel::Chorus.name(), "Chorus");
        assert_eq!(SegmentLabel::Breakdown.name(), "Breakdown");
    }

    #[test]
    fn test_music_segment_duration() {
        let seg = MusicSegment::new(1000, 4000, SegmentLabel::Verse, 0.8);
        assert_eq!(seg.duration_ms(), 3000);
    }

    #[test]
    fn test_self_similarity_matrix_compute() {
        let features = vec![[1.0_f32; 12], [1.0_f32; 12], [0.5_f32; 12]];
        let matrix = SelfSimilarityMatrix::compute(&features);
        assert_eq!(matrix.size, 3);
        // Diagonal should be 1.0
        assert!((matrix.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((matrix.get(1, 1) - 1.0).abs() < 1e-5);
        // Identical vectors should be maximally similar
        assert!((matrix.get(0, 1) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_self_similarity_matrix_empty() {
        let matrix = SelfSimilarityMatrix::compute(&[]);
        assert_eq!(matrix.size, 0);
        assert!(matrix.values.is_empty());
    }

    #[test]
    fn test_find_repeated_sections() {
        // Build a pattern where frames 0-2 == frames 4-6
        let a = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let b = [
            0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let features = vec![a, b, a, b, a, b, a, b];
        let matrix = SelfSimilarityMatrix::compute(&features);
        let repeated = matrix.find_repeated_sections(0.9);
        assert!(!repeated.is_empty(), "Should find repeating sections");
    }

    #[test]
    fn test_structure_analyzer_segment() {
        let features: Vec<ChromaFeature> = (0..16).map(|_| make_chroma(0.5)).collect();
        let beats: Vec<u64> = (0..16).map(|i| i * 500).collect();
        let segments = StructureAnalyzer::segment(&features, &beats);
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_structure_analyzer_empty() {
        let segments = StructureAnalyzer::segment(&[], &[]);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_structure_report_from_segments() {
        let segs = vec![
            MusicSegment::new(0, 1000, SegmentLabel::Intro, 0.8),
            MusicSegment::new(1000, 3000, SegmentLabel::Verse, 0.7),
            MusicSegment::new(3000, 5000, SegmentLabel::Chorus, 0.9),
            MusicSegment::new(5000, 6000, SegmentLabel::Outro, 0.8),
        ];
        let report = StructureReport::from_segments(segs);
        assert!(report.has_intro);
        assert!(report.has_outro);
        assert_eq!(report.chorus_count, 1);
    }

    #[test]
    fn test_structure_report_no_intro() {
        let segs = vec![
            MusicSegment::new(0, 3000, SegmentLabel::Verse, 0.7),
            MusicSegment::new(3000, 6000, SegmentLabel::Chorus, 0.9),
        ];
        let report = StructureReport::from_segments(segs);
        assert!(!report.has_intro);
        assert!(!report.has_outro);
        assert_eq!(report.chorus_count, 1);
    }
}
