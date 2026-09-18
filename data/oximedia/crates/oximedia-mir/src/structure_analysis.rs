//! Music structure analysis: verse, chorus, bridge detection and segment boundaries.
//!
//! Uses self-similarity matrices and novelty curves to locate structural
//! boundaries in audio, then labels sections heuristically.

#![allow(dead_code)]

use std::fmt;

/// A structural section label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionLabel {
    /// Song introduction.
    Intro,
    /// Verse section.
    Verse,
    /// Pre-chorus (build-up).
    PreChorus,
    /// Chorus / refrain.
    Chorus,
    /// Bridge section.
    Bridge,
    /// Instrumental break.
    Instrumental,
    /// Outro / coda.
    Outro,
    /// Unknown / undetermined section type.
    Unknown,
}

impl SectionLabel {
    /// Returns the human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Intro => "intro",
            Self::Verse => "verse",
            Self::PreChorus => "pre-chorus",
            Self::Chorus => "chorus",
            Self::Bridge => "bridge",
            Self::Instrumental => "instrumental",
            Self::Outro => "outro",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for SectionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A detected structural segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Section label.
    pub label: SectionLabel,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Repetition group index (segments with the same index are repeats).
    pub group: Option<usize>,
}

impl Segment {
    /// Duration of this segment in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Whether this segment overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && self.end > other.start
    }
}

/// Compute a self-similarity matrix from a sequence of feature vectors.
///
/// Each row in `features` is a feature vector (e.g., chroma or MFCC frame).
/// Returns a square matrix `S` where `S[i][j]` is the cosine similarity
/// between frames `i` and `j`.
///
/// # Arguments
///
/// * `features` - Slice of feature vectors, all of equal length.
#[must_use]
pub fn self_similarity_matrix(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = features.len();
    if n == 0 {
        return vec![];
    }

    // Precompute norms
    let norms: Vec<f64> = features
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();

    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            if norms[i] < 1e-12 || norms[j] < 1e-12 {
                mat[i][j] = 0.0;
                mat[j][i] = 0.0;
                continue;
            }
            let dot: f64 = features[i]
                .iter()
                .zip(features[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let sim = dot / (norms[i] * norms[j]);
            mat[i][j] = sim;
            mat[j][i] = sim;
        }
    }
    mat
}

/// Compute a novelty curve from a self-similarity matrix using a Gaussian checkerboard kernel.
///
/// High values in the novelty curve indicate structural boundaries.
///
/// # Arguments
///
/// * `ssm` - Square self-similarity matrix.
/// * `kernel_size` - Half-size of the checkerboard kernel in frames.
#[must_use]
#[allow(clippy::needless_range_loop, clippy::cast_possible_wrap)]
pub fn novelty_curve(ssm: &[Vec<f64>], kernel_size: usize) -> Vec<f64> {
    let n = ssm.len();
    if n == 0 || kernel_size == 0 {
        return vec![0.0; n];
    }

    // Build Gaussian checkerboard kernel
    let k = kernel_size as f64;
    let kernel: Vec<Vec<f64>> = (0..kernel_size)
        .map(|i| {
            (0..kernel_size)
                .map(|j| {
                    let di = (i as f64 + 0.5 - k / 2.0) / (k / 4.0);
                    let dj = (j as f64 + 0.5 - k / 2.0) / (k / 4.0);
                    let gauss = (-0.5 * (di * di + dj * dj)).exp();
                    // Checkerboard: upper-right and lower-left negative
                    let sign = if (i < kernel_size / 2) == (j < kernel_size / 2) {
                        1.0
                    } else {
                        -1.0
                    };
                    gauss * sign
                })
                .collect()
        })
        .collect();

    let mut novelty = vec![0.0_f64; n];
    for t in 0..n {
        let mut score = 0.0_f64;
        for ki in 0..kernel_size {
            for kj in 0..kernel_size {
                let row = t as i64 + ki as i64 - kernel_size as i64 / 2;
                let col = t as i64 + kj as i64 - kernel_size as i64 / 2;
                if row >= 0 && row < n as i64 && col >= 0 && col < n as i64 {
                    score += ssm[row as usize][col as usize] * kernel[ki][kj];
                }
            }
        }
        novelty[t] = score.max(0.0);
    }
    novelty
}

/// Detect boundary positions from a novelty curve using simple peak picking.
///
/// A frame is a boundary if its novelty score exceeds `threshold` and
/// it is a local maximum within a `min_gap` radius.
///
/// # Arguments
///
/// * `novelty` - Novelty curve values.
/// * `threshold` - Minimum novelty score to consider a boundary.
/// * `min_gap` - Minimum gap between boundaries in frames.
#[must_use]
pub fn pick_boundaries(novelty: &[f64], threshold: f64, min_gap: usize) -> Vec<usize> {
    let n = novelty.len();
    if n == 0 {
        return vec![];
    }

    let mut boundaries = Vec::new();
    let mut last_boundary: Option<usize> = None;

    for t in 0..n {
        if novelty[t] < threshold {
            continue;
        }
        // Check local maximum in window [t-min_gap/2, t+min_gap/2]
        let lo = t.saturating_sub(min_gap / 2);
        let hi = (t + min_gap / 2 + 1).min(n);
        let is_local_max = novelty[lo..hi].iter().all(|&v| v <= novelty[t]);
        if !is_local_max {
            continue;
        }
        // Enforce minimum gap
        if let Some(last) = last_boundary {
            if t - last < min_gap {
                continue;
            }
        }
        boundaries.push(t);
        last_boundary = Some(t);
    }
    boundaries
}

/// Convert frame-level boundary positions to time-stamped segments.
///
/// # Arguments
///
/// * `boundaries` - Frame indices of detected boundaries.
/// * `total_frames` - Total number of frames in the audio.
/// * `hop_duration` - Duration of a single hop frame in seconds.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn boundaries_to_segments(
    boundaries: &[usize],
    total_frames: usize,
    hop_duration: f64,
) -> Vec<Segment> {
    if total_frames == 0 {
        return vec![];
    }

    // Build boundary list including 0 and total_frames
    let mut bnd = vec![0usize];
    bnd.extend_from_slice(boundaries);
    if *bnd.last().unwrap_or(&0) != total_frames {
        bnd.push(total_frames);
    }

    bnd.windows(2)
        .map(|w| Segment {
            start: w[0] as f64 * hop_duration,
            end: w[1] as f64 * hop_duration,
            label: SectionLabel::Unknown,
            confidence: 0.5,
            group: None,
        })
        .collect()
}

/// Assign heuristic section labels to a sequence of segments.
///
/// Simple rule-based labelling based on position in the track:
/// - First segment: Intro
/// - Last segment: Outro
/// - Long segments near the middle: Chorus
/// - Short segments: Verse or Bridge
pub fn label_segments(segments: &mut [Segment]) {
    let n = segments.len();
    if n == 0 {
        return;
    }

    let total_dur: f64 = segments.last().map_or(0.0, |s| s.end);
    let avg_dur = if n > 0 { total_dur / n as f64 } else { 0.0 };

    for (i, seg) in segments.iter_mut().enumerate() {
        let mid = (seg.start + seg.end) / 2.0;
        let relative_pos = if total_dur > 0.0 {
            mid / total_dur
        } else {
            0.0
        };
        let is_long = seg.duration() > avg_dur * 1.3;

        seg.label = if i == 0 {
            SectionLabel::Intro
        } else if i == n - 1 {
            SectionLabel::Outro
        } else if is_long && (0.25..0.75).contains(&relative_pos) {
            SectionLabel::Chorus
        } else if relative_pos > 0.75 {
            SectionLabel::Bridge
        } else {
            SectionLabel::Verse
        };
    }
}

/// Summary of structural analysis results.
#[derive(Debug, Clone)]
pub struct StructureAnalysisResult {
    /// Detected segments with labels.
    pub segments: Vec<Segment>,
    /// Number of distinct section groups (repetitions detected).
    pub num_groups: usize,
    /// Estimated song form as a string (e.g. "ABABCB").
    pub form_string: String,
}

impl StructureAnalysisResult {
    /// Create a result from segments.
    #[must_use]
    pub fn from_segments(segments: Vec<Segment>) -> Self {
        let form_string = segments
            .iter()
            .map(|s| {
                s.label
                    .name()
                    .chars()
                    .next()
                    .unwrap_or('?')
                    .to_ascii_uppercase()
            })
            .collect();
        let num_groups = segments
            .iter()
            .filter_map(|s| s.group)
            .max()
            .map_or(0, |m| m + 1);
        Self {
            segments,
            num_groups,
            form_string,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_section_label_names() {
        assert_eq!(SectionLabel::Chorus.name(), "chorus");
        assert_eq!(SectionLabel::Verse.name(), "verse");
        assert_eq!(SectionLabel::Unknown.name(), "unknown");
    }

    #[test]
    fn test_section_label_display() {
        assert_eq!(SectionLabel::Intro.to_string(), "intro");
        assert_eq!(SectionLabel::Outro.to_string(), "outro");
    }

    #[test]
    fn test_segment_duration() {
        let seg = Segment {
            start: 1.0,
            end: 5.5,
            label: SectionLabel::Verse,
            confidence: 0.8,
            group: None,
        };
        assert!(approx_eq(seg.duration(), 4.5, 1e-10));
    }

    #[test]
    fn test_segment_overlaps() {
        let a = Segment {
            start: 0.0,
            end: 3.0,
            label: SectionLabel::Verse,
            confidence: 0.8,
            group: None,
        };
        let b = Segment {
            start: 2.0,
            end: 5.0,
            label: SectionLabel::Chorus,
            confidence: 0.8,
            group: None,
        };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_segment_no_overlap() {
        let a = Segment {
            start: 0.0,
            end: 2.0,
            label: SectionLabel::Verse,
            confidence: 0.8,
            group: None,
        };
        let b = Segment {
            start: 3.0,
            end: 5.0,
            label: SectionLabel::Chorus,
            confidence: 0.8,
            group: None,
        };
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_self_similarity_matrix_diagonal_is_one() {
        let features = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ssm = self_similarity_matrix(&features);
        for i in 0..3 {
            assert!(approx_eq(ssm[i][i], 1.0, 1e-10));
        }
    }

    #[test]
    fn test_self_similarity_matrix_orthogonal_off_diagonal() {
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ssm = self_similarity_matrix(&features);
        assert!(approx_eq(ssm[0][1], 0.0, 1e-10));
    }

    #[test]
    fn test_self_similarity_matrix_symmetric() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let ssm = self_similarity_matrix(&features);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(ssm[i][j], ssm[j][i], 1e-10));
            }
        }
    }

    #[test]
    fn test_self_similarity_matrix_empty() {
        let ssm = self_similarity_matrix(&[]);
        assert!(ssm.is_empty());
    }

    #[test]
    fn test_novelty_curve_length_matches_ssm() {
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let ssm = self_similarity_matrix(&features);
        let nc = novelty_curve(&ssm, 2);
        assert_eq!(nc.len(), features.len());
    }

    #[test]
    fn test_pick_boundaries_empty_novelty() {
        let bounds = pick_boundaries(&[], 0.5, 2);
        assert!(bounds.is_empty());
    }

    #[test]
    fn test_pick_boundaries_below_threshold() {
        let novelty = vec![0.1, 0.2, 0.1, 0.2];
        let bounds = pick_boundaries(&novelty, 0.5, 1);
        assert!(bounds.is_empty());
    }

    #[test]
    fn test_boundaries_to_segments_count() {
        let boundaries = vec![3, 7];
        let segs = boundaries_to_segments(&boundaries, 10, 0.5);
        // Boundaries: [0, 3, 7, 10] -> 3 segments
        assert_eq!(segs.len(), 3);
    }

    #[test]
    fn test_boundaries_to_segments_timing() {
        let segs = boundaries_to_segments(&[5], 10, 1.0);
        assert!(approx_eq(segs[0].start, 0.0, 1e-10));
        assert!(approx_eq(segs[0].end, 5.0, 1e-10));
        assert!(approx_eq(segs[1].start, 5.0, 1e-10));
        assert!(approx_eq(segs[1].end, 10.0, 1e-10));
    }

    #[test]
    fn test_label_segments_first_is_intro() {
        let mut segs = boundaries_to_segments(&[5, 10], 15, 1.0);
        label_segments(&mut segs);
        assert_eq!(segs[0].label, SectionLabel::Intro);
    }

    #[test]
    fn test_label_segments_last_is_outro() {
        let mut segs = boundaries_to_segments(&[5, 10], 15, 1.0);
        label_segments(&mut segs);
        let last = segs.last().expect("should succeed in test");
        assert_eq!(last.label, SectionLabel::Outro);
    }

    #[test]
    fn test_structure_analysis_result_form_string() {
        let mut segs = boundaries_to_segments(&[5, 10], 15, 1.0);
        label_segments(&mut segs);
        let result = StructureAnalysisResult::from_segments(segs);
        // Should start with 'I' for intro
        assert!(result.form_string.starts_with('I'));
    }
}
