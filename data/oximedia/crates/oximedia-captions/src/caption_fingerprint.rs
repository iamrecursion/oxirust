#![allow(dead_code)]

//! Caption fingerprinting for deduplication and matching.
//!
//! Generates compact fingerprints from caption tracks that can be used
//! to identify duplicate or near-duplicate caption files, match translated
//! versions, and detect unauthorized copies.

use std::collections::HashMap;
use std::fmt;

/// Fingerprint algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FingerprintAlgorithm {
    /// Hash based on timing structure only.
    TimingOnly,
    /// Hash based on text content only.
    TextOnly,
    /// Combined timing and text hash.
    Combined,
    /// Structural fingerprint based on line counts, durations, gaps.
    Structural,
}

impl fmt::Display for FingerprintAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimingOnly => write!(f, "timing_only"),
            Self::TextOnly => write!(f, "text_only"),
            Self::Combined => write!(f, "combined"),
            Self::Structural => write!(f, "structural"),
        }
    }
}

/// A caption entry for fingerprinting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FingerprintCaption {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// The text content.
    pub text: String,
}

impl FingerprintCaption {
    /// Create a new caption for fingerprinting.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: String) -> Self {
        Self {
            start_ms,
            end_ms,
            text,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of lines in the text.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.text.lines().count().max(1)
    }

    /// Character count (excluding whitespace).
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().filter(|c| !c.is_whitespace()).count()
    }
}

/// A computed fingerprint for a caption track.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CaptionFingerprint {
    /// The algorithm used to generate this fingerprint.
    pub algorithm: FingerprintAlgorithm,
    /// The fingerprint hash value (hex string).
    pub hash: String,
    /// Number of captions in the track.
    pub caption_count: usize,
    /// Total duration covered by captions in milliseconds.
    pub total_duration_ms: u64,
}

impl CaptionFingerprint {
    /// Check if two fingerprints match.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.algorithm == other.algorithm && self.hash == other.hash
    }
}

impl fmt::Display for CaptionFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} ({}cues, {}ms)",
            self.algorithm, self.hash, self.caption_count, self.total_duration_ms
        )
    }
}

/// Structural features extracted from a caption track.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuralFeatures {
    /// Number of captions.
    pub count: usize,
    /// Average duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Standard deviation of duration.
    pub std_duration_ms: f64,
    /// Average gap between captions in milliseconds.
    pub avg_gap_ms: f64,
    /// Average characters per caption.
    pub avg_chars: f64,
    /// Average lines per caption.
    pub avg_lines: f64,
    /// Total duration span from first start to last end.
    pub total_span_ms: u64,
}

/// Simple deterministic hash function (FNV-1a variant).
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Generate a timing-only fingerprint.
fn timing_fingerprint(captions: &[FingerprintCaption]) -> String {
    let mut data = Vec::new();
    for cap in captions {
        data.extend_from_slice(&cap.start_ms.to_le_bytes());
        data.extend_from_slice(&cap.end_ms.to_le_bytes());
    }
    let h = fnv1a_hash(&data);
    format!("{h:016x}")
}

/// Generate a text-only fingerprint.
fn text_fingerprint(captions: &[FingerprintCaption]) -> String {
    let mut data = Vec::new();
    for cap in captions {
        let normalized = cap
            .text
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        data.extend_from_slice(normalized.as_bytes());
        data.push(0); // separator
    }
    let h = fnv1a_hash(&data);
    format!("{h:016x}")
}

/// Generate a combined timing + text fingerprint.
fn combined_fingerprint(captions: &[FingerprintCaption]) -> String {
    let mut data = Vec::new();
    for cap in captions {
        data.extend_from_slice(&cap.start_ms.to_le_bytes());
        data.extend_from_slice(&cap.end_ms.to_le_bytes());
        let normalized = cap
            .text
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        data.extend_from_slice(normalized.as_bytes());
        data.push(0);
    }
    let h = fnv1a_hash(&data);
    format!("{h:016x}")
}

/// Extract structural features from a caption track.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn extract_features(captions: &[FingerprintCaption]) -> StructuralFeatures {
    if captions.is_empty() {
        return StructuralFeatures {
            count: 0,
            avg_duration_ms: 0.0,
            std_duration_ms: 0.0,
            avg_gap_ms: 0.0,
            avg_chars: 0.0,
            avg_lines: 0.0,
            total_span_ms: 0,
        };
    }

    let count = captions.len();
    let durations: Vec<f64> = captions.iter().map(|c| c.duration_ms() as f64).collect();
    let avg_duration = durations.iter().sum::<f64>() / count as f64;
    let variance = durations
        .iter()
        .map(|d| (d - avg_duration).powi(2))
        .sum::<f64>()
        / count as f64;
    let std_duration = variance.sqrt();

    let gaps: Vec<f64> = captions
        .windows(2)
        .map(|w| w[1].start_ms.saturating_sub(w[0].end_ms) as f64)
        .collect();
    let avg_gap = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f64>() / gaps.len() as f64
    };

    let avg_chars = captions.iter().map(|c| c.char_count() as f64).sum::<f64>() / count as f64;
    let avg_lines = captions.iter().map(|c| c.line_count() as f64).sum::<f64>() / count as f64;

    let first_start = captions.first().map_or(0, |c| c.start_ms);
    let last_end = captions.last().map_or(0, |c| c.end_ms);
    let total_span = last_end.saturating_sub(first_start);

    StructuralFeatures {
        count,
        avg_duration_ms: avg_duration,
        std_duration_ms: std_duration,
        avg_gap_ms: avg_gap,
        avg_chars,
        avg_lines,
        total_span_ms: total_span,
    }
}

/// Generate a structural fingerprint from features.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn structural_fingerprint(captions: &[FingerprintCaption]) -> String {
    let features = extract_features(captions);
    let mut data = Vec::new();
    data.extend_from_slice(&(features.count as u64).to_le_bytes());
    // Quantize continuous values to reduce sensitivity
    let q_avg_dur = (features.avg_duration_ms / 100.0) as u64;
    let q_avg_gap = (features.avg_gap_ms / 100.0) as u64;
    let q_avg_chars = features.avg_chars as u64;
    data.extend_from_slice(&q_avg_dur.to_le_bytes());
    data.extend_from_slice(&q_avg_gap.to_le_bytes());
    data.extend_from_slice(&q_avg_chars.to_le_bytes());
    data.extend_from_slice(&features.total_span_ms.to_le_bytes());
    let h = fnv1a_hash(&data);
    format!("{h:016x}")
}

/// Compute a fingerprint for a caption track.
#[must_use]
pub fn fingerprint(
    captions: &[FingerprintCaption],
    algorithm: FingerprintAlgorithm,
) -> CaptionFingerprint {
    let hash = match algorithm {
        FingerprintAlgorithm::TimingOnly => timing_fingerprint(captions),
        FingerprintAlgorithm::TextOnly => text_fingerprint(captions),
        FingerprintAlgorithm::Combined => combined_fingerprint(captions),
        FingerprintAlgorithm::Structural => structural_fingerprint(captions),
    };

    let total_duration_ms: u64 = captions.iter().map(FingerprintCaption::duration_ms).sum();

    CaptionFingerprint {
        algorithm,
        hash,
        caption_count: captions.len(),
        total_duration_ms,
    }
}

/// Compare two caption tracks and return similarity score (0.0 to 1.0).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn similarity(left: &[FingerprintCaption], right: &[FingerprintCaption]) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let lf = extract_features(left);
    let rf = extract_features(right);

    // Compare structural features with normalized distances
    let count_sim =
        1.0 - (lf.count as f64 - rf.count as f64).abs() / (lf.count.max(rf.count) as f64).max(1.0);
    let dur_sim = if lf.avg_duration_ms + rf.avg_duration_ms > 0.0 {
        1.0 - (lf.avg_duration_ms - rf.avg_duration_ms).abs()
            / lf.avg_duration_ms.max(rf.avg_duration_ms).max(1.0)
    } else {
        1.0
    };
    let span_sim = if lf.total_span_ms + rf.total_span_ms > 0 {
        let max_span = lf.total_span_ms.max(rf.total_span_ms) as f64;
        let diff = if lf.total_span_ms > rf.total_span_ms {
            lf.total_span_ms - rf.total_span_ms
        } else {
            rf.total_span_ms - lf.total_span_ms
        };
        1.0 - diff as f64 / max_span.max(1.0)
    } else {
        1.0
    };

    (count_sim * 0.4 + dur_sim * 0.3 + span_sim * 0.3).clamp(0.0, 1.0)
}

/// Batch fingerprint multiple tracks and find duplicates.
#[must_use]
pub fn find_duplicates(
    tracks: &[Vec<FingerprintCaption>],
    algorithm: FingerprintAlgorithm,
) -> Vec<(usize, usize)> {
    let fingerprints: Vec<CaptionFingerprint> =
        tracks.iter().map(|t| fingerprint(t, algorithm)).collect();

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, fp) in fingerprints.iter().enumerate() {
        groups.entry(fp.hash.clone()).or_default().push(i);
    }

    let mut pairs = Vec::new();
    for indices in groups.values() {
        if indices.len() > 1 {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    pairs.push((indices[i], indices[j]));
                }
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(start: u64, end: u64, text: &str) -> FingerprintCaption {
        FingerprintCaption::new(start, end, text.to_string())
    }

    #[test]
    fn test_timing_fingerprint_deterministic() {
        let captions = vec![cap(0, 1000, "Hello"), cap(2000, 3000, "World")];
        let fp1 = fingerprint(&captions, FingerprintAlgorithm::TimingOnly);
        let fp2 = fingerprint(&captions, FingerprintAlgorithm::TimingOnly);
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_text_fingerprint_deterministic() {
        let captions = vec![cap(0, 1000, "Hello"), cap(2000, 3000, "World")];
        let fp1 = fingerprint(&captions, FingerprintAlgorithm::TextOnly);
        let fp2 = fingerprint(&captions, FingerprintAlgorithm::TextOnly);
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_combined_fingerprint_deterministic() {
        let captions = vec![cap(0, 1000, "Hello"), cap(2000, 3000, "World")];
        let fp1 = fingerprint(&captions, FingerprintAlgorithm::Combined);
        let fp2 = fingerprint(&captions, FingerprintAlgorithm::Combined);
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_different_text_different_hash() {
        let a = vec![cap(0, 1000, "Hello")];
        let b = vec![cap(0, 1000, "Goodbye")];
        let fp_a = fingerprint(&a, FingerprintAlgorithm::TextOnly);
        let fp_b = fingerprint(&b, FingerprintAlgorithm::TextOnly);
        assert_ne!(fp_a.hash, fp_b.hash);
    }

    #[test]
    fn test_timing_ignores_text() {
        let a = vec![cap(0, 1000, "Hello")];
        let b = vec![cap(0, 1000, "Goodbye")];
        let fp_a = fingerprint(&a, FingerprintAlgorithm::TimingOnly);
        let fp_b = fingerprint(&b, FingerprintAlgorithm::TimingOnly);
        assert_eq!(fp_a.hash, fp_b.hash);
    }

    #[test]
    fn test_empty_track() {
        let fp = fingerprint(&[], FingerprintAlgorithm::Combined);
        assert_eq!(fp.caption_count, 0);
        assert_eq!(fp.total_duration_ms, 0);
    }

    #[test]
    fn test_structural_fingerprint() {
        let captions = vec![
            cap(0, 1000, "Hello"),
            cap(2000, 3000, "World"),
            cap(4000, 5000, "Test"),
        ];
        let fp = fingerprint(&captions, FingerprintAlgorithm::Structural);
        assert_eq!(fp.caption_count, 3);
        assert!(!fp.hash.is_empty());
    }

    #[test]
    fn test_extract_features_empty() {
        let features = extract_features(&[]);
        assert_eq!(features.count, 0);
        assert!((features.avg_duration_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_features_basic() {
        let captions = vec![cap(0, 1000, "A"), cap(2000, 3000, "B")];
        let features = extract_features(&captions);
        assert_eq!(features.count, 2);
        assert!((features.avg_duration_ms - 1000.0).abs() < f64::EPSILON);
        assert_eq!(features.total_span_ms, 3000);
    }

    #[test]
    fn test_similarity_identical() {
        let captions = vec![cap(0, 1000, "Hello"), cap(2000, 3000, "World")];
        let sim = similarity(&captions, &captions);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_empty() {
        assert!((similarity(&[], &[]) - 1.0).abs() < f64::EPSILON);
        assert!(similarity(&[], &[cap(0, 1000, "A")]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_duplicates() {
        let track_a = vec![cap(0, 1000, "Hello"), cap(2000, 3000, "World")];
        let track_b = track_a.clone();
        let track_c = vec![cap(0, 1000, "Different")];
        let tracks = vec![track_a, track_b, track_c];
        let dups = find_duplicates(&tracks, FingerprintAlgorithm::Combined);
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0], (0, 1));
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = CaptionFingerprint {
            algorithm: FingerprintAlgorithm::Combined,
            hash: "abc123".to_string(),
            caption_count: 5,
            total_duration_ms: 10000,
        };
        let display = fp.to_string();
        assert!(display.contains("combined"));
        assert!(display.contains("abc123"));
    }

    #[test]
    fn test_fingerprint_matches() {
        let fp1 = CaptionFingerprint {
            algorithm: FingerprintAlgorithm::TextOnly,
            hash: "abc".to_string(),
            caption_count: 1,
            total_duration_ms: 1000,
        };
        let fp2 = fp1.clone();
        assert!(fp1.matches(&fp2));
    }

    #[test]
    fn test_caption_char_count() {
        let c = cap(0, 1000, "Hello World");
        assert_eq!(c.char_count(), 10); // no space counted
    }
}
