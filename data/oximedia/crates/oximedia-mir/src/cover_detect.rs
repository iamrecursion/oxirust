#![allow(dead_code)]
//! Cover version detection for music information retrieval.
//!
//! Detects whether two recordings are versions of the same underlying
//! composition by comparing tonal sequences (chroma features) that are
//! invariant to tempo, key, and timbral differences.

use std::collections::HashMap;

/// Chroma-based fingerprint of a musical passage.
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaFingerprint {
    /// Sequence of 12-dimensional chroma vectors (one per analysis frame).
    pub frames: Vec<[f64; 12]>,
    /// Duration in seconds represented by this fingerprint.
    pub duration_s: f64,
}

impl ChromaFingerprint {
    /// Create a fingerprint from a sequence of chroma vectors.
    #[must_use]
    pub fn new(frames: Vec<[f64; 12]>, duration_s: f64) -> Self {
        Self { frames, duration_s }
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the fingerprint is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Transpose all chroma vectors by `semitones` (key normalisation).
    #[must_use]
    pub fn transposed(&self, semitones: i32) -> Self {
        let shift = ((semitones % 12) + 12) as usize % 12;
        let frames = self
            .frames
            .iter()
            .map(|f| {
                let mut out = [0.0; 12];
                for i in 0..12 {
                    out[(i + shift) % 12] = f[i];
                }
                out
            })
            .collect();
        Self {
            frames,
            duration_s: self.duration_s,
        }
    }

    /// Compute average chroma vector across all frames.
    #[must_use]
    pub fn average_chroma(&self) -> [f64; 12] {
        let mut avg = [0.0_f64; 12];
        if self.frames.is_empty() {
            return avg;
        }
        for f in &self.frames {
            for (i, v) in f.iter().enumerate() {
                avg[i] += v;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.frames.len() as f64;
        for v in &mut avg {
            *v /= n;
        }
        avg
    }
}

/// A known cover version entry.
#[derive(Debug, Clone)]
pub struct CoverVersion {
    /// Unique identifier.
    pub id: String,
    /// Title of the recording.
    pub title: String,
    /// Artist name.
    pub artist: String,
    /// Chroma fingerprint.
    pub fingerprint: ChromaFingerprint,
    /// Estimated key (0..11, where 0 = C).
    pub estimated_key: u8,
}

impl CoverVersion {
    /// Create a new cover version entry.
    #[must_use]
    pub fn new(
        id: &str,
        title: &str,
        artist: &str,
        fingerprint: ChromaFingerprint,
        key: u8,
    ) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            artist: artist.to_string(),
            fingerprint,
            estimated_key: key % 12,
        }
    }
}

/// Detection result when comparing two recordings.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// Query track ID.
    pub query_id: String,
    /// Reference track ID.
    pub reference_id: String,
    /// Similarity score (0.0..1.0).
    pub similarity: f64,
    /// Best key transposition applied (semitones).
    pub key_shift: i32,
    /// Whether the pair is classified as a cover.
    pub is_cover: bool,
}

/// Database of known tracks for cover detection.
#[derive(Debug, Clone)]
pub struct CoverDatabase {
    /// Stored tracks.
    entries: Vec<CoverVersion>,
    /// Index from ID to position.
    index: HashMap<String, usize>,
}

impl CoverDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a track to the database.
    pub fn add(&mut self, entry: CoverVersion) {
        let idx = self.entries.len();
        self.index.insert(entry.id.clone(), idx);
        self.entries.push(entry);
    }

    /// Look up an entry by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CoverVersion> {
        self.index.get(id).map(|&i| &self.entries[i])
    }
}

impl Default for CoverDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Detector that finds cover versions by comparing chroma fingerprints.
#[derive(Debug)]
pub struct CoverDetector {
    /// Similarity threshold above which a pair is considered a cover.
    pub threshold: f64,
}

impl Default for CoverDetector {
    fn default() -> Self {
        Self { threshold: 0.7 }
    }
}

impl CoverDetector {
    /// Create a detector with the given threshold.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Compare two fingerprints, trying all 12 key transpositions.
    #[must_use]
    pub fn compare(&self, a: &ChromaFingerprint, b: &ChromaFingerprint) -> (f64, i32) {
        let mut best_sim = 0.0_f64;
        let mut best_shift = 0_i32;

        for shift in 0..12_i32 {
            let b_transposed = b.transposed(shift);
            let sim = Self::cosine_similarity_sequence(a, &b_transposed);
            if sim > best_sim {
                best_sim = sim;
                best_shift = shift;
            }
        }
        (best_sim, best_shift)
    }

    /// Detect whether a query is a cover of any entry in the database.
    #[must_use]
    pub fn detect(&self, query: &CoverVersion, db: &CoverDatabase) -> Vec<DetectionResult> {
        let mut results = Vec::new();
        for entry in &db.entries {
            if entry.id == query.id {
                continue;
            }
            let (sim, shift) = self.compare(&query.fingerprint, &entry.fingerprint);
            results.push(DetectionResult {
                query_id: query.id.clone(),
                reference_id: entry.id.clone(),
                similarity: sim,
                key_shift: shift,
                is_cover: sim >= self.threshold,
            });
        }
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Average cosine similarity across aligned frames.
    fn cosine_similarity_sequence(a: &ChromaFingerprint, b: &ChromaFingerprint) -> f64 {
        let n = a.frames.len().min(b.frames.len());
        if n == 0 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .map(|i| Self::cosine_sim_12(&a.frames[i], &b.frames[i]))
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let avg = total / n as f64;
        avg
    }

    /// Cosine similarity of two 12-dimensional vectors.
    fn cosine_sim_12(a: &[f64; 12], b: &[f64; 12]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fingerprint(value: f64, n_frames: usize) -> ChromaFingerprint {
        let frame = [value; 12];
        ChromaFingerprint::new(vec![frame; n_frames], n_frames as f64 * 0.1)
    }

    fn make_c_major_fingerprint(n_frames: usize) -> ChromaFingerprint {
        // C major chord: C=1, E=1, G=1, rest=0
        let mut frame = [0.0_f64; 12];
        frame[0] = 1.0; // C
        frame[4] = 1.0; // E
        frame[7] = 1.0; // G
        ChromaFingerprint::new(vec![frame; n_frames], n_frames as f64 * 0.1)
    }

    #[test]
    fn test_fingerprint_creation() {
        let fp = make_fingerprint(0.5, 10);
        assert_eq!(fp.len(), 10);
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_fingerprint_empty() {
        let fp = ChromaFingerprint::new(vec![], 0.0);
        assert!(fp.is_empty());
    }

    #[test]
    fn test_fingerprint_transpose() {
        let fp = make_c_major_fingerprint(5);
        let transposed = fp.transposed(3); // Transpose up 3 semitones
                                           // C -> Eb, E -> G, G -> Bb
        assert!((transposed.frames[0][3] - 1.0).abs() < f64::EPSILON); // Eb
        assert!((transposed.frames[0][7] - 1.0).abs() < f64::EPSILON); // G
        assert!((transposed.frames[0][10] - 1.0).abs() < f64::EPSILON); // Bb
    }

    #[test]
    fn test_average_chroma() {
        let fp = make_fingerprint(0.5, 10);
        let avg = fp.average_chroma();
        for v in &avg {
            assert!((v - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_cover_version_creation() {
        let fp = make_fingerprint(0.5, 10);
        let cv = CoverVersion::new("t1", "Song A", "Artist 1", fp, 0);
        assert_eq!(cv.id, "t1");
        assert_eq!(cv.estimated_key, 0);
    }

    #[test]
    fn test_database_operations() {
        let mut db = CoverDatabase::new();
        assert!(db.is_empty());
        let fp = make_fingerprint(0.5, 10);
        db.add(CoverVersion::new("t1", "Song A", "Artist 1", fp, 0));
        assert_eq!(db.len(), 1);
        assert!(db.get("t1").is_some());
        assert!(db.get("nonexistent").is_none());
    }

    #[test]
    fn test_detector_identical() {
        let det = CoverDetector::new(0.7);
        let fp = make_c_major_fingerprint(20);
        let (sim, _) = det.compare(&fp, &fp);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_detector_transposed_cover() {
        let det = CoverDetector::new(0.7);
        let original = make_c_major_fingerprint(20);
        let transposed = original.transposed(5); // up a fourth
        let (sim, shift) = det.compare(&original, &transposed);
        // Should find high similarity at shift=7 (12-5)
        assert!(sim > 0.9, "similarity was {sim}");
        assert!(shift == 7 || shift == 0, "shift was {shift}");
    }

    #[test]
    fn test_detector_different() {
        let det = CoverDetector::new(0.7);
        let a = make_c_major_fingerprint(20);
        // D minor: D=1, F=1, A=1
        let mut frame = [0.0_f64; 12];
        frame[2] = 1.0;
        frame[5] = 1.0;
        frame[9] = 1.0;
        let b = ChromaFingerprint::new(vec![frame; 20], 2.0);
        let (sim, _) = det.compare(&a, &b);
        // Different chord: similarity should be lower
        assert!(sim < 1.0);
    }

    #[test]
    fn test_detect_in_database() {
        let det = CoverDetector::new(0.5);
        let mut db = CoverDatabase::new();
        let fp1 = make_c_major_fingerprint(20);
        db.add(CoverVersion::new(
            "t1",
            "Original",
            "Artist A",
            fp1.clone(),
            0,
        ));
        let fp2 = make_fingerprint(0.1, 20);
        db.add(CoverVersion::new("t2", "Other", "Artist B", fp2, 3));

        let query_fp = fp1.transposed(2);
        let query = CoverVersion::new("q1", "Cover", "Artist C", query_fp, 2);
        let results = det.detect(&query, &db);
        assert_eq!(results.len(), 2);
        // First result should match original
        assert_eq!(results[0].reference_id, "t1");
    }

    #[test]
    fn test_detection_result_fields() {
        let det = CoverDetector::new(0.5);
        let fp = make_c_major_fingerprint(10);
        let mut db = CoverDatabase::new();
        db.add(CoverVersion::new("ref", "Ref", "A", fp.clone(), 0));
        let query = CoverVersion::new("q", "Query", "B", fp, 0);
        let results = det.detect(&query, &db);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_cover);
        assert!(results[0].similarity > 0.9);
    }

    #[test]
    fn test_empty_fingerprint_compare() {
        let det = CoverDetector::new(0.5);
        let a = ChromaFingerprint::new(vec![], 0.0);
        let b = make_c_major_fingerprint(10);
        let (sim, _) = det.compare(&a, &b);
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_average_chroma_empty() {
        let fp = ChromaFingerprint::new(vec![], 0.0);
        let avg = fp.average_chroma();
        for v in &avg {
            assert!((*v - 0.0).abs() < f64::EPSILON);
        }
    }
}
