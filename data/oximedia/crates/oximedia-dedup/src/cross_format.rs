//! Cross-format duplicate detection: same content in different containers/codecs.
//!
//! Detects when the same media content exists in multiple file formats
//! (e.g., the same video in MP4/MKV/WebM or the same audio in FLAC/OGG/WAV).
//!
//! # Approach
//!
//! Cross-format duplicates cannot be found by cryptographic hashing because
//! different containers/codecs produce entirely different byte streams.  Instead
//! we combine multiple format-agnostic signals:
//!
//! 1. **Duration matching** -- content in different containers should have
//!    nearly identical duration.
//! 2. **Perceptual hash matching** -- visual content produces similar pHash /
//!    dHash regardless of codec.
//! 3. **Audio fingerprint matching** -- spectral fingerprints survive
//!    re-encoding.
//! 4. **Resolution / channel layout matching** -- same content typically
//!    retains the same frame size and audio channel count.
//!
//! The module assigns a **cross-format confidence** score (0.0 - 1.0) to
//! each candidate pair and groups files above a configurable threshold.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FormatInfo
// ---------------------------------------------------------------------------

/// Normalised, format-agnostic content descriptor.
#[derive(Debug, Clone)]
pub struct FormatInfo {
    /// File path.
    pub path: String,
    /// Container format label (e.g. "mp4", "mkv", "webm", "flac").
    pub container: String,
    /// Video codec label (e.g. "av1", "vp9"), if present.
    pub video_codec: Option<String>,
    /// Audio codec label (e.g. "opus", "vorbis", "flac"), if present.
    pub audio_codec: Option<String>,
    /// Duration in seconds.
    pub duration_secs: Option<f64>,
    /// Video width in pixels.
    pub width: Option<u32>,
    /// Video height in pixels.
    pub height: Option<u32>,
    /// Audio sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    pub audio_channels: Option<u32>,
    /// 64-bit perceptual hash of a representative frame (if available).
    pub phash: Option<u64>,
    /// Audio fingerprint bytes (if available).
    pub audio_fingerprint: Option<Vec<u8>>,
}

impl FormatInfo {
    /// Create a minimal `FormatInfo` with only path and container.
    #[must_use]
    pub fn new(path: impl Into<String>, container: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            container: container.into(),
            video_codec: None,
            audio_codec: None,
            duration_secs: None,
            width: None,
            height: None,
            sample_rate: None,
            audio_channels: None,
            phash: None,
            audio_fingerprint: None,
        }
    }

    /// Builder: set duration.
    #[must_use]
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Builder: set video dimensions.
    #[must_use]
    pub fn with_resolution(mut self, w: u32, h: u32) -> Self {
        self.width = Some(w);
        self.height = Some(h);
        self
    }

    /// Builder: set codecs.
    #[must_use]
    pub fn with_codecs(mut self, video: Option<String>, audio: Option<String>) -> Self {
        self.video_codec = video;
        self.audio_codec = audio;
        self
    }

    /// Builder: set perceptual hash.
    #[must_use]
    pub fn with_phash(mut self, hash: u64) -> Self {
        self.phash = Some(hash);
        self
    }

    /// Builder: set audio fingerprint.
    #[must_use]
    pub fn with_audio_fingerprint(mut self, fp: Vec<u8>) -> Self {
        self.audio_fingerprint = Some(fp);
        self
    }

    /// Builder: set audio info.
    #[must_use]
    pub fn with_audio_info(mut self, sample_rate: u32, channels: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self.audio_channels = Some(channels);
        self
    }

    /// Returns `true` if the containers differ from `other`.
    #[must_use]
    pub fn is_different_format(&self, other: &Self) -> bool {
        self.container.to_lowercase() != other.container.to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// CrossFormatConfig
// ---------------------------------------------------------------------------

/// Configuration for cross-format detection.
#[derive(Debug, Clone)]
pub struct CrossFormatConfig {
    /// Maximum allowed duration difference in seconds.
    pub max_duration_diff_secs: f64,
    /// Maximum Hamming distance for perceptual hash match (out of 64 bits).
    pub max_phash_distance: u32,
    /// Minimum audio fingerprint similarity (0.0 - 1.0).
    pub min_audio_similarity: f64,
    /// Overall confidence threshold for declaring a cross-format duplicate.
    pub confidence_threshold: f64,
    /// Weight for duration similarity in composite score.
    pub weight_duration: f64,
    /// Weight for resolution match in composite score.
    pub weight_resolution: f64,
    /// Weight for perceptual hash match in composite score.
    pub weight_phash: f64,
    /// Weight for audio fingerprint match in composite score.
    pub weight_audio: f64,
}

impl Default for CrossFormatConfig {
    fn default() -> Self {
        Self {
            max_duration_diff_secs: 0.5,
            max_phash_distance: 8,
            min_audio_similarity: 0.80,
            confidence_threshold: 0.75,
            weight_duration: 0.25,
            weight_resolution: 0.15,
            weight_phash: 0.35,
            weight_audio: 0.25,
        }
    }
}

impl CrossFormatConfig {
    /// Normalise weights so they sum to 1.0.
    #[must_use]
    pub fn normalised_weights(&self) -> (f64, f64, f64, f64) {
        let total =
            self.weight_duration + self.weight_resolution + self.weight_phash + self.weight_audio;
        if total < f64::EPSILON {
            return (0.25, 0.25, 0.25, 0.25);
        }
        (
            self.weight_duration / total,
            self.weight_resolution / total,
            self.weight_phash / total,
            self.weight_audio / total,
        )
    }
}

// ---------------------------------------------------------------------------
// CrossFormatMatch
// ---------------------------------------------------------------------------

/// A confirmed cross-format duplicate pair.
#[derive(Debug, Clone)]
pub struct CrossFormatMatch {
    /// Path of the first file.
    pub path_a: String,
    /// Path of the second file.
    pub path_b: String,
    /// Container of the first file.
    pub container_a: String,
    /// Container of the second file.
    pub container_b: String,
    /// Overall confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Individual signal scores.
    pub signal_scores: SignalScores,
}

/// Individual signal similarity scores.
#[derive(Debug, Clone)]
pub struct SignalScores {
    /// Duration similarity (1.0 = identical).
    pub duration: Option<f64>,
    /// Resolution match (1.0 = same, 0.0 = different or missing).
    pub resolution: Option<f64>,
    /// Perceptual hash similarity (0.0 - 1.0).
    pub phash: Option<f64>,
    /// Audio fingerprint similarity (0.0 - 1.0).
    pub audio: Option<f64>,
}

// ---------------------------------------------------------------------------
// CrossFormatGroup
// ---------------------------------------------------------------------------

/// A group of files that contain the same content in different formats.
#[derive(Debug, Clone)]
pub struct CrossFormatGroup {
    /// Files in this group.
    pub files: Vec<String>,
    /// Containers present in this group.
    pub containers: Vec<String>,
    /// Best confidence score among all pairs in this group.
    pub best_confidence: f64,
}

// ---------------------------------------------------------------------------
// Comparison functions
// ---------------------------------------------------------------------------

/// Compare duration similarity.
///
/// Returns 1.0 for identical durations, tapering to 0.0 at `max_diff` seconds.
fn duration_similarity(a: Option<f64>, b: Option<f64>, max_diff: f64) -> Option<f64> {
    match (a, b) {
        (Some(da), Some(db)) => {
            let diff = (da - db).abs();
            if max_diff < f64::EPSILON {
                return Some(if diff < f64::EPSILON { 1.0 } else { 0.0 });
            }
            Some((1.0 - diff / max_diff).max(0.0))
        }
        _ => None,
    }
}

/// Compare resolution.
///
/// Returns 1.0 if both width and height match, 0.5 if only one matches, 0.0 otherwise.
fn resolution_similarity(
    w_a: Option<u32>,
    h_a: Option<u32>,
    w_b: Option<u32>,
    h_b: Option<u32>,
) -> Option<f64> {
    match (w_a, h_a, w_b, h_b) {
        (Some(wa), Some(ha), Some(wb), Some(hb)) => {
            // Compute per-dimension ratio similarity.
            let w_ratio = wa.min(wb) as f64 / wa.max(wb).max(1) as f64;
            let h_ratio = ha.min(hb) as f64 / ha.max(hb).max(1) as f64;

            let score = if wa == wb && ha == hb {
                1.0
            } else if w_ratio > 0.99 && h_ratio > 0.99 {
                // Near-identical (e.g. 1920 vs 1918 due to encoding quirks).
                0.95
            } else if w_ratio > 0.95 && h_ratio > 0.95 {
                // Very close resolutions.
                0.85
            } else if (wa == wb) || (ha == hb) {
                // One dimension matches exactly.
                0.5
            } else {
                0.0
            };

            Some(score)
        }
        _ => None,
    }
}

/// Compare perceptual hashes via Hamming distance.
fn phash_similarity(a: Option<u64>, b: Option<u64>, max_distance: u32) -> Option<f64> {
    match (a, b) {
        (Some(ha), Some(hb)) => {
            let dist = (ha ^ hb).count_ones();
            if dist > max_distance {
                Some(0.0)
            } else {
                Some(1.0 - dist as f64 / 64.0)
            }
        }
        _ => None,
    }
}

/// Compare audio fingerprints using bit-level Hamming similarity.
fn audio_fingerprint_similarity(a: &Option<Vec<u8>>, b: &Option<Vec<u8>>) -> Option<f64> {
    match (a.as_ref(), b.as_ref()) {
        (Some(fa), Some(fb)) => {
            if fa.is_empty() || fb.is_empty() {
                return Some(0.0);
            }
            let len = fa.len().min(fb.len());
            let total_bits = len * 8;
            if total_bits == 0 {
                return Some(0.0);
            }
            let differing_bits: u32 = fa
                .iter()
                .zip(fb.iter())
                .take(len)
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            Some(1.0 - differing_bits as f64 / total_bits as f64)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CrossFormatDetector
// ---------------------------------------------------------------------------

/// Detector for cross-format duplicates.
#[derive(Debug)]
pub struct CrossFormatDetector {
    config: CrossFormatConfig,
    items: Vec<FormatInfo>,
}

impl CrossFormatDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new(config: CrossFormatConfig) -> Self {
        Self {
            config,
            items: Vec::new(),
        }
    }

    /// Create a detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CrossFormatConfig::default())
    }

    /// Add a file to the detection pool.
    pub fn add(&mut self, info: FormatInfo) {
        self.items.push(info);
    }

    /// Add multiple files.
    pub fn add_batch(&mut self, infos: impl IntoIterator<Item = FormatInfo>) {
        self.items.extend(infos);
    }

    /// Number of files in the pool.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Compare two items and return a match if above threshold.
    fn compare_pair(&self, a: &FormatInfo, b: &FormatInfo) -> Option<CrossFormatMatch> {
        // Only compare files in different formats.
        if !a.is_different_format(b) {
            return None;
        }

        // Quick rejection: duration must be close.
        if let (Some(da), Some(db)) = (a.duration_secs, b.duration_secs) {
            if (da - db).abs() > self.config.max_duration_diff_secs * 2.0 {
                return None;
            }
        }

        let dur_sim = duration_similarity(
            a.duration_secs,
            b.duration_secs,
            self.config.max_duration_diff_secs,
        );
        let res_sim = resolution_similarity(a.width, a.height, b.width, b.height);
        let phash_sim = phash_similarity(a.phash, b.phash, self.config.max_phash_distance);
        let audio_sim = audio_fingerprint_similarity(&a.audio_fingerprint, &b.audio_fingerprint);

        let signal_scores = SignalScores {
            duration: dur_sim,
            resolution: res_sim,
            phash: phash_sim,
            audio: audio_sim,
        };

        // Compute weighted confidence.
        let (wd, wr, wp, wa) = self.config.normalised_weights();
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        if let Some(s) = dur_sim {
            weighted_sum += s * wd;
            weight_sum += wd;
        }
        if let Some(s) = res_sim {
            weighted_sum += s * wr;
            weight_sum += wr;
        }
        if let Some(s) = phash_sim {
            weighted_sum += s * wp;
            weight_sum += wp;
        }
        if let Some(s) = audio_sim {
            weighted_sum += s * wa;
            weight_sum += wa;
        }

        if weight_sum < f64::EPSILON {
            return None;
        }

        let confidence = weighted_sum / weight_sum;

        if confidence >= self.config.confidence_threshold {
            Some(CrossFormatMatch {
                path_a: a.path.clone(),
                path_b: b.path.clone(),
                container_a: a.container.clone(),
                container_b: b.container.clone(),
                confidence,
                signal_scores,
            })
        } else {
            None
        }
    }

    /// Find all cross-format duplicate pairs.
    #[must_use]
    pub fn find_matches(&self) -> Vec<CrossFormatMatch> {
        let mut matches = Vec::new();
        let mut seen_pairs = std::collections::HashSet::new();

        // Group by approximate duration to reduce comparisons.
        let buckets = self.bucket_by_duration();

        for bucket in buckets.values() {
            if bucket.len() < 2 {
                continue;
            }
            for i in 0..bucket.len() {
                for j in (i + 1)..bucket.len() {
                    let (lo, hi) = if bucket[i] < bucket[j] {
                        (bucket[i], bucket[j])
                    } else {
                        (bucket[j], bucket[i])
                    };
                    if !seen_pairs.insert((lo, hi)) {
                        continue; // already checked this pair
                    }
                    if let Some(m) = self.compare_pair(&self.items[lo], &self.items[hi]) {
                        matches.push(m);
                    }
                }
            }
        }

        // Sort by confidence descending.
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Find and group cross-format duplicates using transitive closure.
    #[must_use]
    pub fn find_groups(&self) -> Vec<CrossFormatGroup> {
        let matches = self.find_matches();
        if matches.is_empty() {
            return Vec::new();
        }

        // Build path index.
        let mut path_to_idx: HashMap<&str, usize> = HashMap::new();
        for (i, item) in self.items.iter().enumerate() {
            path_to_idx.insert(&item.path, i);
        }

        // Union-Find for grouping.
        let n = self.items.len();
        let mut parent: Vec<usize> = (0..n).collect();

        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        };

        let mut best_confidence: Vec<f64> = vec![0.0; n];

        for m in &matches {
            if let (Some(&ia), Some(&ib)) = (
                path_to_idx.get(m.path_a.as_str()),
                path_to_idx.get(m.path_b.as_str()),
            ) {
                let ra = find(&mut parent, ia);
                let rb = find(&mut parent, ib);
                if ra != rb {
                    parent[ra] = rb;
                }
                best_confidence[ia] = best_confidence[ia].max(m.confidence);
                best_confidence[ib] = best_confidence[ib].max(m.confidence);
            }
        }

        // Collect groups.
        let mut groups_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups_map.entry(root).or_default().push(i);
        }

        groups_map
            .into_values()
            .filter(|g| g.len() > 1)
            .filter(|g| {
                // Ensure at least 2 different containers.
                let containers: std::collections::HashSet<&str> = g
                    .iter()
                    .map(|&i| self.items[i].container.as_str())
                    .collect();
                containers.len() > 1
            })
            .map(|g| {
                let mut containers: Vec<String> =
                    g.iter().map(|&i| self.items[i].container.clone()).collect();
                containers.sort();
                containers.dedup();

                let bc = g.iter().map(|&i| best_confidence[i]).fold(0.0f64, f64::max);

                CrossFormatGroup {
                    files: g.iter().map(|&i| self.items[i].path.clone()).collect(),
                    containers,
                    best_confidence: bc,
                }
            })
            .collect()
    }

    /// Bucket items by rounded duration for efficient comparison.
    fn bucket_by_duration(&self) -> HashMap<i64, Vec<usize>> {
        let mut buckets: HashMap<i64, Vec<usize>> = HashMap::new();
        let bucket_width = self.config.max_duration_diff_secs.max(0.5);

        for (idx, item) in self.items.iter().enumerate() {
            match item.duration_secs {
                Some(d) => {
                    // Insert into the primary bucket and adjacent buckets
                    // to handle boundary cases.
                    let primary = (d / bucket_width) as i64;
                    for offset in -1..=1 {
                        buckets.entry(primary + offset).or_default().push(idx);
                    }
                }
                None => {
                    // Items without duration go into a special bucket.
                    buckets.entry(i64::MIN).or_default().push(idx);
                }
            }
        }

        // Deduplicate indices within each bucket.
        for bucket in buckets.values_mut() {
            bucket.sort_unstable();
            bucket.dedup();
        }

        buckets
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_info_creation() {
        let info = FormatInfo::new("video.mp4", "mp4")
            .with_duration(120.5)
            .with_resolution(1920, 1080);
        assert_eq!(info.path, "video.mp4");
        assert_eq!(info.container, "mp4");
        assert_eq!(info.duration_secs, Some(120.5));
        assert_eq!(info.width, Some(1920));
        assert_eq!(info.height, Some(1080));
    }

    #[test]
    fn test_is_different_format() {
        let a = FormatInfo::new("a.mp4", "mp4");
        let b = FormatInfo::new("b.mkv", "mkv");
        let c = FormatInfo::new("c.mp4", "MP4");

        assert!(a.is_different_format(&b));
        assert!(!a.is_different_format(&c)); // case-insensitive
    }

    #[test]
    fn test_duration_similarity_identical() {
        let sim = duration_similarity(Some(120.0), Some(120.0), 0.5);
        assert_eq!(sim, Some(1.0));
    }

    #[test]
    fn test_duration_similarity_close() {
        let sim = duration_similarity(Some(120.0), Some(120.3), 0.5);
        let s = sim.expect("should be Some");
        assert!(s > 0.3 && s < 1.0, "sim = {s}");
    }

    #[test]
    fn test_duration_similarity_too_far() {
        let sim = duration_similarity(Some(120.0), Some(121.0), 0.5);
        let s = sim.expect("should be Some");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_duration_similarity_missing() {
        assert!(duration_similarity(None, Some(120.0), 0.5).is_none());
        assert!(duration_similarity(Some(120.0), None, 0.5).is_none());
    }

    #[test]
    fn test_resolution_similarity_exact() {
        let sim = resolution_similarity(Some(1920), Some(1080), Some(1920), Some(1080));
        assert_eq!(sim, Some(1.0));
    }

    #[test]
    fn test_resolution_similarity_different() {
        let sim = resolution_similarity(Some(1920), Some(1080), Some(1280), Some(720));
        let s = sim.expect("should be Some");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_resolution_similarity_partial() {
        let sim = resolution_similarity(Some(1920), Some(1080), Some(1920), Some(720));
        let s = sim.expect("should be Some");
        assert_eq!(s, 0.5);
    }

    #[test]
    fn test_resolution_similarity_missing() {
        assert!(resolution_similarity(None, Some(1080), Some(1920), Some(1080)).is_none());
    }

    #[test]
    fn test_phash_similarity_identical() {
        let sim = phash_similarity(Some(0xDEADBEEF), Some(0xDEADBEEF), 8);
        assert_eq!(sim, Some(1.0));
    }

    #[test]
    fn test_phash_similarity_close() {
        let a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let b = a ^ 0b1111; // 4 bits different
        let sim = phash_similarity(Some(a), Some(b), 8);
        let s = sim.expect("should be Some");
        assert!(s > 0.9, "sim = {s}");
    }

    #[test]
    fn test_phash_similarity_too_far() {
        let sim = phash_similarity(Some(0x0), Some(0xFFFF_FFFF_FFFF_FFFF), 8);
        let s = sim.expect("should be Some");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_audio_fingerprint_similarity_identical() {
        let fp = vec![0xAB, 0xCD, 0xEF, 0x01];
        let sim = audio_fingerprint_similarity(&Some(fp.clone()), &Some(fp));
        assert_eq!(sim, Some(1.0));
    }

    #[test]
    fn test_audio_fingerprint_similarity_different() {
        let a = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let b = vec![0x00, 0x00, 0x00, 0x00];
        let sim = audio_fingerprint_similarity(&Some(a), &Some(b));
        assert_eq!(sim, Some(0.0));
    }

    #[test]
    fn test_audio_fingerprint_similarity_missing() {
        let fp = vec![0xAB];
        assert!(audio_fingerprint_similarity(&None, &Some(fp)).is_none());
    }

    #[test]
    fn test_cross_format_config_normalised_weights() {
        let config = CrossFormatConfig::default();
        let (wd, wr, wp, wa) = config.normalised_weights();
        let total = wd + wr + wp + wa;
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detector_identical_content_different_format() {
        let mut detector = CrossFormatDetector::with_defaults();

        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        let fp = vec![0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89];

        detector.add(
            FormatInfo::new("video.mp4", "mp4")
                .with_duration(120.0)
                .with_resolution(1920, 1080)
                .with_phash(hash)
                .with_audio_fingerprint(fp.clone()),
        );
        detector.add(
            FormatInfo::new("video.mkv", "mkv")
                .with_duration(120.0)
                .with_resolution(1920, 1080)
                .with_phash(hash)
                .with_audio_fingerprint(fp),
        );

        let matches = detector.find_matches();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].confidence > 0.99);
    }

    #[test]
    fn test_detector_same_format_not_matched() {
        let mut detector = CrossFormatDetector::with_defaults();

        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        detector.add(
            FormatInfo::new("a.mp4", "mp4")
                .with_duration(120.0)
                .with_phash(hash),
        );
        detector.add(
            FormatInfo::new("b.mp4", "mp4")
                .with_duration(120.0)
                .with_phash(hash),
        );

        let matches = detector.find_matches();
        assert!(matches.is_empty(), "same format should not be matched");
    }

    #[test]
    fn test_detector_duration_too_different() {
        let mut detector = CrossFormatDetector::with_defaults();

        detector.add(
            FormatInfo::new("short.mp4", "mp4")
                .with_duration(60.0)
                .with_resolution(1920, 1080),
        );
        detector.add(
            FormatInfo::new("long.mkv", "mkv")
                .with_duration(120.0)
                .with_resolution(1920, 1080),
        );

        let matches = detector.find_matches();
        assert!(
            matches.is_empty(),
            "very different durations should not match"
        );
    }

    #[test]
    fn test_detector_find_groups() {
        let mut detector = CrossFormatDetector::with_defaults();

        let hash = 0xAAAA_BBBB_CCCC_DDDDu64;
        let fp = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

        for (path, container) in &[
            ("video.mp4", "mp4"),
            ("video.mkv", "mkv"),
            ("video.webm", "webm"),
        ] {
            detector.add(
                FormatInfo::new(*path, *container)
                    .with_duration(90.0)
                    .with_resolution(1280, 720)
                    .with_phash(hash)
                    .with_audio_fingerprint(fp.clone()),
            );
        }

        let groups = detector.find_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 3);
        assert!(groups[0].containers.len() >= 2);
        assert!(groups[0].best_confidence > 0.9);
    }

    #[test]
    fn test_detector_two_separate_groups() {
        let mut detector = CrossFormatDetector::with_defaults();

        // Group 1
        detector.add(
            FormatInfo::new("a.mp4", "mp4")
                .with_duration(60.0)
                .with_resolution(1920, 1080)
                .with_phash(0x1111_1111_1111_1111),
        );
        detector.add(
            FormatInfo::new("a.mkv", "mkv")
                .with_duration(60.0)
                .with_resolution(1920, 1080)
                .with_phash(0x1111_1111_1111_1111),
        );

        // Group 2 (different content)
        detector.add(
            FormatInfo::new("b.mp4", "mp4")
                .with_duration(300.0)
                .with_resolution(1280, 720)
                .with_phash(0xFFFF_FFFF_FFFF_FFFF),
        );
        detector.add(
            FormatInfo::new("b.webm", "webm")
                .with_duration(300.0)
                .with_resolution(1280, 720)
                .with_phash(0xFFFF_FFFF_FFFF_FFFF),
        );

        let groups = detector.find_groups();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_detector_empty_pool() {
        let detector = CrossFormatDetector::with_defaults();
        assert!(detector.find_matches().is_empty());
        assert!(detector.find_groups().is_empty());
    }

    #[test]
    fn test_detector_single_item() {
        let mut detector = CrossFormatDetector::with_defaults();
        detector.add(FormatInfo::new("only.mp4", "mp4").with_duration(60.0));
        assert!(detector.find_matches().is_empty());
    }

    #[test]
    fn test_detector_partial_signals() {
        // Only duration and resolution, no phash/audio
        let mut detector = CrossFormatDetector::new(CrossFormatConfig {
            confidence_threshold: 0.5, // lower threshold since we have fewer signals
            ..CrossFormatConfig::default()
        });

        detector.add(
            FormatInfo::new("video.mp4", "mp4")
                .with_duration(120.0)
                .with_resolution(1920, 1080),
        );
        detector.add(
            FormatInfo::new("video.mkv", "mkv")
                .with_duration(120.0)
                .with_resolution(1920, 1080),
        );

        let matches = detector.find_matches();
        assert_eq!(matches.len(), 1);
        // Score should reflect only the available signals
        assert!(matches[0].confidence >= 0.5);
    }

    #[test]
    fn test_detector_audio_only_content() {
        let mut detector = CrossFormatDetector::with_defaults();

        let fp = vec![0xAA; 32];
        detector.add(
            FormatInfo::new("song.flac", "flac")
                .with_duration(180.0)
                .with_audio_fingerprint(fp.clone())
                .with_audio_info(44100, 2),
        );
        detector.add(
            FormatInfo::new("song.ogg", "ogg")
                .with_duration(180.0)
                .with_audio_fingerprint(fp)
                .with_audio_info(44100, 2),
        );

        let matches = detector.find_matches();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].confidence > 0.7);
    }

    #[test]
    fn test_signal_scores_populated() {
        let mut detector = CrossFormatDetector::with_defaults();

        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        detector.add(
            FormatInfo::new("a.mp4", "mp4")
                .with_duration(100.0)
                .with_resolution(1920, 1080)
                .with_phash(hash),
        );
        detector.add(
            FormatInfo::new("a.mkv", "mkv")
                .with_duration(100.0)
                .with_resolution(1920, 1080)
                .with_phash(hash),
        );

        let matches = detector.find_matches();
        assert_eq!(matches.len(), 1);

        let scores = &matches[0].signal_scores;
        assert_eq!(scores.duration, Some(1.0));
        assert_eq!(scores.resolution, Some(1.0));
        assert_eq!(scores.phash, Some(1.0));
        assert!(scores.audio.is_none()); // no audio fingerprint provided
    }

    #[test]
    fn test_item_count() {
        let mut detector = CrossFormatDetector::with_defaults();
        assert_eq!(detector.item_count(), 0);
        detector.add(FormatInfo::new("a.mp4", "mp4"));
        detector.add(FormatInfo::new("b.mkv", "mkv"));
        assert_eq!(detector.item_count(), 2);
    }

    #[test]
    fn test_add_batch() {
        let mut detector = CrossFormatDetector::with_defaults();
        detector.add_batch(vec![
            FormatInfo::new("a.mp4", "mp4"),
            FormatInfo::new("b.mkv", "mkv"),
            FormatInfo::new("c.webm", "webm"),
        ]);
        assert_eq!(detector.item_count(), 3);
    }

    #[test]
    fn test_resolution_similarity_near_identical() {
        // Encoding quirk: 1920 vs 1918
        let sim = resolution_similarity(Some(1920), Some(1080), Some(1918), Some(1080));
        let s = sim.expect("should be Some");
        assert!(s > 0.8, "near-identical resolution should score high: {s}");
    }

    #[test]
    fn test_matches_sorted_by_confidence() {
        let mut detector = CrossFormatDetector::new(CrossFormatConfig {
            confidence_threshold: 0.3,
            ..CrossFormatConfig::default()
        });

        // High confidence pair
        detector.add(
            FormatInfo::new("a.mp4", "mp4")
                .with_duration(100.0)
                .with_resolution(1920, 1080)
                .with_phash(0xAAAA),
        );
        detector.add(
            FormatInfo::new("a.mkv", "mkv")
                .with_duration(100.0)
                .with_resolution(1920, 1080)
                .with_phash(0xAAAA),
        );

        // Lower confidence pair (duration slightly off)
        detector.add(
            FormatInfo::new("b.mp4", "mp4")
                .with_duration(200.0)
                .with_resolution(1280, 720)
                .with_phash(0xBBBB),
        );
        detector.add(
            FormatInfo::new("b.webm", "webm")
                .with_duration(200.2)
                .with_resolution(1280, 720)
                .with_phash(0xBBBB),
        );

        let matches = detector.find_matches();
        assert!(matches.len() >= 2);
        // Should be sorted descending by confidence
        for i in 1..matches.len() {
            assert!(matches[i - 1].confidence >= matches[i].confidence);
        }
    }

    #[test]
    fn test_format_info_builders() {
        let info = FormatInfo::new("test.mp4", "mp4")
            .with_codecs(Some("av1".into()), Some("opus".into()))
            .with_audio_info(48000, 6);
        assert_eq!(info.video_codec.as_deref(), Some("av1"));
        assert_eq!(info.audio_codec.as_deref(), Some("opus"));
        assert_eq!(info.sample_rate, Some(48000));
        assert_eq!(info.audio_channels, Some(6));
    }
}
