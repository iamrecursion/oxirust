//! Content indexing for automated media search and retrieval.
//!
//! Provides per-frame feature extraction, scene boundary detection, KNN
//! similarity search, keyframe extraction, and content summarization.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ContentFeatures
// ---------------------------------------------------------------------------

/// Per-frame feature descriptor used for indexing and retrieval.
#[derive(Debug, Clone)]
pub struct ContentFeatures {
    /// Index of the frame in the source video.
    pub frame_number: u64,
    /// Timestamp of the frame in milliseconds.
    pub timestamp_ms: i64,
    /// 16-bucket luma (brightness) histogram. Values sum to 1.0.
    pub luma_histogram: [f32; 16],
    /// 36-bucket hue histogram. Values sum to 1.0.
    pub hue_histogram: [f32; 36],
    /// Fraction of pixels considered edges (0.0–1.0).
    pub edge_density: f32,
    /// Dominant motion vector `(dx, dy)` in pixels per frame.
    pub motion_vector: (f32, f32),
    /// Identifier of the scene this frame belongs to.
    pub scene_id: u32,
}

impl ContentFeatures {
    /// Create a new, zeroed feature descriptor.
    pub fn new(frame_number: u64, timestamp_ms: i64) -> Self {
        Self {
            frame_number,
            timestamp_ms,
            luma_histogram: [0.0; 16],
            hue_histogram: [0.0; 36],
            edge_density: 0.0,
            motion_vector: (0.0, 0.0),
            scene_id: 0,
        }
    }

    /// Magnitude of the motion vector.
    pub fn motion_magnitude(&self) -> f32 {
        let (dx, dy) = self.motion_vector;
        (dx * dx + dy * dy).sqrt()
    }

    /// Clamp all histogram values to [0, 1] and re-normalize to sum to 1.
    pub fn normalize_histograms(&mut self) {
        normalize_histogram(&mut self.luma_histogram);
        normalize_histogram(&mut self.hue_histogram);
    }

    /// Dominant luma bucket index (0–15).
    pub fn dominant_luma_bucket(&self) -> usize {
        dominant_bucket(&self.luma_histogram)
    }

    /// Dominant hue bucket index (0–35).
    pub fn dominant_hue_bucket(&self) -> usize {
        dominant_bucket(&self.hue_histogram)
    }
}

fn normalize_histogram<const N: usize>(hist: &mut [f32; N]) {
    for v in hist.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    let total: f32 = hist.iter().sum();
    if total > 1e-10 {
        for v in hist.iter_mut() {
            *v /= total;
        }
    }
}

fn dominant_bucket<const N: usize>(hist: &[f32; N]) -> usize {
    hist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Feature distance
// ---------------------------------------------------------------------------

/// Compute the weighted distance between two feature descriptors.
///
/// Distance = 0.4 × luma histogram SAD + 0.3 × motion magnitude diff + 0.3 × edge diff
pub fn feature_distance(a: &ContentFeatures, b: &ContentFeatures) -> f32 {
    let hist_diff = histogram_sad(&a.luma_histogram, &b.luma_histogram);
    let motion_diff = (a.motion_magnitude() - b.motion_magnitude()).abs();
    let edge_diff = (a.edge_density - b.edge_density).abs();

    0.4 * hist_diff + 0.3 * motion_diff + 0.3 * edge_diff
}

fn histogram_sad<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / N as f32
}

// ---------------------------------------------------------------------------
// ContentIndex and IndexBuilder
// ---------------------------------------------------------------------------

/// An indexed collection of frame features with scene boundary information.
#[derive(Debug, Clone)]
pub struct ContentIndex {
    /// Feature descriptors for each indexed frame.
    pub entries: Vec<ContentFeatures>,
    /// Scene boundary intervals `(start_frame, end_frame)` (inclusive).
    pub frame_intervals: Vec<(u64, u64)>,
}

impl ContentIndex {
    /// Number of indexed frames.
    pub fn frame_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of scenes.
    pub fn scene_count(&self) -> usize {
        self.frame_intervals.len()
    }

    /// Total duration in milliseconds (from first to last frame).
    pub fn duration_ms(&self) -> i64 {
        match (self.entries.first(), self.entries.last()) {
            (Some(first), Some(last)) => last.timestamp_ms - first.timestamp_ms,
            _ => 0,
        }
    }

    /// Average motion magnitude across all indexed frames.
    pub fn avg_motion(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let total: f32 = self.entries.iter().map(|e| e.motion_magnitude()).sum();
        total / self.entries.len() as f32
    }
}

/// Incrementally builds a `ContentIndex` from a stream of frame features.
#[derive(Debug, Default)]
pub struct IndexBuilder {
    entries: Vec<ContentFeatures>,
}

impl IndexBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame's features to the index.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_frame(&mut self, features: ContentFeatures) -> &mut Self {
        self.entries.push(features);
        self
    }

    /// Detect scene boundaries by measuring the feature distance between
    /// consecutive frames against `threshold`.
    ///
    /// Returns the frame numbers where a scene boundary is detected.
    pub fn detect_scene_boundaries(&self, threshold: f32) -> Vec<u64> {
        self.entries
            .windows(2)
            .filter_map(|pair| {
                let dist = feature_distance(&pair[0], &pair[1]);
                if dist > threshold {
                    Some(pair[1].frame_number)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Finalize the index.
    ///
    /// Detects scene boundaries using a default threshold of `0.15` and
    /// computes scene intervals. Call `detect_scene_boundaries` first if you
    /// need a custom threshold before calling `build`.
    pub fn build(self) -> ContentIndex {
        self.build_with_threshold(0.15)
    }

    /// Finalize the index using a specific scene-boundary threshold.
    pub fn build_with_threshold(self, threshold: f32) -> ContentIndex {
        let boundaries = self
            .entries
            .windows(2)
            .filter_map(|pair| {
                let dist = feature_distance(&pair[0], &pair[1]);
                if dist > threshold {
                    Some(pair[1].frame_number)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let frame_intervals = build_intervals(&self.entries, &boundaries);

        ContentIndex {
            entries: self.entries,
            frame_intervals,
        }
    }
}

fn build_intervals(entries: &[ContentFeatures], boundaries: &[u64]) -> Vec<(u64, u64)> {
    if entries.is_empty() {
        return Vec::new();
    }

    let first_frame = entries.first().map(|e| e.frame_number).unwrap_or(0);
    let last_frame = entries.last().map(|e| e.frame_number).unwrap_or(0);

    let mut intervals = Vec::new();
    let mut scene_start = first_frame;

    for &boundary in boundaries {
        if boundary > scene_start {
            intervals.push((scene_start, boundary - 1));
            scene_start = boundary;
        }
    }
    intervals.push((scene_start, last_frame));
    intervals
}

// ---------------------------------------------------------------------------
// KNN search
// ---------------------------------------------------------------------------

/// Query the index for the `top_k` frames most similar to `query`.
///
/// Returns `(frame_number, distance)` pairs sorted by ascending distance.
pub fn query_similar_frames(
    index: &ContentIndex,
    query: &ContentFeatures,
    top_k: usize,
) -> Vec<(u64, f32)> {
    if index.entries.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(u64, f32)> = index
        .entries
        .iter()
        .map(|e| (e.frame_number, feature_distance(e, query)))
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

// ---------------------------------------------------------------------------
// Keyframe extraction
// ---------------------------------------------------------------------------

/// Extract up to `max_keyframes` representative frame numbers from the index.
///
/// Strategy:
/// 1. Always include the first frame of each scene (boundary keyframes).
/// 2. Fill remaining slots with uniformly sampled frames.
/// 3. Sort and deduplicate the result.
pub fn extract_keyframes(index: &ContentIndex, max_keyframes: usize) -> Vec<u64> {
    if index.entries.is_empty() || max_keyframes == 0 {
        return Vec::new();
    }

    // Collect scene boundary frames
    let mut keyframes: Vec<u64> = index
        .frame_intervals
        .iter()
        .map(|(start, _)| *start)
        .collect();

    // Fill remaining slots with uniform sampling
    let remaining = max_keyframes.saturating_sub(keyframes.len());
    if remaining > 0 && index.entries.len() > 1 {
        let step = index.entries.len() / (remaining + 1);
        if step > 0 {
            for i in 1..=remaining {
                let idx = (i * step).min(index.entries.len() - 1);
                keyframes.push(index.entries[idx].frame_number);
            }
        }
    }

    // Sort and deduplicate
    keyframes.sort_unstable();
    keyframes.dedup();
    keyframes.truncate(max_keyframes);
    keyframes
}

// ---------------------------------------------------------------------------
// ContentSummary
// ---------------------------------------------------------------------------

/// High-level summary of a content index.
#[derive(Debug, Clone)]
pub struct ContentSummary {
    /// Total duration in milliseconds.
    pub duration_ms: i64,
    /// Number of detected scenes.
    pub scene_count: u32,
    /// Average motion magnitude across all frames.
    pub avg_motion: f32,
    /// Dominant colors per scene as approximate RGB triples.
    pub dominant_colors: Vec<[u8; 3]>,
    /// Frame numbers selected as representative keyframes.
    pub keyframes: Vec<u64>,
}

/// Generate a `ContentSummary` from a content index.
pub fn summarize(index: &ContentIndex) -> ContentSummary {
    let duration_ms = index.duration_ms();
    let scene_count = index.scene_count() as u32;
    let avg_motion = index.avg_motion();

    // Derive one dominant color per scene from the dominant luma bucket
    let dominant_colors: Vec<[u8; 3]> = index
        .frame_intervals
        .iter()
        .map(|(start, end)| {
            // Find the first frame in this interval
            let representative = index
                .entries
                .iter()
                .find(|e| e.frame_number >= *start && e.frame_number <= *end);

            if let Some(frame) = representative {
                let luma_bucket = frame.dominant_luma_bucket();
                // Map bucket (0–15) to a gray luma value
                let luma = ((luma_bucket as f32 / 15.0) * 255.0).round() as u8;

                // Map hue bucket (0–35) to a hue angle, then to approximate RGB
                let hue_bucket = frame.dominant_hue_bucket();
                let hue_deg = (hue_bucket as f32 / 36.0) * 360.0;
                hue_luma_to_rgb(hue_deg, luma)
            } else {
                [128, 128, 128]
            }
        })
        .collect();

    let keyframes = extract_keyframes(index, (scene_count as usize * 2).max(8));

    ContentSummary {
        duration_ms,
        scene_count,
        avg_motion,
        dominant_colors,
        keyframes,
    }
}

/// Convert a hue angle and luma value to an approximate RGB triple.
fn hue_luma_to_rgb(hue_deg: f32, luma: u8) -> [u8; 3] {
    // Convert hue to RGB using the HSV model with S=1, V=luma/255
    let h = hue_deg % 360.0;
    let v = luma as f32 / 255.0;
    let s = 0.8_f32; // fixed saturation

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(frame: u64, luma_peak: usize) -> ContentFeatures {
        let mut f = ContentFeatures::new(frame, frame as i64 * 33);
        f.luma_histogram[luma_peak] = 1.0;
        f.hue_histogram[0] = 1.0;
        f.edge_density = 0.2;
        f.motion_vector = (1.0, 0.0);
        f
    }

    fn make_features_with_motion(frame: u64, dx: f32, dy: f32) -> ContentFeatures {
        let mut f = ContentFeatures::new(frame, frame as i64 * 33);
        f.luma_histogram[0] = 1.0;
        f.hue_histogram[0] = 1.0;
        f.edge_density = 0.1;
        f.motion_vector = (dx, dy);
        f
    }

    fn build_index(n: usize, threshold: f32) -> ContentIndex {
        let mut builder = IndexBuilder::new();
        for i in 0..n {
            // Alternate luma peaks to create detectable scene boundaries
            let luma_peak = if i < n / 2 { 0 } else { 15 };
            builder.add_frame(make_features(i as u64, luma_peak));
        }
        builder.build_with_threshold(threshold)
    }

    // -- ContentFeatures tests --

    #[test]
    fn test_motion_magnitude() {
        let f = make_features_with_motion(0, 3.0, 4.0);
        assert!((f.motion_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_histograms() {
        let mut f = ContentFeatures::new(0, 0);
        f.luma_histogram[0] = 2.0;
        f.luma_histogram[1] = 2.0;
        f.normalize_histograms();
        let sum: f32 = f.luma_histogram.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dominant_luma_bucket() {
        let mut f = ContentFeatures::new(0, 0);
        f.luma_histogram[7] = 0.9;
        assert_eq!(f.dominant_luma_bucket(), 7);
    }

    #[test]
    fn test_dominant_hue_bucket() {
        let mut f = ContentFeatures::new(0, 0);
        f.hue_histogram[20] = 0.8;
        assert_eq!(f.dominant_hue_bucket(), 20);
    }

    // -- feature_distance tests --

    #[test]
    fn test_feature_distance_identical() {
        let a = make_features(0, 5);
        let b = a.clone();
        let d = feature_distance(&a, &b);
        assert!(
            d < 1e-5,
            "identical features should have ~0 distance, got {d}"
        );
    }

    #[test]
    fn test_feature_distance_different_luma() {
        let a = make_features(0, 0);
        let b = make_features(1, 15);
        let d = feature_distance(&a, &b);
        assert!(
            d > 0.0,
            "different luma peaks should have non-zero distance"
        );
    }

    #[test]
    fn test_feature_distance_symmetry() {
        let a = make_features(0, 0);
        let b = make_features(1, 8);
        assert!((feature_distance(&a, &b) - feature_distance(&b, &a)).abs() < 1e-6);
    }

    // -- IndexBuilder tests --

    #[test]
    fn test_index_builder_frame_count() {
        let index = build_index(10, 0.5);
        assert_eq!(index.frame_count(), 10);
    }

    #[test]
    fn test_index_builder_detect_scene_boundaries() {
        let mut builder = IndexBuilder::new();
        for i in 0..10usize {
            let luma = if i < 5 { 0 } else { 15 };
            builder.add_frame(make_features(i as u64, luma));
        }
        let boundaries = builder.detect_scene_boundaries(0.01);
        // Should detect a boundary at frame 5 where luma peak changes
        assert!(
            boundaries.contains(&5),
            "expected boundary at 5, got {:?}",
            boundaries
        );
    }

    #[test]
    fn test_index_builder_build_creates_intervals() {
        let index = build_index(10, 0.01);
        // At least one interval expected
        assert!(!index.frame_intervals.is_empty());
    }

    #[test]
    fn test_index_no_boundary_single_scene() {
        let mut builder = IndexBuilder::new();
        for i in 0..5u64 {
            builder.add_frame(make_features(i, 5)); // all same
        }
        let index = builder.build_with_threshold(0.5);
        // No boundary → single scene interval
        assert_eq!(index.scene_count(), 1);
    }

    // -- ContentIndex tests --

    #[test]
    fn test_content_index_duration_ms() {
        let mut builder = IndexBuilder::new();
        builder.add_frame(make_features(0, 0)); // ts = 0
        builder.add_frame(make_features(30, 0)); // ts = 990ms
        let index = builder.build();
        assert!(index.duration_ms() > 0);
    }

    #[test]
    fn test_content_index_avg_motion() {
        let mut builder = IndexBuilder::new();
        for i in 0..4u64 {
            let f = make_features_with_motion(i, 3.0, 4.0); // magnitude = 5.0
            builder.add_frame(f);
        }
        let index = builder.build();
        assert!((index.avg_motion() - 5.0).abs() < 1e-4);
    }

    // -- query_similar_frames tests --

    #[test]
    fn test_query_similar_frames_top1() {
        let index = build_index(10, 0.5);
        let query = make_features(0, 0); // luma peak 0
        let results = query_similar_frames(&index, &query, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_similar_frames_sorted() {
        let index = build_index(10, 0.5);
        let query = make_features(0, 0);
        let results = query_similar_frames(&index, &query, 5);
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by distance");
        }
    }

    #[test]
    fn test_query_similar_frames_empty_index() {
        let index = ContentIndex {
            entries: Vec::new(),
            frame_intervals: Vec::new(),
        };
        let query = make_features(0, 0);
        let results = query_similar_frames(&index, &query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_similar_frames_top_k_zero() {
        let index = build_index(5, 0.5);
        let query = make_features(0, 0);
        let results = query_similar_frames(&index, &query, 0);
        assert!(results.is_empty());
    }

    // -- extract_keyframes tests --

    #[test]
    fn test_extract_keyframes_respects_max() {
        let index = build_index(30, 0.01);
        let kf = extract_keyframes(&index, 5);
        assert!(kf.len() <= 5);
    }

    #[test]
    fn test_extract_keyframes_no_duplicates() {
        let index = build_index(20, 0.01);
        let kf = extract_keyframes(&index, 10);
        let unique: std::collections::HashSet<u64> = kf.iter().copied().collect();
        assert_eq!(unique.len(), kf.len(), "duplicate keyframes found");
    }

    #[test]
    fn test_extract_keyframes_sorted() {
        let index = build_index(20, 0.01);
        let kf = extract_keyframes(&index, 10);
        for w in kf.windows(2) {
            assert!(w[0] < w[1], "keyframes not sorted");
        }
    }

    #[test]
    fn test_extract_keyframes_zero_max() {
        let index = build_index(10, 0.5);
        let kf = extract_keyframes(&index, 0);
        assert!(kf.is_empty());
    }

    #[test]
    fn test_extract_keyframes_empty_index() {
        let index = ContentIndex {
            entries: Vec::new(),
            frame_intervals: Vec::new(),
        };
        let kf = extract_keyframes(&index, 10);
        assert!(kf.is_empty());
    }

    // -- summarize tests --

    #[test]
    fn test_summarize_basic() {
        let index = build_index(20, 0.01);
        let summary = summarize(&index);
        assert!(summary.duration_ms >= 0);
        assert!(summary.scene_count >= 1);
        assert!(!summary.keyframes.is_empty());
    }

    #[test]
    fn test_summarize_dominant_colors_count() {
        let index = build_index(20, 0.01);
        let summary = summarize(&index);
        // One dominant color per scene
        assert_eq!(summary.dominant_colors.len(), summary.scene_count as usize);
    }

    #[test]
    fn test_summarize_empty_index() {
        let index = ContentIndex {
            entries: Vec::new(),
            frame_intervals: Vec::new(),
        };
        let summary = summarize(&index);
        assert_eq!(summary.duration_ms, 0);
        assert_eq!(summary.scene_count, 0);
        assert!(summary.keyframes.is_empty());
    }
}
