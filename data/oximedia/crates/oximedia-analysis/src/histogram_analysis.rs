//! Histogram-based image analysis: contrast detection and clipping checks.

#![allow(dead_code)]

/// A single histogram bucket covering a contiguous intensity range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramBucket {
    /// Inclusive lower bound of the bucket (0–255).
    pub lower: u8,
    /// Inclusive upper bound of the bucket (0–255).
    pub upper: u8,
    /// Number of pixels that fell in this bucket.
    pub count: u64,
    /// Total number of pixels considered.
    pub total: u64,
}

impl HistogramBucket {
    /// Create a new bucket.
    #[must_use]
    pub fn new(lower: u8, upper: u8, count: u64, total: u64) -> Self {
        Self {
            lower,
            upper,
            count,
            total,
        }
    }

    /// Fraction of pixels in this bucket (0.0 – 1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fill_ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.count as f64 / self.total as f64
        }
    }
}

/// A 256-bin luminance histogram for a single image or frame.
#[derive(Debug, Clone)]
pub struct ImageHistogram {
    /// Bins indexed 0–255 by luma value.
    pub bins: [u64; 256],
    /// Total number of pixels.
    pub total_pixels: u64,
}

impl ImageHistogram {
    /// Build a histogram from an 8-bit luma plane.
    #[must_use]
    pub fn from_luma(data: &[u8]) -> Self {
        let mut bins = [0u64; 256];
        for &px in data {
            bins[px as usize] += 1;
        }
        Self {
            bins,
            total_pixels: data.len() as u64,
        }
    }

    /// Weighted mean luma value (0–255).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| i as u64 * c)
            .sum();
        sum as f64 / self.total_pixels as f64
    }

    /// Standard deviation of luma values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let diff = i as f64 - mean;
                diff * diff * c as f64
            })
            .sum::<f64>()
            / self.total_pixels as f64;
        variance.sqrt()
    }

    /// `true` when the image appears low-contrast (std dev < threshold).
    #[must_use]
    pub fn is_low_contrast(&self, threshold: f64) -> bool {
        self.std_dev() < threshold
    }

    /// Fraction of pixels at or above `level` (potential highlight clipping).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn highlight_clip_ratio(&self, level: u8) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let clipped: u64 = self.bins[level as usize..].iter().sum();
        clipped as f64 / self.total_pixels as f64
    }

    /// Fraction of pixels at or below `level` (potential shadow clipping).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn shadow_clip_ratio(&self, level: u8) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let clipped: u64 = self.bins[..=level as usize].iter().sum();
        clipped as f64 / self.total_pixels as f64
    }
}

impl Default for ImageHistogram {
    fn default() -> Self {
        Self {
            bins: [0; 256],
            total_pixels: 0,
        }
    }
}

/// Result of a histogram analysis pass.
#[derive(Debug, Clone)]
pub struct HistogramAnalysisResult {
    /// Mean luma.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Shadow clipping ratio (below level 16).
    pub shadow_clip: f64,
    /// Highlight clipping ratio (above level 235).
    pub highlight_clip: f64,
    /// Whether the image is low-contrast.
    pub low_contrast: bool,
}

/// Stateless analyzer that processes `ImageHistogram` values.
#[derive(Debug, Default)]
pub struct HistogramAnalyzer {
    /// Threshold below which an image is considered low-contrast.
    pub contrast_threshold: f64,
    /// Luma level above which pixels count as highlight clipping.
    pub highlight_level: u8,
    /// Luma level below which pixels count as shadow clipping.
    pub shadow_level: u8,
}

impl HistogramAnalyzer {
    /// Create a new analyzer with broadcast-safe defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            contrast_threshold: 20.0,
            highlight_level: 235,
            shadow_level: 16,
        }
    }

    /// Analyze a histogram and return the result.
    #[must_use]
    pub fn analyze(&self, h: &ImageHistogram) -> HistogramAnalysisResult {
        HistogramAnalysisResult {
            mean: h.mean(),
            std_dev: h.std_dev(),
            shadow_clip: h.shadow_clip_ratio(self.shadow_level),
            highlight_clip: h.highlight_clip_ratio(self.highlight_level),
            low_contrast: h.is_low_contrast(self.contrast_threshold),
        }
    }

    /// Detect clipping: returns `(shadow_clipped, highlight_clipped)`.
    #[must_use]
    pub fn detect_clipping(&self, h: &ImageHistogram) -> (bool, bool) {
        let shadow = h.shadow_clip_ratio(self.shadow_level) > 0.01;
        let highlight = h.highlight_clip_ratio(self.highlight_level) > 0.01;
        (shadow, highlight)
    }
}

// ─────────────────────────────────────────────────────────────
// FrameHistogramCache — memoized per-frame histogram
// ─────────────────────────────────────────────────────────────

/// Generic histogram payload that can be cached.
///
/// Wraps an [`ImageHistogram`] so callers can store it by value and look it
/// up without recomputing on each sub-analyzer that processes the same frame.
#[derive(Debug, Clone)]
pub struct HistogramData {
    /// The underlying 256-bin histogram.
    pub histogram: ImageHistogram,
}

impl From<ImageHistogram> for HistogramData {
    fn from(h: ImageHistogram) -> Self {
        Self { histogram: h }
    }
}

/// Compute a fast, low-collision hash key for a pixel buffer.
///
/// Hashes `(width, height)` plus up to 64 pixels from the front and 64 from
/// the back of the buffer.  This is intentionally a *sample* — it runs in
/// O(1) regardless of frame size.
///
/// # Collision note
/// Two different frames with the same dimensions, identical first 64 pixels,
/// and identical last 64 pixels will hash identically.  For media analysis
/// use-cases (sub-analyzers processing the **same** frame object) this is
/// sufficient.  Do not use this as a cryptographic identifier.
#[must_use]
pub fn frame_hash_key(pixels: &[u8], width: usize, height: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    width.hash(&mut h);
    height.hash(&mut h);
    // Sample: first 64 bytes + last 64 bytes.
    let head_end = pixels.len().min(64);
    let tail_start = pixels.len().saturating_sub(64);
    pixels[..head_end].hash(&mut h);
    if tail_start > head_end {
        pixels[tail_start..].hash(&mut h);
    }
    h.finish()
}

/// A bounded LRU-style cache that memoises [`HistogramData`] per frame.
///
/// Keyed on a [`u64`] frame hash produced by [`frame_hash_key`].  When the
/// cache reaches `max_entries` the oldest entry (by insertion order) is
/// evicted before the new one is inserted.
///
/// # Example
/// ```rust
/// use oximedia_analysis::histogram_analysis::{
///     FrameHistogramCache, HistogramData, ImageHistogram, frame_hash_key,
/// };
///
/// let mut cache = FrameHistogramCache::new(8);
/// let pixels = vec![128u8; 1920 * 1080];
/// let key = frame_hash_key(&pixels, 1920, 1080);
/// let data = cache.get_or_compute(key, || {
///     HistogramData::from(ImageHistogram::from_luma(&pixels))
/// });
/// assert_eq!(data.histogram.total_pixels, 1920 * 1080);
/// ```
pub struct FrameHistogramCache {
    map: std::collections::HashMap<u64, HistogramData>,
    max_entries: usize,
    insertion_order: std::collections::VecDeque<u64>,
}

impl FrameHistogramCache {
    /// Create a new cache that retains at most `max_entries` histograms.
    ///
    /// `max_entries` must be ≥ 1; if 0 is passed it is silently raised to 1.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.max(1);
        Self {
            map: std::collections::HashMap::with_capacity(cap),
            max_entries: cap,
            insertion_order: std::collections::VecDeque::with_capacity(cap),
        }
    }

    /// Return the cached [`HistogramData`] for `frame_key`, or compute and
    /// cache it if not present.
    ///
    /// The closure `compute` is called **at most once** per unique key.
    pub fn get_or_compute<F>(&mut self, frame_key: u64, compute: F) -> &HistogramData
    where
        F: FnOnce() -> HistogramData,
    {
        // Avoid double-lookup: check presence first, then insert if missing.
        if !self.map.contains_key(&frame_key) {
            // Evict oldest entry if at capacity.
            if self.map.len() >= self.max_entries {
                if let Some(evict_key) = self.insertion_order.pop_front() {
                    self.map.remove(&evict_key);
                }
            }
            let data = compute();
            self.map.insert(frame_key, data);
            self.insertion_order.push_back(frame_key);
        }
        // SAFETY: we just inserted if missing; the key is present.
        &self.map[&frame_key]
    }

    /// Returns the number of entries currently held in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_histogram(value: u8, count: u64) -> ImageHistogram {
        let data: Vec<u8> = vec![value; count as usize];
        ImageHistogram::from_luma(&data)
    }

    #[test]
    fn test_bucket_fill_ratio_zero_total() {
        let b = HistogramBucket::new(0, 10, 0, 0);
        assert_eq!(b.fill_ratio(), 0.0);
    }

    #[test]
    fn test_bucket_fill_ratio() {
        let b = HistogramBucket::new(0, 10, 50, 100);
        assert!((b.fill_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_from_luma_single_value() {
        let h = uniform_histogram(128, 10);
        assert_eq!(h.bins[128], 10);
        assert_eq!(h.total_pixels, 10);
    }

    #[test]
    fn test_histogram_mean_uniform() {
        let h = uniform_histogram(100, 100);
        assert!((h.mean() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_std_dev_uniform() {
        // All pixels same value → std dev = 0.
        let h = uniform_histogram(128, 50);
        assert!(h.std_dev() < 1e-9);
    }

    #[test]
    fn test_histogram_std_dev_bimodal() {
        // Half black, half white.
        let mut data = vec![0u8; 50];
        data.extend(vec![255u8; 50]);
        let h = ImageHistogram::from_luma(&data);
        assert!(h.std_dev() > 100.0);
    }

    #[test]
    fn test_is_low_contrast_true() {
        let h = uniform_histogram(128, 100);
        assert!(h.is_low_contrast(20.0));
    }

    #[test]
    fn test_is_low_contrast_false() {
        let mut data = vec![0u8; 50];
        data.extend(vec![255u8; 50]);
        let h = ImageHistogram::from_luma(&data);
        assert!(!h.is_low_contrast(20.0));
    }

    #[test]
    fn test_highlight_clip_ratio_all_white() {
        let h = uniform_histogram(255, 100);
        assert!(h.highlight_clip_ratio(235) > 0.99);
    }

    #[test]
    fn test_shadow_clip_ratio_all_black() {
        let h = uniform_histogram(0, 100);
        assert!(h.shadow_clip_ratio(16) > 0.99);
    }

    #[test]
    fn test_analyzer_low_contrast_detected() {
        let analyzer = HistogramAnalyzer::new();
        let h = uniform_histogram(128, 100);
        let result = analyzer.analyze(&h);
        assert!(result.low_contrast);
    }

    #[test]
    fn test_analyzer_detect_highlight_clipping() {
        let analyzer = HistogramAnalyzer::new();
        let h = uniform_histogram(255, 100);
        let (_, highlight) = analyzer.detect_clipping(&h);
        assert!(highlight);
    }

    #[test]
    fn test_analyzer_detect_shadow_clipping() {
        let analyzer = HistogramAnalyzer::new();
        let h = uniform_histogram(0, 100);
        let (shadow, _) = analyzer.detect_clipping(&h);
        assert!(shadow);
    }

    #[test]
    fn test_default_histogram_empty() {
        let h = ImageHistogram::default();
        assert_eq!(h.total_pixels, 0);
        assert_eq!(h.mean(), 0.0);
    }

    // ── FrameHistogramCache ───────────────────────────────────

    /// The compute closure must be called exactly once on the first access and
    /// zero times on subsequent accesses with the same key.
    #[test]
    fn test_histogram_cache_hit_skips_recompute() {
        let mut cache = FrameHistogramCache::new(4);
        let pixels = vec![100u8; 64 * 64];
        let key = frame_hash_key(&pixels, 64, 64);

        let mut call_count = 0usize;

        // First call: compute closure fires.
        {
            let _data = cache.get_or_compute(key, || {
                call_count += 1;
                HistogramData::from(ImageHistogram::from_luma(&pixels))
            });
        }
        assert_eq!(call_count, 1, "compute must be called once on cache miss");

        // Second call with identical key: compute closure must NOT fire.
        {
            let _data = cache.get_or_compute(key, || {
                call_count += 1;
                HistogramData::from(ImageHistogram::from_luma(&pixels))
            });
        }
        assert_eq!(call_count, 1, "compute must be skipped on cache hit");

        // The cached entry has the correct pixel count.
        let data = cache.get_or_compute(key, || {
            call_count += 1;
            HistogramData::from(ImageHistogram::from_luma(&pixels))
        });
        assert_eq!(data.histogram.total_pixels, 64 * 64);
        assert_eq!(call_count, 1, "total compute calls must remain 1");
    }

    /// A different frame (different pixel data) must produce a different hash
    /// and therefore trigger a new compute.
    #[test]
    fn test_histogram_cache_miss_on_new_frame() {
        let mut cache = FrameHistogramCache::new(4);

        let pixels_a = vec![50u8; 64 * 64];
        let pixels_b = vec![200u8; 64 * 64];
        let key_a = frame_hash_key(&pixels_a, 64, 64);
        let key_b = frame_hash_key(&pixels_b, 64, 64);

        // The two frames must hash differently.
        assert_ne!(key_a, key_b, "distinct frames must have distinct hash keys");

        let mut calls = 0usize;

        cache.get_or_compute(key_a, || {
            calls += 1;
            HistogramData::from(ImageHistogram::from_luma(&pixels_a))
        });
        assert_eq!(calls, 1);

        // New frame → cache miss → compute fires again.
        cache.get_or_compute(key_b, || {
            calls += 1;
            HistogramData::from(ImageHistogram::from_luma(&pixels_b))
        });
        assert_eq!(calls, 2, "cache miss on new frame must trigger compute");

        // Both entries are present in the cache.
        assert_eq!(cache.len(), 2);
    }
}
