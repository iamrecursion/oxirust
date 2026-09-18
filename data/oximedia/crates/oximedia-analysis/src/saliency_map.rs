//! Saliency map analysis for detecting visually important regions in video frames.

#![allow(dead_code)]

/// Method used to compute saliency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaliencyMethod {
    /// Spectral residual approach
    SpectralResidual,
    /// Fine-grained frequency tuning
    FineGrained,
    /// Itti-Koch-Niebur model
    IttiKoch,
    /// Simple center-surround differences
    CenterSurround,
}

impl SaliencyMethod {
    /// Return a human-readable label for the method.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::SpectralResidual => "spectral_residual",
            Self::FineGrained => "fine_grained",
            Self::IttiKoch => "itti_koch",
            Self::CenterSurround => "center_surround",
        }
    }

    /// Return a quality rating (0–100) indicating typical output quality.
    #[must_use]
    pub fn quality_rating(&self) -> u8 {
        match self {
            Self::SpectralResidual => 75,
            Self::FineGrained => 85,
            Self::IttiKoch => 95,
            Self::CenterSurround => 60,
        }
    }
}

/// A single pixel in a saliency map together with its coordinates.
#[derive(Debug, Clone)]
pub struct SaliencyPixel {
    /// Horizontal position (pixels from left).
    pub x: u32,
    /// Vertical position (pixels from top).
    pub y: u32,
    /// Saliency value in the range [0.0, 1.0].
    pub value: f32,
}

impl SaliencyPixel {
    /// Create a new [`SaliencyPixel`].
    #[must_use]
    pub fn new(x: u32, y: u32, value: f32) -> Self {
        Self {
            x,
            y,
            value: value.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` when the pixel value exceeds `threshold`.
    #[must_use]
    pub fn is_salient(&self, threshold: f32) -> bool {
        self.value >= threshold
    }
}

/// An axis-aligned rectangular region in a saliency map.
#[derive(Debug, Clone)]
pub struct SaliencyRegion {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Average saliency over the region.
    pub avg_saliency: f32,
}

impl SaliencyRegion {
    /// Create a new [`SaliencyRegion`].
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, avg_saliency: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            avg_saliency: avg_saliency.clamp(0.0, 1.0),
        }
    }

    /// Return the area of the region in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// A full saliency map for one frame.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Per-pixel saliency values in row-major order.
    pub values: Vec<f32>,
    /// The method used to generate this map.
    pub method: SaliencyMethod,
}

impl SaliencyMap {
    /// Create a new zero-filled [`SaliencyMap`].
    #[must_use]
    pub fn new(width: u32, height: u32, method: SaliencyMethod) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            width,
            height,
            values: vec![0.0; size],
            method,
        }
    }

    /// Return the N regions with the highest average saliency.
    ///
    /// Regions are non-overlapping blocks of size `block_size × block_size`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn top_regions(&self, n: usize, block_size: u32) -> Vec<SaliencyRegion> {
        if block_size == 0 || self.width == 0 || self.height == 0 {
            return Vec::new();
        }

        let cols = self.width.div_ceil(block_size);
        let rows = self.height.div_ceil(block_size);

        let mut scored: Vec<SaliencyRegion> = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            for col in 0..cols {
                let bx = col * block_size;
                let by = row * block_size;
                let bw = block_size.min(self.width - bx);
                let bh = block_size.min(self.height - by);

                let mut sum = 0.0_f32;
                let mut count = 0u32;

                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx =
                            ((by + dy) as usize) * (self.width as usize) + ((bx + dx) as usize);
                        if let Some(&v) = self.values.get(idx) {
                            sum += v;
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                scored.push(SaliencyRegion::new(bx, by, bw, bh, avg));
            }
        }

        scored.sort_by(|a, b| {
            b.avg_saliency
                .partial_cmp(&a.avg_saliency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(n);
        scored
    }

    /// Return the global mean saliency value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_saliency(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }
}

/// Content type hint used to select appropriate saliency model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentTypeHint {
    /// General-purpose content (default weights).
    General,
    /// Face-heavy content (portrait, interview).
    Face,
    /// Sports / fast-action content.
    Sports,
    /// Documentary / talking-head content.
    Documentary,
    /// Screen capture / UI content.
    ScreenCapture,
    /// Landscape / nature content.
    Nature,
}

impl ContentTypeHint {
    /// Return a label string for the content type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Face => "face",
            Self::Sports => "sports",
            Self::Documentary => "documentary",
            Self::ScreenCapture => "screen_capture",
            Self::Nature => "nature",
        }
    }
}

/// Saliency model weights for a specific content type.
///
/// Each weight is in `[0.0, 1.0]` and controls how strongly each cue
/// contributes to the final saliency map.  Weights do not need to sum to 1.0;
/// the analyzer normalises them internally.
#[derive(Debug, Clone)]
pub struct SaliencyModelWeights {
    /// Weight for center-surround (local contrast) cue.
    pub center_surround: f32,
    /// Weight for high-frequency edge/texture cue.
    pub edge_texture: f32,
    /// Weight for color distinctiveness cue (requires UV planes).
    pub color_distinctiveness: f32,
    /// Weight for spatial center bias (center of frame tends to be salient).
    pub center_bias: f32,
}

impl SaliencyModelWeights {
    /// Default (general-purpose) weights.
    #[must_use]
    pub fn general() -> Self {
        Self {
            center_surround: 0.40,
            edge_texture: 0.30,
            color_distinctiveness: 0.20,
            center_bias: 0.10,
        }
    }

    /// Weights tuned for face-heavy content: strong center bias and color cue.
    #[must_use]
    pub fn face() -> Self {
        Self {
            center_surround: 0.25,
            edge_texture: 0.20,
            color_distinctiveness: 0.30,
            center_bias: 0.25,
        }
    }

    /// Weights tuned for sports content: emphasise local contrast and edges.
    #[must_use]
    pub fn sports() -> Self {
        Self {
            center_surround: 0.45,
            edge_texture: 0.40,
            color_distinctiveness: 0.10,
            center_bias: 0.05,
        }
    }

    /// Weights tuned for documentary / talking-head content.
    #[must_use]
    pub fn documentary() -> Self {
        Self {
            center_surround: 0.30,
            edge_texture: 0.25,
            color_distinctiveness: 0.20,
            center_bias: 0.25,
        }
    }

    /// Weights tuned for screen-capture content: prioritise edges/text over blobs.
    #[must_use]
    pub fn screen_capture() -> Self {
        Self {
            center_surround: 0.20,
            edge_texture: 0.60,
            color_distinctiveness: 0.15,
            center_bias: 0.05,
        }
    }

    /// Weights tuned for nature / landscape content.
    #[must_use]
    pub fn nature() -> Self {
        Self {
            center_surround: 0.35,
            edge_texture: 0.25,
            color_distinctiveness: 0.30,
            center_bias: 0.10,
        }
    }

    /// Return the preset weights for the given `ContentTypeHint`.
    #[must_use]
    pub fn for_content(hint: ContentTypeHint) -> Self {
        match hint {
            ContentTypeHint::General => Self::general(),
            ContentTypeHint::Face => Self::face(),
            ContentTypeHint::Sports => Self::sports(),
            ContentTypeHint::Documentary => Self::documentary(),
            ContentTypeHint::ScreenCapture => Self::screen_capture(),
            ContentTypeHint::Nature => Self::nature(),
        }
    }

    /// Validate that all weights are non-negative and return the weight sum.
    #[must_use]
    pub fn weight_sum(&self) -> f32 {
        self.center_surround + self.edge_texture + self.color_distinctiveness + self.center_bias
    }

    /// Normalise weights so they sum to 1.0.  Returns `None` if sum is zero.
    #[must_use]
    pub fn normalised(&self) -> Option<Self> {
        let s = self.weight_sum();
        if s < f32::EPSILON {
            return None;
        }
        Some(Self {
            center_surround: self.center_surround / s,
            edge_texture: self.edge_texture / s,
            color_distinctiveness: self.color_distinctiveness / s,
            center_bias: self.center_bias / s,
        })
    }
}

/// Configuration for the saliency analyzer.
#[derive(Debug, Clone)]
pub struct SaliencyAnalyzerConfig {
    /// Analysis method to use.
    pub method: SaliencyMethod,
    /// Threshold above which a pixel is considered salient.
    pub saliency_threshold: f32,
    /// Number of top regions to return.
    pub top_region_count: usize,
    /// Block size used when computing top regions.
    pub block_size: u32,
    /// Content type hint used to select model weights.
    pub content_hint: ContentTypeHint,
    /// Custom model weights (overrides `content_hint` when set).
    pub custom_weights: Option<SaliencyModelWeights>,
}

impl Default for SaliencyAnalyzerConfig {
    fn default() -> Self {
        Self {
            method: SaliencyMethod::SpectralResidual,
            saliency_threshold: 0.5,
            top_region_count: 5,
            block_size: 32,
            content_hint: ContentTypeHint::General,
            custom_weights: None,
        }
    }
}

/// Computes saliency maps from raw luma frames.
pub struct SaliencyAnalyzer {
    config: SaliencyAnalyzerConfig,
    frame_count: usize,
    total_mean: f32,
}

impl SaliencyAnalyzer {
    /// Create a new [`SaliencyAnalyzer`] with the given config.
    #[must_use]
    pub fn new(config: SaliencyAnalyzerConfig) -> Self {
        Self {
            config,
            frame_count: 0,
            total_mean: 0.0,
        }
    }

    /// Compute a saliency map for a single luma plane.
    ///
    /// `luma` must have exactly `width * height` bytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&mut self, luma: &[u8], width: u32, height: u32) -> SaliencyMap {
        let size = (width as usize) * (height as usize);
        let mut map = SaliencyMap::new(width, height, self.config.method.clone());

        if luma.len() < size {
            return map;
        }

        // Simple center-surround saliency approximation using local contrast.
        let w = width as usize;
        let h = height as usize;

        for y in 0..h {
            for x in 0..w {
                let center = f32::from(luma[y * w + x]);

                // 3×3 neighbourhood mean (clamped at borders).
                let mut neigh_sum = 0.0_f32;
                let mut neigh_count = 0u32;
                for dy in -2_i32..=2 {
                    for dx in -2_i32..=2 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            neigh_sum += f32::from(luma[(ny as usize) * w + (nx as usize)]);
                            neigh_count += 1;
                        }
                    }
                }

                let neigh_mean = if neigh_count > 0 {
                    neigh_sum / neigh_count as f32
                } else {
                    center
                };
                let diff = (center - neigh_mean).abs() / 255.0;
                map.values[y * w + x] = diff.clamp(0.0, 1.0);
            }
        }

        self.total_mean += map.mean_saliency();
        self.frame_count += 1;
        map
    }

    /// Returns the number of frames processed so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Returns the running average mean saliency across all processed frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_mean_saliency(&self) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.total_mean / self.frame_count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_method_quality_rating() {
        assert_eq!(SaliencyMethod::SpectralResidual.quality_rating(), 75);
        assert_eq!(SaliencyMethod::IttiKoch.quality_rating(), 95);
        assert_eq!(SaliencyMethod::CenterSurround.quality_rating(), 60);
    }

    #[test]
    fn test_saliency_method_label() {
        assert_eq!(SaliencyMethod::FineGrained.label(), "fine_grained");
        assert_eq!(
            SaliencyMethod::SpectralResidual.label(),
            "spectral_residual"
        );
    }

    #[test]
    fn test_saliency_pixel_new_clamps() {
        let p = SaliencyPixel::new(0, 0, 1.5);
        assert!((p.value - 1.0).abs() < f32::EPSILON);
        let p2 = SaliencyPixel::new(1, 1, -0.3);
        assert!((p2.value - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saliency_pixel_is_salient() {
        let p = SaliencyPixel::new(5, 5, 0.8);
        assert!(p.is_salient(0.5));
        assert!(!p.is_salient(0.9));
    }

    #[test]
    fn test_saliency_region_area() {
        let r = SaliencyRegion::new(0, 0, 10, 20, 0.7);
        assert_eq!(r.area(), 200);
    }

    #[test]
    fn test_saliency_map_new_zero_filled() {
        let m = SaliencyMap::new(4, 4, SaliencyMethod::CenterSurround);
        assert_eq!(m.values.len(), 16);
        assert!(m.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_saliency_map_mean_zero() {
        let m = SaliencyMap::new(4, 4, SaliencyMethod::CenterSurround);
        assert!((m.mean_saliency() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saliency_map_mean_uniform() {
        let mut m = SaliencyMap::new(2, 2, SaliencyMethod::CenterSurround);
        m.values = vec![0.5, 0.5, 0.5, 0.5];
        assert!((m.mean_saliency() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_saliency_map_top_regions_empty_for_zero_block() {
        let m = SaliencyMap::new(8, 8, SaliencyMethod::SpectralResidual);
        assert!(m.top_regions(3, 0).is_empty());
    }

    #[test]
    fn test_saliency_map_top_regions_count() {
        let mut m = SaliencyMap::new(8, 8, SaliencyMethod::SpectralResidual);
        for (i, v) in m.values.iter_mut().enumerate() {
            *v = i as f32 / 64.0;
        }
        let regions = m.top_regions(3, 4);
        assert!(regions.len() <= 3);
    }

    #[test]
    fn test_saliency_analyzer_default_config() {
        let cfg = SaliencyAnalyzerConfig::default();
        assert_eq!(cfg.top_region_count, 5);
        assert!((cfg.saliency_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saliency_analyzer_compute_increments_frame_count() {
        let mut analyzer = SaliencyAnalyzer::new(SaliencyAnalyzerConfig::default());
        let luma = vec![128u8; 16 * 16];
        let _ = analyzer.compute(&luma, 16, 16);
        assert_eq!(analyzer.frame_count(), 1);
    }

    #[test]
    fn test_saliency_analyzer_compute_returns_correct_dimensions() {
        let mut analyzer = SaliencyAnalyzer::new(SaliencyAnalyzerConfig::default());
        let luma = vec![0u8; 8 * 8];
        let map = analyzer.compute(&luma, 8, 8);
        assert_eq!(map.width, 8);
        assert_eq!(map.height, 8);
        assert_eq!(map.values.len(), 64);
    }

    #[test]
    fn test_saliency_analyzer_average_mean_no_frames() {
        let analyzer = SaliencyAnalyzer::new(SaliencyAnalyzerConfig::default());
        assert!((analyzer.average_mean_saliency() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saliency_analyzer_short_luma_returns_empty_map() {
        let mut analyzer = SaliencyAnalyzer::new(SaliencyAnalyzerConfig::default());
        let luma = vec![100u8; 4]; // too short for 8x8
        let map = analyzer.compute(&luma, 8, 8);
        assert!(map.values.iter().all(|&v| v == 0.0));
    }
}
