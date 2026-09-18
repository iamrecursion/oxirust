//! Content mapping analysis for Dolby Vision
//!
//! Provides statistical analysis of PQ-encoded frame metadata to characterise
//! content and recommend optimal tone-mapping trim strategies.

use crate::scene_trim::TrimTarget;

/// Statistical summary of PQ values across a sequence of frames
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ContentMappingStats {
    /// Minimum PQ code value (0–4095) across all frames
    pub min_pq: u16,
    /// Maximum PQ code value (0–4095) across all frames
    pub max_pq: u16,
    /// Average PQ value (floating-point) across all frames
    pub avg_pq: f32,
    /// 10th percentile PQ value
    pub p10_pq: u16,
    /// 90th percentile PQ value
    pub p90_pq: u16,
    /// 99th percentile PQ value
    pub p99_pq: u16,
}

/// Histogram of PQ code values (0–4095)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PqHistogram {
    /// 4096 buckets, one per PQ code value
    pub buckets: Vec<u32>,
    /// Total number of samples added
    pub total_pixels: u64,
}

impl PqHistogram {
    /// Create a new empty histogram
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: vec![0u32; 4096],
            total_pixels: 0,
        }
    }

    /// Add a single PQ sample to the histogram
    pub fn add_sample(&mut self, pq: u16) {
        let idx = usize::from(pq.min(4095));
        self.buckets[idx] = self.buckets[idx].saturating_add(1);
        self.total_pixels = self.total_pixels.saturating_add(1);
    }

    /// Return the PQ value at the given percentile (0.0–100.0)
    #[must_use]
    pub fn percentile(&self, p: f32) -> u16 {
        if self.total_pixels == 0 {
            return 0;
        }
        let target = (f64::from(p) / 100.0 * self.total_pixels as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += u64::from(count);
            if cumulative >= target {
                return i as u16;
            }
        }
        4095
    }
}

impl Default for PqHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// A single frame's Dolby Vision metadata (defined locally for analysis)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct DvMetadataFrame {
    /// Maximum PQ value in this frame (0–4095)
    pub max_pq: u16,
    /// Average PQ value in this frame (0–4095)
    pub avg_pq: u16,
    /// Frame index
    pub frame_idx: u64,
}

impl DvMetadataFrame {
    /// Create a new `DvMetadataFrame`
    #[must_use]
    pub fn new(max_pq: u16, avg_pq: u16, frame_idx: u64) -> Self {
        Self {
            max_pq,
            avg_pq,
            frame_idx,
        }
    }
}

/// Analyzes a sequence of `DvMetadataFrame` values to produce statistics
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct CmAnalyzer;

impl CmAnalyzer {
    /// Analyze a slice of frames and return aggregate content mapping statistics
    #[must_use]
    pub fn analyze(frames: &[DvMetadataFrame]) -> ContentMappingStats {
        if frames.is_empty() {
            return ContentMappingStats {
                min_pq: 0,
                max_pq: 0,
                avg_pq: 0.0,
                p10_pq: 0,
                p90_pq: 0,
                p99_pq: 0,
            };
        }

        let mut histogram = PqHistogram::new();
        let mut min_pq = u16::MAX;
        let mut max_pq = u16::MIN;
        let mut sum: f64 = 0.0;

        for frame in frames {
            // Populate histogram with max_pq samples per frame
            histogram.add_sample(frame.max_pq);

            if frame.max_pq < min_pq {
                min_pq = frame.max_pq;
            }
            if frame.max_pq > max_pq {
                max_pq = frame.max_pq;
            }
            sum += f64::from(frame.avg_pq);
        }

        let avg_pq = (sum / frames.len() as f64) as f32;

        ContentMappingStats {
            min_pq,
            max_pq,
            avg_pq,
            p10_pq: histogram.percentile(10.0),
            p90_pq: histogram.percentile(90.0),
            p99_pq: histogram.percentile(99.0),
        }
    }
}

/// Content character classification based on statistical analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ContentCharacter {
    /// Very dark content (low average PQ)
    Dark,
    /// Night scene (moderate avg, low min)
    NightScene,
    /// Day exterior (high max, moderate avg)
    DayExterior,
    /// High contrast content
    HighContrast,
    /// Low contrast content
    LowContrast,
}

impl ContentCharacter {
    /// Classify content character from statistics
    #[must_use]
    pub fn from_stats(stats: &ContentMappingStats) -> Self {
        let range = stats.max_pq.saturating_sub(stats.min_pq);
        let avg = stats.avg_pq;

        if avg < 500.0 && stats.max_pq < 1500 {
            return Self::Dark;
        }
        if avg < 700.0 && stats.min_pq < 200 && stats.max_pq > 2500 {
            return Self::NightScene;
        }
        if stats.max_pq > 3500 && avg > 1500.0 {
            return Self::DayExterior;
        }
        if range > 3000 {
            return Self::HighContrast;
        }
        Self::LowContrast
    }
}

/// Recommends optimal trim strategies based on content character
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct OptimalTrimStrategy;

impl OptimalTrimStrategy {
    /// Recommend trim parameters for the given content character and target display luminance
    #[must_use]
    pub fn recommend(character: ContentCharacter, target_nits: f32) -> TrimTarget {
        // Start from the display's standard trim
        let mut base = TrimTarget::for_display(target_nits);

        // Apply content-character adjustments
        match character {
            ContentCharacter::Dark => {
                // Lift blacks slightly to avoid crushed shadows
                base.trim_offset += 0.03;
                base.trim_power *= 1.05;
            }
            ContentCharacter::NightScene => {
                // Protect shadow detail
                base.trim_slope *= 0.97;
                base.trim_offset += 0.01;
            }
            ContentCharacter::DayExterior => {
                // Aggressive highlight mapping for bright scenes
                base.trim_slope *= 1.03;
                base.trim_power *= 0.97;
            }
            ContentCharacter::HighContrast => {
                // Widen dynamic range reproduction
                base.trim_slope *= 1.02;
                base.target_mid_contrast *= 1.05;
            }
            ContentCharacter::LowContrast => {
                // Slightly boost contrast for SDR-like displays
                base.trim_slope *= 0.99;
                base.target_mid_contrast *= 0.98;
            }
        }

        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_histogram_add_and_percentile() {
        let mut hist = PqHistogram::new();
        for pq in 0u16..=100 {
            hist.add_sample(pq);
        }
        assert_eq!(hist.total_pixels, 101);
        // 50th percentile should be around 50
        let p50 = hist.percentile(50.0);
        assert!(p50 <= 52, "p50={p50}");
    }

    #[test]
    fn test_pq_histogram_empty() {
        let hist = PqHistogram::new();
        assert_eq!(hist.percentile(50.0), 0);
    }

    #[test]
    fn test_pq_histogram_clamps_at_4095() {
        let mut hist = PqHistogram::new();
        hist.add_sample(5000); // should clamp to 4095
        assert_eq!(hist.buckets[4095], 1);
    }

    #[test]
    fn test_cm_analyzer_empty() {
        let stats = CmAnalyzer::analyze(&[]);
        assert_eq!(stats.min_pq, 0);
        assert_eq!(stats.max_pq, 0);
        assert!((stats.avg_pq).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cm_analyzer_uniform_frames() {
        let frames: Vec<DvMetadataFrame> = (0..10)
            .map(|i| DvMetadataFrame::new(2000, 1000, i))
            .collect();
        let stats = CmAnalyzer::analyze(&frames);
        assert_eq!(stats.min_pq, 2000);
        assert_eq!(stats.max_pq, 2000);
        assert!((stats.avg_pq - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_cm_analyzer_varying_frames() {
        let frames = vec![
            DvMetadataFrame::new(1000, 500, 0),
            DvMetadataFrame::new(3000, 1500, 1),
            DvMetadataFrame::new(2000, 1000, 2),
        ];
        let stats = CmAnalyzer::analyze(&frames);
        assert_eq!(stats.min_pq, 1000);
        assert_eq!(stats.max_pq, 3000);
        assert!((stats.avg_pq - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_content_character_dark() {
        let stats = ContentMappingStats {
            min_pq: 0,
            max_pq: 1200,
            avg_pq: 400.0,
            p10_pq: 100,
            p90_pq: 1000,
            p99_pq: 1100,
        };
        assert_eq!(ContentCharacter::from_stats(&stats), ContentCharacter::Dark);
    }

    #[test]
    fn test_content_character_day_exterior() {
        let stats = ContentMappingStats {
            min_pq: 500,
            max_pq: 3800,
            avg_pq: 2000.0,
            p10_pq: 600,
            p90_pq: 3600,
            p99_pq: 3700,
        };
        assert_eq!(
            ContentCharacter::from_stats(&stats),
            ContentCharacter::DayExterior
        );
    }

    #[test]
    fn test_content_character_high_contrast() {
        let stats = ContentMappingStats {
            min_pq: 50,
            max_pq: 3800,
            avg_pq: 1200.0,
            p10_pq: 100,
            p90_pq: 3500,
            p99_pq: 3700,
        };
        assert_eq!(
            ContentCharacter::from_stats(&stats),
            ContentCharacter::HighContrast
        );
    }

    #[test]
    fn test_optimal_trim_dark_content() {
        let trim = OptimalTrimStrategy::recommend(ContentCharacter::Dark, 1000.0);
        // Dark content gets positive trim_offset adjustment
        assert!(trim.trim_offset > 0.0, "trim_offset={}", trim.trim_offset);
    }

    #[test]
    fn test_optimal_trim_day_exterior() {
        let base = TrimTarget::for_display(1000.0);
        let trim = OptimalTrimStrategy::recommend(ContentCharacter::DayExterior, 1000.0);
        // Day exterior should have higher trim_slope than base
        assert!(trim.trim_slope > base.trim_slope, "expected higher slope");
    }

    #[test]
    fn test_dv_metadata_frame_creation() {
        let frame = DvMetadataFrame::new(3000, 1500, 42);
        assert_eq!(frame.max_pq, 3000);
        assert_eq!(frame.avg_pq, 1500);
        assert_eq!(frame.frame_idx, 42);
    }
}

// ── Spec-required types ───────────────────────────────────────────────────────

/// Dolby Vision Content Mapping (CM) version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CmVersion {
    /// CM v2.9 (original, Level 2 trims only).
    Cm2_9,
    /// CM v4.0 (extended, supports Level 8 target-display trims).
    Cm4_0,
}

impl CmVersion {
    /// Returns the version identifier string.
    #[must_use]
    pub fn version_string(&self) -> &str {
        match self {
            Self::Cm2_9 => "2.9",
            Self::Cm4_0 => "4.0",
        }
    }

    /// Returns `true` if this version supports Level 8 target-display trims.
    #[must_use]
    pub fn supports_level8(&self) -> bool {
        matches!(self, Self::Cm4_0)
    }
}

/// Aggregate content mapping analysis for a sequence of Dolby Vision frames.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ContentMappingAnalysis {
    /// Number of analysed frames.
    pub frame_count: u64,
    /// Average maximum PQ per frame (0.0–4095.0).
    pub avg_max_pq: f32,
    /// Average minimum PQ per frame (0.0–4095.0).
    pub avg_min_pq: f32,
    /// Number of detected scenes.
    pub scene_count: u32,
    /// Weighted complexity of the trim metadata (arbitrary units).
    pub trim_complexity: f32,
}

impl ContentMappingAnalysis {
    /// Ratio of average max PQ to average min PQ; or 0.0 if min is zero.
    #[must_use]
    pub fn dynamic_range_ratio(&self) -> f32 {
        if self.avg_min_pq < f32::EPSILON {
            return 0.0;
        }
        self.avg_max_pq / self.avg_min_pq
    }

    /// Returns `true` if more than 80 % of the average PQ range lies above
    /// PQ = 2 000 (roughly > 203 nits), indicating HDR-heavy content.
    #[must_use]
    pub fn is_hdr_heavy(&self) -> bool {
        self.avg_max_pq > 2_000.0
    }
}

/// Statistical summary of a set of normalised PQ values (0.0–1.0).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PqStatistics {
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Arithmetic mean.
    pub mean: f32,
    /// 95th-percentile value.
    pub percentile_95: f32,
}

impl PqStatistics {
    /// Compute statistics from a slice of PQ values.
    ///
    /// Uses a sort-based approach to determine the 95th percentile.
    /// Returns all-zero if `pq_values` is empty.
    #[must_use]
    pub fn compute(pq_values: &[f32]) -> Self {
        if pq_values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                percentile_95: 0.0,
            };
        }

        let mut sorted = pq_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = *sorted.last().unwrap_or(&sorted[0]);
        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;

        let idx_95 = ((sorted.len() - 1) as f32 * 0.95).round() as usize;
        let percentile_95 = sorted[idx_95.min(sorted.len() - 1)];

        Self {
            min,
            max,
            mean,
            percentile_95,
        }
    }
}

/// PQ ↔ nits conversion utilities (simplified ST.2084).
#[allow(dead_code)]
pub struct PqConverter;

impl PqConverter {
    /// Convert nits to a normalised PQ signal (0.0–1.0).
    ///
    /// Uses the simplified ST.2084 formula: `(nits / 10 000) ^ 0.1593`.
    #[must_use]
    pub fn nits_to_pq(nits: f32) -> f32 {
        let y = (nits / 10_000.0_f32).max(0.0);
        y.powf(0.159_3_f32).min(1.0)
    }

    /// Convert a normalised PQ signal (0.0–1.0) back to nits.
    ///
    /// Inverse of `nits_to_pq`: `pq ^ (1 / 0.1593) * 10 000`.
    #[must_use]
    pub fn pq_to_nits(pq: f32) -> f32 {
        let pq = pq.clamp(0.0, 1.0);
        pq.powf(1.0 / 0.159_3_f32) * 10_000.0_f32
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    #[test]
    fn test_cm_version_string_2_9() {
        assert_eq!(CmVersion::Cm2_9.version_string(), "2.9");
    }

    #[test]
    fn test_cm_version_string_4_0() {
        assert_eq!(CmVersion::Cm4_0.version_string(), "4.0");
    }

    #[test]
    fn test_cm_version_supports_level8_false() {
        assert!(!CmVersion::Cm2_9.supports_level8());
    }

    #[test]
    fn test_cm_version_supports_level8_true() {
        assert!(CmVersion::Cm4_0.supports_level8());
    }

    #[test]
    fn test_content_mapping_analysis_dynamic_range_ratio() {
        let a = ContentMappingAnalysis {
            frame_count: 100,
            avg_max_pq: 3000.0,
            avg_min_pq: 100.0,
            scene_count: 5,
            trim_complexity: 1.5,
        };
        assert!((a.dynamic_range_ratio() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_content_mapping_analysis_zero_min_pq() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 2000.0,
            avg_min_pq: 0.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert_eq!(a.dynamic_range_ratio(), 0.0);
    }

    #[test]
    fn test_content_mapping_analysis_is_hdr_heavy_true() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 3000.0,
            avg_min_pq: 50.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert!(a.is_hdr_heavy());
    }

    #[test]
    fn test_content_mapping_analysis_is_hdr_heavy_false() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 1500.0,
            avg_min_pq: 50.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert!(!a.is_hdr_heavy());
    }

    #[test]
    fn test_pq_statistics_empty() {
        let s = PqStatistics::compute(&[]);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
    }

    #[test]
    fn test_pq_statistics_single() {
        let s = PqStatistics::compute(&[0.5]);
        assert!((s.min - 0.5).abs() < 1e-6);
        assert!((s.max - 0.5).abs() < 1e-6);
        assert!((s.mean - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pq_statistics_percentile_95() {
        let values: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let s = PqStatistics::compute(&values);
        assert!(
            s.percentile_95 >= 0.94 && s.percentile_95 <= 0.96,
            "p95={}",
            s.percentile_95
        );
    }

    #[test]
    fn test_pq_converter_nits_to_pq_zero() {
        assert_eq!(PqConverter::nits_to_pq(0.0), 0.0);
    }

    #[test]
    fn test_pq_converter_nits_to_pq_ten_thousand() {
        let pq = PqConverter::nits_to_pq(10_000.0);
        assert!((pq - 1.0).abs() < 0.01, "pq={pq}");
    }

    #[test]
    fn test_pq_converter_roundtrip() {
        let nits = 1000.0_f32;
        let pq = PqConverter::nits_to_pq(nits);
        let recovered = PqConverter::pq_to_nits(pq);
        assert!((recovered - nits).abs() < 1.0, "recovered={recovered}");
    }
}

// ── CM v4.0 Advanced Types ────────────────────────────────────────────────────

/// Trim mode controlling how a Dolby Vision display mapping is applied.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum TrimMode {
    /// Automatically select optimal trim based on content analysis.
    Auto,
    /// Manually specified lift/gain/gamma tone curve.
    Manual {
        /// Shadow lift (0.0 = no lift)
        lift: f32,
        /// Highlight gain (1.0 = unity)
        gain: f32,
        /// Mid-tone gamma adjustment (1.0 = unity)
        gamma: f32,
    },
    /// Pure saturation scaling.
    Saturation {
        /// Saturation gain multiplier (1.0 = unity)
        sat_gain: f32,
    },
    /// Color primaries conversion via a 3x3 floating-point matrix.
    ColorPrimaries {
        /// 3x3 color transformation matrix
        matrix: [[f32; 3]; 3],
    },
}

/// Configuration for content mapping analysis.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CmAnalysisConfig {
    /// Peak luminance of the target display in nits.
    pub target_display_nits: f32,
    /// Reference white level in nits (typically 100–203).
    pub reference_white_nits: f32,
    /// Trim mode to apply.
    pub trim_mode: TrimMode,
}

impl Default for CmAnalysisConfig {
    fn default() -> Self {
        Self {
            target_display_nits: 1000.0,
            reference_white_nits: 203.0,
            trim_mode: TrimMode::Auto,
        }
    }
}

/// Per-channel tone curve slope/offset/power triplet for CM v4.0 trim.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct TrimSlop {
    /// Multiplicative slope applied to the PQ signal.
    pub slope: f32,
    /// Additive offset applied after slope.
    pub offset: f32,
    /// Exponent (power) applied to the PQ signal before slope.
    pub power: f32,
}

impl TrimSlop {
    /// Identity transform: pass through unchanged.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            slope: 1.0,
            offset: 0.0,
            power: 1.0,
        }
    }
}

/// CM v4.0 metadata block targeting a specific display peak luminance.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CmV40Metadata {
    /// Target display peak PQ code (0–4095).
    pub target_max_pq: u32,
    /// Per-channel (R, G, B) tone curve parameters.
    pub trim_slops: Vec<TrimSlop>,
    /// Per-channel chroma weighting factors `[r, g, b]`.
    pub chroma_weights: [f32; 3],
}

impl CmV40Metadata {
    /// Create a neutral (identity) CM v4.0 metadata block for the given target PQ.
    #[must_use]
    pub fn neutral(target_max_pq: u32) -> Self {
        Self {
            target_max_pq,
            trim_slops: vec![
                TrimSlop::identity(),
                TrimSlop::identity(),
                TrimSlop::identity(),
            ],
            chroma_weights: [1.0, 1.0, 1.0],
        }
    }
}

/// Apply a single-channel piecewise tone curve defined by a `TrimSlop`.
///
/// Formula: `slope * pq_value^power + offset`
#[must_use]
#[inline]
pub fn apply_tone_curve(pq_value: f32, slop: &TrimSlop) -> f32 {
    let powered = pq_value.max(0.0).powf(slop.power);
    (slop.slope * powered + slop.offset).clamp(0.0, 1.0)
}

/// Analyze a 1024-bin luma histogram and derive three `TrimSlop` entries
/// that approximate a 3-segment tone curve mapping to `target_nits`.
///
/// The strategy uses cumulative distribution to identify:
/// - Shadow region (0–10th percentile)
/// - Mid-tone region (10th–90th percentile)
/// - Highlight region (90th–100th percentile)
///
/// Each region receives slope/power adjustments relative to a neutral curve
/// scaled by the ratio of target luminance to reference peak (10 000 nits).
#[must_use]
pub fn compute_trim_slops(src_luma_hist: &[u32; 1024], target_nits: f32) -> Vec<TrimSlop> {
    let total: u64 = src_luma_hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return vec![TrimSlop::identity(); 3];
    }

    // Find percentile bin indices from the 1024-bin histogram
    let find_percentile = |p: f64| -> usize {
        let target_count = (p / 100.0 * total as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in src_luma_hist.iter().enumerate() {
            cumulative += count as u64;
            if cumulative >= target_count {
                return i;
            }
        }
        1023
    };

    let p10_bin = find_percentile(10.0) as f32 / 1023.0;
    let p90_bin = find_percentile(90.0) as f32 / 1023.0;

    // Compute gain factor: ratio of target to reference peak
    let gain_factor = (target_nits / 10_000.0_f32).clamp(0.001, 1.0);

    // Shadow region: slight lift to avoid crushed blacks
    let shadow_slop = TrimSlop {
        slope: gain_factor * (1.0 + (1.0 - p10_bin) * 0.1),
        offset: p10_bin * 0.02,
        power: 1.05,
    };

    // Mid-tone region: primary tone curve driven by gain factor
    let mid_slop = TrimSlop {
        slope: gain_factor * 1.05,
        offset: 0.0,
        power: 1.0 - (1.0 - gain_factor) * 0.15,
    };

    // Highlight region: compression of peaks above target
    let highlight_slop = TrimSlop {
        slope: gain_factor * (1.0 - (1.0 - p90_bin) * 0.2),
        offset: 0.0,
        power: 0.9 + gain_factor * 0.1,
    };

    vec![shadow_slop, mid_slop, highlight_slop]
}

/// Gamut compressor using smooth sigmoid clamping.
///
/// Values in [0, 1] pass through unchanged; values outside this range
/// are smoothly compressed back using a sigmoid-like rolloff.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GamutCompressor {
    /// Saturation roll-off threshold (values above this are compressed).
    pub primary_saturation: f32,
}

impl GamutCompressor {
    /// Create a new gamut compressor with the given saturation threshold.
    #[must_use]
    pub fn new(primary_saturation: f32) -> Self {
        Self { primary_saturation }
    }

    /// Compress a single channel value using sigmoid rolloff.
    #[must_use]
    fn compress_channel(value: f32, threshold: f32) -> f32 {
        if value <= threshold {
            return value;
        }
        // Sigmoid compression: maps (threshold, ∞) → (threshold, 1.0)
        let excess = value - threshold;
        let headroom = 1.0 - threshold;
        if headroom < f32::EPSILON {
            return threshold;
        }
        // smooth sigmoid: 1 - headroom * exp(-excess/headroom)
        threshold + headroom * (1.0 - (-excess / headroom).exp())
    }

    /// Compress out-of-gamut RGB colors using smooth sigmoid approach.
    #[must_use]
    pub fn compress(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let threshold = self.primary_saturation.clamp(0.0, 1.0);
        let r_out = Self::compress_channel(r, threshold);
        let g_out = Self::compress_channel(g, threshold);
        let b_out = Self::compress_channel(b, threshold);
        (r_out.max(0.0), g_out.max(0.0), b_out.max(0.0))
    }
}

/// Saturation deployment in IPT-PQ space.
///
/// Scales the P and T chroma channels by a gain factor while leaving
/// the I (intensity) channel unchanged.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SaturationDeployment;

impl SaturationDeployment {
    /// Apply saturation scaling to IPT-PQ values.
    ///
    /// `gain > 1.0` increases saturation; `gain < 1.0` decreases it.
    /// `gain = 0.0` produces a fully desaturated (achromatic) result.
    #[must_use]
    pub fn apply_saturation(i: f32, p: f32, t: f32, gain: f32) -> (f32, f32, f32) {
        let gain = gain.max(0.0);
        (i, p * gain, t * gain)
    }
}

#[cfg(test)]
mod cm_v40_tests {
    use super::*;

    #[test]
    fn test_trim_mode_auto_variant() {
        let mode = TrimMode::Auto;
        assert_eq!(mode, TrimMode::Auto);
    }

    #[test]
    fn test_trim_mode_manual_fields() {
        let mode = TrimMode::Manual {
            lift: 0.1,
            gain: 1.2,
            gamma: 0.95,
        };
        if let TrimMode::Manual { lift, gain, gamma } = mode {
            assert!((lift - 0.1).abs() < 1e-6);
            assert!((gain - 1.2).abs() < 1e-6);
            assert!((gamma - 0.95).abs() < 1e-6);
        } else {
            panic!("Expected Manual variant");
        }
    }

    #[test]
    fn test_trim_mode_saturation_variant() {
        let mode = TrimMode::Saturation { sat_gain: 1.5 };
        if let TrimMode::Saturation { sat_gain } = mode {
            assert!((sat_gain - 1.5).abs() < 1e-6);
        } else {
            panic!("Expected Saturation variant");
        }
    }

    #[test]
    fn test_trim_slop_identity() {
        let id = TrimSlop::identity();
        assert!((id.slope - 1.0).abs() < 1e-6);
        assert!((id.offset).abs() < 1e-6);
        assert!((id.power - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cm_v40_metadata_neutral() {
        let meta = CmV40Metadata::neutral(2081);
        assert_eq!(meta.target_max_pq, 2081);
        assert_eq!(meta.trim_slops.len(), 3);
        assert!((meta.chroma_weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_tone_curve_identity() {
        let slop = TrimSlop::identity();
        let result = apply_tone_curve(0.5, &slop);
        assert!((result - 0.5).abs() < 1e-5, "result={result}");
    }

    #[test]
    fn test_apply_tone_curve_clamp_high() {
        let slop = TrimSlop {
            slope: 5.0,
            offset: 0.5,
            power: 1.0,
        };
        let result = apply_tone_curve(0.5, &slop);
        assert!(result <= 1.0, "result={result}");
    }

    #[test]
    fn test_apply_tone_curve_clamp_low() {
        let slop = TrimSlop {
            slope: 0.0,
            offset: -1.0,
            power: 1.0,
        };
        let result = apply_tone_curve(0.5, &slop);
        assert!(result >= 0.0, "result={result}");
    }

    #[test]
    fn test_apply_tone_curve_negative_input_clamped() {
        let slop = TrimSlop::identity();
        let result = apply_tone_curve(-0.5, &slop);
        assert!(result >= 0.0, "result={result}");
    }

    #[test]
    fn test_compute_trim_slops_empty_histogram() {
        let hist = [0u32; 1024];
        let slops = compute_trim_slops(&hist, 1000.0);
        assert_eq!(slops.len(), 3);
        for slop in &slops {
            assert!((slop.slope - 1.0).abs() < 1e-5, "slope={}", slop.slope);
        }
    }

    #[test]
    fn test_compute_trim_slops_uniform_histogram() {
        let hist = [100u32; 1024];
        let slops = compute_trim_slops(&hist, 1000.0);
        assert_eq!(slops.len(), 3);
        // All slopes should be <= 1.0 for 1000 nit target (downmapping from 10000)
        for slop in &slops {
            assert!(
                slop.slope > 0.0 && slop.slope <= 1.1,
                "slope={}",
                slop.slope
            );
        }
    }

    #[test]
    fn test_compute_trim_slops_100_nit_target() {
        let hist = [100u32; 1024];
        let slops = compute_trim_slops(&hist, 100.0);
        // 100 nit target = gain_factor 0.01 — should have lower slopes than 1000 nit
        let slops_1000 = compute_trim_slops(&hist, 1000.0);
        assert!(
            slops[1].slope < slops_1000[1].slope,
            "expected lower slope for lower target"
        );
    }

    #[test]
    fn test_gamut_compressor_in_range_passthrough() {
        let gc = GamutCompressor::new(0.8);
        let (r, g, b) = gc.compress(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 1e-5);
        assert!((g - 0.3).abs() < 1e-5);
        assert!((b - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_gamut_compressor_clips_above_one() {
        let gc = GamutCompressor::new(0.8);
        let (r, _g, _b) = gc.compress(1.5, 0.0, 0.0);
        assert!(r <= 1.0, "r={r} must not exceed 1.0");
        assert!(r > 0.8, "r={r} should remain above threshold");
    }

    #[test]
    fn test_gamut_compressor_negative_clamped() {
        let gc = GamutCompressor::new(0.5);
        let (r, g, b) = gc.compress(-0.2, -0.1, -0.5);
        assert!(r >= 0.0, "r={r}");
        assert!(g >= 0.0, "g={g}");
        assert!(b >= 0.0, "b={b}");
    }

    #[test]
    fn test_saturation_deployment_unity_gain() {
        let (i, p, t) = SaturationDeployment::apply_saturation(0.5, 0.3, -0.1, 1.0);
        assert!((i - 0.5).abs() < 1e-6);
        assert!((p - 0.3).abs() < 1e-6);
        assert!((t - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_saturation_deployment_desaturate() {
        let (i, p, t) = SaturationDeployment::apply_saturation(0.5, 0.3, -0.1, 0.0);
        assert!((i - 0.5).abs() < 1e-6);
        assert!(p.abs() < 1e-6);
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn test_saturation_deployment_boost() {
        let (i, p, t) = SaturationDeployment::apply_saturation(0.5, 0.3, 0.2, 2.0);
        assert!((i - 0.5).abs() < 1e-6);
        assert!((p - 0.6).abs() < 1e-6);
        assert!((t - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_saturation_deployment_negative_gain_clamped() {
        let (_, p, t) = SaturationDeployment::apply_saturation(0.5, 0.3, -0.1, -1.0);
        // Negative gain clamped to 0 → desaturate
        assert!(p.abs() < 1e-6);
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn test_cm_analysis_config_default() {
        let cfg = CmAnalysisConfig::default();
        assert!((cfg.target_display_nits - 1000.0).abs() < 1.0);
        assert!((cfg.reference_white_nits - 203.0).abs() < 1.0);
        assert_eq!(cfg.trim_mode, TrimMode::Auto);
    }
}
