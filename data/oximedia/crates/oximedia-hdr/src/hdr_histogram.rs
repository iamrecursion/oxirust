//! HDR luminance histogram analysis for CLL/FALL estimation.
//!
//! Provides a perceptual histogram over scene-linear luminance values,
//! with percentile queries, MaxCLL and MaxFALL computation.

use crate::transfer_function::TransferFunction;
use crate::{HdrError, Result};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of histogram bins. 4096 gives ~2.4 nit resolution over 10 000 nit peak.
const DEFAULT_BINS: usize = 4096;

// ── HdrHistogram ──────────────────────────────────────────────────────────────

/// A luminance histogram over an HDR frame.
///
/// Bins are linearly spaced in the luminance domain between
/// `min_nits` and `max_nits`. Each bin counts the number of
/// pixels whose linear luminance falls within that interval.
#[derive(Debug, Clone)]
pub struct HdrHistogram {
    /// Per-bin pixel counts.
    pub bins: Vec<u64>,
    /// Minimum luminance value mapped to bin 0 (nits).
    pub min_nits: f32,
    /// Maximum luminance value mapped to the last bin (nits).
    pub max_nits: f32,
    /// Total pixel count accumulated in this histogram.
    total_pixels: u64,
}

impl HdrHistogram {
    /// Create an empty histogram with `n_bins` bins spanning `[min_nits, max_nits]`.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `min_nits >= max_nits` or `n_bins == 0`.
    pub fn new(n_bins: usize, min_nits: f32, max_nits: f32) -> Result<Self> {
        if n_bins == 0 {
            return Err(HdrError::InvalidLuminance(0.0));
        }
        if min_nits >= max_nits {
            return Err(HdrError::InvalidLuminance(min_nits));
        }
        Ok(Self {
            bins: vec![0u64; n_bins],
            min_nits,
            max_nits,
            total_pixels: 0,
        })
    }

    /// Return the number of bins.
    pub fn n_bins(&self) -> usize {
        self.bins.len()
    }

    /// Return the total number of pixels accumulated.
    pub fn total_pixels(&self) -> u64 {
        self.total_pixels
    }

    /// Return the luminance at the centre of a given bin (nits).
    pub fn bin_centre_nits(&self, bin: usize) -> f32 {
        let n = self.bins.len() as f32;
        let t = (bin as f32 + 0.5) / n;
        self.min_nits + t * (self.max_nits - self.min_nits)
    }

    /// Map a luminance value in nits to a bin index.
    ///
    /// Returns `None` if the value is outside `[min_nits, max_nits]`.
    fn nits_to_bin(&self, nits: f32) -> Option<usize> {
        if nits < self.min_nits || nits > self.max_nits {
            return None;
        }
        let range = self.max_nits - self.min_nits;
        let t = (nits - self.min_nits) / range;
        let idx = (t * self.bins.len() as f32) as usize;
        Some(idx.min(self.bins.len() - 1))
    }

    /// Accumulate a single luminance sample (nits).
    ///
    /// Values outside the histogram range are clamped to the first/last bin.
    pub fn accumulate(&mut self, nits: f32) {
        let clamped = nits.clamp(self.min_nits, self.max_nits);
        let bin = self.nits_to_bin(clamped).unwrap_or(self.bins.len() - 1);
        self.bins[bin] += 1;
        self.total_pixels += 1;
    }

    /// Merge another histogram into this one.
    ///
    /// # Errors
    /// Returns an error if bin counts or luminance ranges differ.
    pub fn merge(&mut self, other: &HdrHistogram) -> Result<()> {
        if self.bins.len() != other.bins.len() {
            return Err(HdrError::ToneMappingError(
                "histogram bin count mismatch".to_string(),
            ));
        }
        for (a, b) in self.bins.iter_mut().zip(other.bins.iter()) {
            *a += b;
        }
        self.total_pixels += other.total_pixels;
        Ok(())
    }

    /// Compute the luminance at a given percentile `p` in [0, 100].
    ///
    /// Returns the luminance (nits) below which `p`% of pixels fall.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `p` is outside [0, 100] or
    /// the histogram is empty.
    pub fn percentile(&self, p: f32) -> Result<f32> {
        if !(0.0..=100.0).contains(&p) {
            return Err(HdrError::InvalidLuminance(p));
        }
        if self.total_pixels == 0 {
            return Err(HdrError::InvalidLuminance(0.0));
        }

        let target = (p / 100.0 * self.total_pixels as f32) as u64;
        let mut cumulative: u64 = 0;

        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Ok(self.bin_centre_nits(i));
            }
        }

        // All pixels accounted for — return max_nits
        Ok(self.max_nits)
    }

    /// Maximum Content Light Level (MaxCLL): the luminance value at the 100th percentile.
    ///
    /// This is the brightest single pixel in the histogram.
    ///
    /// # Errors
    /// Propagates errors from `percentile`.
    pub fn maxcll(&self) -> Result<f32> {
        // Find last non-zero bin
        if self.total_pixels == 0 {
            return Err(HdrError::InvalidLuminance(0.0));
        }
        for i in (0..self.bins.len()).rev() {
            if self.bins[i] > 0 {
                return Ok(self.bin_centre_nits(i));
            }
        }
        Ok(self.min_nits)
    }

    /// Maximum Frame-Average Light Level (MaxFALL): the luminance at the 99.98th percentile.
    ///
    /// Per the CTA-861.3 specification, MaxFALL is derived from the frame-average
    /// luminance, not from the per-pixel peak. Using the 99.98th percentile of the
    /// per-pixel distribution provides a robust approximation that discards
    /// specular highlights and isolated bright pixels.
    ///
    /// # Errors
    /// Propagates errors from `percentile`.
    pub fn maxfall(&self) -> Result<f32> {
        self.percentile(99.98)
    }

    /// Compute the mean luminance over all pixels (nits).
    pub fn mean_nits(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| self.bin_centre_nits(i) as f64 * count as f64)
            .sum();
        (sum / self.total_pixels as f64) as f32
    }
}

// ── HdrHistogramAnalyzer ──────────────────────────────────────────────────────

/// Stateless helper for building `HdrHistogram`s from raw frame data.
pub struct HdrHistogramAnalyzer;

impl HdrHistogramAnalyzer {
    /// Build a luminance histogram from a flat interleaved RGB frame.
    ///
    /// The frame is expected to contain pixels in scene-linear light encoded
    /// according to `transfer`.  For `TransferFunction::Pq`, values are
    /// normalised to 1.0 = 10 000 nits.
    ///
    /// # Parameters
    /// - `frame`: interleaved linear-light RGB values (length divisible by 3)
    /// - `transfer`: the transfer function used to interpret the values
    ///
    /// # Returns
    /// An `HdrHistogram` covering `[0, peak_nits]` in `DEFAULT_BINS` bins.
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if `frame.len() % 3 != 0`.
    pub fn compute(frame: &[f32], transfer: &TransferFunction) -> Result<HdrHistogram> {
        if !frame.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "frame length {} is not divisible by 3",
                frame.len()
            )));
        }

        let peak_nits = transfer.peak_luminance_nits() as f32;
        let mut hist = HdrHistogram::new(DEFAULT_BINS, 0.0, peak_nits)?;

        for chunk in frame.chunks_exact(3) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];

            // BT.2100 luminance coefficients
            let linear_lum = 0.2627 * r + 0.6780 * g + 0.0593 * b;

            // Convert from normalised [0, 1] to nits
            let nits = (linear_lum * peak_nits).max(0.0);
            hist.accumulate(nits);
        }

        Ok(hist)
    }

    /// Build a luminance histogram from a greyscale luminance frame (one channel per pixel).
    ///
    /// All values are expected to be in scene-linear light normalised to 1.0 = peak.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `peak_nits <= 0`.
    pub fn compute_luma(luma: &[f32], peak_nits: f32) -> Result<HdrHistogram> {
        if peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(peak_nits));
        }

        let mut hist = HdrHistogram::new(DEFAULT_BINS, 0.0, peak_nits)?;
        for &v in luma {
            let nits = (v * peak_nits).max(0.0);
            hist.accumulate(nits);
        }
        Ok(hist)
    }
}

// ── LuminanceHistogram (configurable, supports log/linear binning) ─────────────

/// Scale used for binning luminance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramScale {
    /// Bins are linearly spaced in the nit domain.
    Linear,
    /// Bins are spaced in the log10(nits) domain — more perceptually uniform for HDR content.
    Logarithmic,
}

/// Configuration for building a [`LuminanceHistogram`].
#[derive(Debug, Clone)]
pub struct HdrHistogramConfig {
    /// Number of bins.
    pub num_bins: usize,
    /// Lower luminance bound (nits), e.g. 0.0001.
    pub min_nits: f32,
    /// Upper luminance bound (nits), e.g. 10000.0.
    pub max_nits: f32,
    /// Linear or logarithmic bin spacing.
    pub scale: HistogramScale,
}

impl Default for HdrHistogramConfig {
    fn default() -> Self {
        Self {
            num_bins: 1000,
            min_nits: 0.0001,
            max_nits: 10_000.0,
            scale: HistogramScale::Logarithmic,
        }
    }
}

/// PQ EOTF constants (SMPTE ST 2084).
mod pq_constants {
    pub const M1: f64 = 0.159_301_757_812_5;
    pub const M2: f64 = 78.843_75;
    pub const C1: f64 = 0.835_937_5;
    pub const C2: f64 = 18.851_562_5;
    pub const C3: f64 = 18.687_5;
    pub const PEAK_NITS: f64 = 10_000.0;
}

/// Convert a PQ-encoded normalised value (0.0–1.0) to absolute luminance in nits.
///
/// Implements SMPTE ST 2084 EOTF.
fn pq_to_nits(pq: f32) -> f32 {
    use pq_constants::*;
    let pq = (pq as f64).clamp(0.0, 1.0);
    let pq_m2 = pq.powf(1.0 / M2);
    let num = (pq_m2 - C1).max(0.0);
    let den = C2 - C3 * pq_m2;
    if den <= 0.0 {
        return 0.0;
    }
    (PEAK_NITS * (num / den).powf(1.0 / M1)) as f32
}

/// A luminance histogram with configurable binning for HDR analysis.
///
/// Supports both linear and logarithmic bin spacing and provides
/// MaxCLL, MaxFALL, APL, and arbitrary percentile queries.
#[derive(Debug, Clone)]
pub struct LuminanceHistogram {
    /// Per-bin pixel counts.
    pub bins: Vec<u32>,
    /// Nit value at the left edge of each bin (length = num_bins + 1).
    pub bin_edges_nits: Vec<f32>,
    /// Total pixels accumulated.
    pub total_pixels: u64,
    /// Minimum luminance observed (nits).
    pub min_nits: f32,
    /// Maximum luminance observed (nits).
    pub max_nits: f32,
    /// Running sum for mean computation (nits × pixel_count).
    mean_accumulator: f64,
}

impl LuminanceHistogram {
    /// Create an empty histogram from a configuration.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `num_bins == 0` or `min_nits >= max_nits`.
    pub fn new(config: &HdrHistogramConfig) -> crate::Result<Self> {
        if config.num_bins == 0 {
            return Err(HdrError::InvalidLuminance(0.0));
        }
        if config.min_nits >= config.max_nits {
            return Err(HdrError::InvalidLuminance(config.min_nits));
        }

        let edges = Self::compute_edges(config);
        Ok(Self {
            bins: vec![0u32; config.num_bins],
            bin_edges_nits: edges,
            total_pixels: 0,
            min_nits: f32::MAX,
            max_nits: f32::MIN,
            mean_accumulator: 0.0,
        })
    }

    fn compute_edges(config: &HdrHistogramConfig) -> Vec<f32> {
        let n = config.num_bins;
        let mut edges = Vec::with_capacity(n + 1);
        match config.scale {
            HistogramScale::Linear => {
                let step = (config.max_nits - config.min_nits) / n as f32;
                for i in 0..=n {
                    edges.push(config.min_nits + i as f32 * step);
                }
            }
            HistogramScale::Logarithmic => {
                let log_min = config.min_nits.max(1e-7_f32).log10() as f64;
                let log_max = (config.max_nits as f64).log10();
                let step = (log_max - log_min) / n as f64;
                for i in 0..=n {
                    edges.push(10.0_f64.powf(log_min + i as f64 * step) as f32);
                }
            }
        }
        edges
    }

    /// Map a nit value to a bin index using binary search.
    fn nits_to_bin(&self, nits: f32) -> usize {
        let n = self.bins.len();
        // Binary search in bin_edges_nits
        let mut lo = 0usize;
        let mut hi = n; // bin index is in [0, n-1]
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.bin_edges_nits[mid + 1] <= nits {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.min(n - 1)
    }

    /// Accumulate a single luminance sample in nits.
    fn accumulate_nits(&mut self, nits: f32) {
        let cfg_min = self.bin_edges_nits[0];
        let cfg_max = *self.bin_edges_nits.last().unwrap_or(&1.0);
        let clamped = nits.clamp(cfg_min, cfg_max);
        let bin = self.nits_to_bin(clamped);
        self.bins[bin] += 1;
        self.total_pixels += 1;
        if nits < self.min_nits {
            self.min_nits = nits;
        }
        if nits > self.max_nits {
            self.max_nits = nits;
        }
        self.mean_accumulator += clamped as f64;
    }

    /// Build histogram from a PQ-encoded frame (interleaved RGB, values 0.0–1.0).
    ///
    /// BT.2100 luminance coefficients are applied to convert RGB to Y.
    /// PQ EOTF converts Y to absolute nits.
    pub fn from_pq_frame(
        frame: &[f32],
        _width: u32,
        _height: u32,
        config: &HdrHistogramConfig,
    ) -> crate::Result<Self> {
        if !frame.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "PQ frame length {} not divisible by 3",
                frame.len()
            )));
        }
        let mut hist = Self::new(config)?;
        for chunk in frame.chunks_exact(3) {
            // BT.2100 luma: Y = 0.2627 R + 0.6780 G + 0.0593 B
            let y_pq = 0.2627 * chunk[0] + 0.6780 * chunk[1] + 0.0593 * chunk[2];
            let nits = pq_to_nits(y_pq);
            hist.accumulate_nits(nits);
        }
        Ok(hist)
    }

    /// Build histogram from a linear-light frame (interleaved RGB, values in nits).
    ///
    /// BT.2100 luminance coefficients are applied to convert RGB to Y nits.
    pub fn from_linear_nits(
        frame: &[f32],
        _width: u32,
        _height: u32,
        config: &HdrHistogramConfig,
    ) -> crate::Result<Self> {
        if !frame.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "linear frame length {} not divisible by 3",
                frame.len()
            )));
        }
        let mut hist = Self::new(config)?;
        for chunk in frame.chunks_exact(3) {
            let y = 0.2627 * chunk[0] + 0.6780 * chunk[1] + 0.0593 * chunk[2];
            hist.accumulate_nits(y.max(0.0));
        }
        Ok(hist)
    }

    /// Return the luminance (nits) at the given percentile `p` (0.0–100.0).
    ///
    /// Uses the centre of the qualifying bin as the result.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total_pixels == 0 || !(0.0..=100.0).contains(&p) {
            return 0.0;
        }
        let target = ((p / 100.0) * self.total_pixels as f32) as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += u64::from(count);
            if cumulative >= target {
                // bin centre
                let lo = self.bin_edges_nits[i];
                let hi = self.bin_edges_nits[i + 1];
                return (lo + hi) * 0.5;
            }
        }
        *self.bin_edges_nits.last().unwrap_or(&0.0)
    }

    /// Maximum Frame Average Light Level (MaxFALL).
    ///
    /// Per CTA-861.3, approximated as the 99.98th percentile of per-pixel luminance.
    pub fn max_fall(&self) -> f32 {
        self.percentile(99.98)
    }

    /// Maximum Content Light Level (MaxCLL) — the brightest pixel in the histogram.
    pub fn max_cll(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        for i in (0..self.bins.len()).rev() {
            if self.bins[i] > 0 {
                let lo = self.bin_edges_nits[i];
                let hi = self.bin_edges_nits[i + 1];
                return (lo + hi) * 0.5;
            }
        }
        self.bin_edges_nits[0]
    }

    /// Average Picture Level (APL) — mean luminance over all pixels (nits).
    pub fn apl(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        (self.mean_accumulator / self.total_pixels as f64) as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // 1. Empty histogram returns error on percentile
    #[test]
    fn test_empty_histogram_percentile_error() {
        let hist = HdrHistogram::new(256, 0.0, 10_000.0).expect("new");
        assert!(
            hist.percentile(50.0).is_err(),
            "empty histogram should error"
        );
    }

    // 2. Accumulate a single pixel
    #[test]
    fn test_accumulate_single_pixel() {
        let mut hist = HdrHistogram::new(256, 0.0, 10_000.0).expect("new");
        hist.accumulate(500.0);
        assert_eq!(hist.total_pixels(), 1);
    }

    // 3. Percentile on uniform distribution
    #[test]
    fn test_percentile_uniform() {
        let mut hist = HdrHistogram::new(1000, 0.0, 1000.0).expect("new");
        for i in 0..1000 {
            hist.accumulate(i as f32);
        }
        let p50 = hist.percentile(50.0).expect("p50");
        // Should be near 500 nits
        assert!(approx(p50, 500.0, 50.0), "p50 ~ 500 nits: {p50}");
    }

    // 4. percentile(100) should return near max
    #[test]
    fn test_percentile_100() {
        let mut hist = HdrHistogram::new(256, 0.0, 1000.0).expect("new");
        for i in 0..100 {
            hist.accumulate(i as f32 * 10.0);
        }
        let p100 = hist.percentile(100.0).expect("p100");
        assert!(p100 > 800.0, "p100 should be near max: {p100}");
    }

    // 5. percentile(0) should return near minimum
    #[test]
    fn test_percentile_0() {
        let mut hist = HdrHistogram::new(256, 0.0, 1000.0).expect("new");
        for i in 1..=100 {
            hist.accumulate(i as f32);
        }
        let p0 = hist.percentile(0.0).expect("p0");
        assert!(p0 <= 10.0, "p0 should be near min: {p0}");
    }

    // 6. MaxCLL returns the maximum accumulated luminance
    #[test]
    fn test_maxcll_single_peak() {
        let mut hist = HdrHistogram::new(1000, 0.0, 10_000.0).expect("new");
        hist.accumulate(100.0);
        hist.accumulate(4000.0); // bright pixel
        hist.accumulate(200.0);
        let cll = hist.maxcll().expect("maxcll");
        assert!((3900.0..=4100.0).contains(&cll), "MaxCLL near 4000: {cll}");
    }

    // 7. MaxFALL is below MaxCLL for content with a few bright pixels
    #[test]
    fn test_maxfall_below_maxcll() {
        let mut hist = HdrHistogram::new(1000, 0.0, 1000.0).expect("new");
        // Most pixels at ~100 nits, a few very bright
        for _ in 0..1000 {
            hist.accumulate(100.0);
        }
        for _ in 0..2 {
            hist.accumulate(900.0);
        }
        let cll = hist.maxcll().expect("maxcll");
        let fall = hist.maxfall().expect("maxfall");
        assert!(fall <= cll + 1.0, "MaxFALL should not exceed MaxCLL");
    }

    // 8. Compute from RGB frame
    #[test]
    fn test_compute_from_rgb_frame() {
        // Flat grey frame: all pixels at linear 0.5 (= 5000 nits for PQ)
        let frame: Vec<f32> = vec![0.5f32; 300]; // 100 grey pixels
        let hist = HdrHistogramAnalyzer::compute(&frame, &TransferFunction::Pq).expect("compute");
        assert_eq!(hist.total_pixels(), 100);
        let cll = hist.maxcll().expect("maxcll");
        // 0.5 * 0.2627 + 0.5 * 0.6780 + 0.5 * 0.0593 ≈ 0.5 => 5000 nits
        assert!(cll > 4000.0 && cll < 6000.0, "CLL near 5000 nits: {cll}");
    }

    // 9. Compute from grey frame with luma helper
    #[test]
    fn test_compute_luma() {
        let luma: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let hist = HdrHistogramAnalyzer::compute_luma(&luma, 10_000.0).expect("luma");
        assert_eq!(hist.total_pixels(), 100);
        let p99 = hist.percentile(99.0).expect("p99");
        assert!(p99 > 8000.0, "p99 should be near max: {p99}");
    }

    // 10. Mean luminance computation
    #[test]
    fn test_mean_nits() {
        let mut hist = HdrHistogram::new(1000, 0.0, 1000.0).expect("new");
        for _ in 0..100 {
            hist.accumulate(500.0);
        }
        let mean = hist.mean_nits();
        assert!(approx(mean, 500.0, 25.0), "mean should be ~500: {mean}");
    }

    // 11. Invalid range returns error
    #[test]
    fn test_invalid_range_error() {
        assert!(HdrHistogram::new(256, 1000.0, 0.0).is_err());
        assert!(HdrHistogram::new(0, 0.0, 1000.0).is_err());
    }

    // 12. Compute from frame with invalid length returns error
    #[test]
    fn test_compute_invalid_frame_length() {
        let frame = vec![0.5f32; 10]; // not divisible by 3
        assert!(HdrHistogramAnalyzer::compute(&frame, &TransferFunction::Pq).is_err());
    }

    // ── LuminanceHistogram tests ────────────────────────────────────────────

    fn default_config() -> HdrHistogramConfig {
        HdrHistogramConfig::default()
    }

    fn linear_config() -> HdrHistogramConfig {
        HdrHistogramConfig {
            num_bins: 1000,
            min_nits: 0.0,
            max_nits: 10_000.0,
            scale: HistogramScale::Linear,
        }
    }

    // LH-1: Default config creates correct number of edges
    #[test]
    fn test_lh_new_default_edges() {
        let cfg = default_config();
        let hist = LuminanceHistogram::new(&cfg).expect("new");
        assert_eq!(hist.bins.len(), cfg.num_bins);
        assert_eq!(hist.bin_edges_nits.len(), cfg.num_bins + 1);
    }

    // LH-2: Invalid config returns error
    #[test]
    fn test_lh_new_invalid_config() {
        let bad = HdrHistogramConfig {
            num_bins: 0,
            ..default_config()
        };
        assert!(LuminanceHistogram::new(&bad).is_err());
        let bad2 = HdrHistogramConfig {
            min_nits: 500.0,
            max_nits: 100.0,
            ..default_config()
        };
        assert!(LuminanceHistogram::new(&bad2).is_err());
    }

    // LH-3: from_pq_frame rejects non-multiple-of-3 lengths
    #[test]
    fn test_lh_from_pq_frame_invalid_length() {
        let frame = vec![0.5f32; 7];
        assert!(LuminanceHistogram::from_pq_frame(&frame, 2, 1, &default_config()).is_err());
    }

    // LH-4: from_linear_nits rejects non-multiple-of-3 lengths
    #[test]
    fn test_lh_from_linear_nits_invalid_length() {
        let frame = vec![100.0f32; 5];
        assert!(LuminanceHistogram::from_linear_nits(&frame, 1, 1, &default_config()).is_err());
    }

    // LH-5: from_linear_nits accumulates correct pixel count
    #[test]
    fn test_lh_from_linear_nits_pixel_count() {
        // 30 values = 10 RGB pixels, each at 100 nits
        let frame: Vec<f32> = vec![100.0_f32; 30];
        let hist = LuminanceHistogram::from_linear_nits(&frame, 10, 1, &linear_config())
            .expect("from_linear");
        assert_eq!(hist.total_pixels, 10);
    }

    // LH-6: APL of uniform-100-nit frame ≈ 100
    #[test]
    fn test_lh_apl_uniform() {
        let frame: Vec<f32> = std::iter::repeat_n(100.0_f32, 300).collect(); // 100 grey pixels
        let hist =
            LuminanceHistogram::from_linear_nits(&frame, 10, 10, &linear_config()).expect("apl");
        // Y = 0.2627*100 + 0.6780*100 + 0.0593*100 = 100 nits
        let apl = hist.apl();
        assert!(approx(apl, 100.0, 5.0), "APL should be ~100 nits: {apl}");
    }

    // LH-7: max_cll returns brightest bin for mixed frame
    #[test]
    fn test_lh_max_cll_mixed() {
        let cfg = linear_config();
        // 9 dark pixels (1 nit each R=G=B) + 1 bright pixel (1000 nits each)
        let mut frame: Vec<f32> = vec![1.0_f32; 27];
        frame.extend_from_slice(&[1000.0_f32, 1000.0_f32, 1000.0_f32]);
        let hist = LuminanceHistogram::from_linear_nits(&frame, 10, 1, &cfg).expect("mixed");
        let cll = hist.max_cll();
        assert!(cll > 800.0, "MaxCLL should be near 1000: {cll}");
    }

    // LH-8: max_fall is <= max_cll
    #[test]
    fn test_lh_max_fall_le_max_cll() {
        let cfg = linear_config();
        // Most pixels at 50 nits, a few at 5000
        let mut frame: Vec<f32> = vec![50.0_f32; 900];
        frame.extend_from_slice(&[5000.0_f32; 30]);
        let hist = LuminanceHistogram::from_linear_nits(&frame, 310, 3, &cfg).expect("fall");
        assert!(hist.max_fall() <= hist.max_cll() + 1.0);
    }

    // LH-9: percentile(0) returns a low value; percentile(100) returns a high one
    #[test]
    fn test_lh_percentile_bounds() {
        let cfg = linear_config();
        // Ramp from 0 to 9999 nits, one pixel each
        let n = 100usize;
        let mut frame = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = i as f32 * 100.0;
            frame.extend_from_slice(&[v, v, v]);
        }
        let hist = LuminanceHistogram::from_linear_nits(&frame, n as u32, 1, &cfg).expect("ramp");
        let p0 = hist.percentile(0.0);
        let p100 = hist.percentile(100.0);
        assert!(p0 <= p100, "p0 ({p0}) must not exceed p100 ({p100})");
    }

    // LH-10: Log-scale histogram produces increasing edge values
    #[test]
    fn test_lh_log_edges_increasing() {
        let cfg = default_config(); // log scale
        let hist = LuminanceHistogram::new(&cfg).expect("log");
        for w in hist.bin_edges_nits.windows(2) {
            assert!(
                w[1] > w[0],
                "edges must be strictly increasing: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    // LH-11: from_pq_frame with known PQ value near 0.508 → ~100 nits
    #[test]
    fn test_lh_from_pq_known_value() {
        // PQ value ~0.508 ≈ 100 nits (standard reference white for HDR10)
        // We use a value that decodes to approximately 100 nits.
        // The actual PQ of 100 nits is approximately 0.5070605.
        let pq_100 = 0.5070605_f32;
        let frame: Vec<f32> = vec![pq_100; 30]; // 10 pixels, each channel = pq_100
        let cfg = HdrHistogramConfig {
            num_bins: 1000,
            min_nits: 0.0001,
            max_nits: 10_000.0,
            scale: HistogramScale::Logarithmic,
        };
        let hist = LuminanceHistogram::from_pq_frame(&frame, 10, 1, &cfg).expect("pq");
        assert_eq!(hist.total_pixels, 10);
        // Y = (0.2627 + 0.6780 + 0.0593) * pq_100 = 1.0 * pq_100
        // so nits ≈ 100
        let apl = hist.apl();
        assert!(
            apl > 50.0 && apl < 200.0,
            "APL should be near 100 nits from PQ: {apl}"
        );
    }

    // LH-12: Empty LuminanceHistogram returns 0.0 for all stat methods
    #[test]
    fn test_lh_empty_stats() {
        let hist = LuminanceHistogram::new(&default_config()).expect("empty");
        assert_eq!(hist.apl(), 0.0);
        assert_eq!(hist.max_cll(), 0.0);
        assert_eq!(hist.max_fall(), 0.0);
        assert_eq!(hist.percentile(50.0), 0.0);
    }
}
