//! Detection and reduction of colour banding artefacts in video/image content.

#![allow(dead_code)]

/// A detected banding pattern within a region of an image.
#[derive(Debug, Clone)]
pub struct BandingPattern {
    /// Pixel row at which the band starts.
    pub start_row: u32,
    /// Pixel row at which the band ends (exclusive).
    pub end_row: u32,
    /// Pixel column at which the band starts.
    pub start_col: u32,
    /// Pixel column at which the band ends (exclusive).
    pub end_col: u32,
    /// Mean luminance value of this band (in the range `[0.0, 1.0]`).
    pub mean_luma: f32,
    /// Sharpness of the transition at the band edge (higher → harder edge).
    pub edge_sharpness: f32,
}

impl BandingPattern {
    /// Create a new banding pattern.
    #[must_use]
    pub fn new(start_row: u32, end_row: u32, start_col: u32, end_col: u32, mean_luma: f32) -> Self {
        Self {
            start_row,
            end_row,
            start_col,
            end_col,
            mean_luma,
            edge_sharpness: 1.0,
        }
    }

    /// Return the height of the band in pixels.
    #[must_use]
    pub fn band_height(&self) -> u32 {
        self.end_row.saturating_sub(self.start_row)
    }

    /// Return the width of the band in pixels.
    #[must_use]
    pub fn band_width(&self) -> u32 {
        self.end_col.saturating_sub(self.start_col)
    }

    /// Return the total area of the band in pixels².
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.band_height()) * u64::from(self.band_width())
    }

    /// Return `true` if this is a narrow band (height < 4 rows or width < 4 cols).
    #[must_use]
    pub fn is_narrow(&self) -> bool {
        self.band_height() < 4 || self.band_width() < 4
    }
}

/// Severity classification for a banding report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BandingSeverity {
    /// No perceptible banding.
    None,
    /// Barely visible under close inspection.
    Mild,
    /// Clearly visible in gradients.
    Moderate,
    /// Prominently visible, affecting perceived quality.
    Severe,
}

impl BandingSeverity {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Mild => "mild",
            Self::Moderate => "moderate",
            Self::Severe => "severe",
        }
    }
}

/// Applies debanding smoothing to a region of pixels.
pub struct DeBandingFilter {
    /// Threshold below which adjacent pixel differences are smoothed.
    pub threshold: f32,
    /// Radius of the smoothing kernel (in pixels).
    pub radius: u32,
    /// Strength of the correction in `[0.0, 1.0]`.
    pub strength: f32,
}

impl DeBandingFilter {
    /// Create a new debanding filter.
    #[must_use]
    pub fn new(threshold: f32, radius: u32, strength: f32) -> Self {
        Self {
            threshold,
            radius,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Default filter suitable for 8-bit SDR content.
    #[must_use]
    pub fn default_sdr() -> Self {
        Self::new(0.02, 3, 0.6)
    }

    /// Smooth a 1-D row of pixel values in-place where adjacent differences fall below
    /// the configured threshold.
    ///
    /// Returns the smoothed row.
    #[must_use]
    pub fn smooth_region(&self, row: &[f32]) -> Vec<f32> {
        if row.is_empty() {
            return Vec::new();
        }
        let mut out = row.to_vec();
        let r = self.radius as usize;
        for i in r..row.len().saturating_sub(r) {
            let window: Vec<f32> = row[i.saturating_sub(r)..=(i + r).min(row.len() - 1)].to_vec();
            let diff = (row[i] - row[i.saturating_sub(1)]).abs();
            if diff < self.threshold {
                #[allow(clippy::cast_precision_loss)]
                let mean: f32 = window.iter().sum::<f32>() / window.len() as f32;
                out[i] = row[i] * (1.0 - self.strength) + mean * self.strength;
            }
        }
        out
    }
}

/// Scans image data for banding patterns.
pub struct BandingDetector {
    /// Minimum luma step between adjacent rows/cols to be counted as a band edge (0..1 range).
    pub edge_threshold: f32,
    /// Minimum band width in pixels to be considered a real band.
    pub min_band_width: u32,
}

impl BandingDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new(edge_threshold: f32, min_band_width: u32) -> Self {
        Self {
            edge_threshold,
            min_band_width,
        }
    }

    /// Scan a row of pixel values and return detected banding patterns.
    ///
    /// `row` is expected to be a 1-D slice of normalised luma values in `[0.0, 1.0]`.
    /// `row_index` is the row coordinate used to populate the returned patterns.
    #[must_use]
    pub fn scan(&self, row: &[f32], row_index: u32) -> Vec<BandingPattern> {
        if row.len() < 2 {
            return Vec::new();
        }
        let mut patterns = Vec::new();
        let mut band_start: Option<usize> = None;
        let mut band_sum = 0.0_f32;
        let mut band_len = 0_usize;

        for i in 1..row.len() {
            let diff = (row[i] - row[i - 1]).abs();
            if diff < self.edge_threshold {
                if band_start.is_none() {
                    band_start = Some(i - 1);
                    band_sum = row[i - 1];
                    band_len = 1;
                }
                band_sum += row[i];
                band_len += 1;
            } else if let Some(start) = band_start.take() {
                #[allow(clippy::cast_precision_loss)]
                let mean = band_sum / band_len as f32;
                let width = (i - start) as u32;
                if width >= self.min_band_width {
                    patterns.push(BandingPattern::new(
                        row_index,
                        row_index + 1,
                        start as u32,
                        i as u32,
                        mean,
                    ));
                }
                band_sum = 0.0;
                band_len = 0;
            }
        }
        // Close any open band at the end
        if let Some(start) = band_start {
            #[allow(clippy::cast_precision_loss)]
            let mean = band_sum / band_len as f32;
            let width = (row.len() - start) as u32;
            if width >= self.min_band_width {
                patterns.push(BandingPattern::new(
                    row_index,
                    row_index + 1,
                    start as u32,
                    row.len() as u32,
                    mean,
                ));
            }
        }
        patterns
    }
}

/// Summary report of banding detected in a frame or image.
#[derive(Debug, Clone)]
pub struct BandingReport {
    /// All detected banding patterns.
    pub patterns: Vec<BandingPattern>,
    /// Total pixel area analysed.
    pub total_area: u64,
}

impl BandingReport {
    /// Create a new report.
    #[must_use]
    pub fn new(patterns: Vec<BandingPattern>, total_area: u64) -> Self {
        Self {
            patterns,
            total_area,
        }
    }

    /// Total pixel area covered by detected banding.
    #[must_use]
    pub fn banded_area(&self) -> u64 {
        self.patterns.iter().map(|p| p.area()).sum()
    }

    /// Fraction of the total area that is banded (`[0.0, 1.0]`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn banded_fraction(&self) -> f32 {
        if self.total_area == 0 {
            return 0.0;
        }
        self.banded_area() as f32 / self.total_area as f32
    }

    /// Classify the banding severity based on the banded fraction.
    #[must_use]
    pub fn severity(&self) -> BandingSeverity {
        let frac = self.banded_fraction();
        if frac < 0.02 {
            BandingSeverity::None
        } else if frac < 0.10 {
            BandingSeverity::Mild
        } else if frac < 0.30 {
            BandingSeverity::Moderate
        } else {
            BandingSeverity::Severe
        }
    }

    /// Return the number of detected bands.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BandingPattern ---

    #[test]
    fn test_band_width() {
        let p = BandingPattern::new(0, 1, 10, 50, 0.5);
        assert_eq!(p.band_width(), 40);
    }

    #[test]
    fn test_band_height() {
        let p = BandingPattern::new(5, 15, 0, 100, 0.5);
        assert_eq!(p.band_height(), 10);
    }

    #[test]
    fn test_band_area() {
        let p = BandingPattern::new(0, 10, 0, 20, 0.5);
        assert_eq!(p.area(), 200);
    }

    #[test]
    fn test_band_is_narrow_narrow() {
        let p = BandingPattern::new(0, 2, 0, 100, 0.5);
        assert!(p.is_narrow()); // height < 4
    }

    #[test]
    fn test_band_is_narrow_wide() {
        let p = BandingPattern::new(0, 10, 0, 100, 0.5);
        assert!(!p.is_narrow());
    }

    // --- BandingSeverity ---

    #[test]
    fn test_severity_order() {
        assert!(BandingSeverity::None < BandingSeverity::Mild);
        assert!(BandingSeverity::Mild < BandingSeverity::Moderate);
        assert!(BandingSeverity::Moderate < BandingSeverity::Severe);
    }

    #[test]
    fn test_severity_label() {
        assert_eq!(BandingSeverity::Severe.label(), "severe");
        assert_eq!(BandingSeverity::None.label(), "none");
    }

    // --- DeBandingFilter ---

    #[test]
    fn test_smooth_region_length_preserved() {
        let filter = DeBandingFilter::default_sdr();
        let row: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let out = filter.smooth_region(&row);
        assert_eq!(out.len(), row.len());
    }

    #[test]
    fn test_smooth_region_empty() {
        let filter = DeBandingFilter::default_sdr();
        assert!(filter.smooth_region(&[]).is_empty());
    }

    #[test]
    fn test_smooth_region_flat_row_unchanged() {
        let filter = DeBandingFilter::new(0.5, 2, 1.0);
        let row = vec![0.5_f32; 20];
        let out = filter.smooth_region(&row);
        // All values should remain close to 0.5
        for v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected ~0.5 but got {v}");
        }
    }

    #[test]
    fn test_deband_filter_strength_clamped() {
        let filter = DeBandingFilter::new(0.02, 3, 2.5);
        assert!(filter.strength <= 1.0);
    }

    // --- BandingDetector ---

    #[test]
    fn test_detector_scan_flat_row_one_band() {
        let detector = BandingDetector::new(0.05, 4);
        let row = vec![0.5_f32; 100];
        let bands = detector.scan(&row, 0);
        assert_eq!(bands.len(), 1);
    }

    #[test]
    fn test_detector_scan_no_bands_on_short_row() {
        let detector = BandingDetector::new(0.05, 4);
        let bands = detector.scan(&[0.5], 0);
        assert!(bands.is_empty());
    }

    #[test]
    fn test_detector_scan_band_row_index_preserved() {
        let detector = BandingDetector::new(0.05, 4);
        let row = vec![0.3_f32; 50];
        let bands = detector.scan(&row, 7);
        for b in &bands {
            assert_eq!(b.start_row, 7);
        }
    }

    // --- BandingReport ---

    #[test]
    fn test_report_no_bands_severity_none() {
        let report = BandingReport::new(vec![], 1920 * 1080);
        assert_eq!(report.severity(), BandingSeverity::None);
    }

    #[test]
    fn test_report_banded_fraction() {
        let p = BandingPattern::new(0, 10, 0, 50, 0.5); // area = 500
        let report = BandingReport::new(vec![p], 1000);
        assert!((report.banded_fraction() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_report_severity_severe() {
        let p = BandingPattern::new(0, 100, 0, 500, 0.5); // area = 50000
        let report = BandingReport::new(vec![p], 100_000);
        assert_eq!(report.severity(), BandingSeverity::Severe);
    }

    #[test]
    fn test_report_band_count() {
        let patterns = vec![
            BandingPattern::new(0, 1, 0, 10, 0.5),
            BandingPattern::new(1, 2, 0, 10, 0.6),
        ];
        let report = BandingReport::new(patterns, 1000);
        assert_eq!(report.band_count(), 2);
    }

    #[test]
    fn test_report_zero_total_area() {
        let report = BandingReport::new(vec![], 0);
        assert_eq!(report.banded_fraction(), 0.0);
    }
}
