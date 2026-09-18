//! Logo detection for identifying brand logos within video frames.

#![allow(dead_code)]

/// A detected logo region within a frame.
#[derive(Debug, Clone)]
pub struct LogoRegion {
    /// Horizontal centre of the logo (pixels).
    pub cx: f32,
    /// Vertical centre of the logo (pixels).
    pub cy: f32,
    /// Width of the bounding box (pixels).
    pub width: f32,
    /// Height of the bounding box (pixels).
    pub height: f32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Optional label identifying the logo.
    pub label: Option<String>,
}

impl LogoRegion {
    /// Create a new [`LogoRegion`].
    #[must_use]
    pub fn new(cx: f32, cy: f32, width: f32, height: f32, confidence: f32) -> Self {
        Self {
            cx,
            cy,
            width,
            height,
            confidence: confidence.clamp(0.0, 1.0),
            label: None,
        }
    }

    /// Returns `true` when confidence is above the given minimum.
    #[must_use]
    pub fn confidence_ok(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }

    /// Return the area of the bounding box.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Configuration for logo detection.
#[derive(Debug, Clone)]
pub struct LogoDetectionConfig {
    /// Minimum confidence to keep a detection.
    pub min_confidence: f32,
    /// Maximum number of detections to return per frame.
    pub max_per_frame: usize,
    /// Minimum bounding-box area in pixels².
    pub min_area: f32,
}

impl Default for LogoDetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_per_frame: 10,
            min_area: 100.0,
        }
    }
}

impl LogoDetectionConfig {
    /// Return the configured minimum confidence threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.min_confidence
    }
}

/// Aggregated logo detection report for a sequence of frames.
#[derive(Debug, Clone)]
pub struct LogoReport {
    /// Regions accepted across all scanned frames.
    pub regions: Vec<LogoRegion>,
    /// Total number of frames scanned.
    pub frames_scanned: usize,
}

impl LogoReport {
    /// Create an empty [`LogoReport`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            frames_scanned: 0,
        }
    }

    /// Return the number of accepted logo regions.
    #[must_use]
    pub fn found_count(&self) -> usize {
        self.regions.len()
    }

    /// Return the average confidence across all regions, or 0.0 when empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_confidence(&self) -> f32 {
        if self.regions.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.regions.iter().map(|r| r.confidence).sum();
        sum / self.regions.len() as f32
    }
}

impl Default for LogoReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects logos in video frames using a simple heuristic model.
pub struct LogoDetector {
    config: LogoDetectionConfig,
    report: LogoReport,
}

impl LogoDetector {
    /// Create a new [`LogoDetector`].
    #[must_use]
    pub fn new(config: LogoDetectionConfig) -> Self {
        Self {
            config,
            report: LogoReport::new(),
        }
    }

    /// Scan a single luma frame and collect logo detections.
    ///
    /// This stub uses synthetic detections based on local brightness peaks.
    #[allow(clippy::cast_precision_loss)]
    pub fn scan_frame(&mut self, luma: &[u8], width: u32, height: u32) {
        self.report.frames_scanned += 1;
        if luma.len() < (width as usize) * (height as usize) {
            return;
        }

        // Heuristic: find the brightest 16×16 block as a candidate logo region.
        let bw = 16_u32;
        let bh = 16_u32;
        let cols = width.div_ceil(bw);
        let rows = height.div_ceil(bh);
        let w = width as usize;

        let mut best_mean = 0.0_f32;
        let mut best_bx = 0_u32;
        let mut best_by = 0_u32;

        for row in 0..rows {
            for col in 0..cols {
                let bx = col * bw;
                let by = row * bh;
                let actual_bw = bw.min(width - bx);
                let actual_bh = bh.min(height - by);

                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in 0..actual_bh {
                    for dx in 0..actual_bw {
                        let idx = ((by + dy) as usize) * w + ((bx + dx) as usize);
                        sum += u32::from(luma[idx]);
                        count += 1;
                    }
                }

                let mean = if count > 0 {
                    sum as f32 / count as f32
                } else {
                    0.0
                };
                if mean > best_mean {
                    best_mean = mean;
                    best_bx = bx;
                    best_by = by;
                }
            }
        }

        // Normalise the mean to a confidence value.
        let confidence = (best_mean / 255.0).clamp(0.0, 1.0);
        if confidence >= self.config.min_confidence {
            let region = LogoRegion::new(
                best_bx as f32 + 8.0,
                best_by as f32 + 8.0,
                bw as f32,
                bh as f32,
                confidence,
            );
            if region.area() >= self.config.min_area
                && self.report.regions.len()
                    < self.config.max_per_frame * self.report.frames_scanned
            {
                self.report.regions.push(region);
            }
        }
    }

    /// Return the accumulated report.
    #[must_use]
    pub fn report(&self) -> &LogoReport {
        &self.report
    }

    /// Consume the detector and return the final report.
    #[must_use]
    pub fn finalize(self) -> LogoReport {
        self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logo_region_confidence_clamped() {
        let r = LogoRegion::new(50.0, 50.0, 40.0, 40.0, 1.5);
        assert!((r.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_logo_region_confidence_ok_true() {
        let r = LogoRegion::new(0.0, 0.0, 20.0, 20.0, 0.8);
        assert!(r.confidence_ok(0.5));
    }

    #[test]
    fn test_logo_region_confidence_ok_false() {
        let r = LogoRegion::new(0.0, 0.0, 20.0, 20.0, 0.3);
        assert!(!r.confidence_ok(0.5));
    }

    #[test]
    fn test_logo_region_area() {
        let r = LogoRegion::new(0.0, 0.0, 10.0, 20.0, 0.6);
        assert!((r.area() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_logo_detection_config_default_threshold() {
        let cfg = LogoDetectionConfig::default();
        assert!((cfg.threshold() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_logo_report_new_empty() {
        let report = LogoReport::new();
        assert_eq!(report.found_count(), 0);
        assert_eq!(report.frames_scanned, 0);
    }

    #[test]
    fn test_logo_report_avg_confidence_empty() {
        let report = LogoReport::new();
        assert!((report.avg_confidence() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_logo_report_avg_confidence_non_empty() {
        let mut report = LogoReport::new();
        report
            .regions
            .push(LogoRegion::new(0.0, 0.0, 20.0, 20.0, 0.8));
        report
            .regions
            .push(LogoRegion::new(0.0, 0.0, 20.0, 20.0, 0.6));
        let avg = report.avg_confidence();
        assert!((avg - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_logo_detector_scan_increments_frames() {
        let mut det = LogoDetector::new(LogoDetectionConfig::default());
        let luma = vec![200u8; 32 * 32];
        det.scan_frame(&luma, 32, 32);
        assert_eq!(det.report().frames_scanned, 1);
    }

    #[test]
    fn test_logo_detector_scan_too_short_luma() {
        let mut det = LogoDetector::new(LogoDetectionConfig::default());
        let luma = vec![200u8; 10]; // shorter than 32*32
        det.scan_frame(&luma, 32, 32);
        // Frame is counted but no region is added.
        assert_eq!(det.report().frames_scanned, 1);
        assert_eq!(det.report().found_count(), 0);
    }

    #[test]
    fn test_logo_detector_bright_frame_detects_region() {
        let mut det = LogoDetector::new(LogoDetectionConfig::default());
        // Uniform bright frame — confidence will be ~200/255 ≈ 0.78 > 0.5
        let luma = vec![200u8; 64 * 64];
        det.scan_frame(&luma, 64, 64);
        assert!(det.report().found_count() > 0);
    }

    #[test]
    fn test_logo_detector_dark_frame_no_region() {
        let mut cfg = LogoDetectionConfig::default();
        cfg.min_confidence = 0.9; // very high threshold
        let mut det = LogoDetector::new(cfg);
        // Dim frame — confidence ~50/255 ≈ 0.20 < 0.9
        let luma = vec![50u8; 64 * 64];
        det.scan_frame(&luma, 64, 64);
        assert_eq!(det.report().found_count(), 0);
    }

    #[test]
    fn test_logo_detector_finalize_moves_report() {
        let mut det = LogoDetector::new(LogoDetectionConfig::default());
        let luma = vec![200u8; 32 * 32];
        det.scan_frame(&luma, 32, 32);
        let report = det.finalize();
        assert_eq!(report.frames_scanned, 1);
    }
}
