#![allow(dead_code)]
//! Watermark detection for forensic analysis.
//!
//! Provides types and a detector for identifying visible, invisible, and
//! forensic watermarks within frame data.

/// Category of watermark that was (or may be) present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkType {
    /// Visually apparent branding or overlay.
    Visible,
    /// Steganographically embedded, imperceptible to the eye.
    Invisible,
    /// Forensic / fingerprinting watermark embedded for provenance tracking.
    Forensic,
}

impl WatermarkType {
    /// Returns `true` for types that can be detected by simple visual inspection.
    pub fn is_detectable(&self) -> bool {
        matches!(self, WatermarkType::Visible)
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            WatermarkType::Visible => "Visible",
            WatermarkType::Invisible => "Invisible",
            WatermarkType::Forensic => "Forensic",
        }
    }
}

/// A candidate watermark signal returned by the detector.
#[derive(Debug, Clone)]
pub struct WatermarkSignal {
    /// What kind of watermark this signal represents.
    pub watermark_type: WatermarkType,
    /// Detection confidence in the range [0.0, 1.0].
    pub confidence: f64,
    /// Optional spatial location hint (x, y) within the frame.
    pub location: Option<(u32, u32)>,
    /// Additional detail string produced by the detector.
    pub detail: String,
}

impl WatermarkSignal {
    /// Construct a new signal.
    pub fn new(watermark_type: WatermarkType, confidence: f64) -> Self {
        Self {
            watermark_type,
            confidence: confidence.clamp(0.0, 1.0),
            location: None,
            detail: String::new(),
        }
    }

    /// Attach spatial location information.
    pub fn with_location(mut self, x: u32, y: u32) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Attach a detail message.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Returns `true` when confidence meets or exceeds the given threshold.
    pub fn confidence_ok(&self, threshold: f64) -> bool {
        self.confidence >= threshold.clamp(0.0, 1.0)
    }
}

/// Simulated per-frame scan result.
#[derive(Debug)]
pub struct FrameScanResult {
    /// Frame index within the video.
    pub frame_index: u64,
    /// All signals found in this frame.
    pub signals: Vec<WatermarkSignal>,
}

impl FrameScanResult {
    /// Whether any signal meets the confidence threshold.
    pub fn any_confident(&self, threshold: f64) -> bool {
        self.signals.iter().any(|s| s.confidence_ok(threshold))
    }

    /// Signals above the threshold.
    pub fn confident_signals(&self, threshold: f64) -> Vec<&WatermarkSignal> {
        self.signals
            .iter()
            .filter(|s| s.confidence_ok(threshold))
            .collect()
    }
}

/// Watermark detector that operates on raw luminance byte slices.
#[derive(Debug)]
pub struct WatermarkDetector {
    /// Minimum confidence to record a signal.
    pub confidence_threshold: f64,
    /// Accumulated scan results.
    scan_results: Vec<FrameScanResult>,
}

impl WatermarkDetector {
    /// Create a detector with the given confidence threshold.
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
            scan_results: Vec::new(),
        }
    }

    /// Scan a single frame represented as raw luma bytes.
    ///
    /// This implementation uses a heuristic:
    /// - High mean brightness → possible visible watermark.
    /// - Low variance → possible invisible watermark.
    /// - Bit-pattern modulation → possible forensic watermark.
    #[allow(clippy::cast_precision_loss)]
    pub fn scan_frame(&mut self, frame_index: u64, luma: &[u8]) -> &FrameScanResult {
        let mut signals = Vec::new();

        if !luma.is_empty() {
            let mean = luma.iter().map(|&v| v as f64).sum::<f64>() / luma.len() as f64;
            let variance = luma
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / luma.len() as f64;

            // Heuristic: bright region with low variance → visible overlay.
            if mean > 200.0 && variance < 50.0 {
                let conf = ((mean - 200.0) / 55.0).clamp(0.0, 1.0);
                let sig = WatermarkSignal::new(WatermarkType::Visible, conf)
                    .with_detail(format!("mean={mean:.1}, var={variance:.1}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }

            // Heuristic: low mean with very low variance → invisible embed.
            if mean < 30.0 && variance < 5.0 {
                let conf = (1.0 - mean / 30.0).clamp(0.0, 1.0);
                let sig = WatermarkSignal::new(WatermarkType::Invisible, conf)
                    .with_detail(format!("mean={mean:.1}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }

            // Heuristic: LSB modulation → forensic mark.
            let lsb_ones = luma.iter().filter(|&&v| v & 1 == 1).count();
            let lsb_ratio = lsb_ones as f64 / luma.len() as f64;
            if (lsb_ratio - 0.5).abs() < 0.05 {
                // Near-perfect 50/50 → likely structured embed.
                let conf = 1.0 - (lsb_ratio - 0.5).abs() / 0.05;
                let sig = WatermarkSignal::new(WatermarkType::Forensic, conf)
                    .with_detail(format!("lsb_ratio={lsb_ratio:.3}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }
        }

        self.scan_results.push(FrameScanResult {
            frame_index,
            signals,
        });
        // Safety: we just pushed an element, so last() is always Some.
        self.scan_results
            .last()
            .expect("scan_results is non-empty after push")
    }

    /// Total number of detected signals across all scanned frames.
    pub fn detected_count(&self) -> usize {
        self.scan_results.iter().map(|r| r.signals.len()).sum()
    }

    /// Number of frames scanned so far.
    pub fn frames_scanned(&self) -> usize {
        self.scan_results.len()
    }

    /// Reset all accumulated results.
    pub fn reset(&mut self) {
        self.scan_results.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_type_visible_is_detectable() {
        assert!(WatermarkType::Visible.is_detectable());
    }

    #[test]
    fn test_watermark_type_invisible_not_detectable() {
        assert!(!WatermarkType::Invisible.is_detectable());
    }

    #[test]
    fn test_watermark_type_forensic_not_detectable() {
        assert!(!WatermarkType::Forensic.is_detectable());
    }

    #[test]
    fn test_watermark_type_labels() {
        assert_eq!(WatermarkType::Visible.label(), "Visible");
        assert_eq!(WatermarkType::Invisible.label(), "Invisible");
        assert_eq!(WatermarkType::Forensic.label(), "Forensic");
    }

    #[test]
    fn test_signal_confidence_clamps() {
        let s = WatermarkSignal::new(WatermarkType::Visible, 2.0);
        assert_eq!(s.confidence, 1.0);
        let s2 = WatermarkSignal::new(WatermarkType::Visible, -1.0);
        assert_eq!(s2.confidence, 0.0);
    }

    #[test]
    fn test_signal_confidence_ok_true() {
        let s = WatermarkSignal::new(WatermarkType::Forensic, 0.8);
        assert!(s.confidence_ok(0.7));
    }

    #[test]
    fn test_signal_confidence_ok_false() {
        let s = WatermarkSignal::new(WatermarkType::Forensic, 0.3);
        assert!(!s.confidence_ok(0.5));
    }

    #[test]
    fn test_signal_with_location() {
        let s = WatermarkSignal::new(WatermarkType::Visible, 0.9).with_location(10, 20);
        assert_eq!(s.location, Some((10, 20)));
    }

    #[test]
    fn test_signal_with_detail() {
        let s = WatermarkSignal::new(WatermarkType::Invisible, 0.6).with_detail("embedded payload");
        assert!(s.detail.contains("embedded payload"));
    }

    #[test]
    fn test_detector_empty_frame_no_signals() {
        let mut det = WatermarkDetector::new(0.0);
        let result = det.scan_frame(0, &[]);
        assert!(result.signals.is_empty());
    }

    #[test]
    fn test_detector_detected_count_zero_initially() {
        let det = WatermarkDetector::new(0.5);
        assert_eq!(det.detected_count(), 0);
        assert_eq!(det.frames_scanned(), 0);
    }

    #[test]
    fn test_detector_frames_scanned_increases() {
        let mut det = WatermarkDetector::new(0.5);
        det.scan_frame(0, &[128u8; 100]);
        det.scan_frame(1, &[128u8; 100]);
        assert_eq!(det.frames_scanned(), 2);
    }

    #[test]
    fn test_detector_reset_clears_results() {
        let mut det = WatermarkDetector::new(0.0);
        det.scan_frame(0, &[255u8; 50]);
        assert!(det.frames_scanned() > 0);
        det.reset();
        assert_eq!(det.frames_scanned(), 0);
        assert_eq!(det.detected_count(), 0);
    }

    #[test]
    fn test_frame_scan_result_any_confident() {
        let result = FrameScanResult {
            frame_index: 0,
            signals: vec![WatermarkSignal::new(WatermarkType::Visible, 0.9)],
        };
        assert!(result.any_confident(0.8));
        assert!(!result.any_confident(0.95));
    }

    #[test]
    fn test_frame_scan_result_confident_signals() {
        let result = FrameScanResult {
            frame_index: 0,
            signals: vec![
                WatermarkSignal::new(WatermarkType::Visible, 0.9),
                WatermarkSignal::new(WatermarkType::Invisible, 0.3),
            ],
        };
        let conf = result.confident_signals(0.5);
        assert_eq!(conf.len(), 1);
    }

    // ── Parametrized embedding-strength tests ──────────────────────────────────
    //
    // These tests sweep a synthetic "embedding strength" parameter for each of
    // the three watermark heuristics implemented in `scan_frame` and assert
    // that detection confidence follows the expected monotonic trend: weak
    // embeds should not fire (or should fire with low confidence), and strong
    // embeds should fire with confidence approaching 1.0.

    /// Build a uniform (zero-variance) luma frame of the given value.
    fn uniform_frame(value: u8, len: usize) -> Vec<u8> {
        vec![value; len]
    }

    /// Visible watermark: embedding strength maps to overlay brightness.
    /// `strength=0.0` -> mean 150 (below the 200 detection floor); the
    /// heuristic requires `mean > 200.0 && variance < 50.0` (see
    /// `scan_frame`), so as `strength` rises toward `1.0` the frame mean
    /// crosses 200 and confidence should increase monotonically thereafter.
    #[test]
    fn test_visible_watermark_strength_trend() {
        let strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut prev_confidence = 0.0;
        let mut saw_detection = false;

        for &strength in &strengths {
            let mean_value = (150.0 + strength * 100.0) as u8; // 150 .. 250
            let frame = uniform_frame(mean_value, 256);

            let mut det = WatermarkDetector::new(0.0);
            let result = det.scan_frame(0, &frame);
            let visible: Vec<&WatermarkSignal> = result
                .signals
                .iter()
                .filter(|s| s.watermark_type == WatermarkType::Visible)
                .collect();

            if mean_value as f64 > 200.0 {
                assert_eq!(
                    visible.len(),
                    1,
                    "strength={strength} (mean={mean_value}) should yield exactly one visible signal"
                );
                let confidence = visible[0].confidence;
                assert!(
                    confidence >= prev_confidence - 1e-9,
                    "visible watermark confidence must not decrease as embedding \
                     strength increases: strength={strength}, confidence={confidence}, \
                     prev={prev_confidence}"
                );
                prev_confidence = confidence;
                saw_detection = true;
            } else {
                assert!(
                    visible.is_empty(),
                    "strength={strength} (mean={mean_value}) is below the visible \
                     detection floor and must not yield a signal"
                );
            }
        }

        assert!(
            saw_detection,
            "at least one high-strength sample must trigger visible watermark detection"
        );
        assert!(
            prev_confidence > 0.9,
            "the strongest embedding (mean=250) should be near-maximal confidence, got {prev_confidence}"
        );
    }

    /// Invisible watermark: embedding strength maps to how far the mean is
    /// pushed down toward the "dark, low-variance" invisible-embed region
    /// (`mean < 30.0 && variance < 5.0`). Weak embeds (near-mid brightness)
    /// should not trigger; as strength increases toward `1.0` the mean falls
    /// well under the threshold and confidence should trend upward.
    #[test]
    fn test_invisible_watermark_strength_trend() {
        let strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut prev_confidence = 0.0;
        let mut saw_detection = false;

        for &strength in &strengths {
            // strength=0.0 -> mean=100 (well above threshold, no detection)
            // strength=1.0 -> mean=5   (deep in the invisible-embed zone)
            let mean_value = (100.0_f64 - strength * 95.0).round().clamp(0.0, 255.0) as u8;
            let frame = uniform_frame(mean_value, 256);

            let mut det = WatermarkDetector::new(0.0);
            let result = det.scan_frame(0, &frame);
            let invisible: Vec<&WatermarkSignal> = result
                .signals
                .iter()
                .filter(|s| s.watermark_type == WatermarkType::Invisible)
                .collect();

            if (mean_value as f64) < 30.0 {
                assert_eq!(
                    invisible.len(),
                    1,
                    "strength={strength} (mean={mean_value}) should yield exactly one invisible signal"
                );
                let confidence = invisible[0].confidence;
                assert!(
                    confidence >= prev_confidence - 1e-9,
                    "invisible watermark confidence must not decrease as embedding \
                     strength increases: strength={strength}, confidence={confidence}, \
                     prev={prev_confidence}"
                );
                prev_confidence = confidence;
                saw_detection = true;
            } else {
                assert!(
                    invisible.is_empty(),
                    "strength={strength} (mean={mean_value}) is above the invisible \
                     detection ceiling and must not yield a signal"
                );
            }
        }

        assert!(
            saw_detection,
            "at least one high-strength sample must trigger invisible watermark detection"
        );
        assert!(
            prev_confidence > 0.8,
            "the strongest embedding (mean=5) should be near-maximal confidence, got {prev_confidence}"
        );
    }

    /// Forensic (LSB) watermark: embedding strength maps to how precisely the
    /// LSB ratio is steered toward the 50/50 structured-embed signature.
    /// `diff` is how far the engineered ratio sits from 0.5; smaller `diff`
    /// means a *stronger*, more precise embed. The heuristic only fires when
    /// `diff < 0.05`, and confidence is `1.0 - diff / 0.05`.
    #[test]
    fn test_forensic_watermark_strength_trend() {
        // Ordered from weakest (largest diff, no detection) to strongest
        // (diff == 0.0, perfect 50/50 split).
        let diffs = [0.06, 0.04, 0.03, 0.02, 0.01, 0.0];
        let len = 1000usize;
        let mut prev_confidence = 0.0;
        let mut saw_detection = false;

        for &diff in &diffs {
            let ratio = 0.5 - diff;
            let odd_count = (ratio * len as f64).round() as usize;

            let mut frame = Vec::with_capacity(len);
            for i in 0..len {
                // Odd (LSB=1) for the first `odd_count` bytes, even otherwise.
                // All values kept mid-range so the visible/invisible
                // heuristics never fire and only the forensic signal is
                // exercised.
                if i < odd_count {
                    frame.push(101u8); // odd
                } else {
                    frame.push(100u8); // even
                }
            }

            let mut det = WatermarkDetector::new(0.0);
            let result = det.scan_frame(0, &frame);
            let forensic: Vec<&WatermarkSignal> = result
                .signals
                .iter()
                .filter(|s| s.watermark_type == WatermarkType::Forensic)
                .collect();

            if diff < 0.05 {
                assert_eq!(
                    forensic.len(),
                    1,
                    "diff={diff} (strong embed) should yield exactly one forensic signal"
                );
                let confidence = forensic[0].confidence;
                assert!(
                    confidence >= prev_confidence - 1e-9,
                    "forensic watermark confidence must increase as embedding \
                     precision improves (diff shrinks): diff={diff}, confidence={confidence}, \
                     prev={prev_confidence}"
                );
                prev_confidence = confidence;
                saw_detection = true;
            } else {
                assert!(
                    forensic.is_empty(),
                    "diff={diff} (weak embed) must not yield a forensic signal"
                );
            }
        }

        assert!(
            saw_detection,
            "at least one precise embed must trigger forensic watermark detection"
        );
        assert!(
            prev_confidence > 0.95,
            "a perfect 50/50 LSB split (diff=0.0) should be near-maximal confidence, got {prev_confidence}"
        );
    }
}
