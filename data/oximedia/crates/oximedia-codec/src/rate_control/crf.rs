//! Constant Rate Factor rate control.
//!
//! CRF (Constant Rate Factor) is a quality-based rate control mode that
//! attempts to maintain consistent perceptual quality throughout the encode.
//! It adjusts the QP based on frame complexity to achieve this goal.
//!
//! # CRF Values
//!
//! - 0: Lossless (best quality, largest files)
//! - 18-23: Visually lossless range
//! - 23-28: Good quality (recommended)
//! - 28-35: Medium quality
//! - 35-51: Low quality
//! - 63: Maximum compression (worst quality)

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

use super::types::{FrameStats, RcConfig, RcOutput};

/// Constant Rate Factor controller.
///
/// Maintains consistent perceptual quality by adjusting QP based on
/// frame complexity.
#[derive(Clone, Debug)]
pub struct CrfController {
    /// Base CRF value.
    crf: f32,
    /// Current QP.
    current_qp: f32,
    /// Minimum QP.
    min_qp: u8,
    /// Maximum QP.
    max_qp: u8,
    /// I-frame QP offset.
    i_qp_offset: i8,
    /// B-frame QP offset.
    b_qp_offset: i8,
    /// Enable adaptive quantization integration.
    enable_aq: bool,
    /// AQ strength.
    aq_strength: f32,
    /// Frame counter.
    frame_count: u64,
    /// Total bits encoded.
    total_bits: u64,
    /// Running average complexity.
    average_complexity: f32,
    /// Complexity weight for averaging.
    complexity_weight: f32,
    /// QP adjustment history for smoothing.
    qp_history: Vec<f32>,
    /// Maximum QP history size.
    max_qp_history: usize,
}

impl CrfController {
    /// Create a new CRF controller from configuration.
    #[must_use]
    pub fn new(config: &RcConfig) -> Self {
        Self {
            crf: config.crf.clamp(0.0, 63.0),
            current_qp: config.crf,
            min_qp: config.min_qp,
            max_qp: config.max_qp,
            i_qp_offset: config.i_qp_offset,
            b_qp_offset: config.b_qp_offset,
            enable_aq: config.enable_aq,
            aq_strength: config.aq_strength,
            frame_count: 0,
            total_bits: 0,
            average_complexity: 1.0,
            complexity_weight: 0.1,
            qp_history: Vec::with_capacity(30),
            max_qp_history: 30,
        }
    }

    /// Create a CRF controller with a specific CRF value.
    #[must_use]
    pub fn with_crf(crf: f32) -> Self {
        Self {
            crf: crf.clamp(0.0, 63.0),
            current_qp: crf,
            ..Default::default()
        }
    }

    /// Set the CRF value.
    pub fn set_crf(&mut self, crf: f32) {
        self.crf = crf.clamp(0.0, 63.0);
    }

    /// Enable or disable adaptive quantization.
    pub fn set_enable_aq(&mut self, enable: bool) {
        self.enable_aq = enable;
    }

    /// Set AQ strength.
    pub fn set_aq_strength(&mut self, strength: f32) {
        self.aq_strength = strength.clamp(0.0, 2.0);
    }

    /// Get rate control output for a frame.
    #[must_use]
    pub fn get_rc(&self, frame_type: FrameType, complexity: f32) -> RcOutput {
        // Map CRF to base QP
        let base_qp = self.crf_to_qp(self.crf);

        // Apply complexity-based adjustment
        let complexity_adjustment = self.calculate_complexity_adjustment(complexity);

        // Apply frame type offset
        let offset = match frame_type {
            FrameType::Key => self.i_qp_offset,
            FrameType::BiDir => self.b_qp_offset,
            FrameType::Inter | FrameType::Switch => 0,
        };

        let final_qp = (base_qp + complexity_adjustment + offset as f32)
            .clamp(self.min_qp as f32, self.max_qp as f32);
        let qp = final_qp.round() as u8;

        let mut output = RcOutput {
            qp,
            qp_f: final_qp,
            ..Default::default()
        };
        output.compute_lambda();

        // If AQ is enabled, we would add qp_offsets here
        // (handled by the AQ module externally)

        output
    }

    /// Map CRF value to QP.
    ///
    /// CRF and QP have a roughly linear relationship, but CRF accounts
    /// for frame complexity whereas QP is absolute.
    fn crf_to_qp(&self, crf: f32) -> f32 {
        // For most codecs, CRF maps directly to QP as a baseline
        // The actual QP varies based on frame content
        crf
    }

    /// Calculate QP to CRF mapping (inverse).
    #[must_use]
    pub fn qp_to_crf(&self, qp: f32) -> f32 {
        qp
    }

    /// Calculate complexity-based QP adjustment.
    fn calculate_complexity_adjustment(&self, complexity: f32) -> f32 {
        if complexity <= 0.0 || self.average_complexity <= 0.0 {
            return 0.0;
        }

        // Calculate ratio to average complexity
        let ratio = complexity / self.average_complexity;

        // Higher complexity -> higher QP (fewer bits per pixel)
        // Lower complexity -> lower QP (more bits per pixel, better quality)
        //
        // The adjustment is logarithmic to smooth extreme values
        let log_ratio = ratio.ln();

        // Scale the adjustment
        // Typical range: -3 to +3 QP
        let adjustment = log_ratio * 2.0 * self.aq_strength;
        adjustment.clamp(-4.0, 4.0)
    }

    /// Update controller with frame statistics.
    pub fn update(&mut self, stats: &FrameStats) {
        self.frame_count += 1;
        self.total_bits += stats.bits;

        // Update running average complexity
        if stats.complexity > 0.0 {
            self.average_complexity = self.average_complexity * (1.0 - self.complexity_weight)
                + stats.complexity * self.complexity_weight;
        }

        // Update QP history
        self.qp_history.push(stats.qp_f);
        if self.qp_history.len() > self.max_qp_history {
            self.qp_history.remove(0);
        }

        self.current_qp = stats.qp_f;
    }

    /// Get the CRF value.
    #[must_use]
    pub fn crf(&self) -> f32 {
        self.crf
    }

    /// Get current QP.
    #[must_use]
    pub fn current_qp(&self) -> f32 {
        self.current_qp
    }

    /// Get average QP from history.
    #[must_use]
    pub fn average_qp(&self) -> f32 {
        if self.qp_history.is_empty() {
            return self.crf;
        }
        self.qp_history.iter().sum::<f32>() / self.qp_history.len() as f32
    }

    /// Get average complexity.
    #[must_use]
    pub fn average_complexity(&self) -> f32 {
        self.average_complexity
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get total bits produced.
    #[must_use]
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Estimate bitrate at current quality.
    #[must_use]
    pub fn estimated_bitrate(&self, fps: f64) -> f64 {
        if self.frame_count == 0 || fps <= 0.0 {
            return 0.0;
        }

        let avg_bits_per_frame = self.total_bits as f64 / self.frame_count as f64;
        avg_bits_per_frame * fps
    }

    /// Get QP variance (quality stability metric).
    #[must_use]
    pub fn qp_variance(&self) -> f32 {
        if self.qp_history.len() < 2 {
            return 0.0;
        }

        let avg = self.average_qp();
        let variance: f32 = self
            .qp_history
            .iter()
            .map(|qp| (qp - avg).powi(2))
            .sum::<f32>()
            / self.qp_history.len() as f32;

        variance.sqrt()
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.total_bits = 0;
        self.current_qp = self.crf;
        self.average_complexity = 1.0;
        self.qp_history.clear();
    }

    /// Calculate lambda for RDO from CRF.
    #[must_use]
    pub fn crf_to_lambda(&self) -> f64 {
        RcOutput::qp_to_lambda(self.crf)
    }
}

impl Default for CrfController {
    fn default() -> Self {
        Self {
            crf: 23.0,
            current_qp: 23.0,
            min_qp: 1,
            max_qp: 63,
            i_qp_offset: -2,
            b_qp_offset: 2,
            enable_aq: true,
            aq_strength: 1.0,
            frame_count: 0,
            total_bits: 0,
            average_complexity: 1.0,
            complexity_weight: 0.1,
            qp_history: Vec::with_capacity(30),
            max_qp_history: 30,
        }
    }
}

/// Quality presets for CRF encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityPreset {
    /// Highest quality, largest files.
    Lossless,
    /// Visually lossless, very high quality.
    VisuallyLossless,
    /// High quality (recommended for archival).
    High,
    /// Good quality (recommended for general use).
    Good,
    /// Medium quality (balanced).
    Medium,
    /// Low quality (for size constraints).
    Low,
    /// Lowest quality, smallest files.
    Minimum,
}

impl QualityPreset {
    /// Get CRF value for this preset.
    #[must_use]
    pub fn to_crf(self) -> f32 {
        match self {
            Self::Lossless => 0.0,
            Self::VisuallyLossless => 18.0,
            Self::High => 20.0,
            Self::Good => 23.0,
            Self::Medium => 28.0,
            Self::Low => 35.0,
            Self::Minimum => 51.0,
        }
    }

    /// Create a CRF controller with this preset.
    #[must_use]
    pub fn to_controller(self) -> CrfController {
        CrfController::with_crf(self.to_crf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crf_creation() {
        let config = RcConfig::crf(23.0);
        let controller = CrfController::new(&config);
        assert!((controller.crf() - 23.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crf_with_value() {
        let controller = CrfController::with_crf(28.0);
        assert!((controller.crf() - 28.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_rc_i_frame() {
        let controller = CrfController::default();
        let output = controller.get_rc(FrameType::Key, 1.0);

        assert!(!output.drop_frame);
        assert!(output.qp > 0);
    }

    #[test]
    fn test_frame_type_qp_offsets() {
        let mut controller = CrfController::default();
        controller.i_qp_offset = -4;
        controller.b_qp_offset = 4;

        let i_output = controller.get_rc(FrameType::Key, 1.0);
        let p_output = controller.get_rc(FrameType::Inter, 1.0);
        let b_output = controller.get_rc(FrameType::BiDir, 1.0);

        assert!(i_output.qp < p_output.qp);
        assert!(b_output.qp > p_output.qp);
    }

    #[test]
    fn test_complexity_adjustment() {
        let controller = CrfController::default();

        let low_complexity = controller.get_rc(FrameType::Inter, 0.5);
        let high_complexity = controller.get_rc(FrameType::Inter, 2.0);

        // Higher complexity should get higher QP
        assert!(high_complexity.qp_f > low_complexity.qp_f);
    }

    #[test]
    fn test_statistics_update() {
        let mut controller = CrfController::default();

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        stats.qp_f = 24.0;
        stats.complexity = 1.2;
        controller.update(&stats);

        assert_eq!(controller.frame_count(), 1);
        assert_eq!(controller.total_bits(), 100_000);
    }

    #[test]
    fn test_average_qp() {
        let mut controller = CrfController::default();

        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.qp_f = 24.0 + (i as f32 * 0.5);
            controller.update(&stats);
        }

        let avg = controller.average_qp();
        // Average of 24.0, 24.5, 25.0, ..., 28.5 = 26.25
        assert!((avg - 26.25).abs() < 0.1);
    }

    #[test]
    fn test_qp_variance() {
        let mut controller = CrfController::default();

        // Constant QP should have zero variance
        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.qp_f = 24.0;
            controller.update(&stats);
        }

        assert!(controller.qp_variance() < 0.01);

        // Reset and add varying QP
        controller.reset();
        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.qp_f = 20.0 + (i as f32 * 2.0);
            controller.update(&stats);
        }

        assert!(controller.qp_variance() > 1.0);
    }

    #[test]
    fn test_quality_presets() {
        assert!((QualityPreset::Lossless.to_crf() - 0.0).abs() < f32::EPSILON);
        assert!((QualityPreset::Good.to_crf() - 23.0).abs() < f32::EPSILON);
        assert!((QualityPreset::Minimum.to_crf() - 51.0).abs() < f32::EPSILON);

        let controller = QualityPreset::High.to_controller();
        assert!((controller.crf() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lambda_calculation() {
        let controller = CrfController::default();
        let lambda = controller.crf_to_lambda();
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut controller = CrfController::default();

        for i in 0..10 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 100_000;
            stats.qp_f = 24.0;
            stats.complexity = 1.0;
            controller.update(&stats);
        }

        controller.reset();

        assert_eq!(controller.frame_count(), 0);
        assert_eq!(controller.total_bits(), 0);
        assert!(controller.qp_history.is_empty());
    }

    #[test]
    fn test_estimated_bitrate() {
        let mut controller = CrfController::default();

        for i in 0..30 {
            let mut stats = FrameStats::new(i, FrameType::Inter);
            stats.bits = 100_000; // 100 kb per frame
            controller.update(&stats);
        }

        // At 30 fps, should be approximately 3 Mbps
        let bitrate = controller.estimated_bitrate(30.0);
        assert!((bitrate - 3_000_000.0).abs() < 1.0);
    }
}
