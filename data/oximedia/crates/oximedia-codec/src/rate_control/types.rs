//! Rate control types and data structures.
//!
//! This module defines the core types used throughout the rate control system:
//! - Configuration structures for different rate control modes
//! - Statistics structures for frames and GOPs
//! - Output structures for rate control decisions

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::double_must_use)]
#![allow(clippy::match_same_arms)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

/// Rate control mode selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RateControlMode {
    /// Constant Quantization Parameter.
    /// Uses a fixed QP for all frames.
    Cqp,
    /// Constant Bitrate.
    /// Maintains a steady bitrate over time.
    Cbr,
    /// Variable Bitrate.
    /// Allows bitrate to vary while staying within bounds.
    Vbr,
    /// Average Bitrate.
    /// Targets an average bitrate over the entire encode.
    Abr,
    /// Constant Rate Factor (quality-based).
    /// Maintains consistent perceptual quality.
    #[default]
    Crf,
}

impl RateControlMode {
    /// Returns true if this mode targets a specific bitrate.
    #[must_use]
    pub fn is_bitrate_based(&self) -> bool {
        matches!(self, Self::Cbr | Self::Vbr | Self::Abr)
    }

    /// Returns true if this mode targets consistent quality.
    #[must_use]
    pub fn is_quality_based(&self) -> bool {
        matches!(self, Self::Cqp | Self::Crf)
    }
}

/// Rate control configuration.
#[derive(Clone, Debug)]
pub struct RcConfig {
    /// Rate control mode.
    pub mode: RateControlMode,
    /// Target bitrate in bits per second.
    pub target_bitrate: u64,
    /// Maximum bitrate in bits per second (for VBR).
    pub max_bitrate: Option<u64>,
    /// Minimum bitrate in bits per second (for VBR).
    pub min_bitrate: Option<u64>,
    /// Minimum QP value (higher quality limit).
    pub min_qp: u8,
    /// Maximum QP value (lower quality limit).
    pub max_qp: u8,
    /// Initial QP for the first frame.
    pub initial_qp: u8,
    /// CRF value for quality-based modes (0-63).
    pub crf: f32,
    /// VBV/HRD buffer size in bits.
    pub buffer_size: u64,
    /// Initial buffer fullness as a fraction (0.0-1.0).
    pub initial_buffer_fullness: f32,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
    /// GOP (keyframe interval) length.
    pub gop_length: u32,
    /// Enable adaptive quantization.
    pub enable_aq: bool,
    /// AQ strength (0.0-2.0).
    pub aq_strength: f32,
    /// Lookahead depth (number of frames).
    pub lookahead_depth: usize,
    /// B-frame QP offset from P-frame.
    pub b_qp_offset: i8,
    /// I-frame QP offset from P-frame (typically negative).
    pub i_qp_offset: i8,
    /// Enable scene cut detection.
    pub scene_cut_detection: bool,
    /// Scene cut threshold (0.0-1.0).
    pub scene_cut_threshold: f32,
}

impl Default for RcConfig {
    fn default() -> Self {
        Self {
            mode: RateControlMode::Crf,
            target_bitrate: 5_000_000, // 5 Mbps
            max_bitrate: None,
            min_bitrate: None,
            min_qp: 1,
            max_qp: 63,
            initial_qp: 28,
            crf: 23.0,
            buffer_size: 10_000_000, // 10 Mb
            initial_buffer_fullness: 0.75,
            framerate_num: 30,
            framerate_den: 1,
            gop_length: 250,
            enable_aq: true,
            aq_strength: 1.0,
            lookahead_depth: 40,
            b_qp_offset: 2,
            i_qp_offset: -2,
            scene_cut_detection: true,
            scene_cut_threshold: 0.4,
        }
    }
}

impl RcConfig {
    /// Create a CQP configuration.
    #[must_use]
    pub fn cqp(qp: u8) -> Self {
        Self {
            mode: RateControlMode::Cqp,
            initial_qp: qp,
            ..Default::default()
        }
    }

    /// Create a CBR configuration.
    #[must_use]
    pub fn cbr(bitrate: u64) -> Self {
        Self {
            mode: RateControlMode::Cbr,
            target_bitrate: bitrate,
            max_bitrate: Some(bitrate),
            buffer_size: bitrate, // 1 second buffer
            ..Default::default()
        }
    }

    /// Create a VBR configuration.
    #[must_use]
    pub fn vbr(target: u64, max: u64) -> Self {
        Self {
            mode: RateControlMode::Vbr,
            target_bitrate: target,
            max_bitrate: Some(max),
            buffer_size: max * 2, // 2 second buffer at max rate
            ..Default::default()
        }
    }

    /// Create a CRF configuration.
    #[must_use]
    pub fn crf(crf_value: f32) -> Self {
        Self {
            mode: RateControlMode::Crf,
            crf: crf_value.clamp(0.0, 63.0),
            ..Default::default()
        }
    }

    /// Get the frame rate as a floating point value.
    #[must_use]
    pub fn framerate(&self) -> f64 {
        if self.framerate_den == 0 {
            30.0
        } else {
            f64::from(self.framerate_num) / f64::from(self.framerate_den)
        }
    }

    /// Get the target bits per frame.
    #[must_use]
    pub fn target_bits_per_frame(&self) -> u64 {
        let fps = self.framerate();
        if fps <= 0.0 {
            return 0;
        }
        (self.target_bitrate as f64 / fps) as u64
    }

    /// Validate configuration parameters.
    #[must_use]
    pub fn validate(&self) -> Result<(), RcConfigError> {
        if self.min_qp > self.max_qp {
            return Err(RcConfigError::InvalidQpRange);
        }
        if self.initial_qp < self.min_qp || self.initial_qp > self.max_qp {
            return Err(RcConfigError::InitialQpOutOfRange);
        }
        if self.crf < 0.0 || self.crf > 63.0 {
            return Err(RcConfigError::InvalidCrf);
        }
        if self.framerate_den == 0 {
            return Err(RcConfigError::InvalidFramerate);
        }
        if self.mode.is_bitrate_based() && self.target_bitrate == 0 {
            return Err(RcConfigError::ZeroBitrate);
        }
        Ok(())
    }
}

/// Configuration validation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RcConfigError {
    /// `min_qp` is greater than `max_qp`.
    InvalidQpRange,
    /// `initial_qp` is outside `min_qp/max_qp` range.
    InitialQpOutOfRange,
    /// CRF value is out of valid range.
    InvalidCrf,
    /// Frame rate denominator is zero.
    InvalidFramerate,
    /// Target bitrate is zero for bitrate-based mode.
    ZeroBitrate,
}

/// Statistics for a single encoded frame.
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// Frame number (display order).
    pub frame_num: u64,
    /// Frame type (I/P/B).
    pub frame_type: FrameType,
    /// Bits used for this frame.
    pub bits: u64,
    /// QP used for this frame.
    pub qp: u8,
    /// QP as floating point for averaging.
    pub qp_f: f32,
    /// Spatial complexity estimate.
    pub spatial_complexity: f32,
    /// Temporal complexity estimate (motion).
    pub temporal_complexity: f32,
    /// Combined complexity metric.
    pub complexity: f32,
    /// PSNR if calculated.
    pub psnr: Option<f32>,
    /// SSIM if calculated.
    pub ssim: Option<f32>,
    /// Was this frame a scene cut.
    pub scene_cut: bool,
    /// Target bits for this frame.
    pub target_bits: u64,
    /// Actual bits / target bits ratio.
    pub bit_accuracy: f32,
    /// Encoding time in microseconds.
    pub encode_time_us: u64,
}

impl FrameStats {
    /// Create new frame statistics.
    #[must_use]
    pub fn new(frame_num: u64, frame_type: FrameType) -> Self {
        Self {
            frame_num,
            frame_type,
            ..Default::default()
        }
    }

    /// Calculate bits per pixel.
    #[must_use]
    pub fn bits_per_pixel(&self, width: u32, height: u32) -> f32 {
        let pixels = u64::from(width) * u64::from(height);
        if pixels == 0 {
            return 0.0;
        }
        self.bits as f32 / pixels as f32
    }

    /// Check if frame exceeded target significantly.
    #[must_use]
    pub fn exceeded_target(&self, threshold: f32) -> bool {
        self.target_bits > 0 && self.bit_accuracy > (1.0 + threshold)
    }

    /// Check if frame was under target significantly.
    #[must_use]
    pub fn under_target(&self, threshold: f32) -> bool {
        self.target_bits > 0 && self.bit_accuracy < (1.0 - threshold)
    }
}

/// Statistics for a Group of Pictures (GOP).
#[derive(Clone, Debug, Default)]
pub struct GopStats {
    /// GOP index.
    pub gop_index: u64,
    /// Number of frames in this GOP.
    pub frame_count: u32,
    /// Number of I-frames.
    pub i_frame_count: u32,
    /// Number of P-frames.
    pub p_frame_count: u32,
    /// Number of B-frames.
    pub b_frame_count: u32,
    /// Total bits for this GOP.
    pub total_bits: u64,
    /// Target bits for this GOP.
    pub target_bits: u64,
    /// Average QP across all frames.
    pub average_qp: f32,
    /// Average complexity.
    pub average_complexity: f32,
    /// Total complexity (sum of frame complexities).
    pub total_complexity: f32,
    /// First frame number in this GOP.
    pub first_frame: u64,
    /// Last frame number in this GOP.
    pub last_frame: u64,
    /// Accumulated frame statistics.
    frames: Vec<FrameStats>,
}

impl GopStats {
    /// Create new GOP statistics.
    #[must_use]
    pub fn new(gop_index: u64, first_frame: u64) -> Self {
        Self {
            gop_index,
            first_frame,
            last_frame: first_frame,
            ..Default::default()
        }
    }

    /// Add frame statistics to this GOP.
    pub fn add_frame(&mut self, stats: FrameStats) {
        self.last_frame = stats.frame_num;
        self.total_bits += stats.bits;
        self.total_complexity += stats.complexity;

        match stats.frame_type {
            FrameType::Key => self.i_frame_count += 1,
            FrameType::Inter => self.p_frame_count += 1,
            FrameType::BiDir => self.b_frame_count += 1,
            FrameType::Switch => self.p_frame_count += 1,
        }

        self.frames.push(stats);
        self.frame_count = self.frames.len() as u32;

        // Update averages
        if self.frame_count > 0 {
            let fc = self.frame_count as f32;
            self.average_qp = self.frames.iter().map(|f| f.qp_f).sum::<f32>() / fc;
            self.average_complexity = self.total_complexity / fc;
        }
    }

    /// Get frame statistics.
    #[must_use]
    pub fn frames(&self) -> &[FrameStats] {
        &self.frames
    }

    /// Get bit accuracy for this GOP.
    #[must_use]
    pub fn bit_accuracy(&self) -> f32 {
        if self.target_bits == 0 {
            return 1.0;
        }
        self.total_bits as f32 / self.target_bits as f32
    }

    /// Get bits per I-frame.
    #[must_use]
    pub fn bits_per_i_frame(&self) -> f64 {
        if self.i_frame_count == 0 {
            return 0.0;
        }
        let i_bits: u64 = self
            .frames
            .iter()
            .filter(|f| f.frame_type == FrameType::Key)
            .map(|f| f.bits)
            .sum();
        i_bits as f64 / f64::from(self.i_frame_count)
    }

    /// Get bits per P-frame.
    #[must_use]
    pub fn bits_per_p_frame(&self) -> f64 {
        if self.p_frame_count == 0 {
            return 0.0;
        }
        let p_bits: u64 = self
            .frames
            .iter()
            .filter(|f| f.frame_type == FrameType::Inter)
            .map(|f| f.bits)
            .sum();
        p_bits as f64 / f64::from(self.p_frame_count)
    }

    /// Get bits per B-frame.
    #[must_use]
    pub fn bits_per_b_frame(&self) -> f64 {
        if self.b_frame_count == 0 {
            return 0.0;
        }
        let b_bits: u64 = self
            .frames
            .iter()
            .filter(|f| f.frame_type == FrameType::BiDir)
            .map(|f| f.bits)
            .sum();
        b_bits as f64 / f64::from(self.b_frame_count)
    }
}

/// Rate control output for a single frame.
#[derive(Clone, Debug, Default)]
pub struct RcOutput {
    /// Quantization parameter to use.
    pub qp: u8,
    /// QP as floating point (before rounding).
    pub qp_f: f32,
    /// Target bits for this frame.
    pub target_bits: u64,
    /// Minimum bits (for underflow prevention).
    pub min_bits: u64,
    /// Maximum bits (for overflow prevention).
    pub max_bits: u64,
    /// Whether to drop this frame.
    pub drop_frame: bool,
    /// Whether to force a keyframe.
    pub force_keyframe: bool,
    /// Block-level QP offsets for AQ.
    pub qp_offsets: Option<Vec<f32>>,
    /// Lambda for RDO (rate-distortion optimization).
    pub lambda: f64,
    /// Lambda for motion estimation.
    pub lambda_me: f64,
}

impl RcOutput {
    /// Create a new rate control output with the given QP.
    #[must_use]
    pub fn with_qp(qp: u8) -> Self {
        Self {
            qp,
            qp_f: qp as f32,
            ..Default::default()
        }
    }

    /// Create output that drops the frame.
    #[must_use]
    pub fn drop() -> Self {
        Self {
            drop_frame: true,
            ..Default::default()
        }
    }

    /// Calculate lambda from QP using standard formula.
    /// Lambda = 0.85 * 2^((QP-12)/3)
    #[must_use]
    pub fn qp_to_lambda(qp: f32) -> f64 {
        0.85 * 2.0_f64.powf((f64::from(qp) - 12.0) / 3.0)
    }

    /// Calculate motion estimation lambda from lambda.
    #[must_use]
    pub fn lambda_to_lambda_me(lambda: f64) -> f64 {
        lambda.sqrt()
    }

    /// Set lambda values based on QP.
    pub fn compute_lambda(&mut self) {
        self.lambda = Self::qp_to_lambda(self.qp_f);
        self.lambda_me = Self::lambda_to_lambda_me(self.lambda);
    }
}

/// Rate control state summary.
#[derive(Clone, Debug, Default)]
pub struct RcState {
    /// Total frames processed.
    pub frames_encoded: u64,
    /// Total bits produced.
    pub total_bits: u64,
    /// Current buffer level.
    pub buffer_level: i64,
    /// Average bitrate so far.
    pub average_bitrate: f64,
    /// Average QP so far.
    pub average_qp: f32,
    /// Frames dropped due to rate control.
    pub frames_dropped: u64,
    /// Keyframes forced by scene detection.
    pub scene_cuts_detected: u64,
    /// Current GOP index.
    pub current_gop: u64,
}

impl RcState {
    /// Update state with new frame statistics.
    pub fn update(&mut self, stats: &FrameStats, elapsed_time: f64) {
        self.frames_encoded += 1;
        self.total_bits += stats.bits;

        if elapsed_time > 0.0 {
            self.average_bitrate = self.total_bits as f64 / elapsed_time;
        }

        // Running average of QP
        let n = self.frames_encoded as f32;
        self.average_qp = self.average_qp * (n - 1.0) / n + stats.qp_f / n;

        if stats.scene_cut {
            self.scene_cuts_detected += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_control_mode() {
        assert!(RateControlMode::Cbr.is_bitrate_based());
        assert!(RateControlMode::Vbr.is_bitrate_based());
        assert!(RateControlMode::Abr.is_bitrate_based());
        assert!(!RateControlMode::Cqp.is_bitrate_based());
        assert!(!RateControlMode::Crf.is_bitrate_based());

        assert!(RateControlMode::Cqp.is_quality_based());
        assert!(RateControlMode::Crf.is_quality_based());
        assert!(!RateControlMode::Cbr.is_quality_based());
    }

    #[test]
    fn test_rc_config_creation() {
        let config = RcConfig::cqp(28);
        assert_eq!(config.mode, RateControlMode::Cqp);
        assert_eq!(config.initial_qp, 28);

        let config = RcConfig::cbr(5_000_000);
        assert_eq!(config.mode, RateControlMode::Cbr);
        assert_eq!(config.target_bitrate, 5_000_000);

        let config = RcConfig::vbr(5_000_000, 10_000_000);
        assert_eq!(config.mode, RateControlMode::Vbr);
        assert_eq!(config.max_bitrate, Some(10_000_000));

        let config = RcConfig::crf(23.0);
        assert_eq!(config.mode, RateControlMode::Crf);
        assert!((config.crf - 23.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rc_config_validation() {
        let mut config = RcConfig::default();
        assert!(config.validate().is_ok());

        config.min_qp = 50;
        config.max_qp = 30;
        assert_eq!(config.validate(), Err(RcConfigError::InvalidQpRange));

        config.min_qp = 10;
        config.max_qp = 40;
        config.initial_qp = 5;
        assert_eq!(config.validate(), Err(RcConfigError::InitialQpOutOfRange));

        config.initial_qp = 25;
        config.crf = 100.0;
        assert_eq!(config.validate(), Err(RcConfigError::InvalidCrf));

        config.crf = 23.0;
        config.framerate_den = 0;
        assert_eq!(config.validate(), Err(RcConfigError::InvalidFramerate));

        config.framerate_den = 1;
        config.mode = RateControlMode::Cbr;
        config.target_bitrate = 0;
        assert_eq!(config.validate(), Err(RcConfigError::ZeroBitrate));
    }

    #[test]
    fn test_target_bits_per_frame() {
        let config = RcConfig {
            target_bitrate: 3_000_000,
            framerate_num: 30,
            framerate_den: 1,
            ..Default::default()
        };
        assert_eq!(config.target_bits_per_frame(), 100_000);

        let config = RcConfig {
            target_bitrate: 6_000_000,
            framerate_num: 60,
            framerate_den: 1,
            ..Default::default()
        };
        assert_eq!(config.target_bits_per_frame(), 100_000);
    }

    #[test]
    fn test_frame_stats() {
        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        stats.target_bits = 80_000;
        stats.bit_accuracy = stats.bits as f32 / stats.target_bits as f32;

        assert!(stats.exceeded_target(0.1));
        assert!(!stats.under_target(0.1));

        let bpp = stats.bits_per_pixel(1920, 1080);
        assert!(bpp > 0.0);
    }

    #[test]
    fn test_gop_stats() {
        let mut gop = GopStats::new(0, 0);

        let mut i_frame = FrameStats::new(0, FrameType::Key);
        i_frame.bits = 200_000;
        i_frame.qp_f = 24.0;
        i_frame.complexity = 1.5;
        gop.add_frame(i_frame);

        let mut p_frame = FrameStats::new(1, FrameType::Inter);
        p_frame.bits = 50_000;
        p_frame.qp_f = 26.0;
        p_frame.complexity = 1.0;
        gop.add_frame(p_frame);

        assert_eq!(gop.frame_count, 2);
        assert_eq!(gop.i_frame_count, 1);
        assert_eq!(gop.p_frame_count, 1);
        assert_eq!(gop.total_bits, 250_000);
        assert!((gop.average_qp - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rc_output_lambda() {
        let lambda = RcOutput::qp_to_lambda(28.0);
        assert!(lambda > 0.0);

        let lambda_me = RcOutput::lambda_to_lambda_me(lambda);
        assert!((lambda_me - lambda.sqrt()).abs() < f64::EPSILON);

        let mut output = RcOutput::with_qp(28);
        output.compute_lambda();
        assert!(output.lambda > 0.0);
        assert!(output.lambda_me > 0.0);
    }

    #[test]
    fn test_rc_state_update() {
        let mut state = RcState::default();
        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        stats.qp_f = 28.0;

        state.update(&stats, 1.0 / 30.0);

        assert_eq!(state.frames_encoded, 1);
        assert_eq!(state.total_bits, 100_000);
        assert!((state.average_qp - 28.0).abs() < f32::EPSILON);
    }
}
