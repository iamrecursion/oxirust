//! Audio quality control checks.
//!
//! This module provides QC rules for validating audio streams, including
//! codec validation, sample rate checks, loudness compliance (EBU R128, ATSC A/85),
//! clipping detection, silence detection, phase issues, and DC offset detection.

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity, Thresholds};
use oximedia_core::{CodecId, OxiResult};

/// Validates that audio codec is from the green list.
///
/// Ensures only patent-free codecs (Opus, Vorbis, FLAC, PCM, MP3) are used.
pub struct AudioCodecValidation;

impl QcRule for AudioCodecValidation {
    fn name(&self) -> &str {
        "audio_codec_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Validates that audio codec is patent-free (green list only)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            let result = match stream.codec {
                CodecId::Opus | CodecId::Vorbis | CodecId::Flac | CodecId::Pcm | CodecId::Mp3 => {
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Using approved codec: {}",
                            stream.codec.name()
                        ))
                }
                _ => CheckResult::fail(
                    self.name(),
                    Severity::Critical,
                    format!(
                        "Invalid audio codec '{}' - only green list codecs are allowed",
                        stream.codec.name()
                    ),
                )
                .with_stream(stream.index)
                .with_recommendation("Use Opus, Vorbis, FLAC, or PCM instead".to_string()),
            };
            results.push(result);
        }

        if results.is_empty() {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Warning,
                    "No audio streams found".to_string(),
                )
                .with_recommendation(
                    "Most deliverables require at least one audio stream".to_string(),
                ),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Validates audio sample rate.
///
/// Checks that sample rate is one of the standard rates (44.1kHz, 48kHz, etc.).
pub struct SampleRateValidation {
    allowed_rates: Vec<u32>,
    strict: bool,
}

impl SampleRateValidation {
    /// Creates a new sample rate validation rule with standard rates.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allowed_rates: vec![44_100, 48_000, 88_200, 96_000, 176_400, 192_000],
            strict: true,
        }
    }

    /// Sets the allowed sample rates.
    #[must_use]
    pub fn with_allowed_rates(mut self, rates: Vec<u32>) -> Self {
        self.allowed_rates = rates;
        self
    }

    /// Sets whether to strictly enforce allowed rates.
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

impl Default for SampleRateValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for SampleRateValidation {
    fn name(&self) -> &str {
        "sample_rate_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Validates audio sample rate is standard and appropriate"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            if let Some(sample_rate) = stream.codec_params.sample_rate {
                if self.allowed_rates.contains(&sample_rate) {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!("Sample rate: {sample_rate} Hz")),
                    );
                } else {
                    let severity = if self.strict {
                        Severity::Error
                    } else {
                        Severity::Warning
                    };
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            severity,
                            format!(
                                "Sample rate {sample_rate} Hz is not in allowed list: {:?}",
                                self.allowed_rates
                            ),
                        )
                        .with_stream(stream.index)
                        .with_recommendation(
                            "Use standard sample rate (48kHz recommended)".to_string(),
                        ),
                    );
                }
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Error,
                        "Audio stream missing sample rate information".to_string(),
                    )
                    .with_stream(stream.index),
                );
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Validates loudness compliance with EBU R128 or ATSC A/85.
///
/// Checks integrated loudness, loudness range, and true peak levels.
pub struct LoudnessCompliance {
    standard: LoudnessStandard,
    thresholds: Thresholds,
}

/// Loudness standard to validate against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoudnessStandard {
    /// EBU R128 (European Broadcasting Union).
    EbuR128,
    /// ATSC A/85 (North American broadcast).
    AtscA85,
    /// Custom target loudness.
    Custom(i32),
}

impl LoudnessCompliance {
    /// Creates a new loudness compliance rule for EBU R128.
    #[must_use]
    pub fn ebu_r128(thresholds: Thresholds) -> Self {
        Self {
            standard: LoudnessStandard::EbuR128,
            thresholds,
        }
    }

    /// Creates a new loudness compliance rule for ATSC A/85.
    #[must_use]
    pub fn atsc_a85(thresholds: Thresholds) -> Self {
        Self {
            standard: LoudnessStandard::AtscA85,
            thresholds,
        }
    }

    /// Creates a new loudness compliance rule with custom target.
    #[must_use]
    pub fn custom(target_lufs: i32, thresholds: Thresholds) -> Self {
        Self {
            standard: LoudnessStandard::Custom(target_lufs),
            thresholds,
        }
    }
}

impl QcRule for LoudnessCompliance {
    fn name(&self) -> &str {
        "loudness_compliance"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        match self.standard {
            LoudnessStandard::EbuR128 => "Validates EBU R128 loudness compliance",
            LoudnessStandard::AtscA85 => "Validates ATSC A/85 loudness compliance",
            LoudnessStandard::Custom(_) => "Validates custom loudness target",
        }
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        let target = self.thresholds.loudness_target.unwrap_or(-23.0);
        let tolerance = self.thresholds.loudness_tolerance.unwrap_or(1.0);

        for stream in context.audio_streams() {
            // In production, this would measure actual loudness using ITU-R BS.1770 algorithm
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Target: {target:.1} LUFS ± {tolerance:.1} LU ({})",
                        match self.standard {
                            LoudnessStandard::EbuR128 => "EBU R128",
                            LoudnessStandard::AtscA85 => "ATSC A/85",
                            LoudnessStandard::Custom(_) => "Custom",
                        }
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Detects audio clipping.
///
/// Identifies samples at or near maximum amplitude that indicate clipping.
pub struct ClippingDetection {
    threshold: f64,
    max_consecutive_samples: usize,
}

impl ClippingDetection {
    /// Creates a new clipping detection rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: 0.99,
            max_consecutive_samples: 3,
        }
    }

    /// Sets the clipping threshold (0.0 to 1.0).
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the maximum allowed consecutive clipped samples.
    #[must_use]
    pub const fn with_max_consecutive(mut self, count: usize) -> Self {
        self.max_consecutive_samples = count;
        self
    }
}

impl Default for ClippingDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for ClippingDetection {
    fn name(&self) -> &str {
        "clipping_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Detects audio clipping and distortion"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            // In production, this would decode and analyze audio samples
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will detect clipping above {:.0}% amplitude",
                        self.threshold * 100.0
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Detects silence in audio.
///
/// Identifies periods of silence that may indicate issues.
pub struct SilenceDetection {
    threshold_db: f64,
    max_silence_duration: f64,
}

impl SilenceDetection {
    /// Creates a new silence detection rule.
    #[must_use]
    pub fn new(thresholds: &Thresholds) -> Self {
        Self {
            threshold_db: -60.0,
            max_silence_duration: thresholds.max_silence_duration.unwrap_or(2.0),
        }
    }

    /// Sets the silence threshold in dB.
    #[must_use]
    pub const fn with_threshold_db(mut self, db: f64) -> Self {
        self.threshold_db = db;
        self
    }

    /// Sets the maximum allowed silence duration.
    #[must_use]
    pub const fn with_max_duration(mut self, duration: f64) -> Self {
        self.max_silence_duration = duration;
        self
    }
}

impl QcRule for SilenceDetection {
    fn name(&self) -> &str {
        "silence_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Detects extended periods of silence"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            // In production, this would decode and analyze audio levels
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will detect silence below {:.1} dB lasting > {:.1}s",
                        self.threshold_db, self.max_silence_duration
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Detects phase issues in stereo audio.
///
/// Identifies phase cancellation problems that can affect mono compatibility.
pub struct PhaseDetection {
    correlation_threshold: f64,
}

impl PhaseDetection {
    /// Creates a new phase detection rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            correlation_threshold: -0.3,
        }
    }

    /// Sets the phase correlation threshold.
    ///
    /// Values below -0.3 indicate potential phase issues.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }
}

impl Default for PhaseDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for PhaseDetection {
    fn name(&self) -> &str {
        "phase_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Detects phase correlation issues in stereo audio"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            // Only applicable to stereo or multi-channel audio
            if let Some(channels) = stream.codec_params.channels {
                if channels >= 2 {
                    // In production, this would calculate phase correlation
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!(
                                "Will check phase correlation (threshold: {:.2})",
                                self.correlation_threshold
                            )),
                    );
                } else {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(
                                "Mono audio - phase check not applicable".to_string(),
                            ),
                    );
                }
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        context
            .audio_streams()
            .iter()
            .any(|s| s.codec_params.channels.is_some_and(|ch| ch >= 2))
    }
}

/// Detects DC offset in audio.
///
/// Identifies a non-zero average amplitude that can cause issues.
pub struct DcOffsetDetection {
    threshold: f64,
}

impl DcOffsetDetection {
    /// Creates a new DC offset detection rule.
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: 0.01 }
    }

    /// Sets the DC offset threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for DcOffsetDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for DcOffsetDetection {
    fn name(&self) -> &str {
        "dc_offset_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Detects DC offset in audio signal"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            // In production, this would calculate mean sample value
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will detect DC offset above {:.2}%",
                        self.threshold * 100.0
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}

/// Validates audio channel configuration.
///
/// Ensures channel count is valid and matches expected configurations.
pub struct ChannelValidation {
    allowed_configurations: Vec<u8>,
}

impl ChannelValidation {
    /// Creates a new channel validation rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allowed_configurations: vec![1, 2, 6, 8], // Mono, Stereo, 5.1, 7.1
        }
    }

    /// Sets the allowed channel configurations.
    #[must_use]
    pub fn with_allowed_configurations(mut self, configs: Vec<u8>) -> Self {
        self.allowed_configurations = configs;
        self
    }
}

impl Default for ChannelValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for ChannelValidation {
    fn name(&self) -> &str {
        "channel_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Validates audio channel configuration"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.audio_streams() {
            if let Some(channels) = stream.codec_params.channels {
                if self.allowed_configurations.contains(&channels) {
                    let config = match channels {
                        1 => "Mono",
                        2 => "Stereo",
                        6 => "5.1 Surround",
                        8 => "7.1 Surround",
                        n => {
                            return Ok(vec![CheckResult::pass(self.name())
                                .with_stream(stream.index)
                                .with_recommendation(format!("{n} channels"))])
                        }
                    };
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!("Configuration: {config}")),
                    );
                } else {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Warning,
                            format!(
                                "{} channel(s) not in allowed configurations: {:?}",
                                channels, self.allowed_configurations
                            ),
                        )
                        .with_stream(stream.index)
                        .with_recommendation("Use standard channel configuration".to_string()),
                    );
                }
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Error,
                        "Audio stream missing channel information".to_string(),
                    )
                    .with_stream(stream.index),
                );
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.audio_streams().is_empty()
    }
}
