//! Preset compatibility matrix — validate source media against preset requirements.
//!
//! This module models the requirements that each preset places on its input
//! source and provides a structured compatibility check so that callers can
//! know *before* encoding whether a given source will work with a given preset.
//!
//! # Core concepts
//!
//! * [`SourceMediaSpec`] — describes the properties of the source media to be
//!   encoded (resolution, frame rate, codec, audio sample rate, etc.).
//! * [`PresetRequirements`] — the constraints a preset imposes on the source.
//!   Each constraint is a closed range or an enumerated set of allowed values.
//! * [`CompatibilityMatrix`] — a registry that stores [`PresetRequirements`]
//!   keyed by preset ID, and provides [`CompatibilityMatrix::check`] to
//!   evaluate a source against a preset.
//! * [`CompatibilityReport`] — the result of a compatibility check, listing
//!   which constraints passed and which failed (with human-readable messages).
//!
//! # Example
//!
//! ```rust
//! use oximedia_presets::preset_compatibility::{
//!     CompatibilityMatrix, PresetRequirements, SourceMediaSpec,
//! };
//!
//! let mut matrix = CompatibilityMatrix::new();
//! let reqs = PresetRequirements::new("youtube-1080p")
//!     .with_max_width(3840)
//!     .with_max_height(2160)
//!     .with_allowed_container("mp4")
//!     .with_allowed_container("mov");
//! matrix.register("youtube-1080p", reqs);
//!
//! let source = SourceMediaSpec::new()
//!     .with_resolution(1920, 1080)
//!     .with_container("mp4");
//!
//! let report = matrix.check("youtube-1080p", &source).unwrap();
//! assert!(report.is_compatible());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by [`CompatibilityMatrix`].
#[derive(Debug, Error, Clone)]
pub enum CompatibilityError {
    /// No requirements registered for the requested preset ID.
    #[error("No requirements registered for preset: {0}")]
    UnknownPreset(String),
}

// ── SourceMediaSpec ─────────────────────────────────────────────────────────

/// Properties of a source media asset that will be fed into an encoding preset.
///
/// All fields are optional; unset fields are treated as "unknown" and are
/// skipped during compatibility checking.
#[derive(Debug, Clone, Default)]
pub struct SourceMediaSpec {
    /// Width of the video in pixels.
    pub width: Option<u32>,
    /// Height of the video in pixels.
    pub height: Option<u32>,
    /// Frame rate as `(numerator, denominator)`.
    pub frame_rate: Option<(u32, u32)>,
    /// Video codec name (e.g. `"h264"`, `"hevc"`, `"prores"`).
    pub video_codec: Option<String>,
    /// Audio codec name (e.g. `"aac"`, `"pcm_s16le"`).
    pub audio_codec: Option<String>,
    /// Container format (e.g. `"mp4"`, `"mov"`, `"mkv"`).
    pub container: Option<String>,
    /// Video bit depth (e.g. `8`, `10`, `12`).
    pub bit_depth: Option<u8>,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: Option<u32>,
    /// Number of audio channels.
    pub audio_channels: Option<u8>,
    /// Duration of the source in seconds.
    pub duration_secs: Option<f64>,
    /// Approximate video bitrate of the source in bits per second.
    pub video_bitrate: Option<u64>,
}

impl SourceMediaSpec {
    /// Create a new, empty spec.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set the frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, numerator: u32, denominator: u32) -> Self {
        self.frame_rate = Some((numerator, denominator));
        self
    }

    /// Set the video codec.
    #[must_use]
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = Some(codec.into());
        self
    }

    /// Set the audio codec.
    #[must_use]
    pub fn with_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codec = Some(codec.into());
        self
    }

    /// Set the container format.
    #[must_use]
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = Some(container.into());
        self
    }

    /// Set the bit depth.
    #[must_use]
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = Some(bit_depth);
        self
    }

    /// Set the audio sample rate.
    #[must_use]
    pub fn with_audio_sample_rate(mut self, sample_rate: u32) -> Self {
        self.audio_sample_rate = Some(sample_rate);
        self
    }

    /// Set the number of audio channels.
    #[must_use]
    pub fn with_audio_channels(mut self, channels: u8) -> Self {
        self.audio_channels = Some(channels);
        self
    }

    /// Set the source duration in seconds.
    #[must_use]
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = Some(secs);
        self
    }

    /// Set the source video bitrate.
    #[must_use]
    pub fn with_video_bitrate(mut self, bitrate: u64) -> Self {
        self.video_bitrate = Some(bitrate);
        self
    }

    /// Compute frame rate as a floating-point value, or `None` if unset.
    #[must_use]
    pub fn frame_rate_f64(&self) -> Option<f64> {
        self.frame_rate.map(|(n, d)| {
            if d == 0 {
                0.0
            } else {
                f64::from(n) / f64::from(d)
            }
        })
    }
}

// ── ConstraintViolation ─────────────────────────────────────────────────────

/// A single failed requirement during a compatibility check.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Short identifier for the violated constraint (e.g. `"max_height"`).
    pub constraint: String,
    /// Human-readable explanation.
    pub message: String,
}

impl ConstraintViolation {
    fn new(constraint: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            constraint: constraint.into(),
            message: message.into(),
        }
    }
}

// ── CompatibilityReport ─────────────────────────────────────────────────────

/// The result of checking a source against a preset's requirements.
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// Preset ID that was checked.
    pub preset_id: String,
    /// Whether all constraints passed.
    compatible: bool,
    /// List of constraint violations (empty when compatible).
    violations: Vec<ConstraintViolation>,
    /// Number of constraints that were evaluated (including passing ones).
    evaluated: usize,
}

impl CompatibilityReport {
    fn new(preset_id: impl Into<String>) -> Self {
        Self {
            preset_id: preset_id.into(),
            compatible: true,
            violations: Vec::new(),
            evaluated: 0,
        }
    }

    fn record_pass(&mut self) {
        self.evaluated += 1;
    }

    fn record_fail(&mut self, violation: ConstraintViolation) {
        self.compatible = false;
        self.evaluated += 1;
        self.violations.push(violation);
    }

    /// Whether the source is fully compatible with the preset.
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        self.compatible
    }

    /// All constraint violations found (empty when compatible).
    #[must_use]
    pub fn violations(&self) -> &[ConstraintViolation] {
        &self.violations
    }

    /// Number of violations.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Total number of constraints evaluated (passing + failing).
    #[must_use]
    pub fn evaluated_count(&self) -> usize {
        self.evaluated
    }

    /// Collect all violation messages as a `Vec<String>`.
    #[must_use]
    pub fn violation_messages(&self) -> Vec<String> {
        self.violations.iter().map(|v| v.message.clone()).collect()
    }
}

// ── PresetRequirements ──────────────────────────────────────────────────────

/// The constraints that a preset imposes on its input source.
///
/// Each constraint is optional; absent constraints are skipped during
/// evaluation so that requirements can be as broad or narrow as needed.
#[derive(Debug, Clone, Default)]
pub struct PresetRequirements {
    /// Preset identifier these requirements belong to.
    pub preset_id: String,

    // ── Resolution constraints ──────────────────────────────────────────
    /// Minimum allowed source width.
    pub min_width: Option<u32>,
    /// Maximum allowed source width.
    pub max_width: Option<u32>,
    /// Minimum allowed source height.
    pub min_height: Option<u32>,
    /// Maximum allowed source height.
    pub max_height: Option<u32>,

    // ── Frame rate constraints ──────────────────────────────────────────
    /// Minimum allowed frame rate (fps as float).
    pub min_fps: Option<f64>,
    /// Maximum allowed frame rate (fps as float).
    pub max_fps: Option<f64>,

    // ── Codec / container constraints ───────────────────────────────────
    /// Allowed source video codecs (empty = any allowed).
    pub allowed_video_codecs: Vec<String>,
    /// Allowed source audio codecs (empty = any allowed).
    pub allowed_audio_codecs: Vec<String>,
    /// Allowed container formats (empty = any allowed).
    pub allowed_containers: Vec<String>,

    // ── Bit-depth / sample rate ─────────────────────────────────────────
    /// Minimum bit depth.
    pub min_bit_depth: Option<u8>,
    /// Maximum bit depth.
    pub max_bit_depth: Option<u8>,
    /// Allowed audio sample rates in Hz (empty = any allowed).
    pub allowed_sample_rates: Vec<u32>,
    /// Minimum audio channels.
    pub min_audio_channels: Option<u8>,
    /// Maximum audio channels.
    pub max_audio_channels: Option<u8>,

    // ── Duration constraint ─────────────────────────────────────────────
    /// Maximum allowed duration in seconds (e.g. Instagram 60s limit).
    pub max_duration_secs: Option<f64>,
}

impl PresetRequirements {
    /// Create a new (empty) requirements set for a preset.
    #[must_use]
    pub fn new(preset_id: impl Into<String>) -> Self {
        Self {
            preset_id: preset_id.into(),
            ..Self::default()
        }
    }

    // ── Builder helpers ──────────────────────────────────────────────────

    /// Set minimum source width.
    #[must_use]
    pub fn with_min_width(mut self, v: u32) -> Self {
        self.min_width = Some(v);
        self
    }
    /// Set maximum source width.
    #[must_use]
    pub fn with_max_width(mut self, v: u32) -> Self {
        self.max_width = Some(v);
        self
    }
    /// Set minimum source height.
    #[must_use]
    pub fn with_min_height(mut self, v: u32) -> Self {
        self.min_height = Some(v);
        self
    }
    /// Set maximum source height.
    #[must_use]
    pub fn with_max_height(mut self, v: u32) -> Self {
        self.max_height = Some(v);
        self
    }
    /// Set minimum frame rate.
    #[must_use]
    pub fn with_min_fps(mut self, v: f64) -> Self {
        self.min_fps = Some(v);
        self
    }
    /// Set maximum frame rate.
    #[must_use]
    pub fn with_max_fps(mut self, v: f64) -> Self {
        self.max_fps = Some(v);
        self
    }
    /// Add an allowed video codec.
    #[must_use]
    pub fn with_allowed_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.allowed_video_codecs.push(codec.into());
        self
    }
    /// Add an allowed audio codec.
    #[must_use]
    pub fn with_allowed_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.allowed_audio_codecs.push(codec.into());
        self
    }
    /// Add an allowed container format.
    #[must_use]
    pub fn with_allowed_container(mut self, container: impl Into<String>) -> Self {
        self.allowed_containers.push(container.into());
        self
    }
    /// Set minimum bit depth.
    #[must_use]
    pub fn with_min_bit_depth(mut self, v: u8) -> Self {
        self.min_bit_depth = Some(v);
        self
    }
    /// Set maximum bit depth.
    #[must_use]
    pub fn with_max_bit_depth(mut self, v: u8) -> Self {
        self.max_bit_depth = Some(v);
        self
    }
    /// Add an allowed audio sample rate.
    #[must_use]
    pub fn with_allowed_sample_rate(mut self, rate: u32) -> Self {
        self.allowed_sample_rates.push(rate);
        self
    }
    /// Set minimum audio channels.
    #[must_use]
    pub fn with_min_audio_channels(mut self, v: u8) -> Self {
        self.min_audio_channels = Some(v);
        self
    }
    /// Set maximum audio channels.
    #[must_use]
    pub fn with_max_audio_channels(mut self, v: u8) -> Self {
        self.max_audio_channels = Some(v);
        self
    }
    /// Set maximum source duration.
    #[must_use]
    pub fn with_max_duration(mut self, secs: f64) -> Self {
        self.max_duration_secs = Some(secs);
        self
    }

    // ── Evaluation ───────────────────────────────────────────────────────

    /// Evaluate the requirements against a source spec and produce a report.
    #[must_use]
    pub fn evaluate(&self, source: &SourceMediaSpec) -> CompatibilityReport {
        let mut report = CompatibilityReport::new(&self.preset_id);

        // Resolution
        self.check_u32_min(
            source.width,
            self.min_width,
            "min_width",
            "Source width",
            &mut report,
        );
        self.check_u32_max(
            source.width,
            self.max_width,
            "max_width",
            "Source width",
            &mut report,
        );
        self.check_u32_min(
            source.height,
            self.min_height,
            "min_height",
            "Source height",
            &mut report,
        );
        self.check_u32_max(
            source.height,
            self.max_height,
            "max_height",
            "Source height",
            &mut report,
        );

        // Frame rate
        if let Some(fps) = source.frame_rate_f64() {
            if let Some(min) = self.min_fps {
                report.evaluated += 1;
                if fps < min {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "min_fps",
                        format!("Frame rate {fps:.3} fps is below minimum {min:.3} fps"),
                    ));
                }
            }
            if let Some(max) = self.max_fps {
                report.evaluated += 1;
                if fps > max {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "max_fps",
                        format!("Frame rate {fps:.3} fps exceeds maximum {max:.3} fps"),
                    ));
                }
            }
        }

        // Video codec
        if !self.allowed_video_codecs.is_empty() {
            if let Some(ref codec) = source.video_codec {
                report.evaluated += 1;
                let codec_lower = codec.to_lowercase();
                let allowed = self
                    .allowed_video_codecs
                    .iter()
                    .any(|c| c.to_lowercase() == codec_lower);
                if !allowed {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "allowed_video_codecs",
                        format!(
                            "Video codec '{}' not in allowed list: {:?}",
                            codec, self.allowed_video_codecs
                        ),
                    ));
                }
            }
        }

        // Audio codec
        if !self.allowed_audio_codecs.is_empty() {
            if let Some(ref codec) = source.audio_codec {
                report.evaluated += 1;
                let codec_lower = codec.to_lowercase();
                let allowed = self
                    .allowed_audio_codecs
                    .iter()
                    .any(|c| c.to_lowercase() == codec_lower);
                if !allowed {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "allowed_audio_codecs",
                        format!(
                            "Audio codec '{}' not in allowed list: {:?}",
                            codec, self.allowed_audio_codecs
                        ),
                    ));
                }
            }
        }

        // Container
        if !self.allowed_containers.is_empty() {
            if let Some(ref container) = source.container {
                report.evaluated += 1;
                let c_lower = container.to_lowercase();
                let allowed = self
                    .allowed_containers
                    .iter()
                    .any(|c| c.to_lowercase() == c_lower);
                if !allowed {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "allowed_containers",
                        format!(
                            "Container '{}' not in allowed list: {:?}",
                            container, self.allowed_containers
                        ),
                    ));
                }
            }
        }

        // Bit depth
        self.check_u8_min(
            source.bit_depth,
            self.min_bit_depth,
            "min_bit_depth",
            "Bit depth",
            &mut report,
        );
        self.check_u8_max(
            source.bit_depth,
            self.max_bit_depth,
            "max_bit_depth",
            "Bit depth",
            &mut report,
        );

        // Audio sample rate
        if !self.allowed_sample_rates.is_empty() {
            if let Some(rate) = source.audio_sample_rate {
                report.evaluated += 1;
                if !self.allowed_sample_rates.contains(&rate) {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "allowed_sample_rates",
                        format!(
                            "Audio sample rate {}Hz not in allowed list: {:?}",
                            rate, self.allowed_sample_rates
                        ),
                    ));
                }
            }
        }

        // Audio channels
        self.check_u8_min(
            source.audio_channels,
            self.min_audio_channels,
            "min_audio_channels",
            "Audio channels",
            &mut report,
        );
        self.check_u8_max(
            source.audio_channels,
            self.max_audio_channels,
            "max_audio_channels",
            "Audio channels",
            &mut report,
        );

        // Duration
        if let Some(max_dur) = self.max_duration_secs {
            if let Some(dur) = source.duration_secs {
                report.evaluated += 1;
                if dur > max_dur {
                    report.compatible = false;
                    report.violations.push(ConstraintViolation::new(
                        "max_duration_secs",
                        format!("Duration {dur:.1}s exceeds maximum {max_dur:.1}s"),
                    ));
                }
            }
        }

        report
    }

    // ── Private check helpers ────────────────────────────────────────────

    fn check_u32_min(
        &self,
        value: Option<u32>,
        limit: Option<u32>,
        constraint: &str,
        label: &str,
        report: &mut CompatibilityReport,
    ) {
        if let (Some(v), Some(min)) = (value, limit) {
            report.evaluated += 1;
            if v < min {
                report.compatible = false;
                report.violations.push(ConstraintViolation::new(
                    constraint,
                    format!("{label} {v} is below minimum {min}"),
                ));
            }
        }
    }

    fn check_u32_max(
        &self,
        value: Option<u32>,
        limit: Option<u32>,
        constraint: &str,
        label: &str,
        report: &mut CompatibilityReport,
    ) {
        if let (Some(v), Some(max)) = (value, limit) {
            report.evaluated += 1;
            if v > max {
                report.compatible = false;
                report.violations.push(ConstraintViolation::new(
                    constraint,
                    format!("{label} {v} exceeds maximum {max}"),
                ));
            }
        }
    }

    fn check_u8_min(
        &self,
        value: Option<u8>,
        limit: Option<u8>,
        constraint: &str,
        label: &str,
        report: &mut CompatibilityReport,
    ) {
        if let (Some(v), Some(min)) = (value, limit) {
            report.evaluated += 1;
            if v < min {
                report.compatible = false;
                report.violations.push(ConstraintViolation::new(
                    constraint,
                    format!("{label} {v} is below minimum {min}"),
                ));
            }
        }
    }

    fn check_u8_max(
        &self,
        value: Option<u8>,
        limit: Option<u8>,
        constraint: &str,
        label: &str,
        report: &mut CompatibilityReport,
    ) {
        if let (Some(v), Some(max)) = (value, limit) {
            report.evaluated += 1;
            if v > max {
                report.compatible = false;
                report.violations.push(ConstraintViolation::new(
                    constraint,
                    format!("{label} {v} exceeds maximum {max}"),
                ));
            }
        }
    }
}

// ── CompatibilityMatrix ─────────────────────────────────────────────────────

/// A registry of [`PresetRequirements`] keyed by preset ID.
///
/// Provides bulk compatibility checking for multiple presets against a
/// single source spec, which is useful for surfacing which presets are
/// valid choices for a given input file.
#[derive(Debug, Default)]
pub struct CompatibilityMatrix {
    requirements: HashMap<String, PresetRequirements>,
}

impl CompatibilityMatrix {
    /// Create an empty matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register requirements for a preset.
    pub fn register(&mut self, preset_id: &str, requirements: PresetRequirements) {
        self.requirements
            .insert(preset_id.to_string(), requirements);
    }

    /// Check the source against the requirements for a specific preset.
    ///
    /// # Errors
    ///
    /// Returns [`CompatibilityError::UnknownPreset`] if no requirements are
    /// registered for `preset_id`.
    pub fn check(
        &self,
        preset_id: &str,
        source: &SourceMediaSpec,
    ) -> Result<CompatibilityReport, CompatibilityError> {
        let reqs = self
            .requirements
            .get(preset_id)
            .ok_or_else(|| CompatibilityError::UnknownPreset(preset_id.to_string()))?;
        Ok(reqs.evaluate(source))
    }

    /// Check the source against **all** registered presets and return a map
    /// of preset ID → report, sorted by ID alphabetically.
    #[must_use]
    pub fn check_all(&self, source: &SourceMediaSpec) -> Vec<(String, CompatibilityReport)> {
        let mut results: Vec<(String, CompatibilityReport)> = self
            .requirements
            .iter()
            .map(|(id, reqs)| (id.clone(), reqs.evaluate(source)))
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }

    /// Return all preset IDs for which the source is fully compatible.
    #[must_use]
    pub fn compatible_presets(&self, source: &SourceMediaSpec) -> Vec<String> {
        let mut ids: Vec<String> = self
            .requirements
            .iter()
            .filter_map(|(id, reqs)| {
                if reqs.evaluate(source).is_compatible() {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        ids.sort();
        ids
    }

    /// Number of registered preset requirement sets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.requirements.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn instagram_reqs() -> PresetRequirements {
        PresetRequirements::new("instagram-reel")
            .with_max_duration(60.0)
            .with_min_height(720)
            .with_max_height(1920)
            .with_min_width(720)
            .with_max_width(1080)
            .with_allowed_container("mp4")
            .with_allowed_container("mov")
            .with_max_fps(60.0)
    }

    fn youtube_reqs() -> PresetRequirements {
        PresetRequirements::new("youtube-1080p")
            .with_max_width(3840)
            .with_max_height(2160)
            .with_max_fps(60.0)
    }

    // ── SourceMediaSpec ──

    #[test]
    fn test_source_spec_frame_rate_f64() {
        let spec = SourceMediaSpec::new().with_frame_rate(60, 1);
        assert!((spec.frame_rate_f64().expect("fps should be set") - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_source_spec_frame_rate_ntsc() {
        let spec = SourceMediaSpec::new().with_frame_rate(30000, 1001);
        let fps = spec.frame_rate_f64().expect("fps should be set");
        assert!((fps - 29.970_029_97).abs() < 1e-6);
    }

    #[test]
    fn test_source_spec_zero_denominator() {
        let spec = SourceMediaSpec::new().with_frame_rate(30, 0);
        assert_eq!(spec.frame_rate_f64(), Some(0.0));
    }

    // ── PresetRequirements::evaluate ──

    #[test]
    fn test_compatible_source_no_violations() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(1080, 1080)
            .with_container("mp4")
            .with_frame_rate(30, 1)
            .with_duration(45.0);
        let report = reqs.evaluate(&source);
        assert!(report.is_compatible());
        assert_eq!(report.violation_count(), 0);
    }

    #[test]
    fn test_duration_exceeds_maximum() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(1080, 1080)
            .with_container("mp4")
            .with_duration(120.0);
        let report = reqs.evaluate(&source);
        assert!(!report.is_compatible());
        let violations = report.violations();
        assert!(violations
            .iter()
            .any(|v| v.constraint == "max_duration_secs"));
    }

    #[test]
    fn test_container_not_allowed() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(1080, 1080)
            .with_container("mkv")
            .with_duration(30.0);
        let report = reqs.evaluate(&source);
        assert!(!report.is_compatible());
        assert!(report
            .violations()
            .iter()
            .any(|v| v.constraint == "allowed_containers"));
    }

    #[test]
    fn test_height_below_minimum() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(480, 320)
            .with_container("mp4")
            .with_duration(10.0);
        let report = reqs.evaluate(&source);
        assert!(!report.is_compatible());
        assert!(report
            .violations()
            .iter()
            .any(|v| v.constraint == "min_height"));
    }

    #[test]
    fn test_fps_exceeds_maximum() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(1080, 1920)
            .with_frame_rate(120, 1)
            .with_container("mp4")
            .with_duration(10.0);
        let report = reqs.evaluate(&source);
        assert!(!report.is_compatible());
        assert!(report
            .violations()
            .iter()
            .any(|v| v.constraint == "max_fps"));
    }

    #[test]
    fn test_multiple_violations_accumulated() {
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new()
            .with_resolution(320, 240) // too small
            .with_container("avi") // wrong container
            .with_duration(200.0); // too long
        let report = reqs.evaluate(&source);
        assert!(!report.is_compatible());
        // Expect violations: min_height + min_width + container + duration
        assert!(report.violation_count() >= 3);
    }

    #[test]
    fn test_no_constraints_always_compatible() {
        let reqs = PresetRequirements::new("unconstrained");
        let source = SourceMediaSpec::new()
            .with_resolution(3840, 2160)
            .with_container("mkv")
            .with_frame_rate(240, 1)
            .with_duration(7200.0);
        let report = reqs.evaluate(&source);
        assert!(report.is_compatible());
    }

    #[test]
    fn test_unknown_source_fields_skipped() {
        // If source has no container set, the container constraint is not evaluated
        let reqs = instagram_reqs();
        let source = SourceMediaSpec::new().with_duration(30.0);
        // Width/height min constraints will not fire if source has no width/height
        let report = reqs.evaluate(&source);
        // Only duration is checked and it's fine
        assert!(report.is_compatible());
    }

    #[test]
    fn test_video_codec_constraint() {
        let reqs = PresetRequirements::new("web-delivery")
            .with_allowed_video_codec("h264")
            .with_allowed_video_codec("vp9");
        let ok_source = SourceMediaSpec::new().with_video_codec("h264");
        let bad_source = SourceMediaSpec::new().with_video_codec("prores");
        assert!(reqs.evaluate(&ok_source).is_compatible());
        assert!(!reqs.evaluate(&bad_source).is_compatible());
    }

    #[test]
    fn test_codec_check_case_insensitive() {
        let reqs = PresetRequirements::new("test").with_allowed_video_codec("H264");
        let source = SourceMediaSpec::new().with_video_codec("h264");
        assert!(reqs.evaluate(&source).is_compatible());
    }

    #[test]
    fn test_bit_depth_range() {
        let reqs = PresetRequirements::new("hdr")
            .with_min_bit_depth(10)
            .with_max_bit_depth(12);
        assert!(reqs
            .evaluate(&SourceMediaSpec::new().with_bit_depth(10))
            .is_compatible());
        assert!(reqs
            .evaluate(&SourceMediaSpec::new().with_bit_depth(12))
            .is_compatible());
        assert!(!reqs
            .evaluate(&SourceMediaSpec::new().with_bit_depth(8))
            .is_compatible());
        assert!(!reqs
            .evaluate(&SourceMediaSpec::new().with_bit_depth(16))
            .is_compatible());
    }

    #[test]
    fn test_audio_sample_rate_allowed_list() {
        let reqs = PresetRequirements::new("podcast")
            .with_allowed_sample_rate(44100)
            .with_allowed_sample_rate(48000);
        assert!(reqs
            .evaluate(&SourceMediaSpec::new().with_audio_sample_rate(44100))
            .is_compatible());
        assert!(!reqs
            .evaluate(&SourceMediaSpec::new().with_audio_sample_rate(32000))
            .is_compatible());
    }

    #[test]
    fn test_evaluated_count_increases_per_checked_constraint() {
        let reqs = PresetRequirements::new("counted")
            .with_min_width(100)
            .with_max_width(4000)
            .with_min_height(100)
            .with_max_height(4000);
        let source = SourceMediaSpec::new().with_resolution(1920, 1080);
        let report = reqs.evaluate(&source);
        assert_eq!(report.evaluated_count(), 4); // 4 constraints evaluated
    }

    // ── CompatibilityMatrix ──

    #[test]
    fn test_matrix_register_and_check() {
        let mut matrix = CompatibilityMatrix::new();
        matrix.register("youtube-1080p", youtube_reqs());
        let source = SourceMediaSpec::new().with_resolution(1920, 1080);
        let report = matrix
            .check("youtube-1080p", &source)
            .expect("check should succeed");
        assert!(report.is_compatible());
    }

    #[test]
    fn test_matrix_unknown_preset_returns_error() {
        let matrix = CompatibilityMatrix::new();
        let source = SourceMediaSpec::new();
        let result = matrix.check("nonexistent", &source);
        assert!(matches!(result, Err(CompatibilityError::UnknownPreset(_))));
    }

    #[test]
    fn test_matrix_compatible_presets_filters_correctly() {
        let mut matrix = CompatibilityMatrix::new();
        matrix.register(
            "strict",
            PresetRequirements::new("strict")
                .with_max_duration(10.0)
                .with_allowed_container("mp4"),
        );
        matrix.register(
            "lenient",
            PresetRequirements::new("lenient").with_max_duration(3600.0),
        );

        // Source: 30s duration, avi container
        let source = SourceMediaSpec::new()
            .with_duration(30.0)
            .with_container("avi");

        let compatible = matrix.compatible_presets(&source);
        // "strict" fails on container; "lenient" passes
        assert_eq!(compatible, vec!["lenient".to_string()]);
    }

    #[test]
    fn test_matrix_check_all_returns_all_presets() {
        let mut matrix = CompatibilityMatrix::new();
        matrix.register("a", PresetRequirements::new("a"));
        matrix.register("b", PresetRequirements::new("b"));
        let source = SourceMediaSpec::new();
        let results = matrix.check_all(&source);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_matrix_count() {
        let mut matrix = CompatibilityMatrix::new();
        assert_eq!(matrix.count(), 0);
        matrix.register("p1", PresetRequirements::new("p1"));
        matrix.register("p2", PresetRequirements::new("p2"));
        assert_eq!(matrix.count(), 2);
    }

    #[test]
    fn test_violation_messages_nonempty_on_fail() {
        let reqs = PresetRequirements::new("test").with_max_duration(5.0);
        let source = SourceMediaSpec::new().with_duration(60.0);
        let report = reqs.evaluate(&source);
        assert!(!report.violation_messages().is_empty());
    }
}
