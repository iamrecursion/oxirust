#![allow(dead_code)]
//! Concatenation and joining of multiple media sources into a single output.
//!
//! Handles cross-format joining with optional transition effects between
//! segments, automatic audio/video alignment, and gap filling.
//!
//! # Mixed-Source Concatenation
//!
//! [`MixedSourceConcatenator`] handles sequences where sources have different
//! resolutions, frame rates, or codecs.  It analyses each source segment,
//! determines whether re-encoding is required, and produces a [`ConcatPlan`]
//! that the caller can execute.

use std::fmt;

/// Strategy for handling format mismatches between segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformStrategy {
    /// Re-encode every segment to match the first segment's format.
    ReEncodeAll,
    /// Re-encode only segments that differ from the target format.
    ReEncodeDiffers,
    /// Attempt stream-copy where possible (fastest, may fail on mismatches).
    StreamCopy,
}

impl fmt::Display for ConformStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReEncodeAll => write!(f, "re-encode-all"),
            Self::ReEncodeDiffers => write!(f, "re-encode-differs"),
            Self::StreamCopy => write!(f, "stream-copy"),
        }
    }
}

/// Transition type between consecutive segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionKind {
    /// Hard cut with no transition.
    Cut,
    /// Crossfade of the specified duration in seconds.
    Crossfade(f64),
    /// Fade to black then fade from black.
    FadeThrough(f64),
}

impl TransitionKind {
    /// Return the duration in seconds (0.0 for a hard cut).
    #[must_use]
    pub fn duration(&self) -> f64 {
        match self {
            Self::Cut => 0.0,
            Self::Crossfade(d) | Self::FadeThrough(d) => *d,
        }
    }
}

/// A single input segment in the concat list.
#[derive(Debug, Clone)]
pub struct ConcatSegment {
    /// Path or URI to the source media.
    pub source: String,
    /// Optional in-point in seconds (trim start).
    pub in_point: Option<f64>,
    /// Optional out-point in seconds (trim end).
    pub out_point: Option<f64>,
    /// Transition to apply *after* this segment (before the next).
    pub transition: TransitionKind,
}

impl ConcatSegment {
    /// Create a segment from a source path with defaults (full duration, hard cut).
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            in_point: None,
            out_point: None,
            transition: TransitionKind::Cut,
        }
    }

    /// Set in-point.
    #[must_use]
    pub fn with_in_point(mut self, seconds: f64) -> Self {
        self.in_point = Some(seconds);
        self
    }

    /// Set out-point.
    #[must_use]
    pub fn with_out_point(mut self, seconds: f64) -> Self {
        self.out_point = Some(seconds);
        self
    }

    /// Set transition after this segment.
    #[must_use]
    pub fn with_transition(mut self, t: TransitionKind) -> Self {
        self.transition = t;
        self
    }

    /// Compute effective duration (returns `None` when both points are absent).
    #[must_use]
    pub fn effective_duration(&self) -> Option<f64> {
        match (self.in_point, self.out_point) {
            (Some(i), Some(o)) => Some((o - i).max(0.0)),
            _ => None,
        }
    }
}

/// Overall concat job configuration.
#[derive(Debug, Clone)]
pub struct ConcatConfig {
    /// Ordered list of segments.
    pub segments: Vec<ConcatSegment>,
    /// Output path.
    pub output: String,
    /// Conforming strategy.
    pub conform: ConformStrategy,
    /// Target video width (if re-encoding).
    pub target_width: Option<u32>,
    /// Target video height (if re-encoding).
    pub target_height: Option<u32>,
    /// Target frame rate numerator / denominator (if re-encoding).
    pub target_fps: Option<(u32, u32)>,
    /// Target audio sample rate.
    pub target_sample_rate: Option<u32>,
}

impl ConcatConfig {
    /// Create a new concat configuration.
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            segments: Vec::new(),
            output: output.into(),
            conform: ConformStrategy::ReEncodeDiffers,
            target_width: None,
            target_height: None,
            target_fps: None,
            target_sample_rate: None,
        }
    }

    /// Add a segment.
    pub fn add_segment(&mut self, seg: ConcatSegment) {
        self.segments.push(seg);
    }

    /// Set conforming strategy.
    #[must_use]
    pub fn with_conform(mut self, strategy: ConformStrategy) -> Self {
        self.conform = strategy;
        self
    }

    /// Set target resolution.
    #[must_use]
    pub fn with_resolution(mut self, w: u32, h: u32) -> Self {
        self.target_width = Some(w);
        self.target_height = Some(h);
        self
    }

    /// Set target frame rate.
    #[must_use]
    pub fn with_fps(mut self, num: u32, den: u32) -> Self {
        self.target_fps = Some((num, den));
        self
    }

    /// Set target audio sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.target_sample_rate = Some(rate);
        self
    }

    /// Return total number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Compute the total transition time between segments.
    #[must_use]
    pub fn total_transition_time(&self) -> f64 {
        self.segments.iter().map(|s| s.transition.duration()).sum()
    }

    /// Compute the total known content duration (sum of effective durations).
    /// Segments without known duration are excluded.
    #[must_use]
    pub fn total_known_duration(&self) -> f64 {
        self.segments
            .iter()
            .filter_map(ConcatSegment::effective_duration)
            .sum()
    }
}

/// Result of a concat operation.
#[derive(Debug, Clone)]
pub struct ConcatResult {
    /// Output file path.
    pub output_path: String,
    /// Number of segments joined.
    pub segments_joined: usize,
    /// Total output duration in seconds.
    pub total_duration: f64,
    /// Number of segments that required re-encoding.
    pub re_encoded_count: usize,
}

/// Validate a concat configuration and return a list of issues (empty = valid).
#[must_use]
pub fn validate_concat(config: &ConcatConfig) -> Vec<String> {
    let mut issues = Vec::new();
    if config.segments.is_empty() {
        issues.push("No segments specified".to_string());
    }
    if config.output.is_empty() {
        issues.push("Output path is empty".to_string());
    }
    for (i, seg) in config.segments.iter().enumerate() {
        if seg.source.is_empty() {
            issues.push(format!("Segment {i} has empty source path"));
        }
        if let (Some(inp), Some(out)) = (seg.in_point, seg.out_point) {
            if out <= inp {
                issues.push(format!("Segment {i} out-point ({out}) <= in-point ({inp})"));
            }
        }
    }
    issues
}

// ─── Mixed-source concatenation ───────────────────────────────────────────────

/// Codec and format properties of a single source segment.
///
/// Obtained by probing the source file before planning.  Filling in accurate
/// values is the caller's responsibility; the planner uses these to decide
/// whether re-encoding is necessary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceProperties {
    /// Video codec identifier (e.g. `"h264"`, `"vp9"`, `"av1"`).
    pub codec: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Audio codec identifier (e.g. `"aac"`, `"opus"`).
    pub audio_codec: String,
}

impl SourceProperties {
    /// Returns `(width, height)`.
    #[must_use]
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the frame rate as a floating-point number.
    ///
    /// Returns `0.0` when `fps_den` is zero.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            return 0.0;
        }
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Returns `true` when `other` has the same video codec, resolution,
    /// frame rate, audio codec, and sample rate.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.codec == other.codec
            && self.width == other.width
            && self.height == other.height
            && self.fps_num == other.fps_num
            && self.fps_den == other.fps_den
            && self.sample_rate == other.sample_rate
            && self.audio_codec == other.audio_codec
    }
}

/// A [`ConcatSegment`] annotated with its probed source properties.
#[derive(Debug, Clone)]
pub struct AnnotatedSegment {
    /// The segment specification.
    pub segment: ConcatSegment,
    /// Probed source properties.
    pub properties: SourceProperties,
}

impl AnnotatedSegment {
    /// Creates an annotated segment.
    #[must_use]
    pub fn new(segment: ConcatSegment, properties: SourceProperties) -> Self {
        Self {
            segment,
            properties,
        }
    }
}

/// A single step in a [`ConcatPlan`].
#[derive(Debug, Clone)]
pub struct ConcatStep {
    /// Source file path / URI.
    pub source: String,
    /// Whether this segment needs re-encoding to match the target parameters.
    pub requires_reencode: bool,
    /// Target width after possible rescaling.
    pub target_width: u32,
    /// Target height after possible rescaling.
    pub target_height: u32,
    /// Target frame rate numerator.
    pub target_fps_num: u32,
    /// Target frame rate denominator.
    pub target_fps_den: u32,
}

impl ConcatStep {
    /// Returns the target resolution as `(width, height)`.
    #[must_use]
    pub fn target_resolution(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    /// Returns the target frame rate as a float.
    #[must_use]
    pub fn target_fps(&self) -> f64 {
        if self.target_fps_den == 0 {
            return 0.0;
        }
        f64::from(self.target_fps_num) / f64::from(self.target_fps_den)
    }
}

/// A fully-resolved plan for concatenating mixed-source segments.
///
/// Produced by [`MixedSourceConcatenator::build_plan`] and consumed by the
/// caller's encoding engine.
#[derive(Debug, Clone)]
pub struct ConcatPlan {
    /// Ordered processing steps, one per source segment.
    pub steps: Vec<ConcatStep>,
    /// Resolved target width for the output.
    pub target_width: u32,
    /// Resolved target height for the output.
    pub target_height: u32,
    /// Resolved target frame rate numerator.
    pub target_fps_num: u32,
    /// Resolved target frame rate denominator.
    pub target_fps_den: u32,
    /// Output file path.
    pub output: String,
}

impl ConcatPlan {
    /// Returns the number of segments that will be re-encoded.
    #[must_use]
    pub fn reencode_count(&self) -> usize {
        self.steps.iter().filter(|s| s.requires_reencode).count()
    }

    /// Returns the number of segments that will be stream-copied.
    #[must_use]
    pub fn stream_copy_count(&self) -> usize {
        self.steps.iter().filter(|s| !s.requires_reencode).count()
    }

    /// Returns `true` if all segments can be stream-copied (no re-encoding).
    #[must_use]
    pub fn all_stream_copy(&self) -> bool {
        self.steps.iter().all(|s| !s.requires_reencode)
    }
}

/// Analyses a list of annotated source segments and produces a [`ConcatPlan`]
/// that handles mixed resolutions, frame rates, and codecs.
///
/// # Algorithm
///
/// 1. Determine the **reference properties** (from the first segment, or from
///    the `ConcatConfig` target resolution / fps overrides).
/// 2. For each segment compare its properties to the reference.
/// 3. Under `ReEncodeAll`: every segment is re-encoded.
/// 4. Under `ReEncodeDiffers`: only segments that differ from the reference.
/// 5. Under `StreamCopy`: all segments are stream-copied (caller assumes
///    compatible sources).
pub struct MixedSourceConcatenator {
    config: ConcatConfig,
    sources: Vec<AnnotatedSegment>,
}

impl MixedSourceConcatenator {
    /// Creates a new concatenator.
    ///
    /// `sources` must be in the same order as `config.segments`.
    #[must_use]
    pub fn new(config: ConcatConfig, sources: Vec<AnnotatedSegment>) -> Self {
        Self { config, sources }
    }

    /// Returns the reference `SourceProperties` against which all segments are
    /// compared.
    ///
    /// Priority: explicit config target → first source segment.
    fn reference_properties(&self) -> SourceProperties {
        let first = self.sources.first().map(|s| s.properties.clone());

        let width = self
            .config
            .target_width
            .or_else(|| first.as_ref().map(|p| p.width))
            .unwrap_or(1920);
        let height = self
            .config
            .target_height
            .or_else(|| first.as_ref().map(|p| p.height))
            .unwrap_or(1080);
        let (fps_num, fps_den) = self
            .config
            .target_fps
            .or_else(|| first.as_ref().map(|p| (p.fps_num, p.fps_den)))
            .unwrap_or((30, 1));
        let sample_rate = self
            .config
            .target_sample_rate
            .or_else(|| first.as_ref().map(|p| p.sample_rate))
            .unwrap_or(48_000);
        let codec = first
            .as_ref()
            .map(|p| p.codec.clone())
            .unwrap_or_else(|| "h264".into());
        let audio_codec = first
            .as_ref()
            .map(|p| p.audio_codec.clone())
            .unwrap_or_else(|| "aac".into());

        SourceProperties {
            codec,
            width,
            height,
            fps_num,
            fps_den,
            sample_rate,
            audio_codec,
        }
    }

    /// Builds the [`ConcatPlan`] from the configured sources and strategy.
    #[must_use]
    pub fn build_plan(&self) -> ConcatPlan {
        let reference = self.reference_properties();

        let steps: Vec<ConcatStep> = self
            .sources
            .iter()
            .map(|ann| {
                let requires_reencode = match self.config.conform {
                    ConformStrategy::ReEncodeAll => true,
                    ConformStrategy::StreamCopy => false,
                    ConformStrategy::ReEncodeDiffers => {
                        !ann.properties.is_compatible_with(&reference)
                    }
                };

                ConcatStep {
                    source: ann.segment.source.clone(),
                    requires_reencode,
                    target_width: reference.width,
                    target_height: reference.height,
                    target_fps_num: reference.fps_num,
                    target_fps_den: reference.fps_den,
                }
            })
            .collect();

        ConcatPlan {
            steps,
            target_width: reference.width,
            target_height: reference.height,
            target_fps_num: reference.fps_num,
            target_fps_den: reference.fps_den,
            output: self.config.output.clone(),
        }
    }

    /// Returns the number of sources that require re-encoding.
    #[must_use]
    pub fn reencode_count(&self) -> usize {
        self.build_plan().reencode_count()
    }

    /// Returns `true` if all sources are format-compatible (no re-encoding).
    #[must_use]
    pub fn all_compatible(&self) -> bool {
        self.build_plan().all_stream_copy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conform_strategy_display() {
        assert_eq!(ConformStrategy::ReEncodeAll.to_string(), "re-encode-all");
        assert_eq!(
            ConformStrategy::ReEncodeDiffers.to_string(),
            "re-encode-differs"
        );
        assert_eq!(ConformStrategy::StreamCopy.to_string(), "stream-copy");
    }

    #[test]
    fn test_transition_duration() {
        assert!((TransitionKind::Cut.duration() - 0.0).abs() < f64::EPSILON);
        assert!((TransitionKind::Crossfade(1.5).duration() - 1.5).abs() < f64::EPSILON);
        assert!((TransitionKind::FadeThrough(2.0).duration() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segment_new() {
        let seg = ConcatSegment::new("clip.mp4");
        assert_eq!(seg.source, "clip.mp4");
        assert!(seg.in_point.is_none());
        assert!(seg.out_point.is_none());
        assert_eq!(seg.transition, TransitionKind::Cut);
    }

    #[test]
    fn test_segment_trim() {
        let seg = ConcatSegment::new("clip.mp4")
            .with_in_point(5.0)
            .with_out_point(15.0);
        assert!(
            (seg.effective_duration().expect("should succeed in test") - 10.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_segment_no_duration() {
        let seg = ConcatSegment::new("clip.mp4").with_in_point(5.0);
        assert!(seg.effective_duration().is_none());
    }

    #[test]
    fn test_concat_config_builder() {
        let mut config = ConcatConfig::new("output.mp4")
            .with_conform(ConformStrategy::StreamCopy)
            .with_resolution(1920, 1080)
            .with_fps(30, 1)
            .with_sample_rate(48000);
        config.add_segment(ConcatSegment::new("a.mp4"));
        config.add_segment(ConcatSegment::new("b.mp4"));

        assert_eq!(config.segment_count(), 2);
        assert_eq!(config.conform, ConformStrategy::StreamCopy);
        assert_eq!(config.target_width, Some(1920));
        assert_eq!(config.target_height, Some(1080));
        assert_eq!(config.target_fps, Some((30, 1)));
        assert_eq!(config.target_sample_rate, Some(48000));
    }

    #[test]
    fn test_total_transition_time() {
        let mut config = ConcatConfig::new("out.mp4");
        config.add_segment(
            ConcatSegment::new("a.mp4").with_transition(TransitionKind::Crossfade(1.0)),
        );
        config.add_segment(
            ConcatSegment::new("b.mp4").with_transition(TransitionKind::FadeThrough(0.5)),
        );
        config.add_segment(ConcatSegment::new("c.mp4"));
        assert!((config.total_transition_time() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_total_known_duration() {
        let mut config = ConcatConfig::new("out.mp4");
        config.add_segment(
            ConcatSegment::new("a.mp4")
                .with_in_point(0.0)
                .with_out_point(10.0),
        );
        config.add_segment(ConcatSegment::new("b.mp4")); // unknown duration
        config.add_segment(
            ConcatSegment::new("c.mp4")
                .with_in_point(5.0)
                .with_out_point(20.0),
        );
        assert!((config.total_known_duration() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_empty_segments() {
        let config = ConcatConfig::new("out.mp4");
        let issues = validate_concat(&config);
        assert!(issues.iter().any(|i| i.contains("No segments")));
    }

    #[test]
    fn test_validate_empty_output() {
        let mut config = ConcatConfig::new("");
        config.add_segment(ConcatSegment::new("a.mp4"));
        let issues = validate_concat(&config);
        assert!(issues.iter().any(|i| i.contains("Output path")));
    }

    #[test]
    fn test_validate_bad_trim() {
        let mut config = ConcatConfig::new("out.mp4");
        config.add_segment(
            ConcatSegment::new("a.mp4")
                .with_in_point(20.0)
                .with_out_point(5.0),
        );
        let issues = validate_concat(&config);
        assert!(issues.iter().any(|i| i.contains("out-point")));
    }

    #[test]
    fn test_validate_valid_config() {
        let mut config = ConcatConfig::new("out.mp4");
        config.add_segment(
            ConcatSegment::new("a.mp4")
                .with_in_point(0.0)
                .with_out_point(10.0),
        );
        let issues = validate_concat(&config);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_concat_result_fields() {
        let result = ConcatResult {
            output_path: "out.mp4".to_string(),
            segments_joined: 3,
            total_duration: 30.0,
            re_encoded_count: 1,
        };
        assert_eq!(result.segments_joined, 3);
        assert!((result.total_duration - 30.0).abs() < f64::EPSILON);
    }

    // ── MixedSourceConcatenator tests ─────────────────────────────────────────

    #[test]
    fn test_source_properties_default() {
        let props = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        assert_eq!(props.codec, "h264");
        assert_eq!(props.resolution(), (1920, 1080));
    }

    #[test]
    fn test_source_properties_compatible() {
        let a = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let b = a.clone();
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_source_properties_incompatible_resolution() {
        let a = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let b = SourceProperties {
            width: 1280,
            height: 720,
            ..a.clone()
        };
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_source_properties_incompatible_codec() {
        let a = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let b = SourceProperties {
            codec: "vp9".into(),
            ..a.clone()
        };
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_mixed_concatenator_uniform_sources() {
        let props = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let sources = vec![
            AnnotatedSegment {
                segment: ConcatSegment::new("a.mp4"),
                properties: props.clone(),
            },
            AnnotatedSegment {
                segment: ConcatSegment::new("b.mp4"),
                properties: props.clone(),
            },
        ];
        let mut config =
            ConcatConfig::new("out.mp4").with_conform(ConformStrategy::ReEncodeDiffers);
        config.add_segment(ConcatSegment::new("a.mp4"));
        config.add_segment(ConcatSegment::new("b.mp4"));

        let concatenator = MixedSourceConcatenator::new(config.clone(), sources);
        let plan = concatenator.build_plan();

        // Uniform sources: no re-encoding needed
        assert_eq!(plan.steps.len(), 2);
        assert!(
            plan.steps.iter().all(|s| !s.requires_reencode),
            "Uniform sources should not require re-encoding"
        );
    }

    #[test]
    fn test_mixed_concatenator_mixed_resolution() {
        let base = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let different = SourceProperties {
            width: 1280,
            height: 720,
            ..base.clone()
        };

        let sources = vec![
            AnnotatedSegment {
                segment: ConcatSegment::new("hd.mp4"),
                properties: base.clone(),
            },
            AnnotatedSegment {
                segment: ConcatSegment::new("sd.mp4"),
                properties: different.clone(),
            },
        ];
        let mut config = ConcatConfig::new("out.mp4")
            .with_conform(ConformStrategy::ReEncodeDiffers)
            .with_resolution(1920, 1080);
        config.add_segment(ConcatSegment::new("hd.mp4"));
        config.add_segment(ConcatSegment::new("sd.mp4"));

        let concatenator = MixedSourceConcatenator::new(config, sources);
        let plan = concatenator.build_plan();

        assert_eq!(plan.steps.len(), 2);
        // Second segment is 720p — should be marked for re-encoding
        assert!(
            plan.steps[1].requires_reencode,
            "Mixed-resolution segment should require re-encoding"
        );
        // Target resolution set to match the first/config resolution
        assert_eq!(plan.target_width, 1920);
        assert_eq!(plan.target_height, 1080);
    }

    #[test]
    fn test_mixed_concatenator_reencode_all() {
        let props = SourceProperties {
            codec: "h264".into(),
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            sample_rate: 48_000,
            audio_codec: "aac".into(),
        };
        let sources = vec![
            AnnotatedSegment {
                segment: ConcatSegment::new("a.mp4"),
                properties: props.clone(),
            },
            AnnotatedSegment {
                segment: ConcatSegment::new("b.mp4"),
                properties: props.clone(),
            },
        ];
        let mut config = ConcatConfig::new("out.mp4").with_conform(ConformStrategy::ReEncodeAll);
        config.add_segment(ConcatSegment::new("a.mp4"));
        config.add_segment(ConcatSegment::new("b.mp4"));

        let concatenator = MixedSourceConcatenator::new(config, sources);
        let plan = concatenator.build_plan();

        // ReEncodeAll forces re-encoding even for compatible sources.
        assert!(plan.steps.iter().all(|s| s.requires_reencode));
    }

    #[test]
    fn test_concat_plan_reencode_count() {
        let plan = ConcatPlan {
            steps: vec![
                ConcatStep {
                    source: "a.mp4".into(),
                    requires_reencode: false,
                    target_width: 1920,
                    target_height: 1080,
                    target_fps_num: 30,
                    target_fps_den: 1,
                },
                ConcatStep {
                    source: "b.mp4".into(),
                    requires_reencode: true,
                    target_width: 1920,
                    target_height: 1080,
                    target_fps_num: 30,
                    target_fps_den: 1,
                },
            ],
            target_width: 1920,
            target_height: 1080,
            target_fps_num: 30,
            target_fps_den: 1,
            output: "out.mp4".into(),
        };
        assert_eq!(plan.reencode_count(), 1);
        assert_eq!(plan.stream_copy_count(), 1);
    }

    #[test]
    fn test_concat_plan_all_stream_copy() {
        let plan = ConcatPlan {
            steps: vec![ConcatStep {
                source: "a.mp4".into(),
                requires_reencode: false,
                target_width: 1920,
                target_height: 1080,
                target_fps_num: 30,
                target_fps_den: 1,
            }],
            target_width: 1920,
            target_height: 1080,
            target_fps_num: 30,
            target_fps_den: 1,
            output: "out.mp4".into(),
        };
        assert!(plan.all_stream_copy());
    }

    #[test]
    fn test_concat_step_resolution() {
        let step = ConcatStep {
            source: "x.mp4".into(),
            requires_reencode: true,
            target_width: 3840,
            target_height: 2160,
            target_fps_num: 60,
            target_fps_den: 1,
        };
        assert_eq!(step.target_resolution(), (3840, 2160));
        assert!((step.target_fps() - 60.0).abs() < 1e-9);
    }
}
