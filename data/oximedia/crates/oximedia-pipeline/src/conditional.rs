//! Conditional branching support for the pipeline DSL.
//!
//! This module provides runtime-evaluated conditions that select which set of
//! pipeline operations to apply based on stream properties observed in a
//! [`PipelineContext`].  It is designed for declarative, data-driven pipelines
//! where the processing path depends on properties like resolution, frame-rate,
//! audio channel count, or the presence of HDR metadata.
//!
//! # Design
//!
//! A [`ConditionalBranch`] holds a [`PipelineCondition`] together with two
//! operation lists: `then_ops` (applied when the condition is true) and
//! `else_ops` (applied otherwise).  The [`PipelineDsl`] assembles a flat op
//! sequence and an ordered list of branches; at runtime, callers feed it a
//! [`PipelineContext`] to discover which ops are active.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::conditional::{
//!     PipelineContext, PipelineCondition, PipelineOp, PipelineDsl,
//! };
//!
//! let mut dsl = PipelineDsl::new();
//! dsl.if_then_else(
//!     PipelineCondition::Resolution { min_w: 1280, min_h: 720 },
//!     vec![PipelineOp::Scale { width: 1920, height: 1080 }],
//!     vec![PipelineOp::Scale { width: 640, height: 360 }],
//! );
//!
//! let ctx = PipelineContext::video(1920, 1080, 30.0, false);
//! let ops = dsl.evaluate_branches(&ctx);
//! assert_eq!(ops.len(), 1);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

// ── PipelineContext ───────────────────────────────────────────────────────────

/// Runtime snapshot of the stream properties against which conditions are
/// evaluated.
///
/// All numeric fields default to zero / false; use the builder helpers
/// ([`PipelineContext::video`], [`PipelineContext::audio`]) or construct the
/// struct directly.
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Video frame width in pixels (0 for audio-only streams).
    pub width: u32,
    /// Video frame height in pixels (0 for audio-only streams).
    pub height: u32,
    /// Video frame-rate in frames per second (0.0 for audio-only streams).
    pub frame_rate: f32,
    /// Number of audio channels (0 for video-only streams).
    pub channels: u8,
    /// Audio sample rate in Hz (0 for video-only streams).
    pub sample_rate: u32,
    /// Whether the stream carries HDR metadata (HDR10, HLG, Dolby Vision, …).
    pub has_hdr: bool,
    /// Arbitrary key → value properties for user-defined conditions.
    pub custom: HashMap<String, String>,
}

impl PipelineContext {
    /// Create a context for a video stream.
    pub fn video(width: u32, height: u32, frame_rate: f32, has_hdr: bool) -> Self {
        Self {
            width,
            height,
            frame_rate,
            channels: 0,
            sample_rate: 0,
            has_hdr,
            custom: HashMap::new(),
        }
    }

    /// Create a context for an audio stream.
    pub fn audio(channels: u8, sample_rate: u32) -> Self {
        Self {
            width: 0,
            height: 0,
            frame_rate: 0.0,
            channels,
            sample_rate,
            has_hdr: false,
            custom: HashMap::new(),
        }
    }

    /// Insert a custom key-value property.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Retrieve a custom property by key.
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            frame_rate: 0.0,
            channels: 0,
            sample_rate: 0,
            has_hdr: false,
            custom: HashMap::new(),
        }
    }
}

// ── PipelineCondition ─────────────────────────────────────────────────────────

/// A boolean predicate evaluated against a [`PipelineContext`].
///
/// The `Custom` variant accepts any `Fn` closure, enabling arbitrary
/// runtime logic without enumerating every possible stream attribute.
pub enum PipelineCondition {
    /// True when the stream frame-rate is **at least** the given value (fps).
    FrameRate(f32),

    /// True when the video frame is at least `min_w × min_h` pixels.
    Resolution { min_w: u32, min_h: u32 },

    /// True when the audio stream has **exactly** the given channel count.
    AudioChannels(u8),

    /// True when the stream carries HDR metadata.
    HasHdr,

    /// True when the audio sample rate is **at least** the given value (Hz).
    SampleRate(u32),

    /// True when the given custom property key exists in the context.
    HasCustomProperty(String),

    /// True when the given custom property equals the given value.
    CustomPropertyEquals(String, String),

    /// Logical AND: true only when **both** inner conditions are true.
    And(Box<PipelineCondition>, Box<PipelineCondition>),

    /// Logical OR: true when **either** inner condition is true.
    Or(Box<PipelineCondition>, Box<PipelineCondition>),

    /// Logical NOT: inverts the inner condition.
    Not(Box<PipelineCondition>),

    /// User-supplied closure evaluated against the full [`PipelineContext`].
    Custom(Arc<dyn Fn(&PipelineContext) -> bool + Send + Sync>),
}

impl Clone for PipelineCondition {
    fn clone(&self) -> Self {
        match self {
            PipelineCondition::FrameRate(v) => PipelineCondition::FrameRate(*v),
            PipelineCondition::Resolution { min_w, min_h } => PipelineCondition::Resolution {
                min_w: *min_w,
                min_h: *min_h,
            },
            PipelineCondition::AudioChannels(ch) => PipelineCondition::AudioChannels(*ch),
            PipelineCondition::HasHdr => PipelineCondition::HasHdr,
            PipelineCondition::SampleRate(hz) => PipelineCondition::SampleRate(*hz),
            PipelineCondition::HasCustomProperty(k) => {
                PipelineCondition::HasCustomProperty(k.clone())
            }
            PipelineCondition::CustomPropertyEquals(k, v) => {
                PipelineCondition::CustomPropertyEquals(k.clone(), v.clone())
            }
            PipelineCondition::And(a, b) => PipelineCondition::And(a.clone(), b.clone()),
            PipelineCondition::Or(a, b) => PipelineCondition::Or(a.clone(), b.clone()),
            PipelineCondition::Not(inner) => PipelineCondition::Not(inner.clone()),
            PipelineCondition::Custom(f) => PipelineCondition::Custom(Arc::clone(f)),
        }
    }
}

impl std::fmt::Debug for PipelineCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineCondition::FrameRate(fps) => write!(f, "FrameRate({fps})"),
            PipelineCondition::Resolution { min_w, min_h } => {
                write!(f, "Resolution {{ min_w: {min_w}, min_h: {min_h} }}")
            }
            PipelineCondition::AudioChannels(ch) => write!(f, "AudioChannels({ch})"),
            PipelineCondition::HasHdr => write!(f, "HasHdr"),
            PipelineCondition::SampleRate(hz) => write!(f, "SampleRate({hz})"),
            PipelineCondition::HasCustomProperty(k) => write!(f, "HasCustomProperty({k})"),
            PipelineCondition::CustomPropertyEquals(k, v) => {
                write!(f, "CustomPropertyEquals({k}, {v})")
            }
            PipelineCondition::And(a, b) => write!(f, "And({a:?}, {b:?})"),
            PipelineCondition::Or(a, b) => write!(f, "Or({a:?}, {b:?})"),
            PipelineCondition::Not(inner) => write!(f, "Not({inner:?})"),
            PipelineCondition::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl PipelineCondition {
    /// Evaluate the condition against the given context, returning `true` or
    /// `false`.
    pub fn evaluate(&self, ctx: &PipelineContext) -> bool {
        match self {
            PipelineCondition::FrameRate(min_fps) => ctx.frame_rate >= *min_fps,
            PipelineCondition::Resolution { min_w, min_h } => {
                ctx.width >= *min_w && ctx.height >= *min_h
            }
            PipelineCondition::AudioChannels(expected) => ctx.channels == *expected,
            PipelineCondition::HasHdr => ctx.has_hdr,
            PipelineCondition::SampleRate(min_hz) => ctx.sample_rate >= *min_hz,
            PipelineCondition::HasCustomProperty(key) => ctx.custom.contains_key(key.as_str()),
            PipelineCondition::CustomPropertyEquals(key, value) => ctx
                .custom
                .get(key.as_str())
                .map(|v| v == value)
                .unwrap_or(false),
            PipelineCondition::And(a, b) => a.evaluate(ctx) && b.evaluate(ctx),
            PipelineCondition::Or(a, b) => a.evaluate(ctx) || b.evaluate(ctx),
            PipelineCondition::Not(inner) => !inner.evaluate(ctx),
            PipelineCondition::Custom(f) => f(ctx),
        }
    }
}

// ── PipelineOp ────────────────────────────────────────────────────────────────

/// A single operation that can appear inside a `then_ops` or `else_ops` list
/// of a [`ConditionalBranch`], or in the flat op sequence of a
/// [`PipelineDsl`].
///
/// These mirror the filter variants in `FilterConfig` but are kept
/// intentionally lightweight — they carry only the parameters needed to
/// describe the operation declaratively, not execution logic.
#[derive(Debug, Clone)]
pub enum PipelineOp {
    /// Resize video to `width × height`.
    Scale { width: u32, height: u32 },
    /// Crop a rectangular region.
    Crop { x: u32, y: u32, w: u32, h: u32 },
    /// Flip video horizontally.
    Hflip,
    /// Flip video vertically.
    Vflip,
    /// Force a constant output frame-rate (fps).
    Fps(f32),
    /// Adjust audio gain in dB.
    Volume(f32),
    /// Trim the stream to a time window in milliseconds.
    Trim { start_ms: i64, end_ms: i64 },
    /// Convert to the named pixel / sample format (e.g. `"yuv420p"`).
    Format(String),
    /// Insert a nested conditional branch.
    Branch(Box<ConditionalBranch>),
}

impl PipelineOp {
    /// Return a short human-readable name for this operation.
    pub fn name(&self) -> &str {
        match self {
            PipelineOp::Scale { .. } => "scale",
            PipelineOp::Crop { .. } => "crop",
            PipelineOp::Hflip => "hflip",
            PipelineOp::Vflip => "vflip",
            PipelineOp::Fps(_) => "fps",
            PipelineOp::Volume(_) => "volume",
            PipelineOp::Trim { .. } => "trim",
            PipelineOp::Format(_) => "format",
            PipelineOp::Branch(_) => "branch",
        }
    }
}

impl PartialEq for PipelineOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                PipelineOp::Scale {
                    width: w1,
                    height: h1,
                },
                PipelineOp::Scale {
                    width: w2,
                    height: h2,
                },
            ) => w1 == w2 && h1 == h2,
            (
                PipelineOp::Crop {
                    x: x1,
                    y: y1,
                    w: w1,
                    h: h1,
                },
                PipelineOp::Crop {
                    x: x2,
                    y: y2,
                    w: w2,
                    h: h2,
                },
            ) => x1 == x2 && y1 == y2 && w1 == w2 && h1 == h2,
            (PipelineOp::Hflip, PipelineOp::Hflip) => true,
            (PipelineOp::Vflip, PipelineOp::Vflip) => true,
            (PipelineOp::Fps(a), PipelineOp::Fps(b)) => a.to_bits() == b.to_bits(),
            (PipelineOp::Volume(a), PipelineOp::Volume(b)) => a.to_bits() == b.to_bits(),
            (
                PipelineOp::Trim {
                    start_ms: s1,
                    end_ms: e1,
                },
                PipelineOp::Trim {
                    start_ms: s2,
                    end_ms: e2,
                },
            ) => s1 == s2 && e1 == e2,
            (PipelineOp::Format(a), PipelineOp::Format(b)) => a == b,
            // Branch variants are considered unequal (closures can't be compared).
            (PipelineOp::Branch(_), PipelineOp::Branch(_)) => false,
            _ => false,
        }
    }
}

// ── ConditionalBranch ─────────────────────────────────────────────────────────

/// A conditional pipeline branch: applies `then_ops` when `condition` is true,
/// otherwise applies `else_ops`.
#[derive(Debug, Clone)]
pub struct ConditionalBranch {
    /// The predicate to evaluate against the runtime context.
    pub condition: PipelineCondition,
    /// Operations applied when the condition evaluates to `true`.
    pub then_ops: Vec<PipelineOp>,
    /// Operations applied when the condition evaluates to `false`.
    pub else_ops: Vec<PipelineOp>,
}

impl ConditionalBranch {
    /// Build a new `ConditionalBranch`.
    pub fn new(
        condition: PipelineCondition,
        then_ops: Vec<PipelineOp>,
        else_ops: Vec<PipelineOp>,
    ) -> Self {
        Self {
            condition,
            then_ops,
            else_ops,
        }
    }

    /// Evaluate the branch against `ctx` and return a slice of the active ops.
    pub fn evaluate<'a>(&'a self, ctx: &PipelineContext) -> &'a [PipelineOp] {
        if self.condition.evaluate(ctx) {
            &self.then_ops
        } else {
            &self.else_ops
        }
    }
}

// ── PipelineDsl ───────────────────────────────────────────────────────────────

/// A declarative pipeline description composed of unconditional operations and
/// conditional branches.
///
/// Call [`PipelineDsl::add_op`] to append flat operations and
/// [`PipelineDsl::if_then_else`] to append conditional branches.  At runtime,
/// pass a [`PipelineContext`] to [`PipelineDsl::evaluate_branches`] to resolve
/// which operations are active for a particular stream.
///
/// # Example
///
/// ```rust
/// use oximedia_pipeline::conditional::{
///     PipelineContext, PipelineCondition, PipelineOp, PipelineDsl,
/// };
///
/// let mut dsl = PipelineDsl::new();
/// dsl.add_op(PipelineOp::Fps(25.0))
///    .if_then_else(
///        PipelineCondition::HasHdr,
///        vec![PipelineOp::Format("yuv420p10le".into())],
///        vec![PipelineOp::Format("yuv420p".into())],
///    );
///
/// let hdr_ctx = PipelineContext::video(1920, 1080, 25.0, true);
/// let ops = dsl.evaluate_branches(&hdr_ctx);
/// // Fps(25.0) + Format("yuv420p10le")
/// assert_eq!(ops.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct PipelineDsl {
    /// Flat (unconditional) operations in insertion order.
    pub ops: Vec<PipelineOp>,
    /// Conditional branches in insertion order.
    pub branches: Vec<ConditionalBranch>,
}

impl PipelineDsl {
    /// Create an empty DSL description.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a single unconditional operation.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn add_op(&mut self, op: PipelineOp) -> &mut Self {
        self.ops.push(op);
        self
    }

    /// Append a conditional branch.
    ///
    /// When evaluated, the branch applies `then_ops` if `condition` is true,
    /// otherwise `else_ops`.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn if_then_else(
        &mut self,
        cond: PipelineCondition,
        then_ops: Vec<PipelineOp>,
        else_ops: Vec<PipelineOp>,
    ) -> &mut Self {
        self.branches
            .push(ConditionalBranch::new(cond, then_ops, else_ops));
        self
    }

    /// Evaluate all branches against `ctx` and collect the active operations.
    ///
    /// The returned `Vec` contains:
    /// 1. All unconditional `ops` in order.
    /// 2. For each branch (in order), the ops selected by its condition.
    ///
    /// Nested [`PipelineOp::Branch`] items inside `then_ops` / `else_ops` are
    /// **not** recursively resolved here; callers can unwrap them manually if
    /// needed.
    pub fn evaluate_branches<'a>(&'a self, ctx: &PipelineContext) -> Vec<&'a PipelineOp> {
        let mut result: Vec<&'a PipelineOp> = self.ops.iter().collect();
        for branch in &self.branches {
            let selected = branch.evaluate(ctx);
            result.extend(selected.iter());
        }
        result
    }

    /// Return the total number of flat ops (not counting branch contents).
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Return the number of registered conditional branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Returns `true` when there are no ops and no branches.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.branches.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_ctx() -> PipelineContext {
        PipelineContext::video(1920, 1080, 30.0, false)
    }

    fn sd_ctx() -> PipelineContext {
        PipelineContext::video(640, 480, 25.0, false)
    }

    fn hdr_ctx() -> PipelineContext {
        PipelineContext::video(3840, 2160, 60.0, true)
    }

    fn stereo_audio() -> PipelineContext {
        PipelineContext::audio(2, 48000)
    }

    fn surround_audio() -> PipelineContext {
        PipelineContext::audio(6, 48000)
    }

    // ── PipelineContext ──────────────────────────────────────────────────────

    #[test]
    fn context_video_defaults() {
        let ctx = hd_ctx();
        assert_eq!(ctx.width, 1920);
        assert_eq!(ctx.height, 1080);
        assert!((ctx.frame_rate - 30.0).abs() < f32::EPSILON);
        assert!(!ctx.has_hdr);
        assert_eq!(ctx.channels, 0);
    }

    #[test]
    fn context_audio_defaults() {
        let ctx = stereo_audio();
        assert_eq!(ctx.channels, 2);
        assert_eq!(ctx.sample_rate, 48000);
        assert_eq!(ctx.width, 0);
        assert!(!ctx.has_hdr);
    }

    #[test]
    fn context_custom_properties() {
        let ctx = PipelineContext::default()
            .with_custom("codec", "av1")
            .with_custom("profile", "main");
        assert_eq!(ctx.get_custom("codec"), Some("av1"));
        assert_eq!(ctx.get_custom("profile"), Some("main"));
        assert_eq!(ctx.get_custom("missing"), None);
    }

    // ── PipelineCondition ────────────────────────────────────────────────────

    #[test]
    fn condition_frame_rate_at_threshold() {
        let cond = PipelineCondition::FrameRate(30.0);
        assert!(cond.evaluate(&hd_ctx()));
        let low = PipelineContext::video(1920, 1080, 24.0, false);
        assert!(!cond.evaluate(&low));
    }

    #[test]
    fn condition_resolution_meets_minimum() {
        let cond = PipelineCondition::Resolution {
            min_w: 1280,
            min_h: 720,
        };
        assert!(cond.evaluate(&hd_ctx()));
        assert!(!cond.evaluate(&sd_ctx()));
    }

    #[test]
    fn condition_resolution_exact_boundary() {
        let ctx = PipelineContext::video(1280, 720, 25.0, false);
        let cond = PipelineCondition::Resolution {
            min_w: 1280,
            min_h: 720,
        };
        assert!(cond.evaluate(&ctx));
        let ctx_below = PipelineContext::video(1279, 720, 25.0, false);
        assert!(!cond.evaluate(&ctx_below));
    }

    #[test]
    fn condition_audio_channels_exact() {
        let cond = PipelineCondition::AudioChannels(2);
        assert!(cond.evaluate(&stereo_audio()));
        assert!(!cond.evaluate(&surround_audio()));
    }

    #[test]
    fn condition_has_hdr_true_and_false() {
        let cond = PipelineCondition::HasHdr;
        assert!(cond.evaluate(&hdr_ctx()));
        assert!(!cond.evaluate(&hd_ctx()));
    }

    #[test]
    fn condition_sample_rate_at_least() {
        let cond = PipelineCondition::SampleRate(44100);
        assert!(cond.evaluate(&stereo_audio())); // 48000 >= 44100
        let low = PipelineContext::audio(2, 22050);
        assert!(!cond.evaluate(&low)); // 22050 < 44100
    }

    #[test]
    fn condition_has_custom_property() {
        let ctx = PipelineContext::default().with_custom("hdr_format", "hdr10");
        let cond = PipelineCondition::HasCustomProperty("hdr_format".into());
        assert!(cond.evaluate(&ctx));
        let absent = PipelineCondition::HasCustomProperty("missing_key".into());
        assert!(!absent.evaluate(&ctx));
    }

    #[test]
    fn condition_custom_property_equals() {
        let ctx = PipelineContext::default().with_custom("codec", "av1");
        let cond = PipelineCondition::CustomPropertyEquals("codec".into(), "av1".into());
        assert!(cond.evaluate(&ctx));
        let wrong = PipelineCondition::CustomPropertyEquals("codec".into(), "h264".into());
        assert!(!wrong.evaluate(&ctx));
    }

    #[test]
    fn condition_and_both_true() {
        let cond = PipelineCondition::And(
            Box::new(PipelineCondition::HasHdr),
            Box::new(PipelineCondition::Resolution {
                min_w: 3840,
                min_h: 2160,
            }),
        );
        assert!(cond.evaluate(&hdr_ctx()));
        // HDR but lower res
        let partial = PipelineContext::video(1920, 1080, 30.0, true);
        assert!(!cond.evaluate(&partial));
    }

    #[test]
    fn condition_or_either_true() {
        let cond = PipelineCondition::Or(
            Box::new(PipelineCondition::HasHdr),
            Box::new(PipelineCondition::Resolution {
                min_w: 1920,
                min_h: 1080,
            }),
        );
        assert!(cond.evaluate(&hd_ctx())); // HD (no HDR)
        assert!(cond.evaluate(&hdr_ctx())); // HDR 4K
        let neither = PipelineContext::video(640, 480, 25.0, false);
        assert!(!cond.evaluate(&neither));
    }

    #[test]
    fn condition_not_inverts() {
        let cond = PipelineCondition::Not(Box::new(PipelineCondition::HasHdr));
        assert!(cond.evaluate(&hd_ctx()));
        assert!(!cond.evaluate(&hdr_ctx()));
    }

    #[test]
    fn condition_custom_closure() {
        let cond = PipelineCondition::Custom(Arc::new(|ctx: &PipelineContext| {
            ctx.width > 1000 && ctx.height > 500
        }));
        assert!(cond.evaluate(&hd_ctx()));
        assert!(!cond.evaluate(&sd_ctx()));
    }

    // ── ConditionalBranch ────────────────────────────────────────────────────

    #[test]
    fn branch_selects_then_ops_when_true() {
        let branch = ConditionalBranch::new(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![PipelineOp::Format("yuv420p".into())],
        );
        let selected = branch.evaluate(&hdr_ctx());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], PipelineOp::Format("yuv420p10le".into()));
    }

    #[test]
    fn branch_selects_else_ops_when_false() {
        let branch = ConditionalBranch::new(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![PipelineOp::Format("yuv420p".into())],
        );
        let selected = branch.evaluate(&hd_ctx());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], PipelineOp::Format("yuv420p".into()));
    }

    #[test]
    fn branch_empty_else_ops() {
        let branch = ConditionalBranch::new(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![],
        );
        let else_ops = branch.evaluate(&hd_ctx());
        assert!(else_ops.is_empty());
    }

    // ── PipelineDsl ──────────────────────────────────────────────────────────

    #[test]
    fn dsl_new_is_empty() {
        let dsl = PipelineDsl::new();
        assert!(dsl.is_empty());
        assert_eq!(dsl.op_count(), 0);
        assert_eq!(dsl.branch_count(), 0);
    }

    #[test]
    fn dsl_add_op_increments_count() {
        let mut dsl = PipelineDsl::new();
        dsl.add_op(PipelineOp::Hflip).add_op(PipelineOp::Vflip);
        assert_eq!(dsl.op_count(), 2);
        assert!(!dsl.is_empty());
    }

    #[test]
    fn dsl_if_then_else_increments_branch_count() {
        let mut dsl = PipelineDsl::new();
        dsl.if_then_else(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Hflip],
            vec![PipelineOp::Vflip],
        );
        assert_eq!(dsl.branch_count(), 1);
    }

    #[test]
    fn dsl_evaluate_branches_flat_ops_only() {
        let mut dsl = PipelineDsl::new();
        dsl.add_op(PipelineOp::Hflip).add_op(PipelineOp::Scale {
            width: 1280,
            height: 720,
        });
        let ops = dsl.evaluate_branches(&hd_ctx());
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name(), "hflip");
        assert_eq!(ops[1].name(), "scale");
    }

    #[test]
    fn dsl_evaluate_branches_selects_correct_branch() {
        let mut dsl = PipelineDsl::new();
        dsl.if_then_else(
            PipelineCondition::Resolution {
                min_w: 1280,
                min_h: 720,
            },
            vec![PipelineOp::Scale {
                width: 1920,
                height: 1080,
            }],
            vec![PipelineOp::Scale {
                width: 640,
                height: 360,
            }],
        );
        let hd_ops = dsl.evaluate_branches(&hd_ctx());
        assert_eq!(hd_ops.len(), 1);
        assert_eq!(
            hd_ops[0],
            &PipelineOp::Scale {
                width: 1920,
                height: 1080
            }
        );

        let sd_ops = dsl.evaluate_branches(&sd_ctx());
        assert_eq!(sd_ops.len(), 1);
        assert_eq!(
            sd_ops[0],
            &PipelineOp::Scale {
                width: 640,
                height: 360
            }
        );
    }

    #[test]
    fn dsl_evaluate_branches_flat_and_conditional_combined() {
        let mut dsl = PipelineDsl::new();
        dsl.add_op(PipelineOp::Fps(25.0)).if_then_else(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![PipelineOp::Format("yuv420p".into())],
        );
        let ops = dsl.evaluate_branches(&hdr_ctx());
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name(), "fps");
        assert_eq!(ops[1], &PipelineOp::Format("yuv420p10le".into()));
    }

    #[test]
    fn dsl_multiple_branches_evaluated_in_order() {
        let mut dsl = PipelineDsl::new();
        dsl.if_then_else(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![PipelineOp::Format("yuv420p".into())],
        )
        .if_then_else(
            PipelineCondition::AudioChannels(6),
            vec![PipelineOp::Volume(0.5)],
            vec![PipelineOp::Volume(1.0)],
        );

        // HDR video context → SDR audio channels=0 (not 6)
        let ctx = hdr_ctx();
        let ops = dsl.evaluate_branches(&ctx);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0], &PipelineOp::Format("yuv420p10le".into()));
        // channels=0 ≠ 6 → else branch
        assert_eq!(ops[1], &PipelineOp::Volume(1.0));
    }

    #[test]
    fn dsl_chaining_methods_return_self() {
        let mut dsl = PipelineDsl::new();
        let r = dsl
            .add_op(PipelineOp::Hflip)
            .add_op(PipelineOp::Vflip)
            .if_then_else(PipelineCondition::HasHdr, vec![], vec![]);
        // Just verify the chain compiles and the DSL has the expected counts
        let _ = r;
        assert_eq!(dsl.op_count(), 2);
        assert_eq!(dsl.branch_count(), 1);
    }

    #[test]
    fn pipeline_op_name() {
        assert_eq!(PipelineOp::Hflip.name(), "hflip");
        assert_eq!(PipelineOp::Vflip.name(), "vflip");
        assert_eq!(PipelineOp::Fps(30.0).name(), "fps");
        assert_eq!(PipelineOp::Volume(0.0).name(), "volume");
        assert_eq!(
            PipelineOp::Scale {
                width: 1920,
                height: 1080
            }
            .name(),
            "scale"
        );
        assert_eq!(
            PipelineOp::Crop {
                x: 0,
                y: 0,
                w: 100,
                h: 100
            }
            .name(),
            "crop"
        );
        assert_eq!(
            PipelineOp::Trim {
                start_ms: 0,
                end_ms: 5000
            }
            .name(),
            "trim"
        );
        assert_eq!(PipelineOp::Format("yuv420p".into()).name(), "format");
    }

    #[test]
    fn condition_debug_format_does_not_panic() {
        let c = PipelineCondition::FrameRate(30.0);
        let s = format!("{c:?}");
        assert!(!s.is_empty());

        let c2 = PipelineCondition::Custom(Arc::new(|_ctx: &PipelineContext| true));
        let s2 = format!("{c2:?}");
        assert!(s2.contains("Custom"));
    }

    #[test]
    fn nested_branch_op() {
        let inner = ConditionalBranch::new(
            PipelineCondition::HasHdr,
            vec![PipelineOp::Format("yuv420p10le".into())],
            vec![PipelineOp::Format("yuv420p".into())],
        );
        let mut dsl = PipelineDsl::new();
        dsl.add_op(PipelineOp::Branch(Box::new(inner)));
        let ops = dsl.evaluate_branches(&hdr_ctx());
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].name(), "branch");
    }
}
