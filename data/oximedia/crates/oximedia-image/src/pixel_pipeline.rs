//! Pixel pipeline: composable per-pixel transformations applied in sequence.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ── Pixel value wrapper ───────────────────────────────────────────────────────

/// A 32-bit float RGBA pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelF32 {
    /// Red component [0, 1].
    pub r: f32,
    /// Green component [0, 1].
    pub g: f32,
    /// Blue component [0, 1].
    pub b: f32,
    /// Alpha component [0, 1].
    pub a: f32,
}

impl PixelF32 {
    /// Create a new pixel.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque black.
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Opaque white.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    /// Clamp all components to [0, 1].
    #[must_use]
    pub fn clamped(self) -> Self {
        Self::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
            self.a.clamp(0.0, 1.0),
        )
    }

    /// Component-wise addition.
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            self.a + other.a,
        )
    }

    /// Component-wise multiplication.
    #[must_use]
    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.r * other.r,
            self.g * other.g,
            self.b * other.b,
            self.a * other.a,
        )
    }
}

impl Default for PixelF32 {
    fn default() -> Self {
        Self::black()
    }
}

// ── Stage trait ───────────────────────────────────────────────────────────────

/// A single, stateless pixel-processing stage.
pub trait PixelStage: Send + Sync {
    /// Process one pixel and return the transformed result.
    fn process(&self, p: PixelF32) -> PixelF32;

    /// Human-readable name of this stage.
    fn name(&self) -> &str;
}

// ── Built-in stages ───────────────────────────────────────────────────────────

/// Gamma encode/decode stage.
#[derive(Clone, Debug)]
pub struct GammaStage {
    /// Exponent applied to each colour channel.
    pub exponent: f32,
}

impl GammaStage {
    /// Create a gamma stage with `exponent`.
    #[must_use]
    pub fn new(exponent: f32) -> Self {
        Self { exponent }
    }

    /// sRGB display gamma (2.2 approximation).
    #[must_use]
    pub fn srgb_display() -> Self {
        Self::new(1.0 / 2.2)
    }

    /// sRGB linearisation.
    #[must_use]
    pub fn srgb_linear() -> Self {
        Self::new(2.2)
    }
}

impl PixelStage for GammaStage {
    fn process(&self, p: PixelF32) -> PixelF32 {
        PixelF32::new(
            p.r.max(0.0).powf(self.exponent),
            p.g.max(0.0).powf(self.exponent),
            p.b.max(0.0).powf(self.exponent),
            p.a,
        )
    }

    fn name(&self) -> &str {
        "GammaStage"
    }
}

/// Per-channel gain stage.
#[derive(Clone, Debug)]
pub struct GainStage {
    /// Red gain.
    pub r: f32,
    /// Green gain.
    pub g: f32,
    /// Blue gain.
    pub b: f32,
}

impl GainStage {
    /// Create a uniform gain stage.
    #[must_use]
    pub fn uniform(gain: f32) -> Self {
        Self {
            r: gain,
            g: gain,
            b: gain,
        }
    }

    /// Create a per-channel gain stage.
    #[must_use]
    pub fn per_channel(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl PixelStage for GainStage {
    fn process(&self, p: PixelF32) -> PixelF32 {
        PixelF32::new(p.r * self.r, p.g * self.g, p.b * self.b, p.a)
    }

    fn name(&self) -> &str {
        "GainStage"
    }
}

/// Invert colours (negative).
#[derive(Clone, Debug, Default)]
pub struct InvertStage;

impl PixelStage for InvertStage {
    fn process(&self, p: PixelF32) -> PixelF32 {
        PixelF32::new(1.0 - p.r, 1.0 - p.g, 1.0 - p.b, p.a)
    }

    fn name(&self) -> &str {
        "InvertStage"
    }
}

/// Clamp stage – forces all components to [0, 1].
#[derive(Clone, Debug, Default)]
pub struct ClampStage;

impl PixelStage for ClampStage {
    fn process(&self, p: PixelF32) -> PixelF32 {
        p.clamped()
    }

    fn name(&self) -> &str {
        "ClampStage"
    }
}

/// Desaturate (convert to luminance using BT.709 coefficients).
#[derive(Clone, Debug, Default)]
pub struct DesaturateStage;

impl PixelStage for DesaturateStage {
    fn process(&self, p: PixelF32) -> PixelF32 {
        let luma = 0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b;
        PixelF32::new(luma, luma, luma, p.a)
    }

    fn name(&self) -> &str {
        "DesaturateStage"
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// A sequential pipeline of [`PixelStage`] operations.
pub struct PixelPipeline {
    stages: Vec<Box<dyn PixelStage>>,
}

impl PixelPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the pipeline.
    pub fn push<S: PixelStage + 'static>(&mut self, stage: S) {
        self.stages.push(Box::new(stage));
    }

    /// Process a single pixel through all stages in order.
    #[must_use]
    pub fn run(&self, p: PixelF32) -> PixelF32 {
        self.stages.iter().fold(p, |acc, s| s.process(acc))
    }

    /// Process an entire image buffer in-place.
    ///
    /// `buf` must be a flat sequence of RGBA f32 values (4 floats per pixel).
    pub fn process_buffer(&self, buf: &mut [f32]) {
        for chunk in buf.chunks_exact_mut(4) {
            let p = PixelF32::new(chunk[0], chunk[1], chunk[2], chunk[3]);
            let out = self.run(p);
            chunk[0] = out.r;
            chunk[1] = out.g;
            chunk[2] = out.b;
            chunk[3] = out.a;
        }
    }

    /// Returns the number of stages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the pipeline has no stages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Returns the names of all stages.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

impl Default for PixelPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_f32_new() {
        let p = PixelF32::new(0.1, 0.2, 0.3, 1.0);
        assert!((p.r - 0.1).abs() < 1e-6);
        assert!((p.g - 0.2).abs() < 1e-6);
        assert!((p.b - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_f32_black() {
        let p = PixelF32::black();
        assert_eq!(p.r, 0.0);
        assert_eq!(p.g, 0.0);
        assert_eq!(p.b, 0.0);
        assert_eq!(p.a, 1.0);
    }

    #[test]
    fn test_pixel_f32_white() {
        let p = PixelF32::white();
        assert_eq!(p.r, 1.0);
        assert_eq!(p.a, 1.0);
    }

    #[test]
    fn test_pixel_f32_clamped() {
        let p = PixelF32::new(-0.5, 1.5, 0.5, 2.0).clamped();
        assert_eq!(p.r, 0.0);
        assert_eq!(p.g, 1.0);
        assert_eq!(p.b, 0.5);
        assert_eq!(p.a, 1.0);
    }

    #[test]
    fn test_pixel_f32_add() {
        let a = PixelF32::new(0.1, 0.2, 0.3, 0.5);
        let b = PixelF32::new(0.4, 0.3, 0.2, 0.5);
        let c = a.add(b);
        assert!((c.r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_pixel_f32_mul() {
        let a = PixelF32::new(1.0, 0.5, 0.0, 1.0);
        let b = PixelF32::new(0.5, 2.0, 1.0, 1.0);
        let c = a.mul(b);
        assert!((c.r - 0.5).abs() < 1e-6);
        assert!((c.g - 1.0).abs() < 1e-6);
        assert!((c.b - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_stage_identity() {
        let s = GammaStage::new(1.0);
        let p = PixelF32::new(0.5, 0.5, 0.5, 1.0);
        let out = s.process(p);
        assert!((out.r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_invert_stage() {
        let s = InvertStage;
        let p = PixelF32::new(1.0, 0.0, 0.5, 1.0);
        let out = s.process(p);
        assert!((out.r - 0.0).abs() < 1e-6);
        assert!((out.g - 1.0).abs() < 1e-6);
        assert!((out.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gain_stage_uniform() {
        let s = GainStage::uniform(2.0);
        let p = PixelF32::new(0.5, 0.25, 0.1, 1.0);
        let out = s.process(p);
        assert!((out.r - 1.0).abs() < 1e-5);
        assert!((out.g - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_clamp_stage() {
        let s = ClampStage;
        let p = PixelF32::new(-1.0, 2.0, 0.5, 1.5);
        let out = s.process(p);
        assert_eq!(out.r, 0.0);
        assert_eq!(out.g, 1.0);
    }

    #[test]
    fn test_desaturate_stage_luma() {
        let s = DesaturateStage;
        let p = PixelF32::new(1.0, 0.0, 0.0, 1.0); // Pure red
        let out = s.process(p);
        // BT.709 red coefficient is 0.2126
        assert!((out.r - 0.2126).abs() < 1e-4);
        assert!((out.r - out.g).abs() < 1e-6);
        assert!((out.r - out.b).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = PixelPipeline::new();
        let p = PixelF32::new(0.5, 0.5, 0.5, 1.0);
        let out = pipeline.run(p);
        assert!((out.r - 0.5).abs() < 1e-6);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_stage_names() {
        let mut pipeline = PixelPipeline::new();
        pipeline.push(GainStage::uniform(1.0));
        pipeline.push(InvertStage);
        let names = pipeline.stage_names();
        assert_eq!(names, vec!["GainStage", "InvertStage"]);
    }

    #[test]
    fn test_pipeline_process_buffer() {
        let mut buf = vec![1.0f32, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0];
        let mut pipeline = PixelPipeline::new();
        pipeline.push(InvertStage);
        pipeline.process_buffer(&mut buf);
        assert!((buf[0] - 0.0).abs() < 1e-5);
        assert!((buf[4] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_len() {
        let mut pipeline = PixelPipeline::new();
        pipeline.push(ClampStage);
        pipeline.push(DesaturateStage);
        assert_eq!(pipeline.len(), 2);
    }
}
