//! Professional color grading tools.
//!
//! Provides CDL (Color Decision List) grading, color wheel lift/gamma/gain,
//! and a composable grading pipeline.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Color wheel for lift/gamma/gain (shadows/midtones/highlights) correction.
///
/// Each field holds RGB offsets applied based on luminance weighting.
#[derive(Debug, Clone)]
pub struct ColorWheel {
    /// Shadows offset (lift) - applied to dark areas.
    pub shadows: [f32; 3],
    /// Midtones offset (gamma) - applied to mid-tones.
    pub midtones: [f32; 3],
    /// Highlights offset (gain) - applied to bright areas.
    pub highlights: [f32; 3],
}

impl ColorWheel {
    /// Create a new color wheel with the given lift/gamma/gain offsets.
    #[must_use]
    pub fn new(shadows: [f32; 3], midtones: [f32; 3], highlights: [f32; 3]) -> Self {
        Self {
            shadows,
            midtones,
            highlights,
        }
    }

    /// Create an identity (no-op) color wheel.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            shadows: [0.0, 0.0, 0.0],
            midtones: [0.0, 0.0, 0.0],
            highlights: [0.0, 0.0, 0.0],
        }
    }

    /// Apply color wheel correction to a pixel.
    ///
    /// Weighting is based on luma: shadows dominate at low luma, highlights at high luma,
    /// midtones peak at mid luma with smooth transitions using cosine weighting.
    ///
    /// # Arguments
    ///
    /// * `r`, `g`, `b` - Input pixel RGB (0..1)
    /// * `luma` - Luminance of the pixel (0..1), used for weighting
    #[must_use]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32, luma: f32) -> (f32, f32, f32) {
        let luma = luma.clamp(0.0, 1.0);

        // Shadow weight: peaks at 0, falls off toward 1
        let shadow_w = (1.0 - luma).powf(2.0);
        // Highlight weight: peaks at 1, falls off toward 0
        let highlight_w = luma.powf(2.0);
        // Midtone weight: peaks at 0.5
        let mid_w = 1.0 - shadow_w - highlight_w;
        let mid_w = mid_w.max(0.0);

        let r_out = (r
            + self.shadows[0] * shadow_w
            + self.midtones[0] * mid_w
            + self.highlights[0] * highlight_w)
            .clamp(0.0, 1.0);
        let g_out = (g
            + self.shadows[1] * shadow_w
            + self.midtones[1] * mid_w
            + self.highlights[1] * highlight_w)
            .clamp(0.0, 1.0);
        let b_out = (b
            + self.shadows[2] * shadow_w
            + self.midtones[2] * mid_w
            + self.highlights[2] * highlight_w)
            .clamp(0.0, 1.0);

        (r_out, g_out, b_out)
    }
}

impl Default for ColorWheel {
    fn default() -> Self {
        Self::identity()
    }
}

/// Color Decision List (CDL) grade node.
///
/// Implements the ASC CDL formula: `out = clamp((in * slope + offset)^power)`
/// followed by saturation adjustment.
#[derive(Debug, Clone)]
pub struct CdlGrade {
    /// Per-channel slope (multiplicative, positive).
    pub slope: [f32; 3],
    /// Per-channel offset (additive).
    pub offset: [f32; 3],
    /// Per-channel power (gamma, positive).
    pub power: [f32; 3],
    /// Global saturation (1.0 = no change, 0.0 = grayscale).
    pub saturation: f32,
}

impl CdlGrade {
    /// Create a new CDL grade.
    #[must_use]
    pub fn new(slope: [f32; 3], offset: [f32; 3], power: [f32; 3], saturation: f32) -> Self {
        Self {
            slope,
            offset,
            power,
            saturation,
        }
    }

    /// Identity CDL (no change).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
            saturation: 1.0,
        }
    }

    /// Apply the CDL grade to a pixel.
    ///
    /// Formula: `out = clamp((in * slope + offset)^power)` then saturation.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Apply slope, offset, power per channel
        let r_cdl = ((r * self.slope[0] + self.offset[0]).max(0.0))
            .powf(self.power[0])
            .min(1.0);
        let g_cdl = ((g * self.slope[1] + self.offset[1]).max(0.0))
            .powf(self.power[1])
            .min(1.0);
        let b_cdl = ((b * self.slope[2] + self.offset[2]).max(0.0))
            .powf(self.power[2])
            .min(1.0);

        // Apply saturation using Rec.709 luma weights
        let luma = 0.2126 * r_cdl + 0.7152 * g_cdl + 0.0722 * b_cdl;
        let r_out = luma + self.saturation * (r_cdl - luma);
        let g_out = luma + self.saturation * (g_cdl - luma);
        let b_out = luma + self.saturation * (b_cdl - luma);

        (
            r_out.clamp(0.0, 1.0),
            g_out.clamp(0.0, 1.0),
            b_out.clamp(0.0, 1.0),
        )
    }

    /// Serialize to a simple string format.
    ///
    /// Format: `slope=R,G,B offset=R,G,B power=R,G,B sat=S`
    #[must_use]
    pub fn to_cdl_string(&self) -> String {
        format!(
            "slope={:.6},{:.6},{:.6} offset={:.6},{:.6},{:.6} power={:.6},{:.6},{:.6} sat={:.6}",
            self.slope[0],
            self.slope[1],
            self.slope[2],
            self.offset[0],
            self.offset[1],
            self.offset[2],
            self.power[0],
            self.power[1],
            self.power[2],
            self.saturation
        )
    }

    /// Parse from the simple string format produced by `to_cdl_string`.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if parsing fails.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut slope = [1.0f32; 3];
        let mut offset = [0.0f32; 3];
        let mut power = [1.0f32; 3];
        let mut saturation = 1.0f32;

        for part in s.split_whitespace() {
            if let Some(val) = part.strip_prefix("slope=") {
                let vals = parse_triple(val)?;
                slope = vals;
            } else if let Some(val) = part.strip_prefix("offset=") {
                let vals = parse_triple(val)?;
                offset = vals;
            } else if let Some(val) = part.strip_prefix("power=") {
                let vals = parse_triple(val)?;
                power = vals;
            } else if let Some(val) = part.strip_prefix("sat=") {
                saturation = val
                    .parse::<f32>()
                    .map_err(|e| format!("sat parse error: {e}"))?;
            }
        }

        Ok(Self {
            slope,
            offset,
            power,
            saturation,
        })
    }
}

/// Parse three comma-separated floats.
fn parse_triple(s: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("Expected 3 values, got {}: '{s}'", parts.len()));
    }
    let a = parts[0]
        .parse::<f32>()
        .map_err(|e| format!("parse error: {e}"))?;
    let b = parts[1]
        .parse::<f32>()
        .map_err(|e| format!("parse error: {e}"))?;
    let c = parts[2]
        .parse::<f32>()
        .map_err(|e| format!("parse error: {e}"))?;
    Ok([a, b, c])
}

impl Default for CdlGrade {
    fn default() -> Self {
        Self::identity()
    }
}

/// Individual node type for a grading pipeline.
#[derive(Debug, Clone)]
pub enum GradingNode {
    /// CDL grade node.
    Cdl(CdlGrade),
    /// Color wheel lift/gamma/gain with luma weighting.
    ColorWheel(ColorWheel),
    /// Master + per-channel tone curves (see [`crate::curves::RgbCurves`]),
    /// applied to each pixel via [`crate::curves::RgbCurves::apply_pixel`].
    /// Use [`crate::curves::RgbCurves::identity`] for a no-op curve node.
    Curves(crate::curves::RgbCurves),
    /// Global saturation adjustment.
    Saturation(f32),
    /// Exposure adjustment in stops.
    Exposure(f32),
}

impl GradingNode {
    /// Apply this node to a pixel.
    #[must_use]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self {
            GradingNode::Cdl(cdl) => cdl.apply(r, g, b),
            GradingNode::ColorWheel(wheel) => {
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                wheel.apply_pixel(r, g, b, luma)
            }
            GradingNode::Curves(curves) => curves.apply_pixel(r, g, b),
            GradingNode::Saturation(sat) => {
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let r_out = luma + sat * (r - luma);
                let g_out = luma + sat * (g - luma);
                let b_out = luma + sat * (b - luma);
                (
                    r_out.clamp(0.0, 1.0),
                    g_out.clamp(0.0, 1.0),
                    b_out.clamp(0.0, 1.0),
                )
            }
            GradingNode::Exposure(stops) => {
                let gain = 2.0_f32.powf(*stops);
                (
                    (r * gain).clamp(0.0, 1.0),
                    (g * gain).clamp(0.0, 1.0),
                    (b * gain).clamp(0.0, 1.0),
                )
            }
        }
    }
}

/// A composable grading pipeline of nodes applied in sequence.
#[derive(Debug, Clone, Default)]
pub struct GradingPipeline {
    /// The ordered list of grading nodes.
    pub nodes: Vec<GradingNode>,
}

impl GradingPipeline {
    /// Create a new empty grading pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node to the pipeline.
    pub fn add_node(&mut self, node: GradingNode) {
        self.nodes.push(node);
    }

    /// Apply all nodes to a pixel.
    #[must_use]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let mut rgb = (r, g, b);
        for node in &self.nodes {
            rgb = node.apply_pixel(rgb.0, rgb.1, rgb.2);
        }
        rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_wheel_identity() {
        let wheel = ColorWheel::identity();
        let (r, g, b) = wheel.apply_pixel(0.5, 0.4, 0.3, 0.4);
        assert!((r - 0.5).abs() < 0.001);
        assert!((g - 0.4).abs() < 0.001);
        assert!((b - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_color_wheel_shadow_boost() {
        // Boost shadows red
        let wheel = ColorWheel::new([0.1, 0.0, 0.0], [0.0; 3], [0.0; 3]);
        // Dark pixel → mostly in shadow zone
        let (r, g, b) = wheel.apply_pixel(0.1, 0.1, 0.1, 0.1);
        assert!(r > g, "Shadow red should be boosted");
        assert!(b < r);
    }

    #[test]
    fn test_color_wheel_highlight_boost() {
        // Boost highlights blue
        let wheel = ColorWheel::new([0.0; 3], [0.0; 3], [0.0, 0.0, 0.2]);
        // Bright pixel → mostly in highlight zone
        let (r, _g, b) = wheel.apply_pixel(0.9, 0.9, 0.9, 0.9);
        assert!(
            b > r + 0.01,
            "Highlight blue should be boosted: b={b} r={r}"
        );
    }

    #[test]
    fn test_cdl_identity() {
        let cdl = CdlGrade::identity();
        let (r, g, b) = cdl.apply(0.5, 0.3, 0.2);
        assert!((r - 0.5).abs() < 1e-4);
        assert!((g - 0.3).abs() < 1e-4);
        assert!((b - 0.2).abs() < 1e-4);
    }

    #[test]
    fn test_cdl_slope() {
        let cdl = CdlGrade::new([2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let (r, _, _) = cdl.apply(0.3, 0.3, 0.3);
        assert!(r > 0.3, "Slope > 1 should increase value");
    }

    #[test]
    fn test_cdl_offset() {
        let cdl = CdlGrade::new([1.0, 1.0, 1.0], [0.1, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let (r, g, _) = cdl.apply(0.5, 0.5, 0.5);
        assert!(r > g, "Positive offset should increase channel");
    }

    #[test]
    fn test_cdl_power() {
        let cdl = CdlGrade::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], 1.0);
        let (r, g, _) = cdl.apply(0.5, 0.5, 0.5);
        assert!(r < g, "Power > 1 should darken midtones");
    }

    #[test]
    fn test_cdl_saturation_desaturate() {
        let cdl = CdlGrade::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let (r, g, b) = cdl.apply(0.8, 0.2, 0.4);
        // At sat=0 all channels should equal luma
        assert!((r - g).abs() < 1e-4);
        assert!((g - b).abs() < 1e-4);
    }

    #[test]
    fn test_cdl_string_roundtrip() {
        let cdl = CdlGrade::new([1.1, 0.9, 1.0], [0.02, -0.01, 0.0], [0.95, 1.05, 1.0], 1.1);
        let s = cdl.to_cdl_string();
        let parsed = CdlGrade::parse(&s).expect("parse should succeed");
        assert!((parsed.slope[0] - cdl.slope[0]).abs() < 1e-4);
        assert!((parsed.offset[1] - cdl.offset[1]).abs() < 1e-4);
        assert!((parsed.power[2] - cdl.power[2]).abs() < 1e-4);
        assert!((parsed.saturation - cdl.saturation).abs() < 1e-4);
    }

    #[test]
    fn test_cdl_parse_invalid() {
        assert!(CdlGrade::parse("slope=1,2 offset=0,0,0 power=1,1,1 sat=1").is_err());
    }

    #[test]
    fn test_grading_node_saturation() {
        let node = GradingNode::Saturation(0.0);
        let (r, g, b) = node.apply_pixel(0.8, 0.2, 0.5);
        assert!((r - g).abs() < 1e-4);
        assert!((g - b).abs() < 1e-4);
    }

    #[test]
    fn test_grading_node_exposure() {
        let node = GradingNode::Exposure(1.0); // +1 stop = double
        let (r, _, _) = node.apply_pixel(0.3, 0.3, 0.3);
        assert!((r - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_grading_pipeline_empty() {
        let pipeline = GradingPipeline::new();
        let (r, g, b) = pipeline.apply_pixel(0.5, 0.4, 0.3);
        assert!((r - 0.5).abs() < 1e-4);
        assert!((g - 0.4).abs() < 1e-4);
        assert!((b - 0.3).abs() < 1e-4);
    }

    #[test]
    fn test_grading_pipeline_multi_node() {
        let mut pipeline = GradingPipeline::new();
        pipeline.add_node(GradingNode::Cdl(CdlGrade::identity()));
        pipeline.add_node(GradingNode::Saturation(1.2));
        pipeline.add_node(GradingNode::Exposure(0.0));
        let (r, g, b) = pipeline.apply_pixel(0.5, 0.4, 0.3);
        assert!(r.is_finite() && g.is_finite() && b.is_finite());
    }

    #[test]
    fn test_grading_node_curves_identity_passthrough() {
        // An explicit identity curve node must still be a passthrough.
        let node = GradingNode::Curves(crate::curves::RgbCurves::identity());
        let (r, g, b) = node.apply_pixel(0.5, 0.4, 0.3);
        assert!((r - 0.5).abs() < 1e-6);
        assert!((g - 0.4).abs() < 1e-6);
        assert!((b - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_grading_node_curves_nonidentity_changes_pixel() {
        use crate::curves::{CurvePoint, RgbCurves, ToneCurve};

        // Master curve that strongly lifts shadows (0.25 -> 0.5): a
        // non-identity curve LUT, not a passthrough.
        let lift_curve = ToneCurve::new(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.25, 0.5),
            CurvePoint::new(1.0, 1.0),
        ]);
        let curves = RgbCurves::new(
            ToneCurve::identity(),
            ToneCurve::identity(),
            ToneCurve::identity(),
            lift_curve,
        );
        let node = GradingNode::Curves(curves);
        let (r, g, b) = node.apply_pixel(0.25, 0.25, 0.25);

        // Proves the curve LUT is actually applied to pixels rather than
        // skipped as identity: the output must differ substantially from
        // the un-graded input.
        assert!(
            (r - 0.25).abs() > 0.1,
            "expected curve to change the pixel value, got r={r} (input was 0.25)"
        );
        assert!(
            (r - 0.5).abs() < 1e-4,
            "expected curve-lifted value ≈ 0.5, got r={r}"
        );
        assert!(
            (r - g).abs() < 1e-6,
            "all channels fed the same input must match"
        );
        assert!(
            (g - b).abs() < 1e-6,
            "all channels fed the same input must match"
        );
    }

    #[test]
    fn test_grading_pipeline_with_curves_node_applies_curve() {
        use crate::curves::{CurvePoint, RgbCurves, ToneCurve};

        // A per-channel curve that only boosts the red channel.
        let red_boost = ToneCurve::new(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.5, 0.9),
            CurvePoint::new(1.0, 1.0),
        ]);
        let curves = RgbCurves::new(
            red_boost,
            ToneCurve::identity(),
            ToneCurve::identity(),
            ToneCurve::identity(),
        );

        let mut pipeline = GradingPipeline::new();
        pipeline.add_node(GradingNode::Curves(curves));

        let (r, g, b) = pipeline.apply_pixel(0.5, 0.5, 0.5);
        assert!(
            r > 0.7,
            "expected the pipeline to apply the red-boosting curve, got r={r}"
        );
        // Green/blue used identity curves, so they pass through unchanged.
        assert!((g - 0.5).abs() < 1e-4);
        assert!((b - 0.5).abs() < 1e-4);
    }
}
