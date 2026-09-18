#![allow(dead_code)]
//! Colour grading pipeline for per-frame image processing.
//!
//! Implements a node-based colour grading pipeline with operations inspired by
//! the ASC CDL standard as well as common DCC tools (lift/gamma/gain, HSL,
//! curves, vignette, temperature/tint).

// ─── Colour operations ───────────────────────────────────────────────────────

/// A single colour grading operation applied to every pixel.
#[derive(Clone, Debug)]
pub enum ColorOperation {
    /// Add to shadows: `out = in + lift * (1 – in)`.
    Lift(f32, f32, f32),
    /// Per-channel power: `out = in.powf(1 / gamma)`.
    Gamma(f32, f32, f32),
    /// Multiply: `out = in * gain`.
    Gain(f32, f32, f32),
    /// Saturation adjustment (1.0 = no change, 0.0 = greyscale).
    Saturation(f32),
    /// Colour temperature shift in Kelvin offset (positive = warm, negative = cool).
    Temperature(f32),
    /// Green–magenta tint in `[-1, 1]`.
    Tint(f32),
    /// Per-channel curves: sorted `(input, output)` control points for R / G / B.
    Curves(Vec<(f32, f32)>, Vec<(f32, f32)>, Vec<(f32, f32)>),
    /// Hue rotation (°), saturation scale, lightness scale.
    Hsl(f32, f32, f32),
    /// Radial darkening at the frame edges.
    Vignette {
        /// Maximum darkening factor in `[0, 1]`.
        strength: f32,
        /// Normalised radius at which darkening begins (0 = centre, 1 = corner).
        radius: f32,
        /// Width of the fade in normalised units.
        feather: f32,
    },
}

// ─── Node wrapper ────────────────────────────────────────────────────────────

/// A named, enable-able colour grading node.
#[derive(Clone, Debug)]
pub struct ColorGradeNode {
    /// Human-readable node name.
    pub name: String,
    /// Whether this node is applied during `process`.
    pub enabled: bool,
    /// The colour operation this node performs.
    pub operation: ColorOperation,
}

impl ColorGradeNode {
    /// Create an enabled node with the given name and operation.
    pub fn new(name: impl Into<String>, operation: ColorOperation) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            operation,
        }
    }
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

/// A sequential pipeline of colour grading nodes.
pub struct ColorGradePipeline {
    /// Ordered list of grading nodes.
    pub nodes: Vec<ColorGradeNode>,
    /// Frame width required for position-dependent operations (e.g. vignette).
    pub width: u32,
    /// Frame height required for position-dependent operations (e.g. vignette).
    pub height: u32,
}

impl ColorGradePipeline {
    /// Create an empty pipeline for a frame of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            nodes: Vec::new(),
            width,
            height,
        }
    }

    /// Append a named, enabled node.  Returns `&mut Self` for builder chaining.
    pub fn add(&mut self, op: ColorOperation) -> &mut Self {
        let name = format!("node_{}", self.nodes.len());
        self.nodes.push(ColorGradeNode::new(name, op));
        self
    }

    /// Build a pipeline from ASC CDL parameters.
    ///
    /// Applies (in order):
    /// 1. `Gain(slope)` — multiply
    /// 2. `Lift(offset)` — shadow lift
    /// 3. `Gamma(power)` — power function
    pub fn from_cdl(
        slope: (f32, f32, f32),
        offset: (f32, f32, f32),
        power: (f32, f32, f32),
    ) -> Self {
        let mut pip = Self::new(0, 0);
        pip.add(ColorOperation::Gain(slope.0, slope.1, slope.2));
        pip.add(ColorOperation::Lift(offset.0, offset.1, offset.2));
        pip.add(ColorOperation::Gamma(power.0, power.1, power.2));
        pip
    }

    /// Apply all enabled nodes in order to an interleaved RGB `f32` pixel buffer.
    ///
    /// The input buffer must have `width * height * 3` elements.  A new buffer
    /// of the same length is returned.
    pub fn process(&self, pixels: &[f32]) -> Vec<f32> {
        let total = pixels.len();
        let pixel_count = total / 3;
        let mut out: Vec<f32> = pixels.to_vec();

        for node in &self.nodes {
            if !node.enabled {
                continue;
            }
            match &node.operation {
                ColorOperation::Lift(lr, lg, lb) => {
                    apply_lift_buffer(&mut out, *lr, *lg, *lb);
                }
                ColorOperation::Gamma(gr, gg, gb) => {
                    apply_gamma_buffer(&mut out, *gr, *gg, *gb);
                }
                ColorOperation::Gain(gr, gg, gb) => {
                    apply_gain_buffer(&mut out, *gr, *gg, *gb);
                }
                ColorOperation::Saturation(s) => {
                    apply_saturation_buffer(&mut out, *s);
                }
                ColorOperation::Temperature(t) => {
                    apply_temperature_buffer(&mut out, *t);
                }
                ColorOperation::Tint(t) => {
                    apply_tint_buffer(&mut out, *t);
                }
                ColorOperation::Curves(rc, gc, bc) => {
                    apply_curves_buffer(&mut out, rc, gc, bc);
                }
                ColorOperation::Hsl(hue_rot, sat_scale, lum_scale) => {
                    apply_hsl_buffer(&mut out, *hue_rot, *sat_scale, *lum_scale);
                }
                ColorOperation::Vignette {
                    strength,
                    radius,
                    feather,
                } => {
                    apply_vignette_buffer(
                        &mut out,
                        pixel_count,
                        self.width,
                        self.height,
                        *strength,
                        *radius,
                        *feather,
                    );
                }
            }
        }
        out
    }
}

// ─── Buffer-level helpers ────────────────────────────────────────────────────

fn apply_lift_buffer(buf: &mut [f32], lr: f32, lg: f32, lb: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        buf[i] = apply_lift(buf[i], lr);
        buf[i + 1] = apply_lift(buf[i + 1], lg);
        buf[i + 2] = apply_lift(buf[i + 2], lb);
        i += 3;
    }
}

fn apply_gamma_buffer(buf: &mut [f32], gr: f32, gg: f32, gb: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        buf[i] = apply_gamma(buf[i], gr);
        buf[i + 1] = apply_gamma(buf[i + 1], gg);
        buf[i + 2] = apply_gamma(buf[i + 2], gb);
        i += 3;
    }
}

fn apply_gain_buffer(buf: &mut [f32], gr: f32, gg: f32, gb: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        buf[i] = apply_gain(buf[i], gr);
        buf[i + 1] = apply_gain(buf[i + 1], gg);
        buf[i + 2] = apply_gain(buf[i + 2], gb);
        i += 3;
    }
}

fn apply_saturation_buffer(buf: &mut [f32], s: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        let (r, g, b) = apply_saturation(buf[i], buf[i + 1], buf[i + 2], s);
        buf[i] = r;
        buf[i + 1] = g;
        buf[i + 2] = b;
        i += 3;
    }
}

fn apply_temperature_buffer(buf: &mut [f32], t: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        let (r, g, b) = apply_temperature(buf[i], buf[i + 1], buf[i + 2], t);
        buf[i] = r;
        buf[i + 1] = g;
        buf[i + 2] = b;
        i += 3;
    }
}

fn apply_tint_buffer(buf: &mut [f32], t: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        let (r, g, b) = apply_tint(buf[i], buf[i + 1], buf[i + 2], t);
        buf[i] = r;
        buf[i + 1] = g;
        buf[i + 2] = b;
        i += 3;
    }
}

fn apply_curves_buffer(buf: &mut [f32], rc: &[(f32, f32)], gc: &[(f32, f32)], bc: &[(f32, f32)]) {
    let mut i = 0;
    while i + 2 < buf.len() {
        buf[i] = apply_curves_channel(buf[i], rc);
        buf[i + 1] = apply_curves_channel(buf[i + 1], gc);
        buf[i + 2] = apply_curves_channel(buf[i + 2], bc);
        i += 3;
    }
}

fn apply_hsl_buffer(buf: &mut [f32], hue_rot: f32, sat_scale: f32, lum_scale: f32) {
    let mut i = 0;
    while i + 2 < buf.len() {
        let (r, g, b) = apply_hsl(
            buf[i],
            buf[i + 1],
            buf[i + 2],
            hue_rot,
            sat_scale,
            lum_scale,
        );
        buf[i] = r;
        buf[i + 1] = g;
        buf[i + 2] = b;
        i += 3;
    }
}

fn apply_vignette_buffer(
    buf: &mut [f32],
    pixel_count: usize,
    width: u32,
    height: u32,
    strength: f32,
    radius: f32,
    feather: f32,
) {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let cx = 0.5_f32;
    let cy = 0.5_f32;

    for idx in 0..pixel_count {
        // Normalised pixel centre coordinates.
        let px = ((idx as u32 % width.max(1)) as f32 + 0.5) / w;
        let py = ((idx as u32 / width.max(1)) as f32 + 0.5) / h;

        // Elliptical distance from centre (accounts for aspect ratio).
        let dx = (px - cx) * 2.0;
        let dy = (py - cy) * 2.0;
        let dist = (dx * dx + dy * dy).sqrt();

        // Smooth falloff in [radius, radius + feather].
        let feather_end = (radius + feather).max(radius + f32::EPSILON);
        let t = ((dist - radius) / (feather_end - radius)).clamp(0.0, 1.0);
        // Smoothstep.
        let falloff = t * t * (3.0 - 2.0 * t);
        let factor = 1.0 - strength * falloff;

        let i = idx * 3;
        if i + 2 < buf.len() {
            buf[i] *= factor;
            buf[i + 1] *= factor;
            buf[i + 2] *= factor;
        }
    }
}

// ─── Pixel-level operations ──────────────────────────────────────────────────

/// `out = in + lift * (1 – in)`
pub fn apply_lift(v: f32, lift: f32) -> f32 {
    (v + lift * (1.0 - v)).clamp(0.0, 1.0)
}

/// `out = in.powf(1 / gamma)`  (identity when gamma = 1.0)
pub fn apply_gamma(v: f32, gamma: f32) -> f32 {
    if gamma <= 0.0 {
        return v;
    }
    v.max(0.0).powf(1.0 / gamma)
}

/// `out = in * gain`
pub fn apply_gain(v: f32, gain: f32) -> f32 {
    v * gain
}

/// Saturation via BT.709 luma.
pub fn apply_saturation(r: f32, g: f32, b: f32, s: f32) -> (f32, f32, f32) {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    (
        (luma + (r - luma) * s).clamp(0.0, 1.0),
        (luma + (g - luma) * s).clamp(0.0, 1.0),
        (luma + (b - luma) * s).clamp(0.0, 1.0),
    )
}

/// Temperature shift: positive = warm (boost red, reduce blue).
/// Scale is mapped so that ±6500 K roughly equals ±1.0 in linear.
pub fn apply_temperature(r: f32, g: f32, b: f32, t: f32) -> (f32, f32, f32) {
    // Normalise so that Δ6500 K maps to a ±1 full shift.
    let strength = (t / 6500.0).clamp(-1.0, 1.0);
    let r_out = (r + strength * 0.1).clamp(0.0, 1.0);
    let b_out = (b - strength * 0.1).clamp(0.0, 1.0);
    (r_out, g, b_out)
}

/// Green–magenta tint: positive = more green, negative = more magenta.
pub fn apply_tint(r: f32, g: f32, b: f32, t: f32) -> (f32, f32, f32) {
    let t_clamped = t.clamp(-1.0, 1.0);
    let g_out = (g + t_clamped * 0.1).clamp(0.0, 1.0);
    (r, g_out, b)
}

/// Cubic spline interpolation through `(input, output)` control points for
/// one channel.  Control points must be sorted by input.
///
/// Falls back to linear interpolation when fewer than 4 points are provided.
pub fn apply_curves_channel(v: f32, points: &[(f32, f32)]) -> f32 {
    if points.is_empty() {
        return v;
    }
    if points.len() == 1 {
        return points[0].1;
    }

    // Clamp to domain.
    if v <= points[0].0 {
        return points[0].1;
    }
    if v >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }

    // Find the segment containing v.
    let seg = points
        .windows(2)
        .position(|w| v >= w[0].0 && v < w[1].0)
        .unwrap_or(points.len() - 2);

    let p0 = points[seg];
    let p1 = points[seg + 1];

    // With only 2 points in a segment use linear.
    if points.len() < 4 {
        let t = (v - p0.0) / (p1.0 - p0.0).max(f32::EPSILON);
        return p0.1 + t * (p1.1 - p0.1);
    }

    // Catmull-Rom tangents (clamped at endpoints).
    let pm = if seg > 0 { points[seg - 1] } else { p0 };
    let p2 = if seg + 2 < points.len() {
        points[seg + 2]
    } else {
        p1
    };

    let t = (v - p0.0) / (p1.0 - p0.0).max(f32::EPSILON);
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom: m0 and m1 are tangents.
    let m0 = 0.5 * (p1.1 - pm.1);
    let m1 = 0.5 * (p2.1 - p0.1);

    let result = (2.0 * t3 - 3.0 * t2 + 1.0) * p0.1
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1.1
        + (t3 - t2) * m1;

    result.clamp(0.0, 1.0)
}

// ─── HSL helpers ─────────────────────────────────────────────────────────────

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let delta = max - min;
    if delta < f32::EPSILON {
        return (0.0, 0.0, l);
    }
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };
    let h = if (r - max).abs() < f32::EPSILON {
        (g - b) / delta + if g < b { 6.0 } else { 0.0 }
    } else if (g - max).abs() < f32::EPSILON {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };
    (h / 6.0, s, l)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s < f32::EPSILON {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

/// HSL-based hue rotation, saturation scale, lightness scale.
pub fn apply_hsl(
    r: f32,
    g: f32,
    b: f32,
    hue_rot: f32,
    sat_scale: f32,
    lum_scale: f32,
) -> (f32, f32, f32) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    let h2 = (h + hue_rot / 360.0).rem_euclid(1.0);
    let s2 = (s * sat_scale).clamp(0.0, 1.0);
    let l2 = (l * lum_scale).clamp(0.0, 1.0);
    hsl_to_rgb(h2, s2, l2)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pixel-level operations ───────────────────────────────────────────────

    #[test]
    fn test_lift_zero_is_identity() {
        assert!((apply_lift(0.4, 0.0) - 0.4).abs() < 1e-6);
        assert!((apply_lift(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((apply_lift(1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lift_positive_raises_shadows() {
        let v = apply_lift(0.1, 0.5);
        assert!(v > 0.1, "lift should raise value: {v}");
    }

    #[test]
    fn test_gamma_one_is_identity() {
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let out = apply_gamma(v, 1.0);
            assert!((out - v).abs() < 1e-5, "gamma(1.0) identity: {out} vs {v}");
        }
    }

    #[test]
    fn test_gamma_two_darkens_midtones() {
        let v = apply_gamma(0.5, 2.0);
        // 0.5^(1/2) = ~0.707
        assert!(v > 0.5, "gamma > 1 should lighten: {v}");
    }

    #[test]
    fn test_gain_multiplies() {
        let v = apply_gain(0.4, 2.0);
        assert!((v - 0.8).abs() < 1e-6, "gain 2x: {v}");
    }

    #[test]
    fn test_saturation_zero_greyscale() {
        let (r, g, b) = apply_saturation(0.8, 0.2, 0.1, 0.0);
        assert!((r - g).abs() < 1e-4, "r={r} g={g}");
        assert!((g - b).abs() < 1e-4, "g={g} b={b}");
    }

    #[test]
    fn test_saturation_one_identity() {
        let (r, g, b) = apply_saturation(0.6, 0.3, 0.9, 1.0);
        assert!((r - 0.6).abs() < 1e-5);
        assert!((g - 0.3).abs() < 1e-5);
        assert!((b - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_temperature_warm_shifts_red_up() {
        let (r, _, b) = apply_temperature(0.5, 0.5, 0.5, 3250.0);
        assert!(r > 0.5, "warm should boost red: {r}");
        assert!(b < 0.5, "warm should reduce blue: {b}");
    }

    #[test]
    fn test_temperature_cool_shifts_blue_up() {
        let (r, _, b) = apply_temperature(0.5, 0.5, 0.5, -3250.0);
        assert!(r < 0.5, "cool should reduce red: {r}");
        assert!(b > 0.5, "cool should boost blue: {b}");
    }

    #[test]
    fn test_tint_positive_boosts_green() {
        let (_, g, _) = apply_tint(0.5, 0.5, 0.5, 1.0);
        assert!(g > 0.5, "positive tint should boost green: {g}");
    }

    #[test]
    fn test_curves_identity_two_points() {
        let pts = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        for v in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let out = apply_curves_channel(v, &pts);
            assert!((out - v).abs() < 0.01, "curves identity: {out} vs {v}");
        }
    }

    #[test]
    fn test_curves_clamps_to_domain() {
        let pts = vec![(0.2_f32, 0.0), (0.8, 1.0)];
        assert_eq!(apply_curves_channel(0.0, &pts), 0.0);
        assert_eq!(apply_curves_channel(1.0, &pts), 1.0);
    }

    #[test]
    fn test_hsl_zero_rotation_is_identity() {
        let (r, g, b) = apply_hsl(0.7, 0.3, 0.5, 0.0, 1.0, 1.0);
        assert!((r - 0.7).abs() < 0.01, "r={r}");
        assert!((g - 0.3).abs() < 0.01, "g={g}");
        assert!((b - 0.5).abs() < 0.01, "b={b}");
    }

    // ── Pipeline tests ───────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_single_gain() {
        let mut pip = ColorGradePipeline::new(1, 1);
        pip.add(ColorOperation::Gain(2.0, 2.0, 2.0));
        let pixels = vec![0.2_f32, 0.3, 0.4];
        let out = pip.process(&pixels);
        assert!((out[0] - 0.4).abs() < 1e-5, "R={}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-5, "G={}", out[1]);
        assert!((out[2] - 0.8).abs() < 1e-5, "B={}", out[2]);
    }

    #[test]
    fn test_pipeline_disabled_node_skipped() {
        let mut pip = ColorGradePipeline::new(1, 1);
        pip.add(ColorOperation::Gain(10.0, 10.0, 10.0));
        // Disable the node.
        pip.nodes[0].enabled = false;
        let pixels = vec![0.2_f32, 0.3, 0.4];
        let out = pip.process(&pixels);
        assert!((out[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_from_cdl_identity() {
        // slope=1, offset=0, power=1 → identity.
        let pip = ColorGradePipeline::from_cdl((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        let pixels = vec![0.3_f32, 0.5, 0.7];
        let out = pip.process(&pixels);
        // Gain(1)*Lift(0)*Gamma(1) = identity.
        assert!((out[0] - 0.3).abs() < 0.01, "R={}", out[0]);
        assert!((out[1] - 0.5).abs() < 0.01, "G={}", out[1]);
        assert!((out[2] - 0.7).abs() < 0.01, "B={}", out[2]);
    }

    #[test]
    fn test_vignette_centre_unaffected() {
        // 3×3 image; the centre pixel (index 4) should be almost untouched when
        // radius = 1.0 (vignette starts at the very edge).
        let mut pip = ColorGradePipeline::new(3, 3);
        pip.add(ColorOperation::Vignette {
            strength: 1.0,
            radius: 1.0,
            feather: 0.1,
        });
        let pixels: Vec<f32> = vec![1.0; 3 * 3 * 3];
        let out = pip.process(&pixels);
        let centre = out[4 * 3]; // pixel 4, R channel
        assert!(
            (centre - 1.0).abs() < 0.05,
            "centre pixel should be near 1.0: {centre}"
        );
    }
}
