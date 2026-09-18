//! Tone curves for color grading.
//!
//! Provides per-channel and master RGB tone curves with interpolation,
//! LUT generation, and HSL-based curve application.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// A point on a tone curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvePoint {
    /// Input value (normalized 0..1).
    pub input: f32,
    /// Output value (normalized 0..1).
    pub output: f32,
}

impl CurvePoint {
    /// Create a new curve point.
    #[must_use]
    pub fn new(input: f32, output: f32) -> Self {
        Self { input, output }
    }
}

/// A tone curve defined by control points with linear interpolation between them.
#[derive(Debug, Clone)]
pub struct ToneCurve {
    /// Control points, sorted by input value.
    pub points: Vec<CurvePoint>,
}

impl ToneCurve {
    /// Create a new tone curve from a list of control points.
    ///
    /// Points will be sorted by input value.
    #[must_use]
    pub fn new(mut points: Vec<CurvePoint>) -> Self {
        points.sort_by(|a, b| {
            a.input
                .partial_cmp(&b.input)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { points }
    }

    /// Identity curve (output = input).
    #[must_use]
    pub fn identity() -> Self {
        Self::new(vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)])
    }

    /// S-curve (contrast enhancement, darker shadows, brighter highlights).
    #[must_use]
    pub fn s_curve() -> Self {
        Self::new(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.25, 0.18),
            CurvePoint::new(0.5, 0.5),
            CurvePoint::new(0.75, 0.82),
            CurvePoint::new(1.0, 1.0),
        ])
    }

    /// Logarithmic curve (lifts shadows, compresses highlights).
    #[must_use]
    pub fn logarithmic() -> Self {
        Self::new(vec![
            CurvePoint::new(0.0, 0.07),
            CurvePoint::new(0.25, 0.35),
            CurvePoint::new(0.5, 0.58),
            CurvePoint::new(0.75, 0.78),
            CurvePoint::new(1.0, 1.0),
        ])
    }

    /// Apply the tone curve to an input value using linear interpolation between control points.
    #[must_use]
    pub fn apply(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);

        if self.points.is_empty() {
            return x;
        }
        if self.points.len() == 1 {
            return self.points[0].output;
        }

        // Edge cases
        if x <= self.points[0].input {
            return self.points[0].output;
        }
        if x >= self.points[self.points.len() - 1].input {
            return self.points[self.points.len() - 1].output;
        }

        // Find the segment containing x
        for i in 0..self.points.len() - 1 {
            let p0 = &self.points[i];
            let p1 = &self.points[i + 1];
            if x >= p0.input && x <= p1.input {
                let range = p1.input - p0.input;
                if range < 1e-10 {
                    return p0.output;
                }
                let t = (x - p0.input) / range;
                return p0.output + t * (p1.output - p0.output);
            }
        }

        x
    }

    /// Compute a LUT (lookup table) by sampling the curve at `size` evenly spaced points.
    #[must_use]
    pub fn compute_lut(&self, size: usize) -> Vec<f32> {
        if size == 0 {
            return Vec::new();
        }
        (0..size)
            .map(|i| {
                let x = i as f32 / (size - 1).max(1) as f32;
                self.apply(x)
            })
            .collect()
    }
}

impl Default for ToneCurve {
    fn default() -> Self {
        Self::identity()
    }
}

/// RGB tone curves with a master curve applied first.
#[derive(Debug, Clone)]
pub struct RgbCurves {
    /// Red channel curve.
    pub r: ToneCurve,
    /// Green channel curve.
    pub g: ToneCurve,
    /// Blue channel curve.
    pub b: ToneCurve,
    /// Master curve applied before per-channel curves.
    pub master: ToneCurve,
}

impl RgbCurves {
    /// Create new RGB curves with custom master and per-channel curves.
    #[must_use]
    pub fn new(r: ToneCurve, g: ToneCurve, b: ToneCurve, master: ToneCurve) -> Self {
        Self { r, g, b, master }
    }

    /// Identity RGB curves (no change).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            r: ToneCurve::identity(),
            g: ToneCurve::identity(),
            b: ToneCurve::identity(),
            master: ToneCurve::identity(),
        }
    }

    /// Apply master curve first, then per-channel curves.
    #[must_use]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Master curve applied first to all channels
        let r = self.master.apply(r);
        let g = self.master.apply(g);
        let b = self.master.apply(b);

        // Per-channel curves
        let r = self.r.apply(r);
        let g = self.g.apply(g);
        let b = self.b.apply(b);

        (r, g, b)
    }
}

impl Default for RgbCurves {
    fn default() -> Self {
        Self::identity()
    }
}

/// HSL-based tone curves for hue, saturation, and luminance.
#[derive(Debug, Clone)]
pub struct HslCurves {
    /// Hue curve (applied to hue channel, 0..1 maps to 0..360 degrees).
    pub hue: ToneCurve,
    /// Saturation curve.
    pub saturation: ToneCurve,
    /// Luminance curve.
    pub luminance: ToneCurve,
}

impl HslCurves {
    /// Create new HSL curves.
    #[must_use]
    pub fn new(hue: ToneCurve, saturation: ToneCurve, luminance: ToneCurve) -> Self {
        Self {
            hue,
            saturation,
            luminance,
        }
    }

    /// Identity HSL curves.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            hue: ToneCurve::identity(),
            saturation: ToneCurve::identity(),
            luminance: ToneCurve::identity(),
        }
    }

    /// Apply HSL curves to an RGB pixel.
    ///
    /// Converts to HSL, applies curves, converts back to RGB.
    #[must_use]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let (h, s, l) = rgb_to_hsl(r, g, b);

        let h_norm = h / 360.0;
        let h_new = self.hue.apply(h_norm) * 360.0;
        let s_new = self.saturation.apply(s);
        let l_new = self.luminance.apply(l);

        hsl_to_rgb(h_new, s_new, l_new)
    }
}

impl Default for HslCurves {
    fn default() -> Self {
        Self::identity()
    }
}

/// Convert RGB to HSL.
///
/// Returns (hue_degrees, saturation, lightness) all in 0..1 except hue which is 0..360.
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta < 1e-10 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (max - r).abs() < 1e-10 {
        (g - b) / delta + if g < b { 6.0 } else { 0.0 }
    } else if (max - g).abs() < 1e-10 {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    (h * 60.0, s, l)
}

/// Convert HSL to RGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);

    if s < 1e-10 {
        return (l, l, l);
    }

    let h = h.rem_euclid(360.0) / 360.0;

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = t.rem_euclid(1.0);
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tone_curve_identity() {
        let curve = ToneCurve::identity();
        assert!((curve.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((curve.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tone_curve_apply_clamps() {
        let curve = ToneCurve::identity();
        assert!((curve.apply(-0.5) - 0.0).abs() < 1e-6);
        assert!((curve.apply(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tone_curve_s_curve() {
        let curve = ToneCurve::s_curve();
        // S-curve shadows should be darker than identity
        assert!(curve.apply(0.25) < 0.25);
        // S-curve midpoint should be at 0.5
        assert!((curve.apply(0.5) - 0.5).abs() < 0.01);
        // S-curve highlights should be brighter than identity
        assert!(curve.apply(0.75) > 0.75);
    }

    #[test]
    fn test_tone_curve_logarithmic() {
        let curve = ToneCurve::logarithmic();
        // Logarithmic should lift shadows (output > input for low values)
        assert!(curve.apply(0.0) > 0.0);
        // Should still reach 1.0 at the top
        assert!((curve.apply(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_tone_curve_compute_lut() {
        let curve = ToneCurve::identity();
        let lut = curve.compute_lut(256);
        assert_eq!(lut.len(), 256);
        assert!((lut[0] - 0.0).abs() < 1e-6);
        assert!((lut[255] - 1.0).abs() < 1e-6);
        assert!((lut[128] - 128.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn test_tone_curve_compute_lut_zero_size() {
        let curve = ToneCurve::identity();
        let lut = curve.compute_lut(0);
        assert!(lut.is_empty());
    }

    #[test]
    fn test_tone_curve_interpolation() {
        let curve = ToneCurve::new(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.5, 0.8),
            CurvePoint::new(1.0, 1.0),
        ]);
        // At midpoint input=0.5 → output=0.8
        assert!((curve.apply(0.5) - 0.8).abs() < 1e-5);
        // Between 0 and 0.5, should be linearly interpolated toward 0.8
        let mid = curve.apply(0.25);
        assert!((mid - 0.4).abs() < 1e-4);
    }

    #[test]
    fn test_rgb_curves_identity() {
        let curves = RgbCurves::identity();
        let (r, g, b) = curves.apply_pixel(0.6, 0.4, 0.2);
        assert!((r - 0.6).abs() < 1e-5);
        assert!((g - 0.4).abs() < 1e-5);
        assert!((b - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_curves_master_applies_first() {
        // Master S-curve should affect all channels
        let curves = RgbCurves::new(
            ToneCurve::identity(),
            ToneCurve::identity(),
            ToneCurve::identity(),
            ToneCurve::s_curve(),
        );
        let (r, g, b) = curves.apply_pixel(0.25, 0.25, 0.25);
        // S-curve darkens shadows (< 0.25 for input of 0.25)
        assert!(r < 0.25);
        assert!((r - g).abs() < 1e-5);
        assert!((g - b).abs() < 1e-5);
    }

    #[test]
    fn test_hsl_curves_identity() {
        let curves = HslCurves::identity();
        let (r, g, b) = curves.apply_pixel(0.8, 0.4, 0.2);
        assert!((r - 0.8).abs() < 0.01);
        assert!((g - 0.4).abs() < 0.01);
        assert!((b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_rgb_to_hsl_roundtrip() {
        let r = 0.6f32;
        let g = 0.3f32;
        let b = 0.8f32;
        let (h, s, l) = rgb_to_hsl(r, g, b);
        let (r2, g2, b2) = hsl_to_rgb(h, s, l);
        assert!((r - r2).abs() < 0.001);
        assert!((g - g2).abs() < 0.001);
        assert!((b - b2).abs() < 0.001);
    }

    #[test]
    fn test_hsl_gray_roundtrip() {
        let (h, s, l) = rgb_to_hsl(0.5, 0.5, 0.5);
        assert!(s < 1e-6);
        let (r, g, b) = hsl_to_rgb(h, s, l);
        assert!((r - 0.5).abs() < 1e-5);
        assert!((g - 0.5).abs() < 1e-5);
        assert!((b - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_hsl_saturation_curve() {
        // Desaturate everything
        let curves = HslCurves::new(
            ToneCurve::identity(),
            ToneCurve::new(vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 0.0)]),
            ToneCurve::identity(),
        );
        let (r, g, b) = curves.apply_pixel(0.8, 0.2, 0.4);
        // All channels should be equal (grayscale)
        assert!((r - g).abs() < 0.01);
        assert!((g - b).abs() < 0.01);
    }

    #[test]
    fn test_curve_point_new() {
        let p = CurvePoint::new(0.3, 0.7);
        assert!((p.input - 0.3).abs() < 1e-6);
        assert!((p.output - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_tone_curve_empty_points() {
        let curve = ToneCurve::new(vec![]);
        let v = curve.apply(0.5);
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tone_curve_single_point() {
        let curve = ToneCurve::new(vec![CurvePoint::new(0.5, 0.8)]);
        let v = curve.apply(0.3);
        assert!((v - 0.8).abs() < 1e-5);
    }
}
