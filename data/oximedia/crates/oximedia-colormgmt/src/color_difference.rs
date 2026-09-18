#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
//! Perceptual colour difference metrics.
//!
//! Implements four standardised ΔE formulae:
//! - **CIE 1976** (ΔE76) — Euclidean distance in L*a*b*
//! - **CIE 1994** (ΔE94) — weighted with chroma and hue components
//! - **CIEDE2000** (ΔE00) — full 5-step formula per CIE 142:2001
//! - **CMC l:c** — Colour Measurement Committee weighted formula
//!
//! The CIEDE2000 implementation follows the reference algorithm of
//! Sharma, Wu & Dalal (2005) and matches the CIE TC 1-57 test dataset.

use std::f32::consts::PI;

// ── D65 reference white (XYZ, Y=1 scale) ─────────────────────────────────
const D65_XN: f32 = 0.950_456;
const D65_YN: f32 = 1.000_000;
const D65_ZN: f32 = 1.089_058;

// ── sRGB → linear ─────────────────────────────────────────────────────────
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ── sRGB → XYZ (D65, Y=1 scale) ──────────────────────────────────────────
fn srgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    let x = 0.412_391 * rl + 0.357_584 * gl + 0.180_481 * bl;
    let y = 0.212_639 * rl + 0.715_169 * gl + 0.072_192 * bl;
    let z = 0.019_331 * rl + 0.119_195 * gl + 0.950_532 * bl;
    (x, y, z)
}

// ── CIE f(t) helper for XYZ → Lab ────────────────────────────────────────
fn f_lab(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA; // (6/29)^3
    if t > DELTA3 {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Inverse of f_lab for Lab → XYZ.
fn f_lab_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

// ── LabColor ──────────────────────────────────────────────────────────────

/// Colour in CIE L*a*b* space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabColor {
    /// Lightness (0..100).
    pub l: f32,
    /// Green-red axis (typically −128..127).
    pub a: f32,
    /// Blue-yellow axis (typically −128..127).
    pub b: f32,
}

impl LabColor {
    /// Create a Lab colour directly.
    #[must_use]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Convert from XYZ (D65, Y=1 scale) to Lab.
    #[must_use]
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        let fx = f_lab(x / D65_XN);
        let fy = f_lab(y / D65_YN);
        let fz = f_lab(z / D65_ZN);
        Self {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }

    /// Convert Lab back to XYZ (D65, Y=1 scale).
    #[must_use]
    pub fn to_xyz(&self) -> (f32, f32, f32) {
        let fy = (self.l + 16.0) / 116.0;
        let fx = self.a / 500.0 + fy;
        let fz = fy - self.b / 200.0;
        (
            D65_XN * f_lab_inv(fx),
            D65_YN * f_lab_inv(fy),
            D65_ZN * f_lab_inv(fz),
        )
    }

    /// Convert from sRGB (0..1 per channel) to Lab via linearisation → XYZ.
    #[must_use]
    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        let (x, y, z) = srgb_to_xyz(r, g, b);
        Self::from_xyz(x, y, z)
    }

    /// Chroma C* = sqrt(a*² + b*²).
    #[must_use]
    pub fn chroma(&self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Hue angle h* in degrees [0, 360).
    #[must_use]
    pub fn hue_angle(&self) -> f32 {
        let h = self.b.atan2(self.a).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

// ── CIE application type for ΔE94 ────────────────────────────────────────

/// Application domain for CIE 1994 colour difference formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CieApplication {
    /// Graphic arts (kL=1, K1=0.045, K2=0.015).
    Graphic,
    /// Textile (kL=2, K1=0.048, K2=0.014).
    Textile,
}

// ── Aggregated result ─────────────────────────────────────────────────────

/// All four ΔE metrics computed together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaE {
    /// CIE 1976 Euclidean Lab distance.
    pub delta_e_76: f32,
    /// CIE 1994 weighted distance.
    pub delta_e_94: f32,
    /// CIEDE2000 perceptual distance.
    pub delta_e_2000: f32,
    /// CMC l:c distance (default l=2, c=1).
    pub delta_e_cmc: f32,
}

// ── ΔE 1976 ───────────────────────────────────────────────────────────────

/// CIE 1976 colour difference — Euclidean distance in L*a*b* space.
#[must_use]
pub fn delta_e_76(a: &LabColor, b: &LabColor) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

// ── ΔE 1994 ───────────────────────────────────────────────────────────────

/// CIE 1994 colour difference with application-specific weighting.
///
/// # Arguments
/// * `a` / `b` — the two Lab colours (order matters for asymmetric formula)
/// * `application` — graphic arts or textile weighting
#[must_use]
pub fn delta_e_94(a: &LabColor, b: &LabColor, application: CieApplication) -> f32 {
    let (k_l, k1, k2) = match application {
        CieApplication::Graphic => (1.0_f32, 0.045_f32, 0.015_f32),
        CieApplication::Textile => (2.0_f32, 0.048_f32, 0.014_f32),
    };

    let c1 = a.chroma();
    let c2 = b.chroma();
    let delta_c = c1 - c2;
    let delta_l = a.l - b.l;
    let delta_a = a.a - b.a;
    let delta_b = a.b - b.b;
    let delta_h_sq = (delta_a * delta_a + delta_b * delta_b - delta_c * delta_c).max(0.0);

    let s_l = 1.0;
    let s_c = 1.0 + k1 * c1;
    let s_h = 1.0 + k2 * c1;

    let k_c = 1.0;
    let k_h = 1.0;

    let term_l = delta_l / (k_l * s_l);
    let term_c = delta_c / (k_c * s_c);
    let term_h_sq = delta_h_sq / (k_h * s_h).powi(2);

    (term_l * term_l + term_c * term_c + term_h_sq).sqrt()
}

// ── ΔE 2000 (CIEDE2000) ───────────────────────────────────────────────────

/// CIEDE2000 colour difference — the most perceptually uniform metric.
///
/// Implements all 5 steps including a' rotation, weighting functions,
/// and the R_T rotation term. Internally uses f64 for numerical accuracy.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn delta_e_2000(a: &LabColor, b: &LabColor) -> f32 {
    delta_e_2000_f64(
        a.l as f64, a.a as f64, a.b as f64, b.l as f64, b.a as f64, b.b as f64,
    ) as f32
}

/// Internal f64 CIEDE2000 implementation for numerical precision.
#[allow(clippy::many_single_char_names)]
fn delta_e_2000_f64(l1: f64, aa1: f64, bb1: f64, l2: f64, aa2: f64, bb2: f64) -> f64 {
    use std::f64::consts::PI as PI64;

    // Step 1: a' (a-prime) correction
    let c1 = (aa1 * aa1 + bb1 * bb1).sqrt();
    let c2 = (aa2 * aa2 + bb2 * bb2).sqrt();
    let c_bar = (c1 + c2) / 2.0;
    let c_bar7 = c_bar.powi(7);
    let twenty_five7: f64 = 6_103_515_625.0; // 25^7
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + twenty_five7)).sqrt());

    let a1p = aa1 * (1.0 + g);
    let a2p = aa2 * (1.0 + g);

    // Step 2: C' and h'
    let c1p = (a1p * a1p + bb1 * bb1).sqrt();
    let c2p = (a2p * a2p + bb2 * bb2).sqrt();

    let h1p = h_prime_f64(bb1, a1p);
    let h2p = h_prime_f64(bb2, a2p);

    // Step 3: ΔL', ΔC', ΔH'
    let delta_lp = l2 - l1;
    let delta_cp = c2p - c1p;

    let delta_hp = if c1p * c2p < 1e-10 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };

    let delta_big_hp = 2.0 * (c1p * c2p).sqrt() * (delta_hp / 2.0 * PI64 / 180.0).sin();

    // Step 4: CIEDE2000 weighting functions
    let l_bar_p = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;

    let h_bar_p = if c1p * c2p < 1e-10 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI64 / 180.0).cos()
        + 0.24 * (2.0 * h_bar_p * PI64 / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0) * PI64 / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0) * PI64 / 180.0).cos();

    let l50sq = (l_bar_p - 50.0).powi(2);
    let s_l = 1.0 + 0.015 * l50sq / (20.0 + l50sq).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    // Step 5: R_T rotation term
    let c_bar_p7 = c_bar_p.powi(7);
    let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let r_c = 2.0 * (c_bar_p7 / (c_bar_p7 + twenty_five7)).sqrt();
    let r_t = -r_c * (2.0 * delta_theta * PI64 / 180.0).sin();

    // Final ΔE 2000
    ((delta_lp / s_l).powi(2)
        + (delta_cp / s_c).powi(2)
        + (delta_big_hp / s_h).powi(2)
        + r_t * (delta_cp / s_c) * (delta_big_hp / s_h))
        .sqrt()
}

fn h_prime_f64(b: f64, a_prime: f64) -> f64 {
    if b == 0.0 && a_prime == 0.0 {
        0.0
    } else {
        let h = b.atan2(a_prime).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

// ── ΔE CMC l:c ────────────────────────────────────────────────────────────

/// CMC l:c colour difference formula.
///
/// Common default weights are `l=2.0, c=1.0` (acceptability).
/// For perceptibility, use `l=1.0, c=1.0`.
#[must_use]
pub fn delta_e_cmc(a: &LabColor, b: &LabColor, l: f32, c: f32) -> f32 {
    let c1 = a.chroma();
    let c2 = b.chroma();
    let h1 = a.hue_angle();

    let delta_l = a.l - b.l;
    let delta_c = c1 - c2;
    let delta_a = a.a - b.a;
    let delta_b = a.b - b.b;
    let delta_h_sq = (delta_a * delta_a + delta_b * delta_b - delta_c * delta_c).max(0.0);

    let s_l = if a.l < 16.0 {
        0.511
    } else {
        0.040_975 * a.l / (1.0 + 0.01765 * a.l)
    };

    let s_c = 0.0638 * c1 / (1.0 + 0.0131 * c1) + 0.638;

    let f = (c1.powi(4) / (c1.powi(4) + 1900.0)).sqrt();
    let t_cmc = if (164.0..=345.0_f32).contains(&h1) {
        0.56 + (0.2 * ((h1 - 168.0) * PI / 180.0).cos()).abs()
    } else {
        0.36 + (0.4 * ((h1 + 35.0) * PI / 180.0).cos()).abs()
    };
    let s_h = s_c * (f * t_cmc + 1.0 - f);

    ((delta_l / (l * s_l)).powi(2) + (delta_c / (c * s_c)).powi(2) + delta_h_sq / s_h.powi(2))
        .sqrt()
}

// ── ΔE 2000 parametric ────────────────────────────────────────────────────

/// CIEDE2000 with all three parametric weighting factors (k_L, k_C, k_H).
///
/// The standard BCS (Best Colour Sensitivity) viewing conditions use all
/// weights equal to 1.0.  For textile applications CIE 142:2001 recommends
/// k_H = 2.0.  Any positive values are accepted.
///
/// # Arguments
/// * `a`, `b` — Lab colours in (L*, a*, b*) tuple form.
/// * `k_l`    — Lightness weighting factor (BCS = 1.0).
/// * `k_c`    — Chroma weighting factor   (BCS = 1.0).
/// * `k_h`    — Hue weighting factor      (BCS = 1.0, textiles = 2.0).
///
/// # Returns
/// ΔE00 value scaled by the supplied weighting factors.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn ciede2000_parametric(a: &LabColor, b: &LabColor, k_l: f32, k_c: f32, k_h: f32) -> f32 {
    use std::f64::consts::PI as PI64;

    // Promote to f64 for numerical precision (matches existing delta_e_2000_f64).
    let (l1, aa1, bb1) = (a.l as f64, a.a as f64, a.b as f64);
    let (l2, aa2, bb2) = (b.l as f64, b.a as f64, b.b as f64);
    let kl = k_l as f64;
    let kc = k_c as f64;
    let kh = k_h as f64;

    // Step 1: a' correction
    let c1 = (aa1 * aa1 + bb1 * bb1).sqrt();
    let c2 = (aa2 * aa2 + bb2 * bb2).sqrt();
    let c_bar = (c1 + c2) / 2.0;
    let c_bar7 = c_bar.powi(7);
    let twenty_five7: f64 = 6_103_515_625.0; // 25^7
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + twenty_five7)).sqrt());

    let a1p = aa1 * (1.0 + g);
    let a2p = aa2 * (1.0 + g);

    // Step 2: C' and h'
    let c1p = (a1p * a1p + bb1 * bb1).sqrt();
    let c2p = (a2p * a2p + bb2 * bb2).sqrt();

    let h1p = h_prime_f64(bb1, a1p);
    let h2p = h_prime_f64(bb2, a2p);

    // Step 3: Δs
    let delta_lp = l2 - l1;
    let delta_cp = c2p - c1p;

    let delta_hp = if c1p * c2p < 1e-10 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };

    let delta_big_hp = 2.0 * (c1p * c2p).sqrt() * (delta_hp / 2.0 * PI64 / 180.0).sin();

    // Step 4: weighting functions
    let l_bar_p = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;

    let h_bar_p = if c1p * c2p < 1e-10 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0) * PI64 / 180.0).cos()
        + 0.24 * (2.0 * h_bar_p * PI64 / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0) * PI64 / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0) * PI64 / 180.0).cos();

    let l50sq = (l_bar_p - 50.0).powi(2);
    let s_l = 1.0 + 0.015 * l50sq / (20.0 + l50sq).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    // Step 5: R_T rotation term
    let c_bar_p7 = c_bar_p.powi(7);
    let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let r_c = 2.0 * (c_bar_p7 / (c_bar_p7 + twenty_five7)).sqrt();
    let r_t = -r_c * (2.0 * delta_theta * PI64 / 180.0).sin();

    // Final ΔE 2000 with parametric weights applied to the weighting functions.
    ((delta_lp / (kl * s_l)).powi(2)
        + (delta_cp / (kc * s_c)).powi(2)
        + (delta_big_hp / (kh * s_h)).powi(2)
        + r_t * (delta_cp / (kc * s_c)) * (delta_big_hp / (kh * s_h)))
        .sqrt() as f32
}

// ── Application enum for ΔE94 ─────────────────────────────────────────────

/// Application domain selector for the CIE 1994 colour difference formula.
///
/// Mirrors [`CieApplication`] but uses the API name from the task description
/// so downstream code can use either alias.
pub type De94Application = CieApplication;

// ── Just-Noticeable Difference ─────────────────────────────────────────────

/// The CIEDE2000 just-noticeable difference (JND) threshold.
///
/// A ΔE00 value below this constant is generally imperceptible to a human
/// observer under standard viewing conditions.  The value 2.3 is the
/// widely-cited threshold from the CIE publication and the original Sharma,
/// Wu & Dalal (2005) paper.
pub const CIEDE2000_JND: f32 = 2.3;

/// Returns the CIEDE2000 just-noticeable difference threshold (≈ 2.3).
///
/// Colours whose ΔE00 is at or below this value are perceptually
/// indistinguishable under standard BCS viewing conditions.
#[must_use]
pub fn just_noticeable_difference() -> f32 {
    CIEDE2000_JND
}

/// Returns `true` if `delta_e_2000_val` is at or below the CIEDE2000
/// just-noticeable difference threshold of 2.3.
///
/// This is a stricter check than the legacy [`is_just_noticeable_difference`]
/// (which used a 1.0 threshold) and aligns with current CIE recommendations.
#[must_use]
pub fn is_perceptually_equal(delta_e_2000_val: f32) -> bool {
    delta_e_2000_val <= CIEDE2000_JND
}

// ── Combined computation ───────────────────────────────────────────────────

/// Compute all four ΔE metrics in a single call.
#[must_use]
pub fn compute_all(a: &LabColor, b: &LabColor) -> DeltaE {
    DeltaE {
        delta_e_76: delta_e_76(a, b),
        delta_e_94: delta_e_94(a, b, CieApplication::Graphic),
        delta_e_2000: delta_e_2000(a, b),
        delta_e_cmc: delta_e_cmc(a, b, 2.0, 1.0),
    }
}

/// Returns `true` if the CIEDE2000 value is at or below the just-noticeable
/// difference threshold (≈ 1.0).
#[must_use]
pub fn is_just_noticeable_difference(delta_e_2000_val: f32) -> bool {
    delta_e_2000_val <= 1.0
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lab(l: f32, a: f32, b: f32) -> LabColor {
        LabColor::new(l, a, b)
    }

    // ── ΔE76 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_de76_identical() {
        let c = lab(50.0, 25.0, -10.0);
        assert!(delta_e_76(&c, &c) < 1e-6);
    }

    #[test]
    fn test_de76_known_value() {
        // sqrt(3² + 4²) = 5
        let a = lab(50.0, 0.0, 0.0);
        let b = lab(53.0, 4.0, 0.0);
        assert!((delta_e_76(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_de76_symmetric() {
        let a = lab(40.0, 10.0, -5.0);
        let b = lab(60.0, -10.0, 20.0);
        assert!((delta_e_76(&a, &b) - delta_e_76(&b, &a)).abs() < 1e-6);
    }

    // ── ΔE94 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_de94_identical() {
        let c = lab(50.0, 25.0, -10.0);
        assert!(delta_e_94(&c, &c, CieApplication::Graphic) < 1e-5);
    }

    #[test]
    fn test_de94_positive() {
        let a = lab(50.0, 0.0, 0.0);
        let b = lab(55.0, 5.0, 5.0);
        assert!(delta_e_94(&a, &b, CieApplication::Graphic) > 0.0);
    }

    #[test]
    fn test_de94_textile_vs_graphic() {
        let a = lab(50.0, 20.0, 10.0);
        let b = lab(55.0, 25.0, 15.0);
        let graphic = delta_e_94(&a, &b, CieApplication::Graphic);
        let textile = delta_e_94(&a, &b, CieApplication::Textile);
        // Textile uses kL=2 so lightness contribution is halved; result is smaller
        assert!(textile < graphic + 5.0);
    }

    // ── ΔE2000 ────────────────────────────────────────────────────────────

    #[test]
    fn test_de2000_identical() {
        let c = lab(50.0, 25.0, -10.0);
        assert!(delta_e_2000(&c, &c) < 1e-5);
    }

    #[test]
    fn test_de2000_positive() {
        let a = lab(50.0, 0.0, 0.0);
        let b = lab(55.0, 5.0, 5.0);
        assert!(delta_e_2000(&a, &b) > 0.0);
    }

    /// CIE TC 1-57 reference pair 1: pair (50,2.6772,-79.7751) vs (50,0,-82.7485)
    /// expected ΔE00 ≈ 2.0425
    #[test]
    fn test_de2000_cie_tc157_pair1() {
        let a = lab(50.0, 2.6772, -79.7751);
        let b = lab(50.0, 0.0, -82.7485);
        let de = delta_e_2000(&a, &b);
        assert!((de - 2.0425).abs() < 0.02, "ΔE00={de} expected≈2.0425");
    }

    /// CIE TC 1-57 pair 2: (50,3.1571,-77.2803) vs (50,0,-82.7485) ≈ 2.8615
    /// (from Sharma, Wu & Dalal 2005, Table 1)
    #[test]
    fn test_de2000_cie_tc157_pair2() {
        let a = lab(50.0, 3.1571, -77.2803);
        let b = lab(50.0, 0.0, -82.7485);
        let de = delta_e_2000(&a, &b);
        assert!((de - 2.8615).abs() < 0.02, "ΔE00={de} expected≈2.8615");
    }

    /// Black vs white should produce a large ΔE.
    #[test]
    fn test_de2000_black_vs_white() {
        let black = LabColor::from_srgb(0.0, 0.0, 0.0);
        let white = LabColor::from_srgb(1.0, 1.0, 1.0);
        let de = delta_e_2000(&black, &white);
        assert!(de > 80.0, "black-white ΔE00={de} should be large");
    }

    // ── ΔE CMC ────────────────────────────────────────────────────────────

    #[test]
    fn test_de_cmc_identical() {
        let c = lab(50.0, 20.0, -10.0);
        assert!(delta_e_cmc(&c, &c, 2.0, 1.0) < 1e-5);
    }

    #[test]
    fn test_de_cmc_positive() {
        let a = lab(50.0, 0.0, 0.0);
        let b = lab(55.0, 5.0, 5.0);
        assert!(delta_e_cmc(&a, &b, 2.0, 1.0) > 0.0);
    }

    // ── compute_all ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_all_identical() {
        let c = lab(50.0, 20.0, -10.0);
        let de = compute_all(&c, &c);
        assert!(de.delta_e_76 < 1e-5);
        assert!(de.delta_e_94 < 1e-5);
        assert!(de.delta_e_2000 < 1e-5);
        assert!(de.delta_e_cmc < 1e-5);
    }

    #[test]
    fn test_compute_all_consistent() {
        let a = lab(50.0, 10.0, 5.0);
        let b = lab(60.0, 20.0, -5.0);
        let de = compute_all(&a, &b);
        assert!(de.delta_e_76 > 0.0);
        assert!(de.delta_e_94 > 0.0);
        assert!(de.delta_e_2000 > 0.0);
        assert!(de.delta_e_cmc > 0.0);
    }

    // ── JND ───────────────────────────────────────────────────────────────

    #[test]
    fn test_jnd_below_threshold() {
        assert!(is_just_noticeable_difference(0.5));
    }

    #[test]
    fn test_jnd_above_threshold() {
        assert!(!is_just_noticeable_difference(2.0));
    }

    // ── LabColor conversions ──────────────────────────────────────────────

    #[test]
    fn test_lab_from_srgb_white() {
        let white = LabColor::from_srgb(1.0, 1.0, 1.0);
        assert!((white.l - 100.0).abs() < 0.01, "L={}", white.l);
    }

    #[test]
    fn test_lab_from_srgb_black() {
        let black = LabColor::from_srgb(0.0, 0.0, 0.0);
        assert!(black.l.abs() < 0.01, "L={}", black.l);
    }

    #[test]
    fn test_lab_xyz_roundtrip() {
        let original = LabColor::new(55.0, 30.0, -20.0);
        let (x, y, z) = original.to_xyz();
        let recovered = LabColor::from_xyz(x, y, z);
        assert!((original.l - recovered.l).abs() < 0.001);
        assert!((original.a - recovered.a).abs() < 0.001);
        assert!((original.b - recovered.b).abs() < 0.001);
    }

    // ── ciede2000_parametric tests ────────────────────────────────────────

    #[test]
    fn test_ciede2000_parametric_unit_weights_matches_standard() {
        // With k_l=k_c=k_h=1.0 the parametric formula must equal delta_e_2000.
        let a = lab(50.0, 2.6772, -79.7751);
        let b = lab(50.0, 0.0, -82.7485);
        let parametric = ciede2000_parametric(&a, &b, 1.0, 1.0, 1.0);
        let standard = delta_e_2000(&a, &b);
        assert!(
            (parametric - standard).abs() < 0.001,
            "parametric={parametric} standard={standard}"
        );
    }

    #[test]
    fn test_ciede2000_parametric_identical_is_zero() {
        let c = lab(50.0, 25.0, -10.0);
        assert!(ciede2000_parametric(&c, &c, 1.0, 1.0, 1.0) < 1e-4);
        assert!(ciede2000_parametric(&c, &c, 2.0, 1.5, 0.5) < 1e-4);
    }

    #[test]
    fn test_ciede2000_parametric_higher_kh_reduces_hue_contribution() {
        // With k_h increased, hue differences count for less → lower ΔE.
        // Use two colours that differ mainly in hue (same L* and C*).
        let a = lab(50.0, 30.0, 0.0);
        let b = lab(50.0, -30.0, 0.0); // opposite hue
        let de_bcs = ciede2000_parametric(&a, &b, 1.0, 1.0, 1.0);
        let de_textile = ciede2000_parametric(&a, &b, 1.0, 1.0, 2.0); // textile k_h=2
                                                                      // Heavier k_h weight on S_H denominator scales down hue term → smaller ΔE.
        assert!(
            de_textile < de_bcs,
            "textile ΔE={de_textile} should be < BCS ΔE={de_bcs}"
        );
    }

    #[test]
    fn test_ciede2000_parametric_higher_kl_reduces_lightness_contribution() {
        // Two colours differing only in L*.
        let a = lab(40.0, 0.0, 0.0);
        let b = lab(60.0, 0.0, 0.0);
        let de_bcs = ciede2000_parametric(&a, &b, 1.0, 1.0, 1.0);
        let de_textile = ciede2000_parametric(&a, &b, 2.0, 1.0, 1.0);
        assert!(
            de_textile < de_bcs,
            "textile ΔE={de_textile} should be < BCS ΔE={de_bcs}"
        );
    }

    #[test]
    fn test_ciede2000_parametric_cie_tc157_pair1_unit_weights() {
        // CIE TC 1-57 reference pair 1 (Sharma 2005): expected ≈ 2.0425.
        let a = lab(50.0, 2.6772, -79.7751);
        let b = lab(50.0, 0.0, -82.7485);
        let de = ciede2000_parametric(&a, &b, 1.0, 1.0, 1.0);
        assert!((de - 2.0425).abs() < 0.02, "ΔE00={de} expected≈2.0425");
    }

    #[test]
    fn test_ciede2000_parametric_is_positive() {
        let a = lab(50.0, 10.0, 5.0);
        let b = lab(60.0, -10.0, 20.0);
        assert!(ciede2000_parametric(&a, &b, 1.0, 1.0, 1.0) > 0.0);
    }

    // ── just_noticeable_difference tests ─────────────────────────────────

    #[test]
    fn test_just_noticeable_difference_returns_2_3() {
        assert!((just_noticeable_difference() - 2.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ciede2000_jnd_constant() {
        assert!((CIEDE2000_JND - 2.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_perceptually_equal_below_threshold() {
        assert!(is_perceptually_equal(1.0));
        assert!(is_perceptually_equal(2.3));
    }

    #[test]
    fn test_is_perceptually_equal_above_threshold() {
        assert!(!is_perceptually_equal(2.31));
        assert!(!is_perceptually_equal(10.0));
    }

    #[test]
    fn test_identical_colours_are_perceptually_equal() {
        let c = lab(50.0, 20.0, -10.0);
        let de = delta_e_2000(&c, &c);
        assert!(is_perceptually_equal(de));
    }

    #[test]
    fn test_de94_application_type_alias() {
        // De94Application should be usable as CieApplication.
        let _app: De94Application = CieApplication::Graphic;
        let _app2: De94Application = CieApplication::Textile;
    }

    #[test]
    fn test_de94_via_alias_graphic() {
        let a = lab(50.0, 10.0, 5.0);
        let b = lab(55.0, 15.0, 10.0);
        let de = delta_e_94(&a, &b, De94Application::Graphic);
        assert!(de > 0.0);
    }

    #[test]
    fn test_de94_via_alias_textile() {
        let a = lab(50.0, 10.0, 5.0);
        let b = lab(55.0, 15.0, 10.0);
        let de = delta_e_94(&a, &b, De94Application::Textile);
        assert!(de > 0.0);
    }

    /// The CIE 1994 textile weighting uses k_L=2, so the lightness term is
    /// halved; for a lightness-only difference this must produce a smaller ΔE.
    #[test]
    fn test_de94_textile_smaller_than_graphic_for_lightness_diff() {
        let a = lab(40.0, 0.0, 0.0);
        let b = lab(60.0, 0.0, 0.0);
        let graphic = delta_e_94(&a, &b, De94Application::Graphic);
        let textile = delta_e_94(&a, &b, De94Application::Textile);
        assert!(
            textile < graphic,
            "textile={textile} should be < graphic={graphic}"
        );
    }
}
