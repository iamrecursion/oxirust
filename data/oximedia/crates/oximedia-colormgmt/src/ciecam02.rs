#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
//! Full CIECAM02 colour appearance model.
//!
//! Implements the complete CIECAM02 model as specified in:
//! CIE 159:2004 — A Colour Appearance Model for Colour Management Systems.
//!
//! Provides forward (XYZ → appearance correlates) and inverse
//! (appearance correlates → XYZ) transforms.
//!
//! All XYZ values use the Y=100 scale (D65 white point has Y=100).

use std::f32::consts::PI;

// ── D65 white point (CIE standard, Y=100 scale) ───────────────────────────
const D65_X: f32 = 95.047;
const D65_Y: f32 = 100.0;
const D65_Z: f32 = 108.883;

// ── M_CAT02: chromatic adaptation transform (CIE 159:2004 Table 1) ────────
// Maps XYZ → "sharp" RGB
#[rustfmt::skip]
const M_CAT02: [[f32; 3]; 3] = [
    [ 0.7328,  0.4296, -0.1624],
    [-0.7036,  1.6975,  0.0061],
    [ 0.0030,  0.0136,  0.9834],
];

// ── Inverse of M_CAT02 (CIE 159:2004) ────────────────────────────────────
#[rustfmt::skip]
const M_CAT02_INV: [[f32; 3]; 3] = [
    [ 1.096_124, -0.278_869, 0.182_745],
    [ 0.454_369,  0.473_533, 0.072_098],
    [-0.009_628, -0.005_698, 1.015_326],
];

// ── M_HPE: Hunt-Pointer-Estevez (D65-adapted) (CIE 159:2004 Table 1) ─────
#[rustfmt::skip]
const M_HPE: [[f32; 3]; 3] = [
    [ 0.38971,  0.68898, -0.07868],
    [-0.22981,  1.18340,  0.04641],
    [ 0.00000,  0.00000,  1.00000],
];

// ── Inverse of M_HPE (pre-computed) ──────────────────────────────────────
#[rustfmt::skip]
const M_HPE_INV: [[f32; 3]; 3] = [
    [ 1.910_197, -1.112_124,  0.201_908],
    [ 0.370_950,  0.629_054, -0.000_005],
    [ 0.000_000,  0.000_000,  1.000_000],
];

// ── Hue quadrature table (CIE 159:2004 Table 2) ───────────────────────────
// Columns: (index, h_i, e_i, H_i)
const HUE_DATA: [(f32, f32, f32, f32); 5] = [
    (0.0, 20.14, 0.8, 0.0),
    (1.0, 90.00, 0.7, 100.0),
    (2.0, 164.25, 1.0, 200.0),
    (3.0, 237.53, 1.2, 300.0),
    (4.0, 380.14, 0.8, 400.0),
];

// ── Matrix helpers ────────────────────────────────────────────────────────

#[inline]
fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ── Surround condition ─────────────────────────────────────────────────────

/// Surround condition for CIECAM02 viewing environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundCondition {
    /// Average surround (typical indoor monitor viewing).
    Average,
    /// Dim surround (television viewing in a dim room).
    Dim,
    /// Dark surround (cinema or projector in a dark room).
    Dark,
}

impl SurroundCondition {
    /// Returns the surround coefficients (F, c, N_c) per CIE 159 Table 3.
    fn coefficients(self) -> (f32, f32, f32) {
        match self {
            Self::Average => (1.0, 0.69, 1.0),
            Self::Dim => (0.9, 0.59, 0.9),
            Self::Dark => (0.8, 0.525, 0.8),
        }
    }
}

// ── Viewing conditions ─────────────────────────────────────────────────────

/// Viewing conditions for the CIECAM02 model.
#[derive(Debug, Clone)]
pub struct CiecamViewingConditions {
    /// Luminance of the adapting field (cd/m²). Typical indoor value: 64.0.
    pub adapting_luminance_la: f32,
    /// Relative luminance of the background (0..100). Typical: 20.0.
    pub background_relative_lum_yb: f32,
    /// Surround condition.
    pub surround: SurroundCondition,
}

impl CiecamViewingConditions {
    /// Standard D65 indoor viewing conditions (La=64, Yb=20, Average surround).
    #[must_use]
    pub fn standard_d65() -> Self {
        Self {
            adapting_luminance_la: 64.0,
            background_relative_lum_yb: 20.0,
            surround: SurroundCondition::Average,
        }
    }
}

// ── Appearance correlates ─────────────────────────────────────────────────

/// CIECAM02 colour appearance correlates for a single colour sample.
#[derive(Debug, Clone)]
pub struct CiecamAppearance {
    /// Lightness J (0..100).
    pub lightness: f32,
    /// Brightness Q (≥ 0).
    pub brightness: f32,
    /// Chroma C (≥ 0).
    pub chroma: f32,
    /// Colorfulness M (≥ 0).
    pub colorfulness: f32,
    /// Saturation s (0..100).
    pub saturation: f32,
    /// Hue angle h (0..360°).
    pub hue_angle: f32,
    /// Hue quadrature H (0..400).
    pub hue_composition: f32,
}

// ── CIECAM02 model ────────────────────────────────────────────────────────

/// CIECAM02 colour appearance model with pre-computed adaptation factors.
///
/// Create once per viewing condition set, then call `xyz_to_appearance` and
/// `appearance_to_xyz` for individual colour samples.
#[derive(Debug, Clone)]
pub struct CiecamModel {
    // Surround parameters
    #[allow(dead_code)]
    f_big: f32, // F: maximum degree of adaptation
    c: f32,   // c: surround factor
    n_c: f32, // Nc: chromatic induction factor
    // Model parameters
    f_l: f32,  // F_L: luminance-level adaptation factor
    n: f32,    // n: background ratio
    n_bb: f32, // N_bb: background brightness induction factor
    n_cb: f32, // N_cb: chromatic induction factor (same as N_bb in CIECAM02)
    z: f32,    // z: lightness contrast exponent
    // White-point adaptation
    d: f32, // Degree of chromatic adaptation (0..1)
    /// Achromatic response of the adapted white (A_w).
    pub a_w: f32,
    // CAT02 signals of the white point (for forward/inverse adapt)
    rgb_w: [f32; 3],
    // Adapted+compressed white signals (for verification)
    #[allow(dead_code)]
    rgb_aw: [f32; 3],
}

impl CiecamModel {
    /// Build a CIECAM02 model from the given viewing conditions.
    ///
    /// Pre-computes all viewing-condition-dependent constants so that per-colour
    /// conversions are cheap.
    #[must_use]
    pub fn new(conditions: CiecamViewingConditions) -> Self {
        let la = conditions.adapting_luminance_la.max(0.001_f32);
        let yb = conditions.background_relative_lum_yb.clamp(1.0, 100.0);
        let (f_big, c, n_c) = conditions.surround.coefficients();

        // ── Step 1: CAT02 of D65 white ─────────────────────────────────
        let rgb_w = mat3_mul_vec3(&M_CAT02, [D65_X, D65_Y, D65_Z]);

        // ── Step 2: Degree of chromatic adaptation D ───────────────────
        // CIE 159 Eq. 4:  D = F * (1 - (1/3.6)*exp((-La-42)/92))
        let d = (f_big * (1.0 - (1.0 / 3.6) * ((-la - 42.0) / 92.0).exp())).clamp(0.0, 1.0);

        // ── Step 3: F_L (luminance-level adaptation, CIE 159 Eq. 5) ───
        let k = 1.0 / (5.0 * la + 1.0);
        let f_l = 0.2 * k.powi(4) * (5.0 * la)
            + 0.1 * (1.0 - k.powi(4)).powi(2) * (5.0 * la).powf(1.0 / 3.0);

        // ── Step 4: n, N_bb, N_cb, z ──────────────────────────────────
        let n = yb / D65_Y;
        let n_bb = 0.725 * n.powf(-0.2);
        let n_cb = n_bb; // same in CIECAM02
        let z = 1.48 + 0.29 * n.sqrt();

        // ── Step 5: Adapted white signals in CAT02 space ──────────────
        // Cone signals of adapted white: Rc = [D*Yw/Rw + 1 - D]*Rw
        // where Yw=100 (D65 luminance)
        let rgb_wa = [
            (d * D65_Y / rgb_w[0] + 1.0 - d) * rgb_w[0],
            (d * D65_Y / rgb_w[1] + 1.0 - d) * rgb_w[1],
            (d * D65_Y / rgb_w[2] + 1.0 - d) * rgb_w[2],
        ];

        // ── Step 6: Convert adapted white to HPE (M_HPE * M_CAT02^-1) ─
        let xyz_wa = mat3_mul_vec3(&M_CAT02_INV, rgb_wa);
        let hpe_wa = mat3_mul_vec3(&M_HPE, xyz_wa);

        // ── Step 7: Naka-Rushton of white HPE signals ─────────────────
        let ra_w = naka_rushton(hpe_wa[0], f_l);
        let ga_w = naka_rushton(hpe_wa[1], f_l);
        let ba_w = naka_rushton(hpe_wa[2], f_l);

        // ── Step 8: Achromatic response A_w (CIE 159 Eq. 16) ──────────
        let a_w = (2.0 * ra_w + ga_w + (1.0 / 20.0) * ba_w - 0.305) * n_bb;

        Self {
            f_big,
            c,
            n_c,
            f_l,
            n,
            n_bb,
            n_cb,
            z,
            d,
            a_w,
            rgb_w,
            rgb_aw: [ra_w, ga_w, ba_w],
        }
    }

    /// Convert XYZ (D65, Y=100 scale) to CIECAM02 appearance correlates.
    #[must_use]
    pub fn xyz_to_appearance(&self, x: f32, y: f32, z: f32) -> CiecamAppearance {
        // ── Step 1: CAT02 of sample ────────────────────────────────────
        let rgb = mat3_mul_vec3(&M_CAT02, [x, y, z]);

        // ── Step 2: Chromatic adaptation ──────────────────────────────
        let rgb_a = [
            (self.d * D65_Y / self.rgb_w[0] + 1.0 - self.d) * rgb[0],
            (self.d * D65_Y / self.rgb_w[1] + 1.0 - self.d) * rgb[1],
            (self.d * D65_Y / self.rgb_w[2] + 1.0 - self.d) * rgb[2],
        ];

        // ── Step 3: Convert to HPE (M_HPE * M_CAT02^-1 * rgb_a) ──────
        let xyz_a = mat3_mul_vec3(&M_CAT02_INV, rgb_a);
        let hpe = mat3_mul_vec3(&M_HPE, xyz_a);

        // ── Step 4: Naka-Rushton nonlinearity ─────────────────────────
        let ra = naka_rushton_signed(hpe[0], self.f_l);
        let ga = naka_rushton_signed(hpe[1], self.f_l);
        let ba = naka_rushton_signed(hpe[2], self.f_l);

        // ── Step 5: Opponent colour dimensions (CIE 159 Eq. 13-14) ───
        let a_opp = ra - 12.0 * ga / 11.0 + ba / 11.0;
        let b_opp = (ra + ga - 2.0 * ba) / 9.0;

        // ── Step 6: Hue angle h ────────────────────────────────────────
        let mut h = b_opp.atan2(a_opp).to_degrees();
        if h < 0.0 {
            h += 360.0;
        }

        // ── Step 7: Hue quadrature H ───────────────────────────────────
        let hue_comp = hue_quadrature(h);

        // ── Step 8: Achromatic response A (CIE 159 Eq. 16) ───────────
        let a_ach = (2.0 * ra + ga + (1.0 / 20.0) * ba - 0.305) * self.n_bb;

        // ── Step 9: Lightness J (CIE 159 Eq. 17) ─────────────────────
        let j = 100.0 * (a_ach / self.a_w).max(0.0).powf(self.c * self.z);

        // ── Step 10: Brightness Q (CIE 159 Eq. 18) ───────────────────
        let q = (4.0 / self.c) * (j / 100.0).sqrt() * (self.a_w + 4.0) * self.f_l.powf(0.25);

        // ── Step 11: Chroma C (CIE 159 Eq. 19-21) ────────────────────
        let e_t = 0.25 * ((h * PI / 180.0).cos() + 3.8);
        let ab_mag = (a_opp * a_opp + b_opp * b_opp).sqrt();
        let t_num = 50_000.0 / 13.0 * self.n_c * self.n_cb * e_t * ab_mag;
        let t_den = ra + ga + (21.0 / 20.0) * ba;
        let t = if t_den.abs() < 1e-8 {
            0.0
        } else {
            t_num / t_den
        };
        let c_chroma =
            t.max(0.0).powf(0.9) * (j / 100.0).sqrt() * (1.64 - 0.29_f32.powf(self.n)).powi(2);

        // ── Step 12: Colorfulness M (CIE 159 Eq. 22) ─────────────────
        let m_color = c_chroma * self.f_l.powf(0.25);

        // ── Step 13: Saturation s (CIE 159 Eq. 23) ───────────────────
        let s_sat = if q.abs() < 1e-8 {
            0.0
        } else {
            100.0 * (m_color / q).sqrt()
        };

        CiecamAppearance {
            lightness: j.max(0.0),
            brightness: q.max(0.0),
            chroma: c_chroma.max(0.0),
            colorfulness: m_color.max(0.0),
            saturation: s_sat.clamp(0.0, 100.0),
            hue_angle: h,
            hue_composition: hue_comp,
        }
    }

    /// Inverse CIECAM02: convert appearance correlates back to XYZ.
    ///
    /// Uses J (lightness), C (chroma), and h (hue angle) from `appearance`.
    #[must_use]
    pub fn appearance_to_xyz(&self, appearance: &CiecamAppearance) -> (f32, f32, f32) {
        let j = appearance.lightness.max(0.0);
        let c_chroma = appearance.chroma.max(0.0);
        let h = appearance.hue_angle;
        let h_rad = h * PI / 180.0;

        // ── Step 1: Recover t from C ───────────────────────────────────
        // C = t^0.9 * (J/100)^0.5 * (1.64 - 0.29^n)^2
        let scale = (1.64 - 0.29_f32.powf(self.n)).powi(2) * (j / 100.0).sqrt();
        let t = if scale.abs() < 1e-8 || c_chroma < 1e-8 {
            0.0
        } else {
            (c_chroma / scale).powf(1.0 / 0.9)
        };

        // ── Step 2: Recover achromatic response A from J ──────────────
        // J = 100 * (A / A_w)^(c*z)  →  A = A_w * (J/100)^(1/(c*z))
        let a_ach = self.a_w * (j / 100.0).powf(1.0 / (self.c * self.z));

        // ── Step 3: p2 (CIE 159 Eq. 28) ──────────────────────────────
        let p2 = a_ach / self.n_bb + 0.305;

        // ── Step 4: Recover opponent signals a, b ─────────────────────
        // e_t from hue angle (CIE 159 Eq. 21)
        let e_t = 0.25 * (h_rad.cos() + 3.8);

        // From the model equations (CIE 159, working backwards from t):
        // t = (50000/13 * Nc * Ncb * et * sqrt(a²+b²)) / (R+G+21/20*B)
        // p2 = (2R + G + B/20) / n_bb - 0.305 + 0.305 = (R+G+..) /n_bb
        // Let p1 = 50000/13 * Nc * Ncb * et  (numerator factor)
        // Let p3 = 21/20
        // R+G+p3*B = t*sqrt(a²+b²)/p1 * ... => a, b via CIE 159 Eq 29
        // Full inverse (CIE 159 Annex A):
        let (a_opp, b_opp) = if t.abs() < 1e-8 {
            (0.0_f32, 0.0_f32)
        } else {
            let p1 = 50_000.0 / 13.0 * self.n_c * self.n_cb * e_t;
            let p3 = 21.0_f32 / 20.0;
            // From CIE 159 Eq. 29 (with h given):
            // a = sin(h) · t·p2 / (p1 + t*(12/11·sin(h) - (1/9)·sin(h)*...))
            // Use exact system solution:
            // Let sin_h = sin(h), cos_h = cos(h)
            // a·(p1/t + 12/11·cos_h/sin_h) = ...  -- not quite right
            // Use direct approach per Fairchild (2013):
            let sin_h = h_rad.sin();
            let cos_h = h_rad.cos();
            // Solve: t = p1·sqrt(a²+b²) / (R+G+p3·B)
            //        b = a·tan(h) = a·sin_h/cos_h
            // Also from HPE equations:
            //   2R + G + B/20 = (p2+0.305)*n_bb · n_bb⁻¹ = p2 (rewritten)
            //   p2 = (460R + 451G + ... ) / 1403 (per Fairchild Table 9.4)
            //   R  = (460p2 + 451a + 288b) / 1403
            //   G  = (460p2 - 891a - 261b) / 1403
            //   B  = (460p2 - 220a - 6300b) / 1403
            //   => R+G+p3·B = [460(2+p3)p2 + (451-891-p3·220)a + (288-261-p3·6300)b] / 1403
            //   denominator coefficient of a: (451-891-220·21/20) = (451-891-231) = -671
            //   denominator coefficient of b: (288-261-6300·21/20) = (288-261-6615) = -6588
            // So: t·(-671a - 6588b)/1403 = p1·sqrt(a²+b²)
            // With b = a·tan(h):
            //   t·a·(-671 - 6588·tan(h))/1403 = p1·|a|·sqrt(1+tan²(h))
            // For cos_h ≠ 0:
            //   Let a_sq_norm: |a|/sqrt(1+tan²h) = |a|·|cos_h|
            //   => t·|cos_h|·(-671·cos_h - 6588·sin_h)/1403 = p1·sign(a)·|cos_h|
            //   => a sign: if -671·cos_h - 6588·sin_h < 0 → a > 0 (t>0)
            // More robustly: use the two-variable system
            //   p1·sqrt(a²+b²) = t·(Ra + Ga + p3·Ba) where Ra etc from HPE inversion
            //   denom = [460(1+1+p3/20) - ... depends on a/b direction
            //
            // Simpler approximation valid for moderate t:
            let denom_a = 460.0 * (2.0 + p3) / 1403.0;
            // Actually solve the system directly using the standard 2-eq approach:
            // From CIE 159 (Lam & Rigg derivation):
            //   denom = R_a + G_a + p3*B_a
            //         = (460*p2/1403) + a*(451-891-p3*220)/1403 + b*(288-261-p3*6300)/1403
            //   Note 460*(2 + 21/20) = 460*41/20 = 943 (not used below)
            let ka = (451.0 - 891.0 - p3 * 220.0) / 1403.0; // coefficient of a in denom
            let kb = (288.0 - 261.0 - p3 * 6300.0) / 1403.0; // coefficient of b in denom
            let c0 = 460.0 * (1.0 + 1.0 + p3 / 20.0) / 1403.0 * p2; // constant part
                                                                    // Actually the correct expansion:
                                                                    // R_a = (460*p2 + 451*a + 288*b) / 1403
                                                                    // G_a = (460*p2 - 891*a - 261*b) / 1403
                                                                    // B_a = (460*p2 - 220*a - 6300*b) / 1403
                                                                    // R_a+G_a+p3*B_a = (460p2*(1+1+p3) + a*(451-891-p3*220) + b*(288-261-p3*6300)) / 1403
            let _ = (ka, kb, c0, denom_a); // suppress unused warnings

            let r_a_const = 460.0 * p2;
            let g_a_const = 460.0 * p2;
            let b_a_const = 460.0 * p2;
            // We need to find a,b such that b = a*tan(h) and:
            // t * (r_a + g_a + p3*b_a) = p1 * sqrt(a²+b²)
            // where r_a = (r_a_const + 451a + 288b)/1403  etc.
            // Substitute b = a*sin_h/cos_h (for cos_h != 0) or a = b*cos_h/sin_h
            // Let u = |a|, sign of a determined by sign of denominator term
            let (a_v, b_v) = if cos_h.abs() > sin_h.abs() {
                // Express b in terms of a: b = a * sin_h/cos_h
                let tan_h = sin_h / cos_h;
                // denom = [(460*(2+p3)*p2) + a*(ka_full) + a*tan_h*(kb_full)] / 1403
                let ka_full = 451.0 - 891.0 - p3 * 220.0;
                let kb_full = 288.0 - 261.0 - p3 * 6300.0;
                let c0_full = (r_a_const + g_a_const + p3 * b_a_const) / 1403.0; // p2 part
                                                                                 // Note: r_a_const = g_a_const = b_a_const = 460*p2 so:
                                                                                 // c0_full = 460*p2*(1+1+p3)/1403
                let _ = c0_full;
                let const_denom = 460.0 * p2 * (2.0 + p3) / 1403.0;
                let a_coeff = (ka_full + kb_full * tan_h) / 1403.0;
                // t * (const_denom + a*a_coeff) = p1 * |a| * sqrt(1 + tan_h^2)
                let rhs_coeff = p1 * (1.0 + tan_h * tan_h).sqrt();
                // t * const_denom = a * (rhs_coeff - t*a_coeff)
                let a_denom_coeff = rhs_coeff - t * a_coeff;
                let a_v = if a_denom_coeff.abs() < 1e-8 {
                    0.0
                } else {
                    t * const_denom / a_denom_coeff
                };
                (a_v, a_v * tan_h)
            } else {
                // Express a in terms of b: a = b * cos_h/sin_h
                let cot_h = cos_h / sin_h;
                let ka_full = 451.0 - 891.0 - p3 * 220.0;
                let kb_full = 288.0 - 261.0 - p3 * 6300.0;
                let const_denom = 460.0 * p2 * (2.0 + p3) / 1403.0;
                let b_coeff = (kb_full + ka_full * cot_h) / 1403.0;
                let rhs_coeff = p1 * (1.0 + cot_h * cot_h).sqrt();
                let b_denom_coeff = rhs_coeff - t * b_coeff;
                let b_v = if b_denom_coeff.abs() < 1e-8 {
                    0.0
                } else {
                    t * const_denom / b_denom_coeff
                };
                (b_v * cot_h, b_v)
            };
            (a_v, b_v)
        };

        // ── Step 5: Recover post-adapt HPE signals ─────────────────────
        // From opponent signals (CIE 159 Eq. 29–31):
        let ra = (460.0 * p2 + 451.0 * a_opp + 288.0 * b_opp) / 1403.0;
        let ga = (460.0 * p2 - 891.0 * a_opp - 261.0 * b_opp) / 1403.0;
        let ba = (460.0 * p2 - 220.0 * a_opp - 6300.0 * b_opp) / 1403.0;

        // ── Step 6: Invert Naka-Rushton ────────────────────────────────
        let r_hpe = naka_rushton_inv_signed(ra, self.f_l);
        let g_hpe = naka_rushton_inv_signed(ga, self.f_l);
        let b_hpe = naka_rushton_inv_signed(ba, self.f_l);

        // ── Step 7: HPE → XYZ (M_HPE^-1) ─────────────────────────────
        let xyz_from_hpe = mat3_mul_vec3(&M_HPE_INV, [r_hpe, g_hpe, b_hpe]);

        // ── Step 8: XYZ → CAT02 adapted ──────────────────────────────
        let rgb_a_back = mat3_mul_vec3(&M_CAT02, xyz_from_hpe);

        // ── Step 9: Undo chromatic adaptation ─────────────────────────
        let adapt_factor = [
            self.d * D65_Y / self.rgb_w[0] + 1.0 - self.d,
            self.d * D65_Y / self.rgb_w[1] + 1.0 - self.d,
            self.d * D65_Y / self.rgb_w[2] + 1.0 - self.d,
        ];
        let rgb_cat02 = [
            rgb_a_back[0] / adapt_factor[0].max(1e-10),
            rgb_a_back[1] / adapt_factor[1].max(1e-10),
            rgb_a_back[2] / adapt_factor[2].max(1e-10),
        ];

        // ── Step 10: CAT02^-1 → XYZ ───────────────────────────────────
        let xyz = mat3_mul_vec3(&M_CAT02_INV, rgb_cat02);
        (xyz[0], xyz[1], xyz[2])
    }
}

// ── Naka-Rushton nonlinearity ─────────────────────────────────────────────

/// CIECAM02 Naka-Rushton compression (positive input only).
/// CIE 159 Eq. 12: R_a = 400*(F_L*R/100)^0.42 / (27.13 + (F_L*R/100)^0.42) + 0.1
fn naka_rushton(x: f32, f_l: f32) -> f32 {
    let xp = (f_l * x.max(0.0) / 100.0).powf(0.42);
    400.0 * xp / (27.13 + xp) + 0.1
}

/// Signed Naka-Rushton (handles negative x via sign extension).
fn naka_rushton_signed(x: f32, f_l: f32) -> f32 {
    if x >= 0.0 {
        naka_rushton(x, f_l)
    } else {
        -naka_rushton(-x, f_l)
    }
}

/// Inverse of Naka-Rushton (for positive y values).
fn naka_rushton_inv(y: f32, f_l: f32) -> f32 {
    let y_shift = y - 0.1;
    // Guard: 400 - y_shift must be > 0 to avoid division by zero
    let denom = 400.0 - y_shift;
    if denom.abs() < 1e-8 || y_shift < -27.13 {
        return 0.0;
    }
    let ratio = (27.13 * y_shift / denom).max(0.0);
    // xp = ratio^(1/0.42); then x = xp * 100/F_L
    ratio.powf(1.0 / 0.42) * 100.0 / f_l.max(1e-10)
}

/// Signed inverse Naka-Rushton.
fn naka_rushton_inv_signed(y: f32, f_l: f32) -> f32 {
    if y >= 0.0 {
        naka_rushton_inv(y, f_l)
    } else {
        -naka_rushton_inv(-y, f_l)
    }
}

// ── Hue quadrature ────────────────────────────────────────────────────────

/// Compute hue quadrature H ∈ [0, 400] from hue angle h ∈ [0, 360).
fn hue_quadrature(h: f32) -> f32 {
    let h = h.rem_euclid(360.0);
    // Shift h if it falls below the first hue-data entry
    let h_p = if h < HUE_DATA[0].1 { h + 360.0 } else { h };

    // Find the bounding interval [i, i+1]
    let mut idx = 0usize;
    for i in 0..4 {
        if h_p >= HUE_DATA[i].1 && h_p < HUE_DATA[i + 1].1 {
            idx = i;
            break;
        }
    }

    let (_, h_i, e_i, big_h_i) = HUE_DATA[idx];
    let (_, h_i1, e_i1, big_h_i1) = HUE_DATA[idx + 1];

    let t1 = (h_p - h_i) / e_i;
    let t2 = (h_i1 - h_p) / e_i1;
    let denom = t1 + t2;
    if denom.abs() < 1e-10 {
        return big_h_i;
    }
    // CIE 159 Eq. 20: H = H_i + 100 * t1/(t1+t2)
    // Linear interpolation between H_i and H_{i+1} scaled by t1/(t1+t2)
    big_h_i + (big_h_i1 - big_h_i) * t1 / denom
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> CiecamModel {
        CiecamModel::new(CiecamViewingConditions::standard_d65())
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn test_model_construction_does_not_panic() {
        let _m = default_model();
    }

    #[test]
    fn test_model_aw_positive() {
        let m = default_model();
        assert!(m.a_w > 0.0, "A_w should be positive, got {}", m.a_w);
    }

    #[test]
    fn test_model_fl_positive() {
        let m = default_model();
        assert!(m.f_l > 0.0, "F_L should be positive, got {}", m.f_l);
    }

    // ── Forward transform ─────────────────────────────────────────────────

    #[test]
    fn test_d65_white_lightness_near_100() {
        let m = default_model();
        let app = m.xyz_to_appearance(D65_X, D65_Y, D65_Z);
        assert!(
            (app.lightness - 100.0).abs() < 2.0,
            "D65 white J={} not near 100",
            app.lightness
        );
    }

    #[test]
    fn test_black_lightness_near_zero() {
        let m = default_model();
        let app = m.xyz_to_appearance(0.0, 0.0, 0.0);
        assert!(
            app.lightness < 5.0,
            "Black J={} should be near zero",
            app.lightness
        );
    }

    #[test]
    fn test_white_chroma_near_zero() {
        let m = default_model();
        let app = m.xyz_to_appearance(D65_X, D65_Y, D65_Z);
        // D65 white is achromatic
        assert!(
            app.chroma < 2.0,
            "White chroma C={} should be near zero",
            app.chroma
        );
    }

    #[test]
    fn test_hue_angle_range() {
        let m = default_model();
        let app = m.xyz_to_appearance(40.0, 20.0, 5.0);
        assert!(
            app.hue_angle >= 0.0 && app.hue_angle < 360.0,
            "Hue angle {} out of range",
            app.hue_angle
        );
    }

    #[test]
    fn test_hue_quadrature_range() {
        let m = default_model();
        let app = m.xyz_to_appearance(40.0, 20.0, 5.0);
        assert!(
            app.hue_composition >= 0.0 && app.hue_composition <= 400.0,
            "H={} out of [0,400]",
            app.hue_composition
        );
    }

    #[test]
    fn test_brightness_positive() {
        let m = default_model();
        let app = m.xyz_to_appearance(D65_X, D65_Y, D65_Z);
        assert!(app.brightness > 0.0);
    }

    #[test]
    fn test_saturation_range() {
        let m = default_model();
        let app = m.xyz_to_appearance(40.0, 20.0, 5.0);
        assert!(
            app.saturation >= 0.0 && app.saturation <= 100.0,
            "s={} out of range",
            app.saturation
        );
    }

    // ── Inverse transform ─────────────────────────────────────────────────

    #[test]
    fn test_inverse_white_roundtrip() {
        let m = default_model();
        let app = m.xyz_to_appearance(D65_X, D65_Y, D65_Z);
        let (x2, y2, z2) = m.appearance_to_xyz(&app);
        assert!((x2 - D65_X).abs() < 3.0, "X roundtrip: {} vs {}", x2, D65_X);
        assert!((y2 - D65_Y).abs() < 3.0, "Y roundtrip: {} vs {}", y2, D65_Y);
        assert!((z2 - D65_Z).abs() < 3.0, "Z roundtrip: {} vs {}", z2, D65_Z);
    }

    #[test]
    fn test_inverse_midgray_roundtrip() {
        let m = default_model();
        let (x0, y0, z0) = (47.5, 50.0, 54.4);
        let app = m.xyz_to_appearance(x0, y0, z0);
        let (x2, y2, z2) = m.appearance_to_xyz(&app);
        assert!((x2 - x0).abs() < 5.0, "X: {} vs {}", x2, x0);
        assert!((y2 - y0).abs() < 5.0, "Y: {} vs {}", y2, y0);
        assert!((z2 - z0).abs() < 5.0, "Z: {} vs {}", z2, z0);
    }

    // ── Dim / Dark surround conditions ────────────────────────────────────

    #[test]
    fn test_dim_surround_construction() {
        let cond = CiecamViewingConditions {
            adapting_luminance_la: 32.0,
            background_relative_lum_yb: 16.0,
            surround: SurroundCondition::Dim,
        };
        let m = CiecamModel::new(cond);
        assert!(m.a_w > 0.0);
    }

    #[test]
    fn test_dark_surround_construction() {
        let cond = CiecamViewingConditions {
            adapting_luminance_la: 4.0,
            background_relative_lum_yb: 10.0,
            surround: SurroundCondition::Dark,
        };
        let m = CiecamModel::new(cond);
        assert!(m.a_w > 0.0);
    }

    // ── Naka-Rushton monotonicity ─────────────────────────────────────────

    #[test]
    fn test_naka_rushton_monotone() {
        let f_l = 1.0;
        let v1 = naka_rushton(10.0, f_l);
        let v2 = naka_rushton(50.0, f_l);
        let v3 = naka_rushton(100.0, f_l);
        assert!(v1 < v2 && v2 < v3);
    }

    #[test]
    fn test_naka_rushton_inv_roundtrip() {
        let f_l = 1.5;
        let x = 60.0_f32;
        let y = naka_rushton(x, f_l);
        let x2 = naka_rushton_inv(y, f_l);
        assert!((x - x2).abs() < 0.5, "roundtrip: {} -> {} -> {}", x, y, x2);
    }

    // ── Hue quadrature ────────────────────────────────────────────────────

    #[test]
    fn test_hue_quadrature_at_unique_hue() {
        // h=90 should give H=100 exactly (yellow hue)
        let h = hue_quadrature(90.0);
        assert!((h - 100.0).abs() < 1.0, "H at h=90 should be ~100, got {h}");
    }
}
