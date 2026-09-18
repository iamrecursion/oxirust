#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
//! Cusp-based perceptual gamut mapping using CIECAM02.
//!
//! A "cusp" is the point of maximum chroma at a given hue angle in a target
//! gamut boundary. This module pre-computes a dense cusp table for standard
//! broadcast gamuts and uses it to compress out-of-gamut colours toward the
//! cusp of the target gamut, preserving hue while minimising perceptual
//! distortion.
//!
//! ## Algorithm
//!
//! 1. For each hue angle sample the target gamut boundary in CIECAM02 (J, C)
//!    space and locate the cusp — the (J_cusp, C_cusp) with maximum C.
//! 2. Given an input XYZ colour, convert to CIECAM02 (J, C, h).
//! 3. Look up the cusp for that hue (interpolated).
//! 4. If the colour is within the gamut, return it unchanged.
//! 5. Otherwise compress (J, C) along the line toward (J_cusp, C_cusp) until
//!    the result lands on the gamut boundary.
//!
//! ## References
//!
//! - "Gamut Mapping with Perceived Colour Differences", Morovic & Luo (2001)
//! - CIE 159:2004 — CIECAM02 colour appearance model

use crate::ciecam02::{CiecamModel, CiecamViewingConditions, SurroundCondition};

// ── Gamut primaries (CIE xy, Y=1 illuminant D65) ─────────────────────────

/// Rec.709 / sRGB primary triangle in CIE xy (D65 white).
const REC709_PRIMARIES: [[f64; 2]; 3] = [
    [0.640, 0.330], // R
    [0.300, 0.600], // G
    [0.150, 0.060], // B
];

/// DCI-P3 primary triangle in CIE xy (D65 white; display P3).
const DCIP3_PRIMARIES: [[f64; 2]; 3] = [
    [0.680, 0.320], // R
    [0.265, 0.690], // G
    [0.150, 0.060], // B
];

/// Rec.2020 primary triangle in CIE xy (D65 white).
const REC2020_PRIMARIES: [[f64; 2]; 3] = [
    [0.708, 0.292], // R
    [0.170, 0.797], // G
    [0.131, 0.046], // B
];

/// D65 white point (CIE xy).
const D65_WHITE: [f64; 2] = [0.3127, 0.3290];

// ── Number of hue samples used to build the cusp table ───────────────────

const CUSP_TABLE_SIZE: usize = 360;

// ── Target gamut ─────────────────────────────────────────────────────────

/// Target colour gamut for cusp-based gamut mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetGamut {
    /// ITU-R BT.709 / sRGB (standard HD broadcast).
    Rec709,
    /// DCI-P3 display (digital cinema / Apple wide-colour).
    DciP3,
    /// ITU-R BT.2020 (UHD broadcast / HDR mastering).
    Rec2020,
}

impl TargetGamut {
    /// CIE xy primaries [R, G, B] for this gamut.
    fn primaries(self) -> [[f64; 2]; 3] {
        match self {
            TargetGamut::Rec709 => REC709_PRIMARIES,
            TargetGamut::DciP3 => DCIP3_PRIMARIES,
            TargetGamut::Rec2020 => REC2020_PRIMARIES,
        }
    }
}

// ── GamutCusp ─────────────────────────────────────────────────────────────

/// The cusp point of a colour gamut at a given hue angle.
///
/// The cusp is the (J, C) point with maximum CIECAM02 chroma on the gamut
/// boundary at hue angle `hue_angle`.
#[derive(Debug, Clone, Copy)]
pub struct GamutCusp {
    /// Hue angle in degrees (0–360).
    pub hue_angle: f64,
    /// CIECAM02 chroma C at the cusp.
    pub max_chroma: f64,
    /// CIECAM02 lightness J at the cusp.
    pub lightness: f64,
}

// ── CuspGamutMapper ───────────────────────────────────────────────────────

/// Cusp-directed perceptual gamut mapper.
///
/// Pre-computes a [`GamutCusp`] table at `CUSP_TABLE_SIZE` equidistant hue
/// angles and uses linear interpolation to perform per-pixel gamut compression.
#[derive(Debug, Clone)]
pub struct CuspGamutMapper {
    /// Cusp table sampled at uniformly-spaced hue angles.
    pub cusps: Vec<GamutCusp>,
    /// Target gamut for which the cusp table was built.
    pub target_gamut: TargetGamut,
    /// CIECAM02 viewing conditions used during cusp computation and mapping.
    viewing: CiecamViewingConditions,
}

impl CuspGamutMapper {
    /// Create a new mapper with a pre-computed cusp table for `target`.
    ///
    /// Uses standard D65 indoor viewing conditions (La = 64 cd/m², Yb = 20 %,
    /// average surround) for cusp computation.
    #[must_use]
    pub fn new(target: TargetGamut) -> Self {
        let viewing = CiecamViewingConditions {
            adapting_luminance_la: 64.0,
            background_relative_lum_yb: 20.0,
            surround: SurroundCondition::Average,
        };
        Self::with_viewing_conditions(target, viewing)
    }

    /// Create a mapper using custom viewing conditions.
    #[must_use]
    pub fn with_viewing_conditions(target: TargetGamut, viewing: CiecamViewingConditions) -> Self {
        let cusps = build_cusp_table(target, &viewing);
        Self {
            cusps,
            target_gamut: target,
            viewing,
        }
    }

    /// Return the (interpolated) cusp for an arbitrary hue angle (degrees).
    ///
    /// The hue is wrapped into [0, 360) before lookup.
    #[must_use]
    pub fn cusp_at_hue(&self, hue_deg: f64) -> GamutCusp {
        interpolate_cusp(&self.cusps, hue_deg)
    }

    /// Map an XYZ colour (D65, Y = 100 scale) into the target gamut.
    ///
    /// - Hue is preserved exactly.
    /// - Chroma and lightness are compressed linearly toward the cusp along
    ///   the JC line connecting the input to the cusp, stopping at the gamut
    ///   boundary.
    /// - In-gamut colours are returned unchanged.
    ///
    /// Returns the mapped XYZ (D65, Y = 100 scale).
    #[must_use]
    pub fn map_xyz(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let model = CiecamModel::new(self.viewing.clone());

        // Convert to f32 for CIECAM02 (which uses f32 internally).
        let app = model.xyz_to_appearance(x as f32, y as f32, z as f32);

        let j_in = app.lightness as f64;
        let c_in = app.chroma as f64;
        let h = app.hue_angle as f64;

        // Fetch cusp for this hue.
        let cusp = self.cusp_at_hue(h);

        // Check whether the colour is already inside the gamut.
        if is_inside_gamut(j_in, c_in, &cusp) {
            return (x, y, z);
        }

        // Compress (J, C) toward the cusp.
        let (j_out, c_out) = compress_toward_cusp(j_in, c_in, &cusp);

        // Convert compressed appearance back to XYZ.
        let mut app_out = app.clone();
        app_out.lightness = j_out.max(0.0) as f32;
        app_out.chroma = c_out.max(0.0) as f32;

        let (xo, yo, zo) = model.appearance_to_xyz(&app_out);
        (xo as f64, yo as f64, zo as f64)
    }
}

// ── Cusp table construction ───────────────────────────────────────────────

/// Build a cusp table with `CUSP_TABLE_SIZE` entries for `target`.
fn build_cusp_table(target: TargetGamut, viewing: &CiecamViewingConditions) -> Vec<GamutCusp> {
    let model = CiecamModel::new(viewing.clone());
    let primaries = target.primaries();
    let white = D65_WHITE;

    let step = 360.0_f64 / CUSP_TABLE_SIZE as f64;

    (0..CUSP_TABLE_SIZE)
        .map(|i| {
            let hue_target = i as f64 * step;
            find_cusp_at_hue(hue_target, &primaries, white, &model)
        })
        .collect()
}

/// Public entry point: find the CIECAM02 cusp for arbitrary primaries.
///
/// This wrapper exposes the otherwise-private `find_cusp_at_hue` so that
/// other modules (e.g. `aces_gamut`) can compute cusp tables from any set of
/// CIE xy primaries without duplicating the boundary-scan algorithm.
///
/// # Arguments
///
/// * `hue_target` - Target CIECAM02 hue angle in degrees [0, 360).
/// * `primaries` - CIE xy chromaticity coordinates for [R, G, B].
/// * `white` - CIE xy white-point chromaticity (e.g. D65 = [0.3127, 0.3290]).
/// * `model` - Shared CIECAM02 model (viewing conditions already embedded).
///
/// # Returns
///
/// The [`GamutCusp`] with the highest CIECAM02 chroma found near `hue_target`.
#[must_use]
pub fn find_cusp_at_hue_for_primaries(
    hue_target: f64,
    primaries: &[[f64; 2]; 3],
    white: [f64; 2],
    model: &CiecamModel,
) -> GamutCusp {
    find_cusp_at_hue(hue_target, primaries, white, model)
}

/// Find the cusp at a specific target hue angle by scanning the gamut boundary.
///
/// Scans the three edges of the RGB primary triangle (R→G, G→B, B→R) each at
/// multiple luminance levels to locate the point with the highest CIECAM02
/// chroma whose CIECAM02 hue is closest to `hue_target`.
fn find_cusp_at_hue(
    hue_target: f64,
    primaries: &[[f64; 2]; 3],
    white: [f64; 2],
    model: &CiecamModel,
) -> GamutCusp {
    // Build 3×3 RGB→XYZ matrix from primaries.
    let m = rgb_to_xyz_matrix(primaries, white);

    // Sample the boundary: for each primary edge at multiple luminance levels
    // and also sample pure primaries / secondaries.
    let mut best_c = 0.0_f64;
    let mut best_j = 50.0_f64;

    // Number of steps along each hue edge and each luminance slice.
    const EDGE_STEPS: usize = 64;
    const LUM_STEPS: usize = 16;

    // Helper: evaluate one RGB triplet and update best if hue matches.
    let mut try_rgb = |r: f64, g: f64, b: f64| {
        let (x, y, z) = apply_matrix(&m, r, g, b);
        // Scale to Y=100 convention.
        let (x100, y100, z100) = (x * 100.0, y * 100.0, z * 100.0);
        let app = model.xyz_to_appearance(x100 as f32, y100 as f32, z100 as f32);
        let h = app.hue_angle as f64;
        let c = app.chroma as f64;
        let j = app.lightness as f64;

        // Accept this sample if its hue is close enough to hue_target and
        // its chroma exceeds the current best.
        let hue_diff = angle_diff(h, hue_target);
        if hue_diff < 360.0 / CUSP_TABLE_SIZE as f64 * 2.0 && c > best_c {
            best_c = c;
            best_j = j;
        }
    };

    // Scan boundary edges.
    for step_idx in 0..=EDGE_STEPS {
        let t = step_idx as f64 / EDGE_STEPS as f64;
        // R→G edge
        try_rgb(1.0 - t, t, 0.0);
        // G→B edge
        try_rgb(0.0, 1.0 - t, t);
        // B→R edge
        try_rgb(t, 0.0, 1.0 - t);
    }

    // Also scan at different luminance levels on each primary edge.
    for lum_idx in 1..LUM_STEPS {
        let lum = lum_idx as f64 / LUM_STEPS as f64;
        for step_idx in 0..=EDGE_STEPS {
            let t = step_idx as f64 / EDGE_STEPS as f64;
            // Lerp hue edge, then scale by luminance.
            let r = (1.0 - t) * lum;
            let g = t * lum;
            let b = 0.0;
            try_rgb(r, g, b);
            try_rgb(0.0, (1.0 - t) * lum, t * lum);
            try_rgb(t * lum, 0.0, (1.0 - t) * lum);
        }
    }

    // Fallback: if nothing matched, use a neutral grey at J≈50.
    if best_c < 1e-6 {
        best_c = 0.0;
        best_j = 50.0;
    }

    GamutCusp {
        hue_angle: hue_target,
        max_chroma: best_c,
        lightness: best_j,
    }
}

// ── Cusp interpolation ────────────────────────────────────────────────────

/// Linearly interpolate between two adjacent cusp entries.
fn interpolate_cusp(cusps: &[GamutCusp], hue_deg: f64) -> GamutCusp {
    let n = cusps.len();
    if n == 0 {
        return GamutCusp {
            hue_angle: hue_deg,
            max_chroma: 0.0,
            lightness: 50.0,
        };
    }

    let step = 360.0 / n as f64;
    let h = hue_deg.rem_euclid(360.0);
    let idx_lo = (h / step).floor() as usize % n;
    let idx_hi = (idx_lo + 1) % n;
    let t = (h / step) - (h / step).floor();

    let lo = &cusps[idx_lo];
    let hi = &cusps[idx_hi];

    GamutCusp {
        hue_angle: hue_deg,
        max_chroma: lo.max_chroma + t * (hi.max_chroma - lo.max_chroma),
        lightness: lo.lightness + t * (hi.lightness - lo.lightness),
    }
}

// ── In-gamut test ─────────────────────────────────────────────────────────

/// Return `true` if (J, C) is inside the gamut defined by `cusp`.
///
/// We use a simple linear boundary model: at a given J, the maximum chroma
/// is approximated by linear scaling from 0 at J=0 and J=100 toward C_cusp
/// at J=J_cusp.
#[inline]
fn is_inside_gamut(j: f64, c: f64, cusp: &GamutCusp) -> bool {
    c <= max_chroma_at_j(j, cusp) + 1e-6
}

/// Approximate maximum allowed chroma at lightness J for the given cusp.
///
/// Uses a piecewise linear model:
/// - Segment 1: (0, 0) → (J_cusp, C_cusp)
/// - Segment 2: (J_cusp, C_cusp) → (100, 0)
fn max_chroma_at_j(j: f64, cusp: &GamutCusp) -> f64 {
    let j_cusp = cusp.lightness.clamp(1.0, 99.0);
    let c_cusp = cusp.max_chroma.max(0.0);

    if j <= j_cusp {
        c_cusp * (j / j_cusp)
    } else {
        c_cusp * ((100.0 - j) / (100.0 - j_cusp))
    }
}

// ── Cusp-directed compression ─────────────────────────────────────────────

/// Compress (J_in, C_in) toward the cusp until it lands on the gamut boundary.
///
/// The compression is a linear interpolation along the JC line from the input
/// to the cusp:
///   (J, C)(t) = (1−t)·(J_in, C_in) + t·(J_cusp, C_cusp),  t ∈ [0, 1]
///
/// At t=0 the point is outside the gamut; at t=1 it is at the cusp (inside or
/// on the boundary).  We bisect to find the smallest t for which the point is
/// inside the gamut, then return that compressed (J, C).
fn compress_toward_cusp(j_in: f64, c_in: f64, cusp: &GamutCusp) -> (f64, f64) {
    let j_target = cusp.lightness;
    let c_target = cusp.max_chroma;

    // Safety check: if the cusp itself is outside by the model (degenerate),
    // just return the cusp coordinates directly.
    let j_cusp = j_in + 1.0 * (j_target - j_in);
    let c_cusp = (c_in + 1.0 * (c_target - c_in)).max(0.0);
    if !is_inside_gamut(j_cusp, c_cusp, cusp) {
        return (j_cusp, c_cusp);
    }

    // Bisect: t_lo is outside, t_hi is inside.
    let mut t_lo = 0.0_f64;
    let mut t_hi = 1.0_f64;

    for _ in 0..64 {
        let t_mid = (t_lo + t_hi) * 0.5;
        let j_mid = j_in + t_mid * (j_target - j_in);
        let c_mid = (c_in + t_mid * (c_target - c_in)).max(0.0);
        if is_inside_gamut(j_mid, c_mid, cusp) {
            // t_mid is already inside → try a smaller t.
            t_hi = t_mid;
        } else {
            // t_mid is outside → need a larger t.
            t_lo = t_mid;
        }
    }

    // Use t_hi: the smallest t for which the point is inside (or on) the boundary.
    let t = t_hi;
    let j_out = j_in + t * (j_target - j_in);
    let c_out = (c_in + t * (c_target - c_in)).max(0.0);
    (j_out, c_out)
}

// ── RGB ↔ XYZ matrix helpers ──────────────────────────────────────────────

/// Build an RGB→XYZ (Y=1 normalised) matrix from CIE xy primaries and white.
///
/// Follows the standard derivation (Fairchild "Color Appearance Models", App. A).
fn rgb_to_xyz_matrix(primaries: &[[f64; 2]; 3], white: [f64; 2]) -> [[f64; 3]; 3] {
    // Convert xy → XYZ (Y=1) for each primary.
    let xyz_r = xy_to_xyz(primaries[0]);
    let xyz_g = xy_to_xyz(primaries[1]);
    let xyz_b = xy_to_xyz(primaries[2]);

    // Assemble M (column vectors are primaries).
    let m = [
        [xyz_r[0], xyz_g[0], xyz_b[0]],
        [xyz_r[1], xyz_g[1], xyz_b[1]],
        [xyz_r[2], xyz_g[2], xyz_b[2]],
    ];

    // White XYZ (Y=1).
    let w_xyz = xy_to_xyz(white);

    // Solve M·s = w for scaling vector s.
    let m_inv = invert_3x3(&m);
    let s = mat3_vec3(&m_inv, w_xyz);

    // Scale columns by s.
    [
        [m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2]],
        [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2]],
        [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2]],
    ]
}

/// Convert CIE xy → XYZ with Y = 1.
#[inline]
fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    let (x, y) = (xy[0], xy[1]);
    if y.abs() < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Apply a 3×3 matrix to a 3-vector.
#[inline]
fn mat3_vec3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Apply RGB→XYZ matrix and return (X, Y, Z).
#[inline]
fn apply_matrix(m: &[[f64; 3]; 3], r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let xyz = mat3_vec3(m, [r, g, b]);
    (xyz[0], xyz[1], xyz[2])
}

/// Invert a 3×3 matrix; returns identity on near-singular input.
fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let k = m[2][2];

    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);

    if det.abs() < 1e-14 {
        // Fallback: identity.
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;
    [
        [
            (e * k - f * h) * inv_det,
            (c * h - b * k) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * k) * inv_det,
            (a * k - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

// ── Angle difference helper ───────────────────────────────────────────────

/// Smallest absolute angular difference in degrees (result ∈ [0, 180]).
#[inline]
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GamutCusp construction ────────────────────────────────────────────

    #[test]
    fn test_cusp_table_length() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        assert_eq!(mapper.cusps.len(), CUSP_TABLE_SIZE);
    }

    #[test]
    fn test_cusp_hue_angles_span_360() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        let first = mapper.cusps.first().map(|c| c.hue_angle).unwrap_or(0.0);
        let last = mapper.cusps.last().map(|c| c.hue_angle).unwrap_or(0.0);
        assert!(first < 1.0, "first hue should be near 0, got {first}");
        assert!(last > 358.0, "last hue should be near 360, got {last}");
    }

    #[test]
    fn test_cusp_chroma_non_negative() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        for c in &mapper.cusps {
            assert!(
                c.max_chroma >= 0.0,
                "negative cusp chroma at hue {}",
                c.hue_angle
            );
        }
    }

    #[test]
    fn test_cusp_lightness_in_range() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        for c in &mapper.cusps {
            assert!(
                c.lightness >= 0.0 && c.lightness <= 100.0,
                "cusp lightness {} out of [0,100]",
                c.lightness
            );
        }
    }

    // ── Cusp interpolation ────────────────────────────────────────────────

    #[test]
    fn test_cusp_at_hue_wraps_360() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        let c0 = mapper.cusp_at_hue(0.0);
        let c360 = mapper.cusp_at_hue(360.0);
        // Both should land on the same index (0).
        assert!(
            (c0.max_chroma - c360.max_chroma).abs() < 1e-6,
            "0° and 360° cusps differ: {} vs {}",
            c0.max_chroma,
            c360.max_chroma
        );
    }

    #[test]
    fn test_cusp_at_hue_mid_interpolates() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        let step = 360.0 / CUSP_TABLE_SIZE as f64;
        // Ask for a hue exactly halfway between two table entries.
        let h = step * 10.5;
        let cusp = mapper.cusp_at_hue(h);
        // Just verify it returns something sensible.
        assert!(cusp.max_chroma >= 0.0);
        assert!(cusp.lightness >= 0.0 && cusp.lightness <= 100.0);
    }

    // ── max_chroma_at_j ───────────────────────────────────────────────────

    #[test]
    fn test_max_chroma_at_j_zero_at_extremes() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 50.0,
            lightness: 50.0,
        };
        assert!((max_chroma_at_j(0.0, &cusp)).abs() < 1e-9);
        assert!((max_chroma_at_j(100.0, &cusp)).abs() < 1e-9);
    }

    #[test]
    fn test_max_chroma_at_j_peak_at_cusp() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 50.0,
            lightness: 50.0,
        };
        let c_at_cusp = max_chroma_at_j(50.0, &cusp);
        assert!(
            (c_at_cusp - 50.0).abs() < 1e-9,
            "expected 50, got {c_at_cusp}"
        );
    }

    #[test]
    fn test_max_chroma_monotone_below_cusp() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 60.0,
            lightness: 40.0,
        };
        let c20 = max_chroma_at_j(20.0, &cusp);
        let c30 = max_chroma_at_j(30.0, &cusp);
        let c40 = max_chroma_at_j(40.0, &cusp);
        assert!(c20 < c30 && c30 < c40);
    }

    // ── is_inside_gamut ───────────────────────────────────────────────────

    #[test]
    fn test_achromatic_always_inside() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 50.0,
            lightness: 50.0,
        };
        assert!(is_inside_gamut(50.0, 0.0, &cusp));
        assert!(is_inside_gamut(10.0, 0.0, &cusp));
        assert!(is_inside_gamut(90.0, 0.0, &cusp));
    }

    #[test]
    fn test_high_chroma_outside_gamut() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 30.0,
            lightness: 50.0,
        };
        assert!(!is_inside_gamut(50.0, 60.0, &cusp));
    }

    // ── compress_toward_cusp ──────────────────────────────────────────────

    #[test]
    fn test_compression_lands_inside_gamut() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 40.0,
            lightness: 50.0,
        };
        let (j_out, c_out) = compress_toward_cusp(50.0, 80.0, &cusp);
        assert!(is_inside_gamut(j_out, c_out, &cusp));
    }

    #[test]
    fn test_compression_reduces_chroma() {
        let cusp = GamutCusp {
            hue_angle: 0.0,
            max_chroma: 40.0,
            lightness: 50.0,
        };
        let (_, c_out) = compress_toward_cusp(50.0, 80.0, &cusp);
        assert!(c_out < 80.0, "chroma should decrease after compression");
    }

    // ── map_xyz ───────────────────────────────────────────────────────────

    #[test]
    fn test_map_xyz_in_gamut_unchanged() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        // Mid-grey (Y=50 scale): achromatic, clearly inside gamut.
        // D65-proportional grey: X≈47.5, Y=50, Z≈54.4
        let (xo, yo, zo) = mapper.map_xyz(47.524, 50.0, 54.442);
        // The CIECAM02 round-trip introduces small numerical errors; allow 2 units.
        assert!((xo - 47.524).abs() < 2.0, "X changed: {xo}");
        assert!((yo - 50.0).abs() < 2.0, "Y changed: {yo}");
        assert!((zo - 54.442).abs() < 2.0, "Z changed: {zo}");
    }

    #[test]
    fn test_map_xyz_out_of_gamut_returns_valid_xyz() {
        let mapper = CuspGamutMapper::new(TargetGamut::Rec709);
        // Extremely saturated colour (likely out of Rec.709).
        let (xo, yo, zo) = mapper.map_xyz(120.0, 50.0, 5.0);
        // Just verify the output is finite and non-negative Y.
        assert!(xo.is_finite(), "X not finite");
        assert!(yo.is_finite(), "Y not finite");
        assert!(zo.is_finite(), "Z not finite");
        assert!(yo >= 0.0, "Y negative: {yo}");
    }

    #[test]
    fn test_map_xyz_different_gamuts() {
        // DCI-P3 and Rec.2020 mappers should at least construct without panic.
        let _p3 = CuspGamutMapper::new(TargetGamut::DciP3);
        let _r2020 = CuspGamutMapper::new(TargetGamut::Rec2020);
    }

    // ── Matrix helpers ────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_xyz_matrix_white_maps_to_white() {
        let m = rgb_to_xyz_matrix(&REC709_PRIMARIES, D65_WHITE);
        let (x, y, z) = apply_matrix(&m, 1.0, 1.0, 1.0);
        let wx = D65_WHITE[0] / D65_WHITE[1];
        let wz = (1.0 - D65_WHITE[0] - D65_WHITE[1]) / D65_WHITE[1];
        assert!((x - wx).abs() < 1e-4, "X={x} expected {wx}");
        assert!((y - 1.0).abs() < 1e-6, "Y={y} expected 1.0");
        assert!((z - wz).abs() < 1e-4, "Z={z} expected {wz}");
    }

    #[test]
    fn test_invert_3x3_identity_roundtrip() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_angle_diff_symmetry() {
        assert!((angle_diff(10.0, 350.0) - 20.0).abs() < 1e-9);
        assert!((angle_diff(350.0, 10.0) - 20.0).abs() < 1e-9);
    }

    // ── All three gamuts build cusp tables ────────────────────────────────

    #[test]
    fn test_all_target_gamuts_build_cusp_tables() {
        for target in [
            TargetGamut::Rec709,
            TargetGamut::DciP3,
            TargetGamut::Rec2020,
        ] {
            let mapper = CuspGamutMapper::new(target);
            assert_eq!(mapper.cusps.len(), CUSP_TABLE_SIZE);
        }
    }

    // ── Dim / dark viewing conditions ─────────────────────────────────────

    #[test]
    fn test_custom_viewing_conditions() {
        let viewing = CiecamViewingConditions {
            adapting_luminance_la: 32.0,
            background_relative_lum_yb: 15.0,
            surround: SurroundCondition::Dim,
        };
        let mapper = CuspGamutMapper::with_viewing_conditions(TargetGamut::Rec709, viewing);
        assert_eq!(mapper.cusps.len(), CUSP_TABLE_SIZE);
    }
}
