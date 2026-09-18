//! Calibration LUT generation and verification.
//!
//! This module provides tools for generating calibration LUTs from measurements
//! and verifying their accuracy.

pub mod cached;
pub mod generate;
pub mod measure;
pub mod tetrahedral;
pub mod verify;

pub use cached::{Lut3dCached, TetraCoeff};
pub use generate::LutGenerator;
pub use measure::LutMeasurement;
pub use tetrahedral::TetrahedralInterpolator;
pub use verify::LutVerifier;

// ---------------------------------------------------------------------------
// Free-function 3D LUT interpolation API
// ---------------------------------------------------------------------------

/// Sample a 3D LUT using **trilinear** interpolation.
///
/// The LUT data `lut` is a flat `Vec<f32>` in `[R][G][B]` major order, where
/// each entry stores the three output channels consecutively:
/// `lut[((r_idx * lut_size + g_idx) * lut_size + b_idx) * 3 + ch]`.
///
/// Input values `r`, `g`, `b` are clamped to `[0.0, 1.0]` before lookup.
///
/// # Arguments
///
/// * `lut`      - Flat 3-channel LUT data.
/// * `lut_size` - Number of grid points along each axis.
/// * `r`        - Red input (clamped to `[0, 1]`).
/// * `g`        - Green input (clamped to `[0, 1]`).
/// * `b`        - Blue input (clamped to `[0, 1]`).
///
/// # Returns
///
/// `[f32; 3]` interpolated output colour.
#[must_use]
pub fn trilinear_lookup_3d(lut: &[f32], lut_size: usize, r: f32, g: f32, b: f32) -> [f32; 3] {
    if lut_size < 2 || lut.len() < lut_size * lut_size * lut_size * 3 {
        return [0.0, 0.0, 0.0];
    }

    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let scale = (lut_size - 1) as f32;
    let rp = r * scale;
    let gp = g * scale;
    let bp = b * scale;

    let r0 = rp.floor() as usize;
    let g0 = gp.floor() as usize;
    let b0 = bp.floor() as usize;

    let r1 = (r0 + 1).min(lut_size - 1);
    let g1 = (g0 + 1).min(lut_size - 1);
    let b1 = (b0 + 1).min(lut_size - 1);

    let dr = rp - r0 as f32;
    let dg = gp - g0 as f32;
    let db = bp - b0 as f32;

    // Fetch a single channel value from the LUT at lattice coordinates.
    let get = |ri: usize, gi: usize, bi: usize, ch: usize| -> f32 {
        let idx = ((ri * lut_size + gi) * lut_size + bi) * 3 + ch;
        lut[idx]
    };

    let mut out = [0.0_f32; 3];
    for ch in 0..3 {
        let c000 = get(r0, g0, b0, ch);
        let c100 = get(r1, g0, b0, ch);
        let c010 = get(r0, g1, b0, ch);
        let c110 = get(r1, g1, b0, ch);
        let c001 = get(r0, g0, b1, ch);
        let c101 = get(r1, g0, b1, ch);
        let c011 = get(r0, g1, b1, ch);
        let c111 = get(r1, g1, b1, ch);

        let c00 = c000 * (1.0 - dr) + c100 * dr;
        let c10 = c010 * (1.0 - dr) + c110 * dr;
        let c01 = c001 * (1.0 - dr) + c101 * dr;
        let c11 = c011 * (1.0 - dr) + c111 * dr;

        let c0 = c00 * (1.0 - dg) + c10 * dg;
        let c1 = c01 * (1.0 - dg) + c11 * dg;

        out[ch] = c0 * (1.0 - db) + c1 * db;
    }
    out
}

/// Sample a 3D LUT using **tetrahedral** interpolation.
///
/// Uses the same flat LUT layout as [`trilinear_lookup_3d`].  The unit cube
/// around the sample point is decomposed into one of six tetrahedra by
/// sorting `(dr, dg, db)`, and then Barycentric weights are applied to the
/// four enclosing lattice vertices.
///
/// Input values are clamped to `[0.0, 1.0]`.
#[must_use]
pub fn tetrahedral_lookup_3d(lut: &[f32], lut_size: usize, r: f32, g: f32, b: f32) -> [f32; 3] {
    if lut_size < 2 || lut.len() < lut_size * lut_size * lut_size * 3 {
        return [0.0, 0.0, 0.0];
    }

    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let scale = (lut_size - 1) as f32;
    let rp = r * scale;
    let gp = g * scale;
    let bp = b * scale;

    let r0 = rp.floor() as usize;
    let g0 = gp.floor() as usize;
    let b0 = bp.floor() as usize;

    let r1 = (r0 + 1).min(lut_size - 1);
    let g1 = (g0 + 1).min(lut_size - 1);
    let b1 = (b0 + 1).min(lut_size - 1);

    let dr = rp - r0 as f32;
    let dg = gp - g0 as f32;
    let db = bp - b0 as f32;

    let get = |ri: usize, gi: usize, bi: usize, ch: usize| -> f32 {
        let idx = ((ri * lut_size + gi) * lut_size + bi) * 3 + ch;
        lut[idx]
    };

    // Tetrahedral decomposition: six cases based on ordering of (dr, dg, db).
    let mut out = [0.0_f32; 3];
    for ch in 0..3 {
        let v000 = get(r0, g0, b0, ch);
        let v111 = get(r1, g1, b1, ch);

        out[ch] = if dr >= dg && dg >= db {
            // Tetrahedron 1: dr >= dg >= db
            let v100 = get(r1, g0, b0, ch);
            let v110 = get(r1, g1, b0, ch);
            (1.0 - dr) * v000 + (dr - dg) * v100 + (dg - db) * v110 + db * v111
        } else if dr >= db && db >= dg {
            // Tetrahedron 2: dr >= db >= dg
            let v100 = get(r1, g0, b0, ch);
            let v101 = get(r1, g0, b1, ch);
            (1.0 - dr) * v000 + (dr - db) * v100 + (db - dg) * v101 + dg * v111
        } else if db >= dr && dr >= dg {
            // Tetrahedron 3: db >= dr >= dg
            let v001 = get(r0, g0, b1, ch);
            let v101 = get(r1, g0, b1, ch);
            (1.0 - db) * v000 + (db - dr) * v001 + (dr - dg) * v101 + dg * v111
        } else if dg >= dr && dr >= db {
            // Tetrahedron 4: dg >= dr >= db
            let v010 = get(r0, g1, b0, ch);
            let v110 = get(r1, g1, b0, ch);
            (1.0 - dg) * v000 + (dg - dr) * v010 + (dr - db) * v110 + db * v111
        } else if dg >= db && db >= dr {
            // Tetrahedron 5: dg >= db >= dr
            let v010 = get(r0, g1, b0, ch);
            let v011 = get(r0, g1, b1, ch);
            (1.0 - dg) * v000 + (dg - db) * v010 + (db - dr) * v011 + dr * v111
        } else {
            // Tetrahedron 6: db >= dg >= dr
            let v001 = get(r0, g0, b1, ch);
            let v011 = get(r0, g1, b1, ch);
            (1.0 - db) * v000 + (db - dg) * v001 + (dg - dr) * v011 + dr * v111
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Helper: build a flat identity 3D LUT
// ---------------------------------------------------------------------------

/// Build a flat identity 3D LUT of `lut_size³ × 3` f32 values.
///
/// Useful for testing: for any input `(r, g, b)` the lookup should return the
/// same values.
#[must_use]
pub fn build_identity_lut(lut_size: usize) -> Vec<f32> {
    let n = lut_size;
    let mut lut = vec![0.0_f32; n * n * n * 3];
    for ri in 0..n {
        for gi in 0..n {
            for bi in 0..n {
                let idx = ((ri * n + gi) * n + bi) * 3;
                lut[idx] = ri as f32 / (n - 1) as f32;
                lut[idx + 1] = gi as f32 / (n - 1) as f32;
                lut[idx + 2] = bi as f32 / (n - 1) as f32;
            }
        }
    }
    lut
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tetrahedral interpolation of an identity LUT must return the input.
    #[test]
    fn test_tetrahedral_identity_lut() {
        let lut_size = 17_usize;
        let lut = build_identity_lut(lut_size);

        let test_cases: &[(f32, f32, f32)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.25, 0.75, 0.5),
            (0.1, 0.3, 0.9),
        ];

        for &(r, g, b) in test_cases {
            let out = tetrahedral_lookup_3d(&lut, lut_size, r, g, b);
            assert!(
                (out[0] - r).abs() < 1e-4,
                "R: in={r}, out={}, case=({r},{g},{b})",
                out[0]
            );
            assert!(
                (out[1] - g).abs() < 1e-4,
                "G: in={g}, out={}, case=({r},{g},{b})",
                out[1]
            );
            assert!(
                (out[2] - b).abs() < 1e-4,
                "B: in={b}, out={}, case=({r},{g},{b})",
                out[2]
            );
        }
    }

    #[test]
    fn test_trilinear_identity_lut() {
        let lut_size = 17_usize;
        let lut = build_identity_lut(lut_size);

        let out = trilinear_lookup_3d(&lut, lut_size, 0.5, 0.25, 0.75);
        assert!((out[0] - 0.5).abs() < 1e-4, "R: {}", out[0]);
        assert!((out[1] - 0.25).abs() < 1e-4, "G: {}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-4, "B: {}", out[2]);
    }

    #[test]
    fn test_trilinear_clamps_out_of_range() {
        let lut_size = 2_usize;
        let lut = build_identity_lut(lut_size);
        let out = trilinear_lookup_3d(&lut, lut_size, -0.5, 1.5, 0.5);
        assert!(out[0] >= 0.0 && out[0] <= 1.0);
        assert!(out[1] >= 0.0 && out[1] <= 1.0);
    }

    #[test]
    fn test_tetrahedral_matches_trilinear_on_lattice_points() {
        let lut_size = 5_usize;
        let lut = build_identity_lut(lut_size);
        // At exact grid points both methods must agree closely.
        for i in 0..lut_size {
            let v = i as f32 / (lut_size - 1) as f32;
            let tri = trilinear_lookup_3d(&lut, lut_size, v, v, v);
            let tet = tetrahedral_lookup_3d(&lut, lut_size, v, v, v);
            for ch in 0..3 {
                assert!(
                    (tri[ch] - tet[ch]).abs() < 1e-5,
                    "ch={ch} at v={v}: tri={} tet={}",
                    tri[ch],
                    tet[ch]
                );
            }
        }
    }
}
