//! Tetrahedral interpolation for 3-D LUT lookups.
//!
//! Tetrahedral interpolation decomposes the unit cube around each input sample
//! into one of six tetrahedra and performs barycentric interpolation within
//! that tetrahedron. This generally produces higher quality than trilinear
//! interpolation with very similar performance.
//!
//! Reference: "Color and Mastering for Digital Cinema" (Glenn Kennel, 2006).

use crate::Rgb;

// ---------------------------------------------------------------------------
// Tetrahedron selection
// ---------------------------------------------------------------------------

/// Which of the 6 tetrahedra inside the unit cube contains the fractional
/// point `(dr, dg, db)` where all values are in `[0, 1]`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tetrahedron {
    /// Tetrahedron where dr >= dg >= db.
    T0,
    /// Tetrahedron where dr >= db > dg.
    T1,
    /// Tetrahedron where db > dr >= dg.
    T2,
    /// Tetrahedron where dg > dr >= db (and dg >= db).
    T3,
    /// Tetrahedron where dg >= db > dr.
    T4,
    /// Tetrahedron where db > dg >= dr.
    T5,
}

/// Select the tetrahedron for fractional offsets `(dr, dg, db)` in `[0, 1]³`.
#[allow(dead_code)]
#[must_use]
pub fn select_tetrahedron(dr: f64, dg: f64, db: f64) -> Tetrahedron {
    if dr >= dg {
        if dg >= db {
            Tetrahedron::T0 // dr ≥ dg ≥ db
        } else if dr >= db {
            Tetrahedron::T1 // dr ≥ db > dg
        } else {
            Tetrahedron::T2 // db > dr ≥ dg ... but db > dr and dr ≥ dg
        }
    } else {
        // dg > dr
        if dr >= db {
            Tetrahedron::T3 // dg > dr ≥ db
        } else if dg >= db {
            Tetrahedron::T4 // dg ≥ db > dr
        } else {
            Tetrahedron::T5 // db > dg > dr
        }
    }
}

// ---------------------------------------------------------------------------
// Barycentric interpolation
// ---------------------------------------------------------------------------

/// Barycentric weights for a point inside one of the six tetrahedra.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct BarycentricWeights {
    /// Weight for the base (low, low, low) lattice vertex.
    pub w0: f64,
    /// Weight for the first axis vertex.
    pub w1: f64,
    /// Weight for the second axis vertex.
    pub w2: f64,
    /// Weight for the corner (high, high, high) lattice vertex.
    pub w3: f64,
}

/// Compute barycentric weights for the given tetrahedron and fractional offset.
///
/// The four vertices implied by the weights are:
/// * `v0 = lut[r0][g0][b0]` (low corner)
/// * `v1`, `v2` depend on the tetrahedron (one-step axis moves)
/// * `v3 = lut[r1][g1][b1]` (high corner)
///
/// Returns `(BarycentricWeights, axis_order)` where `axis_order` encodes which
/// intermediate vertices to sample: `(0=R, 1=G, 2=B)` in order.
#[allow(dead_code)]
#[must_use]
pub fn barycentric_weights(
    tet: Tetrahedron,
    dr: f64,
    dg: f64,
    db: f64,
) -> (BarycentricWeights, [u8; 2]) {
    match tet {
        Tetrahedron::T0 => {
            // v1 = (+r), v2 = (+g)
            let w3 = db;
            let w2 = dg - db;
            let w1 = dr - dg;
            let w0 = 1.0 - dr;
            (BarycentricWeights { w0, w1, w2, w3 }, [0, 1])
        }
        Tetrahedron::T1 => {
            // v1 = (+r), v2 = (+b)
            let w3 = dg;
            let w2 = db - dg;
            let w1 = dr - db;
            let w0 = 1.0 - dr;
            (BarycentricWeights { w0, w1, w2, w3 }, [0, 2])
        }
        Tetrahedron::T2 => {
            // v1 = (+b), v2 = (+r)
            let w3 = dg;
            let w2 = dr - dg;
            let w1 = db - dr;
            let w0 = 1.0 - db;
            (BarycentricWeights { w0, w1, w2, w3 }, [2, 0])
        }
        Tetrahedron::T3 => {
            // v1 = (+g), v2 = (+r)
            let w3 = db;
            let w2 = dr - db;
            let w1 = dg - dr;
            let w0 = 1.0 - dg;
            (BarycentricWeights { w0, w1, w2, w3 }, [1, 0])
        }
        Tetrahedron::T4 => {
            // v1 = (+g), v2 = (+b)
            let w3 = dr;
            let w2 = db - dr;
            let w1 = dg - db;
            let w0 = 1.0 - dg;
            (BarycentricWeights { w0, w1, w2, w3 }, [1, 2])
        }
        Tetrahedron::T5 => {
            // v1 = (+b), v2 = (+g)
            let w3 = dr;
            let w2 = dg - dr;
            let w1 = db - dg;
            let w0 = 1.0 - db;
            (BarycentricWeights { w0, w1, w2, w3 }, [2, 1])
        }
    }
}

// ---------------------------------------------------------------------------
// Tetrahedral lookup
// ---------------------------------------------------------------------------

/// Perform tetrahedral interpolation on a 3-D LUT.
///
/// # Arguments
///
/// * `lut` – flat slice of `Rgb` values, stored in `[r][g][b]` order.
/// * `size` – number of entries per dimension (e.g. 33 for a 33³ LUT).
/// * `input` – normalised input colour in `[0, 1]³`.
///
/// # Panics
///
/// Panics if `lut.len() != size * size * size` or if `size < 2`.
#[allow(dead_code)]
#[must_use]
pub fn tetrahedral_lookup(lut: &[Rgb], size: usize, input: &Rgb) -> Rgb {
    assert!(size >= 2, "LUT size must be at least 2");
    assert_eq!(lut.len(), size * size * size, "LUT length mismatch");

    let scale = (size - 1) as f64;

    let r = input[0].clamp(0.0, 1.0) * scale;
    let g = input[1].clamp(0.0, 1.0) * scale;
    let b = input[2].clamp(0.0, 1.0) * scale;

    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;

    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let dr = r - r0 as f64;
    let dg = g - g0 as f64;
    let db = b - b0 as f64;

    // Helper to index the flat lut slice.
    let idx = |ri: usize, gi: usize, bi: usize| -> &Rgb { &lut[ri * size * size + gi * size + bi] };

    // The base low-corner vertex.
    let v0 = idx(r0, g0, b0);
    // The high-corner vertex.
    let v3 = idx(r1, g1, b1);

    let tet = select_tetrahedron(dr, dg, db);
    let (bary, axes) = barycentric_weights(tet, dr, dg, db);

    // Resolve the two intermediate vertices from the axis order.
    let (v1, v2) = match axes {
        [0, 1] => (idx(r1, g0, b0), idx(r1, g1, b0)),
        [0, 2] => (idx(r1, g0, b0), idx(r1, g0, b1)),
        [2, 0] => (idx(r0, g0, b1), idx(r1, g0, b1)),
        [1, 0] => (idx(r0, g1, b0), idx(r1, g1, b0)),
        [1, 2] => (idx(r0, g1, b0), idx(r0, g1, b1)),
        [2, 1] => (idx(r0, g0, b1), idx(r0, g1, b1)),
        _ => unreachable!(),
    };

    [
        bary.w0 * v0[0] + bary.w1 * v1[0] + bary.w2 * v2[0] + bary.w3 * v3[0],
        bary.w0 * v0[1] + bary.w1 * v1[1] + bary.w2 * v2[1] + bary.w3 * v3[1],
        bary.w0 * v0[2] + bary.w1 * v1[2] + bary.w2 * v2[2] + bary.w3 * v3[2],
    ]
}

/// Trilinear fallback interpolation (for comparison / validation).
#[allow(dead_code)]
#[must_use]
pub fn trilinear_lookup(lut: &[Rgb], size: usize, input: &Rgb) -> Rgb {
    assert!(size >= 2, "LUT size must be at least 2");
    assert_eq!(lut.len(), size * size * size, "LUT length mismatch");

    let scale = (size - 1) as f64;
    let r = input[0].clamp(0.0, 1.0) * scale;
    let g = input[1].clamp(0.0, 1.0) * scale;
    let b = input[2].clamp(0.0, 1.0) * scale;

    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let dr = r - r0 as f64;
    let dg = g - g0 as f64;
    let db = b - b0 as f64;

    let idx = |ri: usize, gi: usize, bi: usize| -> Rgb { lut[ri * size * size + gi * size + bi] };

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let c000 = idx(r0, g0, b0)[ch];
        let c100 = idx(r1, g0, b0)[ch];
        let c010 = idx(r0, g1, b0)[ch];
        let c110 = idx(r1, g1, b0)[ch];
        let c001 = idx(r0, g0, b1)[ch];
        let c101 = idx(r1, g0, b1)[ch];
        let c011 = idx(r0, g1, b1)[ch];
        let c111 = idx(r1, g1, b1)[ch];

        out[ch] = c000 * (1.0 - dr) * (1.0 - dg) * (1.0 - db)
            + c100 * dr * (1.0 - dg) * (1.0 - db)
            + c010 * (1.0 - dr) * dg * (1.0 - db)
            + c110 * dr * dg * (1.0 - db)
            + c001 * (1.0 - dr) * (1.0 - dg) * db
            + c101 * dr * (1.0 - dg) * db
            + c011 * (1.0 - dr) * dg * db
            + c111 * dr * dg * db;
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an identity LUT of the given size.
    fn identity_lut(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn rgb_approx_eq(a: &Rgb, b: &Rgb) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_identity_lut_corners_tetrahedral() {
        let lut = identity_lut(3);
        let corners: &[Rgb] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        for c in corners {
            let out = tetrahedral_lookup(&lut, 3, c);
            assert!(rgb_approx_eq(&out, c), "{out:?} != {c:?}");
        }
    }

    #[test]
    fn test_identity_lut_midpoint() {
        let lut = identity_lut(3);
        let mid = [0.5, 0.5, 0.5];
        let out = tetrahedral_lookup(&lut, 3, &mid);
        assert!(rgb_approx_eq(&out, &mid), "{out:?}");
    }

    #[test]
    fn test_tetrahedral_vs_trilinear_identity() {
        let lut = identity_lut(5);
        let inputs: &[Rgb] = &[[0.1, 0.2, 0.3], [0.7, 0.5, 0.9], [0.0, 1.0, 0.5]];
        for inp in inputs {
            let tet = tetrahedral_lookup(&lut, 5, inp);
            let tri = trilinear_lookup(&lut, 5, inp);
            // On an identity LUT both methods must agree to high precision.
            assert!(
                rgb_approx_eq(&tet, &tri),
                "tet={tet:?} tri={tri:?} for {inp:?}"
            );
        }
    }

    #[test]
    fn test_select_tetrahedron_t0() {
        // dr ≥ dg ≥ db → T0
        assert_eq!(select_tetrahedron(0.9, 0.5, 0.1), Tetrahedron::T0);
    }

    #[test]
    fn test_select_tetrahedron_t1() {
        // dr ≥ db > dg → T1
        assert_eq!(select_tetrahedron(0.8, 0.1, 0.5), Tetrahedron::T1);
    }

    #[test]
    fn test_select_tetrahedron_t3() {
        // dg > dr ≥ db → T3
        assert_eq!(select_tetrahedron(0.5, 0.9, 0.3), Tetrahedron::T3);
    }

    #[test]
    fn test_select_tetrahedron_t5() {
        // db > dg > dr → T5
        assert_eq!(select_tetrahedron(0.1, 0.4, 0.9), Tetrahedron::T5);
    }

    #[test]
    fn test_barycentric_weights_sum_to_one() {
        let cases: &[(f64, f64, f64)] = &[
            (0.8, 0.5, 0.2),
            (0.3, 0.7, 0.1),
            (0.5, 0.5, 0.5),
            (0.9, 0.1, 0.6),
        ];
        for &(dr, dg, db) in cases {
            let tet = select_tetrahedron(dr, dg, db);
            let (bary, _) = barycentric_weights(tet, dr, dg, db);
            let sum = bary.w0 + bary.w1 + bary.w2 + bary.w3;
            assert!((sum - 1.0).abs() < 1e-12, "sum={sum} for ({dr},{dg},{db})");
        }
    }

    #[test]
    fn test_clamp_out_of_range() {
        let lut = identity_lut(3);
        let out = tetrahedral_lookup(&lut, 3, &[1.5, -0.1, 0.5]);
        assert!(out[0] <= 1.0 + 1e-9 && out[0] >= -1e-9);
    }

    #[test]
    fn test_trilinear_corners() {
        let lut = identity_lut(3);
        let c = [0.0, 0.0, 0.0];
        let out = trilinear_lookup(&lut, 3, &c);
        assert!(rgb_approx_eq(&out, &c));
    }

    #[test]
    fn test_larger_lut_size() {
        let lut = identity_lut(17);
        let inp = [0.33, 0.66, 0.12];
        let out = tetrahedral_lookup(&lut, 17, &inp);
        // With identity LUT the output should be within interpolation tolerance.
        for ch in 0..3 {
            assert!(
                (out[ch] - inp[ch]).abs() < 0.01,
                "ch={ch} out={} inp={}",
                out[ch],
                inp[ch]
            );
        }
    }

    #[test]
    fn test_constant_lut_returns_constant() {
        // A LUT that always maps to [0.5, 0.5, 0.5].
        let size = 3;
        let lut: Vec<Rgb> = vec![[0.5, 0.5, 0.5]; size * size * size];
        let out = tetrahedral_lookup(&lut, size, &[0.3, 0.7, 0.1]);
        assert!(rgb_approx_eq(&out, &[0.5, 0.5, 0.5]));
    }
}
