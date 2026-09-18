//! Pre-computed tetrahedral LUT interpolation coefficient cache.
//!
//! [`Lut3dCached`] wraps a flat 3D LUT (the same layout used by
//! [`super::tetrahedral_lookup_3d`]) and pre-computes per-cell stride data so
//! that the hot per-pixel path replaces index arithmetic with a single table
//! look-up followed by a four-vertex weighted sum.
//!
//! # Data layout
//!
//! The flat LUT stores `lut_size³ × 3` `f32` values in `[R][G][B]` major order:
//! ```text
//! lut[((r_idx * lut_size + g_idx) * lut_size + b_idx) * 3 + ch]
//! ```
//!
//! # Tetrahedron selection
//!
//! Each unit cell is decomposed into one of six tetrahedra by comparing the
//! fractional offsets `(dr, dg, db)`.  The selection depends on the *fractional*
//! offsets of the query point, not on which cell it falls in, so the tetrahedron
//! index cannot be precomputed per-cell.  [`TetraCoeff`] therefore stores the
//! cell's base index and axis strides; the dominant-axis selection happens at
//! query time using the same six-case logic as [`super::tetrahedral_lookup_3d`].

// ---------------------------------------------------------------------------
// TetraCoeff
// ---------------------------------------------------------------------------

/// Per-cell precomputed index data for tetrahedral interpolation.
///
/// Stores the flat index of the `(ri, gi, bi)` corner and the stride for each
/// axis.  Combined with the fractional offsets computed at query time, these
/// allow a direct look-up without repeating the index formula.
#[derive(Debug, Clone, Copy)]
pub struct TetraCoeff {
    /// Flat index into the LUT for the `(ri, gi, bi)` corner:
    /// `((ri * lut_size + gi) * lut_size + bi) * 3`.
    pub base_idx: usize,
    /// Stride for the R axis: `lut_size * lut_size * 3`.
    pub step_r: usize,
    /// Stride for the G axis: `lut_size * 3`.
    pub step_g: usize,
    /// Stride for the B axis: `3`.
    pub step_b: usize,
}

// ---------------------------------------------------------------------------
// Lut3dCached
// ---------------------------------------------------------------------------

/// 3D LUT with precomputed per-cell tetrahedral interpolation coefficients.
///
/// Construct once with [`Lut3dCached::new`] and call [`Lut3dCached::interpolate`]
/// for every pixel.  The per-pixel cost is one table look-up + four multiply-
/// accumulate operations per channel, with no repeated index arithmetic.
///
/// # Example
///
/// ```
/// use oximedia_calibrate::lut::cached::Lut3dCached;
/// use oximedia_calibrate::lut::build_identity_lut;
///
/// let lut_size = 17_usize;
/// let data = build_identity_lut(lut_size);
/// let cache = Lut3dCached::new(data, lut_size);
/// let out = cache.interpolate(0.5, 0.3, 0.7);
/// ```
pub struct Lut3dCached {
    /// The raw flat LUT data (same layout as [`super::tetrahedral_lookup_3d`]).
    pub lut: Vec<f32>,
    /// Number of grid points per axis.
    pub lut_size: usize,
    /// Precomputed coefficients, length `(lut_size - 1)³`.
    pub coeffs: Box<[TetraCoeff]>,
}

impl Lut3dCached {
    /// Build a cached wrapper around a flat 3D LUT.
    ///
    /// # Arguments
    ///
    /// * `lut`      - Flat LUT data in `[R][G][B]` major order (`lut_size³ × 3`
    ///                `f32` values).
    /// * `lut_size` - Number of grid points along each axis (must be ≥ 2).
    ///
    /// # Panics
    ///
    /// Does not panic.  If `lut_size < 2` or `lut` is too short, returns a
    /// degenerate cache that always produces `[0,0,0]`.
    #[must_use]
    pub fn new(lut: Vec<f32>, lut_size: usize) -> Self {
        if lut_size < 2 || lut.len() < lut_size * lut_size * lut_size * 3 {
            return Self {
                lut,
                lut_size,
                coeffs: Box::new([]),
            };
        }

        let n = lut_size;
        let n1 = n - 1;

        // Strides in terms of f32 elements (not cells).
        let step_r = n * n * 3;
        let step_g = n * 3;
        let step_b = 3;

        let mut coeffs = Vec::with_capacity(n1 * n1 * n1);
        for ri in 0..n1 {
            for gi in 0..n1 {
                for bi in 0..n1 {
                    // Base index of the (ri, gi, bi) lattice corner in the flat LUT.
                    let base_idx = (ri * n * n + gi * n + bi) * 3;
                    coeffs.push(TetraCoeff {
                        base_idx,
                        step_r,
                        step_g,
                        step_b,
                    });
                }
            }
        }

        Self {
            lut,
            lut_size,
            coeffs: coeffs.into_boxed_slice(),
        }
    }

    /// Interpolate a colour through the cached LUT using tetrahedral interpolation.
    ///
    /// The output is bit-identical (within f32 rounding) to calling
    /// [`super::tetrahedral_lookup_3d`] with the same data and size.
    ///
    /// Input values outside `[0.0, 1.0]` are clamped.
    ///
    /// # Arguments
    ///
    /// * `r` - Red channel in `[0.0, 1.0]`.
    /// * `g` - Green channel in `[0.0, 1.0]`.
    /// * `b` - Blue channel in `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// `[f32; 3]` interpolated output colour.
    #[must_use]
    pub fn interpolate(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = self.lut_size;
        if n < 2 || self.lut.len() < n * n * n * 3 {
            return [0.0, 0.0, 0.0];
        }

        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        let scale = (n - 1) as f32;
        let rp = r * scale;
        let gp = g * scale;
        let bp = b * scale;

        let ri = (rp as usize).min(n - 2);
        let gi = (gp as usize).min(n - 2);
        let bi = (bp as usize).min(n - 2);

        let dr = rp - ri as f32;
        let dg = gp - gi as f32;
        let db = bp - bi as f32;

        let n1 = n - 1;
        let coeff = &self.coeffs[ri * n1 * n1 + gi * n1 + bi];

        interpolate_from_coeff(&self.lut, coeff, dr, dg, db)
    }
}

// ---------------------------------------------------------------------------
// Core interpolation with precomputed strides
// ---------------------------------------------------------------------------

/// Inner tetrahedral interpolation using precomputed [`TetraCoeff`] strides.
///
/// Reproduces the exact six-case tetrahedron decomposition from
/// [`super::tetrahedral_lookup_3d`].  One "vertex" is always the `(0,0,0)`
/// corner (`v000`) and another is always the `(1,1,1)` corner (`v111`); the
/// two intermediate vertices depend on the dominant-axis ordering.
#[inline]
fn interpolate_from_coeff(lut: &[f32], coeff: &TetraCoeff, dr: f32, dg: f32, db: f32) -> [f32; 3] {
    let base = coeff.base_idx;
    let sr = coeff.step_r;
    let sg = coeff.step_g;
    let sb = coeff.step_b;

    // Helper: fetch one channel from a lattice vertex given its offset from base.
    // `off` is the sum of axis strides for that vertex.
    macro_rules! v {
        ($off:expr, $ch:expr) => {
            lut[base + $off + $ch]
        };
    }

    let mut out = [0.0_f32; 3];

    for ch in 0..3_usize {
        let v000 = v!(0, ch);
        let v111 = v!(sr + sg + sb, ch);

        out[ch] = if dr >= dg && dg >= db {
            // Tetrahedron 1: dr >= dg >= db
            let v100 = v!(sr, ch);
            let v110 = v!(sr + sg, ch);
            (1.0 - dr) * v000 + (dr - dg) * v100 + (dg - db) * v110 + db * v111
        } else if dr >= db && db >= dg {
            // Tetrahedron 2: dr >= db >= dg
            let v100 = v!(sr, ch);
            let v101 = v!(sr + sb, ch);
            (1.0 - dr) * v000 + (dr - db) * v100 + (db - dg) * v101 + dg * v111
        } else if db >= dr && dr >= dg {
            // Tetrahedron 3: db >= dr >= dg
            let v001 = v!(sb, ch);
            let v101 = v!(sr + sb, ch);
            (1.0 - db) * v000 + (db - dr) * v001 + (dr - dg) * v101 + dg * v111
        } else if dg >= dr && dr >= db {
            // Tetrahedron 4: dg >= dr >= db
            let v010 = v!(sg, ch);
            let v110 = v!(sr + sg, ch);
            (1.0 - dg) * v000 + (dg - dr) * v010 + (dr - db) * v110 + db * v111
        } else if dg >= db && db >= dr {
            // Tetrahedron 5: dg >= db >= dr
            let v010 = v!(sg, ch);
            let v011 = v!(sg + sb, ch);
            (1.0 - dg) * v000 + (dg - db) * v010 + (db - dr) * v011 + dr * v111
        } else {
            // Tetrahedron 6: db >= dg >= dr
            let v001 = v!(sb, ch);
            let v011 = v!(sg + sb, ch);
            (1.0 - db) * v000 + (db - dg) * v001 + (dg - dr) * v011 + dr * v111
        };
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut::{build_identity_lut, tetrahedral_lookup_3d};

    const LUT_SIZE: usize = 17;

    // ── helper ───────────────────────────────────────────────────────────────

    fn rgb_close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    // ── 1. Cached vs direct – random grid ───────────────────────────────────

    /// Verify that `Lut3dCached::interpolate` matches `tetrahedral_lookup_3d`
    /// on 10 representative points, within f32 floating-point tolerance.
    #[test]
    fn test_lut3d_cached_matches_direct() {
        let lut = build_identity_lut(LUT_SIZE);
        let cache = Lut3dCached::new(lut.clone(), LUT_SIZE);

        let tol = 4.0 * f32::EPSILON * 255.0;

        let points: &[(f32, f32, f32)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.1, 0.2, 0.3),
            (0.7, 0.5, 0.9),
            (0.33, 0.66, 0.12),
            (0.01, 0.99, 0.5),
            (0.8, 0.15, 0.4),
            (0.45, 0.72, 0.63),
            (0.98, 0.02, 0.87),
        ];

        for &(r, g, b) in points {
            let direct = tetrahedral_lookup_3d(&lut, LUT_SIZE, r, g, b);
            let cached = cache.interpolate(r, g, b);
            assert!(
                rgb_close(direct, cached, tol),
                "Mismatch at ({r},{g},{b}): direct={direct:?} cached={cached:?} tol={tol}",
            );
        }
    }

    // ── 2. Identity LUT maps RGB to self ─────────────────────────────────────

    /// An identity LUT must return the input colour unchanged.
    #[test]
    fn test_identity_lut_maps_rgb_to_self() {
        let lut = build_identity_lut(LUT_SIZE);
        let cache = Lut3dCached::new(lut, LUT_SIZE);

        let inp = (0.3_f32, 0.5_f32, 0.7_f32);
        let out = cache.interpolate(inp.0, inp.1, inp.2);

        let tol = 1e-4_f32;
        assert!(
            (out[0] - inp.0).abs() < tol,
            "R: expected {} got {}",
            inp.0,
            out[0]
        );
        assert!(
            (out[1] - inp.1).abs() < tol,
            "G: expected {} got {}",
            inp.1,
            out[1]
        );
        assert!(
            (out[2] - inp.2).abs() < tol,
            "B: expected {} got {}",
            inp.2,
            out[2]
        );
    }

    // ── 3. Out-of-range inputs clamp and don't panic ─────────────────────────

    /// Inputs outside `[0,1]` must not panic and outputs must stay in `[0,1]`.
    #[test]
    fn test_out_of_range_clamps() {
        let lut = build_identity_lut(LUT_SIZE);
        let cache = Lut3dCached::new(lut, LUT_SIZE);

        let cases: &[(f32, f32, f32)] = &[
            (1.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
            (0.5, 1.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 1.5),
            (0.5, 0.5, -0.5),
            (2.0, -1.0, 3.0),
        ];

        for &(r, g, b) in cases {
            let out = cache.interpolate(r, g, b);
            for (ch, &v) in out.iter().enumerate() {
                assert!(
                    v >= 0.0 && v <= 1.0 + 1e-5,
                    "Channel {ch} out of range for ({r},{g},{b}): {v}"
                );
            }
        }
    }

    // ── 4. Throughput: 10 000 calls complete in < 200 ms ─────────────────────

    /// The cache must handle 10 000 interpolations in well under 200 ms.
    #[test]
    fn test_perf_10k_calls() {
        let lut = build_identity_lut(33);
        let cache = Lut3dCached::new(lut, 33);

        // Generate a varied input sequence.
        let mut inputs = Vec::with_capacity(10_000);
        for i in 0..10_000_usize {
            let t = i as f32 / 9_999.0;
            let r = (t * 3.7).fract();
            let g = (t * 5.3).fract();
            let b = (t * 7.1).fract();
            inputs.push((r, g, b));
        }

        let start = std::time::Instant::now();
        for &(r, g, b) in &inputs {
            let _ = cache.interpolate(r, g, b);
        }
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 200,
            "10 000 interpolations took {}ms (limit 200ms)",
            elapsed.as_millis()
        );
    }
}
