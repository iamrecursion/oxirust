//! Tetrahedral interpolation for 3D LUTs.
//!
//! Provides a high-accuracy [`TetrahedralInterpolator`] that splits each RGB cube cell
//! into six tetrahedra and uses barycentric (Sakamoto) interpolation within the
//! appropriate tetrahedron.  Tetrahedral interpolation is the industry-standard
//! method used by ICC colour management engines and professional LUT processors
//! because it is exact at all eight cube corners and produces smoother results
//! than trilinear interpolation for non-linear LUT data.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::lut_interp::{Lut3d, TetrahedralInterpolator, interpolate_3d};
//!
//! // Build a 2-point identity LUT
//! let lut = Lut3d::identity(2);
//! let interp = TetrahedralInterpolator::new(lut);
//!
//! let out = interp.interpolate(0.5, 0.5, 0.5);
//! assert!((out[0] - 0.5).abs() < 1e-4);
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::error::{ColorError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Lut3d
// ──────────────────────────────────────────────────────────────────────────────

/// A 3-dimensional colour lookup table.
///
/// Data is stored in R-major, G-minor, B-innermost order: the value at lattice
/// point `(r, g, b)` starts at index `(r * size * size + g * size + b) * 3`.
/// Each triplet is `[R_out, G_out, B_out]` as `f32`.
#[derive(Debug, Clone)]
pub struct Lut3d {
    /// Raw LUT data: `size³ × 3` values.
    pub data: Vec<f32>,
    /// Number of grid points along each axis.
    pub size: usize,
}

impl Lut3d {
    /// Creates a new 3D LUT, validating that `data.len() == size³ × 3`.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::Lut`] when the data length is wrong or `size < 2`.
    pub fn new(data: Vec<f32>, size: usize) -> Result<Self> {
        if size < 2 {
            return Err(ColorError::Lut(format!(
                "Lut3d size must be at least 2, got {size}"
            )));
        }
        let expected = size * size * size * 3;
        if data.len() != expected {
            return Err(ColorError::Lut(format!(
                "Lut3d data length mismatch: expected {expected}, got {}",
                data.len()
            )));
        }
        Ok(Self { data, size })
    }

    /// Creates an identity 3D LUT of the given size.
    ///
    /// An identity LUT maps every input RGB to the same output RGB.
    ///
    /// # Panics
    ///
    /// Panics if `size < 2`.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        assert!(size >= 2, "Lut3d identity size must be at least 2");
        let n = size * size * size * 3;
        let mut data = Vec::with_capacity(n);
        let scale = (size - 1) as f32;
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push(r as f32 / scale);
                    data.push(g as f32 / scale);
                    data.push(b as f32 / scale);
                }
            }
        }
        Self { data, size }
    }

    /// Returns the output triplet at lattice point `(ri, gi, bi)`.
    ///
    /// This is a low-level accessor used by the interpolation kernel.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::Lut`] if `ri`, `gi`, or `bi` is out of bounds.
    #[inline(always)]
    fn cell(&self, ri: usize, gi: usize, bi: usize) -> [f32; 3] {
        let idx = (ri * self.size * self.size + gi * self.size + bi) * 3;
        // Safety: callers guarantee ri, gi, bi < self.size, which means
        // idx+2 < size^3*3 == data.len().  The get() calls will never fail
        // for well-formed LUTs constructed via new() or identity().
        let r_val = self.data.get(idx).copied().unwrap_or(0.0);
        let g_val = self.data.get(idx + 1).copied().unwrap_or(0.0);
        let b_val = self.data.get(idx + 2).copied().unwrap_or(0.0);
        [r_val, g_val, b_val]
    }

    /// Safe cell access returning a `Result`.
    pub fn cell_checked(&self, ri: usize, gi: usize, bi: usize) -> Result<[f32; 3]> {
        let idx = (ri * self.size * self.size + gi * self.size + bi) * 3;
        let r_val = *self.data.get(idx).ok_or_else(|| {
            ColorError::Lut(format!("cell index out of bounds: ({ri},{gi},{bi})"))
        })?;
        let g_val = *self.data.get(idx + 1).ok_or_else(|| {
            ColorError::Lut(format!("cell index out of bounds: ({ri},{gi},{bi})"))
        })?;
        let b_val = *self.data.get(idx + 2).ok_or_else(|| {
            ColorError::Lut(format!("cell index out of bounds: ({ri},{gi},{bi})"))
        })?;
        Ok([r_val, g_val, b_val])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Free function: interpolate_3d
// ──────────────────────────────────────────────────────────────────────────────

/// Tetrahedral interpolation into a 3D LUT.
///
/// The RGB cube cell containing `(r, g, b)` is decomposed into six tetrahedra
/// using the Sakamoto ordering.  The point is localised to one tetrahedron by
/// comparing the fractional coordinates, and the output is computed as a
/// weighted sum of four lattice-point values.
///
/// # Arguments
///
/// * `r`, `g`, `b` — Input coordinates in **[0, 1]**.  Values outside this
///   range are clamped silently.
/// * `lut` — Reference to the 3D LUT.
///
/// # Returns
///
/// Interpolated `[R, G, B]` as `f32`.
#[must_use]
pub fn interpolate_3d(r: f32, g: f32, b: f32, lut: &Lut3d) -> [f32; 3] {
    let size = lut.size;

    // Clamp inputs to [0, 1]
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let scale = (size - 1) as f32;

    let pr = r * scale;
    let pg = g * scale;
    let pb = b * scale;

    // Base (floor) lattice indices, clamped so that r0+1 is always valid.
    let r0 = (pr.floor() as usize).min(size - 2);
    let g0 = (pg.floor() as usize).min(size - 2);
    let b0 = (pb.floor() as usize).min(size - 2);

    let r1 = r0 + 1;
    let g1 = g0 + 1;
    let b1 = b0 + 1;

    // Fractional parts inside the cell: rf, gf, bf ∈ [0, 1]
    let rf = pr - r0 as f32;
    let gf = pg - g0 as f32;
    let bf = pb - b0 as f32;

    // Fetch the eight corner values of the unit cube.
    // Only the four needed for the selected tetrahedron are actually used; the
    // rest are fetched lazily inside each branch.
    let c000 = lut.cell(r0, g0, b0);

    // Sakamoto tetrahedral decomposition — 6 tetrahedra.
    // Each branch computes:  result = c000 + α*(cA-c000) + β*(cB-cA) + γ*(c111-cB)
    // where the dominant ordering determines the branch.
    if rf >= gf {
        if gf >= bf {
            // Tetrahedron 1:  rf ≥ gf ≥ bf
            let c100 = lut.cell(r1, g0, b0);
            let c110 = lut.cell(r1, g1, b0);
            let c111 = lut.cell(r1, g1, b1);
            interp4(c000, c100, c110, c111, rf, gf, bf)
        } else if rf >= bf {
            // Tetrahedron 2:  rf ≥ bf > gf
            let c100 = lut.cell(r1, g0, b0);
            let c101 = lut.cell(r1, g0, b1);
            let c111 = lut.cell(r1, g1, b1);
            interp4(c000, c100, c101, c111, rf, bf, gf)
        } else {
            // Tetrahedron 3:  bf > rf ≥ gf
            let c001 = lut.cell(r0, g0, b1);
            let c101 = lut.cell(r1, g0, b1);
            let c111 = lut.cell(r1, g1, b1);
            interp4(c000, c001, c101, c111, bf, rf, gf)
        }
    } else if bf >= gf {
        // Tetrahedron 4:  bf ≥ gf > rf  (i.e., gf > rf and bf ≥ gf)
        let c001 = lut.cell(r0, g0, b1);
        let c011 = lut.cell(r0, g1, b1);
        let c111 = lut.cell(r1, g1, b1);
        interp4(c000, c001, c011, c111, bf, gf, rf)
    } else if gf >= bf {
        // Tetrahedron 5:  gf > bf and gf > rf  →  gf ≥ bf > rf
        // (covers gf > bf > rf and gf ≥ bf when the other conditions not met)
        if bf > rf {
            // gf > bf > rf
            let c010 = lut.cell(r0, g1, b0);
            let c011 = lut.cell(r0, g1, b1);
            let c111 = lut.cell(r1, g1, b1);
            interp4(c000, c010, c011, c111, gf, bf, rf)
        } else {
            // gf > rf ≥ bf  (and gf > rf already known from outer else)
            let c010 = lut.cell(r0, g1, b0);
            let c110 = lut.cell(r1, g1, b0);
            let c111 = lut.cell(r1, g1, b1);
            interp4(c000, c010, c110, c111, gf, rf, bf)
        }
    } else {
        // Tetrahedron 6:  bf > gf > rf
        let c001 = lut.cell(r0, g0, b1);
        let c011 = lut.cell(r0, g1, b1);
        let c111 = lut.cell(r1, g1, b1);
        interp4(c000, c001, c011, c111, bf, gf, rf)
    }
}

/// Sakamoto barycentric interpolation over one tetrahedron defined by four
/// vertices `v0, v1, v2, v3` with weights `w1, w2, w3` (barycentric along the
/// path v0→v1→v2→v3).
///
/// Formula:  result = v0 + w1*(v1-v0) + w2*(v2-v1) + w3*(v3-v2)
#[inline(always)]
fn interp4(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    w1: f32,
    w2: f32,
    w3: f32,
) -> [f32; 3] {
    [
        v0[0] + w1 * (v1[0] - v0[0]) + w2 * (v2[0] - v1[0]) + w3 * (v3[0] - v2[0]),
        v0[1] + w1 * (v1[1] - v0[1]) + w2 * (v2[1] - v1[1]) + w3 * (v3[1] - v2[1]),
        v0[2] + w1 * (v1[2] - v0[2]) + w2 * (v2[2] - v1[2]) + w3 * (v3[2] - v2[2]),
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// TetrahedralInterpolator
// ──────────────────────────────────────────────────────────────────────────────

/// Owns a [`Lut3d`] and provides a convenient `interpolate` method.
///
/// # Example
///
/// ```
/// use oximedia_colormgmt::lut_interp::{Lut3d, TetrahedralInterpolator};
///
/// let lut = Lut3d::identity(17);
/// let interp = TetrahedralInterpolator::new(lut);
/// let out = interp.interpolate(0.25, 0.5, 0.75);
/// assert!((out[0] - 0.25).abs() < 1e-4);
/// assert!((out[1] - 0.5).abs() < 1e-4);
/// assert!((out[2] - 0.75).abs() < 1e-4);
/// ```
#[derive(Debug, Clone)]
pub struct TetrahedralInterpolator {
    lut: Lut3d,
}

impl TetrahedralInterpolator {
    /// Creates a new `TetrahedralInterpolator` that owns the given LUT.
    #[must_use]
    pub fn new(lut: Lut3d) -> Self {
        Self { lut }
    }

    /// Returns a reference to the underlying [`Lut3d`].
    #[must_use]
    pub fn lut(&self) -> &Lut3d {
        &self.lut
    }

    /// Interpolates the LUT at `(r, g, b)`.
    ///
    /// Inputs outside `[0, 1]` are clamped silently.
    #[must_use]
    pub fn interpolate(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        interpolate_3d(r, g, b, &self.lut)
    }

    /// Processes a batch of pixels in place.
    ///
    /// Each pixel is `[R, G, B]` with values in `[0, 1]`.  Out-of-range values
    /// are clamped.
    pub fn apply_batch(&self, pixels: &mut [[f32; 3]]) {
        for px in pixels.iter_mut() {
            let out = self.interpolate(px[0], px[1], px[2]);
            *px = out;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Prismatic interpolation
// ──────────────────────────────────────────────────────────────────────────────

/// Prismatic interpolation for 3D LUTs.
///
/// Decomposes each RGB cube cell into 6 triangular prisms (one per face diagonal)
/// and interpolates within the containing prism.  This method provides a different
/// interpolation topology than tetrahedral, which can be preferable for LUTs with
/// strong diagonal features.
#[derive(Debug, Clone)]
pub struct PrismaticInterpolator {
    lut: Lut3d,
}

impl PrismaticInterpolator {
    /// Creates a new prismatic interpolator.
    #[must_use]
    pub fn new(lut: Lut3d) -> Self {
        Self { lut }
    }

    /// Returns a reference to the underlying LUT.
    #[must_use]
    pub fn lut(&self) -> &Lut3d {
        &self.lut
    }

    /// Interpolate at `(r, g, b)` using prismatic decomposition.
    ///
    /// The cube is split into 6 prisms by partitioning the RG face into two
    /// triangles (above and below the diagonal) and sweeping along the B axis.
    /// Within each prism, we do a bilinear-in-triangle + linear-along-B blend.
    #[must_use]
    pub fn interpolate(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        interpolate_prismatic(r, g, b, &self.lut)
    }

    /// Processes a batch of pixels in place.
    pub fn apply_batch(&self, pixels: &mut [[f32; 3]]) {
        for px in pixels.iter_mut() {
            *px = self.interpolate(px[0], px[1], px[2]);
        }
    }
}

/// Prismatic interpolation into a 3D LUT.
///
/// Decomposes each cube cell into 6 triangular prisms along the three
/// axis-pair diagonals and blends linearly along the remaining axis.
#[must_use]
pub fn interpolate_prismatic(r: f32, g: f32, b: f32, lut: &Lut3d) -> [f32; 3] {
    let size = lut.size;
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    let scale = (size - 1) as f32;
    let pr = r * scale;
    let pg = g * scale;
    let pb = b * scale;

    let r0 = (pr.floor() as usize).min(size - 2);
    let g0 = (pg.floor() as usize).min(size - 2);
    let b0 = (pb.floor() as usize).min(size - 2);
    let r1 = r0 + 1;
    let g1 = g0 + 1;
    let b1 = b0 + 1;

    let rf = pr - r0 as f32;
    let gf = pg - g0 as f32;
    let bf = pb - b0 as f32;

    // Prismatic: split RG face into 2 triangles, sweep along B.
    // Triangle 1: rf >= gf  (lower-right triangle)
    // Triangle 2: gf > rf   (upper-left triangle)
    // Blend along B axis linearly between b0 and b1 planes.

    let interp_rg_plane = |bi: usize| -> [f32; 3] {
        if rf >= gf {
            // Triangle: (r0,g0) (r1,g0) (r1,g1)
            let c00 = lut.cell(r0, g0, bi);
            let c10 = lut.cell(r1, g0, bi);
            let c11 = lut.cell(r1, g1, bi);
            // Barycentric in this triangle
            let w_c11 = gf;
            let w_c10 = rf - gf;
            let w_c00 = 1.0 - rf;
            [
                w_c00 * c00[0] + w_c10 * c10[0] + w_c11 * c11[0],
                w_c00 * c00[1] + w_c10 * c10[1] + w_c11 * c11[1],
                w_c00 * c00[2] + w_c10 * c10[2] + w_c11 * c11[2],
            ]
        } else {
            // Triangle: (r0,g0) (r0,g1) (r1,g1)
            let c00 = lut.cell(r0, g0, bi);
            let c01 = lut.cell(r0, g1, bi);
            let c11 = lut.cell(r1, g1, bi);
            let w_c11 = rf;
            let w_c01 = gf - rf;
            let w_c00 = 1.0 - gf;
            [
                w_c00 * c00[0] + w_c01 * c01[0] + w_c11 * c11[0],
                w_c00 * c00[1] + w_c01 * c01[1] + w_c11 * c11[1],
                w_c00 * c00[2] + w_c01 * c01[2] + w_c11 * c11[2],
            ]
        }
    };

    let plane0 = interp_rg_plane(b0);
    let plane1 = interp_rg_plane(b1);

    // Linear blend along B
    [
        plane0[0] + bf * (plane1[0] - plane0[0]),
        plane0[1] + bf * (plane1[1] - plane0[1]),
        plane0[2] + bf * (plane1[2] - plane0[2]),
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// LUT size conversion
// ──────────────────────────────────────────────────────────────────────────────

/// Converts a 3D LUT to a different grid size using tetrahedral interpolation.
///
/// For example, converts a 17³ LUT to a 33³ or 65³ LUT.  The new LUT is created
/// by sampling the original at evenly-spaced positions using tetrahedral
/// interpolation.
///
/// # Errors
///
/// Returns [`ColorError::Lut`] if `new_size < 2`.
pub fn lut_resize(lut: &Lut3d, new_size: usize) -> Result<Lut3d> {
    if new_size < 2 {
        return Err(ColorError::Lut(format!(
            "New LUT size must be at least 2, got {new_size}"
        )));
    }
    let n = new_size * new_size * new_size * 3;
    let mut data = Vec::with_capacity(n);
    let scale = (new_size - 1) as f32;

    for ri in 0..new_size {
        let r = ri as f32 / scale;
        for gi in 0..new_size {
            let g = gi as f32 / scale;
            for bi in 0..new_size {
                let b = bi as f32 / scale;
                let out = interpolate_3d(r, g, b, lut);
                data.push(out[0]);
                data.push(out[1]);
                data.push(out[2]);
            }
        }
    }

    Lut3d::new(data, new_size)
}

// ──────────────────────────────────────────────────────────────────────────────
// LUT composition
// ──────────────────────────────────────────────────────────────────────────────

/// Composes two 3D LUTs into a single LUT: `result(x) = lut_b(lut_a(x))`.
///
/// The output LUT has `output_size` grid points per axis.  Each lattice point
/// is evaluated by first passing through `lut_a`, then feeding the result into
/// `lut_b`.
///
/// # Errors
///
/// Returns [`ColorError::Lut`] if `output_size < 2`.
pub fn lut_compose(lut_a: &Lut3d, lut_b: &Lut3d, output_size: usize) -> Result<Lut3d> {
    if output_size < 2 {
        return Err(ColorError::Lut(format!(
            "Output LUT size must be at least 2, got {output_size}"
        )));
    }
    let n = output_size * output_size * output_size * 3;
    let mut data = Vec::with_capacity(n);
    let scale = (output_size - 1) as f32;

    for ri in 0..output_size {
        let r = ri as f32 / scale;
        for gi in 0..output_size {
            let g = gi as f32 / scale;
            for bi in 0..output_size {
                let b = bi as f32 / scale;
                // First pass through lut_a
                let mid = interpolate_3d(r, g, b, lut_a);
                // Then through lut_b
                let out = interpolate_3d(mid[0], mid[1], mid[2], lut_b);
                data.push(out[0]);
                data.push(out[1]);
                data.push(out[2]);
            }
        }
    }

    Lut3d::new(data, output_size)
}

// ──────────────────────────────────────────────────────────────────────────────
// LUT inversion (Newton's method)
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for iterative LUT inversion.
#[derive(Debug, Clone)]
pub struct LutInversionConfig {
    /// Grid size of the output inverse LUT.
    pub output_size: usize,
    /// Maximum Newton iterations per point.
    pub max_iterations: usize,
    /// Convergence tolerance (Euclidean distance).
    pub tolerance: f32,
    /// Step size for finite-difference Jacobian estimation.
    pub jacobian_step: f32,
}

impl Default for LutInversionConfig {
    fn default() -> Self {
        Self {
            output_size: 33,
            max_iterations: 20,
            tolerance: 1e-5,
            jacobian_step: 1e-4,
        }
    }
}

/// Computes an approximate inverse of a 3D LUT using iterative Newton's method.
///
/// For each lattice point in the output grid, we seek an input `x` such that
/// `lut(x) ≈ target` using Newton-Raphson with a finite-difference Jacobian.
///
/// # Errors
///
/// Returns [`ColorError::Lut`] if the config output size is < 2.
pub fn lut_invert(lut: &Lut3d, config: &LutInversionConfig) -> Result<Lut3d> {
    let out_size = config.output_size;
    if out_size < 2 {
        return Err(ColorError::Lut(format!(
            "Inverse LUT size must be at least 2, got {out_size}"
        )));
    }

    let n = out_size * out_size * out_size * 3;
    let mut data = Vec::with_capacity(n);
    let scale = (out_size - 1) as f32;
    let h = config.jacobian_step;

    for ri in 0..out_size {
        let target_r = ri as f32 / scale;
        for gi in 0..out_size {
            let target_g = gi as f32 / scale;
            for bi in 0..out_size {
                let target_b = bi as f32 / scale;
                let target = [target_r, target_g, target_b];

                // Initial guess: the target itself (works well for near-identity LUTs)
                let mut x = target;

                for _ in 0..config.max_iterations {
                    let fx = interpolate_3d(x[0], x[1], x[2], lut);
                    let err = [fx[0] - target[0], fx[1] - target[1], fx[2] - target[2]];

                    let err_norm = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]).sqrt();
                    if err_norm < config.tolerance {
                        break;
                    }

                    // Finite-difference Jacobian (3×3)
                    let j = compute_jacobian(lut, x, h);

                    // Solve J * delta = -err using Cramer's rule
                    if let Some(delta) = solve_3x3(j, [-err[0], -err[1], -err[2]]) {
                        x[0] = (x[0] + delta[0]).clamp(0.0, 1.0);
                        x[1] = (x[1] + delta[1]).clamp(0.0, 1.0);
                        x[2] = (x[2] + delta[2]).clamp(0.0, 1.0);
                    } else {
                        // Singular Jacobian — stop iterating
                        break;
                    }
                }

                data.push(x[0]);
                data.push(x[1]);
                data.push(x[2]);
            }
        }
    }

    Lut3d::new(data, out_size)
}

/// Compute the 3×3 Jacobian of the LUT at point `x` using central differences.
fn compute_jacobian(lut: &Lut3d, x: [f32; 3], h: f32) -> [[f32; 3]; 3] {
    let mut j = [[0.0f32; 3]; 3];
    for col in 0..3 {
        let mut xp = x;
        let mut xm = x;
        xp[col] = (x[col] + h).clamp(0.0, 1.0);
        xm[col] = (x[col] - h).clamp(0.0, 1.0);
        let fp = interpolate_3d(xp[0], xp[1], xp[2], lut);
        let fm = interpolate_3d(xm[0], xm[1], xm[2], lut);
        let denom = xp[col] - xm[col];
        if denom.abs() > 1e-10 {
            for row in 0..3 {
                j[row][col] = (fp[row] - fm[row]) / denom;
            }
        }
    }
    j
}

/// Solve a 3×3 linear system `A * x = b` using Cramer's rule.
///
/// Returns `None` if the determinant is too small (singular matrix).
fn solve_3x3(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;

    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv_det;

    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv_det;

    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;

    Some([x0, x1, x2])
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn assert_close3(a: [f32; 3], b: [f32; 3], tol: f32, msg: &str) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "{msg}: channel {i}: got {}, expected {} (tol {tol})",
                a[i],
                b[i]
            );
        }
    }

    fn identity_lut(size: usize) -> Lut3d {
        Lut3d::identity(size)
    }

    // ── Lut3d construction ─────────────────────────────────────────────────────

    #[test]
    fn test_lut3d_new_valid() {
        let data = vec![0.0f32; 2 * 2 * 2 * 3]; // 24 floats for size=2
        let lut = Lut3d::new(data, 2).expect("Lut3d::new with correct size should succeed");
        assert_eq!(lut.size, 2);
    }

    #[test]
    fn test_lut3d_new_wrong_length_returns_error() {
        let data = vec![0.0f32; 10]; // wrong length
        let result = Lut3d::new(data, 2);
        assert!(
            result.is_err(),
            "Lut3d::new with wrong data length should return Err"
        );
    }

    #[test]
    fn test_lut3d_new_size_too_small_returns_error() {
        let data = vec![0.0f32; 3];
        let result = Lut3d::new(data, 1);
        assert!(
            result.is_err(),
            "Lut3d::new with size < 2 should return Err"
        );
    }

    // ── Identity corners ───────────────────────────────────────────────────────

    #[test]
    fn test_identity_lut2_corner_black() {
        let lut = identity_lut(2);
        let out = interpolate_3d(0.0, 0.0, 0.0, &lut);
        assert_close3(out, [0.0, 0.0, 0.0], 1e-5, "corner black");
    }

    #[test]
    fn test_identity_lut2_corner_white() {
        let lut = identity_lut(2);
        let out = interpolate_3d(1.0, 1.0, 1.0, &lut);
        assert_close3(out, [1.0, 1.0, 1.0], 1e-5, "corner white");
    }

    #[test]
    fn test_identity_lut2_corner_red() {
        let lut = identity_lut(2);
        let out = interpolate_3d(1.0, 0.0, 0.0, &lut);
        assert_close3(out, [1.0, 0.0, 0.0], 1e-5, "corner red");
    }

    #[test]
    fn test_identity_lut2_corner_green() {
        let lut = identity_lut(2);
        let out = interpolate_3d(0.0, 1.0, 0.0, &lut);
        assert_close3(out, [0.0, 1.0, 0.0], 1e-5, "corner green");
    }

    #[test]
    fn test_identity_lut2_corner_blue() {
        let lut = identity_lut(2);
        let out = interpolate_3d(0.0, 0.0, 1.0, &lut);
        assert_close3(out, [0.0, 0.0, 1.0], 1e-5, "corner blue");
    }

    #[test]
    fn test_identity_lut2_midpoint() {
        let lut = identity_lut(2);
        let out = interpolate_3d(0.5, 0.5, 0.5, &lut);
        assert_close3(out, [0.5, 0.5, 0.5], 1e-4, "midpoint");
    }

    // ── Identity with larger LUT ───────────────────────────────────────────────

    #[test]
    fn test_identity_lut17_various_points() {
        let lut = identity_lut(17);
        let interp = TetrahedralInterpolator::new(lut);

        let test_cases: &[(f32, f32, f32)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.25, 0.5, 0.75),
            (0.1, 0.3, 0.9),
            (0.6, 0.2, 0.8),
            (0.333, 0.666, 0.999),
        ];
        for &(r, g, b) in test_cases {
            let out = interp.interpolate(r, g, b);
            assert_close3(
                out,
                [r, g, b],
                1e-4,
                &format!("identity lut17 at ({r}, {g}, {b})"),
            );
        }
    }

    // ── All 6 tetrahedra exercised ─────────────────────────────────────────────
    // By construction the 6 branches are selected by ordering of rf, gf, bf.
    // We use a size-2 identity LUT so the expected output equals the input.

    #[test]
    fn test_tetra1_rf_ge_gf_ge_bf() {
        // rf=0.6 >= gf=0.4 >= bf=0.2
        let lut = identity_lut(2);
        let out = interpolate_3d(0.6, 0.4, 0.2, &lut);
        assert_close3(out, [0.6, 0.4, 0.2], 1e-4, "tetra1");
    }

    #[test]
    fn test_tetra2_rf_ge_bf_gt_gf() {
        // rf=0.7 >= bf=0.5 > gf=0.1
        let lut = identity_lut(2);
        let out = interpolate_3d(0.7, 0.1, 0.5, &lut);
        assert_close3(out, [0.7, 0.1, 0.5], 1e-4, "tetra2");
    }

    #[test]
    fn test_tetra3_bf_gt_rf_ge_gf() {
        // bf=0.9 > rf=0.5 >= gf=0.3
        let lut = identity_lut(2);
        let out = interpolate_3d(0.5, 0.3, 0.9, &lut);
        assert_close3(out, [0.5, 0.3, 0.9], 1e-4, "tetra3");
    }

    #[test]
    fn test_tetra4_bf_ge_gf_gt_rf() {
        // bf=0.8 >= gf=0.6 > rf=0.2
        let lut = identity_lut(2);
        let out = interpolate_3d(0.2, 0.6, 0.8, &lut);
        assert_close3(out, [0.2, 0.6, 0.8], 1e-4, "tetra4");
    }

    #[test]
    fn test_tetra5_gf_gt_bf_gt_rf() {
        // gf=0.8 > bf=0.5 > rf=0.1
        let lut = identity_lut(2);
        let out = interpolate_3d(0.1, 0.8, 0.5, &lut);
        assert_close3(out, [0.1, 0.8, 0.5], 1e-4, "tetra5");
    }

    #[test]
    fn test_tetra6_gf_gt_rf_ge_bf() {
        // gf=0.9 > rf=0.4 >= bf=0.1
        let lut = identity_lut(2);
        let out = interpolate_3d(0.4, 0.9, 0.1, &lut);
        assert_close3(out, [0.4, 0.9, 0.1], 1e-4, "tetra6");
    }

    // ── Out-of-range clamping ─────────────────────────────────────────────────

    #[test]
    fn test_clamping_high() {
        let lut = identity_lut(2);
        let out_clamped = interpolate_3d(1.5, 1.5, 1.5, &lut);
        let out_one = interpolate_3d(1.0, 1.0, 1.0, &lut);
        assert_close3(out_clamped, out_one, 1e-6, "clamped high == 1.0");
    }

    #[test]
    fn test_clamping_low() {
        let lut = identity_lut(2);
        let out_clamped = interpolate_3d(-0.5, -0.1, -1.0, &lut);
        let out_zero = interpolate_3d(0.0, 0.0, 0.0, &lut);
        assert_close3(out_clamped, out_zero, 1e-6, "clamped low == 0.0");
    }

    // ── TetrahedralInterpolator API ───────────────────────────────────────────

    #[test]
    fn test_interpolator_new_and_interpolate() {
        let lut = identity_lut(17);
        let interp = TetrahedralInterpolator::new(lut);
        let out = interp.interpolate(0.3, 0.6, 0.9);
        assert_close3(out, [0.3, 0.6, 0.9], 1e-4, "interpolator.interpolate");
    }

    #[test]
    fn test_interpolator_lut_accessor() {
        let lut = identity_lut(5);
        let interp = TetrahedralInterpolator::new(lut);
        assert_eq!(interp.lut().size, 5);
    }

    #[test]
    fn test_interpolator_apply_batch() {
        let lut = identity_lut(17);
        let interp = TetrahedralInterpolator::new(lut);
        let mut pixels: Vec<[f32; 3]> = vec![
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let original = pixels.clone();
        interp.apply_batch(&mut pixels);
        for (i, (out, orig)) in pixels.iter().zip(original.iter()).enumerate() {
            assert_close3(*out, *orig, 1e-4, &format!("batch pixel {i}"));
        }
    }

    // ── Batch of 10 diverse colours through identity LUT ─────────────────────

    #[test]
    fn test_batch_diverse_colors_identity() {
        let lut = identity_lut(33);
        let test_cases: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.1, 0.9, 0.5],
            [0.9, 0.1, 0.5],
            [0.5, 0.1, 0.9],
            [0.25, 0.75, 0.0],
            [0.333, 0.333, 0.333],
            [0.8, 0.4, 0.2],
            [0.15, 0.55, 0.85],
        ];
        for &[r, g, b] in test_cases {
            let out = interpolate_3d(r, g, b, &lut);
            assert_close3(
                out,
                [r, g, b],
                1e-4,
                &format!("diverse color ({r}, {g}, {b})"),
            );
        }
    }

    // ── Accuracy: tetrahedral is exact at cube corners ─────────────────────────

    #[test]
    fn test_exact_at_corners_non_linear_lut() {
        // Build a simple non-linear LUT (gamma-like) for size=2
        // Corner values: 0→0, 1→1 so all 8 corners are exact identity
        // We verify that all 8 corners are reproduced exactly.
        let size: usize = 2;
        let mut data = vec![0.0f32; size * size * size * 3];
        let scale = (size - 1) as f32;
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let idx = (r * size * size + g * size + b) * 3;
                    // Apply a non-linear transform: square root
                    data[idx] = (r as f32 / scale).sqrt();
                    data[idx + 1] = (g as f32 / scale).sqrt();
                    data[idx + 2] = (b as f32 / scale).sqrt();
                }
            }
        }
        let lut = Lut3d::new(data, size).expect("non-linear lut construction should succeed");

        // All 8 corners must be exact
        let corners: &[(f32, f32, f32, [f32; 3])] = &[
            (0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
            (1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
            (0.0, 1.0, 0.0, [0.0, 1.0, 0.0]),
            (0.0, 0.0, 1.0, [0.0, 0.0, 1.0]),
            (1.0, 1.0, 0.0, [1.0, 1.0, 0.0]),
            (1.0, 0.0, 1.0, [1.0, 0.0, 1.0]),
            (0.0, 1.0, 1.0, [0.0, 1.0, 1.0]),
            (1.0, 1.0, 1.0, [1.0, 1.0, 1.0]),
        ];
        for &(r, g, b, expected) in corners {
            let out = interpolate_3d(r, g, b, &lut);
            assert_close3(
                out,
                expected,
                1e-5,
                &format!("non-linear lut corner ({r},{g},{b})"),
            );
        }
    }

    // ── Prismatic interpolation tests ────────────────────────────────────────

    #[test]
    fn test_prismatic_identity_corners() {
        let lut = identity_lut(2);
        let interp = PrismaticInterpolator::new(lut);
        assert_close3(
            interp.interpolate(0.0, 0.0, 0.0),
            [0.0, 0.0, 0.0],
            1e-5,
            "prismatic black",
        );
        assert_close3(
            interp.interpolate(1.0, 1.0, 1.0),
            [1.0, 1.0, 1.0],
            1e-5,
            "prismatic white",
        );
        assert_close3(
            interp.interpolate(1.0, 0.0, 0.0),
            [1.0, 0.0, 0.0],
            1e-5,
            "prismatic red",
        );
    }

    #[test]
    fn test_prismatic_identity_midpoint() {
        let lut = identity_lut(17);
        let interp = PrismaticInterpolator::new(lut);
        let out = interp.interpolate(0.5, 0.5, 0.5);
        assert_close3(out, [0.5, 0.5, 0.5], 1e-4, "prismatic midpoint");
    }

    #[test]
    fn test_prismatic_identity_various() {
        let lut = identity_lut(33);
        let interp = PrismaticInterpolator::new(lut);
        for &(r, g, b) in &[(0.1, 0.9, 0.5), (0.7, 0.2, 0.8), (0.3, 0.6, 0.1)] {
            let out = interp.interpolate(r, g, b);
            assert_close3(out, [r, g, b], 1e-3, &format!("prismatic ({r},{g},{b})"));
        }
    }

    #[test]
    fn test_prismatic_batch() {
        let lut = identity_lut(17);
        let interp = PrismaticInterpolator::new(lut);
        let mut pixels = vec![[0.2, 0.4, 0.6], [0.8, 0.1, 0.5]];
        let orig = pixels.clone();
        interp.apply_batch(&mut pixels);
        for (i, (out, expected)) in pixels.iter().zip(orig.iter()).enumerate() {
            assert_close3(*out, *expected, 1e-3, &format!("prismatic batch {i}"));
        }
    }

    #[test]
    fn test_prismatic_clamping() {
        let lut = identity_lut(2);
        let interp = PrismaticInterpolator::new(lut);
        let out = interp.interpolate(1.5, -0.5, 2.0);
        assert_close3(out, [1.0, 0.0, 1.0], 1e-5, "prismatic clamping");
    }

    #[test]
    fn test_prismatic_lut_accessor() {
        let lut = identity_lut(5);
        let interp = PrismaticInterpolator::new(lut);
        assert_eq!(interp.lut().size, 5);
    }

    // ── LUT size conversion tests ────────────────────────────────────────────

    #[test]
    fn test_lut_resize_identity_17_to_33() {
        let lut17 = identity_lut(17);
        let lut33 = lut_resize(&lut17, 33).expect("resize should succeed");
        assert_eq!(lut33.size, 33);
        // Sample some points to verify identity is preserved
        for &(r, g, b) in &[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.3, 0.7, 0.1),
        ] {
            let out = interpolate_3d(r, g, b, &lut33);
            assert_close3(
                out,
                [r, g, b],
                1e-3,
                &format!("resize identity ({r},{g},{b})"),
            );
        }
    }

    #[test]
    fn test_lut_resize_identity_33_to_65() {
        let lut33 = identity_lut(33);
        let lut65 = lut_resize(&lut33, 65).expect("resize should succeed");
        assert_eq!(lut65.size, 65);
        let out = interpolate_3d(0.25, 0.75, 0.5, &lut65);
        assert_close3(out, [0.25, 0.75, 0.5], 1e-3, "resize 33->65");
    }

    #[test]
    fn test_lut_resize_error_size_too_small() {
        let lut = identity_lut(2);
        assert!(lut_resize(&lut, 1).is_err());
    }

    // ── LUT composition tests ────────────────────────────────────────────────

    #[test]
    fn test_lut_compose_identity_identity() {
        let id_a = identity_lut(17);
        let id_b = identity_lut(17);
        let composed = lut_compose(&id_a, &id_b, 17).expect("compose should succeed");
        assert_eq!(composed.size, 17);
        let out = interpolate_3d(0.4, 0.5, 0.6, &composed);
        assert_close3(out, [0.4, 0.5, 0.6], 1e-3, "compose identity+identity");
    }

    #[test]
    fn test_lut_compose_with_non_identity() {
        // Build a simple "halving" LUT (output = input * 0.5) for size=2
        let size = 2;
        let scale = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size * 3);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push(r as f32 / scale * 0.5);
                    data.push(g as f32 / scale * 0.5);
                    data.push(b as f32 / scale * 0.5);
                }
            }
        }
        let half_lut = Lut3d::new(data, size).expect("half lut construction");
        let id_lut = identity_lut(17);
        // compose(half, identity) should be half
        let composed = lut_compose(&half_lut, &id_lut, 17).expect("compose should succeed");
        let out = interpolate_3d(0.8, 0.6, 0.4, &composed);
        assert_close3(out, [0.4, 0.3, 0.2], 3e-2, "compose half+id");
    }

    #[test]
    fn test_lut_compose_error_size_too_small() {
        let id = identity_lut(2);
        assert!(lut_compose(&id, &id, 1).is_err());
    }

    // ── LUT inversion tests ─────────────────────────────────────────────────

    #[test]
    fn test_lut_invert_identity() {
        let id_lut = identity_lut(17);
        let config = LutInversionConfig {
            output_size: 17,
            max_iterations: 20,
            tolerance: 1e-5,
            jacobian_step: 1e-4,
        };
        let inv = lut_invert(&id_lut, &config).expect("invert should succeed");
        // Inverse of identity should be identity
        let out = interpolate_3d(0.5, 0.5, 0.5, &inv);
        assert_close3(out, [0.5, 0.5, 0.5], 1e-3, "inverse of identity");
    }

    #[test]
    fn test_lut_invert_roundtrip() {
        // Build a mild non-linear LUT (gamma 0.5 = sqrt)
        let size = 17;
        let mut data = Vec::with_capacity(size * size * size * 3);
        let scale = (size - 1) as f32;
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push((r as f32 / scale).sqrt());
                    data.push((g as f32 / scale).sqrt());
                    data.push((b as f32 / scale).sqrt());
                }
            }
        }
        let gamma_lut = Lut3d::new(data, size).expect("gamma lut construction");
        let config = LutInversionConfig {
            output_size: 33,
            max_iterations: 30,
            tolerance: 1e-6,
            jacobian_step: 1e-4,
        };
        let inv = lut_invert(&gamma_lut, &config).expect("invert should succeed");
        // Apply gamma_lut then inverse: should get back roughly to input
        let input = [0.5, 0.3, 0.8];
        let mid = interpolate_3d(input[0], input[1], input[2], &gamma_lut);
        let roundtrip = interpolate_3d(mid[0], mid[1], mid[2], &inv);
        assert_close3(roundtrip, input, 5e-2, "inversion roundtrip");
    }

    #[test]
    fn test_lut_invert_error_size_too_small() {
        let id = identity_lut(2);
        let config = LutInversionConfig {
            output_size: 1,
            ..LutInversionConfig::default()
        };
        assert!(lut_invert(&id, &config).is_err());
    }

    #[test]
    fn test_lut_invert_default_config() {
        let config = LutInversionConfig::default();
        assert_eq!(config.output_size, 33);
        assert_eq!(config.max_iterations, 20);
        assert!(config.tolerance > 0.0);
        assert!(config.jacobian_step > 0.0);
    }

    // ── solve_3x3 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_solve_3x3_identity_matrix() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 3.0];
        let x = solve_3x3(a, b).expect("should solve identity system");
        assert_close3(x, b, 1e-6, "solve 3x3 identity");
    }

    #[test]
    fn test_solve_3x3_singular_returns_none() {
        let a = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 3.0];
        assert!(
            solve_3x3(a, b).is_none(),
            "singular matrix should return None"
        );
    }

    #[test]
    fn test_cell_checked_valid() {
        let lut = identity_lut(3);
        let cell = lut.cell_checked(0, 0, 0);
        assert!(cell.is_ok());
        assert_close3(
            cell.expect("should be ok"),
            [0.0, 0.0, 0.0],
            1e-6,
            "cell_checked (0,0,0)",
        );
    }
}
