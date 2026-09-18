//! Pre-computed lookup tables for fisheye-to-equirect mapping.
//!
//! Building a lookup table (LUT) amortises the cost of the trigonometric
//! functions (`sin`, `cos`, `atan2`, `asin`) over many frames: each entry in
//! the LUT stores the pre-computed source UV for one output pixel, so
//! converting a frame becomes a simple indexed bilinear-sample loop with no
//! trigonometry at all.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::fisheye_lut::FisheyeLut;
//! use oximedia_360::fisheye::FisheyeParams;
//!
//! // Build a LUT for a 128×64 output equirectangular image
//! let params = FisheyeParams::equidistant(180.0);
//! let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
//!
//! // Apply to a fisheye source image
//! let src = vec![128u8; 64 * 64 * 3];
//! let out = lut.apply_u8(&src, 64, 64).expect("apply");
//! assert_eq!(out.len(), 128 * 64 * 3);
//! ```

use crate::{
    fisheye::FisheyeParams,
    projection::{bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, UvCoord},
    VrError,
};

// ─── LUT entry ───────────────────────────────────────────────────────────────

/// One entry in a fisheye-to-equirect lookup table.
///
/// Stores the normalised source UV coordinate and whether the output pixel is
/// inside the fisheye circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisheyeLutEntry {
    /// Normalised source U coordinate in `[0, 1]`.  Valid only when `valid`.
    pub src_u: f32,
    /// Normalised source V coordinate in `[0, 1]`.  Valid only when `valid`.
    pub src_v: f32,
    /// `true` if the output pixel falls inside the fisheye circle; `false` for
    /// pixels outside the lens coverage (these become black in the output).
    pub valid: bool,
}

// ─── LUT ─────────────────────────────────────────────────────────────────────

/// Pre-computed mapping from every output pixel of an equirectangular image
/// back to a source UV coordinate in a fisheye image.
///
/// Build once per unique `(params, src_dims, out_dims)` combination and reuse
/// across frames.
#[derive(Debug, Clone)]
pub struct FisheyeLut {
    /// Output image width in pixels.
    pub out_width: u32,
    /// Output image height in pixels.
    pub out_height: u32,
    /// Row-major table: one entry per output pixel.
    entries: Vec<FisheyeLutEntry>,
}

impl FisheyeLut {
    /// Build a lookup table for the given fisheye parameters and image sizes.
    ///
    /// * `params`     — fisheye lens parameters (model, FOV, centre, radius)
    /// * `src_width`  — source fisheye image width in pixels
    /// * `src_height` — source fisheye image height in pixels
    /// * `out_width`  — output equirectangular width in pixels
    /// * `out_height` — output equirectangular height in pixels
    ///
    /// This is a one-time O(out_width × out_height) operation that computes
    /// and stores all `sin`/`cos`/`atan2` evaluations needed for the mapping.
    pub fn build(
        params: &FisheyeParams,
        src_width: u32,
        src_height: u32,
        out_width: u32,
        out_height: u32,
    ) -> Self {
        let n = (out_width * out_height) as usize;
        let mut entries = Vec::with_capacity(n);

        let half_min_dim = (src_width.min(src_height) as f32) * 0.5;
        let circle_r_px = params.radius * half_min_dim;
        let cx_px = params.center_x * src_width as f32;
        let cy_px = params.center_y * src_height as f32;

        for oy in 0..out_height {
            for ox in 0..out_width {
                let u = (ox as f32 + 0.5) / out_width as f32;
                let v = (oy as f32 + 0.5) / out_height as f32;

                let sphere = equirect_to_sphere(&UvCoord { u, v });
                let theta = std::f32::consts::FRAC_PI_2 - sphere.elevation_rad;
                let phi = sphere.azimuth_rad;

                if let Some(r_norm) = params.theta_to_r(theta) {
                    let r_px = r_norm * circle_r_px;
                    let fx = cx_px + r_px * phi.sin();
                    let fy = cy_px - r_px * phi.cos();
                    let fu = fx / src_width as f32;
                    let fv = fy / src_height as f32;
                    if fu >= 0.0 && fu <= 1.0 && fv >= 0.0 && fv <= 1.0 {
                        entries.push(FisheyeLutEntry {
                            src_u: fu,
                            src_v: fv,
                            valid: true,
                        });
                    } else {
                        entries.push(FisheyeLutEntry {
                            src_u: 0.0,
                            src_v: 0.0,
                            valid: false,
                        });
                    }
                } else {
                    entries.push(FisheyeLutEntry {
                        src_u: 0.0,
                        src_v: 0.0,
                        valid: false,
                    });
                }
            }
        }

        Self {
            out_width,
            out_height,
            entries,
        }
    }

    /// Number of entries in the LUT (`out_width × out_height`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the LUT has no entries (zero-size output).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Access a specific LUT entry by output pixel index (row-major).
    ///
    /// Returns `None` if `index` is out of range.
    pub fn entry(&self, index: usize) -> Option<&FisheyeLutEntry> {
        self.entries.get(index)
    }

    /// Apply this LUT to produce an equirectangular image from a fisheye source.
    ///
    /// * `src`        — source fisheye pixel data (RGB, 3 bpp, row-major)
    /// * `src_width`  — source image width in pixels
    /// * `src_height` — source image height in pixels
    ///
    /// Returns an RGB equirectangular image of size `out_width × out_height`.
    /// Pixels outside the fisheye circle are black.
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if `src_width` or `src_height` is
    /// zero.
    /// Returns [`VrError::BufferTooSmall`] if `src` is smaller than expected.
    pub fn apply_u8(
        &self,
        src: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> Result<Vec<u8>, VrError> {
        if src_width == 0 || src_height == 0 {
            return Err(VrError::InvalidDimensions(
                "src_width and src_height must be > 0".into(),
            ));
        }
        let expected = src_width as usize * src_height as usize * 3;
        if src.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: src.len(),
            });
        }

        const CH: u32 = 3;
        let out_n = self.out_width * self.out_height;
        let mut out = vec![0u8; out_n as usize * CH as usize];

        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.valid {
                let sample =
                    bilinear_sample_u8(src, src_width, src_height, entry.src_u, entry.src_v, CH);
                let dst = idx * CH as usize;
                out[dst..dst + CH as usize].copy_from_slice(&sample);
            }
            // invalid entries remain black (already zero-initialised)
        }

        Ok(out)
    }

    /// Apply this LUT using a pre-built equirectangular source inverse mapping.
    ///
    /// Unlike `apply_u8`, which uses the LUT to look up the fisheye source,
    /// this variant applies the equirectangular-to-sphere mapping stored in the
    /// LUT to sample from an **equirectangular** source image into a virtual
    /// viewport, enabling fast re-projection of already-stitched frames.
    ///
    /// The LUT must have been built with the same `out_width × out_height` as
    /// the desired output size; `src` must be a valid equirectangular image of
    /// `src_width × src_height` pixels.
    ///
    /// # Errors
    /// Same as `apply_u8`.
    pub fn apply_equirect_u8(
        &self,
        src: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> Result<Vec<u8>, VrError> {
        if src_width == 0 || src_height == 0 {
            return Err(VrError::InvalidDimensions(
                "src_width and src_height must be > 0".into(),
            ));
        }
        let expected = src_width as usize * src_height as usize * 3;
        if src.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: src.len(),
            });
        }

        // For this variant we use the stored src_u/src_v to index the equirect directly.
        const CH: u32 = 3;
        let out_n = self.out_width * self.out_height;
        let mut out = vec![0u8; out_n as usize * CH as usize];

        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.valid {
                // Re-map the fisheye source UV through equirectangular sphere
                let sphere = crate::projection::equirect_to_sphere(&UvCoord {
                    u: entry.src_u,
                    v: entry.src_v,
                });
                let eq_uv = sphere_to_equirect(&sphere);
                let sample = bilinear_sample_u8(src, src_width, src_height, eq_uv.u, eq_uv.v, CH);
                let dst = idx * CH as usize;
                out[dst..dst + CH as usize].copy_from_slice(&sample);
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fisheye::FisheyeParams;

    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── Build tests ───────────────────────────────────────────────────────────

    #[test]
    fn lut_build_produces_correct_entry_count() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        assert_eq!(lut.len(), 128 * 64);
        assert_eq!(lut.out_width, 128);
        assert_eq!(lut.out_height, 64);
    }

    #[test]
    fn lut_build_has_valid_entries() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        // At least some entries should be valid (the fisheye covers most of the sphere)
        let valid_count = lut.entries.iter().filter(|e| e.valid).count();
        assert!(
            valid_count > 128 * 64 / 4,
            "expected many valid entries, got {valid_count}"
        );
    }

    #[test]
    fn lut_valid_entries_have_in_range_uv() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        for entry in lut.entries.iter().filter(|e| e.valid) {
            assert!(
                entry.src_u >= 0.0 && entry.src_u <= 1.0,
                "src_u out of range: {}",
                entry.src_u
            );
            assert!(
                entry.src_v >= 0.0 && entry.src_v <= 1.0,
                "src_v out of range: {}",
                entry.src_v
            );
        }
    }

    // ── Apply tests ──────────────────────────────────────────────────────────

    #[test]
    fn lut_apply_output_size() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        let src = solid(64, 64, 128, 64, 32);
        let out = lut.apply_u8(&src, 64, 64).expect("apply");
        assert_eq!(out.len(), 128 * 64 * 3);
    }

    #[test]
    fn lut_apply_solid_colour_centre_correct() {
        // A solid-colour fisheye: the equirectangular output should mostly
        // match the input colour at the equatorial centre.
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        let src = solid(64, 64, 200, 100, 50);
        let out = lut.apply_u8(&src, 64, 64).expect("apply");

        // Pixel at (row=16, col=64) is well inside the 180° fisheye circle
        // (theta≈46°, r_norm≈0.52); row=32 maps to theta≈91° which is outside.
        let cx = 64usize;
        let cy = 16usize;
        let base = (cy * 128 + cx) * 3;
        let err = (out[base] as i32 - 200).abs();
        assert!(err <= 5, "R error at (row=16, col=64): {err}");
    }

    #[test]
    fn lut_apply_matches_direct_fisheye_conversion() {
        use crate::fisheye::fisheye_to_equirect;
        let params = FisheyeParams::equidistant(180.0);
        let src = solid(32, 32, 150, 75, 25);

        // Direct conversion
        let direct = fisheye_to_equirect(&src, 32, 32, &params, 64, 32).expect("direct");

        // LUT-based conversion
        let lut = FisheyeLut::build(&params, 32, 32, 64, 32);
        let via_lut = lut.apply_u8(&src, 32, 32).expect("lut");

        // They should agree at most pixels (bilinear rounding may differ by ≤2)
        let mut max_err = 0i32;
        for i in 0..direct.len() {
            let e = (direct[i] as i32 - via_lut[i] as i32).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err <= 3, "max error between direct and LUT: {max_err}");
    }

    #[test]
    fn lut_apply_zero_src_dim_error() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        let src = solid(64, 64, 100, 100, 100);
        assert!(lut.apply_u8(&src, 0, 64).is_err());
        assert!(lut.apply_u8(&src, 64, 0).is_err());
    }

    #[test]
    fn lut_apply_buffer_too_small_error() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 128, 64);
        assert!(lut.apply_u8(&[0u8; 5], 64, 64).is_err());
    }

    #[test]
    fn lut_is_empty_for_zero_output() {
        // Building with 0-width output is degenerate but should give empty LUT
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 64, 64, 0, 0);
        assert!(lut.is_empty());
    }

    // ── entry() accessor ─────────────────────────────────────────────────────

    #[test]
    fn lut_entry_accessor_in_range() {
        let params = FisheyeParams::equidistant(180.0);
        let lut = FisheyeLut::build(&params, 8, 8, 16, 8);
        assert!(lut.entry(0).is_some());
        assert!(lut.entry(16 * 8 - 1).is_some());
        assert!(lut.entry(16 * 8).is_none());
    }
}
