//! Octahedral projection mapping for efficient VR video compression.
//!
//! The octahedral projection maps the unit sphere onto a unit square (or an
//! octahedron unfolded into a diamond then rotated 45°).  Compared with the
//! equirectangular projection it offers:
//!
//! * **Uniform area distribution** — far fewer wasted pixels at the poles.
//! * **Continuous UV domain** — the entire sphere tiles into a single square,
//!   making it amenable to block-based video codecs.
//! * **Simple GPU decode** — the forward/inverse transforms use only basic
//!   arithmetic (no trigonometric functions) at decode time.
//!
//! ## References
//!
//! * Meyer et al. "Survey of Sphere Mapping Techniques" (2010)
//! * Cigolle et al. "A Survey of Efficient Representations for Independent
//!   Unit Vectors" (2014) — octahedral normal encoding
//!
//! ## Coordinate convention
//!
//! The octahedral UV domain covers `[0, 1] × [0, 1]`.  The unit sphere is
//! parameterised in the standard way:
//! * `x = cos(el) · sin(az)`  (right)
//! * `y = sin(el)`             (up)
//! * `z = cos(el) · cos(az)`  (forward)
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::octahedral::{equirect_to_octahedral, octahedral_to_equirect};
//!
//! let src = vec![128u8; 128 * 64 * 3]; // 128×64 equirectangular
//! let oct = equirect_to_octahedral(&src, 128, 64, 64).expect("ok");
//! let back = octahedral_to_equirect(&oct, 64, 128, 64).expect("ok");
//! ```

use crate::{
    projection::{
        bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, SphericalCoord, UvCoord,
    },
    VrError,
};

// ─── Forward transform: sphere → octahedral UV ───────────────────────────────

/// Convert a unit-sphere direction to octahedral UV in `[0,1]²`.
///
/// **Algorithm:**
/// 1. Project onto the L¹ octahedron: divide by L¹ norm (sum of absolutes).
/// 2. Unfold the lower hemisphere by reflecting the x/z components.
/// 3. Map `[-1,1]²` to `[0,1]²`.
pub fn sphere_to_octahedral_uv(s: &SphericalCoord) -> UvCoord {
    // Cartesian direction
    let x = s.elevation_rad.cos() * s.azimuth_rad.sin();
    let y = s.elevation_rad.sin();
    let z = s.elevation_rad.cos() * s.azimuth_rad.cos();

    let l1 = x.abs() + y.abs() + z.abs();
    let (nx, ny, nz) = if l1 < f32::EPSILON {
        (0.0, 1.0, 0.0) // degenerate: treat as north pole
    } else {
        (x / l1, y / l1, z / l1)
    };

    // Project to top face of octahedron (nx, nz in [-1,1] for y > 0)
    let (px, pz) = if ny >= 0.0 {
        (nx, nz)
    } else {
        // Lower hemisphere: unfold via diamond reflection
        let ux = (1.0 - nz.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 };
        let uz = (1.0 - nx.abs()) * if nz >= 0.0 { 1.0 } else { -1.0 };
        (ux, uz)
    };

    // Map [-1,1] → [0,1]
    UvCoord {
        u: (px + 1.0) * 0.5,
        v: (pz + 1.0) * 0.5,
    }
}

// ─── Inverse transform: octahedral UV → sphere ───────────────────────────────

/// Convert an octahedral UV in `[0,1]²` back to a unit-sphere direction.
pub fn octahedral_uv_to_sphere(uv: &UvCoord) -> SphericalCoord {
    // Map [0,1] → [-1,1]
    let px = uv.u * 2.0 - 1.0;
    let pz = uv.v * 2.0 - 1.0;

    // L¹ norm of the projected point on the octahedron
    let l1 = px.abs() + pz.abs();

    let (nx, ny, nz) = if l1 <= 1.0 {
        // Upper hemisphere
        (px, 1.0 - l1, pz)
    } else {
        // Lower hemisphere: invert the fold
        let ux = (1.0 - pz.abs()) * if px >= 0.0 { 1.0 } else { -1.0 };
        let uz = (1.0 - px.abs()) * if pz >= 0.0 { 1.0 } else { -1.0 };
        (ux, -(1.0 - (ux.abs() + uz.abs())), uz)
    };

    // Normalise to unit sphere
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    let (nx, ny, nz) = if len < f32::EPSILON {
        (0.0, 1.0, 0.0)
    } else {
        (nx / len, ny / len, nz / len)
    };

    let elevation_rad = ny.clamp(-1.0, 1.0).asin();
    let azimuth_rad = nx.atan2(nz);

    SphericalCoord {
        azimuth_rad,
        elevation_rad,
    }
}

// ─── Full-image conversions ───────────────────────────────────────────────────

/// Convert an equirectangular image to the octahedral projection.
///
/// * `src`       — source pixel data (RGB, 3 bpp, row-major)
/// * `src_w`     — source image width in pixels
/// * `src_h`     — source image height in pixels
/// * `out_size`  — output octahedral image size (square, `out_size × out_size`)
///
/// Returns an RGB image of `out_size × out_size` pixels.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero or `src` is
/// too small for the declared dimensions.
pub fn equirect_to_octahedral(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_size: u32,
) -> Result<Vec<u8>, VrError> {
    validate_input(src, src_w, src_h, out_size)?;

    const CH: u32 = 3;
    let total_pixels = (out_size * out_size * CH) as usize;
    let mut out = vec![0u8; total_pixels];

    for oy in 0..out_size {
        for ox in 0..out_size {
            let ou = (ox as f32 + 0.5) / out_size as f32;
            let ov = (oy as f32 + 0.5) / out_size as f32;

            // Octahedral UV → sphere → equirect UV → sample
            let oct_uv = UvCoord { u: ou, v: ov };
            let sphere = octahedral_uv_to_sphere(&oct_uv);
            let src_uv = sphere_to_equirect(&sphere);

            let sample = bilinear_sample_u8(src, src_w, src_h, src_uv.u, src_uv.v, CH);
            let dst_base = (oy * out_size + ox) as usize * CH as usize;
            out[dst_base..dst_base + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

/// Convert an octahedral projection image back to equirectangular.
///
/// * `oct`       — octahedral pixel data (RGB, 3 bpp, square, row-major)
/// * `oct_size`  — side length of the octahedral image in pixels
/// * `out_w`     — output equirectangular image width
/// * `out_h`     — output equirectangular image height
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
/// Returns [`VrError::BufferTooSmall`] if `oct` is too small.
pub fn octahedral_to_equirect(
    oct: &[u8],
    oct_size: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, VrError> {
    if oct_size == 0 || out_w == 0 || out_h == 0 {
        return Err(VrError::InvalidDimensions(
            "oct_size, out_w and out_h must be > 0".into(),
        ));
    }
    let expected = oct_size as usize * oct_size as usize * 3;
    if oct.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: oct.len(),
        });
    }

    const CH: u32 = 3;
    let mut out = vec![0u8; (out_w * out_h * CH) as usize];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let eu = (ox as f32 + 0.5) / out_w as f32;
            let ev = (oy as f32 + 0.5) / out_h as f32;

            // Equirect UV → sphere → octahedral UV → sample
            let eq_uv = UvCoord { u: eu, v: ev };
            let sphere = equirect_to_sphere(&eq_uv);
            let oct_uv = sphere_to_octahedral_uv(&sphere);

            let sample = bilinear_sample_u8(oct, oct_size, oct_size, oct_uv.u, oct_uv.v, CH);
            let dst_base = (oy * out_w + ox) as usize * CH as usize;
            out[dst_base..dst_base + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Internal validation ──────────────────────────────────────────────────────

fn validate_input(src: &[u8], w: u32, h: u32, out_size: u32) -> Result<(), VrError> {
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    if out_size == 0 {
        return Err(VrError::InvalidDimensions("out_size must be > 0".into()));
    }
    let expected = w as usize * h as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    const EPSILON: f32 = 0.04;

    fn sphere(az: f32, el: f32) -> SphericalCoord {
        SphericalCoord {
            azimuth_rad: az,
            elevation_rad: el,
        }
    }

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── Coordinate round-trips ───────────────────────────────────────────────

    fn check_roundtrip(az: f32, el: f32) {
        let s_in = sphere(az, el);
        let uv = sphere_to_octahedral_uv(&s_in);
        let s_out = octahedral_uv_to_sphere(&uv);

        // UV must be in [0,1]
        assert!(uv.u >= 0.0 && uv.u <= 1.0, "u={}", uv.u);
        assert!(uv.v >= 0.0 && uv.v <= 1.0, "v={}", uv.v);

        assert!(
            (s_out.elevation_rad - s_in.elevation_rad).abs() < EPSILON,
            "el in={:.4} out={:.4}",
            s_in.elevation_rad,
            s_out.elevation_rad
        );

        // Allow azimuth wrap-around
        let az_diff = (s_out.azimuth_rad - s_in.azimuth_rad).abs();
        let az_diff = az_diff
            .min((az_diff - 2.0 * PI).abs())
            .min((az_diff + 2.0 * PI).abs());
        assert!(
            az_diff < EPSILON,
            "az in={:.4} out={:.4}",
            s_in.azimuth_rad,
            s_out.azimuth_rad
        );
    }

    #[test]
    fn octahedral_roundtrip_north_pole() {
        check_roundtrip(0.0, FRAC_PI_2 * 0.95);
    }

    #[test]
    fn octahedral_roundtrip_south_pole() {
        check_roundtrip(0.0, -FRAC_PI_2 * 0.95);
    }

    #[test]
    fn octahedral_roundtrip_front() {
        check_roundtrip(0.0, 0.0);
    }

    #[test]
    fn octahedral_roundtrip_right() {
        check_roundtrip(FRAC_PI_2, 0.0);
    }

    #[test]
    fn octahedral_roundtrip_left() {
        check_roundtrip(-FRAC_PI_2, 0.0);
    }

    #[test]
    fn octahedral_roundtrip_back() {
        check_roundtrip(PI * 0.9, 0.0);
    }

    #[test]
    fn octahedral_roundtrip_diagonal() {
        check_roundtrip(PI / 4.0, PI / 6.0);
    }

    #[test]
    fn octahedral_roundtrip_lower_hemisphere() {
        check_roundtrip(PI / 3.0, -PI / 5.0);
    }

    // ── North pole maps to UV = (0.5, 0.5) ──────────────────────────────────

    #[test]
    fn north_pole_maps_to_centre() {
        let s = sphere(0.0, FRAC_PI_2);
        let uv = sphere_to_octahedral_uv(&s);
        assert!((uv.u - 0.5).abs() < 0.01, "u={}", uv.u);
        assert!((uv.v - 0.5).abs() < 0.01, "v={}", uv.v);
    }

    // ── UV is always in [0,1] ────────────────────────────────────────────────

    #[test]
    fn uv_always_in_unit_square() {
        for az_i in 0..=16 {
            for el_i in 0..=8 {
                let az = (az_i as f32 - 8.0) * PI / 8.0;
                let el = (el_i as f32 - 4.0) * FRAC_PI_2 / 4.0;
                let el = el.clamp(-FRAC_PI_2 * 0.99, FRAC_PI_2 * 0.99);
                let s = sphere(az, el);
                let uv = sphere_to_octahedral_uv(&s);
                assert!(
                    uv.u >= -0.001 && uv.u <= 1.001,
                    "az={az} el={el} u={}",
                    uv.u
                );
                assert!(
                    uv.v >= -0.001 && uv.v <= 1.001,
                    "az={az} el={el} v={}",
                    uv.v
                );
            }
        }
    }

    // ── Image conversion ─────────────────────────────────────────────────────

    #[test]
    fn equirect_to_octahedral_correct_size() {
        let src = solid_rgb(64, 32, 100, 150, 200);
        let out = equirect_to_octahedral(&src, 64, 32, 32).expect("ok");
        assert_eq!(out.len(), 32 * 32 * 3);
    }

    #[test]
    fn octahedral_to_equirect_correct_size() {
        let oct = solid_rgb(32, 32, 200, 100, 50);
        let out = octahedral_to_equirect(&oct, 32, 128, 64).expect("ok");
        assert_eq!(out.len(), 128 * 64 * 3);
    }

    #[test]
    fn equirect_to_octahedral_zero_dimensions_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        assert!(equirect_to_octahedral(&src, 0, 32, 32).is_err());
        assert!(equirect_to_octahedral(&src, 64, 0, 32).is_err());
        assert!(equirect_to_octahedral(&src, 64, 32, 0).is_err());
    }

    #[test]
    fn octahedral_to_equirect_zero_dimensions_error() {
        let oct = solid_rgb(32, 32, 0, 0, 0);
        assert!(octahedral_to_equirect(&oct, 0, 128, 64).is_err());
        assert!(octahedral_to_equirect(&oct, 32, 0, 64).is_err());
        assert!(octahedral_to_equirect(&oct, 32, 128, 0).is_err());
    }

    #[test]
    fn equirect_to_octahedral_buffer_too_small_error() {
        assert!(equirect_to_octahedral(&[0u8; 5], 64, 32, 16).is_err());
    }

    #[test]
    fn octahedral_to_equirect_buffer_too_small_error() {
        assert!(octahedral_to_equirect(&[0u8; 5], 32, 64, 32).is_err());
    }

    #[test]
    fn octahedral_solid_colour_round_trip_centre() {
        let src = solid_rgb(128, 64, 180, 90, 45);
        let oct = equirect_to_octahedral(&src, 128, 64, 64).expect("to oct");
        let back = octahedral_to_equirect(&oct, 64, 128, 64).expect("to equirect");

        // Sample a central pixel — should be close to the original colour
        let base = (32 * 128 + 64) * 3;
        let err_r = (back[base] as i32 - 180).abs();
        let err_g = (back[base + 1] as i32 - 90).abs();
        assert!(err_r <= 10, "R error={err_r}");
        assert!(err_g <= 10, "G error={err_g}");
    }
}
