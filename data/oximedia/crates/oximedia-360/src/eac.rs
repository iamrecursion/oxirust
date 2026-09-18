//! Equi-Angular Cubemap (EAC) projection.
//!
//! EAC is the projection format used by YouTube and Google for 360° VR content.
//! Unlike the standard cubemap where face UV coordinates map linearly to the
//! 3-D direction, EAC applies an `atan`-based remapping so that each face pixel
//! subtends an equal solid angle — greatly improving sampling uniformity in the
//! face corners.
//!
//! ## Coordinate conventions
//!
//! The EAC remapping transforms a face-local UV coordinate `s ∈ [0,1]` to the
//! actual face tangent `t` used when computing the 3-D direction:
//!
//! ```text
//! t = tan(π/4 * (2*s - 1))
//! ```
//!
//! and its inverse:
//!
//! ```text
//! s = (atan(t) / (π/4) + 1) / 2
//! ```
//!
//! ## Provided conversions
//!
//! * [`equirect_to_eac`]      — equirectangular image → EAC 6-face layout
//! * [`eac_to_equirect`]      — EAC 6-face layout → equirectangular image
//! * [`eac_uv_to_sphere`]     — coordinate-only (EAC face UV → sphere)
//! * [`sphere_to_eac_face`]   — coordinate-only (sphere → EAC face UV)

use std::collections::HashMap;

use crate::{
    projection::{
        bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, CubeFace, SphericalCoord,
        UvCoord,
    },
    VrError,
};

// ─── EAC coordinate mapping helpers ─────────────────────────────────────────

/// Apply EAC remapping: face-local UV in \[0,1\] → tangent value.
///
/// The formula is `t = tan(π/4 · (2·s − 1))`.
#[inline]
pub fn eac_uv_to_tangent(s: f64) -> f64 {
    let angle = std::f64::consts::FRAC_PI_4 * (2.0 * s - 1.0);
    angle.tan()
}

/// Invert EAC remapping: tangent value → face-local UV in \[0,1\].
///
/// The formula is `s = (atan(t) / (π/4) + 1) / 2`.
#[inline]
pub fn eac_tangent_to_uv(t: f64) -> f64 {
    let angle = t.atan();
    (angle / std::f64::consts::FRAC_PI_4 + 1.0) * 0.5
}

// ─── EAC face UV → sphere ─────────────────────────────────────────────────────

/// Convert an EAC face UV coordinate to a unit-sphere direction.
///
/// `u` and `v` are in `[0, 1]` (EAC-remapped face-local coordinates).
pub fn eac_uv_to_sphere(face: CubeFace, u: f64, v: f64) -> SphericalCoord {
    // Apply EAC remapping to get tangent-space coordinates
    let ts = eac_uv_to_tangent(u); // horizontal tangent
    let tt = eac_uv_to_tangent(v); // vertical tangent

    // Reconstruct the 3-D direction vector using the same face convention as
    // the standard cubemap, but using EAC tangent coordinates instead of linear ones.
    let (x, y, z) = match face {
        CubeFace::Front => (ts, -tt, 1.0),
        CubeFace::Back => (-ts, -tt, -1.0),
        CubeFace::Right => (1.0, -tt, -ts),
        CubeFace::Left => (-1.0, -tt, ts),
        CubeFace::Top => (ts, 1.0, tt),
        CubeFace::Bottom => (ts, -1.0, -tt),
    };

    let len = (x * x + y * y + z * z).sqrt();
    let nx = x / len;
    let ny = y / len;
    let nz = z / len;

    let elevation_rad = (ny as f32).asin();
    let azimuth_rad = (nx as f32).atan2(nz as f32);

    SphericalCoord {
        azimuth_rad,
        elevation_rad,
    }
}

// ─── Sphere → EAC face UV ─────────────────────────────────────────────────────

/// Convert a unit-sphere direction to an EAC face and face-local UV.
///
/// Returns `(face, u, v)` where `u`, `v` are EAC-remapped coordinates in `[0,1]`.
pub fn sphere_to_eac_face(s: &SphericalCoord) -> (CubeFace, f64, f64) {
    // Convert to 3-D Cartesian
    let x = (s.elevation_rad.cos() * s.azimuth_rad.sin()) as f64;
    let y = s.elevation_rad.sin() as f64;
    let z = (s.elevation_rad.cos() * s.azimuth_rad.cos()) as f64;

    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();

    // Select dominant face and compute face-tangent coordinates
    let (face, ts, tt) = if ax >= ay && ax >= az {
        if x > 0.0 {
            // Right: dominant +X
            (CubeFace::Right, -z / x, -y / x)
        } else {
            // Left: dominant -X
            (CubeFace::Left, z / (-x), -y / (-x))
        }
    } else if ay >= ax && ay >= az {
        if y > 0.0 {
            // Top: dominant +Y
            (CubeFace::Top, x / y, z / y)
        } else {
            // Bottom: dominant -Y
            (CubeFace::Bottom, x / (-y), -z / (-y))
        }
    } else {
        if z > 0.0 {
            // Front: dominant +Z
            (CubeFace::Front, x / z, -y / z)
        } else {
            // Back: dominant -Z
            (CubeFace::Back, -x / (-z), -y / (-z))
        }
    };

    let u = eac_tangent_to_uv(ts).clamp(0.0, 1.0);
    let v = eac_tangent_to_uv(tt).clamp(0.0, 1.0);

    (face, u, v)
}

// ─── Full-image conversions ───────────────────────────────────────────────────

/// Convert an equirectangular image to six EAC cube-face images.
///
/// * `src`       — source pixel data (RGB, 3 bpp, row-major)
/// * `width`     — source image width in pixels
/// * `height`    — source image height in pixels
/// * `face_size` — output EAC face size (square); each output is `face_size × face_size`
///
/// Returns a [`HashMap`] mapping each [`CubeFace`] to its RGB pixel data.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero or `src` is
/// too small.
pub fn equirect_to_eac(
    src: &[u8],
    width: u32,
    height: u32,
    face_size: u32,
) -> Result<HashMap<CubeFace, Vec<u8>>, VrError> {
    validate_eac_input(src, width, height, face_size)?;

    const CH: u32 = 3;
    let face_pixels = (face_size * face_size * CH) as usize;
    let mut map: HashMap<CubeFace, Vec<u8>> = HashMap::new();

    for face in CubeFace::all() {
        let mut face_data = vec![0u8; face_pixels];

        for fy in 0..face_size {
            for fx in 0..face_size {
                // EAC face-local UV in [0,1]
                let eu = (fx as f64 + 0.5) / face_size as f64;
                let ev = (fy as f64 + 0.5) / face_size as f64;

                let sphere = eac_uv_to_sphere(face, eu, ev);
                let uv = sphere_to_equirect(&sphere);

                let sample = bilinear_sample_u8(src, width, height, uv.u, uv.v, CH);
                let dst_base = (fy * face_size + fx) as usize * CH as usize;
                face_data[dst_base..dst_base + CH as usize].copy_from_slice(&sample);
            }
        }

        map.insert(face, face_data);
    }

    Ok(map)
}

/// Convert six EAC face images back to an equirectangular image.
///
/// * `faces`      — map from [`CubeFace`] to EAC pixel data (RGB, 3 bpp, square)
/// * `face_size`  — side length of each EAC face in pixels
/// * `out_width`  — output equirectangular width in pixels
/// * `out_height` — output equirectangular height in pixels
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if dimensions are zero.
/// Returns [`VrError::MissingFace`] if any face is absent.
pub fn eac_to_equirect(
    faces: &HashMap<CubeFace, Vec<u8>>,
    face_size: u32,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, VrError> {
    if face_size == 0 || out_width == 0 || out_height == 0 {
        return Err(VrError::InvalidDimensions(
            "face_size, out_width and out_height must be > 0".into(),
        ));
    }
    for face in CubeFace::all() {
        if !faces.contains_key(&face) {
            return Err(VrError::MissingFace(format!("{face:?}")));
        }
    }

    const CH: u32 = 3;
    let mut out = vec![0u8; (out_width * out_height * CH) as usize];

    for oy in 0..out_height {
        for ox in 0..out_width {
            let u = (ox as f32 + 0.5) / out_width as f32;
            let v = (oy as f32 + 0.5) / out_height as f32;

            let sphere = equirect_to_sphere(&UvCoord { u, v });
            let (face, fu, fv) = sphere_to_eac_face(&sphere);

            let face_data = &faces[&face];
            let sample =
                bilinear_sample_u8(face_data, face_size, face_size, fu as f32, fv as f32, CH);
            let dst_base = (oy * out_width + ox) as usize * CH as usize;
            out[dst_base..dst_base + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Internal validation ──────────────────────────────────────────────────────

fn validate_eac_input(src: &[u8], w: u32, h: u32, face_size: u32) -> Result<(), VrError> {
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    if face_size == 0 {
        return Err(VrError::InvalidDimensions("face_size must be > 0".into()));
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
    use std::f32::consts::PI;

    const EPSILON_ANGLE: f32 = 0.03;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── EAC tangent mapping ──────────────────────────────────────────────────

    #[test]
    fn eac_tangent_roundtrip_centre() {
        let t = eac_uv_to_tangent(0.5);
        let s = eac_tangent_to_uv(t);
        assert!((s - 0.5).abs() < 1e-10, "s={s}");
    }

    #[test]
    fn eac_tangent_roundtrip_quarter() {
        let s0 = 0.25;
        let t = eac_uv_to_tangent(s0);
        let s1 = eac_tangent_to_uv(t);
        assert!((s1 - s0).abs() < 1e-10, "s1={s1}");
    }

    #[test]
    fn eac_tangent_centre_gives_zero() {
        let t = eac_uv_to_tangent(0.5);
        assert!(t.abs() < 1e-12, "t={t}");
    }

    #[test]
    fn eac_tangent_to_uv_zero_gives_half() {
        let s = eac_tangent_to_uv(0.0);
        assert!((s - 0.5).abs() < 1e-12, "s={s}");
    }

    // ── sphere ↔ EAC face round-trips ────────────────────────────────────────

    fn check_sphere_eac_roundtrip(az: f32, el: f32) {
        let sphere_in = SphericalCoord {
            azimuth_rad: az,
            elevation_rad: el,
        };
        let (face, u, v) = sphere_to_eac_face(&sphere_in);
        let sphere_out = eac_uv_to_sphere(face, u, v);

        // Azimuth may wrap; check the minimal angular distance
        let az_diff = (sphere_out.azimuth_rad - az).abs();
        let az_diff = az_diff
            .min((az_diff - 2.0 * PI).abs())
            .min((az_diff + 2.0 * PI).abs());
        assert!(
            az_diff < EPSILON_ANGLE,
            "az mismatch: in={:.4} out={:.4}",
            az,
            sphere_out.azimuth_rad
        );
        assert!(
            (sphere_out.elevation_rad - el).abs() < EPSILON_ANGLE,
            "el mismatch: in={:.4} out={:.4}",
            el,
            sphere_out.elevation_rad
        );
    }

    #[test]
    fn eac_roundtrip_front_centre() {
        check_sphere_eac_roundtrip(0.0, 0.0);
    }

    #[test]
    fn eac_roundtrip_right() {
        check_sphere_eac_roundtrip(PI / 2.0, 0.0);
    }

    #[test]
    fn eac_roundtrip_left() {
        check_sphere_eac_roundtrip(-PI / 2.0, 0.0);
    }

    #[test]
    fn eac_roundtrip_top() {
        check_sphere_eac_roundtrip(0.0, PI / 2.0 * 0.85);
    }

    #[test]
    fn eac_roundtrip_bottom() {
        check_sphere_eac_roundtrip(0.0, -PI / 2.0 * 0.85);
    }

    #[test]
    fn eac_roundtrip_back() {
        check_sphere_eac_roundtrip(PI * 0.9, 0.0);
    }

    #[test]
    fn eac_roundtrip_diagonal() {
        check_sphere_eac_roundtrip(PI / 4.0, PI / 8.0);
    }

    // ── equirect_to_eac ──────────────────────────────────────────────────────

    #[test]
    fn equirect_to_eac_produces_six_faces() {
        let src = solid_rgb(64, 32, 100, 150, 200);
        let faces = equirect_to_eac(&src, 64, 32, 8).expect("ok");
        assert_eq!(faces.len(), 6);
        for face in CubeFace::all() {
            assert!(faces.contains_key(&face), "missing {face:?}");
            assert_eq!(faces[&face].len(), 8 * 8 * 3);
        }
    }

    #[test]
    fn equirect_to_eac_zero_dimensions_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        assert!(equirect_to_eac(&src, 0, 32, 8).is_err());
        assert!(equirect_to_eac(&src, 64, 0, 8).is_err());
        assert!(equirect_to_eac(&src, 64, 32, 0).is_err());
    }

    #[test]
    fn equirect_to_eac_buffer_too_small_error() {
        assert!(equirect_to_eac(&[0u8; 5], 64, 32, 8).is_err());
    }

    #[test]
    fn equirect_to_eac_solid_colour_preserved() {
        // A uniform-colour equirect: every EAC face pixel should equal input colour
        let src = solid_rgb(64, 32, 200, 100, 50);
        let faces = equirect_to_eac(&src, 64, 32, 8).expect("ok");
        for face in CubeFace::all() {
            let data = &faces[&face];
            // Check a centre pixel
            let cx = 4usize;
            let cy = 4usize;
            let base = (cy * 8 + cx) * 3;
            let err = (data[base] as i32 - 200).abs();
            assert!(err <= 5, "face={face:?} R error={err}");
        }
    }

    // ── eac_to_equirect ──────────────────────────────────────────────────────

    #[test]
    fn eac_to_equirect_correct_size() {
        let src = solid_rgb(64, 32, 150, 75, 200);
        let faces = equirect_to_eac(&src, 64, 32, 16).expect("ok");
        let out = eac_to_equirect(&faces, 16, 64, 32).expect("ok");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn eac_to_equirect_missing_face_error() {
        let mut faces: HashMap<CubeFace, Vec<u8>> = HashMap::new();
        for face in [
            CubeFace::Front,
            CubeFace::Left,
            CubeFace::Right,
            CubeFace::Top,
            CubeFace::Bottom,
        ] {
            faces.insert(face, vec![0u8; 8 * 8 * 3]);
        }
        assert!(eac_to_equirect(&faces, 8, 32, 16).is_err());
    }

    #[test]
    fn eac_to_equirect_zero_dimensions_error() {
        let faces: HashMap<CubeFace, Vec<u8>> = HashMap::new();
        assert!(eac_to_equirect(&faces, 0, 32, 16).is_err());
        assert!(eac_to_equirect(&faces, 8, 0, 16).is_err());
        assert!(eac_to_equirect(&faces, 8, 32, 0).is_err());
    }

    #[test]
    fn eac_equirect_roundtrip_colour_preservation() {
        let src = solid_rgb(128, 64, 180, 90, 45);
        let faces = equirect_to_eac(&src, 128, 64, 32).expect("to eac");
        let out = eac_to_equirect(&faces, 32, 128, 64).expect("to equirect");
        // Centre pixel should be close to input colour
        let base = (32 * 128 + 64) * 3;
        let err_r = (out[base] as i32 - 180).abs();
        let err_g = (out[base + 1] as i32 - 90).abs();
        assert!(err_r <= 10, "R error={err_r}");
        assert!(err_g <= 10, "G error={err_g}");
    }

    // ── EAC mapping properties ────────────────────────────────────────────────

    #[test]
    fn eac_remapping_properties() {
        // EAC maps equal angular increments to equal UV increments (angle-linear),
        // which is more uniform than the standard cubemap that maps equal tangent
        // increments (cos-nonlinear) to equal UV increments.
        //
        // Verify core invariants:
        // 1. Centre direction (angle 0) → UV = 0.5
        // 2. Edge direction (angle 45°) → UV = 1.0
        // 3. At 22.5°, EAC UV = 0.75 (exactly half-way, confirming angle-linearity)
        // 4. Standard cubemap UV at 22.5° < 0.75 (compressed near edges)

        let u_centre = eac_tangent_to_uv(0.0);
        assert!((u_centre - 0.5).abs() < 1e-9, "centre→0.5: {u_centre}");

        let u_edge = eac_tangent_to_uv(1.0); // tan(45°) = 1.0
        assert!((u_edge - 1.0).abs() < 1e-9, "edge→1.0: {u_edge}");

        // At 22.5° (halfway in angle): EAC UV = 0.75 exactly (angle-linear)
        let t_22 = (22.5_f64.to_radians()).tan();
        let u_22_eac = eac_tangent_to_uv(t_22);
        assert!((u_22_eac - 0.75).abs() < 1e-9, "22.5°→0.75: {u_22_eac}");

        // Standard cubemap linear UV at 22.5° = tan(22.5°)*0.5 + 0.5 ≈ 0.707
        let u_22_std = t_22 * 0.5 + 0.5;
        assert!(
            u_22_std < u_22_eac,
            "EAC should give more UV space near centre than standard: std={u_22_std}, eac={u_22_eac}"
        );
    }

    #[test]
    fn eac_face_centre_maps_to_uv_half() {
        // The direction directly at the front face centre (az=0, el=0) should
        // map to EAC UV = (0.5, 0.5)
        let sphere = SphericalCoord {
            azimuth_rad: 0.0,
            elevation_rad: 0.0,
        };
        let (face, eu, ev) = sphere_to_eac_face(&sphere);
        assert_eq!(face, crate::CubeFace::Front);
        assert!((eu - 0.5).abs() < 0.01, "eu={eu}");
        assert!((ev - 0.5).abs() < 0.01, "ev={ev}");
    }
}
