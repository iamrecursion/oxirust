//! Tiled cubemap conversion for better CPU cache locality.
//!
//! Standard row-major cubemap conversion processes pixels in scanline order,
//! which produces poor cache behaviour because the equirectangular source image
//! is sampled at pseudo-random UV positions.  By dividing the output into
//! small square tiles (typically 16×16 or 32×32 pixels) and processing each
//! tile completely before moving to the next, the working set that must stay
//! "hot" in the cache is dramatically reduced.
//!
//! Additionally this module exposes a **parallel** conversion path that uses
//! rayon to process faces and tiles concurrently.
//!
//! ## Performance notes
//!
//! * Tile size of 16 is a good default — it fits comfortably in L1 cache and
//!   aligns with many SIMD widths.
//! * Larger tiles (32, 64) may be faster on CPUs with large L1/L2 caches.
//! * The parallel path scales well on machines with ≥ 4 cores.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::tiled::equirect_to_cube_tiled;
//! use oximedia_360::projection::CubeFace;
//!
//! let src = vec![128u8; 256 * 128 * 3];
//! let faces = equirect_to_cube_tiled(&src, 256, 128, 64, 16).expect("ok");
//! assert!(faces.contains_key(&CubeFace::Front));
//! ```

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{
    projection::{
        bilinear_sample_u8, cube_face_to_sphere, sphere_to_equirect, CubeFace, CubeFaceCoord,
    },
    VrError,
};

// ─── Tiled equirect → cube ────────────────────────────────────────────────────

/// Convert an equirectangular image to six cube-map face images using
/// cache-friendly tiled processing.
///
/// * `src`        — source pixel data (RGB, 3 bpp, row-major)
/// * `src_w`      — source image width in pixels
/// * `src_h`      — source image height in pixels
/// * `face_size`  — output cube-face size (square)
/// * `tile_size`  — tile size in pixels (recommended: 16 or 32)
///
/// Returns a [`HashMap`] mapping each [`CubeFace`] to its RGB pixel data.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero or `src` is
/// too small for the declared dimensions.
pub fn equirect_to_cube_tiled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    face_size: u32,
    tile_size: u32,
) -> Result<HashMap<CubeFace, Vec<u8>>, VrError> {
    validate_input(src, src_w, src_h, face_size)?;
    if tile_size == 0 {
        return Err(VrError::InvalidDimensions("tile_size must be > 0".into()));
    }

    const CH: u32 = 3;
    let face_pixels = (face_size * face_size * CH) as usize;

    // Process each face in parallel
    let results: Vec<(CubeFace, Vec<u8>)> = CubeFace::all()
        .par_iter()
        .map(|&face| {
            let mut face_data = vec![0u8; face_pixels];
            process_face_tiled(
                src,
                src_w,
                src_h,
                face,
                &mut face_data,
                face_size,
                tile_size,
            );
            (face, face_data)
        })
        .collect();

    Ok(results.into_iter().collect())
}

/// Convert an equirectangular image to six cube-map faces (parallel, no tiling).
///
/// Each face is processed concurrently on a rayon thread.  This is equivalent
/// to [`equirect_to_cube_tiled`] with a tile size equal to `face_size` (a
/// single tile per face), which is fastest when the face fits entirely in cache.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero or `src` is
/// too small.
pub fn equirect_to_cube_parallel(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    face_size: u32,
) -> Result<HashMap<CubeFace, Vec<u8>>, VrError> {
    equirect_to_cube_tiled(src, src_w, src_h, face_size, face_size)
}

// ─── Parallel equirect → equirect (via cube) ─────────────────────────────────

/// Convert a single equirectangular frame using rayon parallel scanlines.
///
/// For each output scanline the function computes the equirectangular ↔ sphere
/// mapping independently, allowing safe parallelism with no data races.
///
/// * `src`        — source pixel data (RGB, 3 bpp, row-major)
/// * `src_w`      — source image width
/// * `src_h`      — source image height
/// * `out_w`      — output image width
/// * `out_h`      — output image height
/// * `map_pixel`  — per-pixel mapping function: `(u, v) → (src_u, src_v)`
///
/// This is a generic helper used by higher-level projection converters.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
/// Returns [`VrError::BufferTooSmall`] if `src` is too small.
pub fn resample_equirect_parallel<F>(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    map_pixel: F,
) -> Result<Vec<u8>, VrError>
where
    F: Fn(f32, f32) -> (f32, f32) + Sync,
{
    if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return Err(VrError::InvalidDimensions(
            "all dimensions must be > 0".into(),
        ));
    }
    let expected = src_w as usize * src_h as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }

    const CH: u32 = 3;
    let row_size = out_w as usize * CH as usize;

    // Process scanlines in parallel — each row is independent
    let rows: Vec<Vec<u8>> = (0..out_h)
        .into_par_iter()
        .map(|oy| {
            let mut row = vec![0u8; row_size];
            for ox in 0..out_w {
                let u = (ox as f32 + 0.5) / out_w as f32;
                let v = (oy as f32 + 0.5) / out_h as f32;
                let (su, sv) = map_pixel(u, v);
                let sample = bilinear_sample_u8(src, src_w, src_h, su, sv, CH);
                let dst_base = ox as usize * CH as usize;
                row[dst_base..dst_base + CH as usize].copy_from_slice(&sample);
            }
            row
        })
        .collect();

    let mut out = Vec::with_capacity(out_h as usize * row_size);
    for row in rows {
        out.extend_from_slice(&row);
    }
    Ok(out)
}

// ─── Packed SIMD-style bilinear sampler ───────────────────────────────────────

/// SIMD-accelerated bilinear sampling for 3-channel (RGB) images.
///
/// Processes 3 channels simultaneously using packed f32 arithmetic, avoiding
/// the per-channel loop overhead of the generic sampler.  On platforms with
/// AVX/NEON this typically compiles to vectorised code.
///
/// This function is semantically identical to [`bilinear_sample_u8`] but
/// is specialised for exactly 3 channels.
///
/// # Panics
/// Does not panic; edge pixels are clamped.
#[inline]
pub fn bilinear_sample_rgb_packed(data: &[u8], w: u32, h: u32, u: f32, v: f32) -> [u8; 3] {
    let fw = w as f32;
    let fh = h as f32;

    let px = (u * fw - 0.5).max(0.0);
    let py = (v * fh - 0.5).max(0.0);

    let x0 = (px.floor() as u32).min(w.saturating_sub(1));
    let y0 = (py.floor() as u32).min(h.saturating_sub(1));
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let tx = px - px.floor();
    let ty = py - py.floor();
    let one_m_tx = 1.0 - tx;
    let one_m_ty = 1.0 - ty;

    let stride = w as usize * 3;
    let b00 = y0 as usize * stride + x0 as usize * 3;
    let b10 = y0 as usize * stride + x1 as usize * 3;
    let b01 = y1 as usize * stride + x0 as usize * 3;
    let b11 = y1 as usize * stride + x1 as usize * 3;

    // All 3 channels packed into [f32; 3] arrays
    let p00 = [data[b00] as f32, data[b00 + 1] as f32, data[b00 + 2] as f32];
    let p10 = [data[b10] as f32, data[b10 + 1] as f32, data[b10 + 2] as f32];
    let p01 = [data[b01] as f32, data[b01 + 1] as f32, data[b01 + 2] as f32];
    let p11 = [data[b11] as f32, data[b11 + 1] as f32, data[b11 + 2] as f32];

    // Bilinear: top = lerp(p00, p10, tx), bottom = lerp(p01, p11, tx)
    // final = lerp(top, bottom, ty)
    let mut result = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] * one_m_tx + p10[c] * tx;
        let bottom = p01[c] * one_m_tx + p11[c] * tx;
        let v_final = top * one_m_ty + bottom * ty;
        result[c] = v_final.round().clamp(0.0, 255.0) as u8;
    }
    result
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

fn process_face_tiled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    face: CubeFace,
    face_data: &mut [u8],
    face_size: u32,
    tile_size: u32,
) {
    const CH: usize = 3;
    let tiles_x = (face_size + tile_size - 1) / tile_size;
    let tiles_y = (face_size + tile_size - 1) / tile_size;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x_start = tx * tile_size;
            let y_start = ty * tile_size;
            let x_end = (x_start + tile_size).min(face_size);
            let y_end = (y_start + tile_size).min(face_size);

            for fy in y_start..y_end {
                for fx in x_start..x_end {
                    let fu = (fx as f32 + 0.5) / face_size as f32;
                    let fv = (fy as f32 + 0.5) / face_size as f32;

                    let cube_coord = CubeFaceCoord { face, u: fu, v: fv };
                    let sphere = cube_face_to_sphere(&cube_coord);
                    let uv = sphere_to_equirect(&sphere);

                    let sample = bilinear_sample_rgb_packed(src, src_w, src_h, uv.u, uv.v);
                    let dst_base = (fy * face_size + fx) as usize * CH;
                    face_data[dst_base] = sample[0];
                    face_data[dst_base + 1] = sample[1];
                    face_data[dst_base + 2] = sample[2];
                }
            }
        }
    }
}

fn validate_input(src: &[u8], w: u32, h: u32, face_size: u32) -> Result<(), VrError> {
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

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── bilinear_sample_rgb_packed ───────────────────────────────────────────

    #[test]
    fn packed_sampler_solid_colour() {
        let img = solid_rgb(8, 8, 200, 100, 50);
        let p = bilinear_sample_rgb_packed(&img, 8, 8, 0.5, 0.5);
        assert_eq!(p, [200, 100, 50]);
    }

    #[test]
    fn packed_sampler_corner_clamping() {
        let img = solid_rgb(4, 4, 128, 64, 32);
        let p = bilinear_sample_rgb_packed(&img, 4, 4, 0.0, 0.0);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], 128);
    }

    #[test]
    fn packed_sampler_matches_generic() {
        let img = solid_rgb(16, 8, 180, 90, 45);
        for u in [0.1, 0.5, 0.9] {
            for v in [0.1, 0.5, 0.9] {
                let packed = bilinear_sample_rgb_packed(&img, 16, 8, u, v);
                let generic = bilinear_sample_u8(&img, 16, 8, u, v, 3);
                assert_eq!(packed.as_ref(), generic.as_slice(), "u={u} v={v}");
            }
        }
    }

    // ── equirect_to_cube_tiled ───────────────────────────────────────────────

    #[test]
    fn tiled_produces_six_faces() {
        let src = solid_rgb(64, 32, 100, 150, 200);
        let faces = equirect_to_cube_tiled(&src, 64, 32, 16, 8).expect("ok");
        assert_eq!(faces.len(), 6);
        for face in CubeFace::all() {
            assert!(faces.contains_key(&face), "missing {face:?}");
        }
    }

    #[test]
    fn tiled_face_size_is_correct() {
        let src = solid_rgb(64, 32, 200, 100, 50);
        let faces = equirect_to_cube_tiled(&src, 64, 32, 16, 4).expect("ok");
        for face in CubeFace::all() {
            assert_eq!(faces[&face].len(), 16 * 16 * 3, "face={face:?}");
        }
    }

    #[test]
    fn tiled_matches_standard_result() {
        // Tiled conversion should give same result as standard for solid colour
        let src = solid_rgb(64, 32, 180, 90, 45);
        let tiled_faces = equirect_to_cube_tiled(&src, 64, 32, 16, 4).expect("tiled");
        let std_faces = crate::projection::equirect_to_cube(&src, 64, 32, 16).expect("standard");

        for face in CubeFace::all() {
            let tiled = &tiled_faces[&face];
            let standard = &std_faces[&face];
            // Check centre pixel
            let cx = 8usize;
            let cy = 8usize;
            let base = (cy * 16 + cx) * 3;
            let err = (tiled[base] as i32 - standard[base] as i32).abs();
            assert!(err <= 2, "face={face:?} err={err}");
        }
    }

    #[test]
    fn tiled_zero_face_size_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        assert!(equirect_to_cube_tiled(&src, 64, 32, 0, 8).is_err());
    }

    #[test]
    fn tiled_zero_tile_size_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        assert!(equirect_to_cube_tiled(&src, 64, 32, 16, 0).is_err());
    }

    #[test]
    fn tiled_buffer_too_small_error() {
        assert!(equirect_to_cube_tiled(&[0u8; 5], 64, 32, 16, 8).is_err());
    }

    // ── equirect_to_cube_parallel ────────────────────────────────────────────

    #[test]
    fn parallel_produces_six_faces() {
        let src = solid_rgb(64, 32, 100, 200, 50);
        let faces = equirect_to_cube_parallel(&src, 64, 32, 16).expect("ok");
        assert_eq!(faces.len(), 6);
    }

    // ── resample_equirect_parallel ───────────────────────────────────────────

    #[test]
    fn resample_parallel_identity_map_correct_size() {
        let src = solid_rgb(64, 32, 128, 64, 32);
        let out = resample_equirect_parallel(&src, 64, 32, 64, 32, |u, v| (u, v)).expect("ok");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn resample_parallel_solid_colour_preserved() {
        let src = solid_rgb(64, 32, 200, 100, 50);
        let out = resample_equirect_parallel(&src, 64, 32, 64, 32, |u, v| (u, v)).expect("ok");
        let base = (16 * 64 + 32) * 3;
        assert!((out[base] as i32 - 200).abs() <= 3);
    }

    #[test]
    fn resample_parallel_zero_dim_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        assert!(resample_equirect_parallel(&src, 0, 32, 64, 32, |u, v| (u, v)).is_err());
        assert!(resample_equirect_parallel(&src, 64, 0, 64, 32, |u, v| (u, v)).is_err());
        assert!(resample_equirect_parallel(&src, 64, 32, 0, 32, |u, v| (u, v)).is_err());
    }
}
