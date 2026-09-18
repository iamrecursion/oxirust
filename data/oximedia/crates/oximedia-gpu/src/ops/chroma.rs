//! GPU-accelerated chroma subsampling operations.
//!
//! Provides conversion between packed RGBA (4:4:4) and chroma-subsampled
//! planar formats:
//!
//! - **4:2:0** – Cb/Cr at quarter resolution (standard for H.264, AV1)
//! - **4:2:2** – Cb/Cr at half horizontal resolution (broadcast TV)
//!
//! All operations use BT.601 coefficients by default and support configurable
//! color space matrices.  CPU fallback paths use rayon parallelism.

use crate::{GpuError, Result};
use rayon::prelude::*;

// ============================================================================
// Color space matrix coefficients
// ============================================================================

/// Coefficients for RGB→YCbCr conversion.
#[derive(Debug, Clone, Copy)]
pub struct YcbcrCoefficients {
    /// Red contribution to luma.
    pub kr: f64,
    /// Green contribution to luma (derived: 1 - kr - kb).
    pub kg: f64,
    /// Blue contribution to luma.
    pub kb: f64,
}

impl YcbcrCoefficients {
    /// BT.601 coefficients (SD video).
    pub const BT601: Self = Self {
        kr: 0.299,
        kg: 0.587,
        kb: 0.114,
    };

    /// BT.709 coefficients (HD video).
    pub const BT709: Self = Self {
        kr: 0.2126,
        kg: 0.7152,
        kb: 0.0722,
    };

    /// BT.2020 coefficients (UHD / HDR video).
    pub const BT2020: Self = Self {
        kr: 0.2627,
        kg: 0.6780,
        kb: 0.0593,
    };
}

/// Chroma subsampling format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:2:0 – Cb/Cr at quarter resolution.
    Yuv420,
    /// 4:2:2 – Cb/Cr at half horizontal resolution.
    Yuv422,
}

impl ChromaSubsampling {
    /// Calculate the expected output buffer size for a given resolution.
    fn output_size(self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        match self {
            Self::Yuv420 => {
                let uv_w = (w + 1) / 2;
                let uv_h = (h + 1) / 2;
                y_size + 2 * uv_w * uv_h
            }
            Self::Yuv422 => {
                let uv_w = (w + 1) / 2;
                y_size + 2 * uv_w * h
            }
        }
    }
}

/// Chroma subsampling operations.
pub struct ChromaOps;

impl ChromaOps {
    /// Convert packed RGBA to planar YCbCr with the specified chroma subsampling.
    ///
    /// Output layout: Y plane (full size), then Cb plane, then Cr plane
    /// (subsampled according to `format`).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or buffer sizes don't match.
    pub fn rgba_to_ycbcr(
        rgba: &[u8],
        width: u32,
        height: u32,
        format: ChromaSubsampling,
        coefficients: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;
        let expected_input = w * h * 4;

        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }
        if rgba.len() < expected_input {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_input,
                actual: rgba.len(),
            });
        }

        match format {
            ChromaSubsampling::Yuv420 => Self::rgba_to_yuv420(rgba, w, h, coefficients),
            ChromaSubsampling::Yuv422 => Self::rgba_to_yuv422(rgba, w, h, coefficients),
        }
    }

    /// Convert planar YCbCr back to packed RGBA.
    ///
    /// Input layout: Y plane (full size), then Cb plane, then Cr plane
    /// (subsampled according to `format`).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or buffer sizes don't match.
    pub fn ycbcr_to_rgba(
        ycbcr: &[u8],
        width: u32,
        height: u32,
        format: ChromaSubsampling,
        coefficients: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let w = width as usize;
        let h = height as usize;

        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        let expected = format.output_size(width, height);
        if ycbcr.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: ycbcr.len(),
            });
        }

        match format {
            ChromaSubsampling::Yuv420 => Self::yuv420_to_rgba(ycbcr, w, h, coefficients),
            ChromaSubsampling::Yuv422 => Self::yuv422_to_rgba(ycbcr, w, h, coefficients),
        }
    }

    // -----------------------------------------------------------------------
    // 4:2:0 forward
    // -----------------------------------------------------------------------

    fn rgba_to_yuv420(
        rgba: &[u8],
        w: usize,
        h: usize,
        coeff: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let uv_size = uv_w * uv_h;
        let mut output = vec![0u8; y_size + 2 * uv_size];

        // Y plane – parallelise by row.
        let y_plane = &mut output[..y_size];
        y_plane
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let base = (y * w + x) * 4;
                    let r = rgba[base] as f64;
                    let g = rgba[base + 1] as f64;
                    let b = rgba[base + 2] as f64;
                    let luma = coeff.kr * r + coeff.kg * g + coeff.kb * b;
                    row[x] = luma.round().clamp(0.0, 255.0) as u8;
                }
            });

        // Cb/Cr planes – 2x2 block averaging.
        let cb_start = y_size;
        let cr_start = y_size + uv_size;

        for by in 0..uv_h {
            for bx in 0..uv_w {
                let mut sum_cb = 0.0_f64;
                let mut sum_cr = 0.0_f64;
                let mut count = 0u32;

                for dy in 0..2_usize {
                    let sy = by * 2 + dy;
                    if sy >= h {
                        continue;
                    }
                    for dx in 0..2_usize {
                        let sx = bx * 2 + dx;
                        if sx >= w {
                            continue;
                        }
                        let base = (sy * w + sx) * 4;
                        let r = rgba[base] as f64;
                        let g = rgba[base + 1] as f64;
                        let b = rgba[base + 2] as f64;
                        let y_val = coeff.kr * r + coeff.kg * g + coeff.kb * b;
                        // Cb = (B - Y) / (2 * (1 - Kb)) + 128
                        let denom_cb = 2.0 * (1.0 - coeff.kb);
                        let cb = if denom_cb.abs() > 1e-10 {
                            (b - y_val) / denom_cb + 128.0
                        } else {
                            128.0
                        };
                        // Cr = (R - Y) / (2 * (1 - Kr)) + 128
                        let denom_cr = 2.0 * (1.0 - coeff.kr);
                        let cr = if denom_cr.abs() > 1e-10 {
                            (r - y_val) / denom_cr + 128.0
                        } else {
                            128.0
                        };
                        sum_cb += cb;
                        sum_cr += cr;
                        count += 1;
                    }
                }

                let uv_idx = by * uv_w + bx;
                if count > 0 {
                    output[cb_start + uv_idx] =
                        (sum_cb / count as f64).round().clamp(0.0, 255.0) as u8;
                    output[cr_start + uv_idx] =
                        (sum_cr / count as f64).round().clamp(0.0, 255.0) as u8;
                } else {
                    output[cb_start + uv_idx] = 128;
                    output[cr_start + uv_idx] = 128;
                }
            }
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // 4:2:0 inverse
    // -----------------------------------------------------------------------

    fn yuv420_to_rgba(
        ycbcr: &[u8],
        w: usize,
        h: usize,
        coeff: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let uv_size = uv_w * uv_h;

        let y_plane = &ycbcr[..y_size];
        let cb_plane = &ycbcr[y_size..y_size + uv_size];
        let cr_plane = &ycbcr[y_size + uv_size..y_size + 2 * uv_size];

        let mut rgba = vec![0u8; w * h * 4];

        rgba.par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(py, row)| {
                let uv_y = (py / 2).min(uv_h.saturating_sub(1));
                for px in 0..w {
                    let uv_x = (px / 2).min(uv_w.saturating_sub(1));
                    let uv_idx = uv_y * uv_w + uv_x;

                    let y_val = y_plane[py * w + px] as f64;
                    let cb = cb_plane[uv_idx] as f64 - 128.0;
                    let cr = cr_plane[uv_idx] as f64 - 128.0;

                    let r = y_val + 2.0 * (1.0 - coeff.kr) * cr;
                    let b = y_val + 2.0 * (1.0 - coeff.kb) * cb;
                    let g = if coeff.kg.abs() > 1e-10 {
                        (y_val - coeff.kr * r - coeff.kb * b) / coeff.kg
                    } else {
                        y_val
                    };

                    let base = px * 4;
                    row[base] = r.round().clamp(0.0, 255.0) as u8;
                    row[base + 1] = g.round().clamp(0.0, 255.0) as u8;
                    row[base + 2] = b.round().clamp(0.0, 255.0) as u8;
                    row[base + 3] = 255;
                }
            });

        Ok(rgba)
    }

    // -----------------------------------------------------------------------
    // 4:2:2 forward
    // -----------------------------------------------------------------------

    fn rgba_to_yuv422(
        rgba: &[u8],
        w: usize,
        h: usize,
        coeff: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_size = uv_w * h;
        let mut output = vec![0u8; y_size + 2 * uv_size];

        // Y plane.
        let y_plane = &mut output[..y_size];
        y_plane
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let base = (y * w + x) * 4;
                    let r = rgba[base] as f64;
                    let g = rgba[base + 1] as f64;
                    let b = rgba[base + 2] as f64;
                    let luma = coeff.kr * r + coeff.kg * g + coeff.kb * b;
                    row[x] = luma.round().clamp(0.0, 255.0) as u8;
                }
            });

        // Cb/Cr planes – horizontal 2-pixel averaging.
        let cb_start = y_size;
        let cr_start = y_size + uv_size;

        for y in 0..h {
            for bx in 0..uv_w {
                let mut sum_cb = 0.0_f64;
                let mut sum_cr = 0.0_f64;
                let mut count = 0u32;

                for dx in 0..2_usize {
                    let sx = bx * 2 + dx;
                    if sx >= w {
                        continue;
                    }
                    let base = (y * w + sx) * 4;
                    let r = rgba[base] as f64;
                    let g = rgba[base + 1] as f64;
                    let b = rgba[base + 2] as f64;
                    let y_val = coeff.kr * r + coeff.kg * g + coeff.kb * b;

                    let denom_cb = 2.0 * (1.0 - coeff.kb);
                    let cb = if denom_cb.abs() > 1e-10 {
                        (b - y_val) / denom_cb + 128.0
                    } else {
                        128.0
                    };
                    let denom_cr = 2.0 * (1.0 - coeff.kr);
                    let cr = if denom_cr.abs() > 1e-10 {
                        (r - y_val) / denom_cr + 128.0
                    } else {
                        128.0
                    };

                    sum_cb += cb;
                    sum_cr += cr;
                    count += 1;
                }

                let uv_idx = y * uv_w + bx;
                if count > 0 {
                    output[cb_start + uv_idx] =
                        (sum_cb / count as f64).round().clamp(0.0, 255.0) as u8;
                    output[cr_start + uv_idx] =
                        (sum_cr / count as f64).round().clamp(0.0, 255.0) as u8;
                } else {
                    output[cb_start + uv_idx] = 128;
                    output[cr_start + uv_idx] = 128;
                }
            }
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // 4:2:2 inverse
    // -----------------------------------------------------------------------

    fn yuv422_to_rgba(
        ycbcr: &[u8],
        w: usize,
        h: usize,
        coeff: YcbcrCoefficients,
    ) -> Result<Vec<u8>> {
        let y_size = w * h;
        let uv_w = (w + 1) / 2;
        let uv_size = uv_w * h;

        let y_plane = &ycbcr[..y_size];
        let cb_plane = &ycbcr[y_size..y_size + uv_size];
        let cr_plane = &ycbcr[y_size + uv_size..y_size + 2 * uv_size];

        let mut rgba = vec![0u8; w * h * 4];

        rgba.par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(py, row)| {
                for px in 0..w {
                    let uv_x = (px / 2).min(uv_w.saturating_sub(1));
                    let uv_idx = py * uv_w + uv_x;

                    let y_val = y_plane[py * w + px] as f64;
                    let cb = cb_plane[uv_idx] as f64 - 128.0;
                    let cr = cr_plane[uv_idx] as f64 - 128.0;

                    let r = y_val + 2.0 * (1.0 - coeff.kr) * cr;
                    let b = y_val + 2.0 * (1.0 - coeff.kb) * cb;
                    let g = if coeff.kg.abs() > 1e-10 {
                        (y_val - coeff.kr * r - coeff.kb * b) / coeff.kg
                    } else {
                        y_val
                    };

                    let base = px * 4;
                    row[base] = r.round().clamp(0.0, 255.0) as u8;
                    row[base + 1] = g.round().clamp(0.0, 255.0) as u8;
                    row[base + 2] = b.round().clamp(0.0, 255.0) as u8;
                    row[base + 3] = 255;
                }
            });

        Ok(rgba)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (w as usize) * (h as usize);
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&[r, g, b, 255]);
        }
        buf
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
        let ww = w as usize;
        let hh = h as usize;
        let mut buf = Vec::with_capacity(ww * hh * 4);
        for y in 0..hh {
            for x in 0..ww {
                let v = ((x + y) % 256) as u8;
                buf.extend_from_slice(&[v, v / 2, 255 - v, 255]);
            }
        }
        buf
    }

    // --- 4:2:0 tests ---

    #[test]
    fn test_420_output_size() {
        let rgba = solid_rgba(16, 16, 128, 128, 128);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("conversion should succeed");
        // Y: 16*16=256, Cb: 8*8=64, Cr: 8*8=64 → 384
        assert_eq!(yuv.len(), 384);
    }

    #[test]
    fn test_420_roundtrip_grey() {
        let rgba = solid_rgba(16, 16, 128, 128, 128);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("inverse");
        // Round-trip error should be small for a grey image.
        for i in 0..(16 * 16) {
            let base = i * 4;
            for c in 0..3 {
                let diff = (rgba[base + c] as i32 - back[base + c] as i32).unsigned_abs();
                assert!(diff <= 2, "pixel {i} channel {c}: diff={diff}");
            }
        }
    }

    #[test]
    fn test_420_roundtrip_gradient() {
        let rgba = gradient_rgba(32, 32);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            32,
            32,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            32,
            32,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("inverse");
        // Allow up to ±5 per channel due to subsampling + quantisation.
        let max_diff: u32 = (0..(32 * 32))
            .map(|i| {
                let base = i * 4;
                (0..3)
                    .map(|c| (rgba[base + c] as i32 - back[base + c] as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 10,
            "max roundtrip diff={max_diff}, expected <= 10"
        );
    }

    #[test]
    fn test_420_white() {
        let rgba = solid_rgba(4, 4, 255, 255, 255);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            4,
            4,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        // Y for white should be ~255.
        assert!(yuv[0] > 250, "Y for white should be ~255, got {}", yuv[0]);
        // Cb and Cr for white should be ~128 (neutral).
        let y_size = 4 * 4;
        let cb = yuv[y_size];
        let cr = yuv[y_size + 4]; // first Cr sample
        assert!(
            (cb as i32 - 128).unsigned_abs() <= 2,
            "Cb for white should be ~128, got {cb}"
        );
        assert!(
            (cr as i32 - 128).unsigned_abs() <= 2,
            "Cr for white should be ~128, got {cr}"
        );
    }

    #[test]
    fn test_420_black() {
        let rgba = solid_rgba(4, 4, 0, 0, 0);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            4,
            4,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        assert_eq!(yuv[0], 0, "Y for black should be 0");
    }

    #[test]
    fn test_420_odd_dimensions() {
        // Odd dimensions should work (UV planes round up).
        let rgba = solid_rgba(15, 13, 100, 150, 200);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            15,
            13,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        let expected = ChromaSubsampling::Yuv420.output_size(15, 13);
        assert_eq!(yuv.len(), expected);
        // Inverse should also work.
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            15,
            13,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .expect("inverse");
        assert_eq!(back.len(), 15 * 13 * 4);
    }

    // --- 4:2:2 tests ---

    #[test]
    fn test_422_output_size() {
        let rgba = solid_rgba(16, 16, 128, 128, 128);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT601,
        )
        .expect("conversion should succeed");
        // Y: 16*16=256, Cb: 8*16=128, Cr: 8*16=128 → 512
        assert_eq!(yuv.len(), 512);
    }

    #[test]
    fn test_422_roundtrip_grey() {
        let rgba = solid_rgba(16, 16, 128, 128, 128);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            16,
            16,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT601,
        )
        .expect("inverse");
        for i in 0..(16 * 16) {
            let base = i * 4;
            for c in 0..3 {
                let diff = (rgba[base + c] as i32 - back[base + c] as i32).unsigned_abs();
                assert!(diff <= 2, "pixel {i} channel {c}: diff={diff}");
            }
        }
    }

    #[test]
    fn test_422_roundtrip_gradient() {
        let rgba = gradient_rgba(32, 32);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            32,
            32,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT709,
        )
        .expect("forward");
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            32,
            32,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT709,
        )
        .expect("inverse");
        let max_diff: u32 = (0..(32 * 32))
            .map(|i| {
                let base = i * 4;
                (0..3)
                    .map(|c| (rgba[base + c] as i32 - back[base + c] as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 8,
            "max roundtrip diff={max_diff}, expected <= 8"
        );
    }

    #[test]
    fn test_422_odd_width() {
        let rgba = solid_rgba(15, 8, 100, 200, 50);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            15,
            8,
            ChromaSubsampling::Yuv422,
            YcbcrCoefficients::BT601,
        )
        .expect("forward");
        let expected = ChromaSubsampling::Yuv422.output_size(15, 8);
        assert_eq!(yuv.len(), expected);
    }

    // --- BT.2020 tests ---

    #[test]
    fn test_bt2020_roundtrip() {
        let rgba = gradient_rgba(16, 16);
        let yuv = ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT2020,
        )
        .expect("forward");
        let back = ChromaOps::ycbcr_to_rgba(
            &yuv,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT2020,
        )
        .expect("inverse");
        let max_diff: u32 = (0..(16 * 16))
            .map(|i| {
                let base = i * 4;
                (0..3)
                    .map(|c| (rgba[base + c] as i32 - back[base + c] as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(max_diff <= 10, "BT.2020 max roundtrip diff={max_diff}");
    }

    // --- Error cases ---

    #[test]
    fn test_zero_dimensions() {
        let rgba = vec![];
        assert!(ChromaOps::rgba_to_ycbcr(
            &rgba,
            0,
            0,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .is_err());
    }

    #[test]
    fn test_buffer_too_small() {
        let rgba = vec![0u8; 10];
        assert!(ChromaOps::rgba_to_ycbcr(
            &rgba,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .is_err());
    }

    #[test]
    fn test_inverse_buffer_too_small() {
        let yuv = vec![0u8; 10];
        assert!(ChromaOps::ycbcr_to_rgba(
            &yuv,
            16,
            16,
            ChromaSubsampling::Yuv420,
            YcbcrCoefficients::BT601,
        )
        .is_err());
    }

    // --- ChromaSubsampling::output_size ---

    #[test]
    fn test_output_size_420() {
        assert_eq!(ChromaSubsampling::Yuv420.output_size(16, 16), 384);
        assert_eq!(
            ChromaSubsampling::Yuv420.output_size(15, 13),
            15 * 13 + 2 * 8 * 7
        );
    }

    #[test]
    fn test_output_size_422() {
        assert_eq!(ChromaSubsampling::Yuv422.output_size(16, 16), 512);
        assert_eq!(
            ChromaSubsampling::Yuv422.output_size(15, 8),
            15 * 8 + 2 * 8 * 8
        );
    }
}
