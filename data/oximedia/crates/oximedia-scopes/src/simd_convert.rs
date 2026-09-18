//! SIMD-optimized RGB-to-YCbCr conversion for waveform and vectorscope generation.
//!
//! This module provides high-throughput batch conversion between color spaces,
//! structured for auto-vectorization by the compiler. All operations are strictly
//! pure Rust with no `unsafe` blocks; the data layout and loop structure are
//! chosen so that LLVM can apply SIMD widening automatically.
//!
//! # Color Space Mathematics
//!
//! Conversion follows ITU-R BT.709 (HD) primaries:
//!
//! ```text
//! Y  =  0.2126 R + 0.7152 G + 0.0722 B
//! Cb = -0.1146 R - 0.3854 G + 0.5000 B  (+128 bias for 8-bit)
//! Cr =  0.5000 R - 0.4542 G - 0.0458 B  (+128 bias for 8-bit)
//! ```
//!
//! Integer fixed-point coefficients scaled by 2^15 (32768) are used to avoid
//! floating-point in the hot path while maintaining sub-0.5 LSB accuracy.

/// Number of pixels processed per "SIMD lane" in the batch path.
/// Matches typical AVX2 width for 32-bit lanes.
pub const BATCH_SIZE: usize = 8;

/// Converted YCbCr triplet (8-bit, broadcast-legal 16–235 / 16–240 range
/// is **not** applied here; raw full-range values are returned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YcbcrPixel {
    /// Luma component (0–255, BT.709 full range).
    pub y: u8,
    /// Cb (blue-difference chroma) component, biased by 128.
    pub cb: u8,
    /// Cr (red-difference chroma) component, biased by 128.
    pub cr: u8,
}

/// BT.709 fixed-point coefficients scaled by 2^15.
///
/// Values chosen so that rounding errors do not exceed ±0.5 LSB across
/// the full 0–255 input range.
mod coeff709 {
    /// Y  from R: round(0.2126 * 32768)
    pub const KR: i32 = 6967;
    /// Y  from G: round(0.7152 * 32768)
    pub const KG: i32 = 23434;
    /// Y  from B: round(0.0722 * 32768)
    pub const KB: i32 = 2367;

    /// Cb from R: round(-0.1146 * 32768)
    pub const CB_R: i32 = -3755;
    /// Cb from G: round(-0.3854 * 32768)
    pub const CB_G: i32 = -12629;
    /// Cb from B: round( 0.5000 * 32768)
    pub const CB_B: i32 = 16384;

    /// Cr from R: round( 0.5000 * 32768)
    pub const CR_R: i32 = 16384;
    /// Cr from G: round(-0.4542 * 32768)
    pub const CR_G: i32 = -14882;
    /// Cr from B: round(-0.0458 * 32768)
    pub const CR_B: i32 = -1502;
}

/// Convert a single RGB triplet to BT.709 YCbCr.
///
/// Returns a [`YcbcrPixel`] with full-range (0–255) Y and biased (0–255)
/// Cb/Cr values (neutral grey maps to Cb=128, Cr=128).
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::rgb_to_ycbcr_pixel;
/// let p = rgb_to_ycbcr_pixel(255, 255, 255);
/// assert_eq!(p.y, 255);
/// ```
#[must_use]
pub fn rgb_to_ycbcr_pixel(r: u8, g: u8, b: u8) -> YcbcrPixel {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = (coeff709::KR * ri + coeff709::KG * gi + coeff709::KB * bi + (1 << 14)) >> 15;
    let cb = (coeff709::CB_R * ri + coeff709::CB_G * gi + coeff709::CB_B * bi + (1 << 14)) >> 15;
    let cr = (coeff709::CR_R * ri + coeff709::CR_G * gi + coeff709::CR_B * bi + (1 << 14)) >> 15;

    YcbcrPixel {
        y: y.clamp(0, 255) as u8,
        cb: (cb + 128).clamp(0, 255) as u8,
        cr: (cr + 128).clamp(0, 255) as u8,
    }
}

/// Convert a batch of RGB pixels to YCbCr in one call.
///
/// The input slice must be RGB-interleaved (`[R, G, B, R, G, B, …]`).
/// The output vector will contain one [`YcbcrPixel`] per input pixel.
/// Incomplete trailing pixels (i.e. `frame.len() % 3 != 0`) are silently
/// ignored.
///
/// This function is structured for LLVM auto-vectorization: the inner loop
/// over a fixed `BATCH_SIZE` window processes 8 pixels in parallel, enabling
/// 256-bit AVX2 or 128-bit NEON widening without any `unsafe` code.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::convert_batch;
/// let rgb = vec![255u8, 0, 0,  0, 255, 0,  0, 0, 255];
/// let out = convert_batch(&rgb);
/// assert_eq!(out.len(), 3);
/// ```
#[must_use]
pub fn convert_batch(frame: &[u8]) -> Vec<YcbcrPixel> {
    let pixel_count = frame.len() / 3;
    let mut output = Vec::with_capacity(pixel_count);

    // Process pixels in chunks of BATCH_SIZE for auto-vectorization.
    let chunk_pixels = pixel_count / BATCH_SIZE;
    let remainder_start = chunk_pixels * BATCH_SIZE;

    for chunk_idx in 0..chunk_pixels {
        let base = chunk_idx * BATCH_SIZE;

        // Unrolled fixed-size array so LLVM can see the trip-count.
        let mut chunk_out = [YcbcrPixel {
            y: 0,
            cb: 128,
            cr: 128,
        }; BATCH_SIZE];
        for lane in 0..BATCH_SIZE {
            let off = (base + lane) * 3;
            let r = frame[off];
            let g = frame[off + 1];
            let b = frame[off + 2];
            chunk_out[lane] = rgb_to_ycbcr_pixel(r, g, b);
        }
        output.extend_from_slice(&chunk_out);
    }

    // Handle the remaining pixels (< BATCH_SIZE).
    for px in remainder_start..pixel_count {
        let off = px * 3;
        output.push(rgb_to_ycbcr_pixel(
            frame[off],
            frame[off + 1],
            frame[off + 2],
        ));
    }

    output
}

/// Extract the luma channel from a batch conversion result into a flat `Vec<u8>`.
///
/// This is a convenience wrapper for the common case of waveform generation
/// where only the Y channel is needed.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::extract_luma;
/// let rgb = vec![128u8, 128, 128, 128, 128, 128];
/// let luma = extract_luma(&rgb);
/// assert_eq!(luma.len(), 2);
/// ```
#[must_use]
pub fn extract_luma(frame: &[u8]) -> Vec<u8> {
    convert_batch(frame).into_iter().map(|p| p.y).collect()
}

/// Extract Cb and Cr channels (interleaved `[Cb, Cr, Cb, Cr, …]`) from a
/// full RGB frame.  Used by the vectorscope for fast UV-plane construction.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::extract_cbcr;
/// let rgb = vec![128u8, 128, 128];
/// let cbcr = extract_cbcr(&rgb);
/// assert_eq!(cbcr.len(), 2); // one Cb + one Cr per pixel
/// ```
#[must_use]
pub fn extract_cbcr(frame: &[u8]) -> Vec<u8> {
    let pixels = convert_batch(frame);
    let mut out = Vec::with_capacity(pixels.len() * 2);
    for p in &pixels {
        out.push(p.cb);
        out.push(p.cr);
    }
    out
}

/// BT.2020 fixed-point coefficients scaled by 2^15 for wide-gamut HDR content.
mod coeff2020 {
    /// Y  from R: round(0.2627 * 32768)
    pub const KR: i32 = 8610;
    /// Y  from G: round(0.6780 * 32768)
    pub const KG: i32 = 22216;
    /// Y  from B: round(0.0593 * 32768)
    pub const KB: i32 = 1943;

    /// Cb from R: round(-0.1396 * 32768)
    pub const CB_R: i32 = -4574;
    /// Cb from G: round(-0.3604 * 32768)
    pub const CB_G: i32 = -11810;
    /// Cb from B: round( 0.5000 * 32768)
    pub const CB_B: i32 = 16384;

    /// Cr from R: round( 0.5000 * 32768)
    pub const CR_R: i32 = 16384;
    /// Cr from G: round(-0.4598 * 32768)
    pub const CR_G: i32 = -15066;
    /// Cr from B: round(-0.0402 * 32768)
    pub const CR_B: i32 = -1317;
}

/// Convert a single RGB triplet to BT.2020 YCbCr (for HDR / wide-gamut scopes).
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::rgb_to_ycbcr_bt2020;
/// let p = rgb_to_ycbcr_bt2020(0, 0, 0);
/// assert_eq!(p.y, 0);
/// assert_eq!(p.cb, 128);
/// assert_eq!(p.cr, 128);
/// ```
#[must_use]
pub fn rgb_to_ycbcr_bt2020(r: u8, g: u8, b: u8) -> YcbcrPixel {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = (coeff2020::KR * ri + coeff2020::KG * gi + coeff2020::KB * bi + (1 << 14)) >> 15;
    let cb = (coeff2020::CB_R * ri + coeff2020::CB_G * gi + coeff2020::CB_B * bi + (1 << 14)) >> 15;
    let cr = (coeff2020::CR_R * ri + coeff2020::CR_G * gi + coeff2020::CR_B * bi + (1 << 14)) >> 15;

    YcbcrPixel {
        y: y.clamp(0, 255) as u8,
        cb: (cb + 128).clamp(0, 255) as u8,
        cr: (cr + 128).clamp(0, 255) as u8,
    }
}

/// Batch-convert a frame using BT.2020 primaries.
///
/// Semantics are identical to [`convert_batch`] but using BT.2020 coefficients.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::convert_batch_bt2020;
/// let rgb = vec![255u8, 255, 255, 0, 0, 0];
/// let out = convert_batch_bt2020(&rgb);
/// assert_eq!(out.len(), 2);
/// assert_eq!(out[0].y, 255);
/// assert_eq!(out[1].y, 0);
/// ```
#[must_use]
pub fn convert_batch_bt2020(frame: &[u8]) -> Vec<YcbcrPixel> {
    let pixel_count = frame.len() / 3;
    let mut output = Vec::with_capacity(pixel_count);

    let chunk_pixels = pixel_count / BATCH_SIZE;
    let remainder_start = chunk_pixels * BATCH_SIZE;

    for chunk_idx in 0..chunk_pixels {
        let base = chunk_idx * BATCH_SIZE;
        let mut chunk_out = [YcbcrPixel {
            y: 0,
            cb: 128,
            cr: 128,
        }; BATCH_SIZE];
        for lane in 0..BATCH_SIZE {
            let off = (base + lane) * 3;
            chunk_out[lane] = rgb_to_ycbcr_bt2020(frame[off], frame[off + 1], frame[off + 2]);
        }
        output.extend_from_slice(&chunk_out);
    }

    for px in remainder_start..pixel_count {
        let off = px * 3;
        output.push(rgb_to_ycbcr_bt2020(
            frame[off],
            frame[off + 1],
            frame[off + 2],
        ));
    }

    output
}

// ── BT.601 SIMD-dispatched RGB-to-YCbCr ────────────────────────────────────

/// BT.601 fixed-point coefficients scaled by 2^15.
///
/// BT.601 is used in standard-definition broadcast and is the traditional
/// coefficients for waveform/vectorscope analysis:
/// ```text
/// Y  =  0.299  R + 0.587  G + 0.114  B
/// Cb = -0.16874 R - 0.33126 G + 0.5    B  (+128 bias)
/// Cr =  0.5    R - 0.41869 G - 0.08131 B  (+128 bias)
/// ```
mod coeff601 {
    /// Y from R: round(0.299 * 32768)
    pub const KR: i32 = 9798;
    /// Y from G: round(0.587 * 32768)
    pub const KG: i32 = 19235;
    /// Y from B: round(0.114 * 32768)
    pub const KB: i32 = 3736;

    /// Cb from R: round(-0.16874 * 32768)
    pub const CB_R: i32 = -5529;
    /// Cb from G: round(-0.33126 * 32768)
    pub const CB_G: i32 = -10855;
    /// Cb from B: round(0.5 * 32768)
    pub const CB_B: i32 = 16384;

    /// Cr from R: round(0.5 * 32768)
    pub const CR_R: i32 = 16384;
    /// Cr from G: round(-0.41869 * 32768)
    pub const CR_G: i32 = -13717;
    /// Cr from B: round(-0.08131 * 32768)
    pub const CR_B: i32 = -2664;
}

/// Convert a single RGB triplet to BT.601 YCbCr (scalar path).
///
/// Returns packed `[Y, Cb, Cr]` as `[u8; 3]`.
/// Neutral grey maps to `Cb = 128, Cr = 128`.
#[must_use]
pub fn rgb_to_ycbcr_bt601_pixel(r: u8, g: u8, b: u8) -> [u8; 3] {
    let ri = i32::from(r);
    let gi = i32::from(g);
    let bi = i32::from(b);

    let y = (coeff601::KR * ri + coeff601::KG * gi + coeff601::KB * bi + (1 << 14)) >> 15;
    let cb = (coeff601::CB_R * ri + coeff601::CB_G * gi + coeff601::CB_B * bi + (1 << 14)) >> 15;
    let cr = (coeff601::CR_R * ri + coeff601::CR_G * gi + coeff601::CR_B * bi + (1 << 14)) >> 15;

    [
        y.clamp(0, 255) as u8,
        (cb + 128).clamp(0, 255) as u8,
        (cr + 128).clamp(0, 255) as u8,
    ]
}

/// Scalar BT.601 batch conversion — processes `pixels` slice without any
/// architecture-specific intrinsics.  Used in tests as the reference path.
#[must_use]
#[cfg(test)]
fn rgb_to_ycbcr_scalar_bt601(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    pixels
        .iter()
        .map(|&[r, g, b]| rgb_to_ycbcr_bt601_pixel(r, g, b))
        .collect()
}

/// Number of pixels per SIMD batch for auto-vectorized BT.601 conversion.
/// A width-8 window matches AVX2's 8×i32 lane count; LLVM will widen
/// scalar fixed-point code to 256-bit vectors on x86-64 and 128-bit on AArch64.
const BT601_BATCH: usize = 8;

/// Process exactly `BT601_BATCH` pixels of BT.601 conversion in a single
/// tight, fixed-trip-count inner loop that LLVM can auto-vectorize.
///
/// The function is `#[inline(always)]` so the caller's loop is the outer
/// iterator; LLVM sees a perfectly nested loop with a constant inner bound
/// and emits SIMD instructions automatically.
#[inline(always)]
fn convert_bt601_batch(chunk: &[[u8; 3]; BT601_BATCH]) -> [[u8; 3]; BT601_BATCH] {
    let mut out = [[0u8; 3]; BT601_BATCH];
    for lane in 0..BT601_BATCH {
        out[lane] = rgb_to_ycbcr_bt601_pixel(chunk[lane][0], chunk[lane][1], chunk[lane][2]);
    }
    out
}

/// BT.601 SIMD-optimized RGB-to-YCbCr conversion.
///
/// Returns packed `[Y, Cb, Cr]` for each input pixel, using BT.601 primaries
/// (traditional broadcast SD coefficients, also used for waveform/vectorscope
/// displays in many broadcast monitors).
///
/// The implementation uses an 8-pixel fixed-width inner loop that the Rust
/// compiler (LLVM) auto-vectorizes to AVX2 (x86-64) or NEON (AArch64)
/// without requiring `unsafe` code.  The trailing scalar path handles
/// any residual pixels when the total count is not a multiple of 8.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::rgb_to_ycbcr_simd;
/// let pixels: Vec<[u8; 3]> = vec![[255, 255, 255], [0, 0, 0]];
/// let out = rgb_to_ycbcr_simd(&pixels);
/// assert_eq!(out.len(), 2);
/// // White maps to Y ≈ 255, neutral Cb/Cr.
/// assert_eq!(out[0][0], 255);
/// assert!((out[0][1] as i32 - 128).abs() <= 1);
/// ```
#[must_use]
pub fn rgb_to_ycbcr_simd(pixels: &[[u8; 3]]) -> Vec<[u8; 3]> {
    let n = pixels.len();
    let mut output = Vec::with_capacity(n);

    let full_chunks = n / BT601_BATCH;
    for c in 0..full_chunks {
        let base = c * BT601_BATCH;
        // Build a fixed-size array so LLVM can determine the inner trip-count
        // at compile time and emit vectorized code.
        let mut chunk = [[0u8; 3]; BT601_BATCH];
        for lane in 0..BT601_BATCH {
            chunk[lane] = pixels[base + lane];
        }
        let converted = convert_bt601_batch(&chunk);
        output.extend_from_slice(&converted);
    }

    // Scalar tail for any remaining pixels.
    for px in (full_chunks * BT601_BATCH)..n {
        output.push(rgb_to_ycbcr_bt601_pixel(
            pixels[px][0],
            pixels[px][1],
            pixels[px][2],
        ));
    }

    output
}

/// Convert a flat RGB-interleaved frame buffer to BT.601 YCbCr.
///
/// Returns packed `[Y, Cb, Cr]` triplets, one per source pixel.
/// The implementation uses the same 8-pixel auto-vectorizable inner loop as
/// [`rgb_to_ycbcr_simd`] but reads directly from the flat `&[u8]` slice,
/// avoiding an intermediate allocation.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::simd_convert::convert_batch_bt601_simd;
/// let rgb = vec![255u8, 0, 0,  0, 255, 0,  0, 0, 255];
/// let out = convert_batch_bt601_simd(&rgb);
/// assert_eq!(out.len(), 3);
/// ```
#[must_use]
pub fn convert_batch_bt601_simd(frame: &[u8]) -> Vec<[u8; 3]> {
    let pixel_count = frame.len() / 3;
    let mut output = Vec::with_capacity(pixel_count);

    let full_chunks = pixel_count / BT601_BATCH;
    for c in 0..full_chunks {
        let base = c * BT601_BATCH * 3;
        let mut chunk = [[0u8; 3]; BT601_BATCH];
        for lane in 0..BT601_BATCH {
            let off = base + lane * 3;
            chunk[lane] = [frame[off], frame[off + 1], frame[off + 2]];
        }
        let converted = convert_bt601_batch(&chunk);
        output.extend_from_slice(&converted);
    }

    // Scalar tail.
    for px in (full_chunks * BT601_BATCH)..pixel_count {
        let off = px * 3;
        output.push(rgb_to_ycbcr_bt601_pixel(
            frame[off],
            frame[off + 1],
            frame[off + 2],
        ));
    }

    output
}

/// YCbCr conversion accuracy metric used in benchmarks and validation.
///
/// Returns the maximum absolute difference between scalar reference and
/// fixed-point batch conversion for a given test frame.
#[must_use]
pub fn max_error_vs_float(frame: &[u8]) -> f64 {
    let pixel_count = frame.len() / 3;
    let mut max_err: f64 = 0.0;

    for px in 0..pixel_count {
        let off = px * 3;
        let r = frame[off] as f64;
        let g = frame[off + 1] as f64;
        let b = frame[off + 2] as f64;

        let y_ref = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let cb_ref = -0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0;
        let cr_ref = 0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0;

        let fp = rgb_to_ycbcr_pixel(frame[off], frame[off + 1], frame[off + 2]);

        let dy = (fp.y as f64 - y_ref.clamp(0.0, 255.0)).abs();
        let dcb = (fp.cb as f64 - cb_ref.clamp(0.0, 255.0)).abs();
        let dcr = (fp.cr as f64 - cr_ref.clamp(0.0, 255.0)).abs();

        max_err = max_err.max(dy).max(dcb).max(dcr);
    }

    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure-black should give Y=0, Cb=128, Cr=128.
    #[test]
    fn test_black_pixel() {
        let p = rgb_to_ycbcr_pixel(0, 0, 0);
        assert_eq!(p.y, 0);
        assert_eq!(p.cb, 128);
        assert_eq!(p.cr, 128);
    }

    /// Pure-white should give Y=255, Cb≈128, Cr≈128.
    #[test]
    fn test_white_pixel() {
        let p = rgb_to_ycbcr_pixel(255, 255, 255);
        assert_eq!(p.y, 255);
        // Chroma should stay near neutral (128) for achromatic colours.
        assert!((p.cb as i32 - 128).abs() <= 1, "cb={}", p.cb);
        assert!((p.cr as i32 - 128).abs() <= 1, "cr={}", p.cr);
    }

    /// Pure-red in BT.709 has significant positive Cr.
    #[test]
    fn test_red_pixel_chroma() {
        let p = rgb_to_ycbcr_pixel(255, 0, 0);
        // Y ≈ 54 (0.2126 * 255)
        assert!((p.y as i32 - 54).abs() <= 1, "y={}", p.y);
        // Cr should be the highest chroma component for red.
        assert!(
            p.cr > p.cb,
            "expected Cr > Cb for red, got cr={} cb={}",
            p.cr,
            p.cb
        );
    }

    /// Pure-blue in BT.709 has significant positive Cb.
    #[test]
    fn test_blue_pixel_chroma() {
        let p = rgb_to_ycbcr_pixel(0, 0, 255);
        // Cb should be the highest chroma component for blue.
        assert!(
            p.cb > p.cr,
            "expected Cb > Cr for blue, got cb={} cr={}",
            p.cb,
            p.cr
        );
    }

    /// `convert_batch` on an empty frame returns an empty vec.
    #[test]
    fn test_batch_empty() {
        let out = convert_batch(&[]);
        assert!(out.is_empty());
    }

    /// `convert_batch` on a single pixel equals `rgb_to_ycbcr_pixel`.
    #[test]
    fn test_batch_single_pixel_matches_scalar() {
        let rgb = [200u8, 100, 50];
        let batch = convert_batch(&rgb);
        let scalar = rgb_to_ycbcr_pixel(200, 100, 50);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], scalar);
    }

    /// Batch conversion with a stride that is not a multiple of BATCH_SIZE
    /// must still produce the right number of pixels.
    #[test]
    fn test_batch_non_multiple_of_batch_size() {
        // 11 pixels → not divisible by BATCH_SIZE (8)
        let pixel_count = 11_usize;
        let rgb: Vec<u8> = (0..pixel_count * 3).map(|i| (i % 256) as u8).collect();
        let out = convert_batch(&rgb);
        assert_eq!(out.len(), pixel_count);
    }

    /// Fixed-point batch results must agree with reference float within 1 LSB.
    #[test]
    fn test_batch_accuracy_within_one_lsb() {
        // Ramp through all grey levels (achromatic – hardest for chroma).
        let rgb: Vec<u8> = (0..=255).flat_map(|v| [v, v, v]).collect();
        let err = max_error_vs_float(&rgb);
        assert!(err <= 1.0, "max error {err} exceeds 1 LSB");
    }

    /// BT.2020 black should also be Y=0, Cb=128, Cr=128.
    #[test]
    fn test_bt2020_black_pixel() {
        let p = rgb_to_ycbcr_bt2020(0, 0, 0);
        assert_eq!(p.y, 0);
        assert_eq!(p.cb, 128);
        assert_eq!(p.cr, 128);
    }

    /// BT.2020 white should be Y=255, neutral chroma.
    #[test]
    fn test_bt2020_white_pixel() {
        let p = rgb_to_ycbcr_bt2020(255, 255, 255);
        assert_eq!(p.y, 255);
        assert!((p.cb as i32 - 128).abs() <= 1, "cb={}", p.cb);
        assert!((p.cr as i32 - 128).abs() <= 1, "cr={}", p.cr);
    }

    /// `extract_luma` length must equal pixel_count.
    #[test]
    fn test_extract_luma_length() {
        let rgb = vec![0u8; 30]; // 10 pixels
        let luma = extract_luma(&rgb);
        assert_eq!(luma.len(), 10);
    }

    /// `extract_cbcr` length must be 2× pixel_count.
    #[test]
    fn test_extract_cbcr_length() {
        let rgb = vec![0u8; 30]; // 10 pixels
        let cbcr = extract_cbcr(&rgb);
        assert_eq!(cbcr.len(), 20);
    }

    /// BT.2020 batch on 9 pixels (non-aligned) returns 9 pixels.
    #[test]
    fn test_bt2020_batch_non_aligned() {
        let rgb: Vec<u8> = (0..27).collect(); // 9 pixels
        let out = convert_batch_bt2020(&rgb);
        assert_eq!(out.len(), 9);
    }

    // ── Item 1 tests (BT.601 SIMD dispatch) ──────────────────────────────────

    /// SIMD dispatch result must match the scalar reference for every pixel
    /// across a comprehensive ramp (all grey levels × representative colors).
    #[test]
    fn test_rgb_to_ycbcr_simd_matches_scalar() {
        // Grey ramp
        let mut pixels: Vec<[u8; 3]> = (0u8..=255).map(|v| [v, v, v]).collect();
        // Add primaries and complementaries
        pixels.extend_from_slice(&[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [128, 64, 32],
        ]);

        let simd_out = rgb_to_ycbcr_simd(&pixels);
        let scalar_out = rgb_to_ycbcr_scalar_bt601(&pixels);

        assert_eq!(simd_out.len(), scalar_out.len());
        for (i, (s, r)) in simd_out.iter().zip(scalar_out.iter()).enumerate() {
            assert_eq!(s, r, "pixel {i}: SIMD {:?} != scalar {:?}", s, r);
        }
    }

    /// White pixel with BT.601 must produce Y=255, Cb≈128, Cr≈128.
    #[test]
    fn test_rgb_to_ycbcr_white_pixel() {
        let pixels = vec![[255u8, 255, 255]];
        let out = rgb_to_ycbcr_simd(&pixels);
        assert_eq!(out.len(), 1);
        let [y, cb, cr] = out[0];
        assert_eq!(y, 255, "white Y should be 255");
        assert!(
            (cb as i32 - 128).abs() <= 1,
            "white Cb should be ~128, got {cb}"
        );
        assert!(
            (cr as i32 - 128).abs() <= 1,
            "white Cr should be ~128, got {cr}"
        );
    }

    /// Black pixel with BT.601 must produce Y=0, Cb=128, Cr=128.
    #[test]
    fn test_rgb_to_ycbcr_black_pixel_bt601() {
        let pixels = vec![[0u8, 0, 0]];
        let out = rgb_to_ycbcr_simd(&pixels);
        let [y, cb, cr] = out[0];
        assert_eq!(y, 0, "black Y should be 0");
        assert_eq!(cb, 128, "black Cb should be 128");
        assert_eq!(cr, 128, "black Cr should be 128");
    }

    /// `convert_batch_bt601_simd` flat-frame wrapper must agree with per-pixel scalar.
    #[test]
    fn test_convert_batch_bt601_simd_wrapper() {
        let frame: Vec<u8> = (0..30).collect(); // 10 pixels
        let batch = convert_batch_bt601_simd(&frame);
        let scalar: Vec<[u8; 3]> = (0..10)
            .map(|i| rgb_to_ycbcr_bt601_pixel(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]))
            .collect();
        assert_eq!(batch, scalar);
    }

    /// SIMD path on an empty slice returns an empty vec without panicking.
    #[test]
    fn test_rgb_to_ycbcr_simd_empty() {
        let out = rgb_to_ycbcr_simd(&[]);
        assert!(out.is_empty());
    }
}
