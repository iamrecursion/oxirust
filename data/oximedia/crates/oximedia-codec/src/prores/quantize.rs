//! Forward quantization for ProRes 422 encoding.
//!
//! Each DCT coefficient is divided by `matrix[i] * qscale` using a
//! deadzone quantizer that matches the FFmpeg `prorescenc.c` approach:
//!
//! ```text
//!   q[i] = (coeff[i] * 4 + sign(coeff[i]) * (matrix[i] * qscale) / 2)
//!          / (matrix[i] * qscale)
//! ```
//!
//! This is equivalent to rounding toward the nearest quantization level
//! with a slight deadzone around zero, which is standard for DCT codecs.
//!
//! The input coefficients should be in raster order (not zigzag).

/// Quantize one 8×8 block of DCT coefficients.
///
/// `coeffs` is in **raster order** (same as the output of [`super::fdct::fdct_8x8`]).
/// `matrix` is the per-position quantization matrix (raster order), values 1..=255.
/// `qscale` is the per-slice quantization parameter; 0 is treated as 1 to avoid
/// division by zero.
///
/// Returns an array of quantized coefficients as `i16`, in raster order.
///
/// Uses a standard deadzone (round-half-away-from-zero) quantizer:
/// `q = (coeff + sign(coeff) * step/2) / step`
/// which matches the dequantize formula `coeff ≈ q * step`.
#[must_use]
pub fn quantize_block(coeffs: &[i32; 64], matrix: &[u8; 64], qscale: u8) -> [i16; 64] {
    let qs = i32::from(if qscale == 0 { 1 } else { qscale });
    let mut out = [0i16; 64];
    for i in 0..64 {
        let m = i32::from(matrix[i]);
        let step = m * qs;
        let c = coeffs[i];
        if step == 0 || c == 0 {
            out[i] = 0;
            continue;
        }
        // Deadzone quantizer: round half-steps away from zero.
        // sign(c) * (step / 2) biases the rounding away from zero.
        let sign = if c > 0 { 1i32 } else { -1i32 };
        let biased = c + sign * (step / 2);
        // Integer division truncates toward zero (Rust's default).
        let q = biased / step;
        out[i] = q.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::dequant::dequantize_block;

    #[test]
    fn quantize_zero_block_stays_zero() {
        let coeffs = [0i32; 64];
        let matrix = [4u8; 64];
        let q = quantize_block(&coeffs, &matrix, 6);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn quantize_then_dequantize_small_error() {
        // Create a test block with values in normal ProRes 422 range.
        let coeffs: [i32; 64] = std::array::from_fn(|i| (i as i32 * 7) - 224);
        let matrix: [u8; 64] = crate::prores::quant::DEFAULT_LUMA_QUANT_MATRIX;
        let qscale = 4u8;

        let quantized = quantize_block(&coeffs, &matrix, qscale);

        // Dequantize expects i32.
        let quantized_i32: [i32; 64] = std::array::from_fn(|i| quantized[i] as i32);
        let dequantized = dequantize_block(&quantized_i32, &matrix, qscale);

        // Error should be within ±matrix[i]*qscale for each position.
        for i in 0..64 {
            let step = i32::from(matrix[i]) * i32::from(qscale);
            let err = (dequantized[i] - coeffs[i]).abs();
            assert!(
                err <= step,
                "quant/dequant error at [{i}]: orig={}, dequant={}, step={}",
                coeffs[i],
                dequantized[i],
                step
            );
        }
    }

    #[test]
    fn quantize_preserves_sign() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 200;
        coeffs[1] = -200;
        let matrix = [4u8; 64];
        let q = quantize_block(&coeffs, &matrix, 4);
        assert!(q[0] > 0, "positive coeff should stay positive");
        assert!(q[1] < 0, "negative coeff should stay negative");
    }

    #[test]
    fn quantize_rounds_toward_zero() {
        // Very small coefficient compared to step → quantizes to zero.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1;
        let matrix = [4u8; 64];
        let q = quantize_block(&coeffs, &matrix, 16);
        // step = 4 * 16 = 64; coeff = 1 * 4 + 32 = 36; 36 / 64 = 0
        assert_eq!(q[0], 0, "small coeff should quantize to 0");
    }

    #[test]
    fn quantize_qscale_zero_treated_as_one() {
        let coeffs: [i32; 64] = std::array::from_fn(|i| i as i32 * 10);
        let matrix = [4u8; 64];
        let q0 = quantize_block(&coeffs, &matrix, 0);
        let q1 = quantize_block(&coeffs, &matrix, 1);
        assert_eq!(q0, q1, "qscale=0 should behave like qscale=1");
    }
}
