//! Forward (encoder-side) scalar quantisation for the CDF 9/7 lossy path.
//!
//! Inverse of [`super::tier1::CodeBlock::dequantize`]: given the f64 wavelet
//! coefficients of a single subband and the QCD-derived step size, produce the
//! signed integer (sign-magnitude as i32) coefficients that the existing
//! decoder dequantises back to the same f64 values.
//!
//! ## Mathematical model (ISO/IEC 15444-1 §E.1.1)
//!
//! The decoder dequantise formula (at [`super::tier1::CodeBlock::dequantize`])
//! reads:
//!
//! ```text
//!   if step_size ≈ 1.0:        c_f64 = v_i32  (lossless shortcut, cast)
//!   else:                       c_f64 = sign(v_i32) * |v_i32| * step_size * 2^(-N_b)
//! ```
//!
//! where `N_b = num_bit_planes` is the number of magnitude bit-planes coded.
//! The forward quantiser is the **mid-tread** integer mapper that is the
//! exact inverse of this formula:
//!
//! ```text
//!   if step_size ≈ 1.0:        v_i32 = sign(c_f64) * round(|c_f64|)
//!   else:                       v_i32 = sign(c_f64) * floor(|c_f64| * 2^N_b / step_size + 0.5)
//! ```
//!
//! With Wave 10's chosen QCD policy (ε=8, μ=0 globally and the decoder's
//! `R_b = bit_depth` per [`super::markers::QcdMarker::step_size_for_subband`]),
//! `step_size = 2^(bit_depth − 8) · 1.0 = 1.0` for `bit_depth = 8`. That
//! activates the lossless shortcut path: the quantised value is the rounded
//! integer of the f64 wavelet coefficient.
//!
//! The output magnitudes are guaranteed to fit in `2^num_bit_planes` magnitude
//! bit-planes provided `|c_f64| < 2^num_bit_planes` (LL after forward 9/7
//! attenuates by `1/K²` per level, so this is satisfied for `bit_depth`-range
//! inputs).

/// Forward-quantise the f64 wavelet coefficients of one subband to signed
/// integer coefficients suitable for the EBCOT Tier-1 encoder.
///
/// `step_size` is the QCD-derived `Δ_b` (see
/// [`super::markers::QcdMarker::step_size_for_subband`]).
/// `num_bit_planes` is the number of magnitude bit-planes that will be coded
/// for this subband (matches the value the decoder passes to
/// [`super::tier1::CodeBlock::dequantize`]).
///
/// Returns a `Vec<i32>` of the same length as `coeffs`.
#[must_use]
pub fn quantize_subband_97(coeffs: &[f64], step_size: f64, num_bit_planes: u8) -> Vec<i32> {
    // The decoder applies a lossless shortcut when step_size is (approximately)
    // 1.0: it just casts the integer coefficient to f64. The forward inverse is
    // therefore a plain rounding to nearest integer.
    if (step_size - 1.0).abs() < 1e-10 {
        return coeffs
            .iter()
            .map(|&c| {
                let mag = c.abs();
                let q = (mag + 0.5).floor();
                let qi = q as i32;
                if c < 0.0 {
                    -qi
                } else {
                    qi
                }
            })
            .collect();
    }
    // Lossy mid-tread quantiser. The decoder dequant scale is
    //   scale = step_size * 2^(-N_b).
    // Its inverse is
    //   scale_inv = 2^N_b / step_size = 1 / scale.
    let scale_inv = (2.0f64).powi(i32::from(num_bit_planes)) / step_size;
    coeffs
        .iter()
        .map(|&c| {
            let mag = c.abs() * scale_inv;
            let q = (mag + 0.5).floor();
            let qi = q as i32;
            if c < 0.0 {
                -qi
            } else {
                qi
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::tier1::CodeBlock;

    #[test]
    fn quantize_then_dequantize_lossless_shortcut() {
        // step_size = 1.0 activates the decoder shortcut: encoder just rounds.
        let coeffs = [0.0, 1.7, -3.2, 100.4, -200.6, 255.0];
        let q = quantize_subband_97(&coeffs, 1.0, 8);
        assert_eq!(q, vec![0, 2, -3, 100, -201, 255]);

        // Decoder dequantize with step_size = 1.0 returns the same i32 cast to f64.
        let block = CodeBlock {
            coeffs: q.clone(),
            width: q.len(),
            height: 1,
        };
        let recovered = block.dequantize(1.0, 8);
        for (i, (&orig, &rec)) in coeffs.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= 0.5 + 1e-9,
                "sample {i}: orig {orig}, recovered {rec}"
            );
        }
    }

    #[test]
    fn quantize_then_dequantize_general_step_size() {
        let nbp = 8u8;
        let step_size = 2.0f64;
        let coeffs: Vec<f64> = (0..16).map(|i| (i as f64) * 0.01 - 0.07).collect();
        let q = quantize_subband_97(&coeffs, step_size, nbp);
        let block = CodeBlock {
            coeffs: q.clone(),
            width: q.len(),
            height: 1,
        };
        let recovered = block.dequantize(step_size, usize::from(nbp));
        // Mid-tread quantisation error is bounded by step/2, where
        // step = step_size * 2^(-N_b).
        let max_err = 0.5 * step_size * (0.5f64).powi(i32::from(nbp));
        for (i, (&orig, &rec)) in coeffs.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= max_err + 1e-9,
                "sample {i}: orig {orig}, rec {rec}, max_err {max_err}"
            );
        }
    }

    #[test]
    fn quantize_zero_input_yields_zero() {
        let coeffs = vec![0.0f64; 5];
        let q = quantize_subband_97(&coeffs, 1.0, 8);
        assert_eq!(q, vec![0i32; 5]);
        let q2 = quantize_subband_97(&coeffs, 2.5, 8);
        assert_eq!(q2, vec![0i32; 5]);
    }

    #[test]
    fn quantize_sign_preservation() {
        let coeffs = [1.5_f64, -1.5, 7.4, -7.4, 8.0, -8.0];
        let q = quantize_subband_97(&coeffs, 1.0, 8);
        for (c, qv) in coeffs.iter().zip(q.iter()) {
            assert_eq!(c.signum() as i32, qv.signum(), "sign mismatch for {c}");
        }
    }
}
