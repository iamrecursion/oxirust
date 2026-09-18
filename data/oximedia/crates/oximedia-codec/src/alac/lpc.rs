//! Dynamic predictor for ALAC (`dp_dec`/`dp_enc` in Apple's reference).
//!
//! ALAC predicts each sample from `order` previous samples using a fixed-point
//! FIR filter whose coefficients adapt sample-by-sample with a sign-sign LMS
//! rule. The decoder integrates residuals into samples; the encoder differences
//! samples into residuals using the **identical** adaptation, so it is the
//! exact inverse.
//!
//! # Structure
//!
//! For each output index `i >= order + 1` the predictor uses a base sample
//! `top = out[i - order - 1]` and predicts:
//!
//! ```text
//! acc  = (1 << (denshift - 1)) + Σ_j coefs[j] * (out[i-1-j] - top)
//! pred = top + (acc >> denshift)
//! ```
//!
//! Decode reconstructs `out[i] = pred + residual[i]`; encode produces
//! `residual[i] = out[i] - pred`. Both then adapt the coefficients using only
//! the (shared) history and the sign of the residual, so the two stay in
//! lockstep. The first `order + 1` samples are warmed up with a first-order
//! difference integrator, exactly as Apple does.

use super::{AlacError, AlacResult};

/// Maximum predictor order supported (Apple uses up to 31; we keep a generous
/// bound that comfortably covers the standard coefficient sets).
pub const MAX_COEFS: usize = 32;

/// Arithmetic right shift that rounds toward negative infinity (matches C's
/// `>>` on signed integers, which ALAC relies upon).
#[inline]
fn arith_shift(value: i64, shift: u32) -> i64 {
    value >> shift
}

/// Clamp/sign-extend a reconstructed sample to `chan_bits` bits.
#[inline]
fn extend_sample(value: i64, chan_bits: u32) -> i32 {
    super::bitstream::sign_extend((value as u32) & mask_bits(chan_bits), chan_bits)
}

#[inline]
fn mask_bits(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[inline]
fn sign_of(v: i64) -> i32 {
    if v > 0 {
        1
    } else if v < 0 {
        -1
    } else {
        0
    }
}

/// Decode a block of residuals into samples (`unpc_block`).
///
/// * `residuals` — the entropy-decoded prediction residuals (length `num`).
/// * `coefs` — initial predictor coefficients (length = predictor order). The
///   slice is mutated in place as it adapts; callers pass a fresh copy per
///   subframe.
/// * `chan_bits` — sample width used for sign-extension/wrap.
/// * `denshift` — fixed-point denominator shift for the FIR sum.
///
/// Returns the reconstructed samples.
pub fn predict_decode(
    residuals: &[i32],
    coefs: &mut [i32],
    chan_bits: u32,
    denshift: u32,
) -> AlacResult<Vec<i32>> {
    let num = residuals.len();
    let order = coefs.len();
    if order > MAX_COEFS {
        return Err(AlacError::InvalidBitstream(format!(
            "predictor order {order} exceeds {MAX_COEFS}"
        )));
    }
    let mut out = vec![0i32; num];
    if num == 0 {
        return Ok(out);
    }

    // First residual is stored verbatim.
    out[0] = residuals[0];

    if order == 0 {
        // No prediction: residuals are the samples.
        out[1..].copy_from_slice(&residuals[1..]);
        return Ok(out);
    }

    // Warm-up: integrate first-order differences for the first `order` samples.
    let mut i = 1usize;
    while i <= order && i < num {
        let prev = i64::from(out[i - 1]);
        let val = prev + i64::from(residuals[i]);
        out[i] = extend_sample(val, chan_bits);
        i += 1;
    }

    let denhalf: i64 = if denshift == 0 {
        0
    } else {
        1i64 << (denshift - 1)
    };

    // Main adaptive loop.
    while i < num {
        let top = i64::from(out[i - order - 1]);
        let mut acc = denhalf;
        for j in 0..order {
            acc += i64::from(coefs[j]) * (i64::from(out[i - 1 - j]) - top);
        }
        let pred = top + arith_shift(acc, denshift);
        let residual = i64::from(residuals[i]);
        let reconstructed = pred + residual;
        out[i] = extend_sample(reconstructed, chan_bits);

        adapt_coefs(coefs, &out, i, top, residuals[i]);
        i += 1;
    }

    Ok(out)
}

/// Encode a block of samples into residuals (`pc_block`), mirroring
/// [`predict_decode`] exactly so the pair round-trips losslessly.
pub fn predict_encode(
    samples: &[i32],
    coefs: &mut [i32],
    chan_bits: u32,
    denshift: u32,
) -> Vec<i32> {
    let num = samples.len();
    let order = coefs.len();
    let mut residuals = vec![0i32; num];
    if num == 0 {
        return residuals;
    }

    residuals[0] = samples[0];

    if order == 0 {
        residuals[1..].copy_from_slice(&samples[1..]);
        return residuals;
    }

    // Warm-up: first-order difference.
    let mut i = 1usize;
    while i <= order && i < num {
        let diff = i64::from(samples[i]) - i64::from(samples[i - 1]);
        residuals[i] = extend_sample(diff, chan_bits);
        i += 1;
    }

    let denhalf: i64 = if denshift == 0 {
        0
    } else {
        1i64 << (denshift - 1)
    };

    while i < num {
        let top = i64::from(samples[i - order - 1]);
        let mut acc = denhalf;
        for j in 0..order {
            acc += i64::from(coefs[j]) * (i64::from(samples[i - 1 - j]) - top);
        }
        let pred = top + arith_shift(acc, denshift);
        let residual = i64::from(samples[i]) - pred;
        let residual = extend_sample(residual, chan_bits);
        residuals[i] = residual;

        adapt_coefs(coefs, samples, i, top, residual);
        i += 1;
    }

    residuals
}

/// Sign-sign LMS coefficient adaptation, shared by encode and decode.
///
/// `history` is `out`/`samples` (identical on both sides up to index `i`),
/// `top` is the base sample, and `residual` is the just-computed prediction
/// error. Each coefficient nudges by `sign(residual) * sign(history_delta)`.
#[inline]
fn adapt_coefs(coefs: &mut [i32], history: &[i32], i: usize, top: i64, residual: i32) {
    let err_sign = sign_of(i64::from(residual));
    if err_sign == 0 {
        return;
    }
    let order = coefs.len();
    for j in 0..order {
        let delta = i64::from(history[i - 1 - j]) - top;
        let d_sign = sign_of(delta);
        coefs[j] += err_sign * d_sign;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(samples: &[i32], order: usize, chan_bits: u32, denshift: u32) {
        let init = vec![0i32; order];
        let mut enc_coefs = init.clone();
        let residuals = predict_encode(samples, &mut enc_coefs, chan_bits, denshift);
        let mut dec_coefs = init;
        let decoded = predict_decode(&residuals, &mut dec_coefs, chan_bits, denshift).expect("dec");
        assert_eq!(decoded, samples, "predictor round-trip mismatch");
        // Coefficients must have evolved identically on both sides.
        assert_eq!(enc_coefs, dec_coefs, "coef divergence");
    }

    #[test]
    fn test_roundtrip_ramp_order4() {
        let samples: Vec<i32> = (0..256).map(|i| i * 7 - 400).collect();
        roundtrip(&samples, 4, 16, 4);
    }

    #[test]
    fn test_roundtrip_sine_order8() {
        let samples: Vec<i32> = (0..512)
            .map(|i| ((i as f64 * 0.05).sin() * 9000.0) as i32)
            .collect();
        roundtrip(&samples, 8, 16, 4);
    }

    #[test]
    fn test_roundtrip_order0_copy() {
        let samples: Vec<i32> = (0..64).map(|i| i * 3).collect();
        roundtrip(&samples, 0, 16, 4);
    }

    #[test]
    fn test_roundtrip_constant() {
        let samples = vec![1234i32; 200];
        roundtrip(&samples, 8, 16, 4);
    }

    #[test]
    fn test_roundtrip_24bit() {
        let samples: Vec<i32> = (0..300)
            .map(|i| ((i as f64 * 0.02).sin() * 4_000_000.0) as i32)
            .collect();
        roundtrip(&samples, 8, 24, 4);
    }

    #[test]
    fn test_arith_shift_negative() {
        assert_eq!(arith_shift(-8, 1), -4);
        assert_eq!(arith_shift(-1, 1), -1); // floor division
        assert_eq!(arith_shift(7, 2), 1);
    }

    #[test]
    fn test_order_too_large_errors() {
        let mut coefs = vec![0i32; MAX_COEFS + 1];
        let res = predict_decode(&[0i32; 4], &mut coefs, 16, 4);
        assert!(res.is_err());
    }
}
