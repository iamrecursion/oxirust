//! Inter-channel decorrelation for ALAC stereo pairs (`matrix_dec`/`matrix_enc`).
//!
//! For a 2-channel element ALAC recombines left/right into two coded channels
//! using a weighted mid/side predictor parameterised by `mix_bits` (the shift)
//! and `mix_res` (the weight). The transform has an exact integer inverse.
//!
//! # Forward ([`mix_stereo`])
//!
//! With `mod = 1 << mix_bits` and `m2 = mod - mix_res`:
//!
//! ```text
//! u[j] = (mix_res * l + m2 * r) >> mix_bits     // weighted "mid"
//! v[j] = l - r                                   // "side"
//! ```
//!
//! When `mix_res == 0` the channels pass through unchanged (`u = l`, `v = r`).
//!
//! # Inverse ([`unmix_stereo`])
//!
//! ```text
//! l = u[j] + v[j] - ((mix_res * v[j]) >> mix_bits)
//! r = l - v[j]
//! ```
//!
//! This reproduces the original `l`/`r` exactly because `mod * r` contributes
//! zero to the low `mix_bits` bits, so the arithmetic shift distributes over
//! the addition.

/// Decorrelate an interleaved stereo block into two coded channels.
///
/// `interleaved` holds `2 * num` samples (`l0, r0, l1, r1, …`). On return
/// `u`/`v` each hold `num` coded samples.
pub fn mix_stereo(
    interleaved: &[i32],
    num: usize,
    mix_bits: u32,
    mix_res: i32,
    u: &mut [i32],
    v: &mut [i32],
) {
    if mix_res != 0 {
        let mod_v: i64 = 1i64 << mix_bits;
        let m2: i64 = mod_v - i64::from(mix_res);
        let res = i64::from(mix_res);
        for j in 0..num {
            let l = i64::from(interleaved[2 * j]);
            let r = i64::from(interleaved[2 * j + 1]);
            u[j] = ((res * l + m2 * r) >> mix_bits) as i32;
            v[j] = (l - r) as i32;
        }
    } else {
        for j in 0..num {
            u[j] = interleaved[2 * j];
            v[j] = interleaved[2 * j + 1];
        }
    }
}

/// Inverse of [`mix_stereo`]: recombine the two coded channels into
/// interleaved stereo.
pub fn unmix_stereo(
    u: &[i32],
    v: &[i32],
    num: usize,
    mix_bits: u32,
    mix_res: i32,
    interleaved: &mut [i32],
) {
    if mix_res != 0 {
        let res = i64::from(mix_res);
        // Defend against a shift overflow: `mix_bits` comes from the frame
        // header and shifting an i64 by >= 64 panics. Valid ALAC mix_bits is
        // small (<= 31), so clamping to 63 never affects a well-formed stream.
        let shift = mix_bits.min(63);
        for j in 0..num {
            let uj = i64::from(u[j]);
            let vj = i64::from(v[j]);
            let l = uj + vj - ((res * vj) >> shift);
            let r = l - vj;
            interleaved[2 * j] = l as i32;
            interleaved[2 * j + 1] = r as i32;
        }
    } else {
        for j in 0..num {
            interleaved[2 * j] = u[j];
            interleaved[2 * j + 1] = v[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(interleaved: &[i32], mix_bits: u32, mix_res: i32) {
        let num = interleaved.len() / 2;
        let mut u = vec![0i32; num];
        let mut v = vec![0i32; num];
        mix_stereo(interleaved, num, mix_bits, mix_res, &mut u, &mut v);
        let mut out = vec![0i32; interleaved.len()];
        unmix_stereo(&u, &v, num, mix_bits, mix_res, &mut out);
        assert_eq!(out, interleaved, "mix round-trip mismatch (res={mix_res})");
    }

    #[test]
    fn test_passthrough_res0() {
        let data: Vec<i32> = (0..64).collect();
        roundtrip(&data, 8, 0);
    }

    #[test]
    fn test_matrixed_various_res() {
        let data: Vec<i32> = (0..128).map(|i| (i * 37) % 5000 - 2500).collect();
        for res in [1i32, 2, 4, 8, 100, 255] {
            roundtrip(&data, 8, res);
        }
    }

    #[test]
    fn test_matrixed_24bit_range() {
        let data: Vec<i32> = (0..200)
            .map(|i| {
                if i % 2 == 0 {
                    (i as i32) * 40_000 - 4_000_000
                } else {
                    -(i as i32) * 30_000 + 3_000_000
                }
            })
            .collect();
        for res in [1i32, 3, 7, 200] {
            roundtrip(&data, 8, res);
        }
    }

    #[test]
    fn test_identical_channels() {
        // L == R ⇒ side is zero; round-trip must still be exact.
        let data: Vec<i32> = (0..64).flat_map(|i| [i * 11, i * 11]).collect();
        roundtrip(&data, 8, 4);
    }
}
