// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS Nonlinear Transform (NLT) reverse application.
//!
//! The NLT is an optional pre-processing step defined in ISO 21122-1:2019 §A.2.
//! When present (indicated by the NLT marker `0xFF15`), it maps the original
//! sample values through a piecewise nonlinear function before wavelet encoding.
//! The decoder must apply the inverse mapping after wavelet reconstruction.
//!
//! # NLT types
//!
//! - **None**: passthrough (most JPEG XS streams in broadcast use no NLT)
//! - **Quadratic**: piecewise-quadratic mapping to extend the dynamic range of
//!   quantised coefficients near the boundaries of the sample range (ISO 21122-1 §A.2.2)
//! - **Extended**: reserved for future nonlinear profiles
//!
//! ## Quadratic NLT — forward transform (applied by encoder)
//!
//! Given thresholds `T1 < T2 ≤ MaxVal` where `MaxVal = (1 << bit_depth) - 1`:
//!
//! ```text
//! forward(s):
//!   if s ≤ T1:   s' = s
//!   if T1 < s ≤ T2:
//!       s' = T1 + (s - T1)² / (T2 - T1)          [quadratic compression]
//!   if s > T2:
//!       s' = MaxVal - (MaxVal - s)² / (MaxVal + 1 - T2)   [symmetric upper region]
//! ```
//!
//! The decoder applies the exact inverse of each region.  Because the encoder
//! uses integer (floor) division, the inverse must use ceiling-sqrt to find
//! the minimum source value that round-trips correctly (see `isqrt64_ceil`).

use super::JxsError;

/// NLT type selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NltType {
    /// No nonlinear transform applied — passthrough.
    None,
    /// Piecewise-quadratic NLT (ISO 21122-1 §A.2.2).
    Quadratic,
    /// Extended NLT (reserved).
    Extended,
}

/// Parameters for the Nonlinear Transform.
#[derive(Debug, Clone, Copy)]
pub struct NltParams {
    /// NLT variant selector.
    pub nlt_type: NltType,
    /// Lower threshold `T1` for the quadratic mapping.
    pub t1: u16,
    /// Upper threshold `T2` for the quadratic mapping.
    pub t2: u16,
}

impl NltParams {
    /// Construct a passthrough (no NLT) parameter set.
    pub fn none() -> Self {
        Self {
            nlt_type: NltType::None,
            t1: 0,
            t2: 0,
        }
    }

    /// Construct a quadratic NLT parameter set with specified thresholds.
    pub fn quadratic(t1: u16, t2: u16) -> Self {
        Self {
            nlt_type: NltType::Quadratic,
            t1,
            t2,
        }
    }
}

/// Integer floor-square-root for `u64`: largest `x` with `x² ≤ n`.
///
/// Uses a floating-point seed followed by at most two Newton correction steps
/// to guarantee the exact floor.  No `unsafe` code; the final result is always
/// verified algebraically.
fn isqrt64_floor(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = (n as f64).sqrt() as u64;
    // Pull down if seed is too high (f64 rounding can produce x where x²>n).
    while x > 0 && x.saturating_mul(x) > n {
        x = (x + n / x) / 2;
    }
    // Push up if floor is actually higher than the seed.
    while let Some(xp1) = x.checked_add(1) {
        if xp1.saturating_mul(xp1) <= n {
            x = xp1;
        } else {
            break;
        }
    }
    x
}

/// Integer ceiling-square-root for `u64`: smallest `x` with `x² ≥ n`.
///
/// Defined as `floor(sqrt(n - 1)) + 1` for `n ≥ 1`, which correctly handles
/// perfect squares (e.g. `isqrt64_ceil(9) = 3`, `isqrt64_ceil(10) = 4`).
fn isqrt64_ceil(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    // floor(sqrt(n - 1)) + 1 gives the ceiling for all n >= 1.
    isqrt64_floor(n - 1) + 1
}

/// Apply the inverse quadratic NLT to a single sample value `s_prime`.
///
/// The encoder applied integer-division forward transform:
/// - Mid:  `s' = T1 + floor((s - T1)² / (T2 - T1))`
/// - High: `s' = MaxVal - floor((MaxVal - s)² / (MaxVal + 1 - T2))`
///
/// The inverse must find the smallest `s` in each region whose forward transform
/// equals `s'`.  For integer-division encoders this requires **ceiling** sqrt:
///
/// - Low  (`s' ≤ T1`): `s = s'`  (identity)
/// - Mid  (`T1 < s' ≤ T2`):
///   `s = T1 + ceil(sqrt((s' - T1) * (T2 - T1)))`
/// - High (`s' > T2`):
///   `s = MaxVal - ceil(sqrt((MaxVal - s') * (MaxVal + 1 - T2)))`
///
/// This guarantees `forward(s) == s'` for every integer s' that the encoder
/// could have produced.  All intermediates use `u64` arithmetic to avoid
/// overflow for 16-bit sample depths.
fn nlt_quadratic_inverse(s_prime: i32, t1: i32, t2: i32, max_val: i32) -> i32 {
    if s_prime <= t1 {
        // Low region — encoder applied identity.
        s_prime
    } else if s_prime <= t2 {
        // Mid region:  s' - T1 = floor((s - T1)^2 / scale)
        // => s = T1 + ceil(sqrt((s' - T1) * scale))
        let delta = (s_prime - t1) as u64;
        let scale = (t2 - t1) as u64;
        t1 + isqrt64_ceil(delta * scale) as i32
    } else {
        // High region: MaxVal - s' = floor((MaxVal - s)^2 / A)
        // => s = MaxVal - ceil(sqrt((MaxVal - s') * A))
        let delta = (max_val - s_prime) as u64;
        let denom = (max_val + 1 - t2) as u64;
        max_val - isqrt64_ceil(delta * denom) as i32
    }
}

/// Apply the NLT reverse transform to a buffer of reconstructed wavelet samples.
///
/// The transform is applied in-place. For `NltType::None`, the function is a
/// no-op and returns `Ok(())` immediately.
///
/// # Arguments
/// - `samples`: mutable slice of reconstructed integer samples (one component).
/// - `params`: NLT parameters including type, T1, T2.
/// - `bit_depth`: sample bit depth (used to compute `MaxVal` and validate params).
///
/// # Errors
/// - `JxsError::InvalidHeader` — quadratic params violate `T1 < T2 ≤ MaxVal`.
/// - `JxsError::Unsupported`   — `NltType::Extended` is not yet implemented.
pub fn apply_nlt_reverse(
    samples: &mut [i32],
    params: &NltParams,
    bit_depth: u8,
) -> Result<(), JxsError> {
    match params.nlt_type {
        NltType::None => Ok(()),
        NltType::Quadratic => {
            // MaxVal = (1 << bit_depth) - 1. saturating_sub guards bit_depth == 0
            // (which is already rejected upstream, but defensive here).
            let max_val = ((1u32 << bit_depth) as i32).saturating_sub(1);
            let t1 = i32::from(params.t1);
            let t2 = i32::from(params.t2);
            // ISO 21122-1 §A.2.2 mandates T1 < T2 and T2 ≤ MaxVal.
            if t1 >= t2 || t2 > max_val {
                return Err(JxsError::InvalidHeader(format!(
                    "NLT quadratic: T1={t1} T2={t2} MaxVal={max_val} — must have T1 < T2 ≤ MaxVal"
                )));
            }
            for s in samples.iter_mut() {
                *s = nlt_quadratic_inverse(*s, t1, t2, max_val);
            }
            Ok(())
        }
        NltType::Extended => {
            let _ = (samples, bit_depth);
            Err(JxsError::Unsupported(
                "NLT extended reverse transform not yet implemented".to_string(),
            ))
        }
    }
}

/// Parse an NLT marker payload to extract `NltParams`.
///
/// ISO 21122-1 NLT payload layout (after the 2-byte length field):
/// - `Tnlt` 1 byte — NLT type (0 = quadratic, 1 = extended; absence of NLT marker = none)
/// - `T1`   2 bytes — lower threshold
/// - `T2`   2 bytes — upper threshold
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if the payload is too short.
pub fn parse_nlt_payload(payload: &[u8]) -> Result<NltParams, JxsError> {
    if payload.len() < 5 {
        return Err(JxsError::InvalidHeader(format!(
            "NLT payload too short: {} < 5 bytes",
            payload.len()
        )));
    }
    let tnlt = payload[0];
    let t1 = u16::from_be_bytes([payload[1], payload[2]]);
    let t2 = u16::from_be_bytes([payload[3], payload[4]]);
    let nlt_type = match tnlt {
        0 => NltType::Quadratic,
        1 => NltType::Extended,
        _ => {
            return Err(JxsError::InvalidHeader(format!(
                "NLT Tnlt={tnlt}: unknown NLT type"
            )));
        }
    };
    Ok(NltParams { nlt_type, t1, t2 })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────────

    /// Apply the forward quadratic NLT (encoder side) to a single sample,
    /// using the same three-region piecewise formula as ISO 21122-1 §A.2.2.
    /// Used exclusively in tests to generate s' from s for round-trip checks.
    fn nlt_quadratic_forward(s: i32, t1: i32, t2: i32, max_val: i32) -> i32 {
        if s <= t1 {
            s
        } else if s <= t2 {
            // mid: s' = T1 + (s - T1)² / (T2 - T1)
            let num = (s - t1) as i64 * (s - t1) as i64;
            let den = (t2 - t1) as i64;
            t1 + (num / den) as i32
        } else {
            // high: s' = MaxVal - (MaxVal - s)² / (MaxVal + 1 - T2)
            let num = (max_val - s) as i64 * (max_val - s) as i64;
            let den = (max_val + 1 - t2) as i64;
            max_val - (num / den) as i32
        }
    }

    // ── isqrt64_floor ────────────────────────────────────────────────────────

    #[test]
    fn isqrt64_floor_zero() {
        assert_eq!(isqrt64_floor(0), 0);
    }

    #[test]
    fn isqrt64_floor_perfect_squares() {
        for k in 0u64..=1024 {
            let n = k * k;
            assert_eq!(isqrt64_floor(n), k, "isqrt64_floor({n}) should be {k}");
        }
    }

    #[test]
    fn isqrt64_floor_non_perfect() {
        assert_eq!(isqrt64_floor(2), 1);
        assert_eq!(isqrt64_floor(3), 1);
        assert_eq!(isqrt64_floor(8), 2);
        assert_eq!(isqrt64_floor(10), 3);
        // Large value: floor(sqrt(65535^2)) = 65535
        assert_eq!(isqrt64_floor(65535 * 65535), 65535);
    }

    // ── isqrt64_ceil ─────────────────────────────────────────────────────────

    #[test]
    fn isqrt64_ceil_zero() {
        assert_eq!(isqrt64_ceil(0), 0);
    }

    #[test]
    fn isqrt64_ceil_perfect_squares() {
        // ceil(sqrt(k²)) = k for all k.
        for k in 0u64..=1024 {
            let n = k * k;
            assert_eq!(isqrt64_ceil(n), k, "isqrt64_ceil({n}) should be {k}");
        }
    }

    #[test]
    fn isqrt64_ceil_non_perfect() {
        // ceil(sqrt(2)) = 2
        assert_eq!(isqrt64_ceil(2), 2);
        // ceil(sqrt(3)) = 2
        assert_eq!(isqrt64_ceil(3), 2);
        // ceil(sqrt(5)) = 3
        assert_eq!(isqrt64_ceil(5), 3);
        // ceil(sqrt(8)) = 3
        assert_eq!(isqrt64_ceil(8), 3);
        // ceil(sqrt(10)) = 4
        assert_eq!(isqrt64_ceil(10), 4);
        // ceil(sqrt(128)) = 12 (since 11²=121<128<144=12²)
        assert_eq!(isqrt64_ceil(128), 12);
    }

    // ── nlt_none ─────────────────────────────────────────────────────────────

    #[test]
    fn nlt_none_is_passthrough() {
        let mut samples = vec![100i32, 200, 300];
        let params = NltParams::none();
        apply_nlt_reverse(&mut samples, &params, 8).unwrap();
        assert_eq!(samples, vec![100, 200, 300]);
    }

    // ── nlt_quadratic identity at boundaries ─────────────────────────────────

    #[test]
    fn nlt_quadratic_identity_below_t1() {
        // s ≤ T1: low region is identity, so inverse(0) = 0.
        let params = NltParams::quadratic(64, 192);
        let mut samples = vec![0i32, 32, 64];
        apply_nlt_reverse(&mut samples, &params, 8).unwrap();
        assert_eq!(samples, vec![0, 32, 64]);
    }

    #[test]
    fn nlt_quadratic_identity_at_t1_boundary() {
        // s' = T1 falls exactly on the low-region boundary.
        // forward(T1) = T1, so inverse(T1) = T1.
        let t1: u16 = 64;
        let t2: u16 = 192;
        let params = NltParams::quadratic(t1, t2);
        let mut samples = vec![i32::from(t1)];
        apply_nlt_reverse(&mut samples, &params, 8).unwrap();
        assert_eq!(samples[0], i32::from(t1));
    }

    // ── nlt_quadratic source-driven round-trip ────────────────────────────────
    //
    // The encoder's integer-division forward transform is NOT injective (multiple
    // source values can map to the same s').  The correct way to test the inverse
    // is therefore source-driven:
    //
    //   for each source value s:
    //     compute s' = forward(s)    (encoder side, uses integer div)
    //     compute r  = inverse(s')   (decoder — our implementation)
    //     verify:  forward(r) == s'  (r is a valid preimage of s')
    //
    // This guarantees that every reconstructed value the decoder produces
    // faithfully represents the encoded coefficient.

    #[test]
    fn nlt_quadratic_middle_region_roundtrip_8bit() {
        // 8-bit: MaxVal = 255, T1 = 64, T2 = 192
        let bit_depth = 8u8;
        let max_val = 255i32;
        let t1 = 64i32;
        let t2 = 192i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in (t1 + 1)..=t2 {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            // r must be in [0, MaxVal].
            assert!(
                r >= 0 && r <= max_val,
                "s={s}, s'={s_prime}: r={r} out of range"
            );
            // forward(r) must reproduce s' exactly.
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "s={s}, s'={s_prime}: inverse gave r={r}, forward(r)={reencoded} ≠ s'"
            );
        }
    }

    #[test]
    fn nlt_quadratic_upper_region_roundtrip_8bit() {
        // 8-bit: MaxVal = 255, T1 = 64, T2 = 192
        let bit_depth = 8u8;
        let max_val = 255i32;
        let t1 = 64i32;
        let t2 = 192i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in (t2 + 1)..=max_val {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            assert!(
                r >= 0 && r <= max_val,
                "s={s}, s'={s_prime}: r={r} out of range"
            );
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "s={s}, s'={s_prime}: inverse gave r={r}, forward(r)={reencoded} ≠ s'"
            );
        }
    }

    #[test]
    fn nlt_quadratic_middle_region_roundtrip_10bit() {
        // 10-bit: MaxVal = 1023, T1 = 128, T2 = 768
        let bit_depth = 10u8;
        let max_val = 1023i32;
        let t1 = 128i32;
        let t2 = 768i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in ((t1 + 1)..=t2).step_by(7) {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            assert!(
                r >= 0 && r <= max_val,
                "10-bit s={s}, s'={s_prime}: r={r} out of range"
            );
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "10-bit s={s}, s'={s_prime}: inverse gave r={r}, forward(r)={reencoded} ≠ s'"
            );
        }
    }

    #[test]
    fn nlt_quadratic_upper_region_roundtrip_10bit() {
        // 10-bit: MaxVal = 1023, T1 = 128, T2 = 768
        let bit_depth = 10u8;
        let max_val = 1023i32;
        let t1 = 128i32;
        let t2 = 768i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in ((t2 + 1)..=max_val).step_by(7) {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            assert!(
                r >= 0 && r <= max_val,
                "10-bit s={s}, s'={s_prime}: r={r} out of range"
            );
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "10-bit s={s}, s'={s_prime}: inverse gave r={r}, forward(r)={reencoded} ≠ s'"
            );
        }
    }

    #[test]
    fn nlt_quadratic_t1_zero_low_region_identity() {
        // T1 = 0: the low-region degenerates to only s' = 0, which is identity.
        let bit_depth = 8u8;
        let params = NltParams::quadratic(0, 128);

        let mut buf = vec![0i32];
        apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn nlt_quadratic_t1_zero_mid_region_roundtrip() {
        // T1 = 0, T2 = 128, MaxVal = 255: source-driven round-trip for mid region.
        let bit_depth = 8u8;
        let max_val = 255i32;
        let t1 = 0i32;
        let t2 = 128i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in 1..=t2 {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "t1=0 mid: s={s}, s'={s_prime}, r={r}, forward(r)={reencoded}"
            );
        }
    }

    #[test]
    fn nlt_quadratic_t1_zero_upper_region_roundtrip() {
        // T1 = 0, T2 = 128, MaxVal = 255: source-driven round-trip for upper region.
        let bit_depth = 8u8;
        let max_val = 255i32;
        let t1 = 0i32;
        let t2 = 128i32;
        let params = NltParams::quadratic(t1 as u16, t2 as u16);

        for s in (t2 + 1)..=max_val {
            let s_prime = nlt_quadratic_forward(s, t1, t2, max_val);
            let mut buf = vec![s_prime];
            apply_nlt_reverse(&mut buf, &params, bit_depth).unwrap();
            let r = buf[0];
            let reencoded = nlt_quadratic_forward(r, t1, t2, max_val);
            assert_eq!(
                reencoded, s_prime,
                "t1=0 upper: s={s}, s'={s_prime}, r={r}, forward(r)={reencoded}"
            );
        }
    }

    // ── nlt_quadratic invalid params ─────────────────────────────────────────

    #[test]
    fn nlt_invalid_params_t1_equals_t2_returns_error() {
        let params = NltParams::quadratic(128, 128); // T1 == T2: invalid
        let mut samples = vec![50i32];
        let result = apply_nlt_reverse(&mut samples, &params, 8);
        assert!(
            matches!(result, Err(JxsError::InvalidHeader(_))),
            "expected InvalidHeader, got {result:?}"
        );
    }

    #[test]
    fn nlt_invalid_params_t1_greater_than_t2_returns_error() {
        let params = NltParams::quadratic(200, 100); // T1 > T2: invalid
        let mut samples = vec![50i32];
        let result = apply_nlt_reverse(&mut samples, &params, 8);
        assert!(
            matches!(result, Err(JxsError::InvalidHeader(_))),
            "expected InvalidHeader, got {result:?}"
        );
    }

    #[test]
    fn nlt_invalid_params_t2_exceeds_max_val_returns_error() {
        // T2 = 256 > MaxVal = 255 for 8-bit
        let params = NltParams::quadratic(64, 256);
        let mut samples = vec![50i32];
        let result = apply_nlt_reverse(&mut samples, &params, 8);
        assert!(
            matches!(result, Err(JxsError::InvalidHeader(_))),
            "expected InvalidHeader, got {result:?}"
        );
    }

    // ── nlt_extended still returns Unsupported ────────────────────────────────

    #[test]
    fn nlt_extended_returns_unsupported() {
        let mut samples = vec![10i32];
        let params = NltParams {
            nlt_type: NltType::Extended,
            t1: 0,
            t2: 0,
        };
        let result = apply_nlt_reverse(&mut samples, &params, 8);
        assert!(result.is_err());
        assert!(matches!(result, Err(JxsError::Unsupported(_))));
    }

    // ── parse_nlt_payload ────────────────────────────────────────────────────

    #[test]
    fn parse_nlt_payload_quadratic() {
        // Tnlt=0 (quadratic), T1=100, T2=900
        let payload = [0x00u8, 0x00, 0x64, 0x03, 0x84];
        let params = parse_nlt_payload(&payload).unwrap();
        assert_eq!(params.nlt_type, NltType::Quadratic);
        assert_eq!(params.t1, 100);
        assert_eq!(params.t2, 900);
    }

    #[test]
    fn parse_nlt_payload_extended() {
        // Tnlt=1 (extended), T1=0, T2=0
        let payload = [0x01u8, 0x00, 0x00, 0x00, 0x00];
        let params = parse_nlt_payload(&payload).unwrap();
        assert_eq!(params.nlt_type, NltType::Extended);
    }

    #[test]
    fn parse_nlt_payload_unknown_tnlt_returns_error() {
        let payload = [0x05u8, 0x00, 0x00, 0x00, 0x00];
        assert!(parse_nlt_payload(&payload).is_err());
    }

    #[test]
    fn parse_nlt_payload_too_short_returns_error() {
        let payload = [0x00u8, 0x01]; // only 2 bytes
        assert!(parse_nlt_payload(&payload).is_err());
    }
}
