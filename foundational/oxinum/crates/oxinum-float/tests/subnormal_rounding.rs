//! `BigFloat::to_f64` at the IEEE-754 underflow and overflow boundaries.
//!
//! Converting to `f64` is the one place in the crate where an arbitrary-precision
//! value has to be squeezed onto a *fixed* grid, and the grid is not uniform:
//! normals carry 53 significant bits, while every subnormal shares the single
//! grid `n * 2^-1074`, so a value just under `f64::MIN_POSITIVE` is stored with
//! 52 bits, one just under `2^-1073` with 51, and so on down to a single bit at
//! `2^-1074`. Two things go wrong there and both are silent:
//!
//! * **Flush to zero.** Rounding is not the same as truncating the exponent
//!   range. A magnitude in `(2^-1075, 2^-1074)` is below every representable
//!   subnormal yet above *half* of the smallest one, so round-to-nearest owes
//!   it `2^-1074`, not zero. IEEE 754 calls the difference gradual underflow.
//!
//! * **Double rounding.** Rounding an already-rounded value narrows twice, and
//!   two correct roundings do not compose into one. The failure is not rare:
//!   for a value in `[2^-1023, 2^-1022)` the halfway points of the 52-bit
//!   subnormal grid are exactly the points of the 53-bit normal grid, so an
//!   intermediate 53-bit rounding lands on one about half the time, and
//!   ties-to-even then guesses wrong about half of *those*.
//!
//! The fixtures below pin both boundaries by construction, and the sweeps at
//! the end re-derive them from a fixed seed so the whole band — not just the
//! points someone thought to name — stays covered.

use oxinum_core::Sign;
use oxinum_float::native::{BigFloat, RoundingMode};
use oxinum_int::native::BigUint;

/// The exact value `m * 2^e`.
///
/// Precision is fixed at 64 so `from_parts` can only ever *pad* `m`: padding
/// preserves the value bit for bit and records no rounding direction, which is
/// what makes these fixtures usable as genuinely un-rounded inputs to the
/// conversion under test.
fn exact(m: u64, e: i64) -> BigFloat {
    BigFloat::from_parts(
        Sign::Positive,
        BigUint::from_u64(m),
        e,
        64,
        RoundingMode::HalfEven,
    )
}

/// `2^-1074`, the smallest positive `f64`.
const MIN_SUBNORMAL: f64 = 5e-324;
/// `(2^52 - 1) * 2^-1074`, the largest subnormal.
const MAX_SUBNORMAL_BITS: u64 = 0x000f_ffff_ffff_ffff;

// ---------------------------------------------------------------------------
// The underflow boundary: 2^-1075 (half the smallest subnormal)
// ---------------------------------------------------------------------------

#[test]
fn exact_half_of_min_subnormal_ties_to_even_zero() {
    // 2^-1075 is exactly halfway between 0 and 2^-1074. Zero has an even
    // significand, so ties-to-even keeps it.
    assert_eq!(exact(1, -1075).to_f64(), 0.0);
    // The same value written with a wider mantissa must decide the same way.
    assert_eq!(exact(1 << 20, -1095).to_f64(), 0.0);
    // Sign survives the underflow.
    let negative = exact(1, -1075).neg().to_f64();
    assert_eq!(negative, 0.0);
    assert!(
        negative.is_sign_negative(),
        "negative underflow must give -0.0"
    );
}

#[test]
fn just_above_half_of_min_subnormal_rounds_up() {
    // 3 * 2^-1076 = 1.5 * 2^-1075.
    assert_eq!(exact(3, -1076).to_f64(), MIN_SUBNORMAL);
    // The narrowest possible excess over the tie: 2^-1075 + 2^-1128.
    assert_eq!(exact((1u64 << 53) + 1, -1128).to_f64(), MIN_SUBNORMAL);
    assert_eq!(exact(3, -1076).neg().to_f64(), -MIN_SUBNORMAL);
}

#[test]
fn just_below_half_of_min_subnormal_rounds_to_zero() {
    // 2^-1075 - 2^-1128.
    assert_eq!(exact((1u64 << 53) - 1, -1128).to_f64(), 0.0);
    assert_eq!(exact(1, -1076).to_f64(), 0.0);
    // Deep underflow takes the shortcut path; it must agree with the general
    // one, sign included.
    assert_eq!(exact(1, -5000).to_f64(), 0.0);
    assert!(exact(1, -5000).neg().to_f64().is_sign_negative());
}

// ---------------------------------------------------------------------------
// Inside the subnormal grid
// ---------------------------------------------------------------------------

#[test]
fn subnormal_halfway_points_tie_to_even() {
    // Counted in units of 2^-1074, the halfway points are the odd multiples
    // of one half, and the tie must go to the even neighbour in *both*
    // directions rather than always up or always down.
    assert_eq!(exact(3, -1075).to_f64().to_bits(), 2, "1.5 -> 2");
    assert_eq!(exact(5, -1075).to_f64().to_bits(), 2, "2.5 -> 2");
    assert_eq!(exact(7, -1075).to_f64().to_bits(), 4, "3.5 -> 4");
    assert_eq!(exact(9, -1075).to_f64().to_bits(), 4, "4.5 -> 4");
}

#[test]
fn subnormal_near_ties_resolve_by_magnitude_not_parity() {
    // 2.5 units + 2^-11 of a unit, and 2.5 units - 2^-11 of a unit (a unit is
    // 2^-1074, and 2.5 of them is `5 << 10` at 2^-1085). Parity would send
    // both to 2; the true nearest neighbours are 3 and 2.
    assert_eq!(exact((5 << 10) + 1, -1085).to_f64().to_bits(), 3);
    assert_eq!(exact((5 << 10) - 1, -1085).to_f64().to_bits(), 2);
}

#[test]
fn every_subnormal_bit_pattern_round_trips() {
    // One representative per subnormal binade, plus the extremes: `from_f64`
    // then `to_f64` must be the identity across the whole range.
    for shift in 0..52 {
        let bits = 1u64 << shift;
        let x = f64::from_bits(bits);
        assert_eq!(
            BigFloat::from_f64(x, 53)
                .expect("finite")
                .to_f64()
                .to_bits(),
            bits,
            "round-trip failed for subnormal 2^{}",
            shift as i64 - 1074,
        );
    }
    for bits in [1u64, 2, 3, MAX_SUBNORMAL_BITS, MAX_SUBNORMAL_BITS - 1] {
        let x = f64::from_bits(bits);
        assert_eq!(
            BigFloat::from_f64(x, 53)
                .expect("finite")
                .to_f64()
                .to_bits(),
            bits
        );
    }
}

// ---------------------------------------------------------------------------
// The subnormal/normal seam: 2^-1022
// ---------------------------------------------------------------------------

#[test]
fn largest_subnormal_boundary_carries_into_the_smallest_normal() {
    let max_subnormal = f64::from_bits(MAX_SUBNORMAL_BITS);
    assert_eq!(exact((1u64 << 52) - 1, -1074).to_f64(), max_subnormal);
    // Exactly halfway between the largest subnormal and MIN_POSITIVE. The
    // even neighbour is the normal one, so the tie *promotes* the value
    // across the seam — the rounding carry that grows the significand out of
    // the subnormal window.
    assert_eq!(exact((1u64 << 53) - 1, -1075).to_f64(), f64::MIN_POSITIVE);
    // A quarter of a ULP below that tie stays subnormal.
    assert_eq!(exact((1u64 << 54) - 3, -1076).to_f64(), max_subnormal);
    // And the first two normals convert exactly.
    assert_eq!(exact(1 << 52, -1074).to_f64(), f64::MIN_POSITIVE);
    assert_eq!(
        exact((1u64 << 52) + 1, -1074).to_f64().to_bits(),
        0x0010_0000_0000_0001,
    );
}

// ---------------------------------------------------------------------------
// The overflow boundary: 2^1024 - 2^970
// ---------------------------------------------------------------------------

#[test]
fn overflow_threshold_is_the_ieee_halfway_point() {
    // f64::MAX = (2^53 - 1) * 2^971.
    assert_eq!(exact((1u64 << 53) - 1, 971).to_f64(), f64::MAX);
    // (2^54 - 1) * 2^970 is exactly halfway between f64::MAX and 2^1024.
    // Ties-to-even prefers 2^1024, which is not representable: infinity.
    assert!(exact((1u64 << 54) - 1, 970).to_f64().is_infinite());
    assert_eq!(
        exact((1u64 << 54) - 1, 970).neg().to_f64(),
        f64::NEG_INFINITY
    );
    // Three quarters of a ULP below the threshold still fits.
    assert_eq!(exact((1u64 << 55) - 3, 969).to_f64(), f64::MAX);
}

// ---------------------------------------------------------------------------
// Deterministic sweeps
//
// Fixed-seed xorshift rather than `proptest`: these have to cover a band that
// a uniform `f64` generator reaches only about one time in a hundred, and a
// failure has to name the same pair on every machine.
// ---------------------------------------------------------------------------

/// Iterations per sweep. Sized for the debug profile the test suite runs
/// under — enough to cross every halfway point in the band many times over
/// without turning `cargo nextest run` into a coffee break.
const SWEEP_CASES: u32 = 50_000;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Assemble a finite `f64` from a biased exponent and a fraction.
fn assemble(biased_exp: u64, fraction: u64) -> f64 {
    f64::from_bits((biased_exp << 52) | (fraction & ((1u64 << 52) - 1)))
}

#[test]
fn sweep_division_into_the_subnormal_band_matches_ieee() {
    let mut state = 0x243f_6a88_85a3_08d3;
    let mut covered = 0u32;
    for _ in 0..SWEEP_CASES {
        let num_frac = xorshift(&mut state);
        let den_frac = xorshift(&mut state);
        let pick = xorshift(&mut state);
        // Steer the quotient across 2^-1015 .. 2^-1085: the whole subnormal
        // range plus the flush-to-zero cliff just below it.
        let den_exp = 900 + (pick % 700);
        let gap = 1015 + ((pick >> 20) % 70);
        if den_exp <= gap {
            continue;
        }
        let num_exp = den_exp - gap;
        if num_exp == 0 || den_exp > 2046 {
            continue;
        }
        let x = assemble(num_exp, num_frac);
        let y = assemble(den_exp, den_frac);
        if !x.is_normal() || !y.is_normal() {
            continue;
        }
        covered += 1;
        let q = BigFloat::from_f64(x, 53)
            .expect("finite")
            .div_ref(&BigFloat::from_f64(y, 53).expect("finite"))
            .expect("nonzero divisor");
        assert_eq!(
            q.to_f64().to_bits(),
            (x / y).to_bits(),
            "div {:#x} / {:#x} disagrees with the hardware quotient",
            x.to_bits(),
            y.to_bits(),
        );
    }
    assert!(covered > SWEEP_CASES / 2, "sweep filtered out too much");
}

#[test]
fn sweep_multiplication_into_the_subnormal_band_matches_ieee() {
    let mut state = 0x0bad_c0de_dead_beef;
    let mut covered = 0u32;
    for _ in 0..SWEEP_CASES {
        let lhs_frac = xorshift(&mut state);
        let rhs_frac = xorshift(&mut state);
        let pick = xorshift(&mut state);
        let lhs_exp = 1 + (pick % 900);
        // Aim the product's binade at the subnormal range and just below it.
        let product_exp = 1023 - (1015 + ((pick >> 20) % 70)) as i64;
        let rhs_exp = product_exp + 1023 - lhs_exp as i64;
        if !(1..=2046).contains(&rhs_exp) {
            continue;
        }
        let x = assemble(lhs_exp, lhs_frac);
        let y = assemble(rhs_exp as u64, rhs_frac);
        if !x.is_normal() || !y.is_normal() {
            continue;
        }
        covered += 1;
        let p = &BigFloat::from_f64(x, 53).expect("finite")
            * &BigFloat::from_f64(y, 53).expect("finite");
        assert_eq!(
            p.to_f64().to_bits(),
            (x * y).to_bits(),
            "mul {:#x} * {:#x} disagrees with the hardware product",
            x.to_bits(),
            y.to_bits(),
        );
    }
    assert!(covered > SWEEP_CASES / 2, "sweep filtered out too much");
}

#[test]
fn sweep_addition_across_the_subnormal_seam_matches_ieee() {
    // Sums of subnormals are exact — every operand is a multiple of 2^-1074
    // and so is the result — so this sweep is a guard against the conversion
    // mangling values addition handed it correctly, not against addition.
    let mut state = 0xfeed_face_cafe_1234;
    for _ in 0..SWEEP_CASES {
        let lhs_bits = xorshift(&mut state) & ((1u64 << 52) - 1);
        let rhs_bits = xorshift(&mut state) & ((1u64 << 52) - 1);
        if lhs_bits == 0 || rhs_bits == 0 {
            continue;
        }
        let x = f64::from_bits(lhs_bits);
        let y = f64::from_bits(rhs_bits);
        for (lhs, rhs) in [(x, y), (x, -y), (-x, y)] {
            let sum = &BigFloat::from_f64(lhs, 53).expect("finite")
                + &BigFloat::from_f64(rhs, 53).expect("finite");
            assert_eq!(
                sum.to_f64().to_bits(),
                (lhs + rhs).to_bits(),
                "add {lhs:e} + {rhs:e} disagrees with the hardware sum",
            );
        }
    }
}

#[test]
fn sweep_division_at_precision_54_matches_ieee() {
    // Nothing subnormal here: dividing at 54 bits and converting to f64
    // narrows by exactly one bit, so a quotient with its low bit set sits on
    // a 53-bit halfway point. That is the same double-rounding trap the
    // subnormal grid springs, reproduced in the normal range — roughly a
    // quarter of these pairs land on it.
    let mut state = 0x9e37_79b9_7f4a_7c15;
    let mut covered = 0u32;
    for _ in 0..SWEEP_CASES {
        let lhs = xorshift(&mut state);
        let rhs = xorshift(&mut state);
        // Clear the sign bit and force a mid-range exponent so the quotient
        // can neither overflow nor go subnormal.
        let x = f64::from_bits((lhs >> 1) | (1u64 << 62));
        let y = f64::from_bits((rhs >> 1) | (1u64 << 62));
        if !x.is_normal() || !y.is_normal() {
            continue;
        }
        let expected = x / y;
        if !expected.is_normal() {
            continue;
        }
        covered += 1;
        let q = BigFloat::from_f64(x, 54)
            .expect("finite")
            .div_ref(&BigFloat::from_f64(y, 54).expect("finite"))
            .expect("nonzero divisor");
        assert_eq!(
            q.to_f64().to_bits(),
            expected.to_bits(),
            "54-bit div {:#x} / {:#x} double-rounded on the way to f64",
            x.to_bits(),
            y.to_bits(),
        );
    }
    assert!(covered > SWEEP_CASES / 2, "sweep filtered out too much");
}

#[test]
fn sweep_subnormal_round_trip_is_the_identity() {
    let mut state = 0x1357_9bdf_2468_ace0;
    for _ in 0..SWEEP_CASES {
        let bits = xorshift(&mut state) & ((1u64 << 52) - 1);
        let x = f64::from_bits(bits);
        for value in [x, -x] {
            assert_eq!(
                BigFloat::from_f64(value, 53)
                    .expect("finite")
                    .to_f64()
                    .to_bits(),
                value.to_bits(),
                "round-trip changed the subnormal {bits:#x}",
            );
        }
    }
}

#[test]
fn sweep_sqrt_near_the_subnormal_seam_stays_within_one_ulp() {
    // `sqrt` rounds an integer-floor approximation rather than an exact
    // intermediate, so it is held to a 1-ULP bound rather than to bit
    // equality; what this pins is that the conversion never flushes its
    // result to zero or promotes it past the seam.
    let mut state = 0x2545_f491_4f6c_dd1d;
    for _ in 0..SWEEP_CASES / 4 {
        let bits = xorshift(&mut state) & ((1u64 << 52) - 1);
        if bits == 0 {
            continue;
        }
        let x = f64::from_bits(bits);
        let s = BigFloat::from_f64(x, 53)
            .expect("finite")
            .sqrt(53, RoundingMode::HalfEven)
            .expect("non-negative");
        let got = s.to_f64();
        let want = x.sqrt();
        assert!(
            got.is_normal(),
            "sqrt of a subnormal must be normal: {got:e}"
        );
        let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
        assert!(
            ulps <= 1,
            "sqrt({x:e}) = {got:e}, expected {want:e} ({ulps} ULP)"
        );
    }
}

// ---------------------------------------------------------------------------
// Which roundings hand a tie-break down to the conversion
// ---------------------------------------------------------------------------

/// A value one ULP-fraction above `5 * 2^-1075`, carried exactly at 64 bits.
///
/// `5 * 2^-1075` is a halfway point of the subnormal grid whose neighbours are
/// 2 and 3 units of `2^-1074`; ties-to-even prefers 2, so a value that is
/// really *above* the halfway point is a case where the two answers differ.
/// Rounding this to 53 bits lands exactly on that halfway point.
fn just_above_the_five_halves_tie() -> BigFloat {
    exact(5 * (1u64 << 55) + 1, -1130)
}

/// Mirror image: just below `7 * 2^-1075`, whose neighbours are 3 and 4 units.
/// Ties-to-even prefers 4, so "really below" is again the distinguishable case.
fn just_below_the_seven_halves_tie() -> BigFloat {
    exact(7 * (1u64 << 55) - 1, -1130)
}

#[test]
fn nearest_rounding_hands_its_direction_to_the_conversion() {
    // Rounded to nearest at 53 bits, both collapse onto the halfway point —
    // but the conversion still knows which side they came from.
    let above = just_above_the_five_halves_tie().round_to_precision(53, RoundingMode::HalfEven);
    assert_eq!(above.to_f64().to_bits(), 3, "came from above the tie");
    let below = just_below_the_seven_halves_tie().round_to_precision(53, RoundingMode::HalfEven);
    assert_eq!(below.to_f64().to_bits(), 3, "came from below the tie");

    // The same holds for the other two round-to-nearest tie-break policies:
    // what matters is that the rounding was to *nearest*, not how it broke
    // its own ties.
    for mode in [RoundingMode::HalfAway, RoundingMode::HalfToZero] {
        let above = just_above_the_five_halves_tie().round_to_precision(53, mode);
        assert_eq!(above.to_f64().to_bits(), 3, "{mode:?} from above");
        let below = just_below_the_seven_halves_tie().round_to_precision(53, mode);
        assert_eq!(below.to_f64().to_bits(), 3, "{mode:?} from below");
    }
}

#[test]
fn directed_rounding_leaves_the_conversion_on_ties_to_even() {
    // A directed rounding is a deliberate bias, not a failed attempt at the
    // nearest value, so the conversion must not use it to re-centre the
    // result. Where a directed mode does land *on* the halfway point, the
    // conversion falls back on ties-to-even and takes the even neighbour —
    // 2 for the 5/2 tie, 4 for the 7/2 tie — which is exactly the answer it
    // gave before any direction was recorded. Each of these would come back
    // one ULP away if the directed mode's bias were treated as a
    // nearest-rounding residue.
    for mode in [RoundingMode::ToZero, RoundingMode::ToNegInf] {
        let on_tie = just_above_the_five_halves_tie().round_to_precision(53, mode);
        assert_eq!(on_tie.to_f64().to_bits(), 2, "{mode:?} must tie to even");
    }
    for mode in [RoundingMode::ToInf, RoundingMode::AwayFromZero] {
        let on_tie = just_below_the_seven_halves_tie().round_to_precision(53, mode);
        assert_eq!(on_tie.to_f64().to_bits(), 4, "{mode:?} must tie to even");
    }

    // The complementary pairings never reach a tie: the directed rounding
    // carries them clear of the halfway point, leaving the conversion an
    // unambiguous nearest neighbour. Pinned so that a future change to the
    // tie-break cannot quietly reclassify these as tie cases.
    for mode in [RoundingMode::ToInf, RoundingMode::AwayFromZero] {
        let clear = just_above_the_five_halves_tie().round_to_precision(53, mode);
        assert_eq!(
            clear.to_f64().to_bits(),
            3,
            "{mode:?} rounds clear of the tie"
        );
    }
    for mode in [RoundingMode::ToZero, RoundingMode::ToNegInf] {
        let clear = just_below_the_seven_halves_tie().round_to_precision(53, mode);
        assert_eq!(
            clear.to_f64().to_bits(),
            3,
            "{mode:?} rounds clear of the tie"
        );
    }
}
