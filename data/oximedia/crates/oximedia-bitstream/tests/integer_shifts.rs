// Tests for `UnsignedInteger::shl_default` / `shr_default`.
//
// The macro override in `define_unsigned_integer!` (enabled once
// `unbounded_shl`/`unbounded_shr` became stable in Rust 1.87) ensures
// shifts whose amount is >= the type's bit-width return 0 rather than
// panicking or wrapping.  These tests exercise:
//
//   • normal shifts (rhs < BITS)     — must match `<<`/`>>`
//   • boundary shift (rhs == BITS)   — must return 0, not panic
//   • over-width shift (rhs >> BITS) — must return 0, not panic
//
// The four unsigned primitive types covered by the macro are tested.

use oximedia_bitstream::UnsignedInteger;

// ── u8 ───────────────────────────────────────────────────────────────────────

#[test]
fn u8_shl_default_normal() {
    assert_eq!(0b0000_0011_u8.shl_default(2), 0b0000_1100_u8);
    assert_eq!(1_u8.shl_default(0), 1_u8);
    assert_eq!(1_u8.shl_default(7), 0x80_u8);
}

#[test]
fn u8_shl_default_at_boundary() {
    // rhs == 8 (== u8::BITS): no panic, result must be 0
    assert_eq!(1_u8.shl_default(8), 0_u8);
    assert_eq!(0xFF_u8.shl_default(8), 0_u8);
}

#[test]
fn u8_shl_default_over_width() {
    assert_eq!(1_u8.shl_default(100), 0_u8);
    assert_eq!(0xFF_u8.shl_default(u32::MAX), 0_u8);
}

#[test]
fn u8_shr_default_normal() {
    assert_eq!(0b1100_0000_u8.shr_default(2), 0b0011_0000_u8);
    assert_eq!(0x80_u8.shr_default(7), 1_u8);
    assert_eq!(1_u8.shr_default(0), 1_u8);
}

#[test]
fn u8_shr_default_at_boundary() {
    assert_eq!(0xFF_u8.shr_default(8), 0_u8);
    assert_eq!(1_u8.shr_default(8), 0_u8);
}

#[test]
fn u8_shr_default_over_width() {
    assert_eq!(0xFF_u8.shr_default(100), 0_u8);
    assert_eq!(0xFF_u8.shr_default(u32::MAX), 0_u8);
}

// ── u16 ──────────────────────────────────────────────────────────────────────

#[test]
fn u16_shl_default_normal() {
    assert_eq!(1_u16.shl_default(15), 0x8000_u16);
    assert_eq!(0x00FF_u16.shl_default(8), 0xFF00_u16);
}

#[test]
fn u16_shl_default_at_boundary() {
    assert_eq!(1_u16.shl_default(16), 0_u16);
    assert_eq!(0xFFFF_u16.shl_default(16), 0_u16);
}

#[test]
fn u16_shl_default_over_width() {
    assert_eq!(0xFFFF_u16.shl_default(200), 0_u16);
}

#[test]
fn u16_shr_default_normal() {
    assert_eq!(0xFF00_u16.shr_default(8), 0x00FF_u16);
    assert_eq!(0x8000_u16.shr_default(15), 1_u16);
}

#[test]
fn u16_shr_default_at_boundary() {
    assert_eq!(0xFFFF_u16.shr_default(16), 0_u16);
}

#[test]
fn u16_shr_default_over_width() {
    assert_eq!(0xFFFF_u16.shr_default(200), 0_u16);
}

// ── u32 ──────────────────────────────────────────────────────────────────────

#[test]
fn u32_shl_default_normal() {
    assert_eq!(1_u32.shl_default(31), 0x8000_0000_u32);
    assert_eq!(0x0000_FFFF_u32.shl_default(16), 0xFFFF_0000_u32);
}

#[test]
fn u32_shl_default_at_boundary() {
    assert_eq!(1_u32.shl_default(32), 0_u32);
    assert_eq!(u32::MAX.shl_default(32), 0_u32);
}

#[test]
fn u32_shl_default_over_width() {
    assert_eq!(u32::MAX.shl_default(500), 0_u32);
}

#[test]
fn u32_shr_default_normal() {
    assert_eq!(0xFFFF_0000_u32.shr_default(16), 0x0000_FFFF_u32);
    assert_eq!(0x8000_0000_u32.shr_default(31), 1_u32);
}

#[test]
fn u32_shr_default_at_boundary() {
    assert_eq!(u32::MAX.shr_default(32), 0_u32);
}

#[test]
fn u32_shr_default_over_width() {
    assert_eq!(u32::MAX.shr_default(500), 0_u32);
}

// ── u64 ──────────────────────────────────────────────────────────────────────

#[test]
fn u64_shl_default_normal() {
    assert_eq!(1_u64.shl_default(63), 0x8000_0000_0000_0000_u64);
    assert_eq!(0xFFFF_FFFF_u64.shl_default(32), 0xFFFF_FFFF_0000_0000_u64);
}

#[test]
fn u64_shl_default_at_boundary() {
    assert_eq!(1_u64.shl_default(64), 0_u64);
    assert_eq!(u64::MAX.shl_default(64), 0_u64);
}

#[test]
fn u64_shl_default_over_width() {
    assert_eq!(u64::MAX.shl_default(1000), 0_u64);
}

#[test]
fn u64_shr_default_normal() {
    assert_eq!(0x8000_0000_0000_0000_u64.shr_default(63), 1_u64);
    assert_eq!(0xFFFF_FFFF_0000_0000_u64.shr_default(32), 0xFFFF_FFFF_u64);
}

#[test]
fn u64_shr_default_at_boundary() {
    assert_eq!(u64::MAX.shr_default(64), 0_u64);
}

#[test]
fn u64_shr_default_over_width() {
    assert_eq!(u64::MAX.shr_default(1000), 0_u64);
}

// ── u128 ─────────────────────────────────────────────────────────────────────

#[test]
fn u128_shl_default_at_boundary() {
    assert_eq!(1_u128.shl_default(128), 0_u128);
    assert_eq!(u128::MAX.shl_default(128), 0_u128);
}

#[test]
fn u128_shr_default_at_boundary() {
    assert_eq!(u128::MAX.shr_default(128), 0_u128);
}

#[test]
fn u128_shl_default_normal() {
    assert_eq!(1_u128.shl_default(127), 1_u128 << 127);
}

#[test]
fn u128_shr_default_normal() {
    assert_eq!((1_u128 << 127).shr_default(127), 1_u128);
}

// ── consistency: shl_default / shr_default agree with `checked_*` ───────────

#[test]
fn consistency_shl_matches_checked() {
    for rhs in [0_u32, 1, 4, 7, 8, 9, 31, 32, 63, 64, 127, 128, 255] {
        let checked = 0xAB_u8.checked_shl(rhs).unwrap_or(0);
        let unbounded = 0xAB_u8.shl_default(rhs);
        assert_eq!(checked, unbounded, "u8.shl_default({rhs}) mismatch");
    }
    for rhs in [0_u32, 1, 15, 16, 17, 31, 32, 63, 64, 127, 128] {
        let checked = 0xABCD_u16.checked_shl(rhs).unwrap_or(0);
        let unbounded = 0xABCD_u16.shl_default(rhs);
        assert_eq!(checked, unbounded, "u16.shl_default({rhs}) mismatch");
    }
}

#[test]
fn consistency_shr_matches_checked() {
    for rhs in [0_u32, 1, 4, 7, 8, 9, 31, 32, 63, 64, 127, 128, 255] {
        let checked = 0xAB_u8.checked_shr(rhs).unwrap_or(0);
        let unbounded = 0xAB_u8.shr_default(rhs);
        assert_eq!(checked, unbounded, "u8.shr_default({rhs}) mismatch");
    }
    for rhs in [0_u32, 1, 15, 16, 17, 31, 32, 63, 64, 127, 128] {
        let checked = 0xABCD_u16.checked_shr(rhs).unwrap_or(0);
        let unbounded = 0xABCD_u16.shr_default(rhs);
        assert_eq!(checked, unbounded, "u16.shr_default({rhs}) mismatch");
    }
}
