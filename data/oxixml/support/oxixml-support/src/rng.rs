//! Deterministic, **non-cryptographic** pseudo-random number generators.
//!
//! These generators exist for reproducible property testing, fuzzing seeds and any other place
//! where a small, fast, self-contained source of pseudo-randomness is useful. They are **not**
//! suitable for cryptography, security tokens, or anything where predictability is a hazard:
//! their internal state is trivially recoverable from a handful of outputs.
//!
//! Two algorithms are provided:
//!
//! * [`SplitMix64`] — the SplitMix64 mixing function (as popularised alongside the xoshiro
//!   family). A single 64-bit state word; excellent as a seeder for other generators.
//! * [`Pcg32`] — the PCG-XSH-RR 64/32 permuted congruential generator, which produces 32-bit
//!   outputs from a 64-bit LCG state with an additional per-stream increment.
//!
//! Both implement the [`Rng`] trait, which supplies [`Rng::fill_bytes`] and an unbiased
//! [`Rng::gen_range`] on top of the primitive `next_u32` / `next_u64` operations.

use core::ops::Range;

/// A source of pseudo-random 32- and 64-bit words, with convenience helpers layered on top.
///
/// Implementors need only provide [`Rng::next_u32`] and [`Rng::next_u64`]; the byte-filling and
/// ranged-generation helpers have correct default implementations.
pub trait Rng {
    /// Produce the next pseudo-random [`u32`].
    fn next_u32(&mut self) -> u32;

    /// Produce the next pseudo-random [`u64`].
    fn next_u64(&mut self) -> u64;

    /// Fill `dest` completely with pseudo-random bytes.
    ///
    /// Bytes are drawn eight at a time (little-endian) from [`Rng::next_u64`], with a partial
    /// final chunk handled correctly.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            remainder.copy_from_slice(&bytes[..remainder.len()]);
        }
    }

    /// Generate a uniformly distributed value in the half-open range `start..end`.
    ///
    /// Uses rejection sampling so the result is **free of modulo bias**: values in the top
    /// `2^64 mod width` slice of the raw output are discarded and re-drawn.
    ///
    /// # Panics
    ///
    /// Panics if the range is empty (`start >= end`), mirroring the behaviour of the standard
    /// library and the `rand` crate.
    fn gen_range(&mut self, range: Range<u64>) -> u64 {
        let Range { start, end } = range;
        assert!(
            start < end,
            "gen_range: empty or inverted range {start}..{end}"
        );
        let width = end - start;
        // `threshold` is `2^64 mod width`: the number of low values to reject so that the
        // remaining accepted values are an exact multiple of `width`.
        let threshold = width.wrapping_neg() % width;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return start + (value % width);
            }
        }
    }
}

/// The SplitMix64 pseudo-random generator (non-cryptographic).
///
/// A single 64-bit state word advanced by the golden-ratio increment, followed by a strong
/// finalising mix. Fast, tiny, and well-distributed; commonly used to seed other generators.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

/// The golden-ratio odd increment used by SplitMix64.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

impl SplitMix64 {
    /// Create a generator seeded with `seed`. Every seed is valid.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Rng for SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_INCREMENT);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Take the high 32 bits, which are the best-mixed.
        (self.next_u64() >> 32) as u32
    }
}

/// The PCG-XSH-RR 64/32 permuted congruential generator (non-cryptographic).
///
/// An LCG over a 64-bit state produces 32-bit outputs through an xor-shift followed by a
/// data-dependent rotation. Each instance carries an odd increment ("stream") so that two
/// generators seeded identically but with different streams produce distinct sequences.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

/// The LCG multiplier used by PCG (the value recommended by the reference implementation).
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// A default stream selector used by [`Pcg32::new`].
const PCG_DEFAULT_STREAM: u64 = 0x0A02_BDBF_7BB3_C0A7;

impl Pcg32 {
    /// Create a generator seeded with `seed`, using a fixed default stream.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, PCG_DEFAULT_STREAM)
    }

    /// Create a generator seeded with `seed` on the independent stream selected by `stream`.
    ///
    /// Any two generators with the same `seed` but different `stream` values yield different,
    /// non-overlapping-in-practice sequences.
    #[inline]
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        // The increment must be odd; the low bit of `stream` is dropped in favour of a set bit.
        let increment = (stream << 1) | 1;
        let mut rng = Self {
            state: 0,
            increment,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    /// Advance the underlying LCG by one step.
    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
    }
}

impl Rng for Pcg32 {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.step();
        // XSH-RR output permutation.
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rotation = (old_state >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32());
        (high << 32) | low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_known_sequence() {
        // Reference outputs for SplitMix64 seeded with 0 (the canonical reference stream).
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::new(12345);
        let mut b = SplitMix64::new(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn pcg32_is_deterministic_and_distinct_streams() {
        let mut a = Pcg32::new(42);
        let mut b = Pcg32::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        // Different streams diverge.
        let mut s0 = Pcg32::with_stream(42, 1);
        let mut s1 = Pcg32::with_stream(42, 2);
        let mut differs = false;
        for _ in 0..16 {
            if s0.next_u32() != s1.next_u32() {
                differs = true;
                break;
            }
        }
        assert!(differs, "distinct streams should produce distinct output");
    }

    #[test]
    fn fill_bytes_is_reproducible_and_full() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        let mut buf_a = [0u8; 37];
        let mut buf_b = [0u8; 37];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
        // Extremely unlikely to be all zero for a full 37-byte fill.
        assert!(buf_a.iter().any(|&byte| byte != 0));
    }

    #[test]
    fn gen_range_stays_in_bounds() {
        let mut rng = Pcg32::new(99);
        for _ in 0..10_000 {
            let value = rng.gen_range(10..20);
            assert!((10..20).contains(&value));
        }
        // A width-1 range always yields its single value.
        for _ in 0..100 {
            assert_eq!(rng.gen_range(5..6), 5);
        }
    }

    #[test]
    fn gen_range_is_reasonably_uniform() {
        let mut rng = SplitMix64::new(0xABCD);
        let mut counts = [0u32; 6];
        for _ in 0..60_000 {
            let value = rng.gen_range(0..6) as usize;
            counts[value] += 1;
        }
        // Each bucket should be near 10_000; allow a generous tolerance.
        for &count in &counts {
            assert!(count > 9_000 && count < 11_000, "bucket skew: {counts:?}");
        }
    }

    #[test]
    #[should_panic(expected = "empty or inverted range")]
    fn gen_range_rejects_empty_range() {
        let mut rng = SplitMix64::new(1);
        let _ = rng.gen_range(5..5);
    }
}
