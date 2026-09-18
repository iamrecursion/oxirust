//! A minimal deterministic PRNG (SplitMix64).
//!
//! phop avoids a `rand` dependency (which would risk pulling conflicting versions into the
//! `scirs2` tree) and instead uses this tiny, well-known generator wherever reproducible
//! pseudo-randomness is needed: Gumbel-noise sampling in the topology search and shuffled
//! minibatching of the data axis. Seeded from [`crate::Config`], it makes every run repeatable.

/// SplitMix64 — a fast, statistically solid 64-bit generator with a 64-bit state.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    /// Create a generator from a seed.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Draw the next raw 64-bit value.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in the open interval `(0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v.clamp(1e-12, 1.0 - 1e-12)
    }

    /// A standard Gumbel(0, 1) sample: `-ln(-ln(u))`.
    pub(crate) fn gumbel(&mut self) -> f64 {
        -(-self.unit().ln()).ln()
    }

    /// Uniform integer in `[0, n)` (via rejection-free modulo; bias is negligible for our `n`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Fisher–Yates shuffle of `slice` in place.
    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.below(i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn unit_in_open_interval() {
        let mut r = SplitMix64::new(1);
        for _ in 0..1000 {
            let u = r.unit();
            assert!(u > 0.0 && u < 1.0);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = SplitMix64::new(7);
        let mut v: Vec<usize> = (0..50).collect();
        r.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        // With overwhelming probability the shuffle changed the order.
        assert_ne!(v, (0..50).collect::<Vec<_>>());
    }
}
