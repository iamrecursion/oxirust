//! Object-safe RNG shim for the speculative decoder.
//!
//! The core [`DraftModel`](crate::speculative_decoding::traits::DraftModel)
//! trait takes `&mut dyn SpecRng` so that callers can pass a `StdRng` (the
//! default, reproducible, seedable option) without any of the acceptance
//! / engine code being forced to generic on the PRNG type.
//!
//! This trait exposes **only** what speculative decoding needs: a `[0, 1)`
//! float.  That one primitive is enough to derive categorical samples,
//! Bernoulli trials and raw `u64` draws (via 64 independent bits).
//!
//! We provide a blanket impl for any SciRS2 `Rng + RngExt + Send` type so
//! callers can pass a `StdRng` unchanged.

use scirs2_core::random::{Rng, RngExt};

/// Minimal object-safe RNG interface consumed by
/// [`DraftModel::propose`](crate::speculative_decoding::traits::DraftModel::propose)
/// and [`crate::speculative_decoding::acceptance`].
pub trait SpecRng: Send {
    /// Draw a uniform float in `[0, 1)`.
    fn next_unit_f64(&mut self) -> f64;
}

/// Blanket impl: any `Send` SciRS2 RNG satisfying [`Rng`] + [`RngExt`] is a
/// [`SpecRng`].
impl<R> SpecRng for R
where
    R: Rng + RngExt + Send,
{
    fn next_unit_f64(&mut self) -> f64 {
        // `RngExt::random::<f64>()` returns a uniform float in `[0, 1)` per
        // SciRS2 docs (matches the convention of `rand::Rng::gen::<f64>()`).
        RngExt::random::<f64>(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{SeedableRng, StdRng};

    #[test]
    fn std_rng_is_spec_rng() {
        let mut rng = StdRng::seed_from_u64(123);
        let u = rng.next_unit_f64();
        assert!((0.0..1.0).contains(&u));
    }

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut a = StdRng::seed_from_u64(99);
        let mut b = StdRng::seed_from_u64(99);
        assert_eq!(
            (0..5).map(|_| a.next_unit_f64()).collect::<Vec<_>>(),
            (0..5).map(|_| b.next_unit_f64()).collect::<Vec<_>>()
        );
    }
}
