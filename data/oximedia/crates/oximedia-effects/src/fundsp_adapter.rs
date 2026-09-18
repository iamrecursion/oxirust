//! FunDSP interoperability adapter for `AudioEffect` implementations.
//!
//! Enable with the `fundsp` feature gate:
//! ```toml
//! oximedia-effects = { features = ["fundsp"] }
//! ```
//!
//! # Overview
//!
//! `FunDspAdapter` wraps any `AudioEffect` as a FunDSP stereo `AudioNode`
//! (2-in / 2-out).  This lets `AudioEffect` implementations participate in
//! FunDSP signal graphs without any manual FFI or intermediate conversion.
//!
//! # Node Identity
//!
//! FunDSP's `AudioNode::ID` must be a `const u64`.  Because `const` associated
//! items cannot depend on a runtime generic type parameter, we set `ID = 0` in
//! the trait impl and provide `FunDspAdapter::effect_node_id` as the correct
//! per-type identifier (an FNV-1a hash of `E::EFFECT_ID`).  Callers that
//! construct FunDSP graphs programmatically should use `effect_node_id()` when
//! they need a stable, non-zero node tag.

/// FNV-1a 64-bit hash of a string, evaluated at compile time.
///
/// Produces a stable `u64` identifier from a string slice.  Suitable for
/// deriving FunDSP node IDs from `EFFECT_ID` slugs.
///
/// # Example
/// ```
/// use oximedia_effects::fundsp_adapter::fnv1a_u64;
/// let id = fnv1a_u64("freeverb");
/// assert_ne!(id, 0);
/// ```
pub const fn fnv1a_u64(s: &str) -> u64 {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    let bytes = s.as_bytes();
    let mut hash = OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

#[cfg(feature = "fundsp")]
mod adapter_impl {
    use super::fnv1a_u64;
    use crate::AudioEffect;
    use fundsp::prelude::*;

    /// Wraps any `AudioEffect + Clone + Send + Sync` as a FunDSP stereo `AudioNode`.
    ///
    /// The node has 2 inputs (`U2`) and 2 outputs (`U2`), corresponding to the
    /// stereo left/right channel pair.
    ///
    /// # Cloning + thread-safety requirement
    ///
    /// FunDSP requires all `AudioNode` implementations to be `Clone + Send + Sync`
    /// so that signal graphs can be duplicated and shared across threads.
    /// Derive or implement those traits on your effect struct before wrapping.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use oximedia_effects::{AudioEffect, reverb::Freeverb, ReverbConfig};
    /// use oximedia_effects::fundsp_adapter::FunDspAdapter;
    /// use fundsp::prelude::*;
    ///
    /// let reverb = Freeverb::new(ReverbConfig::default(), 44100.0);
    /// let mut node = FunDspAdapter::new(reverb);
    /// node.set_sample_rate(44100.0);
    ///
    /// let frame: Frame<f32, U2> = [0.5f32, 0.5].into();
    /// let out = node.tick(&frame);
    /// assert!(out[0].is_finite() && out[1].is_finite());
    /// ```
    pub struct FunDspAdapter<E>
    where
        E: AudioEffect + Clone + Send + Sync,
    {
        inner: E,
        sample_rate: f64,
    }

    impl<E> FunDspAdapter<E>
    where
        E: AudioEffect + Clone + Send + Sync,
    {
        /// Wrap an `AudioEffect` as a FunDSP stereo node.
        pub fn new(effect: E) -> Self {
            Self {
                inner: effect,
                sample_rate: 44100.0,
            }
        }

        /// Immutable reference to the wrapped effect.
        pub fn inner(&self) -> &E {
            &self.inner
        }

        /// Mutable reference to the wrapped effect.
        pub fn inner_mut(&mut self) -> &mut E {
            &mut self.inner
        }

        /// Consume the adapter, returning the inner effect.
        pub fn into_inner(self) -> E {
            self.inner
        }

        /// Returns the stable FNV-1a hash of this effect's `EFFECT_ID`.
        ///
        /// Use this as the node tag when constructing FunDSP signal graphs that
        /// require unique per-type identifiers.
        ///
        /// # Example
        /// ```ignore
        /// let id = adapter.effect_node_id();
        /// assert_eq!(id, oximedia_effects::fundsp_adapter::fnv1a_u64(E::EFFECT_ID));
        /// ```
        pub fn effect_node_id(&self) -> u64 {
            fnv1a_u64(E::EFFECT_ID)
        }
    }

    impl<E> Clone for FunDspAdapter<E>
    where
        E: AudioEffect + Clone + Send + Sync,
    {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
                sample_rate: self.sample_rate,
            }
        }
    }

    impl<E> AudioNode for FunDspAdapter<E>
    where
        E: AudioEffect + Clone + Send + Sync + 'static,
    {
        /// Const node ID required by FunDSP.
        ///
        /// Set to `0` because const associated items cannot depend on generic
        /// type parameters.  Use [`FunDspAdapter::effect_node_id`] for the
        /// correct per-type FNV-1a hash ID.
        const ID: u64 = 0;

        /// Stereo input arity (L + R).
        type Inputs = U2;

        /// Stereo output arity (L + R).
        type Outputs = U2;

        fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
            let (l, r) = self.inner.process_sample_stereo(input[0], input[1]);
            [l, r].into()
        }

        fn set_sample_rate(&mut self, sample_rate: f64) {
            self.sample_rate = sample_rate;
            #[allow(clippy::cast_possible_truncation)]
            self.inner.set_sample_rate(sample_rate as f32);
        }

        fn reset(&mut self) {
            self.inner.reset();
        }
    }
}

#[cfg(feature = "fundsp")]
pub use adapter_impl::FunDspAdapter;

#[cfg(test)]
mod tests {
    use super::fnv1a_u64;

    #[test]
    fn test_fnv1a_u64_non_zero() {
        assert_ne!(fnv1a_u64("freeverb"), 0);
        assert_ne!(fnv1a_u64("plate_reverb"), 0);
    }

    #[test]
    fn test_fnv1a_u64_distinct() {
        let a = fnv1a_u64("freeverb");
        let b = fnv1a_u64("plate_reverb");
        let c = fnv1a_u64("spring_reverb");
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn test_fnv1a_u64_stable() {
        // Same input always produces the same output.
        assert_eq!(fnv1a_u64("analog_delay"), fnv1a_u64("analog_delay"));
    }

    #[test]
    fn test_fnv1a_u64_empty() {
        // The empty string produces the FNV offset basis, which is non-zero.
        let empty_hash = fnv1a_u64("");
        assert_eq!(empty_hash, 14_695_981_039_346_656_037u64);
    }
}
