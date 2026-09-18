//! Audio fingerprinting and matching.

pub mod fingerprint;
pub mod match_impl;

pub use fingerprint::AudioFingerprintIndex;
pub use match_impl::AudioMatcher;
