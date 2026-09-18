//! Audio fingerprinting module for `oximedia-mir`.
//!
//! Provides Chromaprint/AcoustID-inspired acoustic fingerprinting.

pub mod acoustid;

pub use acoustid::{
    AcoustidEncoder, AcoustidFingerprint, ChromaExtractor, ChromaFeature, FingerprintMatcher,
};
