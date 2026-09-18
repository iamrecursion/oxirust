//! Chromatic adaptation transforms.
//!
//! This module provides tools for adapting colors to different illuminants.

pub mod adapt;
pub mod cat16;

pub use adapt::{ChromaticAdaptation, ChromaticAdaptationMethod};
pub use cat16::Cat16Adapter;

/// Adapt an XYZ colour from `src_wp` to `dst_wp` using the CAT16 transform.
///
/// This is a free-function convenience wrapper around [`Cat16Adapter::adapt`].
///
/// # Arguments
///
/// * `xyz`    - Input XYZ colour.
/// * `src_wp` - XYZ white-point of the source illuminant.
/// * `dst_wp` - XYZ white-point of the destination illuminant.
///
/// # Returns
///
/// Adapted XYZ colour under `dst_wp`.
#[must_use]
pub fn cat16_adapt(xyz: [f64; 3], src_wp: [f64; 3], dst_wp: [f64; 3]) -> [f64; 3] {
    Cat16Adapter::adapt(&xyz, &src_wp, &dst_wp)
}

// ---------------------------------------------------------------------------
// Tests for the cat16_adapt free function
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Illuminant, Xyz};

    /// CAT16 D65→D65 must be an identity transform.
    #[test]
    fn test_cat16_d65_to_d65() {
        let d65: Xyz = Illuminant::D65.xyz();
        let xyz: Xyz = [0.4505, 0.3290, 0.0736];
        let result = cat16_adapt(xyz, d65, d65);
        for ch in 0..3 {
            assert!(
                (result[ch] - xyz[ch]).abs() < 1e-6,
                "channel {ch}: expected {}, got {}",
                xyz[ch],
                result[ch]
            );
        }
    }
}
