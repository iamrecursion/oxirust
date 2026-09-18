//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};

/// Helper to convert f64 constants to generic Float type with better error messages
#[inline(always)]
pub(super) fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value)
        .expect(
            "Failed to convert constant to target float type - this indicates an incompatible numeric type",
        )
}
