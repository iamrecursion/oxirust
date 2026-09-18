//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
pub(super) fn const_f64<F: Float + NumCast>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}
