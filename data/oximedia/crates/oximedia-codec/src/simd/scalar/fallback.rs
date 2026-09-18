//! Pure Rust scalar fallback implementation.
//!
//! This implementation uses no SIMD instructions and works on all platforms.
//! It serves as a reference implementation and fallback when SIMD is unavailable.

use crate::simd::traits::{SimdOps, SimdOpsExt};
use crate::simd::types::{I16x8, I32x4, U8x16};

/// Scalar fallback SIMD implementation.
#[derive(Clone, Copy, Debug)]
pub struct ScalarFallback;

impl ScalarFallback {
    /// Create a new scalar fallback instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for ScalarFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdOps for ScalarFallback {
    #[inline]
    fn name(&self) -> &'static str {
        "scalar"
    }

    #[inline]
    fn is_available(&self) -> bool {
        true // Always available
    }

    #[inline]
    fn add_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    fn sub_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_sub(b[i]);
        }
        result
    }

    #[inline]
    fn mul_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_mul(b[i]);
        }
        result
    }

    #[inline]
    fn add_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }

    #[inline]
    fn sub_i32x4(&self, a: I32x4, b: I32x4) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = a[i].wrapping_sub(b[i]);
        }
        result
    }

    #[inline]
    fn min_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    fn max_i16x8(&self, a: I16x8, b: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[inline]
    fn clamp_i16x8(&self, v: I16x8, min: i16, max: i16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i].clamp(min, max);
        }
        result
    }

    #[inline]
    fn min_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[inline]
    fn max_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[inline]
    fn clamp_u8x16(&self, v: U8x16, min: u8, max: u8) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = v[i].clamp(min, max);
        }
        result
    }

    #[inline]
    fn horizontal_sum_i16x8(&self, v: I16x8) -> i32 {
        v.iter().map(|&x| i32::from(x)).sum()
    }

    #[inline]
    fn horizontal_sum_i32x4(&self, v: I32x4) -> i32 {
        v.iter().sum()
    }

    #[inline]
    fn sad_u8x16(&self, a: U8x16, b: U8x16) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y): (&u8, &u8)| u32::from(x.abs_diff(y)))
            .sum()
    }

    #[inline]
    fn sad_8(&self, a: &[u8], b: &[u8]) -> u32 {
        assert!(a.len() >= 8 && b.len() >= 8);
        a[..8]
            .iter()
            .zip(b[..8].iter())
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }

    #[inline]
    fn sad_16(&self, a: &[u8], b: &[u8]) -> u32 {
        assert!(a.len() >= 16 && b.len() >= 16);
        a[..16]
            .iter()
            .zip(b[..16].iter())
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }

    #[inline]
    fn widen_low_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i]);
        }
        result
    }

    #[inline]
    fn widen_high_u8_to_i16(&self, v: U8x16) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(v[i + 8]);
        }
        result
    }

    #[inline]
    fn narrow_i32x4_to_i16x8(&self, low: I32x4, high: I32x4) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..4 {
            result[i] = low[i].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            result[i + 4] = high[i].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        result
    }

    #[inline]
    fn madd_i16x8(&self, a: I16x8, b: I16x8, c: I16x8) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = a[i].wrapping_mul(b[i]).wrapping_add(c[i]);
        }
        result
    }

    #[inline]
    fn pmaddwd(&self, a: I16x8, b: I16x8) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = i32::from(a[i * 2]) * i32::from(b[i * 2])
                + i32::from(a[i * 2 + 1]) * i32::from(b[i * 2 + 1]);
        }
        result
    }

    #[inline]
    fn shr_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    fn shl_i16x8(&self, v: I16x8, shift: u32) -> I16x8 {
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = v[i] << shift;
        }
        result
    }

    #[inline]
    fn shr_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = v[i] >> shift;
        }
        result
    }

    #[inline]
    fn shl_i32x4(&self, v: I32x4, shift: u32) -> I32x4 {
        let mut result = I32x4::zero();
        for i in 0..4 {
            result[i] = v[i] << shift;
        }
        result
    }

    #[inline]
    fn avg_u8x16(&self, a: U8x16, b: U8x16) -> U8x16 {
        let mut result = U8x16::zero();
        for i in 0..16 {
            result[i] = ((u16::from(a[i]) + u16::from(b[i]) + 1) / 2) as u8;
        }
        result
    }
}

impl SimdOpsExt for ScalarFallback {
    #[inline]
    fn load4_u8_to_i16x8(&self, src: &[u8]) -> I16x8 {
        assert!(src.len() >= 4);
        let mut result = I16x8::zero();
        for i in 0..4 {
            result[i] = i16::from(src[i]);
        }
        result
    }

    #[inline]
    fn load8_u8_to_i16x8(&self, src: &[u8]) -> I16x8 {
        assert!(src.len() >= 8);
        let mut result = I16x8::zero();
        for i in 0..8 {
            result[i] = i16::from(src[i]);
        }
        result
    }

    #[inline]
    fn store4_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]) {
        assert!(dst.len() >= 4);
        for i in 0..4 {
            dst[i] = v[i].clamp(0, 255) as u8;
        }
    }

    #[inline]
    fn store8_i16x8_as_u8(&self, v: I16x8, dst: &mut [u8]) {
        assert!(dst.len() >= 8);
        for i in 0..8 {
            dst[i] = v[i].clamp(0, 255) as u8;
        }
    }

    #[inline]
    fn transpose_4x4_i16(&self, rows: &[I16x8; 4]) -> [I16x8; 4] {
        let mut out = [I16x8::zero(); 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = rows[j][i];
            }
        }
        out
    }

    #[inline]
    fn transpose_8x8_i16(&self, rows: &[I16x8; 8]) -> [I16x8; 8] {
        let mut out = [I16x8::zero(); 8];
        for i in 0..8 {
            for j in 0..8 {
                out[i][j] = rows[j][i];
            }
        }
        out
    }

    #[inline]
    fn butterfly_i16x8(&self, a: I16x8, b: I16x8) -> (I16x8, I16x8) {
        let sum = self.add_i16x8(a, b);
        let diff = self.sub_i16x8(a, b);
        (sum, diff)
    }

    #[inline]
    fn butterfly_i32x4(&self, a: I32x4, b: I32x4) -> (I32x4, I32x4) {
        let sum = self.add_i32x4(a, b);
        let diff = self.sub_i32x4(a, b);
        (sum, diff)
    }
}
