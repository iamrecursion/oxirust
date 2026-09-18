//! Utilities for casting between f32 and f64 tensors, and a dual-precision bridge.
//!
//! This module provides:
//! - `cast_f64_to_f32`: lossy downcast of an `ArrayD<f64>` to `ArrayD<f32>`
//! - `cast_f32_to_f64`: lossless upcast of an `ArrayD<f32>` to `ArrayD<f64>`
//! - `DualPrecisionBridge`: runs forward einsum in f32 but accumulates gradients in f64

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use tensorlogic_infer::ExecutorError;

use crate::executor_f32::Scirs2Exec32;
use tensorlogic_infer::TlExecutor;

/// Cast an `ArrayD<f64>` to `ArrayD<f32>` (lossy — values outside f32 range saturate).
pub fn cast_f64_to_f32(t: &ArrayD<f64>) -> ArrayD<f32> {
    t.mapv(|v| v as f32)
}

/// Cast an `ArrayD<f32>` to `ArrayD<f64>` (lossless for values representable in f32).
pub fn cast_f32_to_f64(t: &ArrayD<f32>) -> ArrayD<f64> {
    t.mapv(|v| v as f64)
}

/// A bridge that runs forward einsum operations in f32 precision but stores
/// and accumulates gradients in f64 precision.
///
/// This is useful for mixed-precision training pipelines where forward
/// activations can tolerate f32 rounding but gradient accumulation requires f64.
pub struct DualPrecisionBridge {
    exec32: Scirs2Exec32,
    f64_accumulator: HashMap<String, ArrayD<f64>>,
}

impl Default for DualPrecisionBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl DualPrecisionBridge {
    /// Create a new, empty bridge.
    pub fn new() -> Self {
        DualPrecisionBridge {
            exec32: Scirs2Exec32::new(),
            f64_accumulator: HashMap::new(),
        }
    }

    /// Insert an f64 tensor (auto-cast to f32 for forward operations).
    pub fn add_f64_tensor(&mut self, name: impl Into<String>, t: ArrayD<f64>) {
        let key: String = name.into();
        let t32 = cast_f64_to_f32(&t);
        self.exec32.add_tensor(key, t32);
    }

    /// Run an einsum in f32 precision and return the result cast back to f64.
    ///
    /// `input_names` must all be tensors previously added via `add_f64_tensor`.
    pub fn einsum_f32_result_f64(
        &mut self,
        spec: &str,
        input_names: &[&str],
    ) -> Result<ArrayD<f64>, ExecutorError> {
        let inputs: Result<Vec<_>, ExecutorError> = input_names
            .iter()
            .map(|name| {
                self.exec32.tensors.get(*name).cloned().ok_or_else(|| {
                    ExecutorError::InvalidEinsumSpec(format!(
                        "Tensor '{}' not found in bridge",
                        name
                    ))
                })
            })
            .collect();
        let inputs = inputs?;
        let result32 = self.exec32.einsum(spec, &inputs)?;
        Ok(cast_f32_to_f64(&result32))
    }

    /// Accumulate a gradient tensor (f64) by name.
    ///
    /// If a gradient for `name` already exists it is element-wise added to
    /// the incoming gradient (standard gradient accumulation semantics).
    pub fn accumulate_grad(&mut self, name: &str, grad: ArrayD<f64>) {
        let entry = self
            .f64_accumulator
            .entry(name.to_string())
            .or_insert_with(|| ArrayD::zeros(grad.raw_dim()));
        *entry = &*entry + &grad;
    }

    /// Get the accumulated gradient for a tensor, if any.
    pub fn get_grad(&self, name: &str) -> Option<&ArrayD<f64>> {
        self.f64_accumulator.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn f64_tensor(shape: &[usize], data: Vec<f64>) -> ArrayD<f64> {
        ArrayD::from_shape_vec(shape, data).expect("valid shape/data for test tensor")
    }

    fn f32_tensor(shape: &[usize], data: Vec<f32>) -> ArrayD<f32> {
        ArrayD::from_shape_vec(shape, data).expect("valid shape/data for test tensor")
    }

    #[test]
    fn test_cast_f64_to_f32_values() {
        let t = f64_tensor(&[3], vec![1.0, 2.0, std::f64::consts::PI]);
        let t32 = cast_f64_to_f32(&t);
        let pi32 = t32[[2]] as f64;
        // f32 PI is accurate to ~7 significant digits
        assert!(
            (pi32 - std::f64::consts::PI).abs() < 1e-6,
            "pi approx failed: {}",
            pi32
        );
        assert_eq!(t32[[0]], 1.0_f32);
        assert_eq!(t32[[1]], 2.0_f32);
    }

    #[test]
    fn test_cast_f32_to_f64_lossless() {
        // Integer values that fit exactly in f32 should survive round-trip perfectly.
        let t = f32_tensor(&[4], vec![1.0, 2.0, 4.0, 8.0]);
        let t64 = cast_f32_to_f64(&t);
        let expected = [1.0_f64, 2.0, 4.0, 8.0];
        for (got, exp) in t64.iter().zip(expected.iter()) {
            assert_eq!(*got, *exp, "lossless upcast failed");
        }
    }

    #[test]
    fn test_cast_shape_preserved() {
        let t = f64_tensor(&[3, 4, 5], (0..60).map(|i| i as f64).collect());
        let t32 = cast_f64_to_f32(&t);
        let t64 = cast_f32_to_f64(&t32);
        assert_eq!(t.shape(), t32.shape());
        assert_eq!(t32.shape(), t64.shape());
    }

    #[test]
    fn test_dual_bridge_einsum() {
        // 2×2 identity × 2×2 matrix -> same matrix
        let identity = f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let matrix = f64_tensor(&[2, 2], vec![3.0, 4.0, 5.0, 6.0]);
        let mut bridge = DualPrecisionBridge::new();
        bridge.add_f64_tensor("I", identity);
        bridge.add_f64_tensor("M", matrix);
        let result = bridge
            .einsum_f32_result_f64("ij,jk->ik", &["I", "M"])
            .expect("dual bridge einsum");
        assert_eq!(result.shape(), &[2, 2]);
        let data: Vec<f64> = result.iter().copied().collect();
        // Result should be identity × matrix = matrix
        assert!(
            (data[0] - 3.0).abs() < 1e-4,
            "expected 3.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 4.0).abs() < 1e-4,
            "expected 4.0, got {}",
            data[1]
        );
    }

    #[test]
    fn test_dual_bridge_accumulate_grad() {
        let grad1 = f64_tensor(&[2], vec![1.0, 2.0]);
        let grad2 = f64_tensor(&[2], vec![3.0, 4.0]);
        let mut bridge = DualPrecisionBridge::new();
        bridge.accumulate_grad("w", grad1);
        bridge.accumulate_grad("w", grad2);
        let acc = bridge
            .get_grad("w")
            .expect("grad should exist after accumulation");
        let data: Vec<f64> = acc.iter().copied().collect();
        assert!(
            (data[0] - 4.0).abs() < 1e-10,
            "accumulated grad[0]={}",
            data[0]
        );
        assert!(
            (data[1] - 6.0).abs() < 1e-10,
            "accumulated grad[1]={}",
            data[1]
        );
    }

    #[test]
    fn test_dual_bridge_get_nonexistent_grad() {
        let bridge = DualPrecisionBridge::new();
        assert!(bridge.get_grad("nonexistent").is_none());
    }

    #[test]
    fn test_cast_zeros() {
        let zeros = f64_tensor(&[3, 3], vec![0.0; 9]);
        let z32 = cast_f64_to_f32(&zeros);
        assert!(z32.iter().all(|&v| v == 0.0_f32));
        let z64 = cast_f32_to_f64(&z32);
        assert!(z64.iter().all(|&v| v == 0.0_f64));
    }

    #[test]
    fn test_cast_large_values() {
        // f32::MAX ≈ 3.4e38 should survive a round-trip through as-cast
        let large = f32::MAX;
        let t32 = f32_tensor(&[1], vec![large]);
        let t64 = cast_f32_to_f64(&t32);
        // Cast back to compare — value should be exactly f32::MAX as f64
        assert!(
            (t64[[0]] - (large as f64)).abs() < 1.0,
            "large value cast failed"
        );
    }

    #[test]
    fn test_dual_bridge_multiple_einsums() {
        // Two sequential 2×2 matmuls through the bridge
        let a = f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 2.0]); // diag(1,2)
        let b = f64_tensor(&[2, 2], vec![3.0, 0.0, 0.0, 4.0]); // diag(3,4)
        let c = f64_tensor(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]); // all-ones

        let mut bridge = DualPrecisionBridge::new();
        bridge.add_f64_tensor("A", a);
        bridge.add_f64_tensor("B", b);
        bridge.add_f64_tensor("C", c);

        // First: AB = diag(3, 8)
        let ab = bridge
            .einsum_f32_result_f64("ij,jk->ik", &["A", "B"])
            .expect("first einsum");
        let ab_data: Vec<f64> = ab.iter().copied().collect();
        assert!((ab_data[0] - 3.0).abs() < 1e-4);
        assert!((ab_data[3] - 8.0).abs() < 1e-4);

        // Second: AC = [[1,1],[2,2]]
        let ac = bridge
            .einsum_f32_result_f64("ij,jk->ik", &["A", "C"])
            .expect("second einsum");
        let ac_data: Vec<f64> = ac.iter().copied().collect();
        assert!((ac_data[0] - 1.0).abs() < 1e-4);
        assert!((ac_data[1] - 1.0).abs() < 1e-4);
        assert!((ac_data[2] - 2.0).abs() < 1e-4);
        assert!((ac_data[3] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_dual_bridge_default() {
        let bridge = DualPrecisionBridge::default();
        // A fresh bridge has no gradients.
        assert!(bridge.get_grad("any_name").is_none());
    }
}
