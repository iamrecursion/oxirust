//! SciRS2 f32 executor implementation.
//!
//! This module provides `Scirs2Exec32`, an f32-precision executor that mirrors
//! `Scirs2Exec` but uses `ArrayD<f32>` for all tensor operations.

use scirs2_core::ndarray::{ArrayD, Axis};
use std::collections::HashMap;
use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp, TlExecutor};

/// An f32 dynamic-rank tensor backed by ndarray.
pub type Scirs2Tensor32 = ArrayD<f32>;

/// SciRS2-backed executor operating in f32 precision.
///
/// This executor mirrors `Scirs2Exec` but uses `ArrayD<f32>` for all tensor
/// storage and computation, providing 50% memory savings compared to f64
/// at the cost of reduced numerical precision.
pub struct Scirs2Exec32 {
    pub tensors: HashMap<String, Scirs2Tensor32>,
}

impl Default for Scirs2Exec32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Scirs2Exec32 {
    /// Create a new, empty f32 executor.
    pub fn new() -> Self {
        Scirs2Exec32 {
            tensors: HashMap::new(),
        }
    }

    /// Insert a named tensor into the executor's store.
    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: Scirs2Tensor32) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Retrieve a reference to a named tensor.
    pub fn get_tensor(&self, name: &str) -> Option<&Scirs2Tensor32> {
        self.tensors.get(name)
    }
}

impl TlExecutor for Scirs2Exec32 {
    type Tensor = Scirs2Tensor32;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        if inputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "No input tensors provided".to_string(),
            ));
        }

        let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let view_refs: Vec<_> = views.iter().collect();

        scirs2_linalg::einsum(spec, &view_refs)
            .map_err(|e| ExecutorError::InvalidEinsumSpec(format!("Einsum error: {}", e)))
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        let result = match op {
            ElemOp::Relu => x.mapv(|v| v.max(0.0_f32)),
            ElemOp::Sigmoid => x.mapv(|v| 1.0_f32 / (1.0_f32 + (-v).exp())),
            ElemOp::OneMinus => x.mapv(|v| 1.0_f32 - v),
            _ => {
                return Err(ExecutorError::UnsupportedOperation(format!(
                    "Unary operation {:?} not supported",
                    op
                )))
            }
        };

        Ok(result)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        // Handle scalar broadcasting: if one tensor is scalar (shape []) and the other isn't,
        // broadcast the scalar to match the shape of the other tensor.
        let x_is_scalar = x.ndim() == 0;
        let y_is_scalar = y.ndim() == 0;

        let (x_broadcast, y_broadcast);
        let (x_ref, y_ref) = if x_is_scalar && !y_is_scalar {
            let scalar_value = x
                .iter()
                .next()
                .expect("scalar tensor has at least one element");
            x_broadcast = scirs2_core::ndarray::Array::from_elem(y.raw_dim(), *scalar_value);
            (&x_broadcast.view(), &y.view())
        } else if y_is_scalar && !x_is_scalar {
            let scalar_value = y
                .iter()
                .next()
                .expect("scalar tensor has at least one element");
            y_broadcast = scirs2_core::ndarray::Array::from_elem(x.raw_dim(), *scalar_value);
            (&x.view(), &y_broadcast.view())
        } else if x.shape() != y.shape() {
            return Err(ExecutorError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        } else {
            (&x.view(), &y.view())
        };

        let result = match op {
            ElemOp::Add => x_ref + y_ref,
            ElemOp::Subtract => x_ref - y_ref,
            ElemOp::Multiply => x_ref * y_ref,
            ElemOp::Divide => x_ref / y_ref,
            ElemOp::Min => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a.min(b)),
            ElemOp::Max => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a.max(b)),

            ElemOp::Eq => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if (a - b).abs() < 1e-7_f32 { 1.0 } else { 0.0 }),
            ElemOp::Lt => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a < b { 1.0 } else { 0.0 }),
            ElemOp::Gt => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a > b { 1.0 } else { 0.0 }),
            ElemOp::Lte => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a <= b { 1.0 } else { 0.0 }),
            ElemOp::Gte => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a >= b { 1.0 } else { 0.0 }),

            ElemOp::OrMax => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a.max(b)),
            ElemOp::OrProbSum => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a + b - a * b),
            ElemOp::Nand => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| 1.0_f32 - a * b),
            ElemOp::Nor => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| 1.0_f32 - a.max(b)),
            ElemOp::Xor => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a + b - 2.0_f32 * a * b),

            _ => {
                return Err(ExecutorError::UnsupportedOperation(format!(
                    "Binary operation {:?} not supported",
                    op
                )))
            }
        };

        Ok(result)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        if axes.is_empty() {
            return Ok(x.clone());
        }

        for &axis in axes {
            if axis >= x.ndim() {
                return Err(ExecutorError::ShapeMismatch(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    x.ndim()
                )));
            }
        }

        let mut result = x.clone();
        for &axis in axes.iter().rev() {
            result = match op {
                ReduceOp::Sum => result.sum_axis(Axis(axis)),
                ReduceOp::Max => result.fold_axis(Axis(axis), f32::NEG_INFINITY, |&a, &b| a.max(b)),
                ReduceOp::Min => result.fold_axis(Axis(axis), f32::INFINITY, |&a, &b| a.min(b)),
                ReduceOp::Mean => result
                    .mean_axis(Axis(axis))
                    .expect("axis is valid as validated earlier"),
                ReduceOp::Product => result.fold_axis(Axis(axis), 1.0_f32, |&a, &b| a * b),
            };
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn make_tensor(shape: &[usize], data: Vec<f32>) -> ArrayD<f32> {
        ArrayD::from_shape_vec(shape, data).expect("valid shape/data for test tensor")
    }

    #[test]
    fn test_exec32_einsum_matmul() {
        // 2×3 matrix × 3×2 matrix -> 2×2
        let a = make_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_tensor(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.einsum("ij,jk->ik", &[a, b]).expect("einsum matmul");
        assert_eq!(result.shape(), &[2, 2]);
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let data: Vec<f32> = result.iter().copied().collect();
        assert!(
            (data[0] - 58.0).abs() < 1e-4,
            "expected 58, got {}",
            data[0]
        );
        assert!(
            (data[1] - 64.0).abs() < 1e-4,
            "expected 64, got {}",
            data[1]
        );
        assert!(
            (data[2] - 139.0).abs() < 1e-4,
            "expected 139, got {}",
            data[2]
        );
        assert!(
            (data[3] - 154.0).abs() < 1e-4,
            "expected 154, got {}",
            data[3]
        );
    }

    #[test]
    fn test_exec32_relu() {
        let x = make_tensor(&[4], vec![-1.0, 0.0, 1.0, 2.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op(ElemOp::Relu, &x).expect("relu");
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 1.0);
        assert_eq!(data[3], 2.0);
    }

    #[test]
    fn test_exec32_sigmoid() {
        let x = make_tensor(&[4], vec![-2.0, 0.0, 1.0, 10.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op(ElemOp::Sigmoid, &x).expect("sigmoid");
        for &v in result.iter() {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {} not in (0,1)", v);
        }
        // sigmoid(0) == 0.5
        let data: Vec<f32> = result.iter().copied().collect();
        assert!((data[1] - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5");
    }

    #[test]
    fn test_exec32_one_minus() {
        let x = make_tensor(&[3], vec![0.0, 0.3, 1.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op(ElemOp::OneMinus, &x).expect("one_minus");
        let data: Vec<f32> = result.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.7).abs() < 1e-5);
        assert!((data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exec32_add() {
        let x = make_tensor(&[3], vec![1.0, 2.0, 3.0]);
        let y = make_tensor(&[3], vec![4.0, 5.0, 6.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op_binary(ElemOp::Add, &x, &y).expect("add");
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_exec32_sub() {
        let x = make_tensor(&[3], vec![10.0, 5.0, 3.0]);
        let y = make_tensor(&[3], vec![1.0, 2.0, 3.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op_binary(ElemOp::Subtract, &x, &y).expect("sub");
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![9.0, 3.0, 0.0]);
    }

    #[test]
    fn test_exec32_mul() {
        let x = make_tensor(&[3], vec![2.0, 3.0, 4.0]);
        let y = make_tensor(&[3], vec![5.0, 6.0, 7.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op_binary(ElemOp::Multiply, &x, &y).expect("mul");
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_exec32_div() {
        let x = make_tensor(&[3], vec![6.0, 9.0, 12.0]);
        let y = make_tensor(&[3], vec![2.0, 3.0, 4.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.elem_op_binary(ElemOp::Divide, &x, &y).expect("div");
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_exec32_reduce_sum() {
        // 2×3 matrix, sum along axis 0 -> [1,3] vector
        let x = make_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.reduce(ReduceOp::Sum, &x, &[0]).expect("reduce_sum");
        assert_eq!(result.shape(), &[3]);
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_exec32_reduce_max() {
        let x = make_tensor(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.reduce(ReduceOp::Max, &x, &[0]).expect("reduce_max");
        assert_eq!(result.shape(), &[3]);
        let data: Vec<f32> = result.iter().copied().collect();
        assert_eq!(data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_exec32_reduce_mean() {
        let x = make_tensor(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut exec = Scirs2Exec32::new();
        let result = exec.reduce(ReduceOp::Mean, &x, &[0]).expect("reduce_mean");
        assert_eq!(result.shape(), &[4]);
        let data: Vec<f32> = result.iter().copied().collect();
        for (got, expected) in data.iter().zip([3.0_f32, 4.0, 5.0, 6.0].iter()) {
            assert!(
                (got - expected).abs() < 1e-5,
                "mean mismatch: {} vs {}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_exec32_zeros() {
        let zeros: ArrayD<f32> = ArrayD::zeros(vec![2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert!(zeros.iter().all(|&v| v == 0.0_f32));
    }

    #[test]
    fn test_exec32_ones() {
        let ones: ArrayD<f32> = ArrayD::ones(vec![2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert!(ones.iter().all(|&v| v == 1.0_f32));
    }

    #[test]
    fn test_exec32_from_data() {
        let data = vec![1.5_f32, 2.5, 3.5, 4.5];
        let tensor = ArrayD::from_shape_vec(vec![2, 2], data.clone())
            .expect("valid shape for from_data test");
        assert_eq!(tensor.shape(), &[2, 2]);
        let roundtrip: Vec<f32> = tensor.iter().copied().collect();
        assert_eq!(roundtrip, data);
    }

    #[test]
    fn test_exec32_memory_half_of_f64() {
        // f32 is 4 bytes, f64 is 8 bytes; same element count means half total bytes.
        let f32_tensor: ArrayD<f32> = ArrayD::zeros(vec![4, 4]);
        let f64_tensor: ArrayD<f64> = ArrayD::zeros(vec![4, 4]);
        let f32_bytes = f32_tensor.len() * std::mem::size_of::<f32>();
        let f64_bytes = f64_tensor.len() * std::mem::size_of::<f64>();
        assert_eq!(f32_bytes * 2, f64_bytes);
    }
}
