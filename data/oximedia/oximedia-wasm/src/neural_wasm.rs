//! WebAssembly bindings for basic tensor operations from `oximedia-neural`.
//!
//! Exposes a lightweight WASM-friendly tensor that wraps the pure-Rust
//! `oximedia_neural::Tensor` type. Synchronous, no threads, no file-system.

use wasm_bindgen::prelude::*;

use oximedia_neural::{relu_inplace, Tensor};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// TensorWasm
// ---------------------------------------------------------------------------

/// A lightweight n-dimensional f32 tensor for browser-side ML inference.
///
/// # Example
///
/// ```javascript
/// const t = new TensorWasm([1.0, -2.0, 3.0], [3]);
/// const r = t.relu();
/// console.log(r.get(1)); // 0.0
/// ```
#[wasm_bindgen]
pub struct TensorWasm {
    inner: Tensor,
}

#[wasm_bindgen]
impl TensorWasm {
    /// Create a new tensor from a flat data buffer and a shape.
    ///
    /// `data` must have exactly as many elements as the product of `shape`.
    /// All dimensions must be > 0.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != product(shape)`, or if any dimension is zero.
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<u32>) -> Result<TensorWasm, JsValue> {
        if shape.is_empty() {
            return Err(js_err("shape must have at least one dimension"));
        }
        if shape.iter().any(|&d| d == 0) {
            return Err(js_err("all shape dimensions must be > 0"));
        }
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let expected: usize = shape_usize.iter().product();
        if data.len() != expected {
            return Err(js_err(format!(
                "data length {} does not match shape product {}",
                data.len(),
                expected
            )));
        }
        let inner = Tensor::from_data(data, shape_usize).map_err(|e| js_err(e))?;
        Ok(TensorWasm { inner })
    }

    /// Return the shape of this tensor as a `Vec<u32>`.
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| d as u32).collect()
    }

    /// Get the element at flat index `idx`.
    ///
    /// Returns `f32::NAN` if `idx` is out of bounds.
    pub fn get(&self, idx: u32) -> f32 {
        self.inner
            .data()
            .get(idx as usize)
            .copied()
            .unwrap_or(f32::NAN)
    }

    /// Return the total number of elements.
    pub fn numel(&self) -> u32 {
        self.inner.numel() as u32
    }

    /// Apply the ReLU activation element-wise (max(0, x)) and return a new tensor.
    pub fn relu(&self) -> TensorWasm {
        let mut out = self.inner.clone();
        relu_inplace(&mut out);
        TensorWasm { inner: out }
    }

    /// Return a copy of the flat data buffer.
    pub fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_valid() {
        let t = TensorWasm::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(t.is_ok(), "should construct with matching data and shape");
    }

    #[test]
    fn constructor_rejects_mismatched_shape() {
        let t = TensorWasm::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(t.is_err(), "3 elements with shape [2,2] should fail");
    }

    #[test]
    fn constructor_rejects_zero_dimension() {
        let t = TensorWasm::new(vec![], vec![0, 4]);
        assert!(t.is_err(), "zero dimension should be rejected");
    }

    #[test]
    fn shape_round_trips() {
        let t = TensorWasm::new(vec![0.0; 6], vec![2, 3]).expect("valid");
        assert_eq!(t.shape(), vec![2u32, 3u32]);
    }

    #[test]
    fn get_element() {
        let t = TensorWasm::new(vec![10.0, 20.0, 30.0], vec![3]).expect("valid");
        assert_eq!(t.get(1), 20.0);
    }

    #[test]
    fn get_out_of_bounds_returns_nan() {
        let t = TensorWasm::new(vec![1.0], vec![1]).expect("valid");
        assert!(t.get(99).is_nan(), "OOB get should return NaN");
    }

    #[test]
    fn relu_clamps_negatives() {
        let t = TensorWasm::new(vec![1.0, -2.0, 0.0, 3.0], vec![4]).expect("valid");
        let r = t.relu();
        assert_eq!(r.get(0), 1.0);
        assert_eq!(r.get(1), 0.0, "negative should become 0");
        assert_eq!(r.get(2), 0.0);
        assert_eq!(r.get(3), 3.0);
    }
}
