//! Utility functions for WASM bindings

use crate::error::{WasmError, WasmResult};
use wasm_bindgen::prelude::*;

/// Convert a JavaScript Array to a `Vec<f64>`
pub fn js_array_to_vec_f64(js_array: &js_sys::Array) -> WasmResult<Vec<f64>> {
    let mut vec = Vec::with_capacity(js_array.length() as usize);

    for i in 0..js_array.length() {
        let val = js_array.get(i).as_f64().ok_or_else(|| {
            WasmError::InvalidParameter(format!("Array element at index {} is not a number", i))
        })?;
        vec.push(val);
    }

    Ok(vec)
}

/// Convert a `Vec<f64>` to a JavaScript Array
pub fn vec_f64_to_js_array(vec: &[f64]) -> js_sys::Array {
    let array = js_sys::Array::new_with_length(vec.len() as u32);

    for (i, &val) in vec.iter().enumerate() {
        array.set(i as u32, JsValue::from_f64(val));
    }

    array
}

/// Convert a JavaScript typed array to `Vec<f64>`
pub fn typed_array_to_vec_f64(typed_array: &JsValue) -> WasmResult<Vec<f64>> {
    if typed_array.is_object() {
        // Try Float64Array first
        if js_sys::Float64Array::instanceof(typed_array) {
            let float64_array = js_sys::Float64Array::new(typed_array);
            Ok(float64_array.to_vec())
        } else if js_sys::Float32Array::instanceof(typed_array) {
            let float32_array = js_sys::Float32Array::new(typed_array);
            Ok(float32_array
                .to_vec()
                .into_iter()
                .map(|x| x as f64)
                .collect())
        } else {
            Err(WasmError::InvalidParameter(
                "Expected Float32Array or Float64Array".to_string(),
            ))
        }
    } else {
        Err(WasmError::InvalidParameter(
            "Expected Float32Array or Float64Array".to_string(),
        ))
    }
}

/// Convert `Vec<f64>` to Float64Array (zero-copy when possible)
pub fn vec_f64_to_typed_array(vec: Vec<f64>) -> js_sys::Float64Array {
    let array = js_sys::Float64Array::new_with_length(vec.len() as u32);
    array.copy_from(&vec);
    array
}

/// Parse shape from JavaScript value
pub fn parse_shape(js_value: &JsValue) -> WasmResult<Vec<usize>> {
    if js_value.is_array() {
        let array = js_sys::Array::from(js_value);
        let mut shape = Vec::with_capacity(array.length() as usize);
        for i in 0..array.length() {
            let val = array.get(i).as_f64().ok_or_else(|| {
                WasmError::InvalidDimensions("Shape dimensions must be numbers".to_string())
            })?;

            if val < 0.0 || val.fract() != 0.0 {
                return Err(WasmError::InvalidDimensions(
                    "Shape dimensions must be non-negative integers".to_string(),
                ));
            }

            shape.push(val as usize);
        }
        Ok(shape)
    } else {
        Err(WasmError::InvalidDimensions(
            "Shape must be an array".to_string(),
        ))
    }
}

/// Format a shape for display
pub fn format_shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[2, 3, 4]), "[2, 3, 4]");
        assert_eq!(format_shape(&[]), "[]");
        assert_eq!(format_shape(&[100]), "[100]");
    }
}
