//! Statistical functions for WASM

use crate::array::WasmArray;
use crate::error::WasmError;
use wasm_bindgen::prelude::*;

/// Compute the standard deviation of an array
#[wasm_bindgen]
pub fn std(arr: &WasmArray) -> f64 {
    std_with_ddof(arr, 0)
}

/// Compute the standard deviation with specified degrees of freedom
#[wasm_bindgen]
pub fn std_with_ddof(arr: &WasmArray, ddof: usize) -> f64 {
    let n = arr.len();
    if n <= ddof {
        return f64::NAN;
    }

    let mean_val = super::array::mean(arr);
    let variance = arr
        .data()
        .iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>()
        / (n - ddof) as f64;

    variance.sqrt()
}

/// Compute the variance of an array
#[wasm_bindgen]
pub fn variance(arr: &WasmArray) -> f64 {
    variance_with_ddof(arr, 0)
}

/// Compute the variance with specified degrees of freedom
#[wasm_bindgen]
pub fn variance_with_ddof(arr: &WasmArray, ddof: usize) -> f64 {
    let n = arr.len();
    if n <= ddof {
        return f64::NAN;
    }

    let mean_val = super::array::mean(arr);
    arr.data()
        .iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>()
        / (n - ddof) as f64
}

/// Compute the median of an array
#[wasm_bindgen]
pub fn median(arr: &WasmArray) -> f64 {
    let mut sorted: Vec<f64> = arr.data().iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }

    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute a percentile of an array
#[wasm_bindgen]
pub fn percentile(arr: &WasmArray, q: f64) -> Result<f64, JsValue> {
    if !(0.0..=100.0).contains(&q) {
        return Err(WasmError::InvalidParameter(
            "Percentile must be between 0 and 100".to_string(),
        )
        .into());
    }

    let mut sorted: Vec<f64> = arr.data().iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return Ok(f64::NAN);
    }

    let index = (q / 100.0) * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        Ok(sorted[lower])
    } else {
        let fraction = index - lower as f64;
        Ok(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
    }
}

/// Compute correlation coefficient between two arrays
#[wasm_bindgen]
pub fn corrcoef(x: &WasmArray, y: &WasmArray) -> Result<f64, JsValue> {
    if x.len() != y.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    let n = x.len() as f64;
    if n == 0.0 {
        return Ok(f64::NAN);
    }

    let x_mean = super::array::mean(x);
    let y_mean = super::array::mean(y);

    let mut covariance = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;

    for i in 0..x.len() {
        let x_val = x
            .get(i)
            .map_err(|_| WasmError::ComputationError("Index error".to_string()))?;
        let y_val = y
            .get(i)
            .map_err(|_| WasmError::ComputationError("Index error".to_string()))?;

        let x_diff = x_val - x_mean;
        let y_diff = y_val - y_mean;

        covariance += x_diff * y_diff;
        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
    }

    if x_variance == 0.0 || y_variance == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(covariance / (x_variance * y_variance).sqrt())
}

/// Compute cumulative sum of an array
#[wasm_bindgen]
pub fn cumsum(arr: &WasmArray) -> WasmArray {
    let mut sum = 0.0;
    let vec: Vec<f64> = arr
        .data()
        .iter()
        .map(|&x| {
            sum += x;
            sum
        })
        .collect();

    WasmArray::from_array(
        ndarray::ArrayD::from_shape_vec(arr.data().shape().to_vec(), vec)
            .expect("Shape mismatch in cumsum"),
    )
}

/// Compute cumulative product of an array
#[wasm_bindgen]
pub fn cumprod(arr: &WasmArray) -> WasmArray {
    let mut prod = 1.0;
    let vec: Vec<f64> = arr
        .data()
        .iter()
        .map(|&x| {
            prod *= x;
            prod
        })
        .collect();

    WasmArray::from_array(
        ndarray::ArrayD::from_shape_vec(arr.data().shape().to_vec(), vec)
            .expect("Shape mismatch in cumprod"),
    )
}
