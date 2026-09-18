//! WASM-specific tests using wasm-bindgen-test

use scirs2_wasm::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_version() {
    let v = version();
    assert!(!v.is_empty());
    assert!(v.starts_with("0."));
}

#[wasm_bindgen_test]
fn test_capabilities() {
    let caps = capabilities();
    assert!(!caps.is_null());
    assert!(!caps.is_undefined());
}

#[wasm_bindgen_test]
fn test_array_creation() {
    let js_array = js_sys::Array::new();
    js_array.push(&1.0.into());
    js_array.push(&2.0.into());
    js_array.push(&3.0.into());

    let arr = array::WasmArray::new(&js_array.into()).expect("Failed to create array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.ndim(), 1);
}

#[wasm_bindgen_test]
fn test_array_zeros() {
    let shape = js_sys::Array::new();
    shape.push(&3.0.into());
    shape.push(&4.0.into());

    let zeros = array::WasmArray::zeros(&shape.into()).expect("Failed to create zeros");
    assert_eq!(zeros.len(), 12);
    assert_eq!(zeros.ndim(), 2);
}

#[wasm_bindgen_test]
fn test_array_ones() {
    let shape = js_sys::Array::new();
    shape.push(&5.0.into());

    let ones = array::WasmArray::ones(&shape.into()).expect("Failed to create ones");
    assert_eq!(ones.len(), 5);
    assert_eq!(array::sum(&ones), 5.0);
}

#[wasm_bindgen_test]
fn test_array_linspace() {
    let arr = array::WasmArray::linspace(0.0, 10.0, 11).expect("Failed to create linspace");
    assert_eq!(arr.len(), 11);
    assert!((arr.get(0).unwrap() - 0.0).abs() < 1e-10);
    assert!((arr.get(10).unwrap() - 10.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_array_arange() {
    let arr = array::WasmArray::arange(0.0, 5.0, 1.0).expect("Failed to create arange");
    assert_eq!(arr.len(), 5);
    assert_eq!(arr.get(0).unwrap(), 0.0);
    assert_eq!(arr.get(4).unwrap(), 4.0);
}

#[wasm_bindgen_test]
fn test_array_operations() {
    let js_a = js_sys::Array::new();
    js_a.push(&1.0.into());
    js_a.push(&2.0.into());
    js_a.push(&3.0.into());

    let js_b = js_sys::Array::new();
    js_b.push(&4.0.into());
    js_b.push(&5.0.into());
    js_b.push(&6.0.into());

    let a = array::WasmArray::new(&js_a.into()).expect("Failed to create array a");
    let b = array::WasmArray::new(&js_b.into()).expect("Failed to create array b");

    // Test add
    let sum = array::add(&a, &b).expect("Failed to add");
    assert_eq!(sum.get(0).unwrap(), 5.0);
    assert_eq!(sum.get(1).unwrap(), 7.0);
    assert_eq!(sum.get(2).unwrap(), 9.0);

    // Test subtract
    let diff = array::subtract(&a, &b).expect("Failed to subtract");
    assert_eq!(diff.get(0).unwrap(), -3.0);

    // Test multiply
    let prod = array::multiply(&a, &b).expect("Failed to multiply");
    assert_eq!(prod.get(0).unwrap(), 4.0);
    assert_eq!(prod.get(1).unwrap(), 10.0);
    assert_eq!(prod.get(2).unwrap(), 18.0);
}

#[wasm_bindgen_test]
fn test_array_reductions() {
    let js_array = js_sys::Array::new();
    js_array.push(&1.0.into());
    js_array.push(&2.0.into());
    js_array.push(&3.0.into());
    js_array.push(&4.0.into());
    js_array.push(&5.0.into());

    let arr = array::WasmArray::new(&js_array.into()).expect("Failed to create array");

    assert_eq!(array::sum(&arr), 15.0);
    assert_eq!(array::mean(&arr), 3.0);
    assert_eq!(array::min(&arr), 1.0);
    assert_eq!(array::max(&arr), 5.0);
}

#[wasm_bindgen_test]
fn test_statistics() {
    let js_array = js_sys::Array::new();
    for i in 1..=10 {
        js_array.push(&(i as f64).into());
    }

    let arr = array::WasmArray::new(&js_array.into()).expect("Failed to create array");

    let mean_val = array::mean(&arr);
    assert!((mean_val - 5.5).abs() < 1e-10);

    let std_val = stats::std(&arr);
    assert!(std_val > 0.0);

    let median_val = stats::median(&arr);
    assert_eq!(median_val, 5.5);
}

#[wasm_bindgen_test]
fn test_random_uniform() {
    let shape = js_sys::Array::new();
    shape.push(&100.0.into());

    let arr = random::random_uniform(&shape.into()).expect("Failed to generate random");
    assert_eq!(arr.len(), 100);

    // Check that values are in [0, 1)
    for i in 0..100 {
        let val = arr.get(i).unwrap();
        assert!((0.0..1.0).contains(&val));
    }
}

#[wasm_bindgen_test]
fn test_random_normal() {
    let shape = js_sys::Array::new();
    shape.push(&1000.0.into());

    let arr =
        random::random_normal(&shape.into(), 0.0, 1.0).expect("Failed to generate random normal");
    assert_eq!(arr.len(), 1000);

    // Check approximate mean and std (should be close to 0 and 1)
    let mean_val = array::mean(&arr);
    let std_val = stats::std(&arr);

    assert!((mean_val).abs() < 0.2); // Within reasonable range
    assert!((std_val - 1.0).abs() < 0.2); // Within reasonable range
}

#[wasm_bindgen_test]
fn test_linalg_det() {
    // 2x2 matrix
    let shape = js_sys::Array::new();
    shape.push(&2.0.into());
    shape.push(&2.0.into());

    let data = js_sys::Array::new();
    data.push(&1.0.into());
    data.push(&2.0.into());
    data.push(&3.0.into());
    data.push(&4.0.into());

    let matrix =
        array::WasmArray::from_shape(&shape.into(), &data.into()).expect("Failed to create matrix");

    let det = linalg::det(&matrix).expect("Failed to compute determinant");
    assert!((det - (-2.0)).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_linalg_trace() {
    let shape = js_sys::Array::new();
    shape.push(&3.0.into());
    shape.push(&3.0.into());

    let data = js_sys::Array::new();
    for i in 1..=9 {
        data.push(&(i as f64).into());
    }

    let matrix =
        array::WasmArray::from_shape(&shape.into(), &data.into()).expect("Failed to create matrix");

    let trace = linalg::trace(&matrix).expect("Failed to compute trace");
    assert_eq!(trace, 15.0); // 1 + 5 + 9
}

#[wasm_bindgen_test]
fn test_performance_timer() {
    let timer = PerformanceTimer::new("test".to_string()).expect("Failed to create timer");

    // Do some work
    let shape = js_sys::Array::new();
    shape.push(&1000.0.into());
    let _arr = array::WasmArray::ones(&shape.into()).unwrap();

    let elapsed = timer.elapsed().expect("Failed to get elapsed time");
    assert!(elapsed >= 0.0);
}

#[wasm_bindgen_test]
fn test_array_reshape() {
    let shape = js_sys::Array::new();
    shape.push(&6.0.into());

    let arr = array::WasmArray::ones(&shape.into()).expect("Failed to create array");

    let new_shape = js_sys::Array::new();
    new_shape.push(&2.0.into());
    new_shape.push(&3.0.into());

    let reshaped = arr.reshape(&new_shape.into()).expect("Failed to reshape");
    assert_eq!(reshaped.ndim(), 2);
    assert_eq!(reshaped.len(), 6);
}

#[wasm_bindgen_test]
fn test_cumulative_operations() {
    let js_array = js_sys::Array::new();
    for i in 1..=5 {
        js_array.push(&(i as f64).into());
    }

    let arr = array::WasmArray::new(&js_array.into()).expect("Failed to create array");

    let cumsum = stats::cumsum(&arr);
    assert_eq!(cumsum.get(4).unwrap(), 15.0); // 1+2+3+4+5

    let cumprod = stats::cumprod(&arr);
    assert_eq!(cumprod.get(4).unwrap(), 120.0); // 1*2*3*4*5
}
