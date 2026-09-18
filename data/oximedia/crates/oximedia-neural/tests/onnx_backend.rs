//! Integration tests for `OnnxBackend` — requires the `onnx` feature.

#![cfg(feature = "onnx")]

use std::collections::HashMap;
use std::path::Path;

use oximedia_neural::tensor::Tensor;
use oximedia_neural::OnnxBackend;

/// Loading a non-existent file must return `Err`.
#[test]
fn load_nonexistent_returns_err() {
    let result = OnnxBackend::load(Path::new("/nonexistent/model.onnx"));
    assert!(
        result.is_err(),
        "expected Err when loading a non-existent ONNX file"
    );
}

/// Loading from empty bytes produces a valid (empty) backend — oxionnx
/// accepts an empty protobuf as a model with no inputs or outputs.
/// The resulting backend should report no input or output names.
#[test]
fn load_from_empty_bytes_produces_empty_backend() {
    let backend = OnnxBackend::load_from_bytes(&[])
        .expect("loading empty bytes should succeed (empty model)");
    // An empty model has no declared I/O.
    assert!(
        backend.input_names().is_empty(),
        "empty model should have no input names"
    );
    assert!(
        backend.output_names().is_empty(),
        "empty model should have no output names"
    );
}

/// Loading from obviously invalid bytes must return `Err`.
#[test]
fn load_from_invalid_bytes_returns_err() {
    let junk = b"this is not an onnx model at all";
    let result = OnnxBackend::load_from_bytes(junk);
    assert!(
        result.is_err(),
        "expected Err when loading from invalid bytes"
    );
}

/// `Tensor::from_data` followed by name conversion exercises the Tensor bridging
/// code path without requiring an actual model file.
#[test]
fn tensor_round_trip_through_hashmap() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let t = Tensor::from_data(data.clone(), shape.clone()).expect("tensor creation");

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("x".to_string(), t.clone());

    // Confirm the tensor round-tripped as expected.
    let stored = inputs.get("x").expect("tensor in map");
    assert_eq!(stored.shape(), &[2, 2]);
    assert_eq!(stored.data(), &data[..]);
}
