//! Integration tests for `ModelCache` behaviour.

use oximedia_ml::{DeviceType, MlError, ModelCache, DEFAULT_CAPACITY};

mod fixtures;

#[test]
fn default_cache_has_expected_capacity() {
    let cache = ModelCache::new();
    assert_eq!(cache.capacity(), DEFAULT_CAPACITY);
    assert_eq!(cache.len().expect("cache.len"), 0);
    assert!(cache.is_empty().expect("cache.is_empty"));
}

#[test]
fn zero_capacity_is_rejected() {
    let err = ModelCache::with_capacity(0).expect_err("must fail");
    assert!(matches!(err, MlError::CacheCapacityZero));
}

#[test]
fn missing_file_surfaces_load_error() {
    let cache = ModelCache::new();
    let path = fixtures::missing_model_path("missing_file_surfaces_load_error");
    let result = cache.get_or_load(&path, DeviceType::Cpu);
    // Without the onnx feature this is a FeatureDisabled error.
    // With the onnx feature this is a ModelLoad error from the backend.
    match result {
        Ok(_) => panic!("expected error for non-existent model"),
        Err(MlError::FeatureDisabled(_)) => {
            // Expected when running without the `onnx` feature.
        }
        Err(MlError::ModelLoad { .. }) => {
            // Expected when running with the `onnx` feature — backend can't find the file.
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remove_missing_entry_is_ok() {
    let cache = ModelCache::new();
    let path = fixtures::missing_model_path("remove_missing_entry_is_ok");
    let removed = cache.remove(&path).expect("ok");
    assert!(removed.is_none());
}

#[test]
fn clear_on_empty_cache_is_ok() {
    let cache = ModelCache::new();
    cache.clear().expect("ok");
    assert_eq!(cache.len().expect("cache.len"), 0);
}

#[test]
fn capacity_one_is_valid() {
    let cache = ModelCache::with_capacity(1).expect("ok");
    assert_eq!(cache.capacity(), 1);
}
