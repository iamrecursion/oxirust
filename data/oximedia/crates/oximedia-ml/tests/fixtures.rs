//! Shared test fixtures for integration tests.
//!
//! Fixtures are intentionally "data-only": they do not touch the file
//! system outside of `std::env::temp_dir()` and do not depend on any
//! ONNX model file, so they work in any build configuration.

#![allow(dead_code)]

use std::path::PathBuf;

/// Return a synthetic 2×2 RGB (u8) image — three distinct pixels + one black.
#[must_use]
pub fn tiny_rgb_image() -> (Vec<u8>, u32, u32) {
    let pixels = vec![
        255, 0, 0, 0, 255, 0, // row 0: red, green
        0, 0, 255, 0, 0, 0, // row 1: blue, black
    ];
    (pixels, 2, 2)
}

/// Return a fake ".onnx" path inside the OS temp dir.
///
/// The file is **not** created, so loader tests exercise the error
/// branch cleanly.
#[must_use]
pub fn missing_model_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("oximedia-ml-tests-nonexistent-{name}.onnx"));
    p
}
