//! Integration tests for the `onnx`-gated ML shot boundary detector.
//!
//! These tests deliberately avoid requiring a real ONNX model on disk —
//! they validate the public contract of [`oximedia_shots::MlShotDetector`]
//! (types are `pub`, builder methods chain, error conversion works)
//! without depending on any external fixture.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use oximedia_ml::{DeviceType, MlError};
use oximedia_shots::{MlShotDetector, ShotError, ShotResult};

/// Materialise the re-export path `oximedia_shots::MlShotDetector` so a
/// regression that hides the type surfaces immediately at the crate
/// boundary.
#[test]
fn ml_shot_detector_reexported_at_crate_root() {
    fn assert_type<T>() {}
    assert_type::<MlShotDetector>();
}

/// `MlShotDetector::from_path` against a nonexistent file must return the
/// crate-native [`ShotError`] with the ML error folded in via the
/// `#[from]` conversion.
#[test]
fn from_path_nonexistent_model_returns_shot_ml_error() {
    let temp = std::env::temp_dir().join("oximedia-shots-nonexistent-test-model.onnx");
    // Ensure the file really doesn't exist (in case a prior test left a stub).
    if temp.exists() {
        let _ = std::fs::remove_file(&temp);
    }
    let err: ShotError = MlShotDetector::from_path(&temp, DeviceType::Cpu)
        .expect_err("loading a missing model must fail");
    assert!(
        matches!(err, ShotError::MlError(_)),
        "expected ShotError::MlError, got {err:?}",
    );
}

/// The `From<MlError> for ShotError` conversion is wired up via
/// thiserror's `#[from]`.  Exercising it here guarantees that downstream
/// code can use `?` on ML results inside `ShotResult`-returning
/// functions.
#[test]
fn shot_error_from_ml_error_roundtrip() {
    fn forward() -> ShotResult<()> {
        // Manually synthesise an ML error and let `?` forward it.
        Err::<(), MlError>(MlError::FeatureDisabled("onnx"))?;
        Ok(())
    }
    let err = forward().expect_err("must propagate");
    assert!(
        matches!(err, ShotError::MlError(MlError::FeatureDisabled("onnx"))),
        "expected MlError::FeatureDisabled to round-trip, got {err:?}",
    );
}

/// Builder-style `with_threshold` / `with_window` both return `Self` so
/// callers can chain them immediately after `from_path`.  This test only
/// asserts the signatures compile — it never actually constructs a
/// detector because we have no real ONNX file in-repo.
#[test]
fn with_threshold_and_window_are_chainable_builders() {
    fn chain(detector: MlShotDetector) -> MlShotDetector {
        detector
            .with_threshold(0.5)
            .with_window(100)
            .with_threshold(0.7)
            .with_window(50)
    }
    // Asserting compilation is the contract.  Runtime construction of a
    // detector needs a real model, which is out of scope for in-tree
    // tests; we let `from_path_nonexistent_model_returns_shot_ml_error`
    // cover the fallible path.
    let _ = chain; // suppress dead-code in release profiles
}

/// `detect_boundaries` must reject frames with a non-RGB channel count.
/// Building a detector requires a real model, so this test drives the
/// error-shape contract through a unit-level alternative: it verifies
/// that the public error variant displays a stable prefix the same way
/// users will observe it from the runtime call.
#[test]
fn invalid_frame_error_shape_matches_contract() {
    let err = ShotError::InvalidFrame("frame 0: expected 3 channels (RGB), got 4".to_string());
    let msg = format!("{err}");
    assert!(
        msg.contains("Invalid frame data"),
        "public Display of InvalidFrame should be stable, got {msg}",
    );
}

/// Using a nonexistent path under `std::env::temp_dir()` proves the
/// error path is clean (not a panic / abort).  This complements
/// `from_path_nonexistent_model_returns_shot_ml_error` by exercising a
/// realistic filesystem location.
#[test]
fn missing_model_under_temp_dir_returns_clean_error() {
    let path: PathBuf = std::env::temp_dir()
        .join("oximedia-shots-ml-test")
        .join("never-created.onnx");
    let result = MlShotDetector::from_path(&path, DeviceType::Cpu);
    assert!(
        matches!(result, Err(ShotError::MlError(_))),
        "expected ShotError::MlError from missing model, got {result:?}",
    );
}
