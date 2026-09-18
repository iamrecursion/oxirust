//! Integration tests for the `onnx`-gated ML scene enricher.
//!
//! These tests deliberately avoid requiring a real ONNX model on disk —
//! they validate the public contract of [`oximedia_scene::MlSceneEnricher`]
//! (types are `pub`, builder methods chain, error conversion works)
//! without depending on any external fixture.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use oximedia_ml::{DeviceType, MlError};
use oximedia_scene::{MlSceneEnricher, SceneError, SceneResult};

/// Materialise the re-export path `oximedia_scene::MlSceneEnricher` so a
/// regression that hides the type surfaces immediately at the crate
/// boundary.
#[test]
fn ml_scene_enricher_reexported_at_crate_root() {
    fn assert_type<T>() {}
    assert_type::<MlSceneEnricher>();
}

/// `MlSceneEnricher::from_path` against a nonexistent file must return the
/// crate-native [`SceneError`] with the ML error folded in via the
/// `#[from]` conversion.
#[test]
fn from_path_nonexistent_model_returns_scene_ml_error() {
    let labels = vec!["indoor".to_string(), "outdoor".to_string()];
    let bogus = PathBuf::from("/nonexistent-oximedia-scene-test-model.onnx");
    let err: SceneError = MlSceneEnricher::from_path(&bogus, labels, DeviceType::Cpu)
        .expect_err("loading a missing model must fail");
    assert!(
        matches!(err, SceneError::MlError(_)),
        "expected SceneError::MlError, got {err:?}",
    );
}

/// The `From<MlError> for SceneError` conversion is wired up via
/// thiserror's `#[from]`.  Exercising it here guarantees that downstream
/// code can use `?` on ML results inside `SceneResult`-returning
/// functions.
#[test]
fn scene_error_from_ml_error_roundtrip() {
    fn forward() -> SceneResult<()> {
        // Manually synthesise an ML error and let `?` forward it.
        Err::<(), MlError>(MlError::FeatureDisabled("onnx"))?;
        Ok(())
    }
    let err = forward().expect_err("must propagate");
    assert!(
        matches!(err, SceneError::MlError(MlError::FeatureDisabled("onnx"))),
        "expected MlError::FeatureDisabled to round-trip, got {err:?}",
    );
}

/// Builder-style `with_top_k` returns `Self` so callers can chain it
/// immediately after `from_path`.  This test only asserts the signature
/// compiles — it never actually constructs an enricher because we have no
/// real ONNX file in-repo.
#[test]
fn with_top_k_is_a_chainable_builder() {
    fn chain(enricher: MlSceneEnricher) -> MlSceneEnricher {
        enricher.with_top_k(3).with_top_k(5)
    }
    // Asserting compilation is the contract.  Runtime construction of an
    // enricher needs a real model, which is out of scope for in-tree
    // tests; we let `from_path_nonexistent_model_returns_scene_ml_error`
    // cover the fallible path.
    let _ = chain; // suppress dead-code in release profiles
}

/// `classify_frame` must reject buffers whose length disagrees with
/// `width * height * 3`.  Building an enricher requires a real model, so
/// this test drives the dimension check through a unit-level alternative:
/// it verifies the error-shape contract by constructing a `SceneError`
/// directly and pattern matching, which is the same error users will see
/// from `classify_frame`.
#[test]
fn invalid_dimensions_error_shape_matches_contract() {
    let err = SceneError::InvalidDimensions("expected 12 bytes, got 10".to_string());
    let msg = format!("{err}");
    assert!(
        msg.contains("Invalid dimensions"),
        "public Display of InvalidDimensions should be stable, got {msg}",
    );
}
