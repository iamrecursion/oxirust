//! Integration tests for the `onnx`-gated ML embedding extractor.
//!
//! These tests deliberately avoid requiring a real ONNX model on disk —
//! they validate the public contract of
//! [`oximedia_recommend::ml::EmbeddingExtractor`] and
//! [`oximedia_recommend::ml::ContentEmbedding`] (types are `pub`,
//! builder methods chain, error conversion works, rank_by_similarity
//! contract holds) without depending on any external fixture.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use oximedia_ml::{DeviceType, MlError};
use oximedia_recommend::{
    rank_by_similarity, ContentEmbedding, EmbeddingExtractor, RecommendError, RecommendResult,
};

/// Materialise the re-export path `oximedia_recommend::EmbeddingExtractor`
/// so a regression that hides the type surfaces immediately at the crate
/// boundary.
#[test]
fn embedding_extractor_reexported_at_crate_root() {
    fn assert_type<T>() {}
    assert_type::<EmbeddingExtractor>();
    assert_type::<ContentEmbedding>();
}

/// `EmbeddingExtractor::from_path` against a nonexistent file must return
/// the crate-native [`RecommendError`] with the ML error folded in via
/// the `#[from]` conversion.
#[test]
fn from_path_nonexistent_model_returns_ml_error() {
    let temp = std::env::temp_dir().join("oximedia-recommend-nonexistent-test-model.onnx");
    // Ensure the file really doesn't exist (in case a prior test left a stub).
    if temp.exists() {
        let _ = std::fs::remove_file(&temp);
    }
    let err: RecommendError = EmbeddingExtractor::from_path(&temp, DeviceType::Cpu)
        .expect_err("loading a missing model must fail");
    assert!(
        matches!(err, RecommendError::Ml(_)),
        "expected RecommendError::Ml, got {err:?}",
    );
}

/// The `From<MlError> for RecommendError` conversion is wired up via
/// thiserror's `#[from]`.  Exercising it here guarantees that downstream
/// code can use `?` on ML results inside `RecommendResult`-returning
/// functions.
#[test]
fn recommend_error_from_ml_error_roundtrip() {
    fn forward() -> RecommendResult<()> {
        // Manually synthesise an ML error and let `?` forward it.
        Err::<(), MlError>(MlError::FeatureDisabled("onnx"))?;
        Ok(())
    }
    let err = forward().expect_err("must propagate");
    assert!(
        matches!(err, RecommendError::Ml(MlError::FeatureDisabled("onnx"))),
        "expected MlError::FeatureDisabled to round-trip, got {err:?}",
    );
}

/// Builder-style `with_input_name` / `with_output_name` both return `Self`
/// so callers can chain them immediately after `from_path`.  This test
/// only asserts the signatures compile — it never actually constructs a
/// detector because we have no real ONNX file in-repo.
#[test]
fn with_input_and_output_names_are_chainable_builders() {
    fn chain(extractor: EmbeddingExtractor) -> EmbeddingExtractor {
        extractor
            .with_input_name("images")
            .with_output_name("features")
            .with_input_name("image_tensor")
            .with_output_name("pooled_output")
    }
    // Asserting compilation is the contract.  Runtime construction of an
    // extractor needs a real model, which is out of scope for in-tree
    // tests; we let `from_path_nonexistent_model_returns_ml_error` cover
    // the fallible path.
    let _ = chain; // suppress dead-code in release profiles
}

/// `ContentEmbedding::new` on an empty vector must reject the input
/// cleanly via [`RecommendError::InvalidSimilarity`] so downstream code
/// can distinguish malformed embeddings from a model-load failure.
#[test]
fn content_embedding_empty_input_shape_matches_contract() {
    let err = ContentEmbedding::new(Vec::<f32>::new()).expect_err("empty buffers must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("Invalid similarity"),
        "public Display of InvalidSimilarity should be stable, got {msg}",
    );
}

/// `rank_by_similarity` must sort by descending cosine similarity,
/// truncate to `top_k`, and report indices into the original candidate
/// slice — this is the single most load-bearing invariant for
/// recommendation callers.
#[test]
fn rank_by_similarity_orders_candidates_correctly() {
    let query = ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("unit-norm query must build");
    let candidates = vec![
        ContentEmbedding::new(vec![0.0_f32, 1.0, 0.0]).expect("ok"),
        ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok"),
        ContentEmbedding::new(vec![0.7071_f32, 0.7071, 0.0]).expect("ok"),
    ];
    let ranked = rank_by_similarity(&query, &candidates, 3);
    assert_eq!(ranked.len(), 3);
    // Highest similarity: the identical vector at candidate index 1.
    assert_eq!(ranked[0].0, 1);
    assert!((ranked[0].1 - 1.0).abs() < 1e-5);
    // Second: the 45-degree vector at index 2.
    assert_eq!(ranked[1].0, 2);
    assert!((ranked[1].1 - 0.7071).abs() < 1e-3);
    // Third: the orthogonal vector at index 0.
    assert_eq!(ranked[2].0, 0);
    assert!(ranked[2].1.abs() < 1e-5);
}

/// Using a nonexistent path under `std::env::temp_dir()` proves the
/// error path is clean (not a panic / abort).  This complements
/// `from_path_nonexistent_model_returns_ml_error` by exercising a
/// realistic filesystem location.
#[test]
fn missing_model_under_temp_dir_returns_clean_error() {
    let path: PathBuf = std::env::temp_dir()
        .join("oximedia-recommend-ml-test")
        .join("never-created.onnx");
    let result = EmbeddingExtractor::from_path(&path, DeviceType::Cpu);
    assert!(
        matches!(result, Err(RecommendError::Ml(_))),
        "expected RecommendError::Ml from missing model, got {result:?}",
    );
}
