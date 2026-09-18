//! Integration tests for the `onnx`-gated ML music-tagging pipeline.
//!
//! These tests deliberately avoid requiring a real ONNX model on disk —
//! they validate the public contract of
//! [`oximedia_mir::ml::MusicTagger`], [`oximedia_mir::ml::MusicTags`],
//! and [`oximedia_mir::ml::TagActivation`] (types are `pub`, builder
//! methods chain, error conversion works, `activate_and_rank` contract
//! holds) without depending on any external fixture.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use oximedia_mir::{
    activate_and_rank, apply_activation, MirError, MirResult, MusicTagger, MusicTags,
    TagActivation, TagActivationScore,
};
use oximedia_ml::{DeviceType, MlError};

/// Materialise the re-export paths at the crate root so a regression
/// that hides any of the ML types surfaces immediately at the public
/// boundary.
#[test]
fn ml_types_reexported_at_crate_root() {
    fn assert_type<T>() {}
    assert_type::<MusicTagger>();
    assert_type::<MusicTags>();
    assert_type::<TagActivation>();
    assert_type::<TagActivationScore>();
}

/// `MusicTagger::from_path` against a nonexistent file must return the
/// crate-native [`MirError`] with the ML error folded in via the
/// `#[from]` conversion.
#[test]
fn from_path_nonexistent_model_returns_ml_error() {
    let temp = std::env::temp_dir().join("oximedia-mir-nonexistent-music-tagger.onnx");
    if temp.exists() {
        let _ = std::fs::remove_file(&temp);
    }
    let err: MirError = MusicTagger::from_path(&temp, DeviceType::Cpu)
        .expect_err("loading a missing model must fail");
    assert!(
        matches!(err, MirError::Ml(_)),
        "expected MirError::Ml, got {err:?}",
    );
}

/// The `From<MlError> for MirError` conversion is wired up via
/// thiserror's `#[from]`.  Exercising it here guarantees that
/// downstream code can use `?` on ML results inside
/// `MirResult`-returning functions.
#[test]
fn mir_error_from_ml_error_roundtrip() {
    fn forward() -> MirResult<()> {
        Err::<(), MlError>(MlError::FeatureDisabled("onnx"))?;
        Ok(())
    }
    let err = forward().expect_err("must propagate");
    assert!(
        matches!(err, MirError::Ml(MlError::FeatureDisabled("onnx"))),
        "expected MlError::FeatureDisabled to round-trip, got {err:?}",
    );
}

/// Builder-style `with_*` methods all return `Self` so callers can
/// chain them fluently after `from_path` / `from_shared_model`.  This
/// test only asserts the signatures compile — it never actually
/// constructs a tagger because we have no real ONNX file in-repo.
#[test]
fn builder_methods_are_chainable() {
    fn chain(tagger: MusicTagger) -> MusicTagger {
        tagger
            .with_input_name("audio")
            .with_output_name("logits")
            .with_labels(vec!["rock".to_string(), "jazz".to_string()])
            .with_top_k(3)
            .with_activation(TagActivation::Sigmoid)
            .with_activation(TagActivation::None)
            .with_top_k(1)
    }
    // Asserting compilation is the contract.  Runtime construction of
    // a tagger needs a real model, which is out of scope for in-tree
    // tests; the `from_path_nonexistent_model_returns_ml_error` test
    // covers the fallible path.
    let _ = chain; // suppress dead-code in release profiles
}

/// Using a nonexistent path under `std::env::temp_dir()` proves the
/// error path is clean (not a panic / abort) in a realistic filesystem
/// location.
#[test]
fn missing_model_under_temp_dir_returns_clean_error() {
    let path: PathBuf = std::env::temp_dir()
        .join("oximedia-mir-ml-test")
        .join("never-created.onnx");
    let result = MusicTagger::from_path(&path, DeviceType::Cpu);
    assert!(
        matches!(result, Err(MirError::Ml(_))),
        "expected MirError::Ml from missing model, got {result:?}",
    );
}

/// `activate_and_rank` is the pure post-processing helper that the
/// `MusicTagger::classify` method routes through.  Exercising it at
/// the integration boundary guarantees the re-export is stable and the
/// sort-descending contract holds across crate API surface.
#[test]
fn activate_and_rank_via_crate_surface_sorts_descending() {
    let labels = vec![
        "blues".to_string(),
        "classical".to_string(),
        "country".to_string(),
        "disco".to_string(),
        "hiphop".to_string(),
    ];
    let logits = vec![0.05_f32, 0.85, 0.15, 0.75, 0.25];
    let ranked = activate_and_rank(&logits, &labels, 3, TagActivation::Softmax).expect("ok");
    assert_eq!(ranked.len(), 3);
    // Top must be class 1 (the largest logit) and scores descending.
    assert_eq!(ranked[0].label, "classical");
    assert_eq!(ranked[0].index, 1);
    for w in ranked.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "ranking violates descending invariant: {w:?}",
        );
    }
}

/// `apply_activation` produces deterministic outputs depending on the
/// selected activation. This test also confirms the helper is exported
/// at the crate root for downstream users who want to reuse it
/// standalone.
#[test]
fn apply_activation_modes_are_deterministic() {
    let logits = vec![1.0_f32, 2.0, 3.0];

    let softmax_out = apply_activation(&logits, TagActivation::Softmax);
    let sum: f32 = softmax_out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    let sigmoid_out = apply_activation(&logits, TagActivation::Sigmoid);
    for v in &sigmoid_out {
        assert!((0.0..=1.0).contains(v));
    }

    let none_out = apply_activation(&logits, TagActivation::None);
    assert_eq!(none_out, logits);
}

/// `MusicTags` has getter-only access; the struct's invariants should
/// remain stable even when default-constructed (empty result).  This
/// verifies the public contract of `best`, `len`, `is_empty`,
/// `activation`, and `into_inner`.
#[test]
fn music_tags_default_empty_contract_matches() {
    let tags = MusicTags::default();
    assert!(tags.is_empty());
    assert_eq!(tags.len(), 0);
    assert!(tags.best().is_none());
    // Default activation matches `TagActivation::default` = Softmax.
    assert_eq!(tags.activation(), TagActivation::Softmax);
    let owned = tags.into_inner();
    assert!(owned.is_empty());
}
