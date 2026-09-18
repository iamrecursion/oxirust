//! Integration tests for the `onnx`-gated ML caption-encoding pipeline.
//!
//! These tests deliberately avoid requiring a real ONNX model on disk —
//! they validate the public contract of
//! [`oximedia_caption_gen::CaptionEncoder`],
//! [`oximedia_caption_gen::EncoderOutput`],
//! [`oximedia_caption_gen::greedy_decode`], and
//! [`oximedia_caption_gen::top_k_sample`] (types are `pub`, builder methods
//! chain, error conversion works, decoding contracts hold) without
//! depending on any external fixture.

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use oximedia_caption_gen::{
    greedy_decode, top_k_sample, CaptionEncoder, CaptionGenError, CaptionGenResult, EncoderOutput,
};
use oximedia_ml::{DeviceType, MlError};

/// Materialise the re-export paths at the crate root so a regression that
/// hides any of the ML types surfaces immediately at the public boundary.
#[test]
fn ml_types_reexported_at_crate_root() {
    fn assert_type<T>() {}
    assert_type::<CaptionEncoder>();
    assert_type::<EncoderOutput>();
}

/// `CaptionEncoder::from_path` against a nonexistent file must return the
/// crate-native [`CaptionGenError`] with the ML error folded in via the
/// `#[from]` conversion.
#[test]
fn from_path_nonexistent_model_returns_ml_error() {
    let temp = std::env::temp_dir().join("oximedia-caption-gen-nonexistent-encoder.onnx");
    if temp.exists() {
        let _ = std::fs::remove_file(&temp);
    }
    let err: CaptionGenError = CaptionEncoder::from_path(&temp, DeviceType::Cpu)
        .expect_err("loading a missing model must fail");
    assert!(
        matches!(err, CaptionGenError::Ml(_)),
        "expected CaptionGenError::Ml, got {err:?}",
    );
}

/// The `From<MlError> for CaptionGenError` conversion is wired up via
/// thiserror's `#[from]`.  Exercising it here guarantees that downstream
/// code can use `?` on ML results inside `CaptionGenResult`-returning
/// functions.
#[test]
fn caption_gen_error_from_ml_error_roundtrip() {
    fn forward() -> CaptionGenResult<()> {
        Err(MlError::FeatureDisabled("onnx"))?;
        Ok(())
    }
    let err = forward().expect_err("must propagate");
    assert!(
        matches!(err, CaptionGenError::Ml(MlError::FeatureDisabled("onnx"))),
        "expected MlError::FeatureDisabled to round-trip, got {err:?}",
    );
}

/// Builder-style `with_*` methods all return `Self` so callers can chain
/// them fluently after `from_path` / `from_shared_model`.  This test only
/// asserts the signatures compile — it never actually constructs an
/// encoder because we have no real ONNX file in-repo.
#[test]
fn builder_methods_are_chainable() {
    fn chain(encoder: CaptionEncoder) -> CaptionEncoder {
        encoder
            .with_input_name("audio")
            .with_output_name("logits")
            .with_input_name("mel")
            .with_output_name("tokens")
    }
    // Asserting compilation is the contract.  Runtime construction of an
    // encoder needs a real model, which is out of scope for in-tree tests;
    // the `from_path_nonexistent_model_returns_ml_error` test covers the
    // fallible path.
    let _ = chain; // suppress dead-code in release profiles
}

/// Using a nonexistent path under `std::env::temp_dir()` proves the error
/// path is clean (not a panic / abort) in a realistic filesystem location.
#[test]
fn missing_model_under_temp_dir_returns_clean_error() {
    let path: PathBuf = std::env::temp_dir()
        .join("oximedia-caption-gen-ml-test")
        .join("never-created.onnx");
    let result = CaptionEncoder::from_path(&path, DeviceType::Cpu);
    assert!(
        matches!(result, Err(CaptionGenError::Ml(_))),
        "expected CaptionGenError::Ml from missing model, got {result:?}",
    );
}

/// `greedy_decode` is the pure post-processing helper that downstream
/// caption decoders route through.  Exercising it at the integration
/// boundary guarantees the re-export is stable and the argmax-per-row
/// contract holds across the crate API surface.
#[test]
fn greedy_decode_via_crate_surface_picks_argmax() {
    // 3 steps, 4-token vocab.
    let logits = vec![
        0.1, 0.2, 0.9, 0.1, // step 0 → 2
        0.3, 0.7, 0.2, 0.1, // step 1 → 1
        0.4, 0.4, 0.1, 0.4, // step 2 → 0 (tie → lowest index)
    ];
    let out = greedy_decode(&logits, 4, 3).expect("ok");
    assert_eq!(out, vec![2_u32, 1_u32, 0_u32]);
}

/// `top_k_sample` produces deterministic output for identical seeds and
/// only emits tokens from the top-`k` set.  This test also confirms the
/// helper is exported at the crate root for downstream users.
#[test]
fn top_k_sample_is_deterministic_and_bounded() {
    // Vocab=5, seq_len=2, k=2.
    // Step 0: top-2 = [1, 3] (values 0.9, 0.7).
    // Step 1: top-2 = [0, 2] (values 0.6, 0.5).
    let logits = vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.2, 0.1];
    let a = top_k_sample(&logits, 5, 2, 2, 7).expect("ok");
    let b = top_k_sample(&logits, 5, 2, 2, 7).expect("ok");
    assert_eq!(a, b, "same seed must give same output");
    assert!(a[0] == 1 || a[0] == 3, "step 0 outside top-2: {a:?}");
    assert!(a[1] == 0 || a[1] == 2, "step 1 outside top-2: {a:?}");
}

/// `EncoderOutput` has straightforward inherent methods; this test pins
/// their semantics at the integration boundary so refactors do not change
/// behaviour silently.
#[test]
fn encoder_output_public_contract_matches() {
    let e = EncoderOutput {
        logits: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![1, 2, 2],
    };
    assert_eq!(e.len(), 4);
    assert!(!e.is_empty());
    // `shape` is `pub` — callers must be able to read it directly.
    assert_eq!(e.shape, vec![1_usize, 2, 2]);
    // `logits` is `pub` — same.
    assert_eq!(e.logits.len(), 4);
}
