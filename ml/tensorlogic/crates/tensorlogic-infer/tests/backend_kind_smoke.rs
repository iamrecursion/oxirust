//! Smoke tests for [`tensorlogic_infer::BackendKind`] and [`tensorlogic_infer::BackendKindError`].

use std::str::FromStr;

use tensorlogic_infer::{BackendKind, BackendKindError};

// ─── default_backend ─────────────────────────────────────────────────────────

/// Without the env-var set the default must be `Scirs`.
#[test]
fn default_is_scirs_without_env() {
    // Guarantee the env-var is absent for this test.
    std::env::remove_var("TENSORLOGIC_BACKEND");
    assert_eq!(BackendKind::default_backend(), BackendKind::Scirs);
}

// ─── from_env ────────────────────────────────────────────────────────────────

/// Setting `TENSORLOGIC_BACKEND=oxicuda` must yield `OxiCuda`.
///
/// We restore the variable afterwards so that test order does not matter.
#[test]
fn from_env_oxicuda() {
    let key = "TENSORLOGIC_BACKEND";
    // Save whatever was there before (if anything).
    let previous = std::env::var(key).ok();

    std::env::set_var(key, "oxicuda");
    let result = BackendKind::from_env();

    // Restore env state before any assertion so a panic cannot leave it dirty.
    match &previous {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }

    assert_eq!(result, BackendKind::OxiCuda);
}

// ─── validate ────────────────────────────────────────────────────────────────

#[test]
fn validate_scirs_ok() {
    assert!(BackendKind::Scirs.validate().is_ok());
}

#[test]
fn validate_metal_err() {
    assert!(BackendKind::Metal.validate().is_err());
}

// ─── supports_autodiff ───────────────────────────────────────────────────────

#[test]
fn supports_autodiff_scirs() {
    assert!(BackendKind::Scirs.supports_autodiff());
}

#[test]
fn supports_autodiff_levelzero() {
    assert!(!BackendKind::Levelzero.supports_autodiff());
}

// ─── from_str roundtrip ──────────────────────────────────────────────────────

#[test]
fn from_str_roundtrip() {
    assert_eq!(BackendKind::from_str("scirs").unwrap(), BackendKind::Scirs);
}

// ─── from_str aliases ────────────────────────────────────────────────────────

#[test]
fn from_str_cuda_alias() {
    assert_eq!(BackendKind::from_str("cuda").unwrap(), BackendKind::OxiCuda);
}

#[test]
fn from_str_unknown_is_err() {
    let err = BackendKind::from_str("nonexistent_backend_xyz");
    assert!(err.is_err());
    let err = err.unwrap_err();
    // Must be the UnknownName variant.
    assert!(
        matches!(err, BackendKindError::UnknownName(_)),
        "expected UnknownName, got: {err:?}"
    );
}

// ─── is_gpu ──────────────────────────────────────────────────────────────────

#[test]
fn is_gpu_flags() {
    assert!(!BackendKind::Scirs.is_gpu());
    assert!(BackendKind::OxiCuda.is_gpu());
    assert!(BackendKind::Vulkan.is_gpu());
    assert!(BackendKind::Metal.is_gpu());
}

// ─── available_backends includes Scirs ───────────────────────────────────────

#[test]
fn available_backends_contains_scirs() {
    let backends = BackendKind::available_backends();
    assert!(
        backends.contains(&BackendKind::Scirs),
        "Scirs must always be listed"
    );
}

// ─── as_str stability ────────────────────────────────────────────────────────

#[test]
fn as_str_values() {
    assert_eq!(BackendKind::Scirs.as_str(), "scirs");
    assert_eq!(BackendKind::OxiCuda.as_str(), "oxicuda");
    assert_eq!(BackendKind::Metal.as_str(), "metal");
    assert_eq!(BackendKind::Vulkan.as_str(), "vulkan");
    assert_eq!(BackendKind::Rocm.as_str(), "rocm");
    assert_eq!(BackendKind::Webgpu.as_str(), "webgpu");
    assert_eq!(BackendKind::Levelzero.as_str(), "levelzero");
}

// ─── validate error message ──────────────────────────────────────────────────

#[test]
fn validate_unimplemented_error_message() {
    let err = BackendKind::Vulkan.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("vulkan"),
        "error message should mention the backend name; got: {msg}"
    );
    assert!(
        msg.contains("Round 6"),
        "error message should mention Round 6; got: {msg}"
    );
}
