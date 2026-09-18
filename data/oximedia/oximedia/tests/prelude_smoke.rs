//! Facade prelude smoke test — default-feature-only.
//!
//! Guards the default-feature surface of the `oximedia` facade crate: with
//! *zero* optional features enabled, `oximedia::prelude::*` must still bring
//! in a usable set of always-on core types (from `oximedia-core`,
//! `oximedia-container`, `oximedia-io`, and `oximedia-cv`) and compile/run
//! correctly. This complements `tests/integration.rs`'s `core_tests` module
//! (which imports items directly from the crate root) by exercising the
//! `prelude` re-export path specifically, since that is the documented
//! "import everything at once" entry point (see `src/prelude.rs` docs).
//!
//! Intentionally minimal: no opt-in feature is enabled, so this test must
//! keep compiling even if every optional subsystem is stripped out.

use oximedia::prelude::*;

/// `PixelFormat` is constructible via the prelude and its pure accessor
/// methods return the expected, well-known values.
#[test]
fn test_prelude_pixel_format_construction_and_accessors() {
    let fmt = PixelFormat::Yuv420p;
    assert_eq!(fmt.plane_count(), 3, "YUV420P must have 3 planes");
    assert!(fmt.is_planar(), "YUV420P must be planar");
    assert_eq!(
        fmt.bits_per_component(),
        8,
        "YUV420P must be 8 bits per component"
    );

    let fmt_10bit = PixelFormat::P010;
    assert_eq!(
        fmt_10bit.bits_per_component(),
        10,
        "P010 must be 10 bits per component"
    );
    assert_eq!(fmt_10bit.plane_count(), 2, "P010 must have 2 planes");
}

/// `Rational` is constructible via the prelude and preserves its fields.
#[test]
fn test_prelude_rational_construction() {
    let r = Rational::new(30_000, 1_001);
    assert_eq!(r.num, 30_000, "Numerator must be preserved");
    assert_eq!(r.den, 1_001, "Denominator must be preserved");
}

/// `Timestamp` is constructible via the prelude and its pure `to_seconds`
/// conversion is correct.
#[test]
fn test_prelude_timestamp_construction_and_to_seconds() {
    let ts = Timestamp::new(90_000, Rational::new(1, 90_000));
    assert!(
        (ts.to_seconds() - 1.0).abs() < f64::EPSILON,
        "90000 pts at 1/90000 timebase must be exactly 1.0 seconds, got {}",
        ts.to_seconds()
    );
}

/// `OxiError` is constructible via the prelude and displays its payload.
#[test]
fn test_prelude_oxi_error_display() {
    let err = OxiError::InvalidData("prelude smoke payload".to_string());
    let msg = err.to_string();
    assert!(
        msg.contains("prelude smoke payload"),
        "OxiError display should include the payload, got: {msg}"
    );
}

/// `OxiResult` propagation via `?` works when imported from the prelude.
#[test]
fn test_prelude_oxi_result_propagation() -> OxiResult<()> {
    fn inner() -> OxiResult<u32> {
        Ok(7)
    }
    assert_eq!(inner()?, 7, "OxiResult<u32> should carry the inner value");
    Ok(())
}

/// `probe_format` (re-exported from `oximedia-container` via the prelude)
/// must not panic on empty or garbage input.
#[test]
fn test_prelude_probe_format_does_not_panic() {
    let empty_result = probe_format(&[]);
    match empty_result {
        Ok(_) | Err(_) => {} // either outcome is acceptable; must not panic
    }

    let garbage: Vec<u8> = (0u8..=255).cycle().take(64).collect();
    let garbage_result = probe_format(&garbage);
    match garbage_result {
        Ok(_) | Err(_) => {}
    }
}

/// The always-available `cv` module alias is reachable through the prelude
/// without requiring any optional feature flag.
#[test]
fn test_prelude_cv_module_reachable() {
    let err = cv::CvError::InvalidDimensions {
        width: 0,
        height: 0,
    };
    let msg = err.to_string();
    assert!(
        msg.contains('0'),
        "CvError::InvalidDimensions display should mention the dimensions, got: {msg}"
    );
}
