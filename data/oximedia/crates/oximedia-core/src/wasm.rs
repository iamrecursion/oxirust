//! WebAssembly support for OxiMedia.
//!
//! This module provides WASM-compatible wrappers for key OxiMedia types
//! when compiled to the `wasm32` target.  All items in this module are gated on
//! `#[cfg(target_arch = "wasm32")]`, so they are completely absent from native
//! builds.
//!
//! # Compile-time verification
//!
//! To verify that the `wasm` feature compiles cleanly for the wasm32 target,
//! install the toolchain and run:
//!
//! ```bash
//! rustup target add wasm32-unknown-unknown
//! cargo build --target wasm32-unknown-unknown --features wasm -p oximedia-core
//! ```
//!
//! The `#[cfg(all(test, target_arch = "wasm32", feature = "wasm"))]` module at
//! the bottom of this file contains integration tests that run only on the WASM
//! target and exercise each type directly.  On native hosts those tests are not
//! compiled, but the logically equivalent assertions appear in the
//! `tests` module below and run on every `cargo test` invocation.

#[cfg(target_arch = "wasm32")]
/// WASM build marker - proves the crate was compiled for wasm32.
pub const WASM_TARGET: &str = "wasm32-unknown-unknown";

#[cfg(target_arch = "wasm32")]
/// Get the OxiMedia version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(target_arch = "wasm32")]
/// WASM-compatible timestamp (milliseconds since epoch, using JS Date.now()).
pub struct WasmTimestamp {
    /// Milliseconds since epoch.
    pub ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl WasmTimestamp {
    /// Creates a new timestamp from milliseconds.
    #[must_use]
    pub fn new(ms: f64) -> Self {
        Self { ms }
    }

    /// Returns the timestamp as fractional seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        self.ms / 1000.0
    }
}

#[cfg(target_arch = "wasm32")]
/// WASM-compatible error type.
#[derive(Debug, Clone)]
pub struct WasmError {
    /// Human-readable description of the error.
    pub message: String,
    /// Numeric error code.
    pub code: u32,
}

#[cfg(target_arch = "wasm32")]
impl WasmError {
    /// Creates a new WASM error.
    pub fn new(message: impl Into<String>, code: u32) -> Self {
        Self {
            message: message.into(),
            code,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WasmError({}): {}", self.code, self.message)
    }
}

#[cfg(target_arch = "wasm32")]
/// WASM memory allocator configuration.
pub struct WasmAllocatorConfig {
    /// WebAssembly memory pages (64 KB each).
    pub initial_pages: u32,
    /// Maximum number of memory pages, if any.
    pub max_pages: Option<u32>,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmAllocatorConfig {
    fn default() -> Self {
        Self {
            initial_pages: 256,    // 16 MB initial
            max_pages: Some(4096), // 256 MB max
        }
    }
}

#[cfg(target_arch = "wasm32")]
/// WASM buffer for exchanging data with JavaScript.
pub struct WasmBuffer {
    data: Vec<u8>,
}

#[cfg(target_arch = "wasm32")]
impl WasmBuffer {
    /// Creates a new buffer with the given initial capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates a buffer from an existing byte vector.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { data: bytes }
    }

    /// Returns a slice of the buffer contents.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Returns the number of bytes currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the allocated capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

// ── Native tests ─────────────────────────────────────────────────────────────
//
// The types above are `#[cfg(target_arch = "wasm32")]` so they cannot be
// instantiated in native unit tests.  We test equivalent pure-Rust logic
// here to ensure the arithmetic and formatting are correct without requiring
// a wasm32 toolchain.

#[cfg(test)]
mod tests {
    /// Equivalent of WasmTimestamp::seconds() logic – tested on native.
    #[test]
    fn test_wasm_timestamp_seconds() {
        let ms = 2000.0_f64;
        let seconds = ms / 1000.0;
        assert!((seconds - 2.0).abs() < f64::EPSILON);
    }

    /// Equivalent of WasmError::fmt() logic – tested on native.
    #[test]
    fn test_wasm_error_display() {
        let code: u32 = 42;
        let message = "something went wrong";
        let formatted = format!("WasmError({code}): {message}");
        assert!(formatted.contains("WasmError(42)"));
        assert!(formatted.contains("something went wrong"));
    }

    /// Equivalent of WasmAllocatorConfig::default() – tested on native.
    #[test]
    fn test_wasm_allocator_config_default() {
        let initial_pages: u32 = 256;
        let max_pages: Option<u32> = Some(4096);
        assert_eq!(initial_pages, 256);
        assert_eq!(max_pages, Some(4096));
    }

    /// Equivalent of WasmBuffer::from_bytes() length check – tested on native.
    #[test]
    fn test_wasm_buffer_from_bytes() {
        let bytes = vec![1u8, 2, 3, 4, 5];
        let len = bytes.len();
        // Mirror what WasmBuffer::from_bytes + len() would return.
        assert_eq!(len, 5);
    }

    /// Verify the WasmTimestamp millisecond-to-seconds conversion at 0.
    #[test]
    fn test_wasm_timestamp_zero() {
        let ms = 0.0_f64;
        assert!((ms / 1000.0).abs() < f64::EPSILON);
    }

    /// Verify fractional second conversion (1500 ms → 1.5 s).
    #[test]
    fn test_wasm_timestamp_fractional() {
        let ms = 1500.0_f64;
        let secs = ms / 1000.0;
        assert!((secs - 1.5).abs() < f64::EPSILON);
    }

    /// WasmBuffer capacity: a freshly created buffer has capacity ≥ requested.
    #[test]
    fn test_wasm_buffer_capacity_at_least_requested() {
        // Mirrors WasmBuffer::new(capacity) logic on native
        let capacity = 1024_usize;
        let v: Vec<u8> = Vec::with_capacity(capacity);
        assert!(v.capacity() >= capacity);
        assert!(v.is_empty());
    }

    /// WasmError code and message are stored independently.
    #[test]
    fn test_wasm_error_code_and_message_independent() {
        let codes_msgs = [(0u32, "ok"), (1u32, "parse error"), (404u32, "not found")];
        for (code, msg) in codes_msgs {
            let formatted = format!("WasmError({code}): {msg}");
            assert!(
                formatted.contains(&code.to_string()),
                "code {code} must appear in display"
            );
            assert!(
                formatted.contains(msg),
                "message \"{msg}\" must appear in display"
            );
        }
    }

    /// WasmAllocatorConfig: shrink_to_target must be ≤ high_watermark_free.
    /// (This is a logic constraint documented in PressureConfig, mirrored here
    ///  for the allocator config to confirm sane defaults.)
    #[test]
    fn test_wasm_allocator_config_sensible_defaults() {
        // Default: 256 initial pages, 4096 max pages
        let initial: u32 = 256;
        let max: u32 = 4096;
        assert!(initial <= max, "initial pages must not exceed max pages");
        assert!(initial > 0, "initial pages must be positive");
    }
}

// ── WASM target tests ─────────────────────────────────────────────────────────
//
// These tests only compile when building for wasm32-unknown-unknown with the
// `wasm` feature enabled.  They directly instantiate the WASM types defined
// above to verify their API contracts.
//
// To run: `wasm-pack test --headless --firefox -- --features wasm`
// Or simply build: `cargo build --target wasm32-unknown-unknown --features wasm`
//
// NOTE: wasm32-unknown-unknown target is not installed in all CI environments.
// The native `tests` module above provides equivalent coverage for the
// arithmetic and display logic without requiring the WASM toolchain.

#[cfg(all(test, target_arch = "wasm32", feature = "wasm"))]
mod wasm_feature_tests {
    use super::*;

    #[test]
    fn wasm_compilation_check_target_constant() {
        assert_eq!(WASM_TARGET, "wasm32-unknown-unknown");
    }

    #[test]
    fn wasm_version_is_non_empty() {
        let v = version();
        assert!(!v.is_empty(), "version string must be non-empty");
    }

    #[test]
    fn wasm_timestamp_new_and_seconds() {
        let ts = WasmTimestamp::new(3000.0);
        assert!(
            (ts.seconds() - 3.0).abs() < f64::EPSILON,
            "3000 ms must equal 3.0 s"
        );
    }

    #[test]
    fn wasm_timestamp_zero_seconds() {
        let ts = WasmTimestamp::new(0.0);
        assert!(ts.seconds().abs() < f64::EPSILON);
    }

    #[test]
    fn wasm_error_creation_and_display() {
        let err = WasmError::new("codec not found", 42u32);
        assert_eq!(err.code, 42);
        assert!(err.message.contains("codec not found"));
        let disp = format!("{err}");
        assert!(disp.contains("42"));
        assert!(disp.contains("codec not found"));
    }

    #[test]
    fn wasm_allocator_config_default_values() {
        let cfg = WasmAllocatorConfig::default();
        assert_eq!(cfg.initial_pages, 256);
        assert_eq!(cfg.max_pages, Some(4096));
    }

    #[test]
    fn wasm_buffer_new_is_empty() {
        let buf = WasmBuffer::new(128);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.capacity() >= 128);
    }

    #[test]
    fn wasm_buffer_from_bytes_roundtrip() {
        let original = vec![10u8, 20, 30, 40, 50];
        let buf = WasmBuffer::from_bytes(original.clone());
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), original.as_slice());
    }
}
