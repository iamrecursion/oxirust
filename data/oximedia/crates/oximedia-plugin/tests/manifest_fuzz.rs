//! Additional manifest fuzz tests: injection, large input, null bytes.
//!
//! These complement the main fuzz corpus in `fuzz_manifest.rs` and focus on
//! adversarial inputs (SQL-injection-style strings, large allocations, null
//! bytes) that should never panic — only return `Result::Err` or `Ok`.

use oximedia_plugin::manifest::PluginManifest;

// ── Structural parse failures ─────────────────────────────────────────────────

#[test]
fn test_manifest_fuzz_empty_json() {
    assert!(PluginManifest::from_json("").is_err());
}

#[test]
fn test_manifest_fuzz_partial_json() {
    // Missing closing brace — truncated stream.
    assert!(PluginManifest::from_json("{\"name\":\"x\"").is_err());
}

// ── Adversarial content — must not panic ─────────────────────────────────────

/// SQL-injection-style value: the parser must accept or reject without panic.
#[test]
fn test_manifest_fuzz_injection_json() {
    // This may parse as a JSON object (valid JSON string containing SQL) but
    // will fail manifest validation (missing required fields).  What matters
    // is that there is no panic.
    let _ =
        PluginManifest::from_json("{\"name\":\"'; DROP TABLE plugins; --\",\"version\":\"1.0.0\"}");
}

/// Embedded null bytes in the JSON payload.  serde_json may reject or accept
/// them; either outcome is acceptable as long as the call does not panic.
#[test]
fn test_manifest_fuzz_nullbytes() {
    let _ = PluginManifest::from_json("\0\0\0");
}

/// One-million-character name field.  Verifies that the parser does not
/// allocate unbounded memory in a way that causes an OOM abort or panic.
#[test]
fn test_manifest_fuzz_large_input() {
    let big = "x".repeat(1_000_000);
    let _ = PluginManifest::from_json(&format!("{{\"name\":\"{big}\"}}"));
}
