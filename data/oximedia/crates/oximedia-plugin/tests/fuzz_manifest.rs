//! Fuzz-style tests for `PluginManifest` parsing with malformed JSON/TOML.
//!
//! These tests exercise the robustness of the manifest parser against a broad
//! range of malformed, truncated, and adversarially-crafted inputs.  They do
//! not use a formal fuzzing harness (which would require `cargo fuzz`) but
//! rather a curated corpus of known-bad inputs combined with property-based
//! checks.

use oximedia_plugin::manifest::PluginManifest;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Verify that parsing `json` does not panic and returns an error.
fn must_fail(json: &str) {
    let result = PluginManifest::from_json(json);
    assert!(
        result.is_err(),
        "Expected parse failure for input: {json:?}"
    );
}

/// Verify that parsing `json` does not panic (result can be Ok or Err).
fn must_not_panic(json: &str) {
    let _ = PluginManifest::from_json(json);
}

// ── Clearly invalid inputs ─────────────────────────────────────────────────

#[test]
fn test_empty_string() {
    must_fail("");
}

#[test]
fn test_whitespace_only() {
    must_fail("   \t\n  ");
}

#[test]
fn test_not_json() {
    must_fail("not json at all");
}

#[test]
fn test_yaml_like() {
    must_fail("name: my-plugin\nversion: 1.0.0");
}

#[test]
fn test_toml_like() {
    must_fail("[package]\nname = \"my-plugin\"");
}

#[test]
fn test_single_number() {
    must_fail("42");
}

#[test]
fn test_json_array_not_object() {
    must_fail("[\"not\", \"an\", \"object\"]");
}

#[test]
fn test_json_null() {
    must_fail("null");
}

#[test]
fn test_json_boolean() {
    must_fail("true");
}

// ── Truncated / partial JSON ─────────────────────────────────────────────────

#[test]
fn test_truncated_at_open_brace() {
    must_fail("{");
}

#[test]
fn test_truncated_at_key() {
    must_fail(r#"{"name":"#);
}

#[test]
fn test_truncated_mid_value() {
    must_fail(r#"{"name":"my-plu"#);
}

#[test]
fn test_truncated_mid_object() {
    must_fail(r#"{"name":"p","version":"1.0"#);
}

// ── Wrong types for required fields ──────────────────────────────────────────

#[test]
fn test_name_is_number() {
    must_fail(
        r#"{"name":42,"version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_version_is_array() {
    must_fail(
        r#"{"name":"p","version":[1,0,0],"api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_api_version_is_string() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":"one","description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_patent_encumbered_is_string() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":"yes","library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_codecs_is_not_array() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":"h264"}"#,
    );
}

// ── Missing required fields ──────────────────────────────────────────────────

#[test]
fn test_missing_name() {
    must_fail(
        r#"{"version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_missing_api_version() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_missing_library() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"codecs":[]}"#,
    );
}

#[test]
fn test_missing_codecs() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so"}"#,
    );
}

// ── Adversarial content ──────────────────────────────────────────────────────

#[test]
fn test_deeply_nested_json() {
    // Deeply nested objects should not cause a stack overflow.
    let mut json = String::new();
    for _ in 0..1000 {
        json.push('{');
    }
    for _ in 0..1000 {
        json.push('}');
    }
    must_not_panic(&json);
}

#[test]
fn test_very_long_string_value() {
    let long_name = "x".repeat(100_000);
    let json = format!(
        r#"{{"name":"{long_name}","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}}"#
    );
    // Should not panic — may or may not parse correctly.
    must_not_panic(&json);
}

#[test]
fn test_unicode_null_bytes() {
    must_not_panic("{\"name\":\"p\x00q\",\"version\":\"1.0.0\"}");
}

#[test]
fn test_control_characters_in_value() {
    let json = "{\"name\":\"p\\u0001\\u0002\",\"version\":\"1.0.0\",\"api_version\":1,\"description\":\"D\",\"author\":\"A\",\"license\":\"MIT\",\"patent_encumbered\":false,\"library\":\"l.so\",\"codecs\":[]}";
    must_not_panic(json);
}

#[test]
fn test_duplicate_keys_last_wins() {
    // JSON spec says behaviour is undefined for duplicate keys; serde_json
    // typically takes the last value.  We just assert no panic.
    let json = r#"{"name":"first","name":"second","version":"1.0.0","api_version":1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#;
    must_not_panic(json);
}

#[test]
fn test_negative_api_version_rejected() {
    // Negative numbers cannot deserialise into u32.
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":-1,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

#[test]
fn test_floating_point_api_version_rejected() {
    must_fail(
        r#"{"name":"p","version":"1.0.0","api_version":1.5,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[]}"#,
    );
}

// ── Valid-parse but invalid-validate inputs ──────────────────────────────────
//
// These inputs parse without error but should fail `validate()`.

#[test]
fn test_parses_ok_but_empty_name_fails_validate() {
    use oximedia_plugin::traits::PLUGIN_API_VERSION;
    let json = format!(
        r#"{{"name":"","version":"1.0.0","api_version":{PLUGIN_API_VERSION},"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[{{"name":"c","decode":true,"encode":false,"description":"C"}}]}}"#
    );
    let m = PluginManifest::from_json(&json).expect("should parse");
    assert!(m.validate().is_err());
}

#[test]
fn test_parses_ok_but_wrong_api_version_fails_validate() {
    let json = r#"{"name":"p","version":"1.0.0","api_version":999,"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[{"name":"c","decode":true,"encode":false,"description":"C"}]}"#;
    let m = PluginManifest::from_json(json).expect("should parse");
    assert!(m.validate().is_err());
}

// ── min_host_version field ───────────────────────────────────────────────────

#[test]
fn test_min_oximedia_version_parses_from_json() {
    use oximedia_plugin::traits::PLUGIN_API_VERSION;
    let json = format!(
        r#"{{"name":"p","version":"1.0.0","api_version":{PLUGIN_API_VERSION},"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[{{"name":"c","decode":true,"encode":false,"description":"C"}}],"min_host_version":">=0.1.2"}}"#
    );
    let m = PluginManifest::from_json(&json).expect("should parse");
    assert_eq!(m.min_host_version.as_deref(), Some(">=0.1.2"));
}

#[test]
fn test_min_oximedia_version_absent_defaults_to_none() {
    use oximedia_plugin::traits::PLUGIN_API_VERSION;
    let json = format!(
        r#"{{"name":"p","version":"1.0.0","api_version":{PLUGIN_API_VERSION},"description":"D","author":"A","license":"MIT","patent_encumbered":false,"library":"l.so","codecs":[{{"name":"c","decode":true,"encode":false,"description":"C"}}]}}"#
    );
    let m = PluginManifest::from_json(&json).expect("should parse");
    assert!(m.min_host_version.is_none());
}
