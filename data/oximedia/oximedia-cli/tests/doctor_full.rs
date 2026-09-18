//! Integration tests for `oximedia doctor --full`.
//!
//! These verify the three new diagnostic sections (codec matrix, plugin path
//! validation, OxiCUDA probe) added by Run 4 / Slice B, while leaving the
//! existing five default-output sections strictly untouched. The default-output
//! schema is exercised separately by `tests/doctor_smoke.rs` so a regression
//! there would surface independently.

use assert_cmd::Command;
use std::path::PathBuf;
use tempfile::TempDir;

/// Build a `Command` for `oximedia` with the OxiCUDA env stripped — CI
/// environments occasionally leave `OXICUDA_HOME` set, which would change the
/// JSON shape and produce flaky results. Tests that need a controlled value
/// can re-add it with `.env(...)`.
fn doctor_cmd() -> Command {
    let mut cmd = Command::cargo_bin("oximedia").expect("oximedia binary should exist");
    cmd.env_remove("OXICUDA_HOME");
    cmd.env_remove("OXIMEDIA_PLUGIN_PATH");
    cmd
}

#[test]
fn doctor_full_exits_zero_with_six_sections() {
    let output = doctor_cmd()
        .args(["doctor", "--full"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("plain output should be UTF-8");
    // Default 4 sections (header + rust + temp + gpu) + 3 new ones.
    assert!(
        s.contains("OxiMedia Doctor Report"),
        "missing report header: {s}"
    );
    assert!(s.contains("Rust version:"), "missing rust version line");
    assert!(s.contains("Temp dir:"), "missing temp dir line");
    assert!(s.contains("GPU adapters"), "missing GPU adapters line");
    assert!(s.contains("Codec matrix:"), "missing Codec matrix section");
    assert!(s.contains("Plugin paths"), "missing Plugin paths section");
    assert!(s.contains("OxiCUDA"), "missing OxiCUDA section");
    // OxiCUDA in the no-env case should explicitly tell the user it's optional.
    assert!(
        s.contains("not configured"),
        "OxiCUDA absence should surface 'not configured': {s}"
    );
}

#[test]
fn doctor_full_json_has_three_new_keys() {
    let output = doctor_cmd()
        .args(["doctor", "--full", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("doctor --full --json output should be UTF-8");
    let v: serde_json::Value =
        serde_json::from_str(&s).expect("doctor --full --json should produce valid JSON");

    // Existing keys still present.
    assert!(v.get("rust_version").is_some(), "JSON missing rust_version");
    assert!(v.get("temp_dir").is_some(), "JSON missing temp_dir");
    assert!(v.get("gpu_adapters").is_some(), "JSON missing gpu_adapters");

    // New keys.
    let cm = v
        .get("codec_matrix")
        .expect("JSON missing codec_matrix")
        .as_array()
        .expect("codec_matrix should be an array");
    assert!(!cm.is_empty(), "codec_matrix should be non-empty");
    let names: Vec<&str> = cm
        .iter()
        .filter_map(|row| row.get("codec").and_then(|v| v.as_str()))
        .collect();
    for required in ["av1", "vp9", "vp8", "opus", "ffv1"] {
        assert!(
            names.contains(&required),
            "codec_matrix missing `{required}`: {names:?}"
        );
    }

    // plugin_paths key must exist (even when empty array).
    assert!(v.get("plugin_paths").is_some(), "JSON missing plugin_paths");

    let cuda = v.get("oxicuda").expect("JSON missing oxicuda");
    let configured = cuda
        .get("configured")
        .and_then(|v| v.as_bool())
        .expect("oxicuda.configured should be a bool");
    assert!(!configured, "oxicuda should not be configured here");
}

#[test]
fn doctor_full_reports_nonexistent_plugin_path() {
    let bogus = "/this/path/does/not/exist/oximedia_doctor_test";
    let output = doctor_cmd()
        .args(["doctor", "--full", "--json"])
        .env("OXIMEDIA_PLUGIN_PATH", bogus)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("UTF-8");
    let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
    let entries = v
        .get("plugin_paths")
        .and_then(|x| x.as_array())
        .expect("plugin_paths array");
    assert_eq!(entries.len(), 1, "expected single plugin path entry");
    let entry = &entries[0];
    assert_eq!(
        entry.get("exists").and_then(|x| x.as_bool()),
        Some(false),
        "exists should be false for missing path"
    );
    assert_eq!(entry.get("readable").and_then(|x| x.as_bool()), Some(false));
    assert_eq!(entry.get("dylibs_found").and_then(|x| x.as_u64()), Some(0));
    let path_str = entry
        .get("path")
        .and_then(|x| x.as_str())
        .expect("path string");
    assert_eq!(PathBuf::from(path_str), PathBuf::from(bogus));
}

#[test]
fn doctor_full_tempdir_plugin_path_zero_dylibs() {
    let dir = TempDir::new().expect("create tempdir");
    let path_str = dir.path().to_string_lossy().to_string();

    let output = doctor_cmd()
        .args(["doctor", "--full", "--json"])
        .env("OXIMEDIA_PLUGIN_PATH", &path_str)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("UTF-8");
    let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
    let entries = v
        .get("plugin_paths")
        .and_then(|x| x.as_array())
        .expect("plugin_paths array");
    assert_eq!(entries.len(), 1, "expected single tempdir entry");
    let entry = &entries[0];
    assert_eq!(
        entry.get("exists").and_then(|x| x.as_bool()),
        Some(true),
        "tempdir should exist"
    );
    assert_eq!(
        entry.get("readable").and_then(|x| x.as_bool()),
        Some(true),
        "tempdir should be readable"
    );
    assert_eq!(
        entry.get("dylibs_found").and_then(|x| x.as_u64()),
        Some(0),
        "empty tempdir should have zero dylibs"
    );
}

#[test]
fn doctor_default_json_schema_is_unchanged() {
    // Belt-and-braces guard: the three new keys must NOT appear in the default
    // (non-`--full`) JSON output. Existing downstream consumers depend on this.
    let output = doctor_cmd()
        .args(["doctor", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("UTF-8");
    let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
    assert!(
        v.get("codec_matrix").is_none(),
        "default JSON must omit codec_matrix"
    );
    assert!(
        v.get("plugin_paths").is_none(),
        "default JSON must omit plugin_paths"
    );
    assert!(v.get("oxicuda").is_none(), "default JSON must omit oxicuda");
}

#[test]
fn doctor_full_oxicuda_recognises_version_file() {
    let dir = TempDir::new().expect("create tempdir");
    let version_file = dir.path().join("version.txt");
    std::fs::write(&version_file, "CUDA Version 12.4.1\n").expect("write version.txt");

    let output = doctor_cmd()
        .args(["doctor", "--full", "--json"])
        .env("OXICUDA_HOME", dir.path())
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("UTF-8");
    let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
    let cuda = v.get("oxicuda").expect("oxicuda key");
    assert_eq!(cuda.get("configured").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(
        cuda.get("version").and_then(|x| x.as_str()),
        Some("CUDA Version 12.4.1")
    );
}
