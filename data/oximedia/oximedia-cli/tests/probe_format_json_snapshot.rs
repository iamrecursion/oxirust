//! Path-stable snapshot tests for `oximedia probe --format json`.
//!
//! Each test writes a tiny magic-byte fixture (synthetic, just enough for the
//! container probe in `oximedia-container::probe::probe_format`) into a
//! per-test [`tempfile::TempDir`] using a deterministic file stem
//! (`fixture.<ext>`) so the `metadata.filename` field stays stable across
//! runs.
//!
//! The snapshot pipeline normalizes:
//! - `file` → `<stripped>` (absolute tempdir path)
//! - `file_name` (left as the deterministic fixture stem; intentionally not
//!   stripped so we can detect schema regressions)
//! - `file_size_bytes` → `0`
//! - `confidence` → rounded to 4 decimals (defends against `f32` formatting
//!   drift across libc / libm versions)
//! - `metadata.filename` (when present) → mirrors `file_name` (deterministic)
//!
//! Snapshots live in `tests/probe_snapshots/*.json`. To regenerate them
//! after an intentional schema change, set `OXIMEDIA_UPDATE_SNAPSHOTS=1`.

use assert_cmd::Command;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

// ── Fixture builders ─────────────────────────────────────────────────────────

/// Minimal MP4 (ISOBMFF `ftyp`) blob — 8-byte ftyp box, 24 zero bytes payload.
///
/// `probe_format` only checks `data[4..8] == b"ftyp"` (handler reads 8 KiB).
fn mp4_bytes() -> Vec<u8> {
    let mut v = Vec::with_capacity(32);
    v.extend_from_slice(&0x20u32.to_be_bytes()); // box size = 32
    v.extend_from_slice(b"ftyp");
    v.extend_from_slice(b"isom"); // major brand
    v.extend_from_slice(&0x00u32.to_be_bytes()); // minor version
    v.extend_from_slice(b"isomavc1"); // compatible brands
                                      // Pad out so the 8 KiB read buffer has something to truncate against.
    v.resize(64, 0);
    v
}

/// Minimal WebM/Matroska (EBML magic) blob.
fn webm_bytes() -> Vec<u8> {
    let mut v = vec![0x1A, 0x45, 0xDF, 0xA3];
    v.resize(64, 0);
    v
}

/// Minimal Ogg blob (`OggS` capture pattern).
fn ogg_bytes() -> Vec<u8> {
    let mut v = b"OggS".to_vec();
    v.resize(64, 0);
    v
}

/// Minimal FLAC blob (`fLaC` stream marker).
fn flac_bytes() -> Vec<u8> {
    let mut v = b"fLaC".to_vec();
    v.resize(64, 0);
    v
}

/// Write `bytes` into `dir/fixture.<ext>` and return the path.
fn write_fixture(dir: &Path, ext: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(format!("fixture.{ext}"));
    std::fs::write(&path, bytes).expect("failed to write fixture");
    path
}

// ── Probe runner ─────────────────────────────────────────────────────────────

fn run_probe_json(fixture: &Path, extra_flags: &[&str]) -> Value {
    let mut cmd = Command::cargo_bin("oximedia").expect("oximedia binary not found");
    // `--quiet` (global) suppresses the `tracing` `INFO` line that the probe
    // handler emits to stdout — without it the stdout starts with "INFO …\n{"
    // and `serde_json::from_str` rejects the leading non-JSON noise.
    cmd.args([
        "--quiet",
        "probe",
        "-i",
        fixture.to_str().expect("fixture path is utf-8"),
        "--format",
        "json",
    ]);
    for flag in extra_flags {
        cmd.arg(flag);
    }
    let output = cmd.assert().success().get_output().clone();
    let stdout = String::from_utf8(output.stdout).expect("probe stdout is utf-8");
    serde_json::from_str(&stdout).expect("probe stdout is valid JSON")
}

// ── Normalization ────────────────────────────────────────────────────────────

fn round_confidence(v: &Value) -> Value {
    // Round f32 → 4 decimals; emit as JSON number so snapshot diffs read clean.
    let f = v.as_f64().unwrap_or(0.0);
    let r = (f * 10_000.0).round() / 10_000.0;
    serde_json::Number::from_f64(r)
        .map(Value::Number)
        .unwrap_or_else(|| json!(0.0))
}

fn normalize(mut v: Value) -> Value {
    if let Some(obj) = v.as_object_mut() {
        if obj.contains_key("file") {
            obj.insert("file".into(), Value::String("<stripped>".into()));
        }
        if obj.contains_key("file_size_bytes") {
            obj.insert("file_size_bytes".into(), json!(0));
        }
        if let Some(c) = obj.get("confidence").cloned() {
            obj.insert("confidence".into(), round_confidence(&c));
        }
        // metadata.filename mirrors `file_name`; keep it as-is (deterministic
        // because we always use `fixture.<ext>` as the file name).
        if let Some(Value::Object(metadata)) = obj.get_mut("metadata") {
            normalize_metadata_paths(metadata);
        }
    }
    v
}

fn normalize_metadata_paths(metadata: &mut Map<String, Value>) {
    // Future-proof: if the handler ever adds path-shaped keys to metadata,
    // strip them here. Today only `filename` is emitted (deterministic via
    // our fixture naming).
    for key in ["path", "absolute_path", "full_path"] {
        if metadata.contains_key(key) {
            metadata.insert(key.into(), Value::String("<stripped>".into()));
        }
    }
}

// ── Snapshot persistence ─────────────────────────────────────────────────────

fn snapshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("probe_snapshots")
        .join(name)
}

fn check_snapshot(name: &str, actual: &Value) {
    let path = snapshot_path(name);
    let actual_pretty = serde_json::to_string_pretty(actual).expect("serialize actual snapshot");

    let update = std::env::var("OXIMEDIA_UPDATE_SNAPSHOTS").is_ok();
    if update || !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create snapshot dir");
        }
        let mut content = actual_pretty.clone();
        content.push('\n');
        std::fs::write(&path, content).expect("write snapshot");
        eprintln!("snapshot written: {}", path.display());
        return;
    }

    let expected_raw = std::fs::read_to_string(&path).expect("read snapshot");
    let expected: Value = serde_json::from_str(&expected_raw).expect("parse snapshot JSON");
    assert_eq!(
        actual,
        &expected,
        "probe JSON drift in {}; rerun with OXIMEDIA_UPDATE_SNAPSHOTS=1 to refresh",
        path.display()
    );
}

// ── Per-format basic snapshots ───────────────────────────────────────────────

#[test]
fn probe_mp4_basic_json_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "mp4", &mp4_bytes());
    let actual = normalize(run_probe_json(&path, &[]));
    check_snapshot("mp4_basic.json", &actual);
}

#[test]
fn probe_webm_basic_json_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "webm", &webm_bytes());
    let actual = normalize(run_probe_json(&path, &[]));
    check_snapshot("webm_basic.json", &actual);
}

#[test]
fn probe_ogg_basic_json_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "ogg", &ogg_bytes());
    let actual = normalize(run_probe_json(&path, &[]));
    check_snapshot("ogg_basic.json", &actual);
}

#[test]
fn probe_flac_basic_json_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "flac", &flac_bytes());
    let actual = normalize(run_probe_json(&path, &[]));
    check_snapshot("flac_basic.json", &actual);
}

// ── Per-flag schema snapshots (one fixture, exhaustive flag coverage) ────────
//
// The handler's `streams[]`, `chapters[]`, `metadata{}` payloads are currently
// hardcoded placeholders — the JSON shape is identical regardless of the input
// container. We snapshot the schema once per flag combo on a single fixture
// (mp4) so future codec-aware probe rewrites get a clean diff.

#[test]
fn probe_streams_flag_schema_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "mp4", &mp4_bytes());
    let actual = normalize(run_probe_json(&path, &["--streams"]));
    check_snapshot("mp4_streams.json", &actual);
}

#[test]
fn probe_chapters_flag_schema_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "mp4", &mp4_bytes());
    let actual = normalize(run_probe_json(&path, &["--chapters"]));
    check_snapshot("mp4_chapters.json", &actual);
}

#[test]
fn probe_metadata_flag_schema_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "mp4", &mp4_bytes());
    let actual = normalize(run_probe_json(&path, &["--metadata"]));
    check_snapshot("mp4_metadata.json", &actual);
}

#[test]
fn probe_all_flags_schema_matches_snapshot() {
    let dir = TempDir::new().expect("tempdir");
    let path = write_fixture(dir.path(), "mp4", &mp4_bytes());
    let actual = normalize(run_probe_json(
        &path,
        &["--streams", "--chapters", "--metadata"],
    ));
    check_snapshot("mp4_all_flags.json", &actual);
}
