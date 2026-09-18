//! Integration tests for rayon-parallel batch processing.
//!
//! Uses `--dry-run` to avoid real transcoding while still exercising the job
//! discovery and parallelization code paths.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Create a batch TOML config that matches *.mkv files.
fn write_batch_config(dir: &std::path::Path) -> std::path::PathBuf {
    let config_path = dir.join("batch_test.toml");
    let config_content = r#"
# Batch config for integration test
patterns = ["*.mkv"]
video_codec = "vp9"
audio_codec = "opus"
output_extension = "webm"
preset = "fast"
overwrite = true
recursive = false
"#;
    fs::write(&config_path, config_content).expect("failed to write batch config");
    config_path
}

#[test]
fn batch_dry_run_j0_no_panic() {
    let tmp = TempDir::new().expect("tempdir creation should succeed");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");

    fs::create_dir_all(&input_dir).expect("mkdir input");
    fs::create_dir_all(&output_dir).expect("mkdir output");

    // Create a couple of dummy .mkv files to discover.
    fs::write(input_dir.join("clip1.mkv"), b"not really an mkv").expect("write dummy clip1");
    fs::write(input_dir.join("clip2.mkv"), b"not really an mkv").expect("write dummy clip2");

    let config = write_batch_config(tmp.path());

    // `--dry-run` skips actual transcoding; `-j 0` means auto num_cpus.
    Command::cargo_bin("oximedia")
        .expect("oximedia binary not found")
        .args([
            "batch",
            input_dir.to_str().expect("input_dir utf8"),
            output_dir.to_str().expect("output_dir utf8"),
            config.to_str().expect("config utf8"),
            "-j",
            "0",
            "--dry-run",
        ])
        .assert()
        .success()
        // Must not panic.
        .stderr(predicates::str::contains("thread 'main' panicked").not())
        // Dry-run output should mention the discovered jobs.
        .stdout(predicates::str::contains("Dry Run").or(predicates::str::contains("job")));
}

#[test]
fn batch_dry_run_j1_no_panic() {
    let tmp = TempDir::new().expect("tempdir creation should succeed");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");

    fs::create_dir_all(&input_dir).expect("mkdir input");
    fs::create_dir_all(&output_dir).expect("mkdir output");

    fs::write(input_dir.join("a.mkv"), b"dummy").expect("write dummy a");
    fs::write(input_dir.join("b.mkv"), b"dummy").expect("write dummy b");
    fs::write(input_dir.join("c.mkv"), b"dummy").expect("write dummy c");

    let config = write_batch_config(tmp.path());

    Command::cargo_bin("oximedia")
        .expect("oximedia binary not found")
        .args([
            "batch",
            input_dir.to_str().expect("input_dir utf8"),
            output_dir.to_str().expect("output_dir utf8"),
            config.to_str().expect("config utf8"),
            "-j",
            "1",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicates::str::contains("thread 'main' panicked").not());
}
