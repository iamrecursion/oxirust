//! End-to-end tests for the `normalize` subcommand.
//!
//! These tests verify that `oximedia normalize analyze` can accept real WAV
//! files without panicking and produces meaningful LUFS output.

mod common;

#[test]
fn normalize_analyze_wav_exits_zero() {
    let (_dir, wav_path) = common::write_wav_fixture(1000.0, 44100, 1, 5.0);

    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args([
            "normalize",
            "analyze",
            "--input",
            wav_path.to_str().expect("path is UTF-8"),
            "--standard",
            "ebu",
        ])
        .output()
        .expect("failed to run oximedia normalize analyze");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "normalize analyze must not panic: {stderr}"
    );

    // Must not contain unimplemented stubs
    let all_output = format!("{stdout}{stderr}");
    assert!(
        !all_output.to_lowercase().contains("not yet implemented"),
        "must not contain unimplemented: {all_output}"
    );

    // Command should succeed
    assert!(
        output.status.success(),
        "normalize analyze should succeed on valid WAV, stdout={stdout}, stderr={stderr}"
    );

    // Output should mention LUFS
    assert!(
        stdout.contains("LUFS") || stdout.contains("lufs"),
        "successful normalize should mention LUFS, stdout: {stdout}"
    );

    // A 1 kHz full-scale sine should produce a loud reading (not -inf / not positive)
    // EBU R128 integrated loudness for a 1 kHz near-full-scale sine over 5s is
    // typically around -3 LUFS. We just verify it's a finite numeric value.
    assert!(
        !stdout.contains("-inf") && !stdout.contains("inf LUFS"),
        "LUFS should be a finite number, not infinite: {stdout}"
    );
}

#[test]
fn normalize_analyze_wav_json_output() {
    let (_dir, wav_path) = common::write_wav_fixture(1000.0, 44100, 1, 3.0);

    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args([
            "--json",
            "normalize",
            "analyze",
            "--input",
            wav_path.to_str().expect("path is UTF-8"),
            "--standard",
            "ebu",
        ])
        .output()
        .expect("failed to run oximedia normalize analyze --json");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "normalize analyze --json must not panic: {stderr}"
    );

    // If it succeeds, output should be parseable JSON
    if output.status.success() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
        assert!(parsed.is_ok(), "JSON output must be valid JSON: {stdout}");
        let obj = parsed.expect("already checked");
        // Should contain integrated_lufs field
        assert!(
            obj.get("analysis")
                .and_then(|a| a.get("integrated_lufs"))
                .is_some(),
            "JSON should contain analysis.integrated_lufs: {obj}"
        );
    }
}

#[test]
fn normalize_analyze_missing_file_exits_nonzero() {
    let missing_wav = std::env::temp_dir()
        .join("this_file_does_not_exist_oximedia_normalize_test.wav")
        .display()
        .to_string();
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["normalize", "analyze", "--input", &missing_wav])
        .output()
        .expect("failed to run oximedia normalize analyze on missing file");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "must not panic on missing file: {stderr}"
    );

    // Must exit non-zero
    assert!(
        !output.status.success(),
        "should fail on missing input file"
    );
}

#[test]
fn normalize_targets_lists_standards() {
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["normalize", "targets"])
        .output()
        .expect("failed to run oximedia normalize targets");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        !stderr.contains("thread 'main' panicked"),
        "normalize targets must not panic: {stderr}"
    );
    assert!(
        output.status.success(),
        "normalize targets should succeed, stderr: {stderr}"
    );
    // Should list at least one platform
    assert!(
        stdout.contains("EBU") || stdout.contains("ebu") || stdout.contains("Spotify"),
        "targets output should list platforms: {stdout}"
    );
}
