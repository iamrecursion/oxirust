//! Integration tests for the `oximedia ml` subcommand family.
//!
//! These spawn the real `oximedia` binary via `assert_cmd` so we exercise
//! the whole clap → handler → renderer path. Both feature-on and
//! feature-off configurations are covered — tests conditionally gate
//! themselves on `cfg!(feature = "ml")`.
//!
//! Build notes:
//! * `assert_cmd` auto-locates the binary built for the current profile.
//! * Integration tests compile against `oximedia-cli`'s feature set, so
//!   when the test harness runs with `--features ml`, the binary under
//!   test is also built with that feature.

use assert_cmd::Command;
use predicates::prelude::*;

// ---------------------------------------------------------------------------
// feature = "ml" OFF — the subcommand should surface a clear rebuild hint
// ---------------------------------------------------------------------------

#[cfg(not(feature = "ml"))]
#[test]
fn ml_list_without_ml_feature() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("list");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--features ml"));
}

#[cfg(not(feature = "ml"))]
#[test]
fn ml_probe_without_ml_feature() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("probe");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--features ml"));
}

#[cfg(not(feature = "ml"))]
#[test]
fn ml_run_without_ml_feature() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    let missing_model = std::env::temp_dir()
        .join("does-not-exist.onnx")
        .display()
        .to_string();
    let missing_input = std::env::temp_dir()
        .join("also-absent.png")
        .display()
        .to_string();
    cmd.arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("scene-classifier")
        .arg("--model")
        .arg(&missing_model)
        .arg("--input")
        .arg(&missing_input)
        .arg("--dry-run");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--features ml"));
}

// ---------------------------------------------------------------------------
// feature = "ml" ON — full dispatch tests
// ---------------------------------------------------------------------------

#[cfg(feature = "ml")]
#[test]
fn ml_list_lists_built_in_pipelines() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("list");
    // The list output mentions each of the five pipeline IDs.
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("scene-classifier/places365"))
        .stdout(predicate::str::contains("shot-boundary/transnet-v2"))
        .stdout(predicate::str::contains("aesthetic-score/nima"))
        .stdout(predicate::str::contains("object-detector/yolov8"))
        .stdout(predicate::str::contains("face-embedder/arcface"))
        // The default model zoo is also surfaced.
        .stdout(predicate::str::contains("places365/resnet18"))
        .stdout(predicate::str::contains("transnet-v2"));
}

#[cfg(feature = "ml")]
#[test]
fn ml_list_json_parses() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("list").arg("--json");
    let output = cmd.output().expect("spawn oximedia");
    assert!(
        output.status.success(),
        "oximedia ml list --json exit != 0: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("ml list --json produced valid JSON");
    assert_eq!(parsed["command"], "ml list");
    assert!(parsed["pipelines"].is_array());
    assert!(parsed["model_zoo"].is_array());
    let pipelines = parsed["pipelines"].as_array().expect("pipelines array");
    assert!(
        pipelines.len() >= 5,
        "expected >= 5 pipelines, got {}",
        pipelines.len()
    );
}

#[cfg(feature = "ml")]
#[test]
fn ml_probe_shows_cpu() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("probe");
    cmd.assert()
        .success()
        // The text renderer always lists the CPU row (CPU is always available).
        .stdout(predicate::str::contains("CPU"))
        .stdout(predicate::str::contains("cpu"));
}

#[cfg(feature = "ml")]
#[test]
fn ml_probe_json_parses() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("probe").arg("--json");
    let output = cmd.output().expect("spawn oximedia");
    assert!(
        output.status.success(),
        "oximedia ml probe --json exit != 0: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("ml probe --json produced valid JSON");
    assert_eq!(parsed["command"], "ml probe");
    let devices = parsed["devices"].as_array().expect("devices array");
    assert!(
        !devices.is_empty(),
        "expected >= 1 device entry in probe output"
    );
    // Every device entry must carry the canonical fields.
    for (idx, dev) in devices.iter().enumerate() {
        assert!(
            dev["device_type"].is_string(),
            "devices[{idx}].device_type not a string: {dev}"
        );
        assert!(
            dev["is_available"].is_boolean(),
            "devices[{idx}].is_available not a bool: {dev}"
        );
        assert!(
            dev["device_name"].is_string(),
            "devices[{idx}].device_name not a string: {dev}"
        );
    }
    // One of the entries must be CPU and it must be available.
    let cpu = devices
        .iter()
        .find(|d| d["device_type"] == "cpu")
        .expect("cpu entry present");
    assert_eq!(cpu["is_available"], true);
}

#[cfg(feature = "ml")]
#[test]
fn ml_probe_device_filter() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml")
        .arg("probe")
        .arg("--device")
        .arg("cpu")
        .arg("--json");
    let output = cmd.output().expect("spawn oximedia");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let devices = parsed["devices"].as_array().expect("devices array");
    assert_eq!(devices.len(), 1, "filter should yield exactly one device");
    assert_eq!(devices[0]["device_type"], "cpu");
}

#[cfg(feature = "ml")]
#[test]
fn ml_probe_rejects_unknown_device() {
    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml").arg("probe").arg("--device").arg("bogus-gpu");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unknown device"));
}

#[cfg(feature = "ml")]
#[test]
fn ml_run_dry_run_scene_classifier() {
    // Create temp files so the --model and --input paths resolve. The
    // contents are irrelevant because --dry-run short-circuits before any
    // ONNX parsing / image decoding happens.
    let tmp = std::env::temp_dir();
    let model_path = tmp.join("oximedia-ml-test-model.onnx");
    let input_path = tmp.join("oximedia-ml-test-input.png");
    std::fs::write(&model_path, b"not a real model").expect("write model placeholder");
    std::fs::write(&input_path, b"not a real image").expect("write input placeholder");

    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("scene-classifier")
        .arg("--model")
        .arg(&model_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--device")
        .arg("cpu")
        .arg("--top-k")
        .arg("5")
        .arg("--dry-run");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("scene-classifier/places365"))
        .stdout(predicate::str::contains("CPU"))
        .stdout(predicate::str::contains("dry-run"));

    // Also exercise --json + --dry-run so the structured output path is covered.
    let mut json_cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    json_cmd
        .arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("scene-classifier")
        .arg("--model")
        .arg(&model_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--device")
        .arg("cpu")
        .arg("--dry-run")
        .arg("--json");
    let output = json_cmd.output().expect("spawn oximedia");
    assert!(output.status.success(), "dry-run --json should succeed");
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["command"], "ml run");
    assert_eq!(parsed["mode"], "dry-run");
    assert_eq!(parsed["pipeline"]["id"], "scene-classifier/places365");
    assert_eq!(parsed["device"], "cpu");

    // Best-effort cleanup; ignored on failure.
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&input_path);
}

#[cfg(feature = "ml")]
#[test]
fn ml_run_rejects_unknown_pipeline() {
    let tmp = std::env::temp_dir();
    let model_path = tmp.join("oximedia-ml-test-model-bad-pipeline.onnx");
    let input_path = tmp.join("oximedia-ml-test-input-bad-pipeline.png");
    std::fs::write(&model_path, b"x").expect("write model placeholder");
    std::fs::write(&input_path, b"x").expect("write input placeholder");

    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("no-such-pipeline")
        .arg("--model")
        .arg(&model_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--dry-run");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unknown pipeline"));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&input_path);
}

#[cfg(feature = "ml")]
#[test]
fn ml_run_rejects_unknown_device() {
    let tmp = std::env::temp_dir();
    let model_path = tmp.join("oximedia-ml-test-model-bad-device.onnx");
    let input_path = tmp.join("oximedia-ml-test-input-bad-device.png");
    std::fs::write(&model_path, b"x").expect("write model placeholder");
    std::fs::write(&input_path, b"x").expect("write input placeholder");

    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("scene-classifier")
        .arg("--model")
        .arg(&model_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--device")
        .arg("quantum-annealer")
        .arg("--dry-run");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unknown device"));

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&input_path);
}

#[cfg(feature = "ml")]
#[test]
fn ml_run_rejects_missing_model_file() {
    let tmp = std::env::temp_dir();
    let input_path = tmp.join("oximedia-ml-test-input-missing-model.png");
    std::fs::write(&input_path, b"x").expect("write input placeholder");
    // Intentionally do NOT create the model file.
    let model_path = tmp.join("oximedia-ml-test-model-missing-2026.onnx");
    let _ = std::fs::remove_file(&model_path);

    let mut cmd = Command::cargo_bin("oximedia").expect("locate oximedia binary");
    cmd.arg("ml")
        .arg("run")
        .arg("--pipeline")
        .arg("scene-classifier")
        .arg("--model")
        .arg(&model_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--dry-run");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("model file not found"));

    let _ = std::fs::remove_file(&input_path);
}
