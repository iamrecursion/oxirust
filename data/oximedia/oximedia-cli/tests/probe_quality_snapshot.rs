//! Regression tests for `oximedia probe --quality-snapshot`.
//!
//! Prior to this fix, `--quality-snapshot` was parsed by clap
//! (`commands/mod.rs`) but discarded in `main.rs`
//! (`quality_snapshot: _quality_snapshot`) and `handlers/inspect.rs::probe_file`
//! had no parameter for it — the flag was a complete no-op.
//!
//! These tests prove the flag now performs a **real decode of frame 0** of
//! the probed file (via `frame_extract::extract_video_frame_rgb`) and scores
//! the five no-reference metrics (blur/noise/blockiness/BRISQUE/NIQE) on the
//! actual pixels — NOT on a synthetic constant-grey stand-in frame like
//! `quality_cmd::make_grey_frame`.
//!
//! Anti-regression design: the Y4M fixture uses neutral chroma (U = V = 128),
//! which makes the Y4M -> RGB -> BT.601-luma round trip *lossless* — the
//! luma plane the CLI scores is bit-identical to the fixture's Y plane. Each
//! CLI-reported score is therefore cross-checked for near-exact equality
//! against an independent in-process `QualityAssessor` run over the same
//! pattern, and additionally asserted to *differ* from the scores of a
//! solid-grey frame of the same size. A `make_grey_frame`-style shortcut
//! (or any fake/placeholder value) fails both checks immediately.

mod common;

use assert_cmd::Command;
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::path::PathBuf;
use tempfile::TempDir;

/// Fixture dimensions: 96x96 is the smallest size at which *all five*
/// no-reference metrics clear their minimum-size guards (NIQE needs 96x96).
const W: usize = 96;
const H: usize = 96;

/// Metric keys in the JSON output, in snapshot order.
const METRIC_KEYS: [&str; 5] = ["blur", "noise", "blockiness", "brisque", "niqe"];

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

/// Deterministic high-frequency texture. A flat or purely linear pattern
/// would zero out Laplacian variance and weaken the "differs from grey"
/// assertion, so use a pseudo-textured modulo pattern instead.
fn textured_luma(x: usize, y: usize) -> u8 {
    ((x * 7 + y * 13) % 251) as u8
}

/// Write a one-frame 96x96 C420jpeg Y4M fixture with the textured luma
/// plane and neutral chroma (U = V = 128).
fn write_y4m_fixture() -> (TempDir, PathBuf) {
    let mut data = Vec::new();
    data.extend_from_slice(format!("YUV4MPEG2 W{W} H{H} F25:1 Ip C420jpeg\n").as_bytes());
    data.extend_from_slice(b"FRAME\n");
    for y in 0..H {
        for x in 0..W {
            data.push(textured_luma(x, y));
        }
    }
    let chroma_len = W.div_ceil(2) * H.div_ceil(2);
    data.extend(std::iter::repeat_n(128u8, chroma_len * 2)); // U then V

    let dir = TempDir::new().expect("failed to create TempDir");
    let path = dir.path().join("quality_fixture.y4m");
    std::fs::write(&path, &data).expect("failed to write Y4M fixture");
    (dir, path)
}

/// Independently score all five metrics in-process on a Gray8 frame whose
/// luma plane is produced by `fill`. Returns scores keyed like the CLI JSON.
fn score_in_process(fill: impl Fn(usize, usize) -> u8) -> Vec<(&'static str, Option<f64>)> {
    let mut frame =
        Frame::new(W, H, PixelFormat::Gray8).expect("Gray8 frame allocation must succeed");
    for (i, px) in frame.luma_mut().iter_mut().enumerate() {
        *px = fill(i % W, i / W);
    }
    let assessor = QualityAssessor::new();
    let mut scores = Vec::new();
    for (key, metric) in [
        ("blur", MetricType::Blur),
        ("noise", MetricType::Noise),
        ("blockiness", MetricType::Blockiness),
        ("brisque", MetricType::Brisque),
        ("niqe", MetricType::Niqe),
    ] {
        scores.push((
            key,
            assessor
                .assess_no_reference(&frame, metric)
                .ok()
                .map(|s| s.score),
        ));
    }
    scores
}

/// Run `probe --quality-snapshot --format json` on `path` and return the
/// parsed JSON output, asserting the process succeeded.
fn probe_quality_json(path: &std::path::Path, extra_args: &[&str]) -> serde_json::Value {
    let mut args = vec![
        "--quiet",
        "probe",
        "-i",
        path.to_str().expect("fixture path is UTF-8"),
        "--quality-snapshot",
    ];
    args.extend_from_slice(extra_args);

    let output = oximedia()
        .args(&args)
        .output()
        .expect("failed to run oximedia probe --quality-snapshot");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "probe --quality-snapshot must not panic: {stderr}"
    );
    assert!(
        output.status.success(),
        "probe --quality-snapshot should succeed; stderr: {stderr}"
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    serde_json::from_str(&stdout).expect("probe JSON output should parse")
}

#[test]
fn quality_snapshot_json_scores_match_independent_assessor() {
    let (_dir, path) = write_y4m_fixture();
    let json = probe_quality_json(&path, &["--format", "json"]);

    let snapshot = json
        .get("quality_snapshot")
        .unwrap_or_else(|| panic!("JSON output must contain quality_snapshot: {json}"));
    assert_eq!(
        snapshot["available"],
        serde_json::json!(true),
        "snapshot must be available on a decodable video: {snapshot}"
    );
    assert_eq!(snapshot["frame_index"], serde_json::json!(0));
    assert_eq!(snapshot["width"], serde_json::json!(W));
    assert_eq!(snapshot["height"], serde_json::json!(H));

    // Independent oracle: same pattern, scored in-process. Neutral chroma
    // makes the decode round trip lossless, so scores must match near-exactly.
    let expected = score_in_process(textured_luma);
    for (key, expected_score) in &expected {
        let cli_score = snapshot["metrics"][key]["score"].as_f64();
        let want = expected_score
            .unwrap_or_else(|| panic!("{key} must be computable in-process at {W}x{H}"));
        let got = cli_score
            .unwrap_or_else(|| panic!("{key} score must be a number: {}", snapshot["metrics"]));
        assert!(
            (got - want).abs() < 1e-9,
            "{key}: CLI score {got} != independent score {want}"
        );
        assert!(got.is_finite(), "{key} score must be finite: {got}");
    }
}

#[test]
fn quality_snapshot_scores_differ_from_synthetic_grey_frame() {
    // Anti-regression against the `quality_cmd::make_grey_frame` shortcut:
    // scoring a constant-grey frame instead of the real file must be caught.
    let (_dir, path) = write_y4m_fixture();
    let json = probe_quality_json(&path, &["--format", "json"]);
    let metrics = &json["quality_snapshot"]["metrics"];

    let grey = score_in_process(|_x, _y| 128);
    let mut any_differs = false;
    for (key, grey_score) in &grey {
        let cli_score = metrics[key]["score"].as_f64();
        match (cli_score, grey_score) {
            (Some(a), Some(b)) => {
                if (a - b).abs() > 1e-6 {
                    any_differs = true;
                }
            }
            // Score present for one but not the other also proves they differ.
            (Some(_), None) | (None, Some(_)) => any_differs = true,
            (None, None) => {}
        }
    }
    assert!(
        any_differs,
        "scores on the textured fixture must differ from a solid-grey frame; \
         identical scores mean a synthetic frame was scored instead of the real file. \
         metrics: {metrics}, grey: {grey:?}"
    );
}

#[test]
fn quality_snapshot_audio_only_wav_probe_still_succeeds() {
    // A real WAV file has no video stream: the probe must still exit 0 and
    // report the snapshot as unavailable with a clear reason.
    let (_dir, path) = common::write_wav_fixture(440.0, 8_000, 1, 0.05);
    let json = probe_quality_json(&path, &["--format", "json"]);

    let snapshot = json
        .get("quality_snapshot")
        .unwrap_or_else(|| panic!("JSON output must contain quality_snapshot: {json}"));
    assert_eq!(
        snapshot["available"],
        serde_json::json!(false),
        "audio-only input cannot yield a video frame: {snapshot}"
    );
    let reason = snapshot["reason"].as_str().unwrap_or("");
    assert!(
        !reason.is_empty(),
        "unavailable snapshot must carry a reason: {snapshot}"
    );
    assert_eq!(snapshot["metrics"], serde_json::Value::Null);
    assert_eq!(snapshot["width"], serde_json::Value::Null);
}

#[test]
fn quality_snapshot_audio_only_wav_text_output_succeeds() {
    let (_dir, path) = common::write_wav_fixture(440.0, 8_000, 1, 0.05);

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--quality-snapshot",
        ])
        .output()
        .expect("failed to run oximedia probe --quality-snapshot (text)");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "must not panic: {stderr}"
    );
    assert!(
        output.status.success(),
        "text-format probe --quality-snapshot must succeed on audio-only input; stderr: {stderr}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Quality Snapshot"),
        "text output must contain the quality section header: {stdout}"
    );
}

#[test]
fn probe_without_quality_snapshot_flag_omits_field() {
    let (_dir, path) = write_y4m_fixture();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run oximedia probe --format json");

    assert!(
        output.status.success(),
        "probe without --quality-snapshot should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("probe JSON output should parse");
    assert!(
        json.get("quality_snapshot").is_none(),
        "quality_snapshot must be absent without the flag: {json}"
    );
}

#[test]
fn quality_snapshot_csv_appends_metric_columns() {
    let (_dir, path) = write_y4m_fixture();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--quality-snapshot",
            "--format",
            "csv",
        ])
        .output()
        .expect("failed to run oximedia probe --quality-snapshot --format csv");

    assert!(
        output.status.success(),
        "csv probe --quality-snapshot should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let mut lines = stdout.lines();
    let header = lines.next().expect("csv output must have a header line");
    for column in [
        "quality_available",
        "quality_blur",
        "quality_noise",
        "quality_blockiness",
        "quality_brisque",
        "quality_niqe",
    ] {
        assert!(
            header.contains(column),
            "csv header must include {column}: {header}"
        );
    }

    let row = lines.next().expect("csv output must have a data row");
    assert_eq!(
        row.split(',').count(),
        header.split(',').count(),
        "csv data row must have as many cells as the header: {row}"
    );
    assert!(
        row.contains(",true,"),
        "csv row must mark the snapshot available: {row}"
    );
}

#[test]
fn quality_snapshot_ndjson_record_contains_metrics() {
    let (_dir, path) = write_y4m_fixture();

    let output = oximedia()
        .args([
            "--quiet",
            "--ndjson",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--quality-snapshot",
        ])
        .output()
        .expect("failed to run oximedia probe --quality-snapshot --ndjson");

    assert!(
        output.status.success(),
        "ndjson probe --quality-snapshot should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let line = stdout
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("ndjson output must contain a record");
    let record: serde_json::Value =
        serde_json::from_str(line).expect("ndjson line must be valid JSON");

    let snapshot = record
        .get("quality_snapshot")
        .unwrap_or_else(|| panic!("ndjson record must contain quality_snapshot: {record}"));
    assert_eq!(snapshot["available"], serde_json::json!(true));
    for key in METRIC_KEYS {
        assert!(
            snapshot["metrics"][key]["score"].as_f64().is_some(),
            "{key} score must be present in ndjson output: {snapshot}"
        );
    }
}
