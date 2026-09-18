//! Integration tests for the `phop` binary.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

/// Write a tiny CSV where the last column is the target (`y = x0 + x1`).
fn write_last_target_csv() -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".csv").expect("create temp csv");
    writeln!(f, "x0,x1,y").unwrap();
    for i in 0..16 {
        let a = i as f64;
        let b = (i % 4) as f64;
        writeln!(f, "{a},{b},{}", a + b).unwrap();
    }
    f.flush().unwrap();
    f
}

/// Write a tiny CSV where the FIRST column is the target (`y = x0 + x1`),
/// stored in column 0 (`y`).
fn write_first_target_csv() -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".csv").expect("create temp csv");
    writeln!(f, "y,x0,x1").unwrap();
    for i in 0..16 {
        let a = i as f64;
        let b = (i % 4) as f64;
        writeln!(f, "{},{a},{b}", a + b).unwrap();
    }
    f.flush().unwrap();
    f
}

/// Write a CSV for a multiplicative law `y = x0 * x1` (positive inputs) — only the rich-leaf
/// (log-linear) engine recovers it; the shallow EML core cannot assemble a product.
fn write_product_csv() -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".csv").expect("create temp csv");
    writeln!(f, "x0,x1,y").unwrap();
    for i in 0..40 {
        let a = 1.0 + 0.1 * i as f64;
        let b = 0.5 + 0.07 * i as f64;
        writeln!(f, "{a},{b},{}", a * b).unwrap();
    }
    f.flush().unwrap();
    f
}

/// Write a CSV for `y = exp(x0)` — recovered exactly as `eml(x0, 1)` by the enumerate method.
fn write_exp_csv() -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".csv").expect("create temp csv");
    writeln!(f, "x0,y").unwrap();
    for i in 0..30 {
        let x = i as f64 * 0.1;
        writeln!(f, "{x},{}", x.exp()).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn predict_round_trips_a_discovered_law() {
    // Discover an exp law to a JSON model, then `predict` with it on the same data: the predictions
    // must reproduce the law (R² ≈ 1), exercising the EML-tree serialize → deserialize round-trip.
    let csv = write_exp_csv();
    let model = NamedTempFile::with_suffix(".json").expect("create temp model");
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--method", "enumerate"])
        .args(["--format", "json"])
        .args(["--max-epochs", "150"])
        .arg("--quiet")
        .args(["--output", model.path().to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("phop")
        .unwrap()
        .arg("predict")
        .arg(model.path())
        .arg(csv.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("prediction"))
        .stderr(predicate::str::contains("R²"));
}

#[test]
fn certify_prints_certified_range() {
    // `--certify` must print a guaranteed range enclosure of the best law to stderr.
    let csv = write_exp_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--method", "enumerate"])
        .args(["--max-epochs", "120"])
        .arg("--certify")
        .assert()
        .success()
        .stderr(predicate::str::contains("certified range"));
}

#[test]
fn help_succeeds_and_mentions_discover() {
    Command::cargo_bin("phop")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("discover"));
}

#[test]
fn discover_json_format_parses_with_mse() {
    let csv = write_last_target_csv();
    let assert = Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--format", "json"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .args(["--top-k", "3"])
        .assert()
        .success();

    let out = assert.get_output();
    let stdout = String::from_utf8(out.stdout.clone()).expect("utf8 stdout");
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert!(parsed.is_array(), "JSON output should be an array");
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty(), "expected at least one solution");
    assert!(
        arr[0].get("mse").is_some(),
        "each solution object must contain an 'mse' field"
    );
}

#[test]
fn target_zero_selects_first_column() {
    let csv = write_first_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--target", "0"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .args(["--top-k", "3"])
        .assert()
        .success();
}

#[test]
fn target_by_name_selects_column() {
    let csv = write_first_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--target", "y"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .args(["--top-k", "3"])
        .assert()
        .success();
}

#[test]
fn unknown_target_column_errors() {
    let csv = write_last_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--target", "nope"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown column"));
}

#[test]
fn analyze_prints_cas_analysis() {
    // --analyze should print the symbolic CAS analysis (derivative etc.) of the best law to stderr.
    let csv = write_last_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .arg("--analyze")
        .assert()
        .success()
        .stderr(predicate::str::contains("d/dx0"));
}

#[test]
fn features_subset_by_name_runs() {
    // Keep only feature `x0` (drop `x1`); target defaults to the last column `y`.
    let csv = write_last_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--features", "x0"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .args(["--verbose"])
        .assert()
        .success()
        // The verbose line should report a single feature column.
        .stderr(predicate::str::contains("1 feature(s) [x0]"));
}

#[test]
fn auto_method_reaches_rich_engine() {
    // `--method auto` must now include the rich-leaf (affine) engine, so a multiplicative law the
    // shallow EML core can't assemble is still recovered. The merged JSON carries a `source` field,
    // and at least one accurate `affine` member must appear.
    let csv = write_product_csv();
    let assert = Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .args(["--method", "auto"])
        .args(["--format", "json"])
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .args(["--top-k", "5"])
        .assert()
        .success();

    let out = assert.get_output();
    let stdout = String::from_utf8(out.stdout.clone()).expect("utf8 stdout");
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    let arr = parsed.as_array().expect("JSON output should be an array");
    assert!(!arr.is_empty(), "expected at least one solution");
    assert!(
        arr[0].get("source").is_some(),
        "each merged solution must carry a 'source' field"
    );
    let best_affine_r2 = arr
        .iter()
        .filter(|s| s.get("source").and_then(serde_json::Value::as_str) == Some("affine"))
        .filter_map(|s| s.get("r2").and_then(serde_json::Value::as_f64))
        .fold(f64::MIN, f64::max);
    assert!(
        best_affine_r2 > 0.99,
        "auto must reach an accurate affine member (best affine r2 = {best_affine_r2})"
    );
}

#[test]
fn missing_csv_errors() {
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg("/this/path/does/not/exist.csv")
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .assert()
        .failure();
}

#[test]
fn quiet_suppresses_loaded_line() {
    let csv = write_last_target_csv();
    Command::cargo_bin("phop")
        .unwrap()
        .arg("discover")
        .arg(csv.path())
        .arg("--quiet")
        .args(["--max-epochs", "50"])
        .args(["--population", "16"])
        .assert()
        .success()
        .stderr(predicate::str::contains("phop: loaded").not());
}
