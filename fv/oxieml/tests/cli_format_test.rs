//! Integration tests for `--format` and `--output` CLI flags.
//!
//! Tests exercise `--lower`, `--symreg`, and the default eval path
//! with each of the three output formats (pretty, latex, json) and
//! the `--output <path>` flag.

use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;

// ================================================================
// Helper: build a fast symreg dataset (y = x0, 5 points)
// ================================================================

fn linear_dataset() -> String {
    let mut s = String::from("# y = x0\n");
    for i in 0..5 {
        let x0 = i as f64;
        s.push_str(&format!("{x0} {x0}\n"));
    }
    s
}

// ================================================================
// 1. format_pretty_default
// ================================================================

/// Running `--lower` without `--format` should print the pretty form.
#[test]
fn format_pretty_default() {
    // E(x0,1) = exp(x0).  After lower+simplify the pretty form is "exp(x0)".
    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--lower").arg("E(x0,1)");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("exp"));
}

// ================================================================
// 2. format_latex
// ================================================================

/// `--lower --format latex` should produce LaTeX markers.
#[test]
fn format_latex() {
    // E(x0,1) = exp(x0).  LaTeX for Exp(x0) is "e^{x_{0}}".
    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--lower")
        .arg("E(x0,1)")
        .arg("--format")
        .arg("latex");

    cmd.assert()
        .success()
        // LaTeX delimiters present
        .stdout(predicate::str::contains("$$"))
        // Variable subscript notation
        .stdout(predicate::str::contains("x_{"));
}

// ================================================================
// 3. format_json_lower
// ================================================================

/// `--lower --format json` should produce a JSON envelope with version and formulas.
#[test]
fn format_json_lower() {
    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--lower")
        .arg("E(x0,1)")
        .arg("--format")
        .arg("json");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"version\":1"))
        .stdout(predicate::str::contains("\"formulas\""))
        .stdout(predicate::str::contains("\"pretty\""))
        .stdout(predicate::str::contains("\"latex\""));
}

// ================================================================
// 4. output_to_file
// ================================================================

/// `--lower --format pretty --output <path>` should create the file with content.
#[test]
fn output_to_file() {
    // Include the process ID to avoid collisions when tests run in parallel.
    let filename = format!("oxieml_test_out_format_{}.txt", std::process::id());
    let out_path = env::temp_dir().join(&filename);

    // Clean up any leftover from a prior run.
    let _ = fs::remove_file(&out_path);

    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--lower")
        .arg("E(x0,1)")
        .arg("--format")
        .arg("pretty")
        .arg("--output")
        .arg(&out_path);

    // stdout should be empty (content went to file), exit 0.
    cmd.assert().success().stdout(predicate::str::is_empty());

    // The file must exist and contain the pretty representation.
    let content = fs::read_to_string(&out_path).expect("output file should exist");
    assert!(
        content.contains("exp"),
        "output file should contain 'exp', got: {content:?}"
    );

    // Clean up.
    let _ = fs::remove_file(&out_path);
}

// ================================================================
// 5. format_json_symreg
// ================================================================

/// `--symreg --format json` should emit valid JSON with `"version":1`.
#[test]
fn format_json_symreg() {
    let data = linear_dataset();

    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--symreg")
        .arg("--vars")
        .arg("1")
        .arg("--max-depth")
        .arg("1")
        .arg("--max-iter")
        .arg("50")
        .arg("--num-restarts")
        .arg("1")
        .arg("--top")
        .arg("1")
        .arg("--format")
        .arg("json")
        .write_stdin(data);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"version\":1"))
        .stdout(predicate::str::contains("\"formulas\""))
        .stdout(predicate::str::contains("\"rank\""))
        .stdout(predicate::str::contains("\"mse\""));
}

// ================================================================
// Additional coverage: latex output for symreg
// ================================================================

/// `--symreg --format latex` should produce `$$...$$` markers.
#[test]
fn format_latex_symreg() {
    let data = linear_dataset();

    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--symreg")
        .arg("--vars")
        .arg("1")
        .arg("--max-depth")
        .arg("1")
        .arg("--max-iter")
        .arg("50")
        .arg("--num-restarts")
        .arg("1")
        .arg("--top")
        .arg("1")
        .arg("--format")
        .arg("latex")
        .write_stdin(data);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Rank 1:"))
        .stdout(predicate::str::contains("$$"));
}
