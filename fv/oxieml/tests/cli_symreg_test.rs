//! Integration tests for the `oxieml --symreg` CLI subcommand.
//!
//! These tests drive the compiled `oxieml` binary end-to-end via
//! `assert_cmd` and are intentionally lenient on numerical quality:
//! symbolic regression is stochastic, so we assert process-level
//! shape ("runs, exits 0, prints a `Rank 1:` line") rather than
//! exact recovered formulas.

use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;

/// (a) A linear target `y = x0 + x1` should produce at least one ranked
///     formula. We don't assert the exact formula — only that the CLI
///     runs, exits 0, and prints a `Rank 1:` line.
#[test]
fn recovery_linear() {
    // 20 deterministic points covering an integer grid for x0, x1.
    let mut data = String::new();
    data.push_str("# y = x0 + x1\n");
    for i in 0..4 {
        for j in 0..5 {
            let x0 = i as f64;
            let x1 = j as f64;
            let y = x0 + x1;
            data.push_str(&format!("{x0} {x1} {y}\n"));
        }
    }

    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--symreg")
        .arg("--vars")
        .arg("2")
        .arg("--max-depth")
        .arg("1")
        .arg("--max-iter")
        .arg("50")
        .arg("--num-restarts")
        .arg("1")
        .arg("--top")
        .arg("1")
        .write_stdin(data);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Rank 1:"));
}

/// (b) Empty stdin with no --file should yield a non-zero exit and a
///     stderr message containing "no data".
#[test]
fn no_data_errors() {
    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--symreg").arg("--vars").arg("2").write_stdin("");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no data"));
}

/// (c) `oxieml --help` stdout should mention `--symreg`.
#[test]
fn help_mentions_symreg() {
    let mut cmd = Command::cargo_bin("oxieml").expect("binary built");
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--symreg"));
}

/// (d) Writing a dataset to a temp file and passing `--file <path>` runs
///     to completion with exit code 0.
#[test]
fn file_input_works() {
    // Simple y = 2*x0 dataset. Any pattern works; we just need valid rows.
    let mut data = String::new();
    data.push_str("# y = 2*x0\n");
    for i in 0..10 {
        let x0 = i as f64;
        let y = 2.0 * x0;
        data.push_str(&format!("{x0} {y}\n"));
    }

    let dir = env::temp_dir();
    let path = dir.join("oxieml_cli_symreg_test_file_input.txt");
    fs::write(&path, &data).expect("failed to write temp dataset");

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
        .arg("--file")
        .arg(&path);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Rank 1:"));

    // Clean up (best-effort; OS cleans temp dirs).
    let _ = fs::remove_file(&path);
}
