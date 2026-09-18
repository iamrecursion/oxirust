//! Smoke tests for the `doctor` subcommand.

use assert_cmd::Command;

#[test]
fn doctor_exits_zero() {
    Command::cargo_bin("oximedia")
        .expect("oximedia binary should exist")
        .args(["doctor"])
        .assert()
        .success();
}

#[test]
fn doctor_json_is_valid() {
    let output = Command::cargo_bin("oximedia")
        .expect("oximedia binary should exist")
        .args(["doctor", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let s = String::from_utf8(output).expect("doctor --json output should be UTF-8");
    let v: serde_json::Value =
        serde_json::from_str(&s).expect("doctor --json should produce valid JSON");
    assert!(v.get("rust_version").is_some(), "JSON missing rust_version");
    assert!(v.get("temp_dir").is_some(), "JSON missing temp_dir");
}
