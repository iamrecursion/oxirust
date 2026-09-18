use assert_cmd::Command;
use predicates::prelude::*;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("binary not found")
}

#[test]
fn list_functions_has_enough_entries() {
    let output = cmd()
        .arg("--list-functions")
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Count non-header lines (entries have the function name followed by spaces)
    let count = stdout
        .lines()
        .filter(|l| {
            !l.starts_with('-')
                && !l.starts_with("Function")
                && !l.starts_with("Total")
                && !l.is_empty()
        })
        .count();
    assert!(count >= 30, "expected >=30 functions, got {count}");
}

#[test]
fn list_constants_has_enough_entries() {
    let output = cmd()
        .arg("--list-constants")
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Lines with " = " are constant entries
    let count = stdout.lines().filter(|l| l.contains(" = ")).count();
    assert!(count >= 130, "expected >=130 constants, got {count}");
}

#[test]
fn list_functions_contains_known_functions() {
    cmd()
        .arg("--list-functions")
        .assert()
        .success()
        .stdout(predicate::str::contains("imread"))
        .stdout(predicate::str::contains("gaussian_blur"))
        .stdout(predicate::str::contains("canny"))
        .stdout(predicate::str::contains("threshold"));
}

#[test]
fn list_constants_contains_known_constants() {
    cmd()
        .arg("--list-constants")
        .assert()
        .success()
        .stdout(predicate::str::contains("IMREAD_COLOR"))
        .stdout(predicate::str::contains("COLOR_BGR2GRAY"))
        .stdout(predicate::str::contains("THRESH_BINARY"))
        .stdout(predicate::str::contains("INTER_LINEAR"));
}
