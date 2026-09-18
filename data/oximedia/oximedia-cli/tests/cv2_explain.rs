use assert_cmd::Command;
use std::env;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("binary not found")
}

#[test]
fn explain_probe_prints_dispatch_no_output_file() {
    let fake_output = env::temp_dir().join("cv2_explain_should_not_exist.png");
    let _ = std::fs::remove_file(&fake_output); // ensure clean state

    cmd()
        .args(["--explain", "probe", "nonexistent.png"])
        .assert()
        .success()
        .stdout(predicates::str::contains("dispatch:"))
        .stdout(predicates::str::contains("explain"));

    // --explain must NOT create an output file
    assert!(
        !fake_output.exists(),
        "--explain should not write output files"
    );
}

#[test]
fn explain_canny_contains_dispatch_info() {
    cmd()
        .args([
            "--explain",
            "canny",
            "in.png",
            "out.png",
            "--threshold1",
            "100",
            "--threshold2",
            "200",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("canny"));
}
