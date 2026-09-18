//! Smoke tests for structured exit codes.

#[test]
fn probe_nonexistent_file_exit_code_3() {
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["probe", "-i", "/absolutely/nonexistent/path/file.mp4"])
        .status()
        .expect("spawn oximedia");
    // IoError = 3
    assert_eq!(
        status.code(),
        Some(3),
        "expected exit code 3 for nonexistent file"
    );
}

#[test]
fn unknown_subcommand_exit_code_2() {
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["this-subcommand-does-not-exist"])
        .status()
        .expect("spawn oximedia");
    // UsageError = 2 (clap error)
    assert_eq!(
        status.code(),
        Some(2),
        "expected exit code 2 for unknown subcommand"
    );
}
