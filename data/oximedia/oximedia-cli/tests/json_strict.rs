//! Verify basic JSON output strictness for select subcommands.

/// Sanity check: `--help` still works (basic CLI health).
#[test]
fn help_works() {
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .arg("--help")
        .status()
        .expect("spawn oximedia");
    assert!(status.success(), "--help must exit 0");
}

/// `--json` flag is accepted globally without causing a parse error.
#[test]
fn global_json_flag_is_recognized() {
    // This will fail because the file doesn't exist, but the exit code
    // should be 3 (IoError), NOT 2 (usage/parse error).
    let nonexistent = std::env::temp_dir()
        .join("__nonexistent_oximedia.mp4")
        .display()
        .to_string();
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["--json", "probe", "-i", &nonexistent])
        .status()
        .expect("spawn oximedia");
    // Should not be a clap error (2); 3 = IoError is acceptable.
    assert_ne!(
        status.code(),
        Some(2),
        "--json flag must not cause a parse error"
    );
}

/// `--ndjson` flag is accepted globally without causing a parse error.
#[test]
fn global_ndjson_flag_is_recognized() {
    let nonexistent2 = std::env::temp_dir()
        .join("__nonexistent_oximedia.mp4")
        .display()
        .to_string();
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["--ndjson", "probe", "-i", &nonexistent2])
        .status()
        .expect("spawn oximedia");
    assert_ne!(
        status.code(),
        Some(2),
        "--ndjson flag must not cause a parse error"
    );
}
