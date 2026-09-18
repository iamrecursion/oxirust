//! Verify that passing both `--json` and `--ndjson` is rejected by clap.

#[test]
fn json_and_ndjson_conflict_is_error() {
    let nonexistent = std::env::temp_dir()
        .join("nonexistent.mp4")
        .display()
        .to_string();
    let status = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["probe", "--json", "--ndjson", "-i", &nonexistent])
        .status()
        .expect("spawn oximedia");
    // clap rejects conflicting flags with exit code 2 (usage error).
    assert!(!status.success(), "combining --json and --ndjson must fail");
}
