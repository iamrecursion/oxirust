//! Smoke test for `--ndjson` flag on the `probe` subcommand.

#[test]
fn probe_ndjson_emits_valid_json_lines() {
    // /dev/null exists on Unix and is a valid (empty) file.
    // probe may succeed or fail; either way, if it prints anything it
    // must be a valid NDJSON line.
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args(["probe", "--ndjson", "-i", "/dev/null"])
        .output()
        .expect("spawn oximedia");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.trim().lines() {
            if !line.is_empty() {
                let _: serde_json::Value =
                    serde_json::from_str(line).expect("each stdout line must be valid JSON");
            }
        }
    }
    // Non-zero exit (e.g. probe error on /dev/null) is also acceptable.
}
