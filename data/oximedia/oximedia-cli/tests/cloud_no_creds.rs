//! Integration test: cloud upload without credentials must give a clean error,
//! never a panic and never a "simulation" message.

/// Test that cloud upload without credentials gives a clean error.
#[test]
fn cloud_upload_s3_no_creds_clean_error() {
    let input_path = std::env::temp_dir()
        .join("nonexistent_oximedia_test_file.txt")
        .display()
        .to_string();
    // Remove any AWS env vars to ensure no credentials are available.
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .env_remove("AWS_ACCESS_KEY_ID")
        .env_remove("AWS_SECRET_ACCESS_KEY")
        .env_remove("AWS_PROFILE")
        .env_remove("AWS_DEFAULT_PROFILE")
        .env_remove("AWS_SHARED_CREDENTIALS_FILE")
        .args([
            "cloud",
            "upload",
            "--provider",
            "s3",
            "--bucket",
            "test-bucket",
            "--key",
            "test/key",
            "--input",
            &input_path,
        ])
        .output()
        .expect("failed to spawn oximedia");

    // The command must fail (no credentials → no successful upload).
    assert!(
        !output.status.success(),
        "should fail without credentials; status = {:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let all_output = format!("{stderr}{stdout}");

    // Must NOT be a panic.
    assert!(
        !all_output.contains("thread 'main' panicked"),
        "must not panic, got: {all_output}"
    );

    // Must NOT contain simulation language.
    assert!(
        !all_output.to_lowercase().contains("simulation"),
        "must not use simulation strings, got: {all_output}"
    );
    assert!(
        !all_output.contains("cloud credentials required for execution"),
        "must not use old simulation string, got: {all_output}"
    );

    // Should mention the missing env var or credentials.
    assert!(
        all_output.contains("AWS_ACCESS_KEY_ID")
            || all_output.contains("credentials")
            || all_output.contains("not found"),
        "error should mention AWS_ACCESS_KEY_ID, credentials, or not found; got: {all_output}"
    );
}

/// Test that an unknown provider gives a clear, non-panicking error.
#[test]
fn cloud_upload_unknown_provider_clean_error() {
    let input_path2 = std::env::temp_dir()
        .join("nonexistent_oximedia_test_file2.txt")
        .display()
        .to_string();
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args([
            "cloud",
            "upload",
            "--provider",
            "unknowncloud999",
            "--bucket",
            "bucket",
            "--key",
            "key",
            "--input",
            &input_path2,
        ])
        .output()
        .expect("failed to spawn oximedia");

    assert!(
        !output.status.success(),
        "should fail with unknown provider"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let all_output = format!("{stderr}{stdout}");

    assert!(
        !all_output.contains("thread 'main' panicked"),
        "must not panic: {all_output}"
    );
    assert!(
        all_output.contains("Unknown")
            || all_output.contains("unknown")
            || all_output.contains("provider"),
        "error should mention unknown provider: {all_output}"
    );
}
