//! Integration tests for the `oximedia-cv2 dnn-forward` subcommand.
//! Gated on `cargo --features dnn` because the subcommand itself is only
//! compiled into the binary when that feature is enabled.

#![cfg(feature = "dnn")]

use assert_cmd::Command;
use oximedia_compat_cv2::{image_io::imwrite, mat::Mat};
use tempfile::TempDir;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("oximedia-cv2 binary not found")
}

#[test]
fn dnn_forward_help_exits_zero() {
    cmd().args(["dnn-forward", "--help"]).assert().success();
}

/// When given a non-existent ONNX model path, `dnn-forward` must exit
/// non-zero with a clean error string — never panic, never produce a
/// rust-style backtrace as the only output.
#[test]
fn dnn_forward_missing_model_returns_clean_error() {
    let dir = TempDir::new().expect("tempdir");
    // Synthesise a 32×32 BGR PNG so the imread step succeeds and we land in
    // the actual model load path.
    let h = 32usize;
    let w = 32usize;
    let data = vec![128u8; h * w * 3];
    let input = dir.path().join("dnn_in.png");
    imwrite(&input, &Mat::from_bgr_bytes(data, h, w)).expect("write dnn input");

    let bogus_model = dir.path().join("does_not_exist.onnx");

    let out = cmd()
        .args([
            "dnn-forward",
            "--model",
            bogus_model.to_str().expect("bogus path utf8"),
            "--input",
            input.to_str().expect("input path utf8"),
            "--resize",
            "32x32",
        ])
        .output()
        .expect("run dnn-forward");

    assert!(
        !out.status.success(),
        "dnn-forward must fail on missing model"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Must be a clean message, not a panic or `RUST_BACKTRACE` dump.
    assert!(
        stderr.contains("read_net_from_onnx") || stderr.contains("model"),
        "expected error to mention the model load failure, got: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "stderr must not be a panic backtrace: {stderr}"
    );
    assert!(
        !stderr.contains("panicked at"),
        "stderr must not contain a panic, got: {stderr}"
    );
}
