//! Integration tests for the new `oximedia-cv2` Slice F op subcommands and
//! cross-cutting smoke checks for all 10 Wave 2 additions.
//!
//! Each op test:
//!   1. Synthesises a small (32×32) PNG with `oximedia-compat-cv2` `imwrite`.
//!   2. Runs the corresponding `oximedia-cv2 <op> ...` invocation.
//!   3. Asserts the produced output file is decodable via `imread` and has
//!      the expected dimensions / channel count.

use assert_cmd::Command;
use oximedia_compat_cv2::{
    image_io::{imread, imwrite},
    mat::Mat,
    IMREAD_UNCHANGED,
};
use std::path::PathBuf;
use tempfile::TempDir;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("oximedia-cv2 binary not found")
}

/// Build a 32×32 BGR PNG with a vertical step (left dark, right bright) so
/// gradient ops have something to find.
fn write_gradient_bgr_png(dir: &TempDir, name: &str) -> PathBuf {
    let h = 32usize;
    let w = 32usize;
    let mut data = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 3;
            let v = if x >= w / 2 { 220 } else { 30 };
            data[off] = v; // B
            data[off + 1] = v; // G
            data[off + 2] = v; // R
        }
    }
    let path = dir.path().join(name);
    let mat = Mat::from_bgr_bytes(data, h, w);
    imwrite(&path, &mat).expect("imwrite synth gradient PNG");
    path
}

/// Build a 32×32 grayscale PNG with a vertical step.
fn write_gradient_gray_png(dir: &TempDir, name: &str) -> PathBuf {
    let h = 32usize;
    let w = 32usize;
    let mut data = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = if x >= w / 2 { 220 } else { 30 };
        }
    }
    let path = dir.path().join(name);
    let mat = Mat::from_gray_bytes(data, h, w);
    imwrite(&path, &mat).expect("imwrite synth gradient gray PNG");
    path
}

// ── Per-op tests ─────────────────────────────────────────────────────────────

#[test]
fn flip_writes_output_with_same_dimensions() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_bgr_png(&dir, "in.png");
    let output = dir.path().join("flipped.png");
    cmd()
        .args([
            "flip",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--code",
            "y",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read flipped");
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
    assert_eq!(out_mat.channels(), 3);
}

#[test]
fn rotate_swaps_dimensions_for_90_degree() {
    let dir = TempDir::new().unwrap();
    // Use a non-square image so the dim swap is observable.
    let h = 32usize;
    let w = 16usize;
    let mut data = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 3;
            let v = (x * 16) as u8;
            data[off] = v;
            data[off + 1] = v;
            data[off + 2] = v;
        }
    }
    let input = dir.path().join("rect.png");
    imwrite(&input, &Mat::from_bgr_bytes(data, h, w)).expect("write rect");
    let output = dir.path().join("rot.png");
    cmd()
        .args([
            "rotate",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--code",
            "cw90",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read rotated");
    // 90° CW rotation on a 32×16 (rows × cols) input → 16×32.
    assert_eq!(out_mat.rows, 16);
    assert_eq!(out_mat.cols, 32);
}

#[test]
fn sobel_writes_displayable_grayscale() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_gray_png(&dir, "in.png");
    let output = dir.path().join("sobel.png");
    cmd()
        .args([
            "sobel",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--dx",
            "1",
            "--dy",
            "0",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read sobel");
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
    assert_eq!(out_mat.channels(), 1);
    // The vertical step should produce *some* non-zero gradient at the seam.
    assert!(
        out_mat.data.iter().any(|&v| v > 0),
        "expected non-zero sobel response"
    );
}

#[test]
fn laplacian_writes_displayable_grayscale() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_gray_png(&dir, "in.png");
    let output = dir.path().join("lap.png");
    cmd()
        .args([
            "laplacian",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--ksize",
            "3",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read laplacian");
    assert_eq!(out_mat.channels(), 1);
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
}

#[test]
fn adaptive_threshold_returns_binary_image() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_gray_png(&dir, "in.png");
    let output = dir.path().join("adt.png");
    cmd()
        .args([
            "adaptive-threshold",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--max-value",
            "255",
            "--method",
            "mean",
            "--type",
            "binary",
            "--block-size",
            "5",
            "--c",
            "1",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read adaptive thresh");
    assert_eq!(out_mat.channels(), 1);
    // Output must be (mostly) binary — values dominated by 0 and 255 with a
    // very small slack for codec rounding.
    let binary_count = out_mat.data.iter().filter(|&&v| v == 0 || v == 255).count();
    let total = out_mat.data.len();
    assert!(
        binary_count * 100 / total.max(1) >= 95,
        "expected ≥95% strictly-binary pixels, got {} of {}",
        binary_count,
        total
    );
}

#[test]
fn morphology_ex_open_round_trip() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_gray_png(&dir, "in.png");
    let output = dir.path().join("morph.png");
    cmd()
        .args([
            "morphology-ex",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--op",
            "open",
            "--ksize",
            "3",
            "--shape",
            "rect",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read morph");
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
    assert_eq!(out_mat.channels(), 1);
}

#[test]
fn fast_corners_writes_3channel_visualisation() {
    let dir = TempDir::new().unwrap();
    // FAST needs distinctive features; sprinkle bright dots on a dark canvas.
    let h = 32usize;
    let w = 32usize;
    let mut data = vec![20u8; h * w];
    for &(x, y) in &[(8usize, 8usize), (16, 16), (24, 8), (8, 24), (24, 24)] {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    data[ny as usize * w + nx as usize] = 240;
                }
            }
        }
    }
    let input = dir.path().join("fast_in.png");
    imwrite(&input, &Mat::from_gray_bytes(data, h, w)).expect("write fast input");
    let output = dir.path().join("fast_vis.png");
    cmd()
        .args([
            "fast-corners",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--threshold",
            "10",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read fast vis");
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
    // Visualisation is rendered on a 3-channel canvas.
    assert_eq!(out_mat.channels(), 3);
}

#[test]
fn harris_corners_writes_grayscale_response_map() {
    let dir = TempDir::new().unwrap();
    let input = write_gradient_gray_png(&dir, "in.png");
    let output = dir.path().join("harris.png");
    cmd()
        .args([
            "harris-corners",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--block-size",
            "2",
            "--ksize",
            "3",
            "--k",
            "0.04",
        ])
        .assert()
        .success();
    let out_mat = imread(&output, IMREAD_UNCHANGED).expect("read harris");
    assert_eq!(out_mat.rows, 32);
    assert_eq!(out_mat.cols, 32);
    assert_eq!(out_mat.channels(), 1);
}

// ── Cross-cutting smoke tests ────────────────────────────────────────────────

/// `--list-functions` must mention every new subcommand name (hyphenated form).
/// `dnn-forward` is conditionally compiled but its print_functions row is
/// always present so the user can discover the feature.
#[test]
fn list_functions_includes_new_subcommands() {
    let out = cmd()
        .arg("--list-functions")
        .output()
        .expect("run --list-functions");
    assert!(out.status.success(), "--list-functions failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "flip",
        "rotate",
        "sobel",
        "laplacian",
        "adaptive-threshold",
        "morphology-ex",
        "fast-corners",
        "harris-corners",
        "orb-detect",
        "dnn-forward",
    ] {
        assert!(
            stdout.contains(needle),
            "--list-functions output missing {needle:?} — got: {stdout}"
        );
    }
}

/// Every always-built new subcommand exposes a working `--help`. `dnn-forward`
/// is covered separately under `tests/cv2_dnn.rs` since it requires `--features dnn`.
#[test]
fn every_new_subcommand_help_exits_zero() {
    let names = [
        "flip",
        "rotate",
        "sobel",
        "laplacian",
        "adaptive-threshold",
        "morphology-ex",
        "fast-corners",
        "harris-corners",
        "orb-detect",
    ];
    for name in names {
        cmd().args([name, "--help"]).assert().success();
    }
}
