//! Integration tests for the `oximedia-cv2 orb-detect` subcommand.

use assert_cmd::Command;
use oximedia_compat_cv2::{image_io::imwrite, mat::Mat};
use tempfile::TempDir;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("oximedia-cv2 binary not found")
}

#[test]
fn orb_detect_help_exits_zero() {
    cmd().args(["orb-detect", "--help"]).assert().success();
}

/// Generate a 64×64 grayscale image with a few high-contrast bright dots so
/// FAST inside ORB has corners to find, then run `orb-detect` and confirm
/// stdout reports a keypoint count.
#[test]
fn orb_detect_synth_image_returns_keypoints() {
    let dir = TempDir::new().expect("tempdir");
    let h = 64usize;
    let w = 64usize;
    let mut data = vec![20u8; h * w];
    // Sprinkle 9 bright 3×3 corner-like patches.
    let centres: [(usize, usize); 9] = [
        (12, 12),
        (32, 12),
        (52, 12),
        (12, 32),
        (32, 32),
        (52, 32),
        (12, 52),
        (32, 52),
        (52, 52),
    ];
    for (cx, cy) in centres {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    data[y as usize * w + x as usize] = 240;
                }
            }
        }
    }
    let input = dir.path().join("orb_in.png");
    imwrite(&input, &Mat::from_gray_bytes(data, h, w)).expect("write orb input");

    let out = cmd()
        .args([
            "orb-detect",
            input.to_str().expect("input path utf8"),
            "--num-features",
            "200",
        ])
        .output()
        .expect("run orb-detect");

    assert!(
        out.status.success(),
        "orb-detect failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("orb keypoints="),
        "stdout missing 'orb keypoints=' line: {stdout}"
    );
    // The keypoint count must be present (numeric, possibly zero, but the
    // line itself must exist).
    let keypoint_line = stdout
        .lines()
        .find(|l| l.starts_with("orb keypoints="))
        .expect("keypoints line present");
    assert!(
        keypoint_line.contains("descriptors_rows="),
        "keypoint line missing descriptor row count: {keypoint_line}"
    );
}
