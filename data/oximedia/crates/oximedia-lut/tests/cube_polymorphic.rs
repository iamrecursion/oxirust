//! Integration tests for the polymorphic `.cube` loader (`parse_cube_any` / `CubeLut`).
//!
//! These pin the Wave 29 additive loader that closes the latent gap where
//! `Lut3d::from_file` could not load a 1D `.cube` file: `parse_cube_any` inspects
//! the header and dispatches to either [`oximedia_lut::Lut1d`] or
//! [`oximedia_lut::Lut3d`], leaving the existing `parse_cube_file -> Lut3d`
//! contract untouched.

use oximedia_lut::{parse_cube_any, CubeLut, Lut1d, Lut3d, LutError};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Build a process- and call-unique temp path so parallel test runs never collide.
fn unique_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "oximedia_lut_polycube_{tag}_{pid}_{nanos}_{seq}.cube"
    ));
    p
}

/// Write `contents` to `path`, panicking with context on failure (tests only).
fn write_fixture(path: &PathBuf, contents: &str) {
    let mut f = std::fs::File::create(path).expect("create temp .cube fixture");
    f.write_all(contents.as_bytes())
        .expect("write temp .cube fixture");
}

const FIXTURE_1D: &str = "LUT_1D_SIZE 4\n0 0 0\n0.25 0.25 0.25\n0.5 0.5 0.5\n1 1 1\n";

const FIXTURE_3D: &str = "LUT_3D_SIZE 2\n\
0 0 0\n1 0 0\n0 1 0\n1 1 0\n\
0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

const FIXTURE_HEADERLESS: &str = "# a comment\n0 0 0\n0.5 0.5 0.5\n1 1 1\n";

#[test]
fn parse_cube_any_loads_1d_variant() {
    let path = unique_path("p1d");
    write_fixture(&path, FIXTURE_1D);

    let loaded = parse_cube_any(&path).expect("parse_cube_any should accept a 1D .cube");
    assert!(loaded.is_1d(), "header was LUT_1D_SIZE -> must be 1D");
    assert!(!loaded.is_3d());

    let lut = loaded.as_1d().expect("as_1d should return the 1D LUT");
    let expected = [0.0_f64, 0.25, 0.5, 1.0];
    assert_eq!(lut.r.len(), expected.len(), "row count must match size");
    for (got, want) in lut.r.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }
    // as_3d must yield None for a 1D LUT.
    assert!(loaded.as_3d().is_none());

    // Oracle: identical to a direct Lut1d::from_file on the same bytes.
    let direct = Lut1d::from_file(&path).expect("direct Lut1d::from_file");
    assert_eq!(lut.r.len(), direct.r.len());
    for (got, want) in lut.r.iter().zip(direct.r.iter()) {
        assert!((got - want).abs() < 1e-12);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn parse_cube_any_loads_3d_variant() {
    let path = unique_path("p3d");
    write_fixture(&path, FIXTURE_3D);

    let loaded = parse_cube_any(&path).expect("parse_cube_any should accept a 3D .cube");
    assert!(loaded.is_3d(), "header was LUT_3D_SIZE -> must be 3D");
    assert!(!loaded.is_1d());
    assert!(loaded.as_1d().is_none());

    let lut = loaded.as_3d().expect("as_3d should return the 3D LUT");
    // B-outer/G-mid/R-inner fill order corners.
    let c000 = lut.get(0, 0, 0);
    assert!(c000.iter().all(|&v| v.abs() < 1e-12), "origin is black");
    assert!((lut.get(1, 0, 0)[0] - 1.0).abs() < 1e-12, "r-axis max red");
    assert!((lut.get(0, 0, 1)[2] - 1.0).abs() < 1e-12, "b-axis max blue");
    let c111 = lut.get(1, 1, 1);
    for ch in c111 {
        assert!((ch - 1.0).abs() < 1e-12, "far corner is white");
    }

    // Oracle: identical to a direct Lut3d::from_file on the same bytes.
    let direct = Lut3d::from_file(&path).expect("direct Lut3d::from_file");
    assert_eq!(lut.size(), direct.size());
    for b in 0..lut.size() {
        for g in 0..lut.size() {
            for r in 0..lut.size() {
                let a = lut.get(r, g, b);
                let d = direct.get(r, g, b);
                for k in 0..3 {
                    assert!((a[k] - d[k]).abs() < 1e-12);
                }
            }
        }
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn parse_cube_any_rejects_headerless() {
    let path = unique_path("hl");
    write_fixture(&path, FIXTURE_HEADERLESS);

    let err = parse_cube_any(&path).expect_err("headerless .cube must be rejected");
    assert!(
        matches!(err, LutError::UnsupportedFormat(_)),
        "expected UnsupportedFormat, got {err:?}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cube_lut_from_file_dispatches_1d() {
    let path = unique_path("ff1d");
    write_fixture(&path, FIXTURE_1D);

    let loaded = CubeLut::from_file(&path).expect("CubeLut::from_file on 1D fixture");
    assert!(loaded.is_1d());

    let _ = std::fs::remove_file(&path);
}
