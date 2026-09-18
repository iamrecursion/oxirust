//! `.cube` round-trip integration tests for `oximedia-lut`.
//!
//! These tests exercise the string-level `.cube` encode/decode path for both
//! 3-D ([`Lut3d::load_cube`] / [`export::export_3d_lut_cube`] /
//! [`export::parse_cube_3d`]) and 1-D ([`export::export_1d_lut_cube`] /
//! [`export::parse_cube_1d`]) LUTs, plus the file-backed 1-D
//! [`Lut1d::to_file`] / [`Lut1d::from_file`] pair.
//!
//! # Critical `.cube` flattening order
//!
//! `Lut3d::load_cube` and `parse_cube_3d` read entries in **B-outer, G-mid,
//! R-inner** order (`for b { for g { for r { … } } }`).  When flattening a
//! [`Lut3d`] (whose internal storage is R-outer) into the `&[f64]` consumed by
//! `export_3d_lut_cube`, the SAME B-major traversal MUST be used or off-diagonal
//! entries silently fail the per-entry tolerance check.

use oximedia_lut::export::{
    export_1d_lut_cube, export_3d_lut_cube, parse_cube_1d, parse_cube_3d, ExportOptions,
};
use oximedia_lut::{Lut1d, Lut3d, LutSize};

/// Per-channel tolerance for `.cube` string round-trips. The writer formats
/// values with `{:.6}` so any difference must be below 1e-5.
const EPS: f64 = 1e-5;

/// Flatten a [`Lut3d`] into the interleaved RGB `&[f64]` expected by
/// [`export_3d_lut_cube`], using the canonical **B-major** (`b`-outer, `g`-mid,
/// `r`-inner) traversal that matches the `.cube` parsers.
fn flatten_b_major(lut: &Lut3d) -> Vec<f64> {
    let s = lut.size();
    let mut flat = Vec::with_capacity(s * s * s * 3);
    for b in 0..s {
        for g in 0..s {
            for r in 0..s {
                let rgb = lut.get(r, g, b);
                flat.push(rgb[0]);
                flat.push(rgb[1]);
                flat.push(rgb[2]);
            }
        }
    }
    flat
}

#[test]
fn lut3d_identity_17_cube_string_roundtrip() {
    let orig = Lut3d::identity(LutSize::Size17);
    let flat = flatten_b_major(&orig);
    let opts = ExportOptions::default_cube().with_title("identity17");
    let cube = export_3d_lut_cube(&flat, 17, &opts);

    let parsed = Lut3d::load_cube(&cube).expect("load_cube should parse the exported 17^3 cube");
    assert_eq!(parsed.size(), 17, "round-tripped size must be 17");

    for r in 0..17 {
        for g in 0..17 {
            for b in 0..17 {
                let a = orig.get(r, g, b);
                let p = parsed.get(r, g, b);
                for ch in 0..3 {
                    assert!(
                        (p[ch] - a[ch]).abs() < EPS,
                        "identity mismatch at ({r},{g},{b}) ch {ch}: {} vs {}",
                        p[ch],
                        a[ch]
                    );
                }
            }
        }
    }
}

#[test]
fn lut3d_known_transform_17() {
    // Channel-swap transform: output = [b, r, g]. This is strongly off-diagonal,
    // so it only round-trips correctly when the flatten order matches the parser.
    let orig = Lut3d::from_fn(LutSize::Size17, |[r, g, b]| [b, r, g]);
    let flat = flatten_b_major(&orig);
    let opts = ExportOptions::default_cube().with_title("swap_brg17");
    let cube = export_3d_lut_cube(&flat, 17, &opts);

    let parsed = Lut3d::load_cube(&cube).expect("load_cube should parse the channel-swap cube");
    assert_eq!(parsed.size(), 17);

    for r in 0..17 {
        for g in 0..17 {
            for b in 0..17 {
                let a = orig.get(r, g, b);
                let p = parsed.get(r, g, b);
                for ch in 0..3 {
                    assert!(
                        (p[ch] - a[ch]).abs() < EPS,
                        "channel-swap mismatch at ({r},{g},{b}) ch {ch}: {} vs {}",
                        p[ch],
                        a[ch]
                    );
                }
            }
        }
    }
}

#[test]
fn lut3d_boundary_corners_exact() {
    let orig = Lut3d::identity(LutSize::Size33);
    let flat = flatten_b_major(&orig);
    let opts = ExportOptions::default_cube().with_title("identity33");
    let cube = export_3d_lut_cube(&flat, 33, &opts);

    let parsed = Lut3d::load_cube(&cube).expect("load_cube should parse the 33^3 identity cube");
    assert_eq!(parsed.size(), 33);

    // Black corner maps to [0,0,0].
    let black = parsed.get(0, 0, 0);
    for ch in 0..3 {
        assert!(
            black[ch].abs() < EPS,
            "black corner ch {ch} expected 0.0, got {}",
            black[ch]
        );
    }

    // White corner maps to [1,1,1].
    let white = parsed.get(32, 32, 32);
    for ch in 0..3 {
        assert!(
            (white[ch] - 1.0).abs() < EPS,
            "white corner ch {ch} expected 1.0, got {}",
            white[ch]
        );
    }
}

#[test]
fn lut1d_gamma_16_cube_roundtrip() {
    // Use the red channel of a 16-entry gamma-2.2 curve as the 1-D LUT payload.
    // Routed through the raw export::* string fns (NOT Lut1d::to_file/from_file).
    let v = Lut1d::gamma(16, 2.2).r;
    assert_eq!(v.len(), 16);

    let cube = export_1d_lut_cube(&v, "gamma22");
    let parsed = parse_cube_1d(&cube).expect("parse_cube_1d should parse the exported 1D cube");

    assert_eq!(parsed.len(), 16, "round-tripped 1D length must be 16");
    for (i, (a, b)) in v.iter().zip(parsed.iter()).enumerate() {
        assert!((a - b).abs() < EPS, "gamma 1D mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn raw_cube_8cubed_roundtrip() {
    // 8^3 is only reachable through the raw export::* fns because `LutSize` has
    // no `Size8` variant. Build a B-major identity lattice directly.
    let size = 8usize;
    let scale = (size - 1) as f64;
    let mut flat = Vec::with_capacity(size * size * size * 3);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                flat.push(r as f64 / scale);
                flat.push(g as f64 / scale);
                flat.push(b as f64 / scale);
            }
        }
    }
    assert_eq!(flat.len(), 8 * 8 * 8 * 3);

    let opts = ExportOptions::default_cube().with_title("identity8");
    let cube = export_3d_lut_cube(&flat, size, &opts);

    let (parsed, parsed_size) =
        parse_cube_3d(&cube).expect("parse_cube_3d should parse the exported 8^3 cube");
    assert_eq!(parsed_size, 8, "returned size must be 8");
    assert_eq!(parsed.len(), flat.len());

    for (i, (a, b)) in flat.iter().zip(parsed.iter()).enumerate() {
        assert!(
            (a - b).abs() < EPS,
            "8^3 identity mismatch at flat index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn lut1d_to_file_from_file_roundtrip() {
    // Regression guard documenting that the FILE-backed 1-D `.cube` path is
    // self-consistent: `Lut1d::to_file` writes `LUT_1D_SIZE` and
    // `Lut1d::from_file` reads it back. These two methods are a self-contained
    // pair and do NOT route through `formats/cube.rs::parse_cube_file` (which is
    // the `Lut3d` reader). The suspected "1D-asymmetry" bug (writer emits a
    // header the reader rejects) is therefore NOT reproducible on this path —
    // this test pins that fact.
    let original = Lut1d::gamma(17, 2.2);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "oximedia_lut_w28_1d_roundtrip_{}.cube",
        std::process::id()
    ));

    original
        .to_file(&path)
        .expect("Lut1d::to_file should write a 1D .cube in test");
    let loaded = Lut1d::from_file(&path).expect("Lut1d::from_file should read the 1D .cube back");

    assert_eq!(loaded.size(), 17, "round-tripped 1D size must be 17");
    for i in 0..17 {
        assert!(
            (loaded.r[i] - original.r[i]).abs() < EPS,
            "r[{i}] mismatch: {} vs {}",
            loaded.r[i],
            original.r[i]
        );
        assert!(
            (loaded.g[i] - original.g[i]).abs() < EPS,
            "g[{i}] mismatch: {} vs {}",
            loaded.g[i],
            original.g[i]
        );
        assert!(
            (loaded.b[i] - original.b[i]).abs() < EPS,
            "b[{i}] mismatch: {} vs {}",
            loaded.b[i],
            original.b[i]
        );
    }

    let _ = std::fs::remove_file(&path);
}
