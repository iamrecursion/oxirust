//! Cross-check of the binary DE440 header against the public JPL ASCII
//! header `header.440` (GROUP 1030/1040/1041/1050 format, as read by the
//! public-domain JPL `asc2eph.f`).
//!
//! Skips gracefully when `data/ref/header.440` or the DE440 binary is
//! absent.

use std::error::Error;
use std::path::{Path, PathBuf};

use oxiephemeris_de::{DeFile, Series};

type TestResult = Result<(), Box<dyn Error>>;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

/// Parsed subset of the ASCII header.
struct AsciiHeader {
    ksize: usize,
    ncoeff: usize,
    ss: [f64; 3],
    names: Vec<String>,
    values: Vec<f64>,
    /// 15 pointer columns x (start, ncf, na).
    pointers: Vec<[usize; 3]>,
}

/// Lines of one GROUP section (between its header line and the next GROUP).
fn group_lines<'x>(text: &'x str, group: &str) -> Result<Vec<&'x str>, String> {
    let mut lines = text.lines();
    for line in lines.by_ref() {
        if line.trim() == group {
            break;
        }
    }
    let mut out = Vec::new();
    for line in lines {
        if line.trim().starts_with("GROUP") {
            return Ok(out);
        }
        if !line.trim().is_empty() {
            out.push(line);
        }
    }
    if out.is_empty() {
        Err(format!("ASCII header: section {group} not found"))
    } else {
        Ok(out)
    }
}

fn tokens_of(lines: &[&str]) -> Vec<String> {
    lines
        .iter()
        .flat_map(|l| l.split_whitespace())
        .map(str::to_string)
        .collect()
}

/// Parses a Fortran double literal (`0.14959…D+09`).
fn fortran_f64(token: &str) -> Result<f64, String> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|e| format!("bad Fortran float {token}: {e}"))
}

fn parse_ascii_header(text: &str) -> Result<AsciiHeader, String> {
    // First line: "KSIZE= 2036    NCOEFF= 1018".
    let first = text.lines().next().ok_or("empty ASCII header")?;
    let nums: Vec<usize> = first
        .replace('=', " ")
        .split_whitespace()
        .filter_map(|t| t.parse::<usize>().ok())
        .collect();
    let ksize = *nums.first().ok_or("no KSIZE on line 1")?;
    let ncoeff = *nums.get(1).ok_or("no NCOEFF on line 1")?;

    let g1030 = tokens_of(&group_lines(text, "GROUP   1030")?);
    if g1030.len() != 3 {
        return Err(format!(
            "GROUP 1030: expected 3 numbers, got {}",
            g1030.len()
        ));
    }
    let mut ss = [0.0_f64; 3];
    for (slot, tok) in ss.iter_mut().zip(&g1030) {
        *slot = fortran_f64(tok)?;
    }

    let g1040 = tokens_of(&group_lines(text, "GROUP   1040")?);
    let n_names: usize = g1040
        .first()
        .ok_or("GROUP 1040 empty")?
        .parse()
        .map_err(|e| format!("GROUP 1040 count: {e}"))?;
    let names: Vec<String> = g1040.iter().skip(1).cloned().collect();
    if names.len() != n_names {
        return Err(format!(
            "GROUP 1040: expected {n_names} names, got {}",
            names.len()
        ));
    }

    let g1041 = tokens_of(&group_lines(text, "GROUP   1041")?);
    let n_values: usize = g1041
        .first()
        .ok_or("GROUP 1041 empty")?
        .parse()
        .map_err(|e| format!("GROUP 1041 count: {e}"))?;
    let values: Vec<f64> = g1041
        .iter()
        .skip(1)
        .map(|t| fortran_f64(t))
        .collect::<Result<_, _>>()?;
    if values.len() != n_values {
        return Err(format!(
            "GROUP 1041: expected {n_values} values, got {}",
            values.len()
        ));
    }

    // GROUP 1050: three rows (start / ncf / na), 15 columns each
    // (12 IPT + LPT + RPT + TPT).
    let g1050 = group_lines(text, "GROUP   1050")?;
    let rows: Vec<Vec<usize>> = g1050
        .iter()
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<usize>().map_err(|e| format!("GROUP 1050: {e}")))
                .collect()
        })
        .collect::<Result<_, _>>()?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 15) {
        return Err("GROUP 1050: expected 3 rows x 15 columns".to_string());
    }
    let mut pointers = Vec::with_capacity(15);
    for col in 0..15 {
        let mut triple = [0_usize; 3];
        for (slot, row) in triple.iter_mut().zip(&rows) {
            *slot = *row.get(col).ok_or("GROUP 1050: short row")?;
        }
        pointers.push(triple);
    }

    Ok(AsciiHeader {
        ksize,
        ncoeff,
        ss,
        names,
        values,
        pointers,
    })
}

fn assert_close(a: f64, b: f64, rel: f64, what: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        (a - b).abs() <= rel * scale,
        "{what}: binary {a:e} vs ASCII {b:e}"
    );
}

#[test]
fn binary_header_matches_ascii_header() -> TestResult {
    let ascii_path = data_dir().join("ref").join("header.440");
    let bin_path = data_dir().join("de440").join("linux_p1550p2650.440");
    if !ascii_path.exists() {
        println!("skipping: data/ref/header.440 not present");
        return Ok(());
    }
    if !bin_path.exists()
        || data_dir()
            .join("de440")
            .join("linux_p1550p2650.440.aria2")
            .exists()
    {
        println!("skipping: DE440 binary not present/complete");
        return Ok(());
    }
    let ascii = parse_ascii_header(&std::fs::read_to_string(&ascii_path)?)?;
    let bytes = std::fs::read(&bin_path)?;
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE440: {e}"))?;

    // Record shape.
    assert_eq!(de.ncoeff(), ascii.ncoeff, "NCOEFF");
    assert_eq!(2 * de.ncoeff(), ascii.ksize, "KSIZE = 2*NCOEFF");

    // SS: start, stop, step (exact half-integers / small integers).
    let (start, stop) = de.span();
    assert_close(start, ascii.ss[0], 0.0, "SS(1)");
    assert_close(stop, ascii.ss[1], 0.0, "SS(2)");
    assert_close(de.step_days(), ascii.ss[2], 0.0, "SS(3)");

    // NCON and the full constant-name table, including the > 400 tail.
    assert_eq!(de.constant_count(), ascii.names.len(), "NCON");
    assert!(ascii.names.len() > 400, "DE440 exercises NCON > 400");
    for (i, name) in ascii.names.iter().enumerate() {
        assert_eq!(
            de.constant_name(i),
            Some(name.as_str()),
            "constant name {i}"
        );
    }

    // AU / EMRAT come from the header fields; DENUM from NUMDE.
    let idx_of = |name: &str| -> Result<usize, String> {
        ascii
            .names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("ASCII header lacks {name}"))
    };
    let ascii_val = |name: &str| -> Result<f64, String> {
        ascii
            .values
            .get(idx_of(name)?)
            .copied()
            .ok_or_else(|| format!("ASCII header lacks a value for {name}"))
    };
    assert_close(de.au_km(), ascii_val("AU")?, 1e-15, "AU");
    assert_close(de.emrat(), ascii_val("EMRAT")?, 1e-15, "EMRAT");
    assert_close(f64::from(de.numde()), ascii_val("DENUM")?, 0.0, "NUMDE");

    // Spot-check constant values by name against GROUP 1041.
    for name in [
        "AU", "EMRAT", "DENUM", "CLIGHT", "JDEPOC", "GM1", "GMB", "GMS",
    ] {
        let bin = de
            .constant(name)
            .ok_or_else(|| format!("binary header lacks constant {name}"))?;
        assert_close(bin, ascii_val(name)?, 1e-15, name);
    }

    // Pointer table: 12 IPT columns + LPT + RPT + TPT vs GROUP 1050.
    for (series, ascii_col) in Series::ALL.iter().zip(&ascii.pointers) {
        let expected = (*ascii_col)[1] > 0 && (*ascii_col)[2] > 0;
        match de.pointer_triplet(*series) {
            Some((s, ncf, na)) => {
                assert!(expected, "{series:?} present in binary, absent in ASCII");
                assert_eq!([s, ncf, na], *ascii_col, "pointer triplet {series:?}");
            }
            None => assert!(!expected, "{series:?} absent in binary, present in ASCII"),
        }
    }
    println!(
        "DE440 binary header matches header.440 ({} constants, NCOEFF {})",
        de.constant_count(),
        de.ncoeff()
    );
    Ok(())
}
