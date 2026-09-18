//! Unicode normalization conformance tests.
//!
//! The default run validates an embedded every-10th-line subset of the official
//! `NormalizationTest.txt` (Part 0, 1, and 2) generated from the same UCD
//! version the crate's tables were built from. The full file can be run by
//! pointing the `OXIXML_UCD_NORMTEST` environment variable at a local copy of
//! `NormalizationTest.txt`; the test is skipped silently when the variable is
//! unset.

use oxixml_unicode::norm::{nfc, nfd, nfkc, nfkd};

// Embedded fixture: `NORMALIZATION_CASES: &[(&str, &str, &str, &str, &str)]`.
include!("ucd/normalization_cases.rs");

/// Asserts the five NFC/NFD/NFKC/NFKD conformance invariants from the
/// `NormalizationTest.txt` header for one `(c1, c2, c3, c4, c5)` row.
fn check_row(row: usize, c1: &str, c2: &str, c3: &str, c4: &str, c5: &str) {
    // NFC:  c2 == toNFC(c1) == toNFC(c2) == toNFC(c3); c4 == toNFC(c4) == toNFC(c5)
    assert_eq!(nfc(c1), c2, "NFC(c1) row {row}");
    assert_eq!(nfc(c2), c2, "NFC(c2) row {row}");
    assert_eq!(nfc(c3), c2, "NFC(c3) row {row}");
    assert_eq!(nfc(c4), c4, "NFC(c4) row {row}");
    assert_eq!(nfc(c5), c4, "NFC(c5) row {row}");

    // NFD:  c3 == toNFD(c1) == toNFD(c2) == toNFD(c3); c5 == toNFD(c4) == toNFD(c5)
    assert_eq!(nfd(c1), c3, "NFD(c1) row {row}");
    assert_eq!(nfd(c2), c3, "NFD(c2) row {row}");
    assert_eq!(nfd(c3), c3, "NFD(c3) row {row}");
    assert_eq!(nfd(c4), c5, "NFD(c4) row {row}");
    assert_eq!(nfd(c5), c5, "NFD(c5) row {row}");

    // NFKC: c4 == toNFKC(c1..c5)
    for (col, s) in [c1, c2, c3, c4, c5].iter().enumerate() {
        assert_eq!(nfkc(s), c4, "NFKC(c{}) row {row}", col + 1);
    }

    // NFKD: c5 == toNFKD(c1..c5)
    for (col, s) in [c1, c2, c3, c4, c5].iter().enumerate() {
        assert_eq!(nfkd(s), c5, "NFKD(c{}) row {row}", col + 1);
    }
}

#[test]
fn embedded_subset_conformance() {
    for (row, &(c1, c2, c3, c4, c5)) in NORMALIZATION_CASES.iter().enumerate() {
        check_row(row, c1, c2, c3, c4, c5);
    }
}

/// Decodes a `NormalizationTest.txt` column ("0041 0301") into a `String`,
/// returning `None` on any malformed code point.
fn decode_column(field: &str) -> Option<String> {
    let mut out = String::new();
    for token in field.split_whitespace() {
        let cp = u32::from_str_radix(token, 16).ok()?;
        out.push(char::from_u32(cp)?);
    }
    Some(out)
}

#[test]
fn full_file_conformance_when_env_set() {
    let path = match std::env::var_os("OXIXML_UCD_NORMTEST") {
        Some(p) => p,
        None => return, // skipped silently
    };
    let text = std::fs::read_to_string(&path)
        .expect("OXIXML_UCD_NORMTEST must point at a readable NormalizationTest.txt");

    let mut row = 0usize;
    for line in text.lines() {
        if line.starts_with('@') || line.starts_with('#') {
            continue;
        }
        let data = match line.split('#').next() {
            Some(d) => d.trim(),
            None => continue,
        };
        if data.is_empty() {
            continue;
        }
        let cols: Vec<&str> = data.split(';').collect();
        if cols.len() < 5 {
            continue;
        }
        let decoded: Option<Vec<String>> = cols[..5].iter().map(|c| decode_column(c)).collect();
        let decoded = decoded.expect("valid code points in NormalizationTest.txt");
        check_row(
            row,
            &decoded[0],
            &decoded[1],
            &decoded[2],
            &decoded[3],
            &decoded[4],
        );
        row += 1;
    }
    assert!(
        row > 0,
        "the NormalizationTest.txt file contained no data rows"
    );
}
