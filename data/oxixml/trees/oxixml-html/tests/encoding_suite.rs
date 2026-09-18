//! The html5lib-tests encoding conformance harness.
//!
//! html5lib-tests ships an `encoding/` directory of `.dat` files whose blocks
//! pair a byte stream (`#data`) with the encoding an implementation of
//! [HTML § 13.2.3.2](https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding)
//! must arrive at (`#encoding`). They share the `#data` framing of the tree
//! construction suite but assert nothing about the document tree, so they get
//! their own harness rather than being counted as tree construction blocks.
//!
//! Discovery matches the other html5lib harnesses: `OXIXML_HTML5LIB_TESTS_DIR`
//! or `OXIXML_XML_SUITES_DIR`, and a silent skip when neither names a suite.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_encoding::Encoding;
use oxixml_html::sniff_encoding;

/// Resolves one candidate root to the directory holding the encoding `.dat`
/// files.
fn resolve_suite_dir(root: &Path) -> Option<PathBuf> {
    for suffix in [
        Path::new("encoding").to_path_buf(),
        Path::new("html5lib-tests").join("encoding"),
    ] {
        let nested = root.join(suffix);
        if nested.is_dir() {
            return Some(nested);
        }
    }
    None
}

fn suite_dir() -> Option<PathBuf> {
    for variable in ["OXIXML_HTML5LIB_TESTS_DIR", "OXIXML_XML_SUITES_DIR"] {
        let Some(raw) = env::var_os(variable) else {
            continue;
        };
        if let Some(dir) = resolve_suite_dir(Path::new(&raw)) {
            return Some(dir);
        }
    }
    None
}

/// The `.dat` files directly inside `dir`.
///
/// The walk is deliberately not recursive. `encoding/chardet/` holds documents
/// with no declaration at all, whose expected answer comes from the statistical
/// detection HTML § 13.2.3.2 leaves to the user agent as an optional
/// "implementation-defined" step, and `encoding/scripted/` needs a JavaScript
/// engine; both are reported as skips below rather than run.
fn dat_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = BTreeSet::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "dat") {
            files.insert(path);
        }
    }
    files.into_iter().collect()
}

/// One encoding test block: the raw bytes and the expected encoding label.
struct Case {
    data: Vec<u8>,
    label: String,
}

/// Splits an encoding `.dat` file into its blocks.
///
/// The file is read as bytes because the documents under test are deliberately
/// not all UTF-8; only the `#encoding` label is required to be ASCII.
fn parse_dat(bytes: &[u8]) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut data: Option<Vec<u8>> = None;
    let mut in_data = false;
    let mut buffer: Vec<u8> = Vec::new();

    for line in bytes.split(|&byte| byte == b'\n') {
        if line == b"#data" {
            in_data = true;
            buffer.clear();
            continue;
        }
        if line == b"#encoding" {
            // "the test data ... with the final newline removed".
            if !buffer.is_empty() {
                buffer.pop();
            }
            data = Some(std::mem::take(&mut buffer));
            in_data = false;
            continue;
        }
        if in_data {
            buffer.extend_from_slice(line);
            buffer.push(b'\n');
        } else if let Some(data) = data.take() {
            let label = String::from_utf8_lossy(line).trim().to_owned();
            if !label.is_empty() {
                cases.push(Case { data, label });
            }
        }
    }
    cases
}

#[test]
fn encoding_suite() {
    let Some(dir) = suite_dir() else {
        eprintln!(
            "skipping: neither OXIXML_HTML5LIB_TESTS_DIR nor OXIXML_XML_SUITES_DIR \
             resolves to an html5lib encoding suite"
        );
        return;
    };

    let files = dat_files(&dir);
    assert!(!files.is_empty(), "no .dat files under {}", dir.display());

    let mut total = 0_usize;
    let mut passed = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    let mut unsupported = 0_usize;

    for file in &files {
        let Ok(bytes) = fs::read(file) else {
            continue;
        };
        let name = file.file_name().map_or_else(
            || file.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let cases = parse_dat(&bytes);
        let mut file_passed = 0_usize;
        let mut file_total = 0_usize;
        for case in &cases {
            // DOCUMENTED SCOPE GAP, counted as a skip and never as a pass: the
            // block wants an encoding `oxixml_encoding` does not implement.
            // As of this vendored copy of html5lib-tests, every block's
            // `#encoding` label (`euc-jp`, `iso-8859-2`, `utf-8`,
            // `windows-1252`) resolves, so this branch is not currently
            // exercised — it is kept as a safety net against an upstream
            // suite sync adding a block for a label `oxixml-encoding`
            // explicitly leaves unsupported (`gbk`, `gb18030`, `big5`,
            // `euc-kr`, `windows-874`, `iso-8859-11`; see
            // `oxixml_encoding::label`'s own `for_label_unknown_returns_none`
            // test). Sniffing cannot be said to have succeeded or failed when
            // there is no codec to name, and adding codecs belongs to
            // `oxixml-encoding`, not to this crate.
            let Some(expected) = Encoding::for_label(&case.label) else {
                unsupported += 1;
                continue;
            };
            let expected = Some(expected);
            total += 1;
            file_total += 1;
            let actual = sniff_encoding(&case.data, None).encoding;
            if expected == Some(actual) {
                passed += 1;
                file_passed += 1;
            } else if failures.len() < 40 {
                let data = String::from_utf8_lossy(&case.data);
                let head: String = data.chars().take(120).collect();
                failures.push(format!(
                    "{name}: expected {} ({expected:?}), got {actual:?}, for {head:?}",
                    case.label,
                ));
            }
        }
        eprintln!(
            "{name}: {file_passed}/{file_total} passed ({} skipped)",
            cases.len() - file_total
        );
    }

    eprintln!("html5lib encoding: {passed}/{total} passed");
    eprintln!(
        "  skipped {unsupported} case(s): the expected label names an encoding \
         oxixml-encoding does not implement"
    );
    // DOCUMENTED SCOPE GAPS, reported so they are visible rather than silent:
    // `encoding/chardet/` asks for the statistical encoding detection that
    // HTML § 13.2.3.2 step 7 leaves optional and implementation-defined, and
    // `encoding/scripted/` needs a JavaScript engine. Neither is walked.
    eprintln!(
        "  skipped: encoding/chardet/ (optional implementation-defined \
         detection, HTML § 13.2.3.2 step 7) and encoding/scripted/ \
         (needs script execution)"
    );
    for failure in &failures {
        eprintln!("{failure}");
    }

    // Threshold just under the achieved rate — 82/82 of the runnable blocks,
    // i.e. every block whose `#encoding` label `oxixml-encoding` implements —
    // expressed as a ratio so new upstream cases cannot cause a spurious
    // failure while a regression trips immediately.
    let ratio = passed as f64 / total as f64;
    assert!(
        ratio >= 0.999,
        "html5lib encoding pass rate dropped to {passed}/{total}"
    );
}
