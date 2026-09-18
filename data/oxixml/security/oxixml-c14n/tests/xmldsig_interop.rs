//! A smoke test over the W3C XML Signature 1.1 interoperability documents.
//!
//! Those 45 documents are real signatures produced by an independent
//! implementation, and every one of them names a canonicalization algorithm in
//! `<ds:SignedInfo><ds:CanonicalizationMethod Algorithm="…">`. Verifying the
//! signature value is `oxixml-dsig`'s job, but the documents are still the best
//! source of realistic input for this crate, so each `<ds:SignedInfo>` is
//! canonicalized here with the algorithm it names and the two properties that
//! do not need a key are checked:
//!
//! * the output re-parses as well-formed XML, and
//! * canonicalizing it again is a no-op, which is what
//!   [Canonical XML §2.4](https://www.w3.org/TR/xml-c14n#DocSubsets) promises
//!   for a canonical form that is itself well-formed.
//!
//! The suite is discovered through `OXIXML_XML_SUITES_DIR` and skipped silently
//! when that is not set.

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_c14n::{CanonicalizationAlgorithm, NodeSet};
use oxixml_dom::{Document, NodeId, build};

const DSIG_NAMESPACE: &str = "http://www.w3.org/2000/09/xmldsig#";

fn interop_directory() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let directory = root.join("xmldsig-core1-interop");
    directory.is_dir().then_some(directory)
}

/// Collects every `*.xml` file below `directory`.
fn signature_files(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "xml") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Returns the first `<ds:SignedInfo>` element and the algorithm its
/// `<ds:CanonicalizationMethod>` child names.
fn signed_info(document: &Document) -> Option<(NodeId, CanonicalizationAlgorithm)> {
    let signed_info = document
        .get_elements_by_tag_name_ns(document.root(), Some(DSIG_NAMESPACE), "SignedInfo")
        .next()?;
    let method = document
        .get_elements_by_tag_name_ns(signed_info, Some(DSIG_NAMESPACE), "CanonicalizationMethod")
        .next()?;
    let uri = document.get_attribute(method, "Algorithm")?;
    CanonicalizationAlgorithm::from_uri(uri)
        .ok()
        .map(|algorithm| (signed_info, algorithm))
}

#[test]
fn signed_info_canonicalizes_and_is_stable() {
    let Some(directory) = interop_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xmldsig-core1-interop");
        return;
    };

    let files = signature_files(&directory);
    let mut checked = 0_usize;
    let mut skipped = 0_usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Ok(text) = fs::read_to_string(path) else {
            skipped += 1;
            continue;
        };
        let Ok(document) = build::parse_str(&text) else {
            skipped += 1;
            continue;
        };
        let Some((element, algorithm)) = signed_info(&document) else {
            skipped += 1;
            continue;
        };

        let subset = NodeSet::subtree(&document, element);
        let once = match algorithm.apply(&document, &subset) {
            Ok(bytes) => bytes,
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        let Ok(rendered) = String::from_utf8(once.clone()) else {
            failures.push(format!("{name}: the canonical form is not UTF-8"));
            continue;
        };
        let reparsed = match build::parse_str(&rendered) {
            Ok(document) => document,
            Err(error) => {
                failures.push(format!(
                    "{name}: the canonical form did not re-parse: {error}"
                ));
                continue;
            }
        };
        let twice = match algorithm.apply(&reparsed, &NodeSet::document(&reparsed)) {
            Ok(bytes) => bytes,
            Err(error) => {
                failures.push(format!("{name}: second pass failed: {error}"));
                continue;
            }
        };
        if once == twice {
            checked += 1;
        } else {
            failures.push(format!(
                "{name}: not idempotent\n  first:  {rendered}\n  second: {}",
                String::from_utf8_lossy(&twice)
            ));
        }
    }

    println!(
        "XML Signature 1.1 interop: {checked} SignedInfo elements canonicalized, \
         {} failed, {skipped} skipped (of {} files)",
        failures.len(),
        files.len(),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        checked > 0,
        "the interop directory was found but held no signatures"
    );
}
