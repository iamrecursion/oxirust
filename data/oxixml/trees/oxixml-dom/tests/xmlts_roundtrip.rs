//! Round-trips every *valid* document of the W3C XML Conformance Test Suite
//! through the DOM: parse → serialize → re-parse → structural equality.
//!
//! # How the suite is located
//!
//! The suite root comes from the `OXIXML_XML_SUITES_DIR` environment variable.
//! Inside it, the first directory whose name starts with `xmlts` is used — the
//! upstream distribution embeds a release date (`xmlts20130923`) which is
//! deliberately not hardcoded. When the variable is unset, or the expected
//! layout is missing, the test succeeds silently: the suite is an optional,
//! externally provisioned asset.
//!
//! # Which documents are used
//!
//! Every `*.xml` file that sits under a directory component literally named
//! `valid` (and not under `invalid` or `out`) is considered. Rather than parsing
//! the suite catalog, the test simply attempts to build each one; a document
//! this crate cannot build is *skipped*, not failed, because `oxixml-dom` is a
//! non-validating processor that never retrieves the external DTD subset and
//! reads UTF-8 only:
//!
//! * documents in a non-UTF-8 encoding (the suite contains UTF-16, Shift_JIS,
//!   EUC-JP and ISO-2022-JP files);
//! * documents whose entities or default attribute values are declared in an
//!   external subset, which a non-validating processor is explicitly allowed not
//!   to read ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types));
//! * documents that are well-formed XML 1.0 but not namespace-well-formed (the
//!   suite predates Namespaces in XML for many cases), for instance names with
//!   two colons or an undeclared prefix.
//!
//! What is *asserted* is the property that matters for a document model: for
//! every document that does build, serializing it and parsing the result again
//! must yield a structurally identical tree, and serializing that tree again
//! must be byte-identical — the serializer is a fixed point.
//!
//! To keep the test from silently becoming vacuous, it also asserts that a
//! substantial number of documents were actually exercised.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_dom::build;

/// Environment variable pointing at the directory that holds the W3C XML suites.
const SUITES_DIR_ENV: &str = "OXIXML_XML_SUITES_DIR";

/// The smallest number of documents the suite is expected to yield. The 2013
/// distribution provides well over a thousand; this is a floor that only trips
/// when the discovery logic itself breaks.
const MINIMUM_DOCUMENTS: usize = 200;

fn suite_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os(SUITES_DIR_ENV)?);
    let xmlts = fs::read_dir(&root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with("xmlts"))
        })?;
    let conf = xmlts.join("xmlconf");
    conf.is_dir().then_some(conf)
}

/// Collects every `*.xml` file below a path component named `valid`.
fn collect_valid_documents(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        if path.is_dir() {
            if name == "out" {
                continue;
            }
            collect_valid_documents(&path, out);
        } else if path.extension().and_then(OsStr::to_str) == Some("xml")
            && path
                .components()
                .any(|component| component.as_os_str() == OsStr::new("valid"))
        {
            out.push(path);
        }
    }
}

#[test]
fn xmlts_valid_documents_round_trip_through_the_dom() {
    let Some(conf) = suite_root() else {
        eprintln!("{SUITES_DIR_ENV} is not set or has no xmlts*/xmlconf directory; skipping");
        return;
    };

    let mut documents = Vec::new();
    collect_valid_documents(&conf, &mut documents);
    documents.sort();

    let mut round_tripped = 0_usize;
    let mut skipped_encoding = 0_usize;
    let mut skipped_unbuildable = 0_usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &documents {
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            skipped_encoding += 1;
            continue;
        };
        if declares_foreign_encoding(&text) {
            skipped_encoding += 1;
            continue;
        }

        let Ok(original) = build::parse_str(&text) else {
            skipped_unbuildable += 1;
            continue;
        };

        let serialized = original.to_xml();
        match build::parse_str(&serialized) {
            Ok(reparsed) => {
                if let Some(difference) = original.structural_diff(&reparsed) {
                    failures.push(format!("{}: {difference}", path.display()));
                    continue;
                }
                let again = reparsed.to_xml();
                if again != serialized {
                    failures.push(format!(
                        "{}: serialization is not a fixed point\n  first:  {serialized}\n  second: {again}",
                        path.display()
                    ));
                    continue;
                }
                round_tripped += 1;
            }
            Err(error) => failures.push(format!(
                "{}: re-parsing the serialized form failed: {error}\n  {serialized}",
                path.display()
            )),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} documents failed to round-trip:\n{}",
        failures.len(),
        documents.len(),
        failures.join("\n")
    );
    assert!(
        round_tripped >= MINIMUM_DOCUMENTS,
        "only {round_tripped} documents were round-tripped (of {} found, {skipped_encoding} \
         skipped for their encoding and {skipped_unbuildable} not buildable by a \
         non-validating UTF-8 processor); the discovery logic is probably broken",
        documents.len()
    );
    eprintln!(
        "xmlts: {round_tripped} documents round-tripped, {skipped_encoding} skipped for their \
         encoding, {skipped_unbuildable} not buildable by a non-validating processor"
    );
}

/// Returns `true` when the XML declaration names an encoding that is neither
/// UTF-8 nor US-ASCII, both of which this crate reads natively.
fn declares_foreign_encoding(text: &str) -> bool {
    let Some(rest) = text.strip_prefix("<?xml") else {
        return false;
    };
    let Some(end) = rest.find("?>") else {
        return false;
    };
    let declaration = &rest[..end];
    let Some(position) = declaration.find("encoding") else {
        return false;
    };
    let value: String = declaration[position..]
        .chars()
        .skip_while(|&ch| ch != '"' && ch != '\'')
        .skip(1)
        .take_while(|&ch| ch != '"' && ch != '\'')
        .collect();
    !(value.eq_ignore_ascii_case("utf-8")
        || value.eq_ignore_ascii_case("us-ascii")
        || value.eq_ignore_ascii_case("ascii")
        || value.is_empty())
}
