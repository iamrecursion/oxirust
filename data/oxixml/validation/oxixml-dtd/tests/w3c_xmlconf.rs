//! The DTD-driven half of the W3C XML Conformance Test Suite.
//!
//! The suite is not vendored: point `OXIXML_XML_SUITES_DIR` at a checkout whose
//! root holds the `xmlts*` directory (the date suffix is discovered, never
//! hard-coded) and these tests run; leave it unset and they skip silently.
//!
//! Only the `valid` and `invalid` test types are exercised here — they are the
//! ones that turn on a *validating* processor. The `not-wf` and `error` types
//! belong to the well-formedness parser in `oxixml-xml`.

#![allow(clippy::print_stdout, reason = "the harness reports its own tally")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use oxixml_dtd::{DocumentValidator, ExpansionLimits, FilesystemResolver};
use oxixml_xml::XmlVersion;
use oxixml_xml::events::Event;
use oxixml_xml::reader::Reader;

/// One `<TEST>` entry of a suite catalog.
#[derive(Debug, Clone)]
struct TestCase {
    id: String,
    kind: String,
    /// Path of the document, relative to the `xmlconf` root.
    uri: String,
    recommendation: String,
    version: String,
}

/// Cases this implementation knowingly does not handle, with the reason.
///
/// Every entry is a deliberate, documented divergence rather than a silent
/// failure; the harness fails if one of them starts passing, so the list cannot
/// rot.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[];

/// Locates the `xmlconf` directory under `OXIXML_XML_SUITES_DIR`.
fn xmlconf_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let entries = std::fs::read_dir(&root).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("xmlts") {
            continue;
        }
        let candidate = entry.path().join("xmlconf");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Every catalog file below `root`.
///
/// Two passes run: [`collect_catalogs`] first finds every file whose bytes contain a literal
/// `<TESTCASES` or `<TESTSUITE` document-element start, by scanning rather than hard-coding
/// names. That alone misses two shapes this suite uses to split a vendor's cases across files:
///
/// * a handful of `eduni/*/xmlconf.xml` wrappers are themselves `<TESTSUITE>` shells that pull in
///   their vendor's real `<TESTCASES>` content with a `SYSTEM` general entity declared in their
///   own internal DTD subset (`errata-2e/3e/4e`, `namespaces`, `namespaces/errata-1e`,
///   `xml-1.1`) — found by the scan (it does not distinguish `<TESTCASES` from `<TESTSUITE`), but
///   contributing zero cases on their own, since [`read_catalog`] never expands the entity
///   reference (it only recognizes literal `<TEST>` elements);
/// * the Sun-written fixtures (`sun/sun-valid.xml`, `sun/sun-invalid.xml`, …) are bare `<TEST>`
///   siblings with no enclosing root element of their own at all — meant only to be
///   entity-included into another document's `<TESTCASES>` — so the byte scan never finds them as
///   a catalog in the first place.
///
/// The second pass closes both gaps the same way a real XML processor would: for every catalog
/// already found, [`entity_targets`] resolves the `SYSTEM` entities its own bytes declare, and
/// each resolved target that exists on disk is added as a catalog in its own right (and, in turn,
/// scanned for further entities, in case a target itself declares one — none in this suite do,
/// but nothing here assumes only one level of indirection). [`read_catalog`] needs no change to
/// consume the result: a target's own directory is exactly the `xml:base` its declaring document
/// gives it (this suite always colocates a vendor's fragment files with the directory the
/// including document's `xml:base` names), and `read_catalog` does not require a `<TESTCASES>`
/// wrapper to be present — it records a `<TEST>` under whatever base is on top of its stack
/// whether or not one ever pushed.
fn catalogs(root: &Path) -> Vec<PathBuf> {
    let mut found = BTreeSet::new();
    collect_catalogs(root, 0, &mut found);

    let mut frontier: Vec<PathBuf> = found.iter().cloned().collect();
    while let Some(catalog) = frontier.pop() {
        let Ok(bytes) = std::fs::read(&catalog) else {
            continue;
        };
        let directory = catalog.parent().unwrap_or(Path::new(".")).to_path_buf();
        for target in entity_targets(&bytes, &directory) {
            if target.is_file() && found.insert(target.clone()) {
                frontier.push(target);
            }
        }
    }

    let mut found: Vec<PathBuf> = found.into_iter().collect();
    found.sort();
    found
}

/// Recursive helper for [`catalogs`].
fn collect_catalogs(directory: &Path, depth: usize, out: &mut BTreeSet<PathBuf>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_catalogs(&path, depth + 1, out);
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "xml") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.len() > 4 * 1024 * 1024 {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if bytes.windows(10).any(|window| window == b"<TESTCASES")
            || bytes.windows(10).any(|window| window == b"<TESTSUITE")
        {
            out.insert(path);
        }
    }
}

/// The `SYSTEM` general-entity targets a catalog's internal DTD subset declares —
/// `<!ENTITY name SYSTEM "target">`, single- or double-quoted — resolved against `directory`
/// (the declaring file's own directory, which is also the `xml:base` every such entity is used
/// under in this suite; see [`catalogs`]).
///
/// A plain substring scan rather than a DTD parser: every declaration in this suite's catalogs
/// sits inside a `<!DOCTYPE … [ … ]>` internal subset that precedes any document content, so
/// scanning the whole byte stream for the literal `<!ENTITY` marker cannot false-positive on
/// unrelated `<TEST>` content (none of which contains that text).
fn entity_targets(bytes: &[u8], directory: &Path) -> Vec<PathBuf> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::new();
    let mut rest: &str = text.as_ref();
    while let Some(start) = rest.find("<!ENTITY") {
        rest = &rest[start + "<!ENTITY".len()..];
        let mut fields = rest.split_whitespace();
        let (Some(_name), Some(keyword)) = (fields.next(), fields.next()) else {
            break;
        };
        if keyword != "SYSTEM" {
            continue;
        }
        let Some(quote_at) = rest.find(['"', '\'']) else {
            break;
        };
        let quote = rest.as_bytes()[quote_at];
        let after = &rest[quote_at + 1..];
        let Some(end) = after.find(quote as char) else {
            break;
        };
        out.push(directory.join(&after[..end]));
        rest = &after[end + 1..];
    }
    out
}

/// Reads the `<TEST>` entries of one catalog, resolving `xml:base` and the
/// catalog's own location into a path relative to `root`.
fn read_catalog(root: &Path, catalog: &Path) -> Vec<TestCase> {
    let Ok(bytes) = std::fs::read(catalog) else {
        return Vec::new();
    };
    let Some(text) = decode(&bytes) else {
        return Vec::new();
    };
    let start = catalog
        .parent()
        .and_then(|parent| parent.strip_prefix(root).ok())
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();

    let mut cases = Vec::new();
    let mut bases: Vec<String> = vec![start];
    let mut reader = Reader::from_str(&text);
    while let Ok(event) = reader.read_event() {
        match event {
            Event::Start(element) | Event::Empty(element) => {
                let name = element.name();
                let attributes = attribute_map(&element);
                if name.as_ref() == b"TESTCASES" {
                    let current = bases.last().cloned().unwrap_or_default();
                    let next = match attributes.iter().find(|(key, _)| key == "xml:base") {
                        Some((_, value)) => join(&current, value),
                        None => current,
                    };
                    bases.push(next);
                } else if name.as_ref() == b"TEST" {
                    let base = bases.last().cloned().unwrap_or_default();
                    let get = |key: &str| {
                        attributes
                            .iter()
                            .find(|(name, _)| name == key)
                            .map(|(_, value)| value.clone())
                            .unwrap_or_default()
                    };
                    cases.push(TestCase {
                        id: get("ID"),
                        kind: get("TYPE"),
                        uri: join(&base, &get("URI")),
                        recommendation: get("RECOMMENDATION"),
                        version: get("VERSION"),
                    });
                }
            }
            Event::End(element) if element.name().as_ref() == b"TESTCASES" && bases.len() > 1 => {
                bases.pop();
            }
            Event::Eof => break,
            _ => {}
        }
    }
    cases
}

/// The attributes of an element as decoded `(name, value)` pairs.
fn attribute_map(element: &oxixml_xml::events::BytesStart<'_>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for attribute in element.attributes() {
        let Ok(attribute) = attribute else {
            continue;
        };
        let Ok(key) = core::str::from_utf8(attribute.key.as_ref()) else {
            continue;
        };
        let Ok(value) = attribute.normalized_value(XmlVersion::Explicit1_0) else {
            continue;
        };
        out.push((key.to_owned(), value.into_owned()));
    }
    out
}

/// Joins a relative reference onto a directory-style base.
fn join(base: &str, reference: &str) -> String {
    if base.is_empty() {
        return reference.to_owned();
    }
    if reference.is_empty() {
        return base.to_owned();
    }
    let base = base.trim_end_matches('/');
    let mut parts: Vec<&str> = base.split('/').filter(|part| !part.is_empty()).collect();
    for part in reference.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Decodes a suite file, honouring a byte-order mark.
fn decode(bytes: &[u8]) -> Option<String> {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8(rest.to_vec()).ok();
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(rest, true);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return decode_utf16(rest, false);
    }
    String::from_utf8(bytes.to_vec()).ok()
}

/// Decodes UTF-16 of the given endianness.
fn decode_utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).ok()
}

/// What running one case produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The document validated cleanly.
    Valid,
    /// Validity errors were reported.
    Invalid,
    /// The document or its DTD could not be read at all.
    Fatal,
    /// The case could not be attempted (unreadable or non-UTF-8 file).
    Skipped,
}

/// Runs one case, returning its outcome and a human-readable explanation.
fn run_case(root: &Path, case: &TestCase) -> (Outcome, String) {
    let path = root.join(&case.uri);
    let Ok(bytes) = std::fs::read(&path) else {
        return (Outcome::Skipped, "unreadable".to_owned());
    };
    let Some(text) = decode(&bytes) else {
        return (Outcome::Skipped, "not UTF-8 or UTF-16".to_owned());
    };
    let resolver = FilesystemResolver::new(root);
    let mut validator = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .with_limits(ExpansionLimits::RELAXED);
    match validator.validate_str_with_base(&text, Some(&case.uri)) {
        Ok(report) if report.is_valid() => (Outcome::Valid, String::new()),
        Ok(report) => (
            Outcome::Invalid,
            report
                .errors()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
        ),
        Err(error) => (Outcome::Fatal, error.to_string()),
    }
}

/// Whether a case is one this crate is responsible for.
fn is_interesting(case: &TestCase) -> bool {
    if !matches!(case.kind.as_str(), "valid" | "invalid") {
        return false;
    }
    // XML 1.1 and the Namespaces recommendations are other crates' problem —
    // and an XML 1.1 case is XML 1.1 whichever erratum list it appears on.
    if case.version == "1.1" {
        return false;
    }
    case.recommendation.is_empty() || case.recommendation.starts_with("XML1.0")
}

#[test]
fn entity_targets_resolves_system_declarations_relative_to_the_declaring_directory() {
    let dir = std::env::temp_dir().join(format!(
        "oxixml-dtd-entity-targets-test-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let bytes = br#"<!DOCTYPE TESTSUITE SYSTEM "testcases.dtd" [
        <!ENTITY sun-valid   SYSTEM "sun/sun-valid.xml">
        <!ENTITY sun-invalid SYSTEM 'sun/sun-invalid.xml'>
        <!ENTITY not-system  "inline text, not a SYSTEM target">
    ]>
    <TESTSUITE/>"#;

    let targets = entity_targets(bytes, &dir);

    assert_eq!(
        targets,
        vec![
            dir.join("sun/sun-valid.xml"),
            dir.join("sun/sun-invalid.xml")
        ],
        "both single- and double-quoted SYSTEM entities should resolve, and a non-SYSTEM entity \
         should not"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn w3c_xml_conformance_valid_and_invalid_cases() {
    let Some(root) = xmlconf_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is not set; skipping the W3C conformance suite");
        return;
    };
    let known: BTreeSet<&str> = KNOWN_DIVERGENCES.iter().map(|(id, _)| *id).collect();

    let mut attempted = 0_usize;
    let mut passed = 0_usize;
    let mut skipped = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    let mut unexpected_passes: Vec<String> = Vec::new();
    let mut fatal_invalids: Vec<String> = Vec::new();

    for catalog in catalogs(&root) {
        for case in read_catalog(&root, &catalog) {
            if !is_interesting(&case) {
                continue;
            }
            let (outcome, detail) = run_case(&root, &case);
            if outcome == Outcome::Skipped {
                skipped += 1;
                println!("  skipped: {} [{}]: {detail}", case.id, case.uri);
                continue;
            }
            attempted += 1;
            let expected_valid = case.kind == "valid";
            let ok = if expected_valid {
                outcome == Outcome::Valid
            } else {
                matches!(outcome, Outcome::Invalid | Outcome::Fatal)
            };
            if known.contains(case.id.as_str()) {
                if ok {
                    unexpected_passes.push(case.id.clone());
                }
                continue;
            }
            if ok {
                passed += 1;
                if !expected_valid && outcome == Outcome::Fatal {
                    fatal_invalids.push(case.id.clone());
                }
            } else {
                failures.push(format!(
                    "{} ({}) [{}]: {outcome:?} {detail}",
                    case.id, case.kind, case.uri
                ));
            }
        }
    }

    println!(
        "W3C xmlconf: {passed}/{attempted} DTD cases passed ({skipped} skipped, {} known \
         divergences)",
        KNOWN_DIVERGENCES.len()
    );
    for failure in failures.iter().take(60) {
        println!("  failing: {failure}");
    }
    println!(
        "  {} invalid cases were rejected as fatal rather than merely invalid: {fatal_invalids:?}",
        fatal_invalids.len()
    );

    assert!(
        unexpected_passes.is_empty(),
        "these cases are listed as known divergences but now pass: {unexpected_passes:?}"
    );
    assert!(
        attempted > 500,
        "the suite should contribute hundreds of cases, got {attempted}"
    );
    assert!(
        failures.is_empty(),
        "{} of {attempted} W3C conformance cases failed",
        failures.len()
    );
}
