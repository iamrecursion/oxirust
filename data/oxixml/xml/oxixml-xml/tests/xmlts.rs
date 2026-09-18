//! W3C XML Conformance Test Suite (`xmlts`) gate for [`oxixml_xml`].
//!
//! # How the suite is located
//!
//! The suite root is taken from the `OXIXML_XML_SUITES_DIR` environment variable.
//! Inside it, the first directory whose name starts with `xmlts` is used (the
//! upstream distribution embeds a release date, e.g. `xmlts20130923`, which is
//! deliberately *not* hardcoded here). The catalog entry point is
//! `<xmlts*>/xmlconf/xmlconf.xml`.
//!
//! When the variable is unset, or when the expected layout is missing, the test
//! silently succeeds — the suite is an optional, externally provisioned asset.
//!
//! The catalog is parsed with this crate's own [`Reader`], including the manual
//! resolution of the `SYSTEM` general entities declared in the internal DTD
//! subset of `xmlconf.xml` (each of which pulls in one vendor sub-catalog) and
//! of the `xml:base` attributes on the `TESTCASES` elements.
//!
//! # Skip policy
//!
//! `oxixml-xml` is a **non-validating** parser that never retrieves external
//! entities, so only a subset of the suite is applicable. A test case is skipped
//! when any of the following holds:
//!
//! * `TYPE="error"` — the specification leaves the reporting of these optional.
//! * `ENTITIES` is not `none` — the document depends on external parameter or
//!   general entities. Section 5.1 of XML 1.0 explicitly permits a non-validating
//!   parser that does not read them to accept such documents, so neither the
//!   "must fail" nor the "must succeed" expectation can be asserted.
//! * `RECOMMENDATION` is one of the errata revisions (`XML1.0-errata2e`,
//!   `XML1.0-errata3e`, `XML1.0-errata4e`, `NS1.0-errata1e`) — these re-classify
//!   earlier test cases and target a specific edition of the specification.
//! * `EDITION` excludes the 5th edition of XML 1.0 (the edition this crate
//!   implements). XML 1.1 and Namespaces in XML 1.1 cases *are* run: this crate
//!   implements both.
//! * The document uses a character encoding that this crate's byte-oriented
//!   lexer cannot handle. Like `quick-xml` 0.41, `oxixml-xml` scans for ASCII
//!   delimiters in the raw byte stream, so only ASCII-compatible encodings work;
//!   UTF-16 documents (recognized by their BOM or by the `<\0?\0` / `\0<\0?`
//!   signature) and the legacy Japanese `Shift_JIS`, `EUC-JP` and `ISO-2022-JP`
//!   encodings are therefore out of scope. These are transcoding gaps, not
//!   parser conformance gaps.
//! * The referenced document file is absent from the distribution.
//!
//! # Assertions for the applicable cases
//!
//! * `TYPE="not-wf"` must produce an [`Error`](oxixml_xml::Error).
//! * `TYPE="valid"` and `TYPE="invalid"` must parse to `Event::Eof` without an
//!   error — a non-validating parser accepts "invalid" documents because the
//!   violated constraints are validity constraints, not well-formedness ones.
//!
//! # Allowlist
//!
//! A handful of `not-wf` cases are knowingly accepted and are recorded in
//! [`NOT_WF_ALLOWLIST`] together with the reason. Two reasons occur:
//!
//! * **`quick-xml` compatibility.** `oxixml-xml` is a drop-in replacement for
//!   `quick-xml` 0.41 and inherits its deliberately permissive lexer; where
//!   upstream `quick-xml` accepts a document, compatibility wins over conformance.
//! * **Non-validating processor.** Detecting the violation would require applying
//!   the declared attribute types from the DTD, which only a validating processor
//!   does.
//!
//! The allowlist is checked in both directions: an allowlisted case that *is*
//! rejected is reported as a stale entry, so the list cannot rot.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_xml::XmlVersion;
use oxixml_xml::errors::Error;
use oxixml_xml::events::Event;
use oxixml_xml::name::QName;
use oxixml_xml::reader::{NsReader, Reader};

/// Environment variable pointing at the directory that holds the W3C XML suites.
const SUITES_DIR_ENV: &str = "OXIXML_XML_SUITES_DIR";

/// Cases that this parser knowingly accepts even though the W3C suite marks them
/// as not well-formed. See the module documentation for the two admissible
/// reasons.
///
/// Each entry is `(test id, reason)`.
const NOT_WF_ALLOWLIST: &[(&str, &str)] = &[
    (
        "rmt-ns10-012",
        "the two `xmlns` values differ literally and only become equal after the \
         attribute-value normalization that the declared `NMTOKEN` type mandates; \
         applying declared attribute types requires a validating processor",
    ),
    (
        "ibm-1-1-not-wf-P77-ibm77n13.xml",
        "the violated constraint (section 4.3.4) is about the `VersionNum` of an \
         external entity, which a processor that never retrieves external entities \
         cannot see",
    ),
    (
        "ibm-1-1-not-wf-P77-ibm77n14.xml",
        "the violated constraint (section 4.3.4) is about the `VersionNum` of an \
         external entity, which a processor that never retrieves external entities \
         cannot see",
    ),
    (
        "ibm-1-1-not-wf-P77-ibm77n15.xml",
        "the violated constraint (section 4.3.4) is about the `VersionNum` of an \
         external entity, which a processor that never retrieves external entities \
         cannot see",
    ),
];

/// A single `TEST` element of the catalog.
#[derive(Debug, Clone)]
struct TestCase {
    /// Value of the `ID` attribute; unique across the whole suite.
    id: String,
    /// Value of the `TYPE` attribute: `valid`, `invalid`, `not-wf` or `error`.
    ty: String,
    /// Value of the `ENTITIES` attribute; defaults to `none`.
    entities: String,
    /// Value of the `RECOMMENDATION` attribute; defaults to `XML1.0`.
    recommendation: String,
    /// Value of the `EDITION` attribute, if present.
    edition: Option<String>,
    /// Value of the `NAMESPACE` attribute; defaults to `yes`.
    namespace: String,
    /// Value of the `SECTIONS` attribute: the specification sections covered.
    sections: String,
    /// Absolute path of the document under test.
    path: PathBuf,
}

/// Why an applicable-looking case was not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SkipReason {
    /// `TYPE="error"`: reporting is optional.
    OptionalError,
    /// The document needs external entities.
    ExternalEntities,
    /// The case targets an errata revision of the specification.
    Errata,
    /// The case targets XML 1.1 or a non-5th edition of XML 1.0.
    VersionOrEdition,
    /// The document uses an encoding `oxixml-encoding` does not implement.
    UnsupportedEncoding,
    /// The document file is not part of the distribution.
    MissingFile,
}

impl SkipReason {
    /// Human readable label used in the summary table.
    const fn label(self) -> &'static str {
        match self {
            Self::OptionalError => "TYPE=error (optional)",
            Self::ExternalEntities => "external entities",
            Self::Errata => "errata recommendation",
            Self::VersionOrEdition => "XML version / edition",
            Self::UnsupportedEncoding => "unsupported encoding",
            Self::MissingFile => "document file missing",
        }
    }
}

/// Locates `<OXIXML_XML_SUITES_DIR>/xmlts*/xmlconf/xmlconf.xml`.
fn find_catalog() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os(SUITES_DIR_ENV)?);
    let mut candidates: Vec<PathBuf> = fs::read_dir(&root)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with("xmlts"))
        })
        .collect();
    candidates.sort();
    // Prefer the newest distribution when several are provisioned side by side.
    let catalog = candidates.pop()?.join("xmlconf").join("xmlconf.xml");
    catalog.is_file().then_some(catalog)
}

/// Extracts the `<!ENTITY name SYSTEM "file">` declarations of a DOCTYPE
/// internal subset.
fn parse_entity_declarations(doctype: &str) -> BTreeMap<String, String> {
    let mut entities = BTreeMap::new();
    let mut rest = doctype;
    while let Some(start) = rest.find("<!ENTITY") {
        rest = &rest[start + "<!ENTITY".len()..];
        let mut fields = rest.split_whitespace();
        let (Some(name), Some(kind)) = (fields.next(), fields.next()) else {
            break;
        };
        if kind != "SYSTEM" {
            continue;
        }
        // The system literal is the first quoted string after `SYSTEM`.
        let Some(quote_at) = rest.find(['"', '\'']) else {
            break;
        };
        let quote = rest.as_bytes()[quote_at];
        let after = &rest[quote_at + 1..];
        let Some(end) = after.find(quote as char) else {
            break;
        };
        entities.insert(name.to_owned(), after[..end].to_owned());
        rest = &after[end + 1..];
    }
    entities
}

/// Reads a catalog file, normalizing it to UTF-8 (all catalogs are ASCII).
fn read_catalog(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Recursively parses a catalog file, appending every `TEST` element to `out`.
///
/// * `path` — the catalog file to read; its parent directory resolves the
///   `SYSTEM` identifiers of the entities it declares.
/// * `base` — the directory that `URI` attributes are resolved against, i.e. the
///   accumulated `xml:base` of the enclosing `TESTCASES` elements.
fn parse_catalog(path: &Path, base: &Path, out: &mut Vec<TestCase>) -> Result<(), String> {
    let text = read_catalog(path)?;
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut entities = BTreeMap::new();
    let mut bases = vec![base.to_path_buf()];

    let mut reader = Reader::from_str(&text);
    reader.config_mut().trim_text(true);
    loop {
        match reader
            .read_event()
            .map_err(|err| format!("{}: {err}", path.display()))?
        {
            Event::Eof => break,
            Event::DocType(doctype) => {
                let decoded = doctype
                    .decode()
                    .map_err(|err| format!("{}: {err}", path.display()))?;
                entities = parse_entity_declarations(&decoded);
            }
            Event::GeneralRef(reference) => {
                let name = reference
                    .decode()
                    .map_err(|err| format!("{}: {err}", path.display()))?;
                if let Some(system) = entities.get(name.as_ref()) {
                    let current = bases.last().cloned().unwrap_or_default();
                    parse_catalog(&dir.join(system), &current, out)?;
                }
            }
            event @ (Event::Start(_) | Event::Empty(_)) => {
                let is_empty = matches!(event, Event::Empty(_));
                let start = match event {
                    Event::Start(start) | Event::Empty(start) => start,
                    _ => unreachable!("matched above"),
                };
                let name = start.name();
                if !is_empty && (name == QName(b"TESTCASES") || name == QName(b"TESTSUITE")) {
                    let mut current = bases.last().cloned().unwrap_or_default();
                    for attr in start.attributes() {
                        let attr = attr.map_err(|err| format!("{}: {err}", path.display()))?;
                        if attr.key == QName(b"xml:base") {
                            let value = attr
                                .decoded_and_normalized_value(
                                    XmlVersion::Implicit1_0,
                                    reader.decoder(),
                                )
                                .map_err(|err| format!("{}: {err}", path.display()))?;
                            current = current.join(value.as_ref());
                        }
                    }
                    bases.push(current);
                } else if name == QName(b"TEST") {
                    let mut attrs: BTreeMap<String, String> = BTreeMap::new();
                    for attr in start.attributes() {
                        let attr = attr.map_err(|err| format!("{}: {err}", path.display()))?;
                        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                        let value = attr
                            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
                            .map_err(|err| format!("{}: {err}", path.display()))?;
                        attrs.insert(key, value.into_owned());
                    }
                    let base = bases.last().cloned().unwrap_or_default();
                    let uri = attrs.get("URI").cloned().unwrap_or_default();
                    out.push(TestCase {
                        id: attrs.get("ID").cloned().unwrap_or_default(),
                        ty: attrs.get("TYPE").cloned().unwrap_or_default(),
                        entities: attrs
                            .get("ENTITIES")
                            .cloned()
                            .unwrap_or_else(|| "none".to_owned()),
                        recommendation: attrs
                            .get("RECOMMENDATION")
                            .cloned()
                            .unwrap_or_else(|| "XML1.0".to_owned()),
                        edition: attrs.get("EDITION").cloned(),
                        namespace: attrs
                            .get("NAMESPACE")
                            .cloned()
                            .unwrap_or_else(|| "yes".to_owned()),
                        sections: attrs.get("SECTIONS").cloned().unwrap_or_default(),
                        path: base.join(uri),
                    });
                }
            }
            Event::End(end) => {
                let name = end.name();
                if (name == QName(b"TESTCASES") || name == QName(b"TESTSUITE")) && bases.len() > 1 {
                    bases.pop();
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Encodings the suite exercises that `oxixml-encoding` does not implement.
const UNSUPPORTED_ENCODINGS: &[&str] = &["shift_jis", "euc-jp", "iso-2022-jp"];

/// Detects the documents that use an encoding this crate's lexer cannot handle.
fn declares_unsupported_encoding(bytes: &[u8]) -> bool {
    // UTF-16, with or without a byte order mark, is not ASCII compatible.
    if bytes.starts_with(&[0xFF, 0xFE])
        || bytes.starts_with(&[0xFE, 0xFF])
        || bytes.starts_with(&[0x00, b'<'])
        || bytes.starts_with(&[b'<', 0x00])
    {
        return true;
    }
    // The declaration, if present, is ASCII and lives in the first bytes.
    let head = &bytes[..bytes.len().min(256)];
    let head = String::from_utf8_lossy(head).to_ascii_lowercase();
    let Some(decl_end) = head.find("?>") else {
        return false;
    };
    let decl = &head[..decl_end];
    UNSUPPORTED_ENCODINGS
        .iter()
        .any(|encoding| decl.contains(encoding))
}

/// Runs the parser over a document, returning the first error as a string.
fn parse_document(bytes: &[u8], namespaces: bool) -> Result<(), String> {
    let mut buf = Vec::new();
    if namespaces {
        let mut reader = NsReader::from_reader(bytes);
        reader.config_mut().enable_all_checks(true);
        loop {
            match reader.read_resolved_event_into(&mut buf) {
                Ok((_, Event::Eof)) => return Ok(()),
                Ok(_) => buf.clear(),
                Err(err) => return Err(describe(&err)),
            }
        }
    } else {
        let mut reader = Reader::from_reader(bytes);
        reader.config_mut().enable_all_checks(true);
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => return Ok(()),
                Ok(_) => buf.clear(),
                Err(err) => return Err(describe(&err)),
            }
        }
    }
}

/// Renders an error for the failure report.
fn describe(error: &Error) -> String {
    format!("{error:?}: {error}")
}

/// Classifies a case, returning `Some(reason)` when it must not be run.
fn skip_reason(case: &TestCase, bytes: Option<&[u8]>) -> Option<SkipReason> {
    if case.ty == "error" {
        return Some(SkipReason::OptionalError);
    }
    if case.entities != "none" {
        return Some(SkipReason::ExternalEntities);
    }
    if case.recommendation.contains("errata") {
        return Some(SkipReason::Errata);
    }

    if let Some(edition) = &case.edition
        && !edition.split_whitespace().any(|e| e == "5")
    {
        return Some(SkipReason::VersionOrEdition);
    }
    match bytes {
        None => Some(SkipReason::MissingFile),
        Some(bytes) if declares_unsupported_encoding(bytes) => {
            Some(SkipReason::UnsupportedEncoding)
        }
        Some(_) => None,
    }
}

#[test]
fn w3c_xml_conformance_suite() -> Result<(), String> {
    let Some(catalog) = find_catalog() else {
        eprintln!("SKIP: {SUITES_DIR_ENV} is unset or does not contain an `xmlts*` distribution");
        return Ok(());
    };
    let root = catalog.parent().unwrap_or(Path::new(".")).to_path_buf();

    let mut cases = Vec::new();
    parse_catalog(&catalog, &root, &mut cases)?;
    assert!(!cases.is_empty(), "no TEST elements found in {catalog:?}");

    let allowlist: BTreeMap<&str, &str> = NOT_WF_ALLOWLIST.iter().copied().collect();

    let mut skipped: BTreeMap<SkipReason, usize> = BTreeMap::new();
    let mut not_wf_total = 0usize;
    let mut not_wf_detected = 0usize;
    let mut valid_total = 0usize;
    let mut valid_parsed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut stale_allowlist: Vec<String> = Vec::new();
    let mut allowlisted_hits = 0usize;

    for case in &cases {
        let bytes = fs::read(&case.path).ok();
        if let Some(reason) = skip_reason(case, bytes.as_deref()) {
            *skipped.entry(reason).or_default() += 1;
            continue;
        }
        let bytes = bytes.unwrap_or_default();
        let namespaces = case.namespace != "no";
        let outcome = parse_document(&bytes, namespaces);

        match case.ty.as_str() {
            "not-wf" => {
                not_wf_total += 1;
                match (outcome, allowlist.get(case.id.as_str())) {
                    (Err(_), None) => not_wf_detected += 1,
                    (Err(_), Some(reason)) => {
                        not_wf_detected += 1;
                        stale_allowlist.push(format!(
                            "{}: now rejected, drop the allowlist entry ({reason})",
                            case.id
                        ));
                    }
                    (Ok(()), Some(_)) => allowlisted_hits += 1,
                    (Ok(()), None) => failures.push(format!(
                        "not-wf {} ({}) was accepted [{}] SECTIONS={}",
                        case.id,
                        case.path.display(),
                        case.recommendation,
                        case.sections
                    )),
                }
            }
            "valid" | "invalid" => {
                valid_total += 1;
                match outcome {
                    Ok(()) => valid_parsed += 1,
                    Err(err) => failures.push(format!(
                        "{} {} ({}) was rejected: {err}",
                        case.ty,
                        case.id,
                        case.path.display()
                    )),
                }
            }
            other => failures.push(format!("{}: unknown TYPE {other:?}", case.id)),
        }
    }

    let mut report = String::new();
    let _ = writeln!(
        report,
        "\nW3C xmlts conformance report ({} cases)",
        cases.len()
    );
    let _ = writeln!(
        report,
        "  not-wf : {not_wf_detected}/{not_wf_total} detected ({allowlisted_hits} allowlisted)"
    );
    let _ = writeln!(
        report,
        "  valid  : {valid_parsed}/{valid_total} parsed (valid + invalid documents)"
    );
    for (reason, count) in &skipped {
        let _ = writeln!(report, "  skipped: {count:>4}  {}", reason.label());
    }
    for failure in &failures {
        println!("FAIL {failure}");
    }
    for stale in &stale_allowlist {
        println!("STALE {stale}");
    }
    println!("{report}");

    // The gate: at least 99% of the applicable not-wf cases must be detected and
    // every applicable valid/invalid document must parse.
    if not_wf_total > 0 && not_wf_detected * 100 < not_wf_total * 99 {
        return Err(format!(
            "{report}only {not_wf_detected} of {not_wf_total} not-wf cases were detected, \
             which is below the 99% gate"
        ));
    }

    if !failures.is_empty() || !stale_allowlist.is_empty() {
        let mut message = report;
        let _ = writeln!(
            message,
            "{} failures, {} stale allowlist entries (run with `--nocapture` for the list)",
            failures.len(),
            stale_allowlist.len()
        );
        return Err(message);
    }
    Ok(())
}
