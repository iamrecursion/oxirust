//! W3C XML Conformance Test Suite (`xmlconf`) runner.
//!
//! The suite catalog (`xmlconf.xml`, plus every vendor sub-catalog it pulls in through a
//! `SYSTEM` general entity declared in its internal DTD subset) is parsed once with
//! [`oxixml_xml::reader::Reader`], honouring the `xml:base` attribute every `TESTCASES` and
//! `TESTSUITE` element may carry, into a flat list of `TestCase`s. Two independent passes then
//! run over that same list, matching the two dimensions the suite exercises:
//!
//! * **`xmlconf/wellformed`** — every case, run through [`oxixml_xml`]'s pull parser. A
//!   `not-wf` document must be rejected; a `valid` or `invalid` document must still be *accepted*
//!   here, because [XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types) permits a
//!   non-validating processor that does not read external entities to accept a document whose
//!   only problems are validity constraints. `error`-typed cases are skipped (the specification
//!   leaves reporting them optional), as are cases needing external entities, XML 1.1, a
//!   non-5th-edition errata revision, or an encoding this crate's byte-oriented lexer does not
//!   transcode (UTF-16, Shift_JIS, EUC-JP, ISO-2022-JP).
//! * **`xmlconf/validity`** — the `valid`/`invalid` cases only (XML 1.0, non-errata), run
//!   through [`oxixml_dtd::DocumentValidator`]. A `valid` document must validate cleanly; an
//!   `invalid` one must be reported invalid (or rejected outright, which is a stricter but
//!   acceptable response to a validity violation the suite does not distinguish here).
//!
//! Both skip silently when `OXIXML_XML_SUITES_DIR` is unset or has no `xmlts*/xmlconf` layout
//! under it — the date-suffixed distribution name is discovered, never hardcoded.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oxixml_dtd::{DocumentValidator, ExpansionLimits, FilesystemResolver};
use oxixml_xml::XmlVersion;
use oxixml_xml::errors::Error as XmlError;
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::QName;
use oxixml_xml::reader::{NsReader, Reader};

use crate::env;
use crate::report::SuiteReport;

/// One `<TEST>` entry of the catalog.
#[derive(Debug, Clone)]
struct TestCase {
    id: String,
    ty: String,
    entities: String,
    recommendation: String,
    version: Option<String>,
    edition: Option<String>,
    namespace: String,
    path: PathBuf,
}

/// Cases this parser knowingly accepts even though the suite marks them not well-formed, each
/// with the reason: applying a declared attribute type (here, `NMTOKEN` normalization) is a
/// validating processor's job, and this crate's well-formedness parser deliberately matches
/// `quick-xml` 0.41's acceptance of it for drop-in compatibility.
const NOT_WF_ALLOWLIST: &[(&str, &str)] = &[(
    "rmt-ns10-012",
    "the two `xmlns` values only become equal after NMTOKEN attribute-value normalization, which \
     needs a validating processor",
)];

/// Runs both passes over the catalog.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(catalog) = find_catalog() else {
        let reason = match env::xml_suites_dir() {
            None => env::missing(env::XML_SUITES_DIR_VAR),
            Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "xmlts*/xmlconf/xmlconf.xml"),
        };
        return vec![
            SuiteReport::unavailable("xmlconf/wellformed", reason.clone()),
            SuiteReport::unavailable("xmlconf/validity", reason),
        ];
    };
    let root = catalog.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut cases = Vec::new();
    if let Err(message) = parse_catalog(&catalog, &root, &mut cases) {
        let mut report = SuiteReport::new("xmlconf/wellformed");
        report.record_skip(message);
        return vec![
            report,
            SuiteReport::unavailable("xmlconf/validity", "the catalog failed to parse"),
        ];
    }

    vec![run_wellformed(&cases), run_validity(&root, &cases)]
}

// --- catalog discovery and parsing ----------------------------------------------------------

fn find_catalog() -> Option<PathBuf> {
    let root = env::xml_suites_dir()?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("xmlts"))
        })
        .collect();
    candidates.sort();
    let catalog = candidates.pop()?.join("xmlconf").join("xmlconf.xml");
    catalog.is_file().then_some(catalog)
}

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

fn read_catalog(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Recursively parses one catalog file, appending every `TEST` element to `out`, and following
/// any `SYSTEM` general entity its internal DTD subset declares into a sub-catalog.
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
            .map_err(|error| format!("{}: {error}", path.display()))?
        {
            Event::Eof => break,
            Event::DocType(doctype) => {
                let decoded = doctype
                    .decode()
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                entities = parse_entity_declarations(&decoded);
            }
            Event::GeneralRef(reference) => {
                let name = reference
                    .decode()
                    .map_err(|error| format!("{}: {error}", path.display()))?;
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
                    for attribute in attribute_pairs(&start, &reader)? {
                        if attribute.0 == "xml:base" {
                            current = current.join(&attribute.1);
                        }
                    }
                    bases.push(current);
                } else if name == QName(b"TEST") {
                    let attributes: BTreeMap<String, String> =
                        attribute_pairs(&start, &reader)?.into_iter().collect();
                    let base = bases.last().cloned().unwrap_or_default();
                    let uri = attributes.get("URI").cloned().unwrap_or_default();
                    out.push(TestCase {
                        id: attributes.get("ID").cloned().unwrap_or_default(),
                        ty: attributes.get("TYPE").cloned().unwrap_or_default(),
                        entities: attributes
                            .get("ENTITIES")
                            .cloned()
                            .unwrap_or_else(|| "none".to_owned()),
                        recommendation: attributes
                            .get("RECOMMENDATION")
                            .cloned()
                            .unwrap_or_else(|| "XML1.0".to_owned()),
                        version: attributes.get("VERSION").cloned(),
                        edition: attributes.get("EDITION").cloned(),
                        namespace: attributes
                            .get("NAMESPACE")
                            .cloned()
                            .unwrap_or_else(|| "yes".to_owned()),
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

fn attribute_pairs(
    start: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for attribute in start.attributes() {
        let attribute = attribute.map_err(|error| error.to_string())?;
        let key = String::from_utf8_lossy(attribute.key.as_ref()).into_owned();
        let value = attribute
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| error.to_string())?;
        out.push((key, value.into_owned()));
    }
    Ok(out)
}

// --- shared helpers --------------------------------------------------------------------------

/// Decodes a catalog document's bytes, honouring a byte-order mark; falls back to UTF-8.
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

// --- wellformed pass ---------------------------------------------------------------------------

/// Encodings the suite exercises that `oxixml-encoding`'s byte-oriented lexer does not
/// transcode.
const UNSUPPORTED_ENCODINGS: &[&str] = &["shift_jis", "euc-jp", "iso-2022-jp"];

fn declares_unsupported_encoding(bytes: &[u8]) -> bool {
    if bytes.starts_with(&[0xFF, 0xFE])
        || bytes.starts_with(&[0xFE, 0xFF])
        || bytes.starts_with(&[0x00, b'<'])
        || bytes.starts_with(&[b'<', 0x00])
    {
        return true;
    }
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

fn wellformed_skip_reason(case: &TestCase, bytes: Option<&[u8]>) -> Option<&'static str> {
    if case.ty == "error" {
        return Some("TYPE=error (reporting is optional)");
    }
    if case.entities != "none" {
        return Some("document needs external entities");
    }
    if case.recommendation.contains("errata") {
        return Some("targets an errata revision");
    }
    if let Some(version) = &case.version
        && !version.split_whitespace().any(|v| v == "1.0")
    {
        return Some("targets a non-1.0 XML version");
    }
    if case.recommendation == "XML1.1" || case.recommendation == "NS1.1" {
        return Some("targets XML/Namespaces 1.1");
    }
    if let Some(edition) = &case.edition
        && !edition.split_whitespace().any(|e| e == "5")
    {
        return Some("targets a non-5th edition");
    }
    match bytes {
        None => Some("document file missing"),
        Some(bytes) if declares_unsupported_encoding(bytes) => Some("unsupported encoding"),
        Some(_) => None,
    }
}

fn parse_wellformedness(bytes: &[u8], namespaces: bool) -> Result<(), String> {
    let mut buf = Vec::new();
    if namespaces {
        let mut reader = NsReader::from_reader(bytes);
        reader.config_mut().enable_all_checks(true);
        loop {
            match reader.read_resolved_event_into(&mut buf) {
                Ok((_, Event::Eof)) => return Ok(()),
                Ok(_) => buf.clear(),
                Err(error) => return Err(describe(&error)),
            }
        }
    } else {
        let mut reader = Reader::from_reader(bytes);
        reader.config_mut().enable_all_checks(true);
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => return Ok(()),
                Ok(_) => buf.clear(),
                Err(error) => return Err(describe(&error)),
            }
        }
    }
}

fn describe(error: &XmlError) -> String {
    format!("{error:?}: {error}")
}

fn run_wellformed(cases: &[TestCase]) -> SuiteReport {
    let allowlist: BTreeMap<&str, &str> = NOT_WF_ALLOWLIST.iter().copied().collect();
    let mut report = SuiteReport::new("xmlconf/wellformed");
    for case in cases {
        let bytes = std::fs::read(&case.path).ok();
        if let Some(reason) = wellformed_skip_reason(case, bytes.as_deref()) {
            report.record_skip(reason);
            continue;
        }
        let bytes = bytes.unwrap_or_default();
        let namespaces = case.namespace != "no";
        let outcome = parse_wellformedness(&bytes, namespaces);

        match case.ty.as_str() {
            "not-wf" => {
                if allowlist.contains_key(case.id.as_str()) {
                    report.record_skip(format!(
                        "documented divergence: {}",
                        allowlist[case.id.as_str()]
                    ));
                } else if outcome.is_err() {
                    report.record_pass();
                } else {
                    report.record_fail(
                        &case.id,
                        format!("not-wf document was accepted [{}]", case.path.display()),
                    );
                }
            }
            "valid" | "invalid" => match outcome {
                Ok(()) => report.record_pass(),
                Err(message) => report.record_fail(
                    &case.id,
                    format!(
                        "{} {} was rejected: {message}",
                        case.ty,
                        case.path.display()
                    ),
                ),
            },
            other => report.record_fail(&case.id, format!("unknown TYPE {other:?}")),
        }
    }
    report
}

// --- validity pass -----------------------------------------------------------------------------

fn validity_applicable(case: &TestCase) -> bool {
    if !matches!(case.ty.as_str(), "valid" | "invalid") {
        return false;
    }
    if case.version.as_deref() == Some("1.1") {
        return false;
    }
    case.recommendation.is_empty() || case.recommendation.starts_with("XML1.0")
}

fn run_validity(root: &Path, cases: &[TestCase]) -> SuiteReport {
    let mut report = SuiteReport::new("xmlconf/validity");
    for case in cases {
        if !validity_applicable(case) {
            continue;
        }
        let Ok(bytes) = std::fs::read(&case.path) else {
            report.record_skip("document file missing or unreadable");
            continue;
        };
        let Some(text) = decode(&bytes) else {
            report.record_skip("document is not UTF-8 or UTF-16");
            continue;
        };
        let relative_uri = case
            .path
            .strip_prefix(root)
            .unwrap_or(&case.path)
            .to_string_lossy()
            .replace('\\', "/");
        let resolver = FilesystemResolver::new(root);
        let mut validator = DocumentValidator::new()
            .with_resolver(Box::new(resolver))
            .with_limits(ExpansionLimits::RELAXED);

        let outcome = validator.validate_str_with_base(&text, Some(&relative_uri));
        let expected_valid = case.ty == "valid";
        let ok = match &outcome {
            Ok(validation) if expected_valid => validation.is_valid(),
            Ok(validation) => !validation.is_valid(),
            Err(_) => !expected_valid,
        };
        if ok {
            report.record_pass();
        } else {
            let detail = match &outcome {
                Ok(validation) => validation
                    .errors()
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
                Err(error) => error.to_string(),
            };
            report.record_fail(
                &case.id,
                format!("{} [{}]: {detail}", case.ty, case.path.display()),
            );
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entity_declarations_extracts_system_entities_only() {
        let doctype = r#"<!DOCTYPE TESTCASES [
            <!ENTITY vendor-a SYSTEM "vendor-a/catalog.xml">
            <!ENTITY not-system "inline text">
        ]>"#;
        let entities = parse_entity_declarations(doctype);
        assert_eq!(
            entities.get("vendor-a").map(String::as_str),
            Some("vendor-a/catalog.xml")
        );
        assert_eq!(entities.get("not-system"), None);
    }

    #[test]
    fn declares_unsupported_encoding_recognizes_the_documented_encodings_and_utf16() {
        assert!(declares_unsupported_encoding(
            b"<?xml version=\"1.0\" encoding=\"Shift_JIS\"?>"
        ));
        assert!(declares_unsupported_encoding(&[0xFE, 0xFF, 0x00, b'<']));
        assert!(!declares_unsupported_encoding(
            b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        ));
    }

    #[test]
    fn decode_strips_a_utf8_bom() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"<r/>");
        assert_eq!(decode(&bytes).as_deref(), Some("<r/>"));
    }

    #[test]
    fn a_small_embedded_catalog_parses_into_test_cases() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-xmlconf-fixture-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        std::fs::write(dir.join("wf.xml"), b"<r/>").expect("scratch file");
        std::fs::write(dir.join("bad.xml"), b"<r>").expect("scratch file");
        let catalog = dir.join("catalog.xml");
        std::fs::write(
            &catalog,
            br#"<TESTCASES>
                <TEST ID="wf-1" TYPE="not-wf" URI="wf.xml"/>
                <TEST ID="bad-1" TYPE="not-wf" URI="bad.xml"/>
            </TESTCASES>"#,
        )
        .expect("scratch file");

        let mut cases = Vec::new();
        parse_catalog(&catalog, &dir, &mut cases).expect("catalog parses");
        assert_eq!(cases.len(), 2);
        let report = run_wellformed(&cases);
        // `wf.xml` is well-formed but the case claims not-wf, so it is a wrongly-accepted
        // failure; `bad.xml` is genuinely not well-formed, so it correctly passes.
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| !report.is_available()));
    }
}
