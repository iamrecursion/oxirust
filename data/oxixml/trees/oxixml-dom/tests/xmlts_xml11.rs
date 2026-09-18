//! Runs the XML 1.1 sections of the W3C XML Conformance Test Suite through the
//! builder and checks each document against the verdict the suite expects.
//!
//! # How the suite is located
//!
//! The suite root comes from the `OXIXML_XML_SUITES_DIR` environment variable;
//! inside it, the first directory whose name starts with `xmlts` is used. When
//! the variable is unset, or the expected layout is missing, the test succeeds
//! silently: the suite is an optional, externally provisioned asset.
//!
//! # What is asserted
//!
//! For every `<TEST VERSION="1.1" …>` in `ibm/xml-1.1` and `eduni/xml-1.1`:
//! a document of type `valid` or `invalid` must **build** — a non-validating
//! processor accepts both — and one of type `not-wf` must be **rejected**.
//! Nothing may diverge from that.
//!
//! Three groups of tests are deliberately not counted, because they ask for
//! something this crate does not claim to do rather than for XML 1.1:
//!
//! * **`TYPE="error"`** — the suite defines these as cases a processor *may*
//!   report; either answer conforms.
//! * **tests that depend on an external entity or an external DTD subset** —
//!   identified by a non-`none` `ENTITIES` attribute or by a `SYSTEM`/`PUBLIC`
//!   document type declaration. A non-validating processor is allowed not to
//!   retrieve either ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types)),
//!   and this one does not, so a violation that only shows up in the retrieved
//!   text is not visible here. (`build::parse_str_with_resolver` can retrieve
//!   external *general* entities on request; the external subset is still never
//!   read.)
//! * **[`NAMESPACE_DEPENDENT`]** — four documents that are well-formed XML 1.1
//!   but not namespace-well-formed. The builder resolves namespaces with
//!   `NsReader`, so it rejects them; the same class is already documented in
//!   `xmlts_roundtrip.rs`.
//!
//! Non-UTF-8 documents are skipped: this test reads the suite as UTF-8.

use std::path::{Path, PathBuf};

/// Environment variable pointing at the directory that holds the W3C XML suites.
const SUITES_DIR_ENV: &str = "OXIXML_XML_SUITES_DIR";

/// The catalogs read, each with the directory its `URI`s are relative to.
const CATALOGS: &[(&str, &str)] = &[
    ("ibm/xml-1.1/ibm_valid.xml", "ibm/xml-1.1"),
    ("ibm/xml-1.1/ibm_invalid.xml", "ibm/xml-1.1"),
    ("ibm/xml-1.1/ibm_not-wf.xml", "ibm/xml-1.1"),
    ("eduni/xml-1.1/xml11.xml", "eduni/xml-1.1"),
];

/// Documents the suite calls well-formed that are not *namespace*-well-formed:
/// an undeclared default namespace, an undeclared prefix, and a processing
/// instruction whose target contains a colon. The builder is namespace-aware, so
/// it rejects all four.
const NAMESPACE_DEPENDENT: &[&str] = &[
    "valid/P04/ibm04v01.xml",
    "valid/P05/ibm05v01.xml",
    "valid/P05/ibm05v02.xml",
    "valid/P05/ibm05v03.xml",
];

/// The smallest number of documents the run is expected to reach. The 2013
/// distribution yields a little over a hundred and sixty once the groups above
/// are set aside; this is a floor that only trips when the discovery logic
/// itself breaks.
const MINIMUM_DOCUMENTS: usize = 150;

fn suite_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os(SUITES_DIR_ENV)?);
    for entry in std::fs::read_dir(&root).ok()?.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("xmlts"))
        {
            let xmlconf = path.join("xmlconf");
            if xmlconf.is_dir() {
                return Some(xmlconf);
            }
        }
    }
    None
}

/// One `<TEST …>` element of a catalog, reduced to the attributes used here.
struct Case {
    uri: String,
    kind: String,
    entities: String,
    version: String,
}

/// Removes every comment from `catalog`: the suite withdraws a test by
/// commenting it out, and a withdrawn test must not be run.
fn without_comments(catalog: &str) -> String {
    let mut out = String::with_capacity(catalog.len());
    let mut rest = catalog;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + "-->".len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Scans the `<TEST …>` start tags of a catalog. The catalogs use a DTD this
/// crate does not read, so they are scanned rather than parsed.
fn cases(catalog: &str) -> Vec<Case> {
    let mut out = Vec::new();
    let mut rest = catalog;
    while let Some(start) = rest.find("<TEST ") {
        rest = &rest[start..];
        let Some(end) = rest.find('>') else { break };
        let head = &rest[..end];
        rest = &rest[end..];
        let attribute = |name: &str| -> String {
            let key = format!("{name}=\"");
            head.find(&key)
                .and_then(|index| {
                    let after = &head[index + key.len()..];
                    after.find('"').map(|close| after[..close].to_owned())
                })
                .unwrap_or_default()
        };
        out.push(Case {
            uri: attribute("URI"),
            kind: attribute("TYPE"),
            entities: attribute("ENTITIES"),
            version: attribute("VERSION"),
        });
    }
    out
}

/// Whether `source` declares an external DTD subset, whose content this crate
/// never retrieves.
fn has_external_subset(source: &str) -> bool {
    let Some(start) = source.find("<!DOCTYPE") else {
        return false;
    };
    let rest = &source[start..];
    let head = match (rest.find('['), rest.find('>')) {
        (Some(bracket), Some(angle)) => &rest[..bracket.min(angle)],
        (Some(bracket), None) => &rest[..bracket],
        (None, Some(angle)) => &rest[..angle],
        (None, None) => rest,
    };
    head.contains("SYSTEM") || head.contains("PUBLIC")
}

#[test]
fn xml11_conformance_documents_get_the_expected_verdict() {
    let Some(root) = suite_root() else {
        eprintln!("skipping: set {SUITES_DIR_ENV} to the directory holding the W3C `xmlts` suite");
        return;
    };

    let mut checked = 0_usize;
    let mut divergences: Vec<String> = Vec::new();
    for (catalog, base) in CATALOGS {
        let Ok(text) = std::fs::read_to_string(root.join(catalog)) else {
            continue;
        };
        for case in cases(&without_comments(&text)) {
            if case.uri.is_empty() || !case.version.contains("1.1") {
                continue;
            }
            // A case a processor may or may not report is not a measurement.
            if case.kind == "error" {
                continue;
            }
            if NAMESPACE_DEPENDENT.contains(&case.uri.as_str()) {
                continue;
            }
            let path: PathBuf = Path::new(&root).join(base).join(&case.uri);
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue; // Not UTF-8, or not shipped with this distribution.
            };
            let needs_external =
                !matches!(case.entities.as_str(), "" | "none") || has_external_subset(&source);
            if needs_external {
                continue;
            }

            checked += 1;
            let outcome = oxixml_dom::build::parse_str(&source);
            let expected_ok = case.kind != "not-wf";
            match (&outcome, expected_ok) {
                (Ok(_), true) | (Err(_), false) => {}
                (Err(error), true) => divergences.push(format!(
                    "{}: expected {} to build, but it was rejected: {error}",
                    case.uri, case.kind
                )),
                (Ok(_), false) => {
                    divergences.push(format!("{}: expected not-wf, but it built", case.uri))
                }
            }
        }
    }

    assert!(
        divergences.is_empty(),
        "{} XML 1.1 conformance divergences:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
    assert!(
        checked >= MINIMUM_DOCUMENTS,
        "only {checked} documents were exercised, expected at least {MINIMUM_DOCUMENTS}"
    );
}
