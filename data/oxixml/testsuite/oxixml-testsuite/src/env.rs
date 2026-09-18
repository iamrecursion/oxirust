//! Environment-variable driven suite discovery.
//!
//! Every runner in this crate finds its suite through one of the three variables here, and
//! **never** through a hardcoded path — a checkout without the suites downloaded must still
//! build and test cleanly. Nothing here reads a variable other than the three named below.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Root of the vendored W3C XML/XSD/XPath/XSLT/RELAX NG/Schematron/C14N/XMLDSig suites:
/// `xmlts*/`, `xsdtests*/`, `qt3tests*/`, `xslt30-test/`, `jing-trang/`, `schematron/`,
/// `xml-c14n2-testcases/`, `xmldsig-core1-interop/`.
pub const XML_SUITES_DIR_VAR: &str = "OXIXML_XML_SUITES_DIR";

/// Root of the oxigraph testsuite checkout: `rdf-tests/`, `rdf-canon/`, `json-ld-api/`,
/// `json-ld-streaming/`.
pub const TESTSUITE_DIR_VAR: &str = "OXIXML_TESTSUITE_DIR";

/// Root of an RDFa test-suite checkout, if one is provisioned.
pub const RDFA_SUITE_DIR_VAR: &str = "OXIXML_RDFA_SUITE_DIR";

/// Reads an environment variable as a directory path, treating "unset" and "set but empty" the
/// same way (both mean "not configured").
fn dir_var(name: &str) -> Option<PathBuf> {
    let value = std::env::var_os(name)?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

/// The configured root of [`XML_SUITES_DIR_VAR`], regardless of whether it exists on disk.
#[must_use]
pub fn xml_suites_dir() -> Option<PathBuf> {
    dir_var(XML_SUITES_DIR_VAR)
}

/// The configured root of [`TESTSUITE_DIR_VAR`], regardless of whether it exists on disk.
#[must_use]
pub fn testsuite_dir() -> Option<PathBuf> {
    dir_var(TESTSUITE_DIR_VAR)
}

/// The configured root of [`RDFA_SUITE_DIR_VAR`], regardless of whether it exists on disk.
#[must_use]
pub fn rdfa_suite_dir() -> Option<PathBuf> {
    dir_var(RDFA_SUITE_DIR_VAR)
}

/// A directory under [`xml_suites_dir`] with a stable name (`root.join(name)`), falling back to
/// [`find_prefixed`] for the distributions the W3C ships with a release-date suffix
/// (`xmlts20130923`, dated `xsdtests` snapshots, …).
///
/// `probe` is a path, relative to the candidate directory, that must exist for the candidate to
/// be accepted — e.g. `"suite.xml"` for `xsdtests`, `"catalog.xml"` for `qt3tests`. Pass `""` to
/// accept any directory that exists.
#[must_use]
pub fn xml_suite_subdir(name: &str, probe: &str) -> Option<PathBuf> {
    let root = xml_suites_dir()?;
    let direct = root.join(name);
    if probe_exists(&direct, probe) {
        return Some(direct);
    }
    find_prefixed(&root, name, probe)
}

/// Finds the (lexicographically last, so the newest of several dated snapshots wins) entry of
/// `root` whose file name starts with `prefix` and which, joined with `probe`, exists.
#[must_use]
pub fn find_prefixed(root: &Path, prefix: &str, probe: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with(prefix))
                && probe_exists(path, probe)
        })
        .collect();
    candidates.sort();
    candidates.pop()
}

fn probe_exists(candidate: &Path, probe: &str) -> bool {
    if probe.is_empty() {
        candidate.is_dir()
    } else {
        candidate.join(probe).exists()
    }
}

/// Recursively collects every file named exactly `filename` under `root`, sorted.
///
/// Used by runners whose suite spreads its manifests across an arbitrarily deep directory tree
/// (the SPARQL suites, in particular) rather than following an explicit include list.
#[must_use]
pub fn find_files_named(root: &Path, filename: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect_files_named(root, filename, &mut found);
    found.sort();
    found
}

fn collect_files_named(directory: &Path, filename: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_named(&path, filename, out);
        } else if path.file_name().and_then(OsStr::to_str) == Some(filename) {
            out.push(path);
        }
    }
}

/// A human-readable reason to attach to [`crate::report::SuiteReport::unavailable`] when `var`
/// is unset.
#[must_use]
pub fn missing(var: &str) -> String {
    format!("{var} is not set")
}

/// A human-readable reason for when `var` is set but the expected `what` was not found under it.
#[must_use]
pub fn not_found(var: &str, what: &str) -> String {
    format!("{var} is set, but no {what} was found under it")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_prefixed_picks_the_lexicographically_last_match() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-env-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("xmlts20020606")).expect("scratch dir");
        std::fs::create_dir_all(dir.join("xmlts20130923")).expect("scratch dir");
        std::fs::write(dir.join("xmlts20020606").join("marker"), b"").expect("scratch file");
        std::fs::write(dir.join("xmlts20130923").join("marker"), b"").expect("scratch file");

        let found = find_prefixed(&dir, "xmlts", "marker").expect("a match");
        assert_eq!(found, dir.join("xmlts20130923"));

        // A prefix with no matching probe file is not found.
        assert!(find_prefixed(&dir, "xmlts", "does-not-exist").is_none());
        assert!(find_prefixed(&dir, "nope", "marker").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_files_named_recurses_and_sorts() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-env-find-files-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("b").join("nested")).expect("scratch dir");
        std::fs::create_dir_all(dir.join("a")).expect("scratch dir");
        std::fs::write(dir.join("a").join("manifest.ttl"), b"").expect("scratch file");
        std::fs::write(dir.join("b").join("nested").join("manifest.ttl"), b"")
            .expect("scratch file");
        std::fs::write(dir.join("b").join("other.ttl"), b"").expect("scratch file");

        let found = find_files_named(&dir, "manifest.ttl");
        assert_eq!(
            found,
            vec![
                dir.join("a").join("manifest.ttl"),
                dir.join("b").join("nested").join("manifest.ttl"),
            ]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_and_not_found_produce_distinct_messages() {
        assert!(missing("OXIXML_DEMO_DIR").contains("OXIXML_DEMO_DIR"));
        assert_ne!(
            missing("OXIXML_DEMO_DIR"),
            not_found("OXIXML_DEMO_DIR", "demo/")
        );
    }
}
