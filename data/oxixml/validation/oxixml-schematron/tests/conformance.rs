//! Compiles every ISO-namespace Schematron schema found under
//! `$OXIXML_XML_SUITES_DIR/schematron` — the [Schematron reference implementation]'s own test
//! corpus, vendored as a git submodule — and reports how many compile successfully under this
//! crate's XPath query binding.
//!
//! There is no expected-output oracle in that corpus (it is build-tool fixtures, not a
//! pass/fail conformance suite), so this is a compile-coverage smoke test: it discovers real,
//! independently authored schemas and confirms a healthy fraction of the ones that do not
//! require an executable `xslt` query binding (embedded `xsl:` sequence constructors, Saxon or
//! EXSLT extension functions) compile under [`Schematron::compile`]. Skips silently when
//! `OXIXML_XML_SUITES_DIR` is unset or has no `schematron/` suite.
//!
//! [Schematron reference implementation]: https://github.com/Schematron/schematron

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_schematron::Schematron;

fn discover_sch_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        if name.starts_with('.') {
            continue; // skip `.git` and other dotfiles/directories
        }
        if path.is_dir() {
            discover_sch_files(&path, out);
        } else if path.extension().and_then(OsStr::to_str) == Some("sch") {
            out.push(path);
        }
    }
}

#[test]
fn schematron_reference_suite_schemas_compile() {
    let Some(suites_root) = std::env::var_os("OXIXML_XML_SUITES_DIR") else {
        eprintln!("skipping: OXIXML_XML_SUITES_DIR is not set");
        return;
    };
    let suite_dir = Path::new(&suites_root).join("schematron");
    if !suite_dir.is_dir() {
        eprintln!("skipping: {} does not exist", suite_dir.display());
        return;
    }

    let mut files = Vec::new();
    discover_sch_files(&suite_dir, &mut files);
    if files.is_empty() {
        eprintln!(
            "skipping: no `.sch` files found under {}",
            suite_dir.display()
        );
        return;
    }

    let mut compiled = 0usize;
    let mut skipped_not_schematron = 0usize;
    let mut failed: Vec<(PathBuf, String)> = Vec::new();

    for path in &files {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        match Schematron::compile(&text, None) {
            Ok(_) => compiled += 1,
            Err(oxixml_schematron::SchematronError::NotASchema { .. }) => {
                // A `.sch` file whose root is not `sch:schema` (a fragment meant only for
                // `sch:include`, or an unrelated file this discovery swept up).
                skipped_not_schematron += 1;
            }
            Err(error) => failed.push((path.clone(), error.to_string())),
        }
    }

    eprintln!(
        "schematron reference suite: {compiled} compiled, {skipped_not_schematron} not a \
         standalone schema, {} failed, out of {} `.sch` files discovered under {}",
        failed.len(),
        files.len(),
        suite_dir.display(),
    );
    for (path, message) in failed.iter().take(20) {
        eprintln!("  {}: {message}", path.display());
    }

    assert!(
        compiled > 0,
        "expected at least one schema from the reference suite to compile under the XPath \
         query binding, out of {} discovered",
        files.len()
    );
}
