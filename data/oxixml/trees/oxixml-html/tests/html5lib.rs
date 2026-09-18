//! The html5lib-tests tree construction conformance harness.
//!
//! The HTML standard's tree construction tests are distributed as `.dat` files
//! by [html5lib-tests](https://github.com/html5lib/html5lib-tests) (the same
//! files the web-platform-tests project mirrors under
//! `html/syntax/parsing/resources`). They are not vendored into this
//! workspace, so this harness **discovers the suite through environment
//! variables and skips silently when neither names one**:
//!
//! * `OXIXML_HTML5LIB_TESTS_DIR` — a checkout of html5lib-tests, or its
//!   `tree-construction/` subdirectory directly.
//! * `OXIXML_XML_SUITES_DIR` — a directory of vendored conformance suites,
//!   searched for an `html5lib-tests/tree-construction` subdirectory. This is
//!   the variable the rest of this workspace's conformance harnesses share.
//!
//! Run them with, for example:
//!
//! ```sh
//! OXIXML_XML_SUITES_DIR=/path/to/suites \
//!   cargo test -p oxixml-html --test html5lib -- --nocapture
//! ```
//!
//! A second test in this file reuses the same corpus for a different
//! question: `streaming_matches_one_shot_over_the_suite` feeds every one of
//! these documents to [`oxixml_html::StreamingParser`] a few bytes at a time
//! and requires the tree to come out identical to the one-shot parse of the
//! same bytes. It compares the parser against itself rather than against the
//! suite's expected trees, so it runs the `scripted/` blocks too.
//!
//! The harness prints a per-file pass count so that a regression is visible
//! even when the overall assertion still passes, and prints every skipped
//! group with the justification that permits the skip. Only the
//! `tree-construction/` directory is walked: the sibling `encoding/` files use
//! the same `#data` framing but assert the encoding sniffing algorithm of
//! HTML § 13.2.3 rather than a document tree, and are covered by
//! `tests/encoding_suite.rs`.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_html::{
    Namespace, ParseOptions, StreamingParser, dump_tree, parse_document_with, parse_fragment_with,
};

/// Resolves one candidate root to the directory that actually holds the
/// tree construction `.dat` files.
fn resolve_suite_dir(root: &Path) -> Option<PathBuf> {
    for suffix in [
        Path::new("tree-construction").to_path_buf(),
        Path::new("html5lib-tests").join("tree-construction"),
    ] {
        let nested = root.join(suffix);
        if nested.is_dir() {
            return Some(nested);
        }
    }
    // Being pointed straight at a directory of `.dat` files also works.
    if root.is_dir() && !dat_files(root).is_empty() {
        return Some(root.to_path_buf());
    }
    None
}

/// Locates the directory holding the `.dat` files, if the environment names
/// one.
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

fn dat_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "dat") {
                files.insert(path);
            }
        }
    }
    files.into_iter().collect()
}

/// Whether `path` lives under the suite's `scripted/` directory.
fn is_scripted(path: &Path) -> bool {
    path.parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "scripted")
}

/// One `.dat` test case.
#[derive(Debug, Default)]
struct Case {
    data: String,
    document: String,
    fragment_context: Option<String>,
    scripting: Option<bool>,
}

/// Splits a `.dat` file into its test cases.
fn parse_dat(text: &str) -> Vec<Case> {
    const HEADINGS: [&str; 7] = [
        "#data",
        "#errors",
        "#new-errors",
        "#document-fragment",
        "#script-on",
        "#script-off",
        "#document",
    ];

    let mut cases = Vec::new();
    let mut current: Option<Case> = None;
    let mut section = String::new();
    let mut buffer = String::new();

    let flush = |case: &mut Case, section: &str, buffer: &mut String| {
        let content = std::mem::take(buffer);
        match section {
            "#data" => {
                // "the test data ... with the final newline (on the last line)
                // removed" — exactly one, so that a test whose input really
                // ends in a newline keeps it.
                let mut content = content;
                content.pop();
                case.data = content;
            }
            "#document-fragment" => {
                case.fragment_context = Some(content.trim().to_owned());
            }
            "#document" => {
                let mut content = content;
                if content.ends_with("\n\n") {
                    content.pop();
                }
                case.document = content;
            }
            _ => {}
        }
    };

    for line in text.split('\n') {
        if HEADINGS.contains(&line) {
            if let Some(case) = current.as_mut() {
                flush(case, &section, &mut buffer);
            }
            if line == "#data" {
                if let Some(case) = current.take() {
                    cases.push(case);
                }
                current = Some(Case::default());
            }
            if let Some(case) = current.as_mut() {
                match line {
                    "#script-on" => case.scripting = Some(true),
                    "#script-off" => case.scripting = Some(false),
                    _ => {}
                }
            }
            section = line.to_owned();
            buffer.clear();
        } else {
            buffer.push_str(line);
            buffer.push('\n');
        }
    }
    if let Some(mut case) = current {
        flush(&mut case, &section, &mut buffer);
        cases.push(case);
    }
    cases
}

/// Runs one case and returns the actual dump.
fn run_case(case: &Case) -> String {
    // The suite's `#script-on` / `#script-off` headings mark a block as
    // applying only when the parser's scripting flag has that value, so each
    // block runs with the flag it declares; blocks that declare neither run
    // with scripting disabled, the value a non-browser consumer uses.
    let options = ParseOptions::new().with_scripting(case.scripting.unwrap_or(false));
    match &case.fragment_context {
        None => dump_tree(&parse_document_with(&case.data, &options)),
        Some(context) => {
            let (namespace, name) = match context.split_once(' ') {
                Some(("svg", name)) => (Namespace::Svg, name),
                Some(("math", name)) => (Namespace::MathMl, name),
                _ => (Namespace::Html, context.as_str()),
            };
            dump_tree(&parse_fragment_with(&case.data, namespace, name, &options))
        }
    }
}

/// Runs one case through the chunk-fed parser, `size` bytes at a time.
fn run_case_streamed(case: &Case, size: usize) -> String {
    let options = ParseOptions::new().with_scripting(case.scripting.unwrap_or(false));
    let mut parser = match &case.fragment_context {
        None => StreamingParser::with_options(&options),
        Some(context) => {
            let (namespace, name) = match context.split_once(' ') {
                Some(("svg", name)) => (Namespace::Svg, name),
                Some(("math", name)) => (Namespace::MathMl, name),
                _ => (Namespace::Html, context.as_str()),
            };
            StreamingParser::fragment(namespace, name, &options)
        }
    };
    for chunk in case.data.as_bytes().chunks(size) {
        parser.write(chunk);
    }
    dump_tree(&parser.finish())
}

/// A skipped group of cases, with the justification that permits the skip.
struct Skip {
    reason: &'static str,
    cases: usize,
}

#[test]
fn tree_construction_suite() {
    let Some(dir) = suite_dir() else {
        eprintln!(
            "skipping: neither OXIXML_HTML5LIB_TESTS_DIR nor OXIXML_XML_SUITES_DIR \
             resolves to an html5lib tree construction suite"
        );
        return;
    };

    let files = dat_files(&dir);
    assert!(!files.is_empty(), "no .dat files under {}", dir.display());

    let mut total = 0_usize;
    let mut passed = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    let mut skips: Vec<Skip> = Vec::new();

    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let name = file.file_name().map_or_else(
            || file.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let cases = parse_dat(&text);

        // DOCUMENTED SCOPE GAP. The suite's `scripted/` directory is the set of
        // blocks html5lib-tests marks as requiring an implementation that
        // *executes* scripts: every one of them is `#script-on`, and every one
        // asserts a tree only `document.write` or a DOM mutation performed by
        // the script itself can produce (HTML § 13.2.6.4 hands a `script`
        // element to § 4.12.1's execution steps, which need a JavaScript
        // engine). This crate is a parser, not a user agent, and says so in its
        // crate documentation; these blocks are counted as skips, never passes.
        if is_scripted(file) {
            skips.push(Skip {
                reason: "tree-construction/scripted/: blocks whose expected tree is produced by \
                         executing a script (document.write, or a script-driven DOM mutation), \
                         which needs the JavaScript engine this crate deliberately does not have",
                cases: cases.len(),
            });
            eprintln!("scripted/{name}: {} case(s) skipped", cases.len());
            continue;
        }

        let mut file_passed = 0_usize;
        for case in &cases {
            total += 1;
            let actual = run_case(case);
            if actual.trim_end_matches('\n') == case.document.trim_end_matches('\n') {
                passed += 1;
                file_passed += 1;
            } else if failures.len() < 40 {
                failures.push(format!(
                    "{name}: input {:?}{}\n--- expected ---\n{}--- actual ---\n{}",
                    case.data,
                    case.fragment_context
                        .as_ref()
                        .map_or(String::new(), |context| format!(" (fragment: {context})")),
                    case.document,
                    actual
                ));
            }
        }
        eprintln!("{name}: {file_passed}/{} passed", cases.len());
    }

    eprintln!("html5lib tree construction: {passed}/{total} passed");
    for skip in &skips {
        eprintln!("  skipped {} case(s): {}", skip.cases, skip.reason);
    }
    for failure in &failures {
        eprintln!("{failure}");
    }

    // The bar is deliberately expressed as a ratio so that adding new upstream
    // tests cannot turn into a spurious failure, while a real regression in the
    // parser still trips it.
    //
    // At the time of writing the parser passes all 1792 runnable blocks,
    // including the four `<selectedcontent>` blocks of `webkit02.dat` that
    // used to be a known gap: the content they expect inside
    // `<selectedcontent>` is produced by HTML's "clone an option into a
    // selectedcontent" algorithm, which the `option` and `selectedcontent`
    // elements run from their own insertion/connection/pop reactions — DOM
    // behaviour that the tree builder now reproduces (see
    // `trees/oxixml-html/src/tree/select.rs`) even though it happens
    // logically *alongside* rather than inside § 13.2.6 tree construction
    // proper. The floor is ratcheted to the full ratio to lock that in: any
    // regression at all should trip this assertion.
    let ratio = passed as f64 / total as f64;
    assert!(
        ratio >= 1.0,
        "html5lib tree construction pass rate dropped to {passed}/{total}"
    );
}

/// Every document in the suite, fed to the parser in chunks, has to produce
/// the same tree as the same document parsed in one piece.
///
/// The conformance test above proves the parser is right about two thousand
/// documents; this one proves that being handed those documents a few bytes at
/// a time changes nothing at all — which is the whole promise of
/// [`StreamingParser`], and the only way to be sure that neither the
/// tokenizer's resumption points nor the insertion modes' handling of a
/// character run depend on where a chunk happens to end. The suite is the
/// right corpus for it precisely because it is a catalogue of the constructs
/// the two stages find hard.
#[test]
fn streaming_matches_one_shot_over_the_suite() {
    let Some(dir) = suite_dir() else {
        eprintln!(
            "skipping: neither OXIXML_HTML5LIB_TESTS_DIR nor OXIXML_XML_SUITES_DIR \
             resolves to an html5lib tree construction suite"
        );
        return;
    };

    let files = dat_files(&dir);
    assert!(!files.is_empty(), "no .dat files under {}", dir.display());

    let mut compared = 0_usize;
    let mut failures: Vec<String> = Vec::new();

    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let name = file.file_name().map_or_else(
            || file.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        // Unlike the conformance walk, the `scripted/` blocks are worth
        // running here: nothing is compared against an expected tree, only the
        // chunked parse against the one-shot parse of the same bytes, and
        // whatever those blocks make the parser do it has to do identically
        // either way.
        for case in parse_dat(&text) {
            let expected = run_case(&case);
            // One byte at a time is the extreme, where every character
            // reaches tree construction in a token of its own; the others
            // land chunk boundaries on a different offset of every construct.
            for size in [1, 2, 3, 7] {
                compared += 1;
                let actual = run_case_streamed(&case, size);
                if actual != expected && failures.len() < 20 {
                    failures.push(format!(
                        "{name}: {size} byte chunks diverged for {:?}\n\
                         --- one shot ---\n{expected}--- streamed ---\n{actual}",
                        case.data
                    ));
                }
            }
        }
    }

    eprintln!("html5lib streaming differential: {compared} chunked parses compared");
    for failure in &failures {
        eprintln!("{failure}");
    }
    assert!(
        failures.is_empty(),
        "{} chunked parse(s) did not match the one-shot parse",
        failures.len()
    );
}
