//! W3C SPARQL syntax test-suite harness.
//!
//! The suites are discovered through the `OXIXML_TESTSUITE_DIR` environment variable, which must
//! point at the root of a checkout of the `oxigraph` test suite (the directory containing
//! `rdf-tests/`). When the variable is unset the tests silently succeed, so the crate stays
//! testable without any external data.
//!
//! Three properties are checked for every manifest entry:
//!
//! * positive syntax tests must parse;
//! * negative syntax tests must be rejected;
//! * every positive test must round-trip, i.e. `parse(text)` and
//!   `parse(parse(text).to_string())` must yield the very same algebra.

use oxixml_sparql_syntax::SparqlParser;
use std::env::var_os;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

/// The kind of a manifest entry that this harness knows how to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    PositiveQuery,
    NegativeQuery,
    PositiveUpdate,
    NegativeUpdate,
}

impl Kind {
    fn from_manifest_type(name: &str) -> Option<Self> {
        match name {
            "mf:PositiveSyntaxTest" | "mf:PositiveSyntaxTest11" => Some(Self::PositiveQuery),
            "mf:NegativeSyntaxTest" | "mf:NegativeSyntaxTest11" => Some(Self::NegativeQuery),
            "mf:PositiveUpdateSyntaxTest" | "mf:PositiveUpdateSyntaxTest11" => {
                Some(Self::PositiveUpdate)
            }
            "mf:NegativeUpdateSyntaxTest" | "mf:NegativeUpdateSyntaxTest11" => {
                Some(Self::NegativeUpdate)
            }
            _ => None,
        }
    }

    fn is_positive(self) -> bool {
        matches!(self, Self::PositiveQuery | Self::PositiveUpdate)
    }

    fn is_update(self) -> bool {
        matches!(self, Self::PositiveUpdate | Self::NegativeUpdate)
    }
}

/// One runnable manifest entry: the file to parse and how it must behave.
struct Entry {
    kind: Kind,
    path: PathBuf,
}

/// Skips ASCII whitespace and `#` comments starting at `i`.
fn skip_trivia(bytes: &[u8], mut i: usize) -> usize {
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            return i;
        }
    }
}

/// Returns the index just past the end of the Turtle string literal starting at `start`
/// (`bytes[start]` is `"` or `'`), honouring both the short and the triple-quoted long form and
/// backslash escapes. An unterminated literal runs to the end of the buffer rather than panicking
/// — this is a best-effort scanner, not a validator.
fn string_literal_end(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let long = start + 2 < bytes.len() && bytes[start + 1] == quote && bytes[start + 2] == quote;
    let mut i = start + if long { 3 } else { 1 };
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == quote {
            if !long {
                return i + 1;
            }
            if i + 2 < bytes.len() && bytes[i + 1] == quote && bytes[i + 2] == quote {
                return i + 3;
            }
        }
        i += 1;
    }
    bytes.len()
}

/// Returns the index just past the closing `>` of the IRIREF starting at `start`
/// (`bytes[start]` is `<`). Turtle's IRIREF grammar excludes raw `<` and `>` from its body, so
/// the first `>` found always ends it; an unterminated IRIREF runs to the end of the buffer.
fn iri_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() && bytes[i] != b'>' {
        i += 1;
    }
    if i < bytes.len() { i + 1 } else { i }
}

/// Reads the token starting at `i`.
///
/// A bare word is a run of non-whitespace, non-`;`, non-`,`, non-quote, non-`<` bytes. A quoted
/// string literal (`"..."`, `'...'`, or their triple-quoted long forms) and an IRIREF (`<...>`)
/// are each read as a single token spanning the whole literal, so that whitespace-separated words
/// *inside* them — such as a bare `a` in a human-readable `rdfs:comment`/`mf:name` — are never
/// mistaken for the `a`/`rdf:type` keyword (or any other token this scanner looks for), and a
/// `;`/`,`/`#` inside an IRI does not truncate or comment out the rest of it.
fn token_at(bytes: &[u8], i: usize) -> (&str, usize) {
    if i >= bytes.len() {
        return ("", i);
    }
    let end = match bytes[i] {
        b'"' | b'\'' => string_literal_end(bytes, i),
        b'<' => iri_end(bytes, i),
        _ => {
            let mut end = i;
            while end < bytes.len()
                && !bytes[end].is_ascii_whitespace()
                && !matches!(bytes[end], b';' | b',' | b'"' | b'\'' | b'<')
            {
                end += 1;
            }
            end
        }
    };
    (std::str::from_utf8(&bytes[i..end]).unwrap_or_default(), end)
}

/// Extracts the runnable syntax-test entries of a `manifest.ttl`.
///
/// This is a deliberately small, format-specific scan rather than a Turtle parser: the suite
/// manifests always state `rdf:type <kind>` before the matching `mf:action <file>`. It still
/// tokenizes string literals and IRIs as opaque units (see [`token_at`]) so that a description
/// string carrying incidental keyword-shaped words — most commonly a bare `a` — cannot be
/// mistaken for Turtle syntax and desynchronize the scan.
fn parse_manifest(manifest: &Path) -> Vec<Entry> {
    let Ok(text) = read_to_string(manifest) else {
        return Vec::new();
    };
    let dir = manifest.parent().unwrap_or(Path::new("."));
    let bytes = text.as_bytes();
    let mut entries = Vec::new();
    let mut pending: Option<Kind> = None;
    let mut i = 0;
    while i < bytes.len() {
        i = skip_trivia(bytes, i);
        if i >= bytes.len() {
            break;
        }
        let (token, next) = token_at(bytes, i);
        if token.is_empty() {
            i += 1;
            continue;
        }
        match token {
            "rdf:type" | "a" => {
                let j = skip_trivia(bytes, next);
                let (type_name, after) = token_at(bytes, j);
                pending = Kind::from_manifest_type(type_name);
                i = after;
                continue;
            }
            "mf:action" => {
                let j = skip_trivia(bytes, next);
                let (target, after) = token_at(bytes, j);
                if let Some(kind) = pending.take()
                    && let Some(file) = target.strip_prefix('<').and_then(|t| t.strip_suffix('>'))
                    && !file.is_empty()
                    && !file.contains(':')
                {
                    entries.push(Entry {
                        kind,
                        path: dir.join(file),
                    });
                }
                i = after;
                continue;
            }
            _ => {}
        }
        i = next.max(i + 1);
    }
    entries
}

/// Accumulated suite outcome.
#[derive(Default)]
struct Report {
    positive_ok: usize,
    positive_failed: Vec<String>,
    negative_ok: usize,
    negative_failed: Vec<String>,
    roundtrip_ok: usize,
    roundtrip_failed: Vec<String>,
}

impl Report {
    fn run(&mut self, entry: &Entry) {
        let Ok(text) = read_to_string(&entry.path) else {
            return;
        };
        let name = entry.path.display().to_string();
        let parser = SparqlParser::new()
            .with_base_iri("http://example.com/base")
            .ok();
        let parser = parser.unwrap_or_default();
        if entry.kind.is_update() {
            match parser.parse_update(&text) {
                Ok(update) => {
                    if entry.kind.is_positive() {
                        self.positive_ok += 1;
                        let printed = update.to_string();
                        match parser.parse_update(&printed) {
                            Ok(again) if again == update => self.roundtrip_ok += 1,
                            Ok(again) => self.roundtrip_failed.push(format!(
                                "{name}: algebra differs\n      first: {printed}\n     second: {again}"
                            )),
                            Err(e) => self.roundtrip_failed.push(format!("{name}: {e}\n      printed: {printed}")),
                        }
                    } else {
                        self.negative_failed
                            .push(format!("{name}: wrongly accepted"));
                    }
                }
                Err(e) => {
                    if entry.kind.is_positive() {
                        self.positive_failed.push(format!("{name}: {e}"));
                    } else {
                        self.negative_ok += 1;
                    }
                }
            }
        } else {
            match parser.parse_query(&text) {
                Ok(query) => {
                    if entry.kind.is_positive() {
                        self.positive_ok += 1;
                        let printed = query.to_string();
                        match parser.parse_query(&printed) {
                            Ok(again) if again == query => self.roundtrip_ok += 1,
                            Ok(again) => self.roundtrip_failed.push(format!(
                                "{name}: algebra differs\n      first: {printed}\n     second: {again}"
                            )),
                            Err(e) => self.roundtrip_failed.push(format!("{name}: {e}\n      printed: {printed}")),
                        }
                    } else {
                        self.negative_failed
                            .push(format!("{name}: wrongly accepted"));
                    }
                }
                Err(e) => {
                    if entry.kind.is_positive() {
                        self.positive_failed.push(format!("{name}: {e}"));
                    } else {
                        self.negative_ok += 1;
                    }
                }
            }
        }
    }

    fn assert_clean(&self, label: &str) {
        let mut problems = String::new();
        for failure in self
            .positive_failed
            .iter()
            .chain(&self.negative_failed)
            .chain(&self.roundtrip_failed)
        {
            problems.push_str("  - ");
            problems.push_str(failure);
            problems.push('\n');
        }
        assert!(
            problems.is_empty(),
            "{label}: {} positive OK, {} negative OK, {} round-trips OK; failures:\n{problems}",
            self.positive_ok,
            self.negative_ok,
            self.roundtrip_ok
        );
        println!(
            "{label}: {} positive, {} negative, {} round-trips — all OK",
            self.positive_ok, self.negative_ok, self.roundtrip_ok
        );
    }
}

/// Collects the manifests of a suite directory (the directory itself plus its sub-directories).
fn manifests_of(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let direct = root.join("manifest.ttl");
    if direct.is_file() {
        out.push(direct);
    }
    if let Ok(dir) = root.read_dir() {
        let mut children: Vec<_> = dir.flatten().map(|e| e.path()).collect();
        children.sort();
        for child in children {
            if child.is_dir() {
                let manifest = child.join("manifest.ttl");
                if manifest.is_file() {
                    out.push(manifest);
                }
            }
        }
    }
    out
}

fn run_suite(sub_path: &str, label: &str) {
    let Some(root) = var_os("OXIXML_TESTSUITE_DIR") else {
        return;
    };
    let root = PathBuf::from(root).join(sub_path);
    if !root.is_dir() {
        return;
    }
    let mut report = Report::default();
    for manifest in manifests_of(&root) {
        for entry in parse_manifest(&manifest) {
            report.run(&entry);
        }
    }
    report.assert_clean(label);
}

#[test]
fn sparql11_syntax_suite() {
    run_suite("rdf-tests/sparql/sparql11", "SPARQL 1.1 syntax");
}

#[test]
fn sparql10_syntax_suite() {
    run_suite("rdf-tests/sparql/sparql10", "SPARQL 1.0 syntax");
}

/// SPARQL 1.2 needs the `sparql-12` feature: triple terms, reifiers and `VERSION` are simply
/// not part of the grammar without it.
#[cfg(feature = "sparql-12")]
#[test]
fn sparql12_syntax_suite() {
    run_suite("rdf-tests/sparql/sparql12", "SPARQL 1.2 syntax");
}

#[cfg(test)]
mod scanner_tests {
    use super::*;

    #[test]
    fn token_at_reads_a_short_quoted_string_as_a_single_token() {
        let bytes = b"\"a description with a bare a inside\" tail";
        let (token, end) = token_at(bytes, 0);
        assert_eq!(token, "\"a description with a bare a inside\"");
        assert_eq!(&bytes[end..], b" tail");
    }

    #[test]
    fn token_at_reads_a_triple_quoted_string_as_a_single_token() {
        let bytes = b"\"\"\"a triple \"quoted\" a string\"\"\" a";
        let (token, end) = token_at(bytes, 0);
        assert_eq!(token, "\"\"\"a triple \"quoted\" a string\"\"\"");
        assert_eq!(&bytes[end..], b" a");
    }

    #[test]
    fn token_at_honours_backslash_escapes_inside_a_string() {
        let bytes = b"\"escaped \\\" quote a\" tail";
        let (token, end) = token_at(bytes, 0);
        assert_eq!(token, "\"escaped \\\" quote a\"");
        assert_eq!(&bytes[end..], b" tail");
    }

    #[test]
    fn token_at_reads_an_iriref_as_a_single_token_across_semicolons_and_hashes() {
        let bytes = b"<http://example.com/a;b,c#frag> next";
        let (token, end) = token_at(bytes, 0);
        assert_eq!(token, "<http://example.com/a;b,c#frag>");
        assert_eq!(&bytes[end..], b" next");
    }

    #[test]
    fn token_at_still_reads_a_bare_predicate_a_as_its_own_token() {
        let bytes = b"a mf:PositiveSyntaxTest11";
        let (token, end) = token_at(bytes, 0);
        assert_eq!(token, "a");
        assert_eq!(&bytes[end..], b" mf:PositiveSyntaxTest11");
    }

    /// Regression test for the manifest-scanner bug: a bare `a` inside a quoted
    /// `rdfs:comment`/`mf:name` used to be mistaken for the `rdf:type` predicate shorthand,
    /// which cleared the `pending` kind before the real `mf:action` was reached and silently
    /// dropped the entry. With the fix, the whole quoted description is one opaque token and the
    /// entry survives.
    #[test]
    fn parse_manifest_keeps_entries_whose_description_contains_a_bare_a() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-sparql-syntax-suite-scanner-test-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let manifest = dir.join("manifest.ttl");
        std::fs::write(
            &manifest,
            concat!(
                "@prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .\n",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n",
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n",
                "\n",
                ":previously-dropped rdf:type mf:PositiveSyntaxTest11 ;\n",
                "    mf:name \"uses a bare a mid-sentence, like a real description does\" ;\n",
                "    rdfs:comment \"this deletes a thing using a wildcard, as a precaution\" ;\n",
                "    mf:action <previously-dropped.rq> .\n",
                "\n",
                ":control a mf:NegativeSyntaxTest11 ;\n",
                "    mf:action <control.rq> .\n",
            ),
        )
        .expect("scratch manifest");

        let entries = parse_manifest(&manifest);
        assert_eq!(
            entries.len(),
            2,
            "both entries must survive a description containing a bare `a`"
        );
        assert_eq!(entries[0].kind, Kind::PositiveQuery);
        assert!(entries[0].path.ends_with("previously-dropped.rq"));
        assert_eq!(entries[1].kind, Kind::NegativeQuery);
        assert!(entries[1].path.ends_with("control.rq"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
