//! `oxixml svg` — SVG-aware lint, statistics, and pretty/minified re-serialization.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::io::{Read, Write};

use oxixml_dom::{NodeKind, SerializeOptions};
use oxixml_svg::error::ValueError;
use oxixml_svg::lint::diagnostic::Severity;
use oxixml_svg::lint::rules::Linter;
use oxixml_svg::names::elements::ElementId;
use oxixml_svg::ns::SVG_NAMESPACE;
use oxixml_svg::tree::document::SvgDocument;
use oxixml_svg::values::path::PathData;

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

const HELP: &str = "\
oxixml svg — SVG-aware lint, statistics, and pretty/minified re-serialization

USAGE:
    oxixml svg [OPTIONS] [INPUT]

ARGUMENTS:
    <INPUT>    The SVG document to process. A dash (`-`), or omitting <INPUT> entirely, reads
               it from stdin.

OPTIONS:
    --stats             Print a summary report (root width/height/viewBox, element counts by
                         kind, id/reference/path counts) instead of linting.
    --pretty            Pretty-print the document to stdout and exit.
    --min               Re-serialize the document compactly and exit.
    --quiet, --verbose  Control informational output on stderr; --quiet also suppresses
                        info-severity lint findings and the summary line below.
    -h, --help          Print this help and exit.

With none of --stats, --pretty or --min given, checks the document against every SVG-aware
lint rule this crate implements (unknown elements/attributes, value-grammar conformance,
duplicate/dangling/circular references, content-model violations, deprecated constructs) and
prints each finding to stdout, one per line, as `<severity>: <message> [<code>]`, followed by
a summary line `N error(s), M warning(s), K info(s)`.

--stats, --pretty and --min may be combined on the same command line; when more than one is
given, they take precedence in this order: --pretty, then --min, then --stats.

The document must be well-formed XML with an `{http://www.w3.org/2000/svg}svg` document
element — anything else is a processing error, not a lint finding.

EXIT STATUS:
    0    the document processed successfully (linting: no error-severity finding).
    1    the command line was invalid.
    2    the input could not be read, or is not a well-formed SVG document.
    3    linting found at least one error-severity finding.
";

const SUBCOMMAND: &str = "svg";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::flag("stats"),
        FlagSpec::flag("pretty"),
        FlagSpec::flag("min"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml svg`.
pub fn run(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let specs = flags();
    let parsed = match args::parse(args, &specs) {
        Ok(parsed) => parsed,
        Err(error) => return report::usage_error(stderr, SUBCOMMAND, error),
    };
    if parsed.has("help") {
        let _ = write!(stdout, "{HELP}");
        return crate::exit::OK;
    }
    let globals = match global::extract(&parsed, global::Capabilities::NONE) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, SUBCOMMAND, error),
    };
    let input = match parsed.positionals() {
        [] => io_util::STDIN,
        [single] => single.as_str(),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected at most one input path"),
    };
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!("processing `{}`", io_util::display_path(input)),
    );

    let text = match io_util::read_string(input, stdin) {
        Ok(text) => text,
        Err(error) => {
            return report::processing_error(
                stderr,
                SUBCOMMAND,
                format!("cannot read `{}`: {error}", io_util::display_path(input)),
            );
        }
    };
    let document = match SvgDocument::parse_str(&text) {
        Ok(document) => document,
        Err(error) => {
            return report::processing_error(
                stderr,
                SUBCOMMAND,
                format!("{}: {error}", io_util::display_path(input)),
            );
        }
    };

    if parsed.has("pretty") {
        return serialize_and_exit(&document, &SerializeOptions::pretty(), stdout, stderr);
    }
    if parsed.has("min") {
        return serialize_and_exit(&document, &SerializeOptions::new(), stdout, stderr);
    }
    if parsed.has("stats") {
        let report_text = collect_stats(&document);
        return match io_util::write_output(None, report_text.as_bytes(), stdout) {
            Ok(()) => crate::exit::OK,
            Err(error) => report::processing_error(
                stderr,
                SUBCOMMAND,
                format!("cannot write output: {error}"),
            ),
        };
    }
    execute_lint(&document, globals.verbosity.is_quiet(), stdout)
}

/// Serializes `document` with `options` to stdout and reports the outcome — the shared tail of
/// `--pretty` and `--min`, which differ only in which [`SerializeOptions`] they use.
fn serialize_and_exit(
    document: &SvgDocument,
    options: &SerializeOptions,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let mut text = document.to_xml_with(options);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    match io_util::write_output(None, text.as_bytes(), stdout) {
        Ok(()) => crate::exit::OK,
        Err(error) => {
            report::processing_error(stderr, SUBCOMMAND, format!("cannot write output: {error}"))
        }
    }
}

/// The default mode: runs [`Linter`] over `document` and prints its findings.
///
/// `quiet` suppresses info-severity findings from the printed list *and* the trailing summary
/// line — the exit code alone still reports the outcome in that case.
fn execute_lint(document: &SvgDocument, quiet: bool, stdout: &mut dyn Write) -> u8 {
    let diagnostics = Linter::new().lint(document.document());
    let mut errors = 0usize;
    let mut warnings = 0usize;
    let mut infos = 0usize;
    for diagnostic in &diagnostics {
        match diagnostic.severity() {
            Severity::Error => errors += 1,
            Severity::Warning => warnings += 1,
            Severity::Info => infos += 1,
        }
        if quiet && diagnostic.severity() == Severity::Info {
            continue;
        }
        let _ = writeln!(stdout, "{diagnostic}");
    }
    if !quiet {
        let _ = writeln!(
            stdout,
            "{errors} error(s), {warnings} warning(s), {infos} info(s)"
        );
    }
    if errors > 0 {
        crate::exit::VALIDATION_FAILURE
    } else {
        crate::exit::OK
    }
}

/// The `--stats` mode: a compact, best-effort report over the whole document, one line per
/// fact. Every count is a plain structural/lexical scan (see the individual field comments
/// below for exactly what each one measures) — this is a diagnostic aid, not a validator; use
/// the default lint mode for conformance checking.
fn collect_stats(document: &SvgDocument) -> String {
    let doc = document.document();
    let mut out = String::new();

    let _ = writeln!(out, "width: {}", format_optional(document.width()));
    let _ = writeln!(out, "height: {}", format_optional(document.height()));
    let _ = writeln!(out, "viewBox: {}", format_optional(document.view_box()));

    // Per-`ElementId` counts (SVG-namespace, recognized elements only) and the handful of
    // whole-document facts that need only one pass alongside them.
    let mut element_counts: HashMap<ElementId, usize> = HashMap::new();
    let mut total_elements = 0usize;
    let mut id_count = 0usize;
    let mut url_hash_refs = 0usize;
    let mut path_count = 0usize;
    let mut path_segments = 0usize;

    for node in doc.traverse(document.root()) {
        if doc.kind(node) != Some(NodeKind::Element) {
            continue;
        }
        total_elements += 1;

        // "ids": every element (any namespace) carrying a non-empty `id`, duplicates included —
        // the same document-wide scope `crate::tree::refs::IdIndex` uses, but as a raw count
        // rather than an index (duplicate detection is the lint layer's job, not this report's).
        if doc
            .get_attribute(node, "id")
            .is_some_and(|value| !value.is_empty())
        {
            id_count += 1;
        }

        // "url() references": a lexical scan of every attribute value on every element (any
        // namespace), not a semantic one — see `count_url_hash_refs`.
        for attr_id in doc.attributes(node) {
            if let Some(attr) = doc.attr(attr_id) {
                url_hash_refs += count_url_hash_refs(attr.value());
            }
        }

        let element = doc.element(node).expect(
            "Document::kind(node) == Some(NodeKind::Element) implies Document::element(node) is Some",
        );
        if element.namespace_uri() != Some(SVG_NAMESPACE) {
            continue;
        }
        let Some(kind) = ElementId::from_name(element.local_name()) else {
            continue;
        };
        *element_counts.entry(kind).or_insert(0) += 1;

        if kind == ElementId::Path {
            path_count += 1;
            if let Some(d) = doc.get_attribute(node, "d") {
                // Lenient: a malformed `d` still contributes whatever prefix parsed, matching
                // this report's "best-effort, never fails" character.
                let (path, _error) = PathData::parse_recover(d);
                path_segments += path.len();
            }
        }
    }

    let _ = writeln!(
        out,
        "elements: {total_elements} total (every namespace; the breakdown below covers \
         recognized SVG elements only, so it need not sum to this total)"
    );
    let mut ranked: Vec<(ElementId, usize)> = element_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
    let _ = writeln!(
        out,
        "element counts by kind (top 10 of {} recognized kinds):",
        ranked.len()
    );
    for (kind, count) in ranked.iter().take(10) {
        let _ = writeln!(out, "  {kind}: {count}");
    }
    let _ = writeln!(out, "ids: {id_count}");
    let _ = writeln!(out, "url() references: {url_hash_refs}");
    let _ = writeln!(out, "paths: {path_count}");
    let _ = writeln!(out, "path segments: {path_segments}");
    out
}

/// Formats a typed root attribute the way [`collect_stats`] reports it: `"(not set)"` when the
/// attribute is absent, `"(invalid: {error})"` when it is present but fails to parse, or the
/// value's own [`fmt::Display`] otherwise.
fn format_optional<T: fmt::Display>(value: Option<Result<T, ValueError>>) -> String {
    match value {
        None => "(not set)".to_owned(),
        Some(Ok(value)) => value.to_string(),
        Some(Err(error)) => format!("(invalid: {error})"),
    }
}

/// Counts `url(...)` funcIRI occurrences in `value` whose reference is a same-document
/// fragment (`#id`).
///
/// This is a lexical scan of the raw text, not a semantic one: it looks for every
/// case-insensitive occurrence of the `url(` keyword — matching how
/// [`oxixml_svg::values::iri`]'s own `<funciri>` grammar matches it — then, after skipping any
/// white space and at most one opening quote (`'` or `"`), checks whether a `#` follows. It
/// does not otherwise validate the reference (an unterminated or otherwise malformed
/// `url(...)` value still counts here if it starts this way), and — deliberately, for a
/// whole-document report — it is not restricted to attributes whose grammar is actually
/// `<funciri>`/`<paint>`.
fn count_url_hash_refs(value: &str) -> usize {
    let lower = value.to_ascii_lowercase();
    lower
        .match_indices("url(")
        .filter(|&(offset, _)| {
            let rest = value[offset + 4..].trim_start();
            let rest = rest
                .strip_prefix(|c: char| c == '"' || c == '\'')
                .unwrap_or(rest);
            rest.starts_with('#')
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_is_printed_without_touching_stdin() {
        let mut stdin: &[u8] = &[];
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let status = run(&["--help".to_owned()], &mut stdin, &mut stdout, &mut stderr);
        assert_eq!(status, crate::exit::OK);
        assert!(
            String::from_utf8(stdout)
                .expect("utf8")
                .contains("oxixml svg")
        );
        assert!(stderr.is_empty());
    }

    #[test]
    fn count_url_hash_refs_matches_the_unquoted_form() {
        assert_eq!(count_url_hash_refs("url(#a)"), 1);
        assert_eq!(count_url_hash_refs("fill: url(#a); stroke: url(#b)"), 2);
    }

    #[test]
    fn count_url_hash_refs_matches_quoted_and_whitespace_forms() {
        assert_eq!(count_url_hash_refs(r##"url("#a")"##), 1);
        assert_eq!(count_url_hash_refs("url('#a')"), 1);
        assert_eq!(count_url_hash_refs("url(  #a  )"), 1);
    }

    #[test]
    fn count_url_hash_refs_is_case_insensitive_on_the_keyword() {
        assert_eq!(count_url_hash_refs("URL(#a)"), 1);
        assert_eq!(count_url_hash_refs("UrL(#a)"), 1);
    }

    #[test]
    fn count_url_hash_refs_ignores_a_non_fragment_reference() {
        assert_eq!(count_url_hash_refs("url(other.svg)"), 0);
        assert_eq!(count_url_hash_refs("url(http://example.com/x.svg#a)"), 0);
    }

    #[test]
    fn count_url_hash_refs_finds_nothing_in_plain_text() {
        assert_eq!(count_url_hash_refs("red"), 0);
        assert_eq!(count_url_hash_refs(""), 0);
        assert_eq!(
            count_url_hash_refs("myurl(#a)"),
            1,
            "no word-boundary check, by design"
        );
    }

    #[test]
    fn format_optional_covers_absent_valid_and_invalid() {
        assert_eq!(format_optional::<u32>(None), "(not set)");
        assert_eq!(format_optional(Some(Ok(42_u32))), "42");
        let error = format_optional::<u32>(Some(Err(ValueError::Empty)));
        assert!(error.starts_with("(invalid: "), "{error}");
        assert!(error.contains("empty value"), "{error}");
    }
}
