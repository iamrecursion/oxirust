//! `oxixml-cli` — the `oxixml` command-line tool: a single, `xmllint`/`xsltproc`/`rapper`-class
//! binary over the whole [OxiXML](https://github.com/cool-japan/oxixml) workspace.
//!
//! Part of the OxiXML workspace — the COOLJAPAN Pure-Rust foundation for all XML and RDF
//! processing. This crate wires ten subcommands to the sibling library crates that already
//! implement them; it contributes no parsing or processing logic of its own beyond command-line
//! ergonomics.
//!
//! # Subcommands
//!
//! | Subcommand | What it does | Backed by |
//! |---|---|---|
//! | `fmt` | Pretty-print / canonically re-indent an XML document | [`oxixml_dom`] |
//! | `lint` | Well-formedness check, `line:col` diagnostics | [`oxixml_sax`] |
//! | `validate` | Validate against a DTD, XSD, RELAX NG or Schematron schema | [`oxixml_dtd`], [`oxixml_schema`], [`oxixml_relaxng`], [`oxixml_schematron`] |
//! | `transform` | Apply an XSLT stylesheet | [`oxixml_xslt`] |
//! | `query` | Evaluate an XPath expression or an XQuery query | [`oxixml_xpath`], [`oxixml_xquery`] |
//! | `canon` | XML canonicalization, or RDF dataset canonicalization (RDFC-1.0) | [`oxixml_c14n`], [`oxixml_canon`] |
//! | `convert` | RDF format conversion, or SPARQL results format conversion | [`oxixml_io`], [`oxixml_sparql_results`] |
//! | `hdt` | Build, dump, or report statistics about an HDT file | [`oxixml_hdt`] |
//! | `sparql` | Parse a SPARQL query and print its algebra (SSE) | [`oxixml_sparql_syntax`] |
//! | `svg` | SVG-aware lint, statistics, and pretty/minified re-serialization | [`oxixml_svg`] |
//!
//! Run `oxixml <subcommand> --help` for a subcommand's full reference, or see the `HELP`
//! constant at the top of its module (under `commands`, an internal module) for the same text.
//!
//! # Conventions shared by every subcommand
//!
//! * An input path argument of `-` reads that input from stdin.
//! * **Exit codes**: `0` success; `1` the command line itself was invalid; `2` the input, schema
//!   or stylesheet could not be read or processed; `3` the requested check completed and found
//!   the input invalid (`validate`), not well-formed (`lint`), or containing at least one
//!   error-severity finding (`svg`).
//! * **Global flags**, accepted by whichever subcommands they are meaningful for: `--catalog
//!   <FILE>` (OASIS XML Catalog resolution), `--xinclude` (resolve `xi:include` first),
//!   `--base <IRI>` (the base IRI for relative references), `--quiet`/`--verbose`, and
//!   `--help`/`-h`.
//!
//! # Structure
//!
//! This crate has no logic of its own to speak of: [`run`] (and the numeric-status core it
//! wraps, [`run_status`]) parse the subcommand name out of `args` and dispatch to one of the
//! `commands` modules (an internal implementation detail, not part of the public API), each a
//! thin adapter from parsed command-line flags to the sibling crate's own API.
//! [`main`](https://doc.rust-lang.org/reference/crates-and-source-files.html#the-main-function)
//! itself (in `src/main.rs`) is a few lines that call [`run`] with the real process handles;
//! everything else is reachable — and tested — as an ordinary library call.
//!
//! # Example
//!
//! [`run_status`] is the testable core: it takes the same arguments [`run`] does but returns the
//! raw numeric exit status (`std::process::ExitCode` intentionally exposes little of its own
//! representation, which makes it awkward to assert on directly in a test).
//!
//! ```
//! use oxixml_cli::run_status;
//!
//! let args = vec!["fmt".to_owned(), "-".to_owned()];
//! let mut stdin: &[u8] = b"<a><b/></a>";
//! let mut stdout = Vec::new();
//! let mut stderr = Vec::new();
//!
//! let status = run_status(&args, &mut stdin, &mut stdout, &mut stderr);
//!
//! assert_eq!(status, 0);
//! assert_eq!(String::from_utf8(stdout).expect("utf8 output"), "<a>\n  <b/>\n</a>\n");
//! assert!(stderr.is_empty());
//! ```

#![deny(missing_docs)]

pub mod args;
mod catalog_support;
mod commands;
mod diagnostics;
mod exit;
mod global;
mod io_util;
mod report;

use std::io::{Read, Write};
use std::process::ExitCode;

const TOP_LEVEL_HELP: &str = "\
oxixml — the OxiXML command-line tool

USAGE:
    oxixml <SUBCOMMAND> [OPTIONS] [ARGS]
    oxixml --help | -h
    oxixml --version

SUBCOMMANDS:
    fmt         Pretty-print / canonical-indent an XML document.
    lint        Check that an XML document is well-formed.
    validate    Validate against a DTD, XML Schema, RELAX NG or Schematron schema.
    transform   Apply an XSLT stylesheet to a document.
    query       Evaluate an XPath expression or an XQuery query.
    canon       Canonicalize an XML document, or an RDF dataset (RDFC-1.0).
    convert     Convert between RDF serializations, or SPARQL results serializations.
    hdt         Build, dump, or report statistics about an HDT file.
    sparql      Parse a SPARQL query and print its algebra.
    svg         SVG-aware lint, statistics, and pretty/minified re-serialization.

GLOBAL OPTIONS (accepted by whichever subcommands they are meaningful for):
    --catalog <FILE>    Resolve external references through an OASIS XML Catalog.
    --xinclude          Resolve <xi:include> elements before processing.
    --base <IRI>        The base IRI used to resolve relative references.
    --quiet             Suppress informational summary lines.
    --verbose           Print additional progress information to stderr.
    --version           Print the version and exit.
    -h, --help          Print this help and exit. Combine with a subcommand
                        (`oxixml <SUBCOMMAND> --help`) for that subcommand's own reference.

An input path argument of `-` reads that input from stdin.

EXIT STATUS:
    0    success.
    1    the command line itself was invalid.
    2    the input, schema or stylesheet could not be read or processed.
    3    the requested check completed and found the input invalid or not well-formed.

Run 'oxixml <SUBCOMMAND> --help' for full details on any subcommand.
";

/// Runs the `oxixml` command line: parses the subcommand name out of `args` and dispatches to
/// it, returning the raw numeric exit status (`0`, `1`, `2` or `3` — see the
/// [crate-level documentation](crate)).
///
/// `args` holds the subcommand and its own arguments only (`std::env::args().skip(1)`, in a real
/// process) — never the program name itself.
///
/// This is the core [`run`] wraps into a [`std::process::ExitCode`] for `main`; it is exposed
/// directly because `ExitCode` does not support inspecting the value it carries, which makes the
/// numeric status the more convenient thing for a caller — notably a test — to assert on.
#[must_use]
pub fn run_status(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h") => {
            let _ = write!(stdout, "{TOP_LEVEL_HELP}");
            exit::OK
        }
        Some("--version") => {
            let _ = writeln!(stdout, "oxixml {}", env!("CARGO_PKG_VERSION"));
            exit::OK
        }
        Some("fmt") => commands::fmt::run(&args[1..], stdin, stdout, stderr),
        Some("lint") => commands::lint::run(&args[1..], stdin, stdout, stderr),
        Some("validate") => commands::validate::run(&args[1..], stdin, stdout, stderr),
        Some("transform") => commands::transform::run(&args[1..], stdin, stdout, stderr),
        Some("query") => commands::query::run(&args[1..], stdin, stdout, stderr),
        Some("canon") => commands::canon::run(&args[1..], stdin, stdout, stderr),
        Some("convert") => commands::convert::run(&args[1..], stdin, stdout, stderr),
        Some("hdt") => commands::hdt::run(&args[1..], stdin, stdout, stderr),
        Some("sparql") => commands::sparql::run(&args[1..], stdin, stdout, stderr),
        Some("svg") => commands::svg::run(&args[1..], stdin, stdout, stderr),
        Some(other) => {
            let _ = writeln!(stderr, "oxixml: error: unknown subcommand `{other}`");
            let _ = writeln!(stderr, "Try 'oxixml --help' for more information.");
            exit::USAGE_ERROR
        }
    }
}

/// Runs the `oxixml` command line and returns a [`std::process::ExitCode`] for `main` to return.
///
/// See the [crate-level documentation](crate) and [`run_status`] (the function this wraps) for
/// details.
#[must_use]
pub fn run(
    args: Vec<String>,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> ExitCode {
    exit::code(run_status(&args, stdin, stdout, stderr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_prints_top_level_help() {
        let mut stdin: &[u8] = &[];
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let status = run_status(&[], &mut stdin, &mut stdout, &mut stderr);
        assert_eq!(status, exit::OK);
        assert!(String::from_utf8(stdout).expect("utf8").contains("oxixml"));
        assert!(stderr.is_empty());
    }

    #[test]
    fn version_is_printed() {
        let mut stdin: &[u8] = &[];
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let status = run_status(
            &["--version".to_owned()],
            &mut stdin,
            &mut stdout,
            &mut stderr,
        );
        assert_eq!(status, exit::OK);
        let printed = String::from_utf8(stdout).expect("utf8");
        assert!(printed.starts_with("oxixml "), "{printed}");
    }

    #[test]
    fn an_unknown_subcommand_is_a_usage_error() {
        let mut stdin: &[u8] = &[];
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let status = run_status(&["bogus".to_owned()], &mut stdin, &mut stdout, &mut stderr);
        assert_eq!(status, exit::USAGE_ERROR);
        assert!(!stderr.is_empty());
    }
}
