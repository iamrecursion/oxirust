//! `oxixml query` — evaluate an XPath expression or an XQuery query and print the result
//! sequence.

use std::io::{Read, Write};

use oxixml_fn::xmlio::{SerializationParameters, serialize};
use oxixml_infoset::{ParseOptions, Sequence, Tree};
use oxixml_xpath::{StaticContext, XPath, XPathVersion};
use oxixml_xquery::{DynamicQueryContext, StaticQueryContext, XQuery};

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

use super::doc;

const HELP: &str = "\
oxixml query — evaluate an XPath expression or an XQuery query

USAGE:
    oxixml query --xpath <EXPR> [--version <VERSION>] [--ns <PREFIX=URI>]... [OPTIONS] [INPUT]
    oxixml query --xquery <FILE> [--ns <PREFIX=URI>]... [OPTIONS] [INPUT]

ARGUMENTS:
    [INPUT]    The document to evaluate against, or `-` to read it from stdin. May be omitted
               for an expression or query that needs no context item (e.g. `2 + 2`).

OPTIONS:
    --xpath <EXPR>        The XPath expression to evaluate, given directly on the command line.
    --xquery <FILE>        The XQuery query to evaluate, read from a file (or `-` for stdin).
    --version <VERSION>   The XPath language level: 1.0, 2.0 or 3.1 [default: 3.1]. Only valid
                           with --xpath (XQuery is always evaluated as XQuery 3.1).
    --ns <PREFIX=URI>      Binds a namespace prefix for use in the expression or query. May be
                           repeated.
    --catalog <FILE>       Resolve external references through this OASIS XML Catalog.
    --xinclude              Resolve <xi:include> elements in [INPUT] before evaluating.
    --base <IRI>             The base IRI used to resolve relative references.
    --quiet, --verbose      Control informational output on stderr.
    -h, --help              Print this help and exit.

Exactly one of --xpath or --xquery must be given. The result sequence is printed with the
`fn:serialize` XML output method: a node prints as its serialized XML, an atomic value prints as
its string form, and consecutive items are separated by a single space.

EXIT STATUS:
    0    the expression evaluated successfully.
    1    the command line was invalid.
    2    the expression did not compile, the input did not parse, or evaluation raised a
         dynamic error.
";

const SUBCOMMAND: &str = "query";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("xpath"),
        FlagSpec::value("xquery"),
        FlagSpec::value("version"),
        FlagSpec::value("ns"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml query`.
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
    let globals = match global::extract(&parsed, global::Capabilities::DOCUMENT) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, SUBCOMMAND, error),
    };
    match (parsed.value("xpath"), parsed.value("xquery")) {
        (None, None) => {
            return report::usage_error(stderr, SUBCOMMAND, "expected one of --xpath or --xquery");
        }
        (Some(_), Some(_)) => {
            return report::usage_error(
                stderr,
                SUBCOMMAND,
                "--xpath and --xquery may not both be given",
            );
        }
        _ => {}
    }
    if parsed.has("version") && parsed.has("xquery") {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--version may only be used with --xpath",
        );
    }
    let mut namespaces = Vec::new();
    for raw in parsed.all_values("ns") {
        match args::split_key_value(raw) {
            Ok((prefix, uri)) => namespaces.push((prefix.to_owned(), uri.to_owned())),
            Err(message) => {
                return report::usage_error(stderr, SUBCOMMAND, format!("--ns {message}"));
            }
        }
    }
    let input = match parsed.positionals() {
        [] => None,
        [single] => Some(single.clone()),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected at most one input path"),
    };

    let result = if let Some(expr) = parsed.value("xpath") {
        let version = match parsed.value("version") {
            None => XPathVersion::XPath31,
            Some("1.0") => XPathVersion::XPath10,
            Some("2.0") => XPathVersion::XPath20,
            Some("3.1") => XPathVersion::XPath31,
            Some(other) => {
                return report::usage_error(
                    stderr,
                    SUBCOMMAND,
                    format!("--version expects 1.0, 2.0 or 3.1, found `{other}`"),
                );
            }
        };
        report::info(
            stderr,
            globals.verbosity.is_verbose(),
            format!(
                "evaluating XPath `{expr}` against {}",
                context_label(input.as_deref())
            ),
        );
        run_xpath(
            expr,
            version,
            &namespaces,
            input.as_deref(),
            &globals,
            stdin,
        )
    } else {
        let query_path = parsed.value("xquery").expect("checked above");
        report::info(
            stderr,
            globals.verbosity.is_verbose(),
            format!(
                "evaluating XQuery `{}` against {}",
                io_util::display_path(query_path),
                context_label(input.as_deref())
            ),
        );
        run_xquery(query_path, &namespaces, input.as_deref(), &globals, stdin)
    };

    match result {
        Ok(text) => {
            let mut text = text;
            if !text.ends_with('\n') {
                text.push('\n');
            }
            match crate::io_util::write_output(None, text.as_bytes(), stdout) {
                Ok(()) => crate::exit::OK,
                Err(error) => report::processing_error(
                    stderr,
                    SUBCOMMAND,
                    format!("cannot write output: {error}"),
                ),
            }
        }
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

/// A short description of the context item for a `--verbose` progress line.
fn context_label(input: Option<&str>) -> String {
    match input {
        Some(path) => format!("`{}`", io_util::display_path(path)),
        None => "no context item".to_owned(),
    }
}

/// Parses and evaluates `input` (when given) into a context [`Tree`], honoring `--xinclude`.
fn load_context(
    input: Option<&str>,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<Option<Tree>, String> {
    let Some(input) = input else {
        return Ok(None);
    };
    let catalog = doc::load_catalog(globals, stdin)?;
    let base = globals
        .base
        .clone()
        .or_else(|| io_util::file_base_uri(input));
    let text = if globals.xinclude {
        let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
        document.to_xml()
    } else {
        io_util::read_string(input, stdin)
            .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?
    };
    let mut options = ParseOptions::default();
    if let Some(base) = base {
        options = options.document_uri(base);
    }
    let tree = Tree::parse_str_with(&text, options)
        .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
    Ok(Some(tree))
}

fn run_xpath(
    expression: &str,
    version: XPathVersion,
    namespaces: &[(String, String)],
    input: Option<&str>,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<String, String> {
    let mut static_context = StaticContext::new();
    for (prefix, uri) in namespaces {
        static_context = static_context.with_namespace(prefix.clone(), uri.clone());
    }
    let compiled = XPath::compile(expression, version, static_context)
        .map_err(|error| format!("XPath error: {error}"))?;

    let tree = load_context(input, globals, stdin)?;
    let sequence: Sequence = match &tree {
        Some(tree) => compiled.evaluate(tree.root()),
        None => compiled.evaluate_without_context(),
    }
    .map_err(|error| format!("XPath error: {error}"))?;

    serialize(&sequence, &SerializationParameters::default())
        .map_err(|error| format!("cannot serialize the result: {error}"))
}

fn run_xquery(
    query_path: &str,
    namespaces: &[(String, String)],
    input: Option<&str>,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<String, String> {
    let query_text = io_util::read_string(query_path, stdin).map_err(|error| {
        format!(
            "cannot read `{}`: {error}",
            io_util::display_path(query_path)
        )
    })?;
    let mut static_context = StaticQueryContext::new();
    for (prefix, uri) in namespaces {
        static_context = static_context.with_namespace(prefix.clone(), uri.clone());
    }
    let compiled = XQuery::compile(&query_text, static_context)
        .map_err(|error| format!("XQuery error: {error}"))?;

    let tree = load_context(input, globals, stdin)?;
    let mut dynamic = DynamicQueryContext::new();
    if let Some(tree) = &tree {
        dynamic = dynamic.with_context_item(tree.root());
    }
    let sequence = compiled
        .execute(&dynamic)
        .map_err(|error| format!("XQuery error: {error}"))?;

    serialize(&sequence, &SerializationParameters::default())
        .map_err(|error| format!("cannot serialize the result: {error}"))
}
