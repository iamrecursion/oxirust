//! `oxixml canon` — XML canonicalization of a document, or RDF dataset canonicalization
//! (RDFC-1.0) of an RDF graph.

use std::io::{Read, Write};

use oxixml_c14n::{InclusiveNamespaces, NodeSet, c14n_1_0, c14n_1_1, exclusive_c14n};
use oxixml_canon::canonicalize_quads;
use oxixml_io::RdfParser;

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

use super::doc;
use super::rdf_format;

const HELP: &str = "\
oxixml canon — canonicalize an XML document or an RDF dataset

USAGE:
    oxixml canon --algorithm <ALGORITHM> [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The document to canonicalize, or `-` to read it from stdin.

ALGORITHMS:
    c14n        Canonical XML 1.0
    c14n11      Canonical XML 1.1
    exc-c14n    Exclusive XML Canonicalization 1.0
    rdfc10      RDF Dataset Canonicalization 1.0 (URDNA2015) — <INPUT> is RDF, not XML.

OPTIONS:
    --algorithm <ALGORITHM>  One of the algorithms above.
    --with-comments          Keep comments in the output (c14n/c14n11/exc-c14n only).
    --from <FORMAT>          The RDF format of <INPUT> (rdfc10 only): turtle, trig, ntriples,
                              nquads, rdfxml, jsonld, trix or n3. Detected from <INPUT>'s file
                              extension when omitted, defaulting to N-Quads.
    --catalog <FILE>         Resolve external references through this OASIS XML Catalog
                              (XML algorithms only).
    --xinclude                Resolve <xi:include> elements before canonicalizing (XML
                              algorithms only).
    --base <IRI>               The base IRI used to resolve relative references.
    --quiet, --verbose         Control informational output on stderr.
    -h, --help                 Print this help and exit.

EXIT STATUS:
    0    canonicalization succeeded.
    1    the command line was invalid.
    2    the input could not be read or parsed.
";

const SUBCOMMAND: &str = "canon";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    C14n10,
    C14n11,
    ExcC14n,
    Rdfc10,
}

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("algorithm"),
        FlagSpec::flag("with-comments"),
        FlagSpec::value("from"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml canon`.
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
    let Some(algorithm_name) = parsed.value("algorithm") else {
        return report::usage_error(stderr, SUBCOMMAND, "expected --algorithm <ALGORITHM>");
    };
    let algorithm = match algorithm_name {
        "c14n" => Algorithm::C14n10,
        "c14n11" => Algorithm::C14n11,
        "exc-c14n" => Algorithm::ExcC14n,
        "rdfc10" => Algorithm::Rdfc10,
        other => {
            return report::usage_error(
                stderr,
                SUBCOMMAND,
                format!("unknown algorithm `{other}` (expected c14n, c14n11, exc-c14n or rdfc10)"),
            );
        }
    };
    if parsed.has("with-comments") && algorithm == Algorithm::Rdfc10 {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--with-comments does not apply to --algorithm rdfc10",
        );
    }
    if parsed.has("from") && algorithm != Algorithm::Rdfc10 {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--from only applies to --algorithm rdfc10",
        );
    }
    if algorithm == Algorithm::Rdfc10 && (globals.xinclude || globals.catalog.is_some()) {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--xinclude and --catalog do not apply to --algorithm rdfc10",
        );
    }
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };
    let with_comments = parsed.has("with-comments");
    let from = parsed.value("from").map(str::to_owned);
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "canonicalizing `{}` with --algorithm {algorithm_name}",
            io_util::display_path(&input)
        ),
    );

    let outcome = match algorithm {
        Algorithm::Rdfc10 => canonicalize_rdf(&input, from.as_deref(), &globals, stdin),
        _ => canonicalize_xml(algorithm, &input, with_comments, &globals, stdin),
    };
    match outcome {
        Ok(bytes) => match crate::io_util::write_output(None, &bytes, stdout) {
            Ok(()) => crate::exit::OK,
            Err(error) => report::processing_error(
                stderr,
                SUBCOMMAND,
                format!("cannot write output: {error}"),
            ),
        },
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

fn canonicalize_xml(
    algorithm: Algorithm,
    input: &str,
    with_comments: bool,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<Vec<u8>, String> {
    let catalog = doc::load_catalog(globals, stdin)?;
    let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
    let node_set = NodeSet::document(&document);
    let bytes = match algorithm {
        Algorithm::C14n10 => c14n_1_0(&document, &node_set, with_comments),
        Algorithm::C14n11 => c14n_1_1(&document, &node_set, with_comments),
        Algorithm::ExcC14n => exclusive_c14n(
            &document,
            &node_set,
            &InclusiveNamespaces::none(),
            with_comments,
        ),
        Algorithm::Rdfc10 => unreachable!("handled by canonicalize_rdf"),
    };
    bytes.map_err(|error| format!("canonicalization error: {error}"))
}

fn canonicalize_rdf(
    input: &str,
    from: Option<&str>,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<Vec<u8>, String> {
    let format = match from {
        Some(name) => rdf_format::parse(name)?,
        None => rdf_format::from_path(input).unwrap_or(oxixml_io::RdfFormat::NQuads),
    };
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let mut parser = RdfParser::from_format(format);
    if let Some(base) = &globals.base {
        parser = parser
            .with_base_iri(base)
            .map_err(|error| format!("invalid --base: {error}"))?;
    }
    let quads = parser
        .for_reader(bytes.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
    let result =
        canonicalize_quads(quads).map_err(|error| format!("canonicalization error: {error}"))?;
    Ok(result.canonical_n_quads().as_bytes().to_vec())
}
