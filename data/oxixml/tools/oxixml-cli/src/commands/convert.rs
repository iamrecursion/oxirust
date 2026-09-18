//! `oxixml convert` — RDF format conversion, or SPARQL query results format conversion.

use std::io::{Read, Write};

use oxixml_io::{RdfParser, RdfSerializer};
use oxixml_sparql_results::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, ReaderQueryResultsParserOutput,
};

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

use super::rdf_format;

const HELP: &str = "\
oxixml convert — convert between RDF serializations, or between SPARQL results serializations

USAGE:
    oxixml convert --from <FORMAT> --to <FORMAT> [OPTIONS] <INPUT>
    oxixml convert --results <FORMAT> [--from <FORMAT>] [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The document to convert, or `-` to read it from stdin.

OPTIONS:
    --from <FORMAT>       The input format: turtle, trig, ntriples, nquads, rdfxml, jsonld,
                          trix or n3 (RDF mode); xml or json (--results mode; defaults to
                          xml — CSV and TSV cannot be parsed back, only written).
    --to <FORMAT>         The output RDF format (RDF mode only; same names as --from).
    --results <FORMAT>    Switches to SPARQL results conversion: the output format, one of
                          xml, json, csv or tsv.
    --base <IRI>           The base IRI used to resolve relative IRIs (RDF mode only).
    --quiet, --verbose      Control informational output on stderr.
    -h, --help              Print this help and exit.

EXIT STATUS:
    0    the conversion completed and was written successfully.
    1    the command line was invalid.
    2    the input could not be read or did not parse.
";

const SUBCOMMAND: &str = "convert";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("from"),
        FlagSpec::value("to"),
        FlagSpec::value("results"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml convert`.
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
    let globals = match global::extract(&parsed, global::Capabilities::BASE_ONLY) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, SUBCOMMAND, error),
    };
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };

    let outcome = if let Some(results_to) = parsed.value("results") {
        if parsed.has("to") {
            return report::usage_error(
                stderr,
                SUBCOMMAND,
                "--to is not used together with --results",
            );
        }
        match (
            results_format(results_to),
            parsed
                .value("from")
                .map_or(Ok(QueryResultsFormat::Xml), results_format),
        ) {
            (Ok(to), Ok(from)) => {
                if matches!(from, QueryResultsFormat::Csv | QueryResultsFormat::Tsv) {
                    return report::usage_error(
                        stderr,
                        SUBCOMMAND,
                        "--from cannot be csv or tsv when converting SPARQL results: they are write-only",
                    );
                }
                report::info(
                    stderr,
                    globals.verbosity.is_verbose(),
                    format!(
                        "converting SPARQL results in `{}` from {from} to {to}",
                        io_util::display_path(&input)
                    ),
                );
                execute_results(&input, from, to, stdin)
            }
            (Err(message), _) | (_, Err(message)) => {
                return report::usage_error(stderr, SUBCOMMAND, message);
            }
        }
    } else {
        let (Some(from_name), Some(to_name)) = (parsed.value("from"), parsed.value("to")) else {
            return report::usage_error(stderr, SUBCOMMAND, "expected both --from and --to");
        };
        match (rdf_format::parse(from_name), rdf_format::parse(to_name)) {
            (Ok(from), Ok(to)) => {
                report::info(
                    stderr,
                    globals.verbosity.is_verbose(),
                    format!(
                        "converting `{}` from {from} to {to}",
                        io_util::display_path(&input)
                    ),
                );
                execute_rdf(&input, from, to, globals.base.as_deref(), stdin)
            }
            (Err(message), _) | (_, Err(message)) => {
                return report::usage_error(stderr, SUBCOMMAND, message);
            }
        }
    };

    match outcome {
        Ok(bytes) => match io_util::write_output(None, &bytes, stdout) {
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

fn results_format(name: &str) -> Result<QueryResultsFormat, String> {
    match name.to_ascii_lowercase().as_str() {
        "xml" => Ok(QueryResultsFormat::Xml),
        "json" => Ok(QueryResultsFormat::Json),
        "csv" => Ok(QueryResultsFormat::Csv),
        "tsv" => Ok(QueryResultsFormat::Tsv),
        other => Err(format!(
            "unknown SPARQL results format `{other}` (expected xml, json, csv or tsv)"
        )),
    }
}

fn execute_rdf(
    input: &str,
    from: oxixml_io::RdfFormat,
    to: oxixml_io::RdfFormat,
    base: Option<&str>,
    stdin: &mut dyn Read,
) -> Result<Vec<u8>, String> {
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let mut parser = RdfParser::from_format(from);
    if let Some(base) = base {
        parser = parser
            .with_base_iri(base)
            .map_err(|error| format!("invalid --base: {error}"))?;
    }
    let mut serializer = RdfSerializer::from_format(to).for_writer(Vec::new());
    for quad in parser.for_reader(bytes.as_slice()) {
        let quad = quad.map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
        serializer
            .serialize_quad(&quad)
            .map_err(|error| format!("cannot convert to {to}: {error}"))?;
    }
    serializer
        .finish()
        .map_err(|error| format!("cannot write output: {error}"))
}

fn execute_results(
    input: &str,
    from: QueryResultsFormat,
    to: QueryResultsFormat,
    stdin: &mut dyn Read,
) -> Result<Vec<u8>, String> {
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let parsed = QueryResultsParser::from_format(from)
        .for_reader(bytes.as_slice())
        .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
    match parsed {
        ReaderQueryResultsParserOutput::Boolean(value) => QueryResultsSerializer::from_format(to)
            .serialize_boolean_to_writer(Vec::new(), value)
            .map_err(|error| format!("cannot write output: {error}")),
        ReaderQueryResultsParserOutput::Solutions(solutions) => {
            let variables = solutions.variables().to_vec();
            let mut writer = QueryResultsSerializer::from_format(to)
                .serialize_solutions_to_writer(Vec::new(), variables)
                .map_err(|error| format!("cannot write output: {error}"))?;
            for solution in solutions {
                let solution = solution
                    .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
                writer
                    .serialize(solution.iter())
                    .map_err(|error| format!("cannot write output: {error}"))?;
            }
            writer
                .finish()
                .map_err(|error| format!("cannot write output: {error}"))
        }
    }
}
