//! `oxixml sparql` — parse a SPARQL query and print its algebra.

use std::io::{Read, Write};

use oxixml_sparql_syntax::SparqlParser;

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

const HELP: &str = "\
oxixml sparql — parse a SPARQL query and print its algebra

USAGE:
    oxixml sparql --parse <QUERY> [OPTIONS]

OPTIONS:
    --parse <QUERY>     The SPARQL query text, given directly on the command line, or `-` to
                        read it from stdin.
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

The query's algebra is printed to stdout in the SSE (s-expression) notation, e.g.
`(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))`.

EXIT STATUS:
    0    the query parsed successfully.
    1    the command line was invalid.
    2    the query did not parse.
";

const SUBCOMMAND: &str = "sparql";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![FlagSpec::value("parse")];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml sparql`.
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
    if !parsed.positionals().is_empty() {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "oxixml sparql takes no positional arguments",
        );
    }
    let Some(query_arg) = parsed.value("parse") else {
        return report::usage_error(stderr, SUBCOMMAND, "expected --parse <QUERY>");
    };
    report::info(stderr, globals.verbosity.is_verbose(), "parsing the query");

    match execute(query_arg, stdin) {
        Ok(sse) => {
            let mut text = sse;
            text.push('\n');
            match io_util::write_output(None, text.as_bytes(), stdout) {
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

fn execute(query_arg: &str, stdin: &mut dyn Read) -> Result<String, String> {
    let text = if query_arg == io_util::STDIN {
        let mut buffer = String::new();
        stdin
            .read_to_string(&mut buffer)
            .map_err(|error| format!("cannot read the query from stdin: {error}"))?;
        buffer
    } else {
        query_arg.to_owned()
    };
    let query = SparqlParser::new()
        .parse_query(&text)
        .map_err(|error| format!("SPARQL syntax error: {error}"))?;
    Ok(query.to_sse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_literal_query_is_parsed_directly() {
        let mut stdin: &[u8] = &[];
        let sse = execute("SELECT ?s ?p ?o WHERE { ?s ?p ?o . }", &mut stdin).expect("parses");
        assert_eq!(sse, "(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))");
    }

    #[test]
    fn a_dash_reads_from_stdin() {
        let mut stdin: &[u8] = b"SELECT ?s WHERE { ?s ?p ?o . }";
        let sse = execute("-", &mut stdin).expect("parses");
        assert!(sse.contains("bgp"), "{sse}");
    }
}
