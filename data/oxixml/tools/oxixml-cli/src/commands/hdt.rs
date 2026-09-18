//! `oxixml hdt` — build, dump, or report statistics about an HDT (Header-Dictionary-Triples)
//! file.

use std::fmt::Write as _;
use std::io::{Read, Write};

use oxixml_hdt::{Hdt, HdtWriter};
use oxixml_io::RdfParser;
use oxixml_model::Triple;

use crate::args::{self, FlagSpec};
use crate::global;
use crate::io_util;
use crate::report;

use super::rdf_format;

const HELP: &str = "\
oxixml hdt — build, dump, or report statistics about an HDT file

USAGE:
    oxixml hdt build --output <FILE> [--from <FORMAT>] [OPTIONS] <INPUT>
    oxixml hdt dump [OPTIONS] <INPUT.hdt>
    oxixml hdt stats [OPTIONS] <INPUT.hdt>

SUBCOMMANDS:
    build    Read RDF triples and write them out as an HDT file.
    dump     Print every triple in an HDT file as N-Triples.
    stats    Print the triple, term and dictionary counts of an HDT file.

Run `oxixml hdt <subcommand> --help` for that subcommand's own options.
";

const BUILD_HELP: &str = "\
oxixml hdt build — read RDF triples and write them out as an HDT file

USAGE:
    oxixml hdt build [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The RDF document to read, or `-` to read it from stdin.

OPTIONS:
    --output <FILE>    Write the HDT file here instead of stdout.
    --from <FORMAT>    The RDF format of <INPUT>: turtle, trig, ntriples, nquads, rdfxml,
                       jsonld, trix or n3. Detected from <INPUT>'s file extension when
                       omitted, defaulting to N-Triples.
    --base <IRI>        The base IRI used to resolve relative IRIs.
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

<INPUT> must place every triple in the default graph: HDT has no notion of named graphs.

EXIT STATUS:
    0    the HDT file was written successfully.
    1    the command line was invalid.
    2    the input could not be read or parsed, or contained a named graph.
";

const DUMP_HELP: &str = "\
oxixml hdt dump — print every triple in an HDT file as N-Triples

USAGE:
    oxixml hdt dump [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The HDT file to read, or `-` to read it from stdin.

OPTIONS:
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

EXIT STATUS:
    0    the file was read and its triples printed successfully.
    1    the command line was invalid.
    2    the file could not be read or is not a valid HDT file.
";

const STATS_HELP: &str = "\
oxixml hdt stats — print the triple, term and dictionary counts of an HDT file

USAGE:
    oxixml hdt stats [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The HDT file to read, or `-` to read it from stdin.

OPTIONS:
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

EXIT STATUS:
    0    the file was read and its statistics printed successfully.
    1    the command line was invalid.
    2    the file could not be read or is not a valid HDT file.
";

const SUBCOMMAND: &str = "hdt";

/// Runs `oxixml hdt`.
pub fn run(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    match args.first().map(String::as_str) {
        Some("build") => run_build(&args[1..], stdin, stdout, stderr),
        Some("dump") => run_dump(&args[1..], stdin, stdout, stderr),
        Some("stats") => run_stats(&args[1..], stdin, stdout, stderr),
        Some("--help" | "-h") | None => {
            let _ = write!(stdout, "{HELP}");
            crate::exit::OK
        }
        Some(other) => report::usage_error(
            stderr,
            SUBCOMMAND,
            format!("unknown subcommand `{other}` (expected build, dump or stats)"),
        ),
    }
}

fn build_flags() -> Vec<FlagSpec> {
    let mut specs = vec![FlagSpec::value("output"), FlagSpec::value("from")];
    specs.extend_from_slice(global::FLAGS);
    specs
}

fn run_build(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let specs = build_flags();
    let parsed = match args::parse(args, &specs) {
        Ok(parsed) => parsed,
        Err(error) => return report::usage_error(stderr, "hdt build", error),
    };
    if parsed.has("help") {
        let _ = write!(stdout, "{BUILD_HELP}");
        return crate::exit::OK;
    }
    let globals = match global::extract(&parsed, global::Capabilities::BASE_ONLY) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, "hdt build", error),
    };
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, "hdt build", "expected an input path"),
        _ => return report::usage_error(stderr, "hdt build", "expected exactly one input path"),
    };
    let output = parsed.value("output").map(str::to_owned);
    let from = parsed.value("from").map(str::to_owned);
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "building an HDT file from `{}`",
            io_util::display_path(&input)
        ),
    );

    match execute_build(&input, from.as_deref(), globals.base.as_deref(), stdin) {
        Ok(bytes) => match io_util::write_output(output.as_deref(), &bytes, stdout) {
            Ok(()) => crate::exit::OK,
            Err(error) => report::processing_error(
                stderr,
                "hdt build",
                format!("cannot write output: {error}"),
            ),
        },
        Err(message) => report::processing_error(stderr, "hdt build", message),
    }
}

fn execute_build(
    input: &str,
    from: Option<&str>,
    base: Option<&str>,
    stdin: &mut dyn Read,
) -> Result<Vec<u8>, String> {
    let format = match from {
        Some(name) => rdf_format::parse(name)?,
        None => rdf_format::from_path(input).unwrap_or(oxixml_io::RdfFormat::NTriples),
    };
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let mut parser = RdfParser::from_format(format);
    if let Some(base) = base {
        parser = parser
            .with_base_iri(base)
            .map_err(|error| format!("invalid --base: {error}"))?;
    }
    let mut triples = Vec::new();
    for quad in parser.for_reader(bytes.as_slice()) {
        let quad = quad.map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
        if !quad.graph_name.is_default_graph() {
            return Err(format!(
                "`{}` places a triple in a named graph, which HDT cannot represent",
                io_util::display_path(input)
            ));
        }
        triples.push(Triple {
            subject: quad.subject,
            predicate: quad.predicate,
            object: quad.object,
        });
    }
    let mut output = Vec::new();
    HdtWriter::new()
        .write(&mut output, triples)
        .map_err(|error| format!("cannot build the HDT file: {error}"))?;
    Ok(output)
}

fn run_dump(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let specs = global::FLAGS.to_vec();
    let parsed = match args::parse(args, &specs) {
        Ok(parsed) => parsed,
        Err(error) => return report::usage_error(stderr, "hdt dump", error),
    };
    if parsed.has("help") {
        let _ = write!(stdout, "{DUMP_HELP}");
        return crate::exit::OK;
    }
    let globals = match global::extract(&parsed, global::Capabilities::NONE) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, "hdt dump", error),
    };
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, "hdt dump", "expected an input path"),
        _ => return report::usage_error(stderr, "hdt dump", "expected exactly one input path"),
    };
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!("dumping `{}`", io_util::display_path(&input)),
    );
    match execute_dump(&input, stdin) {
        Ok(text) => match io_util::write_output(None, text.as_bytes(), stdout) {
            Ok(()) => crate::exit::OK,
            Err(error) => report::processing_error(
                stderr,
                "hdt dump",
                format!("cannot write output: {error}"),
            ),
        },
        Err(message) => report::processing_error(stderr, "hdt dump", message),
    }
}

fn execute_dump(input: &str, stdin: &mut dyn Read) -> Result<String, String> {
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let hdt = Hdt::from_bytes(&bytes)
        .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
    let mut out = String::new();
    for triple in hdt.triples() {
        let triple =
            triple.map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
        let _ = writeln!(out, "{triple} .");
    }
    Ok(out)
}

fn run_stats(
    args: &[String],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let specs = global::FLAGS.to_vec();
    let parsed = match args::parse(args, &specs) {
        Ok(parsed) => parsed,
        Err(error) => return report::usage_error(stderr, "hdt stats", error),
    };
    if parsed.has("help") {
        let _ = write!(stdout, "{STATS_HELP}");
        return crate::exit::OK;
    }
    let globals = match global::extract(&parsed, global::Capabilities::NONE) {
        Ok(globals) => globals,
        Err(error) => return report::usage_error(stderr, "hdt stats", error),
    };
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, "hdt stats", "expected an input path"),
        _ => return report::usage_error(stderr, "hdt stats", "expected exactly one input path"),
    };
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "reading statistics from `{}`",
            io_util::display_path(&input)
        ),
    );
    match execute_stats(&input, stdin) {
        Ok(text) => match io_util::write_output(None, text.as_bytes(), stdout) {
            Ok(()) => crate::exit::OK,
            Err(error) => report::processing_error(
                stderr,
                "hdt stats",
                format!("cannot write output: {error}"),
            ),
        },
        Err(message) => report::processing_error(stderr, "hdt stats", message),
    }
}

fn execute_stats(input: &str, stdin: &mut dyn Read) -> Result<String, String> {
    let bytes = io_util::read_bytes(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let hdt = Hdt::from_bytes(&bytes)
        .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?;
    let dictionary = hdt.dictionary();
    let mut out = String::new();
    let _ = writeln!(out, "triples: {}", hdt.len());
    let _ = writeln!(out, "subjects: {}", dictionary.subject_count());
    let _ = writeln!(out, "predicates: {}", dictionary.predicate_count());
    let _ = writeln!(out, "objects: {}", dictionary.object_count());
    let _ = writeln!(
        out,
        "shared subject-object terms: {}",
        dictionary.shared_count()
    );
    let _ = writeln!(out, "header bytes: {}", hdt.header().len());
    Ok(out)
}
