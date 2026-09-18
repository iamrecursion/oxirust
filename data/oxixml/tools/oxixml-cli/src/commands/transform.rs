//! `oxixml transform` — apply an XSLT stylesheet to a source document.

use std::io::{Read, Write};
use std::sync::Arc;

use oxixml_infoset::{AtomicValue, ParseOptions, Tree};
use oxixml_xslt::{Parameters, StaticOptions, Stylesheet};

use crate::args::{self, FlagSpec};
use crate::catalog_support;
use crate::global;
use crate::io_util;
use crate::report;

use super::doc;

const HELP: &str = "\
oxixml transform — apply an XSLT stylesheet to a source document

USAGE:
    oxixml transform --xsl <STYLESHEET> [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The source document to transform, or `-` to read it from stdin.

OPTIONS:
    --xsl <STYLESHEET>   The XSLT stylesheet to apply, or `-` to read it from stdin.
    --param <NAME=VALUE>  Binds a stylesheet parameter (as a string). May be repeated.
    --output <FILE>       Write the transformation result here instead of stdout.
    --catalog <FILE>      Resolve xsl:include/xsl:import/document() through this OASIS XML
                           Catalog.
    --xinclude             Resolve <xi:include> elements in <INPUT> before transforming.
    --base <IRI>           The base IRI used to resolve relative references in <INPUT>.
    --quiet, --verbose     Control informational output on stderr.
    -h, --help             Print this help and exit.

EXIT STATUS:
    0    the transformation completed and was written successfully.
    1    the command line was invalid.
    2    the stylesheet did not compile, the source did not parse, or the transformation
         raised a dynamic error.
";

const SUBCOMMAND: &str = "transform";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("xsl"),
        FlagSpec::value("param"),
        FlagSpec::value("output"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml transform`.
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
    let Some(xsl_path) = parsed.value("xsl") else {
        return report::usage_error(stderr, SUBCOMMAND, "expected --xsl <STYLESHEET>");
    };
    let mut params = Vec::new();
    for raw in parsed.all_values("param") {
        match args::split_key_value(raw) {
            Ok((name, value)) => params.push((name.to_owned(), value.to_owned())),
            Err(message) => {
                return report::usage_error(stderr, SUBCOMMAND, format!("--param {message}"));
            }
        }
    }
    let output = parsed.value("output").map(str::to_owned);
    let input = match parsed.positionals() {
        [single] => single.clone(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "applying `{}` to `{}` ({} parameter(s))",
            io_util::display_path(xsl_path),
            io_util::display_path(&input),
            params.len()
        ),
    );

    match execute(
        xsl_path,
        &input,
        &params,
        output.as_deref(),
        &globals,
        stdin,
        stdout,
    ) {
        Ok(()) => crate::exit::OK,
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

fn execute(
    xsl_path: &str,
    input: &str,
    params: &[(String, String)],
    output: Option<&str>,
    globals: &global::Options,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
) -> Result<(), String> {
    let catalog = doc::load_catalog(globals, stdin)?;

    let xsl_text = io_util::read_string(xsl_path, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(xsl_path)))?;
    let stylesheet_dir = io_util::parent_dir(xsl_path);
    let resolver = Arc::new(catalog_support::XsltCatalogResolver::new(
        catalog.clone(),
        stylesheet_dir,
    ));
    let mut static_options = StaticOptions::new().with_resolver(resolver);
    if let Some(base) = io_util::file_base_uri(xsl_path) {
        static_options = static_options.with_base_uri(base);
    }
    let compiled = Stylesheet::compile_str_with(&xsl_text, static_options)
        .map_err(|error| format!("stylesheet error: {error}"))?;

    let source_base = globals
        .base
        .clone()
        .or_else(|| io_util::file_base_uri(input));
    let tree = if globals.xinclude {
        let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
        let mut options = ParseOptions::default();
        if let Some(base) = &source_base {
            options = options.document_uri(base.clone());
        }
        Tree::parse_str_with(&document.to_xml(), options)
            .map_err(|error| format!("cannot re-parse the XInclude-processed source: {error}"))?
    } else {
        let source_text = io_util::read_string(input, stdin)
            .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
        let mut options = ParseOptions::default();
        if let Some(base) = &source_base {
            options = options.document_uri(base.clone());
        }
        Tree::parse_str_with(&source_text, options)
            .map_err(|error| format!("{}: {error}", io_util::display_path(input)))?
    };

    let mut parameters = Parameters::new();
    for (name, value) in params {
        parameters = parameters.with_parameter(name.clone(), AtomicValue::string(value.clone()));
    }

    let transformation = compiled
        .transform_with(&tree.root(), &parameters)
        .map_err(|error| format!("transformation error: {error}"))?;

    io_util::write_output(output, transformation.as_str().as_bytes(), stdout)
        .map_err(|error| format!("cannot write output: {error}"))
}
