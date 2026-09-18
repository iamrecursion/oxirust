//! `oxixml validate` — validate an instance document against a DTD, an XML Schema, a RELAX NG
//! schema (XML or compact syntax) or a Schematron schema.

use std::io::{Read, Write};
use std::path::Path;

use oxixml_catalog::Catalog;
use oxixml_dom::Document;
use oxixml_dtd::{DocumentValidator, FilesystemResolver};
use oxixml_relaxng::{RelaxNg, RelaxNgBuilder};
use oxixml_schema::SchemaSet;
use oxixml_schematron::Schematron;

use crate::args::{self, FlagSpec};
use crate::catalog_support;
use crate::diagnostics::Diagnostic;
use crate::global;
use crate::io_util;
use crate::report;

use super::doc;

const HELP: &str = "\
oxixml validate — validate an XML document against a schema

USAGE:
    oxixml validate --xsd <SCHEMA>... [OPTIONS] <INPUT>
    oxixml validate --dtd <DTD> [OPTIONS] <INPUT>
    oxixml validate --rng <SCHEMA> [OPTIONS] <INPUT>
    oxixml validate --rnc <SCHEMA> [OPTIONS] <INPUT>
    oxixml validate --schematron <SCHEMA> [--phase <NAME>] [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The instance document to validate, or `-` to read it from stdin.

OPTIONS:
    --xsd <SCHEMA>         Validate against a W3C XML Schema document. May be repeated to
                            assemble a schema from several documents.
    --dtd <DTD>            Validate against an external DTD, replacing any DOCTYPE the
                            instance itself declares.
    --rng <SCHEMA>         Validate against a RELAX NG schema in the XML syntax.
    --rnc <SCHEMA>         Validate against a RELAX NG schema in the compact syntax.
    --schematron <SCHEMA>  Validate against a Schematron schema.
    --phase <NAME>         The Schematron phase to activate (only with --schematron).
    --catalog <FILE>       Resolve external references through this OASIS XML Catalog.
    --xinclude              Resolve <xi:include> elements in <INPUT> before validating.
    --base <IRI>            The base IRI used to resolve relative references.
    --quiet, --verbose      Control informational output on stderr.
    -h, --help              Print this help and exit.

Exactly one of --xsd, --dtd, --rng, --rnc or --schematron must be given. Every failed
constraint is printed to stdout as a structured diagnostic.

EXIT STATUS:
    0    the document is valid.
    1    the command line was invalid.
    2    the input or the schema could not be read, or the schema itself is invalid.
    3    the document is not valid; see stdout for the diagnostics.
";

const SUBCOMMAND: &str = "validate";
const KIND_FLAGS: [&str; 5] = ["xsd", "dtd", "rng", "rnc", "schematron"];

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("xsd"),
        FlagSpec::value("dtd"),
        FlagSpec::value("rng"),
        FlagSpec::value("rnc"),
        FlagSpec::value("schematron"),
        FlagSpec::value("phase"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml validate`.
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
    match parsed.count_present(&KIND_FLAGS) {
        0 => {
            return report::usage_error(
                stderr,
                SUBCOMMAND,
                "expected one of --xsd, --dtd, --rng, --rnc, --schematron",
            );
        }
        1 => {}
        _ => {
            return report::usage_error(
                stderr,
                SUBCOMMAND,
                "only one of --xsd, --dtd, --rng, --rnc, --schematron may be given",
            );
        }
    }
    if parsed.has("phase") && !parsed.has("schematron") {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--phase may only be used with --schematron",
        );
    }
    let input = match parsed.positionals() {
        [single] => single.as_str(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };
    let kind = KIND_FLAGS
        .iter()
        .find(|name| parsed.has(name))
        .copied()
        .unwrap_or("?");
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "validating `{}` against --{kind}",
            io_util::display_path(input)
        ),
    );

    match execute(&parsed, input, &globals, stdin) {
        Ok(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stdout, "{diagnostic}");
            }
            if diagnostics.is_empty() {
                if !globals.verbosity.is_quiet() {
                    let _ = writeln!(stdout, "{}: valid", io_util::display_path(input));
                }
                crate::exit::OK
            } else {
                if !globals.verbosity.is_quiet() {
                    let _ = writeln!(
                        stdout,
                        "{}: {} error(s)",
                        io_util::display_path(input),
                        diagnostics.len()
                    );
                }
                crate::exit::VALIDATION_FAILURE
            }
        }
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

fn execute(
    parsed: &args::Parsed,
    input: &str,
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<Vec<Diagnostic>, String> {
    let label = io_util::display_path(input).to_owned();
    let catalog = doc::load_catalog(globals, stdin)?;

    if let Some(dtd_path) = parsed.value("dtd") {
        let text = io_util::read_string(input, stdin)
            .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
        return validate_dtd(dtd_path, &text, &label);
    }

    if parsed.has("xsd") {
        let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
        return validate_xsd(
            &parsed.all_values("xsd"),
            &document,
            &label,
            catalog.as_ref(),
            stdin,
        );
    }

    if let Some(schema_path) = parsed.value("rng") {
        let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
        return validate_relaxng(
            schema_path,
            false,
            &document,
            &label,
            catalog.as_ref(),
            stdin,
        );
    }

    if let Some(schema_path) = parsed.value("rnc") {
        let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;
        return validate_relaxng(
            schema_path,
            true,
            &document,
            &label,
            catalog.as_ref(),
            stdin,
        );
    }

    if let Some(schema_path) = parsed.value("schematron") {
        let text = io_util::read_string(input, stdin)
            .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
        return validate_schematron(schema_path, parsed.value("phase"), &text, &label, stdin);
    }

    unreachable!("run() already checked that exactly one schema kind flag was given")
}

// --- XML Schema ----------------------------------------------------------------------------

fn validate_xsd(
    schema_paths: &[&str],
    instance: &Document,
    label: &str,
    catalog: Option<&Catalog>,
    stdin: &mut dyn Read,
) -> Result<Vec<Diagnostic>, String> {
    let mut set = SchemaSet::new().with_resolver(Box::new(
        catalog_support::SchemaCatalogResolver::new(catalog.cloned()),
    ));
    for schema_path in schema_paths {
        if *schema_path == io_util::STDIN {
            let text = io_util::read_string(schema_path, stdin)
                .map_err(|error| format!("cannot read schema: {error}"))?;
            set.add_document(None, &text)
                .map_err(|error| format!("schema error: {error}"))?;
        } else {
            set.add_file(schema_path)
                .map_err(|error| format!("cannot read schema `{schema_path}`: {error}"))?;
        }
    }
    let schema = set
        .compile()
        .map_err(|error| format!("schema error: {error}"))?;
    let outcome = schema.validate_document(instance);
    Ok(outcome
        .errors
        .iter()
        .map(|error| {
            Diagnostic::error(label, format!("{}: {}", error.path(), error.message()))
                .with_code(error.code())
        })
        .collect())
}

// --- DTD -------------------------------------------------------------------------------------

fn validate_dtd(
    dtd_path: &str,
    instance_text: &str,
    label: &str,
) -> Result<Vec<Diagnostic>, String> {
    if dtd_path == io_util::STDIN {
        return Err("--dtd cannot be `-`: the external DTD must be a real file".to_owned());
    }
    let dtd_path = Path::new(dtd_path);
    if !dtd_path.is_file() {
        return Err(format!("cannot read `{}`: not a file", dtd_path.display()));
    }
    let directory = dtd_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = dtd_path
        .file_name()
        .ok_or_else(|| format!("`{}` has no file name", dtd_path.display()))?
        .to_string_lossy()
        .into_owned();
    if file_name.contains('"') {
        return Err(format!(
            "`{}` cannot be used as a SYSTEM identifier (its name contains a `\"`)",
            dtd_path.display()
        ));
    }

    let root_name = document_element_name(instance_text)?;
    let injected = inject_doctype(instance_text, &root_name, &file_name)?;
    let resolver = FilesystemResolver::new(directory);
    let mut validator = DocumentValidator::new().with_resolver(Box::new(resolver));
    let report = validator
        .validate_str(&injected)
        .map_err(|error| format!("{error}"))?;
    Ok(report
        .errors()
        .iter()
        .map(|error| Diagnostic::error(label, error.to_string()))
        .collect())
}

/// Reads just far enough into `text` to find the qualified name of the document element.
fn document_element_name(text: &str) -> Result<String, String> {
    use oxixml_xml::events::Event;
    use oxixml_xml::reader::Reader;

    let mut reader = Reader::from_str(text);
    loop {
        match reader
            .read_event()
            .map_err(|error| format!("cannot read the document element: {error}"))?
        {
            Event::Start(start) | Event::Empty(start) => {
                return std::str::from_utf8(start.name().as_ref())
                    .map(str::to_owned)
                    .map_err(|error| error.to_string());
            }
            Event::Eof => return Err("the document has no element".to_owned()),
            _ => {}
        }
    }
}

/// Rewrites `text` so that its document type declaration is exactly
/// `<!DOCTYPE root_name SYSTEM "system_id">`, dropping whatever declaration (if any) the
/// document already had. This is what lets [`DocumentValidator`] — which reads the DTD from the
/// document's own declaration — validate against a DTD supplied on the command line instead.
fn inject_doctype(text: &str, root_name: &str, system_id: &str) -> Result<String, String> {
    use oxixml_xml::events::Event;
    use oxixml_xml::reader::Reader;

    let mut reader = Reader::from_str(text);
    let mut prolog_end = 0_usize;
    loop {
        let position_before = reader.buffer_position() as usize;
        match reader
            .read_event()
            .map_err(|error| format!("cannot read the document prolog: {error}"))?
        {
            Event::Decl(_) => prolog_end = reader.buffer_position() as usize,
            Event::Start(_) | Event::Empty(_) => {
                let prefix = &text[..prolog_end];
                let suffix = &text[position_before..];
                return Ok(format!(
                    "{prefix}<!DOCTYPE {root_name} SYSTEM \"{system_id}\">\n{suffix}"
                ));
            }
            Event::Eof => return Err("the document has no element".to_owned()),
            _ => {}
        }
    }
}

// --- RELAX NG --------------------------------------------------------------------------------

fn validate_relaxng(
    schema_path: &str,
    compact: bool,
    instance: &Document,
    label: &str,
    catalog: Option<&Catalog>,
    stdin: &mut dyn Read,
) -> Result<Vec<Diagnostic>, String> {
    let text = io_util::read_string(schema_path, stdin).map_err(|error| {
        format!(
            "cannot read schema `{}`: {error}",
            io_util::display_path(schema_path)
        )
    })?;
    let mut builder = RelaxNgBuilder::new().resolver(Box::new(
        catalog_support::RelaxNgCatalogResolver::new(catalog.cloned()),
    ));
    if let Some(base) = io_util::file_base_uri(schema_path) {
        builder = builder.base_uri(base);
    }
    let schema: RelaxNg = if compact {
        builder.compile_compact(&text)
    } else {
        builder.compile_xml(&text)
    }
    .map_err(|error| format!("schema error: {error}"))?;
    match schema.validate_document(instance) {
        Ok(()) => Ok(Vec::new()),
        Err(errors) => Ok(errors
            .into_iter()
            .map(|error| Diagnostic::error(label, error.to_string()))
            .collect()),
    }
}

// --- Schematron --------------------------------------------------------------------------------

fn validate_schematron(
    schema_path: &str,
    phase: Option<&str>,
    instance_text: &str,
    label: &str,
    stdin: &mut dyn Read,
) -> Result<Vec<Diagnostic>, String> {
    let schema_text = io_util::read_string(schema_path, stdin).map_err(|error| {
        format!(
            "cannot read schema `{}`: {error}",
            io_util::display_path(schema_path)
        )
    })?;
    let compiled = Schematron::compile(&schema_text, phase)
        .map_err(|error| format!("schema error: {error}"))?;
    let report = compiled
        .validate(instance_text)
        .map_err(|error| format!("validation error: {error}"))?;
    Ok(report
        .failed_asserts()
        .map(|check| {
            let mut diagnostic =
                Diagnostic::error(label, format!("{}: {}", check.location, check.message));
            if let Some(id) = &check.id {
                diagnostic = diagnostic.with_code(id.clone());
            }
            diagnostic
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_element_name_finds_a_prefixed_root() {
        let name = document_element_name(r#"<?xml version="1.0"?><a:root xmlns:a="urn:x"/>"#)
            .expect("finds the root");
        assert_eq!(name, "a:root");
    }

    #[test]
    fn inject_doctype_drops_any_existing_declaration() {
        let injected = inject_doctype(
            "<?xml version=\"1.0\"?><!DOCTYPE old [<!ENTITY x \"y\">]><root/>",
            "root",
            "external.dtd",
        )
        .expect("injects");
        assert!(injected.contains("<!DOCTYPE root SYSTEM \"external.dtd\">"));
        assert!(!injected.contains("old"));
        assert!(injected.ends_with("<root/>"));
    }

    #[test]
    fn inject_doctype_works_without_a_prolog() {
        let injected = inject_doctype("<root/>", "root", "external.dtd").expect("injects");
        assert_eq!(injected, "<!DOCTYPE root SYSTEM \"external.dtd\">\n<root/>");
    }
}
