//! `oxixml fmt` — pretty-print or canonically re-indent an XML document.

use std::io::{Read, Write};

use oxixml_dom::{SerializeOptions, XmlDeclaration};

use crate::args::{self, FlagSpec};
use crate::global;
use crate::report;

use super::doc;

const HELP: &str = "\
oxixml fmt — pretty-print / canonical-indent an XML document

USAGE:
    oxixml fmt [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The XML document to format, or `-` to read it from stdin.

OPTIONS:
    --indent <N>        The number of spaces per indentation level [default: 2].
                         May not be combined with --preserve-space.
    --in-place          Write the result back to <INPUT> instead of stdout.
                         Not valid with an <INPUT> of `-`.
    --preserve-space     Do not add or remove any whitespace: write the document compactly,
                         with no indentation at all. Useful for documents that rely on
                         significant whitespace throughout, where any reindentation would
                         change their meaning.
    --catalog <FILE>    Resolve external references through this OASIS XML Catalog.
    --xinclude          Resolve <xi:include> elements before formatting.
    --base <IRI>        The base IRI used to resolve relative references (defaults to
                         <INPUT>'s own location; has no default when reading from stdin).
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

EXIT STATUS:
    0    the document was formatted successfully.
    1    the command line was invalid.
    2    the input could not be read or is not well-formed XML.
";

const SUBCOMMAND: &str = "fmt";

fn flags() -> Vec<FlagSpec> {
    let mut specs = vec![
        FlagSpec::value("indent"),
        FlagSpec::flag("in-place"),
        FlagSpec::flag("preserve-space"),
    ];
    specs.extend_from_slice(global::FLAGS);
    specs
}

/// Runs `oxixml fmt`.
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
    if parsed.has("indent") && parsed.has("preserve-space") {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--indent may not be combined with --preserve-space",
        );
    }
    let indent: usize = match parsed.value("indent") {
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(_) => {
                return report::usage_error(
                    stderr,
                    SUBCOMMAND,
                    format!("--indent expects a non-negative integer, found `{text}`"),
                );
            }
        },
        None => 2,
    };
    let in_place = parsed.has("in-place");
    let preserve_space = parsed.has("preserve-space");
    let inputs = parsed.positionals();
    let input = match inputs {
        [single] => single.as_str(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };
    if in_place && input == crate::io_util::STDIN {
        return report::usage_error(
            stderr,
            SUBCOMMAND,
            "--in-place cannot be used when reading from stdin",
        );
    }
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!(
            "formatting `{}` (indent {indent}, preserve-space {preserve_space})",
            crate::io_util::display_path(input)
        ),
    );

    match execute(
        input,
        indent,
        in_place,
        preserve_space,
        &globals,
        stdin,
        stdout,
    ) {
        Ok(()) => crate::exit::OK,
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

fn execute(
    input: &str,
    indent: usize,
    in_place: bool,
    preserve_space: bool,
    globals: &global::Options,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
) -> Result<(), String> {
    let catalog = doc::load_catalog(globals, stdin)?;
    let document = doc::load_document(input, globals, catalog.as_ref(), true, stdin)?;

    let spaces = " ".repeat(indent);
    let declaration = if document.xml_version().is_some() {
        XmlDeclaration::Include
    } else {
        XmlDeclaration::Omit
    };
    let options = if preserve_space {
        SerializeOptions::new().with_xml_declaration(declaration)
    } else {
        SerializeOptions::new()
            .with_indent(Some(&spaces))
            .with_xml_declaration(declaration)
    };

    let mut text = document.to_xml_with(&options);
    if !text.ends_with('\n') {
        text.push('\n');
    }

    let target = in_place.then_some(input);
    crate::io_util::write_output(target, text.as_bytes(), stdout)
        .map_err(|error| format!("cannot write output: {error}"))
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
                .contains("oxixml fmt")
        );
    }
}
