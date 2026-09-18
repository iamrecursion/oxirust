//! `oxixml lint` — a well-formedness check, reporting `line:col` diagnostics.

use std::io::{Read, Write};
use std::sync::mpsc;
use std::time::Duration;

use oxixml_sax::{ErrorHandler, Handlers, SaxError, SaxParseException, XMLReader};

use crate::args::{self, FlagSpec};
use crate::diagnostics::Diagnostic;
use crate::global;
use crate::io_util;
use crate::report;

const HELP: &str = "\
oxixml lint — check that an XML document is well-formed

USAGE:
    oxixml lint [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    The XML document to check, or `-` to read it from stdin.

OPTIONS:
    --quiet, --verbose  Control informational output on stderr.
    -h, --help          Print this help and exit.

Checks every well-formedness constraint of XML 1.0/1.1 and of Namespaces in XML: document
structure (exactly one root element, no markup outside it), matching start/end tag pairs, the
`Char` and `Name` productions, attribute uniqueness, entity constraints, and more. Each problem
found is printed to stdout as `<path>:<line>:<col>: error: <message>`. A well-formedness
violation is fatal per XML 1.0 §1.2, so parsing stops at the first one: fix it and run again to
see the next.

EXIT STATUS:
    0    the document is well-formed.
    1    the command line was invalid.
    2    the input could not be read, or the check did not complete in time (an internal
         safety net — see the crate's rustdoc for `commands::lint` if this ever triggers).
    3    the document is not well-formed; see stdout for the diagnostics.
";

const SUBCOMMAND: &str = "lint";

/// How long [`execute`]'s check waits to finish before giving up on it and reporting a
/// processing error instead of blocking forever.
///
/// A normal check, of a well-formed or a not-well-formed document of any realistic size,
/// finishes in a small fraction of this budget; it exists purely as a defensive backstop (see
/// [`check_well_formed`]'s own documentation for why one is warranted here specifically).
const CHECK_TIMEOUT: Duration = Duration::from_secs(15);

fn flags() -> Vec<FlagSpec> {
    global::FLAGS.to_vec()
}

/// Runs `oxixml lint`.
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
        [single] => single.as_str(),
        [] => return report::usage_error(stderr, SUBCOMMAND, "expected an input path"),
        _ => return report::usage_error(stderr, SUBCOMMAND, "expected exactly one input path"),
    };
    report::info(
        stderr,
        globals.verbosity.is_verbose(),
        format!("checking `{}`", io_util::display_path(input)),
    );

    match execute(input, stdin, stdout) {
        Ok(true) => {
            if !globals.verbosity.is_quiet() {
                let _ = writeln!(stdout, "{}: well-formed", io_util::display_path(input));
            }
            crate::exit::OK
        }
        Ok(false) => {
            if !globals.verbosity.is_quiet() {
                let _ = writeln!(stdout, "{}: not well-formed", io_util::display_path(input));
            }
            crate::exit::VALIDATION_FAILURE
        }
        Err(message) => report::processing_error(stderr, SUBCOMMAND, message),
    }
}

/// Collects every parse warning/error the SAX reader reports, each paired with its position.
struct Collector {
    path: String,
    diagnostics: Vec<Diagnostic>,
}

impl Collector {
    fn record(&mut self, exception: &SaxParseException, build: fn(String, String) -> Diagnostic) {
        let locator = exception.locator();
        self.diagnostics.push(
            build(self.path.clone(), exception.message().to_owned())
                .at(locator.line_number(), locator.column_number()),
        );
    }
}

impl ErrorHandler for Collector {
    // Never called by `XMLReader` itself today (see the trait's own documentation: this crate
    // does not perform DTD validation, the usual source of a SAX2 warning) — implemented anyway
    // so that a handler layered on top of this one, or a future reader, has somewhere to send a
    // warning rather than it being silently dropped.
    fn warning(&mut self, exception: &SaxParseException) -> Result<(), SaxError> {
        self.record(exception, Diagnostic::warning);
        Ok(())
    }

    fn error(&mut self, exception: &SaxParseException) -> Result<(), SaxError> {
        self.record(exception, Diagnostic::error);
        Ok(())
    }

    fn fatal_error(&mut self, exception: &SaxParseException) -> Result<(), SaxError> {
        self.record(exception, Diagnostic::error);
        Ok(())
    }
}

/// Runs `work` on a worker thread, waiting at most `timeout` for it to finish.
///
/// On timeout, the worker thread is simply abandoned (there is no way to interrupt it
/// mid-computation) rather than joined; it either finishes on its own and finds nothing left to
/// report the result to, or is reclaimed along with every other thread when the process exits
/// shortly after `execute` reports the processing error this produces. `oxixml lint` is a
/// short-lived, one-shot process, so an abandoned worker never outlives anything that matters.
///
/// # Errors
///
/// A human-readable message if `timeout` elapses before `work` finishes.
fn with_timeout<T, F>(timeout: Duration, work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(work());
    });
    receiver
        .recv_timeout(timeout)
        .map_err(|_| "the well-formedness check did not complete in time".to_owned())
}

/// The check: the `line:col`-precise diagnostics `lint` reports, from [`XMLReader`] with
/// [`Config::check_well_formedness`][cwf] enabled — every well-formedness constraint XML 1.0/1.1
/// and Namespaces in XML define (document structure, the `Char`/`Name` productions, matching
/// start/end tags, attribute uniqueness, entity constraints, and more; see
/// [`Config::check_well_formedness`][cwf]'s own documentation for the exact list) — alongside
/// [`Config::check_comments`][cc], which that flag does not itself cover.
///
/// A well-formedness violation is a *fatal* error (XML 1.0 §1.2/§5.1: the processor must not
/// continue normal processing past one), and [`XMLReader`] enforces exactly that, so this
/// reports at most one error per run. Warnings, which are recoverable, would accumulate — but
/// this crate performs no DTD validation, so [`XMLReader`] never produces one today.
///
/// # History
///
/// This used to run in two phases, because `oxixml_xml`'s reader — with `check_well_formedness`
/// enabled — did not terminate on some unbalanced-tag inputs (a mismatched end tag, `<a><b></a>`,
/// and an unclosed one, `<a><b>`): every read past the first violation re-raised the identical
/// error at the identical position forever, rather than ever reaching `Eof`. A second phase, via
/// [`oxixml_dom::build::parse_str_with`] and
/// [`oxixml_dom::BuilderOptions::with_well_formedness_checks`], recovered the coverage this check
/// alone would have missed while `check_well_formedness` stayed off, through a code path the
/// reader defect did not reach. That defect is now fixed
/// (`oxixml_xml::reader::state::ReaderState::check_well_formedness` reaches `Eof` after reporting
/// the violation once — see its own history — rather than re-deriving it on every subsequent
/// read), so a single phase with the flag on now safely covers everything the two phases together
/// used to.
///
/// [cwf]: oxixml_xml::reader::Config::check_well_formedness
/// [cc]: oxixml_xml::reader::Config::check_comments
///
/// # Errors
///
/// The message [`with_timeout`] produces if `timeout` elapses first — kept as a defensive
/// backstop even though nothing in this workspace is known to trip it any more.
fn check_well_formed(
    text: String,
    path: String,
    timeout: Duration,
) -> Result<(bool, Vec<Diagnostic>), String> {
    with_timeout(timeout, move || {
        let mut reader = XMLReader::new();
        reader.xml_config_mut().check_comments = true;
        reader.xml_config_mut().check_well_formedness = true;
        let mut collector = Collector {
            path: path.clone(),
            diagnostics: Vec::new(),
        };
        let outcome = {
            let mut handlers = Handlers::new().with_error(&mut collector);
            reader.parse_str(&text, &mut handlers)
        };
        if collector.diagnostics.is_empty()
            && let Err(error) = outcome
        {
            collector
                .diagnostics
                .push(Diagnostic::error(path, error.to_string()));
        }
        let well_formed = !collector.diagnostics.iter().any(Diagnostic::is_error);
        (well_formed, collector.diagnostics)
    })
}

/// Returns `Ok(true)` when `input` is well-formed, `Ok(false)` with every diagnostic already
/// printed when it is not. A warning alone (see [`ErrorHandler::warning`]) does not fail the
/// check, but is still printed. See [`check_well_formed`] for the check this runs.
fn execute(input: &str, stdin: &mut dyn Read, stdout: &mut dyn Write) -> Result<bool, String> {
    let text = io_util::read_string(input, stdin)
        .map_err(|error| format!("cannot read `{}`: {error}", io_util::display_path(input)))?;
    let path = io_util::display_path(input).to_owned();

    let (well_formed, diagnostics) = check_well_formed(text, path, CHECK_TIMEOUT)?;

    for diagnostic in &diagnostics {
        let _ = writeln!(stdout, "{diagnostic}");
    }
    Ok(well_formed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_formed_input_reports_success() {
        let mut stdin: &[u8] = b"<a><b/></a>";
        let mut stdout = Vec::new();
        assert_eq!(execute("-", &mut stdin, &mut stdout), Ok(true));
        assert!(stdout.is_empty());
    }

    #[test]
    fn a_mismatched_tag_is_reported_with_a_position() {
        let mut stdin: &[u8] = b"<a><b></a>";
        let mut stdout = Vec::new();
        let result = execute("-", &mut stdin, &mut stdout);
        assert_eq!(result, Ok(false));
        let printed = String::from_utf8(stdout).expect("utf8");
        assert!(printed.contains("<stdin>:1:"), "{printed}");
    }

    /// The historical reproducer of the reader hang `check_well_formed`'s documentation
    /// describes: an unclosed element used to make the underlying reader re-derive the same
    /// `MissingEndTag` violation forever instead of ever reaching `Eof`. Now that
    /// `oxixml_xml`'s reader is fixed, this is reported like any other well-formedness
    /// violation — including with a position, which the old two-phase design could not supply
    /// for this particular constraint (its second phase went through
    /// [`oxixml_dom::BuildError`], which carries none).
    #[test]
    fn an_unclosed_tag_is_reported_with_a_position() {
        let mut stdin: &[u8] = b"<a><b>";
        let mut stdout = Vec::new();
        let result = execute("-", &mut stdin, &mut stdout);
        assert_eq!(result, Ok(false));
        let printed = String::from_utf8(stdout).expect("utf8");
        assert!(printed.contains("<stdin>:1:"), "{printed}");
        assert!(printed.contains("not closed"), "{printed}");
    }

    #[test]
    fn a_duplicate_attribute_is_reported() {
        // Attribute-uniqueness is a well-formedness constraint `Config::check_well_formedness`
        // enforces, not something the base tokenizer checks on its own.
        let mut stdin: &[u8] = b"<a x=\"1\" x=\"2\"/>";
        let mut stdout = Vec::new();
        let result = execute("-", &mut stdin, &mut stdout);
        assert_eq!(result, Ok(false));
        let printed = String::from_utf8(stdout).expect("utf8");
        assert!(printed.contains("<stdin>:1:"), "{printed}");
        assert!(printed.contains("more than once"), "{printed}");
    }

    #[test]
    fn a_malformed_comment_is_reported() {
        let mut stdin: &[u8] = b"<a><!-- bad -- comment --></a>";
        let mut stdout = Vec::new();
        assert_eq!(execute("-", &mut stdin, &mut stdout), Ok(false));
    }

    #[test]
    /// A well-formedness violation is fatal (XML 1.0 §1.2), so a document with several of them
    /// is reported one problem at a time: the *first* one, with its position, and nothing after
    /// it. See [`check_well_formed`] for why the run stops there.
    fn the_first_of_several_problems_is_the_one_reported() {
        let mut stdin: &[u8] = b"<a><b></c><d></e></a>";
        let mut stdout = Vec::new();
        let result = execute("-", &mut stdin, &mut stdout);
        assert_eq!(result, Ok(false));
        let printed = String::from_utf8(stdout).expect("utf8");
        assert_eq!(printed.lines().count(), 1, "{printed}");
        assert!(printed.contains("</c>"), "{printed}");
        assert!(!printed.contains("</e>"), "{printed}");
    }

    /// Regression test for the hang [`check_well_formed`]'s documentation describes: the two
    /// historical reproducers — a mismatched end tag and an unclosed one — must never make the
    /// check run past its timeout now that `check_well_formedness` is enabled unconditionally.
    /// A short timeout keeps this fast without changing the inputs under test.
    #[test]
    fn unbalanced_tags_never_hang() {
        for input in [&b"<a><b></a>"[..], &b"<a><b>"[..]] {
            let text = String::from_utf8(input.to_vec()).expect("utf8");
            let result = check_well_formed(text, "<stdin>".to_owned(), Duration::from_millis(500));
            assert!(
                result.is_ok(),
                "expected the check to finish well inside the timeout: {result:?}"
            );
        }
    }

    /// Beyond the two tiny historical reproducers: a much larger unclosed document must not
    /// merely avoid the pre-fix *hang* but stay comfortably fast, confirming the check scales
    /// linearly rather than trading an infinite loop for a quadratic one.
    #[test]
    fn a_large_unclosed_document_stays_well_inside_the_timeout() {
        let text = "<a>".repeat(50_000);
        let generous_but_bounded = Duration::from_secs(5);
        let (well_formed, diagnostics) =
            check_well_formed(text, "<stdin>".to_owned(), generous_but_bounded)
                .expect("50,000 unclosed elements must not come anywhere near a 5s budget");
        assert!(!well_formed);
        assert_eq!(
            diagnostics.len(),
            1,
            "one fatal error, not one per open element"
        );
    }

    #[test]
    fn a_timeout_that_elapses_is_reported_as_a_processing_error() {
        let result = with_timeout(Duration::from_millis(1), || {
            std::thread::sleep(Duration::from_millis(200));
        });
        assert!(result.is_err());
    }
}
