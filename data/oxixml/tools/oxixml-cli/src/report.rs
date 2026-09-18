//! Consistent error reporting: every subcommand ends a failing run by calling one of these, so
//! that the message shape (`oxixml <subcommand>: error: ...`) and the exit code that goes with it
//! are the same everywhere.

use std::fmt;
use std::io::Write;

use crate::exit;

/// Reports a usage error (bad flags, a missing required flag, an invalid combination) to
/// `stderr` and returns [`exit::USAGE_ERROR`].
pub fn usage_error(stderr: &mut dyn Write, subcommand: &str, message: impl fmt::Display) -> u8 {
    let _ = writeln!(stderr, "oxixml {subcommand}: error: {message}");
    let _ = writeln!(
        stderr,
        "Try 'oxixml {subcommand} --help' for more information."
    );
    exit::USAGE_ERROR
}

/// Reports a processing error (a file could not be read, a document did not parse, a stylesheet
/// did not compile, ...) to `stderr` and returns [`exit::PROCESSING_ERROR`].
pub fn processing_error(
    stderr: &mut dyn Write,
    subcommand: &str,
    message: impl fmt::Display,
) -> u8 {
    let _ = writeln!(stderr, "oxixml {subcommand}: error: {message}");
    exit::PROCESSING_ERROR
}

/// Prints an informational line to `stderr` when `verbose` requests it.
pub fn info(stderr: &mut dyn Write, verbose: bool, message: impl fmt::Display) {
    if verbose {
        let _ = writeln!(stderr, "oxixml: {message}");
    }
}
