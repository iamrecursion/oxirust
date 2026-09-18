//! The four exit codes every `oxixml` subcommand uses, and a couple of small conveniences for
//! producing a [`std::process::ExitCode`] from one.

use std::process::ExitCode;

/// Everything requested succeeded.
pub const OK: u8 = 0;
/// The command line itself was wrong: an unknown flag, a missing required flag, an invalid
/// value, or a combination of flags that makes no sense together.
pub const USAGE_ERROR: u8 = 1;
/// The command line was fine, but carrying it out failed: a file could not be read, a document
/// was not well-formed when well-formedness was assumed, a stylesheet or query did not compile,
/// an RDF document did not parse, and so on.
pub const PROCESSING_ERROR: u8 = 2;
/// The requested check ran to completion and found the input invalid: `validate` found a
/// schema-invalid document, or `lint` found the input not well-formed.
pub const VALIDATION_FAILURE: u8 = 3;

/// Turns one of this module's constants into a [`std::process::ExitCode`].
#[must_use]
pub fn code(value: u8) -> ExitCode {
    ExitCode::from(value)
}
