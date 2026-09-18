//! The flags every subcommand shares: `--catalog`, `--xinclude`, `--base`, `--quiet`/`--verbose`
//! (plus `--help`, handled by [`crate`] before a subcommand ever sees its arguments).
//!
//! Not every global flag makes sense for every subcommand — `--base` means nothing to `sparql
//! --parse`, which never loads a document — so a subcommand declares which ones it accepts as a
//! [`Capabilities`] value, and [`extract`] turns a usage of a flag the subcommand does not accept
//! into the same kind of usage error as an unrecognized flag would be.

use crate::args::{FlagSpec, Parsed};

/// The flag specifications for every global flag, ready to be appended to a subcommand's own
/// list before calling [`crate::args::parse`].
pub const FLAGS: &[FlagSpec] = &[
    FlagSpec::value("catalog"),
    FlagSpec::flag("xinclude"),
    FlagSpec::value("base"),
    FlagSpec::flag("quiet"),
    FlagSpec::flag("verbose"),
    FlagSpec::flag("help").with_short('h'),
];

/// How much non-essential status output a subcommand prints, alongside the output it was asked
/// for (which is never suppressed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verbosity {
    /// Suppresses the final human-readable summary line `lint`/`validate` print after their
    /// diagnostics (the diagnostics themselves are the point of those subcommands, not noise,
    /// and are never suppressed).
    Quiet,
    /// The default: the summary line prints, informational progress does not.
    #[default]
    Normal,
    /// Additionally prints informational progress (which schema documents were loaded, how many
    /// triples were parsed, ...) to stderr.
    Verbose,
}

impl Verbosity {
    /// Whether [`Verbosity::Verbose`] progress lines should print.
    #[must_use]
    pub const fn is_verbose(self) -> bool {
        matches!(self, Self::Verbose)
    }

    /// Whether the final summary line should be suppressed.
    #[must_use]
    pub const fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

/// Which global flags a subcommand accepts. A subcommand that does not accept one still has it
/// in its [`FLAGS`] (so it can be *rejected* with a clear message rather than "unrecognized
/// flag") but [`extract`] turns using it into a usage error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Whether `--catalog` is meaningful here.
    pub catalog: bool,
    /// Whether `--xinclude` is meaningful here.
    pub xinclude: bool,
    /// Whether `--base` is meaningful here.
    pub base: bool,
}

impl Capabilities {
    /// No document-related global flags — only `--quiet`/`--verbose` apply. Used by `sparql`,
    /// `hdt dump` and `hdt stats`.
    pub const NONE: Self = Self {
        catalog: false,
        xinclude: false,
        base: false,
    };

    /// `--base` only, for the subcommands that read a document but have no notion of inclusion
    /// or catalog-mediated external references of their own (`hdt build`, `convert`).
    pub const BASE_ONLY: Self = Self {
        catalog: false,
        xinclude: false,
        base: true,
    };

    /// Every document-related global flag: `fmt`, `lint`, `validate`, `transform`, `query` and
    /// `canon` on an XML input.
    pub const DOCUMENT: Self = Self {
        catalog: true,
        xinclude: true,
        base: true,
    };
}

/// The parsed value of every global flag.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The `--catalog` path, not yet loaded.
    pub catalog: Option<String>,
    /// Whether `--xinclude` was given.
    pub xinclude: bool,
    /// The `--base` value, not yet validated as an IRI.
    pub base: Option<String>,
    /// The combined effect of `--quiet`/`--verbose`.
    pub verbosity: Verbosity,
}

/// Pulls the global flags out of `parsed`, rejecting any that `capabilities` does not accept for
/// this subcommand and any use of both `--quiet` and `--verbose` together.
///
/// # Errors
///
/// A human-readable usage error message; the caller wraps it into the crate's usual usage-error
/// reporting.
pub fn extract(parsed: &Parsed, capabilities: Capabilities) -> Result<Options, String> {
    if parsed.has("quiet") && parsed.has("verbose") {
        return Err("--quiet and --verbose cannot both be given".to_owned());
    }
    if !capabilities.catalog && parsed.has("catalog") {
        return Err("--catalog is not applicable to this subcommand".to_owned());
    }
    if !capabilities.xinclude && parsed.has("xinclude") {
        return Err("--xinclude is not applicable to this subcommand".to_owned());
    }
    if !capabilities.base && parsed.has("base") {
        return Err("--base is not applicable to this subcommand".to_owned());
    }
    let verbosity = if parsed.has("quiet") {
        Verbosity::Quiet
    } else if parsed.has("verbose") {
        Verbosity::Verbose
    } else {
        Verbosity::Normal
    };
    Ok(Options {
        catalog: parsed.value("catalog").map(str::to_owned),
        xinclude: parsed.has("xinclude"),
        base: parsed.value("base").map(str::to_owned),
        verbosity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::parse;

    fn owned(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn defaults_are_all_off() {
        let parsed = parse(&owned(&[]), FLAGS).expect("parses");
        let options = extract(&parsed, Capabilities::DOCUMENT).expect("extracts");
        assert!(options.catalog.is_none());
        assert!(!options.xinclude);
        assert!(options.base.is_none());
        assert_eq!(options.verbosity, Verbosity::Normal);
    }

    #[test]
    fn quiet_and_verbose_conflict() {
        let parsed = parse(&owned(&["--quiet", "--verbose"]), FLAGS).expect("parses");
        assert!(extract(&parsed, Capabilities::DOCUMENT).is_err());
    }

    #[test]
    fn a_capability_the_subcommand_does_not_accept_is_rejected() {
        let parsed = parse(&owned(&["--xinclude"]), FLAGS).expect("parses");
        assert!(extract(&parsed, Capabilities::NONE).is_err());
        assert!(extract(&parsed, Capabilities::DOCUMENT).is_ok());
    }

    #[test]
    fn base_only_accepts_base_but_not_catalog() {
        let parsed = parse(&owned(&["--base", "urn:x"]), FLAGS).expect("parses");
        let options = extract(&parsed, Capabilities::BASE_ONLY).expect("extracts");
        assert_eq!(options.base.as_deref(), Some("urn:x"));

        let parsed = parse(&owned(&["--catalog", "cat.xml"]), FLAGS).expect("parses");
        assert!(extract(&parsed, Capabilities::BASE_ONLY).is_err());
    }
}
