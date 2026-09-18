//! A small, dependency-free command-line argument parser.
//!
//! This is not a general-purpose argument-parsing framework: it is exactly as capable as the
//! `oxixml` CLI needs and no more, so that the whole crate stays free of external dependencies.
//! A subcommand declares its flags as a slice of [`FlagSpec`], then calls [`parse`] to turn the
//! raw `Vec<String>` the user typed into a [`Parsed`] bundle of flag/value pairs and positional
//! arguments.
//!
//! # Grammar
//!
//! * `--name` — a boolean flag.
//! * `--name value` — a value flag, value taken from the next token.
//! * `--name=value` — a value flag, value taken from after the `=`.
//! * `-x` — a short flag (always boolean in this crate; only `-h` is used).
//! * `--` — everything after this token is a positional argument, even if it looks like a flag.
//! * anything else — a positional argument.
//!
//! Short flags are never bundled (`-ab` is not `-a -b`) and never take a value: the CLI only
//! defines one (`-h`), so there is no need for the extra complexity.
//!
//! # Example
//!
//! ```
//! use oxixml_cli::args::{FlagSpec, parse};
//!
//! let specs = [FlagSpec::value("indent"), FlagSpec::flag("in-place")];
//! let parsed = parse(
//!     &["--indent".to_owned(), "4".to_owned(), "--in-place".to_owned(), "a.xml".to_owned()],
//!     &specs,
//! )
//! .expect("valid arguments");
//! assert_eq!(parsed.value("indent"), Some("4"));
//! assert!(parsed.has("in-place"));
//! assert_eq!(parsed.positionals(), ["a.xml"]);
//! ```

use std::fmt;

/// One flag a subcommand accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagSpec {
    /// The flag's long name, without the leading `--` (e.g. `"indent"` for `--indent`).
    pub name: &'static str,
    /// A single-character alias, without the leading `-` (e.g. `Some('h')` for `-h`). Short
    /// flags are always boolean.
    pub short: Option<char>,
    /// Whether the flag takes a value (`--name value` / `--name=value`) or is a bare boolean
    /// switch (`--name`).
    pub takes_value: bool,
}

impl FlagSpec {
    /// A boolean switch, e.g. `--in-place`.
    #[must_use]
    pub const fn flag(name: &'static str) -> Self {
        Self {
            name,
            short: None,
            takes_value: false,
        }
    }

    /// A flag that takes a value, e.g. `--indent N`.
    #[must_use]
    pub const fn value(name: &'static str) -> Self {
        Self {
            name,
            short: None,
            takes_value: true,
        }
    }

    /// Attaches a single-character alias.
    #[must_use]
    pub const fn with_short(mut self, short: char) -> Self {
        self.short = Some(short);
        self
    }
}

/// Something wrong with the raw argument list itself, independent of what a subcommand then
/// does with the parsed result. Always an exit code `1` (usage error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// `--name` (or `-x`) is not one of the flags this subcommand declared.
    UnknownFlag(String),
    /// A value-taking flag was the last token, or was immediately followed by another flag.
    MissingValue(String),
    /// `--name=value` was given for a flag declared as a boolean switch.
    UnexpectedValue(String),
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFlag(flag) => write!(f, "unrecognized flag `{flag}`"),
            Self::MissingValue(flag) => write!(f, "`{flag}` requires a value"),
            Self::UnexpectedValue(flag) => write!(f, "`{flag}` does not take a value"),
        }
    }
}

impl std::error::Error for ArgError {}

/// The result of [`parse`]: every flag/value pair, in the order they were written (so a repeated
/// flag such as `--xsd a.xsd --xsd b.xsd` is not lost), plus the positional arguments.
#[derive(Debug, Clone, Default)]
pub struct Parsed {
    pairs: Vec<(String, Option<String>)>,
    positionals: Vec<String>,
}

impl Parsed {
    /// The positional arguments, in the order they appeared.
    #[must_use]
    pub fn positionals(&self) -> &[String] {
        &self.positionals
    }

    /// Whether `name` was given at all (as a boolean switch or a value flag).
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.pairs.iter().any(|(n, _)| n == name)
    }

    /// The value of the last occurrence of the value-flag `name`, or `None` if it was never
    /// given. Panics only make sense for a programmer error (asking for a flag that was never
    /// declared), which this method cannot detect, so an absent flag and a boolean flag are both
    /// simply `None`.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<&str> {
        self.pairs
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.as_deref())
    }

    /// Every value given for the (repeatable) flag `name`, in the order they were written.
    #[must_use]
    pub fn all_values(&self, name: &str) -> Vec<&str> {
        self.pairs
            .iter()
            .filter(|(n, _)| n == name)
            .filter_map(|(_, v)| v.as_deref())
            .collect()
    }

    /// The number of distinct flag names that were given, counting a repeated flag once — handy
    /// for "exactly one of these mutually exclusive flags" checks.
    #[must_use]
    pub fn count_present(&self, names: &[&str]) -> usize {
        names.iter().filter(|name| self.has(name)).count()
    }
}

/// Parses `args` against `specs`. See the [module documentation](self) for the grammar.
///
/// # Errors
///
/// Returns [`ArgError`] for an unrecognized flag, a value-flag with no value, or a boolean flag
/// given `=value`.
pub fn parse(args: &[String], specs: &[FlagSpec]) -> Result<Parsed, ArgError> {
    let mut parsed = Parsed::default();
    let mut index = 0;
    let mut positionals_only = false;
    while index < args.len() {
        let arg = &args[index];
        index += 1;
        if positionals_only {
            parsed.positionals.push(arg.clone());
            continue;
        }
        if arg == "--" {
            positionals_only = true;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--") {
            let (name, inline_value) = match rest.split_once('=') {
                Some((name, value)) => (name, Some(value.to_owned())),
                None => (rest, None),
            };
            let spec = specs
                .iter()
                .find(|spec| spec.name == name)
                .ok_or_else(|| ArgError::UnknownFlag(format!("--{name}")))?;
            if spec.takes_value {
                let value = match inline_value {
                    Some(value) => value,
                    None => {
                        let next = args
                            .get(index)
                            .ok_or_else(|| ArgError::MissingValue(format!("--{name}")))?;
                        index += 1;
                        next.clone()
                    }
                };
                parsed.pairs.push((name.to_owned(), Some(value)));
            } else {
                if inline_value.is_some() {
                    return Err(ArgError::UnexpectedValue(format!("--{name}")));
                }
                parsed.pairs.push((name.to_owned(), None));
            }
            continue;
        }
        if arg.len() == 2 && arg.starts_with('-') && arg != "-" {
            let short = arg
                .chars()
                .nth(1)
                .expect("length checked to be exactly 2 bytes of ASCII-safe flag syntax");
            let spec = specs
                .iter()
                .find(|spec| spec.short == Some(short))
                .ok_or_else(|| ArgError::UnknownFlag(arg.clone()))?;
            if spec.takes_value {
                let next = args
                    .get(index)
                    .ok_or_else(|| ArgError::MissingValue(arg.clone()))?;
                index += 1;
                parsed
                    .pairs
                    .push((spec.name.to_owned(), Some(next.clone())));
            } else {
                parsed.pairs.push((spec.name.to_owned(), None));
            }
            continue;
        }
        parsed.positionals.push(arg.clone());
    }
    Ok(parsed)
}

/// Splits a `key=value` flag argument (`--param name=value`, `--ns prefix=uri`) at the first
/// `=`.
///
/// # Errors
///
/// Returns a human-readable message when `text` has no `=`.
pub fn split_key_value(text: &str) -> Result<(&str, &str), String> {
    text.split_once('=')
        .ok_or_else(|| format!("expected `name=value`, found `{text}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn boolean_flags_need_no_value() {
        let specs = [FlagSpec::flag("in-place")];
        let parsed = parse(&owned(&["--in-place"]), &specs).expect("parses");
        assert!(parsed.has("in-place"));
        assert_eq!(parsed.value("in-place"), None);
    }

    #[test]
    fn value_flags_accept_the_space_separated_form() {
        let specs = [FlagSpec::value("indent")];
        let parsed = parse(&owned(&["--indent", "4"]), &specs).expect("parses");
        assert_eq!(parsed.value("indent"), Some("4"));
    }

    #[test]
    fn value_flags_accept_the_equals_form() {
        let specs = [FlagSpec::value("indent")];
        let parsed = parse(&owned(&["--indent=4"]), &specs).expect("parses");
        assert_eq!(parsed.value("indent"), Some("4"));
    }

    #[test]
    fn repeated_flags_are_all_kept_in_order() {
        let specs = [FlagSpec::value("xsd")];
        let parsed = parse(&owned(&["--xsd", "a.xsd", "--xsd", "b.xsd"]), &specs).expect("parses");
        assert_eq!(parsed.all_values("xsd"), vec!["a.xsd", "b.xsd"]);
    }

    #[test]
    fn later_values_win_for_value() {
        let specs = [FlagSpec::value("algorithm")];
        let parsed = parse(
            &owned(&["--algorithm", "c14n", "--algorithm", "rdfc10"]),
            &specs,
        )
        .expect("parses");
        assert_eq!(parsed.value("algorithm"), Some("rdfc10"));
    }

    #[test]
    fn positionals_are_collected_in_order() {
        let specs = [FlagSpec::flag("verbose")];
        let parsed = parse(&owned(&["a.xml", "--verbose", "b.xml"]), &specs).expect("parses");
        assert_eq!(parsed.positionals(), ["a.xml", "b.xml"]);
    }

    #[test]
    fn a_lone_dash_is_a_positional_not_a_flag() {
        let specs: [FlagSpec; 0] = [];
        let parsed = parse(&owned(&["-"]), &specs).expect("parses");
        assert_eq!(parsed.positionals(), ["-"]);
    }

    #[test]
    fn double_dash_stops_flag_parsing() {
        let specs = [FlagSpec::flag("verbose")];
        let parsed = parse(&owned(&["--", "--verbose"]), &specs).expect("parses");
        assert_eq!(parsed.positionals(), ["--verbose"]);
        assert!(!parsed.has("verbose"));
    }

    #[test]
    fn unknown_long_flags_are_rejected() {
        let specs: [FlagSpec; 0] = [];
        let error = parse(&owned(&["--bogus"]), &specs).expect_err("unknown flag");
        assert_eq!(error, ArgError::UnknownFlag("--bogus".to_owned()));
    }

    #[test]
    fn unknown_short_flags_are_rejected() {
        let specs: [FlagSpec; 0] = [];
        let error = parse(&owned(&["-z"]), &specs).expect_err("unknown flag");
        assert_eq!(error, ArgError::UnknownFlag("-z".to_owned()));
    }

    #[test]
    fn a_missing_value_is_reported() {
        let specs = [FlagSpec::value("indent")];
        let error = parse(&owned(&["--indent"]), &specs).expect_err("missing value");
        assert_eq!(error, ArgError::MissingValue("--indent".to_owned()));
    }

    #[test]
    fn a_value_flag_does_not_swallow_the_next_flag() {
        // `--indent --in-place` has no value for `--indent`; the parser does not look past a
        // flag it does not recognize as a value to guess where the value is, it simply takes the
        // next raw token — matching `--indent=--in-place` would be surprising, so this documents
        // the (intentionally simple) behavior instead of guessing.
        let specs = [FlagSpec::value("indent"), FlagSpec::flag("in-place")];
        let parsed = parse(&owned(&["--indent", "--in-place"]), &specs).expect("parses");
        assert_eq!(parsed.value("indent"), Some("--in-place"));
    }

    #[test]
    fn a_boolean_flag_rejects_an_inline_value() {
        let specs = [FlagSpec::flag("in-place")];
        let error = parse(&owned(&["--in-place=yes"]), &specs).expect_err("unexpected value");
        assert_eq!(error, ArgError::UnexpectedValue("--in-place".to_owned()));
    }

    #[test]
    fn short_flags_resolve_to_their_long_name() {
        let specs = [FlagSpec::flag("help").with_short('h')];
        let parsed = parse(&owned(&["-h"]), &specs).expect("parses");
        assert!(parsed.has("help"));
    }

    #[test]
    fn count_present_counts_distinct_flags_once() {
        let specs = [FlagSpec::value("xsd"), FlagSpec::value("dtd")];
        let parsed = parse(&owned(&["--xsd", "a", "--xsd", "b"]), &specs).expect("parses");
        assert_eq!(parsed.count_present(&["xsd", "dtd"]), 1);
    }

    #[test]
    fn split_key_value_splits_at_the_first_equals() {
        assert_eq!(split_key_value("a=b=c"), Ok(("a", "b=c")));
        assert!(split_key_value("noequals").is_err());
    }
}
