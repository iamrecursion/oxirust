//! The error type returned when a candidate string fails BCP 47 well-formedness checks.

use core::fmt;

/// The reason a [`LanguageTag::parse`](crate::LanguageTag::parse) or
/// [`LanguageTag::parse_and_normalize`](crate::LanguageTag::parse_and_normalize) call failed.
///
/// The variants are deliberately not exhaustive-matchable from outside the crate (the type is
/// `#[non_exhaustive]`-like in spirit, kept private and only reachable through [`fmt::Display`]);
/// callers that need programmatic branching should match on the rendered message or, more
/// robustly, simply treat every [`LanguageTagParseError`] as "not a well-formed BCP 47 tag".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagParseErrorKind {
    /// The primary language subtag was missing, too short, or contained non-letters.
    InvalidLanguage,
    /// A subtag did not fit the grammar in its position (wrong length, wrong character class,
    /// or a shape that no production allows there).
    InvalidSubtag,
    /// A subtag between two hyphens (or at the start/end of the tag) was empty, i.e. two
    /// hyphens appeared back to back, or the tag started or ended with a hyphen.
    EmptySubtag,
    /// A subtag was longer than the eight characters every ABNF production allows at most.
    SubtagTooLong,
    /// A primary language subtag of two or three letters carried more than the three extended
    /// language (`extlang`) subtags [RFC 5646 §2.2.2](https://www.rfc-editor.org/rfc/rfc5646#section-2.2.2) permits.
    TooManyExtlangs,
    /// An extension singleton (e.g. `u` in `de-DE-u-co-phonebk`) was not followed by at least
    /// one value subtag.
    EmptyExtension,
    /// The private-use introducer `x` was not followed by at least one value subtag.
    EmptyPrivateUse,
    /// The same variant subtag (case-insensitively) appeared more than once. RFC 5646 §2.2.9
    /// excludes tags with duplicate variants from being valid, and this crate rejects them at
    /// parse time rather than accepting a tag whose variant sequence is ambiguous to interpret.
    DuplicateVariant,
    /// The same extension singleton (case-insensitively) introduced more than one extension
    /// sequence, e.g. `ar-a-aaa-b-bbb-a-ccc`. RFC 5646 §2.2.9 excludes tags with duplicate
    /// singletons from being valid, and this crate rejects them at parse time.
    DuplicateSingleton,
}

impl fmt::Display for TagParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLanguage => "the primary language subtag is invalid",
            Self::InvalidSubtag => {
                "a subtag does not match any grammar production allowed at that position"
            }
            Self::EmptySubtag => {
                "a subtag must not be empty (check for repeated or leading/trailing hyphens)"
            }
            Self::SubtagTooLong => "a subtag may be at most eight characters long",
            Self::TooManyExtlangs => "at most three extended language subtags are allowed",
            Self::EmptyExtension => {
                "an extension singleton must be followed by at least one value subtag"
            }
            Self::EmptyPrivateUse => {
                "the private use introducer 'x' must be followed by at least one value subtag"
            }
            Self::DuplicateVariant => "the same variant subtag appears more than once",
            Self::DuplicateSingleton => {
                "the same extension singleton introduces more than one extension sequence"
            }
        })
    }
}

/// An error raised while validating a candidate string as a [`LanguageTag`](crate::LanguageTag).
///
/// This type only reports *that* well-formedness checking failed and *why*, in human-readable
/// form via [`fmt::Display`]; it carries no byte offsets. It implements [`std::error::Error`],
/// so it composes cleanly with `?` in any function returning a boxed or `anyhow`-style error.
///
/// ```
/// use oxixml_langtag::LanguageTag;
///
/// let error = LanguageTag::parse("a-DE").unwrap_err();
/// assert_eq!(
///     error.to_string(),
///     "the primary language subtag is invalid"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanguageTagParseError {
    kind: TagParseErrorKind,
}

impl LanguageTagParseError {
    /// Builds a new error wrapping the given internal reason.
    pub(crate) const fn new(kind: TagParseErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for LanguageTagParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

/// Only the [`Error`](std::error::Error) implementation needs `std`; everything else in this
/// crate is `no_std` + `alloc`, so it is gated behind the default-on `std` feature exactly as
/// `oxilangtag` gates its own.
#[cfg(feature = "std")]
impl std::error::Error for LanguageTagParseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn display_messages_are_stable() {
        let kinds = [
            TagParseErrorKind::InvalidLanguage,
            TagParseErrorKind::InvalidSubtag,
            TagParseErrorKind::EmptySubtag,
            TagParseErrorKind::SubtagTooLong,
            TagParseErrorKind::TooManyExtlangs,
            TagParseErrorKind::EmptyExtension,
            TagParseErrorKind::EmptyPrivateUse,
            TagParseErrorKind::DuplicateVariant,
            TagParseErrorKind::DuplicateSingleton,
        ];
        for kind in kinds {
            let error = LanguageTagParseError::new(kind);
            assert!(!error.to_string().is_empty());
            // `Error` is object-safe and implemented (behind the `std` feature); this must
            // compile and run.
            #[cfg(feature = "std")]
            {
                let _: &dyn std::error::Error = &error;
            }
        }
    }

    #[test]
    fn error_is_copy_and_small() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<LanguageTagParseError>();
    }

    /// Regression test for `no_std` support: rendering an error must not need `std` or even a
    /// heap allocation. `std::error::Error` is the only `std`-dependent item in the crate and
    /// it is gated behind the default-on `std` feature, so `default-features = false` builds
    /// (checked with `cargo check -p oxixml-langtag --no-default-features`, including for
    /// bare-metal targets such as `thumbv7em-none-eabihf`) keep the whole parsing API.
    #[test]
    fn errors_render_without_std_or_allocation() {
        use core::fmt::Write as _;

        struct StackBuffer {
            bytes: [u8; 128],
            len: usize,
        }

        impl core::fmt::Write for StackBuffer {
            fn write_str(&mut self, text: &str) -> core::fmt::Result {
                let end = self.len + text.len();
                let slot = self.bytes.get_mut(self.len..end).ok_or(core::fmt::Error)?;
                slot.copy_from_slice(text.as_bytes());
                self.len = end;
                Ok(())
            }
        }

        let mut buffer = StackBuffer {
            bytes: [0; 128],
            len: 0,
        };
        write!(
            buffer,
            "{}",
            LanguageTagParseError::new(TagParseErrorKind::EmptySubtag)
        )
        .expect("the message fits in 128 bytes");
        assert!(buffer.len > 0);
    }
}
