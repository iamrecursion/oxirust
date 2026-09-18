//! Error types returned by [`crate::Iri`] and [`crate::IriRef`] parsing, resolution, and
//! relativization.

use std::error::Error;
use std::fmt;
use std::net::AddrParseError;

/// An error raised while parsing, validating, or resolving an [`crate::Iri`] or
/// [`crate::IriRef`].
///
/// The [`fmt::Display`] implementation describes both the offending character (or fragment) and
/// which grammar production rejected it, e.g. `"Invalid character '?' in scheme"`.
#[derive(Debug)]
pub struct IriParseError {
    pub(crate) kind: IriParseErrorKind,
}

impl fmt::Display for IriParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            IriParseErrorKind::NoScheme => write!(f, "no scheme found in an absolute IRI"),
            IriParseErrorKind::EmptyScheme => write!(f, "empty schemes are not allowed"),
            IriParseErrorKind::InvalidSchemeCharacter(c) => {
                write!(f, "invalid character '{c}' in scheme")
            }
            IriParseErrorKind::InvalidHostCharacter(c) => {
                write!(f, "invalid character '{c}' in host")
            }
            IriParseErrorKind::UnmatchedHostBracket => {
                write!(f, "'[' bracket must be paired with a closing one in host")
            }
            IriParseErrorKind::InvalidHostIp(e) => write!(f, "invalid host IP ({e})"),
            IriParseErrorKind::InvalidPortCharacter(c) => {
                write!(f, "invalid character '{c}' in port")
            }
            IriParseErrorKind::InvalidIriCodePoint(c) => {
                write!(f, "invalid IRI code point '{c}'")
            }
            IriParseErrorKind::InvalidPercentEncoding(cs) => write!(
                f,
                "invalid IRI percent-encoding '{}'",
                cs.iter().flatten().collect::<String>()
            ),
            IriParseErrorKind::PathStartingWithTwoSlashes => {
                write!(f, "an IRI path is not allowed to start with //")
            }
            IriParseErrorKind::ResolvingDotSegmentsNonHierarchical => write!(
                f,
                "cannot resolve dot segments against a relative non-hierarchical IRI path"
            ),
        }
    }
}

impl Error for IriParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            IriParseErrorKind::InvalidHostIp(e) => Some(e),
            _ => None,
        }
    }
}

impl From<IriParseErrorKind> for IriParseError {
    fn from(kind: IriParseErrorKind) -> Self {
        Self { kind }
    }
}

/// The specific grammar production that rejected the input.
///
/// Kept crate-private (only surfaced through [`IriParseError`]'s [`fmt::Display`] and
/// [`Error::source`] implementations) so that new variants can be added without breaking
/// downstream matches, matching `oxiri`'s own stability posture.
#[derive(Debug)]
pub(crate) enum IriParseErrorKind {
    NoScheme,
    EmptyScheme,
    InvalidSchemeCharacter(char),
    UnmatchedHostBracket,
    InvalidHostCharacter(char),
    InvalidHostIp(AddrParseError),
    InvalidPortCharacter(char),
    InvalidIriCodePoint(char),
    InvalidPercentEncoding([Option<char>; 3]),
    PathStartingWithTwoSlashes,
    ResolvingDotSegmentsNonHierarchical,
}

/// An error raised by [`crate::Iri::relativize`].
///
/// It occurs when it is not possible to build a relative IRI reference that resolves back to
/// the target IRI — for example when the target's path contains a `/../` or `/./` segment,
/// since the resolution algorithm ([RFC 3986 §5.2.4]) would consume it and change the meaning.
///
/// [RFC 3986 §5.2.4]: https://www.rfc-editor.org/rfc/rfc3986#section-5.2.4
#[derive(Debug)]
pub struct IriRelativizeError;

impl fmt::Display for IriRelativizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "it is not possible to make this IRI relative because its path contains `/..` or `/.`"
        )
    }
}

impl Error for IriRelativizeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_kind_has_a_non_empty_display_message() {
        let addr_error = "not an ip"
            .parse::<std::net::Ipv6Addr>()
            .expect_err("must fail to parse");
        let kinds = [
            IriParseErrorKind::NoScheme,
            IriParseErrorKind::EmptyScheme,
            IriParseErrorKind::InvalidSchemeCharacter(' '),
            IriParseErrorKind::UnmatchedHostBracket,
            IriParseErrorKind::InvalidHostCharacter(' '),
            IriParseErrorKind::InvalidHostIp(addr_error),
            IriParseErrorKind::InvalidPortCharacter('x'),
            IriParseErrorKind::InvalidIriCodePoint(' '),
            IriParseErrorKind::InvalidPercentEncoding([Some('%'), Some('z'), Some('z')]),
            IriParseErrorKind::PathStartingWithTwoSlashes,
            IriParseErrorKind::ResolvingDotSegmentsNonHierarchical,
        ];
        for kind in kinds {
            let error: IriParseError = kind.into();
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn invalid_host_ip_exposes_the_underlying_error_as_source() {
        let addr_error = "not an ip"
            .parse::<std::net::Ipv6Addr>()
            .expect_err("must fail to parse");
        let error: IriParseError = IriParseErrorKind::InvalidHostIp(addr_error).into();
        assert!(Error::source(&error).is_some());

        let other: IriParseError = IriParseErrorKind::NoScheme.into();
        assert!(Error::source(&other).is_none());
    }

    #[test]
    fn relativize_error_has_a_display_message_and_is_an_error() {
        let error = IriRelativizeError;
        assert!(!error.to_string().is_empty());
        let _: &dyn Error = &error;
    }
}
