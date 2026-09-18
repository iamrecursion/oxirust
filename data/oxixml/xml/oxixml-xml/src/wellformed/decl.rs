//! Grammar checker for the XML declaration.
//!
//! Implements the [`XMLDecl`] production:
//!
//! ```text
//! XMLDecl      ::= '<?xml' VersionInfo EncodingDecl? SDDecl? S? '?>'
//! VersionInfo  ::= S 'version' Eq ("'" VersionNum "'" | '"' VersionNum '"')
//! Eq           ::= S? '=' S?
//! EncodingDecl ::= S 'encoding' Eq ('"' EncName '"' | "'" EncName "'")
//! SDDecl       ::= S 'standalone' Eq (("'" ('yes' | 'no') "'")
//!                                    | ('"' ('yes' | 'no') '"'))
//! ```
//!
//! [`XMLDecl`]: https://www.w3.org/TR/xml/#NT-XMLDecl

use crate::XmlVersion;
use crate::errors::WellFormedError;
use crate::wellformed::chars::{is_enc_name, is_space, is_version_num};

/// Builds an [`InvalidXmlDeclaration`](WellFormedError::InvalidXmlDeclaration) error.
fn err(message: impl Into<String>) -> WellFormedError {
    WellFormedError::InvalidXmlDeclaration(message.into())
}

/// The information the well-formedness checker keeps from an XML declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Declaration {
    /// The version declared by the `version` pseudo-attribute.
    pub version: XmlVersion,
    /// Whether `standalone="yes"` was declared. A standalone document keeps the
    /// [Entity Declared] constraint in force even when an external subset exists.
    ///
    /// [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
    pub standalone: bool,
}

/// Validates the body of an XML declaration — the text between `<?` and `?>`,
/// which always starts with `xml` — and returns the declared version together
/// with the standalone status.
pub(crate) fn validate(body: &str) -> Result<Declaration, WellFormedError> {
    let rest = body
        .strip_prefix("xml")
        .ok_or_else(|| err("the declaration must start with `xml`"))?;

    let mut cursor = Cursor { rest };
    cursor.require_space("before `version`")?;
    let version = cursor.pseudo_attribute("version")?;
    let version = match version {
        "1.0" => XmlVersion::Explicit1_0,
        "1.1" => XmlVersion::Explicit1_1,
        other if is_version_num(other) => XmlVersion::Explicit1_0,
        other => {
            return Err(err(format!(
                "`{}` does not match the `VersionNum` production",
                other
            )));
        }
    };

    let mut standalone = false;
    let mut saw_space = cursor.skip_space();
    if saw_space && cursor.rest.starts_with("encoding") {
        let encoding = cursor.pseudo_attribute("encoding")?;
        if !is_enc_name(encoding) {
            return Err(err(format!(
                "`{}` does not match the `EncName` production",
                encoding
            )));
        }
        saw_space = cursor.skip_space();
    }
    if saw_space && cursor.rest.starts_with("standalone") {
        let value = cursor.pseudo_attribute("standalone")?;
        match value {
            "yes" => standalone = true,
            "no" => standalone = false,
            other => {
                return Err(err(format!(
                    "`standalone` must be `yes` or `no`, found `{}`",
                    other
                )));
            }
        }
        cursor.skip_space();
    }
    if cursor.rest.is_empty() {
        Ok(Declaration {
            version,
            standalone,
        })
    } else {
        Err(err(format!(
            "unexpected `{}` in the XML declaration",
            cursor.rest.chars().take(24).collect::<String>()
        )))
    }
}

/// A cursor over the remaining text of an XML declaration.
struct Cursor<'a> {
    /// Text that has not been consumed yet.
    rest: &'a str,
}

impl<'a> Cursor<'a> {
    /// Consumes zero or more `S` characters, reporting whether any were found.
    fn skip_space(&mut self) -> bool {
        let trimmed = self.rest.trim_start_matches(is_space);
        let found = trimmed.len() != self.rest.len();
        self.rest = trimmed;
        found
    }

    /// Consumes one or more `S` characters or fails.
    fn require_space(&mut self, what: &str) -> Result<(), WellFormedError> {
        if self.skip_space() {
            Ok(())
        } else {
            Err(err(format!("white space is required {}", what)))
        }
    }

    /// Consumes `name Eq QuotedValue` and returns the quoted value.
    fn pseudo_attribute(&mut self, name: &'static str) -> Result<&'a str, WellFormedError> {
        self.rest = self
            .rest
            .strip_prefix(name)
            .ok_or_else(|| err(format!("`{}` is expected", name)))?;
        self.skip_space();
        self.rest = self
            .rest
            .strip_prefix('=')
            .ok_or_else(|| err(format!("`=` is expected after `{}`", name)))?;
        self.skip_space();
        let quote = match self.rest.chars().next() {
            Some(quote @ ('"' | '\'')) => quote,
            _ => return Err(err(format!("the value of `{}` must be quoted", name))),
        };
        let body = &self.rest[1..];
        match body.find(quote) {
            Some(end) => {
                self.rest = &body[end + 1..];
                Ok(&body[..end])
            }
            None => Err(err(format!("the value of `{}` is not closed", name))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_well_formed_declarations() -> Result<(), WellFormedError> {
        assert_eq!(
            validate("xml version=\"1.0\"")?.version,
            XmlVersion::Explicit1_0
        );
        assert_eq!(
            validate("xml version='1.1'")?.version,
            XmlVersion::Explicit1_1
        );
        assert!(
            validate("xml version = \"1.0\" encoding = 'UTF-8' standalone = 'yes' ")?.standalone
        );
        assert!(!validate("xml version=\"1.0\" standalone=\"no\"")?.standalone);
        assert!(!validate("xml version=\"1.0\"")?.standalone);
        Ok(())
    }

    #[test]
    fn rejects_ill_formed_declarations() {
        for body in [
            "xml VERSION=\"1.0\"",
            "xml encoding=\"UTF-8\" version=\"1.0\"",
            "xml version=\"1.0\"encoding=\"UTF-8\" ",
            "xml version=\"1.0\" version=\"1.0\"",
            "xml version=\"1.0\" valid=\"no\" ",
            "xml version=\"1.0\" standalone=\"YES\" ",
            "xml version=\"1.0\" encoding=\" UTF-8\"",
            "xml version=\"1.0 \" ",
            "xml version=\"1.0?\"",
            "xml version = '1.0\"",
        ] {
            assert!(validate(body).is_err(), "expected an error for `{body}`");
        }
    }
}
