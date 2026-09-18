//! [`FnError`] and the [`FnErrorCode`] vocabulary of *XPath and XQuery Functions and Operators
//! 3.1*.
//!
//! Every failure this crate reports carries the qualified error name a conforming processor
//! would raise, so that an XQuery `try`/`catch` or an XSLT `xsl:try` can match on it exactly as
//! the specification describes. The names all live in the standard error namespace
//! [`ERROR_NAMESPACE`], conventionally bound to the prefix `err`.

use oxixml_infoset::{ErrorCode as XdmErrorCode, ExpandedName, Sequence, XdmError};
use oxixml_regex::{ForxCode, RegexError};
use std::error::Error;
use std::fmt;

/// The namespace name of the XPath and XQuery error vocabulary,
/// `http://www.w3.org/2005/xqt-errors`, conventionally bound to the prefix `err`.
pub const ERROR_NAMESPACE: &str = "http://www.w3.org/2005/xqt-errors";

macro_rules! error_codes {
    ($( $(#[$meta:meta])* $variant:ident => $spelling:literal ),+ $(,)?) => {
        /// One of the error codes defined by
        /// [XPath and XQuery Functions and Operators 3.1](https://www.w3.org/TR/xpath-functions-31/#error-summary)
        /// (plus the handful of `XP*`/`XQ*` codes a function can raise).
        ///
        /// # Examples
        ///
        /// ```
        /// use oxixml_fn::FnErrorCode;
        ///
        /// assert_eq!(FnErrorCode::Forg0001.as_str(), "FORG0001");
        /// assert_eq!(FnErrorCode::from_spelling("FOAR0001"), Some(FnErrorCode::Foar0001));
        /// assert_eq!(
        ///     FnErrorCode::Foer0000.expanded_name().to_string(),
        ///     "{http://www.w3.org/2005/xqt-errors}FOER0000"
        /// );
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum FnErrorCode {
            $( $(#[$meta])* $variant, )+
        }

        impl FnErrorCode {
            /// Every code this crate knows about, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];

            /// The bare code, as it is spelled in the specification's error summary.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $spelling, )+
                }
            }

            /// Looks a code up by its specification spelling, which is case-sensitive.
            #[must_use]
            pub fn from_spelling(spelling: &str) -> Option<Self> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|code| code.as_str() == spelling)
            }
        }
    };
}

error_codes! {
    /// `FOAR0001` — division by zero.
    Foar0001 => "FOAR0001",
    /// `FOAR0002` — numeric operation overflow or underflow.
    Foar0002 => "FOAR0002",
    /// `FOCA0001` — input value too large for `xs:decimal`.
    Foca0001 => "FOCA0001",
    /// `FOCA0002` — invalid lexical value.
    Foca0002 => "FOCA0002",
    /// `FOCA0003` — input value too large for `xs:integer`.
    Foca0003 => "FOCA0003",
    /// `FOCA0005` — `NaN` supplied as a float or double value.
    Foca0005 => "FOCA0005",
    /// `FOCA0006` — string to be cast to `xs:decimal` has too much precision.
    Foca0006 => "FOCA0006",
    /// `FOCH0001` — code point not valid.
    Foch0001 => "FOCH0001",
    /// `FOCH0002` — unsupported collation.
    Foch0002 => "FOCH0002",
    /// `FOCH0003` — unsupported normalization form.
    Foch0003 => "FOCH0003",
    /// `FOCH0004` — collation does not support collation units.
    Foch0004 => "FOCH0004",
    /// `FODC0001` — no context document.
    Fodc0001 => "FODC0001",
    /// `FODC0002` — error retrieving resource.
    Fodc0002 => "FODC0002",
    /// `FODC0006` — the string supplied to `fn:parse-xml` is not a well-formed document.
    Fodc0006 => "FODC0006",
    /// `FODC0007` — the base URI passed to `fn:parse-xml` is not a valid absolute URI.
    Fodc0007 => "FODC0007",
    /// `FODF1280` — invalid decimal format name.
    Fodf1280 => "FODF1280",
    /// `FODF1310` — invalid decimal format picture string.
    Fodf1310 => "FODF1310",
    /// `FODT0001` — overflow or underflow in a date/time operation.
    Fodt0001 => "FODT0001",
    /// `FODT0002` — overflow or underflow in a duration operation.
    Fodt0002 => "FODT0002",
    /// `FODT0003` — invalid timezone value.
    Fodt0003 => "FODT0003",
    /// `FOER0000` — unidentified error, the default raised by `fn:error`.
    Foer0000 => "FOER0000",
    /// `FOFD1340` — invalid date/time formatting parameters.
    Fofd1340 => "FOFD1340",
    /// `FOFD1350` — the component specifier is not available in the given date/time value.
    Fofd1350 => "FOFD1350",
    /// `FOJS0001` — the supplied JSON text is not well-formed.
    Fojs0001 => "FOJS0001",
    /// `FOJS0003` — a JSON object has duplicate keys and `duplicates=reject` is in force.
    Fojs0003 => "FOJS0003",
    /// `FOJS0004` — the JSON functions were asked for schema-aware processing.
    Fojs0004 => "FOJS0004",
    /// `FOJS0005` — an invalid option was supplied to one of the JSON functions.
    Fojs0005 => "FOJS0005",
    /// `FOJS0006` — the input to `fn:xml-to-json` is not a valid XML representation of JSON.
    Fojs0006 => "FOJS0006",
    /// `FOJS0007` — a string marked `escaped` contains an undefined escape sequence.
    Fojs0007 => "FOJS0007",
    /// `FONS0004` — no namespace found for the prefix.
    Fons0004 => "FONS0004",
    /// `FONS0005` — base URI not defined in the static context.
    Fons0005 => "FONS0005",
    /// `FOQM0006` — no suitable XQuery processor is available to `fn:load-xquery-module`.
    Foqm0006 => "FOQM0006",
    /// `FORG0001` — invalid value for a cast or constructor.
    Forg0001 => "FORG0001",
    /// `FORG0002` — invalid argument to `fn:resolve-uri`.
    Forg0002 => "FORG0002",
    /// `FORG0003` — `fn:zero-or-one` called with a sequence of more than one item.
    Forg0003 => "FORG0003",
    /// `FORG0004` — `fn:one-or-more` called with the empty sequence.
    Forg0004 => "FORG0004",
    /// `FORG0005` — `fn:exactly-one` called with a sequence of zero or several items.
    Forg0005 => "FORG0005",
    /// `FORG0006` — invalid argument type.
    Forg0006 => "FORG0006",
    /// `FORG0008` — both arguments to `fn:dateTime` carry a timezone, and they differ.
    Forg0008 => "FORG0008",
    /// `FORG0009` — error resolving a relative URI against a base URI in `fn:resolve-uri`.
    Forg0009 => "FORG0009",
    /// `FORG0010` — invalid date/time.
    Forg0010 => "FORG0010",
    /// `FORX0001` — invalid regular-expression flags.
    Forx0001 => "FORX0001",
    /// `FORX0002` — invalid regular expression.
    Forx0002 => "FORX0002",
    /// `FORX0003` — the regular expression matches a zero-length string.
    Forx0003 => "FORX0003",
    /// `FORX0004` — invalid replacement string.
    Forx0004 => "FORX0004",
    /// `FOTY0012` — the argument node does not have a typed value.
    Foty0012 => "FOTY0012",
    /// `FOTY0013` — the argument to `fn:data` contains a function item.
    Foty0013 => "FOTY0013",
    /// `FOTY0014` — `fn:string` applied to a function item.
    Foty0014 => "FOTY0014",
    /// `FOTY0015` — `fn:deep-equal` applied to a function item.
    Foty0015 => "FOTY0015",
    /// `FOUT1170` — the resource named by `fn:unparsed-text` or `fn:json-doc` cannot be
    /// retrieved.
    Fout1170 => "FOUT1170",
    /// `FOUT1190` — the encoding a text resource names is not supported, or is not the one the
    /// resource is actually in.
    Fout1190 => "FOUT1190",
    /// `FOUT1200` — the octets of a text resource cannot be decoded in the stated encoding.
    Fout1200 => "FOUT1200",
    /// `FOXT0004` — the XSLT transformation `fn:transform` was asked for has been disabled.
    Foxt0004 => "FOXT0004",
    /// `SERD0001` — the value cannot be serialized with the output method in force.
    Serd0001 => "SERD0001",
    /// `SEPM0004` — `doctype-system`, or a `standalone` other than `omit`, was asked for while
    /// the value being serialized is not a well-formed document: neither can be written without
    /// exactly one element and no text of its own.
    Sepm0004 => "SEPM0004",
    /// `SEPM0009` — `omit-xml-declaration` is `yes` while `standalone` is not `omit`, or while
    /// `doctype-system` is specified for a version other than `1.0`: the declaration those ask
    /// for is the one being omitted.
    Sepm0009 => "SEPM0009",
    /// `SEPM0016` — a serialization parameter has an invalid value.
    Sepm0016 => "SEPM0016",
    /// `SEPM0017` — a serialization parameter, or the element that carries it, is invalid.
    Sepm0017 => "SEPM0017",
    /// `SEPM0018` — two entries of `use-character-maps` name the same character.
    Sepm0018 => "SEPM0018",
    /// `SEPM0019` — a serialization parameter is specified more than once.
    Sepm0019 => "SEPM0019",
    /// `SENR0001` — the value being serialized contains an item no output method can write:
    /// an attribute or namespace node with no element to hang from, or a function item.
    Senr0001 => "SENR0001",
    /// `SERE0006` — the result would contain a character the XML version being written cannot
    /// represent: a C0 control other than tab, line feed and carriage return is outside XML
    /// 1.0's `Char` production, and XML 1.1 admits it only as a character reference.
    Sere0006 => "SERE0006",
    /// `SERE0020` — a number being written by the `json` output method is `NaN` or an infinity,
    /// neither of which JSON has a literal for.
    Sere0020 => "SERE0020",
    /// `SERE0021` — the value being serialized contains a function item, which no output
    /// method can write.
    Sere0021 => "SERE0021",
    /// `SERE0022` — two keys of a map being written by the `json` output method have the same
    /// string value, and `allow-duplicate-names` is `no`.
    Sere0022 => "SERE0022",
    /// `SERE0023` — the value cannot be serialized by the `json` output method.
    Sere0023 => "SERE0023",
    /// `SESU0011` — the `normalization-form` parameter names a Unicode normalization form this
    /// serializer does not support.
    Sesu0011 => "SESU0011",
    /// `FOAY0001` — array index out of bounds.
    Foay0001 => "FOAY0001",
    /// `FOAY0002` — negative array length.
    Foay0002 => "FOAY0002",
    /// `XPTY0004` — the supplied value has the wrong type for the context it is used in.
    Xpty0004 => "XPTY0004",
    /// `XPTY0117` — an `xs:untypedAtomic` value was supplied where a *namespace-sensitive* type
    /// (`xs:QName`, `xs:NOTATION` or a derivation) is required, which the function conversion
    /// rules of XPath 3.1 §3.1.5.2 forbid: untyped data carries no prefix bindings.
    Xpty0117 => "XPTY0117",
    /// `XPDY0002` — the context item is absent.
    Xpdy0002 => "XPDY0002",
    /// `XPDY0130` — an implementation-defined limit has been exceeded; here, a value or a tree
    /// that nests deeper than a recursive walk over it can afford.
    Xpdy0130 => "XPDY0130",
    /// `XPST0017` — the expanded name and arity do not match a known function.
    Xpst0017 => "XPST0017",
    /// `XQDY0074` — the name used in a computed constructor is not a valid lexical `QName`.
    Xqdy0074 => "XQDY0074",
    /// `XQDY0137` — no two keys in a map may be the same key.
    Xqdy0137 => "XQDY0137",
}

impl FnErrorCode {
    /// The namespace name every code lives in, [`ERROR_NAMESPACE`].
    #[must_use]
    pub const fn namespace(self) -> &'static str {
        ERROR_NAMESPACE
    }

    /// The code's qualified name, `{http://www.w3.org/2005/xqt-errors}CODE`.
    #[must_use]
    pub fn expanded_name(self) -> ExpandedName {
        ExpandedName::new(Some(ERROR_NAMESPACE), self.as_str())
    }
}

impl fmt::Display for FnErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A dynamic error raised by a function in the library.
///
/// The error carries three things a `catch` clause can bind: the qualified [`name`](FnError::name)
/// (always a code from [`FnErrorCode`] except when `fn:error` supplies its own), a human-readable
/// [`description`](FnError::description), and the optional
/// [`error_object`](FnError::error_object) that `fn:error` accepts as its third argument.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{FnError, FnErrorCode};
///
/// let error = FnError::new(FnErrorCode::Foar0001, "division by zero");
/// assert_eq!(error.code(), Some(FnErrorCode::Foar0001));
/// assert_eq!(error.to_string(), "[FOAR0001] division by zero");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnError {
    code: Option<FnErrorCode>,
    name: ExpandedName,
    description: String,
    error_object: Option<Sequence>,
}

impl FnError {
    /// Builds an error with one of the specification's codes.
    #[must_use]
    pub fn new(code: FnErrorCode, description: impl Into<String>) -> Self {
        Self {
            code: Some(code),
            name: code.expanded_name(),
            description: description.into(),
            error_object: None,
        }
    }

    /// Builds an error whose name is not one of the codes this crate knows, as `fn:error`
    /// permits.
    #[must_use]
    pub fn with_name(name: ExpandedName, description: impl Into<String>) -> Self {
        let code = name
            .namespace()
            .filter(|namespace| *namespace == ERROR_NAMESPACE)
            .and_then(|_| FnErrorCode::from_spelling(name.local_name()));
        Self {
            code,
            name,
            description: description.into(),
            error_object: None,
        }
    }

    /// Attaches the error object `fn:error` was given.
    #[must_use]
    pub fn with_error_object(mut self, error_object: Sequence) -> Self {
        self.error_object = Some(error_object);
        self
    }

    /// The specification code, or `None` when `fn:error` supplied a name outside the error
    /// namespace.
    #[must_use]
    pub const fn code(&self) -> Option<FnErrorCode> {
        self.code
    }

    /// The error's qualified name.
    #[must_use]
    pub const fn name(&self) -> &ExpandedName {
        &self.name
    }

    /// The human-readable description. It is diagnostic only and is not part of the contract.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The error object supplied to `fn:error`, if any.
    #[must_use]
    pub fn error_object(&self) -> Option<&Sequence> {
        self.error_object.as_ref()
    }

    /// Whether this error carries the given code.
    #[must_use]
    pub fn is(&self, code: FnErrorCode) -> bool {
        self.code == Some(code)
    }
}

impl fmt::Display for FnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "[{code}] {}", self.description),
            None => write!(f, "[{}] {}", self.name, self.description),
        }
    }
}

impl Error for FnError {}

impl From<XdmError> for FnError {
    /// Maps a data-model error onto the corresponding function-library code, preserving the
    /// message.
    fn from(error: XdmError) -> Self {
        let code = match error.code() {
            XdmErrorCode::Xpty0004 => FnErrorCode::Xpty0004,
            XdmErrorCode::Forg0001 => FnErrorCode::Forg0001,
            XdmErrorCode::Forg0006 => FnErrorCode::Forg0006,
            XdmErrorCode::Foty0012 => FnErrorCode::Foty0012,
            XdmErrorCode::Foty0013 => FnErrorCode::Foty0013,
            XdmErrorCode::Foty0014 => FnErrorCode::Foty0014,
            XdmErrorCode::Foty0015 => FnErrorCode::Foty0015,
            XdmErrorCode::Foay0001 => FnErrorCode::Foay0001,
            XdmErrorCode::Foch0002 => FnErrorCode::Foch0002,
            XdmErrorCode::Xqdy0137 => FnErrorCode::Xqdy0137,
            XdmErrorCode::Fodc0006 => FnErrorCode::Fodc0006,
            XdmErrorCode::Fodt0001 => FnErrorCode::Fodt0001,
            XdmErrorCode::Fodt0002 => FnErrorCode::Fodt0002,
            XdmErrorCode::Foca0002 => FnErrorCode::Foca0002,
            _ => FnErrorCode::Foer0000,
        };
        Self::new(code, error.message().to_owned())
    }
}

impl From<RegexError> for FnError {
    /// Maps a regular-expression failure onto its `FORX` code; the back-tracking budget guard,
    /// which has no specification code, is reported as `FORX0002`.
    fn from(error: RegexError) -> Self {
        let code = match error.forx_code() {
            Some(ForxCode::Forx0001) => FnErrorCode::Forx0001,
            Some(ForxCode::Forx0003) => FnErrorCode::Forx0003,
            Some(ForxCode::Forx0004) => FnErrorCode::Forx0004,
            Some(ForxCode::Forx0002) | None => FnErrorCode::Forx0002,
        };
        Self::new(code, error.to_string())
    }
}

/// The result type every function in this crate returns.
pub type FnResult<T> = Result<T, FnError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn spellings_are_unique_and_round_trip() {
        let mut seen = HashSet::new();
        for &code in FnErrorCode::ALL {
            assert!(seen.insert(code.as_str()), "duplicate {code}");
            assert_eq!(FnErrorCode::from_spelling(code.as_str()), Some(code));
            assert_eq!(code.namespace(), ERROR_NAMESPACE);
        }
        assert_eq!(FnErrorCode::from_spelling("forg0001"), None);
    }

    #[test]
    fn names_outside_the_error_namespace_have_no_code() {
        let name = ExpandedName::new(Some("urn:example"), "boom");
        let error = FnError::with_name(name.clone(), "custom");
        assert_eq!(error.code(), None);
        assert_eq!(error.name(), &name);
        assert!(error.to_string().contains("urn:example"));
    }

    #[test]
    fn names_inside_the_error_namespace_recover_their_code() {
        let error = FnError::with_name(FnErrorCode::Forg0006.expanded_name(), "bad type");
        assert!(error.is(FnErrorCode::Forg0006));
    }

    #[test]
    fn data_model_errors_keep_their_identity() {
        let converted = FnError::from(XdmError::new(XdmErrorCode::Foty0013, "no atomization"));
        assert!(converted.is(FnErrorCode::Foty0013));
    }

    #[test]
    fn regex_errors_map_onto_forx_codes() {
        let converted = FnError::from(RegexError::MatchesEmptyString);
        assert!(converted.is(FnErrorCode::Forx0003));
        let budget = FnError::from(RegexError::BudgetExceeded);
        assert!(budget.is(FnErrorCode::Forx0002));
    }
}
