//! [`XPathError`] and [`XPathErrorCode`] — the qualified errors an XPath processor raises.

use oxixml_fn::{ERROR_NAMESPACE, FnError};
use oxixml_infoset::{ExpandedName, Sequence, XdmError};
use std::error::Error;
use std::fmt;

macro_rules! error_codes {
    ($( $(#[$meta:meta])* $variant:ident => $spelling:literal, $kind:ident ; )+) => {
        /// A static or dynamic error code defined by
        /// [XPath 3.1 Appendix F](https://www.w3.org/TR/xpath-31/#id-errors) or by
        /// [Functions and Operators 3.1](https://www.w3.org/TR/xpath-functions-31/#error-summary).
        ///
        /// # Examples
        ///
        /// ```
        /// use oxixml_xpath::XPathErrorCode;
        ///
        /// assert_eq!(XPathErrorCode::Xpst0003.as_str(), "XPST0003");
        /// assert!(XPathErrorCode::Xpst0003.is_static());
        /// assert!(!XPathErrorCode::Xpdy0002.is_static());
        /// assert_eq!(XPathErrorCode::from_spelling("XPTY0004"), Some(XPathErrorCode::Xpty0004));
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum XPathErrorCode {
            $( $(#[$meta])* $variant, )+
        }

        impl XPathErrorCode {
            /// Every code this crate raises, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];

            /// The bare code, as the specification spells it.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $spelling, )+
                }
            }

            /// Whether the code names a *static* error — one detected while compiling, before
            /// any input document is seen.
            #[must_use]
            pub const fn is_static(self) -> bool {
                match self {
                    $( Self::$variant => error_codes!(@kind $kind), )+
                }
            }

            /// Looks a code up by its (case-sensitive) specification spelling.
            #[must_use]
            pub fn from_spelling(spelling: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|code| code.as_str() == spelling)
            }
        }
    };
    (@kind static) => { true };
    (@kind dynamic) => { false };
}

error_codes! {
    /// `XPST0003` — the expression is not syntactically valid.
    Xpst0003 => "XPST0003", static;
    /// `XPST0008` — the expression refers to a variable or type that is not in scope.
    Xpst0008 => "XPST0008", static;
    /// `XPST0017` — the expanded name and arity do not match a known function.
    Xpst0017 => "XPST0017", static;
    /// `XPST0051` — the name in an `ItemType` is not a known atomic type.
    Xpst0051 => "XPST0051", static;
    /// `XPST0080` — the target of `cast` or `castable` is `xs:NOTATION`, `xs:anySimpleType` or
    /// `xs:anyAtomicType`.
    Xpst0080 => "XPST0080", static;
    /// `XQST0070` — a name is in one of the namespaces the specification reserves.
    Xqst0070 => "XQST0070", static;
    /// `XQST0052` — the target of `cast` or `castable` is not a simple type in the in-scope
    /// schema types.
    ///
    /// The code is spelled `XQ…` because it is XQuery's: from XQuery 3.0 onwards an unknown
    /// cast target is `XQST0052` where XPath reports [`XPathErrorCode::Xpst0051`], and a host
    /// says which reading applies with
    /// [`StaticContext::with_xquery_host_language`](crate::StaticContext::with_xquery_host_language).
    Xqst0052 => "XQST0052", static;
    /// `XPST0081` — a namespace prefix used in the expression is not bound.
    Xpst0081 => "XPST0081", static;
    /// `XQST0039` — an inline function declares the same parameter name twice.
    ///
    /// The code is spelled `XQ…` because XQuery declared it first, for a function *declaration*;
    /// XPath 3.0 adopted it unchanged for the parameter list of an inline function expression.
    Xqst0039 => "XQST0039", static;
    /// `XPDY0002` — the expression depends on a context item, and none is defined.
    Xpdy0002 => "XPDY0002", dynamic;
    /// `XPDY0050` — the operand of `treat as` does not match the required type.
    Xpdy0050 => "XPDY0050", dynamic;
    /// `XPTY0004` — the value supplied has the wrong type for the context it is used in.
    Xpty0004 => "XPTY0004", dynamic;
    /// `XPTY0018` — the last step of a path expression yields both nodes and non-nodes.
    Xpty0018 => "XPTY0018", dynamic;
    /// `XPTY0019` — a step of a path expression is applied to something that is not a node.
    Xpty0019 => "XPTY0019", dynamic;
    /// `XPTY0020` — the context item of an axis step is not a node.
    Xpty0020 => "XPTY0020", dynamic;
    /// `XPTY0117` — the function conversion rules were asked to convert an `xs:untypedAtomic`
    /// value (or the string value of an untyped node) to a *namespace-sensitive* required type:
    /// `xs:QName`, `xs:NOTATION`, or a type derived from either. XPath 3.1 §3.1.5.2 forbids it,
    /// because the namespace bindings the lexical form would need are not those of the caller.
    Xpty0117 => "XPTY0117", dynamic;
    /// `XPDY0130` — an implementation-defined limit has been exceeded.
    Xpdy0130 => "XPDY0130", dynamic;
    /// `FOAR0001` — division by zero.
    Foar0001 => "FOAR0001", dynamic;
    /// `FOAR0002` — numeric operation overflow or underflow.
    Foar0002 => "FOAR0002", dynamic;
    /// `FOAP0001` — the array `fn:apply` is given does not have as many members as the
    /// function's arity, so there is no call to make.
    Foap0001 => "FOAP0001", dynamic;
    /// `FOAY0001` — an array index is out of bounds.
    Foay0001 => "FOAY0001", dynamic;
    /// `FOCA0001` — the value being cast to `xs:decimal` is outside the range this
    /// implementation's `xs:decimal` covers.
    Foca0001 => "FOCA0001", dynamic;
    /// `FOCA0002` — invalid lexical value; raised when a non-finite `xs:double` or `xs:float`
    /// is cast to `xs:decimal` or `xs:integer`.
    Foca0002 => "FOCA0002", dynamic;
    /// `FOCA0003` — the value is too large for `xs:integer`.
    Foca0003 => "FOCA0003", dynamic;
    /// `FODT0001` — overflow or underflow in a date/time operation, including a cast whose
    /// lexical form is well formed but names a point outside the representable timeline.
    Fodt0001 => "FODT0001", dynamic;
    /// `FODT0002` — overflow or underflow in a duration operation, the duration counterpart of
    /// [`XPathErrorCode::Fodt0001`].
    Fodt0002 => "FODT0002", dynamic;
    /// `FOUT1170` — the resource named by `fn:unparsed-text` cannot be retrieved, or its URI is
    /// not valid.
    Fout1170 => "FOUT1170", dynamic;
    /// `FOUT1190` — the encoding named for a text resource is not supported, or is not the one
    /// the resource is in.
    Fout1190 => "FOUT1190", dynamic;
    /// `FOUT1200` — the octets of a text resource cannot be decoded in the stated encoding.
    Fout1200 => "FOUT1200", dynamic;
    /// `FORG0001` — invalid value for a cast or constructor.
    Forg0001 => "FORG0001", dynamic;
    /// `FORG0003` — `fn:zero-or-one` was given a sequence of more than one item.
    Forg0003 => "FORG0003", dynamic;
    /// `FORG0004` — `fn:one-or-more` was given the empty sequence.
    Forg0004 => "FORG0004", dynamic;
    /// `FORG0005` — `fn:exactly-one` was given zero or several items.
    Forg0005 => "FORG0005", dynamic;
    /// `FORG0006` — invalid argument type.
    Forg0006 => "FORG0006", dynamic;
    /// `FOCH0002` — the collation is not supported.
    Foch0002 => "FOCH0002", dynamic;
    /// `FODC0002` — an external resource could not be retrieved.
    Fodc0002 => "FODC0002", dynamic;
    /// `FODC0005` — the argument to `fn:doc` is not a valid URI.
    Fodc0005 => "FODC0005", dynamic;
    /// `FOTY0012` — the argument node does not have a typed value.
    Foty0012 => "FOTY0012", dynamic;
    /// `FOTY0013` — the argument to `fn:data` contains a function item.
    Foty0013 => "FOTY0013", dynamic;
    /// `FOTY0014` — `fn:string` was applied to a function item.
    Foty0014 => "FOTY0014", dynamic;
    /// `FOTY0015` — `fn:deep-equal` was applied to a function item.
    Foty0015 => "FOTY0015", dynamic;
    /// `FONS0004` — no namespace is bound to the prefix.
    Fons0004 => "FONS0004", dynamic;
    /// `FOER0000` — the unidentified error `fn:error` raises by default.
    Foer0000 => "FOER0000", dynamic;
}

impl XPathErrorCode {
    /// The namespace every code lives in, `http://www.w3.org/2005/xqt-errors`.
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

impl fmt::Display for XPathErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error raised while compiling or evaluating an XPath expression.
///
/// The error always carries the qualified name a conforming processor would report, which is
/// what an XQuery `try`/`catch` or an XSLT `xsl:catch` binds to `$err:code`. When the name is
/// one of the codes this crate knows, [`XPathError::code`] recovers it; `fn:error` may supply
/// a name of its own, in which case only [`XPathError::name`] is meaningful.
///
/// Static errors additionally carry the character offset in the source expression where the
/// problem was detected — see [`XPathError::offset`].
///
/// # Examples
///
/// ```
/// use oxixml_xpath::{StaticContext, XPath, XPathErrorCode, XPathVersion};
///
/// let failure = XPath::compile("1 +", XPathVersion::XPath31, StaticContext::new())
///     .expect_err("a truncated expression");
/// assert_eq!(failure.code(), Some(XPathErrorCode::Xpst0003));
/// assert!(failure.is_static());
/// assert!(failure.offset().is_some());
///
/// let compiled = XPath::compile("1 idiv 0", XPathVersion::XPath31, StaticContext::new())?;
/// let failure = compiled.evaluate_without_context().expect_err("division by zero");
/// assert_eq!(failure.code(), Some(XPathErrorCode::Foar0001));
/// assert_eq!(
///     failure.name().to_string(),
///     "{http://www.w3.org/2005/xqt-errors}FOAR0001"
/// );
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XPathError {
    code: Option<XPathErrorCode>,
    name: ExpandedName,
    message: String,
    offset: Option<usize>,
    error_object: Option<Sequence>,
}

impl XPathError {
    /// Builds an error with one of the known codes.
    pub fn new(code: XPathErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: Some(code),
            name: code.expanded_name(),
            message: message.into(),
            offset: None,
            error_object: None,
        }
    }

    /// Builds an error under an arbitrary qualified name, as `fn:error` does.
    pub fn with_name(name: ExpandedName, message: impl Into<String>) -> Self {
        let code = name
            .namespace()
            .filter(|namespace| *namespace == ERROR_NAMESPACE)
            .and_then(|_| XPathErrorCode::from_spelling(name.local_name()));
        Self {
            code,
            name,
            message: message.into(),
            offset: None,
            error_object: None,
        }
    }

    /// Attaches the character offset in the source expression at which the error was detected.
    #[must_use]
    pub const fn at(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Attaches the error object `fn:error` accepts as its third argument.
    #[must_use]
    pub fn with_error_object(mut self, error_object: Sequence) -> Self {
        self.error_object = Some(error_object);
        self
    }

    /// The known code, or `None` when the error carries a name of its own.
    #[must_use]
    pub const fn code(&self) -> Option<XPathErrorCode> {
        self.code
    }

    /// The error's qualified name.
    #[must_use]
    pub const fn name(&self) -> &ExpandedName {
        &self.name
    }

    /// The human-readable description.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The character offset in the source expression, for static errors.
    #[must_use]
    pub const fn offset(&self) -> Option<usize> {
        self.offset
    }

    /// The error object supplied by `fn:error`, if any.
    #[must_use]
    pub const fn error_object(&self) -> Option<&Sequence> {
        self.error_object.as_ref()
    }

    /// Whether this error carries `code`.
    #[must_use]
    pub fn is(&self, code: XPathErrorCode) -> bool {
        self.code == Some(code)
    }

    /// Whether this is a static error, detected at compile time.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.code.is_some_and(XPathErrorCode::is_static)
    }

    /// A static syntax error, `XPST0003`, at `offset`.
    pub(crate) fn syntax(offset: usize, message: impl Into<String>) -> Self {
        Self::new(XPathErrorCode::Xpst0003, message).at(offset)
    }

    /// A dynamic type error, `XPTY0004`.
    pub(crate) fn type_error(message: impl Into<String>) -> Self {
        Self::new(XPathErrorCode::Xpty0004, message)
    }
}

impl fmt::Display for XPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self.code {
            Some(code) => code.as_str().to_owned(),
            None => self.name.to_string(),
        };
        match self.offset {
            Some(offset) => write!(f, "[{label}] {} (at offset {offset})", self.message),
            None => write!(f, "[{label}] {}", self.message),
        }
    }
}

impl Error for XPathError {}

impl From<XdmError> for XPathError {
    fn from(error: XdmError) -> Self {
        let spelling = error.code().as_str();
        match XPathErrorCode::from_spelling(spelling) {
            Some(code) => Self::new(code, error.message().to_owned()),
            None => Self::with_name(
                ExpandedName::new(Some(ERROR_NAMESPACE), spelling),
                error.message().to_owned(),
            ),
        }
    }
}

impl From<FnError> for XPathError {
    fn from(error: FnError) -> Self {
        let mut converted = Self::with_name(error.name().clone(), error.description().to_owned());
        if let Some(object) = error.error_object() {
            converted = converted.with_error_object(object.clone());
        }
        converted
    }
}

/// The result of every fallible operation in this crate.
pub type XPathResult<T> = Result<T, XPathError>;

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_fn::FnErrorCode;
    use oxixml_infoset::ErrorCode;

    #[test]
    fn codes_are_classified_and_round_trip() {
        assert!(XPathErrorCode::Xpst0003.is_static());
        assert!(XPathErrorCode::Xpst0017.is_static());
        assert!(!XPathErrorCode::Xpty0004.is_static());
        for code in XPathErrorCode::ALL {
            assert_eq!(XPathErrorCode::from_spelling(code.as_str()), Some(*code));
            assert_eq!(code.expanded_name().namespace(), Some(ERROR_NAMESPACE));
        }
    }

    #[test]
    fn conversion_preserves_the_code() {
        let from_fn: XPathError = FnError::new(FnErrorCode::Forg0001, "bad cast").into();
        assert_eq!(from_fn.code(), Some(XPathErrorCode::Forg0001));

        let from_xdm: XPathError = XdmError::new(ErrorCode::Foty0013, "a map").into();
        assert_eq!(from_xdm.code(), Some(XPathErrorCode::Foty0013));

        // A code this crate does not model still keeps its qualified name.
        let exotic: XPathError = FnError::new(FnErrorCode::Fodf1310, "picture").into();
        assert_eq!(exotic.code(), None);
        assert_eq!(exotic.name().local_name(), "FODF1310");
    }

    #[test]
    fn display_includes_the_offset_for_static_errors() {
        let error = XPathError::syntax(7, "unexpected end of expression");
        assert!(error.to_string().contains("XPST0003"));
        assert!(error.to_string().contains("offset 7"));
        assert_eq!(error.offset(), Some(7));
    }
}
