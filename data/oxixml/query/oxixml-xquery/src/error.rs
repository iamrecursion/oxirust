//! [`XQueryError`] and [`XQueryErrorCode`] — the qualified errors an XQuery processor raises.
//!
//! XQuery inherits every `XP*`, `FO*` and `XD*` code of XPath and *Functions and Operators*, and
//! adds the `XQST*`, `XQDY*` and `XQTY*` families of
//! [XQuery 3.1 Appendix E](https://www.w3.org/TR/xquery-31/#id-errors). An [`XQueryError`]
//! therefore carries an [`ExpandedName`] — the authoritative identity of the error, which is what
//! a `try`/`catch` binds to `$err:code` — plus, when the name is one this crate knows, a
//! [`XQueryErrorCode`].
//!
//! Every [`XPathError`] converts, so an error raised deep inside the XPath expression core
//! surfaces unchanged.

use oxixml_fn::{ERROR_NAMESPACE, FnError};
use oxixml_infoset::{ExpandedName, Sequence, XdmError};
use oxixml_xpath::{XPathError, XPathErrorCode};
use std::error::Error;
use std::fmt;

macro_rules! xquery_error_codes {
    ($( $(#[$meta:meta])* $variant:ident => $spelling:literal, $kind:ident ; )+) => {
        /// A static, type or dynamic error code defined by
        /// [XQuery 3.1 Appendix E](https://www.w3.org/TR/xquery-31/#id-errors).
        ///
        /// Only the codes *this* crate raises are listed; an error that came out of the XPath
        /// core or the function library keeps its own name and is reported through
        /// [`XQueryError::name`].
        ///
        /// # Examples
        ///
        /// ```
        /// use oxixml_xquery::XQueryErrorCode;
        ///
        /// assert_eq!(XQueryErrorCode::Xqst0033.as_str(), "XQST0033");
        /// assert!(XQueryErrorCode::Xqst0033.is_static());
        /// assert!(!XQueryErrorCode::Xqdy0074.is_static());
        /// assert_eq!(XQueryErrorCode::from_spelling("XQTY0024"), Some(XQueryErrorCode::Xqty0024));
        /// ```
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum XQueryErrorCode {
            $( $(#[$meta])* $variant, )+
        }

        impl XQueryErrorCode {
            /// Every code this crate raises, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];

            /// The bare code, as the specification spells it.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$variant => $spelling, )+
                }
            }

            /// Whether the code names a *static* error — one detected while compiling.
            #[must_use]
            pub const fn is_static(self) -> bool {
                match self {
                    $( Self::$variant => xquery_error_codes!(@kind $kind), )+
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

xquery_error_codes! {
    /// `XPST0003` — the query is not syntactically valid.
    Xpst0003 => "XPST0003", static;
    /// `XPST0008` — a name used in the query is not in scope.
    Xpst0008 => "XPST0008", static;
    /// `XPTY0018` — the last step of a path expression yields both nodes and non-nodes.
    Xpty0018 => "XPTY0018", dynamic;
    /// `XPTY0019` — a step of a path expression is applied to something that is not a node.
    Xpty0019 => "XPTY0019", dynamic;
    /// `XPST0017` — the expanded name and arity do not match a known function.
    Xpst0017 => "XPST0017", static;
    /// `XPST0081` — a namespace prefix used in the query is not bound.
    Xpst0081 => "XPST0081", static;
    /// `XPST0051` — a sequence type names an atomic type the static context does not define.
    Xpst0051 => "XPST0051", static;
    /// `XQST0052` — the type named in a `cast` or `castable` is not in the in-scope schema types.
    Xqst0052 => "XQST0052", static;
    /// `XPST0080` — the target type of a `cast` or `castable` is `xs:NOTATION`,
    /// `xs:anySimpleType` or `xs:anyAtomicType`.
    Xpst0080 => "XPST0080", static;
    /// `XPTY0004` — the value supplied has the wrong type for the context it is used in.
    Xpty0004 => "XPTY0004", dynamic;
    /// `XPDY0002` — the query depends on a context item, and none is defined.
    Xpdy0002 => "XPDY0002", dynamic;
    /// `XPDY0130` — an implementation-defined limit was exceeded; this processor raises it when
    /// function calls nest too deeply.
    Xpdy0130 => "XPDY0130", dynamic;
    /// `XQST0022` — a namespace declaration attribute's value is not a literal.
    Xqst0022 => "XQST0022", static;
    /// `XQTY0024` — an attribute node follows a non-attribute node in element content.
    Xqty0024 => "XQTY0024", dynamic;
    /// `XQDY0025` — a computed element constructor produced two attributes with the same name.
    Xqdy0025 => "XQDY0025", dynamic;
    /// `XQDY0026` — a computed processing-instruction constructor's content contains `?>`.
    Xqdy0026 => "XQDY0026", dynamic;
    /// `XQST0031` — the version named in the version declaration is not supported.
    Xqst0031 => "XQST0031", static;
    /// `XQST0032` — the prolog declares the base URI more than once.
    Xqst0032 => "XQST0032", static;
    /// `XQST0033` — the prolog binds the same namespace prefix twice.
    Xqst0033 => "XQST0033", static;
    /// `XQST0034` — the prolog declares two functions with the same name and arity.
    Xqst0034 => "XQST0034", static;
    /// `XQST0038` — the prolog declares the default collation more than once, or names a
    /// collation the implementation does not support.
    Xqst0038 => "XQST0038", static;
    /// `XQST0039` — a function declaration repeats a parameter name.
    Xqst0039 => "XQST0039", static;
    /// `XQST0040` — a constructed element has two attributes with the same expanded name.
    Xqst0040 => "XQST0040", static;
    /// `XQDY0041` — the name of a computed processing instruction is not an `NCName`.
    Xqdy0041 => "XQDY0041", dynamic;
    /// `XQST0045` — a declared function is in a reserved namespace.
    Xqst0045 => "XQST0045", static;
    /// `XQST0046` — a URI literal is not a valid URI.
    Xqst0046 => "XQST0046", static;
    /// `XQST0047` — two module imports name the same target namespace.
    Xqst0047 => "XQST0047", static;
    /// `XQST0048` — a library module declares a name outside its own target namespace.
    Xqst0048 => "XQST0048", static;
    /// `XQST0049` — the prolog declares the same variable twice.
    Xqst0049 => "XQST0049", static;
    /// `XPDY0050` — the operand of `treat as` does not match the required type.
    Xpdy0050 => "XPDY0050", dynamic;
    /// `XQST0054` — the initializers of the prolog's variables are circular.
    Xqst0054 => "XQST0054", static;
    /// `XQDY0044` — a computed attribute's name is in the `xmlns` namespace.
    Xqdy0044 => "XQDY0044", dynamic;
    /// `XQDY0054` — a cycle was found while initializing the prolog's variables.
    Xqdy0054 => "XQDY0054", dynamic;
    /// `XQST0055` — the prolog declares copy-namespaces more than once.
    Xqst0055 => "XQST0055", static;
    /// `XQST0009` — the query imports a schema, and this processor does not have the Schema
    /// Import Feature.
    Xqst0009 => "XQST0009", static;
    /// `XQST0057` — a schema import binds a prefix, or gives a location hint, for the
    /// zero-length target namespace.
    Xqst0057 => "XQST0057", static;
    /// `XQST0059` — a module or schema import names something that cannot be found.
    Xqst0059 => "XQST0059", static;
    /// `XQST0060` — a declared function's name has no namespace.
    Xqst0060 => "XQST0060", static;
    /// `XQDY0064` — the target of a computed processing instruction is `XML` in any case.
    Xqdy0064 => "XQDY0064", dynamic;
    /// `XQST0065` — the prolog declares the ordering mode more than once.
    Xqst0065 => "XQST0065", static;
    /// `XQST0066` — the prolog declares a default element or function namespace more than once.
    Xqst0066 => "XQST0066", static;
    /// `XQST0067` — the prolog declares the construction mode more than once.
    Xqst0067 => "XQST0067", static;
    /// `XQST0068` — the prolog declares boundary-space more than once.
    Xqst0068 => "XQST0068", static;
    /// `XQST0069` — the prolog declares the default order for empty sequences more than once.
    Xqst0069 => "XQST0069", static;
    /// `XQST0070` — a namespace declaration rebinds a reserved prefix or namespace.
    Xqst0070 => "XQST0070", static;
    /// `XQST0071` — a direct element constructor has two namespace declaration attributes with
    /// the same name.
    Xqst0071 => "XQST0071", static;
    /// `XQDY0072` — the content of a constructed comment contains `--` or ends with `-`.
    Xqdy0072 => "XQDY0072", dynamic;
    /// `XQDY0074` — a computed constructor's name is not a valid lexical `QName`, or its prefix
    /// is not bound.
    Xqdy0074 => "XQDY0074", dynamic;
    /// `XQST0075` — `validate` needs the Schema Import Feature, which this processor has not.
    Xqst0075 => "XQST0075", static;
    /// `XQST0076` — a `collation` in a `group by` or `order by` clause names an unsupported
    /// collation.
    Xqst0076 => "XQST0076", static;
    /// `XQDY0084` — the element validated by `validate` has no matching declaration.
    Xqdy0084 => "XQDY0084", dynamic;
    /// `XQST0085` — a namespace URI in a namespace declaration attribute is a zero-length
    /// string, which XML 1.0 does not allow for a prefix.
    Xqst0085 => "XQST0085", static;
    /// `XQST0087` — the encoding name of the version declaration is not a valid encoding.
    Xqst0087 => "XQST0087", static;
    /// `XQST0088` — the target namespace of a module import is a zero-length string.
    Xqst0088 => "XQST0088", static;
    /// `XQST0089` — a `for` clause repeats a variable name in its positional variable.
    Xqst0089 => "XQST0089", static;
    /// `XQST0090` — a character reference does not identify a valid XML character.
    Xqst0090 => "XQST0090", static;
    /// `XQDY0091` — `xml:id` was constructed with an invalid value.
    Xqdy0091 => "XQDY0091", dynamic;
    /// `XQDY0092` — `xml:space` was constructed with a value other than `preserve` or `default`.
    Xqdy0092 => "XQDY0092", dynamic;
    /// `XQST0079` — an extension expression has no recognized pragma and no expression.
    Xqst0079 => "XQST0079", static;
    /// `XQST0093` — a module imports itself, directly or indirectly.
    Xqst0093 => "XQST0093", static;
    /// `XQST0094` — a `group by` clause names a variable that is not in scope.
    Xqst0094 => "XQST0094", static;
    /// `XQDY0096` — a computed element's name is in a reserved namespace.
    Xqdy0096 => "XQDY0096", dynamic;
    /// `XQST0097` — a decimal-format property has an invalid value.
    Xqst0097 => "XQST0097", static;
    /// `XQST0098` — a decimal format assigns the same character to two different roles.
    Xqst0098 => "XQST0098", static;
    /// `XQST0099` — the prolog declares the context item type more than once.
    Xqst0099 => "XQST0099", static;
    /// `XQTY0105` — the content of an element constructor holds a function item.
    Xqty0105 => "XQTY0105", dynamic;
    /// `XQDY0101` — a computed namespace constructor binds a reserved prefix or namespace.
    Xqdy0101 => "XQDY0101", dynamic;
    /// `XQDY0102` — a constructed element has conflicting default namespace declarations.
    Xqdy0102 => "XQDY0102", dynamic;
    /// `XQST0103` — a window clause repeats a variable name.
    Xqst0103 => "XQST0103", static;
    /// `XQST0106` — a function declaration has conflicting annotations.
    Xqst0106 => "XQST0106", static;
    /// `XQST0108` — an output declaration appears in a library module.
    Xqst0108 => "XQST0108", static;
    /// `XQST0109` — an output declaration names something that is not a serialization
    /// parameter.
    Xqst0109 => "XQST0109", static;
    /// `XQST0110` — the prolog sets the same serialization parameter twice.
    Xqst0110 => "XQST0110", static;
    /// `XQST0111` — the prolog declares the same decimal format twice.
    Xqst0111 => "XQST0111", static;
    /// `XQST0118` — the name in an element constructor's end tag does not match its start tag.
    Xqst0118 => "XQST0118", static;
    /// `XQST0114` — a decimal format declaration repeats a property.
    Xqst0114 => "XQST0114", static;
    /// `XQST0116` — a variable declaration has conflicting annotations.
    Xqst0116 => "XQST0116", static;
    /// `XQST0125` — an inline function is annotated `%public` or `%private`.
    Xqst0125 => "XQST0125", static;
    /// `XQST0113` — a library module's context item declaration supplies a value.
    Xqst0113 => "XQST0113", static;
    /// `XQST0134` — the namespace axis is not supported in XQuery.
    Xqst0134 => "XQST0134", static;
    /// `FOER0000` — the unidentified error `fn:error` raises by default.
    Foer0000 => "FOER0000", dynamic;
}

impl XQueryErrorCode {
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

impl fmt::Display for XQueryErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error raised while compiling or evaluating a query.
///
/// # Examples
///
/// ```
/// use oxixml_xquery::{StaticQueryContext, XQuery, XQueryErrorCode};
///
/// let failure = XQuery::compile("<a>{", StaticQueryContext::new())
///     .expect_err("an unterminated constructor");
/// assert_eq!(failure.code(), Some(XQueryErrorCode::Xpst0003));
/// assert!(failure.is_static());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XQueryError {
    code: Option<XQueryErrorCode>,
    name: ExpandedName,
    message: String,
    offset: Option<usize>,
    /// The parts only `fn:error` and a `catch` clause ever look at, boxed so that an
    /// `XQueryError` stays small enough to travel in a `Result` without a `Box` of its own.
    details: Option<Box<Details>>,
}

/// The rarely-used half of an error.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Details {
    error_object: Option<Sequence>,
    description: Option<String>,
}

/// The result of a fallible XQuery operation.
pub type XQueryResult<T> = Result<T, XQueryError>;

impl XQueryError {
    /// An error with a code this crate knows.
    #[must_use]
    pub fn new(code: XQueryErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: Some(code),
            name: code.expanded_name(),
            message: message.into(),
            offset: None,
            details: None,
        }
    }

    /// An error with an arbitrary qualified name, as `fn:error` may raise.
    #[must_use]
    pub fn with_name(name: ExpandedName, message: impl Into<String>) -> Self {
        let code = (name.namespace() == Some(ERROR_NAMESPACE))
            .then(|| XQueryErrorCode::from_spelling(name.local_name()))
            .flatten();
        Self {
            code,
            name,
            message: message.into(),
            offset: None,
            details: None,
        }
    }

    /// Records the byte offset in the query source where the problem was detected.
    #[must_use]
    pub const fn at(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Attaches the error object a `catch` clause binds to `$err:value`.
    #[must_use]
    pub fn with_error_object(mut self, error_object: Sequence) -> Self {
        self.details.get_or_insert_default().error_object = Some(error_object);
        self
    }

    /// Attaches the description a `catch` clause binds to `$err:description`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.details.get_or_insert_default().description = Some(description.into());
        self
    }

    /// The code, when it is one this crate knows.
    #[must_use]
    pub const fn code(&self) -> Option<XQueryErrorCode> {
        self.code
    }

    /// The error's qualified name, which is always meaningful.
    #[must_use]
    pub const fn name(&self) -> &ExpandedName {
        &self.name
    }

    /// The human-readable message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The description a `catch` clause binds to `$err:description`.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.details
            .as_ref()
            .and_then(|details| details.description.as_deref())
    }

    /// The byte offset in the query source, when one was recorded.
    #[must_use]
    pub const fn offset(&self) -> Option<usize> {
        self.offset
    }

    /// The error object a `catch` clause binds to `$err:value`.
    #[must_use]
    pub fn error_object(&self) -> Option<&Sequence> {
        self.details
            .as_ref()
            .and_then(|details| details.error_object.as_ref())
    }

    /// Whether this error carries `code`.
    #[must_use]
    pub fn is(&self, code: XQueryErrorCode) -> bool {
        self.code == Some(code)
    }

    /// Whether the error was detected while compiling.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.code.is_some_and(XQueryErrorCode::is_static)
    }

    /// A syntax error, `XPST0003`, at `offset`.
    #[must_use]
    pub fn syntax(message: impl Into<String>, offset: usize) -> Self {
        Self::new(XQueryErrorCode::Xpst0003, message).at(offset)
    }
}

impl fmt::Display for XQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "{code}: {}", self.message),
            None => write!(f, "{}: {}", self.name, self.message),
        }
    }
}

impl Error for XQueryError {}

impl From<XPathError> for XQueryError {
    fn from(error: XPathError) -> Self {
        let mut converted = Self::with_name(error.name().clone(), error.message().to_owned());
        if let Some(offset) = error.offset() {
            converted = converted.at(offset);
        }
        if let Some(object) = error.error_object() {
            converted = converted.with_error_object(object.clone());
        }
        converted
    }
}

impl From<XQueryError> for XPathError {
    fn from(error: XQueryError) -> Self {
        let mut converted = Self::with_name(error.name.clone(), error.message.clone());
        if let Some(object) = error.error_object() {
            converted = converted.with_error_object(object.clone());
        }
        converted
    }
}

impl From<FnError> for XQueryError {
    fn from(error: FnError) -> Self {
        Self::from(XPathError::from(error))
    }
}

impl From<XdmError> for XQueryError {
    fn from(error: XdmError) -> Self {
        Self::from(XPathError::from(error))
    }
}

impl From<XPathErrorCode> for XQueryError {
    fn from(code: XPathErrorCode) -> Self {
        Self::with_name(code.expanded_name(), code.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_through_their_spelling() {
        for code in XQueryErrorCode::ALL {
            assert_eq!(XQueryErrorCode::from_spelling(code.as_str()), Some(*code));
            assert_eq!(code.expanded_name().namespace(), Some(ERROR_NAMESPACE));
        }
    }

    #[test]
    fn an_xpath_error_keeps_its_identity() {
        let inner = XPathError::new(XPathErrorCode::Xpty0004, "wrong type");
        let outer = XQueryError::from(inner);
        assert_eq!(outer.code(), Some(XQueryErrorCode::Xpty0004));
        assert_eq!(outer.name().local_name(), "XPTY0004");
        assert!(!outer.is_static());
    }

    #[test]
    fn an_unknown_name_has_no_code() {
        let error = XQueryError::with_name(ExpandedName::new(Some("urn:x"), "boom"), "custom");
        assert!(error.code().is_none());
        assert_eq!(error.to_string(), "{urn:x}boom: custom");
    }

    #[test]
    fn static_errors_report_an_offset() {
        let error = XQueryError::syntax("unexpected `}`", 7);
        assert!(error.is_static());
        assert_eq!(error.offset(), Some(7));
    }
}
