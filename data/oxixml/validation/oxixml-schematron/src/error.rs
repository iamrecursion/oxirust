//! [`SchematronError`] — everything that can go wrong compiling a schema or validating a
//! document against it.

use std::fmt;
use std::sync::Arc;

use oxixml_infoset::XdmError;
use oxixml_xpath::XPathError;

/// The result type this crate's fallible operations return.
pub type SchematronResult<T> = Result<T, SchematronError>;

/// Everything [`crate::Schematron::compile`] and [`crate::Compiled::validate`] can fail with.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SchematronError {
    /// The schema text is not well-formed XML, or an instance document is not well-formed XML.
    Xml(Arc<XdmError>),
    /// The document element of the schema is not `sch:schema` (in the ISO or the 1.5
    /// namespace).
    NotASchema {
        /// The element name found instead, as `{namespace}local` or just `local`.
        found: String,
    },
    /// A required attribute or child element was missing.
    MissingElement {
        /// The element that was expected.
        what: &'static str,
        /// Where it was expected, for a useful message.
        context: String,
    },
    /// A required attribute was missing.
    MissingAttribute {
        /// The attribute name.
        name: &'static str,
        /// The element that was missing it, for a useful message.
        context: String,
    },
    /// `sch:include` was encountered but no [`crate::parse::IncludeResolver`] was supplied.
    UnresolvedInclude {
        /// The `href` that could not be resolved.
        href: String,
    },
    /// `sch:include` nesting exceeded the depth guard, which most likely means a cycle.
    IncludeCycle {
        /// The `href` at which the guard tripped.
        href: String,
    },
    /// An `is-a`, `extends` or `active` attribute named an id that does not exist.
    UnknownReference {
        /// What kind of thing was being looked up (`"pattern"`, `"rule"`, `"phase"`,
        /// `"diagnostic"`).
        kind: &'static str,
        /// The id that could not be found.
        id: String,
    },
    /// `extends` formed a cycle among rules.
    ExtendsCycle {
        /// The rule id at which the cycle was detected.
        id: String,
    },
    /// `extends`/inline-`extends` nesting exceeded the depth guard.
    ///
    /// Unlike [`ExtendsCycle`](Self::ExtendsCycle), this does not mean a rule extends itself —
    /// it means a chain of *distinct* rules nested deeper than the guard allows, which is
    /// exactly as unsafe to leave unbounded as a cycle would be.
    ExtendsDepth {
        /// The depth guard that was exceeded.
        limit: usize,
    },
    /// An XPath expression embedded in the schema (a context, a test, a `let`, a `value-of`
    /// select, …) failed to compile.
    XPath {
        /// The source text that failed to compile.
        expression: String,
        /// The underlying compiler error, boxed because [`XPathError`] is large and this
        /// variant would otherwise dominate `SchematronError`'s size.
        source: Box<XPathError>,
    },
    /// Evaluating a compiled expression against a document failed at run time. Boxed for the
    /// same reason as [`SchematronError::XPath`].
    Evaluation(Box<XPathError>),
}

impl fmt::Display for SchematronError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(source) => write!(f, "not well-formed XML: {source}"),
            Self::NotASchema { found } => {
                write!(f, "the document element is `{found}`, not `schema`")
            }
            Self::MissingElement { what, context } => {
                write!(f, "`{context}` is missing its required `{what}`")
            }
            Self::MissingAttribute { name, context } => {
                write!(f, "`{context}` is missing its required `@{name}`")
            }
            Self::UnresolvedInclude { href } => {
                write!(
                    f,
                    "`sch:include href=\"{href}\"` needs an IncludeResolver, and none was supplied"
                )
            }
            Self::IncludeCycle { href } => {
                write!(
                    f,
                    "`sch:include` nesting is too deep at `{href}`, likely a cycle"
                )
            }
            Self::UnknownReference { kind, id } => {
                write!(f, "no {kind} with id `{id}`")
            }
            Self::ExtendsCycle { id } => {
                write!(f, "`extends` forms a cycle at rule `{id}`")
            }
            Self::ExtendsDepth { limit } => {
                write!(
                    f,
                    "`extends` nests rules more than {limit} levels deep (this limit is not \
                     configurable)"
                )
            }
            Self::XPath { expression, source } => {
                write!(f, "`{expression}` failed to compile: {source}")
            }
            Self::Evaluation(source) => write!(f, "evaluation failed: {source}"),
        }
    }
}

impl std::error::Error for SchematronError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Xml(source) => Some(source.as_ref()),
            Self::XPath { source, .. } => Some(source.as_ref()),
            Self::Evaluation(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<XdmError> for SchematronError {
    fn from(source: XdmError) -> Self {
        Self::Xml(Arc::new(source))
    }
}
