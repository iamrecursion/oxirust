//! Datatype libraries.
//!
//! A RELAX NG datatype library is identified by a URI and offers exactly two
//! services ([§6.2.8](https://relaxng.org/spec-20011203.html#IDA0YZR)):
//!
//! * `datatypeAllows(u, ln, params, s, cx)` — is `s`, read in context `cx`, a
//!   legal value of the datatype `ln` with the parameters `params`?
//! * `datatypeEqual(u, ln, s1, cx1, s2, cx2)` — do `s1` and `s2` denote the same
//!   value?
//!
//! Two libraries are implemented: the built-in one identified by the empty URI
//! ([§6.2.9](https://relaxng.org/spec-20011203.html#IDAK1ZR)), which provides
//! `string` and `token`, and the W3C XML Schema library, which provides the whole
//! XSD 1.0 built-in datatype hierarchy with its constraining facets exposed as
//! `param` elements.

pub(crate) mod decimal;
pub(crate) mod xsd;

use std::fmt;
use std::rc::Rc;

use crate::error::{SchemaError, datatype as datatype_error};
use crate::tree::{Element, NamespaceMap, normalize_whitespace};

pub(crate) use xsd::LIBRARY as XSD_LIBRARY;

/// The part of a RELAX NG *context* that datatypes can see: the namespace map,
/// which `QName` and `NOTATION` need.
#[derive(Debug, Clone)]
pub(crate) struct DatatypeContext {
    /// The namespace bindings in scope.
    pub namespaces: Rc<NamespaceMap>,
}

impl DatatypeContext {
    /// The `context(u, cx)` operation of §6.2.8: the same context with `u` as the
    /// default namespace.
    pub(crate) fn with_default_namespace(&self, namespace: &str) -> Self {
        Self {
            namespaces: Rc::new(self.namespaces.with_default(namespace)),
        }
    }
}

impl Default for DatatypeContext {
    fn default() -> Self {
        Self {
            namespaces: Rc::new(NamespaceMap::default()),
        }
    }
}

/// Why a datatype or one of its parameters could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParamError {
    /// The datatype library is not one this validator implements.
    UnknownLibrary(String),
    /// The library has no datatype with that name.
    UnknownType(String),
    /// The datatype has no such parameter.
    UnknownParam(String),
    /// The parameter cannot be repeated.
    Duplicate(String),
    /// The parameter exists but does not apply to this datatype.
    NotApplicable(String, String),
    /// The parameter value is not legal.
    BadValue(String, String),
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownLibrary(uri) => write!(f, "unsupported datatype library `{uri}`"),
            Self::UnknownType(name) => write!(f, "unknown datatype `{name}`"),
            Self::UnknownParam(name) => write!(f, "unknown parameter `{name}`"),
            Self::Duplicate(name) => write!(f, "the parameter `{name}` may not be repeated"),
            Self::NotApplicable(param, ty) => {
                write!(f, "the parameter `{param}` does not apply to `{ty}`")
            }
            Self::BadValue(param, value) => {
                write!(
                    f,
                    "`{value}` is not a legal value for the parameter `{param}`"
                )
            }
        }
    }
}

impl std::error::Error for ParamError {}

/// A datatype together with the parameters a `data` pattern applied to it.
#[derive(Debug)]
pub(crate) enum Datatype {
    /// The built-in `string` datatype: every string is allowed, and equality is
    /// string identity.
    BuiltinString,
    /// The built-in `token` datatype: every string is allowed, and equality is
    /// identity after whitespace normalization.
    BuiltinToken,
    /// A datatype from the W3C XML Schema library.
    Xsd(Box<xsd::XsdDatatype>),
}

impl Datatype {
    /// Compiles a datatype reference.
    pub(crate) fn compile(
        library: &str,
        name: &str,
        params: &[(String, String)],
        context: &DatatypeContext,
    ) -> Result<Self, ParamError> {
        match library {
            "" => {
                if let Some((param, _)) = params.first() {
                    return Err(ParamError::NotApplicable(param.clone(), name.to_owned()));
                }
                match name {
                    "string" => Ok(Self::BuiltinString),
                    "token" => Ok(Self::BuiltinToken),
                    other => Err(ParamError::UnknownType(other.to_owned())),
                }
            }
            XSD_LIBRARY => xsd::XsdDatatype::compile(name, params, context)
                .map(|datatype| Self::Xsd(Box::new(datatype))),
            other => Err(ParamError::UnknownLibrary(other.to_owned())),
        }
    }

    /// `datatypeAllows`.
    pub(crate) fn allows(&self, value: &str, context: &DatatypeContext) -> bool {
        match self {
            Self::BuiltinString | Self::BuiltinToken => true,
            Self::Xsd(datatype) => datatype.allows(value, context),
        }
    }

    /// `datatypeEqual`.
    pub(crate) fn equal(
        &self,
        left: &str,
        left_context: &DatatypeContext,
        right: &str,
        right_context: &DatatypeContext,
    ) -> bool {
        match self {
            Self::BuiltinString => left == right,
            Self::BuiltinToken => normalize_whitespace(left) == normalize_whitespace(right),
            Self::Xsd(datatype) => {
                match (
                    datatype.value(left, left_context),
                    datatype.value(right, right_context),
                ) {
                    (Some(left), Some(right)) => left == right,
                    _ => false,
                }
            }
        }
    }
}

/// The set of datatype libraries this validator implements.
///
/// The type exists so that rule 4.16 ("a `data` or `value` element must be
/// correct in its use of datatypes") can be checked before the schema is
/// compiled, with the same code that compiles it.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Libraries;

impl Libraries {
    /// Checks the §4.16 datatype constraint on a `data` or `value` element.
    pub(crate) fn check_usage(self, element: &Element) -> Result<(), SchemaError> {
        let library = element.attribute("datatypeLibrary").unwrap_or_default();
        let name = element.attribute("type").unwrap_or_default();
        let params = params_of(element);
        let context = DatatypeContext {
            namespaces: Rc::clone(&element.context.namespaces),
        };
        let datatype = Datatype::compile(library, name, &params, &context)
            .map_err(|error| datatype_error("4.16", error.to_string()).at(element.position))?;
        if element.name == "value" {
            // `datatypeEqual` is required to be reflexive, which presupposes
            // `datatypeAllows`: a literal the datatype does not accept could
            // never match anything, so the schema is not correct in its use of
            // the datatype.
            let literal = element.text();
            let context =
                context.with_default_namespace(element.attribute("ns").unwrap_or_default());
            if !datatype.allows(&literal, &context) {
                return Err(datatype_error(
                    "4.16",
                    format!("`{literal}` is not a legal value of the datatype `{name}`"),
                )
                .at(element.position));
            }
        }
        Ok(())
    }
}

/// Collects the `param` children of a `data` element as name/value pairs.
pub(crate) fn params_of(element: &Element) -> Vec<(String, String)> {
    element
        .child_elements()
        .filter(|child| child.name == "param")
        .map(|child| {
            (
                child.attribute("name").unwrap_or_default().to_owned(),
                child.text(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> DatatypeContext {
        DatatypeContext::default()
    }

    #[test]
    fn the_builtin_library_has_two_datatypes() -> Result<(), ParamError> {
        assert!(Datatype::compile("", "string", &[], &context()).is_ok());
        assert!(Datatype::compile("", "token", &[], &context()).is_ok());
        assert!(Datatype::compile("", "int", &[], &context()).is_err());
        let token = Datatype::compile("", "token", &[], &context())?;
        assert!(token.equal(" a  b ", &context(), "a b", &context()));
        let string = Datatype::compile("", "string", &[], &context())?;
        assert!(!string.equal(" a  b ", &context(), "a b", &context()));
        Ok(())
    }

    #[test]
    fn the_builtin_library_takes_no_parameters() {
        assert!(
            Datatype::compile(
                "",
                "string",
                &[("length".to_owned(), "1".to_owned())],
                &context()
            )
            .is_err()
        );
    }

    #[test]
    fn an_unknown_library_is_reported() {
        let error = Datatype::compile("http://example.com/nope", "x", &[], &context())
            .expect_err("an unsupported library");
        assert_eq!(
            error,
            ParamError::UnknownLibrary("http://example.com/nope".to_owned())
        );
    }
}
