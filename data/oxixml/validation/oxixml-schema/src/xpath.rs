//! The *restricted XPath subset* that identity constraints are written in,
//! specified as a grammar in
//! [XML Schema Part 1, Appendix I](https://www.w3.org/TR/xmlschema-1/#c-selector-xpath).
//!
//! ```text
//! Selector ::= Path ( '|' Path )*
//! Path     ::= ('.//')? Step ( '/' Step )*
//! Step     ::= '.' | NameTest
//! Field    ::= Path ( '|' Path )*  where the last step of a Path may be '@' NameTest
//! NameTest ::= QName | '*' | NCName ':' '*'
//! ```
//!
//! The `child::` and `attribute::` axis names may be written out in full. As the
//! specification requires, an unprefixed `QName` in a selector or field has **no
//! namespace**: the default namespace declaration does not apply to it.

use oxixml_qname::{ExpandedName, is_ncname};

use crate::error::SchemaError;
use crate::value::NamespaceContext;

/// A name test in a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameTest {
    /// `*` — any name.
    Any,
    /// `prefix:*` — any local name in the given namespace.
    AnyInNamespace(String),
    /// A specific expanded name.
    Name(ExpandedName),
}

impl NameTest {
    /// Whether the test matches a node with this expanded name.
    #[must_use]
    pub fn matches(&self, namespace: Option<&str>, local: &str) -> bool {
        match self {
            Self::Any => true,
            Self::AnyInNamespace(uri) => namespace == Some(uri.as_str()),
            Self::Name(name) => name.namespace() == namespace && name.local_name() == local,
        }
    }
}

/// One step of a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// `.` — the context node itself.
    SelfNode,
    /// `child::NameTest`.
    Child(NameTest),
}

/// One `Path` alternative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    /// Whether the path starts with `.//`.
    pub descendant_or_self: bool,
    /// The element steps.
    pub steps: Vec<Step>,
    /// A trailing `@NameTest`, in a *field* path.
    pub attribute: Option<NameTest>,
}

/// A parsed `selector` or `field` XPath: a union of paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathSet {
    /// The alternatives, in the order they were written.
    pub paths: Vec<Path>,
    /// The expression as written, for diagnostics.
    pub source: String,
}

impl PathSet {
    /// Parses a `selector` expression (no attribute steps allowed).
    ///
    /// # Errors
    ///
    /// Returns `c-selector-xpath` when the expression is outside the subset.
    pub fn parse_selector(
        source: &str,
        namespaces: &NamespaceContext,
    ) -> Result<Self, SchemaError> {
        Self::parse(source, namespaces, None, false)
    }

    /// Parses a `selector` expression, resolving unprefixed name tests in
    /// `default_namespace`.
    ///
    /// XSD 1.1 §3.13.1 lets `xs:selector` carry `xpathDefaultNamespace`, which
    /// — unlike an ordinary XPath default namespace — *does* apply to the
    /// unprefixed names of the identity-constraint subset
    /// (`saxonData/Id/id050` … `id053`).
    ///
    /// # Errors
    ///
    /// The same as [`parse_selector`](Self::parse_selector).
    pub fn parse_selector_in(
        source: &str,
        namespaces: &NamespaceContext,
        default_namespace: Option<&str>,
    ) -> Result<Self, SchemaError> {
        Self::parse(source, namespaces, default_namespace, false)
    }

    /// Parses a `field` expression, resolving unprefixed name tests in
    /// `default_namespace`.
    ///
    /// # Errors
    ///
    /// The same as [`parse_field`](Self::parse_field).
    pub fn parse_field_in(
        source: &str,
        namespaces: &NamespaceContext,
        default_namespace: Option<&str>,
    ) -> Result<Self, SchemaError> {
        Self::parse(source, namespaces, default_namespace, true)
    }

    /// Parses a `field` expression (a trailing attribute step is allowed).
    ///
    /// # Errors
    ///
    /// Returns `c-fields-xpaths` when the expression is outside the subset.
    pub fn parse_field(source: &str, namespaces: &NamespaceContext) -> Result<Self, SchemaError> {
        Self::parse(source, namespaces, None, true)
    }

    fn parse(
        source: &str,
        namespaces: &NamespaceContext,
        default_namespace: Option<&str>,
        allow_attribute: bool,
    ) -> Result<Self, SchemaError> {
        let code = if allow_attribute {
            "c-fields-xpaths"
        } else {
            "c-selector-xpath"
        };
        // The subset has no literals, so whitespace is never significant
        // *between* tokens and can be removed before parsing: `child :: a / @b`
        // and `child::a/@b` are the same expression. It is not allowed to fall
        // *inside* a token, though — `xpns :*` splits a `QName`, and `./ /.`
        // splits the `//` of `.//` — so the split points are checked first.
        let condensed = condense(source, code)?;
        let mut paths = Vec::new();
        for alternative in condensed.split('|') {
            paths.push(parse_path(
                alternative,
                namespaces,
                default_namespace,
                allow_attribute,
                code,
            )?);
        }
        if paths.is_empty() {
            return Err(SchemaError::new(code, "an XPath expression is required"));
        }
        Ok(Self {
            paths,
            source: source.to_owned(),
        })
    }
}

/// Strips the whitespace between tokens, rejecting whitespace that would fall
/// inside one.
///
/// XPath 1.0 §3.7 allows `ExprWhitespace` only *between* tokens. The tokens
/// this subset can hold are `.`, `/`, `//`, `|`, `@`, `::`, `*` and `QName`,
/// so whitespace may not sit between the two slashes of a `//`, on either side
/// of the `:` inside a `QName`, or in the middle of an `NCName`. It may sit on
/// either side of a whole `::` axis separator.
fn condense(source: &str, code: &'static str) -> Result<String, SchemaError> {
    let mut characters: Vec<char> = Vec::with_capacity(source.len());
    let mut gap_before: Vec<bool> = Vec::with_capacity(source.len());
    let mut gap = false;
    for character in source.chars() {
        if matches!(character, ' ' | '\t' | '\n' | '\r') {
            gap = !characters.is_empty();
            continue;
        }
        gap_before.push(gap);
        characters.push(character);
        gap = false;
    }
    let broken = |reason: &str| {
        SchemaError::new(code, format!("`{source}` splits {reason} with whitespace"))
    };
    for index in 0..characters.len() {
        if !gap_before[index] || index == 0 {
            continue;
        }
        let previous = characters[index - 1];
        let current = characters[index];
        let opens_axis = current == ':' && characters.get(index + 1) == Some(&':');
        let closes_axis = previous == ':' && index >= 2 && characters[index - 2] == ':';
        if (current == ':' && !opens_axis) || (previous == ':' && !closes_axis) {
            return Err(broken("a QName"));
        }
        if previous == '/' && current == '/' {
            return Err(broken("a `//` step"));
        }
        if is_name_character(previous) && is_name_character(current) {
            return Err(broken("a name"));
        }
    }
    Ok(characters.into_iter().collect())
}

fn is_name_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-' | '.')
}

fn parse_path(
    text: &str,
    namespaces: &NamespaceContext,
    default_namespace: Option<&str>,
    allow_attribute: bool,
    code: &'static str,
) -> Result<Path, SchemaError> {
    let trimmed = text;
    if trimmed.is_empty() {
        return Err(SchemaError::new(code, "empty path in an XPath expression"));
    }
    let (descendant_or_self, rest) = match trimmed.strip_prefix(".//") {
        Some(rest) => (true, rest),
        None => (false, trimmed),
    };
    if rest.starts_with('/') {
        return Err(SchemaError::new(
            code,
            format!("`{text}` is an absolute path, which the subset forbids"),
        ));
    }
    let mut steps = Vec::new();
    let mut attribute = None;
    let raw_steps: Vec<&str> = rest.split('/').collect();
    let last = raw_steps.len() - 1;
    for (index, raw) in raw_steps.iter().enumerate() {
        let step = *raw;
        if step.is_empty() {
            return Err(SchemaError::new(
                code,
                format!("`{text}` contains an empty step"),
            ));
        }
        let is_attribute = step.starts_with('@') || step.starts_with("attribute::");
        if is_attribute {
            if !allow_attribute {
                return Err(SchemaError::new(
                    code,
                    format!("`{text}` uses the attribute axis, which a selector may not"),
                ));
            }
            if index != last {
                return Err(SchemaError::new(
                    code,
                    format!("`{text}` uses the attribute axis before the last step"),
                ));
            }
            let name = step
                .strip_prefix('@')
                .or_else(|| step.strip_prefix("attribute::"))
                .unwrap_or(step);
            attribute = Some(parse_name_test(name, namespaces, None, code)?);
            continue;
        }
        if step == "." {
            steps.push(Step::SelfNode);
            continue;
        }
        let name = step.strip_prefix("child::").unwrap_or(step);
        steps.push(Step::Child(parse_name_test(
            name,
            namespaces,
            default_namespace,
            code,
        )?));
    }
    Ok(Path {
        descendant_or_self,
        steps,
        attribute,
    })
}

fn parse_name_test(
    text: &str,
    namespaces: &NamespaceContext,
    default_namespace: Option<&str>,
    code: &'static str,
) -> Result<NameTest, SchemaError> {
    if text == "*" {
        return Ok(NameTest::Any);
    }
    match text.split_once(':') {
        Some((prefix, local)) => {
            if !is_ncname(prefix) {
                return Err(SchemaError::new(
                    code,
                    format!("`{text}` is not a legal name test"),
                ));
            }
            let namespace = namespaces.lookup(Some(prefix)).ok_or_else(|| {
                SchemaError::new(code, format!("the prefix `{prefix}` is not declared"))
            })?;
            if local == "*" {
                Ok(NameTest::AnyInNamespace(namespace.to_owned()))
            } else if is_ncname(local) {
                Ok(NameTest::Name(ExpandedName::new(Some(namespace), local)))
            } else {
                Err(SchemaError::new(
                    code,
                    format!("`{text}` is not a legal name test"),
                ))
            }
        }
        None => {
            if is_ncname(text) {
                // The default namespace of the schema document deliberately
                // does not apply here — only `xpathDefaultNamespace` does
                // (XSD 1.1 §3.13.1).
                Ok(NameTest::Name(ExpandedName::new(default_namespace, text)))
            } else {
                Err(SchemaError::new(
                    code,
                    format!("`{text}` is not a legal name test"),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> NamespaceContext {
        let mut namespaces = NamespaceContext::new();
        namespaces.declare(Some("p"), "urn:x");
        namespaces.declare(None, "urn:default");
        namespaces
    }

    #[test]
    fn selector_paths() -> Result<(), SchemaError> {
        let set = PathSet::parse_selector(".//a/p:b|c", &context())?;
        assert_eq!(set.paths.len(), 2);
        assert!(set.paths[0].descendant_or_self);
        assert_eq!(set.paths[0].steps.len(), 2);
        assert!(!set.paths[1].descendant_or_self);
        Ok(())
    }

    #[test]
    fn unprefixed_names_have_no_namespace() -> Result<(), SchemaError> {
        let set = PathSet::parse_selector("a", &context())?;
        assert_eq!(
            set.paths[0].steps[0],
            Step::Child(NameTest::Name(ExpandedName::new(None::<String>, "a")))
        );
        Ok(())
    }

    #[test]
    fn fields_may_end_in_an_attribute() -> Result<(), SchemaError> {
        let set = PathSet::parse_field("./a/@id", &context())?;
        assert!(set.paths[0].attribute.is_some());
        assert!(PathSet::parse_selector("a/@id", &context()).is_err());
        assert!(PathSet::parse_field("@id/a", &context()).is_err());
        Ok(())
    }

    #[test]
    fn absolute_paths_are_rejected() {
        assert!(PathSet::parse_selector("/a", &context()).is_err());
        assert!(PathSet::parse_selector("a//b", &context()).is_err());
    }

    #[test]
    fn axis_names_may_be_spelled_out() -> Result<(), SchemaError> {
        let set = PathSet::parse_field("child::a/attribute::b", &context())?;
        assert_eq!(set.paths[0].steps.len(), 1);
        assert!(set.paths[0].attribute.is_some());
        Ok(())
    }
}
