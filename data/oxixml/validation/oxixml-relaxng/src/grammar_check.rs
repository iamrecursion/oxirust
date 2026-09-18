//! The **full syntax** check of
//! [§3](https://relaxng.org/spec-20011203.html#syntax).
//!
//! Simplification assumes the schema already matches the grammar of §3, so the
//! whole grammar is verified up front: which elements may appear where, which
//! attributes each element may carry, how many children each element needs, and
//! which elements may have string children. Getting this right is what makes a
//! validator reject the roughly one hundred deliberately-malformed schemas of the
//! RELAX NG specification test suite.

use crate::error::{SchemaError, grammar};
use crate::name::{is_ncname, is_qname};
use crate::tree::{Child, Element, RNG_NS, is_whitespace};
use crate::uri::{parse_uri_reference, xlink_escape};

/// Which production an element is being checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Pattern,
    NameClass,
    GrammarContent,
    IncludeContent,
}

impl Category {
    const fn describe(self) -> &'static str {
        match self {
            Self::Pattern => "a pattern",
            Self::NameClass => "a name class",
            Self::GrammarContent => "grammar content",
            Self::IncludeContent => "include content",
        }
    }
}

/// Checks that `element` matches the `pattern` production of §3.
pub(crate) fn check_pattern(element: &Element) -> Result<(), SchemaError> {
    check(element, Category::Pattern)
}

/// Checks that `element` matches the `grammar` production of §3.
pub(crate) fn check_grammar(element: &Element) -> Result<(), SchemaError> {
    if element.name != "grammar" {
        return Err(grammar(format!(
            "expected a grammar element, found `{}`",
            element.name
        ))
        .at(element.position));
    }
    check(element, Category::Pattern)
}

fn check(element: &Element, category: Category) -> Result<(), SchemaError> {
    check_common_attributes(element)?;
    if !matches!(element.name.as_str(), "value" | "param" | "name")
        && element
            .children
            .iter()
            .any(|child| matches!(child, Child::Text(text) if !is_whitespace(text)))
    {
        return Err(grammar(format!(
            "the `{}` element must not have string children",
            element.name
        ))
        .at(element.position));
    }

    match category {
        Category::Pattern => check_pattern_element(element),
        Category::NameClass => check_name_class_element(element),
        Category::GrammarContent => check_grammar_content(element, false),
        Category::IncludeContent => check_grammar_content(element, true),
    }
}

fn check_pattern_element(element: &Element) -> Result<(), SchemaError> {
    let children: Vec<&Element> = element.child_elements().collect();
    match element.name.as_str() {
        "element" => {
            allow_attributes(element, &["name"])?;
            check_named_or_name_class(element, &children, 1, None)
        }
        "attribute" => {
            allow_attributes(element, &["name"])?;
            check_named_or_name_class(element, &children, 0, Some(1))
        }
        "group" | "interleave" | "choice" | "optional" | "zeroOrMore" | "oneOrMore" | "list"
        | "mixed" => {
            allow_attributes(element, &[])?;
            require_children(element, children.len(), 1, None)?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::Pattern))
        }
        "ref" | "parentRef" => {
            allow_attributes(element, &["name"])?;
            require_ncname_attribute(element, "name")?;
            require_children(element, children.len(), 0, Some(0))
        }
        "empty" | "text" | "notAllowed" => {
            allow_attributes(element, &[])?;
            require_children(element, children.len(), 0, Some(0))
        }
        "value" => {
            allow_attributes(element, &["type"])?;
            if let Some(value) = element.attribute("type")
                && !is_ncname(value)
            {
                return Err(grammar(format!(
                    "the type attribute of a value element must be an NCName, found `{value}`"
                ))
                .at(element.position));
            }
            require_children(element, children.len(), 0, Some(0))
        }
        "data" => {
            allow_attributes(element, &["type"])?;
            require_ncname_attribute(element, "type")?;
            check_data_children(element, &children)
        }
        "externalRef" => {
            allow_attributes(element, &["href"])?;
            require_uri_attribute(element, "href")?;
            require_children(element, children.len(), 0, Some(0))
        }
        "grammar" => {
            allow_attributes(element, &[])?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::GrammarContent))
        }
        other => Err(
            grammar(format!("`{other}` is not {}", Category::Pattern.describe()))
                .at(element.position),
        ),
    }
}

/// `element`/`attribute`: either a `name` attribute plus patterns, or a leading
/// name class followed by patterns.
fn check_named_or_name_class(
    element: &Element,
    children: &[&Element],
    min_patterns: usize,
    max_patterns: Option<usize>,
) -> Result<(), SchemaError> {
    let patterns = match element.attribute("name") {
        Some(name) => {
            if !is_qname(name) {
                return Err(grammar(format!(
                    "the name attribute of an {} element must be a QName, found `{name}`",
                    element.name
                ))
                .at(element.position));
            }
            children
        }
        None => {
            let Some((first, rest)) = children.split_first() else {
                return Err(grammar(format!(
                    "an {} element without a name attribute must start with a name class",
                    element.name
                ))
                .at(element.position));
            };
            check(first, Category::NameClass)?;
            rest
        }
    };
    require_children(element, patterns.len(), min_patterns, max_patterns)?;
    patterns
        .iter()
        .try_for_each(|child| check(child, Category::Pattern))
}

/// `data`: `param*` followed by an optional `except`.
fn check_data_children(_element: &Element, children: &[&Element]) -> Result<(), SchemaError> {
    let mut seen_except = false;
    for child in children {
        match child.name.as_str() {
            "param" if !seen_except => {
                allow_attributes(child, &["name"])?;
                require_ncname_attribute(child, "name")?;
                if child.child_elements().next().is_some() {
                    return Err(
                        grammar("a param element must not have child elements").at(child.position)
                    );
                }
                check_common_attributes(child)?;
            }
            "param" => {
                return Err(
                    grammar("a param element must come before the except element")
                        .at(child.position),
                );
            }
            "except" if !seen_except => {
                seen_except = true;
                allow_attributes(child, &[])?;
                check_common_attributes(child)?;
                let inner: Vec<&Element> = child.child_elements().collect();
                require_children(child, inner.len(), 1, None)?;
                for pattern in inner {
                    check(pattern, Category::Pattern)?;
                }
            }
            other => {
                return Err(grammar(format!(
                    "a data element may only contain param and except elements, found `{other}`"
                ))
                .at(child.position));
            }
        }
    }
    Ok(())
}

fn check_name_class_element(element: &Element) -> Result<(), SchemaError> {
    let children: Vec<&Element> = element.child_elements().collect();
    match element.name.as_str() {
        "name" => {
            allow_attributes(element, &[])?;
            require_children(element, children.len(), 0, Some(0))?;
            let text = element.text();
            if !is_qname(&text) {
                return Err(grammar(format!(
                    "the content of a name element must be a QName, found `{text}`"
                ))
                .at(element.position));
            }
            Ok(())
        }
        "anyName" | "nsName" => {
            allow_attributes(element, &[])?;
            require_children(element, children.len(), 0, Some(1))?;
            if let Some(except) = children.first() {
                if except.name != "except" {
                    return Err(grammar(format!(
                        "an {} element may only contain an except element, found `{}`",
                        element.name, except.name
                    ))
                    .at(except.position));
                }
                allow_attributes(except, &[])?;
                check_common_attributes(except)?;
                let inner: Vec<&Element> = except.child_elements().collect();
                require_children(except, inner.len(), 1, None)?;
                for name_class in inner {
                    check(name_class, Category::NameClass)?;
                }
            }
            Ok(())
        }
        "choice" => {
            allow_attributes(element, &[])?;
            require_children(element, children.len(), 1, None)?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::NameClass))
        }
        other => Err(grammar(format!(
            "`{other}` is not {}",
            Category::NameClass.describe()
        ))
        .at(element.position)),
    }
}

fn check_grammar_content(element: &Element, include: bool) -> Result<(), SchemaError> {
    let children: Vec<&Element> = element.child_elements().collect();
    let nested = if include {
        Category::IncludeContent
    } else {
        Category::GrammarContent
    };
    match element.name.as_str() {
        "start" => {
            allow_attributes(element, &["combine"])?;
            check_combine(element)?;
            require_children(element, children.len(), 1, Some(1))?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::Pattern))
        }
        "define" => {
            allow_attributes(element, &["name", "combine"])?;
            require_ncname_attribute(element, "name")?;
            check_combine(element)?;
            require_children(element, children.len(), 1, None)?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::Pattern))
        }
        "div" => {
            allow_attributes(element, &[])?;
            children.iter().try_for_each(|child| check(child, nested))
        }
        "include" if !include => {
            allow_attributes(element, &["href"])?;
            require_uri_attribute(element, "href")?;
            children
                .iter()
                .try_for_each(|child| check(child, Category::IncludeContent))
        }
        other => {
            Err(grammar(format!("`{other}` is not {}", nested.describe())).at(element.position))
        }
    }
}

fn check_combine(element: &Element) -> Result<(), SchemaError> {
    match element.attribute("combine") {
        None | Some("choice") | Some("interleave") => Ok(()),
        Some(other) => Err(grammar(format!(
            "the combine attribute must be `choice` or `interleave`, found `{other}`"
        ))
        .at(element.position)),
    }
}

/// Checks the attributes that §3 allows on *any* element: `ns` (any value) and
/// `datatypeLibrary` (an absolute URI reference without a fragment identifier, or
/// the empty string).
fn check_common_attributes(element: &Element) -> Result<(), SchemaError> {
    for attribute in &element.attributes {
        if attribute.namespace == RNG_NS {
            return Err(grammar(format!(
                "an attribute in the RELAX NG namespace is not allowed (`{}` on `{}`)",
                attribute.local, element.name
            ))
            .at(element.position));
        }
    }
    if let Some(value) = element.attribute("datatypeLibrary") {
        check_datatype_library(element, value)?;
    }
    Ok(())
}

fn check_datatype_library(element: &Element, value: &str) -> Result<(), SchemaError> {
    if value.is_empty() {
        return Ok(());
    }
    let escaped = xlink_escape(value);
    let Some(info) = parse_uri_reference(&escaped) else {
        return Err(grammar(format!(
            "the datatypeLibrary attribute must be a URI reference, found `{value}`"
        ))
        .at(element.position));
    };
    if info.has_fragment {
        return Err(grammar(format!(
            "the datatypeLibrary attribute must not have a fragment identifier, found `{value}`"
        ))
        .at(element.position));
    }
    if !info.absolute {
        return Err(grammar(format!(
            "the datatypeLibrary attribute must not be relative, found `{value}`"
        ))
        .at(element.position));
    }
    Ok(())
}

fn allow_attributes(element: &Element, allowed: &[&str]) -> Result<(), SchemaError> {
    for attribute in &element.attributes {
        if !attribute.namespace.is_empty() {
            continue;
        }
        if matches!(attribute.local.as_str(), "ns" | "datatypeLibrary")
            || allowed.contains(&attribute.local.as_str())
        {
            continue;
        }
        return Err(grammar(format!(
            "the `{}` element must not have a `{}` attribute",
            element.name, attribute.local
        ))
        .at(element.position));
    }
    Ok(())
}

fn require_ncname_attribute(element: &Element, name: &str) -> Result<(), SchemaError> {
    let Some(value) = element.attribute(name) else {
        return Err(grammar(format!(
            "the `{}` element requires a `{name}` attribute",
            element.name
        ))
        .at(element.position));
    };
    if !is_ncname(value) {
        return Err(grammar(format!(
            "the {name} attribute of a `{}` element must be an NCName, found `{value}`",
            element.name
        ))
        .at(element.position));
    }
    Ok(())
}

fn require_uri_attribute(element: &Element, name: &str) -> Result<(), SchemaError> {
    let Some(value) = element.attribute(name) else {
        return Err(grammar(format!(
            "the `{}` element requires a `{name}` attribute",
            element.name
        ))
        .at(element.position));
    };
    if parse_uri_reference(&xlink_escape(value)).is_none() {
        return Err(grammar(format!(
            "the {name} attribute of a `{}` element must be a URI reference, found `{value}`",
            element.name
        ))
        .at(element.position));
    }
    Ok(())
}

fn require_children(
    element: &Element,
    count: usize,
    min: usize,
    max: Option<usize>,
) -> Result<(), SchemaError> {
    if count < min {
        return Err(grammar(format!(
            "the `{}` element requires at least {min} child element(s), found {count}",
            element.name
        ))
        .at(element.position));
    }
    if let Some(max) = max
        && count > max
    {
        return Err(grammar(format!(
            "the `{}` element allows at most {max} child element(s), found {count}",
            element.name
        ))
        .at(element.position));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xml_syntax::parse_str;

    fn accepted(schema: &str) -> bool {
        parse_str(schema, None)
            .and_then(|element| check_pattern(&element))
            .is_ok()
    }

    const NS: &str = r#"xmlns="http://relaxng.org/ns/structure/1.0""#;

    #[test]
    fn a_minimal_schema_is_accepted() {
        assert!(accepted(&format!(
            r#"<element {NS} name="foo"><empty/></element>"#
        )));
    }

    #[test]
    fn unknown_elements_are_rejected() {
        assert!(!accepted(&format!("<thisIsJunk {NS}/>")));
    }

    #[test]
    fn a_name_class_may_not_follow_a_name_attribute() {
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo"><name>foo</name><empty/></element>"#
        )));
    }

    #[test]
    fn a_name_class_is_not_a_pattern() {
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo"><group><name>bar</name></group></element>"#
        )));
    }

    #[test]
    fn two_excepts_are_rejected() {
        assert!(!accepted(&format!(
            r#"<element {NS}><anyName><except><name>foo</name></except>
                 <except><name>bar</name></except></anyName><empty/></element>"#
        )));
    }

    #[test]
    fn unknown_attributes_are_rejected() {
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo" extra="bar"><empty/></element>"#
        )));
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo"><empty name="bar"/></element>"#
        )));
    }

    #[test]
    fn empty_has_no_children() {
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo"><empty><empty/></empty></element>"#
        )));
    }

    #[test]
    fn datatype_library_must_be_absolute_without_fragment() {
        assert!(accepted(&format!(
            r#"<element {NS} name="foo" datatypeLibrary="http:ok"><empty/></element>"#
        )));
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo" datatypeLibrary="xyzzy"><empty/></element>"#
        )));
        assert!(!accepted(&format!(
            r#"<element {NS} name="foo" datatypeLibrary="http://www.example.com#"><empty/></element>"#
        )));
    }

    #[test]
    fn combine_values_are_restricted() {
        assert!(!accepted(&format!(
            r#"<grammar {NS}><start combine="both"><empty/></start></grammar>"#
        )));
    }
}
