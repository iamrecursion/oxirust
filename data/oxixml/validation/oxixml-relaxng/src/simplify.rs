//! Rules [4.3](https://relaxng.org/spec-20011203.html#IDATZR) through
//! [4.17](https://relaxng.org/spec-20011203.html#IDA1BAS) of the RELAX NG
//! simplification: datatype library inheritance, `href` resolution, `externalRef`
//! and `include` expansion, `name`/`ns` attribute normalization, QName
//! resolution, `div` flattening, child-count normalization, the `mixed`,
//! `optional` and `zeroOrMore` sugar, the §4.16 constraints, and `combine`.
//!
//! Rules 4.1 and 4.2 are performed while the schema is parsed (see
//! [`crate::xml_syntax`]); rules 4.18 to 4.21 live in [`crate::flatten`].
//!
//! Each rule is a separate pass over the whole tree, because §4 requires that
//! "the effect must be as if each rule was applied to all elements in the schema
//! before the next rule is applied".

use crate::datatype::Libraries;
use crate::error::{SchemaError, grammar, resource, simplify};
use crate::resolver::SchemaResolver;
use crate::tree::{Child, Element, INFOSET_XMLNS_NS};
use crate::uri::xlink_escape;

/// Runs rules 4.3 to 4.17 over `root`.
pub(crate) fn run(
    root: &mut Element,
    resolver: &dyn SchemaResolver,
    libraries: &Libraries,
) -> Result<(), SchemaError> {
    rule_datatype_library(root, "");
    rule_value_type(root);
    let mut loader = Loader {
        resolver,
        stack: Vec::new(),
    };
    loader.resolve_root(root)?;
    rule_name_attribute(root);
    rule_ns_attribute(root, "");
    rule_qnames(root)?;
    rule_div(root);
    rule_child_count(root);
    rule_mixed(root);
    rule_optional(root);
    rule_zero_or_more(root);
    rule_constraints(root, libraries, false)?;
    rule_combine(root)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 4.3 datatypeLibrary attribute
// ---------------------------------------------------------------------------

fn rule_datatype_library(element: &mut Element, inherited: &str) {
    let current = match element.remove_attribute("datatypeLibrary") {
        Some(value) => xlink_escape(&value),
        None => inherited.to_owned(),
    };
    for child in element.child_elements_mut() {
        rule_datatype_library(child, &current);
    }
    if matches!(element.name.as_str(), "data" | "value") {
        element.set_attribute("datatypeLibrary", current);
    }
}

// ---------------------------------------------------------------------------
// 4.4 type attribute of value element
// ---------------------------------------------------------------------------

fn rule_value_type(element: &mut Element) {
    if element.name == "value" && element.attribute("type").is_none() {
        element.set_attribute("type", "token");
        element.set_attribute("datatypeLibrary", "");
    }
    for child in element.child_elements_mut() {
        rule_value_type(child);
    }
}

// ---------------------------------------------------------------------------
// 4.5 href attribute, 4.6 externalRef element, 4.7 include element
// ---------------------------------------------------------------------------

/// Loads and inlines `externalRef` and `include` elements.
struct Loader<'a> {
    resolver: &'a dyn SchemaResolver,
    /// The absolute URIs currently being dereferenced, used to detect the loops
    /// that §4.6 and §4.7 forbid.
    stack: Vec<String>,
}

impl Loader<'_> {
    fn resolve_root(&mut self, root: &mut Element) -> Result<(), SchemaError> {
        match root.name.as_str() {
            "externalRef" => {
                *root = self.external_ref(root)?;
                Ok(())
            }
            "include" => Err(grammar("an include element is not a pattern").at(root.position)),
            _ => self.resolve_children(root),
        }
    }

    fn resolve_children(&mut self, element: &mut Element) -> Result<(), SchemaError> {
        let mut children = Vec::with_capacity(element.children.len());
        for child in std::mem::take(&mut element.children) {
            match child {
                Child::Text(text) => children.push(Child::Text(text)),
                Child::Element(mut child) => match child.name.as_str() {
                    "externalRef" => children.push(Child::Element(self.external_ref(&child)?)),
                    "include" => children.push(Child::Element(self.include(&mut child)?)),
                    _ => {
                        self.resolve_children(&mut child)?;
                        children.push(Child::Element(child));
                    }
                },
            }
        }
        element.children = children;
        Ok(())
    }

    /// Rule 4.5: escape the `href` value and resolve it against the base URI of
    /// the element that bears it.
    fn href(&self, element: &Element) -> Result<String, SchemaError> {
        let raw = element.attribute("href").ok_or_else(|| {
            grammar(format!(
                "the `{}` element requires an href attribute",
                element.name
            ))
            .at(element.position)
        })?;
        let escaped = xlink_escape(raw);
        match element.context.base.as_deref() {
            Some(base) => {
                let base = oxixml_iri::Iri::parse(base).map_err(|error| {
                    resource("4.5", format!("the base URI is not a valid URI: {error}"))
                        .at(element.position)
                })?;
                let reference = oxixml_iri::IriRef::parse(escaped.as_str()).map_err(|error| {
                    resource(
                        "4.5",
                        format!("`{escaped}` is not a URI reference: {error}"),
                    )
                    .at(element.position)
                })?;
                base.resolve(&reference)
                    .map(oxixml_iri::Iri::into_inner)
                    .map_err(|error| {
                        resource("4.5", format!("cannot resolve `{escaped}`: {error}"))
                            .at(element.position)
                    })
            }
            None => {
                if oxixml_iri::Iri::parse(escaped.as_str()).is_ok() {
                    Ok(escaped)
                } else {
                    Err(resource(
                        "4.5",
                        format!(
                            "`{escaped}` is a relative URI reference but the schema has no base URI"
                        ),
                    )
                    .at(element.position))
                }
            }
        }
    }

    /// Loads the document identified by `uri` and applies rules 4.1 to 4.7 to it.
    fn load(
        &mut self,
        uri: &str,
        position: crate::error::Position,
    ) -> Result<Element, SchemaError> {
        if self.stack.iter().any(|seen| seen == uri) {
            return Err(
                resource("4.6", format!("the schema references `{uri}` recursively")).at(position),
            );
        }
        if self.stack.len() > 64 {
            return Err(resource("4.6", "external schema references nest too deeply").at(position));
        }
        let text = self.resolver.resolve(uri).map_err(|error| {
            resource("4.5", format!("cannot retrieve `{uri}`: {error}")).at(position)
        })?;
        let mut element = if uri.ends_with(".rnc") {
            crate::compact::parse(&text, Some(uri))?
        } else {
            crate::xml_syntax::parse_str(&text, Some(uri))?
        };
        rule_datatype_library(&mut element, "");
        rule_value_type(&mut element);
        self.stack.push(uri.to_owned());
        let result = self.resolve_root(&mut element);
        self.stack.pop();
        result?;
        Ok(element)
    }

    /// Rule 4.6.
    fn external_ref(&mut self, element: &Element) -> Result<Element, SchemaError> {
        for attribute in &element.attributes {
            if attribute.namespace.is_empty() && !matches!(attribute.local.as_str(), "href" | "ns")
            {
                return Err(grammar(format!(
                    "the externalRef element must not have a `{}` attribute",
                    attribute.local
                ))
                .at(element.position));
            }
        }
        let uri = self.href(element)?;
        let mut referenced = self.load(&uri, element.position)?;
        crate::grammar_check::check_pattern(&referenced)?;
        if let Some(ns) = element.attribute("ns")
            && referenced.attribute("ns").is_none()
        {
            referenced.set_attribute("ns", ns);
        }
        Ok(referenced)
    }

    /// Rule 4.7.
    fn include(&mut self, element: &mut Element) -> Result<Element, SchemaError> {
        let uri = self.href(element)?;
        let mut grammar_element = self.load(&uri, element.position)?;
        crate::grammar_check::check_grammar(&grammar_element)?;
        self.resolve_children(element)?;

        if has_component(element, |component| component.name == "start") {
            if !has_component(&grammar_element, |component| component.name == "start") {
                return Err(simplify(
                    "4.7",
                    format!("the included grammar `{uri}` has no start component to override"),
                )
                .at(element.position));
            }
            remove_components(&mut grammar_element, |component| component.name == "start");
        }
        let overridden: Vec<String> = components(element)
            .filter(|component| component.name == "define")
            .filter_map(|component| component.attribute("name").map(str::to_owned))
            .collect();
        for name in overridden {
            if !has_component(&grammar_element, |component| {
                component.name == "define" && component.attribute("name") == Some(name.as_str())
            }) {
                return Err(simplify(
                    "4.7",
                    format!(
                        "the included grammar `{uri}` has no define component named `{name}` to override"
                    ),
                )
                .at(element.position));
            }
            remove_components(&mut grammar_element, |component| {
                component.name == "define" && component.attribute("name") == Some(name.as_str())
            });
        }

        grammar_element.name = "div".to_owned();
        let mut div = Element::like(element, "div");
        div.attributes = element
            .attributes
            .iter()
            .filter(|attribute| !(attribute.namespace.is_empty() && attribute.local == "href"))
            .cloned()
            .collect();
        div.children.push(Child::Element(grammar_element));
        div.children.append(&mut element.children);
        Ok(div)
    }
}

/// The *components* of an element: its child elements together with the
/// components of any `div` child elements (§4.7).
fn components(element: &Element) -> Box<dyn Iterator<Item = &Element> + '_> {
    Box::new(element.child_elements().flat_map(|child| {
        if child.name == "div" {
            components(child)
        } else {
            Box::new(std::iter::once(child)) as Box<dyn Iterator<Item = &Element>>
        }
    }))
}

fn has_component(element: &Element, predicate: impl Fn(&Element) -> bool + Copy) -> bool {
    components(element).any(predicate)
}

fn remove_components(element: &mut Element, predicate: impl Fn(&Element) -> bool + Copy) {
    element.children.retain(|child| match child {
        Child::Element(child) => child.name == "div" || !predicate(child),
        Child::Text(_) => true,
    });
    for child in element.child_elements_mut() {
        if child.name == "div" {
            remove_components(child, predicate);
        }
    }
}

// ---------------------------------------------------------------------------
// 4.8 name attribute of element and attribute elements
// ---------------------------------------------------------------------------

fn rule_name_attribute(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_name_attribute(child);
    }
    if !matches!(element.name.as_str(), "element" | "attribute") {
        return;
    }
    let Some(name) = element.remove_attribute("name") else {
        return;
    };
    let mut name_element = Element::like(element, "name");
    name_element.children.push(Child::Text(name));
    if element.name == "attribute" && element.attribute("ns").is_none() {
        name_element.set_attribute("ns", "");
    }
    element.children.insert(0, Child::Element(name_element));
}

// ---------------------------------------------------------------------------
// 4.9 ns attribute
// ---------------------------------------------------------------------------

fn rule_ns_attribute(element: &mut Element, inherited: &str) {
    let current = element
        .remove_attribute("ns")
        .unwrap_or_else(|| inherited.to_owned());
    for child in element.child_elements_mut() {
        rule_ns_attribute(child, &current);
    }
    if matches!(element.name.as_str(), "name" | "nsName" | "value") {
        element.set_attribute("ns", current);
    }
}

// ---------------------------------------------------------------------------
// 4.10 QNames
// ---------------------------------------------------------------------------

fn rule_qnames(element: &mut Element) -> Result<(), SchemaError> {
    if element.name == "name" {
        let text = element.text();
        if let Some((prefix, local)) = text.split_once(':') {
            let Some(namespace) = element.context.namespaces.lookup(prefix) else {
                return Err(simplify(
                    "4.10",
                    format!("the namespace prefix `{prefix}` is not declared"),
                )
                .at(element.position));
            };
            let namespace = namespace.to_owned();
            element.set_attribute("ns", namespace);
            element.children.clear();
            element.children.push(Child::Text(local.to_owned()));
        }
    }
    for child in element.child_elements_mut() {
        rule_qnames(child)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 4.11 div element
// ---------------------------------------------------------------------------

fn rule_div(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_div(child);
    }
    let mut children = Vec::with_capacity(element.children.len());
    for child in std::mem::take(&mut element.children) {
        match child {
            Child::Element(child) if child.name == "div" => {
                children.extend(child.children);
            }
            other => children.push(other),
        }
    }
    element.children = children;
}

// ---------------------------------------------------------------------------
// 4.12 Number of child elements
// ---------------------------------------------------------------------------

/// Builds a left-nested binary tree of `name` elements over `parts`.
///
/// `<choice> p1 p2 p3 </choice>` becomes
/// `<choice> <choice> p1 p2 </choice> p3 </choice>`, which is what §4.12
/// prescribes. A single part is returned unchanged, and an empty list yields an
/// empty element of that name (which the §3 check already rules out).
fn fold_binary(template: &Element, name: &str, parts: Vec<Element>) -> Element {
    let mut parts = parts.into_iter();
    let Some(mut accumulator) = parts.next() else {
        return Element::like(template, name);
    };
    for part in parts {
        let mut combined = Element::like(template, name);
        combined.children.push(Child::Element(accumulator));
        combined.children.push(Child::Element(part));
        accumulator = combined;
    }
    accumulator
}

fn rule_child_count(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_child_count(child);
    }
    match element.name.as_str() {
        "define" | "oneOrMore" | "zeroOrMore" | "optional" | "list" | "mixed" => {
            let children = element.take_child_elements();
            if children.len() > 1 {
                let group = fold_binary(element, "group", children);
                element.set_child_elements(vec![group]);
            } else {
                element.set_child_elements(children);
            }
        }
        "element" | "attribute" => {
            let mut children = element.take_child_elements();
            if children.len() > 2 {
                let rest = children.split_off(1);
                children.push(fold_binary(element, "group", rest));
            } else if children.len() == 1 && element.name == "attribute" {
                children.push(Element::like(element, "text"));
            }
            element.set_child_elements(children);
        }
        "except" => {
            let children = element.take_child_elements();
            if children.len() > 1 {
                let choice = fold_binary(element, "choice", children);
                element.set_child_elements(vec![choice]);
            } else {
                element.set_child_elements(children);
            }
        }
        "choice" | "group" | "interleave" => {
            let mut children = element.take_child_elements();
            match children.len() {
                0 => {}
                1 => {
                    if let Some(only) = children.pop() {
                        *element = only;
                    }
                }
                _ => {
                    let name = element.name.clone();
                    let folded = fold_binary(element, &name, children);
                    element.children = folded.children;
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// 4.13 mixed, 4.14 optional, 4.15 zeroOrMore
// ---------------------------------------------------------------------------

fn rule_mixed(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_mixed(child);
    }
    if element.name == "mixed" {
        element.name = "interleave".to_owned();
        let text = Element::like(element, "text");
        element.children.push(Child::Element(text));
    }
}

fn rule_optional(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_optional(child);
    }
    if element.name == "optional" {
        element.name = "choice".to_owned();
        let empty = Element::like(element, "empty");
        element.children.push(Child::Element(empty));
    }
}

fn rule_zero_or_more(element: &mut Element) {
    for child in element.child_elements_mut() {
        rule_zero_or_more(child);
    }
    if element.name == "zeroOrMore" {
        let mut one_or_more = Element::like(element, "oneOrMore");
        one_or_more.children = std::mem::take(&mut element.children);
        element.name = "choice".to_owned();
        let empty = Element::like(element, "empty");
        element.children.push(Child::Element(one_or_more));
        element.children.push(Child::Element(empty));
    }
}

// ---------------------------------------------------------------------------
// 4.16 Constraints
// ---------------------------------------------------------------------------

/// Checks the §4.16 constraints. `in_attribute_name_class` tracks whether the
/// element is the first child of an `attribute` element or a descendant of it.
fn rule_constraints(
    element: &Element,
    libraries: &Libraries,
    in_attribute_name_class: bool,
) -> Result<(), SchemaError> {
    match element.name.as_str() {
        "anyName" => {
            if let Some(except) = element.child_elements().next()
                && except.has_descendant("anyName")
            {
                return Err(simplify(
                    "4.16",
                    "an except element inside anyName must not contain anyName",
                )
                .at(element.position));
            }
        }
        "nsName" => {
            if let Some(except) = element.child_elements().next()
                && (except.has_descendant("nsName") || except.has_descendant("anyName"))
            {
                return Err(simplify(
                    "4.16",
                    "an except element inside nsName must not contain nsName or anyName",
                )
                .at(element.position));
            }
        }
        _ => {}
    }

    if in_attribute_name_class && matches!(element.name.as_str(), "name" | "nsName") {
        let ns = element.attribute("ns").unwrap_or_default();
        if ns == INFOSET_XMLNS_NS {
            return Err(simplify(
                "4.16",
                format!("an attribute name class must not use the namespace `{INFOSET_XMLNS_NS}`"),
            )
            .at(element.position));
        }
        if element.name == "name" && ns.is_empty() && element.text() == "xmlns" {
            return Err(simplify(
                "4.16",
                "an attribute must not be named `xmlns` in no namespace",
            )
            .at(element.position));
        }
    }

    if matches!(element.name.as_str(), "data" | "value") {
        libraries.check_usage(element)?;
    }

    let attribute_parent = element.name == "attribute";
    for (index, child) in element.child_elements().enumerate() {
        // "the first child of an attribute element, or a descendant of it".
        let nested = if attribute_parent {
            index == 0
        } else {
            in_attribute_name_class
        };
        rule_constraints(child, libraries, nested)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 4.17 combine attribute
// ---------------------------------------------------------------------------

fn rule_combine(element: &mut Element) -> Result<(), SchemaError> {
    for child in element.child_elements_mut() {
        rule_combine(child)?;
    }
    if element.name != "grammar" {
        return Ok(());
    }

    let mut children = std::mem::take(&mut element.children);
    combine_group(element, &mut children, "start", |_| true)?;
    let names: Vec<String> = children
        .iter()
        .filter_map(|child| match child {
            Child::Element(child) if child.name == "define" => {
                child.attribute("name").map(str::to_owned)
            }
            _ => None,
        })
        .collect();
    let mut seen: Vec<String> = Vec::new();
    for name in names {
        if seen.contains(&name) {
            continue;
        }
        seen.push(name.clone());
        combine_group(element, &mut children, "define", |child| {
            child.attribute("name") == Some(name.as_str())
        })?;
    }
    element.children = children;
    Ok(())
}

/// Combines every child of `children` that has local name `name` and satisfies
/// `selector` into a single element, per §4.17.
fn combine_group(
    grammar_element: &Element,
    children: &mut Vec<Child>,
    name: &str,
    selector: impl Fn(&Element) -> bool,
) -> Result<(), SchemaError> {
    let indices: Vec<usize> = children
        .iter()
        .enumerate()
        .filter_map(|(index, child)| match child {
            Child::Element(child) if child.name == name && selector(child) => Some(index),
            _ => None,
        })
        .collect();
    if indices.len() < 2 {
        if let Some(&index) = indices.first()
            && let Child::Element(child) = &mut children[index]
        {
            child.remove_attribute("combine");
        }
        return Ok(());
    }

    let mut method: Option<String> = None;
    let mut without_combine = 0_usize;
    let label = match name {
        "start" => "start".to_owned(),
        _ => {
            let named = children
                .get(indices[0])
                .and_then(|child| match child {
                    Child::Element(child) => child.attribute("name"),
                    Child::Text(_) => None,
                })
                .unwrap_or_default();
            format!("define `{named}`")
        }
    };
    for &index in &indices {
        let Child::Element(child) = &children[index] else {
            continue;
        };
        match child.attribute("combine") {
            None => without_combine += 1,
            Some(value) => match &method {
                None => method = Some(value.to_owned()),
                Some(existing) if existing == value => {}
                Some(existing) => {
                    return Err(simplify(
                        "4.17",
                        format!("{label} is combined both with `{existing}` and with `{value}`"),
                    )
                    .at(child.position));
                }
            },
        }
    }
    if without_combine > 1 {
        return Err(simplify(
            "4.17",
            format!("{label} is defined more than once without a combine attribute"),
        )
        .at(grammar_element.position));
    }
    let Some(method) = method else {
        return Err(simplify(
            "4.17",
            format!("{label} is defined more than once without a combine attribute"),
        )
        .at(grammar_element.position));
    };

    // Collect the bodies in document order, then fold them with the method.
    let mut bodies = Vec::new();
    let mut header: Option<Element> = None;
    for &index in indices.iter().rev() {
        let Child::Element(mut child) = children.remove(index) else {
            continue;
        };
        child.remove_attribute("combine");
        let body = child.take_child_elements();
        bodies.push(fold_binary(&child, &method, body));
        header = Some(child);
    }
    bodies.reverse();
    let Some(mut header) = header else {
        return Ok(());
    };
    let folded = fold_binary(&header, &method, bodies);
    header.set_child_elements(vec![folded]);
    children.insert(indices[0], Child::Element(header));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::Libraries;
    use crate::resolver::NoResolver;
    use crate::xml_syntax::parse_str;

    const NS: &str = r#"xmlns="http://relaxng.org/ns/structure/1.0""#;

    fn simplified(schema: &str) -> Result<Element, SchemaError> {
        let mut root = parse_str(schema, None)?;
        crate::grammar_check::check_pattern(&root)?;
        run(&mut root, &NoResolver, &Libraries)?;
        Ok(root)
    }

    #[test]
    fn datatype_library_is_inherited_by_data_and_value_only() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo" datatypeLibrary="http://www.w3.org/2001/XMLSchema-datatypes">
                 <data type="string"/>
               </element>"#
        ))?;
        assert_eq!(root.attribute("datatypeLibrary"), None);
        let data = root
            .child_elements()
            .find(|child| child.name == "data")
            .expect("the data element survives");
        assert_eq!(
            data.attribute("datatypeLibrary"),
            Some("http://www.w3.org/2001/XMLSchema-datatypes")
        );
        Ok(())
    }

    #[test]
    fn value_without_type_becomes_a_builtin_token() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo" datatypeLibrary="http://example.com/nope">
                 <value>bar</value>
               </element>"#
        ))?;
        let value = root
            .child_elements()
            .find(|child| child.name == "value")
            .expect("the value element survives");
        assert_eq!(value.attribute("type"), Some("token"));
        assert_eq!(value.attribute("datatypeLibrary"), Some(""));
        Ok(())
    }

    #[test]
    fn name_attribute_becomes_a_name_element() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo"><attribute name="bar"/></element>"#
        ))?;
        let name = root
            .child_elements()
            .next()
            .expect("the name class comes first");
        assert_eq!(name.name, "name");
        assert_eq!(name.text(), "foo");
        assert_eq!(name.attribute("ns"), Some(""));
        Ok(())
    }

    #[test]
    fn attribute_without_a_pattern_gains_text() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo"><attribute name="bar"/></element>"#
        ))?;
        let attribute = root
            .child_elements()
            .find(|child| child.name == "attribute")
            .expect("the attribute survives");
        let children: Vec<&str> = attribute
            .child_elements()
            .map(|child| child.name.as_str())
            .collect();
        assert_eq!(children, ["name", "text"]);
        Ok(())
    }

    #[test]
    fn choice_is_made_binary() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo"><choice><empty/><text/><empty/></choice></element>"#
        ))?;
        let choice = root
            .child_elements()
            .find(|child| child.name == "choice")
            .expect("the choice survives");
        assert_eq!(choice.child_elements().count(), 2);
        Ok(())
    }

    #[test]
    fn zero_or_more_expands() -> Result<(), SchemaError> {
        let root = simplified(&format!(
            r#"<element {NS} name="foo"><zeroOrMore><text/></zeroOrMore></element>"#
        ))?;
        let choice = root
            .child_elements()
            .find(|child| child.name == "choice")
            .expect("zeroOrMore became a choice");
        let names: Vec<&str> = choice
            .child_elements()
            .map(|child| child.name.as_str())
            .collect();
        assert_eq!(names, ["oneOrMore", "empty"]);
        Ok(())
    }

    #[test]
    fn duplicate_definitions_without_combine_are_rejected() {
        assert!(
            simplified(&format!(
                r#"<grammar {NS}><start><ref name="a"/></start>
                     <define name="a"><element name="a"><empty/></element></define>
                     <define name="a"><element name="b"><empty/></element></define>
                   </grammar>"#
            ))
            .is_err()
        );
    }

    #[test]
    fn conflicting_combine_methods_are_rejected() {
        assert!(
            simplified(&format!(
                r#"<grammar {NS}><start><ref name="a"/></start>
                     <define name="a" combine="choice"><element name="a"><empty/></element></define>
                     <define name="a" combine="interleave"><element name="b"><empty/></element></define>
                   </grammar>"#
            ))
            .is_err()
        );
    }

    #[test]
    fn an_undeclared_prefix_is_rejected() {
        assert!(
            simplified(&format!(
                r#"<element {NS}><name>p:foo</name><empty/></element>"#
            ))
            .is_err()
        );
    }
}
