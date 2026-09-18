//! The restrictions of [§7](https://relaxng.org/spec-20011203.html#restrictions),
//! checked on the simple form.
//!
//! * §7.1 — the *prohibited paths*, expressed here as a set of inherited context
//!   flags rather than as literal path matching, which is equivalent and needs a
//!   single traversal.
//! * §7.2 — *string sequences*: every element's content pattern must have a
//!   content-type, which rules out a `data`/`value`/`list` pattern sitting next
//!   to anything else that can match a child.
//! * §7.3 — no duplicate attributes across the two halves of a `group` or
//!   `interleave`, and an attribute with an infinite name class must be repeated.
//! * §7.4 — the two halves of an `interleave` must not compete for the same
//!   element name, and `text` may occur in only one of them.

use crate::error::{SchemaError, restriction};
use crate::flatten::SimpleGrammar;
use crate::pattern::{NameClassId, NameClasses};
use crate::tree::Element;

/// Runs every §7 check over the simplified grammar.
pub(crate) fn check(grammar: &SimpleGrammar) -> Result<(), SchemaError> {
    let context = Context {
        in_start: true,
        ..Context::default()
    };
    prohibited_paths(&grammar.start, context, grammar)?;
    for (_, body) in &grammar.defines {
        prohibited_paths(body, Context::default(), grammar)?;
    }

    for (_, body) in &grammar.defines {
        let content = body
            .child_elements()
            .nth(1)
            .ok_or_else(|| crate::flatten::malformed(body, "a content pattern"))?;
        if content_type(content).is_none() {
            return Err(restriction(
                "7.2",
                "the content of this element mixes a pattern that matches a string with one that \
                 matches a child; they must be alternatives to each other",
            )
            .at(content.position));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 7.1 Contextual restrictions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Context {
    in_attribute: bool,
    in_list: bool,
    in_data_except: bool,
    in_start: bool,
    in_one_or_more: bool,
    in_one_or_more_group: bool,
}

fn prohibited_paths(
    element: &Element,
    context: Context,
    grammar: &SimpleGrammar,
) -> Result<(), SchemaError> {
    let name = element.name.as_str();
    let forbid = |section: &'static str, ancestor: &str| {
        Err(restriction(
            section,
            format!("a `{name}` pattern must not appear inside a `{ancestor}` pattern"),
        )
        .at(element.position))
    };

    match name {
        "ref" => {
            if context.in_attribute {
                return forbid("7.1.1", "attribute");
            }
            if context.in_list {
                return forbid("7.1.3", "list");
            }
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
        }
        "attribute" => {
            if context.in_attribute {
                return forbid("7.1.1", "attribute");
            }
            if context.in_one_or_more_group {
                return forbid("7.1.2", "oneOrMore/group or oneOrMore/interleave");
            }
            if context.in_list {
                return forbid("7.1.3", "list");
            }
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
            if context.in_start {
                return forbid("7.1.5", "start");
            }
        }
        "list" => {
            if context.in_list {
                return forbid("7.1.3", "list");
            }
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
            if context.in_start {
                return forbid("7.1.5", "start");
            }
        }
        "text" => {
            if context.in_list {
                return forbid("7.1.3", "list");
            }
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
            if context.in_start {
                return forbid("7.1.5", "start");
            }
        }
        "interleave" => {
            if context.in_list {
                return forbid("7.1.3", "list");
            }
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
            if context.in_start {
                return forbid("7.1.5", "start");
            }
        }
        "group" | "oneOrMore" | "empty" => {
            if context.in_data_except {
                return forbid("7.1.4", "data/except");
            }
            if context.in_start {
                return forbid("7.1.5", "start");
            }
        }
        "data" | "value" if context.in_start => {
            return forbid("7.1.5", "start");
        }
        _ => {}
    }

    if name == "group" || name == "interleave" {
        duplicate_attributes(element, grammar)?;
    }
    if name == "interleave" {
        interleave_restrictions(element, grammar)?;
    }
    if name == "attribute" {
        infinite_attribute_name(element, context)?;
    }

    let mut nested = context;
    match name {
        "attribute" => nested.in_attribute = true,
        "list" => nested.in_list = true,
        "oneOrMore" => nested.in_one_or_more = true,
        "group" | "interleave" if context.in_one_or_more => nested.in_one_or_more_group = true,
        _ => {}
    }
    if name == "element" {
        // `element` only ever appears as the body of a definition, so its content
        // starts a fresh context.
        nested = Context::default();
    }

    for child in element.child_elements() {
        let mut child_context = nested;
        if name == "data" && child.name == "except" {
            child_context.in_data_except = true;
        }
        prohibited_paths(child, child_context, grammar)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 7.2 String sequences
// ---------------------------------------------------------------------------

/// The three content-types of §7.2, ordered `Empty < Complex < Simple`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ContentType {
    Empty,
    Complex,
    Simple,
}

const fn groupable(left: ContentType, right: ContentType) -> bool {
    matches!(
        (left, right),
        (ContentType::Empty, _)
            | (_, ContentType::Empty)
            | (ContentType::Complex, ContentType::Complex)
    )
}

/// Computes the content-type of a pattern, or `None` when it has none.
fn content_type(element: &Element) -> Option<ContentType> {
    match element.name.as_str() {
        "value" | "data" | "list" => Some(ContentType::Simple),
        "text" | "ref" => Some(ContentType::Complex),
        // §4.20 leaves `notAllowed` only under `start` and `element`; it matches
        // nothing at all, so it composes like `empty`.
        "empty" | "attribute" | "notAllowed" => Some(ContentType::Empty),
        "group" | "interleave" => {
            let (left, right) = two(element)?;
            let left = content_type(left)?;
            let right = content_type(right)?;
            groupable(left, right).then(|| left.max(right))
        }
        "oneOrMore" => {
            let inner = content_type(element.child_elements().next()?)?;
            groupable(inner, inner).then_some(inner)
        }
        "choice" => {
            let (left, right) = two(element)?;
            Some(content_type(left)?.max(content_type(right)?))
        }
        _ => None,
    }
}

fn two(element: &Element) -> Option<(&Element, &Element)> {
    let mut children = element.child_elements();
    let left = children.next()?;
    let right = children.next()?;
    Some((left, right))
}

// ---------------------------------------------------------------------------
// 7.3 Restrictions on attributes
// ---------------------------------------------------------------------------

/// The `occurs in` relation of §7.3.
fn occurring<'a>(element: &'a Element, name: &str, out: &mut Vec<&'a Element>) {
    if element.name == name {
        out.push(element);
        return;
    }
    if matches!(
        element.name.as_str(),
        "choice" | "interleave" | "group" | "oneOrMore"
    ) {
        for child in element.child_elements() {
            occurring(child, name, out);
        }
    }
}

fn duplicate_attributes(element: &Element, _grammar: &SimpleGrammar) -> Result<(), SchemaError> {
    let Some((left, right)) = two(element) else {
        return Ok(());
    };
    let mut first = Vec::new();
    let mut second = Vec::new();
    occurring(left, "attribute", &mut first);
    occurring(right, "attribute", &mut second);
    if first.is_empty() || second.is_empty() {
        return Ok(());
    }
    let mut classes = NameClasses::default();
    let first: Vec<NameClassId> = first
        .iter()
        .filter_map(|attribute| name_class_of(attribute, &mut classes))
        .collect();
    let second: Vec<NameClassId> = second
        .iter()
        .filter_map(|attribute| name_class_of(attribute, &mut classes))
        .collect();
    for left in &first {
        for right in &second {
            if classes.overlap(*left, *right) {
                return Err(restriction(
                    "7.3",
                    format!(
                        "the attribute name class `{}` can match the same name as `{}` in the \
                         other half of this {}",
                        classes.describe(*left),
                        classes.describe(*right),
                        element.name
                    ),
                )
                .at(element.position));
            }
        }
    }
    Ok(())
}

fn infinite_attribute_name(element: &Element, context: Context) -> Result<(), SchemaError> {
    if context.in_one_or_more {
        return Ok(());
    }
    let Some(name_class) = element.child_elements().next() else {
        return Ok(());
    };
    if name_class.has_descendant_or_self("anyName") || name_class.has_descendant_or_self("nsName") {
        return Err(restriction(
            "7.3",
            "an attribute whose name class is infinite (it uses anyName or nsName) must have a \
             oneOrMore ancestor",
        )
        .at(element.position));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 7.4 Restrictions on interleave
// ---------------------------------------------------------------------------

fn interleave_restrictions(element: &Element, grammar: &SimpleGrammar) -> Result<(), SchemaError> {
    let Some((left, right)) = two(element) else {
        return Ok(());
    };

    let mut left_text = Vec::new();
    let mut right_text = Vec::new();
    occurring(left, "text", &mut left_text);
    occurring(right, "text", &mut right_text);
    if !left_text.is_empty() && !right_text.is_empty() {
        return Err(restriction(
            "7.4",
            "a text pattern must not occur in both halves of an interleave",
        )
        .at(element.position));
    }

    let mut left_refs = Vec::new();
    let mut right_refs = Vec::new();
    occurring(left, "ref", &mut left_refs);
    occurring(right, "ref", &mut right_refs);
    if left_refs.is_empty() || right_refs.is_empty() {
        return Ok(());
    }
    let mut classes = NameClasses::default();
    let left_names = referenced_names(&left_refs, grammar, &mut classes);
    let right_names = referenced_names(&right_refs, grammar, &mut classes);
    for left in &left_names {
        for right in &right_names {
            if classes.overlap(*left, *right) {
                return Err(restriction(
                    "7.4",
                    format!(
                        "the element name class `{}` can match the same name as `{}` in the other \
                         half of this interleave",
                        classes.describe(*left),
                        classes.describe(*right)
                    ),
                )
                .at(element.position));
            }
        }
    }
    Ok(())
}

fn referenced_names(
    refs: &[&Element],
    grammar: &SimpleGrammar,
    classes: &mut NameClasses,
) -> Vec<NameClassId> {
    refs.iter()
        .filter_map(|reference| {
            let name = reference.attribute("name")?;
            let body = grammar.lookup(name)?;
            name_class_of(body, classes)
        })
        .collect()
}

/// Builds the name class of an `element` or `attribute` pattern in a scratch
/// table, so that overlap can be decided without the compiled schema.
fn name_class_of(element: &Element, classes: &mut NameClasses) -> Option<NameClassId> {
    build_name_class(element.child_elements().next()?, classes)
}

fn build_name_class(element: &Element, classes: &mut NameClasses) -> Option<NameClassId> {
    use crate::pattern::NameClass;

    let class = match element.name.as_str() {
        "name" => NameClass::Name(
            element.attribute("ns").unwrap_or_default().to_owned(),
            element.text(),
        ),
        "anyName" => match element.child_elements().next() {
            Some(except) => {
                let inner = build_name_class(except.child_elements().next()?, classes)?;
                NameClass::AnyNameExcept(inner)
            }
            None => NameClass::AnyName,
        },
        "nsName" => {
            let namespace = element.attribute("ns").unwrap_or_default().to_owned();
            match element.child_elements().next() {
                Some(except) => {
                    let inner = build_name_class(except.child_elements().next()?, classes)?;
                    NameClass::NsNameExcept(namespace, inner)
                }
                None => NameClass::NsName(namespace),
            }
        }
        "choice" => {
            let mut children = element.child_elements();
            let left = build_name_class(children.next()?, classes)?;
            let right = build_name_class(children.next()?, classes)?;
            NameClass::Choice(left, right)
        }
        _ => return None,
    };
    Some(classes.intern(class))
}
