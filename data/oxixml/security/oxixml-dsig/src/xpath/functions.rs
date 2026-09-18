//! The XPath 1.0 core function library, plus XML Signature's `here()`.
//!
//! Every function of
//! [XPath §4](https://www.w3.org/TR/1999/REC-xpath-19991116/#corelib) is
//! implemented. On top of them,
//! [XML Signature §6.6.3](https://www.w3.org/TR/xmldsig-core1/#sec-XPath) adds
//! one:
//!
//! > **Function:** node-set `here()`
//! >
//! > The `here` function returns a node-set containing the attribute or
//! > processing instruction node or the parent element of the text node that
//! > directly bears the XPath expression.
//!
//! That is what makes a self-referential transform possible:
//! `here()/ancestor::ds:Signature[1]` is "the signature I am part of", without
//! the signature needing an `Id`.

use oxixml_dom::{Document, NodeKind};

use crate::xpath::ast::{Expr, Name};
use crate::xpath::eval::{Context, Focus, evaluate};
use crate::xpath::value::{NodeSet, Value, XNode};

/// Dispatches a function call.
///
/// # Errors
///
/// Returns a message for an unknown function, a prefixed name (XPath 1.0 has no
/// namespaced core functions and this crate binds no extensions), a wrong
/// argument count or an argument of the wrong type.
pub(crate) fn call(
    name: &Name,
    arguments: &[Expr],
    context: &Context<'_>,
    focus: &Focus,
) -> Result<Value, String> {
    if name.prefix.is_some() {
        return Err(format!(
            "no extension function is bound for `{}`",
            name.to_qualified_string()
        ));
    }
    let document = context.document;
    let values = arguments
        .iter()
        .map(|argument| evaluate(argument, context, focus))
        .collect::<Result<Vec<_>, _>>()?;

    match name.local.as_str() {
        // --- node-set functions ---------------------------------------------------------------
        "last" => {
            expect_arity(name, &values, 0..=0)?;
            Ok(Value::Number(count_as_f64(focus.size)))
        }
        "position" => {
            expect_arity(name, &values, 0..=0)?;
            Ok(Value::Number(count_as_f64(focus.position)))
        }
        "count" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::Number(count_as_f64(
                node_set_argument(name, &values, 0)?.len(),
            )))
        }
        "id" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::NodeSet(id_function(context, values.first())))
        }
        "local-name" => {
            expect_arity(name, &values, 0..=1)?;
            Ok(Value::Str(
                name_parts(context, focus, name, &values)?
                    .map_or_else(String::new, |(_, local)| local),
            ))
        }
        "namespace-uri" => {
            expect_arity(name, &values, 0..=1)?;
            Ok(Value::Str(
                name_parts(context, focus, name, &values)?
                    .and_then(|(namespace, _)| namespace)
                    .unwrap_or_default(),
            ))
        }
        "name" => {
            expect_arity(name, &values, 0..=1)?;
            Ok(Value::Str(qualified_name(context, focus, name, &values)?))
        }

        // --- string functions -----------------------------------------------------------------
        "string" => {
            expect_arity(name, &values, 0..=1)?;
            Ok(Value::Str(match values.first() {
                Some(value) => value.to_xpath_string(document),
                None => focus.node.string_value(document),
            }))
        }
        "concat" => {
            if values.len() < 2 {
                return Err("concat() needs at least two arguments".to_owned());
            }
            Ok(Value::Str(
                values
                    .iter()
                    .map(|value| value.to_xpath_string(document))
                    .collect(),
            ))
        }
        "starts-with" => {
            expect_arity(name, &values, 2..=2)?;
            let (haystack, needle) = two_strings(document, &values);
            Ok(Value::Boolean(haystack.starts_with(&needle)))
        }
        "contains" => {
            expect_arity(name, &values, 2..=2)?;
            let (haystack, needle) = two_strings(document, &values);
            Ok(Value::Boolean(haystack.contains(&needle)))
        }
        "substring-before" => {
            expect_arity(name, &values, 2..=2)?;
            let (haystack, needle) = two_strings(document, &values);
            Ok(Value::Str(
                haystack
                    .find(&needle)
                    .and_then(|at| haystack.get(..at))
                    .unwrap_or_default()
                    .to_owned(),
            ))
        }
        "substring-after" => {
            expect_arity(name, &values, 2..=2)?;
            let (haystack, needle) = two_strings(document, &values);
            Ok(Value::Str(
                haystack
                    .find(&needle)
                    .and_then(|at| haystack.get(at + needle.len()..))
                    .unwrap_or_default()
                    .to_owned(),
            ))
        }
        "substring" => {
            expect_arity(name, &values, 2..=3)?;
            Ok(Value::Str(substring(document, &values)))
        }
        "string-length" => {
            expect_arity(name, &values, 0..=1)?;
            let text = match values.first() {
                Some(value) => value.to_xpath_string(document),
                None => focus.node.string_value(document),
            };
            Ok(Value::Number(count_as_f64(text.chars().count())))
        }
        "normalize-space" => {
            expect_arity(name, &values, 0..=1)?;
            let text = match values.first() {
                Some(value) => value.to_xpath_string(document),
                None => focus.node.string_value(document),
            };
            Ok(Value::Str(
                text.split_whitespace().collect::<Vec<_>>().join(" "),
            ))
        }
        "translate" => {
            expect_arity(name, &values, 3..=3)?;
            Ok(Value::Str(translate(document, &values)))
        }

        // --- boolean functions ----------------------------------------------------------------
        "boolean" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::Boolean(
                values.first().is_some_and(Value::to_boolean),
            ))
        }
        "not" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::Boolean(
                !values.first().is_some_and(Value::to_boolean),
            ))
        }
        "true" => {
            expect_arity(name, &values, 0..=0)?;
            Ok(Value::Boolean(true))
        }
        "false" => {
            expect_arity(name, &values, 0..=0)?;
            Ok(Value::Boolean(false))
        }
        "lang" => {
            expect_arity(name, &values, 1..=1)?;
            let wanted = values
                .first()
                .map(|value| value.to_xpath_string(document))
                .unwrap_or_default();
            Ok(Value::Boolean(language_matches(document, focus, &wanted)))
        }

        // --- number functions -----------------------------------------------------------------
        "number" => {
            expect_arity(name, &values, 0..=1)?;
            Ok(Value::Number(match values.first() {
                Some(value) => value.to_number(document),
                None => Value::Str(focus.node.string_value(document)).to_number(document),
            }))
        }
        "sum" => {
            expect_arity(name, &values, 1..=1)?;
            let set = node_set_argument(name, &values, 0)?;
            Ok(Value::Number(
                set.nodes().iter().fold(0.0, |total, node| {
                    total + Value::Str(node.string_value(document)).to_number(document)
                }),
            ))
        }
        "floor" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::Number(number_argument(document, &values).floor()))
        }
        "ceiling" => {
            expect_arity(name, &values, 1..=1)?;
            Ok(Value::Number(number_argument(document, &values).ceil()))
        }
        "round" => {
            expect_arity(name, &values, 1..=1)?;
            // XPath rounds halves towards positive infinity, which `f64::round`
            // does not: it rounds halves away from zero.
            let value = number_argument(document, &values);
            Ok(Value::Number(if value.is_finite() {
                (value + 0.5).floor()
            } else {
                value
            }))
        }

        // --- the XML Signature extension --------------------------------------------------------
        "here" => {
            expect_arity(name, &values, 0..=0)?;
            match context.here {
                Some(node) => Ok(Value::NodeSet(NodeSet::singleton(XNode::Tree(node)))),
                None => Err(
                    "here() is only defined for an expression that appears in the document \
                     being evaluated"
                        .to_owned(),
                ),
            }
        }

        other => Err(format!("`{other}()` is not an XPath function")),
    }
}

fn expect_arity(
    name: &Name,
    values: &[Value],
    allowed: std::ops::RangeInclusive<usize>,
) -> Result<(), String> {
    if allowed.contains(&values.len()) {
        Ok(())
    } else {
        Err(format!(
            "{}() takes {} argument(s), not {}",
            name.local,
            if allowed.start() == allowed.end() {
                allowed.start().to_string()
            } else {
                format!("{} to {}", allowed.start(), allowed.end())
            },
            values.len()
        ))
    }
}

fn node_set_argument<'a>(
    name: &Name,
    values: &'a [Value],
    index: usize,
) -> Result<&'a NodeSet, String> {
    match values.get(index) {
        Some(Value::NodeSet(set)) => Ok(set),
        _ => Err(format!("{}() needs a node-set argument", name.local)),
    }
}

fn number_argument(document: &Document, values: &[Value]) -> f64 {
    values
        .first()
        .map_or(f64::NAN, |value| value.to_number(document))
}

fn two_strings(document: &Document, values: &[Value]) -> (String, String) {
    let first = values
        .first()
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default();
    let second = values
        .get(1)
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default();
    (first, second)
}

/// The expanded name of the first node of the argument node-set, or of the
/// context node when there is no argument.
fn name_parts(
    context: &Context<'_>,
    focus: &Focus,
    name: &Name,
    values: &[Value],
) -> Result<Option<(Option<String>, String)>, String> {
    let node = match values.first() {
        None => Some(focus.node.clone()),
        Some(Value::NodeSet(set)) => set.first().cloned(),
        Some(_) => return Err(format!("{}() needs a node-set argument", name.local)),
    };
    Ok(node.and_then(|node| node.expanded_name(context.document)))
}

/// `name()` returns the *qualified* name, prefix included, which is why it
/// cannot be built from the expanded name alone.
fn qualified_name(
    context: &Context<'_>,
    focus: &Focus,
    name: &Name,
    values: &[Value],
) -> Result<String, String> {
    let node = match values.first() {
        None => Some(focus.node.clone()),
        Some(Value::NodeSet(set)) => set.first().cloned(),
        Some(_) => return Err(format!("{}() needs a node-set argument", name.local)),
    };
    let document = context.document;
    Ok(match node {
        Some(XNode::Namespace(namespace)) => namespace.prefix,
        Some(XNode::Tree(id)) => match document.kind(id) {
            Some(NodeKind::Element) => document
                .element(id)
                .map(|element| element.qualified_name())
                .unwrap_or_default(),
            Some(NodeKind::Attr) => document
                .attr(id)
                .map(|attribute| attribute.qualified_name())
                .unwrap_or_default(),
            Some(NodeKind::ProcessingInstruction) => document
                .processing_instruction(id)
                .map(|pi| pi.target().to_owned())
                .unwrap_or_default(),
            _ => String::new(),
        },
        None => String::new(),
    })
}

/// `id()` — the argument is either a node-set (each node's string-value is a
/// whitespace-separated list of identifiers) or a string that is such a list.
fn id_function(context: &Context<'_>, argument: Option<&Value>) -> NodeSet {
    let document = context.document;
    let mut lists: Vec<String> = Vec::new();
    match argument {
        Some(Value::NodeSet(set)) => {
            for node in set.nodes() {
                lists.push(node.string_value(document));
            }
        }
        Some(other) => lists.push(other.to_xpath_string(document)),
        None => {}
    }
    let mut nodes = Vec::new();
    for list in lists {
        for id in list.split_whitespace() {
            if let Some(node) = context.ids.resolve(id) {
                nodes.push(XNode::Tree(node));
            }
        }
    }
    NodeSet::from_nodes(document, nodes)
}

fn substring(document: &Document, values: &[Value]) -> String {
    let text = values
        .first()
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default();
    let characters: Vec<char> = text.chars().collect();
    let start = values
        .get(1)
        .map_or(f64::NAN, |value| value.to_number(document));
    let length = values.get(2).map(|value| value.to_number(document));
    // XPath rounds both arguments and works in one-based positions, so the
    // half-open interval is [round(start), round(start) + round(length)).
    let start = round_half_up(start);
    let end = match length {
        Some(length) => start + round_half_up(length),
        None => f64::INFINITY,
    };
    let mut out = String::new();
    for (index, character) in characters.iter().enumerate() {
        let position = count_as_f64(index + 1);
        if position >= start && position < end {
            out.push(*character);
        }
    }
    out
}

fn translate(document: &Document, values: &[Value]) -> String {
    let text = values
        .first()
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default();
    let from: Vec<char> = values
        .get(1)
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default()
        .chars()
        .collect();
    let to: Vec<char> = values
        .get(2)
        .map(|value| value.to_xpath_string(document))
        .unwrap_or_default()
        .chars()
        .collect();
    let mut out = String::new();
    for character in text.chars() {
        match from.iter().position(|candidate| *candidate == character) {
            None => out.push(character),
            Some(index) => {
                if let Some(replacement) = to.get(index) {
                    out.push(*replacement);
                }
            }
        }
    }
    out
}

/// `lang()` — the nearest `xml:lang` in scope, matched case-insensitively
/// either exactly or up to a `-`.
fn language_matches(document: &Document, focus: &Focus, wanted: &str) -> bool {
    let mut current = focus.node.as_tree();
    while let Some(node) = current {
        if let Some(value) =
            document.get_attribute_ns(node, Some(oxixml_dom::XML_NAMESPACE), "lang")
        {
            let value = value.to_ascii_lowercase();
            let wanted = wanted.to_ascii_lowercase();
            return value == wanted || value.starts_with(&format!("{wanted}-"));
        }
        current = document.parent(node);
    }
    false
}

fn round_half_up(value: f64) -> f64 {
    if value.is_finite() {
        (value + 0.5).floor()
    } else {
        value
    }
}

/// A count as an XPath number; see [`crate::xpath::eval`] for why the
/// conversion is always exact.
#[expect(
    clippy::cast_precision_loss,
    reason = "a node-set or string is far shorter than 2^53"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
