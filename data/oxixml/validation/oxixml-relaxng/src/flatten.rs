//! Rules [4.18](https://relaxng.org/spec-20011203.html#IDA5CAS) through
//! [4.21](https://relaxng.org/spec-20011203.html#IDAEEAS): grammar flattening,
//! `define`/`ref` normalization, and the `notAllowed` and `empty` propagation
//! that together produce the **simple syntax** of
//! [§5](https://relaxng.org/spec-20011203.html#simple-syntax).
//!
//! The output is a [`SimpleGrammar`]: a single `start` pattern plus a flat list of
//! definitions whose bodies are all `element` patterns. That shape is what makes
//! the derivative algorithm terminate — recursion in the schema can only go
//! through an element boundary, which consumes input.

use std::collections::{HashMap, HashSet};

use crate::error::{Position, SchemaError, SchemaErrorKind, simplify};
use crate::tree::{Child, Element};

/// A schema in the simple syntax of §5.
#[derive(Debug, Clone)]
pub(crate) struct SimpleGrammar {
    /// The `start` pattern: either `notAllowed` or a pattern.
    pub start: Element,
    /// The definitions, in the order they were created. Every body is an
    /// `element` pattern.
    pub defines: Vec<(String, Element)>,
}

impl SimpleGrammar {
    /// Looks up a definition body by name.
    pub(crate) fn lookup(&self, name: &str) -> Option<&Element> {
        self.defines
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, body)| body)
    }
}

/// Generates the unique `define` names that rules 4.18 and 4.19 require.
struct Names {
    next: u32,
}

impl Names {
    fn fresh(&mut self) -> String {
        let name = format!("d{}", self.next);
        self.next += 1;
        name
    }
}

/// Runs rules 4.18 to 4.21 over `root`, which must already have been through
/// [`crate::simplify::run`].
pub(crate) fn run(root: Element) -> Result<SimpleGrammar, SchemaError> {
    let mut names = Names { next: 0 };
    let mut grammar = rule_grammar(root, &mut names)?;
    rule_define_and_ref(&mut grammar, &mut names)?;
    rule_not_allowed(&mut grammar);
    rule_empty(&mut grammar);
    prune(&mut grammar);
    Ok(grammar)
}

// ---------------------------------------------------------------------------
// 4.18 grammar element
// ---------------------------------------------------------------------------

fn rule_grammar(root: Element, names: &mut Names) -> Result<SimpleGrammar, SchemaError> {
    // "First, transform the top-level pattern p into <grammar><start>p</start></grammar>."
    let mut start = Element::like(&root, "start");
    let mut top = Element::like(&root, "grammar");
    start.children.push(Child::Element(root));
    top.children.push(Child::Element(start));

    let mut scopes: Vec<HashMap<String, String>> = Vec::new();
    rename(&mut top, &mut scopes, names)?;

    let mut defines = Vec::new();
    let start = extract(top, &mut defines)?;
    Ok(SimpleGrammar { start, defines })
}

/// Renames every `define` so that no two definitions anywhere in the schema share
/// a name, rewriting `ref` and `parentRef` accordingly and turning `parentRef`
/// into `ref`.
fn rename(
    element: &mut Element,
    scopes: &mut Vec<HashMap<String, String>>,
    names: &mut Names,
) -> Result<(), SchemaError> {
    match element.name.as_str() {
        "grammar" => {
            let mut scope = HashMap::new();
            let mut has_start = false;
            for child in element.child_elements() {
                match child.name.as_str() {
                    "start" => has_start = true,
                    "define" => {
                        if let Some(name) = child.attribute("name") {
                            scope
                                .entry(name.to_owned())
                                .or_insert_with(|| names.fresh());
                        }
                    }
                    _ => {}
                }
            }
            if !has_start {
                return Err(
                    simplify("4.18", "a grammar element must have a start child element")
                        .at(element.position),
                );
            }
            scopes.push(scope);
            let result = element
                .child_elements_mut()
                .try_for_each(|child| rename(child, scopes, names));
            scopes.pop();
            result
        }
        "define" => {
            let Some(scope) = scopes.last() else {
                return Err(
                    simplify("4.18", "a define element must be inside a grammar")
                        .at(element.position),
                );
            };
            let current = element.attribute("name").unwrap_or_default();
            let Some(renamed) = scope.get(current).cloned() else {
                return Err(simplify("4.18", format!("unknown definition `{current}`"))
                    .at(element.position));
            };
            element.set_attribute("name", renamed);
            element
                .child_elements_mut()
                .try_for_each(|child| rename(child, scopes, names))
        }
        "ref" | "parentRef" => {
            let depth = scopes.len();
            let index = if element.name == "ref" {
                depth.checked_sub(1)
            } else {
                depth.checked_sub(2)
            };
            let current = element.attribute("name").unwrap_or_default().to_owned();
            let renamed = index
                .and_then(|index| scopes.get(index))
                .and_then(|scope| scope.get(&current))
                .cloned();
            let Some(renamed) = renamed else {
                return Err(simplify(
                    "4.18",
                    format!(
                        "the {} element `{current}` does not refer to a define element",
                        element.name
                    ),
                )
                .at(element.position));
            };
            element.name = "ref".to_owned();
            element.set_attribute("name", renamed);
            Ok(())
        }
        _ => element
            .child_elements_mut()
            .try_for_each(|child| rename(child, scopes, names)),
    }
}

/// Replaces each nested `grammar` element by the child of its `start` element,
/// hoisting every definition into `defines`.
fn extract(
    mut element: Element,
    defines: &mut Vec<(String, Element)>,
) -> Result<Element, SchemaError> {
    if element.name != "grammar" {
        let children = std::mem::take(&mut element.children);
        for child in children {
            match child {
                Child::Element(child) => {
                    element
                        .children
                        .push(Child::Element(extract(child, defines)?));
                }
                Child::Text(text) => element.children.push(Child::Text(text)),
            }
        }
        return Ok(element);
    }

    let position = element.position;
    let mut start_body = None;
    for child in element.take_child_elements() {
        match child.name.as_str() {
            "start" => {
                let mut child = child;
                if let Some(body) = child.take_child_elements().into_iter().next() {
                    start_body = Some(body);
                }
            }
            "define" => {
                let name = child.attribute("name").unwrap_or_default().to_owned();
                let mut child = child;
                if let Some(body) = child.take_child_elements().into_iter().next() {
                    let body = extract(body, defines)?;
                    defines.push((name, body));
                }
            }
            _ => {}
        }
    }
    let Some(start_body) = start_body else {
        return Err(
            simplify("4.18", "a grammar element must have a start child element").at(position),
        );
    };
    extract(start_body, defines)
}

// ---------------------------------------------------------------------------
// 4.19 define and ref elements
// ---------------------------------------------------------------------------

fn rule_define_and_ref(grammar: &mut SimpleGrammar, names: &mut Names) -> Result<(), SchemaError> {
    keep_reachable(grammar);

    // Every `element` pattern that is not already the child of a `define` becomes
    // one, and is replaced in place by a `ref`.
    let mut created = Vec::new();
    hoist_pattern(&mut grammar.start, &mut created, names);
    for (_, body) in &mut grammar.defines {
        hoist_inside(body, &mut created, names);
    }
    grammar.defines.append(&mut created);

    // Definitions whose body is not an `element` pattern are expanded inline.
    let expandable: HashMap<String, Element> = grammar
        .defines
        .iter()
        .filter(|(_, body)| body.name != "element")
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect();
    if !expandable.is_empty() {
        let mut visiting = Vec::new();
        expand_pattern(&mut grammar.start, &expandable, &mut visiting)?;
        for (_, body) in &mut grammar.defines {
            if body.name == "element" {
                expand_inside(body, &expandable, &mut visiting)?;
            }
        }
        grammar.defines.retain(|(_, body)| body.name == "element");
    }

    keep_reachable(grammar);
    Ok(())
}

/// Drops the definitions that no reachable `ref` refers to.
fn keep_reachable(grammar: &mut SimpleGrammar) {
    let mut reachable: HashSet<String> = HashSet::new();
    let mut queue: Vec<String> = Vec::new();
    collect_refs(&grammar.start, &mut queue);
    while let Some(name) = queue.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(body) = grammar.lookup(&name) {
            collect_refs(body, &mut queue);
        }
    }
    grammar
        .defines
        .retain(|(name, _)| reachable.contains(name.as_str()));
}

fn collect_refs(element: &Element, out: &mut Vec<String>) {
    if element.name == "ref"
        && let Some(name) = element.attribute("name")
    {
        out.push(name.to_owned());
    }
    for child in element.child_elements() {
        collect_refs(child, out);
    }
}

/// Replaces every `element` descendant of `element` (but not `element` itself) by
/// a `ref` to a freshly created definition.
fn hoist_inside(element: &mut Element, created: &mut Vec<(String, Element)>, names: &mut Names) {
    for child in element.child_elements_mut() {
        hoist_inside(child, created, names);
        if child.name == "element" {
            let name = names.fresh();
            let mut placeholder = Element::like(child, "ref");
            placeholder.set_attribute("name", name.clone());
            let taken = std::mem::replace(child, placeholder);
            created.push((name, taken));
        }
    }
}

/// Like [`hoist_inside`], but also hoists `pattern` itself when it is an
/// `element` pattern.
fn hoist_pattern(pattern: &mut Element, created: &mut Vec<(String, Element)>, names: &mut Names) {
    hoist_inside(pattern, created, names);
    if pattern.name == "element" {
        let name = names.fresh();
        let mut placeholder = Element::like(pattern, "ref");
        placeholder.set_attribute("name", name.clone());
        let taken = std::mem::replace(pattern, placeholder);
        created.push((name, taken));
    }
}

fn expand_inside(
    element: &mut Element,
    expandable: &HashMap<String, Element>,
    visiting: &mut Vec<String>,
) -> Result<(), SchemaError> {
    for child in element.child_elements_mut() {
        expand_pattern(child, expandable, visiting)?;
    }
    Ok(())
}

fn expand_pattern(
    pattern: &mut Element,
    expandable: &HashMap<String, Element>,
    visiting: &mut Vec<String>,
) -> Result<(), SchemaError> {
    if pattern.name == "ref" {
        let name = pattern.attribute("name").unwrap_or_default().to_owned();
        if let Some(body) = expandable.get(&name) {
            if visiting.contains(&name) {
                return Err(simplify(
                    "4.19",
                    "expanding a ref element requires expanding itself, which is a loop",
                )
                .at(pattern.position));
            }
            visiting.push(name);
            let mut replacement = body.clone();
            let result = expand_pattern(&mut replacement, expandable, visiting);
            visiting.pop();
            result?;
            *pattern = replacement;
            return Ok(());
        }
        return Ok(());
    }
    expand_inside(pattern, expandable, visiting)
}

// ---------------------------------------------------------------------------
// 4.20 notAllowed element
// ---------------------------------------------------------------------------

fn rule_not_allowed(grammar: &mut SimpleGrammar) {
    not_allowed(&mut grammar.start);
    for (_, body) in &mut grammar.defines {
        not_allowed(body);
    }
}

fn not_allowed(element: &mut Element) {
    for child in element.child_elements_mut() {
        not_allowed(child);
    }

    // "An except element that has a notAllowed child element is removed."
    if element.name == "data" {
        element.children.retain(|child| match child {
            Child::Element(child) => {
                child.name != "except" || !child.child_elements().any(|p| p.name == "notAllowed")
            }
            Child::Text(_) => true,
        });
    }

    let has_not_allowed_child = |element: &Element| {
        element
            .child_elements()
            .any(|child| child.name == "notAllowed")
    };

    match element.name.as_str() {
        "attribute" | "list" | "group" | "interleave" | "oneOrMore"
            if has_not_allowed_child(element) =>
        {
            *element = Element::like(element, "notAllowed");
        }
        "choice" => {
            let children = element.take_child_elements();
            let mut kept: Vec<Element> = children
                .into_iter()
                .filter(|child| child.name != "notAllowed")
                .collect();
            match kept.len() {
                0 => *element = Element::like(element, "notAllowed"),
                1 => {
                    if let Some(only) = kept.pop() {
                        *element = only;
                    }
                }
                _ => element.set_child_elements(kept),
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// 4.21 empty element
// ---------------------------------------------------------------------------

fn rule_empty(grammar: &mut SimpleGrammar) {
    empty(&mut grammar.start);
    for (_, body) in &mut grammar.defines {
        empty(body);
    }
}

fn empty(element: &mut Element) {
    for child in element.child_elements_mut() {
        empty(child);
    }
    match element.name.as_str() {
        "group" | "interleave" => {
            let mut children = element.take_child_elements();
            let empties = children.iter().filter(|c| c.name == "empty").count();
            if empties == children.len() && !children.is_empty() {
                *element = Element::like(element, "empty");
            } else {
                children.retain(|child| child.name != "empty");
                if children.len() == 1 {
                    if let Some(only) = children.pop() {
                        *element = only;
                    }
                } else {
                    element.set_child_elements(children);
                }
            }
        }
        "choice" => {
            let children = element.take_child_elements();
            let empties = children.iter().filter(|c| c.name == "empty").count();
            if empties == children.len() && !children.is_empty() {
                *element = Element::like(element, "empty");
            } else {
                let mut children = children;
                if children.len() == 2 && children.get(1).is_some_and(|c| c.name == "empty") {
                    children.swap(0, 1);
                }
                element.set_child_elements(children);
            }
        }
        "oneOrMore" if element.child_elements().any(|child| child.name == "empty") => {
            *element = Element::like(element, "empty");
        }
        _ => {}
    }
}

/// Drops definitions that became unreachable during 4.20 and 4.21.
fn prune(grammar: &mut SimpleGrammar) {
    keep_reachable(grammar);
}

/// Reports a schema error for an unexpected shape in the simple form. Used by the
/// compiler, which trusts that simplification produced §5-conforming output.
pub(crate) fn malformed(element: &Element, expectation: &str) -> SchemaError {
    SchemaError::new(
        SchemaErrorKind::Simplification,
        "5",
        format!(
            "the simplified schema has `{}` where {expectation} was expected",
            element.name
        ),
    )
    .at(if element.position.is_unknown() {
        Position::UNKNOWN
    } else {
        element.position
    })
}
