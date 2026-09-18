//! The content models of the XSLT vocabulary.
//!
//! [XSLT 3.0 §3.7](https://www.w3.org/TR/xslt-30/#element-syntax-summary) gives, for every
//! element in the XSLT namespace, the children it may have and the order they must come in.
//! [§2.9](https://www.w3.org/TR/xslt-30/#stylesheet-element) makes a violation `XTSE0010`:
//! *"it is a static error if an XSLT-defined element is used in a context where it is not
//! permitted, if a required attribute is omitted, or if the content of the element does not
//! correspond to the content that is allowed for the element"*.
//!
//! The tables here are the machine-readable form of that summary. They are consulted by
//! [`crate::validate`], which walks the module once and applies both the attribute rules and
//! the content rules.

use crate::compile::xslt_local_name;
use crate::error::{XsltError, XsltResult};
use crate::names::{XSLT_NAMESPACE, xslt_element};
use oxixml_infoset::{NodeKind, NodeRef, XdmNode};

/// One position in an ordered content model.
#[derive(Clone, Copy)]
pub(crate) struct Slot {
    /// The XSLT element names this position accepts.
    names: &'static [&'static str],
    /// Whether the position accepts at most one element.
    single: bool,
}

/// What an element is allowed to contain.
#[derive(Clone, Copy)]
pub(crate) enum Model {
    /// No element children and no non-whitespace text.
    Empty,
    /// Text only — the model of `xsl:text`.
    TextOnly,
    /// A sequence constructor: text, literal result elements and XSLT instructions.
    Sequence,
    /// A fixed prefix of XSLT elements, optionally followed by a sequence constructor.
    Ordered(&'static [Slot], bool),
    /// The declarations an `xsl:stylesheet` may contain.
    Declarations,
    /// Not checked — the element belongs to a documented gap (packaging, schema-awareness,
    /// streaming, `xsl:merge`, `xsl:try`), whose own rules this processor does not implement.
    Unchecked,
}

/// Builds a repeatable slot.
const fn many(names: &'static [&'static str]) -> Slot {
    Slot {
        names,
        single: false,
    }
}

/// Builds a slot that accepts at most one element.
const fn one(names: &'static [&'static str]) -> Slot {
    Slot {
        names,
        single: true,
    }
}

const TEMPLATE_SLOTS: &[Slot] = &[one(&["context-item"]), many(&["param"])];
const FUNCTION_SLOTS: &[Slot] = &[many(&["param"])];
const ITERATE_SLOTS: &[Slot] = &[many(&["param"]), one(&["on-completion"])];
const SORTED_SLOTS: &[Slot] = &[many(&["sort"])];
const WITH_PARAM_SLOTS: &[Slot] = &[many(&["with-param"])];
const APPLY_TEMPLATES_SLOTS: &[Slot] = &[many(&["sort", "with-param"])];
const NEXT_MATCH_SLOTS: &[Slot] = &[many(&["with-param", "fallback"])];
const CHOOSE_SLOTS: &[Slot] = &[many(&["when"]), one(&["otherwise"])];
const ANALYZE_STRING_SLOTS: &[Slot] = &[
    one(&["matching-substring"]),
    one(&["non-matching-substring"]),
    many(&["fallback"]),
];
const ATTRIBUTE_SET_SLOTS: &[Slot] = &[many(&["attribute"])];
const CHARACTER_MAP_SLOTS: &[Slot] = &[many(&["output-character"])];
const EVALUATE_SLOTS: &[Slot] = &[many(&["with-param"]), many(&["fallback"])];
/// [XSLT 3.0 §15.1](https://www.w3.org/TR/xslt-30/#element-merge) writes the content of
/// `xsl:merge` as `(xsl:merge-source+, xsl:merge-action, xsl:fallback*)`, so an `xsl:fallback`
/// may appear only after the merge action, and nothing else may appear at all.
const MERGE_SLOTS: &[Slot] = &[
    many(&["merge-source"]),
    one(&["merge-action"]),
    many(&["fallback"]),
];

/// The content model of an XSLT element, by local name.
#[must_use]
pub(crate) fn model_of(local: &str) -> Model {
    match local {
        // Empty elements.
        "context-item"
        | "copy-of"
        | "decimal-format"
        | "global-context-item"
        | "import"
        | "include"
        | "mode"
        | "namespace-alias"
        | "number"
        | "output"
        | "output-character"
        | "preserve-space"
        | "strip-space" => Model::Empty,
        "text" => Model::TextOnly,
        // Instructions and declarations whose content is a sequence constructor.
        "assert"
        | "attribute"
        | "break"
        | "catch"
        | "comment"
        | "copy"
        | "document"
        | "element"
        | "fallback"
        | "if"
        | "key"
        | "map"
        | "map-entry"
        | "matching-substring"
        | "message"
        | "namespace"
        | "non-matching-substring"
        | "on-completion"
        | "on-empty"
        | "on-non-empty"
        | "otherwise"
        | "param"
        | "processing-instruction"
        | "result-document"
        | "sequence"
        | "sort"
        | "value-of"
        | "variable"
        | "when"
        | "where-populated"
        | "with-param" => Model::Sequence,
        "template" => Model::Ordered(TEMPLATE_SLOTS, true),
        "function" => Model::Ordered(FUNCTION_SLOTS, true),
        "iterate" => Model::Ordered(ITERATE_SLOTS, true),
        "for-each" | "for-each-group" | "perform-sort" => Model::Ordered(SORTED_SLOTS, true),
        "apply-imports" | "call-template" | "next-iteration" => {
            Model::Ordered(WITH_PARAM_SLOTS, false)
        }
        "apply-templates" => Model::Ordered(APPLY_TEMPLATES_SLOTS, false),
        "next-match" => Model::Ordered(NEXT_MATCH_SLOTS, false),
        "choose" => Model::Ordered(CHOOSE_SLOTS, false),
        "analyze-string" => Model::Ordered(ANALYZE_STRING_SLOTS, false),
        "attribute-set" => Model::Ordered(ATTRIBUTE_SET_SLOTS, false),
        "character-map" => Model::Ordered(CHARACTER_MAP_SLOTS, false),
        "evaluate" => Model::Ordered(EVALUATE_SLOTS, false),
        "merge" => Model::Ordered(MERGE_SLOTS, false),
        "stylesheet" | "transform" => Model::Declarations,
        _ => Model::Unchecked,
    }
}

/// The XSLT elements that may appear in a sequence constructor —
/// [XSLT 3.0 §3.7](https://www.w3.org/TR/xslt-30/#element-syntax-summary) calls these the
/// *instructions*.
#[must_use]
pub(crate) fn is_instruction(local: &str) -> bool {
    matches!(
        local,
        "analyze-string"
            | "apply-imports"
            | "apply-templates"
            | "assert"
            | "attribute"
            | "break"
            | "call-template"
            | "choose"
            | "comment"
            | "copy"
            | "copy-of"
            | "document"
            | "element"
            | "evaluate"
            | "fallback"
            | "for-each"
            | "for-each-group"
            | "fork"
            | "if"
            | "iterate"
            | "map"
            | "map-entry"
            | "merge"
            | "message"
            | "namespace"
            | "next-iteration"
            | "next-match"
            | "number"
            | "on-empty"
            | "on-non-empty"
            | "perform-sort"
            | "processing-instruction"
            | "result-document"
            | "sequence"
            | "source-document"
            | "text"
            | "try"
            | "value-of"
            | "variable"
            | "where-populated"
    )
}

/// The XSLT elements that may appear at the top level of a stylesheet module —
/// [XSLT 3.0 §3.6](https://www.w3.org/TR/xslt-30/#stylesheet-element).
#[must_use]
pub(crate) fn is_declaration(local: &str) -> bool {
    matches!(
        local,
        "accumulator"
            | "attribute-set"
            | "character-map"
            | "decimal-format"
            | "expose"
            | "function"
            | "global-context-item"
            | "import"
            | "import-schema"
            | "include"
            | "key"
            | "mode"
            | "namespace-alias"
            | "output"
            | "override"
            | "param"
            | "preserve-space"
            | "strip-space"
            | "template"
            | "use-package"
            | "variable"
    )
}

/// One child of an element, reduced to what the content model cares about.
enum Item {
    /// An element in the XSLT namespace, with the local name XSLT defines for it.
    Xslt(&'static str),
    /// An element in the XSLT namespace whose local name XSLT does not define.
    UnknownXslt(String),
    /// An element outside the XSLT namespace.
    Other(NodeRef),
    /// A text node that is not entirely whitespace.
    Text,
}

/// The children of `element` that a content model constrains: elements, plus text nodes that
/// carry something other than whitespace. Comments and processing instructions are ignored,
/// as [XSLT 3.0 §4.2](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) directs.
fn items(element: &NodeRef) -> Vec<Item> {
    items_of(element.children())
}

/// [`items`], over a list of children a caller has already filtered.
fn items_of(children: Vec<NodeRef>) -> Vec<Item> {
    let mut out = Vec::new();
    for child in children {
        match child.node_kind() {
            NodeKind::Element => match child.node_name() {
                Some(name) if name.namespace() == Some(XSLT_NAMESPACE) => {
                    match xslt_element(name.local_name()) {
                        Some(local) => out.push(Item::Xslt(local)),
                        // In forwards-compatible mode an element XSLT 3.0 does not define is
                        // not an error: it stands for a later version's instruction, and
                        // XSLT 3.0 §3.9 has the `xsl:fallback` machinery deal with it.
                        None if forwards_compatible(&child) => out.push(Item::Other(child)),
                        None => out.push(Item::UnknownXslt(name.local_name().to_owned())),
                    }
                }
                _ => out.push(Item::Other(child)),
            },
            NodeKind::Text if !child.string_value().trim().is_empty() => out.push(Item::Text),
            _ => {}
        }
    }
    out
}

/// Whether the element states a version later than the one this processor implements, putting
/// it in the forwards-compatible mode of
/// [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#forwards).
fn forwards_compatible(element: &NodeRef) -> bool {
    element.attributes().into_iter().any(|attribute| {
        let Some(name) = attribute.node_name() else {
            return false;
        };
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        owned
            && name.local_name() == "version"
            && attribute
                .string_value()
                .trim()
                .parse::<f64>()
                .is_ok_and(|stated| stated > 3.0)
    })
}

/// Checks that the children of `element` match the content model `model`.
///
/// # Errors
///
/// `XTSE0010` when a child is not permitted where it appears, `XTSE0120` for stray text at the
/// top level of a stylesheet, `XTSE0130` for an unnamespaced top-level element.
pub(crate) fn check(element: &NodeRef, owner: &str, model: Model) -> XsltResult<()> {
    match model {
        Model::Unchecked => Ok(()),
        Model::Empty => {
            // XSLT 3.0 §3.7: "within an XSLT element that is required to be empty, any
            // content other than comments or processing instructions is a static error".
            if !items(element).is_empty() {
                return Err(XsltError::stat1c(
                    "XTSE0260",
                    format!("`xsl:{owner}` must be empty"),
                ));
            }
            Ok(())
        }
        Model::TextOnly => {
            for item in items(element) {
                if !matches!(item, Item::Text) {
                    return Err(wrong_content(owner, &item, "which may contain only text"));
                }
            }
            Ok(())
        }
        Model::Sequence => {
            for item in items(element) {
                check_sequence_item(owner, &item)?;
            }
            Ok(())
        }
        Model::Ordered(slots, tail) => check_ordered(element, owner, slots, tail),
        Model::Declarations => check_declarations(element),
    }
}

fn check_ordered(
    element: &NodeRef,
    owner: &str,
    slots: &'static [Slot],
    tail: bool,
) -> XsltResult<()> {
    let mut cursor = 0usize;
    let mut in_tail = false;
    for item in items(element) {
        if in_tail {
            check_sequence_item(owner, &item)?;
            continue;
        }
        if let Item::Xslt(local) = item
            && let Some(index) = slots
                .iter()
                .enumerate()
                .skip(cursor)
                .find_map(|(index, slot)| slot.names.contains(&local).then_some(index))
        {
            cursor = if slots[index].single {
                index + 1
            } else {
                index
            };
            continue;
        }
        if !tail {
            return Err(wrong_content(owner, &item, "which does not allow it there"));
        }
        in_tail = true;
        check_sequence_item(owner, &item)?;
    }
    Ok(())
}

fn check_sequence_item(owner: &str, item: &Item) -> XsltResult<()> {
    match item {
        Item::Text | Item::Other(_) => Ok(()),
        Item::Xslt(local) if is_instruction(local) => Ok(()),
        // XSLT 3.0 §3.10 makes each of these a top-level declaration under its own code.
        Item::Xslt("include") => Err(XsltError::stat1c(
            "XTSE0170",
            "`xsl:include` must be a top-level declaration",
        )),
        Item::Xslt("import") => Err(XsltError::stat1c(
            "XTSE0190",
            "`xsl:import` must be a top-level declaration",
        )),
        _ => Err(wrong_content(
            owner,
            item,
            "whose content is a sequence constructor",
        )),
    }
}

fn check_declarations(element: &NodeRef) -> XsltResult<()> {
    // [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#forwards): an element whose *effective
    // version* is later than the one this processor implements is read in forwards-compatible
    // mode, and at the top level that means it is ignored — whatever it is. `forwards-005`
    // writes `<xsl:value-of select="$x" version="4.0"/>` beside the template declarations of a
    // 3.0 module: a recognized element in a position 3.0 does not allow, standing for something
    // a later version does.
    let admitted: Vec<NodeRef> = element
        .children()
        .into_iter()
        .filter(|child| !forwards_compatible(child))
        .collect();
    for item in items_of(admitted) {
        match item {
            Item::Text => {
                return Err(XsltError::stat1c(
                    "XTSE0120",
                    "a stylesheet element may not have a text node child",
                ));
            }
            Item::Other(child) => {
                let namespaced = child
                    .node_name()
                    .is_some_and(|name| name.namespace().is_some());
                if !namespaced {
                    return Err(XsltError::stat1c(
                        "XTSE0130",
                        "a top-level element in no namespace is not a declaration",
                    ));
                }
            }
            // XSLT 3.0 §3.10.2 dropped the XSLT 2.0 rule that `xsl:import` must precede
            // every other declaration, so no ordering is imposed here.
            Item::Xslt(local) if is_declaration(local) => {}
            item => {
                return Err(wrong_content(
                    "stylesheet",
                    &item,
                    "which allows only declarations",
                ));
            }
        }
    }
    Ok(())
}

fn wrong_content(owner: &str, item: &Item, tail: &str) -> XsltError {
    let what = match item {
        Item::Xslt(local) => format!("`xsl:{local}`"),
        Item::UnknownXslt(local) => format!("`xsl:{local}`, which XSLT does not define,"),
        Item::Other(node) => match node.node_name() {
            Some(name) => format!("`{}`", name.local_name()),
            None => "an element".to_owned(),
        },
        Item::Text => "text".to_owned(),
    };
    XsltError::stat1c(
        "XTSE0010",
        format!("{what} is not permitted in `xsl:{owner}`, {tail}"),
    )
}

/// Whether the element is an extension instruction, whose content this processor does not
/// constrain: the prefix of its name is named by an `extension-element-prefixes` attribute in
/// scope.
#[must_use]
pub(crate) fn is_extension_element(element: &NodeRef, namespaces: &[String]) -> bool {
    if xslt_local_name(element).is_some() {
        return false;
    }
    element
        .node_name()
        .and_then(|name| name.namespace().map(str::to_owned))
        .is_some_and(|namespace| namespaces.contains(&namespace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_instruction_set_excludes_the_declarations() {
        assert!(is_instruction("value-of"));
        assert!(!is_instruction("param"));
        assert!(!is_instruction("sort"));
        assert!(!is_instruction("with-param"));
    }

    #[test]
    fn the_declaration_set_excludes_the_instructions() {
        assert!(is_declaration("template"));
        assert!(!is_declaration("if"));
    }

    #[test]
    fn the_models_cover_the_common_elements() {
        assert!(matches!(model_of("text"), Model::TextOnly));
        assert!(matches!(model_of("output"), Model::Empty));
        assert!(matches!(model_of("if"), Model::Sequence));
        assert!(matches!(model_of("choose"), Model::Ordered(_, false)));
        assert!(matches!(model_of("template"), Model::Ordered(_, true)));
        // `xsl:merge` is a documented gap, but its *content model* is checked all the same:
        // XSLT 3.0 §15.1 puts `xsl:fallback` after the merge action, and a stylesheet that gets
        // that wrong is `XTSE0010` whether or not the instruction is implemented.
        assert!(matches!(model_of("merge"), Model::Ordered(_, false)));
        assert!(matches!(model_of("merge-action"), Model::Unchecked));
    }
}

/// Where an `xsl:break` or `xsl:next-iteration` element stands relative to the `xsl:iterate`
/// that governs it — [XSLT 3.0 §8.8](https://www.w3.org/TR/xslt-30/#iterate).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BreakPosition {
    /// Not lexically within any `xsl:iterate`.
    Outside,
    /// Within an `xsl:iterate`, but not in a position that can end an iteration.
    Inner,
    /// The last instruction of the `xsl:iterate` body, or of a conditional that is itself in
    /// such a position.
    Tail,
}

/// The position a child of `parent` occupies, given the parent's own position.
#[must_use]
pub(crate) fn child_break_position(
    parent: Option<&str>,
    parent_position: BreakPosition,
    is_last: bool,
) -> BreakPosition {
    match parent {
        // The body of `xsl:iterate` ends with the instruction that decides the next step.
        Some("iterate") if is_last => BreakPosition::Tail,
        Some("iterate") => BreakPosition::Inner,
        // A conditional in that position passes it on: each branch of `xsl:choose` ends the
        // iteration, and so does the last instruction of `xsl:if`, `xsl:when`, `xsl:otherwise`.
        Some("choose") if parent_position == BreakPosition::Tail => BreakPosition::Tail,
        Some("if") | Some("when") | Some("otherwise") | Some("try") | Some("catch")
            if parent_position == BreakPosition::Tail && is_last =>
        {
            BreakPosition::Tail
        }
        _ if parent_position == BreakPosition::Outside => BreakPosition::Outside,
        _ => BreakPosition::Inner,
    }
}

/// Applies the static rules that are about an element's own attributes and children taken
/// together, rather than about either alone.
///
/// # Errors
///
/// `XTSE3125` for a `select` alongside a sequence constructor,
/// `XTSE0580` for two parameters of the same name, `XTSE0670` for two `xsl:with-param`
/// elements of the same name, `XTSE3130` for an `xsl:with-param` on `xsl:next-iteration` that
/// names no parameter of the governing `xsl:iterate`.
pub(crate) fn check_static_rules(element: &NodeRef, local: &str) -> XsltResult<()> {
    // XSLT 3.0 §8.8: `xsl:break` and `xsl:on-completion` take a value either from `select` or
    // from their content, never from both.
    if matches!(local, "break" | "on-completion")
        && has_attribute(element, "select")
        && !items(element).is_empty()
    {
        return Err(XsltError::stat1c(
            "XTSE3125",
            format!("`xsl:{local}` has both a `select` attribute and content"),
        ));
    }
    check_duplicate_names(element, "param", "XTSE0580")?;
    check_duplicate_names(element, "with-param", "XTSE0670")?;
    if local == "next-iteration" {
        check_next_iteration_params(element)?;
    }
    check_xslt_namespace_attributes(element, local)?;
    check_select_and_content(element, local)?;
    check_instruction_rules(element, local)?;
    check_sort_children(element)?;
    Ok(())
}

/// Rejects an attribute in the XSLT namespace written on an XSLT element, which
/// [XSLT 3.0 §3.4](https://www.w3.org/TR/xslt-30/#standard-attributes) allows only in the
/// prefixed form on a literal result element.
fn check_xslt_namespace_attributes(element: &NodeRef, local: &str) -> XsltResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        if name.namespace() == Some(XSLT_NAMESPACE) {
            return Err(XsltError::stat1c(
                "XTSE0090",
                format!(
                    "`xsl:{}` is not an attribute `xsl:{local}` may carry",
                    name.local_name()
                ),
            ));
        }
    }
    Ok(())
}

/// `(element, error code)` for the elements whose `select` attribute forbids content.
///
/// Each entry is one clause of [XSLT 3.0 Appendix G](https://www.w3.org/TR/xslt-30/#error-summary):
/// *"it is a static error if the `select` attribute of the `xsl:…` element is present unless
/// the element has empty content"*.
const SELECT_EXCLUDES_CONTENT: &[(&str, &str)] = &[
    ("attribute", "XTSE0840"),
    ("map-entry", "XTSE3280"),
    ("comment", "XTSE0940"),
    ("catch", "XTSE3150"),
    ("namespace", "XTSE0910"),
    ("param", "XTSE0620"),
    ("processing-instruction", "XTSE0880"),
    ("sort", "XTSE1015"),
    ("value-of", "XTSE0870"),
    ("variable", "XTSE0620"),
    ("with-param", "XTSE0620"),
];

fn check_select_and_content(element: &NodeRef, local: &str) -> XsltResult<()> {
    if let Some((_, code)) = SELECT_EXCLUDES_CONTENT
        .iter()
        .find(|(name, _)| *name == local)
        && has_attribute(element, "select")
        && !items(element).is_empty()
    {
        return Err(XsltError::stat1c(
            *code,
            format!("`xsl:{local}` has both a `select` attribute and content"),
        ));
    }
    // `xsl:sequence` and `xsl:perform-sort` allow one kind of child alongside `select`.
    if local == "sequence"
        && has_attribute(element, "select")
        && items(element)
            .iter()
            .any(|item| !matches!(item, Item::Xslt("fallback")))
    {
        return Err(XsltError::stat1c(
            "XTSE3185",
            "`xsl:sequence` has a `select` attribute and content other than `xsl:fallback`",
        ));
    }
    if local == "try"
        && has_attribute(element, "select")
        && items(element)
            .iter()
            .any(|item| !matches!(item, Item::Xslt("catch") | Item::Xslt("fallback")))
    {
        return Err(XsltError::stat1c(
            "XTSE3140",
            "`xsl:try` has a `select` attribute and content other than `xsl:catch`",
        ));
    }
    if local == "perform-sort"
        && has_attribute(element, "select")
        && items(element)
            .iter()
            .any(|item| !matches!(item, Item::Xslt("sort")))
    {
        return Err(XsltError::stat1c(
            "XTSE1040",
            "`xsl:perform-sort` has a `select` attribute and content other than `xsl:sort`",
        ));
    }
    Ok(())
}

/// The rules that constrain one instruction's own attributes and children.
fn check_instruction_rules(element: &NodeRef, local: &str) -> XsltResult<()> {
    match local {
        // XSLT 3.0 §12.1: `value` computes the number outright, so nothing may select or
        // count nodes alongside it.
        "number" if has_attribute(element, "value") => {
            for other in ["select", "level", "count", "from"] {
                if has_attribute(element, other) {
                    return Err(XsltError::stat1c(
                        "XTSE0975",
                        format!("`xsl:number` has both `value` and `{other}`"),
                    ));
                }
            }
            Ok(())
        }
        // XSLT 3.0 §14.2: the four grouping attributes are mutually exclusive, and exactly one
        // must be present; `collation` and `composite` only mean something for two of them.
        "for-each-group" => {
            let by = ["group-by", "group-adjacent"]
                .iter()
                .filter(|name| has_attribute(element, name))
                .count();
            let population = by
                + ["group-starting-with", "group-ending-with"]
                    .iter()
                    .filter(|name| has_attribute(element, name))
                    .count();
            if population != 1 {
                return Err(XsltError::stat1c(
                    "XTSE1080",
                    "`xsl:for-each-group` needs exactly one grouping attribute",
                ));
            }
            if by == 0
                && ["collation", "composite"]
                    .iter()
                    .any(|name| has_attribute(element, name))
            {
                return Err(XsltError::stat1c(
                    "XTSE1090",
                    "`collation` and `composite` need `group-by` or `group-adjacent`",
                ));
            }
            Ok(())
        }
        // XSLT 3.0 §15.1: an `xsl:analyze-string` that handles neither kind of substring does
        // nothing at all, which the specification makes an error rather than a no-op.
        "analyze-string" => {
            let handled = items(element).iter().any(|item| {
                matches!(
                    item,
                    Item::Xslt("matching-substring") | Item::Xslt("non-matching-substring")
                )
            });
            if handled {
                Ok(())
            } else {
                Err(XsltError::stat1c(
                    "XTSE1130",
                    "`xsl:analyze-string` needs `xsl:matching-substring` or \
                     `xsl:non-matching-substring`",
                ))
            }
        }
        // XSLT 3.0 §15.1: the content of `xsl:merge` is `(xsl:merge-source+, xsl:merge-action,
        // xsl:fallback*)`. The ordering — which is what puts a stray `xsl:fallback` before the
        // action in the wrong place — is the business of `MERGE_SLOTS`; the *number* of each is
        // checked here, because a slot says nothing about how few of an element there may be.
        "merge" => {
            let sources = items(element)
                .iter()
                .filter(|item| matches!(item, Item::Xslt("merge-source")))
                .count();
            let actions = items(element)
                .iter()
                .filter(|item| matches!(item, Item::Xslt("merge-action")))
                .count();
            if sources == 0 {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    "`xsl:merge` needs at least one `xsl:merge-source`",
                ));
            }
            if actions != 1 {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    "`xsl:merge` needs exactly one `xsl:merge-action`",
                ));
            }
            Ok(())
        }
        // XSLT 3.0 §10.3: a key is computed either by `use` or by a sequence constructor.
        "key" => {
            let has_use = has_attribute(element, "use");
            let has_content = !items(element).is_empty();
            if has_use == has_content {
                return Err(XsltError::stat1c(
                    "XTSE1205",
                    "`xsl:key` needs either a `use` attribute or a sequence constructor",
                ));
            }
            Ok(())
        }
        // XSLT 3.0 §10.1: only a parameter of a template may be a tunnel parameter; a global
        // parameter and a function parameter may not.
        "param"
            if matches!(
                element.parent().as_ref().and_then(xslt_local_name),
                Some("function") | Some("stylesheet") | Some("transform")
            ) && element.attributes().into_iter().any(|attribute| {
                attribute
                    .node_name()
                    .is_some_and(|name| name.namespace().is_none() && name.local_name() == "tunnel")
                    && matches!(attribute.string_value().trim(), "yes" | "true" | "1")
            }) =>
        {
            Err(XsltError::stat1c(
                "XTSE0020",
                "only a parameter of a template may be a tunnel parameter",
            ))
        }
        // XSLT 3.0 §8.8: a parameter of `xsl:iterate` always has a value on the first
        // iteration, so it may not be implicitly mandatory.
        "iterate" => {
            for child in element.children() {
                if child.node_kind() != NodeKind::Element
                    || xslt_local_name(&child) != Some("param")
                {
                    continue;
                }
                // Only a required type that excludes the empty sequence makes the parameter
                // implicitly mandatory: otherwise the empty sequence is its first value.
                let mandatory = attribute_value(&child, "as")
                    .is_some_and(|as_type| !as_type.trim_end().ends_with(['?', '*']));
                if mandatory && !has_attribute(&child, "select") && items(&child).is_empty() {
                    return Err(XsltError::stat1c(
                        "XTSE3520",
                        "a parameter of `xsl:iterate` needs a default value",
                    ));
                }
            }
            Ok(())
        }
        // XSLT 3.0 §27.1: a processor without schema support must reject a schema import.
        "import-schema" => Err(XsltError::stat1c(
            "XTSE1650",
            "`xsl:import-schema` needs a schema-aware processor",
        )),
        // XSLT 3.0 §10.3: every argument of a stylesheet function is supplied by the caller,
        // so its parameters may not have defaults.
        "function" => {
            for child in element.children() {
                if child.node_kind() != NodeKind::Element
                    || xslt_local_name(&child) != Some("param")
                {
                    continue;
                }
                if has_attribute(&child, "select") || !items(&child).is_empty() {
                    return Err(XsltError::stat1c(
                        "XTSE0760",
                        "a parameter of `xsl:function` may not have a default value",
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Rejects a `stable` attribute on any but the first of a run of sibling `xsl:sort` elements —
/// [XSLT 3.0 §13.1](https://www.w3.org/TR/xslt-30/#sorting) raises `XTSE1017`.
fn check_sort_children(element: &NodeRef) -> XsltResult<()> {
    let mut seen = false;
    for child in element.children() {
        if child.node_kind() != NodeKind::Element || xslt_local_name(&child) != Some("sort") {
            continue;
        }
        if seen && has_attribute(&child, "stable") {
            return Err(XsltError::stat1c(
                "XTSE1017",
                "only the first of a run of `xsl:sort` elements may have `stable`",
            ));
        }
        seen = true;
    }
    Ok(())
}

/// The attributes in the XSLT namespace a literal result element may carry —
/// [XSLT 3.0 §11.1](https://www.w3.org/TR/xslt-30/#literal-result-element).
const LITERAL_RESULT_ATTRIBUTES: &[&str] = &[
    "default-collation",
    "default-mode",
    "default-validation",
    "exclude-result-prefixes",
    "expand-text",
    "extension-element-prefixes",
    "inherit-namespaces",
    "type",
    "use-attribute-sets",
    "use-when",
    "validation",
    "version",
    "xpath-default-namespace",
];

/// Applies the rules that are about a literal result element rather than an XSLT one.
///
/// # Errors
///
/// `XTSE0805` for an attribute in the XSLT namespace that XSLT does not define there,
/// `XTSE0809` for `#default` in `exclude-result-prefixes` with no default namespace in scope.
pub(crate) fn check_literal_result_element(element: &NodeRef) -> XsltResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        if name.namespace() != Some(XSLT_NAMESPACE) {
            continue;
        }
        let local = name.local_name();
        let local = local.strip_prefix('_').unwrap_or(local);
        if !LITERAL_RESULT_ATTRIBUTES.contains(&local) {
            return Err(XsltError::stat1c(
                "XTSE0805",
                format!("`xsl:{local}` is not an attribute a literal result element may carry"),
            ));
        }
    }
    check_excluded_prefixes(element)
}

/// Rejects `#default` in `[xsl:]exclude-result-prefixes` where no default namespace is in
/// scope — [XSLT 3.0 §11.1.3](https://www.w3.org/TR/xslt-30/#literal-namespace-uri) raises
/// `XTSE0809`.
pub(crate) fn check_excluded_prefixes(element: &NodeRef) -> XsltResult<()> {
    let mentions_default = element.attributes().into_iter().any(|attribute| {
        let Some(name) = attribute.node_name() else {
            return false;
        };
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        owned
            && name.local_name() == "exclude-result-prefixes"
            && attribute
                .string_value()
                .split_whitespace()
                .any(|token| token == "#default")
    });
    if !mentions_default || has_default_namespace(element) {
        return Ok(());
    }
    Err(XsltError::stat1c(
        "XTSE0809",
        "`#default` needs a default namespace to exclude",
    ))
}

/// `(attribute, error code)` for the attributes that name namespace prefixes and whose
/// unbound prefixes have their own error codes in
/// [XSLT 3.0 Appendix G](https://www.w3.org/TR/xslt-30/#error-summary).
const PREFIX_LIST_ATTRIBUTES: &[(&str, &str)] = &[
    ("exclude-result-prefixes", "XTSE0808"),
    ("extension-element-prefixes", "XTSE1430"),
];

/// Rejects a prefix named by `exclude-result-prefixes` or `extension-element-prefixes` that
/// no namespace declaration in scope binds.
///
/// # Errors
///
/// `XTSE0808` for the first attribute, `XTSE1430` for the second.
pub(crate) fn check_prefix_lists(element: &NodeRef) -> XsltResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        if !owned {
            continue;
        }
        let Some((_, code)) = PREFIX_LIST_ATTRIBUTES
            .iter()
            .find(|(candidate, _)| *candidate == name.local_name())
        else {
            continue;
        };
        for token in attribute.string_value().split_whitespace() {
            if matches!(token, "#default" | "#all") || token == "xml" {
                continue;
            }
            if !is_prefix_in_scope(element, token) {
                return Err(XsltError::stat1c(
                    *code,
                    format!("no namespace is bound to the prefix `{token}`"),
                ));
            }
        }
    }
    Ok(())
}

/// Whether a namespace declaration in scope binds `prefix`.
fn is_prefix_in_scope(element: &NodeRef, prefix: &str) -> bool {
    let mut current = Some(element.clone());
    while let Some(node) = current {
        for namespace in node.namespaces() {
            if namespace
                .node_name()
                .is_some_and(|name| name.local_name() == prefix)
            {
                return !namespace.string_value().is_empty();
            }
        }
        current = node.parent();
    }
    false
}

/// Rejects the schema-awareness attributes a non-schema-aware processor may not honour.
///
/// # Errors
///
/// `XTSE1660`, which [XSLT 3.0 §27.1](https://www.w3.org/TR/xslt-30/#conformance) requires a
/// processor without schema support to raise for an `[xsl:]type` attribute, or for a
/// `[xsl:]validation` or `default-validation` attribute asking for anything but `strip`,
/// `preserve` or `lax`.
pub(crate) fn check_schema_awareness(element: &NodeRef, xslt_owned: bool) -> XsltResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        // On a literal result element only the `xsl:`-prefixed form is XSLT's; an unprefixed
        // `type` there is an attribute of the result element itself.
        let owned = match name.namespace() {
            Some(XSLT_NAMESPACE) => true,
            None => xslt_owned,
            _ => false,
        };
        if !owned {
            continue;
        }
        match name.local_name() {
            "type" => {
                return Err(XsltError::stat1c(
                    "XTSE1660",
                    "a `type` attribute needs a schema-aware processor",
                ));
            }
            "validation" | "default-validation" => {
                let value = attribute.string_value();
                if !matches!(value.trim(), "strip" | "preserve" | "lax") {
                    return Err(XsltError::stat1c(
                        "XTSE1660",
                        format!(
                            "`validation=\"{}\"` needs a schema-aware processor",
                            value.trim()
                        ),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether a non-empty default namespace is in scope on the element.
fn has_default_namespace(element: &NodeRef) -> bool {
    let mut current = Some(element.clone());
    while let Some(node) = current {
        for namespace in node.namespaces() {
            if namespace.node_name().is_none() {
                return !namespace.string_value().is_empty();
            }
        }
        current = node.parent();
    }
    false
}

/// Whether the element carries an unprefixed attribute of that name.
fn has_attribute(element: &NodeRef, wanted: &str) -> bool {
    element.attributes().into_iter().any(|attribute| {
        attribute
            .node_name()
            .is_some_and(|name| name.namespace().is_none() && name.local_name() == wanted)
    })
}

/// The value of an unprefixed attribute.
fn attribute_value(element: &NodeRef, wanted: &str) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        let name = attribute.node_name()?;
        (name.namespace().is_none() && name.local_name() == wanted)
            .then(|| attribute.string_value())
    })
}

/// The `name` attributes of the XSLT children of `element` with that local name.
fn child_names(element: &NodeRef, local: &str) -> Vec<String> {
    element
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .filter(|child| xslt_local_name(child) == Some(local))
        .filter_map(|child| {
            child.attributes().into_iter().find_map(|attribute| {
                let name = attribute.node_name()?;
                (name.namespace().is_none() && name.local_name() == "name")
                    .then(|| attribute.string_value().trim().to_owned())
            })
        })
        .collect()
}

fn check_duplicate_names(element: &NodeRef, local: &str, code: &str) -> XsltResult<()> {
    let names = child_names(element, local);
    for (index, name) in names.iter().enumerate() {
        if names[..index].contains(name) {
            return Err(XsltError::stat1c(
                code,
                format!("two `xsl:{local}` elements are both named `{name}`"),
            ));
        }
    }
    Ok(())
}

/// Rejects an `xsl:with-param` on `xsl:next-iteration` that names no parameter of the
/// `xsl:iterate` it belongs to — [XSLT 3.0 §8.8](https://www.w3.org/TR/xslt-30/#iterate)
/// raises `XTSE3130`.
fn check_next_iteration_params(element: &NodeRef) -> XsltResult<()> {
    let mut ancestor = element.parent();
    let iterate = loop {
        let Some(node) = ancestor else { return Ok(()) };
        if xslt_local_name(&node) == Some("iterate") {
            break node;
        }
        ancestor = node.parent();
    };
    let declared = child_names(&iterate, "param");
    for name in child_names(element, "with-param") {
        if !declared.contains(&name) {
            return Err(XsltError::stat1c(
                "XTSE3130",
                format!("`{name}` is not a parameter of the enclosing `xsl:iterate`"),
            ));
        }
    }
    Ok(())
}
