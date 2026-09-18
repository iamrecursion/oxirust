//! Static validation of the XSLT elements a stylesheet module contains.
//!
//! [XSLT 3.0 §3.7](https://www.w3.org/TR/xslt-30/#element-syntax-summary) fixes, for every
//! element in the XSLT namespace, which attributes it may carry and which of those are
//! required; [§3.5](https://www.w3.org/TR/xslt-30/#attribute-values) fixes the value space of
//! the attributes whose values come from a closed set. Violating the first is `XTSE0090` or
//! `XTSE0010`, violating the second is `XTSE0020`.
//!
//! The checks live apart from the compiler proper because they are exhaustive over the
//! vocabulary rather than driven by what the engine happens to need.

use crate::compile::{is_stylesheet_element, xslt_local_name};
use crate::content::{BreakPosition, Model, child_break_position, is_extension_element, model_of};
use crate::error::{XsltError, XsltResult};
use crate::names::XSLT_NAMESPACE;
use oxixml_infoset::{NodeKind, NodeRef, XdmNode};
use oxixml_qname::XML_NAMESPACE;

/// The attributes any element in the XSLT namespace may carry — the *standard attributes* of
/// [XSLT 3.0 §3.4](https://www.w3.org/TR/xslt-30/#standard-attributes).
const STANDARD_ATTRIBUTES: &[&str] = &[
    "default-collation",
    "default-mode",
    "default-validation",
    "exclude-result-prefixes",
    "expand-text",
    "extension-element-prefixes",
    "use-when",
    "version",
    "xpath-default-namespace",
];

/// `(element, required attributes, optional attributes)` for the whole vocabulary.
const ELEMENTS: &[(&str, &[&str], &[&str])] = &[
    ("analyze-string", &["select", "regex"], &["flags"]),
    ("apply-imports", &[], &[]),
    ("apply-templates", &[], &["select", "mode"]),
    ("assert", &["test"], &["select", "error-code"]),
    (
        "attribute",
        &["name"],
        &["namespace", "select", "separator", "type", "validation"],
    ),
    (
        "attribute-set",
        &["name"],
        &["use-attribute-sets", "visibility", "streamable"],
    ),
    ("break", &[], &["select"]),
    ("call-template", &["name"], &[]),
    ("catch", &[], &["errors", "select"]),
    ("character-map", &["name"], &["use-character-maps"]),
    ("choose", &[], &[]),
    ("comment", &[], &["select"]),
    ("context-item", &[], &["as", "use"]),
    (
        "copy",
        &[],
        &[
            "select",
            "copy-namespaces",
            "inherit-namespaces",
            "use-attribute-sets",
            "type",
            "validation",
            "on-empty",
        ],
    ),
    (
        "copy-of",
        &["select"],
        &["copy-namespaces", "copy-accumulators", "type", "validation"],
    ),
    (
        "decimal-format",
        &[],
        &[
            "name",
            "decimal-separator",
            "grouping-separator",
            "infinity",
            "minus-sign",
            "exponent-separator",
            "NaN",
            "percent",
            "per-mille",
            "zero-digit",
            "digit",
            "pattern-separator",
        ],
    ),
    ("document", &[], &["validation", "type"]),
    (
        "element",
        &["name"],
        &[
            "namespace",
            "inherit-namespaces",
            "use-attribute-sets",
            "type",
            "validation",
            "on-empty",
        ],
    ),
    ("fallback", &[], &[]),
    ("for-each", &["select"], &[]),
    (
        "for-each-group",
        &["select"],
        &[
            "group-by",
            "group-adjacent",
            "group-starting-with",
            "group-ending-with",
            "composite",
            "collation",
        ],
    ),
    (
        "function",
        &["name"],
        &[
            "as",
            "override",
            "override-extension-function",
            "visibility",
            "streamability",
            "identity-sensitive",
            "new-each-time",
            "cache",
        ],
    ),
    ("global-context-item", &[], &["as", "use"]),
    ("if", &["test"], &[]),
    ("import", &["href"], &[]),
    ("import-schema", &[], &["namespace", "schema-location"]),
    ("include", &["href"], &[]),
    ("iterate", &["select"], &[]),
    (
        "key",
        &["name", "match"],
        &["use", "composite", "collation"],
    ),
    ("map", &[], &[]),
    ("map-entry", &["key"], &["select"]),
    ("matching-substring", &[], &[]),
    ("message", &[], &["select", "terminate", "error-code"]),
    (
        "mode",
        &[],
        &[
            "name",
            "streamable",
            "use-accumulators",
            "on-no-match",
            "on-multiple-match",
            "warning-on-no-match",
            "warning-on-multiple-match",
            "typed",
            "visibility",
        ],
    ),
    ("namespace", &["name"], &["select"]),
    (
        "namespace-alias",
        &["stylesheet-prefix", "result-prefix"],
        &[],
    ),
    ("next-iteration", &[], &[]),
    ("next-match", &[], &[]),
    ("non-matching-substring", &[], &[]),
    (
        "number",
        &[],
        &[
            "value",
            "select",
            "level",
            "count",
            "from",
            "format",
            "lang",
            "letter-value",
            "ordinal",
            "start-at",
            "grouping-separator",
            "grouping-size",
        ],
    ),
    ("on-completion", &[], &["select"]),
    ("on-empty", &[], &["select"]),
    ("on-non-empty", &[], &["select"]),
    ("otherwise", &[], &[]),
    ("output", &[], SERIALIZATION_ATTRIBUTES),
    ("output-character", &["character", "string"], &[]),
    (
        "param",
        &["name"],
        &["select", "as", "required", "tunnel", "static"],
    ),
    ("perform-sort", &[], &["select"]),
    ("preserve-space", &["elements"], &[]),
    ("processing-instruction", &["name"], &["select"]),
    ("result-document", &[], RESULT_DOCUMENT_ATTRIBUTES),
    // XSLT 2.0 let `xsl:sequence` carry an `as` attribute; XSLT 3.0 §5.7.3 withdrew it, leaving
    // the type check to the construct the sequence is written into (function-1018).
    ("sequence", &[], &["select"]),
    (
        "sort",
        &[],
        &[
            "select",
            "lang",
            "order",
            "collation",
            "stable",
            "case-order",
            "data-type",
        ],
    ),
    ("strip-space", &["elements"], &[]),
    ("stylesheet", &[], &["id", "input-type-annotations"]),
    (
        "template",
        &[],
        &["match", "name", "priority", "mode", "as", "visibility"],
    ),
    ("text", &[], &["disable-output-escaping"]),
    ("transform", &[], &["id", "input-type-annotations"]),
    ("try", &[], &["select", "rollback-output"]),
    (
        "value-of",
        &[],
        &["select", "separator", "disable-output-escaping"],
    ),
    (
        "variable",
        &["name"],
        &["select", "as", "static", "visibility"],
    ),
    ("when", &["test"], &[]),
    ("where-populated", &[], &[]),
    ("with-param", &["name"], &["select", "as", "tunnel"]),
];

/// The serialization parameters `xsl:output` accepts.
const SERIALIZATION_ATTRIBUTES: &[&str] = &[
    "name",
    "method",
    "allow-duplicate-names",
    "build-tree",
    "byte-order-mark",
    "cdata-section-elements",
    "doctype-public",
    "doctype-system",
    "encoding",
    "escape-uri-attributes",
    "html-version",
    "include-content-type",
    "indent",
    "item-separator",
    "json-node-output-method",
    "media-type",
    "normalization-form",
    "omit-xml-declaration",
    "parameter-document",
    "standalone",
    "suppress-indentation",
    "undeclare-prefixes",
    "use-character-maps",
];

/// `xsl:result-document` takes the serialization parameters plus its own four.
const RESULT_DOCUMENT_ATTRIBUTES: &[&str] = &[
    "format",
    "href",
    "validation",
    "type",
    "method",
    "allow-duplicate-names",
    "build-tree",
    "byte-order-mark",
    "cdata-section-elements",
    "doctype-public",
    "doctype-system",
    "encoding",
    "escape-uri-attributes",
    "html-version",
    "include-content-type",
    "indent",
    "item-separator",
    "json-node-output-method",
    "media-type",
    "normalization-form",
    "omit-xml-declaration",
    "parameter-document",
    "standalone",
    "suppress-indentation",
    "undeclare-prefixes",
    "use-character-maps",
];

/// `(attribute, permitted values)` for the attributes whose value space is a closed set and
/// which are not attribute value templates.
const ENUMERATIONS: &[(&str, &[&str])] = &[
    ("cache", YES_NO),
    ("composite", YES_NO),
    ("copy-accumulators", YES_NO),
    ("copy-namespaces", YES_NO),
    ("disable-output-escaping", YES_NO),
    ("identity-sensitive", YES_NO),
    ("inherit-namespaces", YES_NO),
    (
        "input-type-annotations",
        &["preserve", "strip", "unspecified"],
    ),
    ("level", &["single", "multiple", "any"]),
    (
        "new-each-time",
        &["yes", "no", "true", "false", "1", "0", "maybe"],
    ),
    (
        "on-no-match",
        &[
            "deep-copy",
            "shallow-copy",
            "deep-skip",
            "shallow-skip",
            "text-only-copy",
            "fail",
        ],
    ),
    ("on-multiple-match", &["use-last", "fail"]),
    ("override", YES_NO),
    ("override-extension-function", YES_NO),
    ("required", YES_NO),
    ("rollback-output", YES_NO),
    ("static", YES_NO),
    ("streamable", YES_NO),
    ("terminate", YES_NO),
    ("tunnel", YES_NO),
    (
        "typed",
        &["yes", "no", "true", "false", "1", "0", "unspecified"],
    ),
    ("validation", &["strict", "lax", "preserve", "strip"]),
    (
        "visibility",
        &["public", "private", "final", "abstract", "hidden"],
    ),
    ("warning-on-no-match", YES_NO),
    ("warning-on-multiple-match", YES_NO),
];

/// `(element, attribute, permitted values)` for the closed-set attributes whose name is also
/// used, on another element, for something that is not a closed set.
const SCOPED_ENUMERATIONS: &[(&str, &str, &[&str])] = &[
    ("context-item", "use", &["required", "optional", "absent"]),
    (
        "global-context-item",
        "use",
        &["required", "optional", "absent"],
    ),
];

/// The lexical space of the `xs:boolean`-valued XSLT attributes.
const YES_NO: &[&str] = &["yes", "no", "true", "false", "1", "0"];

/// Validates every XSLT element in the subtree rooted at `element`.
///
/// # Errors
///
/// `XTSE0090` for an attribute the element may not carry, `XTSE0010` for a required attribute
/// that is missing, `XTSE0020` for a value outside a closed set.
pub(crate) fn validate_module(element: &NodeRef) -> XsltResult<()> {
    // A stylesheet written for a later version than this processor implements is read in
    // forwards-compatible mode, where an unrecognized attribute is ignored rather than
    // rejected — so nothing is checked there.
    if declared_version(element).is_some_and(|stated| stated > 3.0) {
        return Ok(());
    }
    // Every stylesheet module must state the version it is written for: `version` on
    // `xsl:stylesheet` and `xsl:transform` (XSLT 3.0 §3.6), `xsl:version` on the literal
    // result element of a simplified stylesheet (§3.7).
    check_version_syntax(element)?;
    if is_stylesheet_element(element) {
        if declared_version(element).is_none() {
            return Err(XsltError::stat1c(
                "XTSE0010",
                "a stylesheet module requires a `version` attribute",
            ));
        }
    } else if xslt_local_name(element).is_none()
        && crate::compile::xslt_attribute(element, "version").is_none()
    {
        return Err(XsltError::stat1c(
            "XTSE0150",
            "a simplified stylesheet requires an `xsl:version` attribute",
        ));
    }
    // The passes are separate so that a stylesheet with both a content-model fault and a
    // misplaced `xsl:break` is reported under the content-model code, which is the one the
    // spec's error catalogue lists first for such a module.
    walk(element, &mut Vec::new())?;
    walk_breaks(element, BreakPosition::Outside)
}

/// Checks where every `xsl:break` and `xsl:next-iteration` stands.
///
/// # Errors
///
/// `XTSE0010` outside any `xsl:iterate`, `XTSE3120` in the wrong position within one.
fn walk_breaks(element: &NodeRef, position: BreakPosition) -> XsltResult<()> {
    let local = xslt_local_name(element);
    if let Some(local) = local
        && matches!(local, "break" | "next-iteration")
    {
        match position {
            BreakPosition::Outside => {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    format!("`xsl:{local}` must be lexically within an `xsl:iterate`"),
                ));
            }
            BreakPosition::Inner => {
                return Err(XsltError::stat1c(
                    "XTSE3120",
                    format!("`xsl:{local}` must end the body of its `xsl:iterate`"),
                ));
            }
            BreakPosition::Tail => {}
        }
    }
    // `xsl:fallback` is not part of the sequence constructor of an instruction the processor
    // implements, so it does not displace the instruction that ends an iteration.
    let children: Vec<NodeRef> = element
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .filter(|child| xslt_local_name(child) != Some("fallback"))
        .collect();
    // Every `xsl:catch` of an `xsl:try` ends the instruction just as its body does, so each
    // of them counts as the last child for this purpose.
    let last = children
        .iter()
        .rposition(|child| xslt_local_name(child) != Some("catch"))
        .unwrap_or(children.len().saturating_sub(1));
    for (index, child) in children.iter().enumerate() {
        let is_last =
            index == last || (local == Some("try") && xslt_local_name(child) == Some("catch"));
        walk_breaks(child, child_break_position(local, position, is_last))?;
    }
    Ok(())
}

/// Rejects a `version` attribute that is not an `xs:decimal` —
/// [XSLT 3.0 §3.6](https://www.w3.org/TR/xslt-30/#stylesheet-element) raises `XTSE0110`.
fn check_version_syntax(element: &NodeRef) -> XsltResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        if !owned || name.local_name() != "version" {
            continue;
        }
        let value = attribute.string_value();
        let value = value.trim();
        let digits = value.strip_prefix(['+', '-']).unwrap_or(value);
        let decimal = !digits.is_empty()
            && digits.chars().all(|c| c.is_ascii_digit() || c == '.')
            && digits.matches('.').count() <= 1
            && digits.chars().any(|c| c.is_ascii_digit());
        if !decimal {
            return Err(XsltError::stat1c(
                "XTSE0110",
                format!("`{value}` is not a decimal number"),
            ));
        }
    }
    Ok(())
}

/// The `version` (or `xsl:version`) attribute of the module's outermost element, as a number.
fn declared_version(element: &NodeRef) -> Option<f64> {
    element.attributes().into_iter().find_map(|attribute| {
        let name = attribute.node_name()?;
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        (owned && name.local_name() == "version")
            .then(|| attribute.string_value().trim().parse::<f64>().ok())
            .flatten()
    })
}

fn walk(element: &NodeRef, extensions: &mut Vec<String>) -> XsltResult<()> {
    // A subtree written for a later version is read in forwards-compatible mode, where an
    // element XSLT 3.0 does not define is left to the fallback machinery rather than rejected.
    if declared_version(element).is_some_and(|stated| stated > 3.0) {
        return Ok(());
    }
    let added = collect_extension_namespaces(element, extensions);
    let outcome = walk_checked(element, extensions);
    extensions.truncate(extensions.len() - added);
    outcome
}

fn walk_checked(element: &NodeRef, extensions: &mut Vec<String>) -> XsltResult<()> {
    if is_extension_element(element, extensions) {
        return Ok(());
    }
    let local = xslt_local_name(element);
    let model = match local {
        Some(local) => {
            crate::content::check_schema_awareness(element, true)?;
            validate_element(element, local)?;
            crate::content::check(element, local, model_of(local))?;
            crate::content::check_static_rules(element, local)?;
            crate::content::check_excluded_prefixes(element)?;
            crate::content::check_prefix_lists(element)?;
            model_of(local)
        }
        None => {
            // A literal result element's content is a sequence constructor, exactly like an
            // instruction's.
            crate::content::check(element, "literal result element", Model::Sequence)?;
            crate::content::check_literal_result_element(element)?;
            crate::content::check_prefix_lists(element)?;
            crate::content::check_schema_awareness(element, false)?;
            Model::Sequence
        }
    };
    if matches!(model, Model::Unchecked) {
        return Ok(());
    }
    check_on_empty_is_last(element)?;
    for child in element.children() {
        if child.node_kind() == NodeKind::Element {
            walk(&child, extensions)?;
        }
    }
    Ok(())
}

/// `xsl:on-empty` supplies what to write when the *whole* containing sequence constructor turns
/// out to be empty, so it can only be judged once everything else has been evaluated:
/// [XSLT 3.0 §17.2](https://www.w3.org/TR/xslt-30/#on-empty) requires it to come last, and makes
/// a sibling after it a static error. The suite pins each shape the rule catches — an
/// instruction after it (on-empty-003, on-empty-004), significant text after it (on-empty-007),
/// and an `xsl:on-non-empty` after it (on-non-empty-013), whose own description is "on-empty
/// must come last".
///
/// `xsl:fallback` is exempt for the same reason it is exempt from
/// [`walk_breaks`](fn@walk_breaks): it is never part of the sequence constructor of an
/// instruction this processor implements.
///
/// # Errors
///
/// `XTSE0010`.
fn check_on_empty_is_last(element: &NodeRef) -> XsltResult<()> {
    let mut seen = false;
    for child in element.children() {
        match child.node_kind() {
            NodeKind::Element => {
                if seen && xslt_local_name(&child) != Some("fallback") {
                    return Err(XsltError::stat1c(
                        "XTSE0010",
                        "`xsl:on-empty` must be the last thing in its sequence constructor",
                    ));
                }
                seen |= xslt_local_name(&child) == Some("on-empty");
            }
            NodeKind::Text if seen && !crate::compile::is_xml_whitespace(&child.string_value()) => {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    "`xsl:on-empty` must be the last thing in its sequence constructor, and text \
                     that is not whitespace follows it",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Pushes the namespaces `extension-element-prefixes` names onto `extensions`, returning how
/// many were added so the caller can pop them again.
fn collect_extension_namespaces(element: &NodeRef, extensions: &mut Vec<String>) -> usize {
    let declared: Vec<(Option<String>, String)> = element
        .namespaces()
        .into_iter()
        .map(|namespace| {
            (
                namespace
                    .node_name()
                    .map(|name| name.local_name().to_owned()),
                namespace.string_value(),
            )
        })
        .collect();
    let mut added = 0;
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        let owned = matches!(name.namespace(), None | Some(XSLT_NAMESPACE));
        if !owned || name.local_name() != "extension-element-prefixes" {
            continue;
        }
        for token in attribute.string_value().split_whitespace() {
            let wanted = (token != "#default").then_some(token);
            if let Some((_, uri)) = declared
                .iter()
                .find(|(prefix, _)| prefix.as_deref() == wanted)
            {
                extensions.push(uri.clone());
                added += 1;
            }
        }
    }
    added
}

fn validate_element(element: &NodeRef, local: &str) -> XsltResult<()> {
    let Some((_, required, optional)) = ELEMENTS.iter().find(|(name, _, _)| *name == local) else {
        return Ok(());
    };
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        // Only unprefixed attributes belong to XSLT's own vocabulary; one in a namespace is
        // always allowed, and `xml:` attributes are allowed everywhere.
        if name.namespace().is_some() {
            continue;
        }
        let attribute_name = name.local_name();
        // A *shadow attribute* — `_indent="{$x}"` — supplies the value of the attribute whose
        // name follows the underscore, and is legal wherever that attribute is.
        let shadow = attribute_name.starts_with('_');
        let attribute_name = attribute_name.strip_prefix('_').unwrap_or(attribute_name);
        if !required.contains(&attribute_name)
            && !optional.contains(&attribute_name)
            && !STANDARD_ATTRIBUTES.contains(&attribute_name)
        {
            return Err(XsltError::stat1c(
                "XTSE0090",
                format!("`xsl:{local}` has no attribute named `{attribute_name}`"),
            ));
        }
        let allowed = ENUMERATIONS
            .iter()
            .find(|(candidate, _)| *candidate == attribute_name)
            .map(|(_, values)| *values)
            .or_else(|| {
                SCOPED_ENUMERATIONS
                    .iter()
                    .find(|(owner, candidate, _)| *owner == local && *candidate == attribute_name)
                    .map(|(_, _, values)| *values)
            });
        if let Some(allowed) = allowed.filter(|_| !shadow) {
            let value = attribute.string_value();
            let trimmed = value.trim();
            // A shadow attribute or an attribute value template carries an expression, not a
            // value from the closed set; only the computed result can be judged.
            if trimmed.contains('{') {
                continue;
            }
            if !allowed.contains(&trimmed) {
                return Err(XsltError::stat1c(
                    "XTSE0020",
                    format!(
                        "`{trimmed}` is not a value for the `{attribute_name}` attribute of `xsl:{local}`"
                    ),
                ));
            }
        }
    }
    // A constructed name may not be put into a namespace XSLT reserves. Only a literal
    // `namespace` attribute can be judged statically; an attribute value template is left to
    // the run-time check.
    if matches!(local, "element" | "attribute" | "namespace") {
        for attribute in element.attributes() {
            let Some(name) = attribute.node_name() else {
                continue;
            };
            if name.namespace().is_some() || name.local_name() != "namespace" {
                continue;
            }
            let value = attribute.string_value();
            if !value.contains('{') && is_reserved_namespace(value.trim()) {
                return Err(XsltError::stat1c(
                    "XTSE0080",
                    format!("`{value}` is a namespace a stylesheet may not construct a name in"),
                ));
            }
        }
    }
    if local == "function" {
        check_override_agreement(element)?;
    }
    if local == "param" {
        check_function_parameter(element)?;
    }
    for name in required.iter() {
        let present = element.attributes().into_iter().any(|attribute| {
            attribute.node_name().is_some_and(|found| {
                found.namespace().is_none() && found.local_name().trim_start_matches('_') == *name
            })
        });
        if !present {
            return Err(XsltError::stat1c(
                "XTSE0010",
                format!("`xsl:{local}` requires an attribute named `{name}`"),
            ));
        }
    }
    Ok(())
}

/// The value of the unprefixed XSLT attribute `name`, if the element carries it literally — a
/// shadow attribute or an attribute value template holds an expression, not a value, and only the
/// computed result can be judged.
fn literal_attribute(element: &NodeRef, name: &str) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        let found = attribute.node_name()?;
        (found.namespace().is_none() && found.local_name() == name)
            .then(|| attribute.string_value())
            .filter(|value| !value.contains('{'))
    })
}

/// The truth value of a boolean XSLT attribute — `yes`, `true` and `1` against `no`, `false`
/// and `0` ([XSLT 3.0 §3.5](https://www.w3.org/TR/xslt-30/#attribute-values)).
pub(crate) fn boolean_value(value: &str) -> Option<bool> {
    match value.trim() {
        "yes" | "true" | "1" => Some(true),
        "no" | "false" | "0" => Some(false),
        _ => None,
    }
}

/// `xsl:function` may carry the XSLT 2.0 spelling `override` beside the 3.0
/// `override-extension-function`, of which it is a synonym — but only saying the same thing
/// twice. [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions) keeps the older
/// name "for backwards compatibility" and the suite's function-0116 pins that `yes` beside `true`
/// is accepted, while function-0117 pins that `1` beside `no` is not.
///
/// # Errors
///
/// `XTSE0020` when both attributes are present with opposite values.
fn check_override_agreement(element: &NodeRef) -> XsltResult<()> {
    let Some(new_name) = literal_attribute(element, "override-extension-function") else {
        return Ok(());
    };
    let Some(old_name) = literal_attribute(element, "override") else {
        return Ok(());
    };
    let (Some(new_value), Some(old_value)) = (boolean_value(&new_name), boolean_value(&old_name))
    else {
        // A value outside the boolean vocabulary is rejected by the enumeration check instead.
        return Ok(());
    };
    if new_value != old_value {
        return Err(XsltError::stat1c(
            "XTSE0020",
            "`override` is a synonym for `override-extension-function`, so `xsl:function` may \
             not carry both with opposite values",
        ));
    }
    Ok(())
}

/// The parameters of a stylesheet function are positional arguments the call always supplies, so
/// [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions) leaves an `xsl:param`
/// within `xsl:function` no room to say otherwise: it may not be optional and it may not be
/// tunnelled. Only the values that *contradict* that are rejected — the suite pins each of the
/// three readings, `required="yes"` merely restating what already holds (function-0119b),
/// `required="no"` denying it (function-0120b) and `tunnel="yes"` asking for a channel a call
/// does not have (error-0020j).
///
/// # Errors
///
/// `XTSE0020`.
fn check_function_parameter(element: &NodeRef) -> XsltResult<()> {
    let Some(parent) = element.parent() else {
        return Ok(());
    };
    if xslt_local_name(&parent) != Some("function") {
        return Ok(());
    }
    if literal_attribute(element, "required")
        .and_then(|value| boolean_value(&value))
        .is_some_and(|required| !required)
    {
        return Err(XsltError::stat1c(
            "XTSE0020",
            "an `xsl:param` of an `xsl:function` may not specify `required=\"no\"`: every \
             argument of a stylesheet function is supplied by the call",
        ));
    }
    if literal_attribute(element, "tunnel")
        .and_then(|value| boolean_value(&value))
        .is_some_and(|tunnel| tunnel)
    {
        return Err(XsltError::stat1c(
            "XTSE0020",
            "an `xsl:param` of an `xsl:function` may not specify `tunnel=\"yes\"`: a function \
             call supplies its arguments positionally, with no tunnel channel",
        ));
    }
    Ok(())
}

/// Whether `namespace` is one a stylesheet may not put a constructed name in — the check
/// [XSLT 3.0 §5.6](https://www.w3.org/TR/xslt-30/#reserved-namespaces) makes `XTSE0080`.
#[must_use]
pub(crate) fn is_reserved_namespace(namespace: &str) -> bool {
    matches!(namespace, XSLT_NAMESPACE | XML_NAMESPACE)
        || namespace == "http://www.w3.org/2000/xmlns/"
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    fn check(source: &str) -> XsltResult<()> {
        let tree = Tree::parse_str(source)?;
        let element = tree.document_element().expect("an element");
        validate_module(&element)
    }

    #[test]
    fn an_unknown_attribute_is_rejected() {
        let error = check(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 <xsl:template match="a" nonesuch="1"/>
               </xsl:stylesheet>"#,
        )
        .expect_err("the attribute is not allowed");
        assert_eq!(error.code(), Some("XTSE0090"));
    }

    #[test]
    fn a_missing_required_attribute_is_rejected() {
        let error = check(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 <xsl:template match="a"><xsl:if>x</xsl:if></xsl:template>
               </xsl:stylesheet>"#,
        )
        .expect_err("test is required");
        assert_eq!(error.code(), Some("XTSE0010"));
    }

    #[test]
    fn a_value_outside_a_closed_set_is_rejected() {
        let error = check(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
                 <xsl:template match="a"><xsl:copy-of select="." copy-namespaces="maybe"/>
                 </xsl:template>
               </xsl:stylesheet>"#,
        )
        .expect_err("copy-namespaces is yes or no");
        assert_eq!(error.code(), Some("XTSE0020"));
    }

    #[test]
    fn a_namespaced_attribute_is_left_alone() -> XsltResult<()> {
        check(
            r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                               xmlns:x="urn:x" version="3.0">
                 <xsl:template match="a" x:note="hi"/>
               </xsl:stylesheet>"#,
        )
    }

    #[test]
    fn the_reserved_namespaces_are_known() {
        assert!(is_reserved_namespace(XSLT_NAMESPACE));
        assert!(is_reserved_namespace(XML_NAMESPACE));
        assert!(!is_reserved_namespace("urn:x"));
    }
}
