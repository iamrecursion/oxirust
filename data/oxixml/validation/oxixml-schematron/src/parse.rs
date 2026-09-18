//! Parses Schematron schema XML into the raw `model` object model.
//!
//! `sch:include` is resolved eagerly, by substituting the referenced element in place of the
//! `include` element wherever it occurs (schema, pattern, rule and phase content models all
//! allow it) — see [`IncludeResolver`].

use std::fmt;

use oxixml_infoset::{NodeKind, NodeRef, Tree, XdmNode};

use crate::error::{SchematronError, SchematronResult};
use crate::model::{
    CheckDecl, CheckKind, DiagnosticDecl, LetDecl, MessagePart, NsBinding, ParamDecl, PatternDecl,
    PhaseDecl, RawSchema, RuleDecl, RuleItem, SCHEMATRON_1_5_NS, SCHEMATRON_NS,
};

/// The greatest depth of nested `sch:include` this parser will follow before assuming a cycle.
const MAX_INCLUDE_DEPTH: usize = 64;

/// Fetches the text an `sch:include href="…"` (or `sch:extends href="…"`) points to.
///
/// A Pure-Rust library never reads the filesystem or the network behind its caller's back —
/// nothing is resolvable until the application supplies one of these, exactly as
/// [`oxixml_xpath::DocumentResolver`] gates `fn:doc`.
///
/// # Examples
///
/// ```
/// use oxixml_schematron::parse::IncludeResolver;
/// use oxixml_schematron::SchematronError;
/// use std::collections::HashMap;
///
/// #[derive(Debug)]
/// struct MapResolver(HashMap<String, String>);
///
/// impl IncludeResolver for MapResolver {
///     fn resolve(&self, href: &str) -> Result<String, SchematronError> {
///         self.0
///             .get(href)
///             .cloned()
///             .ok_or_else(|| SchematronError::UnresolvedInclude { href: href.to_owned() })
///     }
/// }
/// ```
pub trait IncludeResolver: fmt::Debug {
    /// Returns the XML text at `href`.
    ///
    /// # Errors
    ///
    /// Any [`SchematronError`]; typically [`SchematronError::UnresolvedInclude`] when `href`
    /// names nothing this resolver knows about.
    fn resolve(&self, href: &str) -> SchematronResult<String>;
}

/// Parses `xml` as a `sch:schema` document, resolving `sch:include` through `resolver`.
///
/// # Errors
///
/// [`SchematronError::Xml`] if `xml` is not well-formed, [`SchematronError::NotASchema`] if its
/// document element is not `schema`, or any of the structural errors raised while walking the
/// content model.
pub fn parse_schema(
    xml: &str,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<RawSchema> {
    let tree = Tree::parse_str(xml)?;
    let root = tree
        .document_element()
        .ok_or_else(|| SchematronError::MissingElement {
            what: "schema",
            context: "document".to_owned(),
        })?;
    if !is_sch(&root, "schema") {
        return Err(SchematronError::NotASchema {
            found: describe(&root),
        });
    }
    parse_schema_element(&root, resolver)
}

/// Whether `node` is a Schematron element named `local`: unprefixed schemas (no default
/// namespace declared, seen in older 1.5-era documents) and both the ISO and 1.5 namespaces are
/// all accepted.
fn is_sch(node: &NodeRef, local: &str) -> bool {
    node.node_kind() == NodeKind::Element
        && node.node_name().is_some_and(|name| {
            name.local_name() == local
                && matches!(
                    name.namespace(),
                    None | Some(SCHEMATRON_NS) | Some(SCHEMATRON_1_5_NS)
                )
        })
}

fn describe(node: &NodeRef) -> String {
    match node.node_name() {
        Some(name) => name.to_string(),
        None => format!("<{:?}>", node.node_kind()),
    }
}

/// The value of unprefixed attribute `local` on `node`, if it has one.
fn attr(node: &NodeRef, local: &str) -> Option<String> {
    node.attributes().into_iter().find_map(|a| {
        let name = a.node_name()?;
        (name.local_name() == local && name.namespace().is_none()).then(|| a.string_value())
    })
}

fn require_attr(node: &NodeRef, local: &'static str, context: &str) -> SchematronResult<String> {
    attr(node, local).ok_or_else(|| SchematronError::MissingAttribute {
        name: local,
        context: context.to_owned(),
    })
}

/// Resolves `node` if it is a `sch:include`, recursively, up to [`MAX_INCLUDE_DEPTH`].
fn resolve_include(
    node: NodeRef,
    resolver: Option<&dyn IncludeResolver>,
    depth: usize,
) -> SchematronResult<NodeRef> {
    if !is_sch(&node, "include") {
        return Ok(node);
    }
    if depth >= MAX_INCLUDE_DEPTH {
        let href = attr(&node, "href").unwrap_or_default();
        return Err(SchematronError::IncludeCycle { href });
    }
    let href = require_attr(&node, "href", "include")?;
    let resolver =
        resolver.ok_or_else(|| SchematronError::UnresolvedInclude { href: href.clone() })?;
    let fetched = resolver.resolve(&href)?;
    let fetched_tree = Tree::parse_str(&fetched)?;
    let fetched_root =
        fetched_tree
            .document_element()
            .ok_or_else(|| SchematronError::MissingElement {
                what: "element",
                context: format!("include href=\"{href}\""),
            })?;
    resolve_include(fetched_root, Some(resolver), depth + 1)
}

/// `node`'s element children, in document order, with every `sch:include` substituted by the
/// element it resolves to.
fn resolved_element_children(
    node: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<Vec<NodeRef>> {
    node.children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .map(|child| resolve_include(child, resolver, 0))
        .collect()
}

fn parse_schema_element(
    root: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<RawSchema> {
    let mut schema = RawSchema {
        title: None,
        schema_version: attr(root, "schemaVersion"),
        default_phase: attr(root, "defaultPhase"),
        query_binding: attr(root, "queryBinding"),
        namespaces: Vec::new(),
        lets: Vec::new(),
        phases: Vec::new(),
        patterns: Vec::new(),
        diagnostics: Vec::new(),
    };

    for child in resolved_element_children(root, resolver)? {
        if is_sch(&child, "title") {
            schema.title = Some(child.string_value());
        } else if is_sch(&child, "ns") {
            schema.namespaces.push(parse_ns(&child)?);
        } else if is_sch(&child, "let") {
            schema.lets.push(parse_let(&child)?);
        } else if is_sch(&child, "phase") {
            schema.phases.push(parse_phase(&child, resolver)?);
        } else if is_sch(&child, "pattern") {
            schema.patterns.push(parse_pattern(&child, resolver)?);
        } else if is_sch(&child, "diagnostics") {
            for diag in resolved_element_children(&child, resolver)? {
                if is_sch(&diag, "diagnostic") {
                    schema.diagnostics.push(parse_diagnostic(&diag)?);
                }
            }
        }
        // `sch:p`, `sch:title` (already handled) and any other documentation/foreign markup
        // carry no validation semantics and are silently skipped.
    }
    Ok(schema)
}

fn parse_ns(node: &NodeRef) -> SchematronResult<NsBinding> {
    Ok(NsBinding {
        prefix: require_attr(node, "prefix", "ns")?,
        uri: require_attr(node, "uri", "ns")?,
    })
}

fn parse_let(node: &NodeRef) -> SchematronResult<LetDecl> {
    let name = require_attr(node, "name", "let")?;
    let value = match attr(node, "value") {
        Some(value) => value,
        None => {
            let text = node.string_value();
            if text.trim().is_empty() {
                return Err(SchematronError::MissingAttribute {
                    name: "value",
                    context: "let".to_owned(),
                });
            }
            text
        }
    };
    Ok(LetDecl { name, value })
}

fn parse_param(node: &NodeRef) -> SchematronResult<ParamDecl> {
    Ok(ParamDecl {
        name: require_attr(node, "name", "param")?,
        value: require_attr(node, "value", "param")?,
    })
}

fn parse_phase(
    node: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<PhaseDecl> {
    let id = require_attr(node, "id", "phase")?;
    let mut active = Vec::new();
    let mut lets = Vec::new();
    for child in resolved_element_children(node, resolver)? {
        if is_sch(&child, "active") {
            active.push(require_attr(&child, "pattern", "active")?);
        } else if is_sch(&child, "let") {
            lets.push(parse_let(&child)?);
        }
    }
    Ok(PhaseDecl { id, active, lets })
}

fn parse_pattern(
    node: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<PatternDecl> {
    let id = attr(node, "id");
    let is_abstract = attr(node, "abstract").as_deref() == Some("true");
    let is_a = attr(node, "is-a");
    let mut title = None;
    let mut lets = Vec::new();
    let mut rules = Vec::new();
    let mut params = Vec::new();
    for child in resolved_element_children(node, resolver)? {
        if is_sch(&child, "title") {
            title = Some(child.string_value());
        } else if is_sch(&child, "let") {
            lets.push(parse_let(&child)?);
        } else if is_sch(&child, "rule") {
            rules.push(parse_rule(&child, resolver)?);
        } else if is_sch(&child, "param") {
            params.push(parse_param(&child)?);
        }
    }
    Ok(PatternDecl {
        id,
        is_abstract,
        is_a,
        title,
        lets,
        rules,
        params,
    })
}

fn parse_rule(
    node: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<RuleDecl> {
    let id = attr(node, "id");
    let context = attr(node, "context");
    let is_abstract = attr(node, "abstract").as_deref() == Some("true");
    let role = attr(node, "role");
    let flag = attr(node, "flag");
    let mut body = Vec::new();
    for child in resolved_element_children(node, resolver)? {
        if is_sch(&child, "let") {
            body.push(RuleItem::Let(parse_let(&child)?));
        } else if is_sch(&child, "assert") {
            body.push(RuleItem::Check(parse_check(&child, CheckKind::Assert)?));
        } else if is_sch(&child, "report") {
            body.push(RuleItem::Check(parse_check(&child, CheckKind::Report)?));
        } else if is_sch(&child, "extends") {
            body.push(parse_extends(&child, resolver)?);
        }
    }
    Ok(RuleDecl {
        id,
        context,
        is_abstract,
        role,
        flag,
        body,
    })
}

fn parse_extends(
    node: &NodeRef,
    resolver: Option<&dyn IncludeResolver>,
) -> SchematronResult<RuleItem> {
    if let Some(rule_id) = attr(node, "rule") {
        return Ok(RuleItem::Extends(rule_id));
    }
    let href = require_attr(node, "href", "extends")?;
    let resolver =
        resolver.ok_or_else(|| SchematronError::UnresolvedInclude { href: href.clone() })?;
    let fetched = resolver.resolve(&href)?;
    let fetched_tree = Tree::parse_str(&fetched)?;
    let fetched_root =
        fetched_tree
            .document_element()
            .ok_or_else(|| SchematronError::MissingElement {
                what: "rule",
                context: format!("extends href=\"{href}\""),
            })?;
    let inline = parse_rule(&fetched_root, Some(resolver))?;
    Ok(RuleItem::InlineExtends(Box::new(inline)))
}

fn parse_check(node: &NodeRef, kind: CheckKind) -> SchematronResult<CheckDecl> {
    let test = require_attr(node, "test", "assert/report")?;
    let diagnostics = attr(node, "diagnostics")
        .map(|value| value.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    Ok(CheckDecl {
        kind,
        test,
        id: attr(node, "id"),
        role: attr(node, "role"),
        flag: attr(node, "flag"),
        subject: attr(node, "subject"),
        diagnostics,
        message: parse_message_parts(node),
    })
}

fn parse_diagnostic(node: &NodeRef) -> SchematronResult<DiagnosticDecl> {
    Ok(DiagnosticDecl {
        id: require_attr(node, "id", "diagnostic")?,
        message: parse_message_parts(node),
    })
}

/// Walks `node`'s mixed content, turning `sch:value-of`/`sch:name` into substitutions and
/// everything else (including foreign markup such as `sch:emph`) into flattened text.
fn parse_message_parts(node: &NodeRef) -> Vec<MessagePart> {
    let mut parts = Vec::new();
    collect_message_parts(node, &mut parts);
    parts
}

fn collect_message_parts(node: &NodeRef, parts: &mut Vec<MessagePart>) {
    for child in node.children() {
        match child.node_kind() {
            NodeKind::Text => {
                let text = child.string_value();
                if !text.is_empty() {
                    push_text(parts, &text);
                }
            }
            NodeKind::Element if is_sch(&child, "value-of") => {
                if let Some(select) = attr(&child, "select") {
                    parts.push(MessagePart::ValueOf(select));
                }
            }
            NodeKind::Element if is_sch(&child, "name") => {
                parts.push(MessagePart::Name(attr(&child, "path")));
            }
            NodeKind::Element => collect_message_parts(&child, parts),
            _ => {}
        }
    }
}

fn push_text(parts: &mut Vec<MessagePart>, text: &str) {
    if let Some(MessagePart::Text(last)) = parts.last_mut() {
        last.push_str(text);
    } else {
        parts.push(MessagePart::Text(text.to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_schema_parses() -> SchematronResult<()> {
        let schema = parse_schema(
            r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
                 <pattern id="p1">
                   <rule context="doc">
                     <assert test="a">must have an a child</assert>
                   </rule>
                 </pattern>
               </schema>"#,
            None,
        )?;
        assert_eq!(schema.patterns.len(), 1);
        assert_eq!(schema.patterns[0].rules.len(), 1);
        assert_eq!(schema.patterns[0].rules[0].context.as_deref(), Some("doc"));
        Ok(())
    }

    #[test]
    fn a_non_schema_root_is_rejected() {
        let error = parse_schema("<not-a-schema/>", None).expect_err("wrong root element");
        assert!(matches!(error, SchematronError::NotASchema { .. }));
    }

    #[test]
    fn an_unresolved_include_is_reported() {
        let error = parse_schema(
            r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
                 <include href="shared.sch"/>
               </schema>"#,
            None,
        )
        .expect_err("no resolver was supplied");
        assert!(matches!(error, SchematronError::UnresolvedInclude { .. }));
    }
}
