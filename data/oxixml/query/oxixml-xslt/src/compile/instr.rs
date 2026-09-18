//! Compiling a *sequence constructor* — the content of a template, of `xsl:for-each`, of a
//! literal result element — into [`Instruction`]s.
//!
//! [XSLT 3.0 §5.6](https://www.w3.org/TR/xslt-30/#sequence-constructors) defines the content
//! model: text nodes become literal text, elements in the XSLT namespace become instructions,
//! everything else becomes a literal result element whose attributes are attribute value
//! templates.

use super::env::{Avt, StaticEnv, XsltVersion};
use super::ir::*;
use super::{
    Compiler, attribute, element_env, is_xml_whitespace, resolve_names, tokens, xslt_local_name,
};
use crate::error::{XsltError, XsltResult};
use crate::names::{XSLT_NAMESPACE, is_available_instruction};
use crate::pattern::Pattern;
use crate::typing::RequiredType;
use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};
use oxixml_qname::XML_NAMESPACE;
use oxixml_xpath::CompiledXPath;
use std::sync::Arc;

/// Compiles the `as` attribute of `element`, when it has one.
///
/// # Errors
///
/// `XPST0003` when the attribute does not hold a sequence type.
pub(crate) fn required_type(
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Option<Arc<RequiredType>>> {
    match env.shadowed_attribute(element, "as")? {
        Some(text) => Ok(Some(Arc::new(RequiredType::compile(env, &text)?))),
        None => Ok(None),
    }
}

/// Compiles the children of `parent` into a sequence constructor.
///
/// # Errors
///
/// Every static error the instructions raise.
pub(crate) fn sequence_constructor(
    compiler: &mut Compiler,
    parent: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Vec<Instruction>> {
    let children = parent.children();
    let preserve = preserves_space(parent);
    let body_start = body_start(&children);
    let mut out = Vec::new();
    let mut pending = String::new();
    for (position, child) in children.iter().enumerate() {
        match child.node_kind() {
            // A run of text nodes that only comments and processing instructions separate is
            // one text node by the time whitespace is stripped, so it is collected whole and
            // tested whole.
            NodeKind::Text if position >= body_start => pending.push_str(&child.string_value()),
            NodeKind::Element => {
                // The run of text that a declaration ends is removed whatever `xml:space` says.
                flush_text(
                    &mut out,
                    &mut pending,
                    env,
                    preserve && !strips_space_before(child),
                )?;
                if let Some(instruction) = compile_element(compiler, child, env)? {
                    out.push(instruction);
                }
            }
            _ => {}
        }
    }
    flush_text(&mut out, &mut pending, env, preserve)?;
    Ok(out)
}

/// Where the sequence constructor begins among `children`.
///
/// The declarations an instruction takes first — `xsl:param` on a template or a function,
/// `xsl:sort` on `xsl:for-each` — are not part of its sequence constructor, and neither is the
/// whitespace laid out around them. `whitespace-015` writes an `xsl:for-each` carrying
/// `xml:space="preserve"` around two `xsl:sort` children and expects only the text that
/// *follows* the last of them to survive.
fn body_start(children: &[NodeRef]) -> usize {
    let mut start = 0;
    for (position, child) in children.iter().enumerate() {
        match child.node_kind() {
            NodeKind::Text if is_xml_whitespace(&child.string_value()) => {}
            NodeKind::Comment | NodeKind::ProcessingInstruction => {}
            NodeKind::Element if is_leading_declaration(child) => start = position + 1,
            _ => break,
        }
    }
    start
}

/// Whether `element` is one of the XSLT elements that may only precede a sequence constructor.
fn is_leading_declaration(element: &NodeRef) -> bool {
    matches!(
        xslt_local_name(element),
        Some("sort" | "param" | "with-param" | "context-item")
    )
}

/// Emits the text collected since the last element child, if
/// [XSLT 3.0 §4.3](https://www.w3.org/TR/xslt-30/#stripping-whitespace) keeps it.
///
/// [§4.2](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) removes comments and processing
/// instructions from a stylesheet module *before* §4.3 strips whitespace, so `whitespace-012`
/// writes `<e>   h<!--c-->   </e>` and expects the content `   h   `: the two text nodes the
/// comment separated are a single, not-whitespace-only text node once it is gone.
fn flush_text(
    out: &mut Vec<Instruction>,
    pending: &mut String,
    env: &StaticEnv,
    preserve: bool,
) -> XsltResult<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let text = std::mem::take(pending);
    if is_xml_whitespace(&text) && !preserve {
        return Ok(());
    }
    out.push(text_instruction(&text, env, false)?);
    Ok(())
}

/// Compiles one text node of a sequence constructor, honouring `expand-text`.
///
/// # Errors
///
/// Any static error an embedded expression raises.
fn text_instruction(
    text: &str,
    env: &StaticEnv,
    disable_escaping: bool,
) -> XsltResult<Instruction> {
    if !env.expands_text() || !text.contains(['{', '}']) {
        return Ok(Instruction::Text {
            text: text.to_owned(),
            disable_escaping,
        });
    }
    let template = env.compile_avt(text)?;
    match template.constant() {
        Some(constant) => Ok(Instruction::Text {
            text: constant.to_owned(),
            disable_escaping,
        }),
        None => Ok(Instruction::TextTemplate {
            template,
            disable_escaping,
        }),
    }
}

/// The XSLT elements whose whitespace-only text children
/// [XSLT 3.0 §4.2](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) removes "regardless of
/// any `xml:space` attributes". Every one of them takes elements rather than text, so whitespace
/// written inside it can only be indentation — `whitespace-001` and `-004` put an
/// `xml:space="preserve"` on such an element on purpose and expect it to change nothing.
const SPACE_STRIPPING_PARENTS: &[&str] = &[
    "accumulator",
    "analyze-string",
    "apply-imports",
    "apply-templates",
    "attribute-set",
    "call-template",
    "character-map",
    "choose",
    "evaluate",
    "fork",
    "merge",
    "merge-source",
    "mode",
    "next-iteration",
    "next-match",
    "override",
    "package",
    "stylesheet",
    "transform",
    "use-package",
];

/// The XSLT declarations a whitespace-only text node may not precede: §4.2 removes such a node
/// "regardless of any `xml:space` attributes", since a declaration is not part of a sequence
/// constructor and the whitespace laid out around it is not part of one either.
/// `whitespace-014` and `-015` are written to pin that.
const DECLARATIONS_STRIPPING_SPACE_BEFORE: &[&str] =
    &["param", "sort", "context-item", "on-completion"];

/// The `xml:space` attribute written on `element`.
fn xml_space(element: &NodeRef) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        attribute
            .node_name()
            .is_some_and(|name| {
                name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "space"
            })
            .then(|| attribute.string_value())
    })
}

/// Whether an element declaration ends a run of whitespace that §4.2 removes whatever
/// `xml:space` says.
fn strips_space_before(element: &NodeRef) -> bool {
    xslt_local_name(element)
        .is_some_and(|local| DECLARATIONS_STRIPPING_SPACE_BEFORE.contains(&local))
}

/// Whether the node after `position` — §4.2 having removed the comments and processing
/// instructions in between first — is one of those declarations.
fn declaration_follows(children: &[NodeRef], position: usize) -> bool {
    children[position + 1..]
        .iter()
        .find(|node| {
            !matches!(
                node.node_kind(),
                NodeKind::Comment | NodeKind::ProcessingInstruction
            )
        })
        .is_some_and(strips_space_before)
}

/// Whether whitespace-only text children of this element are significant.
///
/// [XSLT 3.0 §4.2](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) keeps such a text node
/// when its parent is `xsl:text`, or when it "has an ancestor element that has an `xml:space`
/// attribute with a value of `preserve`, unless there is a closer ancestor element having an
/// `xml:space` attribute with a value of `default`". The attribute is *inherited*
/// ([XML 1.0 §2.10](https://www.w3.org/TR/xml/#sec-white-space)), which is the whole of
/// `choose-0604`: the `xml:space="preserve"` it writes on `xsl:choose` speaks for the `xsl:when`
/// and the `xsl:if` nested inside it, and for nothing that `xsl:choose` itself contains.
fn preserves_space(element: &NodeRef) -> bool {
    if xslt_local_name(element) == Some("text") {
        return true;
    }
    if xslt_local_name(element).is_some_and(|local| SPACE_STRIPPING_PARENTS.contains(&local)) {
        return false;
    }
    let mut ancestor = Some(element.clone());
    while let Some(node) = ancestor {
        if node.node_kind() != NodeKind::Element {
            return false;
        }
        if let Some(value) = xml_space(&node) {
            return value == "preserve";
        }
        ancestor = node.parent();
    }
    false
}

/// Compiles one element of a sequence constructor. Returns `None` for an element that
/// contributes nothing (a stray `xsl:sort`, say, which its parent has already collected).
pub(crate) fn compile_element(
    compiler: &mut Compiler,
    element: &NodeRef,
    parent_env: &StaticEnv,
) -> XsltResult<Option<Instruction>> {
    let env = element_env(element, parent_env)?;
    if !use_when_holds(element, &env)? {
        return Ok(None);
    }
    // XSLT 3.0 §3.9: "[ERR XTDE0160] It is a dynamic error if an element has an effective
    // version of V (with V < 3.0) when the implementation does not support backwards
    // compatible behavior for XSLT version V." §3.9.2 settles V = 2.0 for every 3.0 processor
    // ("no differences are defined for XSLT 2.0 behavior … will therefore produce the same
    // results"), so only an effective version of 1.0 is at stake, and §27.4 leaves it to the
    // processor whether it claims the XSLT 1.0 compatibility feature — which
    // `StaticOptions::with_backwards_compatibility` chooses. Under the strict reading the
    // element's whole subtree collapses into the raising instruction: nothing inside it can be
    // evaluated without evaluating it, and the error stays *dynamic*, so a 1.0 branch that is
    // never taken still costs nothing.
    if env.version() == XsltVersion::V10 && !compiler.backwards_compatibility {
        let name = element
            .node_name()
            .map_or_else(|| "element".to_owned(), |name| name.to_string());
        return Ok(Some(Instruction::BackwardsCompatible { name }));
    }
    let Some(local) = xslt_local_name(element) else {
        if crate::names::is_xslt(element) {
            // An element in the XSLT namespace that XSLT does not define is never a literal
            // result element: it is an unknown instruction, and runs its `xsl:fallback`.
            let name = element
                .node_name()
                .unwrap_or_else(|| ExpandedName::new(Some(XSLT_NAMESPACE), "unknown"));
            let (has_fallback, fallback) = fallback_children(compiler, element, &env)?;
            // An unrecognized XSLT element inside a sequence constructor stays a static
            // error even in forwards-compatible mode: `error-0010bb`, whose module states
            // version 22.0, expects `XTSE0010` and notes that the earlier draft's `XTDE1450`
            // was withdrawn. Only `use-when` and unrecognized *top-level* declarations are
            // relaxed there.
            if !has_fallback {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    format!("`{name}` is not an XSLT element this processor implements"),
                ));
            }
            return Ok(Some(Instruction::Unknown {
                name,
                has_fallback,
                fallback,
            }));
        }
        // XSLT 3.0 §3.7: an element whose prefix an `extension-element-prefixes` attribute in
        // scope names is an *extension instruction*, not a literal result element. This
        // processor implements no extension instructions, so evaluating one runs its
        // `xsl:fallback` children, and `XTDE1450` if it has none.
        if is_extension_instruction(element) {
            let name = element
                .node_name()
                .unwrap_or_else(|| ExpandedName::new(None::<&str>, "unknown"));
            // XSLT 3.0 §3.7: "It is a static error [XTSE0800] if the name of an extension
            // instruction is in a reserved namespace" — a stylesheet that designates one of the
            // namespaces XSLT itself reserves as an extension namespace and then writes an
            // element in it has made a mistake the processor can see without running anything
            // (`extension-functions-0105` designates the XSD namespace).
            if name
                .namespace()
                .is_some_and(crate::names::is_reserved_namespace)
            {
                return Err(XsltError::stat1c(
                    "XTSE0800",
                    format!("`{name}` names an extension instruction in a reserved namespace"),
                ));
            }
            let (has_fallback, fallback) = fallback_children(compiler, element, &env)?;
            return Ok(Some(Instruction::Unknown {
                name,
                has_fallback,
                fallback,
            }));
        }
        return Ok(Some(literal_element(compiler, element, &env)?));
    };
    // The streaming, merging, packaging and dynamic-evaluation instructions are documented
    // gaps. Reporting them under this crate's own code — rather than as a stylesheet error —
    // lets a caller, and the conformance harness, tell "not implemented" from "not valid".
    if UNIMPLEMENTED_INSTRUCTIONS.contains(&local) && !fallback_children(compiler, element, &env)?.0
    {
        return Err(XsltError::stat1c(
            "OXIXML0001",
            format!("`xsl:{local}` is not implemented"),
        ));
    }
    let instruction = match local {
        "text" => {
            let text: String = element
                .children()
                .iter()
                .filter(|child| child.node_kind() == NodeKind::Text)
                .map(XdmNode::string_value)
                .collect();
            let disable_escaping =
                yes(&env.shadowed_attribute(element, "disable-output-escaping")?);
            text_instruction(&text, &env, disable_escaping)?
        }
        "value-of" => Instruction::ValueOf {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
            separator: optional_avt(element, "separator", &env)?,
            disable_escaping: yes(&env.shadowed_attribute(element, "disable-output-escaping")?),
        },
        "sequence" => Instruction::Sequence {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "copy-of" => Instruction::CopyOf {
            select: required_expression(element, "select", &env)?,
            copy_namespaces: !no(&env.shadowed_attribute(element, "copy-namespaces")?),
        },
        "copy" => Instruction::Copy {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
            use_attribute_sets: used_attribute_sets(compiler, element, &env)?,
            copy_namespaces: !no(&env.shadowed_attribute(element, "copy-namespaces")?),
            inherit_namespaces: !no(&env.shadowed_attribute(element, "inherit-namespaces")?),
        },
        "apply-templates" => {
            let select = optional_expression(element, "select", &env)?;
            let mode = mode_reference(element, &env)?;
            if let ModeRef::Named(name) = &mode {
                compiler
                    .referenced_modes
                    .push(super::ir::ModeKey::Named(name.clone()));
            }
            Instruction::ApplyTemplates {
                select,
                mode,
                sort: sort_keys(compiler, element, &env)?,
                params: with_params(compiler, element, &env)?,
            }
        }
        "apply-imports" => Instruction::ApplyImports {
            params: with_params(compiler, element, &env)?,
        },
        "next-match" => Instruction::NextMatch {
            params: with_params(compiler, element, &env)?,
        },
        "call-template" => {
            let name = required_setting(element, "name", &env)?;
            let name = env.resolve_attribute_name(&name)?;
            let params = with_params(compiler, element, &env)?;
            // Whether the template knows each of these parameters, and whether this call site
            // in turn supplies everything the template requires, is only known once every
            // declaration is compiled; `Compiler::check_call_template_params` decides both.
            let supplied = params
                .iter()
                .filter(|param| !param.tunnel)
                .map(|param| param.name.clone())
                .collect();
            compiler.call_template_params.push((name.clone(), supplied));
            Instruction::CallTemplate { name, params }
        }
        "for-each" => Instruction::ForEach {
            select: required_expression(element, "select", &env)?,
            sort: sort_keys(compiler, element, &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "for-each-group" => Instruction::ForEachGroup {
            select: required_expression(element, "select", &env)?,
            grouping: grouping(element, &env)?,
            composite: yes(&env.shadowed_attribute(element, "composite")?),
            collation: collation_avt(element, &env)?,
            sort: sort_keys(compiler, element, &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "if" => Instruction::If {
            test: required_expression(element, "test", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "choose" => choose(compiler, element, &env)?,
        "variable" | "param" => Instruction::Variable(Box::new(VariableDecl {
            name: env.resolve_attribute_name(&required_setting(element, "name", &env)?)?,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
            as_type: required_type(element, &env)?,
            base_uri: env.base_uri().map(Arc::from),
        })),
        "element" => Instruction::Element {
            name: required_avt(element, "name", &env)?,
            namespace: optional_avt(element, "namespace", &env)?,
            prefixes: prefix_snapshot(&env),
            use_attribute_sets: used_attribute_sets(compiler, element, &env)?,
            inherit_namespaces: !no(&env.shadowed_attribute(element, "inherit-namespaces")?),
            content: sequence_constructor(compiler, element, &env)?,
        },
        "attribute" => Instruction::Attribute {
            name: required_avt(element, "name", &env)?,
            namespace: optional_avt(element, "namespace", &env)?,
            prefixes: prefix_snapshot(&env),
            select: optional_expression(element, "select", &env)?,
            separator: optional_avt(element, "separator", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "namespace" => Instruction::Namespace {
            name: required_avt(element, "name", &env)?,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "comment" => Instruction::Comment {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "processing-instruction" => Instruction::ProcessingInstruction {
            name: required_avt(element, "name", &env)?,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "document" => Instruction::Document {
            content: sequence_constructor(compiler, element, &env)?,
        },
        "number" => Instruction::Number(Box::new(number(element, &env)?)),
        "message" => Instruction::Message {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
            terminate: match optional_avt(element, "terminate", &env)? {
                Some(avt) => avt,
                None => Avt::literal("no"),
            },
            error_code: error_code(element, &env)?,
        },
        "analyze-string" => analyze_string(compiler, element, &env)?,
        "perform-sort" => Instruction::PerformSort {
            select: optional_expression(element, "select", &env)?,
            sort: sort_keys(compiler, element, &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "result-document" => result_document(compiler, element, &env)?,
        "try" => try_catch(compiler, element, &env)?,
        "assert" => Instruction::Assert {
            test: required_expression(element, "test", &env)?,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
            error_code: error_code(element, &env)?,
        },
        "iterate" => iterate(compiler, element, &env)?,
        "next-iteration" => Instruction::NextIteration {
            params: with_params(compiler, element, &env)?,
        },
        "break" => Instruction::Break {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "on-empty" => Instruction::OnEmpty {
            expect_empty: true,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "on-non-empty" => Instruction::OnEmpty {
            expect_empty: false,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "where-populated" => Instruction::WherePopulated {
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "map" => Instruction::Map {
            content: sequence_constructor(compiler, element, &env)?,
        },
        "map-entry" => Instruction::MapEntry {
            key: required_expression(element, "key", &env)?,
            select: optional_expression(element, "select", &env)?,
            content: sequence_constructor(compiler, element, &env)?,
        },
        "fallback" => {
            evaluate_use_when_in_ignored_content(element, &env)?;
            Instruction::Fallback
        }
        "sort"
        | "with-param"
        | "matching-substring"
        | "non-matching-substring"
        | "when"
        | "otherwise"
        | "on-completion"
        | "catch"
        | "param-not-here" => return Ok(None),
        other => {
            let (has_fallback, fallback) = fallback_children(compiler, element, &env)?;
            // XSLT 3.0 §3.9: in forwards-compatible mode an unrecognized XSLT instruction is
            // compiled rather than rejected, and only fails — `XTDE1450` — if it is actually
            // evaluated with no `xsl:fallback` to stand in for it.
            if !has_fallback && !is_available_instruction(other) && !env.forwards_compatible() {
                return Err(XsltError::stat1c(
                    "XTSE0010",
                    format!("`xsl:{other}` is not an instruction this processor implements"),
                ));
            }
            Instruction::Unknown {
                name: ExpandedName::new(Some(XSLT_NAMESPACE), other),
                has_fallback,
                fallback,
            }
        }
    };
    Ok(Some(located(element, instruction)))
}

/// Remembers where `instruction` is written, when it is one that can raise on its own account.
///
/// [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch) gives an `xsl:catch` body the
/// line and column of the instruction that failed, so the provenance has to survive compilation.
/// The wrapper goes only around the four instructions a failure surfaces from at run time: the
/// two that evaluate a `select` into the result, the one that opens a second result document, and
/// the one that hands control to another template. Everything else either raises nothing itself or
/// contains an instruction that does, and that inner one is the position to report:
/// [`XsltError::at`](crate::XsltError::at) keeps the innermost as the error travels outwards.
///
/// The position is `None` unless the stylesheet was parsed with
/// [`oxixml_infoset::ParseOptions::record_source_positions`], and an attribute node never has one
/// of its own, which is why the *element* is the node read here.
fn located(element: &NodeRef, instruction: Instruction) -> Instruction {
    if !matches!(
        instruction,
        Instruction::ValueOf { .. }
            | Instruction::Sequence { .. }
            | Instruction::ResultDocument { .. }
            | Instruction::CallTemplate { .. }
    ) {
        return instruction;
    }
    match element.source_position() {
        Some(position) => Instruction::Positioned {
            position,
            inner: Box::new(instruction),
        },
        None => instruction,
    }
}

/// The XSLT instructions and declarations this processor documents as gaps.
const UNIMPLEMENTED_INSTRUCTIONS: &[&str] = &[
    "accept",
    "accumulator",
    "accumulator-rule",
    "evaluate",
    "expose",
    "fork",
    "merge",
    "merge-action",
    "merge-key",
    "merge-source",
    "override",
    "package",
    "source-document",
    "use-package",
];

/// A snapshot of the namespace bindings in scope, for resolving a `QName` computed at run time.
fn prefix_snapshot(env: &StaticEnv) -> std::sync::Arc<Vec<(String, String)>> {
    std::sync::Arc::new(
        env.namespaces()
            .map(|(prefix, uri)| (prefix.to_owned(), uri.to_owned()))
            .collect(),
    )
}

fn literal_element(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Instruction> {
    let name = element.node_name().ok_or_else(|| {
        XsltError::stat1c("XTSE0010", "a literal result element must have a name")
    })?;
    let excluded = excluded_prefixes(element)?;
    let mut namespaces = Vec::new();
    for namespace in element.namespaces() {
        let prefix = namespace.node_name().map(|n| n.local_name().to_owned());
        let uri = namespace.string_value();
        if uri == XSLT_NAMESPACE || uri == XML_NAMESPACE || prefix.as_deref() == Some("xml") {
            continue;
        }
        if excluded.contains(&uri) {
            continue;
        }
        namespaces.push((prefix, uri));
    }

    let mut attributes = Vec::new();
    let mut use_attribute_sets = Vec::new();
    for attribute in element.attributes() {
        let Some(attribute_name) = attribute.node_name() else {
            continue;
        };
        if attribute_name.namespace() == Some(XSLT_NAMESPACE) {
            if attribute_name.local_name() == "use-attribute-sets" {
                use_attribute_sets = resolve_names(&attribute.string_value(), env)?;
            }
            continue;
        }
        let prefix = prefix_of(element, attribute_name.namespace());
        attributes.push((
            attribute_name,
            prefix,
            env.compile_avt(&attribute.string_value())?,
        ));
    }

    compiler
        .attribute_set_references
        .extend(use_attribute_sets.iter().cloned());
    // The prefix the element was *written* with wins over any other binding of the same
    // namespace: `output-0138` declares `my` and `one` for one URI and asserts that
    // `<one:h3>` comes out as `one:h3`. Written with no prefix at all, the element keeps that
    // spelling whenever the default namespace declaration in scope is the one its name is in —
    // `result-document-0217` writes `<item3 xmlns="…"/>` beside a prefixed binding of the same
    // URI and asserts the result element is `<item3 xmlns="…">`, not `<my:item3>`.
    let prefix = match name.prefix() {
        Some(prefix) => Some(prefix.to_owned()),
        None if binds_default_namespace(element, name.namespace()) => None,
        None => prefix_of(element, name.namespace()),
    };
    // On a literal result element the attribute is spelled `xsl:inherit-namespaces`, since an
    // unprefixed one belongs to the result element itself.
    let inherit_namespaces = !no(&super::xslt_attribute(element, "inherit-namespaces"));
    Ok(Instruction::Literal(Box::new(LiteralElement {
        prefix,
        name,
        namespaces,
        attributes,
        use_attribute_sets,
        inherit_namespaces,
        content: sequence_constructor(compiler, element, env)?,
    })))
}

/// The prefix bound to `namespace` on `element`, preferring the one actually written.
fn prefix_of(element: &NodeRef, namespace: Option<&str>) -> Option<String> {
    let namespace = namespace?;
    element.namespaces().into_iter().find_map(|binding| {
        (binding.string_value() == namespace)
            .then(|| binding.node_name().map(|n| n.local_name().to_owned()))
            .flatten()
    })
}

/// Whether the default namespace declaration in scope on `element` binds `namespace`.
///
/// The namespace node standing for `xmlns="…"` is the one with no name — XDM 3.1 §6.2 — which is
/// exactly what [`prefix_of`] skips over when it looks for a prefix to spell a name with.
fn binds_default_namespace(element: &NodeRef, namespace: Option<&str>) -> bool {
    let namespace = namespace.unwrap_or_default();
    element
        .namespaces()
        .into_iter()
        .any(|binding| binding.node_name().is_none() && binding.string_value() == namespace)
}

/// The namespace URI bound to `prefix` on `element`, the empty prefix standing for the default
/// namespace declaration.
fn namespace_in_scope(element: &NodeRef, prefix: &str) -> Option<String> {
    element.namespaces().into_iter().find_map(|binding| {
        let bound = binding
            .node_name()
            .map_or_else(String::new, |name| name.local_name().to_owned());
        (bound == prefix).then(|| binding.string_value())
    })
}

/// The namespaces `exclude-result-prefixes` and `extension-element-prefixes` remove from a
/// literal result element.
///
/// What the two attributes list are *prefixes*, but what they exclude are the namespace **URIs**
/// those prefixes are bound to — [XSLT 3.0 §11.1.3](https://www.w3.org/TR/xslt-30/#literal-namespace-uri)
/// calls the result "the set of excluded namespaces". A second prefix bound to an excluded URI
/// is excluded with it, and a prefix rebound further down to a different URI is not: `lre-028`
/// is written to pin exactly that.
///
/// Which makes *where each token is resolved* the whole of the question. A token is resolved
/// against the namespaces in scope on the element that **carries** the attribute, not on the
/// literal result element being compiled, and `#all` likewise stands for "all the namespaces
/// that are in scope for the element on which the attribute appears". `copy-1220` writes
/// `exclude-result-prefixes="#all"` on `xsl:stylesheet` and then declares further namespaces on
/// the elements below it, which the stylesheet element cannot have meant to exclude — it had
/// never heard of them.
fn excluded_prefixes(element: &NodeRef) -> XsltResult<Vec<String>> {
    let mut excluded = Vec::new();
    let mut collect = |carrier: &NodeRef, value: &str| -> XsltResult<()> {
        for token in tokens(value) {
            match token {
                "#all" => {
                    for binding in carrier.namespaces() {
                        excluded.push(binding.string_value());
                    }
                }
                "#default" => {
                    if let Some(uri) = namespace_in_scope(carrier, "") {
                        excluded.push(uri);
                    }
                }
                prefix => excluded.push(namespace_in_scope(carrier, prefix).ok_or_else(|| {
                    XsltError::stat1c("XTSE0280", format!("the prefix `{prefix}` is not bound"))
                })?),
            }
        }
        Ok(())
    };
    for name in ["exclude-result-prefixes", "extension-element-prefixes"] {
        if let Some(value) = attribute(element, name) {
            collect(element, &value)?;
        }
        if let Some(value) = super::xslt_attribute(element, name) {
            collect(element, &value)?;
        }
    }
    // Inherited from ancestors, which the specification makes cumulative.
    let mut ancestor = element.parent();
    while let Some(node) = ancestor {
        if node.node_kind() == NodeKind::Element {
            for name in ["exclude-result-prefixes", "extension-element-prefixes"] {
                if let Some(value) = super::xslt_attribute(&node, name) {
                    collect(&node, &value)?;
                }
                if crate::names::is_xslt(&node)
                    && let Some(value) = attribute(&node, name)
                {
                    collect(&node, &value)?;
                }
            }
        }
        ancestor = node.parent();
    }
    Ok(excluded)
}

fn choose(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<Instruction> {
    let mut branches = Vec::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element {
            continue;
        }
        let child_env = element_env(&child, env)?;
        // A branch may carry `use-when`, which decides at compile time whether it exists at
        // all — `version-001` guards a branch that names an instruction of a later version.
        if !use_when_holds(&child, &child_env)? {
            continue;
        }
        match xslt_local_name(&child) {
            Some("when") => branches.push((
                Some(required_expression(&child, "test", &child_env)?),
                sequence_constructor(compiler, &child, &child_env)?,
            )),
            Some("otherwise") => {
                branches.push((None, sequence_constructor(compiler, &child, &child_env)?))
            }
            _ => {}
        }
    }
    // XSLT 3.0 §8.2 gives `xsl:choose` the content model `(xsl:when+, xsl:otherwise?)`, so an
    // `xsl:otherwise` with no `xsl:when` in front of it is `XTSE0010` just as an empty
    // `xsl:choose` is — `error-0010n` is exactly that stylesheet.
    if !branches.iter().any(|(test, _)| test.is_some()) {
        return Err(XsltError::stat1c(
            "XTSE0010",
            "xsl:choose must have at least one xsl:when",
        ));
    }
    Ok(Instruction::Choose { branches })
}

fn analyze_string(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Instruction> {
    let mut matching = None;
    let mut non_matching = None;
    for child in element.children() {
        if child.node_kind() != NodeKind::Element {
            continue;
        }
        let child_env = element_env(&child, env)?;
        match xslt_local_name(&child) {
            Some("matching-substring") => {
                matching = Some(sequence_constructor(compiler, &child, &child_env)?);
            }
            Some("non-matching-substring") => {
                non_matching = Some(sequence_constructor(compiler, &child, &child_env)?);
            }
            _ => {}
        }
    }
    Ok(Instruction::AnalyzeString {
        select: required_expression(element, "select", env)?,
        regex: required_avt(element, "regex", env)?,
        flags: match optional_avt(element, "flags", env)? {
            Some(avt) => avt,
            None => Avt::literal(""),
        },
        matching,
        non_matching,
    })
}

fn result_document(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Instruction> {
    let mut output = compiler.ir.output.clone();
    let mut deferred_format = None;
    if let Some(format) = env.shadowed_attribute(element, "format")? {
        // XSLT 3.0 §23.1: `format` is an attribute value template naming an output definition;
        // an *effective* value that is not an EQName, or that names no `xsl:output`, is
        // `XTDE1460`. The specification allows a processor to report it statically when the
        // attribute has no embedded expression, which is what the pending list below does once
        // every `xsl:output` has been seen; anything else waits for the instruction to run.
        let template = env.compile_avt(&format)?;
        match template.constant() {
            Some(constant) => {
                let name = env.resolve_element_name(constant).map_err(|_| {
                    XsltError::stat1c(
                        "XTDE1460",
                        format!("`{constant}` is not a usable output definition name"),
                    )
                })?;
                match compiler.ir.named_outputs.get(&name) {
                    Some(named) => output = (**named).clone(),
                    None => compiler
                        .result_document_formats
                        .push((name, constant.to_owned())),
                }
            }
            None => deferred_format = Some(template),
        }
    }
    // Every serialization parameter on the instruction is an attribute value template too. A
    // constant one is normally folded into the properties here, exactly as `xsl:output`'s is,
    // and one with an embedded expression is carried to the instruction and applied when it
    // runs. A deferred `format` defers all of them: the output definition it names is not known
    // until then, and the instruction's own parameters have to be applied *over* it.
    let mut deferred_parameters = Vec::new();
    for attribute_node in element.attributes() {
        let Some(name) = attribute_node.node_name() else {
            continue;
        };
        if name.namespace().is_some() {
            continue;
        }
        let local = name.local_name();
        if matches!(local, "href" | "format" | "use-character-maps") {
            continue;
        }
        let template = env.compile_avt(&attribute_node.string_value())?;
        match template.constant() {
            Some(constant) if deferred_format.is_none() => {
                crate::compile::decl::apply_output_attribute(
                    &mut output,
                    local,
                    constant,
                    &crate::compile::decl::OutputNames::Static(env),
                    env.version(),
                )?;
            }
            _ => deferred_parameters.push((local.to_owned(), template)),
        }
    }
    // XSLT 3.0 §23.1: `use-character-maps` on the instruction *supplements* the maps the output
    // definition already names, and takes precedence over them where the two disagree —
    // `output-0307` names `format1` on the definition and `format2` on the instruction and
    // expects both, with `format2` winning on the characters they share. Every map is known by
    // now because `Compiler::resolve_character_maps` runs before any instruction is compiled.
    let mut character_map = std::collections::BTreeMap::new();
    if let Some(value) = env.shadowed_attribute(element, "use-character-maps")? {
        for name in crate::compile::resolve_names(&value, env)? {
            let Some(map) = compiler.character_maps.get(&name) else {
                return Err(XsltError::stat1c(
                    "XTSE1590",
                    format!("no `xsl:character-map` is named `{name}`"),
                ));
            };
            character_map.extend(map.iter().map(|(key, value)| (*key, value.clone())));
        }
    }
    if deferred_format.is_none() {
        output.character_map.extend(character_map.clone());
    }
    let deferred = (deferred_format.is_some() || !deferred_parameters.is_empty()).then(|| {
        Box::new(crate::compile::DeferredOutput {
            format: deferred_format,
            parameters: deferred_parameters,
            character_map,
            namespaces: prefix_snapshot(env),
            version: env.version(),
        })
    });
    Ok(Instruction::ResultDocument {
        href: optional_avt(element, "href", env)?,
        output: std::sync::Arc::new(output),
        deferred,
        content: sequence_constructor(compiler, element, env)?,
    })
}

/// Compiles `xsl:try` and the `xsl:catch` clauses that follow its content.
fn try_catch(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Instruction> {
    let mut content = Vec::new();
    let mut catches = Vec::new();
    for child in element.children() {
        if child.node_kind() == NodeKind::Element && xslt_local_name(&child) == Some("catch") {
            let child_env = crate::compile::element_env(&child, env)?;
            let errors = match env.shadowed_attribute(&child, "errors")? {
                Some(list) => catch_tests(&list, &child_env)?,
                None => vec![CatchTest::Any],
            };
            let select = optional_expression(&child, "select", &child_env)?;
            let body = sequence_constructor(compiler, &child, &child_env)?;
            // XSLT 3.0 §12.3: "It is a static error if the select attribute of the xsl:catch
            // element is present unless the element has empty content."
            if select.is_some() && !body.is_empty() {
                return Err(XsltError::stat1c(
                    "XTSE3150",
                    "xsl:catch with a select attribute must have empty content",
                ));
            }
            catches.push(CatchClause {
                errors,
                select,
                content: body,
            });
            continue;
        }
        match child.node_kind() {
            NodeKind::Text => {
                let text = child.string_value();
                if !is_xml_whitespace(&text) || preserves_space(element) {
                    content.push(text_instruction(&text, env, false)?);
                }
            }
            NodeKind::Element => {
                if let Some(instruction) = compile_element(compiler, &child, env)? {
                    content.push(instruction);
                }
            }
            _ => {}
        }
    }
    Ok(Instruction::Try {
        select: optional_expression(element, "select", env)?,
        content,
        catches,
    })
}

/// Parses the name tests of an `xsl:catch/@errors` list.
///
/// [XSLT 3.0 §12.3](https://www.w3.org/TR/xslt-30/#element-catch) makes the attribute a
/// whitespace-separated list of `NameTest`s, so each token takes any of the spellings a name
/// test has: `*`, `prefix:*`, `*:local`, `Q{uri}*`, `Q{uri}local`, `prefix:local` and a bare
/// local name. An unprefixed token is in *no* namespace — neither the `err:` namespace nor the
/// one `xpath-default-namespace` names, which the suite's try-036 and try-037 both turn on —
/// so `errors="too-late"` catches the no-namespace error `QName('', 'too-late')` and a
/// specification error has to be written `err:FOAR0001` or `Q{…}FOAR0001`.
///
/// # Errors
///
/// `XTSE0280` when a prefix is not bound, or when a braced URI literal is not closed.
fn catch_tests(list: &str, env: &StaticEnv) -> XsltResult<Vec<CatchTest>> {
    let mut tests = Vec::new();
    for token in list.split_whitespace() {
        tests.push(catch_test(token, env)?);
    }
    Ok(tests)
}

/// Parses one name test of an `xsl:catch/@errors` list.
fn catch_test(token: &str, env: &StaticEnv) -> XsltResult<CatchTest> {
    if token == "*" {
        return Ok(CatchTest::Any);
    }
    // A `BracedURILiteral` is checked before the prefix split, since the URI it holds carries
    // colons of its own.
    if let Some(rest) = token.strip_prefix("Q{") {
        let Some((uri, local)) = rest.split_once('}') else {
            return Err(XsltError::stat1c(
                "XTSE0280",
                format!("the braced URI literal in `{token}` is not closed"),
            ));
        };
        let namespace = (!uri.is_empty()).then(|| uri.to_owned());
        return Ok(match local {
            "*" => CatchTest::AnyLocal(namespace),
            _ => CatchTest::Name(ExpandedName::new(namespace, local)),
        });
    }
    Ok(match token.split_once(':') {
        Some((prefix, "*")) => CatchTest::AnyLocal(Some(catch_namespace(prefix, env)?)),
        Some(("*", local)) => CatchTest::AnyNamespace(local.to_owned()),
        Some((prefix, local)) => CatchTest::Name(ExpandedName::new(
            Some(catch_namespace(prefix, env)?),
            local,
        )),
        None => CatchTest::Name(ExpandedName::new(None::<String>, token)),
    })
}

/// Resolves a prefix inside `xsl:catch/@errors`, where `err` is bound to the error namespace
/// even when the stylesheet does not declare it.
fn catch_namespace(prefix: &str, env: &StaticEnv) -> XsltResult<String> {
    match env.resolve_prefix(prefix) {
        Ok(uri) => Ok(uri),
        Err(error) if prefix == "err" => {
            let _ = error;
            Ok(crate::error::ERROR_NAMESPACE.to_owned())
        }
        Err(error) => Err(error),
    }
}

fn iterate(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<Instruction> {
    let mut params = Vec::new();
    let mut on_completion = Vec::new();
    let mut content = Vec::new();
    for child in element.children() {
        match child.node_kind() {
            NodeKind::Text => {
                let text = child.string_value();
                if !is_xml_whitespace(&text) {
                    content.push(text_instruction(&text, env, false)?);
                }
            }
            NodeKind::Element => {
                let child_env = element_env(&child, env)?;
                match xslt_local_name(&child) {
                    Some("param") => params.push(parameter(compiler, &child, &child_env)?),
                    Some("on-completion") => {
                        on_completion = on_completion_body(compiler, element, &child, &child_env)?;
                    }
                    _ => {
                        if let Some(instruction) = compile_element(compiler, &child, env)? {
                            content.push(instruction);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Instruction::Iterate {
        select: required_expression(element, "select", env)?,
        params,
        content,
        on_completion,
    })
}

/// Compiles the value an `xsl:on-completion` produces.
///
/// [XSLT 3.0 §8.8](https://www.w3.org/TR/xslt-30/#iterate) gives `xsl:on-completion` — like
/// `xsl:break` — either a `select` attribute or a sequence constructor: "The effect of the
/// element in both cases is obtained by evaluating the `select` expression if present or the
/// contained sequence constructor otherwise". Reading only the content would silently drop the
/// value of `<xsl:on-completion select="…"/>`.
///
/// The expression is compiled in the scope that surrounds the `xsl:iterate`, not in the scope
/// of its body: `xsl:on-completion` is evaluated once the iteration is over, when the body's own
/// `xsl:variable` bindings no longer exist, so a reference to one of them is `XPST0008` rather
/// than a name that happens to resolve. The `xsl:param` declarations of the `xsl:iterate` itself
/// *are* in scope, which is what distinguishes a body variable that shadows a parameter of the
/// same name from one that is only ever local.
///
/// # Errors
///
/// `XPST0008` for a reference to a variable declared in the body of the `xsl:iterate`, plus
/// every static error the expression or the sequence constructor raises.
fn on_completion_body(
    compiler: &mut Compiler,
    iterate: &NodeRef,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Vec<Instruction>> {
    let Some(text) = env.shadowed_attribute(element, "select")? else {
        return sequence_constructor(compiler, element, env);
    };
    let select = match completion_scope(compiler, iterate, env) {
        Some(declared) => env.compile_expression_in_scope(&text, &declared)?,
        None => env.compile_expression(&text)?,
    };
    Ok(vec![Instruction::Sequence {
        select: Some(select),
        content: Vec::new(),
    }])
}

/// The variables an `xsl:on-completion` inside `iterate` may refer to, or `None` when the
/// ordinary permissive scope is good enough.
///
/// Only the names the body of the `xsl:iterate` declares and nothing else does need excluding,
/// so the strict scope is built only when there is at least one of them: every other name the
/// stylesheet declares stays in scope, which keeps this from rejecting a reference the compiler
/// simply cannot see the declaration of.
fn completion_scope(
    compiler: &Compiler,
    iterate: &NodeRef,
    env: &StaticEnv,
) -> Option<Vec<ExpandedName>> {
    let mut body = Vec::new();
    for child in iterate.children() {
        if xslt_local_name(&child) == Some("variable")
            && let Some(name) = declared_name(&child, env)
        {
            body.push(name);
        }
    }
    if body.is_empty() {
        return None;
    }
    let outer = outer_scope(iterate, env)?;
    body.retain(|name| !outer.contains(name));
    if body.is_empty() {
        return None;
    }
    let mut declared = declared_names(iterate, env)?;
    declared.extend(compiler.ir.globals.iter().map(|global| global.name.clone()));
    declared.retain(|name| !body.contains(name));
    Some(declared)
}

/// The name a variable-binding element declares, or `None` when its prefix does not resolve
/// here.
fn declared_name(element: &NodeRef, env: &StaticEnv) -> Option<ExpandedName> {
    let name = env.shadowed_attribute(element, "name").ok()??;
    env.resolve_attribute_name(&name).ok()
}

/// Every variable name that is in scope where `element` is written, walking outwards.
///
/// A variable is in scope for the following siblings of its declaration and their descendants;
/// a parameter is in scope throughout the body of the element that declares it; and every
/// top-level declaration is a global, in scope everywhere. `None` means a name did not resolve,
/// in which case the caller falls back to the permissive scope rather than risk rejecting a
/// legal reference.
fn outer_scope(element: &NodeRef, env: &StaticEnv) -> Option<Vec<ExpandedName>> {
    let mut names = Vec::new();
    for child in element.children() {
        if xslt_local_name(&child) == Some("param") {
            names.push(declared_name(&child, env)?);
        }
    }
    let mut node = element.clone();
    while let Some(parent) = node.parent() {
        let global = crate::compile::is_stylesheet_element(&parent);
        let children = parent.children();
        for child in &children {
            let local = xslt_local_name(child);
            let declares = match local {
                Some("param") => true,
                Some("variable") => global || precedes(&children, child, &node),
                _ => false,
            };
            if declares {
                names.push(declared_name(child, env)?);
            }
        }
        node = parent;
    }
    Some(names)
}

/// Whether `earlier` comes before `later` among `children`.
fn precedes(children: &[NodeRef], earlier: &NodeRef, later: &NodeRef) -> bool {
    let position = |wanted: &NodeRef| children.iter().position(|child| child.is_same_node(wanted));
    match (position(earlier), position(later)) {
        (Some(first), Some(second)) => first < second,
        _ => false,
    }
}

/// Every variable name declared anywhere in the stylesheet module `element` belongs to.
///
/// `None` when one of them does not resolve here, which makes the caller fall back to the
/// permissive scope.
fn declared_names(element: &NodeRef, env: &StaticEnv) -> Option<Vec<ExpandedName>> {
    let mut names = Vec::new();
    let mut stack = vec![element.root()];
    while let Some(node) = stack.pop() {
        if matches!(
            xslt_local_name(&node),
            Some("variable" | "param" | "with-param")
        ) {
            names.push(declared_name(&node, env)?);
        }
        for child in node.children() {
            if child.node_kind() == NodeKind::Element {
                stack.push(child);
            }
        }
    }
    Some(names)
}

fn grouping(element: &NodeRef, env: &StaticEnv) -> XsltResult<Grouping> {
    if let Some(text) = env.shadowed_attribute(element, "group-by")? {
        return Ok(Grouping::By(env.compile_expression(&text)?));
    }
    if let Some(text) = env.shadowed_attribute(element, "group-adjacent")? {
        return Ok(Grouping::Adjacent(env.compile_expression(&text)?));
    }
    if let Some(text) = env.shadowed_attribute(element, "group-starting-with")? {
        return Ok(Grouping::StartingWith(Pattern::compile(&text, env)?));
    }
    if let Some(text) = env.shadowed_attribute(element, "group-ending-with")? {
        return Ok(Grouping::EndingWith(Pattern::compile(&text, env)?));
    }
    Err(XsltError::stat1c(
        "XTSE1080",
        "xsl:for-each-group needs one of group-by, group-adjacent, group-starting-with or \
         group-ending-with",
    ))
}

fn number(element: &NodeRef, env: &StaticEnv) -> XsltResult<NumberSpec> {
    // XSLT 3.0 §12.1: `start-at` is a whitespace-separated list of integers; written without
    // curly brackets, anything else is the static error `XTSE0020`.
    if let Some(value) = env.shadowed_attribute(element, "start-at")?
        && !value.contains('{')
        && (value.split_whitespace().next().is_none()
            || value
                .split_whitespace()
                .any(|token| token.parse::<i64>().is_err()))
    {
        return Err(XsltError::stat1c(
            "XTSE0020",
            format!("`{value}` is not a list of integers for `start-at`"),
        ));
    }
    let level = match env.shadowed_attribute(element, "level")?.as_deref() {
        Some("multiple") => NumberLevel::Multiple,
        Some("any") => NumberLevel::Any,
        Some("single") | None => NumberLevel::Single,
        Some(other) => {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("`{other}` is not a valid xsl:number level"),
            ));
        }
    };
    Ok(NumberSpec {
        value: optional_expression(element, "value", env)?,
        select: optional_expression(element, "select", env)?,
        level,
        count: match env.shadowed_attribute(element, "count")? {
            Some(text) => Some(Pattern::compile(&text, env)?),
            None => None,
        },
        from: match env.shadowed_attribute(element, "from")? {
            Some(text) => Some(Pattern::compile(&text, env)?),
            None => None,
        },
        format: {
            check_literal_language(element, env)?;
            match optional_avt(element, "format", env)? {
                Some(avt) => avt,
                None => Avt::literal("1"),
            }
        },
        letter_value: optional_avt(element, "letter-value", env)?,
        grouping_separator: optional_avt(element, "grouping-separator", env)?,
        grouping_size: optional_avt(element, "grouping-size", env)?,
        ordinal: optional_avt(element, "ordinal", env)?,
        start_at: optional_avt(element, "start-at", env)?,
        language: optional_avt(element, "lang", env)?,
    })
}

/// Rejects a `lang` attribute written literally whose value is not an `xs:language`.
///
/// XSLT 3.0 §3.5 makes a fixed-value-space attribute written without curly brackets a static
/// error when it lies outside that space, which is `XTSE0020`. A value template is only known
/// at run time and is checked there instead, under `XTDE0030`.
fn check_literal_language(element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let Some(value) = env.shadowed_attribute(element, "lang")? else {
        return Ok(());
    };
    if value.contains('{') || crate::names::is_language(value.trim()) {
        return Ok(());
    }
    Err(XsltError::stat1c(
        "XTSE0020",
        format!("`{value}` is not a value for the `lang` attribute"),
    ))
}

pub(crate) fn sort_keys(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Vec<SortKey>> {
    let mut keys = Vec::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element || xslt_local_name(&child) != Some("sort") {
            continue;
        }
        let child_env = element_env(&child, env)?;
        keys.push(SortKey {
            select: optional_expression(&child, "select", &child_env)?,
            content: sequence_constructor(compiler, &child, &child_env)?,
            order: match optional_avt(&child, "order", &child_env)? {
                Some(avt) => avt,
                None => Avt::literal("ascending"),
            },
            // XSLT 1.0 §10 gives `data-type` the default value `text`, and its schema writes that
            // default down; the 2.0 and 3.0 schemas deliberately do not, because from 2.0 on a
            // sort key carries its own type and an absent `data-type` means "compare it as that
            // type" rather than "compare it as a string".
            data_type: match optional_avt(&child, "data-type", &child_env)? {
                Some(avt) => Some(avt),
                None if child_env.version() == XsltVersion::V10 => Some(Avt::literal("text")),
                None => None,
            },
            case_order: optional_avt(&child, "case-order", &child_env)?,
            language: {
                check_literal_language(&child, env)?;
                optional_avt(&child, "lang", &child_env)?
            },
            collation: optional_avt(&child, "collation", &child_env)?,
            default_collation: child_env.default_collation().map(str::to_owned),
            stable: !no(&env.shadowed_attribute(&child, "stable")?),
        });
        // XSLT 3.0 §3.5: a `stable` written without curly brackets must lie inside its fixed
        // value space, which makes anything else the static error `XTSE0020`. A value template
        // is only known at run time and is left alone.
        if let Some(value) = env.shadowed_attribute(&child, "stable")?
            && !value.contains('{')
            && !matches!(value.trim(), "yes" | "no" | "true" | "false" | "1" | "0")
        {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("`{value}` is not a value for the `stable` attribute"),
            ));
        }
    }
    Ok(keys)
}

pub(crate) fn with_params(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Vec<WithParam>> {
    let mut params: Vec<WithParam> = Vec::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element || xslt_local_name(&child) != Some("with-param") {
            continue;
        }
        let child_env = element_env(&child, env)?;
        // §3.14.1 excludes any element whose `use-when` is false, a parameter among them:
        // `use-when-0401` writes three `xsl:with-param` children and guards the middle one off,
        // expecting the receiving `xsl:param` to fall back on its own default.
        if !use_when_holds(&child, &child_env)? {
            continue;
        }
        let name =
            child_env.resolve_attribute_name(&required_setting(&child, "name", &child_env)?)?;
        // XSLT 3.0 §10.1.1: "It is a static error if two or more sibling xsl:with-param
        // elements have name attributes that represent the same expanded QName."
        // `error-0670b` writes the same name under two prefixes bound to one namespace.
        if params.iter().any(|existing| existing.name == name) {
            return Err(XsltError::stat1c(
                "XTSE0670",
                format!("two sibling xsl:with-param elements are both named `{name}`"),
            ));
        }
        params.push(WithParam {
            name,
            select: optional_expression(&child, "select", &child_env)?,
            content: sequence_constructor(compiler, &child, &child_env)?,
            tunnel: yes(&env.shadowed_attribute(&child, "tunnel")?),
            as_type: required_type(&child, &child_env)?,
            base_uri: child_env.base_uri().map(Arc::from),
        });
    }
    Ok(params)
}

pub(crate) fn parameter(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<ParamDecl> {
    let required = yes(&env.shadowed_attribute(element, "required")?);
    let select = optional_expression(element, "select", env)?;
    let content = sequence_constructor(compiler, element, env)?;
    // [XSLT 3.0 §9.2](https://www.w3.org/TR/xslt-30/#element-param) / the suite's param-0111 and
    // param-0113: `required="yes"` promises the caller must supply a value, which a `select`
    // attribute or a non-empty sequence constructor contradicts by giving a default — a static
    // error, `XTSE0010`, rather than deferred to whatever dynamic error an unsupplied value
    // would otherwise raise.
    if required && (select.is_some() || !content.is_empty()) {
        return Err(XsltError::stat1c(
            "XTSE0010",
            "xsl:param with required=\"yes\" must not also have a select attribute or a \
             non-empty sequence constructor",
        ));
    }
    Ok(ParamDecl {
        name: env.resolve_attribute_name(&required_setting(element, "name", env)?)?,
        select,
        content,
        required,
        tunnel: yes(&env.shadowed_attribute(element, "tunnel")?),
        as_type: required_type(element, env)?,
        base_uri: env.base_uri().map(Arc::from),
    })
}

/// Collects the parameters that lead a template or function body, and the body itself.
pub(crate) fn params_and_body(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<(Vec<ParamDecl>, Vec<Instruction>)> {
    let mut params = Vec::new();
    let mut body = Vec::new();
    let children = element.children();
    let preserve = preserves_space(element);
    for (position, child) in children.iter().enumerate() {
        match child.node_kind() {
            NodeKind::Text => {
                let text = child.string_value();
                // A whitespace-only text node survives `xml:space="preserve"` here as anywhere
                // else — `context-item-019` writes a template whose whole body is the two
                // spaces after its `xsl:context-item` — except in front of a declaration, which
                // XSLT 3.0 §4.2 strips whatever the attribute says. That exception is also what
                // keeps a declaration *reachable*: a preserved space in front of an `xsl:param`
                // would start the body, and the parameter after it would no longer be one.
                if is_xml_whitespace(&text)
                    && (!preserve || declaration_follows(&children, position))
                {
                    continue;
                }
                body.push(text_instruction(&text, env, false)?);
            }
            NodeKind::Element => {
                let child_env = element_env(child, env)?;
                if xslt_local_name(child) == Some("param") && body.is_empty() {
                    if use_when_holds(child, &child_env)? {
                        params.push(parameter(compiler, child, &child_env)?);
                    }
                    continue;
                }
                // `xsl:context-item` is a declaration about the template's focus, not an
                // instruction (XSLT 3.0 §10.1.1); `decl::context_item` reads it from the
                // template element directly. Only the one in the position the content model
                // allows — first, before any parameter or body — is passed over here, so that
                // a misplaced one still reaches `compile_element` and is rejected.
                if xslt_local_name(child) == Some("context-item")
                    && xslt_local_name(element) == Some("template")
                    && params.is_empty()
                    && body.is_empty()
                {
                    continue;
                }
                if let Some(instruction) = compile_element(compiler, child, env)? {
                    body.push(instruction);
                }
            }
            _ => {}
        }
    }
    Ok((params, body))
}

/// Evaluates the `[xsl:]use-when` attribute of every element inside content the compiler is
/// about to ignore.
///
/// [XSLT 3.0 §3.14.1](https://www.w3.org/TR/xslt-30/#conditional-inclusion) makes `use-when`
/// part of the stylesheet's *lexical* processing: "the use-when attribute is evaluated even for
/// elements that are otherwise ignored". The content of an `xsl:fallback` guarding an
/// instruction this processor does implement is exactly such content — it is never instantiated
/// and never compiled — so an error in a `use-when` written there would otherwise go unreported
/// (`use-when-0428` writes an unparseable expression, `use-when-0429` one that needs a context
/// item there is none of).
///
/// An element whose own `use-when` is false is excluded together with everything below it, so
/// the walk does not descend into it.
///
/// # Errors
///
/// Whatever evaluating one of those expressions raises.
fn evaluate_use_when_in_ignored_content(element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    for child in element.children() {
        if child.node_kind() != NodeKind::Element {
            continue;
        }
        let child_env = element_env(&child, env)?;
        if !use_when_holds(&child, &child_env)? {
            continue;
        }
        evaluate_use_when_in_ignored_content(&child, &child_env)?;
    }
    Ok(())
}

/// The instructions the `xsl:fallback` children of `element` contribute, and whether there was
/// an `xsl:fallback` child at all.
///
/// The flag is what decides the error: [XSLT 3.0 §3.9](https://www.w3.org/TR/xslt-30/#fallback)
/// instantiates the fallback children of an instruction the processor does not implement, and
/// `XTDE1450` is for an instruction that has *no* `xsl:fallback`. An `<xsl:fallback/>` with empty
/// content contributes no instructions and is still a fallback — the stylesheet's instruction to
/// do nothing (`extension-functions-0106`).
fn fallback_children(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<(bool, Vec<Instruction>)> {
    let mut out = Vec::new();
    let mut present = false;
    for child in element.children() {
        if child.node_kind() != NodeKind::Element || xslt_local_name(&child) != Some("fallback") {
            continue;
        }
        present = true;
        let child_env = element_env(&child, env)?;
        out.extend(sequence_constructor(compiler, &child, &child_env)?);
    }
    Ok((present, out))
}

fn mode_reference(element: &NodeRef, env: &StaticEnv) -> XsltResult<ModeRef> {
    match env
        .shadowed_attribute(element, "mode")?
        .as_deref()
        .map(str::trim)
    {
        // An absent `mode`, or `#default`, selects the mode `default-mode` puts in force —
        // the unnamed mode when no `default-mode` is in scope (XSLT 3.0 §6.6.1).
        None | Some("#default") => Ok(super::default_mode_reference(env)),
        Some("#current") => Ok(ModeRef::Current),
        Some("#unnamed") => Ok(ModeRef::Default),
        Some(name) => Ok(ModeRef::Named(env.resolve_attribute_name(name)?)),
    }
}

/// Reads an `error-code` attribute together with the namespaces that resolve its prefix.
///
/// # Errors
///
/// Any static error the attribute value template raises.
fn error_code(element: &NodeRef, env: &StaticEnv) -> XsltResult<Option<ErrorCode>> {
    let Some(template) = optional_avt(element, "error-code", env)? else {
        return Ok(None);
    };
    let namespaces = env
        .namespaces()
        .map(|(prefix, uri)| (prefix.to_owned(), uri.to_owned()))
        .collect();
    Ok(Some(ErrorCode {
        template,
        namespaces,
    }))
}

fn attribute_sets(element: &NodeRef, env: &StaticEnv) -> XsltResult<Vec<ExpandedName>> {
    match env.shadowed_attribute(element, "use-attribute-sets")? {
        Some(value) => resolve_names(&value, env),
        None => Ok(Vec::new()),
    }
}

/// Reads `use-attribute-sets` and records the reference, so that a name no
/// `xsl:attribute-set` declares can be reported once every declaration is known.
fn used_attribute_sets(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Vec<ExpandedName>> {
    let names = attribute_sets(element, env)?;
    compiler
        .attribute_set_references
        .extend(names.iter().cloned());
    Ok(names)
}

// --- small attribute helpers ------------------------------------------------------------

pub(crate) fn yes(value: &Option<String>) -> bool {
    matches!(
        value.as_deref().map(str::trim),
        Some("yes") | Some("true") | Some("1")
    )
}

pub(crate) fn no(value: &Option<String>) -> bool {
    matches!(
        value.as_deref().map(str::trim),
        Some("no") | Some("false") | Some("0")
    )
}

/// `XTSE0010` for an attribute an element cannot do without.
fn missing_attribute(element: &NodeRef, name: &str) -> XsltError {
    let owner = element
        .node_name()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default();
    XsltError::stat1c(
        "XTSE0010",
        format!("`xsl:{owner}` requires the `{name}` attribute"),
    )
}

pub(crate) fn optional_expression(
    element: &NodeRef,
    name: &str,
    env: &StaticEnv,
) -> XsltResult<Option<CompiledXPath>> {
    match env.shadowed_attribute(element, name)? {
        Some(text) => Ok(Some(env.compile_expression(&text)?)),
        None => Ok(None),
    }
}

pub(crate) fn required_expression(
    element: &NodeRef,
    name: &str,
    env: &StaticEnv,
) -> XsltResult<CompiledXPath> {
    env.compile_expression(&required_setting(element, name, env)?)
}

pub(crate) fn optional_avt(
    element: &NodeRef,
    name: &str,
    env: &StaticEnv,
) -> XsltResult<Option<Avt>> {
    match env.shadowed_attribute(element, name)? {
        Some(text) => Ok(Some(env.compile_avt(&text)?)),
        None => Ok(None),
    }
}

/// The `collation` attribute of a collation-sensitive instruction, falling back on the default
/// collation in force where the instruction is written.
///
/// [XSLT 3.0 §13.1](https://www.w3.org/TR/xslt-30/#sorting) and §14.6 both say that with the
/// attribute absent "the default collation is used", and §5.4.4 makes that a *static* property:
/// the nearest `[xsl:]default-collation` ancestor decides it, not the stylesheet element alone.
/// Binding it here rather than at run time is what keeps the two in step.
///
/// # Errors
///
/// `XTSE0350`/`XTSE0370` from a malformed attribute value template.
fn collation_avt(element: &NodeRef, env: &StaticEnv) -> XsltResult<Option<Avt>> {
    match optional_avt(element, "collation", env)? {
        Some(avt) => Ok(Some(avt)),
        None => Ok(env.default_collation().map(Avt::literal)),
    }
}

pub(crate) fn required_avt(element: &NodeRef, name: &str, env: &StaticEnv) -> XsltResult<Avt> {
    env.compile_avt(&required_setting(element, name, env)?)
}

/// A required attribute, read through any shadow attribute that supplies it.
///
/// [XSLT 3.0 §3.14.2](https://www.w3.org/TR/xslt-30/#shadow-attributes) makes the shadow
/// spelling available for *every* no-namespace attribute of an element in the XSLT namespace —
/// "this mechanism applies to all attributes in the stylesheet where the attribute name is in no
/// namespace and the name of the parent element is in the XSLT namespace" — so every attribute
/// an XSLT element requires is read through here rather than off the element directly.
///
/// # Errors
///
/// `XTSE0010` when neither the attribute nor a shadow attribute for it is present, plus every
/// static error expanding a shadow attribute raises.
pub(crate) fn required_setting(
    element: &NodeRef,
    name: &str,
    env: &StaticEnv,
) -> XsltResult<String> {
    env.shadowed_attribute(element, name)?
        .ok_or_else(|| missing_attribute(element, name))
}

/// Whether `element` is an extension instruction: some ancestor-or-self carries an
/// `extension-element-prefixes` attribute — spelled `xsl:extension-element-prefixes` on a
/// literal result element — that names the prefix the element's own name uses.
fn is_extension_instruction(element: &NodeRef) -> bool {
    let Some(name) = element.node_name() else {
        return false;
    };
    let Some(namespace) = name.namespace() else {
        return false;
    };
    let mut current = Some(element.clone());
    while let Some(node) = current {
        let declared = attribute(&node, "extension-element-prefixes")
            .filter(|_| crate::names::is_xslt(&node))
            .or_else(|| crate::compile::xslt_attribute(&node, "extension-element-prefixes"));
        if let Some(declared) = declared {
            for token in declared.split_whitespace() {
                let bound = node.namespaces().into_iter().find(|binding| {
                    let prefix = binding
                        .node_name()
                        .map(|found| found.local_name().to_owned())
                        .unwrap_or_default();
                    if token == "#default" {
                        prefix.is_empty()
                    } else {
                        prefix == token
                    }
                });
                if bound.is_some_and(|binding| binding.string_value() == namespace) {
                    return true;
                }
            }
        }
        current = node.parent();
    }
    false
}

/// Evaluates a `use-when` attribute, which decides at compile time whether the element is part
/// of the stylesheet at all.
///
/// [XSLT 3.0 §3.14.1](https://www.w3.org/TR/xslt-30/#conditional-inclusion) discards the
/// element — with everything inside it — when the effective boolean value of the expression is
/// false. That is how a stylesheet guards a construct a later version defines: the guarded
/// element never reaches the compiler, so it cannot be a static error. "No static or dynamic
/// errors will be reported in respect of the element and its contents, **other than errors in
/// the `use-when` attribute itself**" — so an error the expression raises is reported, not
/// swallowed, and the restricted context of §9.7 ([`StaticEnv::compile_static_expression`]) is
/// what makes several of them raise at all.
///
/// # Errors
///
/// Every error the expression raises, statically or on evaluation.
pub(crate) fn use_when_holds(element: &NodeRef, env: &StaticEnv) -> XsltResult<bool> {
    let text = match env
        .shadowed_attribute(element, "use-when")?
        .filter(|_| crate::names::is_xslt(element))
    {
        Some(text) => text,
        None => match crate::compile::xslt_attribute(element, "use-when") {
            Some(text) => text,
            None => return Ok(true),
        },
    };
    let compiled = env.compile_static_expression(&text)?;
    // §9.7 puts the static variables declared before this element into the expression's
    // context, and nothing else.
    let dynamic = env.bind_statics(compiled.context());
    // The availability functions expand their argument against the namespaces in scope for the
    // expression, which is the only place they can be read from before a transformation exists.
    let _scope = crate::run::functions::enter_static_expression(std::sync::Arc::clone(
        compiled.static_context(),
    ));
    compiled
        .evaluate_to_boolean_in(&dynamic)
        .map_err(|error| XsltError::from(error).as_static())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_attributes_are_parsed_loosely() {
        assert!(yes(&Some("yes".to_owned())));
        assert!(yes(&Some(" true ".to_owned())));
        assert!(!yes(&Some("no".to_owned())));
        assert!(!yes(&None));
        assert!(no(&Some("no".to_owned())));
        assert!(!no(&None));
    }
}
