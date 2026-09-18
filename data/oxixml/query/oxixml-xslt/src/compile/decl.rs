//! Compiling the *declarations* that may appear at the top level of a stylesheet module.

use super::env::{StaticEnv, XsltVersion};
use super::instr::{
    optional_expression, params_and_body, required_setting, sequence_constructor, yes,
};
use super::ir::*;
use super::{Compiler, attribute, is_xml_whitespace, resolve_names, tokens, xslt_local_name};
use crate::error::{XsltError, XsltResult};
use crate::names::XSLT_NAMESPACE;
use crate::output::{OutputMethod, OutputProperties, Standalone};
use crate::pattern::Pattern;
use oxixml_infoset::{NodeKind, NodeRef, XdmNode};
use std::sync::Arc;

/// Compiles one top-level declaration into the compiler's IR.
///
/// # Errors
///
/// Every static error the declaration raises.
pub(crate) fn compile_declaration(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<()> {
    // XSLT 3.0 §3.9 ignores a top-level element that is not a declaration this processor
    // recognizes when the element's effective version is a later one — and ignores it *whole*.
    // Its `use-when` is not evaluated either: the expression belongs to the same later version
    // and may name a function that does not exist here, which is precisely what `forwards-008`
    // writes (`<xsl:new-declaration use-when="fn:new-function()"/>`).
    if env.forwards_compatible()
        && crate::names::is_xslt(element)
        && !xslt_local_name(element).is_some_and(crate::content::is_declaration)
    {
        return Ok(());
    }
    if !super::instr::use_when_holds(element, env)? {
        return Ok(());
    }
    let Some(local) = xslt_local_name(element) else {
        if crate::names::is_xslt(element) {
            // A top-level element in the XSLT namespace whose name XSLT does not define.
            // XSLT 3.0 §3.9 ignores it, with its content, in forwards-compatible mode; outside
            // that mode it is `XTSE0010`.
            if env.forwards_compatible() {
                return Ok(());
            }
            return Err(XsltError::stat1c(
                "XTSE0010",
                "an element in the XSLT namespace is not a top-level declaration this processor implements",
            ));
        }
        // A top-level element in a namespace of its own is a *user-defined data element*
        // ([XSLT 3.0 §3.6.2](https://www.w3.org/TR/xslt-30/#user-defined-top-level)): data the
        // stylesheet carries for itself, which "must be ignored by the XSLT processor" and is
        // reached — content and all — through `document('')`. It declares nothing.
        //
        // A literal result element that is the module's *own document element* is a different
        // construct: a simplified stylesheet (§3.7), whose whole element is the body of a
        // template matching the root. The two are told apart by what encloses the element,
        // because `Expansion::begin` hands the module element itself here when the module has no
        // `xsl:stylesheet`.
        if element
            .parent()
            .is_some_and(|parent| super::is_stylesheet_element(&parent))
        {
            // §3.6.2: "It is a static error if the name of the element has a null namespace URI."
            // Such an element is neither an XSLT declaration nor stylesheet data.
            if element
                .node_name()
                .is_none_or(|name| name.namespace().is_none())
            {
                return Err(XsltError::stat1c(
                    "XTSE0130",
                    "a top-level element that is not in the XSLT namespace must be in a \
                     namespace of its own",
                ));
            }
            return Ok(());
        }
        // A simplified stylesheet: the literal result element is the body of a template that
        // matches the root.
        return simplified_stylesheet(compiler, element, env);
    };
    match local {
        "template" => template(compiler, element, env),
        "variable" | "param" => global_variable(compiler, element, env, local == "param"),
        "key" => key(compiler, element, env),
        "output" => output(compiler, element, env),
        "decimal-format" => decimal_format(compiler, element, env),
        "strip-space" => space_control(compiler, element, env, true),
        "preserve-space" => space_control(compiler, element, env, false),
        "attribute-set" => attribute_set(compiler, element, env),
        "function" => function(compiler, element, env),
        "namespace-alias" => namespace_alias(compiler, element, env),
        "mode" => mode(compiler, element, env),
        "character-map" => character_map(compiler, element, env),
        "global-context-item" => global_context_item(compiler, element, env),
        "import-schema" | "include" | "import" | "context-item" => Ok(()),
        // Documented gaps: packaging, accumulators and streaming declarations. The static rules
        // a package manifest can break inside its *own* declaration are still checked, since
        // they need no library of packages to decide.
        "use-package" => {
            check_use_package(element, env)?;
            Err(XsltError::stat1c(
                "OXIXML0001",
                "`xsl:use-package` is not implemented",
            ))
        }
        "accumulator" | "expose" | "override" | "accept" | "merge" | "fork" | "source-document" => {
            Err(XsltError::stat1c(
                "OXIXML0001",
                format!("`xsl:{local}` is not implemented"),
            ))
        }
        // XSLT 3.0 §3.9: a top-level element the processor does not recognize is ignored,
        // together with its content, when the module states a later version than this one.
        other if env.forwards_compatible() => {
            let _ = other;
            Ok(())
        }
        other => Err(XsltError::stat1c(
            "XTSE0010",
            format!("`xsl:{other}` is not a top-level declaration this processor implements"),
        )),
    }
}

/// The static rules an `xsl:use-package` declaration can break on its own.
///
/// Packaging itself is a documented gap, but the rules that hold *within one manifest* need no
/// library of packages to decide, so they are reported as the specification's own static errors
/// rather than as this crate's "not implemented".
///
/// # Errors
///
/// `XTSE3051` — [XSLT 3.0 §3.5.2](https://www.w3.org/TR/xslt-30/#element-accept): "It is a
/// static error if a token in the `names` attribute of `xsl:accept`, other than a wildcard,
/// matches the symbolic name of a component declared within an `xsl:override` child of the same
/// `xsl:use-package` element."
fn check_use_package(element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let mut overridden: Vec<(&'static str, String)> = Vec::new();
    for child in element.children() {
        if xslt_local_name(&child) != Some("override") {
            continue;
        }
        for declaration in child.children() {
            let Some(kind) = xslt_local_name(&declaration).and_then(component_kind) else {
                continue;
            };
            let Some(name) = attribute(&declaration, "name") else {
                continue;
            };
            overridden.push((kind, env.resolve_attribute_name(&name)?.to_string()));
        }
    }
    if overridden.is_empty() {
        return Ok(());
    }
    for child in element.children() {
        if xslt_local_name(&child) != Some("accept") {
            continue;
        }
        let component = attribute(&child, "component").unwrap_or_else(|| "*".to_owned());
        let names = attribute(&child, "names").unwrap_or_default();
        for token in tokens(&names) {
            // A wildcard is explicitly exempt: it names no component in particular, so an
            // overriding declaration it happens to cover is not a conflict.
            if token.contains('*') {
                continue;
            }
            // A function is named with its arity, which no overriding declaration repeats.
            let token = token.split('#').next().unwrap_or(token);
            let expanded = env.resolve_attribute_name(token)?.to_string();
            if overridden.iter().any(|(kind, name)| {
                *name == expanded && (component.trim() == "*" || component.trim() == *kind)
            }) {
                return Err(XsltError::stat1c(
                    "XTSE3051",
                    format!(
                        "`{token}` is accepted by name and also overridden in the same \
                         xsl:use-package"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The `component` value an overriding declaration's element name stands for.
fn component_kind(local: &str) -> Option<&'static str> {
    match local {
        "template" => Some("template"),
        "function" => Some("function"),
        "variable" | "param" => Some("variable"),
        "attribute-set" => Some("attribute-set"),
        "mode" => Some("mode"),
        _ => None,
    }
}

fn simplified_stylesheet(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<()> {
    let mut root_env = env.clone();
    if let Some(version) = super::xslt_attribute(element, "version") {
        super::env::check_version_text(&version)?;
        root_env.set_version_text(&version);
    }
    let body = super::instr::compile_element(compiler, element, &root_env)?
        .into_iter()
        .collect::<Vec<_>>();
    let pattern = Pattern::compile("/", &root_env)?;
    let priority = pattern.default_priority();
    let index = compiler.ir.templates.len();
    compiler.ir.templates.push(Template {
        pattern: Some(Arc::new(pattern)),
        as_type: None,
        alternative: 0,
        priority,
        import_precedence: compiler.precedence,
        declaration_order: compiler.order,
        params: Arc::new(Vec::new()),
        context_item: None,
        content: Arc::new(body),
    });
    compiler
        .ir
        .modes
        .entry(ModeKey::Default)
        .or_default()
        .rules
        .push(index);
    compiler.ir.version = root_env.version();
    Ok(())
}

/// Compiles the `xsl:context-item` child of `xsl:template`, when it has one.
///
/// [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#context-item). Where the element may
/// stand (first child element, at most one, only under `xsl:template`) and which values `use`
/// admits are settled by [`crate::validate`] before compilation starts; what is left here is
/// the interaction between the two attributes and the containing template.
///
/// # Errors
///
/// `XTSE3088` for `as` together with `use="absent"`, `XTSE0020` for an `as` that is not a
/// single `ItemType` or for a `use` other than `required` on a template rule with no name.
fn context_item(element: &NodeRef, env: &StaticEnv) -> XsltResult<Option<ContextItemDecl>> {
    let Some(declaration) = element.children().into_iter().find(|child| {
        child.node_kind() == NodeKind::Element
            || (child.node_kind() == NodeKind::Text && !is_xml_whitespace(&child.string_value()))
    }) else {
        return Ok(None);
    };
    if xslt_local_name(&declaration) != Some("context-item") {
        return Ok(None);
    }
    let env = super::element_env(&declaration, env)?;
    let usage = match env
        .shadowed_attribute(&declaration, "use")?
        .as_deref()
        .map(str::trim)
    {
        None | Some("optional") => ContextItemUse::Optional,
        Some("required") => ContextItemUse::Required,
        Some("absent") => ContextItemUse::Absent,
        Some(other) => {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("`{other}` is not a value the `use` attribute of xsl:context-item takes"),
            ));
        }
    };
    // "If the containing xsl:template element has no name attribute then the only permitted
    // value is required" — an unnamed template rule always runs with the matched item in
    // focus, so neither `optional` nor `absent` can describe it.
    if env.shadowed_attribute(element, "name")?.is_none() && usage != ContextItemUse::Required {
        return Err(XsltError::stat1c(
            "XTSE0020",
            "on an xsl:template with no name attribute, xsl:context-item may only say \
             use=\"required\"",
        ));
    }
    let Some(text) = env.shadowed_attribute(&declaration, "as")? else {
        return Ok(Some(ContextItemDecl {
            as_type: None,
            usage,
        }));
    };
    // [ERR XTSE3088] It is a static error if the as attribute is present when use="absent" is
    // specified.
    if usage == ContextItemUse::Absent {
        return Err(XsltError::stat1c(
            "XTSE3088",
            "xsl:context-item may not have an `as` attribute together with use=\"absent\"",
        ));
    }
    let as_type = crate::typing::RequiredType::compile(&env, &text)?;
    // The attribute holds an `ItemType`, not a `SequenceType`: an occurrence indicator, or a
    // type this processor cannot parse at all, is outside the permitted value space.
    if as_type.occurrence() != Some(oxixml_xpath::Occurrence::One) {
        return Err(XsltError::stat1c(
            "XTSE0020",
            format!("`{text}` is not a single item type xsl:context-item may require"),
        ));
    }
    Ok(Some(ContextItemDecl {
        as_type: Some(Arc::new(as_type)),
        usage,
    }))
}

/// Compiles the `xsl:global-context-item` declaration.
///
/// [XSLT 3.0 §3.13](https://www.w3.org/TR/xslt-30/#global-context-item) states what the
/// invocation must supply as the *global context item* — the focus every global variable and
/// parameter is evaluated with — and, unlike the per-template `xsl:context-item`, there may be
/// only one of it per module and every module of the package must agree.
///
/// # Errors
///
/// `XTSE3089` for `as` together with `use="absent"`, `XTSE3087` for a second, disagreeing
/// declaration, `XTSE0020` for an `as` that is not a single `ItemType` or an unknown `use`.
fn global_context_item(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<()> {
    let env = super::element_env(element, env)?;
    let usage = match env
        .shadowed_attribute(element, "use")?
        .as_deref()
        .map(str::trim)
    {
        None | Some("optional") => ContextItemUse::Optional,
        Some("required") => ContextItemUse::Required,
        Some("absent") => ContextItemUse::Absent,
        Some(other) => {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!(
                    "`{other}` is not a value the `use` attribute of xsl:global-context-item takes"
                ),
            ));
        }
    };
    let text = env.shadowed_attribute(element, "as")?;
    // [ERR XTSE3089] It is a static error if the as attribute is present when use="absent" is
    // specified.
    if text.is_some() && usage == ContextItemUse::Absent {
        return Err(XsltError::stat1c(
            "XTSE3089",
            "xsl:global-context-item may not have an `as` attribute together with use=\"absent\"",
        ));
    }
    let as_type = match &text {
        Some(text) => {
            let as_type = crate::typing::RequiredType::compile(&env, text)?;
            // The attribute holds an `ItemType`, not a `SequenceType`.
            if as_type.occurrence() != Some(oxixml_xpath::Occurrence::One) {
                return Err(XsltError::stat1c(
                    "XTSE0020",
                    format!(
                        "`{text}` is not a single item type xsl:global-context-item may require"
                    ),
                ));
            }
            Some(Arc::new(as_type))
        }
        None => None,
    };
    // [ERR XTSE3087] It is a static error if a stylesheet module contains more than one
    // xsl:global-context-item declaration, or if two modules of one package declare different
    // ones. Two modules that agree are explicitly allowed.
    let written = (usage, text.map(|text| text.trim().to_owned()));
    if let Some((previous, precedence)) = compiler.global_context_items.last()
        && (previous != &written || *precedence == compiler.precedence)
    {
        return Err(XsltError::stat1c(
            "XTSE3087",
            "the stylesheet has more than one xsl:global-context-item declaration",
        ));
    }
    compiler
        .global_context_items
        .push((written, compiler.precedence));
    compiler.ir.global_context_item = Some(Arc::new(ContextItemDecl { as_type, usage }));
    Ok(())
}

fn template(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let context_item = context_item(element, env)?.map(Arc::new);
    let (params, content) = params_and_body(compiler, element, env)?;
    let as_type = super::instr::required_type(element, env)?;
    let name = match env.shadowed_attribute(element, "name")? {
        Some(text) => Some(env.resolve_attribute_name(&text)?),
        None => None,
    };
    // XSLT 3.0 §3.3: a stylesheet may not declare a name in a namespace XSLT reserves, except
    // for the two names the specification itself defines there.
    if let Some(name) = &name
        && name.namespace() == Some(XSLT_NAMESPACE)
        && !matches!(name.local_name(), "initial-template" | "original")
    {
        return Err(XsltError::stat1c(
            "XTSE0080",
            format!("`{name}` is in a namespace a stylesheet may not declare a name in"),
        ));
    }
    let explicit_priority = match env.shadowed_attribute(element, "priority")? {
        Some(text) => {
            // The value space is `xs:decimal`, which has no exponent.
            let trimmed = text.trim();
            let digits = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
            let decimal = !digits.is_empty()
                && digits.chars().all(|c| c.is_ascii_digit() || c == '.')
                && digits.matches('.').count() <= 1
                && digits.chars().any(|c| c.is_ascii_digit());
            if !decimal {
                return Err(XsltError::stat1c(
                    "XTSE0530",
                    format!("`{text}` is not a valid priority"),
                ));
            }
            Some(trimmed.parse::<f64>().map_err(|_| {
                XsltError::stat1c("XTSE0530", format!("`{text}` is not a valid priority"))
            })?)
        }
        None => None,
    };
    let (modes, all_modes) = template_modes(element, env)?;
    let content = Arc::new(content);
    let params = Arc::new(params);

    match env.shadowed_attribute(element, "match")? {
        Some(text) => {
            let pattern = Pattern::compile(&text, env)?;
            // XSLT 3.0 §6.5: a union pattern is treated as several template rules, one per
            // alternative — but "the splitting of a template rule into multiple rules occurs
            // only if there is no explicit `priority` attribute". A declared priority governs
            // the whole pattern, so the rule stays single.
            let pattern = match explicit_priority {
                Some(_) => pattern.into_single_rule(),
                None => pattern,
            };
            let priorities = pattern.alternative_priorities();
            let pattern = Arc::new(pattern);
            for (alternative, default) in priorities.into_iter().enumerate() {
                let index = compiler.ir.templates.len();
                compiler.ir.templates.push(Template {
                    pattern: Some(Arc::clone(&pattern)),
                    as_type: as_type.clone(),
                    alternative,
                    priority: explicit_priority.unwrap_or(default),
                    import_precedence: compiler.precedence,
                    declaration_order: compiler.order,
                    params: Arc::clone(&params),
                    context_item: context_item.clone(),
                    content: Arc::clone(&content),
                });
                for mode in &modes {
                    compiler
                        .ir
                        .modes
                        .entry(mode.clone())
                        .or_default()
                        .rules
                        .push(index);
                }
                if all_modes {
                    compiler.all_mode_rules.push(index);
                }
                if alternative == 0
                    && let Some(name) = &name
                {
                    compiler.ir.named_templates.insert(name.clone(), index);
                }
            }
        }
        None => {
            let Some(name) = name else {
                return Err(XsltError::stat1c(
                    "XTSE0500",
                    "xsl:template needs a match or a name attribute",
                ));
            };
            // XSLT 3.0 §6.3: without `match` there is no rule to place, so `mode` and
            // `priority` have nothing to say.
            for forbidden in ["mode", "priority"] {
                if attribute(element, forbidden).is_some() {
                    return Err(XsltError::stat1c(
                        "XTSE0500",
                        format!("`xsl:template` without `match` may not have `{forbidden}`"),
                    ));
                }
            }
            compiler.declare("XTSE0660", "template", name.to_string());
            let index = compiler.ir.templates.len();
            compiler.ir.templates.push(Template {
                pattern: None,
                as_type: as_type.clone(),
                alternative: 0,
                priority: explicit_priority.unwrap_or(0.0),
                import_precedence: compiler.precedence,
                declaration_order: compiler.order,
                params: Arc::clone(&params),
                context_item,
                content: Arc::clone(&content),
            });
            let replace = compiler
                .ir
                .named_templates
                .get(&name)
                .and_then(|existing| compiler.ir.templates.get(*existing))
                .is_none_or(|existing| existing.import_precedence <= compiler.precedence);
            if replace {
                compiler.ir.named_templates.insert(name, index);
            }
        }
    }
    Ok(())
}

/// The modes a template rule belongs to, and whether it was written with `mode="#all"`.
///
/// [XSLT 3.0 §6.6](https://www.w3.org/TR/xslt-30/#modes): an absent `mode` and the token
/// `#default` both name the mode `default-mode` puts in force, `#unnamed` always names the
/// unnamed mode, and `#all` names every mode the package uses.
fn template_modes(element: &NodeRef, env: &StaticEnv) -> XsltResult<(Vec<ModeKey>, bool)> {
    let Some(value) = env.shadowed_attribute(element, "mode")? else {
        return Ok((vec![super::default_mode_key(env)], false));
    };
    // XSLT 3.0 §6.6 makes every fault in this list `XTSE0550`: an empty list, a repeated
    // token, a token that is not a mode name, or `#all` alongside anything else.
    let bad = |what: &str| XsltError::stat1c("XTSE0550", what.to_owned());
    let written = tokens(&value);
    if written.is_empty() {
        return Err(bad(
            "the `mode` attribute of `xsl:template` may not be empty",
        ));
    }
    for (index, token) in written.iter().enumerate() {
        if written[..index].contains(token) {
            return Err(bad("the `mode` attribute lists a mode twice"));
        }
    }
    let mut modes = Vec::new();
    let mut all = false;
    for token in written {
        match token {
            "#default" => modes.push(super::default_mode_key(env)),
            "#unnamed" => modes.push(ModeKey::Default),
            "#all" => all = true,
            name => {
                let resolved = env
                    .resolve_attribute_name(name)
                    .map_err(|_| bad("the `mode` attribute lists something that is not a name"))?;
                modes.push(ModeKey::Named(resolved));
            }
        }
    }
    if all && !modes.is_empty() {
        return Err(bad(
            "`#all` is the only token the `mode` attribute may then carry",
        ));
    }
    if modes.is_empty() && !all {
        modes.push(super::default_mode_key(env));
    }
    Ok((modes, all))
}

fn global_variable(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
    is_parameter: bool,
) -> XsltResult<()> {
    let name = env.resolve_attribute_name(&required_setting(element, "name", env)?)?;
    let precedence = compiler.precedence;
    compiler.declare("XTSE0630", "global", name.to_string());
    let required = is_parameter && yes(&env.shadowed_attribute(element, "required")?);
    let select = optional_expression(element, "select", env)?;
    // Kept for `Compiler::check_global_self_reference`, which can only judge the scope of this
    // declaration's own `select` once every global name in the package is known.
    if let Some(text) = env.shadowed_attribute(element, "select")? {
        compiler
            .global_selects
            .push((name.clone(), text, env.clone()));
    }
    let content = sequence_constructor(compiler, element, env)?;
    // As `super::instr::parameter` for a template's own `xsl:param`: a global parameter
    // declared `required="yes"` must not also carry a default (param-0111/param-0113's rule
    // applies at either scope), which is `XTSE0010` rather than a run-time surprise.
    if required && (select.is_some() || !content.is_empty()) {
        return Err(XsltError::stat1c(
            "XTSE0010",
            "xsl:param with required=\"yes\" must not also have a select attribute or a \
             non-empty sequence constructor",
        ));
    }
    let declaration = GlobalVariable {
        name: name.clone(),
        select,
        content,
        is_parameter,
        is_static: yes(&env.shadowed_attribute(element, "static")?),
        required,
        as_type: super::instr::required_type(element, env)?,
        import_precedence: precedence,
        base_uri: env.base_uri().map(Arc::from),
    };
    match compiler
        .ir
        .globals
        .iter_mut()
        .find(|existing| existing.name == name)
    {
        Some(existing) if existing.import_precedence <= precedence => *existing = declaration,
        Some(_) => {}
        None => compiler.ir.globals.push(declaration),
    }
    Ok(())
}

fn key(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = env.resolve_attribute_name(&required_setting(element, "name", env)?)?;
    let pattern = Pattern::compile(&required_setting(element, "match", env)?, env)?;
    let use_expression = optional_expression(element, "use", env)?;
    let content = sequence_constructor(compiler, element, env)?;
    // XSLT 3.0 §17.1: the collation of an `xsl:key` must be one the processor recognizes.
    let collation = match env.shadowed_attribute(element, "collation")? {
        Some(uri) => {
            let uri = uri.trim().to_owned();
            if !crate::run::is_recognized_collation(&uri) {
                return Err(XsltError::stat1c(
                    "XTSE1210",
                    format!("`{uri}` is not a collation this processor recognizes"),
                ));
            }
            Some(uri)
        }
        None => env.default_collation().map(str::to_owned),
    };
    let composite = yes(&env.shadowed_attribute(element, "composite")?);
    // XSLT 3.0 §17.1: every `xsl:key` of one name must agree about its collation and about
    // `composite`.
    for existing in &compiler.ir.keys {
        if existing.name != name {
            continue;
        }
        if existing.collation != collation {
            return Err(XsltError::stat1c(
                "XTSE1220",
                format!("the xsl:key declarations named `{name}` disagree about the collation"),
            ));
        }
        if existing.composite != composite {
            return Err(XsltError::stat1c(
                "XTSE1222",
                format!("the xsl:key declarations named `{name}` disagree about `composite`"),
            ));
        }
    }
    compiler.ir.keys.push(KeyDefinition {
        name,
        pattern,
        use_expression,
        content,
        collation,
        composite,
    });
    Ok(())
}

/// Reads a serialization parameter whose value must be `yes` or `no`.
///
/// # Errors
///
/// `XTSE0020`, which [XSLT 3.0 §3.5](https://www.w3.org/TR/xslt-30/#attribute-values) makes a
/// static error for any attribute whose value is constrained to a fixed set.
fn boolean_parameter(
    name: &str,
    value: &str,
    names: &OutputNames<'_>,
    version: XsltVersion,
) -> XsltResult<bool> {
    // XSLT 2.0 admits only `yes` and `no`; XSLT 3.0 widened these attributes to the whole
    // `xs:boolean` lexical space.
    let extended = version >= XsltVersion::V30;
    match value.trim() {
        "yes" => Ok(true),
        "no" => Ok(false),
        "true" | "1" if extended => Ok(true),
        "false" | "0" if extended => Ok(false),
        other => Err(names.reject(
            "XTSE0020",
            "SEPM0016",
            format!("`{other}` is not a value for the `{name}` parameter"),
        )),
    }
}

/// Whether `character` may appear in a `PubidLiteral`, per
/// [XML 1.0 §2.3](https://www.w3.org/TR/xml/#NT-PubidChar).
fn is_pubid_char(character: char) -> bool {
    matches!(character, ' ' | '\r' | '\n')
        || character.is_ascii_alphanumeric()
        || matches!(
            character,
            '-' | '\''
                | '('
                | ')'
                | '+'
                | ','
                | '.'
                | '/'
                | ':'
                | '='
                | '?'
                | ';'
                | '!'
                | '*'
                | '#'
                | '@'
                | '$'
                | '_'
                | '%'
        )
}

/// Resolves a list of `QName`s written on a serialization parameter, where an unprefixed name
/// takes the default namespace declared on the element rather than no namespace.
///
/// # Errors
///
/// `XTSE0280` when a prefix is not bound.
fn resolve_output_names(
    value: &str,
    names: &OutputNames<'_>,
) -> XsltResult<Vec<oxixml_infoset::ExpandedName>> {
    tokens(value)
        .into_iter()
        .map(|token| names.resolve(token))
        .collect()
}

/// How a `QName` written in a serialization parameter is expanded, and which kind of error a
/// value outside the parameter's lexical space raises.
///
/// An `xsl:output` declaration — and an `xsl:result-document` attribute that turns out to be a
/// constant — is settled while the stylesheet is read, so the compiler's own environment expands
/// the name and a bad value is a *static* error. The effective value of an attribute value
/// template is only known once the instruction runs, so it is expanded against a snapshot of the
/// bindings that were in scope where the instruction is written — [XSLT 3.0
/// §23.1](https://www.w3.org/TR/xslt-30/#element-result-document) — and a bad value is a
/// *dynamic* error under the code the serialization specification gives that parameter.
pub(crate) enum OutputNames<'a> {
    /// The compiler's static environment.
    Static(&'a StaticEnv),
    /// The prefix-to-namespace bindings in scope where the instruction was written, with the
    /// empty prefix carrying the default namespace declaration.
    Snapshot(&'a [(String, String)]),
}

impl OutputNames<'_> {
    /// Expands one lexical `QName` written in a serialization parameter.
    ///
    /// # Errors
    ///
    /// `XTSE0280` (statically) or `SEPM0017` (dynamically) when the prefix is not bound.
    fn resolve(&self, lexical: &str) -> XsltResult<oxixml_infoset::ExpandedName> {
        match self {
            Self::Static(env) => env.resolve_output_name(lexical),
            Self::Snapshot(bindings) => {
                let lexical = lexical.trim();
                if let Some(rest) = lexical.strip_prefix("Q{")
                    && let Some((namespace, local)) = rest.split_once('}')
                {
                    let namespace = namespace.trim();
                    return Ok(oxixml_infoset::ExpandedName::new(
                        (!namespace.is_empty()).then_some(namespace),
                        local.trim(),
                    ));
                }
                let (prefix, local) = match lexical.split_once(':') {
                    // An unprefixed name in one of these lists takes the default namespace
                    // declaration in scope, not no namespace.
                    None => ("", lexical),
                    Some((prefix, local)) => (prefix, local),
                };
                let namespace = if prefix == "xml" {
                    Some(oxixml_qname::XML_NAMESPACE.to_owned())
                } else {
                    bindings
                        .iter()
                        .find(|(bound, _)| bound == prefix)
                        .map(|(_, uri)| uri.clone())
                };
                match (prefix.is_empty(), namespace) {
                    (_, Some(uri)) if !uri.is_empty() => {
                        Ok(oxixml_infoset::ExpandedName::new(Some(uri), local))
                    }
                    (true, _) => Ok(oxixml_infoset::ExpandedName::new(None::<&str>, local)),
                    (false, _) => Err(XsltError::dynamic(
                        "SEPM0017",
                        format!("the prefix `{prefix}` is not bound"),
                    )),
                }
            }
        }
    }

    /// Expands the effective value of an `xsl:result-document/@format`.
    ///
    /// # Errors
    ///
    /// `XTDE1460` when the value is not an `EQName`, or carries a prefix that is not bound —
    /// which [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#element-result-document) makes the
    /// same fault as naming no output definition at all (`error-0290a`, `error-1460d`).
    pub(crate) fn resolve_output_definition(
        &self,
        lexical: &str,
    ) -> XsltResult<oxixml_infoset::ExpandedName> {
        let lexical = lexical.trim();
        // An unprefixed `format` is in no namespace, not in the default namespace: it names an
        // `xsl:output` whose own `name` was expanded the same way.
        let expanded = match self {
            Self::Static(env) => env.resolve_attribute_name(lexical),
            Self::Snapshot(_) if lexical.contains(':') || lexical.starts_with("Q{") => {
                self.resolve(lexical)
            }
            Self::Snapshot(_) => Ok(oxixml_infoset::ExpandedName::new(None::<&str>, lexical)),
        };
        expanded.map_err(|_| {
            XsltError::dynamic(
                "XTDE1460",
                format!("`{lexical}` is not a usable output definition name"),
            )
        })
    }

    /// Reports a value that is outside a serialization parameter's lexical space, under the
    /// code that fits where the value was settled.
    fn reject(&self, static_code: &str, dynamic_code: &str, message: String) -> XsltError {
        match self {
            Self::Static(_) => XsltError::stat1c(static_code, message),
            Self::Snapshot(_) => XsltError::dynamic(dynamic_code, message),
        }
    }
}

/// Applies the serialization attributes an `xsl:output` or `xsl:result-document` carries.
///
/// # Errors
///
/// `XTSE1570` for an unknown method or a malformed `QName` list.
pub(crate) fn apply_output_attributes(
    properties: &mut OutputProperties,
    element: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<()> {
    for attribute_node in element.attributes() {
        let Some(name) = attribute_node.node_name() else {
            continue;
        };
        if name.namespace().is_some() {
            continue;
        }
        apply_output_attribute(
            properties,
            name.local_name(),
            &attribute_node.string_value(),
            &OutputNames::Static(env),
            env.version(),
        )?;
    }
    Ok(())
}

/// Applies one serialization parameter, given its effective value.
///
/// # Errors
///
/// `XTSE1570`/`SERE0022` for an unknown method, `XTSE0020`/`SEPM0016` for a value outside a
/// parameter's lexical space, and `OXIXML0001` for one of this processor's documented gaps.
pub(crate) fn apply_output_attribute(
    properties: &mut OutputProperties,
    local: &str,
    value: &str,
    names: &OutputNames<'_>,
    version: XsltVersion,
) -> XsltResult<()> {
    let owned = value.to_owned();
    let value = value.trim();
    match local {
        // A documented gap: `parameter-document` names a document of serialization parameters to
        // be read at run time, which this processor does not implement. It is reported under this
        // crate's own code so that a caller — and the conformance harness — can tell "not
        // implemented" from "your stylesheet is wrong".
        "parameter-document" => {
            return Err(XsltError::stat1c(
                "OXIXML0001",
                format!("the `{local}` serialization parameter is not implemented"),
            ));
        }
        // XSLT 3.0 §26: `build-tree="no"` hands the destination's content to the serialization
        // method as the raw sequence it is, rather than normalizing it into a result tree first.
        // Written out it overrides the default the method carries.
        "build-tree" => {
            properties.build_tree = Some(boolean_parameter("build-tree", value, names, version)?);
        }
        "method" => {
            // Serialization 3.1 §3 gives the run-time form of this fault its own code:
            // `result-document-1405` states the method through an attribute value template,
            // so the value is not there to be judged until the instruction runs.
            let method = OutputMethod::parse(value).ok_or_else(|| {
                names.reject(
                    "XTSE1570",
                    "SERE0022",
                    format!("`{owned}` is not a serialization method"),
                )
            })?;
            properties.method = method;
            properties.method_specified = true;
        }
        "version" => properties.version = Some(owned),
        "html-version" => {
            if value.parse::<f64>().is_err() {
                return Err(names.reject(
                    "XTSE0020",
                    "SEPM0016",
                    format!("`{owned}` is not a value for the `html-version` parameter"),
                ));
            }
            properties.html_version = Some(owned);
        }
        "encoding" => properties.encoding = owned,
        "omit-xml-declaration" => {
            properties.omit_xml_declaration =
                boolean_parameter("omit-xml-declaration", value, names, version)?;
        }
        "standalone" => {
            let extended = version >= XsltVersion::V30;
            properties.standalone = match value {
                "yes" => Standalone::Yes,
                "no" => Standalone::No,
                "true" | "1" if extended => Standalone::Yes,
                "false" | "0" if extended => Standalone::No,
                "omit" => Standalone::Omit,
                other => {
                    return Err(names.reject(
                        "XTSE0020",
                        "SEPM0016",
                        format!("`{other}` is not a value for the `standalone` parameter"),
                    ));
                }
            };
        }
        "doctype-public" => {
            // XSLT 2.0 erratum E3: `doctype-public` supplies the public identifier of a
            // document type declaration, so its value has to be a `PubidLiteral` — the
            // characters XML 1.0 §2.3 allows there. `output-0284` writes `£[~~~]` and
            // expects the value to be rejected.
            if !owned.chars().all(is_pubid_char) {
                return Err(names.reject(
                    "XTSE0020",
                    "SEPM0016",
                    format!("`{owned}` is not a public identifier"),
                ));
            }
            properties.doctype_public = Some(owned);
        }
        "doctype-system" => properties.doctype_system = Some(owned),
        // [XSLT 3.0 §26.1](https://www.w3.org/TR/xslt-30/#serialization-parameters): an
        // unprefixed name in these lists is in the default namespace of the element the
        // attribute is written on, not in no namespace.
        "cdata-section-elements" => {
            properties
                .cdata_section_elements
                .extend(resolve_output_names(&owned, names)?);
        }
        "suppress-indentation" => {
            properties
                .suppress_indentation
                .extend(resolve_output_names(&owned, names)?);
        }
        "indent" => properties.indent = boolean_parameter("indent", value, names, version)?,
        "media-type" => properties.media_type = Some(owned),
        // XSLT 3.0 §26.2: "To set the parameter to absent, the `item-separator` attribute can
        // either be omitted, or set to the special value `item-separator="#absent"`; it is not
        // possible to set the value of the serialization parameter to the literal 7-character
        // string". Written on an `xsl:result-document`, that is how a separator the named output
        // definition set is taken away again (`result-document-0305`).
        "item-separator" => {
            properties.item_separator = match owned.as_str() {
                "#absent" => None,
                _ => Some(owned),
            };
        }
        "normalization-form" => properties.normalization_form = Some(owned),
        "byte-order-mark" => {
            properties.byte_order_mark =
                boolean_parameter("byte-order-mark", value, names, version)?;
        }
        "include-content-type" => {
            properties.include_content_type =
                boolean_parameter("include-content-type", value, names, version)?;
        }
        "escape-uri-attributes" => {
            properties.escape_uri_attributes =
                boolean_parameter("escape-uri-attributes", value, names, version)?;
        }
        "undeclare-prefixes" => {
            properties.undeclare_prefixes =
                boolean_parameter("undeclare-prefixes", value, names, version)?;
        }
        // Serialization 3.1 §10: the nodes a JSON value holds are serialized with this method
        // and the text written as a JSON string. `result-document-1402b` asks for `html`, whose
        // `include-content-type` rule is what puts the `<meta http-equiv>` in the string.
        "json-node-output-method" => {
            properties.json_node_output_method = OutputMethod::parse(value).ok_or_else(|| {
                names.reject(
                    "XTSE1570",
                    "SERE0022",
                    format!("`{owned}` is not a serialization method"),
                )
            })?;
        }
        "allow-duplicate-names" => {
            properties.allow_duplicate_names =
                boolean_parameter("allow-duplicate-names", value, names, version)?;
        }
        _ => {}
    }
    Ok(())
}

fn output(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = match env.shadowed_attribute(element, "name")? {
        Some(text) => Some(env.resolve_element_name(&text)?),
        None => None,
    };
    match name.clone() {
        None => {
            let mut properties = compiler.ir.output.clone();
            apply_output_attributes(&mut properties, element, env)?;
            compiler.ir.output = properties;
        }
        Some(name) => {
            let mut properties = compiler
                .ir
                .named_outputs
                .get(&name)
                .map(|existing| (**existing).clone())
                .unwrap_or_default();
            apply_output_attributes(&mut properties, element, env)?;
            compiler.ir.named_outputs.insert(name, Arc::new(properties));
        }
    }
    // XSLT 3.0 §26.1 lets several `xsl:output` declarations contribute to one output
    // definition, but not disagree; the judgement waits until every module is compiled.
    for attribute_node in element.attributes() {
        let Some(attribute_name) = attribute_node.node_name() else {
            continue;
        };
        if attribute_name.namespace().is_some() || attribute_name.local_name() == "name" {
            continue;
        }
        compiler.output_attributes.push((
            name.clone(),
            attribute_name.local_name().to_owned(),
            attribute_node.string_value(),
            compiler.precedence,
        ));
    }
    // `use-character-maps` is resolved after all character maps have been seen.
    if let Some(value) = env.shadowed_attribute(element, "use-character-maps")? {
        let names = resolve_names(&value, env)?;
        compiler.pending_character_maps.push((
            match env.shadowed_attribute(element, "name")? {
                Some(text) => Some(env.resolve_element_name(&text)?),
                None => None,
            },
            names,
        ));
    }
    Ok(())
}

fn decimal_format(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = match env.shadowed_attribute(element, "name")? {
        Some(text) => Some(env.resolve_attribute_name(&text)?),
        None => None,
    };
    let mut format = compiler
        .ir
        .decimal_formats
        .get(&name)
        .cloned()
        .unwrap_or_default();
    let single = |value: &str, what: &str| -> XsltResult<char> {
        let mut characters = value.chars();
        match (characters.next(), characters.next()) {
            (Some(character), None) => Ok(character),
            _ => Err(XsltError::stat1c(
                "XTSE0020",
                format!("the `{what}` of xsl:decimal-format must be one character"),
            )),
        }
    };
    for attribute_node in element.attributes() {
        let Some(attribute_name) = attribute_node.node_name() else {
            continue;
        };
        if attribute_name.namespace().is_some() {
            continue;
        }
        let value = attribute_node.string_value();
        format = match attribute_name.local_name() {
            "decimal-separator" => {
                format.with_decimal_separator(single(&value, "decimal-separator")?)
            }
            "grouping-separator" => {
                format.with_grouping_separator(single(&value, "grouping-separator")?)
            }
            "exponent-separator" => {
                format.with_exponent_separator(single(&value, "exponent-separator")?)
            }
            "infinity" => format.with_infinity(value),
            "minus-sign" => format.with_minus_sign(single(&value, "minus-sign")?),
            "NaN" => format.with_nan(value),
            "percent" => format.with_percent(single(&value, "percent")?),
            "per-mille" => format.with_per_mille(single(&value, "per-mille")?),
            "zero-digit" => format.with_zero_digit(single(&value, "zero-digit")?),
            "digit" => format.with_digit(single(&value, "digit")?),
            "pattern-separator" => {
                format.with_pattern_separator(single(&value, "pattern-separator")?)
            }
            _ => format,
        };
    }
    // XSLT 3.0 §18.3.1: two `xsl:decimal-format` declarations for the same name and the same
    // import precedence may not give the same attribute different values. The judgement is
    // deferred to `Compiler::finish`, where the whole set is known.
    for attribute_node in element.attributes() {
        let Some(attribute_name) = attribute_node.node_name() else {
            continue;
        };
        if attribute_name.namespace().is_some() || attribute_name.local_name() == "name" {
            continue;
        }
        compiler.decimal_format_attributes.push((
            name.clone(),
            attribute_name.local_name().to_owned(),
            attribute_node.string_value(),
            compiler.precedence,
        ));
    }
    compiler.ir.decimal_formats.insert(name, format);
    Ok(())
}

/// Applies the two consistency rules a decimal format must satisfy.
///
/// # Errors
///
/// `XTSE1295` when `zero-digit` is not the zero of a decimal digit family, `XTSE1300` when two
/// of the characters a picture string uses are the same
/// ([XSLT 3.0 §18.3.1](https://www.w3.org/TR/xslt-30/#defining-decimal-format)).
pub(crate) fn check_decimal_format(format: &oxixml_fn::numeric::DecimalFormat) -> XsltResult<()> {
    let zero = format.zero_digit();
    // A digit family is ten consecutive code points in category `Nd`, the first of them the
    // zero; that is what `zero-digit` must name.
    let decimal = |c: char| {
        oxixml_unicode::gc::general_category(c)
            == oxixml_unicode::gc::GeneralCategory::DecimalNumber
    };
    let digit_zero = decimal(zero)
        && (0..10).all(|offset| char::from_u32(zero as u32 + offset).is_some_and(decimal))
        && !char::from_u32((zero as u32).wrapping_sub(1)).is_some_and(decimal);
    if !digit_zero {
        return Err(XsltError::stat1c(
            "XTSE1295",
            format!("`{zero}` is not the zero digit of a decimal digit family"),
        ));
    }
    let characters = [
        format.decimal_separator(),
        format.grouping_separator(),
        format.percent(),
        format.per_mille(),
        format.zero_digit(),
        format.digit(),
        format.pattern_separator(),
        format.exponent_separator(),
    ];
    for (index, character) in characters.iter().enumerate() {
        if characters[..index].contains(character) {
            return Err(XsltError::stat1c(
                "XTSE1300",
                format!("`{character}` has two roles in the same decimal format"),
            ));
        }
    }
    Ok(())
}

/// Splits the `elements` attribute of `xsl:strip-space` or `xsl:preserve-space` into name tests.
///
/// The list is whitespace-separated, but the `BracedURILiteral` of an `EQName` may itself contain
/// whitespace ([XPath 3.1 §2.4.5](https://www.w3.org/TR/xpath-31/#doc-xpath31-BracedURILiteral)),
/// so the braces are what a token really ends at. A `}` with no `Q{` before it is ordinary text
/// and splits as usual, which keeps a malformed value from swallowing the rest of the list.
fn name_tests(value: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = None;
    let mut braced = false;
    for (offset, character) in value.char_indices() {
        if braced {
            braced = character != '}';
            continue;
        }
        if character.is_whitespace() {
            if let Some(from) = start.take() {
                out.push(&value[from..offset]);
            }
            continue;
        }
        let from = *start.get_or_insert(offset);
        braced = character == '{' && value[from..offset].ends_with('Q');
    }
    if let Some(from) = start {
        out.push(&value[from..]);
    }
    out
}

/// The namespace of a `Q{uri}*` name test, which the `elements` attribute admits wherever a
/// `prefix:*` wildcard is allowed — [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stylesheet-stripping)
/// types the attribute as a list of `NameTest`s, and XPath 3.1's `Wildcard` production includes
/// the braced form. An empty `Q{}` names no namespace at all, which the space tests cannot yet
/// express, so it is left to the ordinary name path rather than being silently widened.
fn braced_wildcard(token: &str) -> Option<&str> {
    let (namespace, local) = token.strip_prefix("Q{")?.split_once('}')?;
    let namespace = namespace.trim();
    (local == "*" && !namespace.is_empty()).then_some(namespace)
}

fn space_control(
    compiler: &mut Compiler,
    element: &NodeRef,
    env: &StaticEnv,
    strip: bool,
) -> XsltResult<()> {
    // XSLT 3.0 §3.7: "Within an XSLT element that is required to be empty, any content other
    // than comments or processing instructions, including any whitespace text node preserved
    // using the xml:space="preserve" attribute, is a static error." `error-0260d` writes an
    // `xsl:strip-space` carrying `xml:space="preserve"` around a run of blank lines.
    if element
        .children()
        .iter()
        .any(|child| matches!(child.node_kind(), NodeKind::Text | NodeKind::Element))
    {
        return Err(XsltError::stat1c(
            "XTSE0260",
            format!(
                "xsl:{} must be empty",
                xslt_local_name(element).unwrap_or("")
            ),
        ));
    }
    let value = required_setting(element, "elements", env)?;
    for token in name_tests(&value) {
        let (test, priority) = if token == "*" {
            (SpaceTest::Any, -0.5)
        } else if let Some(prefix) = token.strip_suffix(":*") {
            (SpaceTest::AnyLocal(env.resolve_prefix(prefix)?), -0.25)
        } else if let Some(local) = token.strip_prefix("*:") {
            (SpaceTest::AnyNamespace(local.to_owned()), -0.25)
        } else if let Some(namespace) = braced_wildcard(token) {
            (SpaceTest::AnyLocal(namespace.to_owned()), -0.25)
        } else {
            (SpaceTest::Name(env.resolve_element_name(token)?), 0.0)
        };
        compiler.ir.space_controls.push(SpaceControl {
            test,
            strip,
            priority,
            import_precedence: compiler.precedence,
        });
    }
    Ok(())
}

fn attribute_set(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = env.resolve_attribute_name(&required_setting(element, "name", env)?)?;
    let uses = match env.shadowed_attribute(element, "use-attribute-sets")? {
        Some(value) => resolve_names(&value, env)?,
        None => Vec::new(),
    };
    compiler
        .attribute_set_references
        .extend(uses.iter().cloned());
    let content = sequence_constructor(compiler, element, env)?;
    compiler
        .ir
        .attribute_sets
        .entry(name)
        .or_default()
        .push(AttributeSet {
            uses,
            content,
            import_precedence: compiler.precedence,
        });
    Ok(())
}

fn function(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = env.resolve_attribute_name(&required_setting(element, "name", env)?)?;
    if name.namespace().is_none() {
        return Err(XsltError::stat1c(
            "XTSE0740",
            "the name of an xsl:function must be in a namespace",
        ));
    }
    // XSLT 3.0 §3.1: a stylesheet function may not be declared in a reserved namespace.
    if name
        .namespace()
        .is_some_and(crate::names::is_reserved_namespace)
    {
        return Err(XsltError::stat1c(
            "XTSE0080",
            format!("`{name}` is in a namespace a stylesheet may not declare a name in"),
        ));
    }
    let (params, content) = params_and_body(compiler, element, env)?;
    let arity = params.len();
    let precedence = compiler.precedence;
    // XSLT 3.0 §10.3 spells the memoization hint `new-each-time="yes|no|maybe"`: `no` promises
    // that two calls with the same arguments may share one result, which is exactly what the
    // draft-era `cache="yes"` asked for. Both spellings are honoured, and `maybe` — the default
    // — leaves the choice to the processor, which takes it as "do not memoize".
    let cache = super::instr::yes(&env.shadowed_attribute(element, "cache")?)
        || super::instr::no(&env.shadowed_attribute(element, "new-each-time")?);
    compiler.declare("XTSE0770", "function", format!("{name}#{arity}"));
    let existing_precedence = compiler
        .ir
        .functions
        .get(&(name.clone(), arity))
        .map(|existing| existing.import_precedence);
    if existing_precedence.is_none_or(|existing| existing <= precedence) {
        compiler.ir.functions.insert(
            (name, arity),
            UserFunction {
                as_type: super::instr::required_type(element, env)?,
                params,
                content,
                import_precedence: precedence,
                cache,
            },
        );
    }
    Ok(())
}

fn namespace_alias(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let stylesheet = required_setting(element, "stylesheet-prefix", env)?;
    let result = required_setting(element, "result-prefix", env)?;
    // XSLT 3.0 §11.1.5 gives an unbound prefix here its own code.
    let resolve = |prefix: &str| -> XsltResult<String> {
        if prefix == "#default" {
            return env.resolve_prefix("").or(Ok(String::new()));
        }
        env.resolve_prefix(prefix).map_err(|_| {
            XsltError::stat1c(
                "XTSE0812",
                format!("the prefix `{prefix}` of an `xsl:namespace-alias` is not bound"),
            )
        })
    };
    compiler.ir.namespace_aliases.push((
        resolve(&stylesheet)?,
        resolve(&result)?,
        compiler.precedence,
    ));
    Ok(())
}

fn mode(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let key = match env.shadowed_attribute(element, "name")? {
        Some(text) if !matches!(text.trim(), "#default" | "#unnamed") => {
            ModeKey::Named(env.resolve_attribute_name(&text)?)
        }
        _ => ModeKey::Default,
    };
    // XSLT 3.0 §6.6.1: the unnamed mode is always private, and no mode declared in a
    // stylesheet (as opposed to a package's `xsl:expose`) may be abstract.
    let mut private = false;
    if let Some(visibility) = env.shadowed_attribute(element, "visibility")? {
        let visibility = visibility.trim().to_owned();
        if visibility == "abstract" {
            return Err(XsltError::stat1c(
                "XTSE0020",
                "`abstract` is not a visibility an `xsl:mode` declaration may have",
            ));
        }
        if key == ModeKey::Default && visibility != "private" {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("the unnamed mode may not be declared `{visibility}`"),
            ));
        }
        private = visibility == "private";
    }
    // Two `xsl:mode` declarations for the same mode are only an error when they share the
    // *highest* import precedence for that mode, so the judgement waits until every module is
    // compiled; see `Compiler::check_mode_declarations`.
    compiler
        .mode_declarations
        .push((key.clone(), compiler.precedence));
    let entry = compiler.ir.modes.entry(key).or_default();
    // Declarations are compiled lowest import precedence first, so the last one to state a
    // property is the one that wins — which is how an importing module overrides the visibility
    // an imported module's mode was declared with (`mode-1902`).
    entry.declaration.private = private;
    if let Some(policy) = env
        .shadowed_attribute(element, "on-no-match")?
        .as_deref()
        .and_then(NoMatchPolicy::parse)
    {
        entry.declaration.on_no_match = policy;
    }
    if env
        .shadowed_attribute(element, "on-multiple-match")?
        .as_deref()
        .map(str::trim)
        == Some("fail")
    {
        entry.declaration.on_multiple_match_fails = true;
    }
    if let Some(typed) = env.shadowed_attribute(element, "typed")? {
        entry.declaration.typed = super::instr::yes(&Some(typed));
    }
    Ok(())
}

fn character_map(compiler: &mut Compiler, element: &NodeRef, env: &StaticEnv) -> XsltResult<()> {
    let name = env.resolve_element_name(&required_setting(element, "name", env)?)?;
    compiler.declare("XTSE1580", "character-map", name.to_string());
    let mut map = std::collections::BTreeMap::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element
            || child.node_name().is_none_or(|name| {
                name.namespace() != Some(XSLT_NAMESPACE) || name.local_name() != "output-character"
            })
        {
            continue;
        }
        let character = required_setting(&child, "character", env)?;
        let string = required_setting(&child, "string", env)?;
        let mut characters = character.chars();
        match (characters.next(), characters.next()) {
            (Some(character), None) => {
                map.insert(character, string);
            }
            _ => {
                return Err(XsltError::stat1c(
                    "XTSE0020",
                    "the `character` of xsl:output-character must be one character",
                ));
            }
        }
    }
    if let Some(value) = env.shadowed_attribute(element, "use-character-maps")? {
        compiler
            .pending_character_map_uses
            .push((name.clone(), resolve_names(&value, env)?));
    }
    compiler.character_maps.insert(name, map);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn decimal_formats_take_single_characters() -> XsltResult<()> {
        let tree = Tree::parse_str(
            r#"<xsl:decimal-format xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                 decimal-separator="," grouping-separator="." NaN="nicht"/>"#,
        )?;
        let element = tree.document_element().expect("an element");
        let env = StaticEnv::new(XsltVersion::V20);
        let mut compiler = Compiler::new(None);
        decimal_format(&mut compiler, &element, &env)?;
        let format = compiler
            .ir
            .decimal_formats
            .get(&None)
            .expect("the unnamed format");
        assert_eq!(format.decimal_separator(), ',');
        assert_eq!(format.nan(), "nicht");
        Ok(())
    }
}
