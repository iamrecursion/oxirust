//! Static variables, static parameters and the shadow attributes they drive.
//!
//! [XSLT 3.0 §9.6](https://www.w3.org/TR/xslt-30/#static-params) makes a top-level
//! `xsl:variable` or `xsl:param` carrying `static="yes"` a *static variable*: its value is known
//! while the stylesheet is being compiled, so it can decide what the stylesheet even *is*.
//! Two constructs consume it — the `[xsl:]use-when` attribute of
//! [§3.14.1](https://www.w3.org/TR/xslt-30/#conditional-inclusion), and the *shadow attributes*
//! of [§3.14.2](https://www.w3.org/TR/xslt-30/#shadow-attributes), where `_href="lib{$V}.xsl"`
//! supplies the value of the `href` attribute the element does not carry.
//!
//! The bindings are collected in *stylesheet tree order* — the order the declarations appear in
//! once every `xsl:import` and `xsl:include` has been replaced by the module it names — because
//! [§9.7](https://www.w3.org/TR/xslt-30/#static-expression) puts only the statics declared
//! *before* an element into the static context of its static expressions. That is what makes a
//! forward reference `XPST0008` rather than a circularity, and it is why the collection runs
//! during module expansion rather than after it.

use std::collections::HashMap;

use oxixml_infoset::{ExpandedName, Item, NodeKind, NodeRef, XdmNode};
use oxixml_xpath::XpathValue;

use super::env::StaticEnv;
use super::{attribute, xslt_local_name};
use crate::error::{XsltError, XsltResult};

/// One static variable or parameter, as bound.
#[derive(Debug, Clone)]
struct Binding {
    name: ExpandedName,
    value: XpathValue,
    /// Whether the declaration was an `xsl:param` rather than an `xsl:variable`; two
    /// declarations of one name are only consistent when they agree about this.
    is_parameter: bool,
    /// Which *import group* declared it. A module and everything it includes share one group;
    /// an imported module gets its own. A module is expanded where it is referenced, so a group
    /// seen earlier is always of lower import precedence — which is exactly the comparison
    /// `XTSE3450` asks for.
    group: usize,
}

/// The static variables bound so far, in stylesheet tree order.
#[derive(Debug, Default)]
pub(crate) struct StaticBindings {
    bindings: Vec<Binding>,
    /// The values the caller supplied for static parameters, which take precedence over the
    /// declarations' own `select` expressions.
    supplied: HashMap<ExpandedName, XpathValue>,
}

impl StaticBindings {
    /// Installs the values the caller supplied for static parameters.
    pub(crate) fn set_supplied(&mut self, supplied: HashMap<ExpandedName, XpathValue>) {
        self.supplied = supplied;
    }

    /// The value bound to `name`, if any.
    pub(crate) fn value(&self, name: &ExpandedName) -> Option<XpathValue> {
        self.bindings
            .iter()
            .rev()
            .find(|binding| &binding.name == name)
            .map(|binding| binding.value.clone())
    }

    /// Every static variable and parameter, with the value the compilation settled on.
    ///
    /// §9.6 makes a static variable an ordinary global variable as well, whose value is the one
    /// it was compiled with — so these travel with the compiled stylesheet and are bound again
    /// when it runs, rather than being recomputed from a `select` that may no longer be there.
    pub(crate) fn values(&self) -> Vec<(ExpandedName, XpathValue)> {
        let mut values: Vec<(ExpandedName, XpathValue)> = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            match values.iter_mut().find(|(name, _)| name == &binding.name) {
                // The declaration with the highest import precedence wins, and that is the one
                // latest in stylesheet tree order among consistent declarations.
                Some((_, value)) => value.clone_from(&binding.value),
                None => values.push((binding.name.clone(), binding.value.clone())),
            }
        }
        values
    }

    /// The names a static expression written after everything bound so far may refer to.
    pub(crate) fn names(&self) -> Vec<ExpandedName> {
        let mut names: Vec<ExpandedName> = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            if !names.contains(&binding.name) {
                names.push(binding.name.clone());
            }
        }
        names
    }

    /// The value the caller supplied for a static parameter.
    fn supplied(&self, name: &ExpandedName) -> Option<XpathValue> {
        self.supplied.get(name).cloned()
    }

    /// Adds a binding, having checked it against the earlier declarations of the same name.
    fn push(&mut self, binding: Binding) -> XsltResult<()> {
        for earlier in &self.bindings {
            if earlier.name != binding.name || earlier.group == binding.group {
                continue;
            }
            // §9.6: "if the declaration with higher import precedence occurs after the one with
            // lower import precedence in stylesheet tree order, then the two declarations must
            // be consistent" — the same kind of declaration, holding the same value.
            if earlier.is_parameter != binding.is_parameter
                || !identical(&earlier.value, &binding.value)
            {
                return Err(XsltError::stat1c(
                    "XTSE3450",
                    format!(
                        "the static declarations of `{}` are inconsistent with each other",
                        binding.name
                    ),
                ));
            }
        }
        self.bindings.push(binding);
        Ok(())
    }
}

/// Binds the declaration `element` makes, when it is a static variable or parameter.
///
/// The declaration is left in place for the ordinary declaration compiler as well: §9.6 says
/// "static variables and parameters are global variables and can be used in the same way as
/// other global variables", so the name has to be readable from a non-static expression too.
///
/// # Errors
///
/// `XPST0008` when the `select` expression refers to a static variable that is not yet
/// declared, `XTDE0050` when a `required="yes"` static parameter has no value, and `XTSE3450`
/// when the declaration contradicts an earlier one of lower import precedence.
pub(crate) fn collect_declaration(
    element: &NodeRef,
    env: &StaticEnv,
    group: usize,
) -> XsltResult<()> {
    let Some(local) = xslt_local_name(element) else {
        return Ok(());
    };
    if !matches!(local, "variable" | "param") {
        return Ok(());
    }
    // The `static` attribute may itself be shadowed, and `_static="{$X}"` is well founded:
    // `$X` can only be a static variable declared earlier still.
    if !super::instr::yes(&env.shadowed_attribute(element, "static")?) {
        return Ok(());
    }
    // A declaration the stylesheet has conditioned away is not a declaration at all.
    if !super::instr::use_when_holds(element, env)? {
        return Ok(());
    }
    let Some(lexical) = env.shadowed_attribute(element, "name")? else {
        return Err(XsltError::stat1c(
            "XTSE0010",
            "a static xsl:variable or xsl:param needs a name attribute",
        ));
    };
    let name = env.resolve_attribute_name(&lexical)?;
    let is_parameter = local == "param";
    // XSLT 3.0 §9.6: "When the attribute `static="yes"` is specified, the `xsl:variable` or
    // `xsl:param` element must have empty content. In the case of `xsl:variable` the `select`
    // attribute must be present to define the value of the variable." A value that is not
    // available before the stylesheet is compiled is no use to a `use-when` expression, and a
    // sequence constructor is exactly such a value.
    if element.children().into_iter().any(|child| {
        !matches!(child.node_kind(), NodeKind::Text)
            || !super::is_xml_whitespace(&child.string_value())
    }) {
        return Err(XsltError::stat1c(
            "XTSE0010",
            format!("the static `{name}` must have empty content"),
        ));
    }
    if !is_parameter && env.shadowed_attribute(element, "select")?.is_none() {
        return Err(XsltError::stat1c(
            "XTSE0010",
            format!("the static variable `{name}` requires a select attribute"),
        ));
    }
    let supplied = env
        .statics()
        .borrow()
        .supplied(&name)
        .filter(|_| is_parameter);
    // §9.7 compiles the expression whether or not its value is wanted: which variables are in
    // scope is a *static* constraint, so `error-0640o-2` supplies the parameter externally and
    // still expects the forward reference in the now-unused default to be reported.
    let default = match env.shadowed_attribute(element, "select")? {
        Some(text) => Some(evaluate(&text, env)?),
        None => None,
    };
    let value = match (supplied, default) {
        (Some(value), _) | (None, Some(value)) => value,
        (None, None)
            if is_parameter && super::instr::yes(&env.shadowed_attribute(element, "required")?) =>
        {
            return Err(XsltError::dynamic(
                "XTDE0050",
                format!("no value was supplied for the required static parameter `{name}`"),
            ));
        }
        // [XSLT 3.0 §9.3](https://www.w3.org/TR/xslt-30/#parameters) settles the implicit default
        // of a declaration with neither a `select` attribute nor content, and the two halves of
        // the rule are both measured: "the implicit default value is the empty sequence if there
        // is an `as` attribute" (`static-009` … `-011`, whose `as` clauses expect `empty($p)` to
        // be true and `10 * $p` to raise nothing), and otherwise the zero-length string
        // (`static-008`, which expects `empty($p)` to be false and `$p = ''` to hold).
        (None, None) => match env.shadowed_attribute(element, "as")? {
            Some(_) => XpathValue::default(),
            None => XpathValue::from(oxixml_infoset::AtomicValue::string("")),
        },
    };
    env.statics().borrow_mut().push(Binding {
        name,
        value,
        is_parameter,
        group,
    })
}

/// Whether two static values are identical in the sense [§9.6] requires.
///
/// The specification asks for *identical* values — the same type and the same value — and bars
/// function items from a static value outright. Anything that is not a sequence of atomic
/// values is therefore reported as differing rather than guessed at.
fn identical(left: &XpathValue, right: &XpathValue) -> bool {
    let (Ok(left), Ok(right)) = (left.clone().into_sequence(), right.clone().into_sequence())
    else {
        return false;
    };
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(left, right)| match (left, right) {
            (Item::Atomic(left), Item::Atomic(right)) => {
                left.atomic_type() == right.atomic_type()
                    && left.string_value() == right.string_value()
            }
            _ => false,
        })
}

/// Evaluates a static expression against the bindings collected so far.
///
/// # Errors
///
/// `XPST0008` when the expression refers to a variable that is not a previously declared static
/// variable, plus whatever the expression itself raises.
pub(crate) fn evaluate(text: &str, env: &StaticEnv) -> XsltResult<XpathValue> {
    let names = env.statics().borrow().names();
    let compiled = env.compile_expression_in_scope(text, &names)?;
    let context = env.bind_statics(compiled.context());
    // A static expression may ask what is available, and the availability functions expand the
    // name they are given against the namespaces in scope for the expression they are written
    // in — which is all the context there is before a transformation exists.
    let _scope = crate::run::functions::enter_static_expression(std::sync::Arc::clone(
        compiled.static_context(),
    ));
    compiled
        .evaluate_value_in(&context)
        .map_err(|error| XsltError::from(error).as_static())
}

/// Expands a shadow attribute's value template.
///
/// §3.14.2 makes the value of a shadow attribute "a value template in which all expressions
/// enclosed between curly braces must be static expressions", so it is expanded exactly like an
/// attribute value template — doubled braces and all — except that the expressions inside it are
/// evaluated now, against the static variables, rather than against a source document later.
///
/// # Errors
///
/// `XTSE0350` or `XTSE0370` for a malformed value template, plus every static error an embedded
/// expression raises.
pub(crate) fn expand_shadow(text: &str, env: &StaticEnv) -> XsltResult<String> {
    let mut out = String::with_capacity(text.len());
    for piece in crate::avt::parse(text)? {
        match piece {
            crate::avt::AvtPiece::Literal(literal) => out.push_str(&literal),
            crate::avt::AvtPiece::Expression(expression) => {
                let value = evaluate(&expression, env)?;
                out.push_str(&string_value(&value)?);
            }
        }
    }
    Ok(out)
}

/// The string an expanded static expression contributes to a shadow attribute: its items'
/// string values, separated by single spaces, as for an attribute value template.
fn string_value(value: &XpathValue) -> XsltResult<String> {
    let sequence = value.clone().into_sequence().map_err(XsltError::from)?;
    let mut parts = Vec::with_capacity(sequence.len());
    for item in sequence.iter() {
        parts.push(match item {
            Item::Atomic(atomic) => atomic.string_value(),
            Item::Node(node) => node.string_value(),
            _ => {
                return Err(XsltError::stat1c(
                    "XTSE0350",
                    "a shadow attribute may not be given a function item",
                ));
            }
        });
    }
    Ok(parts.join(" "))
}

/// The value an XSLT element's attribute has once shadow attributes are applied.
///
/// §3.14.2 restricts the mechanism to "attributes in the stylesheet where the attribute name is
/// in no namespace and the name of the parent element is in the XSLT namespace": an `_x`
/// attribute on a literal result element is an ordinary attribute called `_x`, and must be
/// copied through untouched.
///
/// # Errors
///
/// Every static error expanding the shadow attribute raises.
pub(crate) fn shadowed_attribute(
    element: &NodeRef,
    name: &str,
    env: &StaticEnv,
) -> XsltResult<Option<String>> {
    if crate::names::is_xslt(element)
        && let Some(text) = shadow_text(element, name)
    {
        // "If a shadow attribute and its corresponding target attribute are both present in the
        // stylesheet, the non-shadow attribute is ignored."
        return expand_shadow(&text, env).map(Some);
    }
    Ok(attribute(element, name))
}

/// The unexpanded text of the shadow attribute that stands in for `name`, if there is one.
///
/// Written out rather than deferring to [`attribute`] with a built-up `_name` so that reading
/// an attribute — which happens several times for every element of every stylesheet — costs no
/// allocation when there is no shadow attribute, which is the overwhelmingly common case.
fn shadow_text(element: &NodeRef, name: &str) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        let attribute_name = attribute.node_name()?;
        if attribute_name.namespace().is_some() {
            return None;
        }
        let shadowed = attribute_name.local_name().strip_prefix('_')?;
        (shadowed == name).then(|| attribute.string_value())
    })
}
