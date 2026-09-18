//! Template-rule dispatch: finding the rule a node or atomic value matches, instantiating a
//! template, and binding its parameters and the built-in rules.

use super::super::{Bindings, Context, Emitter, Transformer};
use super::Flow;
use crate::compile::{
    Instruction, ModeRef, NoMatchPolicy, ParamDecl, SortKey, Template, WithParam,
};
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{ExpandedName, Item, NodeKind, NodeRef, Sequence, XdmNode};
use oxixml_xpath::XpathValue;
use std::sync::Arc;

/// Whether a template rule carries a `priority` attribute, which keeps a union pattern one
/// rule rather than several.
fn has_declared_priority(template: &Template) -> bool {
    let Some(pattern) = &template.pattern else {
        return false;
    };
    pattern
        .alternative_priorities()
        .get(template.alternative)
        .is_none_or(|default| *default != template.priority)
}

/// The rule the conflict-resolution algorithm has settled on, kept while the search goes on
/// looking for a rule that ties with it — which only a mode declaring `on-multiple-match="fail"`
/// asks for.
struct ChosenRule {
    /// The rule's position in the mode's rule list, which `xsl:next-match` resumes from.
    position: usize,
    /// The index of the template in the stylesheet.
    index: usize,
    import_precedence: usize,
    priority: f64,
    /// The `xsl:template` the rule came from, as `(import precedence, declaration order)`. Two
    /// alternatives of one union pattern share it, and are therefore not in conflict.
    template: (usize, usize),
}

/// The band of import precedences `xsl:apply-imports` may draw a template rule from.
///
/// [XSLT 3.0 §6.8](https://www.w3.org/TR/xslt-30/#apply-imports) restricts the search to
/// "template rules contained in stylesheet levels that are **descendants in the import tree** of
/// the stylesheet level that contains the current template rule", and warns in the same breath
/// that "this is not the same as saying that the search considers all template rules whose import
/// precedence is lower than that of the current template rule".
///
/// The two coincide only for the principal module. A module's own import subtree is a *contiguous*
/// band of precedences — the compiler numbers modules in post-order, so every descendant of a
/// module is numbered before it and after every module outside the subtree — which is why a
/// `floor` and a `limit` express the rule exactly. See `StylesheetIr::import_subtree_floor`.
#[derive(Debug, Clone, Copy)]
struct PrecedenceWindow {
    /// The lowest precedence in the current rule's import subtree.
    floor: usize,
    /// The precedence of the current rule, which the search stays strictly below: a rule that
    /// matched once would otherwise match again and the instruction would not terminate.
    limit: usize,
}

impl PrecedenceWindow {
    /// Whether a rule of this import precedence is in the band.
    fn admits(self, precedence: usize) -> bool {
        precedence >= self.floor && precedence < self.limit
    }
}

impl Transformer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_templates(
        &mut self,
        select: Option<&oxixml_xpath::CompiledXPath>,
        mode: &ModeRef,
        sort: &[SortKey],
        params: &[WithParam],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let items: Vec<Item> = match select {
            Some(select) => self
                .evaluate(select, context)?
                .into_sequence()?
                .into_items(),
            None => {
                // XSLT 3.0 §6.3: `xsl:apply-templates` with no `select` needs a node in focus.
                let Some(Item::Node(node)) = &context.item else {
                    return Err(XsltError::dynamic(
                        "XTTE0510",
                        "xsl:apply-templates has no select attribute and the context item is not a node",
                    ));
                };
                node.children().into_iter().map(Item::Node).collect()
            }
        };
        let items = if sort.is_empty() {
            items
        } else {
            self.sort_items(items, sort, context)?
        };
        let (bindings, tunnel) = self.parameter_bindings(params, context)?;
        let mode_key = self.ir().resolve_mode(mode, &context.mode);
        let size = items.len();
        for (offset, item) in items.into_iter().enumerate() {
            let mut scoped = context.with_focus(item, offset + 1, size);
            scoped.mode = mode_key.clone();
            scoped.bindings = Bindings::default();
            scoped.tunnel = tunnel.clone();
            scoped.rule = None;
            self.apply_one(&scoped, &bindings, out, 0)?;
        }
        Ok(Flow::Normal)
    }

    /// Applies the template rules to the initial context node, which is how a transformation
    /// starts when no initial template was named.
    ///
    /// `supplied` carries the non-tunnel parameters the invocation states —
    /// [XSLT 3.0 §2.3.3](https://www.w3.org/TR/xslt-30/#invoking-initial-mode) lets the *apply
    /// templates* entry point supply the initial mode's parameters, and `bind_parameters_over`
    /// reads them from here. The tunnel half travels on the context instead.
    ///
    /// # Errors
    ///
    /// Any dynamic error the matched rule raises.
    pub(crate) fn apply_to_initial(
        &mut self,
        context: &Context,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<()> {
        // XSLT 3.0 §4.4 strips whitespace from every document the transformation reads, the
        // initial match selection included: `mode-1802` supplies nodes of a document the engine
        // has not seen and reads `/*/node()[2]` from the tree they belong to.
        let scoped = match &context.item {
            Some(item) => {
                let mut scoped = context.clone();
                scoped.item = Some(self.stripped_selection_item(item));
                scoped
            }
            None => context.clone(),
        };
        self.apply_one(&scoped, supplied, out, 0)?;
        Ok(())
    }

    /// Applies the best-matching rule at or after `start` in the mode's rule list.
    fn apply_one(
        &mut self,
        context: &Context,
        supplied: &Bindings,
        out: &mut Emitter,
        start: usize,
    ) -> XsltResult<Flow> {
        self.check_declared_typing(context)?;
        let node = match &context.item {
            Some(Item::Node(node)) => node.clone(),
            // XSLT 3.0 §6.3 lets any item be the one a template rule is applied to — an atomic
            // value, a map, an array, a function item — and §5.5.2's predicate pattern is the one
            // form that can match one. The built-in rule decides when none does.
            Some(item) => {
                let item = item.clone();
                return match self.find_item_rule(context, &item, start, None, None)? {
                    Some((position, index)) => {
                        let mut scoped = context.clone();
                        scoped.rule = Some(position);
                        self.instantiate(index, &scoped, supplied, out)
                    }
                    None => self.built_in_item_rule(context, &item, supplied, out),
                };
            }
            None => return Ok(Flow::Normal),
        };
        match self.find_rule(context, &node, start, None, None)? {
            Some((position, index)) => {
                let mut scoped = context.clone();
                scoped.rule = Some(position);
                self.instantiate(index, &scoped, supplied, out)
            }
            None => self.built_in_rule(context, &node, supplied, out),
        }
    }

    /// Enforces `xsl:mode/@typed` on the item a template rule is about to be applied to.
    ///
    /// [XSLT 3.0 §6.6.2](https://www.w3.org/TR/xslt-30/#mode-declarations) lets a mode declare
    /// that the nodes processed in it carry type annotations. Applying such a mode to an element
    /// or attribute that was never validated is the type error `XTTE3100`; `mode-1439` declares
    /// `typed="yes"` over an ordinary source document and expects exactly that.
    ///
    /// The converse declaration, `typed="no"`, is not enforced here because it cannot be
    /// violated: this processor annotates no node with anything but `xs:untyped`.
    ///
    /// # Errors
    ///
    /// `XTTE3100`.
    fn check_declared_typing(&self, context: &Context) -> XsltResult<()> {
        let ir = &self.runtime.ir;
        if !ir
            .modes
            .get(&context.mode)
            .is_some_and(|mode| mode.declaration.typed)
        {
            return Ok(());
        }
        let Some(Item::Node(node)) = &context.item else {
            return Ok(());
        };
        if !matches!(node.node_kind(), NodeKind::Element | NodeKind::Attribute) {
            return Ok(());
        }
        const XSD_NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema";
        let annotated = node.type_name().is_some_and(|name| {
            !matches!(
                (name.namespace(), name.local_name()),
                (Some(XSD_NAMESPACE), "untyped" | "untypedAtomic")
            )
        });
        if annotated {
            return Ok(());
        }
        Err(XsltError::dynamic(
            "XTTE3100",
            "the mode is declared typed=\"yes\", and the node being processed is untyped",
        ))
    }

    /// Whether the mode in force turns an ambiguous template-rule match into an error.
    fn mode_fails_on_multiple_match(&self, context: &Context) -> bool {
        self.runtime
            .ir
            .modes
            .get(&context.mode)
            .is_some_and(|mode| mode.declaration.on_multiple_match_fails)
    }

    /// The error a mode declaring `on-multiple-match="fail"` raises for an ambiguous match.
    fn multiple_match_error() -> XsltError {
        XsltError::dynamic(
            "XTDE0540",
            "more than one template rule matches, and the mode declares \
             on-multiple-match=\"fail\"",
        )
    }

    /// The first rule of `mode` at or after `start` whose pattern an item that is not a node
    /// matches.
    ///
    /// `below` keeps the search under an import precedence, which is what `xsl:apply-imports`
    /// asks for; `besides` names one `xsl:template` — as `(import precedence, declaration
    /// order)` — whose rules are skipped, keeping a union pattern's other alternatives out of an
    /// `xsl:next-match` chain.
    fn find_item_rule(
        &mut self,
        context: &Context,
        item: &Item,
        start: usize,
        below: Option<PrecedenceWindow>,
        besides: Option<(usize, usize)>,
    ) -> XsltResult<Option<(usize, usize)>> {
        let ir = Arc::clone(&self.runtime.ir);
        let Some(rules) = ir.modes.get(&context.mode) else {
            return Ok(None);
        };
        let fails_on_multiple_match = self.mode_fails_on_multiple_match(context);
        let mut chosen: Option<ChosenRule> = None;
        for (position, index) in rules.rules.iter().enumerate().skip(start) {
            let Some(template) = ir.templates.get(*index) else {
                continue;
            };
            if besides == Some((template.import_precedence, template.declaration_order)) {
                continue;
            }
            if let Some(window) = below
                && !window.admits(template.import_precedence)
            {
                continue;
            }
            // Once a rule has been chosen, only a rule the conflict-resolution algorithm ranks
            // equally with it can still be ambiguous, and the list is ordered best-first.
            if let Some(chosen) = &chosen
                && (template.import_precedence != chosen.import_precedence
                    || template.priority != chosen.priority)
            {
                break;
            }
            let Some(pattern) = &template.pattern else {
                continue;
            };
            let dynamic = self.pattern_context_for(context);
            if self.without_regex_groups(|_| {
                pattern.matches_item_alternative(template.alternative, item, &dynamic)
            })? {
                if let Some(chosen) = &chosen {
                    // Two alternatives of one `xsl:template`'s union pattern are compiled as
                    // two rules, but they are still one template rule and matching both of them
                    // is not a conflict at all (`mode-1516`).
                    if chosen.template != (template.import_precedence, template.declaration_order) {
                        return Err(Self::multiple_match_error());
                    }
                    continue;
                }
                if !fails_on_multiple_match {
                    return Ok(Some((position, *index)));
                }
                chosen = Some(ChosenRule {
                    position,
                    index: *index,
                    import_precedence: template.import_precedence,
                    priority: template.priority,
                    template: (template.import_precedence, template.declaration_order),
                });
            }
        }
        Ok(chosen.map(|chosen| (chosen.position, chosen.index)))
    }

    /// The first rule of `mode` at or after `start` that `node` matches.
    ///
    /// `below` keeps the search under an import precedence, which is what `xsl:apply-imports`
    /// asks for; `besides` names one `xsl:template` — as `(import precedence, declaration
    /// order)` — whose rules are skipped, which is how a union pattern's other alternatives are
    /// kept out of an `xsl:next-match` chain.
    fn find_rule(
        &mut self,
        context: &Context,
        node: &NodeRef,
        start: usize,
        below: Option<PrecedenceWindow>,
        besides: Option<(usize, usize)>,
    ) -> XsltResult<Option<(usize, usize)>> {
        let ir = Arc::clone(&self.runtime.ir);
        let Some(rules) = ir.modes.get(&context.mode) else {
            return Ok(None);
        };
        let fails_on_multiple_match = self.mode_fails_on_multiple_match(context);
        let mut context = context.clone();
        context.item = Some(Item::Node(node.clone()));
        let mut chosen: Option<ChosenRule> = None;
        for (position, index) in rules.rules.iter().enumerate().skip(start) {
            let Some(template) = ir.templates.get(*index) else {
                continue;
            };
            if besides == Some((template.import_precedence, template.declaration_order)) {
                continue;
            }
            if let Some(window) = below
                && !window.admits(template.import_precedence)
            {
                continue;
            }
            // Once a rule has been chosen, only a rule the conflict-resolution algorithm ranks
            // equally with it can still be ambiguous, and the list is ordered best-first.
            if let Some(chosen) = &chosen
                && (template.import_precedence != chosen.import_precedence
                    || template.priority != chosen.priority)
            {
                break;
            }
            let Some(pattern) = &template.pattern else {
                continue;
            };
            let dynamic = self.pattern_context_for(&context);
            if self.without_regex_groups(|_| {
                pattern.matches_alternative(template.alternative, node, &dynamic)
            })? {
                // XSLT 3.0 §6.4: the conflict-resolution algorithm leaving more than one rule is
                // recovered from by taking the last of them, unless the mode says otherwise.
                if let Some(chosen) = &chosen {
                    // Two alternatives of one `xsl:template`'s union pattern are compiled as
                    // two rules, but they are still one template rule and matching both of them
                    // is not a conflict at all (`mode-1516`).
                    if chosen.template != (template.import_precedence, template.declaration_order) {
                        return Err(Self::multiple_match_error());
                    }
                    continue;
                }
                if !fails_on_multiple_match {
                    return Ok(Some((position, *index)));
                }
                chosen = Some(ChosenRule {
                    position,
                    index: *index,
                    import_precedence: template.import_precedence,
                    priority: template.priority,
                    template: (template.import_precedence, template.declaration_order),
                });
            }
        }
        Ok(chosen.map(|chosen| (chosen.position, chosen.index)))
    }

    pub(super) fn apply_next(
        &mut self,
        params: &[WithParam],
        context: &Context,
        out: &mut Emitter,
        imports_only: bool,
    ) -> XsltResult<Flow> {
        // XSLT 3.0 §6.7: `xsl:apply-imports` and `xsl:next-match` need a current template
        // rule; without one they are `XTDE0560` rather than a fresh search from the top.
        if context.rule.is_none() {
            return Err(XsltError::dynamic(
                "XTDE0560",
                "there is no current template rule for xsl:apply-imports or xsl:next-match",
            ));
        }
        let (bindings, tunnel) = self.parameter_bindings(params, context)?;
        let start = context.rule.map_or(0, |position| position + 1);
        let ir = Arc::clone(&self.runtime.ir);
        // The rule the instruction is written in, as an index into `ir.templates`. XSLT 3.0
        // §6.4 treats a union pattern as several template rules, one per alternative, but they
        // are alternatives of *one* `xsl:template`: an `xsl:next-match` written inside it moves
        // on to another template, never back into the one it is written in through a second
        // alternative that the same item also matches (the suite's next-match-024).
        let current = context
            .rule
            .and_then(|position| ir.modes.get(&context.mode).map(|mode| (position, mode)))
            .and_then(|(position, mode)| mode.rules.get(position).copied())
            .and_then(|index| ir.templates.get(index));
        // XSLT 3.0 §6.4: "if the match pattern is a union pattern and no `priority`
        // attribute is specified, the template rule is treated as if it were several template
        // rules, one for each alternative, each with its own default priority". With a
        // `priority` there is no such split, so an `xsl:next-match` written in such a template
        // moves on to another template rather than back into a second alternative of the one it
        // is written in (the suite's next-match-024 against next-match-021/023/039, which do
        // want the split). What tells the two apart at run time is that a declared priority
        // displaces the alternative's own default.
        let besides = current
            .filter(|template| has_declared_priority(template))
            .map(|template| (template.import_precedence, template.declaration_order));
        // XSLT 3.0 §6.8: `xsl:apply-imports` looks only within the import subtree of the
        // stylesheet level the rule it is written in belongs to, and strictly below that rule's
        // own import precedence — an item matched there is matched again, so without the upper
        // bound the search would find the very rule that is running.
        let below = if imports_only {
            current.map(|template| PrecedenceWindow {
                floor: ir.import_subtree_floor(template.import_precedence),
                limit: template.import_precedence,
            })
        } else {
            None
        };
        let node = match &context.item {
            Some(Item::Node(node)) => node.clone(),
            Some(item) => {
                let item = item.clone();
                return match self.find_item_rule(context, &item, start, below, besides)? {
                    Some((position, index)) => {
                        let mut scoped = context.clone();
                        scoped.rule = Some(position);
                        scoped.bindings = Bindings::default();
                        scoped.tunnel = tunnel;
                        self.instantiate(index, &scoped, &bindings, out)
                    }
                    None => {
                        let mut scoped = context.clone();
                        scoped.tunnel = tunnel;
                        self.built_in_item_rule(&scoped, &item, &bindings, out)
                    }
                };
            }
            None => return Ok(Flow::Normal),
        };
        match self.find_rule(context, &node, start, below, besides)? {
            Some((position, index)) => {
                let mut scoped = context.clone();
                scoped.rule = Some(position);
                scoped.bindings = Bindings::default();
                scoped.tunnel = tunnel;
                self.instantiate(index, &scoped, &bindings, out)
            }
            None => {
                // The built-in rule that `xsl:next-match` falls back on still receives the
                // parameters the instruction supplied, tunnel ones included.
                let mut scoped = context.clone();
                scoped.tunnel = tunnel;
                self.built_in_rule(&scoped, &node, &bindings, out)
            }
        }
    }

    pub(super) fn call_template(
        &mut self,
        name: &ExpandedName,
        params: &[WithParam],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let ir = Arc::clone(&self.runtime.ir);
        let Some(index) = ir.named_templates.get(name).copied() else {
            return Err(XsltError::dynamic(
                "XTSE0650",
                format!("no template named `{name}` is declared"),
            ));
        };
        let (bindings, tunnel) = self.parameter_bindings(params, context)?;
        let mut scoped = context.clone();
        scoped.bindings = Bindings::default();
        scoped.tunnel = tunnel;
        self.instantiate(index, &scoped, &bindings, out)
    }

    /// Instantiates the template at `index` with the supplied non-tunnel parameters.
    pub(crate) fn instantiate(
        &mut self,
        index: usize,
        context: &Context,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let ir = Arc::clone(&self.runtime.ir);
        let Some(template) = ir.templates.get(index) else {
            return Ok(Flow::Normal);
        };
        self.enter()?;
        let result = self.instantiate_body(template, context, supplied, out);
        self.leave();
        result
    }

    fn instantiate_body(
        &mut self,
        template: &Template,
        context: &Context,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let mut scoped = context.clone();
        // The focus the `xsl:context-item` declaration asks for is settled before the
        // parameters are bound: with `use="absent"`, XSLT 3.0 §10.1.1 evaluates "the contained
        // sequence constructor, *and any xsl:param elements*, with an absent focus".
        super::super::context_item::apply_context_item(
            template.context_item.as_deref(),
            &mut scoped,
        )?;
        scoped.bindings = self.bind_parameters(&template.params, supplied, &scoped)?;
        self.runtime.current.borrow_mut().push(scoped.item.clone());
        let result = match &template.as_type {
            // XSLT 3.0 §6.7: with `as` on `xsl:template` the body's result is converted to
            // the required type before it is written, and a failure is `XTTE0505`.
            Some(required) if required.is_checked() => self.typed_body(
                &template.content,
                required,
                "XTTE0505",
                "template",
                &scoped,
                out,
            ),
            _ => self.run_sequence(&template.content, &scoped, out),
        };
        self.runtime.current.borrow_mut().pop();
        result
    }

    /// Runs a body whose declared type must be checked: the result is collected, converted and
    /// only then written to `out`.
    fn typed_body(
        &mut self,
        content: &[Instruction],
        required: &crate::typing::RequiredType,
        code: &str,
        what: &str,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        // Collecting the body is only a means of checking it: the body still writes to the
        // destination the caller was writing to, so it is not in temporary output state. The
        // body of a declared-type template, function or variable is not an `xsl:iterate` body,
        // so no `xsl:break` or `xsl:next-iteration` can end it.
        let (_, _, sequence) = self.run_to_sequence_in_place(content, context)?;
        let value = required.convert(
            XpathValue::from(sequence),
            code,
            &format!("the result of the {what}"),
        )?;
        for item in value.into_sequence()?.iter() {
            out.item(item)?;
        }
        Ok(Flow::Normal)
    }

    /// Binds a template's declared parameters from the supplied values, falling back to each
    /// parameter's own default.
    pub(crate) fn bind_parameters(
        &mut self,
        declared: &[ParamDecl],
        supplied: &Bindings,
        context: &Context,
    ) -> XsltResult<Bindings> {
        self.bind_parameters_over(declared, supplied, context, &Bindings::default())
    }

    /// Binds the declared parameters on top of `base`, which stays visible to their default
    /// values. `xsl:iterate` needs this: its parameters and its body see the variables in
    /// scope where the instruction is written, unlike a template's.
    ///
    /// # Errors
    ///
    /// `XTDE0700` when a required parameter has no value, plus anything a default raises.
    pub(crate) fn bind_parameters_over(
        &mut self,
        declared: &[ParamDecl],
        supplied: &Bindings,
        context: &Context,
        base: &Bindings,
    ) -> XsltResult<Bindings> {
        let mut bindings = base.clone();
        for parameter in declared {
            let source = if parameter.tunnel {
                context.tunnel.get(&parameter.name)
            } else {
                supplied.get(&parameter.name)
            };
            // XSLT 3.0 §9.3: a value a caller *supplied* converts to the declared type under
            // `XTTE0590`; the parameter's own *default* — used only when nothing was supplied —
            // converts under the distinct `XTTE0600` instead, since a default that cannot
            // satisfy its own parameter's type is the declaration's fault, not the caller's
            // (the suite's error-0600a turns on exactly this distinction).
            let (mut value, conversion_code, description) = match source {
                Some(value) => (
                    value.clone(),
                    "XTTE0590",
                    format!("the value of the parameter `{}`", parameter.name),
                ),
                // XSLT 3.0 §9.3 makes a parameter with no explicit default whose `as` does not
                // admit the empty sequence *implicitly mandatory* — "treated as if
                // required="yes" had been specified" — so an unsupplied value is `XTDE0700`,
                // the missing-argument error, not `XTTE0600`, the bad-default one.
                None if parameter.required || parameter.is_implicitly_mandatory() => {
                    return Err(XsltError::dynamic(
                        "XTDE0700",
                        format!(
                            "no value was supplied for the parameter `{}`",
                            parameter.name
                        ),
                    ));
                }
                None => {
                    let mut scoped = context.clone();
                    scoped.bindings = bindings.clone();
                    let default = self.bound_value(
                        parameter.select.as_ref(),
                        &parameter.content,
                        parameter.as_type.is_some(),
                        parameter.base_uri.as_deref(),
                        &scoped,
                    )?;
                    (
                        default,
                        "XTTE0600",
                        format!("the default value of the parameter `{}`", parameter.name),
                    )
                }
            };
            if let Some(required) = &parameter.as_type {
                value = required.convert(value, conversion_code, &description)?;
            }
            bindings = bindings.bind(parameter.name.clone(), value);
        }
        Ok(bindings)
    }

    /// Evaluates `xsl:with-param`, splitting tunnel parameters from ordinary ones.
    fn parameter_bindings(
        &mut self,
        params: &[WithParam],
        context: &Context,
    ) -> XsltResult<(Bindings, Bindings)> {
        let mut ordinary = Bindings::default();
        let mut tunnel = context.tunnel.clone();
        for parameter in params {
            let value = self.with_param_value(parameter, context)?;
            if parameter.tunnel {
                tunnel = tunnel.bind(parameter.name.clone(), value);
            } else {
                ordinary = ordinary.bind(parameter.name.clone(), value);
            }
        }
        Ok((ordinary, tunnel))
    }

    pub(crate) fn with_param_value(
        &mut self,
        parameter: &WithParam,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        if parameter.select.is_none() && parameter.content.is_empty() {
            // XSLT 3.0 §9.3: with neither a `select` attribute nor content the value supplied
            // is "the empty sequence if there is an `as` attribute, or a zero-length string if
            // not" — `built-in-templates-0201` writes `<xsl:with-param name="empty_string"/>`
            // and asserts `$empty_string eq ''`, which the empty sequence does not satisfy.
            return Ok(match parameter.as_type {
                Some(_) => XpathValue::from(Sequence::empty()),
                None => XpathValue::from(oxixml_infoset::AtomicValue::string("")),
            });
        }
        let value = self.bound_value(
            parameter.select.as_ref(),
            &parameter.content,
            parameter.as_type.is_some(),
            parameter.base_uri.as_deref(),
            context,
        )?;
        match &parameter.as_type {
            // XSLT 3.0 §10.1.1: `xsl:with-param` with `as` converts the value it supplies,
            // under the same code as the receiving `xsl:param`.
            Some(required) => required.convert(
                value,
                "XTTE0590",
                &format!("the value supplied for the parameter `{}`", parameter.name),
            ),
            None => Ok(value),
        }
    }

    /// The built-in template rules of
    /// [XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#built-in-rule).
    ///
    /// XSLT 3.0 §6.7.1 and §6.7.3: when the built-in rule for a document or element node
    /// recurses, "those parameters are passed on in the implicit `xsl:apply-templates`
    /// instruction", so `supplied` travels down with the recursion rather than being dropped.
    fn built_in_rule(
        &mut self,
        context: &Context,
        node: &NodeRef,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        match self.no_match_policy(context) {
            NoMatchPolicy::Fail => Err(XsltError::dynamic(
                "XTDE0555",
                "no template rule matches, and the mode declares on-no-match=\"fail\"",
            )),
            // XSLT 3.0 §6.7.4 states the document node apart from every other item: its rule
            // "is equivalent to calling xsl:apply-templates with no select attribute, and with
            // the mode attribute set to #current", while "the built-in rule for all items other
            // than document nodes … is to do nothing". It is the one policy under which a
            // document node and an element node behave differently, which is what
            // `mode-1437a` ("deep skip should not skip document node") is written for.
            NoMatchPolicy::DeepSkip => match node.node_kind() {
                NodeKind::Document => self.recurse_children(context, node, supplied, out, false),
                _ => Ok(Flow::Normal),
            },
            NoMatchPolicy::ShallowSkip => match node.node_kind() {
                NodeKind::Document | NodeKind::Element => {
                    self.recurse_children(context, node, supplied, out, true)
                }
                _ => Ok(Flow::Normal),
            },
            NoMatchPolicy::DeepCopy => {
                out.copy_node(node)?;
                Ok(Flow::Normal)
            }
            NoMatchPolicy::ShallowCopy => match node.node_kind() {
                NodeKind::Element => {
                    let name = node
                        .node_name()
                        .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
                    out.start_element(name, None);
                    // The built-in rule is `<xsl:copy>` over the node, so §11.9.1's rule that a
                    // copy keeps the original's base URI applies to it too.
                    out.note_copied_base_uri(node.base_uri());
                    for binding in node.namespaces() {
                        let prefix = binding.node_name().map(|n| n.local_name().to_owned());
                        if prefix.as_deref() == Some("xml") {
                            continue;
                        }
                        out.namespace(prefix, &binding.string_value())?;
                    }
                    let flow = self.recurse_children(context, node, supplied, out, true)?;
                    out.end_element()?;
                    Ok(flow)
                }
                // [XSLT 3.0 §6.7.3](https://www.w3.org/TR/xslt-30/#built-in-rule-shallow-copy)
                // makes the built-in rule an `xsl:copy` over every item, a document node
                // included, and §11.9.1 makes `xsl:copy` of a document node a document node of
                // its own — carrying the original's base URI, document URI and DTD declarations
                // — wherever one can stand, which is the top level of a sequence. `base-uri-053`
                // reads the base URI back off exactly such a copy. Inside constructed content a
                // document node is replaced by its children, so there the walk is the whole of
                // it.
                NodeKind::Document if out.wants_document_item() => {
                    let mut copy = Emitter::tree();
                    copy.carry_document_properties(node);
                    let flow = {
                        let _temporary = crate::run::TemporaryOutput::enter(&self.runtime);
                        self.recurse_children(context, node, supplied, &mut copy, true)?
                    };
                    let (tree, _) = copy.finish()?;
                    out.append_node(&tree.root())?;
                    Ok(flow)
                }
                NodeKind::Document => self.recurse_children(context, node, supplied, out, true),
                _ => {
                    out.copy_node(node)?;
                    Ok(Flow::Normal)
                }
            },
            NoMatchPolicy::TextOnlyCopy | NoMatchPolicy::DeepCopyText => match node.node_kind() {
                NodeKind::Document | NodeKind::Element => {
                    self.recurse_children(context, node, supplied, out, false)
                }
                NodeKind::Text | NodeKind::Attribute => {
                    out.text(&node.string_value(), false);
                    Ok(Flow::Normal)
                }
                _ => Ok(Flow::Normal),
            },
        }
    }

    /// The `on-no-match` policy of the mode in force, defaulting to the XSLT 1.0 built-in rules
    /// for a mode no `xsl:mode` declaration names.
    fn no_match_policy(&self, context: &Context) -> NoMatchPolicy {
        self.runtime
            .ir
            .modes
            .get(&context.mode)
            .map_or(NoMatchPolicy::DeepCopyText, |mode| {
                mode.declaration.on_no_match
            })
    }

    /// The built-in template rule for an item that is not a node
    /// ([XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#built-in-rule)).
    ///
    /// Every one of the six policies says what it does with an atomic value and with a function
    /// item (a map among them), and they do not agree. `text-only-copy` and the XSLT 1.0 rules
    /// write a text node holding an atomic value and nothing at all for a function;
    /// `deep-copy` and `shallow-copy` return the item itself, since `xsl:copy-of` and
    /// `xsl:copy` of an item that is not a node both answer that item; `deep-skip` and
    /// `shallow-skip` return the empty sequence for every item alike — which is what
    /// `match-256` asserts of the atomic value `93`.
    ///
    /// An **array** is the exception, and it behaves like an element rather than like a map:
    /// the policies that recurse into an element's children recurse into an array's members,
    /// `<xsl:template match=".[. instance of array(*)]"><xsl:apply-templates select="?*"/>`.
    /// The suite's `arrays-301` (the default rules) and `arrays-302` (`shallow-skip`) both
    /// apply templates to `array{$data}` and expect the four elements it holds to be matched.
    fn built_in_item_rule(
        &mut self,
        context: &Context,
        item: &Item,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let policy = self.no_match_policy(context);
        if let (
            Item::Array(array),
            NoMatchPolicy::TextOnlyCopy | NoMatchPolicy::DeepCopyText | NoMatchPolicy::ShallowSkip,
        ) = (item, policy)
        {
            return self.recurse_members(context, array, supplied, out);
        }
        match policy {
            NoMatchPolicy::Fail => Err(XsltError::dynamic(
                "XTDE0555",
                "no template rule matches, and the mode declares on-no-match=\"fail\"",
            )),
            NoMatchPolicy::DeepSkip | NoMatchPolicy::ShallowSkip => Ok(Flow::Normal),
            NoMatchPolicy::DeepCopy | NoMatchPolicy::ShallowCopy => {
                out.item(item)?;
                Ok(Flow::Normal)
            }
            NoMatchPolicy::TextOnlyCopy | NoMatchPolicy::DeepCopyText => match item {
                Item::Atomic(value) => {
                    out.atomic(value.clone())?;
                    Ok(Flow::Normal)
                }
                _ => Ok(Flow::Normal),
            },
        }
    }

    /// Applies the template rules to the items an array's members hold, in order.
    fn recurse_members(
        &mut self,
        context: &Context,
        array: &oxixml_infoset::XdmArray,
        supplied: &Bindings,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let items: Vec<Item> = array
            .members()
            .flat_map(|member| member.iter().cloned())
            .collect();
        let size = items.len();
        for (offset, member_item) in items.into_iter().enumerate() {
            let mut scoped = context.with_focus(member_item, offset + 1, size);
            scoped.rule = None;
            scoped.bindings = Bindings::default();
            self.apply_one(&scoped, supplied, out, 0)?;
        }
        Ok(Flow::Normal)
    }

    /// Applies the template rules to a node's children, and — when the policy says so — to its
    /// attributes first.
    fn recurse_children(
        &mut self,
        context: &Context,
        node: &NodeRef,
        supplied: &Bindings,
        out: &mut Emitter,
        include_attributes: bool,
    ) -> XsltResult<Flow> {
        if include_attributes {
            let attributes = node.attributes();
            let size = attributes.len();
            for (offset, attribute) in attributes.into_iter().enumerate() {
                let mut scoped = context.with_focus(Item::Node(attribute), offset + 1, size);
                scoped.rule = None;
                scoped.bindings = Bindings::default();
                self.apply_one(&scoped, supplied, out, 0)?;
            }
        }
        let children = node.children();
        let size = children.len();
        for (offset, child) in children.into_iter().enumerate() {
            let mut scoped = context.with_focus(Item::Node(child), offset + 1, size);
            scoped.rule = None;
            scoped.bindings = Bindings::default();
            self.apply_one(&scoped, supplied, out, 0)?;
        }
        Ok(Flow::Normal)
    }
}
