//! The iteration and grouping instructions, `xsl:iterate` and `xsl:for-each-group`.

use super::super::{Bindings, Context, Emitter, Transformer};
use super::Flow;
use crate::compile::{Grouping, Instruction, ParamDecl, SortKey};
use crate::error::XsltResult;
use oxixml_infoset::Sequence;
use oxixml_xpath::XpathValue;

impl Transformer {
    pub(super) fn iterate(
        &mut self,
        select: &oxixml_xpath::CompiledXPath,
        params: &[ParamDecl],
        content: &[Instruction],
        on_completion: &[Instruction],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let items = self
            .evaluate(select, context)?
            .into_sequence()?
            .into_items();
        let size = items.len();
        // The parameters and the body of `xsl:iterate` see the variables in scope where the
        // instruction is written, so they are bound on top of the current scope rather than in
        // a fresh one (XSLT 3.0 §8.8).
        let outer = context.bindings.clone();
        let mut carried =
            self.bind_parameters_over(params, &Bindings::default(), context, &outer)?;
        for (offset, item) in items.into_iter().enumerate() {
            let mut scoped = context.with_focus(item, offset + 1, size);
            scoped.bindings = carried.clone();
            match self.run_sequence(content, &scoped, out)? {
                Flow::Normal => {}
                Flow::Break(value) => {
                    for item in value.iter() {
                        out.item(item)?;
                    }
                    return Ok(Flow::Normal);
                }
                Flow::NextIteration(values) => {
                    let mut next = outer.clone();
                    for parameter in params {
                        let supplied = values
                            .iter()
                            .find(|(name, _)| *name == parameter.name)
                            .map(|(_, value)| value.clone());
                        let carried_over = supplied.is_none();
                        let mut value = supplied
                            .or_else(|| carried.get(&parameter.name).cloned())
                            .unwrap_or_else(|| XpathValue::from(Sequence::empty()));
                        // XSLT 3.0 §9.10: an `xsl:with-param` supplies its value "in the same
                        // way as for xsl:variable and xsl:param", so a value the
                        // `xsl:next-iteration` computes is converted to the type the
                        // `xsl:param` of the `xsl:iterate` declares — `@date` reaching an
                        // `as="xs:date?"` parameter is cast, not carried as an untyped node
                        // that a later `eq` would refuse to compare (the suite's iterate-027).
                        // A parameter that simply keeps its previous value was converted when
                        // it was first bound and is left alone.
                        if let Some(required) = &parameter.as_type
                            && !carried_over
                        {
                            value = required.convert(
                                value,
                                "XTTE0590",
                                &format!(
                                    "the value supplied for the parameter `{}`",
                                    parameter.name
                                ),
                            )?;
                        }
                        next = next.bind(parameter.name.clone(), value);
                    }
                    carried = next;
                }
            }
        }
        let mut completion = context.clone();
        completion.bindings = carried;
        completion.item = None;
        self.run_sequence(on_completion, &completion, out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_each_group(
        &mut self,
        select: &oxixml_xpath::CompiledXPath,
        grouping: &Grouping,
        composite: bool,
        collation: Option<&crate::compile::Avt>,
        sort: &[SortKey],
        content: &[Instruction],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let population = self
            .evaluate(select, context)?
            .into_sequence()?
            .into_items();
        let collation = match collation {
            Some(avt) => Some(self.evaluate_avt(avt, context)?),
            None => None,
        };
        let groups = self.form_groups(
            population,
            grouping,
            composite,
            collation.as_deref(),
            context,
        )?;
        let groups = if sort.is_empty() {
            groups
        } else {
            self.sort_groups(groups, sort, context)?
        };
        // XSLT 3.0 §14.5: `group-starting-with` and `group-ending-with` partition by position,
        // not by value, so inside them there is no current grouping key at all and
        // `fn:current-grouping-key()` is `XTDE1071` rather than an empty sequence.
        let by_value = !matches!(
            grouping,
            Grouping::StartingWith(_) | Grouping::EndingWith(_)
        );
        let size = groups.len();
        for (offset, (key, members)) in groups.into_iter().enumerate() {
            let first = members.first().cloned();
            self.runtime
                .groups
                .borrow_mut()
                .push((members.clone(), by_value.then(|| key.clone())));
            let mut scoped = context.clone();
            if let Some(item) = first {
                scoped = context.with_focus(item, offset + 1, size);
            }
            let flow = self.run_sequence(content, &scoped, out);
            self.runtime.groups.borrow_mut().pop();
            let flow = flow?;
            if !flow.is_normal() {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }
}
