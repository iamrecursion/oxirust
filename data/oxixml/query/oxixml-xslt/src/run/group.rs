//! `xsl:for-each-group`: the four ways
//! [XSLT 3.0 §14](https://www.w3.org/TR/xslt-30/#grouping) partitions a population.
//!
//! * `group-by` — one group per distinct key value, in order of first appearance, and an item
//!   whose key expression yields several values joins every one of those groups;
//! * `group-adjacent` — a new group whenever the key changes between neighbours;
//! * `group-starting-with` — a new group at every item that matches the pattern;
//! * `group-ending-with` — a group ends at every item that matches the pattern.

use super::{Context, Transformer};
use crate::compile::Grouping;
use crate::error::XsltResult;
use oxixml_infoset::{AtomicValue, Item, Sequence};
use oxixml_xsd::TimezoneOffset;
use std::sync::Arc;

impl Transformer {
    /// Partitions `population` into `(grouping key, members)` pairs.
    ///
    /// # Errors
    ///
    /// Any dynamic error the key expression or the pattern raises.
    pub(crate) fn form_groups(
        &mut self,
        population: Vec<Item>,
        grouping: &Grouping,
        composite: bool,
        collation: Option<&str>,
        context: &Context,
    ) -> XsltResult<Vec<(Sequence, Vec<Item>)>> {
        // XSLT 3.0 §14.6: an unrecognized collation URI on `xsl:for-each-group` is `XTDE1110`.
        let collation = match collation {
            Some(uri) => oxixml_fn::collation::Collations::new()
                .resolve(uri)
                .map_err(|_| {
                    crate::error::XsltError::dynamic(
                        "XTDE1110",
                        format!("`{uri}` is not a collation this processor recognizes"),
                    )
                })?,
            None => Arc::clone(self.runtime.fn_context.default_collation()),
        };
        match grouping {
            Grouping::By(select) => {
                self.group_by(population, select, composite, &collation, context)
            }
            Grouping::Adjacent(select) => {
                self.group_adjacent(population, select, composite, &collation, context)
            }
            Grouping::StartingWith(pattern) => {
                self.group_at_boundary(population, pattern, context, true)
            }
            Grouping::EndingWith(pattern) => {
                self.group_at_boundary(population, pattern, context, false)
            }
        }
    }

    fn group_by(
        &mut self,
        population: Vec<Item>,
        select: &oxixml_xpath::CompiledXPath,
        composite: bool,
        collation: &Arc<dyn oxixml_fn::Collation>,
        context: &Context,
    ) -> XsltResult<Vec<(Sequence, Vec<Item>)>> {
        let size = population.len();
        let implicit_timezone = self.runtime.fn_context.implicit_timezone();
        let mut order: Vec<Vec<AtomicValue>> = Vec::new();
        let mut groups: Vec<(Sequence, Vec<Item>)> = Vec::new();
        for (offset, item) in population.into_iter().enumerate() {
            let scoped = context.with_focus(item.clone(), offset + 1, size);
            let keys = self.evaluate(select, &scoped)?.into_sequence()?.atomize()?;
            // XSLT 3.0 §14.2: with `composite="yes"` the whole atomized sequence is one
            // grouping key, so an item joins exactly one group whatever the key's length.
            if composite {
                let key = atomics(&keys);
                match order
                    .iter()
                    .position(|candidate| keys_equal(candidate, &key, collation, implicit_timezone))
                {
                    Some(index) => {
                        if let Some((_, members)) = groups.get_mut(index) {
                            members.push(item);
                        }
                    }
                    None => {
                        order.push(key);
                        groups.push((keys.clone(), vec![item]));
                    }
                }
                continue;
            }
            let mut seen: Vec<Vec<AtomicValue>> = Vec::new();
            for key in keys.iter() {
                let Some(atomic) = key.as_atomic() else {
                    continue;
                };
                let identity = vec![atomic.clone()];
                if seen
                    .iter()
                    .any(|candidate| keys_equal(candidate, &identity, collation, implicit_timezone))
                {
                    continue;
                }
                seen.push(identity.clone());
                match order.iter().position(|candidate| {
                    keys_equal(candidate, &identity, collation, implicit_timezone)
                }) {
                    Some(index) => {
                        if let Some((_, members)) = groups.get_mut(index) {
                            members.push(item.clone());
                        }
                    }
                    None => {
                        order.push(identity);
                        groups.push((Sequence::from(atomic.clone()), vec![item.clone()]));
                    }
                }
            }
        }
        Ok(groups)
    }

    fn group_adjacent(
        &mut self,
        population: Vec<Item>,
        select: &oxixml_xpath::CompiledXPath,
        composite: bool,
        collation: &Arc<dyn oxixml_fn::Collation>,
        context: &Context,
    ) -> XsltResult<Vec<(Sequence, Vec<Item>)>> {
        let size = population.len();
        let implicit_timezone = self.runtime.fn_context.implicit_timezone();
        let mut groups: Vec<(Sequence, Vec<Item>)> = Vec::new();
        let mut previous: Option<Vec<AtomicValue>> = None;
        for (offset, item) in population.into_iter().enumerate() {
            let scoped = context.with_focus(item.clone(), offset + 1, size);
            let keys = self.evaluate(select, &scoped)?.into_sequence()?.atomize()?;
            // XSLT 3.0 §14.3: without `composite="yes"` the key of `group-adjacent` must be
            // exactly one item.
            if !composite && keys.len() != 1 {
                return Err(crate::error::XsltError::dynamic(
                    "XTTE1100",
                    format!(
                        "the group-adjacent expression produced {} items, and composite=\"yes\" was not specified",
                        keys.len()
                    ),
                ));
            }
            let (identity, key) = if composite {
                (atomics(&keys), keys.clone())
            } else {
                let Some(atomic) = keys.first().and_then(oxixml_infoset::Item::as_atomic) else {
                    continue;
                };
                (vec![atomic.clone()], Sequence::from(atomic.clone()))
            };
            match (&previous, groups.last_mut()) {
                (Some(previous), Some((_, members)))
                    if keys_equal(previous, &identity, collation, implicit_timezone) =>
                {
                    members.push(item);
                }
                _ => {
                    groups.push((key, vec![item]));
                }
            }
            previous = Some(identity);
        }
        Ok(groups)
    }

    fn group_at_boundary(
        &mut self,
        population: Vec<Item>,
        pattern: &crate::pattern::Pattern,
        context: &Context,
        starting: bool,
    ) -> XsltResult<Vec<(Sequence, Vec<Item>)>> {
        let mut groups: Vec<(Sequence, Vec<Item>)> = Vec::new();
        let mut current: Vec<Item> = Vec::new();
        let dynamic = self.pattern_context_for(context);
        for item in population {
            // XSLT 3.0 §18.4: a pattern is evaluated with the captured substrings of
            // `fn:regex-group` empty, whatever `xsl:analyze-string` is in force around it.
            let matched = self.without_regex_groups(|_| match &item {
                Item::Node(node) => pattern.matches(node, &dynamic),
                other => pattern.matches_item(other, &dynamic),
            })?;
            if starting {
                if matched && !current.is_empty() {
                    groups.push((Sequence::empty(), std::mem::take(&mut current)));
                }
                current.push(item);
            } else {
                current.push(item);
                if matched {
                    groups.push((Sequence::empty(), std::mem::take(&mut current)));
                }
            }
        }
        if !current.is_empty() {
            groups.push((Sequence::empty(), current));
        }
        Ok(groups)
    }
}

/// The atomic values of a grouping key sequence.
fn atomics(keys: &Sequence) -> Vec<AtomicValue> {
    keys.iter()
        .filter_map(|item| item.as_atomic().cloned())
        .collect()
}

/// Whether two grouping keys are the same.
///
/// [XSLT 3.0 §14.2](https://www.w3.org/TR/xslt-30/#grouping) compares key values "using the
/// `fn:deep-equal` function with the collation in use", which is the `eq` operator with
/// `xs:untypedAtomic` treated as a string and two `NaN` values counted equal. Comparing by
/// value rather than by lexical form is what puts two `xs:dateTime` values written in different
/// time zones — or two `xs:QName` values written with different prefixes — in one group, and it
/// is also what makes the comparison follow XPath's numeric type promotion: an `xs:float` and
/// an `xs:decimal` are compared as `xs:float`, not as two `xs:double`s.
fn keys_equal(
    left: &[AtomicValue],
    right: &[AtomicValue],
    collation: &Arc<dyn oxixml_fn::Collation>,
    implicit_timezone: TimezoneOffset,
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| values_equal(a, b, collation, implicit_timezone))
}

fn values_equal(
    left: &AtomicValue,
    right: &AtomicValue,
    collation: &Arc<dyn oxixml_fn::Collation>,
    implicit_timezone: TimezoneOffset,
) -> bool {
    // `fn:deep-equal` counts two `NaN` values equal even though `NaN eq NaN` is false.
    if left.is_nan() && right.is_nan() {
        return true;
    }
    // A collation decides string comparison and nothing else, so it is applied through the
    // collation object the instruction resolved — which knows tailorings a URI-driven lookup
    // inside the value comparison need not.
    if is_string_like(left) && is_string_like(right) {
        return collation.equal(&left.string_value(), &right.string_value());
    }
    // Two values of unrelated types are `XPTY0004` under `eq`; `fn:deep-equal` reports them as
    // simply different, which is what puts them in separate groups.
    oxixml_infoset::atomic_equal_in_context(left, right, None, Some(implicit_timezone))
        .unwrap_or(false)
}

/// Whether `eq` compares a value as a string, which is true of `xs:string`, `xs:anyURI`,
/// `xs:untypedAtomic` and their restrictions, and of nothing else.
fn is_string_like(value: &AtomicValue) -> bool {
    oxixml_infoset::atomic_equal(value, &AtomicValue::string(value.string_value())).unwrap_or(false)
}

impl Transformer {
    /// A dynamic context for matching a pattern against a node, sharing the engine's globals.
    pub(crate) fn pattern_context_for(&self, context: &Context) -> oxixml_xpath::DynamicContext {
        // The globals are already bound on the context the runtime shares, which is built over
        // the very static context a pattern is evaluated against — see
        // [`crate::run::Runtime::globals_context`].
        let mut dynamic = self
            .runtime
            .globals_bound_context()
            .with_fn_context(self.runtime.fn_context.clone());
        if let Some(item) = &context.item {
            dynamic = dynamic.with_context_item(item.clone());
        }
        if let Some(resolver) = &self.runtime.xpath_resolver {
            dynamic = dynamic.with_document_resolver(std::sync::Arc::clone(resolver));
        }
        for (name, value) in context.bindings.visible().into_iter().rev() {
            dynamic = dynamic.with_variable_name(name.clone(), value.clone());
        }
        dynamic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codepoint() -> Arc<dyn oxixml_fn::Collation> {
        oxixml_fn::collation::Collations::new().codepoint()
    }

    #[test]
    fn numeric_keys_compare_by_value() {
        let one = vec![AtomicValue::integer(1_i64)];
        let also_one = vec![AtomicValue::double(1.0)];
        assert!(keys_equal(
            &one,
            &also_one,
            &codepoint(),
            TimezoneOffset::UTC
        ));
    }

    #[test]
    fn composite_keys_compare_component_by_component() {
        let collation = codepoint();
        let left = vec![AtomicValue::string("a"), AtomicValue::string("bc")];
        let right = vec![AtomicValue::string("ab"), AtomicValue::string("c")];
        assert!(!keys_equal(&left, &right, &collation, TimezoneOffset::UTC));
        assert!(keys_equal(
            &left,
            &left.clone(),
            &collation,
            TimezoneOffset::UTC
        ));
    }
}
