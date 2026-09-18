//! [`render_message`] — interpolates a compiled `sch:assert`/`sch:report`/`sch:diagnostic`
//! message against a firing context.

use oxixml_infoset::{NodeKind, NodeRef, XdmNode};
use oxixml_xpath::DynamicContext;

use crate::compile::CompiledMessagePart;
use crate::error::{SchematronError, SchematronResult};
use crate::location::prefixed_name;

/// Renders `parts`, evaluating `sch:value-of` and `sch:name` against `dynamic_context` (which
/// must already have `node` as its focus and every visible `let` bound).
///
/// # Errors
///
/// [`SchematronError::Evaluation`] if a `sch:value-of` or `sch:name` expression fails at run
/// time.
pub(crate) fn render_message(
    parts: &[CompiledMessagePart],
    dynamic_context: &DynamicContext,
    node: &NodeRef,
) -> SchematronResult<String> {
    let mut out = String::new();
    for part in parts {
        match part {
            CompiledMessagePart::Text(text) => out.push_str(text),
            CompiledMessagePart::ValueOf(expr) => {
                let value = expr
                    .evaluate_to_string_in(dynamic_context)
                    .map_err(|error| SchematronError::Evaluation(Box::new(error)))?;
                out.push_str(&value);
            }
            CompiledMessagePart::Name(select) => {
                let named = match select {
                    Some(expr) => expr
                        .evaluate_to_nodes_in(dynamic_context)
                        .map_err(|error| SchematronError::Evaluation(Box::new(error)))?
                        .into_iter()
                        .next(),
                    None => Some(node.clone()),
                };
                if let Some(named) = named {
                    if let Some(name) = named.node_name() {
                        let scope = if named.node_kind() == NodeKind::Attribute {
                            named.parent().unwrap_or_else(|| named.clone())
                        } else {
                            named.clone()
                        };
                        out.push_str(&prefixed_name(&scope, &name));
                    }
                }
            }
        }
    }
    Ok(out)
}
