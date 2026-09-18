//! Path expressions, axis steps, predicates and the simple map operator.
//!
//! Three rules do all the work here:
//!
//! * In `E1/E2`, `E2` is evaluated once for every item of `E1`, with that item as the context
//!   item and its ordinal as the context position. Every item of `E1` must be a node
//!   (`XPTY0019`), and the results must be either all nodes or all non-nodes (`XPTY0018`); when
//!   they are all nodes they are merged into document order with duplicates removed.
//! * A predicate is evaluated with the *sequence being filtered* as the focus, so `position()`
//!   counts within that sequence — which, on a reverse axis, counts in reverse document order.
//!   A predicate whose value is a single number is a positional test; anything else is tested by
//!   its effective boolean value.
//! * `E1!E2` is `E1/E2` without the node requirement and without the sort: it keeps duplicates
//!   and keeps the order in which the results were produced.

use super::effective_boolean_value;
use crate::ast::{AxisStep, Expr, NodeTest};
use crate::context::{DynamicContext, Focus};
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::value::{XpathItem, XpathValue};
use oxixml_infoset::{Axis, NodeRef, XdmNode, axis_nodes};

/// Evaluates the path-shaped expressions: [`Expr::Root`], [`Expr::Path`], [`Expr::Step`] and
/// [`Expr::SimpleMap`].
///
/// # Errors
///
/// `XPTY0019` when a path step is applied to a non-node, `XPTY0018` when a step mixes nodes and
/// non-nodes, `XPTY0020` when an axis step has a non-node context item, and `XPDY0050` when a
/// leading `/` is applied to a tree with no document node.
pub(crate) fn evaluate(expression: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    match expression {
        Expr::Root => root(context),
        Expr::Path(left, right) => path(left, right, context),
        Expr::Step(step) => step_value(step, context),
        Expr::SimpleMap(left, right) => simple_map(left, right, context),
        other => super::evaluate(other, context),
    }
}

fn root(context: &DynamicContext) -> XPathResult<XpathValue> {
    let node = context.context_node()?;
    let root = node.root();
    if root.node_kind() != oxixml_infoset::NodeKind::Document {
        return Err(XPathError::new(
            XPathErrorCode::Xpdy0050,
            "a leading `/` requires the context node to be in a tree rooted at a document node",
        ));
    }
    Ok(XpathValue::from(root))
}

/// Evaluates a path expression.
///
/// `/` is left-associative, so `a/b/c` is `Path(Path(a, b), c)` and a path of ten thousand steps
/// leans ten thousand deep to the left. The parser builds that chain with a loop and [`Expr`]'s
/// own `Drop` frees it with one; evaluating it recursively — once per step — would exhaust the
/// thread's stack on a path the grammar does not bound, so the left spine is flattened onto the
/// heap and the steps applied in a loop instead.
fn path(left: &Expr, right: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    // Collected outermost first, so applying them in reverse runs the steps left to right.
    let mut steps: Vec<&Expr> = vec![right];
    let mut base = left;
    while let Expr::Path(inner_left, inner_right) = base {
        steps.push(inner_right);
        base = inner_left;
    }
    let mut value = super::evaluate(base, context)?;
    let last = steps.len().saturating_sub(1);
    for (ordinal, step) in steps.into_iter().rev().enumerate() {
        value = step_over(value, step, context, ordinal == last)?;
    }
    Ok(value)
}

/// Applies one step of a path expression to the sequence the steps so far produced.
///
/// `last` says whether this is the final step: a step that mixes nodes and non-nodes is
/// `XPTY0018` there, but anywhere earlier its result is the operand of the following `/`, which
/// must be a sequence of nodes — `XPTY0019`.
fn step_over(
    input: XpathValue,
    right: &Expr,
    context: &DynamicContext,
    last: bool,
) -> XPathResult<XpathValue> {
    let size = input.len();
    let mut collected: Vec<XpathItem> = Vec::new();
    let mut saw_node = false;
    let mut saw_other = false;
    for (index, item) in input.iter().enumerate() {
        super::EvaluationLimits::check_deadline()?;
        let Some(node) = item.as_node() else {
            return Err(XPathError::new(
                XPathErrorCode::Xpty0019,
                format!(
                    "a path step needs a node on its left, but was given {}",
                    item.describe()
                ),
            ));
        };
        let focus = Focus::new(node.clone(), index + 1, size);
        let stepped = super::evaluate(right, &context.clone().with_focus(focus))?;
        for produced in stepped {
            if produced.is_node() {
                saw_node = true;
            } else {
                saw_other = true;
            }
            collected.push(produced);
        }
    }
    if saw_node && saw_other {
        if !last {
            return Err(XPathError::new(
                XPathErrorCode::Xpty0019,
                "a step of a path expression produced both nodes and non-nodes, and its result \
                 is the input of the following step",
            ));
        }
        return Err(XPathError::new(
            XPathErrorCode::Xpty0018,
            "the last step of a path expression produced both nodes and non-nodes",
        ));
    }
    if !saw_node {
        return Ok(XpathValue::from_items(collected));
    }
    let mut nodes: Vec<NodeRef> = collected
        .into_iter()
        .filter_map(|item| item.as_node().cloned())
        .collect();
    nodes.sort();
    nodes.dedup();
    Ok(XpathValue::from_iter(nodes))
}

fn simple_map(left: &Expr, right: &Expr, context: &DynamicContext) -> XPathResult<XpathValue> {
    let input = super::evaluate(left, context)?;
    let size = input.len();
    let mut out = XpathValue::empty();
    for (index, item) in input.iter().enumerate() {
        super::EvaluationLimits::check_deadline()?;
        let focus = Focus::new(item.clone(), index + 1, size);
        out.append(super::evaluate(right, &context.clone().with_focus(focus))?);
    }
    Ok(out)
}

fn step_value(step: &AxisStep, context: &DynamicContext) -> XPathResult<XpathValue> {
    let node = context.context_node()?;
    let mut nodes: Vec<NodeRef> = axis_nodes(node, step.axis)
        .into_iter()
        .filter(|candidate| matches_test(&step.node_test, candidate, step.axis))
        .collect();
    // The predicates see the axis's own order — reverse document order on a reverse axis — which
    // is what makes `ancestor::*[1]` the *nearest* ancestor.
    for predicate in &step.predicates {
        nodes = filter_nodes(nodes, predicate, context)?;
    }
    // XPath 3.1 §3.3.2: "the resulting node sequence is returned in document order". The axis's
    // own order governs only the numbering inside the step's predicates, so once those have run
    // the value itself is sorted. That is the difference the catalogue pins with parentheses:
    // `ancestor::*[1]` is the nearest ancestor, `(ancestor::*)[1]` the outermost
    // (`K2-ancestorAxis-9`/`-12`, `K2-precedingAxis-4`/`-5`), and it is why
    // `ancestor-or-self::*/name()` reads from the root downwards (`app-FunctxFunctx`).
    if !step.axis.is_forward() {
        nodes.sort();
    }
    Ok(XpathValue::from_items(
        nodes.into_iter().map(XpathItem::from),
    ))
}

/// Whether `node` satisfies `test` on `axis`.
///
/// A *name* test matches only nodes of the axis's principal node kind — elements everywhere
/// except on `attribute` (attributes) and `namespace` (namespace nodes) — which is why
/// `child::a` never selects an attribute and `@a` never selects an element. A *kind* test
/// carries its own node kind and ignores the axis.
fn matches_test(test: &NodeTest, node: &NodeRef, axis: Axis) -> bool {
    match test {
        NodeTest::Name(name) => {
            node.node_kind() == axis.principal_node_kind() && name.matches(node)
        }
        NodeTest::Kind(kind) => kind.matches(node),
    }
}

fn filter_nodes(
    nodes: Vec<NodeRef>,
    predicate: &Expr,
    context: &DynamicContext,
) -> XPathResult<Vec<NodeRef>> {
    let size = nodes.len();
    let mut kept = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.into_iter().enumerate() {
        super::EvaluationLimits::check_deadline()?;
        let focus = Focus::new(node.clone(), index + 1, size);
        let value = super::evaluate(predicate, &context.clone().with_focus(focus))?;
        if predicate_holds(&value, index + 1)? {
            kept.push(node);
        }
    }
    Ok(kept)
}

/// `E[P]` over an arbitrary sequence — the postfix filter, as opposed to a step predicate.
///
/// # Errors
///
/// Whatever the predicate raises.
pub(crate) fn filter(
    base: &Expr,
    predicate: &Expr,
    context: &DynamicContext,
) -> XPathResult<XpathValue> {
    // A constant positional predicate over a range — `(1 to 1000000000)[1]` — selects one item
    // from the range's bounds, so the billion-item sequence is never built.
    if let Some(result) = super::operators::range_at_position(base, predicate, context)? {
        return Ok(result);
    }
    let input = super::evaluate(base, context)?;
    let size = input.len();
    let mut kept = XpathValue::empty();
    for (index, item) in input.into_iter().enumerate() {
        super::EvaluationLimits::check_deadline()?;
        let focus = Focus::new(item.clone(), index + 1, size);
        let value = super::evaluate(predicate, &context.clone().with_focus(focus))?;
        if predicate_holds(&value, index + 1)? {
            kept.push(item);
        }
    }
    Ok(kept)
}

/// A predicate that yields exactly one number is a *positional* test; anything else is tested by
/// its effective boolean value.
fn predicate_holds(value: &XpathValue, position: usize) -> XPathResult<bool> {
    if let [XpathItem::Value(oxixml_infoset::Item::Atomic(atomic))] = value.as_slice() {
        if atomic.is_numeric() {
            let number = atomic.as_f64().unwrap_or(f64::NAN);
            #[allow(clippy::cast_precision_loss)]
            return Ok(number == position as f64);
        }
    }
    effective_boolean_value(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticContext;
    use crate::types::NameTest;
    use oxixml_infoset::{ExpandedName, Tree};
    use std::sync::Arc;

    fn tree() -> XPathResult<Tree> {
        Ok(Tree::parse_str(
            r#"<r><a id="1">x</a><b/><a id="2">y</a></r>"#,
        )?)
    }

    fn context(node: NodeRef) -> DynamicContext {
        DynamicContext::new(Arc::new(StaticContext::new())).with_context_item(node)
    }

    fn child_a() -> AxisStep {
        AxisStep {
            axis: Axis::Child,
            node_test: NodeTest::Name(NameTest::Name(ExpandedName::new(None::<String>, "a"))),
            predicates: Vec::new(),
        }
    }

    #[test]
    fn a_name_test_matches_only_the_principal_node_kind() -> XPathResult<()> {
        let tree = tree()?;
        let root = tree.document_element().expect("a document element");
        let selected = step_value(&child_a(), &context(root.clone()))?;
        assert_eq!(selected.len(), 2);

        let attribute_step = AxisStep {
            axis: Axis::Attribute,
            node_test: NodeTest::Name(NameTest::Any),
            predicates: Vec::new(),
        };
        let first = root.children().first().cloned().expect("a child");
        assert_eq!(step_value(&attribute_step, &context(first))?.len(), 1);
        Ok(())
    }

    #[test]
    fn positional_predicates_and_boolean_predicates() -> XPathResult<()> {
        let tree = tree()?;
        let root = tree.document_element().expect("a document element");
        let mut step = child_a();
        step.predicates
            .push(Expr::Literal(oxixml_infoset::AtomicValue::integer(2_i64)));
        assert_eq!(step_value(&step, &context(root.clone()))?.len(), 1);

        let mut always = child_a();
        always
            .predicates
            .push(Expr::Literal(oxixml_infoset::AtomicValue::boolean(false)));
        assert!(step_value(&always, &context(root))?.is_empty());
        Ok(())
    }

    /// XPath 3.1 §3.3.2: the *value* of an axis step is in document order however the axis runs;
    /// only the numbering inside the step's own predicates follows the axis. That is the whole
    /// difference between `ancestor::*[1]` and `(ancestor::*)[1]`.
    #[test]
    fn a_reverse_axis_step_is_valued_in_document_order() -> XPathResult<()> {
        let tree = Tree::parse_str("<r><a><b><c/></b></a></r>")?;
        let deepest = tree
            .document_element()
            .and_then(|root| root.children().first().cloned())
            .and_then(|a| a.children().first().cloned())
            .and_then(|b| b.children().first().cloned())
            .expect("the `c` element");
        let step = AxisStep {
            axis: Axis::Ancestor,
            node_test: NodeTest::Name(NameTest::Any),
            predicates: Vec::new(),
        };
        let names: Vec<String> = step_value(&step, &context(deepest.clone()))?
            .iter()
            .filter_map(|item| item.as_node().and_then(oxixml_infoset::XdmNode::node_name))
            .map(|name| name.local_name().to_owned())
            .collect();
        assert_eq!(names, ["r", "a", "b"]);

        // The predicate still counts from the nearest ancestor outwards.
        let mut nearest = step;
        nearest
            .predicates
            .push(Expr::Literal(oxixml_infoset::AtomicValue::integer(1_i64)));
        let selected = step_value(&nearest, &context(deepest))?;
        assert_eq!(
            selected
                .first()
                .and_then(|item| item.as_node())
                .and_then(oxixml_infoset::XdmNode::node_name)
                .map(|name| name.local_name().to_owned()),
            Some("b".to_owned())
        );
        Ok(())
    }

    #[test]
    fn a_path_over_non_nodes_is_xpty0019() -> XPathResult<()> {
        let tree = tree()?;
        let root = tree.root();
        let failure = path(
            &Expr::Literal(oxixml_infoset::AtomicValue::integer(1_i64)),
            &Expr::Step(Box::new(child_a())),
            &context(root),
        )
        .expect_err("a number has no children");
        assert_eq!(failure.code(), Some(XPathErrorCode::Xpty0019));
        Ok(())
    }

    #[test]
    fn results_are_deduplicated_and_ordered() -> XPathResult<()> {
        let tree = tree()?;
        let root = tree.document_element().expect("a document element");
        // `a | a` through a union is covered elsewhere; here the same node is reached twice via
        // a two-step path, and must appear once.
        let step = Expr::Step(Box::new(AxisStep {
            axis: Axis::Parent,
            node_test: NodeTest::Kind(crate::types::KindTest::AnyKind),
            predicates: Vec::new(),
        }));
        let children = Expr::Step(Box::new(AxisStep {
            axis: Axis::Child,
            node_test: NodeTest::Kind(crate::types::KindTest::AnyKind),
            predicates: Vec::new(),
        }));
        let parents = path(&children, &step, &context(root))?;
        assert_eq!(parents.len(), 1);
        Ok(())
    }
}
