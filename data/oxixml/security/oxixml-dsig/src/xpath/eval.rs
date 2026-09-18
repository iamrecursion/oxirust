//! The XPath 1.0 evaluator.

use std::collections::BTreeMap;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::xpath::ast::{Axis, BinaryOp, Expr, Name, NodeTest, Step};
use crate::xpath::functions;
use crate::xpath::value::{NodeSet, Value, XNode, is_namespace_declaration, namespace_nodes};

/// How a prefix in an expression is turned into a namespace name.
///
/// [XML Signature §6.6.3](https://www.w3.org/TR/xmldsig-core1/#sec-XPath) says
/// the evaluation context includes "the set of namespace declarations in scope
/// for the XPath expression", which means the declarations on the `<ds:XPath>`
/// element that carries it — not the ones on the element being tested, and not
/// the ones of the document the expression is evaluated against. Those can be
/// two different documents once a transform has re-parsed an octet stream, so
/// the bindings are *snapshotted* when the context is built rather than looked
/// up lazily.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NamespaceContext {
    bindings: BTreeMap<String, String>,
}

impl NamespaceContext {
    /// A context with no bindings at all.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Snapshots every prefix in scope on `element`.
    ///
    /// The default namespace is deliberately left out: an unprefixed name in an
    /// XPath expression has *no* namespace
    /// ([XPath §2.3](https://www.w3.org/TR/1999/REC-xpath-19991116/#node-tests)),
    /// so a default declaration is simply not reachable from an expression.
    pub(crate) fn from_element(document: &Document, element: NodeId) -> Self {
        let mut bindings = BTreeMap::new();
        for node in namespace_nodes(document, element) {
            if !node.prefix.is_empty() {
                bindings.insert(node.prefix, node.uri);
            }
        }
        Self { bindings }
    }

    /// Adds an explicit binding, which wins over the snapshot.
    #[cfg(test)]
    pub(crate) fn bind(mut self, prefix: impl Into<String>, uri: impl Into<String>) -> Self {
        self.bindings.insert(prefix.into(), uri.into());
        self
    }

    fn resolve(&self, prefix: &str) -> Option<&str> {
        self.bindings.get(prefix).map(String::as_str)
    }
}

/// Resolves the argument of `id()` and of a shortname XPointer.
pub(crate) trait IdResolver {
    /// The element carrying that identifier, if any.
    fn resolve(&self, id: &str) -> Option<NodeId>;
}

/// Everything an expression is evaluated against, apart from the focus.
pub(crate) struct Context<'a> {
    /// The document the expression walks.
    pub(crate) document: &'a Document,
    /// How prefixes are resolved.
    pub(crate) namespaces: NamespaceContext,
    /// The node `here()` returns, which XML Signature defines as the attribute,
    /// processing instruction, or parent element of the text node that bears
    /// the expression.
    pub(crate) here: Option<NodeId>,
    /// How `id()` is resolved.
    pub(crate) ids: &'a dyn IdResolver,
}

/// The context node, position and size.
#[derive(Debug, Clone)]
pub(crate) struct Focus {
    /// The context node.
    pub(crate) node: XNode,
    /// The context position, one-based.
    pub(crate) position: usize,
    /// The context size.
    pub(crate) size: usize,
}

impl Focus {
    /// The focus XML Signature's transforms use: one node, position and size 1.
    pub(crate) const fn single(node: XNode) -> Self {
        Self {
            node,
            position: 1,
            size: 1,
        }
    }
}

/// Evaluates an expression.
///
/// # Errors
///
/// Returns a human-readable message for an unbound prefix, an unknown function,
/// a variable reference (XML Signature binds none) or a wrong argument count.
pub(crate) fn evaluate(
    expression: &Expr,
    context: &Context<'_>,
    focus: &Focus,
) -> Result<Value, String> {
    match expression {
        Expr::Number(value) => Ok(Value::Number(*value)),
        Expr::Literal(text) => Ok(Value::Str(text.clone())),
        Expr::Variable(name) => Err(format!(
            "the variable reference ${} cannot be resolved: XML Signature binds no variables",
            name.to_qualified_string()
        )),
        Expr::Negate(inner) => {
            let value = evaluate(inner, context, focus)?;
            Ok(Value::Number(-value.to_number(context.document)))
        }
        Expr::Union(left, right) => {
            let left = expect_node_set(evaluate(left, context, focus)?, "|")?;
            let right = expect_node_set(evaluate(right, context, focus)?, "|")?;
            Ok(Value::NodeSet(left.union(context.document, right)))
        }
        Expr::Binary(operator, left, right) => {
            evaluate_binary(*operator, left, right, context, focus)
        }
        Expr::Function { name, arguments } => functions::call(name, arguments, context, focus),
        Expr::Path { absolute, steps } => {
            let start = if *absolute {
                NodeSet::singleton(XNode::Tree(context.document.root()))
            } else {
                NodeSet::singleton(focus.node.clone())
            };
            Ok(Value::NodeSet(walk_steps(start, steps, context)?))
        }
        Expr::Filter {
            primary,
            predicates,
            steps,
        } => {
            let value = evaluate(primary, context, focus)?;
            let mut set = expect_node_set(value, "a filter expression")?;
            for predicate in predicates {
                set = filter(set, predicate, context, false)?;
            }
            Ok(Value::NodeSet(walk_steps(set, steps, context)?))
        }
    }
}

fn expect_node_set(value: Value, what: &str) -> Result<NodeSet, String> {
    match value {
        Value::NodeSet(set) => Ok(set),
        other => Err(format!(
            "{what} needs a node-set, not {}",
            describe_value(&other)
        )),
    }
}

fn describe_value(value: &Value) -> &'static str {
    match value {
        Value::NodeSet(_) => "a node-set",
        Value::Boolean(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::Str(_) => "a string",
    }
}

fn evaluate_binary(
    operator: BinaryOp,
    left: &Expr,
    right: &Expr,
    context: &Context<'_>,
    focus: &Focus,
) -> Result<Value, String> {
    // `and` and `or` short-circuit, which matters when the other operand would
    // raise an error.
    match operator {
        BinaryOp::And => {
            if !evaluate(left, context, focus)?.to_boolean() {
                return Ok(Value::Boolean(false));
            }
            return Ok(Value::Boolean(
                evaluate(right, context, focus)?.to_boolean(),
            ));
        }
        BinaryOp::Or => {
            if evaluate(left, context, focus)?.to_boolean() {
                return Ok(Value::Boolean(true));
            }
            return Ok(Value::Boolean(
                evaluate(right, context, focus)?.to_boolean(),
            ));
        }
        _ => {}
    }

    let left = evaluate(left, context, focus)?;
    let right = evaluate(right, context, focus)?;
    let document = context.document;
    Ok(match operator {
        BinaryOp::Equal => Value::Boolean(compare_equality(document, &left, &right, false)),
        BinaryOp::NotEqual => Value::Boolean(compare_equality(document, &left, &right, true)),
        BinaryOp::Less | BinaryOp::LessOrEqual | BinaryOp::Greater | BinaryOp::GreaterOrEqual => {
            Value::Boolean(compare_relational(document, operator, &left, &right))
        }
        BinaryOp::Add => Value::Number(left.to_number(document) + right.to_number(document)),
        BinaryOp::Subtract => Value::Number(left.to_number(document) - right.to_number(document)),
        BinaryOp::Multiply => Value::Number(left.to_number(document) * right.to_number(document)),
        BinaryOp::Divide => Value::Number(left.to_number(document) / right.to_number(document)),
        BinaryOp::Modulo => Value::Number(left.to_number(document) % right.to_number(document)),
        BinaryOp::And | BinaryOp::Or => Value::Boolean(false),
    })
}

/// The `=` and `!=` rules of
/// [XPath §3.4](https://www.w3.org/TR/1999/REC-xpath-19991116/#booleans):
/// when either operand is a node-set the comparison is existential.
fn compare_equality(document: &Document, left: &Value, right: &Value, negated: bool) -> bool {
    match (left, right) {
        (Value::NodeSet(a), Value::NodeSet(b)) => {
            let b_strings: Vec<String> = b
                .nodes()
                .iter()
                .map(|node| node.string_value(document))
                .collect();
            a.nodes().iter().any(|node| {
                let value = node.string_value(document);
                b_strings.iter().any(|other| {
                    if negated {
                        *other != value
                    } else {
                        *other == value
                    }
                })
            })
        }
        (Value::NodeSet(set), other) | (other, Value::NodeSet(set)) => {
            set.nodes().iter().any(|node| {
                let node_value = Value::Str(node.string_value(document));
                let equal = match other {
                    Value::Number(_) => {
                        let a = node_value.to_number(document);
                        let b = other.to_number(document);
                        a == b
                    }
                    Value::Boolean(_) => node_value.to_boolean() == other.to_boolean(),
                    _ => node_value.to_xpath_string(document) == other.to_xpath_string(document),
                };
                if negated { !equal } else { equal }
            })
        }
        _ => {
            let equal = if matches!(left, Value::Boolean(_)) || matches!(right, Value::Boolean(_)) {
                left.to_boolean() == right.to_boolean()
            } else if matches!(left, Value::Number(_)) || matches!(right, Value::Number(_)) {
                left.to_number(document) == right.to_number(document)
            } else {
                left.to_xpath_string(document) == right.to_xpath_string(document)
            };
            if negated { !equal } else { equal }
        }
    }
}

fn compare_relational(
    document: &Document,
    operator: BinaryOp,
    left: &Value,
    right: &Value,
) -> bool {
    let compare = |a: f64, b: f64| match operator {
        BinaryOp::Less => a < b,
        BinaryOp::LessOrEqual => a <= b,
        BinaryOp::Greater => a > b,
        BinaryOp::GreaterOrEqual => a >= b,
        _ => false,
    };
    match (left, right) {
        (Value::NodeSet(a), Value::NodeSet(b)) => {
            let b_numbers: Vec<f64> = b
                .nodes()
                .iter()
                .map(|node| Value::Str(node.string_value(document)).to_number(document))
                .collect();
            a.nodes().iter().any(|node| {
                let value = Value::Str(node.string_value(document)).to_number(document);
                b_numbers.iter().any(|other| compare(value, *other))
            })
        }
        (Value::NodeSet(set), other) => {
            let bound = other.to_number(document);
            set.nodes().iter().any(|node| {
                compare(
                    Value::Str(node.string_value(document)).to_number(document),
                    bound,
                )
            })
        }
        (other, Value::NodeSet(set)) => {
            let bound = other.to_number(document);
            set.nodes().iter().any(|node| {
                compare(
                    bound,
                    Value::Str(node.string_value(document)).to_number(document),
                )
            })
        }
        _ => compare(left.to_number(document), right.to_number(document)),
    }
}

/// Walks a sequence of steps from a starting node-set.
fn walk_steps(start: NodeSet, steps: &[Step], context: &Context<'_>) -> Result<NodeSet, String> {
    let mut current = start;
    for step in steps {
        // An unresolvable prefix is an error, not a silent miss, and it does not
        // depend on the context node — so it is checked once per step.
        check_prefixes(context, &step.node_test)?;
        let mut collected = NodeSet::new();
        for node in current.nodes() {
            let matched: Vec<XNode> = axis_nodes(context.document, step.axis, node)
                .into_iter()
                .filter(|candidate| {
                    node_test_matches(context, step.axis, &step.node_test, candidate)
                })
                .collect();
            let mut set = NodeSet::from_nodes(context.document, matched);
            for predicate in &step.predicates {
                set = filter(set, predicate, context, step.axis.is_reverse())?;
            }
            for node in set.into_nodes() {
                collected.push(node);
            }
        }
        collected.normalize(context.document);
        current = collected;
    }
    Ok(current)
}

/// Fails when a node test names a prefix that is not in scope.
fn check_prefixes(context: &Context<'_>, test: &NodeTest) -> Result<(), String> {
    let prefix = match test {
        NodeTest::AnyNameInNamespace(prefix) => prefix.as_str(),
        NodeTest::Name(Name {
            prefix: Some(prefix),
            ..
        }) => prefix.as_str(),
        _ => return Ok(()),
    };
    if context.namespaces.resolve(prefix).is_some() {
        Ok(())
    } else {
        Err(format!(
            "the namespace prefix `{prefix}` is not bound where the XPath expression appears"
        ))
    }
}

/// The nodes on `axis` from `node`, in the axis's own order.
fn axis_nodes(document: &Document, axis: Axis, node: &XNode) -> Vec<XNode> {
    let tree = node.as_tree();
    match axis {
        Axis::SelfAxis => vec![node.clone()],
        Axis::Child => tree
            .filter(|id| {
                matches!(
                    document.kind(*id),
                    Some(NodeKind::Document | NodeKind::Element)
                )
            })
            .map(|id| document.children(id).map(XNode::Tree).collect())
            .unwrap_or_default(),
        Axis::Descendant => tree
            .map(|id| {
                document
                    .descendants(id)
                    .filter(|descendant| *descendant != id)
                    .map(XNode::Tree)
                    .collect()
            })
            .unwrap_or_default(),
        Axis::DescendantOrSelf => match tree {
            Some(id) => document.traverse(id).map(XNode::Tree).collect(),
            None => vec![node.clone()],
        },
        Axis::Parent => parent_of(document, node).into_iter().collect(),
        Axis::Ancestor => {
            let mut nodes = Vec::new();
            let mut current = parent_of(document, node);
            while let Some(XNode::Tree(id)) = current {
                nodes.push(XNode::Tree(id));
                current = document.parent(id).map(XNode::Tree);
            }
            nodes
        }
        Axis::AncestorOrSelf => {
            let mut nodes = vec![node.clone()];
            let mut current = parent_of(document, node);
            while let Some(XNode::Tree(id)) = current {
                nodes.push(XNode::Tree(id));
                current = document.parent(id).map(XNode::Tree);
            }
            nodes
        }
        Axis::Attribute => tree
            .map(|id| {
                document
                    .attributes(id)
                    .filter(|attribute| !is_namespace_declaration(document, *attribute))
                    .map(XNode::Tree)
                    .collect()
            })
            .unwrap_or_default(),
        Axis::Namespace => tree
            .filter(|id| document.kind(*id) == Some(NodeKind::Element))
            .map(|id| {
                namespace_nodes(document, id)
                    .into_iter()
                    .map(XNode::Namespace)
                    .collect()
            })
            .unwrap_or_default(),
        Axis::FollowingSibling => sibling_axis(document, node, true),
        Axis::PrecedingSibling => sibling_axis(document, node, false),
        Axis::Following => tree
            .map(|id| document.following(id).map(XNode::Tree).collect())
            .unwrap_or_default(),
        Axis::Preceding => tree
            .map(|id| document.preceding(id).map(XNode::Tree).collect())
            .unwrap_or_default(),
    }
}

/// The parent of an XPath node. An attribute's parent is its owner element and
/// a namespace node's parent is the element it hangs off, even though neither
/// is a child of that element.
fn parent_of(document: &Document, node: &XNode) -> Option<XNode> {
    match node {
        XNode::Namespace(namespace) => Some(XNode::Tree(namespace.element)),
        XNode::Tree(id) => {
            if document.kind(*id) == Some(NodeKind::Attr) {
                document.owner_element(*id).map(XNode::Tree)
            } else {
                document.parent(*id).map(XNode::Tree)
            }
        }
    }
}

/// The sibling axes, which are empty for attribute and namespace nodes.
fn sibling_axis(document: &Document, node: &XNode, following: bool) -> Vec<XNode> {
    let Some(id) = node.as_tree() else {
        return Vec::new();
    };
    if document.kind(id) == Some(NodeKind::Attr) {
        return Vec::new();
    }
    if following {
        document.following_siblings(id).map(XNode::Tree).collect()
    } else {
        document.preceding_siblings(id).map(XNode::Tree).collect()
    }
}

/// Whether a candidate passes a node test on a given axis.
fn node_test_matches(
    context: &Context<'_>,
    axis: Axis,
    test: &NodeTest,
    candidate: &XNode,
) -> bool {
    let document = context.document;
    let kind = candidate.as_tree().and_then(|id| document.kind(id));
    match test {
        NodeTest::AnyNode => true,
        NodeTest::Text => matches!(kind, Some(NodeKind::Text | NodeKind::CdataSection)),
        NodeTest::Comment => kind == Some(NodeKind::Comment),
        NodeTest::ProcessingInstruction(target) => {
            kind == Some(NodeKind::ProcessingInstruction)
                && match (target, candidate.as_tree()) {
                    (None, _) => true,
                    (Some(wanted), Some(id)) => document
                        .processing_instruction(id)
                        .is_some_and(|pi| pi.target() == wanted),
                    (Some(_), None) => false,
                }
        }
        NodeTest::AnyName => is_principal_type(axis, candidate, kind),
        NodeTest::AnyNameInNamespace(prefix) => {
            if !is_principal_type(axis, candidate, kind) {
                false
            } else {
                let Some(uri) = context.namespaces.resolve(prefix) else {
                    return false;
                };
                candidate
                    .expanded_name(document)
                    .is_some_and(|(namespace, _)| namespace.as_deref() == Some(uri))
            }
        }
        NodeTest::Name(name) => {
            if !is_principal_type(axis, candidate, kind) {
                false
            } else {
                let expected_namespace = match &name.prefix {
                    Some(prefix) => match context.namespaces.resolve(prefix) {
                        Some(uri) => Some(uri.to_owned()),
                        None => return false,
                    },
                    // An unprefixed name has no namespace, even when a default
                    // namespace is in scope — XPath §2.3.
                    None => None,
                };
                candidate
                    .expanded_name(document)
                    .is_some_and(|(namespace, local)| {
                        local == name.local && namespace == expected_namespace
                    })
            }
        }
    }
}

fn is_principal_type(axis: Axis, candidate: &XNode, kind: Option<NodeKind>) -> bool {
    match axis {
        Axis::Attribute => kind == Some(NodeKind::Attr),
        Axis::Namespace => matches!(candidate, XNode::Namespace(_)),
        _ => kind == Some(NodeKind::Element),
    }
}

/// Applies one predicate to a node-set.
///
/// On a reverse axis the context position counts from the end, which is what
/// makes `ancestor::x[1]` the *nearest* ancestor.
fn filter(
    set: NodeSet,
    predicate: &Expr,
    context: &Context<'_>,
    reverse: bool,
) -> Result<NodeSet, String> {
    let nodes = set.into_nodes();
    let size = nodes.len();
    let mut kept = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let position = if reverse { size - index } else { index + 1 };
        let focus = Focus {
            node: node.clone(),
            position,
            size,
        };
        let value = evaluate(predicate, context, &focus)?;
        let keep = match value {
            Value::Number(number) => number == position_as_f64(position),
            other => other.to_boolean(),
        };
        if keep {
            kept.push(node.clone());
        }
    }
    Ok(NodeSet::from_nodes(context.document, kept))
}

/// Builds the node-set XML Signature calls "the whole document":
/// `(//. | //@* | //namespace::*)`.
pub(crate) fn whole_document_node_set(document: &Document) -> NodeSet {
    let mut nodes = Vec::new();
    for node in document.nodes_in_document_order() {
        match document.kind(node) {
            Some(NodeKind::Attr) if !is_namespace_declaration(document, node) => {
                nodes.push(XNode::Tree(node));
            }
            Some(NodeKind::Attr) => {}
            Some(NodeKind::Element) => {
                nodes.push(XNode::Tree(node));
                nodes.extend(
                    namespace_nodes(document, node)
                        .into_iter()
                        .map(XNode::Namespace),
                );
            }
            Some(_) => nodes.push(XNode::Tree(node)),
            None => {}
        }
    }
    NodeSet::from_nodes(document, nodes)
}

/// A context position as an XPath number.
///
/// A node-set can never be long enough for the conversion to lose precision:
/// `f64` represents every integer up to 2^53 exactly, and a document with that
/// many nodes does not fit in memory.
#[expect(
    clippy::cast_precision_loss,
    reason = "a node-set is far shorter than 2^53"
)]
fn position_as_f64(position: usize) -> f64 {
    position as f64
}
