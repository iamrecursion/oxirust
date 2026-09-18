//! `xsl:key` and the `key()` function.
//!
//! [XSLT 3.0 §17](https://www.w3.org/TR/xslt-30/#keys) makes a key a three-part declaration:
//! a name, a pattern saying which nodes the key applies to, and a `use` expression giving the
//! values under which each of those nodes is filed. The index is built lazily, once per
//! `(key name, document)` pair, by walking the whole document; a stylesheet that never calls
//! `key()` therefore pays nothing.
//!
//! Several `xsl:key` declarations may share a name, in which case their indexes are merged.

use super::{Context, Transformer};
use crate::error::{XsltError, XsltResult};
use oxixml_fn::Collation;
use oxixml_infoset::{AtomicValue, ExpandedName, Item, NodeRef, XdmNode};
use oxixml_xsd::TimezoneOffset;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// A key index: the nodes each key value selects, in document order.
///
/// [XSLT 3.0 §17.3](https://www.w3.org/TR/xslt-30/#key-matching) compares a supplied key value
/// with an indexed one "using the `eq` operator, with the collation of the key in force" — a
/// *value* comparison, so an `xs:dateTime` written in two time zones is one key while the
/// integer `4` and the string `"4"` are two, and `NaN` is equal to nothing at all. Key values
/// are therefore held as atomic values rather than as strings, and `composite="yes"` makes the
/// whole atomized `use` sequence one key rather than filing the node once per item.
///
/// A key over string-like values under the code point collation — by far the common case,
/// `use="@id"` over an untyped document — indexes into a hash map, under which `eq` equality
/// *is* string equality. Any other key falls back on a scan.
pub(crate) struct KeyIndex {
    collation: Arc<dyn Collation>,
    implicit_timezone: TimezoneOffset,
    /// Whether the declarations of this key name said `composite="yes"`.
    composite: bool,
    hashed: Option<HashMap<Vec<String>, Vec<NodeRef>>>,
    scanned: Vec<(Vec<AtomicValue>, Vec<NodeRef>)>,
}

impl KeyIndex {
    fn new(
        collation: Arc<dyn Collation>,
        implicit_timezone: TimezoneOffset,
        composite: bool,
    ) -> Self {
        let hashed = (collation.uri() == oxixml_fn::CODEPOINT_COLLATION).then(HashMap::new);
        Self {
            collation,
            implicit_timezone,
            composite,
            hashed,
            scanned: Vec::new(),
        }
    }

    /// Whether a lookup argument is one composite key rather than a list of keys.
    pub(crate) const fn is_composite(&self) -> bool {
        self.composite
    }

    fn insert(&mut self, key: Vec<AtomicValue>, node: NodeRef) {
        if self.hashed.is_some() {
            match hashable(&key) {
                Some(hashable) => {
                    if let Some(hashed) = self.hashed.as_mut() {
                        hashed.entry(hashable).or_default().push(node);
                        return;
                    }
                }
                // One key value that string equality cannot decide retires the fast path: the
                // entries filed so far move into the scan, which compares by value.
                None => {
                    if let Some(hashed) = self.hashed.take() {
                        for (strings, nodes) in hashed {
                            let values: Vec<AtomicValue> =
                                strings.into_iter().map(AtomicValue::string).collect();
                            self.scanned.push((values, nodes));
                        }
                    }
                }
            }
        }
        if let Some(index) = self
            .scanned
            .iter()
            .position(|(candidate, _)| self.equal_keys(candidate, &key))
        {
            if let Some((_, nodes)) = self.scanned.get_mut(index) {
                nodes.push(node);
            }
            return;
        }
        self.scanned.push((key, vec![node]));
    }

    /// The nodes filed under `key`, if any.
    pub(crate) fn get(&self, key: &[AtomicValue]) -> Option<&Vec<NodeRef>> {
        if let Some(hashed) = self.hashed.as_ref() {
            return hashed.get(&hashable(key)?);
        }
        self.scanned
            .iter()
            .find(|(candidate, _)| self.equal_keys(candidate, key))
            .map(|(_, nodes)| nodes)
    }

    /// Whether two key values are equal, component by component, under `eq`.
    fn equal_keys(&self, left: &[AtomicValue], right: &[AtomicValue]) -> bool {
        left.len() == right.len() && left.iter().zip(right).all(|(a, b)| self.equal_value(a, b))
    }

    fn equal_value(&self, left: &AtomicValue, right: &AtomicValue) -> bool {
        // A collation decides string comparison and nothing else, so it is applied through the
        // collation object the key resolved rather than through a URI lookup inside the value
        // comparison.
        if string_like(left) && string_like(right) {
            return self
                .collation
                .equal(&left.string_value(), &right.string_value());
        }
        // Two values of unrelated types are `XPTY0004` under `eq`; a key lookup reports them as
        // simply different rather than failing the transformation.
        oxixml_infoset::atomic_equal_in_context(left, right, None, Some(self.implicit_timezone))
            .unwrap_or(false)
    }
}

/// The string form of a key whose every component `eq` compares as a string, or `None` when
/// some component needs a value comparison.
///
/// A value is string-like exactly when it is equal to a string holding its own string value,
/// which is true of `xs:string`, `xs:anyURI`, `xs:untypedAtomic` and their restrictions, and of
/// nothing else: `4` and `"4"` are a type error under `eq`, and `NaN` is equal to no value at
/// all, itself included.
fn hashable(key: &[AtomicValue]) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(key.len());
    for value in key {
        if !string_like(value) {
            return None;
        }
        out.push(value.string_value());
    }
    Some(out)
}

/// Whether `eq` compares a value as a string.
fn string_like(value: &AtomicValue) -> bool {
    oxixml_infoset::atomic_equal(value, &AtomicValue::string(value.string_value())).unwrap_or(false)
}

impl Transformer {
    /// The nodes `key(name, value)` selects in the document containing `origin`, restricted to
    /// the subtree rooted at `top` when the caller wrote a third argument.
    ///
    /// [XSLT 3.0 §20.2.2](https://www.w3.org/TR/xslt-30/#func-key): "The third argument is used
    /// to identify the *selected subtree*. If the argument is present, the selected subtree is
    /// the set of nodes that have `$top` as an `ancestor-or-self` node. If the argument is
    /// omitted, the selected subtree is the document containing the context node." The index
    /// itself stays per document — it is what makes the lookup cheap and it is shared between
    /// calls that name different subtrees of one tree — so the restriction is a filter over the
    /// nodes it answers, not a second index.
    ///
    /// # Errors
    ///
    /// `XTDE1260` when no key of that name is declared, plus any dynamic error the `use`
    /// expression raises.
    pub(crate) fn key_lookup(
        &mut self,
        name: &ExpandedName,
        values: &[AtomicValue],
        origin: &NodeRef,
        top: Option<&NodeRef>,
    ) -> XsltResult<Vec<NodeRef>> {
        let index = self.key_index(name, origin)?;
        let mut out: Vec<NodeRef> = Vec::new();
        // With `composite="yes"` the whole argument sequence is one key; otherwise each item of
        // it is a key of its own and the results are unioned.
        let mut push = |nodes: Option<&Vec<NodeRef>>| {
            for node in nodes.into_iter().flatten() {
                if !out.contains(node) {
                    out.push(node.clone());
                }
            }
        };
        if index.is_composite() {
            push(index.get(values));
        } else {
            for value in values {
                push(index.get(std::slice::from_ref(value)));
            }
        }
        if let Some(top) = top {
            out.retain(|node| has_ancestor_or_self(node, top));
        }
        oxixml_infoset::sort_document_order(&mut out);
        Ok(out)
    }

    /// The index for `name` over the document containing `origin`, building it if needed.
    ///
    /// # Errors
    ///
    /// `XTDE1260` when no key of that name is declared.
    pub(crate) fn key_index(
        &mut self,
        name: &ExpandedName,
        origin: &NodeRef,
    ) -> XsltResult<Rc<KeyIndex>> {
        let root = origin.root();
        // XSLT 3.0 §18.3: `fn:key` searches a document, so the tree the node it is given
        // belongs to has to be rooted at a document node.
        if root.node_kind() != oxixml_infoset::NodeKind::Document {
            return Err(XsltError::dynamic(
                "XTDE1270",
                "fn:key was called with a node whose root is not a document node",
            ));
        }
        let document = root.document_order_key().tree;
        if let Some(index) = self.runtime.keys.borrow().get(&(name.clone(), document)) {
            return Ok(Rc::clone(index));
        }
        if self
            .runtime
            .keys_in_progress
            .borrow()
            .contains(&(name.clone(), document))
        {
            return Err(XsltError::dynamic(
                "XTDE0640",
                format!("the definition of the key `{name}` is circular"),
            ));
        }
        let ir = Arc::clone(&self.runtime.ir);
        let definitions: Vec<usize> = ir
            .keys
            .iter()
            .enumerate()
            .filter(|(_, definition)| definition.name == *name)
            .map(|(index, _)| index)
            .collect();
        if definitions.is_empty() {
            return Err(XsltError::dynamic(
                "XTDE1260",
                format!("no xsl:key named `{name}` is declared"),
            ));
        }
        self.runtime
            .keys_in_progress
            .borrow_mut()
            .push((name.clone(), document));
        let built = self.build_index(&definitions, &root);
        self.runtime.keys_in_progress.borrow_mut().pop();
        let index = Rc::new(built?);
        self.runtime
            .keys
            .borrow_mut()
            .insert((name.clone(), document), Rc::clone(&index));
        Ok(index)
    }

    /// Walks the document once, filing every node the definitions match.
    fn build_index(&mut self, definitions: &[usize], root: &NodeRef) -> XsltResult<KeyIndex> {
        let ir = Arc::clone(&self.runtime.ir);
        // Every declaration of one key name agrees about the collation and about `composite`
        // (the compiler rejects a disagreement), so the first one settles both.
        let (collation, composite) = match definitions.first().and_then(|at| ir.keys.get(*at)) {
            Some(definition) => (
                self.key_collation(definition.collation.as_deref())?,
                definition.composite,
            ),
            None => (
                Arc::clone(self.runtime.fn_context.default_collation()),
                false,
            ),
        };
        let implicit_timezone = self.runtime.fn_context.implicit_timezone();
        let mut index = KeyIndex::new(collation, implicit_timezone, composite);
        let mut nodes = Vec::new();
        collect(root, &mut nodes);
        for definition_index in definitions.iter().copied() {
            let Some(definition) = ir.keys.get(definition_index) else {
                continue;
            };
            let dynamic = self.pattern_context_for(&Context::new(None));
            for node in &nodes {
                // XSLT 3.0 §18.4: a pattern sees no captured substrings.
                if !self.without_regex_groups(|_| definition.pattern.matches(node, &dynamic))? {
                    continue;
                }
                let context = Context::new(Some(Item::Node(node.clone())));
                // XSLT 3.0 §17.1: the key values of a node are "the result of evaluating
                // the `use` expression — or, where the declaration has a sequence constructor
                // instead, the result of the sequence constructor — with that node as the
                // context item, atomized". A sequence constructor therefore yields as many key
                // values as it produces items, not one string holding all of them (the suite's
                // key-073 and key-074).
                let sequence = match &definition.use_expression {
                    Some(use_expression) => {
                        self.evaluate(use_expression, &context)?.into_sequence()?
                    }
                    None => self.run_to_sequence(&definition.content, None, &context)?.1,
                };
                let values: Vec<AtomicValue> = sequence
                    .atomize()?
                    .iter()
                    .filter_map(|item| item.as_atomic().cloned())
                    .collect();
                if composite {
                    index.insert(values, node.clone());
                } else {
                    for value in values {
                        index.insert(vec![value], node.clone());
                    }
                }
            }
        }
        Ok(index)
    }
}

impl Transformer {
    /// Resolves an `xsl:key` collation URI, falling back on the stylesheet's default.
    ///
    /// # Errors
    ///
    /// `XTSE1220` for a collation this processor does not recognize — which the compiler
    /// normally rejects first.
    fn key_collation(&self, uri: Option<&str>) -> XsltResult<Arc<dyn Collation>> {
        match uri {
            Some(uri) => oxixml_fn::Collations::new().resolve(uri).map_err(|_| {
                XsltError::dynamic(
                    "XTSE1220",
                    format!("`{uri}` is not a collation this processor recognizes"),
                )
            }),
            None => Ok(Arc::clone(self.runtime.fn_context.default_collation())),
        }
    }
}

/// Whether `top` is on `node`'s `ancestor-or-self` axis, which is how
/// [XSLT 3.0 §20.2.2](https://www.w3.org/TR/xslt-30/#func-key) states the selected subtree:
/// "`$N/ancestor-or-self::node() intersect $top` is non-empty".
fn has_ancestor_or_self(node: &NodeRef, top: &NodeRef) -> bool {
    let mut current = Some(node.clone());
    while let Some(candidate) = current {
        if &candidate == top {
            return true;
        }
        current = candidate.parent();
    }
    false
}

/// Every node of a tree that a key may apply to, in document order.
///
/// Namespace nodes are among them: [XSLT 3.0 §6.5](https://www.w3.org/TR/xslt-30/#patterns) lets
/// a pattern match one, and `xsl:key/@match` is a pattern like any other — `key-087` indexes
/// `match="namespace-node()"` and expects the three elements a single `xmlns:ext` declaration is
/// in scope on. They precede the attributes, which is where document order puts them.
fn collect(node: &NodeRef, out: &mut Vec<NodeRef>) {
    out.push(node.clone());
    for namespace in node.namespaces() {
        out.push(namespace);
    }
    for attribute in node.attributes() {
        out.push(attribute);
    }
    for child in node.children() {
        collect(&child, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn collection_visits_attributes_and_children() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str(r#"<a x="1"><b/></a>"#)?;
        let mut nodes = Vec::new();
        collect(&tree.root(), &mut nodes);
        // Document, <a>, its `xml` namespace node, @x, <b>, <b>'s `xml` namespace node.
        assert_eq!(nodes.len(), 6);
        Ok(())
    }

    #[test]
    fn collection_visits_namespace_nodes() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str(r#"<a xmlns:p="urn:p"><b/></a>"#)?;
        let mut nodes = Vec::new();
        collect(&tree.root(), &mut nodes);
        let prefixes: Vec<String> = nodes
            .iter()
            .filter(|node| node.node_kind() == oxixml_infoset::NodeKind::Namespace)
            .filter_map(|node| node.node_name().map(|name| name.local_name().to_owned()))
            .collect();
        // `p` is in scope on `<a>` and, inherited, on `<b>`.
        assert_eq!(
            prefixes.iter().filter(|prefix| *prefix == "p").count(),
            2,
            "{prefixes:?}"
        );
        Ok(())
    }
}
