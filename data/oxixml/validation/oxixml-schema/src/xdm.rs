//! The XDM view an XSD 1.1 assertion is evaluated against.
//!
//! [XSD 1.1 Part 1 §3.13.4.2][xdm] fabricates a *new* XDM instance for every
//! assertion: a copy of the element being validated, **as the root of the
//! tree**, so that the expression can see neither the element's ancestors nor
//! any sibling. Within that copy the nodes carry the type annotations and typed
//! values the post-schema-validation infoset has given them, and everything
//! else is `xs:untypedAtomic`.
//!
//! The root being an *element* rather than a document node is not an
//! incidental detail — it is how the Recommendation stops an assertion from
//! reaching outside the subtree it is written about. XPath's `/` and `//` both
//! begin with `fn:root(.) treat as document-node()`, which raises
//! `err:XPDY0050` on a tree rooted at an element; [`crate::assertion`] turns
//! that dynamic error into an unsatisfied assertion, which is what
//! `ibmData/instance_invalid/D4_3_15/d4_3_15ii31` (`count(//ele1) eq 1`) and
//! `d4_3_15ii32` (`count(//@attr1) eq 1`) — both expected **invalid** under
//! the suite's `Assertions-StayInSubtree` category — are testing.
//!
//! Two shapes are built here.
//!
//! * [`Shape::Assertion`] — the whole subtree, for `xs:assert` (§3.4.4.2) and
//!   for the `assertion` facet (Part 2 §4.3.12).
//! * [`Shape::Alternative`] — the element with its attributes and **no**
//!   children, for conditional type assignment (§3.3.2). Conditional type
//!   assignment runs *before* the element is assessed, so nothing in the view
//!   is typed: every attribute is `xs:untypedAtomic`, exactly as the
//!   Recommendation requires.
//!
//! [xdm]: https://www.w3.org/TR/xmlschema11-1/#key-xdmnode

use std::collections::{HashMap, HashSet};

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_infoset::{
    Annotation, AtomicType, AtomicValue, ExpandedName, QNameValue, Sequence, Tree, TreeBuilder,
    XdmError,
};

use crate::components::{TypeId, Variety};
use crate::names::{XML_NS, XSD_NS};
use crate::schema::Schema;
use crate::validate::{AttributePsvi, ElementPsvi, MAX_DEPTH};

/// Which of the two XDM views §3.13.4.2 asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    /// The whole subtree, typed where the PSVI knows a type.
    Assertion,
    /// The element and its attributes alone, wholly untyped.
    Alternative,
}

/// Everything the builder needs to type a node.
pub(crate) struct XdmView<'a> {
    /// The instance document the element lives in.
    pub doc: &'a Document,
    /// The schema, for resolving a [`TypeId`] to a built-in atomic type.
    pub schema: &'a Schema,
    /// The element annotations gathered so far.
    pub elements: &'a HashMap<NodeId, ElementPsvi>,
    /// The attribute annotations gathered so far.
    pub attributes: &'a HashMap<NodeId, AttributePsvi>,
}

/// Why a fabricated instance could not be written.
///
/// The two cases are kept apart for the reader rather than for the caller —
/// [`XdmView::build`] answers [`None`] either way — because they say different
/// things about the document: one is a shape the data model does not admit, the
/// other a subtree this view declines to copy.
enum BuildFailure {
    /// The tree builder refused something the document contains.
    Xdm,
    /// The subtree nests deeper than [`MAX_DEPTH`]. The copy is written by
    /// recursion, so an unbounded one would consume unbounded stack; refusing
    /// to build the view leaves the assertion unsatisfied, which is the same
    /// outcome as any other view the document does not admit.
    TooDeep,
}

impl From<XdmError> for BuildFailure {
    fn from(_: XdmError) -> Self {
        Self::Xdm
    }
}

impl XdmView<'_> {
    /// Builds the fabricated instance rooted at `node`.
    ///
    /// Returns [`None`] when the tree cannot be built at all — for a document
    /// the DOM would not have produced, or for a subtree nesting deeper than
    /// [`MAX_DEPTH`]; the caller then reports the assertion as unsatisfied
    /// rather than panicking.
    pub(crate) fn build(&self, node: NodeId, shape: Shape) -> Option<Tree> {
        let mut builder = TreeBuilder::new_element_rooted();
        self.write_element(&mut builder, node, shape, true, 0)
            .ok()?;
        builder.build().ok()
    }

    /// Writes one element and, unless the shape forbids it, its content.
    ///
    /// `depth` counts how many elements of the copy are already open, so the
    /// recursion stops at [`MAX_DEPTH`] however deep the instance goes. An
    /// assertion is evaluated at *every* level, so this bound is needed
    /// independently of the one assessment itself observes: a deep subtree is
    /// copied afresh for each of the assertions above it.
    fn write_element(
        &self,
        builder: &mut TreeBuilder,
        node: NodeId,
        shape: Shape,
        root: bool,
        depth: usize,
    ) -> Result<(), BuildFailure> {
        if depth >= MAX_DEPTH {
            return Err(BuildFailure::TooDeep);
        }
        let element = match self.doc.element(node) {
            Some(element) => element,
            None => return Ok(()),
        };
        let name = ExpandedName::new(element.namespace_uri(), element.local_name());
        // §3.13.4.1 clause 1.1 validates `E` "except that for E itself (though
        // not for its element information item descendants), clause 6 of
        // Element Locally Valid (Complex Type) is skipped", so `E` is only
        // *partially* assessed and its [validity] is `notKnown`: "the element
        // node … will normally have the type annotation **anyType**, its
        // atomized value will be a single atomic value of type
        // untypedAtomic". Its descendants and attributes keep the types the
        // partial PSVI gives them (`saxonData/Assert/assert014`).
        let annotation = if shape == Shape::Assertion && !root {
            self.element_annotation(node)
        } else {
            Annotation::default()
        };
        builder.start_annotated_element(name, annotation)?;

        // The copy is detached from its ancestors, so every namespace that was
        // in scope has to be redeclared on the root of the fabricated tree;
        // deeper elements only need what they declare themselves.
        for (prefix, namespace) in self.in_scope_namespaces(node, root) {
            if prefix.as_deref() == Some("xml") && namespace == XML_NS {
                continue;
            }
            builder.declare_namespace(prefix.as_deref(), &namespace)?;
        }

        let mut local_names = HashSet::new();
        for attribute in element.attribute_ids() {
            let Some(data) = self.doc.attr(*attribute) else {
                continue;
            };
            // Namespace declarations are not attributes in the data model, and
            // `xsi:*` attributes are stripped from the fabricated instance by
            // §3.13.4.2 step 2.
            if data.prefix() == Some("xmlns") || data.qualified_name() == "xmlns" {
                continue;
            }
            let attribute_name = ExpandedName::new(data.namespace_uri(), data.local_name());
            local_names.insert(attribute_name.clone());
            let annotation = if shape == Shape::Assertion {
                self.attribute_annotation(node, *attribute)
            } else {
                Annotation::default()
            };
            builder.annotated_attribute(attribute_name, data.value(), annotation)?;
        }

        // XSD 1.1 §3.3.4/§3.3.2 (CTA's `test`) and §3.13.4.2 clause 2
        // (`xs:assert` and the `assertion` facet): an attribute `node` does
        // not carry locally, but that some ancestor does under an
        // `inheritable="true"` declaration, is added to this fabricated view
        // too, exactly as if it were the element's own.
        for (attribute_name, attribute) in self.inherited_attributes(node, &mut local_names) {
            let Some(data) = self.doc.attr(attribute) else {
                continue;
            };
            let annotation = if shape == Shape::Assertion {
                self.attribute_annotation(node, attribute)
            } else {
                Annotation::default()
            };
            builder.annotated_attribute(attribute_name, data.value(), annotation)?;
        }

        if shape == Shape::Assertion {
            for child in self.doc.children(node) {
                match self.doc.kind(child) {
                    Some(NodeKind::Element) => {
                        self.write_element(builder, child, shape, false, depth + 1)?;
                    }
                    Some(NodeKind::Text | NodeKind::CdataSection) => {
                        if let Some(text) = self.doc.character_data(child) {
                            builder.text(text)?;
                        }
                    }
                    // §3.13.4.1 clause 1.2: "By default, comments and
                    // processing instructions are **excluded** from the
                    // partial ·post-schema-validation infoset·" — an
                    // assertion cannot see them, so `empty(.//comment())`
                    // holds even of an element that has one
                    // (`saxonData/Assert/assert023`).
                    _ => {}
                }
            }
        }

        builder.end_element()?;
        Ok(())
    }

    /// The attributes `node` inherits from its ancestors under XSD 1.1's
    /// `inheritable` attribute-use/declaration property ([Part 1
    /// §3.3.4][decl], used by CTA's `test` evaluation ([§3.3.2][cta]) and by
    /// assertion evaluation ([§3.13.4.2][assert], clause 2): walking outward
    /// from `node`'s parent, the nearest ancestor that carries an attribute
    /// of a given expanded name *that was itself validated as inheritable*
    /// contributes it, for every name not already accounted for — `resolved`
    /// starts out seeded with `node`'s own local attribute names, and gains
    /// each name this resolves so a farther ancestor cannot re-contribute it.
    ///
    /// Whether an occurrence counts as inheritable is read from its own
    /// [`AttributePsvi::inheritable`], not re-derived from the declaration
    /// alone: a `ref`'d attribute use's own `inheritable` overrides the
    /// referenced declaration's (`CTA/cta0011`'s schema states this outright:
    /// "inheritability differs between the attribute decl and the attribute
    /// use: the attribute use wins").
    ///
    /// A same-named attribute on a nearer ancestor that is *not* inheritable
    /// does not mask a farther ancestor's inheritable one; it is simply
    /// transparent to the search. `CTA/cta0013`'s schema states that rule
    /// outright too: "a non-inheritable attribute does not mask an
    /// inheritable one on an ancestor".
    ///
    /// [decl]: https://www.w3.org/TR/xmlschema11-1/#e-inheritable
    /// [cta]: https://www.w3.org/TR/xmlschema11-1/#cts-ctable
    /// [assert]: https://www.w3.org/TR/xmlschema11-1/#assertion_satisfied
    fn inherited_attributes(
        &self,
        node: NodeId,
        resolved: &mut HashSet<ExpandedName>,
    ) -> Vec<(ExpandedName, NodeId)> {
        let mut found = Vec::new();
        let mut ancestor = self.doc.parent(node);
        while let Some(current) = ancestor {
            if let Some(element) = self.doc.element(current) {
                for attribute in element.attribute_ids() {
                    let Some(data) = self.doc.attr(*attribute) else {
                        continue;
                    };
                    if data.prefix() == Some("xmlns") || data.qualified_name() == "xmlns" {
                        continue;
                    }
                    let name = ExpandedName::new(data.namespace_uri(), data.local_name());
                    if resolved.contains(&name) {
                        continue;
                    }
                    let is_inheritable = self
                        .attributes
                        .get(attribute)
                        .is_some_and(|psvi| psvi.inheritable);
                    if is_inheritable {
                        resolved.insert(name.clone());
                        found.push((name, *attribute));
                    }
                }
            }
            ancestor = self.doc.parent(current);
        }
        found
    }

    /// The namespace bindings to declare on `node` in the copy: everything in
    /// scope when `node` is the fabricated root, and only its own declarations
    /// otherwise.
    fn in_scope_namespaces(&self, node: NodeId, root: bool) -> Vec<(Option<String>, String)> {
        let mut bindings: Vec<(Option<String>, String)> = Vec::new();
        if !root {
            for (prefix, namespace) in self.doc.namespace_declarations(node) {
                bindings.push((prefix.map(str::to_owned), namespace.to_owned()));
            }
            return bindings;
        }
        // Walking from the element outwards and keeping the *first* binding for
        // each prefix gives the innermost declaration, which is the one in
        // scope.
        let mut current = Some(node);
        while let Some(id) = current {
            if self.doc.kind(id) == Some(NodeKind::Element) {
                for (prefix, namespace) in self.doc.namespace_declarations(id) {
                    let prefix = prefix.map(str::to_owned);
                    if bindings.iter().any(|(seen, _)| *seen == prefix) {
                        continue;
                    }
                    bindings.push((prefix, namespace.to_owned()));
                }
            }
            current = self.doc.parent(id);
        }
        // An undeclaration inherited from an ancestor has nothing to undeclare
        // in the copy, and the builder rejects an empty namespace for a prefix
        // under Namespaces 1.0.
        bindings.retain(|(_, namespace)| !namespace.is_empty());
        bindings
    }

    /// The type annotation and typed value of an assessed element.
    fn element_annotation(&self, node: NodeId) -> Annotation {
        let Some(psvi) = self.elements.get(&node) else {
            return Annotation::default();
        };
        let Some(type_id) = psvi.type_definition else {
            return Annotation::default();
        };
        let typed_value = psvi.normalized_value.as_deref().and_then(|lexical| {
            self.qualified_name_value(node, type_id, lexical)
                .or_else(|| typed_value(self.schema, type_id, lexical))
        });
        Annotation {
            type_name: self.type_annotation_name(type_id),
            typed_value,
            is_id: false,
            is_idref: false,
            nilled: Some(psvi.nil),
        }
    }

    /// The type annotation and typed value of an assessed attribute.
    fn attribute_annotation(&self, owner: NodeId, node: NodeId) -> Annotation {
        let Some(psvi) = self.attributes.get(&node) else {
            return Annotation::default();
        };
        let Some(type_id) = psvi.type_definition else {
            return Annotation::default();
        };
        let typed_value = psvi.normalized_value.as_deref().and_then(|lexical| {
            self.qualified_name_value(owner, type_id, lexical)
                .or_else(|| typed_value(self.schema, type_id, lexical))
        });
        Annotation {
            type_name: self.type_annotation_name(type_id),
            typed_value,
            is_id: false,
            is_idref: false,
            nilled: None,
        }
    }

    /// The typed value of a node whose type is `xs:QName` or `xs:NOTATION`.
    ///
    /// [`typed_value`] cannot compute one on its own: the ·value· of a
    /// `QName` is a namespace name together with a local name, and turning a
    /// lexical form into them takes the namespace bindings in force where the
    /// value was *written* (Datatypes §3.2.18). Those are known here, so the
    /// prefix is resolved against the bindings of `owner` — the element the
    /// value sits on, or that carries the attribute it sits on. Returning
    /// [`None`] leaves the caller with the untyped fallback, which is what an
    /// unresolvable prefix deserves (the facet checks have already reported
    /// it).
    ///
    /// `saxonData/Assert/assert024` is the group this exists for: its
    /// assertion compares an attribute of type `xs:QName` against the value
    /// the `xs:QName` constructor builds, which no comparison of strings can
    /// answer.
    fn qualified_name_value(
        &self,
        owner: NodeId,
        type_id: TypeId,
        lexical: &str,
    ) -> Option<Sequence> {
        if !matches!(
            atomic_type(self.schema, type_id),
            Some(AtomicType::QName | AtomicType::Notation)
        ) {
            return None;
        }
        let lexical = lexical.trim();
        let (prefix, local) = match lexical.split_once(':') {
            Some((prefix, local)) => (Some(prefix), local),
            None => (None, lexical),
        };
        let namespace = match prefix {
            Some("xml") => Some(XML_NS.to_owned()),
            Some(prefix) => Some(
                self.in_scope_namespaces(owner, true)
                    .into_iter()
                    .find(|(seen, _)| seen.as_deref() == Some(prefix))
                    .map(|(_, namespace)| namespace)?,
            ),
            None => self
                .in_scope_namespaces(owner, true)
                .into_iter()
                .find(|(seen, _)| seen.is_none())
                .map(|(_, namespace)| namespace),
        };
        Some(Sequence::from(AtomicValue::qname(QNameValue::new(
            prefix, namespace, local,
        ))))
    }

    /// The name to put on the node, which is the type's own name when it has
    /// one and its nearest named ancestor's otherwise.
    fn type_annotation_name(&self, type_id: TypeId) -> Option<ExpandedName> {
        let mut current = Some(type_id);
        while let Some(id) = current {
            let definition = self.schema.type_definition(id);
            if let Some(name) = &definition.name {
                return Some(ExpandedName::new(name.namespace(), name.local_name()));
            }
            // A type that is its own base ends the walk rather than repeating
            // it, exactly as `Schema::derivation_path` does. Assembly rejects
            // such a definition (`src-type-cycle`), so this is a guard against
            // a schema that was never built rather than a case that arises.
            current = definition.base.filter(|base| *base != id);
        }
        None
    }
}

/// The typed value of a node of type `type_id` whose normalized content is
/// `lexical`.
///
/// A list yields one item per token, a union or an unrecognized type yields
/// `xs:untypedAtomic`, which is what an unvalidated node would carry anyway.
pub(crate) fn typed_value(schema: &Schema, type_id: TypeId, lexical: &str) -> Option<Sequence> {
    let simple = match schema.simple_type(type_id) {
        Some(simple) => simple,
        None => match &schema.complex_type(type_id)?.content {
            crate::components::ContentType::Simple(inner) => {
                let inner = *inner;
                return typed_value(schema, inner, lexical);
            }
            _ => return None,
        },
    };
    match &simple.variety {
        Variety::List(item) => {
            let item_type = atomic_type(schema, *item);
            Some(
                lexical
                    .split_whitespace()
                    .map(|token| atomic(item_type, token))
                    .collect(),
            )
        }
        _ => {
            let atomic_type = atomic_type(schema, type_id);
            Some(Sequence::from(atomic(atomic_type, lexical)))
        }
    }
}

/// The nearest built-in ancestor of `type_id`, which is the atomic type its
/// values are labelled with.
fn atomic_type(schema: &Schema, type_id: TypeId) -> Option<AtomicType> {
    let mut current = Some(type_id);
    while let Some(id) = current {
        let definition = schema.type_definition(id);
        if let Some(name) = &definition.name
            && name.namespace() == Some(XSD_NS)
            && let Some(atomic_type) = AtomicType::from_local_name(name.local_name())
        {
            return Some(atomic_type);
        }
        // As in `XdmView::type_annotation_name`: a type that is its own base
        // ends the walk instead of repeating it forever.
        current = definition.base.filter(|base| *base != id);
    }
    None
}

/// One atomic value of `atomic_type`, falling back to `xs:untypedAtomic` when
/// the type is unknown or the lexical form does not parse — a value that failed
/// to parse has already been reported by the facet checks, and an assertion
/// must not turn that into a second, spurious error.
fn atomic(atomic_type: Option<AtomicType>, lexical: &str) -> AtomicValue {
    match atomic_type {
        Some(AtomicType::QName | AtomicType::Notation) | None => {
            AtomicValue::untyped_atomic(lexical)
        }
        Some(atomic_type) => AtomicValue::parse(atomic_type, lexical)
            .unwrap_or_else(|_| AtomicValue::untyped_atomic(lexical)),
    }
}

/// The XDM instance an `assertion` facet is evaluated against, together with
/// the `$value` it binds (Part 2 §4.3.12).
///
/// The instance is a document node with a single element child that has no
/// attributes, no namespace declarations and no children, carrying the type
/// annotation and typed value of the value being checked; the Recommendation
/// leaves the fabricated element's name implementation-defined, and this one is
/// unqualified so that no `test` can accidentally match it by namespace.
pub(crate) fn facet_instance(
    schema: &Schema,
    type_id: TypeId,
    lexical: &str,
) -> Option<(Tree, Sequence)> {
    let value = typed_value(schema, type_id, lexical)
        .unwrap_or_else(|| Sequence::from(AtomicValue::untyped_atomic(lexical)));
    let mut builder = TreeBuilder::new();
    builder
        .start_annotated_element(
            ExpandedName::new(None::<&str>, FACET_ELEMENT),
            Annotation {
                type_name: None,
                typed_value: Some(value.clone()),
                is_id: false,
                is_idref: false,
                nilled: Some(false),
            },
        )
        .ok()?;
    builder.text(lexical).ok()?;
    builder.end_element().ok()?;
    Some((builder.build().ok()?, value))
}

/// The name of the element fabricated by [`facet_instance`].
const FACET_ELEMENT: &str = "value";

#[cfg(test)]
mod tests {
    use oxixml_infoset::{NodeKind as XdmNodeKind, XdmNode};

    use super::*;

    /// [`crate::assertion::evaluate`] takes the context item an assertion is
    /// evaluated against from [`Tree::document_element`]. A tree with no
    /// document node still has to answer it with the root element, or every
    /// assertion would silently become unsatisfied rather than only the ones
    /// that reach for the root.
    #[test]
    fn an_element_rooted_tree_still_names_its_root_element() {
        let mut builder = TreeBuilder::new_element_rooted();
        builder
            .start_element(ExpandedName::new(None::<&str>, "e"))
            .expect("an element-rooted builder accepts its root element");
        builder
            .end_element()
            .expect("the root element closes cleanly");
        let tree = builder.build().expect("one root element is a whole tree");
        let root = tree
            .document_element()
            .expect("the root element is the document element");
        assert_eq!(root.node_kind(), XdmNodeKind::Element);
        assert_eq!(
            root.node_name().as_ref().map(ExpandedName::local_name),
            Some("e")
        );
    }
}
