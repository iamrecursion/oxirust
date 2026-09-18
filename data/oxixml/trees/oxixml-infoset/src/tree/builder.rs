//! [`TreeBuilder`] — constructs a [`Tree`] node by node, and materializes it into an arena laid
//! out in document order.

use super::dtd::{Notation, NotationDeclaration, UnparsedEntity, UnparsedEntityDeclaration};
use super::position::SourcePosition;
use super::store::{NodeData, TreeStore};
use super::tree_impl::Tree;
use crate::error::{ErrorCode, XdmError};
use crate::node::{NodeKind, NodeRef, XdmNode};
use crate::sequence::Sequence;
use oxixml_iri::{Iri, IriRef};
use oxixml_qname::{ExpandedName, XML_NAMESPACE, XMLNS_NAMESPACE};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The schema-derived properties a builder can attach to an element or attribute.
///
/// A non-validating tree leaves every field at its default, which is what
/// [`TreeBuilder::start_element`] and [`TreeBuilder::attribute`] do. A schema-aware caller
/// supplies the annotations it computed, and [`XdmNode::typed_value`](crate::XdmNode::typed_value),
/// [`XdmNode::type_name`](crate::XdmNode::type_name), [`XdmNode::nilled`](crate::XdmNode::nilled),
/// [`XdmNode::is_id`](crate::XdmNode::is_id) and [`XdmNode::is_idref`](crate::XdmNode::is_idref)
/// report them.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{AtomicValue, Annotation, ExpandedName, Sequence, TreeBuilder, XdmNode};
///
/// let mut builder = TreeBuilder::new();
/// builder.start_element(ExpandedName::new(None::<&str>, "n"))?;
/// builder.annotated_attribute(
///     ExpandedName::new(None::<&str>, "count"),
///     "42",
///     Annotation {
///         typed_value: Some(Sequence::from(AtomicValue::integer(42_i64))),
///         ..Annotation::default()
///     },
/// )?;
/// builder.end_element()?;
/// let tree = builder.build()?;
///
/// let attribute = tree
///     .document_element()
///     .and_then(|element| element.attributes().first().cloned())
///     .expect("one attribute");
/// assert_eq!(attribute.typed_value()?.len(), 1);
/// assert_eq!(attribute.string_value(), "42");
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct Annotation {
    /// The node's type annotation.
    pub type_name: Option<ExpandedName>,
    /// The node's typed value. `None` means "compute the untyped default".
    pub typed_value: Option<Sequence>,
    /// Whether the node is an ID.
    pub is_id: bool,
    /// Whether the node is an IDREF or IDREFS.
    pub is_idref: bool,
    /// Whether an element is nilled.
    pub nilled: Option<bool>,
}

#[derive(Debug, Clone)]
struct RawNode {
    kind: NodeKind,
    name: Option<ExpandedName>,
    value: String,
    children: Vec<usize>,
    attributes: Vec<usize>,
    /// `(prefix, namespace name)` pairs declared *on* this element. An empty namespace name
    /// undeclares the prefix, as Namespaces in XML 1.1 allows.
    declarations: Vec<(Option<String>, String)>,
    annotation: Annotation,
}

impl RawNode {
    fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            name: None,
            value: String::new(),
            children: Vec::new(),
            attributes: Vec::new(),
            declarations: Vec::new(),
            annotation: Annotation::default(),
        }
    }
}

/// One suspended element in [`TreeBuilder::materialize`]'s explicit work-list: everything needed to
/// resume emitting the element's attributes and children, and to write the resulting index lists
/// back into the element once they are all done.
struct ElementFrame {
    /// Index of the element in the output arena.
    self_index: usize,
    /// The namespace bindings in scope inside the element, shared with its descendants.
    scope: Arc<Vec<(Option<String>, String)>>,
    /// The element's base URI, inherited by everything it contains.
    base: Option<String>,
    namespace_ids: Vec<usize>,
    attributes: Vec<usize>,
    next_attribute: usize,
    attribute_ids: Vec<usize>,
    children: Vec<usize>,
    next_child: usize,
    child_ids: Vec<usize>,
}

/// What sits at the root of the tree a [`TreeBuilder`] produces.
///
/// The XDM allows both: [XDM 3.1 §6](https://www.w3.org/TR/xpath-datamodel-31/#Node) requires
/// only that every node belong to a tree with a unique root, and lists the *document node* as
/// one possible root among several. Which one a tree has is chosen when the builder is created
/// and cannot change afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootKind {
    /// A document node, with the element as its child — what parsing an XML document produces.
    Document,
    /// The element itself, with nothing above it.
    Element,
}

/// Builds a [`Tree`].
///
/// The builder mirrors the shape of a SAX event stream: open an element, declare its namespaces
/// and attributes, emit its content, close it. Two data-model axioms are enforced as you go, so
/// a tree that comes out of `build` can never violate them:
///
/// * **no adjacent text nodes** — consecutive [`text`](TreeBuilder::text) calls merge into one
///   node, and empty text is dropped;
/// * **namespace and attribute nodes precede content** — declaring a namespace or adding an
///   attribute after the current element already has children is rejected.
///
/// In-scope namespace nodes are materialized by [`build`](TreeBuilder::build), not as you go:
/// every element ends up with a namespace node for each binding in scope on it, including
/// those inherited from its ancestors and the implicit `xml` binding, exactly as
/// [XDM 3.1 §6.2](https://www.w3.org/TR/xpath-datamodel-31/#NamespaceNode) requires.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{ExpandedName, NodeKind, TreeBuilder, XdmNode};
///
/// let mut builder = TreeBuilder::with_document_uri("http://example.com/doc.xml");
/// builder.start_element(ExpandedName::new(Some("urn:example"), "root"))?;
/// builder.declare_namespace(Some("ex"), "urn:example")?;
/// builder.attribute(ExpandedName::new(None::<&str>, "id"), "r1")?;
/// builder.text("first ")?;
/// builder.text("and second")?; // merged with the previous call
/// builder.comment("a note")?;
/// builder.end_element()?;
/// let tree = builder.build()?;
///
/// let root = tree.document_element().expect("a document element");
/// assert_eq!(root.string_value(), "first and second");
/// assert_eq!(root.children().len(), 2, "one text node and one comment");
/// assert_eq!(root.base_uri().as_deref(), Some("http://example.com/doc.xml"));
///
/// // `ex` and the implicit `xml` binding are both namespace nodes.
/// let prefixes: Vec<_> = root
///     .namespaces()
///     .iter()
///     .map(|namespace| namespace.node_name().map(|name| name.local_name().to_owned()))
///     .collect();
/// assert_eq!(prefixes, [Some("ex".to_owned()), Some("xml".to_owned())]);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
///
/// # Element-rooted trees
///
/// [`new_element_rooted`](TreeBuilder::new_element_rooted) builds a tree whose root *is* the
/// element, with no document node above it. That is the data model an XSD 1.1 assertion is
/// evaluated against ([XML Schema 1.1 Part 1
/// §3.13.4](https://www.w3.org/TR/xmlschema11-1/#cvc-assertion)), where a descendant search must
/// not be able to climb out of the element being asserted.
///
/// ```
/// use oxixml_infoset::{ExpandedName, NodeKind, TreeBuilder, XdmNode};
///
/// let mut builder = TreeBuilder::new_element_rooted();
/// builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
/// builder.text("content")?;
/// builder.end_element()?;
/// let tree = builder.build()?;
///
/// let root = tree.root();
/// assert_eq!(root.node_kind(), NodeKind::Element);
/// assert_eq!(root.parent(), None, "nothing is above the root");
/// assert_eq!(root.root(), root, "fn:root(.) is the element itself");
/// assert_eq!(root.string_value(), "content");
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone)]
pub struct TreeBuilder {
    raw: Vec<RawNode>,
    open: Vec<usize>,
    /// Seeds the base URI of the whole tree, and — for a document-rooted tree — is the
    /// `document-uri` property of its document node.
    document_uri: Option<String>,
    root_kind: RootKind,
    saw_document_element: bool,
    unparsed_entities: BTreeMap<String, UnparsedEntityDeclaration>,
    notations: BTreeMap<String, NotationDeclaration>,
    /// Where the node the next `push_child` creates began in the source, when the caller is
    /// recording source positions. Consumed by that call, so a position can never be attached to
    /// a later node than the one it was set for.
    next_position: Option<SourcePosition>,
    /// Source positions by *raw* node index, which `build` maps to the arena's own indices.
    /// Empty unless the caller sets positions.
    positions: BTreeMap<usize, SourcePosition>,
    /// The base URI of the external parsed entity whose content is being built, if any.
    /// See [`set_entity_base_uri`](TreeBuilder::set_entity_base_uri).
    entity_base: Option<String>,
    /// The entity base URI of each node that *begins* content spliced in from an external
    /// parsed entity, by *raw* node index. A node whose parent already carries the same entity
    /// base inherits it in the ordinary way and is not listed, so the map holds one entry per
    /// splice rather than one per node — and stays empty for a document that splices nothing.
    entity_bases: BTreeMap<usize, String>,
}

impl Default for TreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeBuilder {
    /// A builder for a document with no document URI.
    #[must_use]
    pub fn new() -> Self {
        Self::rooted(RootKind::Document, None)
    }

    /// A builder for a document with the given absolute document URI, which also seeds the
    /// base URI of the whole tree.
    #[must_use]
    pub fn with_document_uri(uri: impl Into<String>) -> Self {
        Self::rooted(RootKind::Document, Some(uri.into()))
    }

    /// A builder whose tree is rooted at the element itself, with **no document node above it**.
    ///
    /// Everything else behaves as it does for a document-rooted builder, with two consequences
    /// that follow from there being no document node:
    ///
    /// * exactly one element must be opened at the top level, and nothing else may appear beside
    ///   it — [`comment`](TreeBuilder::comment) and
    ///   [`processing_instruction`](TreeBuilder::processing_instruction) are rejected while no
    ///   element is open, and [`build`](TreeBuilder::build) fails if no element was ever opened;
    /// * the resulting tree has no `document-uri`, no unparsed entities and no notations —
    ///   [`declare_unparsed_entity`](TreeBuilder::declare_unparsed_entity) and
    ///   [`declare_notation`](TreeBuilder::declare_notation) are rejected.
    ///
    /// The root element's [`parent`](crate::XdmNode::parent) is `None` and its
    /// [`root`](crate::XdmNode::root) — `fn:root(.)` — is the element itself.
    #[must_use]
    pub fn new_element_rooted() -> Self {
        Self::rooted(RootKind::Element, None)
    }

    /// An [element-rooted](TreeBuilder::new_element_rooted) builder whose tree inherits `uri` as
    /// its base URI, as the element would have inherited it from the document it was taken out
    /// of.
    ///
    /// The tree still has no document node and therefore no `document-uri`; only
    /// [`base_uri`](crate::XdmNode::base_uri) is seeded.
    #[must_use]
    pub fn element_rooted_with_base_uri(uri: impl Into<String>) -> Self {
        Self::rooted(RootKind::Element, Some(uri.into()))
    }

    fn rooted(root_kind: RootKind, document_uri: Option<String>) -> Self {
        Self {
            // Index 0 is the builder's own staging parent: it collects whatever is emitted at
            // the top level, and becomes the document node only for a document-rooted tree.
            raw: vec![RawNode::new(NodeKind::Document)],
            open: Vec::new(),
            document_uri,
            root_kind,
            saw_document_element: false,
            unparsed_entities: BTreeMap::new(),
            notations: BTreeMap::new(),
            next_position: None,
            positions: BTreeMap::new(),
            entity_base: None,
            entity_bases: BTreeMap::new(),
        }
    }

    /// Records where the *next* node this builder creates began in the source text.
    ///
    /// `None` clears the pending position, which is how a node whose source coordinates are not
    /// known — one built from an entity's replacement text, say — is left without one.
    pub(crate) const fn set_source_position(&mut self, position: Option<SourcePosition>) {
        self.next_position = position;
    }

    /// The base URI of the external parsed entity whose content is being built, if any.
    pub(crate) fn entity_base_uri(&self) -> Option<&str> {
        self.entity_base.as_deref()
    }

    /// Records that the nodes built from now on come from an external parsed entity retrieved
    /// from `uri`, until this is set again.
    ///
    /// [XML Base §4.2](https://www.w3.org/TR/xmlbase/#granularity) resolves a relative reference
    /// against the entity the node appears in, so content spliced in from an external entity
    /// keeps that entity's URI rather than the referencing document's. Only the nodes that
    /// *begin* the spliced content need it: everything below them inherits it in the ordinary
    /// way, which is what keeps an `xml:base` written inside the entity meaningful.
    ///
    /// `None` — the initial state, and the one to restore when the splice ends — leaves every
    /// node with the base URI it inherits from its parent.
    pub(crate) fn set_entity_base_uri(&mut self, uri: Option<String>) {
        self.entity_base = uri;
    }

    /// Whether this builder produces a tree rooted at the element itself.
    #[must_use]
    pub const fn is_element_rooted(&self) -> bool {
        matches!(self.root_kind, RootKind::Element)
    }

    fn current(&self) -> usize {
        self.open.last().copied().unwrap_or(0)
    }

    fn push_child(&mut self, node: RawNode) -> usize {
        let index = self.raw.len();
        let parent = self.current();
        self.raw.push(node);
        if let Some(data) = self.raw.get_mut(parent) {
            data.children.push(index);
        }
        if let Some(position) = self.next_position.take() {
            self.positions.insert(index, position);
        }
        // The node begins spliced content when its parent does not already stand in the same
        // external entity — which is exactly the case the base URI has to be recorded for.
        let begins_entity = match self.entity_base.as_deref() {
            Some(base) if self.entity_bases.get(&parent).map(String::as_str) != Some(base) => {
                Some(base.to_owned())
            }
            _ => None,
        };
        if let Some(base) = begins_entity {
            self.entity_bases.insert(index, base);
        }
        index
    }

    fn ensure_open_element(&self, what: &str) -> Result<usize, XdmError> {
        self.open.last().copied().ok_or_else(|| {
            XdmError::new(
                ErrorCode::Fodc0006,
                format!("{what} requires an open element"),
            )
        })
    }

    fn ensure_no_children_yet(&self, index: usize, what: &str) -> Result<(), XdmError> {
        let has_children = self
            .raw
            .get(index)
            .is_some_and(|data| !data.children.is_empty());
        if has_children {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("{what} must come before the element's content"),
            ));
        }
        Ok(())
    }

    /// Opens an element with the given expanded name.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if a second document element would be opened.
    pub fn start_element(&mut self, name: ExpandedName) -> Result<&mut Self, XdmError> {
        self.start_annotated_element(name, Annotation::default())
    }

    /// Opens an element carrying schema annotations.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if a second document element would be opened.
    pub fn start_annotated_element(
        &mut self,
        name: ExpandedName,
        annotation: Annotation,
    ) -> Result<&mut Self, XdmError> {
        if self.open.is_empty() {
            if self.saw_document_element {
                return Err(XdmError::new(
                    ErrorCode::Fodc0006,
                    "a document has exactly one document element",
                ));
            }
            self.saw_document_element = true;
        }
        let mut node = RawNode::new(NodeKind::Element);
        node.name = Some(name);
        node.annotation = annotation;
        let index = self.push_child(node);
        self.open.push(index);
        Ok(self)
    }

    /// Closes the innermost open element.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if no element is open.
    pub fn end_element(&mut self) -> Result<&mut Self, XdmError> {
        self.open.pop().ok_or_else(|| {
            XdmError::new(ErrorCode::Fodc0006, "there is no open element to close")
        })?;
        Ok(self)
    }

    /// Declares a namespace binding on the innermost open element.
    ///
    /// `prefix` is `None` for the default namespace. An empty `namespace` *undeclares* the
    /// prefix for the subtree, which Namespaces in XML 1.1 permits for a prefix and Namespaces
    /// in XML 1.0 permits for the default namespace.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if no element is open, if the element already has content, or if
    /// the declaration would rebind the reserved `xml` or `xmlns` prefix to the wrong namespace.
    pub fn declare_namespace(
        &mut self,
        prefix: Option<&str>,
        namespace: &str,
    ) -> Result<&mut Self, XdmError> {
        let index = self.ensure_open_element("a namespace declaration")?;
        self.ensure_no_children_yet(index, "a namespace declaration")?;
        match prefix {
            Some("xml") if namespace != XML_NAMESPACE => {
                return Err(XdmError::new(
                    ErrorCode::Fodc0006,
                    "the `xml` prefix may only be bound to the XML namespace",
                ));
            }
            Some("xmlns") => {
                return Err(XdmError::new(
                    ErrorCode::Fodc0006,
                    "the `xmlns` prefix may not be declared",
                ));
            }
            _ if namespace == XMLNS_NAMESPACE => {
                return Err(XdmError::new(
                    ErrorCode::Fodc0006,
                    "no prefix may be bound to the xmlns namespace",
                ));
            }
            _ => {}
        }
        if let Some(data) = self.raw.get_mut(index) {
            data.declarations
                .push((prefix.map(str::to_owned), namespace.to_owned()));
        }
        Ok(self)
    }

    /// Adds an attribute to the innermost open element.
    ///
    /// An attribute named `xml:id` is automatically marked as an ID.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if no element is open, if the element already has content, if the
    /// attribute name is in the `xmlns` namespace (namespace declarations are not attributes in
    /// the data model), or if the element already has an attribute with this name.
    pub fn attribute(
        &mut self,
        name: ExpandedName,
        value: impl Into<String>,
    ) -> Result<&mut Self, XdmError> {
        let is_id = name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "id";
        self.annotated_attribute(
            name,
            value,
            Annotation {
                is_id,
                ..Annotation::default()
            },
        )
    }

    /// Adds an attribute carrying schema annotations.
    ///
    /// # Errors
    ///
    /// The same conditions as [`TreeBuilder::attribute`].
    pub fn annotated_attribute(
        &mut self,
        name: ExpandedName,
        value: impl Into<String>,
        annotation: Annotation,
    ) -> Result<&mut Self, XdmError> {
        let index = self.ensure_open_element("an attribute")?;
        self.ensure_no_children_yet(index, "an attribute")?;
        if name.namespace() == Some(XMLNS_NAMESPACE) {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                "a namespace declaration is not an attribute; use declare_namespace",
            ));
        }
        let already_present = self.raw.get(index).is_some_and(|element| {
            element.attributes.iter().any(|attribute| {
                self.raw.get(*attribute).and_then(|data| data.name.as_ref()) == Some(&name)
            })
        });
        if already_present {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("the element already has an attribute named `{name}`"),
            ));
        }
        let mut node = RawNode::new(NodeKind::Attribute);
        node.name = Some(name);
        node.value = value.into();
        node.annotation = annotation;
        let attribute_index = self.raw.len();
        self.raw.push(node);
        if let Some(element) = self.raw.get_mut(index) {
            element.attributes.push(attribute_index);
        }
        Ok(self)
    }

    /// Appends character data to the innermost open element.
    ///
    /// Consecutive calls merge into a single text node, and empty content is ignored, so the
    /// "no adjacent text nodes" axiom holds by construction.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if no element is open — a document node has no text children.
    pub fn text(&mut self, content: impl AsRef<str>) -> Result<&mut Self, XdmError> {
        let content = content.as_ref();
        if content.is_empty() {
            return Ok(self);
        }
        let parent = self.ensure_open_element("character data")?;
        let last_child = self
            .raw
            .get(parent)
            .and_then(|data| data.children.last().copied());
        let merge_into = last_child.filter(|child| {
            self.raw
                .get(*child)
                .is_some_and(|data| data.kind == NodeKind::Text)
        });
        match merge_into {
            Some(child) => {
                // The merged-into node keeps the position of its own first character, so the
                // pending one belongs to nothing and must not survive to the next node.
                self.next_position = None;
                if let Some(data) = self.raw.get_mut(child) {
                    data.value.push_str(content);
                }
            }
            None => {
                let mut node = RawNode::new(NodeKind::Text);
                node.value = content.to_owned();
                self.push_child(node);
            }
        }
        Ok(self)
    }

    /// Records an unparsed (`NDATA`) general entity declared by the document's DTD.
    ///
    /// The declaration becomes part of the document node's `unparsed-entities` property
    /// ([XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)), which
    /// [`Tree::unparsed_entity`](crate::Tree::unparsed_entity) and
    /// [`XdmNode::unparsed_entity`](crate::XdmNode::unparsed_entity) read back. `system_id` is
    /// resolved against the tree's document URI by [`build`](TreeBuilder::build), so it may be
    /// the relative reference the declaration wrote.
    ///
    /// As for entity declarations generally (XML 1.0 §4.2), the *first* declaration of a name
    /// binds; a later one for the same name is ignored.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `name` is empty, or if the builder is
    /// [element-rooted](TreeBuilder::new_element_rooted) — such a tree has no document node to
    /// carry the property.
    pub fn declare_unparsed_entity(
        &mut self,
        name: impl Into<String>,
        public_id: Option<&str>,
        system_id: impl Into<String>,
        notation: impl Into<String>,
    ) -> Result<&mut Self, XdmError> {
        let name = name.into();
        self.ensure_declarable(&name, "an unparsed entity declaration")?;
        self.unparsed_entities
            .entry(name)
            .or_insert_with(|| UnparsedEntityDeclaration {
                public_id: public_id.map(str::to_owned),
                system_id: system_id.into(),
                notation: notation.into(),
                uri: None,
            });
        Ok(self)
    }

    /// Records a notation declared by the document's DTD.
    ///
    /// The first declaration of a name binds, as it does for entities.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `name` is empty, or if the builder is
    /// [element-rooted](TreeBuilder::new_element_rooted).
    pub fn declare_notation(
        &mut self,
        name: impl Into<String>,
        public_id: Option<&str>,
        system_id: Option<&str>,
    ) -> Result<&mut Self, XdmError> {
        let name = name.into();
        self.ensure_declarable(&name, "a notation declaration")?;
        self.notations
            .entry(name)
            .or_insert_with(|| NotationDeclaration {
                public_id: public_id.map(str::to_owned),
                system_id: system_id.map(str::to_owned),
                uri: None,
            });
        Ok(self)
    }

    /// Carries the `[unparsed entities]` and `[notations]` of the document `source` belongs to
    /// onto the document node being built.
    ///
    /// The properties are read from `source`'s [`root`](crate::XdmNode::root), which is where
    /// `fn:unparsed-entity-uri` reads them from too; a tree whose root is not a document node
    /// declares neither, and carrying from one is a no-op.
    ///
    /// [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) makes both
    /// properties of the *document node*, so a copy that stands for the same document has to
    /// carry them for `fn:unparsed-entity-uri` and `fn:unparsed-entity-public-id` to answer the
    /// same about the copy as about the original — which is what `fn:copy-of`, `fn:snapshot`
    /// and space stripping produce. Each identifier is carried **already resolved**: the copy
    /// need not have a document URI of its own, and re-resolving a relative system identifier
    /// against a different one — or against none — would change or lose the absolute URI the
    /// data model asks for.
    ///
    /// A name this builder already has a declaration for keeps that declaration, as the first
    /// declaration of a name binds generally (XML 1.0 §4.2).
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if the builder is
    /// [element-rooted](TreeBuilder::new_element_rooted) — such a tree has no document node to
    /// carry the properties.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{ParseOptions, Tree, TreeBuilder, ExpandedName, XdmNode};
    ///
    /// let tree = Tree::parse_str_with(
    ///     r#"<!DOCTYPE r [
    ///          <!NOTATION gif SYSTEM "image/gif">
    ///          <!ENTITY pic SYSTEM "pic.gif" NDATA gif>
    ///        ]><r/>"#,
    ///     ParseOptions::default().document_uri("http://example.com/a/d.xml"),
    /// )?;
    ///
    /// // The copy has no document URI of its own, and still reports the absolute URI.
    /// let mut builder = TreeBuilder::new();
    /// builder.carry_declarations(&tree.root())?;
    /// builder.start_element(ExpandedName::new(None::<&str>, "r"))?.end_element()?;
    /// let copy = builder.build()?;
    ///
    /// let entity = copy.root().unparsed_entity("pic").expect("the entity was carried");
    /// assert_eq!(entity.uri.as_deref(), Some("http://example.com/a/pic.gif"));
    /// assert_eq!(entity.notation, "gif");
    /// assert!(copy.root().notation("gif").is_some());
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn carry_declarations(&mut self, source: &NodeRef) -> Result<&mut Self, XdmError> {
        if self.is_element_rooted() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                "carrying unparsed entities and notations requires a tree rooted at a document \
                 node",
            ));
        }
        let source = source.root();
        for name in source.unparsed_entity_names() {
            let Some(entity) = source.unparsed_entity(&name) else {
                continue;
            };
            self.unparsed_entities
                .entry(name)
                .or_insert_with(|| UnparsedEntityDeclaration {
                    public_id: entity.public_id.clone(),
                    system_id: entity.system_id.clone(),
                    notation: entity.notation.clone(),
                    uri: entity.uri.clone(),
                });
        }
        for name in source.notation_names() {
            let Some(notation) = source.notation(&name) else {
                continue;
            };
            self.notations
                .entry(name)
                .or_insert_with(|| NotationDeclaration {
                    public_id: notation.public_id.clone(),
                    system_id: notation.system_id.clone(),
                    uri: notation.uri.clone(),
                });
        }
        Ok(self)
    }

    /// Checks that `name` names something a document node can carry, and that there is one.
    fn ensure_declarable(&self, name: &str, what: &str) -> Result<(), XdmError> {
        if self.is_element_rooted() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("{what} requires a tree rooted at a document node"),
            ));
        }
        if name.is_empty() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("{what} must name the thing it declares"),
            ));
        }
        Ok(())
    }

    /// Rejects content that may only sit beside a document element, on a tree that has no
    /// document node to hold it.
    fn ensure_top_level_allowed(&self, what: &str) -> Result<(), XdmError> {
        if self.is_element_rooted() && self.open.is_empty() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("{what} beside the root element requires a document node above it"),
            ));
        }
        Ok(())
    }

    /// Appends a comment.
    ///
    /// Comments are legal both inside the document element and beside it.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if the builder is
    /// [element-rooted](TreeBuilder::new_element_rooted) and no element is open, because such a
    /// tree has no node above the root element for the comment to be a child of.
    pub fn comment(&mut self, content: impl Into<String>) -> Result<&mut Self, XdmError> {
        self.ensure_top_level_allowed("a comment")?;
        let mut node = RawNode::new(NodeKind::Comment);
        node.value = content.into();
        self.push_child(node);
        Ok(self)
    }

    /// Appends a processing instruction with the given target and content.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `target` is empty, or if the builder is
    /// [element-rooted](TreeBuilder::new_element_rooted) and no element is open.
    pub fn processing_instruction(
        &mut self,
        target: impl AsRef<str>,
        content: impl Into<String>,
    ) -> Result<&mut Self, XdmError> {
        let target = target.as_ref();
        if target.is_empty() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                "a processing instruction must have a target",
            ));
        }
        self.ensure_top_level_allowed("a processing instruction")?;
        let mut node = RawNode::new(NodeKind::ProcessingInstruction);
        node.name = Some(ExpandedName::new(None::<&str>, target));
        node.value = content.into();
        self.push_child(node);
        Ok(self)
    }

    /// Materializes the tree.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if an element is still open, or — for an
    /// [element-rooted](TreeBuilder::new_element_rooted) builder — if the top level does not
    /// hold exactly one element.
    pub fn build(self) -> Result<Tree, XdmError> {
        if !self.open.is_empty() {
            return Err(XdmError::new(
                ErrorCode::Fodc0006,
                format!("{} element(s) are still open", self.open.len()),
            ));
        }
        let top_level = self
            .raw
            .first()
            .map(|data| data.children.clone())
            .unwrap_or_default();
        let in_scope = vec![(Some("xml".to_owned()), XML_NAMESPACE.to_owned())];
        let base = self.document_uri.clone();
        let mut out: Vec<NodeData> = Vec::with_capacity(self.raw.len());
        let mut emitted_positions: BTreeMap<usize, SourcePosition> = BTreeMap::new();

        match self.root_kind {
            RootKind::Document => {
                let mut document = NodeData::new(NodeKind::Document);
                document.base_uri = base.clone();
                out.push(document);
                let mut children = Vec::new();
                for child in top_level {
                    if let Some(id) = self.materialize(
                        child,
                        Some(0),
                        &in_scope,
                        base.as_deref(),
                        &mut out,
                        &mut emitted_positions,
                    ) {
                        children.push(id);
                    }
                }
                if let Some(document) = out.first_mut() {
                    document.children = children;
                }
            }
            RootKind::Element => {
                // The root element lands at index 0 with no parent above it. Nothing may sit
                // beside it: a second top-level node would be a second root, and an empty top
                // level would leave the store with no nodes at all.
                let [only] = top_level.as_slice() else {
                    return Err(XdmError::new(
                        ErrorCode::Fodc0006,
                        format!(
                            "an element-rooted tree has exactly one top-level node, but {} were \
                             emitted",
                            top_level.len()
                        ),
                    ));
                };
                // The emitted index is 0 by construction, so it is the check below — not the
                // return value — that decides whether the store has a usable root.
                let _ = self.materialize(
                    *only,
                    None,
                    &in_scope,
                    base.as_deref(),
                    &mut out,
                    &mut emitted_positions,
                );
                if out.first().map(|root| root.kind) != Some(NodeKind::Element) {
                    return Err(XdmError::new(
                        ErrorCode::Fodc0006,
                        "the root of an element-rooted tree must be an element",
                    ));
                }
            }
        }

        let document_uri = match self.root_kind {
            RootKind::Document => self.document_uri.clone(),
            // A tree with no document node has no `document-uri`; the URI only seeded the base
            // URI of what it contains.
            RootKind::Element => None,
        };
        Ok(Tree::from_store(Arc::new(TreeStore::new(
            out,
            document_uri,
            self.resolved_unparsed_entities(),
            self.resolved_notations(),
            emitted_positions,
        ))))
    }

    /// Resolves each unparsed entity's system identifier against the tree's document URI, or
    /// keeps the URI a [carried](TreeBuilder::carry_declarations) declaration already had.
    fn resolved_unparsed_entities(&self) -> BTreeMap<String, UnparsedEntity> {
        self.unparsed_entities
            .iter()
            .map(|(name, declaration)| {
                (
                    name.clone(),
                    UnparsedEntity {
                        uri: declaration.uri.clone().or_else(|| {
                            resolve_system_id(self.document_uri.as_deref(), &declaration.system_id)
                        }),
                        system_id: declaration.system_id.clone(),
                        public_id: declaration.public_id.clone(),
                        notation: declaration.notation.clone(),
                    },
                )
            })
            .collect()
    }

    /// Resolves each notation's system identifier against the tree's document URI, or keeps the
    /// URI a [carried](TreeBuilder::carry_declarations) declaration already had.
    fn resolved_notations(&self) -> BTreeMap<String, Notation> {
        self.notations
            .iter()
            .map(|(name, declaration)| {
                (
                    name.clone(),
                    Notation {
                        uri: declaration.uri.clone().or_else(|| {
                            declaration.system_id.as_deref().and_then(|system_id| {
                                resolve_system_id(self.document_uri.as_deref(), system_id)
                            })
                        }),
                        system_id: declaration.system_id.clone(),
                        public_id: declaration.public_id.clone(),
                    },
                )
            })
            .collect()
    }

    /// Emits `index` and its subtree into `out`, in document order, returning the final index of
    /// the node itself, or `None` when `index` names no raw node.
    ///
    /// `parent` is the index the node's parent already occupies in `out`, or `None` for the root
    /// of an [element-rooted](TreeBuilder::new_element_rooted) tree.
    ///
    /// The walk is driven by an explicit work-list rather than by Rust's call stack: a source
    /// document is allowed to be arbitrarily deeply nested, and one native frame per level would
    /// overflow the stack — an *uncatchable* process abort, not a panic — on a document that is
    /// otherwise perfectly ordinary.
    fn materialize(
        &self,
        index: usize,
        parent: Option<usize>,
        in_scope: &[(Option<String>, String)],
        base: Option<&str>,
        out: &mut Vec<NodeData>,
        emitted_positions: &mut BTreeMap<usize, SourcePosition>,
    ) -> Option<usize> {
        let scope = Arc::new(in_scope.to_vec());
        let (root_index, root_frame) = self.emit_node(index, parent, &scope, base, out)?;
        self.carry_position(index, root_index, emitted_positions);

        let mut stack: Vec<ElementFrame> = Vec::new();
        if let Some(frame) = root_frame {
            stack.push(frame);
        }

        while let Some(top) = stack.len().checked_sub(1) {
            // An element's attribute nodes come first in document order, then
            // its children; both are emitted through the same work-list so that
            // no branch of the walk can recurse.
            let next = {
                let frame = &stack[top];
                if frame.next_attribute < frame.attributes.len() {
                    Some((frame.attributes[frame.next_attribute], true))
                } else if frame.next_child < frame.children.len() {
                    Some((frame.children[frame.next_child], false))
                } else {
                    None
                }
            };

            let Some((raw_index, is_attribute)) = next else {
                let Some(frame) = stack.pop() else { break };
                if let Some(element) = out.get_mut(frame.self_index) {
                    element.namespaces = frame.namespace_ids;
                    element.attributes = frame.attribute_ids;
                    element.children = frame.child_ids;
                }
                continue;
            };

            let (self_index, scope, element_base) = {
                let frame = &mut stack[top];
                if is_attribute {
                    frame.next_attribute += 1;
                } else {
                    frame.next_child += 1;
                }
                (
                    frame.self_index,
                    Arc::clone(&frame.scope),
                    frame.base.clone(),
                )
            };

            if let Some((emitted, frame)) = self.emit_node(
                raw_index,
                Some(self_index),
                &scope,
                element_base.as_deref(),
                out,
            ) {
                self.carry_position(raw_index, emitted, emitted_positions);
                let parent_frame = &mut stack[top];
                if is_attribute {
                    parent_frame.attribute_ids.push(emitted);
                } else {
                    parent_frame.child_ids.push(emitted);
                }
                if let Some(frame) = frame {
                    stack.push(frame);
                }
            }
        }

        Some(root_index)
    }

    /// Re-keys the source position of the raw node `raw_index` onto the arena index the node was
    /// emitted to. Does nothing for a builder that recorded no positions.
    fn carry_position(
        &self,
        raw_index: usize,
        emitted: usize,
        emitted_positions: &mut BTreeMap<usize, SourcePosition>,
    ) {
        if let Some(position) = self.positions.get(&raw_index) {
            emitted_positions.insert(emitted, *position);
        }
    }

    /// Emits the node at `index` (and, for an element, its namespace nodes) into `out`.
    ///
    /// Returns the index the node was written to, together with the [`ElementFrame`] that the
    /// caller's work-list must process to emit the element's attributes and children — `None` for
    /// a node that has neither. Returns `None` altogether when `index` names no raw node.
    fn emit_node(
        &self,
        index: usize,
        parent: Option<usize>,
        in_scope: &Arc<Vec<(Option<String>, String)>>,
        base: Option<&str>,
        out: &mut Vec<NodeData>,
    ) -> Option<(usize, Option<ElementFrame>)> {
        let raw = self.raw.get(index)?;
        // Content spliced in from an external parsed entity is based on *that entity*, not on
        // the document that referenced it. An element and a processing instruction each have a
        // `[base URI]` property of their own (XML Infoset §2.2, §2.6) and so take it; a comment
        // and a text node have none and read their parent's, which is what leaving them on the
        // inherited base does.
        let base = match self.entity_bases.get(&index) {
            Some(entity)
                if matches!(
                    raw.kind,
                    NodeKind::Element | NodeKind::ProcessingInstruction
                ) =>
            {
                Some(entity.as_str())
            }
            _ => base,
        };
        let self_index = out.len();
        let mut data = NodeData::new(raw.kind);
        data.parent = parent;
        data.name = raw.name.clone();
        data.value = raw.value.clone();
        data.type_name = raw.annotation.type_name.clone();
        data.typed_value = raw.annotation.typed_value.clone();
        data.nilled = raw.annotation.nilled;
        data.is_id = raw.annotation.is_id;
        data.is_idref = raw.annotation.is_idref;
        data.base_uri = base.map(str::to_owned);

        if raw.kind != NodeKind::Element {
            out.push(data);
            return Some((self_index, None));
        }

        let scope = if raw.declarations.is_empty() {
            Arc::clone(in_scope)
        } else {
            Arc::new(apply_declarations(in_scope, &raw.declarations))
        };
        let element_base = self.element_base_uri(raw, base);
        data.base_uri = element_base.clone();
        out.push(data);

        let mut namespace_ids = Vec::with_capacity(scope.len());
        for (prefix, namespace) in scope.iter() {
            let namespace_index = out.len();
            let mut node = NodeData::new(NodeKind::Namespace);
            node.parent = Some(self_index);
            node.name = prefix
                .as_ref()
                .map(|prefix| ExpandedName::new(None::<&str>, prefix.clone()));
            node.value = namespace.clone();
            // XDM 3.1 §6.2: a namespace node has no base URI.
            out.push(node);
            namespace_ids.push(namespace_index);
        }

        let frame = ElementFrame {
            self_index,
            scope,
            base: element_base,
            namespace_ids,
            attributes: raw.attributes.clone(),
            next_attribute: 0,
            attribute_ids: Vec::with_capacity(raw.attributes.len()),
            children: raw.children.clone(),
            next_child: 0,
            child_ids: Vec::with_capacity(raw.children.len()),
        };
        Some((self_index, Some(frame)))
    }

    /// Resolves the element's `xml:base` attribute, if any, against the inherited base URI.
    fn element_base_uri(&self, raw: &RawNode, inherited: Option<&str>) -> Option<String> {
        let explicit = raw.attributes.iter().find_map(|attribute| {
            let data = self.raw.get(*attribute)?;
            let name = data.name.as_ref()?;
            (name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "base")
                .then(|| data.value.clone())
        });
        match explicit {
            None => inherited.map(str::to_owned),
            Some(explicit) => resolve_base_uri(inherited, &explicit),
        }
    }
}

/// Applies an element's namespace declarations to the bindings inherited from its ancestors.
///
/// The result is sorted by prefix, with the default namespace (if bound) first, which fixes an
/// otherwise unspecified order for the resulting namespace nodes.
fn apply_declarations(
    in_scope: &[(Option<String>, String)],
    declarations: &[(Option<String>, String)],
) -> Vec<(Option<String>, String)> {
    let mut scope = in_scope.to_vec();
    for (prefix, namespace) in declarations {
        scope.retain(|(existing, _)| existing != prefix);
        if !namespace.is_empty() {
            scope.push((prefix.clone(), namespace.clone()));
        }
    }
    scope.sort_by(|left, right| left.0.cmp(&right.0));
    scope
}

/// Resolves an `xml:base` value against the base URI in force, per
/// [XML Base](https://www.w3.org/TR/xmlbase/).
fn resolve_base_uri(inherited: Option<&str>, explicit: &str) -> Option<String> {
    let Some(inherited) = inherited else {
        return Some(explicit.to_owned());
    };
    let (Ok(base), Ok(reference)) = (Iri::parse(inherited), IriRef::parse(explicit)) else {
        // An unparseable base or reference leaves the declared value as-is, which is the most
        // useful behaviour for the non-validating case.
        return Some(explicit.to_owned());
    };
    base.resolve(&reference)
        .map(Iri::into_inner)
        .ok()
        .or_else(|| Some(explicit.to_owned()))
}

/// Resolves the system identifier of a DTD declaration against the base URI in force where the
/// declaration appeared.
///
/// Unlike [`resolve_base_uri`], this never falls back to the value as written: XDM 3.1 §6.1 asks
/// for the *absolute* URI of an unparsed entity, and a relative reference with nothing to
/// resolve it against does not have one.
pub(super) fn resolve_system_id(base: Option<&str>, system_id: &str) -> Option<String> {
    if Iri::parse(system_id).is_ok() {
        return Some(system_id.to_owned());
    }
    let base = Iri::parse(base?).ok()?;
    let reference = IriRef::parse(system_id).ok()?;
    base.resolve(&reference).map(Iri::into_inner).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::XdmNode;

    #[test]
    fn adjacent_text_is_merged_and_empty_text_dropped() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
        builder.text("one ")?;
        builder.text("")?;
        builder.text("two")?;
        builder.end_element()?;
        let tree = builder.build()?;

        let root = tree.document_element().expect("a document element");
        assert_eq!(root.children().len(), 1);
        assert_eq!(root.string_value(), "one two");
        Ok(())
    }

    #[test]
    fn declarations_and_attributes_must_precede_content() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
        builder.text("x")?;
        assert!(builder.declare_namespace(Some("p"), "urn:p").is_err());
        assert!(
            builder
                .attribute(ExpandedName::new(None::<&str>, "k"), "v")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn structural_errors_are_reported() {
        let mut builder = TreeBuilder::new();
        assert!(builder.text("x").is_err(), "no text at document level");
        assert!(builder.end_element().is_err(), "nothing is open");
        assert!(builder.processing_instruction("", "x").is_err());

        let mut builder = TreeBuilder::new();
        assert!(
            builder
                .start_element(ExpandedName::new(None::<&str>, "a"))
                .is_ok()
        );
        assert!(builder.end_element().is_ok());
        assert!(
            builder
                .start_element(ExpandedName::new(None::<&str>, "b"))
                .is_err(),
            "a second document element"
        );

        let mut builder = TreeBuilder::new();
        assert!(
            builder
                .start_element(ExpandedName::new(None::<&str>, "a"))
                .is_ok()
        );
        assert!(builder.clone().build().is_err(), "an element is still open");
    }

    #[test]
    fn duplicate_attributes_and_namespace_pseudo_attributes_are_rejected() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
        builder.attribute(ExpandedName::new(None::<&str>, "k"), "1")?;
        assert!(
            builder
                .attribute(ExpandedName::new(None::<&str>, "k"), "2")
                .is_err()
        );
        assert!(
            builder
                .attribute(ExpandedName::new(Some(XMLNS_NAMESPACE), "p"), "urn:p")
                .is_err()
        );
        assert!(builder.declare_namespace(Some("xmlns"), "urn:x").is_err());
        assert!(builder.declare_namespace(Some("xml"), "urn:x").is_err());
        assert!(
            builder
                .declare_namespace(Some("xml"), XML_NAMESPACE)
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn xml_id_is_detected_automatically() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
        builder.attribute(ExpandedName::new(Some(XML_NAMESPACE), "id"), "x1")?;
        builder.attribute(ExpandedName::new(None::<&str>, "id"), "x2")?;
        builder.end_element()?;
        let tree = builder.build()?;

        let attributes = tree
            .document_element()
            .expect("a document element")
            .attributes();
        assert!(attributes[0].is_id());
        assert!(!attributes[1].is_id());
        Ok(())
    }

    #[test]
    fn namespace_undeclaration_removes_the_binding() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(Some("urn:outer"), "outer"))?;
        builder.declare_namespace(None, "urn:outer")?;
        builder.start_element(ExpandedName::new(None::<&str>, "inner"))?;
        builder.declare_namespace(None, "")?;
        builder.end_element()?;
        builder.end_element()?;
        let tree = builder.build()?;

        let outer = tree.document_element().expect("a document element");
        assert_eq!(outer.namespaces().len(), 2, "the default binding and `xml`");
        let inner = outer.children().first().cloned().expect("one child");
        assert_eq!(inner.namespaces().len(), 1, "only `xml` remains");
        Ok(())
    }

    #[test]
    fn a_system_identifier_is_only_reported_when_it_resolves_to_an_absolute_uri() {
        assert_eq!(
            resolve_system_id(Some("http://example.com/a/b.xml"), "img/logo.png").as_deref(),
            Some("http://example.com/a/img/logo.png")
        );
        assert_eq!(
            resolve_system_id(Some("http://example.com/a/b.xml"), "/logo.png").as_deref(),
            Some("http://example.com/logo.png")
        );
        // Already absolute: the base URI is irrelevant, and there need not be one.
        assert_eq!(
            resolve_system_id(None, "http://example.org/e.bin").as_deref(),
            Some("http://example.org/e.bin")
        );
        // Relative with nothing to resolve against, and an unusable base, have no absolute form.
        assert_eq!(resolve_system_id(None, "logo.png"), None);
        assert_eq!(resolve_system_id(Some("not an iri"), "logo.png"), None);
    }

    #[test]
    fn an_element_rooted_builder_reports_which_kind_it_is() -> Result<(), XdmError> {
        assert!(!TreeBuilder::new().is_element_rooted());
        assert!(!TreeBuilder::with_document_uri("urn:x").is_element_rooted());
        assert!(TreeBuilder::new_element_rooted().is_element_rooted());
        assert!(TreeBuilder::element_rooted_with_base_uri("urn:x").is_element_rooted());

        // The root element is the whole store: nothing precedes it in document order.
        let mut builder = TreeBuilder::new_element_rooted();
        builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
        builder.end_element()?;
        let tree = builder.build()?;
        assert_eq!(tree.root().document_order_key().position, 0);
        assert_eq!(tree.len(), 2, "the element and its `xml` namespace node");
        Ok(())
    }

    #[test]
    fn xml_base_is_resolved_against_the_inherited_base() -> Result<(), XdmError> {
        let mut builder = TreeBuilder::with_document_uri("http://example.com/a/b.xml");
        builder.start_element(ExpandedName::new(None::<&str>, "root"))?;
        builder.attribute(ExpandedName::new(Some(XML_NAMESPACE), "base"), "../c/")?;
        builder.start_element(ExpandedName::new(None::<&str>, "child"))?;
        builder.attribute(ExpandedName::new(Some(XML_NAMESPACE), "base"), "d.xml")?;
        builder.end_element()?;
        builder.end_element()?;
        let tree = builder.build()?;

        let root = tree.document_element().expect("a document element");
        assert_eq!(root.base_uri().as_deref(), Some("http://example.com/c/"));
        let child = root.children().first().cloned().expect("one child");
        assert_eq!(
            child.base_uri().as_deref(),
            Some("http://example.com/c/d.xml")
        );
        // An attribute inherits its element's base URI; a namespace node has none.
        assert_eq!(
            child.attributes()[0].base_uri().as_deref(),
            Some("http://example.com/c/d.xml")
        );
        assert_eq!(child.namespaces()[0].base_uri(), None);
        Ok(())
    }
}
