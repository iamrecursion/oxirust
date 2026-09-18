//! The arena document: storage, node factories and the document-order index.

mod attrs;
mod compare;
mod mutate;
mod navigate;
mod query;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use oxixml_xml::XmlVersion;

use crate::error::{DomError, DomResult, NotFound};
use crate::names::{
    check_attribute_ns, check_cdata, check_chars, check_comment, check_element_ns, check_name,
    check_pi_data, check_pi_target, expanded, split_qualified_name,
};
use crate::node::{
    AttrData, DocumentTypeData, ElementData, NodeContent, NodeData, NodeId, NodeKind,
    ProcessingInstructionData,
};

/// The number of nodes one document's arena can hold.
///
/// A [`NodeId`] is a `u32` index into the arena, and the index `u32::MAX` is
/// deliberately never handed out: it is what
/// [`Document::create_document_fragment`] returns for a full arena, where it
/// names no node at all rather than aliasing the last one, so that every later
/// use of it fails with [`DomError::NotFound`] instead of corrupting the tree.
/// Reaching this ceiling takes hundreds of gigabytes of memory,
/// since every node is well over 64 bytes.
pub(crate) const MAX_NODES: usize = u32::MAX as usize;

/// Returns the identifier the next node pushed into an arena of `len` nodes
/// receives, or `None` once the arena is full.
///
/// Kept apart from [`Document::push`] so that the boundary can be tested without
/// allocating an arena that large.
fn next_id(len: usize) -> Option<NodeId> {
    match u32::try_from(len) {
        Ok(index) if index != u32::MAX => Some(NodeId::from_index(index)),
        _ => None,
    }
}

/// Returns the identifier of the arena slot at `index`.
///
/// Because the arena never grows past [`MAX_NODES`], every index of an existing
/// node fits in a `u32`; an index that does not yields `u32::MAX`, which names
/// no node and so is rejected by every [`Document`] method rather than aliasing
/// the last slot.
fn id_at(index: usize) -> NodeId {
    NodeId::from_index(u32::try_from(index).unwrap_or(u32::MAX))
}

/// The lazily maintained document-order index.
#[derive(Debug, Clone)]
struct OrderIndex {
    dirty: bool,
    values: Vec<u32>,
}

impl Default for OrderIndex {
    fn default() -> Self {
        Self {
            dirty: true,
            values: Vec::new(),
        }
    }
}

/// A DOM Core document: an arena of nodes plus the document-level metadata that
/// the XML declaration and the document type declaration carry.
///
/// All nodes — elements, attributes, text, and the document node itself — live in
/// a single `Vec` owned by the `Document`, and are addressed by [`NodeId`]
/// handles. Removing a node detaches it from the tree but keeps its storage
/// alive, so the returned identifier stays usable and the node can be
/// re-inserted somewhere else; identifiers are therefore never invalidated by a
/// mutation.
///
/// # Example
///
/// ```
/// use oxixml_dom::Document;
///
/// let mut doc = Document::new();
/// let root = doc.create_element("catalog")?;
/// doc.append_child(doc.root(), root)?;
/// doc.set_attribute(root, "count", "0")?;
///
/// assert_eq!(doc.to_xml(), "<catalog count=\"0\"/>");
/// # Ok::<(), oxixml_dom::DomError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Document {
    nodes: Vec<NodeData>,
    xml_version: Option<String>,
    /// The version that selects the `Char` production the node factories apply,
    /// kept in step with `xml_version` and overridable by the builder when the
    /// caller forces a version that the declaration does not name.
    char_version: XmlVersion,
    encoding: Option<String>,
    standalone: Option<bool>,
    /// `element qualified name` (or `None` for "any element") -> attribute
    /// qualified names declared with the `ID` attribute type.
    id_declarations: HashMap<Option<String>, HashSet<String>>,
    order: RefCell<OrderIndex>,
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    /// Creates an empty document that contains nothing but its document node.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: vec![NodeData::new(NodeContent::Document)],
            xml_version: None,
            char_version: XmlVersion::Implicit1_0,
            encoding: None,
            standalone: None,
            id_declarations: HashMap::new(),
            order: RefCell::new(OrderIndex::default()),
        }
    }

    /// Returns the identifier of the document node, which is always
    /// [`NodeId::DOCUMENT`].
    #[must_use]
    pub const fn root(&self) -> NodeId {
        NodeId::DOCUMENT
    }

    /// Returns the number of nodes the arena currently holds, including detached
    /// ones and attribute nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Always returns `false`: a document always contains at least its document
    /// node.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the XML version declared by the document's XML declaration.
    #[must_use]
    pub fn xml_version(&self) -> Option<&str> {
        self.xml_version.as_deref()
    }

    /// Sets the XML version reported by [`Self::xml_version`].
    ///
    /// The version also selects the `Char` production the node factories apply:
    /// `"1.1"` widens it to the [XML 1.1](https://www.w3.org/TR/xml11/) one, so
    /// that a text node or attribute value may hold the C0 controls a 1.1
    /// document writes as character references. Anything else — including no
    /// declaration at all — keeps the XML 1.0 production. Content already in the
    /// tree is not re-validated, so narrowing the version afterwards does not
    /// retroactively reject it.
    pub fn set_xml_version(&mut self, version: Option<impl Into<String>>) {
        let version = version.map(Into::into);
        self.char_version = match version.as_deref() {
            None => XmlVersion::Implicit1_0,
            Some("1.1") => XmlVersion::Explicit1_1,
            Some(_) => XmlVersion::Explicit1_0,
        };
        self.xml_version = version;
    }

    /// The XML version that selects the `Char` production of the node factories.
    pub(crate) const fn char_version(&self) -> XmlVersion {
        self.char_version
    }

    /// Overrides the version that selects the `Char` production, without
    /// touching the declaration reported by [`Self::xml_version`].
    ///
    /// This is how [`BuilderOptions::with_xml_version`](crate::BuilderOptions::with_xml_version)
    /// forces a version onto a document whose declaration says something else,
    /// or says nothing at all.
    pub(crate) const fn set_char_version(&mut self, version: XmlVersion) {
        self.char_version = version;
    }

    /// Returns the character encoding declared by the document's XML
    /// declaration.
    #[must_use]
    pub fn encoding(&self) -> Option<&str> {
        self.encoding.as_deref()
    }

    /// Sets the encoding reported by [`Self::encoding`].
    pub fn set_encoding(&mut self, encoding: Option<impl Into<String>>) {
        self.encoding = encoding.map(Into::into);
    }

    /// Returns the value of the `standalone` pseudo-attribute of the XML
    /// declaration.
    #[must_use]
    pub const fn standalone(&self) -> Option<bool> {
        self.standalone
    }

    /// Sets the value reported by [`Self::standalone`].
    pub const fn set_standalone(&mut self, standalone: Option<bool>) {
        self.standalone = standalone;
    }

    // ---------------------------------------------------------------- storage

    pub(crate) fn get(&self, id: NodeId) -> Option<&NodeData> {
        self.nodes.get(id.index())
    }

    pub(crate) fn get_mut(&mut self, id: NodeId) -> Option<&mut NodeData> {
        self.nodes.get_mut(id.index())
    }

    pub(crate) fn node(&self, id: NodeId) -> DomResult<&NodeData> {
        self.get(id).ok_or(DomError::NotFound(NotFound::Node(id)))
    }

    pub(crate) fn node_mut(&mut self, id: NodeId) -> DomResult<&mut NodeData> {
        self.get_mut(id)
            .ok_or(DomError::NotFound(NotFound::Node(id)))
    }

    /// Adds a node to the arena and returns its identifier.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotSupported`] once the arena is full, which takes
    /// [`MAX_NODES`] nodes and so hundreds of gigabytes of memory.
    pub(crate) fn allocate(&mut self, content: NodeContent) -> DomResult<NodeId> {
        self.push(content).ok_or(DomError::NotSupported {
            detail: "the document already holds the maximum number of nodes",
        })
    }

    /// Adds a node to the arena, or returns `None` when it is full.
    pub(crate) fn push(&mut self, content: NodeContent) -> Option<NodeId> {
        let id = next_id(self.nodes.len())?;
        self.nodes.push(NodeData::new(content));
        self.invalidate_order();
        Some(id)
    }

    pub(crate) fn invalidate_order(&mut self) {
        self.order.get_mut().dirty = true;
    }

    /// Returns `true` when `id` refers to a node of this document.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        id.index() < self.nodes.len()
    }

    // -------------------------------------------------------------- accessors

    /// Returns the kind of the given node, or `None` when the identifier does not
    /// belong to this document.
    #[must_use]
    pub fn kind(&self, id: NodeId) -> Option<NodeKind> {
        self.get(id).map(NodeData::kind)
    }

    /// Returns the element payload of the node, or `None` when it is not an
    /// element.
    #[must_use]
    pub fn element(&self, id: NodeId) -> Option<&ElementData> {
        match &self.get(id)?.content {
            NodeContent::Element(data) => Some(data),
            _ => None,
        }
    }

    /// Returns the attribute payload of the node, or `None` when it is not an
    /// attribute.
    #[must_use]
    pub fn attr(&self, id: NodeId) -> Option<&AttrData> {
        match &self.get(id)?.content {
            NodeContent::Attr(data) => Some(data),
            _ => None,
        }
    }

    /// Returns the document type payload of the node, or `None` when it is not a
    /// document type declaration.
    #[must_use]
    pub fn doctype_data(&self, id: NodeId) -> Option<&DocumentTypeData> {
        match &self.get(id)?.content {
            NodeContent::DocumentType(data) => Some(data),
            _ => None,
        }
    }

    /// Returns the processing-instruction payload of the node, or `None` when it
    /// is not a processing instruction.
    #[must_use]
    pub fn processing_instruction(&self, id: NodeId) -> Option<&ProcessingInstructionData> {
        match &self.get(id)?.content {
            NodeContent::ProcessingInstruction(data) => Some(data),
            _ => None,
        }
    }

    /// Returns the character data of a [`Text`](NodeKind::Text),
    /// [`CdataSection`](NodeKind::CdataSection) or
    /// [`Comment`](NodeKind::Comment) node — the DOM `CharacterData.data`
    /// attribute.
    #[must_use]
    pub fn character_data(&self, id: NodeId) -> Option<&str> {
        match &self.get(id)?.content {
            NodeContent::Text(text)
            | NodeContent::CdataSection(text)
            | NodeContent::Comment(text) => Some(text.as_str()),
            _ => None,
        }
    }

    /// Returns the name of an [`EntityReference`](NodeKind::EntityReference)
    /// node, without the `&` and `;` delimiters.
    #[must_use]
    pub fn entity_reference(&self, id: NodeId) -> Option<&str> {
        match &self.get(id)?.content {
            NodeContent::EntityReference(name) => Some(name.as_str()),
            _ => None,
        }
    }

    /// Returns the DOM `nodeName` of the node: the qualified name for elements
    /// and attributes, the target for processing instructions, the declared name
    /// for the document type, the entity name for entity references, and the
    /// fixed strings `#text`, `#cdata-section`, `#comment`, `#document` and
    /// `#document-fragment` for the remaining kinds.
    #[must_use]
    pub fn node_name(&self, id: NodeId) -> Option<String> {
        Some(match &self.get(id)?.content {
            NodeContent::Document => "#document".to_owned(),
            NodeContent::DocumentFragment => "#document-fragment".to_owned(),
            NodeContent::DocumentType(data) => data.name.clone(),
            NodeContent::Element(data) => data.qualified_name(),
            NodeContent::Attr(data) => data.qualified_name(),
            NodeContent::Text(_) => "#text".to_owned(),
            NodeContent::CdataSection(_) => "#cdata-section".to_owned(),
            NodeContent::Comment(_) => "#comment".to_owned(),
            NodeContent::ProcessingInstruction(data) => data.target.clone(),
            NodeContent::EntityReference(name) => name.clone(),
        })
    }

    /// Returns the DOM `nodeValue` of the node: the character data for text,
    /// CDATA and comment nodes, the value for attributes, the data for processing
    /// instructions, and `None` for every other kind.
    #[must_use]
    pub fn node_value(&self, id: NodeId) -> Option<&str> {
        match &self.get(id)?.content {
            NodeContent::Text(text)
            | NodeContent::CdataSection(text)
            | NodeContent::Comment(text) => Some(text.as_str()),
            NodeContent::Attr(data) => Some(data.value.as_str()),
            NodeContent::ProcessingInstruction(data) => Some(data.data.as_str()),
            _ => None,
        }
    }

    /// Replaces the DOM `nodeValue` of a text, CDATA, comment, attribute or
    /// processing-instruction node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] for an unknown identifier,
    /// [`DomError::NotSupported`] when the node has no value, and
    /// [`DomError::InvalidCharacter`] when the value cannot be represented in
    /// XML.
    pub fn set_node_value(&mut self, id: NodeId, value: impl Into<String>) -> DomResult<()> {
        let value = value.into();
        match self.node(id)?.kind() {
            NodeKind::Text => check_chars(&value, self.char_version)?,
            NodeKind::CdataSection => check_cdata(&value, self.char_version)?,
            NodeKind::Comment => check_comment(&value, self.char_version)?,
            NodeKind::Attr => check_chars(&value, self.char_version)?,
            NodeKind::ProcessingInstruction => check_pi_data(&value, self.char_version)?,
            _ => {
                return Err(DomError::NotSupported {
                    detail: "this kind of node has no node value",
                });
            }
        }
        match &mut self.node_mut(id)?.content {
            NodeContent::Text(text)
            | NodeContent::CdataSection(text)
            | NodeContent::Comment(text) => *text = value,
            NodeContent::Attr(data) => data.value = value,
            NodeContent::ProcessingInstruction(data) => data.data = value,
            _ => unreachable!("kind checked above"),
        }
        Ok(())
    }

    // ------------------------------------------------------------- factories

    /// Creates an element in no namespace with the given name.
    ///
    /// The node is created detached; use [`Self::append_child`] or
    /// [`Self::insert_before`] to place it in the tree.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when `name` is not an XML `Name`.
    pub fn create_element(&mut self, name: &str) -> DomResult<NodeId> {
        check_name(name)?;
        let (prefix, local) = match name.split_once(':') {
            Some((prefix, local)) => (Some(prefix.to_owned()), local.to_owned()),
            None => (None, name.to_owned()),
        };
        self.allocate(NodeContent::Element(ElementData {
            name: expanded(None, &local),
            prefix,
            attributes: Vec::new(),
        }))
    }

    /// Creates a namespace-qualified element.
    ///
    /// `qualified_name` is `prefix:local` or just `local`; `namespace` is the
    /// namespace name the prefix (or the default declaration) is bound to.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when a part of the name is not an
    /// `NCName`, and [`DomError::Namespace`] when the combination violates
    /// [Namespaces in XML](https://www.w3.org/TR/xml-names/) — a prefix without a
    /// namespace, a misuse of the reserved `xml` / `xmlns` prefixes, or the
    /// `xmlns` namespace on an element.
    pub fn create_element_ns(
        &mut self,
        namespace: Option<&str>,
        qualified_name: &str,
    ) -> DomResult<NodeId> {
        let (prefix, local) = split_qualified_name(qualified_name)?;
        check_element_ns(namespace, prefix)?;
        self.allocate(NodeContent::Element(ElementData {
            name: expanded(namespace, local),
            prefix: prefix.map(str::to_owned),
            attributes: Vec::new(),
        }))
    }

    /// Creates a detached attribute node in no namespace.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when `name` is not an XML `Name`.
    pub fn create_attribute(&mut self, name: &str) -> DomResult<NodeId> {
        check_name(name)?;
        let (prefix, local) = match name.split_once(':') {
            Some((prefix, local)) => (Some(prefix.to_owned()), local.to_owned()),
            None => (None, name.to_owned()),
        };
        self.allocate(NodeContent::Attr(AttrData {
            name: expanded(None, &local),
            prefix,
            value: String::new(),
            specified: true,
            is_id: false,
            owner: None,
        }))
    }

    /// Creates a detached namespace-qualified attribute node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] or [`DomError::Namespace`] under
    /// the same conditions as [`Self::create_element_ns`], except that the
    /// `xmlns` namespace *is* permitted for the qualified names `xmlns` and
    /// `xmlns:*`, which is how namespace declarations are represented.
    pub fn create_attribute_ns(
        &mut self,
        namespace: Option<&str>,
        qualified_name: &str,
    ) -> DomResult<NodeId> {
        let (prefix, local) = split_qualified_name(qualified_name)?;
        check_attribute_ns(namespace, prefix, local)?;
        self.allocate(NodeContent::Attr(AttrData {
            name: expanded(namespace, local),
            prefix: prefix.map(str::to_owned),
            value: String::new(),
            specified: true,
            is_id: false,
            owner: None,
        }))
    }

    /// Creates a detached text node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when the data contains a character
    /// outside the XML `Char` production.
    pub fn create_text_node(&mut self, data: impl Into<String>) -> DomResult<NodeId> {
        let data = data.into();
        check_chars(&data, self.char_version)?;
        self.allocate(NodeContent::Text(data))
    }

    /// Creates a detached CDATA section.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when the data contains `]]>` or a
    /// character outside the XML `Char` production.
    pub fn create_cdata_section(&mut self, data: impl Into<String>) -> DomResult<NodeId> {
        let data = data.into();
        check_cdata(&data, self.char_version)?;
        self.allocate(NodeContent::CdataSection(data))
    }

    /// Creates a detached comment node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when the data contains `--`, ends
    /// with `-`, or contains a character outside the XML `Char` production.
    pub fn create_comment(&mut self, data: impl Into<String>) -> DomResult<NodeId> {
        let data = data.into();
        check_comment(&data, self.char_version)?;
        self.allocate(NodeContent::Comment(data))
    }

    /// Creates a detached processing instruction.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when `target` is not a colon-free
    /// `Name`, when it is any capitalization of `xml`, or when `data` contains
    /// `?>`.
    pub fn create_processing_instruction(
        &mut self,
        target: &str,
        data: impl Into<String>,
    ) -> DomResult<NodeId> {
        let data = data.into();
        check_pi_target(target)?;
        check_pi_data(&data, self.char_version)?;
        self.allocate(NodeContent::ProcessingInstruction(
            ProcessingInstructionData {
                target: target.to_owned(),
                data,
            },
        ))
    }

    /// Creates a detached entity-reference node for `&name;`.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when `name` is not an XML `Name`.
    pub fn create_entity_reference(&mut self, name: &str) -> DomResult<NodeId> {
        check_name(name)?;
        self.allocate(NodeContent::EntityReference(name.to_owned()))
    }

    /// Creates a detached document fragment.
    ///
    /// A fragment is the one node whose creation cannot fail, so this is the
    /// only factory that does not return a [`DomResult`]. Should the arena ever
    /// be full — which takes hundreds of gigabytes of nodes — the returned
    /// identifier names no node, and every operation that is given it fails with
    /// [`DomError::NotFound`].
    #[must_use]
    pub fn create_document_fragment(&mut self) -> NodeId {
        debug_assert!(
            self.nodes.len() < MAX_NODES,
            "the node arena must never be allowed to fill up"
        );
        self.push(NodeContent::DocumentFragment)
            .unwrap_or_else(|| id_at(MAX_NODES))
    }

    /// Creates a detached document type declaration node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::InvalidCharacter`] when `name` is not an XML `Name`.
    pub fn create_document_type(
        &mut self,
        name: &str,
        public_id: Option<&str>,
        system_id: Option<&str>,
    ) -> DomResult<NodeId> {
        check_name(name)?;
        self.allocate(NodeContent::DocumentType(DocumentTypeData {
            name: name.to_owned(),
            public_id: public_id.map(str::to_owned),
            system_id: system_id.map(str::to_owned),
            internal_subset: None,
        }))
    }

    /// Replaces the internal subset text of a document type node.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] for an unknown identifier and
    /// [`DomError::NotSupported`] when the node is not a document type.
    pub fn set_internal_subset(
        &mut self,
        id: NodeId,
        subset: Option<impl Into<String>>,
    ) -> DomResult<()> {
        match &mut self.node_mut(id)?.content {
            NodeContent::DocumentType(data) => {
                data.internal_subset = subset.map(Into::into);
                Ok(())
            }
            _ => Err(DomError::NotSupported {
                detail: "only a document type node has an internal subset",
            }),
        }
    }

    // ------------------------------------------------------- ID declarations

    /// Declares that the attribute `attribute_name` of `element_name` has the
    /// DTD attribute type `ID`, so that
    /// [`Self::get_element_by_id`] considers it.
    ///
    /// Pass `None` for `element_name` to declare the attribute name for every
    /// element, which is what a host that only knows the attribute name (for
    /// instance from a schema) can do. Names are compared as *qualified* names,
    /// exactly as an `<!ATTLIST>` declaration spells them.
    pub fn declare_id_attribute(&mut self, element_name: Option<&str>, attribute_name: &str) {
        self.id_declarations
            .entry(element_name.map(str::to_owned))
            .or_default()
            .insert(attribute_name.to_owned());
    }

    /// Returns `true` when an `<!ATTLIST>`-style `ID` declaration covers the
    /// attribute `attribute_name` of an element named `element_name`.
    #[must_use]
    pub fn is_declared_id_attribute(&self, element_name: &str, attribute_name: &str) -> bool {
        let by_element = self
            .id_declarations
            .get(&Some(element_name.to_owned()))
            .is_some_and(|set| set.contains(attribute_name));
        by_element
            || self
                .id_declarations
                .get(&None)
                .is_some_and(|set| set.contains(attribute_name))
    }

    /// Marks (or unmarks) an individual attribute node as being of type `ID`,
    /// mirroring DOM Level 3's `Element.setIdAttributeNode`.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] for an unknown identifier and
    /// [`DomError::NotSupported`] when the node is not an attribute.
    pub fn set_id_attribute_node(&mut self, id: NodeId, is_id: bool) -> DomResult<()> {
        match &mut self.node_mut(id)?.content {
            NodeContent::Attr(data) => {
                data.is_id = is_id;
                Ok(())
            }
            _ => Err(DomError::NotSupported {
                detail: "only an attribute node can be marked as an ID",
            }),
        }
    }

    // ------------------------------------------------------- document order

    /// Recomputes the document-order index if a mutation has invalidated it.
    fn refresh_order(&self) {
        {
            let index = self.order.borrow();
            if !index.dirty && index.values.len() == self.nodes.len() {
                return;
            }
        }

        let mut values = vec![u32::MAX; self.nodes.len()];
        let mut counter: u32 = 0;
        let mut stack: Vec<NodeId> = Vec::new();

        self.walk_order(NodeId::DOCUMENT, &mut values, &mut counter, &mut stack);

        // Detached subtrees keep a stable, self-consistent relative order: each
        // detached root is visited (in arena order) after the live tree.
        for index in 0..self.nodes.len() {
            if values[index] != u32::MAX {
                continue;
            }
            let root = self.detached_root(id_at(index));
            if values[root.index()] == u32::MAX {
                self.walk_order(root, &mut values, &mut counter, &mut stack);
            }
            if values[index] == u32::MAX {
                values[index] = counter;
                counter = counter.saturating_add(1);
            }
        }

        let mut index = self.order.borrow_mut();
        index.values = values;
        index.dirty = false;
    }

    fn walk_order(
        &self,
        from: NodeId,
        values: &mut [u32],
        counter: &mut u32,
        stack: &mut Vec<NodeId>,
    ) {
        stack.clear();
        stack.push(from);
        while let Some(id) = stack.pop() {
            let Some(node) = self.get(id) else { continue };
            if values[id.index()] != u32::MAX {
                continue;
            }
            values[id.index()] = *counter;
            *counter = counter.saturating_add(1);

            let mut children = Vec::new();
            let mut cursor = node.first_child;
            while let Some(child) = cursor {
                children.push(child);
                cursor = self.get(child).and_then(|node| node.next_sibling);
            }
            for child in children.into_iter().rev() {
                stack.push(child);
            }
            if let NodeContent::Element(data) = &node.content {
                for attribute in data.attributes.iter().rev() {
                    stack.push(*attribute);
                }
            }
        }
    }

    fn detached_root(&self, id: NodeId) -> NodeId {
        let mut current = id;
        loop {
            let Some(node) = self.get(current) else {
                return current;
            };
            let up = match &node.content {
                NodeContent::Attr(data) => data.owner,
                _ => node.parent,
            };
            match up {
                Some(parent) if parent != current => current = parent,
                _ => return current,
            }
        }
    }

    /// Returns the document-order position of a node.
    ///
    /// Document order is the order defined by
    /// [XPath 1.0 §5](https://www.w3.org/TR/xpath-10/#data-model): the document
    /// node first, then each node in the order in which its first character
    /// appears in the serialized document, with an element's attributes placed
    /// immediately after the element and before its children.
    ///
    /// The index is cached and recomputed lazily after a structural mutation, so
    /// repeated queries are cheap.
    #[must_use]
    pub fn document_order(&self, id: NodeId) -> Option<u32> {
        if !self.contains(id) {
            return None;
        }
        self.refresh_order();
        self.order.borrow().values.get(id.index()).copied()
    }

    /// Compares two nodes in document order.
    ///
    /// Returns `None` when either identifier is unknown to this document.
    #[must_use]
    pub fn compare_document_order(&self, a: NodeId, b: NodeId) -> Option<std::cmp::Ordering> {
        Some(self.document_order(a)?.cmp(&self.document_order(b)?))
    }

    /// Returns every node of the document, sorted in document order, starting at
    /// the document node. Detached nodes are included after the live tree.
    #[must_use]
    pub fn nodes_in_document_order(&self) -> Vec<NodeId> {
        self.refresh_order();
        let values = &self.order.borrow().values;
        let mut ids: Vec<NodeId> = (0..self.nodes.len()).map(id_at).collect();
        ids.sort_by_key(|id| values.get(id.index()).copied().unwrap_or(u32::MAX));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_NODES, id_at, next_id};
    use crate::node::NodeId;

    #[test]
    fn identifiers_are_handed_out_in_arena_order() {
        assert_eq!(next_id(0), Some(NodeId::DOCUMENT));
        assert_eq!(next_id(1), Some(NodeId::from_index(1)));
        assert_eq!(
            next_id(MAX_NODES - 1),
            Some(NodeId::from_index(u32::MAX - 1))
        );
    }

    #[test]
    fn a_full_arena_hands_out_no_identifier_at_all() {
        // Saturating here would return the identifier of the *last* node again,
        // silently aliasing two nodes onto one arena slot.
        assert_eq!(next_id(MAX_NODES), None);
        assert_eq!(next_id(usize::MAX), None);
    }

    #[test]
    fn an_out_of_range_slot_names_no_node() {
        let index = id_at(MAX_NODES);
        assert_eq!(index.index(), MAX_NODES);
        assert!(
            crate::Document::new().get(index).is_none(),
            "the saturated identifier must never resolve to a node"
        );
    }
}
