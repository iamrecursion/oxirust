//! [`ResultBuilder`] — the receiver a sequence constructor pushes nodes into.
//!
//! The builder mirrors the shape of the *result tree* an XSLT instruction writes: open an
//! element, add its namespaces and attributes, emit its content, close it. Two things happen
//! when the tree is materialized that the caller never has to think about:
//!
//! * **namespace fixup** ([XSLT 3.0 §5.7.3](https://www.w3.org/TR/xslt-30/#namespace-fixup)) —
//!   every element and attribute name is guaranteed a namespace binding, and a prefix that
//!   would collide is replaced by a generated one;
//! * **document order** — the arena is laid out so that an element precedes its namespace
//!   nodes, which precede its attributes, which precede its children.

use super::store::{Declarations, ResultNode, ResultStore, ResultTree};
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};
use oxixml_qname::{XML_NAMESPACE, XMLNS_NAMESPACE};

/// What the element nodes built inside an element's content see of that element's namespaces.
///
/// [XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces) makes namespace
/// inheritance a property of the *instruction* that constructs an element, not of the result
/// tree's parent-child relation, which is why there are three cases and not two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inheritance {
    /// The element's whole in-scope set, which `inherit-namespaces="yes"` — the default of
    /// `xsl:element`, `xsl:copy` and a literal result element — prescribes.
    Scope,
    /// Whatever the element itself inherited, without the namespace nodes it added:
    /// `inherit-namespaces="no"`.
    ///
    /// [XSLT 3.0 §11.4.2](https://www.w3.org/TR/xslt-30/#element-element) is precise about which
    /// nodes stop here — "the namespace nodes created *for the newly constructed element*
    /// (whether these were copied from those of the source node, or generated as a result of
    /// namespace fixup) … are not automatically copied to the children" — and the ones the
    /// element merely inherited from *its* parent are not among them. `element-0306` measures
    /// exactly that distinction: two `xsl:element` instructions under an
    /// `inherit-namespaces="no"` parent, each expected to keep the two bindings the stylesheet's
    /// outermost element declared and to lose only the parent's own.
    ///
    /// A chain of blocked elements therefore still starves: nothing was inherited at the top, so
    /// nothing is passed on (`copy-0613`, `copy-0615`, `copy-0617`…).
    Blocked,
    /// Whatever the element itself inherited. A deep copy has no `inherit-namespaces` attribute
    /// of its own: every element of the copied subtree sees the namespaces in scope *where the
    /// copy was written*, and none of them acquires a namespace merely because its copied parent
    /// needed one. `copy-1221` turns on exactly that — the copy of `<a a:att="A"><aa/></a>`
    /// under `copy-namespaces="no"` gives `a` the binding its attribute needs and `aa` only the
    /// one its own name needs.
    Copied,
}

#[derive(Debug, Clone)]
struct RawNode {
    kind: NodeKind,
    name: Option<ExpandedName>,
    prefix: Option<String>,
    value: String,
    unescaped: bool,
    children: Vec<usize>,
    attributes: Vec<usize>,
    declarations: Vec<(Option<String>, String)>,
    inherit: Inheritance,
    /// The base URI of the node this one is a *copy* of, when it is one.
    ///
    /// [XSLT 3.0 §11.9.1](https://www.w3.org/TR/xslt-30/#element-copy) says that "when a node is
    /// copied, its base URI is copied", and [§11.9.2](https://www.w3.org/TR/xslt-30/#element-copy-of)
    /// completes the rule from the other side: "if the copied node is subsequently attached as a
    /// child to a new element or document node, the final copy of the node inherits its base URI
    /// from its parent node". A builder cannot tell the two apart until it is finished — the
    /// copy is attached exactly when it is not a top-level item of a sequence — so what the
    /// original had is remembered here and consulted in [`ResultBuilder::build`], where the
    /// question is finally settled. `base-uri-024` asks for the parentless half and `base-uri-035`
    /// for the attached one, over almost the same stylesheet.
    copied_base_uri: Option<String>,
    typed_id: bool,
    typed_idref: bool,
}

impl RawNode {
    fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            name: None,
            prefix: None,
            value: String::new(),
            unescaped: false,
            children: Vec::new(),
            attributes: Vec::new(),
            declarations: Vec::new(),
            inherit: Inheritance::Scope,
            copied_base_uri: None,
            typed_id: false,
            typed_idref: false,
        }
    }
}

/// Builds a [`ResultTree`].
#[derive(Debug, Clone)]
pub(crate) struct ResultBuilder {
    raw: Vec<RawNode>,
    open: Vec<usize>,
    document_uri: Option<String>,
    base_uri: Option<String>,
    declarations: Declarations,
    merge_root_text: bool,
}

impl ResultBuilder {
    /// A builder whose top-level text nodes merge, which is what a result tree fragment wants.
    pub(crate) fn new() -> Self {
        Self {
            raw: vec![RawNode::new(NodeKind::Document)],
            open: Vec::new(),
            document_uri: None,
            base_uri: None,
            declarations: Declarations::default(),
            merge_root_text: true,
        }
    }

    /// A builder whose top-level text nodes stay separate, which is what a sequence wants.
    pub(crate) fn for_sequence() -> Self {
        let mut builder = Self::new();
        builder.merge_root_text = false;
        builder
    }

    /// Sets the document URI — what `fn:document-uri` answers for the document node.
    ///
    /// A tree a stylesheet constructs has none: only a document the processor *retrieved* is
    /// identified by a URI. The base URI is a separate property, set through
    /// [`Self::set_base_uri`].
    pub(crate) fn set_document_uri(&mut self, uri: Option<String>) {
        self.document_uri = uri;
    }

    /// Sets the base URI the tree is built under.
    ///
    /// [XSLT 3.0 §9.2](https://www.w3.org/TR/xslt-30/#temporary-trees): the nodes of a temporary
    /// tree take their base URI "as if all the nodes in the temporary tree came from a single
    /// entity whose URI was the base URI of the variable-binding element", so this is the static
    /// base URI of the instruction that builds the tree — the one an `xml:base` in the
    /// stylesheet has already been applied to — and not the URI of any document.
    pub(crate) fn set_base_uri(&mut self, uri: Option<String>) {
        self.base_uri = uri;
    }

    /// Carries the DTD declarations of `source` onto the document node being built.
    ///
    /// Only a copy that *is* the same document has any business doing this — the stripped
    /// counterpart `xsl:strip-space` builds. [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)
    /// makes `[unparsed entities]` and `[notations]` properties of the document node, and space
    /// stripping changes none of them, so `fn:unparsed-entity-uri` must answer the same for the
    /// stripped tree as for the one it was made from.
    pub(crate) fn carry_declarations(&mut self, source: &NodeRef) {
        for name in source.unparsed_entity_names() {
            if let Some(entity) = source.unparsed_entity(&name) {
                self.declarations.unparsed_entities.insert(name, entity);
            }
        }
        for name in source.notation_names() {
            if let Some(notation) = source.notation(&name) {
                self.declarations.notations.insert(name, notation);
            }
        }
    }

    /// Whether an element is currently open.
    pub(crate) fn in_element(&self) -> bool {
        !self.open.is_empty()
    }

    /// The number of nodes the document node has collected so far.
    pub(crate) fn root_child_count(&self) -> usize {
        self.raw.first().map_or(0, |data| data.children.len())
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
        index
    }

    /// Opens an element whose content inherits its namespaces, which is the default.
    pub(crate) fn start_element(&mut self, name: ExpandedName, prefix: Option<String>) {
        self.open_element(name, prefix, Inheritance::Scope);
    }

    /// Opens an element, saying whether the elements built inside its content inherit its
    /// namespaces — the `inherit-namespaces` attribute of
    /// [XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces).
    pub(crate) fn start_element_with(
        &mut self,
        name: ExpandedName,
        prefix: Option<String>,
        inherit: bool,
    ) {
        let inheritance = if inherit {
            Inheritance::Scope
        } else {
            Inheritance::Blocked
        };
        self.open_element(name, prefix, inheritance);
    }

    /// Records, on the element that is open, the base URI of the node it is a copy of.
    ///
    /// A no-op when nothing is open, which is where a caller copying a node that is not an
    /// element ends up: only an element and a document node carry a base URI of their own, and a
    /// document node's is set through [`ResultBuilder::set_base_uri`] on a builder of its own.
    /// See [`RawNode::copied_base_uri`].
    pub(crate) fn note_copied_base_uri(&mut self, uri: Option<String>) {
        let Some(index) = self.open.last().copied() else {
            return;
        };
        if let Some(data) = self.raw.get_mut(index) {
            data.copied_base_uri = uri;
        }
    }

    fn open_element(&mut self, name: ExpandedName, prefix: Option<String>, inherit: Inheritance) {
        let mut node = RawNode::new(NodeKind::Element);
        node.name = Some(name);
        node.prefix = prefix;
        node.inherit = inherit;
        let index = self.push_child(node);
        self.open.push(index);
    }

    /// Closes the innermost open element.
    ///
    /// # Errors
    ///
    /// `XTDE0001` if no element is open.
    pub(crate) fn end_element(&mut self) -> XsltResult<()> {
        self.open
            .pop()
            .map(|_| ())
            .ok_or_else(|| XsltError::dynamic("XTDE0001", "there is no open element to close"))
    }

    fn attributes_still_allowed(&self, what: &str) -> XsltResult<usize> {
        let Some(index) = self.open.last().copied() else {
            // Nothing is open, so the node is being added to the content of the *document*
            // node the builder is filling. XSLT 3.0 §5.7.1 gives that its own code: `XTDE0410`
            // is an attribute misplaced within an element's content, `XTDE0420` an attribute in
            // the content of a document node.
            return Err(XsltError::dynamic(
                "XTDE0420",
                format!("{what} may not appear in the content of a document node"),
            ));
        };
        if self
            .raw
            .get(index)
            .is_some_and(|data| !data.children.is_empty())
        {
            return Err(XsltError::dynamic(
                "XTDE0410",
                format!("{what} must be added before the element has content"),
            ));
        }
        Ok(index)
    }

    /// Adds an attribute to the innermost open element, replacing any attribute of the same
    /// name, as [XSLT 3.0 §11.2](https://www.w3.org/TR/xslt-30/#creating-attributes) requires.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when no element is open or the element already has content.
    pub(crate) fn attribute(
        &mut self,
        name: ExpandedName,
        prefix: Option<String>,
        value: impl Into<String>,
    ) -> XsltResult<()> {
        self.typed_attribute(name, prefix, value, false, false)
    }

    /// Adds an attribute that carries the ID typing of the source attribute it copies.
    ///
    /// Only the re-materialization of a *source* document uses this — a whitespace-stripped tree
    /// is the same document under [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stripping), so
    /// the `[is-id]` and `[is-idrefs]` properties a DTD gave its attributes survive it. An
    /// attribute a stylesheet *constructs* is untyped whatever it is copied from, which is what
    /// [`Self::attribute`] builds.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when no element is open or the element already has content.
    pub(crate) fn typed_attribute(
        &mut self,
        name: ExpandedName,
        prefix: Option<String>,
        value: impl Into<String>,
        is_id: bool,
        is_idref: bool,
    ) -> XsltResult<()> {
        let index = self.attributes_still_allowed("an attribute")?;
        if name.namespace() == Some(XMLNS_NAMESPACE) {
            return Err(XsltError::dynamic(
                "XTDE0865",
                "a namespace declaration is not an attribute",
            ));
        }
        let value = value.into();
        let existing = self.raw.get(index).and_then(|element| {
            element
                .attributes
                .iter()
                .copied()
                .find(|slot| self.raw.get(*slot).and_then(|d| d.name.as_ref()) == Some(&name))
        });
        if let Some(slot) = existing {
            if let Some(data) = self.raw.get_mut(slot) {
                data.value = value;
                data.prefix = prefix;
                data.typed_id = is_id;
                data.typed_idref = is_idref;
            }
            return Ok(());
        }
        let mut node = RawNode::new(NodeKind::Attribute);
        node.name = Some(name);
        node.prefix = prefix;
        node.value = value;
        node.typed_id = is_id;
        node.typed_idref = is_idref;
        let slot = self.raw.len();
        self.raw.push(node);
        if let Some(element) = self.raw.get_mut(index) {
            element.attributes.push(slot);
        }
        Ok(())
    }

    /// Adds a parentless attribute node, which a sequence constructor with `as="attribute()"`
    /// produces.
    pub(crate) fn detached_attribute(
        &mut self,
        name: ExpandedName,
        prefix: Option<String>,
        value: impl Into<String>,
    ) {
        let mut node = RawNode::new(NodeKind::Attribute);
        node.name = Some(name);
        node.prefix = prefix;
        node.value = value.into();
        self.push_child(node);
    }

    /// Adds a parentless namespace node, which a sequence constructor with `as="node()*"`
    /// produces from an `xsl:namespace` instruction that no element encloses.
    ///
    /// [XSLT 3.0 §11.7](https://www.w3.org/TR/xslt-30/#creating-namespace-nodes) constructs a
    /// namespace node whatever surrounds the instruction; it is only *attaching* one to the
    /// content of a document node that §5.7.1 forbids. `snapshot-0103` collects one in a
    /// `node()*` variable beside an element, an attribute, a text node, a comment and a
    /// processing instruction, and takes `fn:snapshot` of each.
    pub(crate) fn detached_namespace(&mut self, prefix: Option<String>, uri: &str) {
        let mut node = RawNode::new(NodeKind::Namespace);
        node.name = prefix.map(|prefix| ExpandedName::new(None::<&str>, prefix));
        node.value = uri.to_owned();
        self.push_child(node);
    }

    /// Adds a zero-length text node at the top level of a sequence.
    ///
    /// [XSLT 3.0 §11.4.2](https://www.w3.org/TR/xslt-30/#value-of) makes `xsl:value-of` and
    /// `xsl:text` construct a text node even when the string is empty; such a node is ignored
    /// only when it goes on to form the content of another node.
    pub(crate) fn empty_text(&mut self) {
        self.push_child(RawNode::new(NodeKind::Text));
    }

    /// Declares a namespace on the innermost open element.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when the element already has content, `XTDE0420` when no element is open at
    /// all, `XTDE0430` when the same prefix is already bound here to another namespace, and
    /// `XTDE0440` when a default namespace is declared on an element that is in no namespace.
    pub(crate) fn namespace(&mut self, prefix: Option<String>, uri: &str) -> XsltResult<()> {
        let index = self.attributes_still_allowed("a namespace node")?;
        // XSLT 3.0 §5.7.1: "it is a non-recoverable dynamic error if the result sequence
        // contains a namespace node with no name and the element node being constructed has a
        // null namespace URI" — declaring a default namespace on an element in no namespace
        // would change what the element's own name means (`error-0440a`, `error-0440b`).
        if prefix.is_none()
            && !uri.is_empty()
            && self.raw.get(index).is_some_and(|element| {
                element
                    .name
                    .as_ref()
                    .is_none_or(|name| name.namespace().is_none())
            })
        {
            return Err(XsltError::dynamic(
                "XTDE0440",
                "a default namespace may not be declared on an element that is in no namespace",
            ));
        }
        if let Some(data) = self.raw.get_mut(index) {
            let already = data
                .declarations
                .iter()
                .find(|(existing, _)| existing.as_deref() == prefix.as_deref());
            match already {
                // Two namespace nodes mapping one prefix to two namespaces cannot both be in
                // scope on the element, which XSLT 3.0 §5.7.1 makes `XTDE0430` (`error-0430a`).
                // Declaring the same binding twice is not a conflict and is simply absorbed.
                Some((_, bound)) if bound != uri => {
                    let bound = bound.clone();
                    return Err(XsltError::dynamic(
                        "XTDE0430",
                        format!(
                            "the prefix `{}` is bound to both `{bound}` and `{uri}` on one element",
                            prefix.as_deref().unwrap_or_default()
                        ),
                    ));
                }
                Some(_) => {}
                None => data.declarations.push((prefix, uri.to_owned())),
            }
        }
        Ok(())
    }

    /// Appends character data.
    pub(crate) fn text(&mut self, content: &str, unescaped: bool) {
        if content.is_empty() {
            return;
        }
        let parent = self.current();
        let merge = parent != 0 || self.merge_root_text;
        let last = self
            .raw
            .get(parent)
            .and_then(|data| data.children.last().copied())
            .filter(|child| {
                self.raw
                    .get(*child)
                    .is_some_and(|data| data.kind == NodeKind::Text && data.unescaped == unescaped)
            });
        match last.filter(|_| merge) {
            Some(child) => {
                if let Some(data) = self.raw.get_mut(child) {
                    data.value.push_str(content);
                }
            }
            None => {
                let mut node = RawNode::new(NodeKind::Text);
                node.value = content.to_owned();
                node.unescaped = unescaped;
                self.push_child(node);
            }
        }
    }

    /// Appends a comment.
    pub(crate) fn comment(&mut self, content: &str) {
        let mut node = RawNode::new(NodeKind::Comment);
        node.value = content.to_owned();
        self.push_child(node);
    }

    /// Appends a processing instruction.
    ///
    /// # Errors
    ///
    /// `XTDE0890` when the target is not an `NCName`.
    pub(crate) fn processing_instruction(&mut self, target: &str, content: &str) -> XsltResult<()> {
        if !oxixml_qname::is_ncname(target) {
            return Err(XsltError::dynamic(
                "XTDE0890",
                format!("`{target}` is not a valid processing-instruction target"),
            ));
        }
        let mut node = RawNode::new(NodeKind::ProcessingInstruction);
        node.name = Some(ExpandedName::new(None::<&str>, target));
        node.value = content.to_owned();
        self.push_child(node);
        Ok(())
    }

    /// Deep-copies `node` and everything below it, as `xsl:copy-of` does.
    ///
    /// # Errors
    ///
    /// Whatever the underlying primitive raises — an attribute copied where no element is open,
    /// for instance.
    /// Deep-copies `node`, optionally dropping its namespace nodes.
    ///
    /// # Errors
    ///
    /// Whatever the underlying primitive raises.
    pub(crate) fn copy_of_with(&mut self, node: &NodeRef, copy_namespaces: bool) -> XsltResult<()> {
        match node.node_kind() {
            NodeKind::Document => {
                for child in node.children() {
                    self.copy_of_with(&child, copy_namespaces)?;
                }
            }
            NodeKind::Element => {
                let name = node
                    .node_name()
                    .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
                let prefix = copied_element_prefix(node, &name);
                self.open_element(name, prefix, Inheritance::Copied);
                self.note_copied_base_uri(node.base_uri());
                for namespace in node.namespaces() {
                    if !copy_namespaces {
                        break;
                    }
                    let prefix = namespace.node_name().map(|n| n.local_name().to_owned());
                    if prefix.as_deref() == Some("xml") {
                        continue;
                    }
                    self.namespace(prefix, &namespace.string_value())?;
                }
                for attribute in node.attributes() {
                    let name = attribute
                        .node_name()
                        .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
                    self.attribute(name, None, attribute.string_value())?;
                }
                for child in node.children() {
                    self.copy_of_with(&child, copy_namespaces)?;
                }
                self.end_element()?;
            }
            NodeKind::Attribute => {
                let name = node
                    .node_name()
                    .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
                self.attribute(name, None, node.string_value())?;
            }
            NodeKind::Namespace => {
                let prefix = node.node_name().map(|n| n.local_name().to_owned());
                self.namespace(prefix, &node.string_value())?;
            }
            NodeKind::Text => self.text(&node.string_value(), carries_disabled_escaping(node)),
            NodeKind::Comment => self.comment(&node.string_value()),
            NodeKind::ProcessingInstruction => {
                let target = node
                    .node_name()
                    .map(|n| n.local_name().to_owned())
                    .unwrap_or_default();
                self.processing_instruction(&target, &node.string_value())?;
            }
        }
        Ok(())
    }

    /// Materializes the tree, applying namespace fixup.
    ///
    /// Returns the tree together with the arena index of each child of the document node, in
    /// order, so a caller building a *sequence* can turn them back into items.
    ///
    /// # Errors
    ///
    /// `XTDE0001` when an element is still open.
    pub(crate) fn build(self) -> XsltResult<(ResultTree, Vec<usize>)> {
        if !self.open.is_empty() {
            return Err(XsltError::dynamic(
                "XTDE0001",
                format!("{} element(s) are still open", self.open.len()),
            ));
        }
        let mut out: Vec<ResultNode> = Vec::with_capacity(self.raw.len());
        let mut document = ResultNode::new(NodeKind::Document);
        // The base URI a constructed tree carries is the one the instruction that built it was
        // written under, which is not the URI of any document — a copy of a *source* document
        // sets both, and says so at the call site.
        document.base_uri = self.base_uri.clone();
        out.push(document);

        let in_scope = vec![(Some("xml".to_owned()), XML_NAMESPACE.to_owned())];
        let mut roots = Vec::new();
        let root_children = self.raw.first().map(|data| data.children.clone());
        for child in root_children.unwrap_or_default() {
            // A top-level node of a *sequence* ends up parentless, so a copy among them keeps
            // the base URI it was copied with rather than the one in force where the copy was
            // written — and passes it on to everything it contains, which is what
            // `base-uri-053`'s `…-deeper` assertions read back. A node the document node keeps
            // as a child inherits from that parent instead, which is `base-uri-035`.
            // See [`RawNode::copied_base_uri`].
            //
            // §11.9.2 states the one exception in as many words: an element "having an
            // `xml:base` attribute" takes its base URI from that attribute "resolved if it is
            // relative against the base URI of the `xsl:copy-of` instruction" — against where
            // the copy was written, that is, and not against where the original stood.
            // `base-uri-046` deep-copies an element whose `xml:base` is `/xml/` out of a document
            // based at `http://www.xmlexample.eu/` and wants it read against the stylesheet's own
            // `http://www.example.org/`.
            let base = match self.merge_root_text {
                true => self.base_uri.clone(),
                false => self
                    .raw
                    .get(child)
                    .filter(|raw| self.declared_base_uri(raw).is_none())
                    .and_then(|raw| raw.copied_base_uri.clone())
                    .or_else(|| self.base_uri.clone()),
            };
            roots.push(self.materialize(child, 0, &in_scope, base.as_deref(), &mut out));
        }
        if self.merge_root_text {
            if let Some(document) = out.first_mut() {
                document.children.clone_from(&roots);
            }
        } else {
            // A sequence constructor's top-level nodes are parentless: `/foo` must not match a
            // constructed element merely because the builder holds it.
            for root in &roots {
                if let Some(node) = out.get_mut(*root) {
                    node.parent = None;
                    // [XDM 3.1 §5.2](https://www.w3.org/TR/xpath-datamodel-31/#dm-base-uri)
                    // gives an attribute, text or comment node the base URI *of its parent*, so
                    // one that has no parent has none at all — `base-uri-015` asks for all three
                    // and expects the empty sequence, and `error-1162a` reads a document name out
                    // of such a text node and expects `XTDE1162` for want of anything to resolve
                    // it against. An element keeps its own, which is why §11.1.4 troubles to say
                    // that the rule "is only relevant when creating a parentless element".
                    if !matches!(node.kind, NodeKind::Element | NodeKind::Document) {
                        node.base_uri = None;
                    }
                }
            }
        }
        Ok((
            ResultTree::from_store(ResultStore::new(out, self.document_uri, self.declarations)),
            roots,
        ))
    }

    fn materialize(
        &self,
        index: usize,
        parent: usize,
        in_scope: &[(Option<String>, String)],
        base: Option<&str>,
        out: &mut Vec<ResultNode>,
    ) -> usize {
        let Some(raw) = self.raw.get(index) else {
            return parent;
        };
        let self_index = out.len();
        let mut data = ResultNode::new(raw.kind);
        data.parent = Some(parent);
        data.name.clone_from(&raw.name);
        data.prefix.clone_from(&raw.prefix);
        data.value.clone_from(&raw.value);
        data.unescaped = raw.unescaped;
        data.typed_id = raw.typed_id;
        data.typed_idref = raw.typed_idref;
        data.base_uri = base.map(str::to_owned);

        if raw.kind != NodeKind::Element {
            out.push(data);
            return self_index;
        }

        // XSLT 3.0 §11.1.4 and §11.2.3: an `xml:base` attribute among the element's content
        // gives the element — and everything under it — that base URI, resolved where it is
        // relative against the base URI in force where the element was written. This is the same
        // rule XML Base gives a parsed document, which is what §9.2 asks a temporary tree to
        // behave like.
        let declared_base = self
            .declared_base_uri(raw)
            .map(|declared| crate::resolve::resolve_uri(&declared, base));
        if let Some(declared) = &declared_base {
            data.base_uri = Some(declared.clone());
        }
        let base = declared_base.as_deref().or(base);

        let mut scope = apply_declarations(in_scope, &raw.declarations);
        let element_prefix = fix_element_prefix(raw, &mut scope);
        data.name = data.name.as_ref().map(|name| retag(name, &element_prefix));
        data.prefix = element_prefix;
        let mut attribute_prefixes = Vec::with_capacity(raw.attributes.len());
        for attribute in &raw.attributes {
            let prefix = self
                .raw
                .get(*attribute)
                .map(|attr| fix_attribute_prefix(attr, &mut scope));
            attribute_prefixes.push(prefix.flatten());
        }
        out.push(data);

        let mut namespace_ids = Vec::with_capacity(scope.len());
        for (prefix, uri) in &scope {
            if uri.is_empty() {
                continue;
            }
            // XML Namespaces 1.0 §3 reserves the XML namespace for the `xml` prefix and forbids
            // binding any other prefix to it, so a binding that arrives under another name — as
            // one does when `xsl:namespace-alias` maps a stylesheet prefix onto the XML
            // namespace — is not a namespace node the result tree may carry
            // (`namespace-alias-1905`, `namespace-alias-2613`).
            if uri == XML_NAMESPACE && prefix.as_deref() != Some("xml") {
                continue;
            }
            let namespace_index = out.len();
            let mut node = ResultNode::new(NodeKind::Namespace);
            node.parent = Some(self_index);
            node.name = prefix
                .as_ref()
                .map(|prefix| ExpandedName::new(None::<&str>, prefix.clone()));
            node.value.clone_from(uri);
            out.push(node);
            namespace_ids.push(namespace_index);
        }

        let mut attribute_ids = Vec::with_capacity(raw.attributes.len());
        for (slot, attribute) in raw.attributes.iter().enumerate() {
            let id = self.materialize(*attribute, self_index, &scope, base, out);
            if let Some(node) = out.get_mut(id) {
                let prefix = attribute_prefixes.get(slot).cloned().unwrap_or(None);
                node.name = node.name.as_ref().map(|name| retag(name, &prefix));
                node.prefix = prefix;
            }
            attribute_ids.push(id);
        }

        // XSLT 3.0 §11.7.1: what the elements written inside this one's content start from.
        // `Blocked` and `Copied` withhold the same thing — the bindings this element added — for
        // two different reasons; see [`Inheritance`].
        let child_scope: &[(Option<String>, String)] = match raw.inherit {
            Inheritance::Scope => &scope,
            Inheritance::Blocked | Inheritance::Copied => in_scope,
        };
        let mut child_ids = Vec::with_capacity(raw.children.len());
        for child in &raw.children {
            child_ids.push(self.materialize(*child, self_index, child_scope, base, out));
        }

        if let Some(element) = out.get_mut(self_index) {
            element.namespaces = namespace_ids;
            element.attributes = attribute_ids;
            element.children = child_ids;
        }
        self_index
    }

    /// The `xml:base` attribute `raw` carries, if it has one.
    fn declared_base_uri(&self, raw: &RawNode) -> Option<String> {
        raw.attributes.iter().find_map(|attribute| {
            let attribute = self.raw.get(*attribute)?;
            let name = attribute.name.as_ref()?;
            (name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "base")
                .then(|| attribute.value.clone())
        })
    }
}

/// The prefix a copied element should be spelled with.
///
/// A copy is spelled the way the original was: the prefix is part of an element's name in the
/// data model, so `<Part>` under `xmlns="some1"` stays `Part` even where a second prefix is also
/// bound to `some1`, and `<ns11:f>` stays `ns11:f` (`copy-4901` asserts both). Only when the
/// source name carries no prefix *and* the source does not bind the default namespace to the
/// name's namespace — a name that lost its prefix on the way in — is the element's own namespace
/// node list consulted for a prefix to reuse.
fn copied_element_prefix(node: &NodeRef, name: &ExpandedName) -> Option<String> {
    if let Some(written) = name.prefix().filter(|prefix| !prefix.is_empty()) {
        return Some(written.to_owned());
    }
    let default_binds_it = node.namespaces().into_iter().any(|binding| {
        binding.node_name().is_none() && Some(binding.string_value().as_str()) == name.namespace()
    });
    if default_binds_it || name.namespace().is_none() {
        return None;
    }
    node.namespaces().into_iter().find_map(|binding| {
        (Some(binding.string_value().as_str()) == name.namespace())
            .then(|| binding.node_name().map(|n| n.local_name().to_owned()))
            .flatten()
    })
}

/// Whether a copy of `node` keeps its request to have output escaping disabled.
///
/// `disable-output-escaping` is a property of the text node an instruction constructs, and a copy
/// of that node keeps the property only for as long as the node has not been made part of a
/// *temporary tree*. The distinction is what the test suite settles: a template declared
/// `as="text()"` hands its text node back as a sequence item, and writing that item into the
/// final result still disables escaping (`doe-0180`, `doe-0181`, `doe-0182`), as does an
/// `xsl:try` body returning one directly or inside a constructed element (`doe-0190`,
/// `doe-0191`); but storing the very same node in a variable that builds a tree drops the
/// request, so the characters come out escaped (`doe-0183`, `doe-0185`).
///
/// The two are told apart by what the node's tree hangs from. A temporary tree is rooted in a
/// document node, whereas the top-level items of a sequence are parentless — an element of a
/// sequence roots its own subtree — so a text node whose root is a document node has been
/// through a temporary tree and a text node whose root is anything else has not.
fn carries_disabled_escaping(node: &NodeRef) -> bool {
    let store = node.store();
    if !store.is_unescaped_text(node.id()) {
        return false;
    }
    store.node_kind(store.root(node.id())) != NodeKind::Document
}

fn apply_declarations(
    in_scope: &[(Option<String>, String)],
    declarations: &[(Option<String>, String)],
) -> Vec<(Option<String>, String)> {
    let mut scope: Vec<(Option<String>, String)> = in_scope.to_vec();
    for (prefix, uri) in declarations {
        match scope
            .iter_mut()
            .find(|(existing, _)| existing.as_deref() == prefix.as_deref())
        {
            Some(entry) => entry.1.clone_from(uri),
            None => scope.push((prefix.clone(), uri.clone())),
        }
    }
    scope.retain(|(prefix, uri)| !(uri.is_empty() && prefix.is_some()));
    scope
}

fn lookup<'a>(scope: &'a [(Option<String>, String)], prefix: Option<&str>) -> Option<&'a str> {
    scope
        .iter()
        .find(|(existing, _)| existing.as_deref() == prefix)
        .map(|(_, uri)| uri.as_str())
}

fn bind(scope: &mut Vec<(Option<String>, String)>, prefix: Option<String>, uri: &str) {
    match scope
        .iter_mut()
        .find(|(existing, _)| existing.as_deref() == prefix.as_deref())
    {
        Some(entry) => entry.1 = uri.to_owned(),
        None => scope.push((prefix, uri.to_owned())),
    }
}

/// Re-tags `name` with the prefix namespace fixup settled on.
///
/// [XSLT 3.0 §5.7.3](https://www.w3.org/TR/xslt-30/#namespace-fixup) lets fixup *change* the
/// prefix of an element or attribute name when the one the stylesheet asked for is already
/// bound to another namespace. The changed prefix is part of the name in the data model, so
/// `fn:name` and `fn:node-name` have to report the prefix that was settled on, not the one that
/// was requested: `bug-1501` copies two `x:aaa` attributes from two different namespaces onto
/// one element and asserts that the two names it reads back are distinct.
fn retag(name: &ExpandedName, prefix: &Option<String>) -> ExpandedName {
    let plain = ExpandedName::new(name.namespace(), name.local_name());
    match prefix.as_deref().filter(|prefix| !prefix.is_empty()) {
        Some(prefix) => plain.with_prefix(prefix),
        None => plain,
    }
}

fn invent_prefix(scope: &[(Option<String>, String)]) -> String {
    for candidate in 1..1000_u32 {
        let prefix = format!("ns{candidate}");
        if lookup(scope, Some(&prefix)).is_none() {
            return prefix;
        }
    }
    "ns".to_owned()
}

/// Chooses the prefix an element is serialized with, adding whatever declaration it needs.
fn fix_element_prefix(raw: &RawNode, scope: &mut Vec<(Option<String>, String)>) -> Option<String> {
    let name = raw.name.as_ref()?;
    let Some(uri) = name.namespace() else {
        // A name in no namespace needs the default namespace to be undeclared.
        if lookup(scope, None).is_some_and(|bound| !bound.is_empty()) {
            bind(scope, None, "");
        }
        return None;
    };
    let wanted = raw.prefix.clone();
    if lookup(scope, wanted.as_deref()) == Some(uri) {
        return wanted;
    }
    // Reuse any prefix already bound to this namespace when no particular one was asked for.
    if wanted.is_none()
        && let Some((prefix, _)) = scope
            .iter()
            .find(|(prefix, bound)| bound == uri && prefix.is_some())
    {
        return prefix.clone();
    }
    if wanted.is_some() && lookup(scope, wanted.as_deref()).is_some() {
        let generated = invent_prefix(scope);
        bind(scope, Some(generated.clone()), uri);
        return Some(generated);
    }
    bind(scope, wanted.clone(), uri);
    wanted
}

/// Chooses the prefix an attribute is serialized with; an attribute in no namespace never
/// takes the default namespace.
fn fix_attribute_prefix(
    raw: &RawNode,
    scope: &mut Vec<(Option<String>, String)>,
) -> Option<String> {
    let name = raw.name.as_ref()?;
    let uri = name.namespace()?;
    if uri == XML_NAMESPACE {
        return Some("xml".to_owned());
    }
    // A copied attribute carries no separate prefix hint — `copy_of_with` has only the source
    // node's name to go on — but that name retains the prefix it was written with, and reusing
    // it keeps a copy spelled the way the original was instead of inventing `ns1`.
    let wanted = raw
        .prefix
        .clone()
        .or_else(|| name.prefix().map(str::to_owned))
        .filter(|prefix| !prefix.is_empty());
    if let Some(prefix) = &wanted
        && lookup(scope, Some(prefix)) == Some(uri)
    {
        return wanted;
    }
    if let Some((prefix, _)) = scope
        .iter()
        .find(|(prefix, bound)| bound == uri && prefix.is_some())
    {
        return prefix.clone();
    }
    let prefix = match wanted {
        Some(prefix) if lookup(scope, Some(&prefix)).is_none() => prefix,
        _ => invent_prefix(scope),
    };
    bind(scope, Some(prefix.clone()), uri);
    Some(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::XdmNode;

    #[test]
    fn a_simple_tree_materializes_in_document_order() -> XsltResult<()> {
        let mut builder = ResultBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "out"), None);
        builder.attribute(ExpandedName::new(None::<&str>, "n"), None, "1")?;
        builder.text("hello", false);
        builder.end_element()?;
        let (tree, roots) = builder.build()?;
        assert_eq!(roots.len(), 1);

        let root = tree.root();
        let out = root.children().first().cloned().expect("one element");
        assert_eq!(out.node_kind(), NodeKind::Element);
        assert_eq!(out.string_value(), "hello");
        assert_eq!(out.attributes().len(), 1);
        Ok(())
    }

    #[test]
    fn namespace_fixup_invents_a_prefix_on_a_collision() -> XsltResult<()> {
        let mut builder = ResultBuilder::new();
        builder.start_element(ExpandedName::new(Some("urn:a"), "e"), Some("p".to_owned()));
        builder.namespace(Some("p".to_owned()), "urn:other")?;
        builder.end_element()?;
        let (tree, _) = builder.build()?;
        let element = tree
            .root()
            .children()
            .first()
            .cloned()
            .expect("one element");
        let bindings: Vec<(Option<String>, String)> = element
            .namespaces()
            .iter()
            .map(|node| {
                (
                    node.node_name().map(|n| n.local_name().to_owned()),
                    node.string_value(),
                )
            })
            .collect();
        assert!(bindings.contains(&(Some("ns1".to_owned()), "urn:a".to_owned())));
        assert!(bindings.contains(&(Some("p".to_owned()), "urn:other".to_owned())));
        Ok(())
    }

    #[test]
    fn adjacent_text_merges_inside_an_element() -> XsltResult<()> {
        let mut builder = ResultBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "e"), None);
        builder.text("a", false);
        builder.text("b", false);
        builder.end_element()?;
        let (tree, _) = builder.build()?;
        let element = tree.root().children().first().cloned().expect("element");
        assert_eq!(element.children().len(), 1);
        assert_eq!(element.string_value(), "ab");
        Ok(())
    }

    #[test]
    fn setting_an_attribute_twice_replaces_it() -> XsltResult<()> {
        let mut builder = ResultBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "e"), None);
        builder.attribute(ExpandedName::new(None::<&str>, "a"), None, "1")?;
        builder.attribute(ExpandedName::new(None::<&str>, "a"), None, "2")?;
        builder.end_element()?;
        let (tree, _) = builder.build()?;
        let element = tree.root().children().first().cloned().expect("element");
        assert_eq!(element.attributes().len(), 1);
        assert_eq!(element.attributes()[0].string_value(), "2");
        Ok(())
    }
}
