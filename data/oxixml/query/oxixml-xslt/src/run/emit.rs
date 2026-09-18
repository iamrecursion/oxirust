//! [`Emitter`] — where a sequence constructor's output goes.
//!
//! XSLT 2.0 made the result of a sequence constructor a *sequence*, not a tree: `xsl:sequence`
//! can hand back the very nodes a path expression selected, and a variable with an `as`
//! attribute keeps atomic values atomic. A variable *without* `as`, and the result of the
//! transformation itself, are still trees, and there the conversion rules of
//! [XSLT 3.0 §5.7.1](https://www.w3.org/TR/xslt-30/#constructing-complex-content) apply:
//! adjacent atomic values are separated by one space, and every node is copied.
//!
//! One emitter serves both: it always owns a [`ResultBuilder`], and in *sequence mode* it also
//! records what each top-level item was, so the tree and the sequence can both come out at the
//! end.

use crate::error::{XsltError, XsltResult};
use crate::tree::{ResultBuilder, ResultTree};
use oxixml_infoset::{AtomicValue, ExpandedName, Item, NodeRef, XdmNode};

/// One item produced at the top level of a sequence constructor.
#[derive(Debug)]
enum Entry {
    /// An atomic value, kept as such.
    Atomic(AtomicValue),
    /// A node the constructor returned rather than built.
    Existing(NodeRef),
    /// A node the constructor built: the ordinal of the document node's child.
    Constructed(usize),
    /// A map, an array or another function item — kept as-is, since none of them corresponds
    /// to anything the tree-shaped [`ResultBuilder`] can represent.
    Function(Item),
}

/// Collects the output of a sequence constructor.
#[derive(Debug)]
pub(crate) struct Emitter {
    builder: ResultBuilder,
    entries: Vec<Entry>,
    sequence_mode: bool,
    /// Whether this emitter collects a *final result tree* — the principal result, or the body
    /// of an `xsl:result-document` with an `href` — which is on its way to the serializer.
    final_result: bool,
    /// Whether the previous thing written at the current level was an atomic value, which
    /// decides whether the next one is preceded by a space.
    previous_was_atomic: Vec<bool>,
    /// The `item-separator` serialization parameter, when the destination declares one, with a
    /// count of the top-level items written so far.
    ///
    /// [Serialization 3.1 §2](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam)
    /// makes the parameter part of *sequence normalization*: with it declared, every adjacent
    /// pair of items in the sequence being serialized is separated by it, not only the adjacent
    /// atomic values that §5.7.1 would otherwise join with a space.
    item_separator: Option<(String, usize)>,
}

impl Emitter {
    /// An emitter whose result is a tree: every item is copied in, atomics become text.
    pub(crate) fn tree() -> Self {
        Self {
            builder: ResultBuilder::new(),
            entries: Vec::new(),
            sequence_mode: false,
            final_result: false,
            previous_was_atomic: vec![false],
            item_separator: None,
        }
    }

    /// An emitter for a *final result tree*: as [`Emitter::tree`], except that an item the
    /// serializer could not write is reported as the serialization error `SENR0001` rather
    /// than as the tree-construction error `XTDE0450` — see [`Emitter::item`].
    pub(crate) fn final_result() -> Self {
        Self {
            final_result: true,
            ..Self::tree()
        }
    }

    /// An emitter whose result is a sequence.
    pub(crate) fn sequence() -> Self {
        Self {
            builder: ResultBuilder::for_sequence(),
            entries: Vec::new(),
            sequence_mode: true,
            final_result: false,
            previous_was_atomic: vec![false],
            item_separator: None,
        }
    }

    /// Declares the `item-separator` the destination serializes under.
    pub(crate) fn set_item_separator(&mut self, separator: Option<&str>) {
        self.item_separator = separator.map(|text| (text.to_owned(), 0));
    }

    /// Writes the `item-separator` before every top-level item but the first, and counts the
    /// item that is about to be written.
    ///
    /// A no-op unless the destination declared one, which is what keeps the ordinary path — and
    /// the single space [XSLT 3.0 §5.7.1](https://www.w3.org/TR/xslt-30/#constructing-complex-content)
    /// puts between adjacent atomic values — exactly as it was.
    fn separate_item(&mut self) {
        if self.nested() {
            return;
        }
        let Some((separator, written)) = &mut self.item_separator else {
            return;
        };
        let separator = (*written > 0).then(|| separator.clone());
        *written += 1;
        if let Some(separator) = separator {
            self.builder.text(&separator, false);
        }
    }

    /// Whether a separator has been declared, which suppresses the single space adjacent atomic
    /// values are otherwise joined with.
    fn separates_items(&self) -> bool {
        self.item_separator.is_some()
    }

    /// Sets the document URI of the tree being built.
    pub(crate) fn set_document_uri(&mut self, uri: Option<String>) {
        self.builder.set_document_uri(uri);
    }

    /// Sets the base URI the tree being built is constructed under — see
    /// [`ResultBuilder::set_base_uri`].
    pub(crate) fn set_base_uri(&mut self, uri: Option<String>) {
        self.builder.set_base_uri(uri);
    }

    /// Carries onto the document node being built the properties a *copy* of `source` keeps.
    ///
    /// [XSLT 3.0 §11.9.1](https://www.w3.org/TR/xslt-30/#element-copy) copies the base URI and
    /// the document URI of a document node onto the copy, and [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)
    /// makes `[unparsed entities]` and `[notations]` properties of that node too — so
    /// `fn:unparsed-entity-uri` has to answer the same for the copy as for the original.
    pub(crate) fn carry_document_properties(&mut self, source: &NodeRef) {
        self.builder.set_document_uri(source.document_uri());
        self.builder.set_base_uri(source.base_uri());
        self.builder.carry_declarations(source);
    }

    /// Whether output is currently going inside a constructed element.
    fn nested(&self) -> bool {
        self.builder.in_element()
    }

    /// Whether what is written next becomes a top-level item of a sequence, where a document
    /// node stays a document node instead of being replaced by its children.
    pub(crate) fn wants_document_item(&self) -> bool {
        self.sequence_mode && !self.nested()
    }

    fn note_non_atomic(&mut self) {
        if let Some(last) = self.previous_was_atomic.last_mut() {
            *last = false;
        }
    }

    /// Opens a constructed element whose content inherits its namespaces.
    pub(crate) fn start_element(&mut self, name: ExpandedName, prefix: Option<String>) {
        self.start_element_with(name, prefix, true);
    }

    /// Records, on the element that is open, the base URI of the node it is a *shallow copy* of.
    ///
    /// The deep-copy path sets it itself, node by node; a shallow copy is opened with
    /// [`Emitter::start_element`] or [`Emitter::start_element_with`] and fills its own content,
    /// so the instruction that opened it says what it copied. See
    /// [`crate::tree::ResultBuilder::note_copied_base_uri`].
    pub(crate) fn note_copied_base_uri(&mut self, uri: Option<String>) {
        self.builder.note_copied_base_uri(uri);
    }

    /// Opens a constructed element, saying whether the elements built inside its content
    /// inherit its namespaces — `inherit-namespaces`
    /// ([XSLT 3.0 §11.7.1](https://www.w3.org/TR/xslt-30/#copy-namespaces)).
    pub(crate) fn start_element_with(
        &mut self,
        name: ExpandedName,
        prefix: Option<String>,
        inherit: bool,
    ) {
        self.separate_item();
        if !self.nested() {
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.note_non_atomic();
        self.builder.start_element_with(name, prefix, inherit);
        self.previous_was_atomic.push(false);
    }

    /// Closes the innermost constructed element.
    ///
    /// # Errors
    ///
    /// `XTDE0001` when no element is open.
    pub(crate) fn end_element(&mut self) -> XsltResult<()> {
        self.previous_was_atomic.pop();
        self.builder.end_element()
    }

    /// Adds an attribute to the innermost open element.
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
        if !self.nested() && self.sequence_mode {
            // A sequence constructor may build a parentless attribute node, which is what a
            // variable declared `as="attribute()"` holds.
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
            self.builder.detached_attribute(name, prefix, value);
            return Ok(());
        }
        self.builder.attribute(name, prefix, value)
    }

    /// Declares a namespace on the innermost open element.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when no element is open or the element already has content.
    pub(crate) fn namespace(&mut self, prefix: Option<String>, uri: &str) -> XsltResult<()> {
        if !self.nested() && self.sequence_mode {
            // A sequence constructor may build a parentless namespace node, exactly as it may
            // build a parentless attribute node above: what §5.7.1 forbids is putting one in the
            // *content of a document node*, not constructing one.
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
            self.builder.detached_namespace(prefix, uri);
            return Ok(());
        }
        self.builder.namespace(prefix, uri)
    }

    /// Writes the text node an instruction constructs.
    ///
    /// Every instruction that reaches here builds a text node rather than merely contributing
    /// characters: `xsl:text` (with or without an expanded body), `xsl:value-of`, `xsl:number`,
    /// and the text-only-copy built-in template rule.
    /// [XSLT 3.0 §11.4.2](https://www.w3.org/TR/xslt-30/#value-of) makes that true even when
    /// the string is zero-length, so such a node is a real item of a sequence — `seqtor-041`
    /// binds three of them to an `as="text()*"` variable and asserts `count()` is 3.
    ///
    /// Inside constructed content the zero-length node is discarded instead, but not before it
    /// has *parted* the atomic values on either side of it. The steps of
    /// [XSLT 3.0 §5.7.1](https://www.w3.org/TR/xslt-30/#constructing-complex-content) run in
    /// order: adjacent atomic values are joined with a single space *first*, and only then are
    /// zero-length text nodes discarded and the surviving text nodes merged. So the two values
    /// in `<xsl:sequence select="'a'"/><xsl:text/><xsl:sequence select="'b'"/>` are not
    /// adjacent, and the content they build is `ab`, not `a b`.
    pub(crate) fn text(&mut self, content: &str, unescaped: bool) {
        self.note_non_atomic();
        if content.is_empty() {
            if self.sequence_mode && !self.nested() {
                self.entries
                    .push(Entry::Constructed(self.builder.root_child_count()));
                self.builder.empty_text();
            }
            return;
        }
        self.separate_item();
        if !self.nested() {
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.builder.text(content, unescaped);
    }

    /// [`Emitter::text`] under the name an instruction that thinks in nodes rather than in
    /// characters calls it by.
    pub(crate) fn text_node(&mut self, content: &str, unescaped: bool) {
        self.text(content, unescaped);
    }

    /// Writes a comment.
    pub(crate) fn comment(&mut self, content: &str) {
        self.separate_item();
        if !self.nested() {
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.note_non_atomic();
        self.builder.comment(content);
    }

    /// Writes a processing instruction.
    ///
    /// # Errors
    ///
    /// `XTDE0890` when the target is not an `NCName`.
    pub(crate) fn processing_instruction(&mut self, target: &str, content: &str) -> XsltResult<()> {
        self.separate_item();
        if !self.nested() {
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.note_non_atomic();
        self.builder.processing_instruction(target, content)
    }

    /// Appends an atomic value: kept atomic at the top level of a sequence, and turned into
    /// text — space-separated from its predecessor — anywhere else.
    pub(crate) fn atomic(&mut self, value: AtomicValue) -> XsltResult<()> {
        if self.sequence_mode && !self.nested() {
            self.entries.push(Entry::Atomic(value));
            return Ok(());
        }
        // `item-separator` belongs to *sequence normalization*, which acts on the top level of
        // the sequence being serialized; inside constructed element content the single space of
        // XSLT 3.0 §5.7.1 still parts adjacent atomic values, whatever the destination declares.
        let separated = (self.nested() || !self.separates_items())
            && self.previous_was_atomic.last().copied().unwrap_or(false);
        let text = value.string_value();
        if separated {
            self.builder.text(" ", false);
        }
        self.separate_item();
        if !self.nested() {
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.builder.text(&text, false);
        if let Some(last) = self.previous_was_atomic.last_mut() {
            *last = true;
        }
        Ok(())
    }

    /// Appends a node the constructor *returned* — `xsl:sequence` — without copying it when
    /// the result is a sequence.
    ///
    /// # Errors
    ///
    /// Whatever copying raises when the node has to be copied.
    pub(crate) fn append_node(&mut self, node: &NodeRef) -> XsltResult<()> {
        if self.sequence_mode && !self.nested() {
            self.entries.push(Entry::Existing(node.clone()));
            return Ok(());
        }
        self.copy_node(node)
    }

    /// Deep-copies a node — `xsl:copy-of`.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when an attribute is copied where no element is open.
    pub(crate) fn copy_node(&mut self, node: &NodeRef) -> XsltResult<()> {
        self.copy_node_with(node, true)
    }

    /// Deep-copies a node, optionally dropping the namespace nodes — the
    /// `copy-namespaces="no"` case of `xsl:copy-of`.
    ///
    /// # Errors
    ///
    /// `XTDE0410` when an attribute is copied where no element is open.
    pub(crate) fn copy_node_with(
        &mut self,
        node: &NodeRef,
        copy_namespaces: bool,
    ) -> XsltResult<()> {
        use oxixml_infoset::NodeKind;
        // A parentless attribute is a legitimate item of a sequence — `as="attribute()"` — so
        // copying one at the top level of a sequence constructor builds a detached copy
        // rather than raising `XTDE0410`.
        // A copy of a document node is a document node, which only survives as such at the
        // top level of a sequence: inside constructed content it is replaced by its children.
        if self.wants_document_item() && node.node_kind() == NodeKind::Document {
            let mut builder = ResultBuilder::new();
            // A copy of a document node is a document node with the same properties, the DTD's
            // unparsed entities and notations among them — the suite's unparsed-entity-05 and
            // -07 read them back out of the copy.
            builder.set_document_uri(node.document_uri());
            builder.set_base_uri(node.base_uri());
            builder.carry_declarations(node);
            for child in node.children() {
                builder.copy_of_with(&child, copy_namespaces)?;
            }
            let (tree, _) = builder.build()?;
            self.entries.push(Entry::Existing(tree.root()));
            return Ok(());
        }
        if self.sequence_mode && !self.nested() {
            // A copy taken at the top level of a *sequence* keeps the kind it had, parentless:
            // an attribute or namespace node there is an item of the sequence rather than
            // something being attached to a node under construction (`snapshot-0103` copies both
            // out of a `node()*` variable).
            match node.node_kind() {
                NodeKind::Attribute => {
                    let name = node
                        .node_name()
                        .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
                    return self.attribute(name, None, node.string_value());
                }
                NodeKind::Namespace => {
                    let prefix = node
                        .node_name()
                        .map(|name| name.local_name().to_owned())
                        .filter(|prefix| !prefix.is_empty());
                    return self.namespace(prefix, &node.string_value());
                }
                _ => {}
            }
        }
        if !self.nested() && !matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace)
        {
            self.separate_item();
            self.entries
                .push(Entry::Constructed(self.builder.root_child_count()));
        }
        self.note_non_atomic();
        self.builder.copy_of_with(node, copy_namespaces)
    }

    /// Appends an item of any kind.
    ///
    /// A map, an array or another function item is a legitimate top-level item of a
    /// *sequence* — `xsl:sequence select="map{...}"` inside a variable declared `as="map(*)"`,
    /// say — since [XSLT 3.0 §5.7.1](https://www.w3.org/TR/xslt-30/#constructing-complex-content)
    /// restricts *complex content construction* (an actual tree — a constructed element, the
    /// principal result, an `xsl:copy`…) to nodes and atomic values; it never restricts what a
    /// sequence constructor not building a tree may return.
    ///
    /// A function item that reaches the *top level of a final result* is a different fault
    /// again. [XSLT 3.0 §2.3.6](https://www.w3.org/TR/xslt-30/#result-trees) keeps result-tree
    /// construction and serialization apart — "the raw result may be serialized", and the
    /// decision to build a tree first "is determined by the rules of the transformation API …
    /// and is not influenced by anything in the stylesheet" — so an item that never enters
    /// constructed complex content but goes straight to the serializer fails under
    /// serialization's own code, `SENR0001`, not `XTDE0450`. That is what the suite's
    /// output-0710/0711/0712 ask for: `<xsl:sequence select="$maps"/>` as the whole body of the
    /// initial template, with `<output serialize="yes"/>` on the case.
    ///
    /// # Errors
    ///
    /// `SENR0001` for a map, an array or a function item at the top level of a final result
    /// tree; `XTDE0450` for one anywhere else that is not the top level of a sequence — nested
    /// inside constructed element content, say, or a temporary tree (`Emitter::tree`).
    /// Otherwise, whatever the underlying primitive raises.
    pub(crate) fn item(&mut self, item: &Item) -> XsltResult<()> {
        match item {
            Item::Node(node) => self.append_node(node),
            Item::Atomic(value) => self.atomic(value.clone()),
            Item::Map(_) | Item::Array(_) | Item::Function(_) => {
                if self.sequence_mode && !self.nested() {
                    self.entries.push(Entry::Function(item.clone()));
                    return Ok(());
                }
                // An *array* is the one function item complex content admits: §5.7.1 replaces
                // it with its members before it looks for an invalid item, so
                // `<out><xsl:sequence select="[1, 2, 3]"/></out>` is the text `1 2 3` rather
                // than a type error (the suite's arrays-304/-305, which pin the same answer for
                // an array of arrays). A map is not — `maps-006` and `lre-025` both make one
                // `XTDE0450` — and neither is any other function item (`lre-026`).
                if let Item::Array(array) = item {
                    for member in array.members() {
                        for member_item in member.iter() {
                            self.item(member_item)?;
                        }
                    }
                    return Ok(());
                }
                if self.final_result && !self.nested() {
                    return Err(XsltError::dynamic(
                        "SENR0001",
                        "a function item cannot be serialized",
                    ));
                }
                Err(XsltError::dynamic(
                    "XTDE0450",
                    "a function item cannot be written to a result tree",
                ))
            }
        }
    }

    /// Whether nothing at all has been written.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.builder.root_child_count() == 0
    }

    /// Materializes the result.
    ///
    /// # Errors
    ///
    /// `XTDE0001` when an element is still open.
    pub(crate) fn finish(self) -> XsltResult<(ResultTree, Vec<Item>)> {
        let sequence_mode = self.sequence_mode;
        let entries = self.entries;
        let (tree, roots) = self.builder.build()?;
        if !sequence_mode {
            return Ok((tree, Vec::new()));
        }
        let mut items = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry {
                Entry::Atomic(value) => items.push(Item::Atomic(value)),
                Entry::Existing(node) => items.push(Item::Node(node)),
                Entry::Constructed(ordinal) => {
                    if let Some(node) = roots.get(ordinal).and_then(|index| tree.node_at(*index)) {
                        items.push(Item::Node(node));
                    }
                }
                Entry::Function(item) => items.push(item),
            }
        }
        Ok((tree, items))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{Sequence, Tree};

    #[test]
    fn a_tree_emitter_turns_atomics_into_separated_text() -> XsltResult<()> {
        let mut emitter = Emitter::tree();
        emitter.start_element(ExpandedName::new(None::<&str>, "e"), None);
        emitter.atomic(AtomicValue::integer(1_i64))?;
        emitter.atomic(AtomicValue::integer(2_i64))?;
        emitter.end_element()?;
        let (tree, items) = emitter.finish()?;
        assert!(items.is_empty());
        let element = tree.root().children().first().cloned().expect("element");
        assert_eq!(element.string_value(), "1 2");
        Ok(())
    }

    #[test]
    fn a_sequence_emitter_keeps_items_apart() -> XsltResult<()> {
        let source = Tree::parse_str("<a/>")?;
        let mut emitter = Emitter::sequence();
        emitter.atomic(AtomicValue::integer(7_i64))?;
        emitter.append_node(&source.document_element().expect("an element"))?;
        emitter.start_element(ExpandedName::new(None::<&str>, "built"), None);
        emitter.end_element()?;
        let (_, items) = emitter.finish()?;
        assert_eq!(items.len(), 3);
        assert!(items[0].is_atomic());
        assert!(items[1].is_node());
        assert_eq!(Sequence::from_items(items).len(), 3);
        Ok(())
    }

    #[test]
    fn emptiness_is_observable() -> XsltResult<()> {
        let mut emitter = Emitter::tree();
        assert!(emitter.is_empty());
        emitter.text("x", false);
        assert!(!emitter.is_empty());
        Ok(())
    }

    /// `seqtor-023`: a zero-length text node between two atomic values keeps them apart, so no
    /// separator is inserted and the surviving text merges to `--`.
    #[test]
    fn a_zero_length_text_node_parts_adjacent_atomics() -> XsltResult<()> {
        let mut emitter = Emitter::tree();
        emitter.atomic(AtomicValue::string("-"))?;
        emitter.text_node("", false);
        emitter.atomic(AtomicValue::string("-"))?;
        let (tree, _) = emitter.finish()?;
        assert_eq!(tree.root().string_value(), "--");
        Ok(())
    }

    /// The negative control: without the intervening text node the two values *are* adjacent
    /// and take the single-space separator §5.7.1 prescribes.
    #[test]
    fn adjacent_atomics_are_still_separated_by_one_space() -> XsltResult<()> {
        let mut emitter = Emitter::tree();
        emitter.atomic(AtomicValue::string("-"))?;
        emitter.atomic(AtomicValue::string("-"))?;
        let (tree, _) = emitter.finish()?;
        assert_eq!(tree.root().string_value(), "- -");
        Ok(())
    }

    /// `seqtor-011`: the parting works inside a constructed element too, where the emitter
    /// tracks the atomic run per open element.
    #[test]
    fn a_zero_length_text_node_parts_atomics_inside_an_element() -> XsltResult<()> {
        let mut emitter = Emitter::tree();
        emitter.start_element(ExpandedName::new(None::<&str>, "e"), None);
        emitter.atomic(AtomicValue::integer(1_i64))?;
        emitter.text_node("", false);
        emitter.atomic(AtomicValue::integer(2_i64))?;
        emitter.end_element()?;
        let (tree, _) = emitter.finish()?;
        let element = tree.root().children().first().cloned().expect("element");
        assert_eq!(element.string_value(), "12");
        Ok(())
    }

    /// `seqtor-041`: three zero-length `xsl:text` results are three items of a sequence, not
    /// one — the emitter must not coalesce them.
    #[test]
    fn three_zero_length_text_nodes_are_three_sequence_items() -> XsltResult<()> {
        let mut emitter = Emitter::sequence();
        emitter.text_node("", false);
        emitter.text_node("", false);
        emitter.text_node("", false);
        let (_, items) = emitter.finish()?;
        assert_eq!(items.len(), 3);
        Ok(())
    }
}
