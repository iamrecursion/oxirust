//! Name- and identifier-based lookups.

use crate::document::Document;
use crate::iter::{ElementFilter, Elements};
use crate::names::XML_NAMESPACE;
use crate::node::{NodeId, NodeKind};

impl Document {
    /// Returns the elements of the subtree rooted at `from`, in document order,
    /// whose qualified name is `name`. The special name `"*"` matches every
    /// element.
    ///
    /// The subtree root itself is *not* returned, matching DOM
    /// `Element.getElementsByTagName`; pass [`Document::root`] to search the
    /// whole document.
    ///
    /// The returned iterator borrows the document, so it always reports the tree
    /// as it currently is.
    #[must_use]
    pub fn get_elements_by_tag_name<'a>(&'a self, from: NodeId, name: &'a str) -> Elements<'a> {
        Elements::new(self.descendants(from), ElementFilter::QualifiedName(name))
    }

    /// Returns the elements of the subtree rooted at `from`, in document order,
    /// with the given expanded name.
    ///
    /// `Some("*")` as the namespace and `"*"` as the local name are wildcards;
    /// `None` as the namespace matches elements in no namespace.
    #[must_use]
    pub fn get_elements_by_tag_name_ns<'a>(
        &'a self,
        from: NodeId,
        namespace: Option<&'a str>,
        local_name: &'a str,
    ) -> Elements<'a> {
        Elements::new(
            self.descendants(from),
            ElementFilter::ExpandedName {
                namespace,
                local: local_name,
            },
        )
    }

    /// Returns the first element of the document, in document order, that has an
    /// `ID`-typed attribute whose value is `id`.
    ///
    /// An attribute counts as `ID`-typed when any of the following holds:
    ///
    /// * it was marked with [`Document::set_id_attribute_node`];
    /// * its expanded name is `xml:id`, which
    ///   [xml:id 1.0](https://www.w3.org/TR/xml-id/) defines as always being of
    ///   type `ID`;
    /// * an `<!ATTLIST … ID …>` declaration was registered for it with
    ///   [`Document::declare_id_attribute`] — the builder does that
    ///   automatically for declarations found in the internal DTD subset.
    #[must_use]
    pub fn get_element_by_id(&self, id: &str) -> Option<NodeId> {
        self.descendants(self.root())
            .filter(|&node| self.kind(node) == Some(NodeKind::Element))
            .find(|&node| {
                self.attributes(node).any(|attribute| {
                    self.attr(attribute).is_some_and(|data| {
                        data.value() == id && self.is_id_attribute(node, attribute)
                    })
                })
            })
    }

    /// Returns `true` when the attribute node is of type `ID` for its element.
    ///
    /// See [`Self::get_element_by_id`] for the three ways an attribute becomes
    /// `ID`-typed.
    #[must_use]
    pub fn is_id_attribute(&self, element: NodeId, attribute: NodeId) -> bool {
        let Some(data) = self.attr(attribute) else {
            return false;
        };
        if data.is_id() {
            return true;
        }
        if data.namespace_uri() == Some(XML_NAMESPACE) && data.local_name() == "id" {
            return true;
        }
        let Some(element_data) = self.element(element) else {
            return false;
        };
        self.is_declared_id_attribute(&element_data.qualified_name(), &data.qualified_name())
    }
}
