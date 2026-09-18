//! Attribute handling and namespace lookup.
//!
//! Namespace declarations are ordinary attribute nodes in the
//! `http://www.w3.org/2000/xmlns/` namespace, exactly as
//! [DOM Level 2 Core §1.1.8](https://www.w3.org/TR/DOM-Level-2-Core/core.html#Namespaces-Considerations)
//! prescribes: `xmlns:p="…"` is an attribute with the prefix `xmlns` and the
//! local name `p`, and `xmlns="…"` is an unprefixed attribute with the local name
//! `xmlns`. Keeping them in the attribute list means nothing has to be kept in
//! sync, and a document round-trips its declarations verbatim.

use crate::document::Document;
use crate::error::{DomError, DomResult, NotFound};
use crate::names::{XMLNS_NAMESPACE, check_attribute_ns, check_chars, split_qualified_name};
use crate::node::{NodeContent, NodeId, NodeKind};

impl Document {
    /// Returns the attribute node of `element` whose qualified name is `name`.
    #[must_use]
    pub fn get_attribute_node(&self, element: NodeId, name: &str) -> Option<NodeId> {
        self.attributes(element).find(|&id| {
            self.attr(id)
                .is_some_and(|data| data.qualified_name() == name)
        })
    }

    /// Returns the attribute node of `element` with the given expanded name.
    ///
    /// Pass `None` for `namespace` to look up an attribute in no namespace.
    #[must_use]
    pub fn get_attribute_node_ns(
        &self,
        element: NodeId,
        namespace: Option<&str>,
        local_name: &str,
    ) -> Option<NodeId> {
        let namespace = namespace.filter(|ns| !ns.is_empty());
        self.attributes(element).find(|&id| {
            self.attr(id).is_some_and(|data| {
                data.local_name() == local_name && data.namespace_uri() == namespace
            })
        })
    }

    /// Returns the value of the attribute of `element` whose qualified name is
    /// `name`.
    #[must_use]
    pub fn get_attribute(&self, element: NodeId, name: &str) -> Option<&str> {
        let id = self.get_attribute_node(element, name)?;
        self.attr(id).map(|data| data.value.as_str())
    }

    /// Returns the value of the attribute of `element` with the given expanded
    /// name.
    #[must_use]
    pub fn get_attribute_ns(
        &self,
        element: NodeId,
        namespace: Option<&str>,
        local_name: &str,
    ) -> Option<&str> {
        let id = self.get_attribute_node_ns(element, namespace, local_name)?;
        self.attr(id).map(|data| data.value.as_str())
    }

    /// Returns `true` when `element` carries an attribute with that qualified
    /// name.
    #[must_use]
    pub fn has_attribute(&self, element: NodeId, name: &str) -> bool {
        self.get_attribute_node(element, name).is_some()
    }

    /// Returns `true` when `element` carries an attribute with that expanded
    /// name.
    #[must_use]
    pub fn has_attribute_ns(
        &self,
        element: NodeId,
        namespace: Option<&str>,
        local_name: &str,
    ) -> bool {
        self.get_attribute_node_ns(element, namespace, local_name)
            .is_some()
    }

    /// Sets the attribute of `element` whose qualified name is `name`, creating
    /// the attribute node when it does not exist yet.
    ///
    /// Returns the attribute node.
    ///
    /// # Errors
    ///
    /// * [`DomError::NotFound`] — `element` is unknown.
    /// * [`DomError::NotSupported`] — `element` is not an element.
    /// * [`DomError::InvalidCharacter`] — `name` is not an XML `Name`, or the
    ///   value contains a character outside the `Char` production.
    pub fn set_attribute(
        &mut self,
        element: NodeId,
        name: &str,
        value: impl Into<String>,
    ) -> DomResult<NodeId> {
        let value = value.into();
        check_chars(&value, self.char_version())?;
        self.require_element(element)?;
        if let Some(existing) = self.get_attribute_node(element, name) {
            if let NodeContent::Attr(data) = &mut self.node_mut(existing)?.content {
                data.value = value;
                data.specified = true;
            }
            return Ok(existing);
        }
        let attribute = self.create_attribute(name)?;
        if let NodeContent::Attr(data) = &mut self.node_mut(attribute)?.content {
            data.value = value;
        }
        self.attach_attribute(element, attribute);
        Ok(attribute)
    }

    /// Sets the namespace-qualified attribute of `element`, creating the
    /// attribute node when no attribute with that expanded name exists yet.
    ///
    /// When the attribute already exists its prefix is updated to the one in
    /// `qualified_name`, as DOM `Element.setAttributeNS` requires.
    ///
    /// # Errors
    ///
    /// The union of the errors of [`Self::set_attribute`] and
    /// [`Self::create_attribute_ns`].
    pub fn set_attribute_ns(
        &mut self,
        element: NodeId,
        namespace: Option<&str>,
        qualified_name: &str,
        value: impl Into<String>,
    ) -> DomResult<NodeId> {
        let value = value.into();
        check_chars(&value, self.char_version())?;
        self.require_element(element)?;

        // Validate the qualified name and resolve the expanded name *before*
        // allocating anything: creating the attribute node first would leak one
        // arena slot per call on the (very common) update-in-place path, because
        // nodes are never reclaimed.
        let (prefix, local) = split_qualified_name(qualified_name)?;
        check_attribute_ns(namespace, prefix, local)?;
        let lookup_namespace = namespace.filter(|ns| !ns.is_empty());

        if let Some(existing) = self.get_attribute_node_ns(element, lookup_namespace, local) {
            let prefix = prefix.map(str::to_owned);
            if let NodeContent::Attr(data) = &mut self.node_mut(existing)?.content {
                data.prefix = prefix;
                data.value = value;
                data.specified = true;
            }
            return Ok(existing);
        }

        let attribute = self.create_attribute_ns(namespace, qualified_name)?;
        if let NodeContent::Attr(data) = &mut self.node_mut(attribute)?.content {
            data.value = value;
        }
        self.attach_attribute(element, attribute);
        Ok(attribute)
    }

    /// Removes the attribute of `element` whose qualified name is `name`.
    ///
    /// Returns the detached attribute node, or `None` when the element had no
    /// such attribute.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when `element` is unknown and
    /// [`DomError::NotSupported`] when it is not an element.
    pub fn remove_attribute(&mut self, element: NodeId, name: &str) -> DomResult<Option<NodeId>> {
        self.require_element(element)?;
        match self.get_attribute_node(element, name) {
            Some(attribute) => {
                self.remove_attribute_node(element, attribute)?;
                Ok(Some(attribute))
            }
            None => Ok(None),
        }
    }

    /// Removes the attribute of `element` with the given expanded name.
    ///
    /// Returns the detached attribute node, or `None` when the element had no
    /// such attribute.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when `element` is unknown and
    /// [`DomError::NotSupported`] when it is not an element.
    pub fn remove_attribute_ns(
        &mut self,
        element: NodeId,
        namespace: Option<&str>,
        local_name: &str,
    ) -> DomResult<Option<NodeId>> {
        self.require_element(element)?;
        match self.get_attribute_node_ns(element, namespace, local_name) {
            Some(attribute) => {
                self.remove_attribute_node(element, attribute)?;
                Ok(Some(attribute))
            }
            None => Ok(None),
        }
    }

    /// Attaches an existing attribute node to `element`, replacing any attribute
    /// that already has the same expanded name.
    ///
    /// Returns the replaced attribute node, if there was one.
    ///
    /// # Errors
    ///
    /// * [`DomError::NotFound`] — either identifier is unknown.
    /// * [`DomError::NotSupported`] — `element` is not an element, or
    ///   `attribute` is not an attribute.
    /// * [`DomError::InUseAttribute`] — the attribute node already belongs to a
    ///   different element.
    pub fn set_attribute_node(
        &mut self,
        element: NodeId,
        attribute: NodeId,
    ) -> DomResult<Option<NodeId>> {
        self.require_element(element)?;
        let (name, qualified) = match &self.node(attribute)?.content {
            NodeContent::Attr(data) => (data.name.clone(), data.qualified_name()),
            _ => {
                return Err(DomError::NotSupported {
                    detail: "only an attribute node can be set as an attribute",
                });
            }
        };
        if let Some(owner) = self.owner_element(attribute) {
            if owner != element {
                return Err(DomError::InUseAttribute {
                    name: qualified,
                    owner,
                });
            }
            return Ok(None);
        }

        let replaced = self.get_attribute_node_ns(element, name.namespace(), name.local_name());
        if let Some(replaced) = replaced {
            self.remove_attribute_node(element, replaced)?;
        }
        self.attach_attribute(element, attribute);
        Ok(replaced)
    }

    /// Detaches an attribute node from its element.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when either identifier is unknown or when
    /// the attribute does not belong to `element`, and
    /// [`DomError::NotSupported`] when `element` is not an element.
    pub fn remove_attribute_node(
        &mut self,
        element: NodeId,
        attribute: NodeId,
    ) -> DomResult<NodeId> {
        self.require_element(element)?;
        self.node(attribute)?;
        let position = self
            .element(element)
            .and_then(|data| data.attributes.iter().position(|&id| id == attribute))
            .ok_or(DomError::NotFound(NotFound::NotAnAttribute {
                element,
                attribute,
            }))?;
        if let NodeContent::Element(data) = &mut self.node_mut(element)?.content {
            data.attributes.remove(position);
        }
        if let NodeContent::Attr(data) = &mut self.node_mut(attribute)?.content {
            data.owner = None;
        }
        self.invalidate_order();
        Ok(attribute)
    }

    /// Declares a namespace on `element` by adding the corresponding
    /// `xmlns` / `xmlns:prefix` attribute.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::set_attribute_ns`].
    pub fn declare_namespace(
        &mut self,
        element: NodeId,
        prefix: Option<&str>,
        namespace: &str,
    ) -> DomResult<NodeId> {
        let qualified_name = match prefix {
            Some(prefix) if !prefix.is_empty() => format!("xmlns:{prefix}"),
            _ => "xmlns".to_owned(),
        };
        self.set_attribute_ns(
            element,
            Some(XMLNS_NAMESPACE),
            &qualified_name,
            namespace.to_owned(),
        )
    }

    /// Returns the namespace declarations made *on* `element` itself, as
    /// `(prefix, namespace name)` pairs. `None` as the prefix is the default
    /// declaration `xmlns="…"`.
    #[must_use]
    pub fn namespace_declarations(&self, element: NodeId) -> Vec<(Option<&str>, &str)> {
        self.attributes(element)
            .filter_map(|id| {
                let data = self.attr(id)?;
                if data.namespace_uri() != Some(XMLNS_NAMESPACE) {
                    return None;
                }
                let prefix = if data.prefix() == Some("xmlns") {
                    Some(data.local_name())
                } else {
                    None
                };
                Some((prefix, data.value()))
            })
            .collect()
    }

    /// Resolves a namespace prefix against the declarations in scope at `node`,
    /// mirroring DOM Level 3 `Node.lookupNamespaceURI`.
    ///
    /// Pass `None` to look up the default namespace. The reserved prefixes `xml`
    /// and `xmlns` always resolve to their fixed namespaces.
    #[must_use]
    pub fn lookup_namespace_uri(&self, node: NodeId, prefix: Option<&str>) -> Option<&str> {
        match prefix {
            Some("xml") => return Some(crate::names::XML_NAMESPACE),
            Some("xmlns") => return Some(XMLNS_NAMESPACE),
            _ => {}
        }
        let start = match self.kind(node)? {
            NodeKind::Element => Some(node),
            NodeKind::Attr => self.owner_element(node),
            _ => self.ancestors(node).next(),
        };
        let mut cursor = start;
        while let Some(current) = cursor {
            if self.kind(current) == Some(NodeKind::Element) {
                for (declared, namespace) in self.namespace_declarations(current) {
                    if declared == prefix.filter(|p| !p.is_empty()) {
                        return if namespace.is_empty() {
                            None
                        } else {
                            Some(namespace)
                        };
                    }
                }
            }
            cursor = self.parent(current);
        }
        None
    }

    /// Returns a prefix that is bound to `namespace` in scope at `node`,
    /// mirroring DOM Level 3 `Node.lookupPrefix`.
    ///
    /// The nearest declaration wins. `None` is returned when only the default
    /// namespace is bound to it (a default declaration has no prefix) or when the
    /// namespace is not in scope at all.
    #[must_use]
    pub fn lookup_prefix(&self, node: NodeId, namespace: &str) -> Option<&str> {
        let start = match self.kind(node)? {
            NodeKind::Element => Some(node),
            NodeKind::Attr => self.owner_element(node),
            _ => self.ancestors(node).next(),
        };
        let mut cursor = start;
        while let Some(current) = cursor {
            if self.kind(current) == Some(NodeKind::Element) {
                for (declared, declared_namespace) in self.namespace_declarations(current) {
                    if let Some(prefix) = declared {
                        if declared_namespace == namespace
                            && self.lookup_namespace_uri(node, Some(prefix)) == Some(namespace)
                        {
                            return Some(prefix);
                        }
                    }
                }
            }
            cursor = self.parent(current);
        }
        None
    }

    /// Returns `true` when `namespace` is the default namespace in scope at
    /// `node`, mirroring DOM Level 3 `Node.isDefaultNamespace`.
    #[must_use]
    pub fn is_default_namespace(&self, node: NodeId, namespace: &str) -> bool {
        self.lookup_namespace_uri(node, None) == Some(namespace)
    }

    // -------------------------------------------------------------- internals

    /// Appends an attribute node to an element's attribute list without any
    /// duplicate or ownership check.
    pub(crate) fn attach_attribute(&mut self, element: NodeId, attribute: NodeId) {
        if let Some(NodeContent::Element(data)) =
            self.get_mut(element).map(|node| &mut node.content)
        {
            data.attributes.push(attribute);
        }
        if let Some(NodeContent::Attr(data)) = self.get_mut(attribute).map(|node| &mut node.content)
        {
            data.owner = Some(element);
        }
        self.invalidate_order();
    }

    pub(crate) fn require_element(&self, id: NodeId) -> DomResult<()> {
        match self.node(id)?.kind() {
            NodeKind::Element => Ok(()),
            _ => Err(DomError::NotSupported {
                detail: "the node is not an element",
            }),
        }
    }
}
