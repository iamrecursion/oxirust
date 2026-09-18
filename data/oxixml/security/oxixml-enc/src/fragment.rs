//! Serializing a subtree so it can be encrypted, and re-parsing it after decryption.
//!
//! This is the part of XML Encryption that is not cryptography at all, and it is where
//! implementations usually go wrong. [XMLEnc §4.5.3.1](https://www.w3.org/TR/xmlenc-core1/#sec-Default-Namespace)
//! spells out the failure: an element serialized out of its document loses the namespace
//! context it was sitting in, and when the plaintext is parsed back inside that context the
//! *same octets* mean something different.
//!
//! The specification's own example is
//!
//! ```xml
//! <Document xmlns="http://example.org/">
//!   <ToBeEncrypted xmlns="" />
//! </Document>
//! ```
//!
//! A canonicalizer serializes `ToBeEncrypted` as `<ToBeEncrypted></ToBeEncrypted>`, because in
//! canonical form a redundant `xmlns=""` is suppressed. Decrypting that inside `Document` puts
//! the element into `http://example.org/`, which is not where it came from.
//!
//! # What this module does
//!
//! **Encrypting** ([`serialize_self_contained`]) copies the subtree into a scratch document, so
//! that it has no ancestors and therefore no inherited context, and then adds back exactly the
//! declarations the specification asks for:
//!
//! * every namespace declaration in scope at the original node that the node does not itself
//!   redeclare — the same set inclusive canonical XML renders at the apex of a subtree;
//! * an explicit `xmlns=""` when the node is in no namespace but a non-empty default namespace
//!   was in scope around it, which is §4.5.3.1's augmentation.
//!
//! **Decrypting** ([`wrap_in_context`]) implements the *text wrapping* of
//! [XMLEnc §4.5.4](https://www.w3.org/TR/xmlenc-core1/#sec-Text-Wrapping): emit a document type
//! declaration for the entities in the parsing context, then a dummy element carrying every
//! in-scope namespace declaration, then the plaintext, then the end tag. Parsing that and
//! discarding the wrapper yields exactly the node set the plaintext denotes *in the context it
//! is being put back into*.
//!
//! # `xml:base`, `xml:lang`, `xml:space`
//!
//! [XMLEnc §4.5.3.2](https://www.w3.org/TR/xmlenc-core1/#sec-XML-Attribute) notes that the same
//! problem exists for the `xml:` attributes, but that it *cannot* be solved the same way: an
//! empty `xml:base` or `xml:space` is undefined rather than a reset, so there is nothing to
//! emit. The specification's advice is to make fragments not depend on inherited `xml:`
//! attributes, and that advice is passed on here rather than papered over: this module does
//! **not** copy inherited `xml:base`, `xml:lang` or `xml:space` onto the fragment.

use std::collections::BTreeSet;

use oxixml_dom::{
    Document, DocumentTypeData, NodeId, NodeKind, SerializeOptions, XMLNS_NAMESPACE, build,
};

use crate::error::Result;

/// The element name the text-wrapping step of XMLEnc §4.5.4 uses for its dummy element.
///
/// The name is arbitrary — the wrapper is discarded — but it must not collide with anything a
/// plaintext could plausibly declare, and it must be a legal `NCName`.
const WRAPPER_NAME: &str = "oxixml-enc-decryption-context";

/// Returns the namespace declarations in force at `node` because of its *ancestors*, nearest
/// declaration winning, excluding any prefix `node` itself redeclares.
///
/// The result is the set inclusive canonical XML renders at the apex of a subtree.
fn inherited_namespaces(doc: &Document, node: NodeId) -> Vec<(Option<String>, String)> {
    let mut own: Vec<Option<String>> = Vec::new();
    if doc.kind(node) == Some(NodeKind::Element) {
        for (prefix, _) in doc.namespace_declarations(node) {
            own.push(prefix.map(str::to_owned));
        }
    }

    // `ancestors` walks outward, so the first declaration found for a prefix is the nearest one.
    let mut bindings: Vec<(Option<String>, String)> = Vec::new();
    for ancestor in doc.ancestors(node) {
        for (prefix, namespace) in doc.namespace_declarations(ancestor) {
            let prefix = prefix.map(str::to_owned);
            if own.contains(&prefix) {
                continue;
            }
            if bindings.iter().any(|(known, _)| *known == prefix) {
                continue;
            }
            bindings.push((prefix, namespace.to_owned()));
        }
    }
    bindings
}

/// Returns the prefixes — with `None` standing for the default binding — that the subtree
/// rooted at `node` actually uses in an element or attribute *name*.
///
/// This is the *visibly utilized* set of
/// [Exclusive XML Canonicalization §3](https://www.w3.org/TR/xml-exc-c14n/#sec-Specification).
/// Declaring only these keeps the ciphertext (and so the document) from growing by every
/// namespace that happened to be in scope, which for a deeply nested document is most of them.
///
/// The known cost — the same one exclusive canonicalization documents in its own Security
/// Considerations — is that a prefix used *only inside an attribute value or in character data*
/// (a QName in content, as XPath expressions and schema `type` attributes contain) is not
/// visible here and will not be carried over. XML Encryption cannot detect that case in general:
/// only the application knows whether a given attribute value is a QName.
fn visibly_utilized(doc: &Document, node: NodeId) -> BTreeSet<Option<String>> {
    let mut used = BTreeSet::new();
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        for child in doc.children(current) {
            stack.push(child);
        }
        let Some(data) = doc.element(current) else {
            continue;
        };
        if data.namespace_uri().is_some() {
            used.insert(data.prefix().map(str::to_owned));
        }
        for attribute in doc.attributes(current) {
            let Some(attr) = doc.attr(attribute) else {
                continue;
            };
            // A namespace declaration is not itself a *use* of the `xmlns` prefix.
            if attr.namespace_uri() == Some(XMLNS_NAMESPACE) {
                continue;
            }
            if attr.namespace_uri().is_some() {
                if let Some(prefix) = attr.prefix() {
                    used.insert(Some(prefix.to_owned()));
                }
            }
        }
    }
    used
}

/// Serializes `node` — an element — as a self-contained XML fragment.
///
/// The fragment carries every namespace declaration it needs, including the `xmlns=""` of
/// XMLEnc §4.5.3.1 when the element is in no namespace but the surrounding document put a
/// default namespace in scope.
///
/// # Errors
///
/// Returns [`EncError::Dom`](crate::EncError::Dom) when the node cannot be imported into the
/// scratch document, which can only happen for an identifier that does not belong to `doc`.
pub fn serialize_self_contained(doc: &Document, node: NodeId) -> Result<String> {
    let inherited = inherited_namespaces(doc, node);
    let utilized = visibly_utilized(doc, node);

    let mut scratch = Document::new();
    let copy = scratch.import_node(doc, node, true)?;

    if scratch.kind(copy) == Some(NodeKind::Element) {
        // Whether the apex is unprefixed and in no namespace at all. Only in that case does
        // re-parsing inside a default namespace change its meaning, and only then does
        // §4.5.3.1's `xmlns=""` augmentation apply.
        let apex_is_namespaceless = scratch
            .element(copy)
            .is_some_and(|data| data.prefix().is_none() && data.namespace_uri().is_none());

        for (prefix, namespace) in &inherited {
            if prefix.is_none() && apex_is_namespaceless {
                // The augmentation of §4.5.3.1: emit the empty declaration the canonicalizer
                // would have suppressed, so the fragment cannot be pulled into the surrounding
                // default namespace when it is parsed back.
                scratch.declare_namespace(copy, None, "")?;
                continue;
            }
            if utilized.contains(prefix) {
                scratch.declare_namespace(copy, prefix.as_deref(), namespace)?;
            }
        }
    }

    Ok(scratch.node_to_xml(copy, &fragment_options()))
}

/// Serializes every child of `element` as one octet string, which is what encrypting element
/// *content* (`Type="…#Content"`) operates on.
///
/// Each element child is made self-contained exactly as [`serialize_self_contained`] does;
/// character data, comments and processing instructions are written as they stand.
///
/// # Errors
///
/// Returns [`EncError::Dom`](crate::EncError::Dom) for an identifier that does not belong to
/// `doc`.
pub fn serialize_children(doc: &Document, element: NodeId) -> Result<String> {
    let mut output = String::new();
    for child in doc.children(element).collect::<Vec<_>>() {
        match doc.kind(child) {
            Some(NodeKind::Element) => output.push_str(&serialize_self_contained(doc, child)?),
            Some(_) => output.push_str(&doc.node_to_xml(child, &fragment_options())),
            None => {}
        }
    }
    Ok(output)
}

/// The serializer settings a fragment is written with: no XML declaration, no indentation (which
/// would add white space to the plaintext), empty elements folded, namespace fixup on.
fn fragment_options() -> SerializeOptions {
    SerializeOptions::new()
}

/// Wraps `plaintext` in the parsing context of `context` and parses it, per XMLEnc §4.5.4.
///
/// `context` is the node the plaintext is going to be inserted *under* — the parent of the
/// `EncryptedData` for `Type="…#Content"`, or the parent of the element being replaced for
/// `Type="…#Element"`. Every namespace declaration in scope there is emitted on the dummy
/// wrapper, and the internal DTD subset of the containing document is emitted ahead of it so
/// that general entity references in the plaintext resolve the way they would have before
/// encryption.
///
/// Returns the parsed wrapper document; the caller takes the wrapper element's children.
///
/// # Errors
///
/// Returns [`EncError::Build`](crate::EncError::Build) when the plaintext is not well-formed
/// XML in that context — which, after a successful authenticated decryption, means the sender
/// produced something invalid rather than that an attacker tampered with it.
pub fn wrap_in_context(doc: &Document, context: NodeId, plaintext: &str) -> Result<Document> {
    let mut wrapped = String::with_capacity(plaintext.len() + 256);

    let internal_subset = doc
        .doctype()
        .and_then(|doctype| doc.doctype_data(doctype))
        .and_then(DocumentTypeData::internal_subset)
        .filter(|subset| !subset.trim().is_empty());
    if let Some(subset) = internal_subset {
        wrapped.push_str("<!DOCTYPE ");
        wrapped.push_str(WRAPPER_NAME);
        wrapped.push_str(" [\n");
        wrapped.push_str(subset);
        wrapped.push_str("\n]>\n");
    }

    wrapped.push('<');
    wrapped.push_str(WRAPPER_NAME);
    for (prefix, namespace) in in_scope_namespaces(doc, context) {
        wrapped.push(' ');
        match prefix {
            Some(prefix) => {
                wrapped.push_str("xmlns:");
                wrapped.push_str(&prefix);
            }
            None => wrapped.push_str("xmlns"),
        }
        wrapped.push_str("=\"");
        escape_attribute_value(&namespace, &mut wrapped);
        wrapped.push('"');
    }
    wrapped.push('>');
    wrapped.push_str(plaintext);
    wrapped.push_str("</");
    wrapped.push_str(WRAPPER_NAME);
    wrapped.push('>');

    Ok(build::parse_str(&wrapped)?)
}

/// Returns the local name of the dummy element [`wrap_in_context`] produces, so that the caller
/// can find it in the parsed document.
#[must_use]
pub fn wrapper_name() -> &'static str {
    WRAPPER_NAME
}

/// Removes namespace declarations in the subtree rooted at `root` that only restate a binding
/// already in force at their position.
///
/// This is infoset-preserving by construction: a declaration is dropped only when looking its
/// prefix up at the element's *parent* yields exactly the same namespace name (or, for
/// `xmlns=""`, when no default namespace is in scope there anyway). Two callers need it:
///
/// * [`crate::model::EncryptedData::write_into`], because the structure builder declares the
///   prefix it uses on every element it creates rather than tracking scope, which is simple and
///   always correct but produces `xmlns:xenc` a dozen times over;
/// * [`crate::decrypt`], because the fragment and the §4.5.4 wrapper both carry declarations the
///   destination already provides — removing them is what makes encrypting and decrypting an
///   element in place reproduce the original tree literally rather than merely equivalently.
///
/// # Errors
///
/// Returns [`EncError::Dom`](crate::EncError::Dom) for an identifier that does not belong to
/// `doc`.
pub fn drop_redundant_namespace_declarations(doc: &mut Document, root: NodeId) -> Result<()> {
    let mut stack = vec![root];
    while let Some(current) = stack.pop() {
        if doc.kind(current) != Some(NodeKind::Element) {
            continue;
        }
        // Children are cleaned after their parent, so each lookup sees an already-minimal
        // ancestor chain. Since only redundant declarations are removed, the bindings a lookup
        // resolves to are the same either way; the order just avoids doing the work twice.
        for child in doc.children(current).collect::<Vec<_>>() {
            stack.push(child);
        }

        let parent = doc.parent(current);
        let declarations: Vec<(Option<String>, String)> = doc
            .namespace_declarations(current)
            .into_iter()
            .map(|(prefix, namespace)| (prefix.map(str::to_owned), namespace.to_owned()))
            .collect();

        for (prefix, namespace) in declarations {
            let in_scope = parent.and_then(|parent| {
                doc.lookup_namespace_uri(parent, prefix.as_deref())
                    .map(str::to_owned)
            });
            let redundant = match in_scope.as_deref() {
                Some(existing) => existing == namespace,
                // `xmlns=""` says "no default namespace", which is already true here.
                None => namespace.is_empty(),
            };
            if redundant {
                let name = match &prefix {
                    Some(prefix) => format!("xmlns:{prefix}"),
                    None => "xmlns".to_owned(),
                };
                doc.remove_attribute(current, &name)?;
            }
        }
    }
    Ok(())
}

/// Returns every namespace declaration in scope at `node`, including the node's own, nearest
/// declaration winning.
///
/// A declaration that undeclares the default namespace (`xmlns=""`) is kept, because emitting it
/// on the wrapper is precisely what stops the wrapper's own context from leaking into the
/// plaintext.
fn in_scope_namespaces(doc: &Document, node: NodeId) -> Vec<(Option<String>, String)> {
    let mut bindings: Vec<(Option<String>, String)> = Vec::new();
    let visit = |element: NodeId, bindings: &mut Vec<(Option<String>, String)>| {
        for (prefix, namespace) in doc.namespace_declarations(element) {
            let prefix = prefix.map(str::to_owned);
            if bindings.iter().any(|(known, _)| *known == prefix) {
                continue;
            }
            bindings.push((prefix, namespace.to_owned()));
        }
    };
    if doc.kind(node) == Some(NodeKind::Element) {
        visit(node, &mut bindings);
    }
    for ancestor in doc.ancestors(node).collect::<Vec<_>>() {
        visit(ancestor, &mut bindings);
    }
    bindings
}

/// Escapes a namespace name for use inside a double-quoted attribute value.
fn escape_attribute_value(value: &str, out: &mut String) {
    for character in value.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(xml: &str) -> Result<Document> {
        Ok(build::parse_str(xml)?)
    }

    fn first_child_element(doc: &Document, parent: NodeId) -> Option<NodeId> {
        doc.children(parent)
            .find(|child| doc.kind(*child) == Some(NodeKind::Element))
    }

    #[test]
    fn the_specification_example_keeps_its_empty_default_namespace() -> Result<()> {
        let doc = parse(r#"<Document xmlns="http://example.org/"><ToBeEncrypted/></Document>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        // The child is *in* the default namespace here, so the declaration must be carried over.
        let serialized = serialize_self_contained(&doc, target)?;
        assert_eq!(
            serialized,
            r#"<ToBeEncrypted xmlns="http://example.org/"/>"#
        );
        Ok(())
    }

    #[test]
    fn an_element_in_no_namespace_gains_the_empty_declaration() -> Result<()> {
        let doc =
            parse(r#"<Document xmlns="http://example.org/"><ToBeEncrypted xmlns=""/></Document>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        let serialized = serialize_self_contained(&doc, target)?;
        assert!(
            serialized.contains(r#"xmlns="""#),
            "§4.5.3.1 requires the empty declaration: {serialized}"
        );
        Ok(())
    }

    #[test]
    fn inherited_prefixes_are_carried_onto_the_fragment() -> Result<()> {
        let doc =
            parse(r#"<r xmlns:p="urn:p" xmlns:q="urn:q"><p:inner q:a="1">text</p:inner></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        let serialized = serialize_self_contained(&doc, target)?;
        assert!(serialized.contains(r#"xmlns:p="urn:p""#), "{serialized}");
        assert!(serialized.contains(r#"xmlns:q="urn:q""#), "{serialized}");
        // The fragment must parse on its own.
        let reparsed = parse(&serialized)?;
        let reparsed_root = reparsed.document_element().expect("a document element");
        assert_eq!(
            reparsed
                .element(reparsed_root)
                .and_then(|data| data.namespace_uri()),
            Some("urn:p")
        );
        Ok(())
    }

    #[test]
    fn a_redeclared_prefix_is_not_overwritten() -> Result<()> {
        let doc = parse(r#"<r xmlns:p="urn:outer"><p:inner xmlns:p="urn:inner"/></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        let serialized = serialize_self_contained(&doc, target)?;
        assert!(serialized.contains("urn:inner"), "{serialized}");
        assert!(!serialized.contains("urn:outer"), "{serialized}");
        Ok(())
    }

    #[test]
    fn children_serialize_in_order_including_character_data() -> Result<()> {
        let doc = parse(r#"<r xmlns:p="urn:p">a<p:x/>b<!--c--><?pi d?></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let serialized = serialize_children(&doc, root)?;
        assert_eq!(serialized, r#"a<p:x xmlns:p="urn:p"/>b<!--c--><?pi d?>"#);
        Ok(())
    }

    #[test]
    fn wrapping_restores_the_parsing_context() -> Result<()> {
        let doc = parse(r#"<r xmlns="urn:d" xmlns:p="urn:p"><child/></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let wrapper_doc = wrap_in_context(&doc, root, "<p:inner/><plain/>")?;
        let wrapper = wrapper_doc.document_element().expect("a wrapper");
        assert_eq!(
            wrapper_doc.element(wrapper).map(|data| data.local_name()),
            Some(wrapper_name())
        );
        let children: Vec<NodeId> = wrapper_doc.children(wrapper).collect();
        assert_eq!(children.len(), 2);
        assert_eq!(
            wrapper_doc
                .element(children[0])
                .and_then(|data| data.namespace_uri()),
            Some("urn:p")
        );
        // The unprefixed element picks the default namespace up from the context, which is the
        // whole point of the wrapper.
        assert_eq!(
            wrapper_doc
                .element(children[1])
                .and_then(|data| data.namespace_uri()),
            Some("urn:d")
        );
        Ok(())
    }

    #[test]
    fn wrapping_carries_the_internal_subset_so_entities_resolve() -> Result<()> {
        let doc =
            parse("<!DOCTYPE r [<!ENTITY greeting \"hello\">]><r xmlns=\"urn:d\">&greeting;</r>")?;
        let root = doc.document_element().expect("a document element");
        let wrapper_doc = wrap_in_context(&doc, root, "<x>&greeting;</x>")?;
        let wrapper = wrapper_doc.document_element().expect("a wrapper");
        let child = first_child_element(&wrapper_doc, wrapper).expect("a child");
        assert_eq!(wrapper_doc.text_content(child).as_deref(), Some("hello"));
        Ok(())
    }

    #[test]
    fn namespace_names_are_escaped_in_the_wrapper() -> Result<()> {
        // A namespace name containing `&` or `"` would otherwise produce a wrapper that is not
        // well-formed, which would turn every decryption in that document into a parse error.
        let doc = parse(r#"<r xmlns:p="urn:a&amp;b&quot;c"/>"#)?;
        let root = doc.document_element().expect("a document element");
        let wrapper_doc = wrap_in_context(&doc, root, "<plain/>")?;
        let wrapper = wrapper_doc.document_element().expect("a wrapper");
        assert_eq!(
            wrapper_doc.namespace_declarations(wrapper),
            [(Some("p"), "urn:a&b\"c")]
        );
        Ok(())
    }

    #[test]
    fn only_visibly_utilized_prefixes_are_carried_over() -> Result<()> {
        let doc = parse(r#"<r xmlns:p="urn:p" xmlns:unused="urn:unused"><p:inner/></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        let serialized = serialize_self_contained(&doc, target)?;
        assert!(serialized.contains(r#"xmlns:p="urn:p""#), "{serialized}");
        assert!(!serialized.contains("urn:unused"), "{serialized}");
        Ok(())
    }

    #[test]
    fn a_descendant_that_uses_a_prefix_keeps_it() -> Result<()> {
        let doc = parse(r#"<r xmlns:p="urn:p"><plain><p:deep/></plain></r>"#)?;
        let root = doc.document_element().expect("a document element");
        let target = first_child_element(&doc, root).expect("a child");
        let serialized = serialize_self_contained(&doc, target)?;
        // The declaration lands once, on the apex, rather than on every descendant.
        assert_eq!(serialized.matches("urn:p").count(), 1, "{serialized}");
        assert!(parse(&serialized).is_ok(), "{serialized}");
        Ok(())
    }

    #[test]
    fn escaping_covers_every_significant_character() {
        let mut out = String::new();
        escape_attribute_value("a<b>c&d\"e", &mut out);
        assert_eq!(out, "a&lt;b&gt;c&amp;d&quot;e");
    }
}
