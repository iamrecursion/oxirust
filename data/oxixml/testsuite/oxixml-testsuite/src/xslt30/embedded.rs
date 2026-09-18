//! The *embedded* stylesheet a `<source defines-stylesheet="true">` puts inside the source
//! document, and how the catalogue points at it.
//!
//! XSLT 3.0 [§3.11](https://www.w3.org/TR/xslt-30/#embedded) describes a stylesheet module that
//! is not a document of its own: it is an `xsl:stylesheet` element somewhere inside another XML
//! document, and the document says which element that is with an
//! [`xml-stylesheet`](https://www.w3.org/TR/xml-stylesheet/) processing instruction whose `href`
//! pseudo-attribute is a bare fragment identifier — `<?xml-stylesheet type="text/xsl"
//! href="#style1"?>` — naming an element of the same document by its ID.
//!
//! Three consequences shape this module:
//!
//! * the *stylesheet* is a node, not a file, so the compiler is handed an element rather than
//!   text (`Stylesheet::compile_with`), and the whole document is still the principal source;
//! * the fragment names an **ID**, which the document's own document type declaration is what
//!   makes one: every case in the catalogue's `embedded-stylesheet` set declares
//!   `<!ATTLIST xsl:stylesheet id ID #REQUIRED>` in its internal subset, so the lookup asks the
//!   data model whether an attribute *is* an ID rather than guessing from its name;
//! * a document may carry several such processing instructions, and the one that governs is the
//!   first that is not marked `alternate="yes"` ([Associating Style Sheets with XML
//!   documents §3](https://www.w3.org/TR/xml-stylesheet/)).

use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};

/// The target of the processing instruction that names an embedded stylesheet.
const XML_STYLESHEET: &str = "xml-stylesheet";

/// The namespace `xml:` is bound to, which no document may rebind — the one an `xml:id`
/// attribute is in.
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";

/// The element an `xml-stylesheet` processing instruction in `document` names, or `None` when
/// the document names no embedded stylesheet this runner can reach.
///
/// `document` is the source document's node; the processing instruction is looked for among its
/// children, which is where the prolog puts one.
pub(super) fn embedded_module(document: &NodeRef) -> Option<NodeRef> {
    let fragment = stylesheet_fragment(document)?;
    element_with_id(document, &fragment)
}

/// The fragment identifier the governing `xml-stylesheet` processing instruction points at,
/// without its `#`.
///
/// Only a *same-document* reference names an embedded stylesheet: an `href` naming another
/// resource is an ordinary stylesheet association, which the catalogue expresses with a
/// `<stylesheet file="…"/>` instead and this runner never reaches through here.
fn stylesheet_fragment(document: &NodeRef) -> Option<String> {
    document
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::ProcessingInstruction)
        .filter(|child| {
            child
                .node_name()
                .is_some_and(|name| name.local_name() == XML_STYLESHEET)
        })
        .find_map(|instruction| {
            let pseudo_attributes = parse_pseudo_attributes(&instruction.string_value());
            if pseudo_attribute(&pseudo_attributes, "alternate").as_deref() == Some("yes") {
                return None;
            }
            let href = pseudo_attribute(&pseudo_attributes, "href")?;
            href.strip_prefix('#').map(str::to_owned)
        })
}

/// The `name="value"` pairs a processing instruction's content holds, in the order they are
/// written.
///
/// The [xml-stylesheet](https://www.w3.org/TR/xml-stylesheet/) specification defines the content
/// as *pseudo-attributes*: they look like attributes but are not, so no XML parser has produced
/// them and the value is taken literally apart from the quotes around it. Anything that does not
/// match the shape ends the scan, exactly as an unparsable pseudo-attribute list ends one there.
fn parse_pseudo_attributes(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut rest = content.trim_start();
    while !rest.is_empty() {
        let Some(equals) = rest.find('=') else {
            break;
        };
        let name = rest[..equals].trim();
        if name.is_empty() || name.contains(char::is_whitespace) {
            break;
        }
        let after = rest[equals + 1..].trim_start();
        let Some(quote) = after
            .chars()
            .next()
            .filter(|mark| *mark == '"' || *mark == '\'')
        else {
            break;
        };
        let value_start = quote.len_utf8();
        let Some(end) = after[value_start..].find(quote) else {
            break;
        };
        pairs.push((
            name.to_owned(),
            after[value_start..value_start + end].to_owned(),
        ));
        rest = after[value_start + end + quote.len_utf8()..].trim_start();
    }
    pairs
}

/// The value of the pseudo-attribute `name`, or `None` when the instruction wrote none.
fn pseudo_attribute(pairs: &[(String, String)], name: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(written, _)| written == name)
        .map(|(_, value)| value.clone())
}

/// The element inside `root` whose ID is `id`.
///
/// The data model's own `[is-id]` property decides which attribute is the identifier — the
/// document type declaration is what makes an attribute an ID, and the catalogue's embedded
/// stylesheets are found through a plain `id` attribute that their internal subset declares as
/// one. `xml:id` is accepted beside it, since [xml:id §4](https://www.w3.org/TR/xml-id/#processing)
/// makes that attribute an identifier with no declaration at all.
fn element_with_id(root: &NodeRef, id: &str) -> Option<NodeRef> {
    if root.node_kind() == NodeKind::Element && names_element(root, id) {
        return Some(root.clone());
    }
    root.children()
        .into_iter()
        .find_map(|child| element_with_id(&child, id))
}

/// Whether `element` is the one `id` names.
fn names_element(element: &NodeRef, id: &str) -> bool {
    element.attributes().into_iter().any(|attribute| {
        attribute.string_value() == id && (attribute.is_id() || is_xml_id(&attribute))
    })
}

/// Whether `attribute` is `xml:id`.
fn is_xml_id(attribute: &NodeRef) -> bool {
    attribute
        .node_name()
        .is_some_and(|name| name == ExpandedName::new(Some(XML_NAMESPACE), "id"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    /// The whole mechanism over the shape the catalogue writes: an internal subset that declares
    /// `id` an ID, a processing instruction naming one, and an `xsl:stylesheet` element carrying
    /// it.
    #[test]
    fn an_xml_stylesheet_instruction_names_an_embedded_module() {
        let tree = Tree::parse_str(
            r##"<?xml-stylesheet type="text/xsl" href="#style1"?>
               <!DOCTYPE doc [
                 <!ELEMENT doc ANY>
                 <!ELEMENT xsl:stylesheet ANY>
                 <!ATTLIST xsl:stylesheet id ID #REQUIRED>
               ]>
               <doc><head><xsl:stylesheet
                     xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                     version="3.0" id="style1"/></head></doc>"##,
        )
        .expect("well-formed fixture");
        let module = embedded_module(&tree.root()).expect("the instruction names a module");
        assert_eq!(
            module
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default(),
            "stylesheet"
        );
    }

    /// An `href` that is not a same-document reference names no embedded module, and neither
    /// does an absent instruction.
    #[test]
    fn only_a_fragment_reference_names_an_embedded_module() {
        let external = Tree::parse_str(
            r##"<?xml-stylesheet type="text/xsl" href="other.xsl"?><doc id="x"/>"##,
        )
        .expect("well-formed fixture");
        assert!(embedded_module(&external.root()).is_none());

        let none = Tree::parse_str("<doc/>").expect("well-formed fixture");
        assert!(embedded_module(&none.root()).is_none());
    }

    /// An `alternate="yes"` association is passed over for the next one, which is the rule
    /// [Associating Style Sheets with XML documents §3](https://www.w3.org/TR/xml-stylesheet/)
    /// states.
    #[test]
    fn an_alternate_association_does_not_govern() {
        let tree = Tree::parse_str(
            r##"<?xml-stylesheet alternate="yes" href="#other"?>
               <?xml-stylesheet href="#wanted"?>
               <doc><a xml:id="other"/><b xml:id="wanted"/></doc>"##,
        )
        .expect("well-formed fixture");
        let module = embedded_module(&tree.root()).expect("the second instruction governs");
        assert_eq!(
            module
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default(),
            "b"
        );
    }

    /// An attribute merely *named* `id` is not an ID: without a declaration making it one, the
    /// fragment names nothing. This is what keeps the lookup from finding the wrong element in a
    /// document whose ordinary data happens to use the name.
    #[test]
    fn an_undeclared_id_attribute_is_not_an_identifier() {
        let tree = Tree::parse_str(r##"<?xml-stylesheet href="#x"?><doc><a id="x"/></doc>"##)
            .expect("well-formed fixture");
        assert!(embedded_module(&tree.root()).is_none());
    }

    /// The pseudo-attribute scan reads both quoting styles and stops at the first thing that is
    /// not a pair, which is all the specification asks of it.
    #[test]
    fn pseudo_attributes_are_read_as_written() {
        assert_eq!(
            parse_pseudo_attributes(r##"type='text/xsl' href="#s" title='a b'"##),
            vec![
                ("type".to_owned(), "text/xsl".to_owned()),
                ("href".to_owned(), "#s".to_owned()),
                ("title".to_owned(), "a b".to_owned()),
            ]
        );
        assert_eq!(parse_pseudo_attributes("href=#s"), Vec::new());
        assert_eq!(parse_pseudo_attributes(""), Vec::new());
    }
}
