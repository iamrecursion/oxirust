//! The declared types an `<!ATTLIST …>` gives its attributes.
//!
//! An attribute-list declaration says two things the data model keeps. Its `AttType` decides how
//! the value is normalized — [XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize): "if the
//! attribute type is not CDATA, then the XML processor MUST further process the normalized
//! attribute value by discarding any leading and trailing space characters, and by replacing
//! sequences of space characters by a single space character" — and, for the three reference
//! types, it decides the `is-id` and `is-idref` properties
//! [XDM 3.1 §5.10](https://www.w3.org/TR/xpath-datamodel-31/#dm-is-id) gives the attribute node,
//! which is what `fn:id` and `fn:idref` search on.
//!
//! `xml:id` is an ID whether or not a declaration mentions it, and its value is normalized as if
//! it had been declared `ID` ([xml:id 1.0 §4](https://www.w3.org/TR/xml-id/#processing)).

use oxixml_infoset::{
    EXTERNAL_SUBSET_NAME, ExternalEntity, ExternalEntityError, ExternalEntityResolver, NodeRef,
    ParseOptions, Tree, XdmError, XdmNode,
};

/// The attributes of `node` as `(name, value, is_id, is_idref)`.
fn attributes(node: &NodeRef) -> Vec<(String, String, bool, bool)> {
    node.attributes()
        .iter()
        .map(|attribute| {
            (
                attribute
                    .node_name()
                    .map_or_else(String::new, |name| name.to_string()),
                attribute.string_value(),
                attribute.is_id(),
                attribute.is_idref(),
            )
        })
        .collect()
}

/// The document element of a parsed document.
fn root(xml: &str) -> Result<NodeRef, XdmError> {
    let tree = Tree::parse_str(xml)?;
    tree.document_element()
        .ok_or_else(|| XdmError::new(oxixml_infoset::ErrorCode::Fodc0006, "no document element"))
}

#[test]
fn a_declared_id_attribute_is_an_id() -> Result<(), XdmError> {
    let root = root(
        r#"<!DOCTYPE r [<!ATTLIST b code ID #REQUIRED>]><r><b code="one"/><b code="two"/></r>"#,
    )?;
    let ids: Vec<(String, bool)> = root
        .children()
        .iter()
        .flat_map(attributes)
        .map(|(_, value, is_id, _)| (value, is_id))
        .collect();
    assert_eq!(ids, [("one".to_owned(), true), ("two".to_owned(), true)]);
    Ok(())
}

#[test]
fn a_declared_idref_attribute_refers_to_an_id() -> Result<(), XdmError> {
    let root = root(
        r#"<!DOCTYPE r [<!ATTLIST b one IDREF #IMPLIED many IDREFS #IMPLIED plain CDATA #IMPLIED>]>
           <r><b one="a" many="a b" plain="a"/></r>"#,
    )?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [
            ("one".to_owned(), "a".to_owned(), false, true),
            ("many".to_owned(), "a b".to_owned(), false, true),
            ("plain".to_owned(), "a".to_owned(), false, false),
        ]
    );
    Ok(())
}

#[test]
fn xml_id_is_an_id_without_a_declaration() -> Result<(), XdmError> {
    let root = root(r#"<r xml:id="here" other="there"/>"#)?;
    assert_eq!(
        attributes(&root),
        [
            (
                "{http://www.w3.org/XML/1998/namespace}id".to_owned(),
                "here".to_owned(),
                true,
                false
            ),
            ("other".to_owned(), "there".to_owned(), false, false),
        ]
    );
    Ok(())
}

#[test]
fn an_xml_id_value_is_normalized_as_a_token() -> Result<(), XdmError> {
    // xml:id 1.0 §4 normalizes the value "as if the attribute were declared to be of type ID",
    // so the surrounding space is not part of the identifier.
    let root = root("<r xml:id=' id3 '><c xml:id='a\tb'/></r>")?;
    assert_eq!(
        root.attributes()
            .first()
            .map(|attribute| attribute.string_value()),
        Some("id3".to_owned())
    );
    // A tab arrives as a space from the first normalization pass, and the token pass keeps one.
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        child
            .attributes()
            .first()
            .map(|attribute| attribute.string_value()),
        Some("a b".to_owned())
    );
    Ok(())
}

#[test]
fn a_declaration_cannot_take_ids_ness_away_from_xml_id() -> Result<(), XdmError> {
    let root = root(r#"<!DOCTYPE r [<!ATTLIST r xml:id CDATA #IMPLIED>]><r xml:id=" x "/>"#)?;
    assert_eq!(
        attributes(&root),
        [(
            "{http://www.w3.org/XML/1998/namespace}id".to_owned(),
            "x".to_owned(),
            true,
            false
        )]
    );
    Ok(())
}

#[test]
fn a_written_tokenized_value_is_collapsed_and_cdata_is_not() -> Result<(), XdmError> {
    // The example is XML C14N's own §3.4: `attr` on `<normNames>` is NMTOKENS and the same value
    // on an undeclared attribute is CDATA. Only the space character counts, so the characters
    // written as references survive both passes.
    let root = root(
        "<!DOCTYPE r [<!ATTLIST n attr NMTOKENS #IMPLIED>]>\
         <r><n attr='   A   &#x20;&#13;&#xa;&#9;   B   ' plain='  A  B  '/></r>",
    )?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [
            ("attr".to_owned(), "A \r\n\t B".to_owned(), false, false),
            ("plain".to_owned(), "  A  B  ".to_owned(), false, false),
        ]
    );
    Ok(())
}

#[test]
fn a_defaulted_id_attribute_is_an_id() -> Result<(), XdmError> {
    let root = root(r#"<!DOCTYPE r [<!ATTLIST b code ID "  fixed  ">]><r><b/></r>"#)?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [("code".to_owned(), "fixed".to_owned(), true, false)]
    );
    Ok(())
}

#[test]
fn a_written_attribute_keeps_the_declared_type_of_the_default_it_replaces() -> Result<(), XdmError>
{
    let root = root(r#"<!DOCTYPE r [<!ATTLIST b code ID "given">]><r><b code=" written "/></r>"#)?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [("code".to_owned(), "written".to_owned(), true, false)]
    );
    Ok(())
}

#[test]
fn the_declaration_is_matched_lexically() -> Result<(), XdmError> {
    // Namespaces in XML leaves attribute-list declarations lexical, so a declaration of `code`
    // says nothing about an attribute written `p:code`.
    let root = root(
        r#"<!DOCTYPE r [<!ATTLIST b code ID #IMPLIED>]>
           <r xmlns:p="urn:p"><b p:code="one" code="two"/></r>"#,
    )?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [
            ("{urn:p}code".to_owned(), "one".to_owned(), false, false),
            ("code".to_owned(), "two".to_owned(), true, false),
        ]
    );
    Ok(())
}

#[test]
fn an_undeclared_attribute_is_neither_an_id_nor_a_reference() -> Result<(), XdmError> {
    let root = root(r#"<r id="looks like one" idref="so does this"/>"#)?;
    assert!(
        attributes(&root)
            .iter()
            .all(|(_, _, is_id, is_idref)| !is_id && !is_idref)
    );
    Ok(())
}

/// A resolver that answers every request with one fixed external subset.
struct Subset(&'static str);

impl ExternalEntityResolver for Subset {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        assert_eq!(entity.name, EXTERNAL_SUBSET_NAME);
        Ok(Some(self.0.to_owned()))
    }
}

#[test]
fn the_external_subset_declares_ids_too() -> Result<(), XdmError> {
    let mut resolver = Subset("<!ATTLIST b code ID #REQUIRED>");
    let tree = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r><b code="one"/></r>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    let child = tree
        .document_element()
        .and_then(|root| root.children().first().cloned())
        .expect("one child");
    assert_eq!(
        attributes(&child),
        [("code".to_owned(), "one".to_owned(), true, false)]
    );
    Ok(())
}

#[test]
fn the_internal_subset_outranks_the_external_one() -> Result<(), XdmError> {
    // XML 1.0 §2.8 has the internal subset read first, and §3.3 makes the first declaration of an
    // attribute the binding one — so the external subset cannot redeclare `code` as CDATA.
    let mut resolver = Subset("<!ATTLIST b code CDATA #REQUIRED>");
    let tree = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r SYSTEM "r.dtd" [<!ATTLIST b code ID #REQUIRED>]><r><b code=" one "/></r>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    let child = tree
        .document_element()
        .and_then(|root| root.children().first().cloned())
        .expect("one child");
    assert_eq!(
        attributes(&child),
        [("code".to_owned(), "one".to_owned(), true, false)]
    );
    Ok(())
}

#[test]
fn an_external_declaration_that_is_not_read_leaves_the_attribute_untyped() -> Result<(), XdmError> {
    // The default is not to read the external subset at all, which is what XML 1.0 §5.1 permits.
    let root = root(r#"<!DOCTYPE r SYSTEM "r.dtd"><r><b code=" one "/></r>"#)?;
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        attributes(&child),
        [("code".to_owned(), " one ".to_owned(), false, false)]
    );
    Ok(())
}

#[test]
fn a_required_id_in_an_internal_subset_with_element_declarations() -> Result<(), XdmError> {
    // The shape the W3C suites use: `<!ELEMENT>` declarations the scanner skips, followed by the
    // `<!ATTLIST>` it does not.
    let root = root(
        "<!DOCTYPE doc [\n  <!ELEMENT doc (a*)>\n  <!ELEMENT a (#PCDATA)>\n  \
         <!ATTLIST a  id ID #REQUIRED>\n]>\n<doc><a id=\"a\">A</a><a id=\"b\">B</a></doc>",
    )?;
    let ids: Vec<(String, bool)> = root
        .children()
        .iter()
        .flat_map(attributes)
        .map(|(_, value, is_id, _)| (value, is_id))
        .collect();
    assert_eq!(ids, [("a".to_owned(), true), ("b".to_owned(), true)]);
    Ok(())
}
