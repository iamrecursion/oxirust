use super::super::{SerializationParameters, serialize};
use super::*;
use crate::error::FnErrorCode;
use oxixml_infoset::{AtomicValue, ExpandedName, Tree, TreeBuilder, XdmArray, XdmMap};

fn document(xml: &str) -> Result<Sequence, Box<dyn std::error::Error>> {
    let tree = Tree::parse_str(xml)?;
    Ok(Sequence::from(
        tree.document_element().ok_or("no document element")?,
    ))
}

fn html(version: &str) -> SerializationParameters {
    SerializationParameters::default()
        .with_method(OutputMethod::Html)
        .with_version(version)
}

#[test]
fn xml_method_round_trips_a_simple_document() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a b='1'><c>text</c><!-- note --></a>")?;
    assert_eq!(
        serialize(&value, &SerializationParameters::default())?,
        r#"<a b="1"><c>text</c><!-- note --></a>"#
    );
    Ok(())
}

#[test]
fn markup_and_attribute_values_are_escaped() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a b='&lt;&amp;&quot;'>1 &lt; 2 &amp; 3</a>")?;
    assert_eq!(
        serialize(&value, &SerializationParameters::default())?,
        r#"<a b="&lt;&amp;&quot;">1 &lt; 2 &amp; 3</a>"#
    );
    Ok(())
}

#[test]
fn namespaces_are_declared_once() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<p:a xmlns:p='urn:p'><p:b/></p:a>")?;
    assert_eq!(
        serialize(&value, &SerializationParameters::default())?,
        r#"<p:a xmlns:p="urn:p"><p:b/></p:a>"#
    );

    let default_namespace = document("<a xmlns='urn:d'><b/></a>")?;
    assert_eq!(
        serialize(&default_namespace, &SerializationParameters::default())?,
        r#"<a xmlns="urn:d"><b/></a>"#
    );
    Ok(())
}

#[test]
fn processing_instructions_and_empty_elements() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a><?target data?><b/></a>")?;
    assert_eq!(
        serialize(&value, &SerializationParameters::default())?,
        "<a><?target data?><b/></a>"
    );
    Ok(())
}

#[test]
fn the_xml_declaration_is_optional() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a/>")?;
    let parameters = SerializationParameters::default()
        .with_omit_xml_declaration(false)
        .with_standalone(Some(true));
    assert_eq!(
        serialize(&value, &parameters)?,
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a/>"#
    );
    Ok(())
}

/// Serialization 3.1 §7: HTML has no XML declaration, so the `html` output method writes none
/// however `omit-xml-declaration` is set.
#[test]
fn the_html_method_writes_no_xml_declaration() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html/>")?;
    let parameters = html("4.0").with_omit_xml_declaration(false);
    assert_eq!(serialize(&value, &parameters)?, "<html></html>");
    Ok(())
}

#[test]
fn a_doctype_declaration_can_be_emitted() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html><body/></html>")?;
    let parameters = SerializationParameters::default()
        .with_doctype_system(Some("about:legacy-compat".to_owned()));
    assert_eq!(
        serialize(&value, &parameters)?,
        r#"<!DOCTYPE html SYSTEM "about:legacy-compat"><html><body/></html>"#
    );
    Ok(())
}

/// Serialization 3.1 §7.2: the `html` output method writes a document type declaration whenever
/// `doctype-public` *or* `doctype-system` is specified, so a public identifier alone is enough —
/// where §5.1.4 gives the `xml` method one only for a system identifier (QT3
/// `Serialization-html-25`, `-27`).
#[test]
fn the_html_method_writes_a_public_doctype() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html/>")?;
    let public = "-//W3C//DTD HTML 4.0//EN";
    for version in ["4.0", "5.0"] {
        let parameters = html(version).with_doctype_public(Some(public.to_owned()));
        assert_eq!(
            serialize(&value, &parameters)?,
            format!("<!DOCTYPE html PUBLIC \"{public}\"><html></html>"),
            "{version}"
        );
    }
    // With both, the system identifier follows the public one.
    let both = html("4.0")
        .with_doctype_public(Some(public.to_owned()))
        .with_doctype_system(Some("http://example.com/html4.dtd".to_owned()));
    assert_eq!(
        serialize(&value, &both)?,
        format!(
            "<!DOCTYPE html PUBLIC \"{public}\" \"http://example.com/html4.dtd\"><html></html>"
        )
    );
    // The `xml` method ignores a public identifier with no system identifier.
    let as_xml = SerializationParameters::default().with_doctype_public(Some(public.to_owned()));
    assert_eq!(serialize(&value, &as_xml)?, "<html/>");
    Ok(())
}

#[test]
fn indentation_is_applied_to_element_only_content() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a><b/><c><d/></c></a>")?;
    let parameters = SerializationParameters::default().with_indent(true);
    assert_eq!(
        serialize(&value, &parameters)?,
        "<a>\n  <b/>\n  <c>\n    <d/>\n  </c>\n</a>"
    );
    Ok(())
}

/// Serialization 3.1 §5.1.1: indentation may only be added where it cannot change what the
/// content means, so an element that holds text is left exactly as it stands — the rule reaches
/// the *whole* of that element's content, not only the text (QT3 `Serialization-html-49`…`-52`,
/// `Serialization-xhtml-61`, `method-xml/K2-Serialization-42`).
#[test]
fn mixed_content_is_never_indented() -> Result<(), Box<dyn std::error::Error>> {
    let indented = SerializationParameters::default().with_indent(true);
    let value = document("<test><a xml:space='preserve'> <x/> </a><b/></test>")?;
    let written = serialize(&value, &indented)?;
    assert!(
        written.contains("<a xml:space=\"preserve\"> <x/> </a>"),
        "{written}"
    );
    assert!(written.contains("<test>\n  <a"), "{written}");
    assert!(written.contains("</a>\n  <b/>\n</test>"), "{written}");

    // The same under the `html` method, where an inline element sits between two pieces of text.
    for (version, inline) in [("4.0", "a"), ("5.0", "mark"), ("4.0", "A"), ("5.0", "Mark")] {
        let page = document(&format!(
            "<html><head/><body><p>a<{inline}>text</{inline}>z</p></body></html>"
        ))?;
        let written = serialize(&page, &html(version).with_indent(true))?;
        assert!(
            written.contains(&format!("<p>a<{inline}>text</{inline}>z</p>")),
            "{written}"
        );
    }
    Ok(())
}

/// Indentation needs something to indent: an element whose content holds no element at all —
/// a lone processing instruction, say — is written as it stands (QT3 `Serialization-html-58`,
/// `-59`).
#[test]
fn indentation_needs_an_element_child() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html><head/><body><p><?pi data?></p></body></html>")?;
    for version in ["4.0", "5"] {
        let written = serialize(&value, &html(version).with_indent(true))?;
        assert!(
            written.contains("<p><?pi data></p>"),
            "{version}: {written}"
        );
    }
    let as_xml = document("<a><?pi data?></a>")?;
    assert_eq!(
        serialize(
            &as_xml,
            &SerializationParameters::default().with_indent(true)
        )?,
        "<a><?pi data?></a>"
    );
    Ok(())
}

/// Serialization 3.1 §5.1.1: within an element carrying `xml:space="preserve"` the serializer
/// may neither add nor remove whitespace, however element-only the content is — and
/// `xml:space="default"` turns the rule off again (QT3 `method-xml/K2-Serialization-40`, `-41`).
#[test]
fn xml_space_preserve_suppresses_indentation() -> Result<(), Box<dyn std::error::Error>> {
    let indented = SerializationParameters::default().with_indent(true);
    let value = document("<test><a xml:space='preserve'><x/></a><b/></test>")?;
    assert_eq!(
        serialize(&value, &indented)?,
        "<test>\n  <a xml:space=\"preserve\"><x/></a>\n  <b/>\n</test>"
    );

    // The rule is inherited by the descendants, and revoked by `xml:space="default"`.
    let nested = document(
        "<test><a xml:space='preserve'><x><y/></x><z xml:space='default'><w/></z></a></test>",
    )?;
    let written = serialize(&nested, &indented)?;
    assert!(written.contains("<x><y/></x>"), "{written}");
    assert!(written.contains("<z xml:space=\"default\">\n"), "{written}");
    Ok(())
}

/// `suppress-indentation` names the elements inside which indentation is not added, and the rule
/// reaches everything they contain.
#[test]
fn suppress_indentation_names_elements() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a><keep><b><c/></b></keep><d/></a>")?;
    let parameters = SerializationParameters::default()
        .with_indent(true)
        .with_suppress_indentation(vec![ExpandedName::new(None::<&str>, "keep")]);
    assert_eq!(
        serialize(&value, &parameters)?,
        "<a>\n  <keep><b><c/></b></keep>\n  <d/>\n</a>"
    );
    // Without it the same tree is indented throughout.
    assert_eq!(
        serialize(
            &value,
            &SerializationParameters::default().with_indent(true)
        )?,
        "<a>\n  <keep>\n    <b>\n      <c/>\n    </b>\n  </keep>\n  <d/>\n</a>"
    );
    Ok(())
}

/// Serialization 3.1 §7: HTML element names are case-insensitive, so the `html` method matches
/// the names `suppress-indentation` lists without regard to case — for a foreign vocabulary
/// carried inside the page as much as for HTML's own (QT3 `Serialization-html-56`, `-57`).
#[test]
fn the_html_method_matches_element_names_case_blind() -> Result<(), Box<dyn std::error::Error>> {
    let table = document(
        "<html><body><table><tr><td><p>One</p></td><td><p>yes</p></td></tr></table></body></html>",
    )?;
    let parameters = html("5.0")
        .with_indent(true)
        .with_suppress_indentation(vec![ExpandedName::new(None::<&str>, "TABLE")]);
    let written = serialize(&table, &parameters)?;
    assert!(
        written.contains("<table><tr><td><p>One</p></td><td>"),
        "{written}"
    );
    assert!(written.contains("<body>\n    <table>"), "{written}");
    // The `xml` method compares exactly, so the same value suppresses nothing.
    let as_xml = SerializationParameters::default()
        .with_indent(true)
        .with_suppress_indentation(vec![ExpandedName::new(None::<&str>, "TABLE")]);
    assert!(serialize(&table, &as_xml)?.contains("<table>\n"), "exact");

    // A foreign vocabulary inside the page is matched the same way, namespace and all.
    let mathml = "http://www.w3.org/1998/Math/MathML";
    let math = document(&format!(
        "<html><body><div>x=<math xmlns='{mathml}'><mfrac><mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow></mfrac></math>=y</div></body></html>"
    ))?;
    let suppressed = html("5.0")
        .with_indent(true)
        .with_suppress_indentation(vec![ExpandedName::new(Some(mathml), "MROW")]);
    let written = serialize(&math, &suppressed)?;
    assert!(
        written.contains("<mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow>"),
        "{written}"
    );
    Ok(())
}

/// Serialization 3.1 §2: an item no output method can write is `SENR0001`.
#[test]
fn function_items_cannot_be_serialized_as_xml() {
    let parameters = SerializationParameters::default();
    let map = Sequence::singleton(Item::from(XdmMap::new()));
    let error = serialize(&map, &parameters).expect_err("a map is not XML");
    assert!(error.is(FnErrorCode::Senr0001));
    assert!(error.to_string().contains("method=\"xml\""), "{error}");

    // The message names the method actually in force, not always `xml`.
    let as_html = parameters.with_method(OutputMethod::Html);
    assert!(
        serialize(&map, &as_html)
            .expect_err("a map is not HTML")
            .to_string()
            .contains("method=\"html\"")
    );
}

#[test]
fn attribute_nodes_cannot_be_serialized_as_xml() -> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str("<a b='1'/>")?;
    let element = tree.document_element().ok_or("no root")?;
    let attribute = element
        .attributes()
        .first()
        .cloned()
        .ok_or("no attribute")?;
    assert!(
        serialize(
            &Sequence::from(attribute),
            &SerializationParameters::default()
        )
        .is_err()
    );
    Ok(())
}

/// The `xml`, `html` and `adaptive` methods all descend the value recursively, and the
/// value is the caller's, so the descent runs out of a budget rather than out of stack.
#[test]
fn a_value_nested_past_the_ceiling_is_xpdy0130() -> Result<(), Box<dyn std::error::Error>> {
    let deep = document("<a><b><c><d/></c></b></a>")?;
    let shallow = SerializationParameters::default().with_max_depth(2);
    let error = serialize(&deep, &shallow).expect_err("past the ceiling");
    assert!(error.is(FnErrorCode::Xpdy0130));
    assert!(error.to_string().contains('2'));
    // Two levels still serialize, so the ceiling is a guard and not a ban.
    let shallow_enough = document("<a><b/></a>")?;
    assert_eq!(serialize(&shallow_enough, &shallow)?, "<a><b/></a>");

    let mut nested = Sequence::from(AtomicValue::integer(1_i64));
    for _ in 0..8 {
        nested = Sequence::singleton(Item::Array(XdmArray::from_members([nested])));
    }
    let adaptive = SerializationParameters::default()
        .with_method(OutputMethod::Adaptive)
        .with_max_depth(4);
    assert!(
        serialize(&nested, &adaptive)
            .expect_err("past the ceiling")
            .is(FnErrorCode::Xpdy0130)
    );

    // The `xml` method flattens an array rather than refusing it, and that descent is on the
    // same budget.
    let flattening = SerializationParameters::default().with_max_depth(4);
    assert!(
        serialize(&nested, &flattening)
            .expect_err("past the ceiling")
            .is(FnErrorCode::Xpdy0130)
    );

    // The `json` method writes the value through a writer of its own, which used to keep
    // its own default ceiling and ignore the one set here.
    let json = SerializationParameters::default()
        .with_method(OutputMethod::Json)
        .with_max_depth(4);
    assert!(
        serialize(&nested, &json)
            .expect_err("past the ceiling")
            .is(FnErrorCode::Sere0023)
    );
    let json_deep_enough = json.with_max_depth(64);
    assert_eq!(serialize(&nested, &json_deep_enough)?, "[[[[[[[[1]]]]]]]]");
    Ok(())
}

/// A CDATA section cannot carry `]]>`, and a mapped character has to leave the section so
/// that its replacement is not escaped.
#[test]
fn a_cdata_section_is_split_where_it_has_to_be() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<a>x]]&gt;y</a>")?;
    let names = vec![ExpandedName::new(None::<&str>, "a")];
    let parameters = SerializationParameters::default().with_cdata_section_elements(names.clone());
    assert_eq!(
        serialize(&value, &parameters)?,
        "<a><![CDATA[x]]]]><![CDATA[>y]]></a>"
    );

    let mapped = parameters.with_character_map(vec![('y', "<br/>".to_owned())]);
    assert_eq!(
        serialize(&value, &mapped)?,
        "<a><![CDATA[x]]]]><![CDATA[>]]><br/></a>"
    );
    Ok(())
}

/// Serialization 3.1 §7.1.4: HTML has no CDATA sections, so `cdata-section-elements` reaches
/// only the elements the `html` method is not writing as HTML — the XML islands. Which elements
/// those are moves with the HTML version, because the XHTML namespace joins the vocabulary at
/// HTML 5 (QT3 `Serialization-html-18`, `-19a`, `-19b`, `-19c`).
#[test]
fn cdata_sections_do_not_reach_html_elements() -> Result<(), Box<dyn std::error::Error>> {
    let islands = document(
        "<html><body><p>No CDATA<em>No CDATA</em>No CDATA<ex:isle1 xmlns:ex='http://www.example.org'>Some CDATA<ex:isle2>No CDATA</ex:isle2>More CDATA</ex:isle1></p></body></html>",
    )?;
    let names = vec![
        ExpandedName::new(None::<&str>, "p"),
        ExpandedName::new(None::<&str>, "em"),
        ExpandedName::new(Some("http://www.example.org"), "isle1"),
    ];
    for version in ["4.0", "5.0"] {
        let written = serialize(
            &islands,
            &html(version).with_cdata_section_elements(names.clone()),
        )?;
        assert!(
            written.contains(
                "<p>No CDATA<em>No CDATA</em>No CDATA<ex:isle1 xmlns:ex=\"http://www.example.org\"><![CDATA[Some CDATA]]><ex:isle2>No CDATA</ex:isle2><![CDATA[More CDATA]]></ex:isle1></p>"
            ),
            "{version}: {written}"
        );
    }

    // An element in the XHTML namespace is an XML island under HTML 4 and an HTML element under
    // HTML 5, and `cdata-section-elements` follows that.
    let xhtml = document(
        "<html><body><p><b>No CDATA</b><em>No CDATA</em><html:b xmlns:html='http://www.w3.org/1999/xhtml'>No CDATA</html:b><html:em xmlns:html='http://www.w3.org/1999/xhtml'>Some CDATA</html:em></p></body></html>",
    )?;
    let mixed = vec![
        ExpandedName::new(None::<&str>, "b"),
        ExpandedName::new(Some("http://www.w3.org/1999/xhtml"), "em"),
    ];
    let four = serialize(
        &xhtml,
        &html("4.0").with_cdata_section_elements(mixed.clone()),
    )?;
    assert!(
        four.contains(
            "<p><b>No CDATA</b><em>No CDATA</em><html:b xmlns:html=\"http://www.w3.org/1999/xhtml\">No CDATA</html:b><html:em xmlns:html=\"http://www.w3.org/1999/xhtml\"><![CDATA[Some CDATA]]></html:em></p>"
        ),
        "{four}"
    );
    let five = serialize(&xhtml, &html("5.0").with_cdata_section_elements(mixed))?;
    assert!(!five.contains("CDATA["), "{five}");
    assert!(
        five.contains("<b xmlns=\"http://www.w3.org/1999/xhtml\">No CDATA</b>"),
        "{five}"
    );
    assert!(
        five.contains(
            "<em xmlns=\"http://www.w3.org/1999/xhtml\">Some CDATA</em></p></body></html>"
        ),
        "{five}"
    );

    // The `xml` method knows no HTML vocabulary, so every named element is wrapped.
    let as_xml = serialize(
        &islands,
        &SerializationParameters::default().with_cdata_section_elements(names),
    )?;
    assert!(as_xml.contains("<em><![CDATA[No CDATA]]></em>"), "{as_xml}");
    Ok(())
}

/// Serialization 3.1 §7.1: an element the `html` method writes as an HTML element carries no
/// prefix, so an element in the XHTML namespace is written under a default namespace declaration
/// instead (QT3 `Serialization-html-19c`).
#[test]
fn an_xhtml_element_is_written_without_a_prefix() -> Result<(), Box<dyn std::error::Error>> {
    let value = document(
        "<html><body><p><html:b xmlns:html='http://www.w3.org/1999/xhtml'>bold</html:b></p></body></html>",
    )?;
    let five = serialize(&value, &html("5.0"))?;
    assert!(
        five.contains("<p><b xmlns=\"http://www.w3.org/1999/xhtml\">bold</b></p>"),
        "{five}"
    );
    // Before HTML 5 the namespace is foreign, so the prefix stays.
    let four = serialize(&value, &html("4.0"))?;
    assert!(
        four.contains("<html:b xmlns:html=\"http://www.w3.org/1999/xhtml\">bold</html:b>"),
        "{four}"
    );
    // An XHTML document that already declares the default namespace gains no second declaration.
    let already = document("<html xmlns='http://www.w3.org/1999/xhtml'><body><p/></body></html>")?;
    let written = serialize(&already, &html("5.0"))?;
    assert!(
        written
            .contains("<html xmlns=\"http://www.w3.org/1999/xhtml\"><body><p></p></body></html>"),
        "{written}"
    );
    Ok(())
}

/// Serialization 3.1 §5.1.1: NEL, LINE SEPARATOR and the C1 controls are written as
/// character references in both text nodes and attribute values, because writing them
/// literally would not survive a round trip (QT3 `method-xml/K2-Serialization-5`, `-6`,
/// `-9`, `-10`).
#[test]
fn control_characters_are_written_as_character_references() -> Result<(), Box<dyn std::error::Error>>
{
    let text = document("<a>&#xD;&#x85;&#x2028;</a>")?;
    assert_eq!(
        serialize(&text, &SerializationParameters::default())?,
        "<a>&#xD;&#x85;&#x2028;</a>"
    );

    let attribute = document("<a attr='&#xD;&#xA;&#x9;&#x85;&#x2028;'/>")?;
    assert_eq!(
        serialize(&attribute, &SerializationParameters::default())?,
        "<a attr=\"&#xD;&#xA;&#x9;&#x85;&#x2028;\"/>"
    );

    // The whole C1 range, in a text node and in an attribute value.
    let range: String = (0x7F..=0x9F_u32)
        .map(|code| format!("&#x{code:X};"))
        .collect();
    let expected: String = (0x7F..=0x9F_u32)
        .map(|code| format!("&#x{code:X};"))
        .collect();
    let c1_text = document(&format!("<a>{range}</a>"))?;
    assert_eq!(
        serialize(&c1_text, &SerializationParameters::default())?,
        format!("<a>{expected}</a>")
    );
    let c1_attribute = document(&format!("<a attr='{range}'/>"))?;
    assert_eq!(
        serialize(&c1_attribute, &SerializationParameters::default())?,
        format!("<a attr=\"{expected}\"/>")
    );

    // The `text` method escapes nothing at all (QT3 `method-text/Serialization-text-5`).
    let as_text = SerializationParameters::default().with_method(OutputMethod::Text);
    assert_eq!(serialize(&text, &as_text)?, "\r\u{85}\u{2028}");
    Ok(())
}

/// Serialization 3.1 §7.1: the `html` method writes an element the HTML vocabulary knows to
/// be empty with a start tag alone, and gives every other element an explicit end tag (QT3
/// `Serialization-html-1`, `-2`, `-7`, `-8`, `-23`, `-24`, `-45`).
#[test]
fn the_html_method_never_self_closes_an_element() -> Result<(), Box<dyn std::error::Error>> {
    let empties = document(
        "<html><body><area/><base/><br/><col/><embed/><frame/><hr/><img/><input/><isindex/><link/><meta/><param/></body></html>",
    )?;
    assert_eq!(
        serialize(&empties, &html("4.0"))?,
        "<html><body><area><base><br><col><embed><frame><hr><img><input><isindex><link><meta><param></body></html>"
    );

    let five = document(
        "<html><body><area/><base/><br/><col/><embed/><hr/><img/><input/><keygen/><link/><meta/><param/><source/><track/><wbr/></body></html>",
    )?;
    assert!(
        serialize(&five, &html("5.0"))?.contains(
            "<html><body><area><base><br><col><embed><hr><img><input><keygen><link><meta><param><source><track><wbr></body></html>"
        )
    );

    // An element the HTML vocabulary does not know keeps its end tag.
    let unknown = document("<html><body><spin/></body></html>")?;
    assert!(serialize(&unknown, &html("5.0"))?.contains("<body><spin></spin></body>"));
    assert_eq!(
        serialize(&document("<foo/>")?, &html("4.0"))?,
        "<foo></foo>"
    );

    // From HTML 5 the XHTML namespace is HTML too; before that it is ordinary XML.
    let xhtml =
        document("<html xmlns='http://www.w3.org/1999/xhtml'><body><spin/><br/></body></html>")?;
    assert!(serialize(&xhtml, &html("5.0"))?.contains("<body><spin></spin><br></body>"));
    assert!(serialize(&xhtml, &html("4.0"))?.contains("<body><spin/><br/></body>"));

    // An element in a foreign namespace is an XML island under either version.
    let island = document(
        "<html xmlns='http://www.w3.org/1999/xhtml'><body><magic xmlns='http://example.org/magic'/></body></html>",
    )?;
    for version in ["4.0", "5.0"] {
        assert!(
            serialize(&island, &html(version))?
                .contains("<magic xmlns=\"http://example.org/magic\"/>"),
            "{version}"
        );
    }
    Ok(())
}

/// Serialization 3.1 §7.3: a boolean attribute — one whose value is its own name, compared
/// without regard to case — is minimized to the name alone (QT3 `Serialization-html-12`,
/// `-13`).
#[test]
fn html_boolean_attributes_are_minimized() -> Result<(), Box<dyn std::error::Error>> {
    let value = document(
        "<html><body><div><option selected='SELECTED'/><option selected='UNSELECTED'/></div></body></html>",
    )?;
    assert!(
        serialize(&value, &html("4.0"))?
            .contains("<option selected></option><option selected=\"UNSELECTED\"></option>")
    );
    // The `xml` method minimizes nothing.
    assert!(
        serialize(&value, &SerializationParameters::default())?
            .contains("<option selected=\"SELECTED\"/>")
    );
    Ok(())
}

/// Serialization 3.1 §7.3: an `&` immediately followed by `{` is left alone in an HTML
/// attribute value (QT3 `Serialization-html-11`).
#[test]
fn an_ampersand_before_a_brace_is_not_escaped_in_html() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html><body><p class='&amp;{entspannend}'>exciting!</p></body></html>")?;
    assert!(serialize(&value, &html("5.0"))?.contains("<p class=\"&{entspannend}\">exciting!</p>"));
    // Anywhere else the ampersand is still escaped, and the `xml` method escapes it always.
    let alone = document("<html><body><p class='a&amp;b'/></body></html>")?;
    assert!(serialize(&alone, &html("5.0"))?.contains("class=\"a&amp;b\""));
    assert!(
        serialize(&value, &SerializationParameters::default())?
            .contains("class=\"&amp;{entspannend}\"")
    );
    Ok(())
}

/// Serialization 3.1 §7.5: `script` and `style` hold CDATA in HTML's own grammar, so nothing
/// inside them is escaped — the markup they contain included, whose attribute values reach an
/// HTML reader as script text — while the `script` element's *own* attributes are escaped like
/// any others (QT3 `Serialization-html-9`, `-10`).
#[test]
fn script_and_style_markup_is_not_escaped() -> Result<(), Box<dyn std::error::Error>> {
    let value = document(
        "<html xmlns='http://www.w3.org/1999/xhtml'><head><script language='Jack&amp;Jill'>go &amp;&amp; write('<p class=\"Bill&amp;Ben\">Jekyll&amp;Hyde</p>');</script></head><body><p>Hi!</p></body></html>",
    )?;
    let written = serialize(&value, &html("5.0"))?;
    assert!(written.contains("\"Jack&amp;Jill\""), "{written}");
    assert!(written.contains("\"Bill&Ben\""), "{written}");
    assert!(written.contains("go && write"), "{written}");
    assert!(written.contains("Jekyll&Hyde"), "{written}");
    // The element name is matched without regard to case.
    let upper = document(
        "<HTML xmlns='http://www.w3.org/1999/xhtml'><head><SCRIPT language='Jack&amp;Jill'>go &amp;&amp; write('<p class=\"Bill&amp;Ben\">Jekyll&amp;Hyde</p>');</SCRIPT></head><body><p>Hi!</p></body></HTML>",
    )?;
    let upper_written = serialize(&upper, &html("5.0"))?;
    assert!(
        upper_written.contains("\"Jack&amp;Jill\""),
        "{upper_written}"
    );
    assert!(upper_written.contains("\"Bill&Ben\""), "{upper_written}");
    assert!(upper_written.contains("go && write"), "{upper_written}");
    assert!(upper_written.contains("Jekyll&Hyde"), "{upper_written}");
    // The `xml` method escapes it, as it escapes everything.
    assert!(
        serialize(&value, &SerializationParameters::default())?.contains("go &amp;&amp; write")
    );
    Ok(())
}

/// Serialization 3.1 §7.5: with `escape-uri-attributes` in force the `html` method
/// percent-encodes the non-ASCII characters of a URI-valued attribute, and leaves everything
/// else — a space included — exactly as it stands (QT3 `Serialization-html-43`, `-43a`, `-44`,
/// `-44a`).
#[test]
fn uri_attributes_are_percent_encoded() -> Result<(), Box<dyn std::error::Error>> {
    let value = document(
        "<html><head/><body><a href='file:///c:/My Documents/temp/b&#xe9;b&#xe9;.xml'>Click here baby!</a></body></html>",
    )?;
    for version in ["4.0", "5.0"] {
        let escaped = serialize(&value, &html(version))?;
        assert!(
            escaped.contains("My Documents/temp/b%C3%A9b%C3%A9.xml"),
            "{version}: {escaped}"
        );
        let raw = serialize(&value, &html(version).with_escape_uri_attributes(false))?;
        assert!(
            raw.contains("My Documents/temp/b\u{e9}b\u{e9}.xml"),
            "{version}: {raw}"
        );
    }
    // An attribute that is not URI-valued keeps its characters, and so does every attribute
    // under the `xml` method.
    let title = document("<html><body><a title='b&#xe9;b&#xe9;'/></body></html>")?;
    assert!(serialize(&title, &html("5.0"))?.contains("title=\"b\u{e9}b\u{e9}\""));
    assert!(
        serialize(&value, &SerializationParameters::default())?.contains("temp/b\u{e9}b\u{e9}.xml")
    );
    Ok(())
}

/// Serialization 3.1 §7.4: `include-content-type` puts a `meta` element declaring the media
/// type and encoding first inside `head`, replacing one the source already declared, and its
/// value of `no` suppresses it (QT3 `Serialization-html-33`, `-34`, `-36`, `-37`, `-39`,
/// `-40`).
#[test]
fn the_content_type_meta_follows_include_content_type() -> Result<(), Box<dyn std::error::Error>> {
    let empty_head = document("<html><head/></html>")?;
    assert_eq!(
        serialize(&empty_head, &html("4.0"))?,
        "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></head></html>"
    );
    assert_eq!(
        serialize(&empty_head, &html("4.0").with_include_content_type(false))?,
        "<html><head></head></html>"
    );

    // A `meta` the source already declared for the content type is replaced, not joined.
    let declared = document(
        "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html;version='3.0'\"/></head></html>",
    )?;
    assert_eq!(
        serialize(&declared, &html("4.0"))?,
        "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></head></html>"
    );
    // Another `meta` is left where it stands.
    let other = document("<html><head><meta name=\"author\"/></head></html>")?;
    assert!(serialize(&other, &html("4.0"))?.contains("<meta name=\"author\">"));
    // The `xml` method adds nothing at all.
    assert_eq!(
        serialize(&empty_head, &SerializationParameters::default())?,
        "<html><head/></html>"
    );
    Ok(())
}

/// Serialization 3.1 §7.6: the `html` method closes a processing instruction with `>` alone
/// (QT3 `Serialization-html-48`, `-58`, `-59`).
#[test]
fn html_processing_instructions_end_with_a_single_bracket() -> Result<(), Box<dyn std::error::Error>>
{
    let value = document("<html><body><p><?pi data?><?bare?></p></body></html>")?;
    assert!(serialize(&value, &html("4.0"))?.contains("<p><?pi data><?bare></p>"));
    assert!(
        serialize(&value, &SerializationParameters::default())?
            .contains("<p><?pi data?><?bare?></p>")
    );
    Ok(())
}

/// Serialization 3.1 §7.2: the HTML 5 document type declaration belongs to HTML 5 alone, and
/// the `version` parameter identifies the HTML version when `html-version` is absent (QT3
/// `Serialization-html-1` against `-2`, `-20` against `-22`).
#[test]
fn the_html5_doctype_follows_the_effective_version() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html/>")?;
    assert_eq!(
        serialize(&value, &html("5.0"))?,
        "<!DOCTYPE html>\n<html></html>"
    );
    assert_eq!(serialize(&value, &html("4.0"))?, "<html></html>");
    // With neither parameter set the default HTML version is 5.
    let bare = SerializationParameters::default().with_method(OutputMethod::Html);
    assert_eq!(serialize(&value, &bare)?, "<!DOCTYPE html>\n<html></html>");
    Ok(())
}

/// Serialization 3.1 §5.1.1: with `undeclare-prefixes` and XML 1.1 the serializer writes the
/// namespace *undeclaration* `xmlns:p=""` for every prefix an ancestor declared that the element
/// no longer has in scope — and writes none at all for XML 1.0, which has no such declaration
/// (QT3 `method-xml/K2-Serialization-20` against `-21`).
#[test]
fn undeclare_prefixes_writes_xml_11_undeclarations() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = TreeBuilder::new();
    builder
        .start_element(ExpandedName::new(Some("http://example.com/p"), "chapter"))?
        .declare_namespace(Some("p"), "http://example.com/p")?
        .start_element(ExpandedName::new(None::<&str>, "section"))?
        .declare_namespace(Some("p"), "")?
        .start_element(ExpandedName::new(None::<&str>, "para"))?
        .end_element()?
        .end_element()?
        .end_element()?;
    let tree = builder.build()?;
    let value = Sequence::from(tree.document_element().ok_or("no document element")?);

    let undeclaring = SerializationParameters::default()
        .with_version("1.1")
        .with_undeclare_prefixes(true);
    let written = serialize(&value, &undeclaring)?;
    assert!(written.contains("<section xmlns:p=\"\">"), "{written}");
    // The undeclaration is not repeated on the descendants it already covers.
    assert!(written.contains("<para/>"), "{written}");

    // Without the parameter, and under XML 1.0, nothing is undeclared.
    for parameters in [
        SerializationParameters::default().with_version("1.1"),
        SerializationParameters::default().with_undeclare_prefixes(true),
    ] {
        let written = serialize(&value, &parameters)?;
        assert!(!written.contains("xmlns:p=\"\""), "{written}");
    }

    // `K2-Serialization-21`'s other half: `undeclare-prefixes="no"` still writes an XML
    // declaration naming version 1.1, which is what makes the case's `not(…)` decidable.
    let declaring = SerializationParameters::serialization_defaults()
        .with_version("1.1")
        .with_undeclare_prefixes(false);
    let written = serialize(&value, &declaring)?;
    assert!(written.starts_with("<?xml version=\"1.1\""), "{written}");
    assert!(!written.contains("xmlns:p=\"\""), "{written}");
    Ok(())
}

/// The `xhtml` output method's own parameter set: `html-version` is the only spelling that names
/// the HTML version, because under this method `version` is the XML version.
fn xhtml(version: &str) -> SerializationParameters {
    SerializationParameters::default()
        .with_method(OutputMethod::Xhtml)
        .with_html_version(version)
}

/// Serialization 3.1 §8: the `xhtml` method minimizes an element of the XHTML vocabulary whose
/// content model is empty to `<br />` — self-closed, with the space an HTML reader needs before
/// the slash — and the vocabulary is the effective HTML version's own, not the union the `html`
/// method works from (QT3 `Serialization-xhtml-1` and `-1a`, `-2`, `-4`).
#[test]
fn xhtml_minimizes_the_empty_elements_of_its_effective_version()
-> Result<(), Box<dyn std::error::Error>> {
    let html4 = "<area/><base/><br/><col/><embed/><frame/><hr/><img/><input/><isindex/><link/>\
                 <meta/><param/>";
    // An element in the XHTML namespace is an XHTML element at every version…
    let namespaced = document(&format!(
        "<html xmlns=\"http://www.w3.org/1999/xhtml\"><body>{html4}</body></html>"
    ))?;
    assert_eq!(
        serialize(&namespaced, &xhtml("4.0"))?,
        "<html xmlns=\"http://www.w3.org/1999/xhtml\"><body><area /><base /><br /><col />\
         <embed /><frame /><hr /><img /><input /><isindex /><link /><meta /><param />\
         </body></html>"
    );
    // …but `frame` and `isindex` left HTML 5, so at 5.0 they are ordinary elements again and
    // carry an explicit end tag.
    assert_eq!(
        serialize(&namespaced, &xhtml("5.0"))?,
        "<!DOCTYPE html>\n<html xmlns=\"http://www.w3.org/1999/xhtml\"><body><area /><base />\
         <br /><col /><embed /><frame></frame><hr /><img /><input /><isindex></isindex>\
         <link /><meta /><param /></body></html>"
    );

    // An element in *no* namespace is an XHTML element from HTML 5 onwards only.
    let bare = document(&format!("<html><body>{html4}</body></html>"))?;
    assert_eq!(
        serialize(&bare, &xhtml("4.0"))?,
        "<html><body><area></area><base></base><br></br><col></col><embed></embed>\
         <frame></frame><hr></hr><img></img><input></input><isindex></isindex><link></link>\
         <meta></meta><param></param></body></html>"
    );
    let html5 = document(
        "<html><body><area/><base/><br/><col/><embed/><hr/><img/><input/><keygen/><link/>\
         <meta/><param/><source/><track/><wbr/></body></html>",
    )?;
    assert_eq!(
        serialize(&html5, &xhtml("5.0"))?,
        "<!DOCTYPE html>\n<html><body><area /><base /><br /><col /><embed /><hr /><img />\
         <input /><keygen /><link /><meta /><param /><source /><track /><wbr /></body></html>"
    );
    Ok(())
}

/// Serialization 3.1 §8: every element the XHTML vocabulary does *not* know to be empty keeps an
/// explicit end tag, because `<div/>` is not something an HTML reader recognizes — whatever
/// namespace the element is in (QT3 `Serialization-xhtml-20`, `-23`, `-24`, `-51`).
#[test]
fn xhtml_never_self_closes_an_element_outside_the_vocabulary()
-> Result<(), Box<dyn std::error::Error>> {
    for version in ["4.0", "5.0"] {
        assert_eq!(
            serialize(&document("<foo/>")?, &xhtml(version))?,
            "<foo></foo>",
            "{version}"
        );
    }
    // `html` is in neither vocabulary, so it is written out even at HTML 5.
    assert_eq!(
        serialize(&document("<html/>")?, &xhtml("5.0"))?,
        "<!DOCTYPE html>\n<html></html>"
    );
    // The `xml` method still self-closes, which is what makes this the `xhtml` method's rule.
    assert_eq!(
        serialize(&document("<foo/>")?, &SerializationParameters::default())?,
        "<foo/>"
    );
    Ok(())
}

/// Serialization 3.1 §8.2: the `xhtml` method writes the HTML 5 document type declaration when
/// the effective version is 5 and no `doctype-system` is given — a `doctype-public` on its own
/// does not displace it, because §5.1.4's XML rule ignores a public identifier without a system
/// one (QT3 `Serialization-xhtml-22`, `-25`…`-30`, against `Serialization-html-27`).
#[test]
fn the_xhtml_doctype_follows_the_xml_rule_and_html5() -> Result<(), Box<dyn std::error::Error>> {
    let value = document("<html/>")?;
    let public = "-//W3C//DTD HTML 4.0//EN".to_owned();
    let system = "http://www.w3.org/TR/REC-html40/strict.dtd".to_owned();

    // `doctype-public` alone: no declaration at 4.0, the HTML 5 one at 5.0.
    assert_eq!(
        serialize(
            &value,
            &xhtml("4.0").with_doctype_public(Some(public.clone()))
        )?,
        "<html></html>"
    );
    assert_eq!(
        serialize(
            &value,
            &xhtml("5.0").with_doctype_public(Some(public.clone()))
        )?,
        "<!DOCTYPE html>\n<html></html>"
    );
    // `doctype-system` is the XML rule at either version, and displaces the HTML 5 declaration.
    assert_eq!(
        serialize(
            &value,
            &xhtml("4.0").with_doctype_system(Some(system.clone()))
        )?,
        format!("<!DOCTYPE html SYSTEM \"{system}\"><html></html>")
    );
    assert_eq!(
        serialize(
            &value,
            &xhtml("5.0").with_doctype_system(Some("about:legacy-compat".to_owned()))
        )?,
        "<!DOCTYPE html SYSTEM \"about:legacy-compat\"><html></html>"
    );
    for version in ["4.0", "5.0"] {
        assert_eq!(
            serialize(
                &value,
                &xhtml(version)
                    .with_doctype_public(Some(public.clone()))
                    .with_doctype_system(Some(system.clone()))
            )?,
            format!("<!DOCTYPE html PUBLIC \"{public}\" \"{system}\"><html></html>"),
            "{version}"
        );
    }
    // Only a document whose outermost element is `html` gets the HTML 5 declaration.
    assert_eq!(
        serialize(&document("<foo/>")?, &xhtml("5.0"))?,
        "<foo></foo>"
    );
    Ok(())
}

/// Serialization 3.1 §8: the three vocabularies an HTML 5 document holds without a prefix —
/// XHTML, SVG and MathML — are written under a default namespace declaration however the tree
/// spells them, and the prefix that bound them is not carried through (QT3
/// `Serialization-xhtml-50`, `-51`, `-52`).
#[test]
fn xhtml_normalizes_the_prefixes_of_the_html5_vocabularies()
-> Result<(), Box<dyn std::error::Error>> {
    let page = document(
        "<h:html xmlns:h=\"http://www.w3.org/1999/xhtml\"><h:head><h:title>Test doc</h:title>\
         </h:head><h:body><h:h1>Header</h:h1><h:p>Content</h:p></h:body></h:html>",
    )?;
    assert_eq!(
        serialize(&page, &xhtml("5.0"))?,
        "<!DOCTYPE html>\n<html xmlns=\"http://www.w3.org/1999/xhtml\"><head>\
         <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />\
         <title>Test doc</title></head><body><h1>Header</h1><p>Content</p></body></html>"
    );

    // SVG and MathML are embedded the same way, and the XHTML prefix does not follow them in.
    let svg = document(
        "<h:html xmlns:h=\"http://www.w3.org/1999/xhtml\"><h:body><svg:svg \
         xmlns:svg=\"http://www.w3.org/2000/svg\"><svg:rectangle/></svg:svg></h:body></h:html>",
    )?;
    assert!(
        serialize(&svg, &xhtml("5.0"))?
            .contains("<svg xmlns=\"http://www.w3.org/2000/svg\"><rectangle></rectangle></svg>"),
        "{}",
        serialize(&svg, &xhtml("5.0"))?
    );
    let math = document(
        "<h:html xmlns:h=\"http://www.w3.org/1999/xhtml\"><h:body><h:p>\
         <m:math xmlns:m=\"http://www.w3.org/1998/Math/MathML\"><m:mrow><m:mi>a</m:mi></m:mrow>\
         </m:math></h:p></h:body></h:html>",
    )?;
    assert!(
        serialize(&math, &xhtml("5.0"))?.contains(
            "<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>a</mi></mrow></math>"
        ),
        "{}",
        serialize(&math, &xhtml("5.0"))?
    );

    // An *attribute* has no default namespace to fall back on, so a prefix one of its own still
    // needs is kept.
    let attribute = document(
        "<h:html xmlns:h=\"http://www.w3.org/1999/xhtml\"><h:body h:class=\"x\"/></h:html>",
    )?;
    assert!(
        serialize(&attribute, &xhtml("5.0"))?
            .contains("<body xmlns:h=\"http://www.w3.org/1999/xhtml\" h:class=\"x\""),
        "{}",
        serialize(&attribute, &xhtml("5.0"))?
    );
    Ok(())
}

/// Serialization 3.1 §7.4: the `xhtml` method adds the same content-type `meta` to `head` as the
/// `html` method — but written as XML, closed by ` />` with no separate end tag, which QT3
/// `Serialization-xhtml-34` and `-37` pin by allowing at most one `meta` in the result
/// (`Serialization-xhtml-33`, `-35`, `-39`).
#[test]
fn the_xhtml_content_type_meta_closes_itself() -> Result<(), Box<dyn std::error::Error>> {
    let namespaced = document("<html xmlns=\"http://www.w3.org/1999/xhtml\"><head/></html>")?;
    assert_eq!(
        serialize(&namespaced, &xhtml("4.0"))?,
        "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head>\
         <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" /></head></html>"
    );
    // The one the source carries is replaced rather than joined.
    let declared = document(
        "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head>\
         <meta http-equiv=\"Content-Type\" content=\"text/html;version='3.0'\"/></head></html>",
    )?;
    let written = serialize(&declared, &xhtml("4.0"))?;
    assert_eq!(written.matches("<meta").count(), 1, "{written}");
    assert!(written.contains("charset=UTF-8\" />"), "{written}");

    // `include-content-type="no"` adds nothing, and neither does a document with no `head`.
    let bare = document("<html><head/></html>")?;
    assert_eq!(
        serialize(&bare, &xhtml("4.0").with_include_content_type(false))?,
        "<html><head></head></html>"
    );
    assert_eq!(
        serialize(&document("<html/>")?, &xhtml("4.0"))?,
        "<html></html>"
    );
    // At 4.0 an element in no namespace is not an XHTML element, so its `head` is not one either.
    assert_eq!(
        serialize(&bare, &xhtml("4.0"))?,
        "<html><head></head></html>"
    );
    assert!(serialize(&bare, &xhtml("5.0"))?.contains("<head><meta "));
    Ok(())
}

/// Serialization 3.1 §8: everything §8 does not name is the `xml` method's — the XML declaration,
/// the `?>` of a processing instruction, XML escaping inside `script`, an attribute whose value
/// repeats its name, and CDATA sections, which reach an XHTML element as they reach any other
/// (QT3 `Serialization-xhtml-19a`, `-19b`, `-58`).
#[test]
fn xhtml_keeps_the_xml_methods_syntax() -> Result<(), Box<dyn std::error::Error>> {
    // The XML declaration is written unless it is omitted, where the `html` method never writes
    // one at all.
    let declaring = SerializationParameters::serialization_defaults()
        .with_method(OutputMethod::Xhtml)
        .with_html_version("4.0");
    assert_eq!(
        serialize(&document("<foo/>")?, &declaring)?,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><foo></foo>"
    );
    // A processing instruction keeps its `?>`, and `script` content is escaped.
    let page = document(
        "<html><body><p><?pi data?></p><script>if (a &lt; b) x()</script>\
         <input checked=\"checked\"/></body></html>",
    )?;
    let written = serialize(&page, &xhtml("5.0"))?;
    assert!(written.contains("<p><?pi data?></p>"), "{written}");
    assert!(
        written.contains("<script>if (a &lt; b) x()</script>"),
        "{written}"
    );
    // A boolean attribute is not minimized: XHTML has no such form.
    assert!(
        written.contains("<input checked=\"checked\" />"),
        "{written}"
    );

    // `cdata-section-elements` reaches an element the method recognizes as XHTML, which is what
    // separates §8 from §7.1.4 — HTML has no CDATA sections, XHTML has XML's.
    let mixed = document(
        "<html xmlns:h=\"http://www.w3.org/1999/xhtml\"><body><p><b>YES</b><em>NO</em>\
         <h:b>NO</h:b><h:em>YES</h:em></p></body></html>",
    )?;
    let parameters = xhtml("4.0").with_cdata_section_elements(vec![
        ExpandedName::new(None::<&str>, "b"),
        ExpandedName::new(Some("http://www.w3.org/1999/xhtml"), "em"),
    ]);
    let written = serialize(&mixed, &parameters)?;
    assert!(written.contains("<b><![CDATA[YES]]></b>"), "{written}");
    assert!(written.contains("<em>NO</em>"), "{written}");
    assert!(
        written.contains("<em xmlns=\"http://www.w3.org/1999/xhtml\"><![CDATA[YES]]></em>"),
        "{written}"
    );
    assert!(!written.contains(">YES<"), "{written}");
    Ok(())
}

/// Serialization 3.1 §8: indentation follows the same content model as the `xml` method's —
/// mixed content is left alone — and `suppress-indentation` names elements by expanded name
/// (QT3 `Serialization-xhtml-57`, `-59`, `-62`, `-65`, `-67`).
#[test]
fn xhtml_indentation_follows_the_content_model() -> Result<(), Box<dyn std::error::Error>> {
    let page = document(
        "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head/><body><p>a<a>text</a>z</p>\
         </body></html>",
    )?;
    let written = serialize(&page, &xhtml("4.0").with_indent(true))?;
    assert!(written.contains("<p>a<a>text</a>z</p>"), "{written}");
    assert!(written.contains("\n  <head>"), "{written}");

    // An XHTML element name is an XML name, so `suppress-indentation` matches it as it stands.
    let list =
        document("<html><body><ul><li><p>One</p></li><li><p>Two</p></li></ul></body></html>")?;
    let suppressed = xhtml("5.0")
        .with_indent(true)
        .with_suppress_indentation(vec![ExpandedName::new(None::<&str>, "li")]);
    let written = serialize(&list, &suppressed)?;
    assert!(written.contains("<li><p>One</p></li>"), "{written}");
    assert!(written.contains("<li><p>Two</p></li>"), "{written}");
    assert!(written.contains("<ul>\n"), "{written}");
    Ok(())
}

/// XML 1.1 admits the C0 controls, but only written as character references, so a serializer
/// asked for `version="1.1"` writes them that way and one asked for `version="1.0"` cannot write
/// them at all.
///
/// The XSLT 3.0 suite pins both halves: `xml-version-007`…`-010` and `-018` put `#x8` and `#x1F`
/// in an attribute under `version="1.1"` and match against the references, and `xml-version-029`
/// and `-030` copy an XML 1.1 document to a `version="1.0"` result and want `SERE0006`.
#[test]
fn the_c0_controls_are_references_under_xml_11_and_an_error_under_1_0()
-> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str("<?xml version=\"1.1\"?><out><a status=\"&#x8;&#x1f;\"/></out>")?;
    let value = Sequence::from(tree.document_element().ok_or("no document element")?);
    let eleven = SerializationParameters::default().with_version("1.1");
    assert_eq!(
        serialize(&value, &eleven)?,
        "<out><a status=\"&#x8;&#x1F;\"/></out>"
    );
    let ten = SerializationParameters::default();
    assert!(
        serialize(&value, &ten)
            .expect_err("XML 1.0 has no spelling for these characters")
            .is(FnErrorCode::Sere0006)
    );

    // In a text node the rule is the same, and a CDATA section — which has no way to spell a
    // reference — is closed around the character rather than swallowing it.
    let bell = Tree::parse_str("<?xml version=\"1.1\"?><bell>a&#7;b</bell>")?;
    let value = Sequence::from(bell.document_element().ok_or("no document element")?);
    assert_eq!(serialize(&value, &eleven)?, "<bell>a&#x7;b</bell>");
    let cdata = eleven
        .clone()
        .with_cdata_section_elements(vec![ExpandedName::new(None::<String>, "bell")]);
    assert_eq!(
        serialize(&value, &cdata)?,
        "<bell><![CDATA[a]]>&#x7;<![CDATA[b]]></bell>"
    );

    // The C1 controls were already written as references whatever the version says, and the
    // three white-space characters XML 1.0 admits are untouched by the rule.
    let plain = Tree::parse_str("<a b=\"x\">m\tn\u{85}</a>")?;
    let value = Sequence::from(plain.document_element().ok_or("no document element")?);
    assert_eq!(serialize(&value, &ten)?, "<a b=\"x\">m\tn&#x85;</a>");
    Ok(())
}

/// The `version` parameter names an XML version only for the methods whose output is XML: under
/// `method="html"` the same parameter is the *HTML* version, so it cannot turn on an XML rule.
#[test]
fn the_html_method_reads_version_as_a_version_of_html() -> Result<(), Box<dyn std::error::Error>> {
    let tree = Tree::parse_str("<?xml version=\"1.1\"?><p>a&#7;b</p>")?;
    let value = Sequence::from(tree.document_element().ok_or("no document element")?);
    assert_eq!(serialize(&value, &html("1.1"))?, "<p>a\u{7}b</p>");
    Ok(())
}
