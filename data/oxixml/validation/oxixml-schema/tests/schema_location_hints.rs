//! Instance-driven schema loading — `xsi:schemaLocation` and
//! `xsi:noNamespaceSchemaLocation`, [§4.3.2][hints].
//!
//! Each test reproduces a W3C XML Schema Test Suite group whose instance names
//! a schema document the assembled schema does not contain: the element or
//! attribute it needs is declared *there*, and without following the hint the
//! instance is rejected for want of a declaration. The fixtures are written to
//! a scratch directory under [`std::env::temp_dir`], because the whole point of
//! the mechanism is that the locations are resolved against the instance's own.
//!
//! [hints]: https://www.w3.org/TR/xmlschema-1/#xsi_schemaLocation

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_schema::{Schema, SchemaSet, ValidationOutcome};

/// A scratch directory holding one group's schema documents and instances.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-schema-location-hints-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch directory");
        Self { dir }
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.dir.join(name);
        fs::write(&path, text).expect("a written fixture");
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn schema_from(path: &Path) -> Schema {
    let mut set = SchemaSet::new();
    set.add_file(path).expect("a schema document");
    set.compile().expect("a compiled schema")
}

fn first_error(outcome: &ValidationOutcome) -> String {
    outcome
        .first_error()
        .map(ToString::to_string)
        .unwrap_or_default()
}

/// `msData/schema/schA1`: a `##any` strict wildcard admits elements from a
/// namespace the instance points at with `xsi:schemaLocation` and from no
/// namespace at all, which it points at with `xsi:noNamespaceSchemaLocation`.
#[test]
fn sch_a1_a_strict_wildcard_is_satisfied_by_a_hinted_declaration() {
    let fixture = Fixture::new("schA1");
    let schema_path = fixture.write(
        "schA1_a.xsd",
        r###"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ns-a" elementFormDefault="qualified" xmlns="ns-a">
             <xsd:complexType name="ct-A">
               <xsd:sequence minOccurs="1">
                 <xsd:element name="a1" type="xsd:string"/>
                 <xsd:element name="a2" type="xsd:string"/>
               </xsd:sequence>
             </xsd:complexType>
             <xsd:element name="e1" type="ct-A"/>
             <xsd:element name="root">
               <xsd:complexType>
                 <xsd:choice maxOccurs="unbounded"><xsd:any namespace="##any"/></xsd:choice>
               </xsd:complexType>
             </xsd:element>
           </xsd:schema>"###,
    );
    fixture.write(
        "schA1_b.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ns-b" xmlns:b="ns-b" elementFormDefault="unqualified">
             <xsd:complexType name="ct-A">
               <xsd:sequence minOccurs="1">
                 <xsd:element name="a1" type="xsd:boolean"/>
                 <xsd:element name="a2" type="xsd:boolean"/>
               </xsd:sequence>
             </xsd:complexType>
             <xsd:element name="e1" type="b:ct-A"/>
           </xsd:schema>"#,
    );
    fixture.write(
        "schA1_c.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:complexType name="ct-A">
               <xsd:sequence minOccurs="1">
                 <xsd:element name="a1" type="xsd:int"/>
                 <xsd:element name="a2" type="xsd:int"/>
               </xsd:sequence>
             </xsd:complexType>
             <xsd:element name="e1" type="ct-A"/>
           </xsd:schema>"#,
    );
    let instance = fixture.write(
        "schA1.xml",
        r#"<a:root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xsi:schemaLocation="ns-b schA1_b.xsd"
                   xsi:noNamespaceSchemaLocation="schA1_c.xsd"
                   xmlns:a="ns-a" xmlns:b="ns-b">
             <a:e1><a:a1>abc</a:a1><a:a2>abc</a:a2></a:e1>
             <b:e1><a1>0</a1><a2>true</a2></b:e1>
             <e1><a1>123</a1><a2>123</a2></e1>
           </a:root>"#,
    );
    let schema = schema_from(&schema_path);
    let outcome = schema.validate_file(&instance);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));

    // The hint is followed only when the instance's own location is known, so
    // the same document assessed without one is still short a declaration.
    let text = fs::read_to_string(&instance).expect("the instance");
    let blind = schema.validate_str(&text);
    assert_eq!(
        blind
            .first_error()
            .map(oxixml_schema::ValidationError::code),
        Some("cvc-complex-type.2.4.c")
    );
}

/// `msData/particles/particlesB013`: a strict `xs:any` over an element the
/// instance's `xsi:schemaLocation` declares.
#[test]
fn particles_b013_a_hinted_namespace_answers_a_strict_any() {
    let fixture = Fixture::new("particlesB013");
    let schema_path = fixture.write(
        "particlesB013.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="doc">
               <xsd:complexType>
                 <xsd:choice><xsd:element name="elem" type="elem"/></xsd:choice>
               </xsd:complexType>
             </xsd:element>
             <xsd:complexType name="elem">
               <xsd:choice><xsd:any maxOccurs="2"/></xsd:choice>
             </xsd:complexType>
           </xsd:schema>"#,
    );
    fixture.write(
        "particlesB013_1.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" targetNamespace="foo">
             <xsd:element name="b"/>
           </xsd:schema>"#,
    );
    let instance = fixture.write(
        "particlesB013.xml",
        r#"<doc xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:noNamespaceSchemaLocation="particlesB013.xsd"
                xsi:schemaLocation="foo particlesB013_1.xsd">
             <elem><b xmlns="foo"/></elem>
           </doc>"#,
    );
    let outcome = schema_from(&schema_path).validate_file(&instance);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));
}

/// `msData/attributeGroup/attgD034`: an `##other` strict *attribute* wildcard
/// admits an attribute the instance's hint declares.
#[test]
fn attg_d034_a_strict_attribute_wildcard_is_satisfied_by_a_hinted_declaration() {
    let fixture = Fixture::new("attgD034");
    let schema_path = fixture.write(
        "attgD034.xsd",
        r###"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:x="http://xsdtesting"
                       targetNamespace="http://xsdtesting" elementFormDefault="qualified"
                       attributeFormDefault="qualified">
             <xsd:element name="doc">
               <xsd:complexType>
                 <xsd:choice><xsd:element name="elem" type="x:attRef"/></xsd:choice>
               </xsd:complexType>
             </xsd:element>
             <xsd:complexType name="attRef"><xsd:attributeGroup ref="x:attG1"/></xsd:complexType>
             <xsd:attributeGroup name="attG1">
               <xsd:attribute ref="x:foo"/>
               <xsd:anyAttribute namespace="##other" processContents="strict"/>
             </xsd:attributeGroup>
             <xsd:attribute name="foo" type="xsd:int"/>
           </xsd:schema>"###,
    );
    fixture.write(
        "attgD034.inc",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" targetNamespace="http://foo"
                       attributeFormDefault="qualified">
             <xsd:attribute name="att" type="xsd:string"/>
           </xsd:schema>"#,
    );
    let instance = fixture.write(
        "attgD034.xml",
        r#"<x:doc xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xmlns:x="http://xsdtesting" xmlns:foo="http://foo"
                  xsi:schemaLocation="http://xsdtesting attgD034.xsd http://foo attgD034.inc">
             <x:elem foo:att="123"/>
           </x:doc>"#,
    );
    let outcome = schema_from(&schema_path).validate_file(&instance);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));
}

/// `msData/complexType/ctL021`, whose hint list names the schema document the
/// assessment already holds *and* the one declaring the wildcard's attribute:
/// re-naming a document already in hand is not an error.
#[test]
fn ct_l021_a_hint_may_repeat_the_schema_already_in_hand() {
    let fixture = Fixture::new("ctL021");
    let schema_path = fixture.write(
        "ctL021.xsd",
        r###"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="root">
               <xsd:complexType>
                 <xsd:sequence><xsd:element ref="fooTest"/></xsd:sequence>
               </xsd:complexType>
             </xsd:element>
             <xsd:element name="fooTest" type="fooType"/>
             <xsd:complexType name="fooType">
               <xsd:sequence>
                 <xsd:element name="child_1" type="xsd:string"/>
                 <xsd:element name="child_2" type="xsd:integer"/>
               </xsd:sequence>
               <xsd:anyAttribute namespace="##other" processContents="strict"/>
             </xsd:complexType>
           </xsd:schema>"###,
    );
    fixture.write(
        "ctL021b.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" targetNamespace="myNamespace">
             <xsd:attribute name="testAttr" type="xsd:string"/>
           </xsd:schema>"#,
    );
    let instance = fixture.write(
        "ctL021.xml",
        r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:noNamespaceSchemaLocation="ctL021.xsd"
                 xsi:schemaLocation="myNamespace ctL021b.xsd" xmlns:myNS="myNamespace">
             <fooTest myNS:testAttr="Test Attribute">
               <child_1>info</child_1><child_2>3</child_2>
             </fooTest>
           </root>"#,
    );
    let outcome = schema_from(&schema_path).validate_file(&instance);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));
}

/// `sunData/ElemDecl/targetNS/targetNS00101m`: the *document element* itself is
/// declared only in a hinted schema document, so the hand-over happens at the
/// validation root.
#[test]
fn target_ns_00101m_the_document_element_may_be_declared_by_a_hint() {
    let fixture = Fixture::new("targetNS00101m");
    let schema_path = fixture.write(
        "targetNS00101m.xsd",
        r#"<xsd:schema xmlns="ElemDecl/targetNS" xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ElemDecl/targetNS">
             <xsd:element name="number" type="xsd:decimal"/>
           </xsd:schema>"#,
    );
    fixture.write(
        "targetNS00101m1a.xsd",
        r#"<xsd:schema xmlns="ElemDecl/targetNSa" xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ElemDecl/targetNSa">
             <xsd:element name="number">
               <xsd:simpleType>
                 <xsd:restriction base="xsd:string"><xsd:pattern value="[IVXCL]*"/></xsd:restriction>
               </xsd:simpleType>
             </xsd:element>
           </xsd:schema>"#,
    );
    let hints = r#"xsi:schemaLocation="ElemDecl/targetNS targetNS00101m.xsd
                                       ElemDecl/targetNSa targetNS00101m1a.xsd""#;
    let valid = fixture.write(
        "targetNS00101m1_p.xml",
        &format!(
            r#"<testa:number xmlns:testa="ElemDecl/targetNSa"
                             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                             {hints}>XII</testa:number>"#
        ),
    );
    let broken = fixture.write(
        "targetNS00101m1_n.xml",
        &format!(
            r#"<testa:number xmlns:testa="ElemDecl/targetNSa"
                             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                             {hints}>12</testa:number>"#
        ),
    );
    let schema = schema_from(&schema_path);
    let outcome = schema.validate_file(&valid);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));
    // The hinted declaration governs the instance rather than merely excusing
    // it: a value its pattern refuses is still invalid.
    assert!(!schema.validate_file(&broken).valid);
    // …and the declaration this schema holds for the *other* namespace is
    // untouched by the hand-over.
    let own = fixture.write(
        "targetNS00101m1_own.xml",
        &format!(
            r#"<test:number xmlns:test="ElemDecl/targetNS"
                            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                            {hints}>1.5</test:number>"#
        ),
    );
    assert!(schema.validate_file(&own).valid);
}

/// `sunData/SType/ST_targetNS/ST_targetNS00101m`: the document element has no
/// declaration anywhere, and `xsi:type` names a simple type that lives in the
/// assembled schema for one instance and in a hinted document for the next.
/// §4.3.2's precedence is what keeps the two apart.
#[test]
fn st_target_ns_00101m_an_xsi_type_may_be_defined_by_a_hint() {
    let fixture = Fixture::new("ST_targetNS00101m");
    let schema_path = fixture.write(
        "ST_targetNS00101m.xsd",
        r#"<xsd:schema xmlns="ST_targetNS" xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ST_targetNS">
             <xsd:simpleType name="Test">
               <xsd:restriction base="xsd:string"><xsd:pattern value="1|2"/></xsd:restriction>
             </xsd:simpleType>
           </xsd:schema>"#,
    );
    fixture.write(
        "ST_targetNS00101ma.xsd",
        r#"<xsd:schema xmlns="ST_targetNSa" xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="ST_targetNSa">
             <xsd:simpleType name="Test">
               <xsd:restriction base="xsd:string"><xsd:pattern value="3|4"/></xsd:restriction>
             </xsd:simpleType>
           </xsd:schema>"#,
    );
    let schema = schema_from(&schema_path);
    let instance = |name: &str, namespace: &str, content: &str| {
        fixture.write(
            name,
            &format!(
                r#"<test xmlns="{namespace}"
                         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                         xsi:schemaLocation="ST_targetNS ST_targetNS00101m.xsd
                                             ST_targetNSa ST_targetNS00101ma.xsd"
                         xsi:type="Test">{content}</test>"#
            ),
        )
    };
    // The type this schema already defines wins, and its pattern is `1|2`.
    let own_valid = instance("m1_p.xml", "ST_targetNS", "1");
    let own_broken = instance("m1_n.xml", "ST_targetNS", "3");
    assert!(schema.validate_file(&own_valid).valid);
    assert!(!schema.validate_file(&own_broken).valid);
    // The hinted one governs the namespace this schema says nothing about, and
    // its pattern is `3|4`.
    let hinted_valid = instance("m2_p.xml", "ST_targetNSa", "3");
    let hinted_broken = instance("m2_n.xml", "ST_targetNSa", "2");
    let outcome = schema.validate_file(&hinted_valid);
    assert!(outcome.valid, "unexpected {}", first_error(&outcome));
    assert!(!schema.validate_file(&hinted_broken).valid);
}

/// A hint that names nothing retrievable is not an error in itself: §4.3.2
/// leaves the whole mechanism to the processor's discretion, so the assessment
/// proceeds and reports what the instance actually breaks.
#[test]
fn an_unretrievable_hint_is_not_fatal() {
    let fixture = Fixture::new("missing-hint");
    let schema_path = fixture.write(
        "root.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="root" type="xsd:int"/>
           </xsd:schema>"#,
    );
    let schema = schema_from(&schema_path);
    let good = fixture.write(
        "good.xml",
        r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:schemaLocation="urn:absent nowhere.xsd
                                     urn:remote http://example.invalid/remote.xsd">7</root>"#,
    );
    assert!(schema.validate_file(&good).valid);
    let bad = fixture.write(
        "bad.xml",
        r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:noNamespaceSchemaLocation="nowhere.xsd">seven</root>"#,
    );
    // The instance is still assessed, and what it breaks is its own content —
    // not the hint that led nowhere.
    let outcome = schema.validate_file(&bad);
    assert!(!outcome.valid);
    assert_ne!(
        outcome
            .first_error()
            .map(oxixml_schema::ValidationError::code),
        Some("cvc-document")
    );
}

/// A file that is not there, or is not XML, is reported like any other
/// assessment failure rather than by panicking.
#[test]
fn an_unreadable_instance_is_reported_not_panicked() {
    let fixture = Fixture::new("unreadable");
    let schema_path = fixture.write(
        "root.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="root" type="xsd:int"/>
           </xsd:schema>"#,
    );
    let schema = schema_from(&schema_path);
    let outcome = schema.validate_file(fixture.dir.join("absent.xml"));
    assert_eq!(
        outcome
            .first_error()
            .map(oxixml_schema::ValidationError::code),
        Some("cvc-document")
    );
    let malformed = fixture.write("malformed.xml", "<root>7");
    assert_eq!(
        schema
            .validate_file(&malformed)
            .first_error()
            .map(oxixml_schema::ValidationError::code),
        Some("cvc-document")
    );
}
