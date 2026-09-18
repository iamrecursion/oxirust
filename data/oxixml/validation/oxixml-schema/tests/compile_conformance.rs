//! Schema-representation and derivation constraints checked during assembly.
//!
//! Every test here is named after the group of the [W3C XML Schema Test
//! Suite][xsts] it reproduces, and carries the schema document of that group
//! inline so that the rule is checked wherever the suite itself is not
//! available. Groups whose expected validity differs between XML Schema 1.0
//! and 1.1 are asserted under both.
//!
//! [xsts]: https://www.w3.org/XML/2004/xml-schema-test-suite/

use std::collections::HashMap;

use oxixml_schema::{ResolvedSchema, SchemaError, SchemaResolver, SchemaSet, XsdVersion};

/// Serves schema documents from a map, so that `xs:import`, `xs:include` and
/// `xs:redefine` are exercised without touching the file system.
#[derive(Debug, Default)]
struct MemoryResolver {
    documents: HashMap<String, String>,
}

impl MemoryResolver {
    fn with(mut self, location: &str, text: &str) -> Self {
        self.documents.insert(location.to_owned(), text.to_owned());
        self
    }
}

impl SchemaResolver for MemoryResolver {
    fn resolve(
        &mut self,
        location: &str,
        _base: Option<&str>,
    ) -> Result<Option<ResolvedSchema>, SchemaError> {
        Ok(self.documents.get(location).map(|text| ResolvedSchema {
            uri: location.to_owned(),
            text: text.clone(),
        }))
    }
}

/// Assembles one document under `version`.
fn assemble(version: XsdVersion, text: &str) -> Result<(), SchemaError> {
    let mut set = SchemaSet::new().with_version(version);
    set.add_document(None, text)?;
    set.compile().map(|_| ())
}

/// Assembles one document under `version`, with `resolver` serving whatever it
/// refers to.
fn assemble_with(
    version: XsdVersion,
    resolver: MemoryResolver,
    text: &str,
) -> Result<(), SchemaError> {
    let mut set = SchemaSet::new()
        .with_version(version)
        .with_resolver(Box::new(resolver));
    set.add_document(None, text)?;
    set.compile().map(|_| ())
}

/// Asserts that both versions reject `text`, and returns the 1.0 error code.
#[track_caller]
fn rejected(what: &str, text: &str) -> String {
    let mut codes = Vec::new();
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        match assemble(version, text) {
            Ok(()) => panic!("{what}: assembled under {version:?}, but it is not a valid schema"),
            Err(error) => codes.push(error.code().to_owned()),
        }
    }
    codes.swap_remove(0)
}

/// Asserts that XML Schema 1.1 rejects `text`, and returns the error code.
///
/// XSD 1.0 usually rejects the same document as well, for want of the 1.1
/// construct it is written with, and its code says so rather than naming the
/// rule under test.
#[track_caller]
fn rejected_by_1_1(what: &str, text: &str) -> String {
    match assemble(XsdVersion::V1_1, text) {
        Ok(()) => panic!("{what}: assembled under XML Schema 1.1, but it is not a valid schema"),
        Err(error) => error.code().to_owned(),
    }
}

/// Asserts that both versions accept `text`.
#[track_caller]
fn accepted(what: &str, text: &str) {
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        if let Err(error) = assemble(version, text) {
            panic!("{what}: rejected under {version:?}: {error}");
        }
    }
}

/// Asserts that XML Schema 1.1 accepts `text`.
///
/// The 1.0 verdict is not asserted: the constructs these groups are about —
/// a wildcard inside an `xs:all` group above all — are 1.1 additions that 1.0
/// rejects for a reason that has nothing to do with the rule under test.
#[track_caller]
fn accepted_by_1_1(what: &str, text: &str) {
    if let Err(error) = assemble(XsdVersion::V1_1, text) {
        panic!("{what}: rejected under XML Schema 1.1: {error}");
    }
}

/// Asserts that 1.0 accepts `text` and 1.1 rejects it.
#[track_caller]
fn accepted_by_1_0_only(what: &str, text: &str) {
    if let Err(error) = assemble(XsdVersion::V1_0, text) {
        panic!("{what}: rejected under XML Schema 1.0: {error}");
    }
    assert!(
        assemble(XsdVersion::V1_1, text).is_err(),
        "{what}: accepted under XML Schema 1.1"
    );
}

/// Assembles `schema` under `version` and reports whether `instance` is valid
/// against it.
///
/// Some component-level rules — the attribute wildcard a complex type ends up
/// with, above all — are only observable through an instance, so the group
/// they belong to is asserted the way the suite states it.
#[track_caller]
fn validates(version: XsdVersion, schema: &str, instance: &str) -> bool {
    let mut set = SchemaSet::new().with_version(version);
    set.add_document(None, schema)
        .expect("the schema document is well-formed");
    let compiled = set.compile().expect("the schema assembles");
    let document = oxixml_dom::build::parse_str(instance).expect("the instance is well-formed");
    compiled.validate_document(&document).valid
}

// ------------------------------------------------------- namespace names

/// `msData/schema/schZ014_a`, `schZ014_b`: the empty string is not a namespace
/// name, so neither `targetNamespace` nor an `xs:import` may write it.
#[test]
fn sch_z014_empty_namespace_names_are_rejected() {
    assert_eq!(
        rejected(
            "schZ014_a",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:import namespace=""/>
               </xsd:schema>"#,
        ),
        "src-import"
    );
    assert_eq!(
        rejected(
            "schZ014_b",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" targetNamespace=""/>"#,
        ),
        "src-schema"
    );
    accepted(
        "a target namespace that names something",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x"/>"#,
    );
}

// ------------------------------------------------------------- notations

/// `msData/notations/notatB001` and `sunData/Notation/publicId00201m1`: XSD 1.0
/// Errata E1-16 made `public` optional, leaving the requirement that a notation
/// name at least one identifier.
#[test]
fn notat_b001_notation_needs_an_identifier() {
    assert_eq!(
        rejected(
            "notatB001",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:notation name="foo"/>
               </xsd:schema>"#,
        ),
        "src-notation"
    );
    // `msData/errata10/errC004`: a `system` alone is enough.
    accepted(
        "errC004",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:notation name="testNotation" system="http://example.invalid/system"/>
           </xs:schema>"#,
    );
}

/// `msData/notations/notatG001`, `notatG003`: every element of the schema
/// vocabulary has element-only content. `notatG005` is the control — whitespace
/// alone is not character content.
#[test]
fn notat_g001_character_content_is_rejected() {
    assert_eq!(
        rejected(
            "notatG001",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:notation name="jpeg" public="image/jpeg" system="viewer.exe">Some Text</xsd:notation>
               </xsd:schema>"#,
        ),
        "src-element"
    );
    rejected(
        "notatG003",
        "<xsd:schema xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">\
           <xsd:notation name=\"jpeg\" public=\"image/jpeg\" system=\"viewer.exe\">\n&amp; \" '\n</xsd:notation>\
         </xsd:schema>",
    );
    accepted(
        "notatG005",
        "<xsd:schema xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">\
           <xsd:notation name=\"jpeg\" public=\"image/jpeg\" system=\"viewer.exe\">\n\t</xsd:notation>\
         </xsd:schema>",
    );
}

/// `msData/annotations/annotF001`, `annotF003`: the `xml:lang` of an
/// `xs:documentation` is an `xs:language`, and neither the empty string nor a
/// blank one is one.
#[test]
fn annot_f001_documentation_language_is_checked() {
    for (name, language) in [("annotF001", ""), ("annotF003", " ")] {
        assert_eq!(
            rejected(
                name,
                &format!(
                    r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                         <xsd:annotation>
                           <xsd:documentation source="http://example.invalid/" xml:lang="{language}"/>
                         </xsd:annotation>
                       </xsd:schema>"#
                ),
            ),
            "src-attribute-value"
        );
    }
    accepted(
        "a documentation that names its language",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:annotation>
               <xsd:documentation xml:lang="en-GB">hello</xsd:documentation>
             </xsd:annotation>
           </xsd:schema>"#,
    );
}

// ------------------------------------------------------------ attributes

/// `msData/attribute/attKb018a`: §3.2.6 *no-xsi* holds for a local declaration
/// that `attributeFormDefault` puts into the XSI namespace, exactly as it does
/// for a top-level one.
#[test]
fn att_kb018a_local_attributes_may_not_be_in_the_xsi_namespace() {
    assert_eq!(
        rejected(
            "attKb018a",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                          targetNamespace="http://www.w3.org/2001/XMLSchema-instance"
                          xmlns:x="http://www.w3.org/2001/XMLSchema-instance"
                          attributeFormDefault="qualified">
                 <xsd:attributeGroup name="attG">
                   <xsd:attribute name="aga1"/>
                 </xsd:attributeGroup>
                 <xsd:complexType name="attRef">
                   <xsd:attributeGroup ref="x:attG"/>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "no-xsi"
    );
}

/// `msData/attribute/attKb009`, `attKc009`, `attP029`: an attribute use that
/// prohibits the attribute has no value to constrain, which XSD 1.1 makes an
/// error and XSD 1.0 tolerated (W3C bug 14245). The three groups differ only in
/// where the prohibited use is written.
#[test]
fn att_p029_prohibited_uses_may_not_be_fixed() {
    accepted_by_1_0_only(
        "attKb009",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
             <xsd:attributeGroup name="attG">
               <xsd:attribute name="aga1" use="prohibited" fixed="abc"/>
             </xsd:attributeGroup>
             <xsd:complexType name="attRef">
               <xsd:attributeGroup ref="x:attG"/>
             </xsd:complexType>
           </xsd:schema>"#,
    );
    accepted_by_1_0_only(
        "attKc009",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
             <xsd:complexType name="attRef">
               <xsd:attribute name="ca1" use="prohibited" fixed="abc"/>
             </xsd:complexType>
           </xsd:schema>"#,
    );
    accepted_by_1_0_only(
        "attP029",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting"
                      attributeFormDefault="qualified">
             <xsd:complexType name="attRef">
               <xsd:attribute name="att" use="prohibited" fixed="37"/>
             </xsd:complexType>
           </xsd:schema>"#,
    );
    accepted(
        "a prohibited use with no value constraint",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:complexType name="attRef">
               <xsd:attribute name="att" use="prohibited"/>
             </xsd:complexType>
           </xsd:schema>"#,
    );
}

/// `msData/errata10/errC009`: XSD 1.0 Errata E1-21 — a restriction may not
/// weaken the `processContents` of the base type's attribute wildcard.
#[test]
fn err_c009_attribute_wildcard_keeps_process_contents() {
    assert_eq!(
        rejected(
            "errC009",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="base">
                   <xs:sequence><xs:element name="foo"/></xs:sequence>
                   <xs:anyAttribute processContents="strict"/>
                 </xs:complexType>
                 <xs:complexType name="derived">
                   <xs:complexContent>
                     <xs:restriction base="base">
                       <xs:sequence><xs:element name="foo"/></xs:sequence>
                       <xs:anyAttribute processContents="skip"/>
                     </xs:restriction>
                   </xs:complexContent>
                 </xs:complexType>
               </xs:schema>"#,
        ),
        "derivation-ok-restriction.4.3"
    );
    accepted(
        "a restriction that keeps the base's `processContents`",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="base">
               <xs:sequence><xs:element name="foo"/></xs:sequence>
               <xs:anyAttribute processContents="skip"/>
             </xs:complexType>
             <xs:complexType name="derived">
               <xs:complexContent>
                 <xs:restriction base="base">
                   <xs:sequence><xs:element name="foo"/></xs:sequence>
                   <xs:anyAttribute processContents="strict"/>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    );
}

// --------------------------------------------- content type and particles

/// `msData/particles/particlesZ039`: §3.4.6 clause 5.2 — a restriction with
/// empty content needs a base whose content is empty too, or mixed and
/// ·emptiable·. Simple content is neither.
#[test]
fn particles_z039_empty_content_may_not_restrict_simple_content() {
    assert_eq!(
        rejected(
            "particlesZ039",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="cTypeBase">
                   <xs:simpleContent>
                     <xs:extension base="xs:string">
                       <xs:attribute name="attr1" type="xs:string"/>
                     </xs:extension>
                   </xs:simpleContent>
                 </xs:complexType>
                 <xs:complexType name="cTypeDerived">
                   <xs:complexContent>
                     <xs:restriction base="cTypeBase"><xs:sequence/></xs:restriction>
                   </xs:complexContent>
                 </xs:complexType>
               </xs:schema>"#,
        ),
        "derivation-ok-restriction.5.2.2"
    );
}

/// `msData/particles/particlesIj008`: *rcase-NameAndTypeOK* clause 7 blocks
/// `extension`, so a restriction may not give the element a type that reaches
/// the base's by extending it.
#[test]
fn particles_ij008_element_types_must_derive_by_restriction() {
    assert_eq!(
        rejected(
            "particlesIj008",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                          targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
                 <xsd:complexType name="foo">
                   <xsd:choice><xsd:element name="f1"/></xsd:choice>
                 </xsd:complexType>
                 <xsd:complexType name="bar">
                   <xsd:complexContent>
                     <xsd:extension base="x:foo">
                       <xsd:choice><xsd:element name="f3"/></xsd:choice>
                     </xsd:extension>
                   </xsd:complexContent>
                 </xsd:complexType>
                 <xsd:complexType name="B">
                   <xsd:choice><xsd:element name="c1" type="x:foo"/></xsd:choice>
                 </xsd:complexType>
                 <xsd:complexType name="R">
                   <xsd:complexContent>
                     <xsd:restriction base="x:B">
                       <xsd:choice><xsd:element name="c1" type="x:bar"/></xsd:choice>
                     </xsd:restriction>
                   </xsd:complexContent>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "rcase-NameAndTypeOK.7"
    );
    // `msData/additional/addB150` is the control: §3.16.6.3 clause 2.2.4 makes a
    // member of an unconstrained union a valid restriction of the union itself.
    accepted(
        "addB150",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="A" mixed="true">
               <xs:sequence>
                 <xs:element name="A" type="deci-string" maxOccurs="unbounded"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="B">
               <xs:complexContent>
                 <xs:restriction base="A">
                   <xs:sequence><xs:element name="A" type="xs:string" maxOccurs="10"/></xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
             <xs:simpleType name="deci-string">
               <xs:union memberTypes="xs:decimal xs:string"/>
             </xs:simpleType>
           </xs:schema>"#,
    );
}

/// `msData/particles/particlesQ012`: the *effective total range* of §3.8.6 sums
/// the ranges of the group particles inside a group, not merely their own
/// occurrence ranges, so the restriction below can match four elements where the
/// base wildcard allows three.
#[test]
fn particles_q012_effective_total_range_recurses() {
    // The wrapper `sequence` is one particle of an enclosing `sequence`, so the
    // failure surfaces through `Recurse`, which is where the pair was compared.
    assert_eq!(
        rejected(
            "particlesQ012",
            r###"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                          targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
                 <xsd:complexType name="B">
                   <xsd:sequence>
                     <xsd:element name="foo"/>
                     <xsd:any namespace="##any" minOccurs="0" maxOccurs="3"/>
                   </xsd:sequence>
                 </xsd:complexType>
                 <xsd:complexType name="R">
                   <xsd:complexContent>
                     <xsd:restriction base="x:B">
                       <xsd:sequence>
                         <xsd:element name="foo"/>
                         <xsd:sequence minOccurs="0" maxOccurs="1">
                           <xsd:group ref="x:g1"/>
                           <xsd:group ref="x:g2"/>
                         </xsd:sequence>
                       </xsd:sequence>
                     </xsd:restriction>
                   </xsd:complexContent>
                 </xsd:complexType>
                 <xsd:group name="g1">
                   <xsd:choice><xsd:element name="e1" maxOccurs="2"/></xsd:choice>
                 </xsd:group>
                 <xsd:group name="g2">
                   <xsd:choice><xsd:element name="e2" maxOccurs="2"/></xsd:choice>
                 </xsd:group>
               </xsd:schema>"###,
        ),
        "rcase-Recurse.2"
    );
}

/// `msData/particles/particlesZ018`: §3.4.6 clause 5.1.1 — the simple type an
/// `xs:simpleContent` restriction writes must be derived from the base's.
#[test]
fn particles_z018_simple_content_restriction_derives_from_the_base() {
    assert_eq!(
        rejected(
            "particlesZ018",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="B1">
                   <xs:simpleContent>
                     <xs:extension base="xs:decimal"><xs:attribute name="foo"/></xs:extension>
                   </xs:simpleContent>
                 </xs:complexType>
                 <xs:complexType name="C2">
                   <xs:simpleContent>
                     <xs:restriction base="B1">
                       <xs:simpleType><xs:list itemType="xs:int"/></xs:simpleType>
                     </xs:restriction>
                   </xs:simpleContent>
                 </xs:complexType>
               </xs:schema>"#,
        ),
        "derivation-ok-restriction.5.1.1"
    );
    accepted(
        "a simple content restriction that narrows the base's type",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="B1">
               <xs:simpleContent>
                 <xs:extension base="xs:decimal"><xs:attribute name="foo"/></xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="C2">
               <xs:simpleContent>
                 <xs:restriction base="B1">
                   <xs:simpleType>
                     <xs:restriction base="xs:decimal"><xs:totalDigits value="3"/></xs:restriction>
                   </xs:simpleType>
                 </xs:restriction>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#,
    );
}

/// `msData/additional/addB033`, `addB036`: *cos-ct-extends* clause 1.4.3.2 — an
/// extension that brings a particle needs a base whose content is empty, mixed
/// or element-only, and simple content is none of those.
#[test]
fn add_b033_extension_may_not_add_elements_to_simple_content() {
    assert_eq!(
        rejected(
            "addB033",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:complexType name="theSimple">
                   <xsd:simpleContent>
                     <xsd:extension base="xsd:decimal">
                       <xsd:attribute name="theAttribute" type="xsd:string"/>
                     </xsd:extension>
                   </xsd:simpleContent>
                 </xsd:complexType>
                 <xsd:complexType name="weird">
                   <xsd:complexContent>
                     <xsd:extension base="theSimple">
                       <xsd:sequence>
                         <xsd:element name="theElement" type="xsd:string"/>
                       </xsd:sequence>
                     </xsd:extension>
                   </xsd:complexContent>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "cos-ct-extends.1.4.3.2"
    );
}

/// `msData/particles/particlesZ031`: extending a simple-content base through
/// `xs:complexContent` is legal in XSD 1.0, which hands the base's content type
/// through unchanged, and an error in XSD 1.1 (W3C bug 15629).
#[test]
fn particles_z031_complex_content_over_simple_content_is_1_0_only() {
    accepted_by_1_0_only(
        "particlesZ031",
        r#"<xs:schema xmlns="urn:x" targetNamespace="urn:x"
                     xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="Type1">
               <xs:simpleContent>
                 <xs:extension base="xs:string">
                   <xs:attribute name="Field1" type="xs:string"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="Type2">
               <xs:complexContent>
                 <xs:extension base="Type1">
                   <xs:attribute name="Field2" type="xs:string"/>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    );
}

// ------------------------------------------------------- `xs:anySimpleType`

/// `msData/additional/addB110`: the ·simple ur-type definition· has no
/// {variety} for a restriction to inherit.
#[test]
fn add_b110_any_simple_type_is_not_a_restriction_base() {
    assert_eq!(
        rejected(
            "addB110",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:simpleType name="ct2">
                   <xs:restriction base="xs:anySimpleType"/>
                 </xs:simpleType>
               </xs:schema>"#,
        ),
        "cos-st-restricts"
    );
    accepted(
        "an element typed `xs:anySimpleType`",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e2" type="xs:anySimpleType"/>
           </xs:schema>"#,
    );
}

/// `msData/simpleType/stE053`: XSD 1.1 forbids naming `xs:anySimpleType` as a
/// member type of a union; XSD 1.0 allowed it.
#[test]
fn st_e053_any_simple_type_is_not_a_union_member_in_1_1() {
    accepted_by_1_0_only(
        "stE053",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="root" fixed="abcdefg">
               <xsd:simpleType>
                 <xsd:union memberTypes="xsd:boolean xsd:int xsd:anySimpleType"/>
               </xsd:simpleType>
             </xsd:element>
           </xsd:schema>"#,
    );
}

// ----------------------------------------------------------- `xs:redefine`

/// `msData/schema/schM10` and `msData/attributeGroup/attgZ003`:
/// *src-redefine.7.2.2* — a redefinition with no self-reference replaces the
/// original outright and must restrict it, so it may neither loosen a `fixed`
/// value nor offer an attribute the original prohibits.
#[test]
fn sch_m10_redefined_attribute_groups_must_restrict_the_original() {
    let fixed_to_default = MemoryResolver::default().with(
        "original.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:attributeGroup name="c-g2">
               <xsd:attribute name="a1"/>
               <xsd:attribute name="a2" type="xsd:string" fixed="abc"/>
             </xsd:attributeGroup>
           </xsd:schema>"#,
    );
    let error = assemble_with(
        XsdVersion::V1_0,
        fixed_to_default,
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:redefine schemaLocation="original.xsd">
               <xsd:attributeGroup name="c-g2">
                 <xsd:attribute name="a1"/>
                 <xsd:attribute name="a2" type="xsd:string" default="abc"/>
               </xsd:attributeGroup>
             </xsd:redefine>
           </xsd:schema>"#,
    )
    .expect_err("schM10 assembled");
    assert_eq!(error.code(), "src-redefine.7.2.2");

    let prohibited = MemoryResolver::default().with(
        "original.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:attributeGroup name="foo">
               <xsd:attribute name="att" type="xsd:int"/>
               <xsd:attribute name="att1" type="xsd:string" use="prohibited"/>
             </xsd:attributeGroup>
           </xsd:schema>"#,
    );
    let error = assemble_with(
        XsdVersion::V1_0,
        prohibited,
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:redefine schemaLocation="original.xsd">
               <xsd:attributeGroup name="foo">
                 <xsd:attribute name="att1" type="xsd:string"/>
               </xsd:attributeGroup>
             </xsd:redefine>
           </xsd:schema>"#,
    )
    .expect_err("attgZ003 assembled");
    assert_eq!(error.code(), "src-redefine.7.2.2");
}

/// `msData/schema/schT3` is the control for the same rule: a redefinition that
/// repeats the original's `use="prohibited"` adds nothing and is a restriction.
#[test]
fn sch_t3_a_repeated_prohibition_still_restricts() {
    let resolver = MemoryResolver::default().with(
        "original.xsd",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:attributeGroup name="b-g2">
               <xsd:attribute name="att9" use="required"/>
               <xsd:attribute name="att10" use="prohibited"/>
               <xsd:attribute name="att16" fixed="fix"/>
             </xsd:attributeGroup>
           </xsd:schema>"#,
    );
    assemble_with(
        XsdVersion::V1_0,
        resolver,
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:redefine schemaLocation="original.xsd">
               <xsd:attributeGroup name="b-g2">
                 <xsd:attribute name="att9" use="required"/>
                 <xsd:attribute name="att10" use="prohibited"/>
                 <xsd:attribute name="att16" fixed="fix"/>
               </xsd:attributeGroup>
             </xsd:redefine>
           </xsd:schema>"#,
    )
    .expect("schT3 rejected");
}

/// `msData/simpleType/stZ033`, `stZ034`: three documents redefine one
/// enumeration in turn, and each level must see the level below it already
/// redefined — so the innermost enumeration is `{1,2,3}`, the middle `{2,3}` and
/// the outermost `{3}`.
#[test]
fn st_z033_chained_redefinitions_apply_from_the_inside_out() {
    let resolver = MemoryResolver::default()
        .with(
            "inner.xsd",
            r#"<xs:schema targetNamespace="foo" xmlns="foo"
                         xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:simpleType name="B">
                   <xs:restriction base="xs:int">
                     <xs:enumeration value="1"/>
                     <xs:enumeration value="2"/>
                     <xs:enumeration value="3"/>
                   </xs:restriction>
                 </xs:simpleType>
               </xs:schema>"#,
        )
        .with(
            "middle.xsd",
            r#"<xs:schema targetNamespace="foo" xmlns="foo"
                         xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:redefine schemaLocation="inner.xsd">
                   <xs:simpleType name="B">
                     <xs:restriction base="B">
                       <xs:enumeration value="2"/>
                       <xs:enumeration value="3"/>
                     </xs:restriction>
                   </xs:simpleType>
                 </xs:redefine>
                 <xs:element name="b2" type="B"/>
               </xs:schema>"#,
        );
    let mut set = SchemaSet::new().with_resolver(Box::new(resolver));
    set.add_document(
        None,
        r#"<xs:schema targetNamespace="foo" xmlns="foo"
                     xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:redefine schemaLocation="middle.xsd">
               <xs:simpleType name="B">
                 <xs:restriction base="B"><xs:enumeration value="3"/></xs:restriction>
               </xs:simpleType>
             </xs:redefine>
             <xs:element name="b1" type="B"/>
           </xs:schema>"#,
    )
    .expect("the chain is well-formed");
    let schema = set.compile().expect("stZ033 rejected");
    // Only the outermost enumeration survives, so `stZ033.i` and `stZ034.i` are
    // both invalid instances and `3` is the one value left.
    assert!(!schema.validate_str(r#"<b1 xmlns="foo">1</b1>"#).valid);
    assert!(!schema.validate_str(r#"<b2 xmlns="foo">2</b2>"#).valid);
    assert!(schema.validate_str(r#"<b1 xmlns="foo">3</b1>"#).valid);
    // …and `b2` is still declared, so the two above fail on the value rather
    // than on a declaration the reordering lost.
    assert!(schema.validate_str(r#"<b2 xmlns="foo">3</b2>"#).valid);
}

// ------------------------------------------------ composition and wildcards

/// `msData/additional/addB187`, `addB188`: a schema may supply its own
/// definitions for the XML namespace, and they replace the ones this crate
/// provides out of the box rather than colliding with them.
#[test]
fn add_b187_a_user_schema_for_the_xml_namespace_wins() {
    let resolver = MemoryResolver::default().with(
        "xml.xsd",
        r#"<xs:schema xmlns:xml="http://www.w3.org/XML/1998/namespace"
                     targetNamespace="http://www.w3.org/XML/1998/namespace"
                     xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:attribute name="lang" type="xs:language"/>
             <xs:attribute name="blah"/>
             <xs:attributeGroup name="specialAttrs">
               <xs:attribute ref="xml:lang"/>
               <xs:attribute ref="xml:blah"/>
             </xs:attributeGroup>
           </xs:schema>"#,
    );
    let mut set = SchemaSet::new().with_resolver(Box::new(resolver));
    set.add_document(
        None,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:import namespace="http://www.w3.org/XML/1998/namespace"
                        schemaLocation="xml.xsd"/>
             <xs:complexType name="t"><xs:attribute ref="xml:blah"/></xs:complexType>
           </xs:schema>"#,
    )
    .expect("the import resolves");
    let schema = set.compile().expect("addB187 rejected");
    assert!(
        schema
            .find_attribute(Some("http://www.w3.org/XML/1998/namespace"), "blah")
            .is_some()
    );
    // `addB187.v` and `addB188.i`: the attribute the user's `specialAttrs`
    // adds is allowed on the type, and one it does not declare is not.
    const INSTANCE: &str = r#"<doc xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                                   xsi:type="t" xml:{attribute}="a"/>"#;
    let declared = schema.validate_str(&INSTANCE.replace("{attribute}", "blah"));
    assert!(declared.valid, "addB187.v: {:?}", declared.first_error());
    let undeclared = schema.validate_str(&INSTANCE.replace("{attribute}", "blah2"));
    assert!(
        !undeclared.valid,
        "addB188.i: the schema declares no `xml:blah2`"
    );
}

/// `msData/schema/schZ012_b`, `schZ012_c`: an `xs:import` whose
/// `schemaLocation` is a second spelling of a document already read contributes
/// nothing, and must not collide with the reading that already happened.
#[test]
fn sch_z012_b_an_import_reached_twice_is_read_once() {
    const ROOT: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                                     targetNamespace="a" xmlns="a">
                            <xs:import schemaLocation="middle.xsd"/>
                            <xs:element name="root"/>
                          </xs:schema>"#;
    let resolver = MemoryResolver::default()
        .with(
            "middle.xsd",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:import namespace="a" schemaLocation="ROOT-again.xsd"/>
                 <xs:element name="root"/>
               </xs:schema>"#,
        )
        // The same text under a second location, as a case-folding file system
        // serves it.
        .with("ROOT-again.xsd", ROOT);
    assemble_with(XsdVersion::V1_0, resolver, ROOT).expect("schZ012_b rejected");
}

/// `msData/wildcards/wildZ013`: XSD 1.0 Errata E1-10 clause 5.3 — uniting
/// `not(N)` with a set that holds ·absent· but not `N` has no XSD 1.0 value.
/// XSD 1.1's negation of a *set* expresses it.
#[test]
fn wild_z013_inexpressible_wildcard_union_is_1_1_only() {
    const SCHEMA: &str = r###"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                                       targetNamespace="a" xmlns:a="a">
             <xs:complexType name="base2">
               <xs:sequence/>
               <xs:attributeGroup ref="a:attG1-54"/>
             </xs:complexType>
             <xs:complexType name="derived2">
               <xs:complexContent>
                 <xs:extension base="a:base2">
                   <xs:sequence/>
                   <xs:attributeGroup ref="a:attG2-54"/>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:attributeGroup name="attG1-54">
               <xs:anyAttribute namespace="##other" processContents="lax"/>
             </xs:attributeGroup>
             <xs:attributeGroup name="attG2-54">
               <xs:anyAttribute namespace="##local b c" processContents="lax"/>
             </xs:attributeGroup>
           </xs:schema>"###;
    let error = assemble(XsdVersion::V1_0, SCHEMA).expect_err("wildZ013 assembled under 1.0");
    assert_eq!(error.code(), "cos-aw-union");
    assemble(XsdVersion::V1_1, SCHEMA).expect("wildZ013 rejected under 1.1");
    // The expressible neighbour: the set holds the negated namespace instead of
    // ·absent·, so the union is `not(absent)` and both versions accept it.
    accepted(
        "an expressible attribute wildcard union",
        &SCHEMA.replace("##local b c", "##targetNamespace b c"),
    );
}

// ------------------------------------------------------------- tripwires

/// The two XSD 1.1 relaxations the changes above run closest to, kept here so
/// that tightening a neighbouring rule cannot quietly undo them:
/// `msData/attributeGroup/attgC010` (an attribute group referring to itself) and
/// `msData/additional/addB118` (a restriction that drops an ·emptiable· branch).
#[test]
fn the_1_1_relaxations_still_hold() {
    assemble(
        XsdVersion::V1_1,
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="T" type="test"/>
             <xsd:complexType name="test"><xsd:attributeGroup ref="test"/></xsd:complexType>
             <xsd:attributeGroup name="test">
               <xsd:attributeGroup ref="test"/>
               <xsd:attribute name="foo" type="xsd:int"/>
               <xsd:attribute name="bar"/>
             </xsd:attributeGroup>
           </xsd:schema>"#,
    )
    .expect("attgC010 rejected under 1.1");
    assemble(
        XsdVersion::V1_1,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="restrictionType">
               <xs:sequence>
                 <xs:element name="f"/>
                 <xs:choice>
                   <xs:choice minOccurs="1">
                     <xs:element name="a"/>
                     <xs:element name="b"/>
                   </xs:choice>
                   <xs:sequence minOccurs="0">
                     <xs:element name="a1"/>
                     <xs:element name="b1"/>
                   </xs:sequence>
                 </xs:choice>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="complexRestrictionType">
               <xs:complexContent>
                 <xs:restriction base="restrictionType">
                   <xs:sequence>
                     <xs:element name="f"/>
                     <xs:choice minOccurs="0">
                       <xs:element name="a"/>
                       <xs:element name="b"/>
                     </xs:choice>
                   </xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    )
    .expect("addB118 rejected under 1.1");
}

// --------------------------------------------- the complete attribute wildcard

/// `sunData/combined/007` (`test.7.n`, `test.8.n`) and
/// `msData/wildcards/wildZ011`: §3.4.2 makes a complex type's {attribute
/// wildcard} the *intersection* of the `<anyAttribute>` written on the type
/// with the wildcards of every attribute group it refers to. Writing both is
/// therefore a narrowing, never a replacement.
#[test]
fn test007_the_complete_attribute_wildcard_is_an_intersection() {
    const SCHEMA: &str = r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                                       xmlns:a="urn:a" xmlns:b="urn:b"
                                       targetNamespace="urn:foo" xmlns="urn:foo"
                                       elementFormDefault="qualified">
             <xs:element name="justA">
               <xs:complexType>
                 <xs:attributeGroup ref="skip_A"/>
                 <xs:anyAttribute processContents="skip" namespace="urn:a urn:b urn:c"/>
               </xs:complexType>
             </xs:element>
             <xs:element name="emptywc">
               <xs:complexType>
                 <xs:attributeGroup ref="skip_A"/>
                 <xs:attributeGroup ref="lax_B"/>
               </xs:complexType>
             </xs:element>
             <xs:attributeGroup name="skip_A">
               <xs:anyAttribute processContents="skip" namespace="urn:a"/>
             </xs:attributeGroup>
             <xs:attributeGroup name="lax_B">
               <xs:anyAttribute processContents="lax" namespace="urn:b"/>
             </xs:attributeGroup>
           </xs:schema>"####;
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        // `urn:a` ∩ {`urn:a`, `urn:b`, `urn:c`} is `urn:a`, so only that one
        // namespace is admitted — `test.5.v` against `test.7.n`, `test.8.n`.
        assert!(
            validates(
                version,
                SCHEMA,
                r#"<justA xmlns="urn:foo" xmlns:a="urn:a" a:foo="local"/>"#,
            ),
            "test.5.v: an attribute in the intersection was refused under {version:?}"
        );
        for instance in [
            r#"<justA xmlns="urn:foo" xmlns:b="urn:b" b:foo="local"/>"#,
            r#"<justA xmlns="urn:foo" xmlns:c="urn:c" c:foo="local"/>"#,
        ] {
            assert!(
                !validates(version, SCHEMA, instance),
                "test.7.n/test.8.n: an attribute outside the intersection was admitted under \
                 {version:?}"
            );
        }
        // Two attribute groups alone intersect to nothing, which is not an
        // error: the type simply admits no wildcard-matched attribute
        // (`test.1.v`).
        assert!(
            validates(version, SCHEMA, r#"<emptywc xmlns="urn:foo"/>"#),
            "test.1.v: the empty intersection was made an error under {version:?}"
        );
        assert!(
            !validates(
                version,
                SCHEMA,
                r#"<emptywc xmlns="urn:foo" xmlns:a="urn:a" a:foo="local"/>"#,
            ),
            "test.2.n: the empty intersection admitted an attribute under {version:?}"
        );
    }
}

/// `msData/wildcards/wildZ011`: the local `<anyAttribute namespace="##any">`
/// is intersected with `##targetNamespace` and `##any` from the two attribute
/// groups, leaving the target namespace alone.
#[test]
fn wild_z011_a_local_wildcard_narrows_the_groups_it_joins() {
    const SCHEMA: &str = r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                                       targetNamespace="a" xmlns:a="a"
                                       elementFormDefault="qualified">
             <xs:element name="doc" type="a:metadata"/>
             <xs:complexType name="metadata">
               <xs:sequence/>
               <xs:attributeGroup ref="a:attG-a"/>
               <xs:anyAttribute namespace="##any" processContents="lax"/>
             </xs:complexType>
             <xs:attribute name="attA"/>
             <xs:attributeGroup name="attG-a">
               <xs:anyAttribute namespace="##targetNamespace" processContents="lax"/>
             </xs:attributeGroup>
           </xs:schema>"####;
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        assert!(
            !validates(
                version,
                SCHEMA,
                r#"<a:doc x:blah="a" xmlns:a="a" xmlns:x="x"/>"#,
            ),
            "wildZ011: `{{x}}blah` was admitted under {version:?}"
        );
        assert!(
            validates(version, SCHEMA, r#"<a:doc a:attA="a" xmlns:a="a"/>"#,),
            "wildZ011: `{{a}}attA` was refused under {version:?}"
        );
    }
}

/// `msData/attribute/attO025`: §3.5.6 *au-props-correct.2* — an attribute use
/// may not contradict the `fixed` value constraint of the declaration it
/// refers to. The use here sits in a *named* complex type, which is reached
/// before the pass over the global attributes.
#[test]
fn att_o025_an_attribute_use_may_not_contradict_a_fixed_declaration() {
    assert_eq!(
        rejected(
            "attO025",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                           targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
                 <xsd:attribute name="foo" fixed="123"/>
                 <xsd:complexType name="attRef">
                   <xsd:attribute ref="x:foo" fixed="456"/>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "au-props-correct.2"
    );
    accepted(
        "attO025 repeating the declaration's own value",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting">
             <xsd:attribute name="foo" fixed="123"/>
             <xsd:complexType name="attRef">
               <xsd:attribute ref="x:foo" fixed="123"/>
             </xsd:complexType>
           </xsd:schema>"#,
    );
}

/// `msData/simpleType/stZ007`, `stZ047`, `stZ055`: Datatypes §3.2.1.3 forbids
/// `xs:anySimpleType` as the base of a facet-based restriction, which is what
/// §3.4.2.2 makes of an `xs:simpleContent` `xs:restriction`. XSD 1.0 had no
/// such rule (W3C bug 14559).
#[test]
fn st_z007_simple_content_may_not_restrict_any_simple_type_in_1_1() {
    accepted_by_1_0_only(
        "stZ007",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="t1">
               <xs:simpleContent><xs:extension base="xs:anySimpleType"/></xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="t2">
               <xs:simpleContent><xs:restriction base="t1"/></xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#,
    );
    // Extending `xs:anySimpleType` on its own stays legal under both
    // versions — only restricting the result is the error.
    accepted(
        "stZ007 without the restriction",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="t1">
               <xs:simpleContent><xs:extension base="xs:anySimpleType"/></xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#,
    );
    // A restriction that writes its own simple type defines nothing in terms
    // of `xs:anySimpleType`, so it is still allowed.
    accepted(
        "stZ007 with an explicit simple type",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="t1">
               <xs:simpleContent><xs:extension base="xs:anySimpleType"/></xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="t2">
               <xs:simpleContent>
                 <xs:restriction base="t1">
                   <xs:simpleType><xs:restriction base="xs:string"/></xs:simpleType>
                 </xs:restriction>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#,
    );
}

// ------------------------------------------- definitions reached through an
// ------------------------------------------- element declaration

/// `msData/additional/addB077` and `msData/particles/particlesZ010`: a model
/// group may be reached again from inside a type written on an element
/// declaration it contains. §3.8.6 clause 2 forbids a group that contains
/// itself "within the {particles} of the group", and the content model of an
/// element declaration is not among them.
#[test]
fn add_b077_a_group_may_be_reached_through_an_element_declaration() {
    accepted(
        "addB077",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:complexType name="complexType">
               <xsd:sequence><xsd:element name="r1"/></xsd:sequence>
             </xsd:complexType>
             <xsd:element name="elem">
               <xsd:complexType><xsd:sequence><xsd:group ref="foo"/></xsd:sequence></xsd:complexType>
             </xsd:element>
             <xsd:group name="foo">
               <xsd:sequence>
                 <xsd:element name="foo1"/>
                 <xsd:element name="x">
                   <xsd:complexType>
                     <xsd:complexContent>
                       <xsd:extension base="complexType">
                         <xsd:group ref="foo"/>
                       </xsd:extension>
                     </xsd:complexContent>
                   </xsd:complexType>
                 </xsd:element>
               </xsd:sequence>
             </xsd:group>
           </xsd:schema>"#,
    );
    accepted(
        "particlesZ010",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:complexType name="a">
               <xsd:sequence>
                 <xsd:group ref="ul" minOccurs="0" maxOccurs="unbounded"/>
               </xsd:sequence>
             </xsd:complexType>
             <xsd:group name="ul">
               <xsd:sequence>
                 <xsd:element name="ul">
                   <xsd:complexType>
                     <xsd:sequence>
                       <xsd:choice minOccurs="0" maxOccurs="unbounded">
                         <xsd:element name="li"/>
                         <xsd:group ref="ul" minOccurs="0" maxOccurs="unbounded"/>
                       </xsd:choice>
                     </xsd:sequence>
                   </xsd:complexType>
                 </xsd:element>
               </xsd:sequence>
             </xsd:group>
           </xsd:schema>"#,
    );
    // A group that reaches itself without crossing an element declaration is
    // still the error §3.8.6 clause 2 names.
    assert_eq!(
        rejected(
            "a group particle that names its own group",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:group name="loop">
                   <xsd:sequence>
                     <xsd:element name="e"/>
                     <xsd:group ref="loop"/>
                   </xsd:sequence>
                 </xsd:group>
                 <xsd:complexType name="t">
                   <xsd:sequence><xsd:group ref="loop"/></xsd:sequence>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "src-group"
    );
}

/// `msData/particles/particlesEb041`: the same for a type definition — an
/// element declaration inside a type may name that very type as the base of a
/// derivation. §3.4.6 *ct-props-correct* clause 3 speaks only of the {base type
/// definition} chain.
#[test]
fn particles_eb041_a_type_may_be_reached_through_an_element_declaration() {
    accepted(
        "particlesEb041",
        r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                       targetNamespace="foo" xmlns="foo">
             <xsd:complexType name="foo">
               <xsd:sequence>
                 <xsd:element name="foo"/>
                 <xsd:element name="bar" minOccurs="0">
                   <xsd:complexType>
                     <xsd:complexContent>
                       <xsd:restriction base="foo">
                         <xsd:sequence><xsd:element name="foo"/></xsd:sequence>
                       </xsd:restriction>
                     </xsd:complexContent>
                   </xsd:complexType>
                 </xsd:element>
               </xsd:sequence>
             </xsd:complexType>
             <xsd:element name="root" type="foo"/>
           </xsd:schema>"#,
    );
    // A type that derives from itself directly is still the error.
    assert_eq!(
        rejected(
            "a type whose base is itself",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:complexType name="loop">
                   <xsd:complexContent>
                     <xsd:restriction base="loop">
                       <xsd:sequence><xsd:element name="e"/></xsd:sequence>
                     </xsd:restriction>
                   </xsd:complexContent>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "src-type-cycle"
    );
    // An error written inside a deferred type is still reported.
    assert_eq!(
        rejected(
            "a deferred element type with an undefined base",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:element name="doc">
                   <xsd:complexType>
                     <xsd:complexContent>
                       <xsd:restriction base="absent">
                         <xsd:sequence><xsd:element name="e"/></xsd:sequence>
                       </xsd:restriction>
                     </xsd:complexContent>
                   </xsd:complexType>
                 </xsd:element>
               </xsd:schema>"#,
        ),
        "src-resolve"
    );
}

// -------------------------------------------- particle restriction, XSD 1.1

/// `msData/particles/particlesHb008`: XSD 1.1 decides *Particle Valid
/// (Restriction)* by language containment, so a `choice` in the restriction may
/// stand for a run of the base's members — the language of one pass is the
/// union of the passes that pick one branch. XSD 1.0's mapping, one member to
/// one member, cannot express that.
#[test]
fn particles_hb008_a_choice_may_span_several_base_members_in_1_1() {
    let text = r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                               targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting"
                               elementFormDefault="qualified">
             <xsd:complexType name="base">
               <xsd:choice>
                 <xsd:element name="e1" minOccurs="1" maxOccurs="3"/>
                 <xsd:sequence maxOccurs="2">
                   <xsd:element name="e2" minOccurs="1" maxOccurs="3"/>
                   <xsd:element name="e3" minOccurs="0" maxOccurs="3"/>
                   <xsd:element name="e4" minOccurs="0" maxOccurs="3"/>
                 </xsd:sequence>
               </xsd:choice>
             </xsd:complexType>
             <xsd:element name="doc">
               <xsd:complexType>
                 <xsd:complexContent>
                   <xsd:restriction base="x:base">
                     <xsd:choice>
                       <xsd:element name="e1" minOccurs="1" maxOccurs="2"/>
                       <xsd:sequence maxOccurs="2">
                         <xsd:element name="e2"/>
                         <xsd:choice>
                           <xsd:element name="e3" minOccurs="2" maxOccurs="3"/>
                           <xsd:element name="e4" minOccurs="1" maxOccurs="3"/>
                         </xsd:choice>
                       </xsd:sequence>
                     </xsd:choice>
                   </xsd:restriction>
                 </xsd:complexContent>
               </xsd:complexType>
             </xsd:element>
           </xsd:schema>"#;
    assert!(
        assemble(XsdVersion::V1_0, text).is_err(),
        "particlesHb008: accepted under XML Schema 1.0"
    );
    if let Err(error) = assemble(XsdVersion::V1_1, text) {
        panic!("particlesHb008: rejected under XML Schema 1.1: {error}");
    }
    // A branch that offers a name the base's run does not is still refused.
    assert!(
        assemble(
            XsdVersion::V1_1,
            &text.replace(
                r#"<xsd:element name="e4" minOccurs="1" maxOccurs="3"/>"#,
                r#"<xsd:element name="e9" minOccurs="1" maxOccurs="3"/>"#
            ),
        )
        .is_err(),
        "particlesHb008: a branch naming an element the base does not have was accepted"
    );
}

/// `msData/particles/particlesHb011`: the same reading lets a *group* restrict
/// a single element particle — every item the group can produce is governed by
/// the base's declaration, and its ·effective total range· fits the base's
/// occurrence range. The §3.9.6 table calls the pair forbidden, which is what
/// XSD 1.0 is held to.
#[test]
fn particles_hb011_a_group_may_restrict_an_element_particle_in_1_1() {
    let text = r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                               targetNamespace="http://xsdtesting" xmlns:x="http://xsdtesting"
                               elementFormDefault="qualified">
             <xsd:complexType name="base">
               <xsd:choice minOccurs="2" maxOccurs="unbounded">
                 <xsd:element name="e1" minOccurs="0" maxOccurs="10"/>
                 <xsd:element name="e2" minOccurs="0"/>
                 <xsd:element name="e3" minOccurs="0"/>
               </xsd:choice>
             </xsd:complexType>
             <xsd:element name="doc">
               <xsd:complexType>
                 <xsd:complexContent>
                   <xsd:restriction base="x:base">
                     <xsd:choice minOccurs="2" maxOccurs="unbounded">
                       <xsd:sequence maxOccurs="2">
                         <xsd:element name="e1" maxOccurs="2"/>
                       </xsd:sequence>
                       <xsd:element name="e2"/>
                       <xsd:element name="e3" minOccurs="1"/>
                     </xsd:choice>
                   </xsd:restriction>
                 </xsd:complexContent>
               </xsd:complexType>
             </xsd:element>
           </xsd:schema>"#;
    assert!(
        assemble(XsdVersion::V1_0, text).is_err(),
        "particlesHb011: accepted under XML Schema 1.0"
    );
    if let Err(error) = assemble(XsdVersion::V1_1, text) {
        panic!("particlesHb011: rejected under XML Schema 1.1: {error}");
    }
    // The group may not contribute more items than the base particle admits…
    assert!(
        assemble(
            XsdVersion::V1_1,
            &text.replace(
                r#"<xsd:element name="e1" maxOccurs="2"/>"#,
                r#"<xsd:element name="e1" maxOccurs="20"/>"#
            ),
        )
        .is_err(),
        "particlesHb011: a group exceeding the base's occurrence range was accepted"
    );
    // …nor items the base's declaration does not govern.
    assert!(
        assemble(
            XsdVersion::V1_1,
            &text.replace(
                r#"<xsd:element name="e1" maxOccurs="2"/>"#,
                r#"<xsd:element name="e9" maxOccurs="2"/>"#
            ),
        )
        .is_err(),
        "particlesHb011: a group holding a foreign element was accepted"
    );
}

// ----------------------------------------------------- `xs:all` extension

/// `saxonData/All/all313` and `all308`: §3.4.6.2 joins two particles into an
/// `all` group only when both are one, and the joined group keeps a single
/// {min occurs}, so the two must agree. A `mixed` base with no content model
/// has an empty *sequence* for its particle, which is why an `all` group
/// cannot extend it (W3C bug 6202).
#[test]
fn all313_an_all_group_extending_an_all_group_keeps_its_min_occurs() {
    assert_eq!(
        rejected_by_1_1(
            "all313",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="ProductType">
                   <xs:all><xs:element name="child1"/></xs:all>
                 </xs:complexType>
                 <xs:complexType name="ExtendedProductType">
                   <xs:complexContent>
                     <xs:extension base="ProductType">
                       <xs:all minOccurs="0"><xs:element name="child2"/></xs:all>
                     </xs:extension>
                   </xs:complexContent>
                 </xs:complexType>
               </xs:schema>"#,
        ),
        "cos-ct-extends.1.4.3.2"
    );
    // The same schema with matching `minOccurs` is a valid extension — under
    // XSD 1.1, which is where an `all` group may be extended at all.
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="ProductType">
               <xs:all><xs:element name="child1"/></xs:all>
             </xs:complexType>
             <xs:complexType name="ExtendedProductType">
               <xs:complexContent>
                 <xs:extension base="ProductType">
                   <xs:all><xs:element name="child2"/></xs:all>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    ) {
        panic!("an `all` group extending an `all` group was rejected under 1.1: {error}");
    }
    assert!(
        assemble(
            XsdVersion::V1_1,
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="b" mixed="true"><xs:all/></xs:complexType>
                 <xs:complexType name="ext">
                   <xs:complexContent mixed="true">
                     <xs:extension base="b">
                       <xs:all>
                         <xs:element name="d" minOccurs="0" maxOccurs="2"/>
                       </xs:all>
                     </xs:extension>
                   </xs:complexContent>
                 </xs:complexType>
               </xs:schema>"#,
        )
        .is_err(),
        "all308: an `all` group extending an empty mixed content type was accepted"
    );
}

// -------------------------------------------------------- open content

/// `saxonData/Complex/complex018`: the schema for schemas requires an
/// `xs:restriction` that writes an `xs:openContent` to write its content model
/// too (W3C bug 16786). An `xs:extension` has the two independently.
#[test]
fn complex018_a_restriction_with_open_content_needs_a_content_model() {
    assert_eq!(
        rejected_by_1_1(
            "complex018",
            r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                   <xs:complexType name="ProductType">
                     <xs:openContent><xs:any namespace="##any"/></xs:openContent>
                     <xs:sequence><xs:element name="child1" minOccurs="0"/></xs:sequence>
                   </xs:complexType>
                   <xs:complexType name="RestrictedProductType">
                     <xs:complexContent>
                       <xs:restriction base="ProductType">
                         <xs:openContent><xs:any namespace="##other"/></xs:openContent>
                       </xs:restriction>
                     </xs:complexContent>
                   </xs:complexType>
                 </xs:schema>"####,
        ),
        "src-ct"
    );
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
               <xs:complexType name="ProductType">
                 <xs:openContent><xs:any namespace="##any"/></xs:openContent>
                 <xs:sequence><xs:element name="child1" minOccurs="0"/></xs:sequence>
               </xs:complexType>
               <xs:complexType name="RestrictedProductType">
                 <xs:complexContent>
                   <xs:restriction base="ProductType">
                     <xs:openContent><xs:any namespace="##other"/></xs:openContent>
                     <xs:sequence><xs:element name="child1" minOccurs="0"/></xs:sequence>
                   </xs:restriction>
                 </xs:complexContent>
               </xs:complexType>
             </xs:schema>"####,
    ) {
        panic!("complex018: the same restriction with a content model was rejected: {error}");
    }
}

/// `saxonData/Open/open048`: the `xs:any` of an `xs:openContent` has type
/// `xs:wildcard` in the schema for schemas, which carries no occurrence
/// attributes — open content says *which* elements may appear beside the
/// content model, not how many (W3C bug 15618).
#[test]
fn open048_open_content_takes_no_occurrence_attributes() {
    assert_eq!(
        rejected_by_1_1(
            "open048",
            r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                   <xs:complexType name="beta">
                     <xs:openContent>
                       <xs:any namespace="##other" processContents="lax" maxOccurs="unbounded"/>
                     </xs:openContent>
                     <xs:sequence/>
                   </xs:complexType>
                 </xs:schema>"####,
        ),
        "src-openContent"
    );
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
               <xs:complexType name="beta">
                 <xs:openContent>
                   <xs:any namespace="##other" processContents="lax"/>
                 </xs:openContent>
                 <xs:sequence/>
               </xs:complexType>
             </xs:schema>"####,
    ) {
        panic!("open048: the same open content without `maxOccurs` was rejected: {error}");
    }
}

/// `saxonData/Open/open046` against its neighbour `open031`: the document's
/// `xs:defaultOpenContent` is applied to the content type a definition ends up
/// with, so an extension whose own particle is empty but whose base's is not
/// takes the default after all — and here the default is `suffix` where the
/// base is `interleave`.
#[test]
fn open046_the_default_open_content_sees_the_derived_content_type() {
    const SCHEMA: &str = r####"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:defaultOpenContent mode="suffix" appliesToEmpty="false">
               <xs:any namespace="http://nonempty.com/" processContents="lax"/>
             </xs:defaultOpenContent>
             <xs:complexType name="B">
               <xs:openContent mode="interleave">
                 <xs:any namespace="http://open.com/" processContents="lax"/>
               </xs:openContent>
               <xs:sequence>
                 <xs:element name="a" maxOccurs="unbounded"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="R">
               <xs:complexContent>
                 <xs:extension base="B">OWN</xs:extension>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"####;
    assert!(
        assemble(XsdVersion::V1_1, &SCHEMA.replace("OWN", "<xs:sequence/>")).is_err(),
        "open046: the default open content was not applied to the extended content type"
    );
    // `open031` is the same schema with the default kept out explicitly.
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        &SCHEMA.replace("OWN", r#"<xs:openContent mode="none"/><xs:sequence/>"#),
    ) {
        panic!("open031: an extension turning open content off was rejected: {error}");
    }
}

// --------------------------------------------- conditional type assignment

/// `saxonData/CTA/cta0001`: §3.4.6 clause 1.4.2 lets an `xs:simpleContent`
/// restriction sit on a base whose content is mixed and ·emptiable· — which is
/// "can match the empty sequence", not "can match nothing else".
/// `msData/complexType/ctD004` is the other half of the clause: the restriction
/// has to write the simple type itself, there being none in the base to derive
/// from.
#[test]
fn cta0001_a_mixed_emptiable_base_may_be_restricted_to_simple_content() {
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="messageType" mixed="true">
               <xs:sequence>
                 <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="messageTypeString">
               <xs:simpleContent>
                 <xs:restriction base="messageType">
                   <xs:simpleType><xs:restriction base="xs:string"/></xs:simpleType>
                 </xs:restriction>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#,
    ) {
        panic!("cta0001: a mixed, emptiable base was refused: {error}");
    }
    assert_eq!(
        rejected(
            "ctD004",
            r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
                 <xsd:complexType name="fooType">
                   <xsd:simpleContent>
                     <xsd:restriction base="xsd:anyType"/>
                   </xsd:simpleContent>
                 </xsd:complexType>
               </xsd:schema>"#,
        ),
        "src-ct.2.2"
    );
}

/// `saxonData/CTA/cta0043`: a restriction whose element declaration carries a
/// {type table} must select, for the same test, a type derived by restriction
/// from the one the base selects. `cta9001err` is the neighbouring
/// representation rule: only the last `xs:alternative` may omit its `test`.
#[test]
fn cta0043_a_type_table_must_restrict_the_base_declarations() {
    const SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="chapType">
               <xs:sequence>
                 <xs:element name="stamp">
                   <xs:alternative test="@type='date'" type="dateWithTypeAttribute"/>
                   <xs:alternative test="@type='dateTime'" type="dateTimeStampWithTypeAttribute"/>
                 </xs:element>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="appendixType">
               <xs:complexContent>
                 <xs:restriction base="chapType">
                   <xs:sequence>
                     <xs:element name="stamp">
                       <xs:alternative test="@type='date'" type="dateWithTypeAttribute"/>
                       <xs:alternative test="@type='dateTime'" type="SELECTED"/>
                     </xs:element>
                   </xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
             <xs:complexType name="dateWithTypeAttribute">
               <xs:simpleContent>
                 <xs:extension base="xs:date">
                   <xs:attribute name="type" type="xs:NCName"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="dateTimeStampWithTypeAttribute">
               <xs:simpleContent>
                 <xs:extension base="xs:dateTimeStamp">
                   <xs:attribute name="type" type="xs:NCName"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="dateTimeWithTypeAttribute">
               <xs:simpleContent>
                 <xs:extension base="xs:dateTime">
                   <xs:attribute name="type" type="xs:NCName"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#;
    assert!(
        assemble(
            XsdVersion::V1_1,
            &SCHEMA.replace("SELECTED", "dateTimeWithTypeAttribute"),
        )
        .is_err(),
        "cta0043: a type table widening the base's was accepted"
    );
    if let Err(error) = assemble(
        XsdVersion::V1_1,
        &SCHEMA.replace("SELECTED", "dateTimeStampWithTypeAttribute"),
    ) {
        panic!("cta0043: the same table repeating the base's type was rejected: {error}");
    }
    assert!(
        assemble(
            XsdVersion::V1_1,
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:complexType name="t"><xs:sequence/></xs:complexType>
                 <xs:element name="e" type="xs:anyType">
                   <xs:alternative type="t"/>
                   <xs:alternative test="@kind='x'" type="t"/>
                 </xs:element>
               </xs:schema>"#,
        )
        .is_err(),
        "cta9001err: an `xs:alternative` without a `test` was accepted before the last one"
    );
}

// ------------------------------------------------------------ assertions

/// `saxonData/Assert/assert024`: an attribute declared `xs:QName` reaches an
/// assertion as a `QName` value, with its prefix resolved against the bindings
/// in force where it was written — a comparison with the `xs:QName`
/// constructor's result cannot be answered any other way.
#[test]
fn assert024_a_qname_attribute_keeps_its_value_in_an_assertion() {
    const SCHEMA: &str = r#"<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">
             <xsd:element name="Test">
               <xsd:complexType>
                 <xsd:sequence>
                   <xsd:element name="Rule" maxOccurs="unbounded">
                     <xsd:complexType>
                       <xsd:attribute name="name" type="xsd:QName" use="required"/>
                       <xsd:assert test="@name eq xsd:QName('xsd:element')"/>
                     </xsd:complexType>
                   </xsd:element>
                 </xsd:sequence>
               </xsd:complexType>
             </xsd:element>
           </xsd:schema>"#;
    assert!(
        validates(
            XsdVersion::V1_1,
            SCHEMA,
            r#"<Test xmlns:xsd="http://www.w3.org/2001/XMLSchema"><Rule name="xsd:element"/></Test>"#,
        ),
        "assert024: the assertion was not satisfied"
    );
    // The same local name in another namespace is a different `QName`, which
    // is the whole point of comparing values rather than lexical forms.
    assert!(
        !validates(
            XsdVersion::V1_1,
            SCHEMA,
            r#"<Test xmlns:other="http://example.invalid/"><Rule name="other:element"/></Test>"#,
        ),
        "assert024: a `QName` in another namespace satisfied the assertion"
    );
    // A prefix spelt differently but bound to the same namespace is the same
    // value.
    assert!(
        validates(
            XsdVersion::V1_1,
            SCHEMA,
            r#"<Test xmlns:s="http://www.w3.org/2001/XMLSchema"><Rule name="s:element"/></Test>"#,
        ),
        "assert024: an equal `QName` written with another prefix was refused"
    );
}

// ---------------------------------------------- xs:all subsumption in XSD 1.1

/// `saxonData/All/all244`: the base group's wildcards overlap — which XSD 1.1
/// permits — and the restricting wildcard straddles both of them, so neither
/// base member is guaranteed anything. The restriction accepts
/// `(one, one, one, three, three)`, which leaves the base's first member three
/// items short of the five it requires.
#[test]
fn all244_a_restriction_that_straddles_two_base_wildcards_guarantees_neither() {
    const SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:complexType name="b">
          <xs:all>
            <xs:any namespace="http://one.uri/ http://two.uri/"
                    minOccurs="5" maxOccurs="unbounded"/>
            <xs:any namespace="http://three.uri/" minOccurs="0" maxOccurs="2"/>
          </xs:all>
        </xs:complexType>
        <xs:complexType name="r">
          <xs:complexContent>
            <xs:restriction base="b">
              <xs:all>
                <xs:any namespace="http://one.uri/" minOccurs="3" maxOccurs="unbounded"/>
                <xs:any namespace="http://two.uri/ http://three.uri/"
                        minOccurs="2" maxOccurs="2"/>
              </xs:all>
            </xs:restriction>
          </xs:complexContent>
        </xs:complexType>
      </xs:schema>"#;
    assert_eq!(
        rejected_by_1_1("all244", SCHEMA),
        "rcase-Recurse.2",
        "all244: the base `all` group's first member can be left short"
    );
    // The control: raising the straddling wildcard's `minOccurs` does not help
    // — its items may all go to the other member — but restricting it to the
    // namespaces of one base member does, because then every item it produces
    // is guaranteed to that member.
    accepted_by_1_1(
        "all244",
        &SCHEMA.replace(
            r#"namespace="http://two.uri/ http://three.uri/""#,
            r#"namespace="http://two.uri/""#,
        ),
    );
}

/// `saxonData/Wild/wild069`: the base attributes an `<e/>` to its element
/// declaration member, so its own `<xs:any namespace="##local"/>` — which is
/// required exactly once — can be left with nothing when the restriction
/// drops the declaration and keeps only the wildcard. `zang` therefore accepts
/// `<e/><f/>`, which `zing` does not.
#[test]
fn wild069_a_wildcard_does_not_guarantee_a_base_wildcard_its_items() {
    const SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:complexType name="zing">
          <xs:all>
            <xs:element name="e" minOccurs="0" maxOccurs="1">
              <xs:simpleType><xs:union memberTypes="xs:date xs:time"/></xs:simpleType>
            </xs:element>
            <xs:element name="f" type="xs:integer"/>
            <xs:any namespace="LOCAL" processContents="lax"/>
          </xs:all>
        </xs:complexType>
        <xs:complexType name="zang">
          <xs:complexContent>
            <xs:restriction base="zing">
              <xs:all>
                <xs:element name="f" type="xs:integer"/>
                <xs:any namespace="LOCAL" processContents="lax"/>
              </xs:all>
            </xs:restriction>
          </xs:complexContent>
        </xs:complexType>
        <xs:element name="doc" type="zang"/>
        <xs:element name="e" type="xs:duration"/>
      </xs:schema>"#;
    let schema = SCHEMA.replace("LOCAL", "##local");
    assert_eq!(
        rejected_by_1_1("wild069", &schema),
        "rcase-Recurse.2",
        "wild069: the base's own wildcard member can be left empty"
    );
    // The control: the very same restriction is legal once the base's own
    // wildcard member — the first of the two, written in `zing` — no longer
    // requires an item, which is precisely the member the rejection is about.
    accepted_by_1_1(
        "wild069",
        &SCHEMA
            .replacen(
                r#"processContents="lax"/>"#,
                r#"processContents="lax" minOccurs="0"/>"#,
                1,
            )
            .replace("LOCAL", "##local"),
    );
}

/// `saxonData/Wild/wild030` (the Recommendation's own example) and `wild047`:
/// a wildcard never competes for the names its **own** siblings declare, so a
/// restriction that repeats the base's element particles beside a wildcard
/// does not thereby widen those members.
#[test]
fn wild030_a_wildcard_leaves_its_siblings_names_to_them() {
    accepted_by_1_1(
        "wild030",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="computer">
               <xs:all>
                 <xs:element name="CPU" type="xs:string"/>
                 <xs:element name="memory" type="xs:string"/>
                 <xs:element name="speaker" type="xs:string" minOccurs="0"/>
                 <xs:any processContents="lax" minOccurs="0" maxOccurs="unbounded"/>
               </xs:all>
             </xs:complexType>
             <xs:complexType name="quietComputer">
               <xs:complexContent>
                 <xs:restriction base="computer">
                   <xs:all>
                     <xs:element name="CPU" type="xs:string"/>
                     <xs:element name="memory" type="xs:string"/>
                     <xs:any processContents="lax" notQName="speaker"
                             minOccurs="0" maxOccurs="unbounded"/>
                   </xs:all>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    );
}
