//! Schema-validity assessment of instance documents.

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_schema::{Schema, SchemaError, SchemaSet, ValidationOutcome, Validity, XsdVersion};

fn schema(text: &str) -> Schema {
    let mut set = SchemaSet::new();
    set.add_document(None, text).expect("a schema document");
    set.compile().expect("a compiled schema")
}

/// Like [`schema`], but under [`XsdVersion::V1_1`], for the tests that
/// exercise a 1.1-only construct.
fn schema_1_1(text: &str) -> Schema {
    let mut set = SchemaSet::new().with_version(XsdVersion::V1_1);
    set.add_document(None, text).expect("a schema document");
    set.compile().expect("a compiled schema")
}

fn code(outcome: &ValidationOutcome) -> Option<&str> {
    outcome
        .first_error()
        .map(oxixml_schema::ValidationError::code)
}

#[test]
fn simple_content_is_checked_against_its_type() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="n" type="xs:int"/>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<n>42</n>").valid);
    assert!(
        schema.validate_str("<n> 42 </n>").valid,
        "whitespace collapses"
    );
    assert!(!schema.validate_str("<n>4.2</n>").valid);
    assert!(!schema.validate_str("<n>99999999999</n>").valid);
    assert_eq!(
        code(&schema.validate_str("<n><b/></n>")),
        Some("cvc-type.3.1.2")
    );
}

#[test]
fn facets_report_the_rule_they_break() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="v">
               <xs:simpleType>
                 <xs:restriction base="xs:string">
                   <xs:minLength value="2"/>
                   <xs:maxLength value="4"/>
                   <xs:pattern value="[a-z]+"/>
                 </xs:restriction>
               </xs:simpleType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<v>abc</v>").valid);
    assert_eq!(
        code(&schema.validate_str("<v>a</v>")),
        Some("cvc-minLength-valid")
    );
    assert_eq!(
        code(&schema.validate_str("<v>abcde</v>")),
        Some("cvc-maxLength-valid")
    );
    assert_eq!(
        code(&schema.validate_str("<v>AB</v>")),
        Some("cvc-pattern-valid")
    );
}

#[test]
fn enumerations_compare_in_the_value_space() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="v">
               <xs:simpleType>
                 <xs:restriction base="xs:decimal">
                   <xs:enumeration value="1.5"/>
                   <xs:enumeration value="2"/>
                 </xs:restriction>
               </xs:simpleType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<v>1.50</v>").valid);
    assert!(schema.validate_str("<v>+2.000</v>").valid);
    assert_eq!(
        code(&schema.validate_str("<v>3</v>")),
        Some("cvc-enumeration-valid")
    );
}

#[test]
fn list_and_union_varieties() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="Ints"><xs:list itemType="xs:int"/></xs:simpleType>
             <xs:simpleType name="IntOrWord">
               <xs:union memberTypes="xs:int xs:NCName"/>
             </xs:simpleType>
             <xs:element name="l" type="Ints"/>
             <xs:element name="u" type="IntOrWord"/>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<l>1 2  3</l>").valid);
    assert!(!schema.validate_str("<l>1 x</l>").valid);
    assert!(schema.validate_str("<u>7</u>").valid);
    assert!(schema.validate_str("<u>seven</u>").valid);
    assert!(!schema.validate_str("<u>7 8</u>").valid);
}

#[test]
fn content_models_accept_only_what_they_describe() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="a" type="xs:string"/>
                   <xs:element name="b" type="xs:string" minOccurs="0" maxOccurs="2"/>
                   <xs:choice minOccurs="0">
                     <xs:element name="c" type="xs:string"/>
                     <xs:element name="d" type="xs:string"/>
                   </xs:choice>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<r><a/></r>").valid);
    assert!(schema.validate_str("<r><a/><b/><b/><d/></r>").valid);
    assert_eq!(
        code(&schema.validate_str("<r><a/><b/><b/><b/></r>")),
        Some("cvc-complex-type.2.4.a")
    );
    assert_eq!(
        code(&schema.validate_str("<r/>")),
        Some("cvc-complex-type.2.4.b")
    );
    assert_eq!(
        code(&schema.validate_str("<r><a/>text</r>")),
        Some("cvc-complex-type.2.3")
    );
}

#[test]
fn an_all_group_ignores_order() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:all>
                   <xs:element name="a" type="xs:string"/>
                   <xs:element name="b" type="xs:string" minOccurs="0"/>
                 </xs:all>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<r><a/><b/></r>").valid);
    assert!(schema.validate_str("<r><b/><a/></r>").valid);
    assert!(schema.validate_str("<r><a/></r>").valid);
    assert!(!schema.validate_str("<r><a/><a/></r>").valid);
    assert!(!schema.validate_str("<r><b/></r>").valid);
}

#[test]
fn mixed_content_allows_text_between_elements() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="p">
               <xs:complexType mixed="true">
                 <xs:sequence>
                   <xs:element name="b" type="xs:string" minOccurs="0" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<p>hello <b>world</b>!</p>").valid);
}

#[test]
fn attributes_are_matched_required_and_fixed() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e">
               <xs:complexType>
                 <xs:attribute name="need" type="xs:int" use="required"/>
                 <xs:attribute name="pin" type="xs:string" fixed="yes"/>
                 <xs:attribute name="opt" type="xs:string" default="d"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str(r#"<e need="1"/>"#).valid);
    assert_eq!(
        code(&schema.validate_str("<e/>")),
        Some("cvc-complex-type.4")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<e need="1" pin="no"/>"#)),
        Some("cvc-attribute.4")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<e need="1" other="x"/>"#)),
        Some("cvc-complex-type.3.2.2")
    );
    assert!(!schema.validate_str(r#"<e need="x"/>"#).valid);
}

#[test]
fn defaults_are_reported_in_the_psvi() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e">
               <xs:complexType>
                 <xs:attribute name="opt" type="xs:string" default="d"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    let document = oxixml_dom::build::parse_str("<e/>").expect("well-formed");
    let outcome = schema.validate_document(&document);
    assert!(outcome.valid);
    let root = document.document_element().expect("a root");
    assert_eq!(
        outcome.psvi.defaulted_attributes(root),
        [("opt".to_owned(), "d".to_owned())]
    );
    let element = outcome.psvi.element(root).expect("annotations");
    assert!(element.declaration.is_some());
    assert!(element.type_definition.is_some());
}

#[test]
fn an_element_default_fills_in_empty_content() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="n" type="xs:int" default="7"/>
           </xs:schema>"#,
    );
    let outcome = schema.validate_str("<n/>");
    assert!(outcome.valid, "{:?}", outcome.errors);
    let document = oxixml_dom::build::parse_str("<n/>").expect("well-formed");
    let outcome = schema.validate_document(&document);
    let root = document.document_element().expect("a root");
    let element = outcome.psvi.element(root).expect("annotations");
    assert!(element.from_default);
    assert_eq!(element.normalized_value.as_deref(), Some("7"));
}

#[test]
fn a_fixed_element_value_must_be_matched() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="n" type="xs:decimal" fixed="1.0"/>
           </xs:schema>"#,
    );
    assert!(
        schema.validate_str("<n>1.00</n>").valid,
        "value-space equality"
    );
    assert!(!schema.validate_str("<n>2</n>").valid);
}

#[test]
fn xsi_type_overrides_the_declared_type() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="Base"><xs:sequence/></xs:complexType>
             <xs:complexType name="Derived">
               <xs:complexContent>
                 <xs:extension base="Base">
                   <xs:sequence><xs:element name="extra" type="xs:string"/></xs:sequence>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="r" type="Base"/>
           </xs:schema>"#,
    );
    let instance = r#"<r xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                         xsi:type="Derived"><extra>x</extra></r>"#;
    assert!(schema.validate_str(instance).valid);
    assert!(!schema.validate_str("<r><extra>x</extra></r>").valid);
    let unknown = r#"<r xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                        xsi:type="Nope"/>"#;
    assert_eq!(code(&schema.validate_str(unknown)), Some("cvc-elt.4.2"));
}

/// Part 1 §3.14.6 *Type Derivation OK (Simple)* clause 3.3.3: a type is
/// validly derived from a union if it is validly derived from *any one* of
/// the union's member types, which is what lets `xsi:type` name a union
/// member directly rather than only the union itself
/// (`union/s3_16_2v05`, `s3_16_2v07`).
#[test]
fn xsi_type_may_name_a_union_member_directly() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="Choice">
               <xs:union memberTypes="xs:int xs:boolean"/>
             </xs:simpleType>
             <xs:element name="r" type="Choice"/>
           </xs:schema>"#,
    );
    let instance = r#"<r xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                         xsi:type="xs:int">42</r>"#;
    assert!(schema.validate_str(instance).valid);
    let other_member = r#"<r xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                             xsi:type="xs:boolean">true</r>"#;
    assert!(schema.validate_str(other_member).valid);
    let not_a_member = r#"<r xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                             xsi:type="xs:string">hello</r>"#;
    assert_eq!(
        code(&schema.validate_str(not_a_member)),
        Some("cvc-elt.4.3")
    );
}

/// XSD 1.1 §3.3.2: an explicit `targetNamespace` on a *local* element or
/// attribute declaration names it in a different namespace than the
/// enclosing schema document's own — restricted to where a restriction is
/// entitled to narrow a wildcard into a named declaration
/// (`TargetNS/target001`, `constraintsOnAttribute/s3_2_3v01`).
#[test]
fn a_local_declarations_own_target_namespace_overrides_the_schemas() {
    let schema = schema_1_1(
        r###"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                       xmlns:x="http://x.example/" targetNamespace="http://x.example/"
                       elementFormDefault="qualified">
             <xs:complexType name="B">
               <xs:sequence>
                 <xs:any namespace="##other" processContents="lax"/>
               </xs:sequence>
             </xs:complexType>
             <xs:element name="r">
               <xs:complexType>
                 <xs:complexContent>
                   <xs:restriction base="x:B">
                     <xs:sequence>
                       <xs:element name="child" targetNamespace="http://y.example/"/>
                     </xs:sequence>
                   </xs:restriction>
                 </xs:complexContent>
               </xs:complexType>
             </xs:element>
           </xs:schema>"###,
    );
    let instance =
        r#"<x:r xmlns:x="http://x.example/" xmlns:y="http://y.example/"><y:child/></x:r>"#;
    assert!(schema.validate_str(instance).valid);
    // The element is in `y`'s namespace, not the schema's own `x`.
    let wrong_namespace = r#"<x:r xmlns:x="http://x.example/"><child/></x:r>"#;
    assert!(!schema.validate_str(wrong_namespace).valid);
}

#[test]
fn xsi_nil_needs_a_nillable_declaration() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a" type="xs:int" nillable="true"/>
             <xs:element name="b" type="xs:int"/>
           </xs:schema>"#,
    );
    let xsi = r#"xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#;
    assert!(
        schema
            .validate_str(&format!(r#"<a {xsi} xsi:nil="true"/>"#))
            .valid
    );
    assert_eq!(
        code(&schema.validate_str(&format!(r#"<a {xsi} xsi:nil="true">1</a>"#))),
        Some("cvc-elt.3.2.1")
    );
    assert_eq!(
        code(&schema.validate_str(&format!(r#"<b {xsi} xsi:nil="true"/>"#))),
        Some("cvc-elt.3.1")
    );
}

#[test]
fn substitution_groups_stand_in_for_their_head() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="head" type="xs:string"/>
             <xs:element name="member" type="xs:string" substitutionGroup="head"/>
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence><xs:element ref="head"/></xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<r><head>x</head></r>").valid);
    assert!(schema.validate_str("<r><member>x</member></r>").valid);
    assert!(!schema.validate_str("<r><other>x</other></r>").valid);
}

#[test]
fn wildcards_honour_their_process_contents() {
    let schema = schema(
        r###"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="urn:x" xmlns="urn:x" elementFormDefault="qualified">
             <xs:element name="strict">
               <xs:complexType>
                 <xs:sequence><xs:any namespace="##other" processContents="strict"/></xs:sequence>
               </xs:complexType>
             </xs:element>
             <xs:element name="skip">
               <xs:complexType>
                 <xs:sequence><xs:any namespace="##other" processContents="skip"/></xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"###,
    );
    let strict = r#"<strict xmlns="urn:x"><o:x xmlns:o="urn:o"/></strict>"#;
    assert_eq!(
        code(&schema.validate_str(strict)),
        Some("cvc-complex-type.2.4.c")
    );
    let skip = r#"<skip xmlns="urn:x"><o:x xmlns:o="urn:o">anything</o:x></skip>"#;
    assert!(schema.validate_str(skip).valid);
    // `##other` excludes the target namespace itself.
    let wrong = r#"<skip xmlns="urn:x"><y/></skip>"#;
    assert!(!schema.validate_str(wrong).valid);
}

#[test]
fn identity_constraints_are_evaluated() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="db">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="row" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="id" type="xs:int" use="required"/>
                       <xs:attribute name="parent" type="xs:int"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:key name="rowKey">
                 <xs:selector xpath="row"/>
                 <xs:field xpath="@id"/>
               </xs:key>
               <xs:keyref name="rowRef" refer="rowKey">
                 <xs:selector xpath="row"/>
                 <xs:field xpath="@parent"/>
               </xs:keyref>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(
        schema
            .validate_str(r#"<db><row id="1"/><row id="2" parent="1"/></db>"#)
            .valid
    );
    assert_eq!(
        code(&schema.validate_str(r#"<db><row id="1"/><row id="1"/></db>"#)),
        Some("cvc-identity-constraint.4.2.2")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<db><row id="1" parent="9"/></db>"#)),
        Some("cvc-identity-constraint.4.3")
    );
    // Keys compare in the value space: `01` and `1` are the same `xs:int`.
    assert_eq!(
        code(&schema.validate_str(r#"<db><row id="1"/><row id="01"/></db>"#)),
        Some("cvc-identity-constraint.4.2.2")
    );
}

#[test]
fn unique_allows_absent_fields_where_key_does_not() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="db">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="row" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="id" type="xs:string"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:unique name="rowUnique">
                 <xs:selector xpath=".//row"/>
                 <xs:field xpath="@id"/>
               </xs:unique>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str(r#"<db><row/><row/></db>"#).valid);
    assert!(
        !schema
            .validate_str(r#"<db><row id="a"/><row id="a"/></db>"#)
            .valid
    );
}

#[test]
fn id_and_idref_are_tracked_across_the_document() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="n" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="id" type="xs:ID"/>
                       <xs:attribute name="see" type="xs:IDREF"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(
        schema
            .validate_str(r#"<r><n id="a"/><n see="a"/></r>"#)
            .valid
    );
    assert_eq!(
        code(&schema.validate_str(r#"<r><n id="a"/><n id="a"/></r>"#)),
        Some("cvc-id.2")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<r><n see="missing"/></r>"#)),
        Some("cvc-id.1")
    );
}

#[test]
fn qname_values_are_read_in_their_namespace_context() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="q" type="xs:QName"/>
           </xs:schema>"#,
    );
    assert!(
        schema
            .validate_str(r#"<q xmlns:p="urn:x">p:name</q>"#)
            .valid
    );
    assert!(!schema.validate_str("<q>p:name</q>").valid);
}

#[test]
fn an_abstract_element_cannot_appear() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a" type="xs:string" abstract="true"/>
             <xs:element name="b" type="xs:string" substitutionGroup="a"/>
             <xs:element name="r">
               <xs:complexType><xs:sequence><xs:element ref="a"/></xs:sequence></xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<r><b>x</b></r>").valid);
    assert_eq!(
        code(&schema.validate_str("<r><a>x</a></r>")),
        Some("cvc-elt.2")
    );
}

#[test]
fn an_undeclared_root_is_reported() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="known" type="xs:string"/>
           </xs:schema>"#,
    );
    assert_eq!(code(&schema.validate_str("<unknown/>")), Some("cvc-elt.1"));
}

#[test]
fn a_malformed_instance_is_an_error_not_a_panic() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a" type="xs:string"/>
           </xs:schema>"#,
    );
    let outcome = schema.validate_str("<a><unclosed></a>");
    assert!(!outcome.valid);
    assert_eq!(code(&outcome), Some("cvc-document"));
}

#[test]
fn empty_content_rejects_children() -> Result<(), SchemaError> {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e">
               <xs:complexType><xs:attribute name="a" type="xs:string"/></xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str(r#"<e a="1"/>"#).valid);
    assert_eq!(
        code(&schema.validate_str("<e>text</e>")),
        Some("cvc-complex-type.2.1")
    );
    assert_eq!(
        code(&schema.validate_str("<e><b/></e>")),
        Some("cvc-complex-type.2.1")
    );
    Ok(())
}

#[test]
fn simple_content_carries_attributes() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="price">
               <xs:complexType>
                 <xs:simpleContent>
                   <xs:extension base="xs:decimal">
                     <xs:attribute name="currency" type="xs:string" use="required"/>
                   </xs:extension>
                 </xs:simpleContent>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(
        schema
            .validate_str(r#"<price currency="JPY">1200</price>"#)
            .valid
    );
    assert!(
        !schema
            .validate_str(r#"<price currency="JPY">free</price>"#)
            .valid
    );
    assert!(!schema.validate_str("<price>1200</price>").valid);
}

#[test]
fn a_keyref_only_matches_the_key_table_of_its_own_scoping_element() {
    // XSD 1.0 §3.11.5: an element's node table for a constraint is assembled *recursively from
    // its own subtree*, so "only element information items within the sub-tree rooted at the
    // element information item being validated can be referenced successfully". A `keyref`
    // scoped to a repeated element must therefore not see a key value that exists only in a
    // sibling occurrence.
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="box" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
             <xs:element name="box">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="item" minOccurs="0" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="id" type="xs:string" use="required"/>
                     </xs:complexType>
                   </xs:element>
                   <xs:element name="ref" minOccurs="0" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="r" type="xs:string" use="required"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:key name="k">
                 <xs:selector xpath="item"/>
                 <xs:field xpath="@id"/>
               </xs:key>
               <xs:keyref name="kr" refer="k">
                 <xs:selector xpath="ref"/>
                 <xs:field xpath="@r"/>
               </xs:keyref>
             </xs:element>
           </xs:schema>"#,
    );
    // Within one box, the reference resolves.
    assert!(
        schema
            .validate_str(r#"<root><box><item id="A"/><ref r="A"/></box></root>"#)
            .valid
    );
    // Across two boxes it must not: box #2's `ref` points at a key that only box #1 declares.
    assert_eq!(
        code(&schema.validate_str(
            r#"<root><box><item id="A"/><ref r="A"/></box><box><item id="B"/><ref r="A"/></box></root>"#
        )),
        Some("cvc-identity-constraint.4.3")
    );
    // The same key value declared independently in each box stays valid — the tables are
    // per-occurrence, not pooled.
    assert!(
        schema
            .validate_str(
                r#"<root><box><item id="A"/><ref r="A"/></box><box><item id="A"/><ref r="A"/></box></root>"#
            )
            .valid
    );
}

#[test]
fn a_key_declared_on_an_ancestor_is_still_visible_to_a_descendant_scope() {
    // The recursive assembly of §3.11.5 propagates a descendant's node table *upward*, so a
    // `keyref` on an outer element does see keys collected inside it.
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="box" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:sequence>
                         <xs:element name="item" maxOccurs="unbounded">
                           <xs:complexType>
                             <xs:attribute name="id" type="xs:string" use="required"/>
                           </xs:complexType>
                         </xs:element>
                       </xs:sequence>
                     </xs:complexType>
                   </xs:element>
                   <xs:element name="ref" minOccurs="0">
                     <xs:complexType>
                       <xs:attribute name="r" type="xs:string" use="required"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:key name="k">
                 <xs:selector xpath=".//item"/>
                 <xs:field xpath="@id"/>
               </xs:key>
               <xs:keyref name="kr" refer="k">
                 <xs:selector xpath="ref"/>
                 <xs:field xpath="@r"/>
               </xs:keyref>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(
        schema
            .validate_str(
                r#"<root><box><item id="A"/></box><box><item id="B"/></box><ref r="B"/></root>"#
            )
            .valid
    );
    assert_eq!(
        code(&schema.validate_str(r#"<root><box><item id="A"/></box><ref r="Z"/></root>"#)),
        Some("cvc-identity-constraint.4.3")
    );
}

#[test]
fn a_field_selecting_an_element_with_complex_content_is_an_error() {
    // XSD 1.0 §3.11.4 clause 3: a `field` must select a node with a *simple* type. Falling back
    // to the element's concatenated character content would silently accept a schema that the
    // Recommendation rejects.
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="i" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:sequence>
                         <xs:element name="k">
                           <xs:complexType>
                             <xs:sequence>
                               <xs:element name="deep" type="xs:string"/>
                             </xs:sequence>
                           </xs:complexType>
                         </xs:element>
                       </xs:sequence>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:unique name="u">
                 <xs:selector xpath="i"/>
                 <xs:field xpath="k"/>
               </xs:unique>
             </xs:element>
           </xs:schema>"#,
    );
    assert_eq!(
        code(
            &schema
                .validate_str(r#"<r><i><k><deep>a</deep></k></i><i><k><deep>b</deep></k></i></r>"#)
        ),
        Some("cvc-identity-constraint.3")
    );
}

/// A recursive schema: `a` may nest inside `a` without limit.
const RECURSIVE: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
     <xs:element name="a">
       <xs:complexType>
         <xs:sequence><xs:element ref="a" minOccurs="0"/></xs:sequence>
       </xs:complexType>
     </xs:element>
   </xs:schema>"#;

/// Builds `<a><a>…</a></a>` nested `levels` deep through the DOM's own
/// mutation API, which is the route that bypasses the parser's nesting cap.
fn nested_document(levels: usize) -> Document {
    let mut doc = Document::new();
    let root = doc.root();
    let mut parent = root;
    for _ in 0..levels {
        let element = doc.create_element("a").expect("an element");
        doc.append_child(parent, element).expect("a child");
        parent = element;
    }
    doc
}

fn first_element_child(doc: &Document, node: NodeId) -> NodeId {
    doc.children(node)
        .find(|child| doc.kind(*child) == Some(NodeKind::Element))
        .expect("an element child")
}

#[test]
fn a_document_nested_past_the_maximum_depth_is_reported_not_fatal() {
    // A `Document` assembled through `create_element`/`append_child` never
    // passes the parser, so nothing else bounds its depth: without a bound of
    // its own, assessment would recurse once per level and overflow the stack,
    // which aborts the process rather than raising anything a caller could
    // catch.
    let schema = schema(RECURSIVE);
    let doc = nested_document(20_000);
    let outcome = schema.validate_document(&doc);
    assert!(!outcome.valid);
    assert!(
        outcome
            .errors
            .iter()
            .any(|error| error.code() == "cvc-max-depth"),
        "expected a depth error, got {:?}",
        outcome.errors.first().map(ToString::to_string)
    );
    // Everything above the cut-off is still assessed, so the PSVI is complete
    // up to the bound rather than empty.
    assert_eq!(
        outcome.psvi.assessed_element_count(),
        oxixml_schema::validate::MAX_DEPTH
    );
}

#[test]
fn a_document_within_the_maximum_depth_is_assessed_as_usual() {
    let schema = schema(RECURSIVE);
    let doc = nested_document(oxixml_schema::validate::MAX_DEPTH);
    let outcome = schema.validate_document(&doc);
    assert!(outcome.valid, "{:?}", outcome.errors.first());
}

#[test]
fn an_assertion_over_a_document_nested_past_the_maximum_depth_is_reported_not_fatal() {
    // The XDM view an `xs:assert` is evaluated against is a *copy* of the
    // subtree, written by its own recursion, so it needs the same bound: an
    // assertion near the root of a deep document would otherwise walk the whole
    // depth however early assessment itself stopped.
    let schema = schema_1_1(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a">
               <xs:complexType>
                 <xs:sequence><xs:element ref="a" minOccurs="0"/></xs:sequence>
                 <xs:assert test="true()"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    let doc = nested_document(20_000);
    let outcome = schema.validate_document(&doc);
    assert!(!outcome.valid);
    assert!(
        outcome
            .errors
            .iter()
            .any(|error| error.code() == "cvc-assertion"),
        "expected the view to be refused, got {:?}",
        outcome.errors.first().map(ToString::to_string)
    );
}

#[test]
fn an_invalid_element_says_so_in_its_own_psvi_entry() {
    // Part 1 §3.3.4: an element is *valid* only when it is locally valid and
    // every attribute and element child of it is valid, so the failure below
    // has to reach the root's own [validity] as well as `n`'s.
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence><xs:element name="n" type="xs:int"/></xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );

    let good = oxixml_dom::build::parse_str("<r><n>42</n></r>").expect("well-formed");
    let outcome = schema.validate_document(&good);
    let root = good.document_element().expect("a root");
    let child = first_element_child(&good, root);
    assert!(outcome.valid);
    assert_eq!(
        outcome.psvi.element(root).map(|psvi| psvi.validity),
        Some(Validity::Valid)
    );
    assert_eq!(
        outcome.psvi.element(child).map(|psvi| psvi.validity),
        Some(Validity::Valid)
    );

    let bad = oxixml_dom::build::parse_str("<r><n>oops</n></r>").expect("well-formed");
    let outcome = schema.validate_document(&bad);
    let root = bad.document_element().expect("a root");
    let child = first_element_child(&bad, root);
    assert!(!outcome.valid);
    assert_eq!(
        outcome.psvi.element(child).map(|psvi| psvi.validity),
        Some(Validity::Invalid)
    );
    assert_eq!(
        outcome.psvi.element(root).map(|psvi| psvi.validity),
        Some(Validity::Invalid),
        "the failure of a child makes the parent invalid too"
    );
}

#[test]
fn an_identity_constraint_failure_reaches_the_psvi() {
    // Identity constraints are evaluated after the whole document has been
    // walked, so their failures have to be attributed to a node explicitly
    // rather than by where they happened to be raised.
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="i" maxOccurs="unbounded">
                     <xs:complexType><xs:attribute name="k" type="xs:string"/></xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
               <xs:unique name="u">
                 <xs:selector xpath="i"/>
                 <xs:field xpath="@k"/>
               </xs:unique>
             </xs:element>
           </xs:schema>"#,
    );
    let doc = oxixml_dom::build::parse_str(r#"<r><i k="x"/><i k="x"/></r>"#).expect("well-formed");
    let outcome = schema.validate_document(&doc);
    assert_eq!(code(&outcome), Some("cvc-identity-constraint.4.1"));
    let root = doc.document_element().expect("a root");
    let duplicate = doc
        .children(root)
        .filter(|child| doc.kind(*child) == Some(NodeKind::Element))
        .nth(1)
        .expect("a second `i`");
    assert_eq!(
        outcome.psvi.element(duplicate).map(|psvi| psvi.validity),
        Some(Validity::Invalid)
    );
    assert_eq!(
        outcome.psvi.element(root).map(|psvi| psvi.validity),
        Some(Validity::Invalid)
    );
}

#[test]
fn a_dangling_idref_reaches_the_psvi() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="r">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="i">
                     <xs:complexType><xs:attribute name="ref" type="xs:IDREF"/></xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    let doc = oxixml_dom::build::parse_str(r#"<r><i ref="nowhere"/></r>"#).expect("well-formed");
    let outcome = schema.validate_document(&doc);
    assert_eq!(code(&outcome), Some("cvc-id.1"));
    let root = doc.document_element().expect("a root");
    let item = first_element_child(&doc, root);
    assert_eq!(
        outcome.psvi.element(item).map(|psvi| psvi.validity),
        Some(Validity::Invalid)
    );
    assert_eq!(
        outcome.psvi.element(root).map(|psvi| psvi.validity),
        Some(Validity::Invalid)
    );
}

#[test]
fn the_psvi_records_the_union_member_type_that_accepted_a_value() {
    // The *[member type definition]* property: without it a caller cannot tell
    // which branch of a union governed the value it is looking at.
    let union = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="u">
               <xs:union memberTypes="xs:int xs:date"/>
             </xs:simpleType>
             <xs:element name="v">
               <xs:complexType>
                 <xs:simpleContent>
                   <xs:extension base="u">
                     <xs:attribute name="w" type="u"/>
                   </xs:extension>
                 </xs:simpleContent>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    let doc = oxixml_dom::build::parse_str(r#"<v w="2001-01-01">42</v>"#).expect("well-formed");
    let outcome = union.validate_document(&doc);
    assert!(outcome.valid, "{:?}", outcome.errors.first());
    let root = doc.document_element().expect("a root");
    assert_eq!(
        outcome.psvi.element(root).and_then(|psvi| psvi.member_type),
        union.builtin("int")
    );
    let attribute = doc.attributes(root).next().expect("an attribute");
    assert_eq!(
        outcome
            .psvi
            .attribute(attribute)
            .and_then(|psvi| psvi.member_type),
        union.builtin("date")
    );
    // A value whose type is not a union has no member type at all.
    let plain = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="n" type="xs:int"/>
           </xs:schema>"#,
    );
    let doc = oxixml_dom::build::parse_str("<n>42</n>").expect("well-formed");
    let outcome = plain.validate_document(&doc);
    let root = doc.document_element().expect("a root");
    assert_eq!(
        outcome.psvi.element(root).and_then(|psvi| psvi.member_type),
        None
    );
}

#[test]
fn an_abstract_substitution_group_member_does_not_match_under_1_0() {
    // XSD 1.0's *actual* substitution group leaves an abstract member out, so
    // a particle referring to the head does not match that name and the local
    // declaration of the same name is free to (`wgData/sg/upa.xsd`, whose
    // instances `e1.xml` and `e1token.xml` are valid under 1.0 for exactly this
    // reason). Attributing the element to the abstract global declaration
    // instead would report it as abstract and reject a valid document.
    let competing = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e" type="xs:string"/>
             <xs:element name="e1" substitutionGroup="e" abstract="true"/>
             <xs:element name="test">
               <xs:complexType>
                 <xs:choice>
                   <xs:element ref="e"/>
                   <xs:element name="e1" type="xs:string"/>
                 </xs:choice>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    let outcome = competing.validate_str("<test><e1>hello</e1></test>");
    assert!(outcome.valid, "{:?}", outcome.errors.first());
    assert!(competing.validate_str("<test><e>hello</e></test>").valid);
    // An element that names the abstract declaration where *nothing else* can
    // match it is still rejected.
    let alone = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="e" type="xs:string"/>
             <xs:element name="e1" substitutionGroup="e" abstract="true"/>
             <xs:element name="test">
               <xs:complexType>
                 <xs:sequence><xs:element ref="e"/></xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(!alone.validate_str("<test><e1>hello</e1></test>").valid);
}
