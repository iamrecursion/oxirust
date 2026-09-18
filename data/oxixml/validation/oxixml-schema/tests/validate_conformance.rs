//! Assessment rules the W3C XML Schema Test Suite exercises through
//! `msData`, `sunData` and `wgData` fixtures: the identity-constraint view of
//! the augmented infoset, `{disallowed substitutions}`, the `fixed` value
//! constraint of *cvc-elt* clause 5.2.2, and the two `ID` rules XSD 1.0
//! imposes and XSD 1.1 drops.
//!
//! Each test names the suite group it reproduces and carries the same schema
//! and instance in miniature, so a failure says which conformance case broke
//! without needing the suite on disk.

use oxixml_schema::{Schema, SchemaSet, ValidationOutcome, XsdVersion};

fn schema(text: &str) -> Schema {
    let mut set = SchemaSet::new();
    set.add_document(None, text).expect("a schema document");
    set.compile().expect("a compiled schema")
}

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

/// The schema of `msData/identityConstraint/idF016` (a `default`) and
/// `idF017`/`idG012` (a `fixed`), which differ only in that one word.
fn unique_over_defaulted_attribute(constraint: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="uid" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
               <xs:unique name="uuid">
                 <xs:selector xpath=".//uid"/>
                 <xs:field xpath="@val"/>
               </xs:unique>
             </xs:element>
             <xs:element name="uid">
               <xs:complexType>
                 <xs:attribute name="val" type="xs:string" {constraint}="test"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )
}

/// `msData/identityConstraint/idF016` and `idG011`: one `uid` writes
/// `val="test"`, the next leaves it to the declaration's `default`. §3.4.5
/// adds the defaulted attribute to the element's \[attributes\], so the
/// `unique` sees the same key twice.
#[test]
fn id_f016_a_defaulted_attribute_is_visible_to_a_field() {
    let schema = schema(&unique_over_defaulted_attribute("default"));
    assert_eq!(
        code(&schema.validate_str(r#"<root><uid val="test"/><uid/></root>"#)),
        Some("cvc-identity-constraint.4.1")
    );
    assert!(
        schema
            .validate_str(r#"<root><uid val="other"/><uid/></root>"#)
            .valid,
        "the default only clashes with an equal written value"
    );
}

/// `msData/identityConstraint/idF017` and `idG012`: a `fixed` value
/// constraint defaults an absent attribute exactly as `default` does.
#[test]
fn id_f017_a_fixed_value_constraint_defaults_the_attribute_too() {
    let schema = schema(&unique_over_defaulted_attribute("fixed"));
    assert_eq!(
        code(&schema.validate_str(r#"<root><uid val="test"/><uid/></root>"#)),
        Some("cvc-identity-constraint.4.1")
    );
}

/// `msData/identityConstraint/idZ011_a`: neither element writes the
/// attribute, so both take the same default and collide with each other.
#[test]
fn id_z011_a_two_defaults_collide_with_each_other() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence><xs:element ref="e" maxOccurs="2"/></xs:sequence>
               </xs:complexType>
               <xs:unique name="idName">
                 <xs:selector xpath=".//e"/>
                 <xs:field xpath="@att"/>
               </xs:unique>
             </xs:element>
             <xs:element name="e">
               <xs:complexType>
                 <xs:attribute name="att" type="xs:string" default="a"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert_eq!(
        code(&schema.validate_str("<root><e/><e/></root>")),
        Some("cvc-identity-constraint.4.1")
    );
    assert!(
        schema
            .validate_str(r#"<root><e att="b"/><e/></root>"#)
            .valid,
        "a written value that differs from the default is no duplicate"
    );
}

/// `msData/identityConstraint/idF018`: `xsi:nil` empties an element's content
/// but does not change its type, so a field selecting a nilled element
/// governed by `xs:anyType` selects a node that has no simple type — which
/// §3.11.4 clause 3 forbids.
#[test]
fn id_f018_a_nilled_element_keeps_its_complex_type() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="uid" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
               <xs:unique name="uuid">
                 <xs:selector xpath=".//uid"/>
                 <xs:field xpath="."/>
               </xs:unique>
             </xs:element>
             <xs:element name="uid" nillable="true"/>
           </xs:schema>"#,
    );
    assert_eq!(
        code(&schema.validate_str(
            r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <uid xsi:nil="true" xsi:type="xs:string"/>
                 <uid xsi:nil="true"/>
               </root>"#
        )),
        Some("cvc-identity-constraint.3")
    );
}

/// `msData/identityConstraint/idZ015`: the only thing that lets `bar` onto
/// `foo` is a `processContents="skip"` wildcard, so the attribute is
/// ·skipped· and takes no part in the constraint — leaving the `key` a field
/// short. W3C bug 4063 settled this reading for XSD 1.0.
#[test]
fn id_z015_a_field_may_not_resolve_to_a_skipped_attribute() {
    let schema = schema(
        r###"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
               <xs:element name="foo">
                 <xs:complexType>
                   <xs:anyAttribute namespace="##any" processContents="skip"/>
                 </xs:complexType>
                 <xs:key name="aKey">
                   <xs:selector xpath="."/>
                   <xs:field xpath="@bar"/>
                 </xs:key>
               </xs:element>
             </xs:schema>"###,
    );
    assert_eq!(
        code(&schema.validate_str(r#"<foo bar="abc"/>"#)),
        Some("cvc-identity-constraint.4.2.1")
    );
}

/// The schema of `sunData/ElemDecl/disallowedSubst/disallowedSubst00503m3`
/// (`block="restriction"`), `…m4` and `…m5` (`block="restriction extension"`),
/// parameterized on what the *head's type* prohibits.
fn substitution_blocked_by_head_type(block: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:t="urn:disallowed" targetNamespace="urn:disallowed"
                      elementFormDefault="qualified">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="t:Head" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
             <xs:element name="Head" type="t:Type"/>
             <xs:complexType name="Type" block="{block}"/>
             <xs:complexType name="derivedFromType">
               <xs:complexContent><xs:restriction base="t:Type"/></xs:complexContent>
             </xs:complexType>
             <xs:element name="Member1" type="t:derivedFromType" substitutionGroup="t:Head"/>
           </xs:schema>"#
    )
}

const SUBSTITUTED: &str = r#"<t:root xmlns:t="urn:disallowed"><t:Member1/></t:root>"#;

/// `disallowedSubst00503m3` … `m5`: the head element blocks nothing, but its
/// *type* prohibits `restriction`, and §3.3.6 clause 3 unions that in — so the
/// member derived by restriction may not stand in for the head.
#[test]
fn disallowed_subst_00503_the_head_types_prohibited_substitutions_count() {
    for block in ["restriction", "restriction extension", "#all"] {
        let schema = schema(&substitution_blocked_by_head_type(block));
        assert_eq!(
            code(&schema.validate_str(SUBSTITUTED)),
            Some("cvc-complex-type.2.4.a"),
            "block=\"{block}\" must block the substitution"
        );
    }
    let schema = schema(&substitution_blocked_by_head_type("extension"));
    assert!(
        schema.validate_str(SUBSTITUTED).valid,
        "prohibiting only `extension` leaves a restriction substitutable"
    );
}

/// §3.4.1 confines a *type*'s {prohibited substitutions} to a subset of
/// `{extension, restriction}`: `block="#all"` on the head's type may not stop
/// a member that shares the head's type outright, since no derivation method
/// is involved at all.
#[test]
fn a_head_types_block_all_does_not_forbid_substitution_itself() {
    let schema = schema(
        r###"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                        xmlns:t="urn:disallowed" targetNamespace="urn:disallowed"
                        elementFormDefault="qualified">
               <xs:element name="root">
                 <xs:complexType>
                   <xs:sequence>
                     <xs:element ref="t:Head" maxOccurs="unbounded"/>
                   </xs:sequence>
                 </xs:complexType>
               </xs:element>
               <xs:element name="Head" type="t:Type"/>
               <xs:complexType name="Type" block="#all"/>
               <xs:element name="Member1" type="t:Type" substitutionGroup="t:Head"/>
             </xs:schema>"###,
    );
    assert!(schema.validate_str(SUBSTITUTED).valid);
}

/// `sunData/ElemDecl/valueConstraint/valueConstraint00701m1`: *cvc-elt*
/// clause 5.2.2.1 — an element whose declaration fixes its value may have no
/// element children, whatever its content type allows.
#[test]
fn value_constraint_00701m1_a_fixed_value_admits_no_element_children() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root" fixed="part1 part2">
               <xs:complexType mixed="true">
                 <xs:sequence minOccurs="0" maxOccurs="unbounded">
                   <xs:element name="separator" minOccurs="0" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<root>part1 part2</root>").valid);
    assert_eq!(
        code(&schema.validate_str("<root>part1 <separator/>part2</root>")),
        Some("cvc-elt.5.2.2.1")
    );
}

/// `sunData/ElemDecl/valueConstraint/valueConstraint00801m1`: clause
/// 5.2.2.2.1 — with a *mixed* content type the ·initial value· of the item
/// must match the fixed value literally.
#[test]
fn value_constraint_00801m1_mixed_content_matches_the_fixed_value_literally() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:t="urn:vc" targetNamespace="urn:vc">
             <xs:element name="root" fixed="alpha beta"/>
             <xs:complexType name="Text" mixed="true"/>
           </xs:schema>"#,
    );
    let instance = |content: &str| {
        format!(
            r#"<t:root xmlns:t="urn:vc"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:type="t:Text">{content}</t:root>"#
        )
    };
    assert!(schema.validate_str(&instance("alpha beta")).valid);
    assert_eq!(
        code(&schema.validate_str(&instance("beta alpha"))),
        Some("cvc-elt.5.2.2.2.1")
    );
}

/// `msData/complexType/ctZ013c`, `ctZ013d` (an element child before or after
/// the text) and `ctZ013e` (the wrong text).
#[test]
fn ct_z013_a_fixed_value_on_a_mixed_type() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:ns="ns" targetNamespace="ns">
             <xs:complexType name="CT" mixed="true">
               <xs:sequence>
                 <xs:element name="a" minOccurs="0" type="xs:byte"/>
               </xs:sequence>
             </xs:complexType>
             <xs:element name="e" type="ns:CT" fixed="abc"/>
           </xs:schema>"#,
    );
    assert!(schema.validate_str(r#"<e xmlns="ns">abc</e>"#).valid);
    assert_eq!(
        code(&schema.validate_str(r#"<e xmlns="ns">abc<a xmlns="">1</a></e>"#)),
        Some("cvc-elt.5.2.2.1")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<e xmlns="ns"><a xmlns="">1</a>abc</e>"#)),
        Some("cvc-elt.5.2.2.1")
    );
    assert_eq!(
        code(&schema.validate_str(r#"<e xmlns="ns">def</e>"#)),
        Some("cvc-elt.5.2.2.2.1")
    );
}

/// `msData/additional/isdefault070`: the fixed element has no type of its
/// own, so it is governed by `xs:anyType`, whose content type is mixed.
#[test]
fn is_default_070_a_fixed_value_on_an_any_type_element() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root" type="ct"/>
             <xs:complexType name="ct">
               <xs:sequence><xs:element name="a" fixed="fixed_value"/></xs:sequence>
             </xs:complexType>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<root><a>fixed_value</a></root>").valid);
    assert_eq!(
        code(&schema.validate_str("<root><a>not fixed</a></root>")),
        Some("cvc-elt.5.2.2.2.1")
    );
}

/// `msData/additional/isdefault076`, the instance of group `isDefault077`.
#[test]
fn is_default_077_a_fixed_value_on_a_mixed_declaration() {
    let schema = schema(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root" fixed="abc">
               <xs:complexType mixed="true">
                 <xs:sequence minOccurs="0">
                   <xs:element name="e1"/>
                   <xs:element name="e2"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    );
    assert!(schema.validate_str("<root>abc</root>").valid);
    assert_eq!(
        code(&schema.validate_str("<root>not_fixed</root>")),
        Some("cvc-elt.5.2.2.2.1")
    );
}

/// The schema of `msData/element/elemT074`, parameterized on what the element
/// blocks: `xsi:type` names a restriction of one of the union's member types,
/// which *Type Derivation OK (Simple)* clause 2.2.4 makes a valid derivation
/// from the union — by restriction, and therefore blocked.
fn union_typed_element(block: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="test" block="{block}" type="Union-AB"/>
             <xs:simpleType name="A">
               <xs:restriction base="xs:int">
                 <xs:enumeration value="1"/>
                 <xs:enumeration value="2"/>
               </xs:restriction>
             </xs:simpleType>
             <xs:simpleType name="B">
               <xs:restriction base="xs:string"><xs:enumeration value="a"/></xs:restriction>
             </xs:simpleType>
             <xs:simpleType name="Union-AB"><xs:union memberTypes="A B"/></xs:simpleType>
             <xs:simpleType name="R-A">
               <xs:restriction base="A"><xs:enumeration value="1"/></xs:restriction>
             </xs:simpleType>
           </xs:schema>"#
    )
}

const UNION_OVERRIDE: &str = r#"<test xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                                      xsi:type="R-A">1</test>"#;

/// `msData/element/elemT074`.
#[test]
fn elem_t074_xsi_type_through_a_union_member_obeys_block() {
    assert_eq!(
        code(&schema(&union_typed_element("restriction")).validate_str(UNION_OVERRIDE)),
        Some("cvc-elt.4.3")
    );
    assert!(
        schema(&union_typed_element("extension"))
            .validate_str(UNION_OVERRIDE)
            .valid,
        "only `restriction` is used along the route, so blocking `extension` says nothing"
    );
}

const ELEM_Z016_SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
     <xs:element name="Root">
       <xs:complexType>
         <xs:sequence>
           <xs:element ref="ID"/>
           <xs:element ref="ID"/>
         </xs:sequence>
       </xs:complexType>
     </xs:element>
     <xs:element name="ID" type="xs:ID"/>
   </xs:schema>"#;

const ELEM_Z016_INSTANCE: &str = "<Root><ID>Address</ID><ID>Address</ID></Root>";

/// `msData/element/elemZ016`: two `xs:ID`-typed children of one element carry
/// the same value. XSD 1.0 wanted every `ID` in a document to be distinct;
/// XSD 1.1 counts the ·binding·, and both children bind the same `Root`.
#[test]
fn elem_z016_repeated_id_values_split_by_version() {
    assert_eq!(
        code(&schema(ELEM_Z016_SCHEMA).validate_str(ELEM_Z016_INSTANCE)),
        Some("cvc-id.2")
    );
    assert!(
        schema_1_1(ELEM_Z016_SCHEMA)
            .validate_str(ELEM_Z016_INSTANCE)
            .valid
    );
}

const ATT_Z014A_SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
     <xs:complexType name="base">
       <xs:attribute name="att1" type="xs:string"/>
       <xs:anyAttribute/>
     </xs:complexType>
     <xs:element name="doc" type="base"/>
     <xs:attribute name="a" type="xs:ID"/>
     <xs:attribute name="b" type="st"/>
     <xs:simpleType name="st"><xs:restriction base="xs:ID"/></xs:simpleType>
   </xs:schema>"#;

const ATT_Z014A_INSTANCE: &str = r#"<doc att1="a" a="x" b="y"/>"#;

/// `msData/attribute/attZ014a`: two attributes of a type derived from `ID`
/// reach the element through its wildcard. *cvc-complex-type* clause 5.1
/// allows at most one such ·wild ID· under XSD 1.0; XSD 1.1 dropped the whole
/// clause.
#[test]
fn att_z014a_two_wild_ids_split_by_version() {
    assert_eq!(
        code(&schema(ATT_Z014A_SCHEMA).validate_str(ATT_Z014A_INSTANCE)),
        Some("cvc-complex-type.5.1")
    );
    assert!(
        schema_1_1(ATT_Z014A_SCHEMA)
            .validate_str(ATT_Z014A_INSTANCE)
            .valid
    );
}

const ATT_Z014B_SCHEMA: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
     <xs:complexType name="base">
       <xs:attribute name="att1" type="xs:string"/>
       <xs:attribute name="att2" type="xs:ID"/>
       <xs:anyAttribute/>
     </xs:complexType>
     <xs:element name="doc" type="base"/>
     <xs:attribute name="b" type="st"/>
     <xs:simpleType name="st"><xs:restriction base="xs:ID"/></xs:simpleType>
   </xs:schema>"#;

const ATT_Z014B_INSTANCE: &str = r#"<doc att1="a" att2="x" b="y"/>"#;

/// `msData/attribute/attZ014b`: one ·wild ID· alongside an `ID` the type
/// declares itself, which clause 5.2 forbids under XSD 1.0.
#[test]
fn att_z014b_a_wild_id_beside_a_declared_one_splits_by_version() {
    assert_eq!(
        code(&schema(ATT_Z014B_SCHEMA).validate_str(ATT_Z014B_INSTANCE)),
        Some("cvc-complex-type.5.2")
    );
    assert!(
        schema_1_1(ATT_Z014B_SCHEMA)
            .validate_str(ATT_Z014B_INSTANCE)
            .valid
    );
}
