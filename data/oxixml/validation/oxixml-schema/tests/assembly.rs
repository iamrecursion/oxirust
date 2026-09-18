//! Schema assembly: the component model, cross-document composition, and the
//! schema component constraints.

use std::collections::HashMap;

use oxixml_schema::components::{ContentType, Derivation, TypeKind, Variety};
use oxixml_schema::{
    ResolvedSchema, Schema, SchemaError, SchemaResolver, SchemaSet, XSD_NS, XsdVersion,
};

/// A resolver that serves schema documents from a map, so that `include`,
/// `import` and `redefine` can be exercised without touching the file system.
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

fn compile(text: &str) -> Result<Schema, SchemaError> {
    let mut set = SchemaSet::new();
    set.add_document(None, text)?;
    set.compile()
}

fn compile_with(resolver: MemoryResolver, text: &str) -> Result<Schema, SchemaError> {
    let mut set = SchemaSet::new().with_resolver(Box::new(resolver));
    set.add_document(None, text)?;
    set.compile()
}

#[test]
fn an_empty_set_yields_the_builtin_hierarchy() -> Result<(), SchemaError> {
    let schema = SchemaSet::new().compile()?;
    for name in ["string", "decimal", "byte", "NCName", "IDREFS", "anyType"] {
        assert!(
            schema.find_type(Some(XSD_NS), name).is_some(),
            "the built-in `{name}` is missing"
        );
    }
    assert!(schema.find_type(Some(XSD_NS), "dateTimeStamp").is_none());
    let one_one = SchemaSet::new().with_version(XsdVersion::V1_1).compile()?;
    assert!(one_one.find_type(Some(XSD_NS), "dateTimeStamp").is_some());
    Ok(())
}

#[test]
fn the_component_model_records_what_the_document_said() -> Result<(), SchemaError> {
    let schema = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="urn:x" xmlns="urn:x"
                      elementFormDefault="qualified">
             <xs:complexType name="Base">
               <xs:sequence>
                 <xs:element name="a" type="xs:string"/>
               </xs:sequence>
               <xs:attribute name="id" type="xs:ID"/>
             </xs:complexType>
             <xs:complexType name="Derived">
               <xs:complexContent>
                 <xs:extension base="Base">
                   <xs:sequence><xs:element name="b" type="xs:int"/></xs:sequence>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="root" type="Derived" nillable="true"/>
           </xs:schema>"#,
    )?;

    let base = schema.find_type(Some("urn:x"), "Base").expect("Base");
    let derived = schema.find_type(Some("urn:x"), "Derived").expect("Derived");
    assert!(schema.is_derived_from(derived, base));
    assert_eq!(
        schema.type_definition(derived).derivation,
        Derivation::Extension
    );
    let complex = schema.complex_type(derived).expect("a complex type");
    assert!(matches!(complex.content, ContentType::ElementOnly(_)));
    // The extension inherits the base's attribute use.
    assert_eq!(complex.attribute_uses.len(), 1);

    let root = schema.find_element(Some("urn:x"), "root").expect("root");
    assert!(schema.element_declaration(root).nillable);
    Ok(())
}

#[test]
fn simple_type_varieties_and_facets() -> Result<(), SchemaError> {
    let schema = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="Digit">
               <xs:restriction base="xs:int">
                 <xs:minInclusive value="0"/>
                 <xs:maxInclusive value="9"/>
               </xs:restriction>
             </xs:simpleType>
             <xs:simpleType name="Digits"><xs:list itemType="Digit"/></xs:simpleType>
             <xs:simpleType name="Either">
               <xs:union memberTypes="Digit xs:string"/>
             </xs:simpleType>
           </xs:schema>"#,
    )?;
    let digit = schema.find_type(None, "Digit").expect("Digit");
    let digits = schema.find_type(None, "Digits").expect("Digits");
    let either = schema.find_type(None, "Either").expect("Either");
    assert!(matches!(
        schema.simple_type(digit).map(|info| &info.variety),
        Some(Variety::Atomic(_))
    ));
    assert_eq!(schema.list_item_type(digits), Some(digit));
    match schema.simple_type(either).map(|info| &info.variety) {
        Some(Variety::Union(members)) => assert_eq!(members.len(), 2),
        other => panic!("expected a union, got {other:?}"),
    }
    // The bounds are merged onto the inherited ones.
    let facets = &schema.simple_type(digit).expect("Digit").facets;
    assert!(facets.min_inclusive.is_some() && facets.max_inclusive.is_some());
    Ok(())
}

#[test]
fn include_composes_one_namespace() -> Result<(), SchemaError> {
    let resolver = MemoryResolver::default().with(
        "part.xsd",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x">
             <xs:simpleType name="Code">
               <xs:restriction base="xs:string"><xs:length value="3"/></xs:restriction>
             </xs:simpleType>
           </xs:schema>"#,
    );
    let schema = compile_with(
        resolver,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="urn:x" xmlns="urn:x">
             <xs:include schemaLocation="part.xsd"/>
             <xs:element name="code" type="Code"/>
           </xs:schema>"#,
    )?;
    assert!(schema.find_type(Some("urn:x"), "Code").is_some());
    assert!(
        schema
            .validate_str(r#"<code xmlns="urn:x">abc</code>"#)
            .valid
    );
    assert!(
        !schema
            .validate_str(r#"<code xmlns="urn:x">abcd</code>"#)
            .valid
    );
    Ok(())
}

#[test]
fn a_chameleon_include_adopts_the_including_namespace() -> Result<(), SchemaError> {
    let resolver = MemoryResolver::default().with(
        "free.xsd",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="Code">
               <xs:restriction base="xs:string"><xs:length value="2"/></xs:restriction>
             </xs:simpleType>
           </xs:schema>"#,
    );
    let schema = compile_with(
        resolver,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="urn:y" xmlns="urn:y">
             <xs:include schemaLocation="free.xsd"/>
             <xs:element name="code" type="Code"/>
           </xs:schema>"#,
    )?;
    assert!(schema.find_type(Some("urn:y"), "Code").is_some());
    assert!(
        schema
            .validate_str(r#"<code xmlns="urn:y">ab</code>"#)
            .valid
    );
    Ok(())
}

#[test]
fn import_crosses_namespaces() -> Result<(), SchemaError> {
    let resolver = MemoryResolver::default().with(
        "other.xsd",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:other">
             <xs:element name="note" type="xs:string"/>
           </xs:schema>"#,
    );
    let schema = compile_with(
        resolver,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:o="urn:other" targetNamespace="urn:x" xmlns="urn:x">
             <xs:import namespace="urn:other" schemaLocation="other.xsd"/>
             <xs:element name="root">
               <xs:complexType><xs:sequence><xs:element ref="o:note"/></xs:sequence></xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )?;
    assert!(
        schema
            .validate_str(r#"<root xmlns="urn:x" xmlns:o="urn:other"><o:note>hi</o:note></root>"#)
            .valid
    );
    Ok(())
}

#[test]
fn redefine_rewrites_a_type_in_terms_of_itself() -> Result<(), SchemaError> {
    let resolver = MemoryResolver::default().with(
        "base.xsd",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x">
             <xs:simpleType name="Code">
               <xs:restriction base="xs:string"><xs:maxLength value="8"/></xs:restriction>
             </xs:simpleType>
           </xs:schema>"#,
    );
    let schema = compile_with(
        resolver,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      targetNamespace="urn:x" xmlns="urn:x">
             <xs:redefine schemaLocation="base.xsd">
               <xs:simpleType name="Code">
                 <xs:restriction base="Code"><xs:maxLength value="3"/></xs:restriction>
               </xs:simpleType>
             </xs:redefine>
             <xs:element name="code" type="Code"/>
           </xs:schema>"#,
    )?;
    assert!(
        schema
            .validate_str(r#"<code xmlns="urn:x">abc</code>"#)
            .valid
    );
    assert!(
        !schema
            .validate_str(r#"<code xmlns="urn:x">abcd</code>"#)
            .valid
    );
    Ok(())
}

#[test]
fn unique_particle_attribution_is_enforced() {
    let error = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:choice>
                     <xs:element name="a" type="xs:string"/>
                     <xs:element name="a" type="xs:string"/>
                   </xs:choice>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )
    .expect_err("an ambiguous content model");
    assert_eq!(error.code(), "cos-nonambig");
}

#[test]
fn repeating_one_particle_is_not_ambiguous() -> Result<(), SchemaError> {
    // `maxOccurs` unrolls into several automaton positions, but they all belong
    // to the same particle, so attribution stays unique.
    compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:choice minOccurs="0" maxOccurs="unbounded">
                   <xs:element name="a" minOccurs="3" maxOccurs="5" type="xs:string"/>
                   <xs:element name="b" minOccurs="3" maxOccurs="5" type="xs:string"/>
                 </xs:choice>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )?;
    Ok(())
}

#[test]
fn element_declarations_must_be_consistent() {
    let error = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="a" type="xs:string"/>
                   <xs:element name="b" minOccurs="0">
                     <xs:complexType>
                       <xs:sequence><xs:element name="c" type="xs:int"/></xs:sequence>
                     </xs:complexType>
                   </xs:element>
                   <xs:element name="a" type="xs:int"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )
    .expect_err("two `a` declarations with different types");
    assert_eq!(error.code(), "cos-element-consistent");
}

#[test]
fn a_recursive_type_is_not_a_cycle() -> Result<(), SchemaError> {
    // An element declaration may name the type it is nested in.
    let schema = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="node" type="Node"/>
             <xs:complexType name="Node">
               <xs:sequence>
                 <xs:element name="child" type="Node" minOccurs="0" maxOccurs="unbounded"/>
               </xs:sequence>
             </xs:complexType>
           </xs:schema>"#,
    )?;
    assert!(
        schema
            .validate_str("<node><child><child/></child></node>")
            .valid
    );
    Ok(())
}

#[test]
fn a_circular_type_definition_is_rejected() {
    let error = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="A"><xs:restriction base="B"/></xs:simpleType>
             <xs:simpleType name="B"><xs:restriction base="A"/></xs:simpleType>
           </xs:schema>"#,
    )
    .expect_err("a cycle");
    assert_eq!(error.code(), "src-type-cycle");
}

#[test]
fn representation_constraints_are_checked() {
    for (text, code) in [
        (
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:element name="a" bogus="1"/>
               </xs:schema>"#,
            "src-element",
        ),
        (
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:element minOccurs="0" name="a"/>
               </xs:schema>"#,
            "src-element",
        ),
        (
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:simpleType name="s"><xs:restriction base="xs:string"/></xs:simpleType>
                 <xs:simpleType name="s"><xs:restriction base="xs:string"/></xs:simpleType>
               </xs:schema>"#,
            "sch-props-correct.2",
        ),
        (
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:element name="a" type="xs:nosuchtype"/>
               </xs:schema>"#,
            "src-resolve",
        ),
    ] {
        let error = compile(text).expect_err("an invalid schema document");
        assert_eq!(error.code(), code, "for {text}");
    }
}

#[test]
fn a_restriction_must_narrow_its_base() {
    let error = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="Base">
               <xs:sequence><xs:element name="a" type="xs:string"/></xs:sequence>
             </xs:complexType>
             <xs:complexType name="Bad">
               <xs:complexContent>
                 <xs:restriction base="Base">
                   <xs:sequence><xs:element name="z" type="xs:string"/></xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
    )
    .expect_err("`z` does not restrict `a`");
    assert!(
        error.code().starts_with("rcase-"),
        "unexpected code {}",
        error.code()
    );
}

#[test]
fn final_blocks_derivation() {
    let error = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="Closed" final="restriction">
               <xs:restriction base="xs:string"/>
             </xs:simpleType>
             <xs:simpleType name="Open">
               <xs:restriction base="Closed"><xs:length value="2"/></xs:restriction>
             </xs:simpleType>
           </xs:schema>"#,
    )
    .expect_err("`final=\"restriction\"`");
    assert_eq!(error.code(), "st-props-correct.3");
}

#[test]
fn the_xml_namespace_attributes_are_available() -> Result<(), SchemaError> {
    let schema = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:xml="http://www.w3.org/XML/1998/namespace">
             <xs:element name="a">
               <xs:complexType><xs:attribute ref="xml:lang"/></xs:complexType>
             </xs:element>
           </xs:schema>"#,
    )?;
    assert!(schema.validate_str(r#"<a xml:lang="en"/>"#).valid);
    Ok(())
}

#[test]
fn anonymous_types_are_reachable_through_their_declarations() -> Result<(), SchemaError> {
    let schema = compile(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a">
               <xs:simpleType><xs:restriction base="xs:string"/></xs:simpleType>
             </xs:element>
           </xs:schema>"#,
    )?;
    let element = schema.find_element(None, "a").expect("a");
    let type_id = schema.element_declaration(element).type_id;
    assert!(schema.type_definition(type_id).name.is_none());
    assert!(matches!(
        schema.type_definition(type_id).kind,
        TypeKind::Simple(_)
    ));
    Ok(())
}

#[test]
fn a_fixed_facet_may_not_be_overridden_by_a_restriction() {
    // XSD Part 2 §4.1.3 / Part 1 §3.16.6.4 (*Simple Type Restriction (Facets)*): a facet the
    // base type marked `fixed="true"` may be restated, but never given a different value.
    for (facet, base_value, new_value) in [
        ("maxLength", "5", "3"),
        ("minLength", "2", "4"),
        ("length", "5", "3"),
    ] {
        let text = format!(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:simpleType name="b">
                   <xs:restriction base="xs:string">
                     <xs:{facet} value="{base_value}" fixed="true"/>
                   </xs:restriction>
                 </xs:simpleType>
                 <xs:simpleType name="d">
                   <xs:restriction base="b">
                     <xs:{facet} value="{new_value}"/>
                   </xs:restriction>
                 </xs:simpleType>
               </xs:schema>"#
        );
        let error = compile(&text)
            .err()
            .unwrap_or_else(|| panic!("overriding a fixed `{facet}` must be rejected"));
        assert_eq!(error.code(), format!("{facet}-valid-restriction"));
    }
}

#[test]
fn a_fixed_bound_facet_may_not_be_overridden_by_a_restriction() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:simpleType name="b">
                      <xs:restriction base="xs:int">
                        <xs:minInclusive value="1" fixed="true"/>
                      </xs:restriction>
                    </xs:simpleType>
                    <xs:simpleType name="d">
                      <xs:restriction base="b">
                        <xs:minInclusive value="2"/>
                      </xs:restriction>
                    </xs:simpleType>
                  </xs:schema>"#;
    let error = compile(text).expect_err("overriding a fixed `minInclusive` must be rejected");
    assert_eq!(error.code(), "minInclusive-valid-restriction");
}

#[test]
fn a_fixed_facet_may_be_restated_with_an_equal_value() {
    // Equality is in the *value* space: `01` and `1` are the same `xs:int`.
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:simpleType name="b">
                      <xs:restriction base="xs:int">
                        <xs:minInclusive value="1" fixed="true"/>
                      </xs:restriction>
                    </xs:simpleType>
                    <xs:simpleType name="d">
                      <xs:restriction base="b">
                        <xs:minInclusive value="01"/>
                      </xs:restriction>
                    </xs:simpleType>
                  </xs:schema>"#;
    assert!(compile(text).is_ok(), "restating the same value is legal");
}

#[test]
fn an_unfixed_facet_may_still_be_narrowed() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:simpleType name="b">
                      <xs:restriction base="xs:string">
                        <xs:maxLength value="5"/>
                      </xs:restriction>
                    </xs:simpleType>
                    <xs:simpleType name="d">
                      <xs:restriction base="b">
                        <xs:maxLength value="3"/>
                      </xs:restriction>
                    </xs:simpleType>
                  </xs:schema>"#;
    assert!(compile(text).is_ok());
}

#[test]
fn a_fixed_facet_stays_fixed_through_an_intervening_restriction() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:simpleType name="b">
                      <xs:restriction base="xs:string">
                        <xs:maxLength value="5" fixed="true"/>
                      </xs:restriction>
                    </xs:simpleType>
                    <xs:simpleType name="mid">
                      <xs:restriction base="b">
                        <xs:minLength value="1"/>
                      </xs:restriction>
                    </xs:simpleType>
                    <xs:simpleType name="d">
                      <xs:restriction base="mid">
                        <xs:maxLength value="3"/>
                      </xs:restriction>
                    </xs:simpleType>
                  </xs:schema>"#;
    let error = compile(text).expect_err("the `fixed` flag is inherited down the chain");
    assert_eq!(error.code(), "maxLength-valid-restriction");
}

/// A `<xs:group ref="…"/>` gives the group's own particles no wrapping of
/// their own beyond the reference itself, so a base `sequence` that merely
/// factors part of itself out into a named group can still be restricted
/// particle-for-particle, as if it had been written out in full
/// (`Group_w3c/groupB003v`).
#[test]
fn a_group_reference_does_not_block_restricting_the_particles_it_stands_for() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="base">
                      <xs:sequence>
                        <xs:group ref="required"/>
                        <xs:group ref="optional" minOccurs="0"/>
                      </xs:sequence>
                    </xs:complexType>
                    <xs:complexType name="derived">
                      <xs:complexContent>
                        <xs:restriction base="base">
                          <xs:sequence>
                            <xs:group ref="required"/>
                          </xs:sequence>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                    <xs:group name="required">
                      <xs:sequence>
                        <xs:element name="r1"/>
                        <xs:element name="r2"/>
                      </xs:sequence>
                    </xs:group>
                    <xs:group name="optional">
                      <xs:sequence>
                        <xs:element name="o1"/>
                        <xs:element name="o2"/>
                      </xs:sequence>
                    </xs:group>
                  </xs:schema>"#;
    assert!(compile(text).is_ok());
}

/// §3.9.6 *Occurrence Range OK*: a particle restricted to occur exactly zero
/// times is always a valid restriction of any occurrence range, however
/// narrow — the standard way to prohibit a particle the base requires
/// outright (`ModelGroups_w3c/mgH014`).
#[test]
fn a_zero_occurs_particle_prohibits_a_base_particle_outright() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="base">
                      <xs:choice>
                        <xs:element name="e1"/>
                        <xs:element name="e2"/>
                      </xs:choice>
                    </xs:complexType>
                    <xs:complexType name="derived">
                      <xs:complexContent>
                        <xs:restriction base="base">
                          <xs:choice>
                            <xs:element name="e1" minOccurs="0" maxOccurs="0"/>
                            <xs:element name="e2"/>
                          </xs:choice>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                  </xs:schema>"#;
    assert!(compile(text).is_ok());
}

/// `RecurseAsIfGroup` (an element particle restricting a group particle) leaves
/// it open whether the element's occurrence range belongs on the fake group or
/// on its single member, and the two readings disagree. XSD 1.0's own wording
/// puts it on the member and the suite holds a 1.0 processor to that
/// (`Particles_w3c/particlesHa161` is marked invalid for 1.0 and valid for 1.1
/// outright); 1.1 clarified it the other way, and needs *both* — `particlesZ001`
/// needs the range on the fake group, `particlesM002` and `particlesIb001` need
/// it on the member, and both are marked valid for 1.1.
#[test]
fn recurse_as_if_group_admits_either_reading_of_the_occurrence_range() {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="base">
                      <xs:sequence>
                        <xs:choice minOccurs="0">
                          <xs:element name="a" type="xs:string"/>
                          <xs:element name="b" type="xs:string"/>
                        </xs:choice>
                      </xs:sequence>
                    </xs:complexType>
                    <xs:complexType name="derived">
                      <xs:complexContent>
                        <xs:restriction base="base">
                          <xs:sequence>
                            <xs:element name="a" type="xs:string" minOccurs="0"/>
                          </xs:sequence>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                  </xs:schema>"#;
    // The restriction accepts only the empty sequence or a single `a`, both of
    // which the base accepts. Only the range-on-the-fake-group reading sees
    // that, so 1.0 — held to its own literal wording — rejects it and 1.1
    // accepts it, which is exactly what the suite says of the same shape in
    // `msData/particles/particlesHa161`.
    assert!(compile(text).is_err());
    compile_1_1(text).expect("the `particlesZ001` shape under 1.1");

    // The mirror image: an element whose range exceeds the base *group*'s while
    // fitting the base *member*'s, which only the range-on-the-member reading
    // sees (`msData/particles/particlesIb001`).
    let mirror = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="base">
                      <xs:choice>
                        <xs:element name="a" minOccurs="2" maxOccurs="6" type="xs:string"/>
                        <xs:element name="b" minOccurs="0" maxOccurs="6" type="xs:string"/>
                      </xs:choice>
                    </xs:complexType>
                    <xs:complexType name="derived">
                      <xs:complexContent>
                        <xs:restriction base="base">
                          <xs:choice>
                            <xs:element name="a" minOccurs="2" maxOccurs="2" type="xs:string"/>
                          </xs:choice>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                  </xs:schema>"#;
    compile(mirror).expect("the `particlesIb001` shape");
    compile_1_1(mirror).expect("the `particlesIb001` shape under 1.1");

    // Neither reading admits an element the base group cannot match at all.
    let unrelated = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="base">
                      <xs:choice>
                        <xs:element name="a" type="xs:string"/>
                        <xs:element name="b" type="xs:string"/>
                      </xs:choice>
                    </xs:complexType>
                    <xs:complexType name="derived">
                      <xs:complexContent>
                        <xs:restriction base="base">
                          <xs:choice>
                            <xs:element name="z" type="xs:string"/>
                          </xs:choice>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                  </xs:schema>"#;
    assert!(compile(unrelated).is_err());
    assert!(compile_1_1(unrelated).is_err());
}

/// A two-level restriction chain from `xs:integer`: `based` carries
/// `base_facet`, `derived` restricts `based` with `derived_facet`. Each level
/// carries exactly one facet, so no `xs:restriction` co-occurrence rule
/// (`maxInclusive`+`maxExclusive`, `minInclusive`+`minExclusive`) is at
/// stake — only the `…-valid-restriction` bound-vs-bound comparison.
fn facet_restriction_chain(base_facet: &str, derived_facet: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:simpleType name="based">
               <xs:restriction base="xs:integer">
                 {base_facet}
               </xs:restriction>
             </xs:simpleType>
             <xs:simpleType name="derived">
               <xs:restriction base="based">
                 {derived_facet}
               </xs:restriction>
             </xs:simpleType>
           </xs:schema>"#
    )
}

/// A restriction facet equal to the base's own same-kind exclusive bound is
/// not a widening — `{v : v < x}` restricted to itself is still
/// `{v : v < x}` — but the ordinary instance check (which enforces the
/// *strict* inequality an instance value must satisfy) wrongly rejects a new
/// facet **declaration** that merely restates the base's own `maxExclusive`
/// or `minExclusive` (`dateTimeStamp/d3_4_28v09`); a `minInclusive`/
/// `maxInclusive` restatement needs no such relaxation, since the ordinary
/// `<=`/`>=` instance check already accepts equality.
#[test]
fn a_bound_facet_equal_to_the_bases_own_same_kind_bound_is_accepted() {
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:minInclusive value="0"/>"#,
            r#"<xs:minInclusive value="0"/>"#
        ))
        .is_ok()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:maxInclusive value="100"/>"#,
            r#"<xs:maxInclusive value="100"/>"#
        ))
        .is_ok()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:minExclusive value="-1"/>"#,
            r#"<xs:minExclusive value="-1"/>"#
        ))
        .is_ok()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:maxExclusive value="101"/>"#,
            r#"<xs:maxExclusive value="101"/>"#
        ))
        .is_ok()
    );
}

/// The relaxation above must not turn the bound-vs-bound check into a no-op:
/// a same-kind bound that actually *widens* past the base's own bound is
/// still rejected, and so is a cross-kind restatement (a new `minInclusive`
/// equal to the base's own `minExclusive`, say) that would admit the
/// excluded boundary value.
#[test]
fn a_bound_facet_that_actually_widens_the_base_is_still_rejected() {
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:maxExclusive value="101"/>"#,
            r#"<xs:maxExclusive value="102"/>"#
        ))
        .is_err()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:minExclusive value="-1"/>"#,
            r#"<xs:minExclusive value="-2"/>"#
        ))
        .is_err()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:minExclusive value="-1"/>"#,
            r#"<xs:minInclusive value="-1"/>"#
        ))
        .is_err()
    );
    assert!(
        compile(&facet_restriction_chain(
            r#"<xs:maxExclusive value="101"/>"#,
            r#"<xs:maxInclusive value="101"/>"#
        ))
        .is_err()
    );
}

fn compile_1_1(text: &str) -> Result<Schema, SchemaError> {
    let mut set = SchemaSet::new().with_version(XsdVersion::V1_1);
    set.add_document(None, text)?;
    set.compile()
}

/// A content model in which a particle refers to `d`, the head of a chain of
/// substitution groups `a` → `b` → `c` → `d`, restricted by a particle
/// referring to one of the members. `blocking` is written on whichever
/// declaration the caller names, and `member` is the one the restriction uses.
fn substitution_chain(blocking: &str, member: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="a" substitutionGroup="b" type="xs:anyType"/>
             <xs:element name="b" substitutionGroup="c" type="xs:anyType"{b}/>
             <xs:element name="c" substitutionGroup="d" type="xs:anyType"{c}/>
             <xs:element name="d" type="xs:anyType"/>
             <xs:complexType name="base">
               <xs:sequence><xs:element ref="d"/></xs:sequence>
             </xs:complexType>
             <xs:complexType name="derived">
               <xs:complexContent>
                 <xs:restriction base="base">
                   <xs:sequence><xs:element ref="{member}"/></xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#,
        b = if blocking == "b" {
            r#" block="substitution""#
        } else {
            ""
        },
        c = if blocking == "c" {
            r#" block="substitution""#
        } else {
            ""
        },
    )
}

#[test]
fn a_restriction_may_narrow_a_particle_to_a_substitution_group_member() -> Result<(), SchemaError> {
    // §3.9.6: a particle referring to `d` admits every member of `d`'s
    // substitution group, so narrowing it to one of them accepts strictly
    // fewer sequences — a valid restriction, even though the two declarations
    // have different names (`msData/element/elemZ027e`, `elemZ027f`).
    compile(&substitution_chain("", "c"))?;
    compile(&substitution_chain("", "a"))?;
    // `block="substitution"` on the member itself says nothing about the member
    // standing in for its own head (`msData/element/elemZ027b`).
    compile(&substitution_chain("c", "c"))?;
    // …but a head part-way up the chain that blocks substitution breaks it, so
    // `a` cannot reach `d` at all (`msData/element/elemZ027c`).
    let error = compile(&substitution_chain("b", "a")).expect_err("a broken chain");
    assert_eq!(error.code(), "rcase-NameAndTypeOK.1");
    // A *choice* of members restricting the head is the same narrowing, written
    // out (`msData/element/elemZ027a`): the table in §3.9.6 has no cell for a
    // group restricting an element, but a particle referring to a head already
    // denotes a choice over its substitution group.
    let choice_of_members = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
         <xs:element name="head"/>
         <xs:element name="m1" substitutionGroup="head"/>
         <xs:element name="m2" substitutionGroup="head"/>
         <xs:complexType name="base">
           <xs:sequence><xs:element ref="head"/></xs:sequence>
         </xs:complexType>
         <xs:complexType name="derived">
           <xs:complexContent>
             <xs:restriction base="base">
               <xs:sequence>
                 <xs:choice>
                   <xs:element ref="m1"/>
                   <xs:element ref="m2"/>
                 </xs:choice>
               </xs:sequence>
             </xs:restriction>
           </xs:complexContent>
         </xs:complexType>
       </xs:schema>"#;
    compile(choice_of_members)?;
    compile_1_1(choice_of_members)?;
    // A head with no substitution group of its own still cannot be restricted
    // by a group: there is no implicit choice to narrow.
    let no_group = choice_of_members.replace(r#" substitutionGroup="head""#, "");
    assert_eq!(
        compile(&no_group).expect_err("nothing to narrow").code(),
        "rcase-Forbidden"
    );

    // An unrelated element is still not a restriction of `d`.
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
         <xs:element name="d" type="xs:anyType"/>
         <xs:element name="z" type="xs:anyType"/>
         <xs:complexType name="base">
           <xs:sequence><xs:element ref="d"/></xs:sequence>
         </xs:complexType>
         <xs:complexType name="derived">
           <xs:complexContent>
             <xs:restriction base="base">
               <xs:sequence><xs:element ref="z"/></xs:sequence>
             </xs:restriction>
           </xs:complexContent>
         </xs:complexType>
       </xs:schema>"#;
    assert_eq!(
        compile(text).expect_err("unrelated names").code(),
        "rcase-NameAndTypeOK.1"
    );
    Ok(())
}

#[test]
fn a_restriction_of_a_substitution_group_member_still_obeys_the_occurrence_range()
-> Result<(), SchemaError> {
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
         <xs:element name="m" substitutionGroup="h" type="xs:anyType"/>
         <xs:element name="h" type="xs:anyType"/>
         <xs:complexType name="base">
           <xs:sequence><xs:element ref="h"/></xs:sequence>
         </xs:complexType>
         <xs:complexType name="derived">
           <xs:complexContent>
             <xs:restriction base="base">
               <xs:sequence><xs:element ref="m" maxOccurs="2"/></xs:sequence>
             </xs:restriction>
           </xs:complexContent>
         </xs:complexType>
       </xs:schema>"#;
    assert_eq!(
        compile(text).expect_err("a widened range").code(),
        "rcase-Recurse.1"
    );
    Ok(())
}

/// An abstract global `e1` in the substitution group of `e`, competing with an
/// unrelated local `e1` in the same content model — the shape of
/// `wgData/sg/upa.xsd` and `edc.xsd`.
const ABSTRACT_MEMBER_COMPETES: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
     <xs:element name="e" type="xs:string"/>
     <xs:element name="e1" substitutionGroup="e" abstract="true"/>
     <xs:complexType name="t">
       <xs:choice>
         <xs:element ref="e"/>
         <xs:element name="e1" type="xs:string"/>
       </xs:choice>
     </xs:complexType>
   </xs:schema>"#;

#[test]
fn an_abstract_substitution_group_member_competes_only_under_1_1() -> Result<(), SchemaError> {
    // XSD 1.0 leaves an abstract element out of the *actual* substitution
    // group — it can never appear in an instance, so the two particles cannot
    // both match the same element — and XSD 1.1 deliberately puts it back in
    // (the resolution of W3C bug 4337).
    compile(ABSTRACT_MEMBER_COMPETES)?;
    let error = compile_1_1(ABSTRACT_MEMBER_COMPETES).expect_err("ambiguous under 1.1");
    assert_eq!(error.code(), "cos-nonambig");
    // The same for *Element Declarations Consistent*, where the two particles
    // sit in a sequence and so never compete for attribution, but would give
    // the same name two types.
    let consistent = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
         <xs:element name="e" type="xs:string"/>
         <xs:element name="e1" substitutionGroup="e" abstract="true"/>
         <xs:complexType name="t">
           <xs:sequence>
             <xs:element ref="e"/>
             <xs:element name="e1" type="xs:int"/>
           </xs:sequence>
         </xs:complexType>
       </xs:schema>"#;
    compile(consistent)?;
    assert_eq!(
        compile_1_1(consistent)
            .expect_err("inconsistent under 1.1")
            .code(),
        "cos-element-consistent"
    );
    // A *concrete* member competes under both versions.
    let concrete = ABSTRACT_MEMBER_COMPETES.replace(r#" abstract="true""#, r#" type="xs:string""#);
    assert_eq!(
        compile(&concrete).expect_err("ambiguous under 1.0").code(),
        "cos-nonambig"
    );
    Ok(())
}

/// Circularity among `xs:redefine`/`xs:override` edges is **not** by itself an
/// error: `saxonData/Override/over023` is titled "A permissible circular
/// `xs:override`" and `msData/schema/schU1` is a pair of documents that redefine
/// each other, both of which the W3C suite marks valid. The loader visits each
/// document once, so the edge that closes the loop finds a document it has
/// already seen and contributes nothing further.
#[test]
fn a_circular_redefinition_is_accepted() -> Result<(), SchemaError> {
    // The shape of `msData/schema/schU1`: `a` redefines an attribute group `b`
    // declares, and `b` redefines one `a` declares.
    let a = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                          targetNamespace="urn:x" xmlns:t="urn:x">
                 <xs:redefine schemaLocation="b.xsd">
                   <xs:attributeGroup name="bg">
                     <xs:attribute name="b1" type="xs:int"/>
                   </xs:attributeGroup>
                 </xs:redefine>
                 <xs:attributeGroup name="ag">
                   <xs:attribute name="a1"/>
                 </xs:attributeGroup>
               </xs:schema>"#;
    let b = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:t="urn:x">
                 <xs:redefine schemaLocation="a.xsd">
                   <xs:attributeGroup name="ag">
                     <xs:attribute name="a1" type="xs:int"/>
                   </xs:attributeGroup>
                 </xs:redefine>
                 <xs:attributeGroup name="bg">
                   <xs:attribute name="b1"/>
                 </xs:attributeGroup>
               </xs:schema>"#;
    let resolver = MemoryResolver::default().with("a.xsd", a).with("b.xsd", b);
    let mut set = SchemaSet::new().with_resolver(Box::new(resolver));
    set.add_document(Some("a.xsd"), a)?;
    set.compile()?;
    Ok(())
}

#[test]
fn a_chain_of_redefinitions_that_does_not_close_is_accepted() -> Result<(), SchemaError> {
    let resolver = MemoryResolver::default()
        .with(
            "a.xsd",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                          xmlns:t="urn:x" targetNamespace="urn:x">
                 <xs:redefine schemaLocation="b.xsd">
                   <xs:complexType name="c">
                     <xs:complexContent>
                       <xs:extension base="t:c">
                         <xs:sequence><xs:element name="extra" type="xs:string"/></xs:sequence>
                       </xs:extension>
                     </xs:complexContent>
                   </xs:complexType>
                 </xs:redefine>
               </xs:schema>"#,
        )
        .with(
            "b.xsd",
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x">
                 <xs:complexType name="c"><xs:sequence/></xs:complexType>
               </xs:schema>"#,
        );
    compile_with(
        resolver,
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x">
             <xs:include schemaLocation="a.xsd"/>
           </xs:schema>"#,
    )?;
    Ok(())
}

/// The `vc:` guards of §4.2.6 exist so that one schema document can serve two
/// versions of XML Schema, which only works if **both** apply them.
#[test]
fn conditional_inclusion_applies_under_xml_schema_1_0_too() -> Result<(), SchemaError> {
    // `xs:assert` is a 1.1 element and `xs:error` a 1.1 datatype, so a 1.0
    // processor that ignored the guards would reject this document twice over;
    // the point of the guards is that it never sees either
    // (`saxonData/VC/vc001`, `vc014`).
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:vc="http://www.w3.org/2007/XMLSchema-versioning">
                    <xs:element name="doc">
                      <xs:complexType>
                        <xs:attribute name="x" use="required" type="xs:integer"/>
                        <xs:attribute name="y" type="xs:error" vc:typeAvailable="xs:error"/>
                        <xs:attribute name="y" type="xs:string" vc:typeUnavailable="xs:error"/>
                        <xs:assert test="@x &gt; 300" vc:minVersion="1.1"/>
                      </xs:complexType>
                    </xs:element>
                  </xs:schema>"#;
    let schema = compile(text)?;
    // The assertion went with its guard, so a value it would have rejected is
    // accepted, and `y` is the string the other guard left behind.
    assert!(schema.validate_str(r#"<doc x="204" y="anything"/>"#).valid);

    // Under 1.1 the other halves are the ones that survive: `xs:error` admits
    // no value at all, and the assertion rejects 204.
    let schema = compile_1_1(text)?;
    assert!(!schema.validate_str(r#"<doc x="204" y="anything"/>"#).valid);
    assert!(schema.validate_str(r#"<doc x="304"/>"#).valid);
    Ok(())
}

#[test]
fn a_facet_guard_names_the_facets_of_the_version_in_force() -> Result<(), SchemaError> {
    // `assertion` is a facet XML Schema 1.1 adds, so the guard holds under 1.1
    // and not under 1.0 (`saxonData/VC/vc024`).
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:vc="http://www.w3.org/2007/XMLSchema-versioning">
                    <xs:element name="doc">
                      <xs:complexType>
                        <xs:attribute name="x" use="required" type="xs:integer"/>
                        <xs:assert test="@x &gt; 300" vc:facetAvailable="xs:assertion"/>
                      </xs:complexType>
                    </xs:element>
                  </xs:schema>"#;
    assert!(compile(text)?.validate_str(r#"<doc x="204"/>"#).valid);
    assert!(!compile_1_1(text)?.validate_str(r#"<doc x="204"/>"#).valid);
    Ok(())
}

#[test]
fn an_uninterpretable_guard_is_an_error_only_where_it_is_declared() {
    // `vc:minVersion` is an `xs:decimal` in the 1.1 schema for schemas and
    // nothing at all in the 1.0 one, where it is simply an attribute from a
    // foreign namespace and no constraint speaks about its value.
    // `saxonData/VC/vc902` and `vc903` are the same document with opposite
    // expectations for exactly this reason.
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:vc="http://www.w3.org/2007/XMLSchema-versioning">
                    <xs:element name="doc" type="xs:string" vc:minVersion="10g"/>
                  </xs:schema>"#;
    let schema = compile(text).expect("1.0 has nothing to say about the value");
    assert!(schema.validate_str("<doc>x</doc>").valid);

    let mut set = SchemaSet::new().with_version(XsdVersion::V1_1);
    let error = set
        .add_document(None, text)
        .expect_err("1.1 declares the attribute an `xs:decimal`");
    assert_eq!(error.code(), "src-cip");
}

#[test]
fn a_schema_document_may_not_nest_without_bound() {
    // The deep part sits inside `xs:documentation`, whose content the
    // Recommendation leaves unconstrained — so nothing but the depth bound
    // stands between it and the recursive walks that read the document.
    fn document(depth: usize) -> String {
        let mut text = String::from(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:annotation><xs:documentation>"#,
        );
        text.push_str(&"<a>".repeat(depth));
        text.push_str(&"</a>".repeat(depth));
        text.push_str(
            r#"</xs:documentation></xs:annotation>
                 <xs:element name="doc" type="xs:string"/>
               </xs:schema>"#,
        );
        text
    }

    let mut set = SchemaSet::new();
    let error = set
        .add_document(None, &document(SchemaSet::MAX_DEPTH + 1))
        .expect_err("a document deeper than the bound");
    assert_eq!(error.code(), "src-max-depth");

    // Everything up to the bound is read as usual.
    let mut set = SchemaSet::new();
    set.add_document(None, &document(SchemaSet::MAX_DEPTH - 4))
        .expect("a document within the bound");
    let schema = set.compile().expect("a document within the bound");
    assert!(schema.validate_str("<doc>x</doc>").valid);
}

#[test]
fn a_choice_may_reorder_its_branches_under_xml_schema_1_1() -> Result<(), SchemaError> {
    // The base offers `c1` or `c2` and the restriction offers `c2` or `c1`, so
    // the two accept exactly the same sequences. XSD 1.0 all the same asks for
    // an order-preserving mapping between the branches and refuses; XSD 1.1
    // asks about the languages, which are equal
    // (`Particles_w3c/particlesT002`, marked invalid for 1.0 and valid for
    // 1.1).
    let text = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                    <xs:complexType name="b">
                      <xs:sequence>
                        <xs:choice>
                          <xs:element name="c1"/>
                          <xs:element name="c2"/>
                        </xs:choice>
                        <xs:element name="foo"/>
                      </xs:sequence>
                    </xs:complexType>
                    <xs:complexType name="r">
                      <xs:complexContent>
                        <xs:restriction base="b">
                          <xs:sequence>
                            <xs:choice>
                              <xs:element name="c2"/>
                              <xs:element name="c1"/>
                            </xs:choice>
                            <xs:element name="foo"/>
                          </xs:sequence>
                        </xs:restriction>
                      </xs:complexContent>
                    </xs:complexType>
                  </xs:schema>"#;
    let code = compile(text)
        .expect_err("1.0 asks for the branch order to be preserved")
        .code()
        .to_owned();
    assert!(code.starts_with("rcase-"), "unexpected code `{code}`");
    compile_1_1(text)?;
    Ok(())
}

#[test]
fn a_chain_of_definitions_may_not_run_on_without_bound() {
    // A cycle among named definitions is already caught, but a *chain* is
    // perfectly legal and is built by recursion — one frame per link — so a
    // document is otherwise free to make it as long as it likes.
    fn groups(links: usize) -> String {
        let mut text = String::from(r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">"#);
        for link in 0..links {
            text.push_str(&format!(r#"<xs:group name="g{link}"><xs:sequence>"#));
            if link + 1 < links {
                text.push_str(&format!(r#"<xs:group ref="g{}"/>"#, link + 1));
            } else {
                text.push_str(r#"<xs:element name="leaf" type="xs:string"/>"#);
            }
            text.push_str("</xs:sequence></xs:group>");
        }
        text.push_str("</xs:schema>");
        text
    }

    let error =
        compile(&groups(SchemaSet::MAX_DEFINITION_CHAIN + 2)).expect_err("a chain past the bound");
    assert_eq!(error.code(), "src-max-depth");
    // Twenty thousand links are what a hostile schema document would actually
    // carry, and the point is that they are refused rather than aborting the
    // process.
    let error = compile(&groups(20_000)).expect_err("a chain far past the bound");
    assert_eq!(error.code(), "src-max-depth");
    // The links are built one after another as well as inside one another, so
    // a chain within the bound must not be refused for the sum of them.
    compile(&groups(SchemaSet::MAX_DEFINITION_CHAIN - 1)).expect("a chain within the bound");
}

/// The bound on how deeply a schema document may nest is what stops the
/// *builders* recursing without limit, not only the readers: a chain of
/// `xs:element` / `xs:complexType` / `xs:sequence` is read by
/// `Compiler::build_particle` calling itself once per level, and that is by
/// some way the most expensive of the recursions the depth governs.
///
/// The work runs on a thread with a stack the size a main thread gets, since
/// that is what the ceiling is chosen against; the harness gives each test a
/// much smaller one.
#[test]
fn the_depth_bound_stops_the_component_builders_too() {
    fn vocabulary(levels: usize) -> String {
        let mut text = String::from(r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">"#);
        text.push_str(&r#"<xs:element name="e"><xs:complexType><xs:sequence>"#.repeat(levels));
        text.push_str(&"</xs:sequence></xs:complexType></xs:element>".repeat(levels));
        text.push_str("</xs:schema>");
        text
    }

    let work = || {
        // Three elements of the document tree per level, so this is past the
        // ceiling with room to spare.
        let error = compile(&vocabulary(SchemaSet::MAX_DEPTH)).expect_err("past the bound");
        assert_eq!(error.code(), "src-max-depth");
        // A depth a real schema might plausibly reach is read as usual.
        compile(&vocabulary(20)).expect("within the bound");
    };
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(work)
        .expect("the probe thread starts")
        .join()
        .expect("the probe thread finishes without aborting");
}
