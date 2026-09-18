//! The XSD 1.1 machinery that needs an XPath engine, plus the 1.1 wildcard,
//! conditional-inclusion and identity rules.
//!
//! Every test here assembles a small schema under [`XsdVersion::V1_1`] and
//! asserts both an accepting and a rejecting instance, so that a rule that
//! stops firing is as visible as one that fires too often.

use oxixml_schema::{SchemaError, SchemaSet, XsdVersion};

/// Compiles `documents` under 1.1 and returns the schema.
fn compile(documents: &[&str]) -> Result<oxixml_schema::Schema, SchemaError> {
    let mut set = SchemaSet::new().with_version(XsdVersion::V1_1);
    for (index, text) in documents.iter().enumerate() {
        set.add_document(Some(&format!("test{index}.xsd")), text)?;
    }
    set.compile()
}

/// The first error code an instance produces, or [`None`] when it is valid.
fn code(schema: &oxixml_schema::Schema, instance: &str) -> Option<String> {
    let outcome = schema.validate_str(instance);
    outcome
        .first_error()
        .map(|error| error.code().to_owned())
        .inspect(|_| assert!(!outcome.valid))
}

/// The `notQName` token for "an element declared in this content model".
///
/// It is a constant because a Rust raw string may not hold `"##`.
const SIBLING: &str = "##definedSibling";

/// The `namespace` token for "no namespace", for the same reason.
const LOCAL: &str = "##local";

const HEAD: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">"#;

#[test]
fn an_assert_on_a_complex_type_is_evaluated() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="range">
               <xs:complexType>
                 <xs:attribute name="min" type="xs:int"/>
                 <xs:attribute name="max" type="xs:int"/>
                 <xs:assert test="@min le @max"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, r#"<range min="1" max="9"/>"#), None);
    assert_eq!(
        code(&schema, r#"<range min="9" max="1"/>"#).as_deref(),
        Some("cvc-assertion")
    );
    Ok(())
}

#[test]
fn an_assert_sees_the_children_but_not_the_ancestors() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="pair" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:sequence>
                         <xs:element name="a" type="xs:string"/>
                         <xs:element name="b" type="xs:string"/>
                       </xs:sequence>
                       <!-- The fabricated instance is rooted at the element
                            under validation, so no ancestor `doc` is
                            reachable. -->
                       <xs:assert test="empty(ancestor::doc) and a != b"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(&schema, "<doc><pair><a>x</a><b>y</b></pair></doc>"),
        None
    );
    assert_eq!(
        code(&schema, "<doc><pair><a>x</a><b>x</b></pair></doc>").as_deref(),
        Some("cvc-assertion")
    );
    Ok(())
}

#[test]
fn the_assertion_facet_binds_value() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="even">
               <xs:simpleType>
                 <xs:restriction base="xs:integer">
                   <xs:assertion test="$value mod 2 = 0"/>
                 </xs:restriction>
               </xs:simpleType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<even>4</even>"), None);
    assert_eq!(
        code(&schema, "<even>5</even>").as_deref(),
        Some("cvc-assertion")
    );
    Ok(())
}

#[test]
fn conditional_type_assignment_chooses_the_type() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <!-- The declared type is `xs:anyType` because §3.3.6.1 clause
                  7 requires every alternative to be validly substitutable for
                  it, and `dated`/`counted` derive from `text` in no way. -->
             <xs:element name="when" type="xs:anyType">
               <xs:alternative test="@kind = 'date'" type="dated"/>
               <xs:alternative test="@kind = 'int'" type="counted"/>
               <xs:alternative type="text"/>
             </xs:element>
             <xs:complexType name="text">
               <xs:simpleContent>
                 <xs:extension base="xs:string">
                   <xs:attribute name="kind" type="xs:string"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="dated">
               <xs:simpleContent>
                 <xs:extension base="xs:date">
                   <xs:attribute name="kind" type="xs:string"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
             <xs:complexType name="counted">
               <xs:simpleContent>
                 <xs:extension base="xs:int">
                   <xs:attribute name="kind" type="xs:string"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(&schema, r#"<when kind="date">2024-02-29</when>"#),
        None
    );
    assert_eq!(code(&schema, r#"<when kind="int">42</when>"#), None);
    assert_eq!(code(&schema, r#"<when kind="other">hello</when>"#), None);
    assert_eq!(
        code(&schema, r#"<when kind="date">hello</when>"#).as_deref(),
        Some("cvc-datatype-valid.1.2.1")
    );
    Ok(())
}

#[test]
fn a_test_that_is_not_an_xpath_expression_is_a_schema_error() {
    let error = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="e">
               <xs:complexType>
                 <xs:assert test="@a +"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])
    .expect_err("the assertion does not parse");
    assert_eq!(error.code(), "src-assert");
}

#[test]
fn not_qname_excludes_names_the_wildcard_would_otherwise_match() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:any notQName="skipped" processContents="skip"
                           minOccurs="0" maxOccurs="unbounded"/>
                 </xs:sequence>
                 <xs:anyAttribute notQName="no" processContents="skip"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<doc><kept/></doc>"), None);
    assert_eq!(
        code(&schema, "<doc><skipped/></doc>").as_deref(),
        Some("cvc-complex-type.2.4.a")
    );
    assert_eq!(code(&schema, r#"<doc yes="1"/>"#), None);
    assert_eq!(
        code(&schema, r#"<doc no="1"/>"#).as_deref(),
        Some("cvc-complex-type.3.2.2")
    );
    Ok(())
}

#[test]
fn defined_sibling_excludes_the_content_models_own_declarations() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="a" type="xs:string"/>
                   <xs:any notQName="{SIBLING}" processContents="skip"
                           minOccurs="0" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<doc><a/><b/></doc>"), None);
    assert_eq!(
        code(&schema, "<doc><a/><a/></doc>").as_deref(),
        Some("cvc-complex-type.2.4.a")
    );
    Ok(())
}

#[test]
fn conditional_inclusion_removes_what_this_version_does_not_want() -> Result<(), SchemaError> {
    // The guarded attribute belongs to a hypothetical XSD 5.0, so it is not
    // there at all — and the unknown element it wraps is never checked.
    let schema = compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:vc="http://www.w3.org/2007/XMLSchema-versioning">
             <xs:element name="doc">
               <xs:complexType>
                 <xs:attribute name="here" type="xs:string"/>
                 <xs:attribute name="gone" type="xs:string" vc:minVersion="5.0"/>
                 <xs:anything-at-all vc:minVersion="5.0"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#])?;
    assert_eq!(code(&schema, r#"<doc here="x"/>"#), None);
    assert_eq!(
        code(&schema, r#"<doc gone="x"/>"#).as_deref(),
        Some("cvc-complex-type.3.2.2")
    );
    Ok(())
}

#[test]
fn a_guard_on_the_schema_element_disables_the_whole_document() -> Result<(), SchemaError> {
    let schema = compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                      xmlns:vc="http://www.w3.org/2007/XMLSchema-versioning"
                      vc:maxVersion="0.9">
             <xs:element name="doc" type="xs:string"/>
           </xs:schema>"#])?;
    // The document contributed nothing, so there is no declaration to find.
    assert_eq!(code(&schema, "<doc/>").as_deref(), Some("cvc-elt.1"));
    Ok(())
}

#[test]
fn an_element_may_carry_two_id_attributes_with_one_value() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="p" maxOccurs="unbounded">
                     <xs:complexType>
                       <xs:attribute name="one" type="xs:ID"/>
                       <xs:attribute name="two" type="xs:ID"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    // 1.1 binds an ID to its element, so one element may bind the same value
    // twice; two different elements may not.
    assert_eq!(code(&schema, r#"<doc><p one="a" two="a"/></doc>"#), None);
    assert_eq!(
        code(&schema, r#"<doc><p one="a"/><p two="a"/></doc>"#).as_deref(),
        Some("cvc-id.2")
    );
    Ok(())
}

#[test]
fn default_attributes_apply_to_every_complex_type() -> Result<(), SchemaError> {
    let schema = compile(&[
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" defaultAttributes="extra">
             <xs:attributeGroup name="extra">
               <xs:attribute name="note" type="xs:string"/>
             </xs:attributeGroup>
             <xs:element name="doc">
               <xs:complexType/>
             </xs:element>
             <xs:element name="closed">
               <xs:complexType defaultAttributesApply="false"/>
             </xs:element>
           </xs:schema>"#,
    ])?;
    assert_eq!(code(&schema, r#"<doc note="hi"/>"#), None);
    assert_eq!(
        code(&schema, r#"<closed note="hi"/>"#).as_deref(),
        Some("cvc-complex-type.3.2.2")
    );
    Ok(())
}

#[test]
fn an_identity_constraint_may_be_referred_to() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:choice maxOccurs="unbounded">
                   <xs:element name="chap" type="part">
                     <xs:unique name="one-number">
                       <xs:selector xpath="section"/>
                       <xs:field xpath="@nr"/>
                     </xs:unique>
                   </xs:element>
                   <xs:element name="appx" type="part">
                     <xs:unique ref="one-number"/>
                   </xs:element>
                 </xs:choice>
               </xs:complexType>
             </xs:element>
             <xs:complexType name="part">
               <xs:sequence>
                 <xs:element name="section" maxOccurs="unbounded">
                   <xs:complexType>
                     <xs:attribute name="nr" type="xs:int"/>
                   </xs:complexType>
                 </xs:element>
               </xs:sequence>
             </xs:complexType>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(
            &schema,
            r#"<doc><appx><section nr="1"/><section nr="2"/></appx></doc>"#
        ),
        None
    );
    assert_eq!(
        code(
            &schema,
            r#"<doc><appx><section nr="1"/><section nr="1"/></appx></doc>"#
        )
        .as_deref(),
        Some("cvc-identity-constraint.4.1")
    );
    Ok(())
}

#[test]
fn a_repeated_group_matches_the_language_not_the_greedy_walk() -> Result<(), SchemaError> {
    // `(a+){2,unbounded}` accepts two `a`s only if the first `a+` is willing
    // to stop after one.
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence minOccurs="2" maxOccurs="unbounded">
                   <xs:element name="a" type="xs:string" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<doc><a/><a/></doc>"), None);
    assert_eq!(
        code(&schema, "<doc><a/></doc>").as_deref(),
        Some("cvc-complex-type.2.4.b")
    );
    Ok(())
}

#[test]
fn xs_error_admits_no_value() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="e" type="text">
               <xs:alternative test="@bad = 'yes'" type="xs:error"/>
               <xs:alternative type="text"/>
             </xs:element>
             <xs:complexType name="text">
               <xs:simpleContent>
                 <xs:extension base="xs:string">
                   <xs:attribute name="bad" type="xs:string"/>
                 </xs:extension>
               </xs:simpleContent>
             </xs:complexType>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, r#"<e bad="no">fine</e>"#), None);
    assert!(code(&schema, r#"<e bad="yes">anything</e>"#).is_some());
    Ok(())
}

#[test]
fn conditional_type_assignment_sees_an_inherited_attribute() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="chap" maxOccurs="unbounded"/>
                 </xs:sequence>
                 <xs:attribute name="lang" inheritable="true"/>
               </xs:complexType>
             </xs:element>
             <!-- `xs:anyType`, so that §3.3.6.1 clause 7 holds of both
                  alternatives. -->
             <xs:element name="chap" type="xs:anyType">
               <xs:alternative test="@lang = 'de'" type="german"/>
               <xs:alternative type="text"/>
             </xs:element>
             <xs:complexType name="text">
               <xs:sequence>
                 <xs:element name="en"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="german">
               <xs:sequence>
                 <xs:element name="de"/>
               </xs:sequence>
             </xs:complexType>
           </xs:schema>"#
    )])?;
    // `chap` carries no `lang` of its own, so CTA's `test` must see `doc`'s
    // (`CTA/cta0009`).
    assert_eq!(
        code(&schema, r#"<doc lang="de"><chap><de/></chap></doc>"#),
        None
    );
    assert_eq!(
        code(&schema, r#"<doc lang="fr"><chap><en/></chap></doc>"#),
        None
    );
    assert_eq!(
        code(&schema, r#"<doc lang="de"><chap><en/></chap></doc>"#).as_deref(),
        Some("cvc-complex-type.2.4.a")
    );
    Ok(())
}

#[test]
fn an_attributes_own_inheritable_overrides_its_declarations() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:attribute name="lang" type="xs:language"/>
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="chap" maxOccurs="unbounded"/>
                 </xs:sequence>
                 <xs:attribute ref="lang" inheritable="true"/>
               </xs:complexType>
             </xs:element>
             <!-- `xs:anyType`, so that §3.3.6.1 clause 7 holds of both
                  alternatives. -->
             <xs:element name="chap" type="xs:anyType">
               <xs:alternative test="@lang = 'de'" type="german"/>
               <xs:alternative type="text"/>
             </xs:element>
             <xs:complexType name="text">
               <xs:sequence>
                 <xs:element name="en"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="german">
               <xs:sequence>
                 <xs:element name="de"/>
               </xs:sequence>
             </xs:complexType>
           </xs:schema>"#
    )])?;
    // The global `lang` declaration has no `inheritable` of its own (so it
    // defaults to `false`); the `ref`'d attribute *use* on `doc` says
    // `inheritable="true"` and must win — `CTA/cta0011`'s schema states this
    // outright: "the attribute use wins".
    assert_eq!(
        code(&schema, r#"<doc lang="de"><chap><de/></chap></doc>"#),
        None
    );
    Ok(())
}

#[test]
fn an_assert_sees_an_inherited_attribute_but_not_a_masked_one() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="part">
                     <xs:complexType>
                       <xs:sequence>
                         <xs:element name="leaf">
                           <xs:complexType>
                             <xs:assert test="@lang = 'de'"/>
                           </xs:complexType>
                         </xs:element>
                       </xs:sequence>
                       <xs:attribute name="lang" inheritable="false"/>
                     </xs:complexType>
                   </xs:element>
                 </xs:sequence>
                 <xs:attribute name="lang" inheritable="true"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    // `part`'s own `lang` is declared *not* inheritable, so it must be
    // transparent to the search rather than masking `doc`'s — `leaf` sees
    // straight past `part`'s `en` to `doc`'s `de` (`CTA/cta0013`'s schema
    // states the rule outright: "a non-inheritable attribute does not mask
    // an inheritable one on an ancestor").
    assert_eq!(
        code(
            &schema,
            r#"<doc lang="de"><part lang="en"><leaf/></part></doc>"#
        ),
        None
    );
    assert_eq!(
        code(
            &schema,
            r#"<doc lang="fr"><part lang="de"><leaf/></part></doc>"#
        )
        .as_deref(),
        Some("cvc-assertion")
    );
    Ok(())
}

#[test]
fn open_content_from_a_base_reaches_past_particles_an_extension_appends() -> Result<(), SchemaError>
{
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="b">
               <xs:openContent mode="suffix">
                 <xs:any namespace="http://open.example/" processContents="lax"/>
               </xs:openContent>
               <xs:sequence>
                 <xs:element name="a"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="r">
               <xs:complexContent>
                 <xs:extension base="b">
                   <xs:sequence>
                     <xs:element name="d" minOccurs="0"/>
                   </xs:sequence>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="doc" type="r"/>
           </xs:schema>"#
    )])?;
    // `r`'s own extension writes no `xs:openContent` of its own, so `b`'s
    // `suffix` wildcard must still apply — now reaching past `d`, the
    // particle the extension appended (`Open/open027`).
    assert_eq!(
        code(
            &schema,
            r#"<doc xmlns:o="http://open.example/"><a/><d/><o:extra/></doc>"#
        ),
        None
    );
    // It is still a *suffix*: the wildcard may not appear before `d`.
    assert!(
        code(
            &schema,
            r#"<doc xmlns:o="http://open.example/"><a/><o:extra/><d/></doc>"#
        )
        .is_some()
    );
    Ok(())
}

#[test]
fn an_extensions_own_open_content_unites_with_the_base_wildcard_it_inherits()
-> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="b">
               <xs:openContent mode="interleave">
                 <xs:any namespace="http://open.example/" processContents="skip"/>
               </xs:openContent>
               <xs:sequence>
                 <xs:element name="a"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="r">
               <xs:complexContent>
                 <xs:extension base="b">
                   <xs:openContent mode="interleave">
                     <xs:any notNamespace="http://open.example/ http://blocked.example/"
                             processContents="skip"/>
                   </xs:openContent>
                   <xs:sequence/>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="doc" type="r"/>
           </xs:schema>"#
    )])?;
    // The two wildcards are *united*, not nested (`Open/open047`): `r`
    // accepts both what `b`'s own wildcard admits...
    assert_eq!(
        code(
            &schema,
            r#"<doc xmlns:o="http://open.example/"><a/><o:extra/></doc>"#
        ),
        None
    );
    // ...and what its own (wider) wildcard admits.
    assert_eq!(
        code(
            &schema,
            r#"<doc xmlns:e="http://elsewhere.example/"><a/><e:extra/></doc>"#
        ),
        None
    );
    // But not the one namespace neither side admits.
    assert!(
        code(
            &schema,
            r#"<doc xmlns:b="http://blocked.example/"><a/><b:extra/></doc>"#
        )
        .is_some()
    );
    Ok(())
}

/// Regression guard for the *compile-time* half of the open-content fix: the
/// `interleave`-forcing rule (`cos-ct-extends.1.4.3.3`) must still see `r`'s
/// own (here, `xs:defaultOpenContent`-defaulted) open content *before* it is
/// merged with `b`'s — merging first would silently widen the mode to
/// `interleave` and make the schema wrongly assemble.
#[test]
fn an_interleave_base_still_rejects_a_defaulted_suffix_extension() {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:defaultOpenContent mode="suffix">
               <xs:any namespace="http://default.example/" processContents="lax"/>
             </xs:defaultOpenContent>
             <xs:complexType name="b">
               <xs:openContent mode="interleave">
                 <xs:any namespace="http://open.example/" processContents="lax"/>
               </xs:openContent>
               <xs:sequence>
                 <xs:element name="a"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="r">
               <xs:complexContent>
                 <xs:extension base="b">
                   <xs:sequence>
                     <xs:element name="d" minOccurs="0"/>
                   </xs:sequence>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="doc" type="r"/>
           </xs:schema>"#
    )]);
    assert!(schema.is_err());
}

// ---------------------------------------------------------------------------
// The XSTS 1.1-contribution uplift: §3.17.5.2's ID/IDREF table, §3.11.4's
// ·target node set·, §3.13.4.1's partial PSVI, §3.4.4.2 clauses 1.1 and 5,
// §3.4.2.3.3's content-type mapping, §3.10.6.3's wildcard union and
// §3.16.6.3 clause 2.2.4.3.
// ---------------------------------------------------------------------------

/// §3.17.5.2: an `ID` written as an element's *content* binds the element's
/// **parent**, so two `ID`-typed children of one element are a one-member
/// ·binding· and not a clash, while an `ID` child and an `ID` attribute
/// elsewhere are two members and are (`saxonData/Id/id003`).
#[test]
fn an_id_element_identifies_its_parent() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="node" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
             <xs:element name="node">
               <xs:complexType>
                 <xs:choice minOccurs="0" maxOccurs="unbounded">
                   <xs:element ref="node"/>
                   <xs:element name="id" type="xs:ID"/>
                 </xs:choice>
                 <xs:attribute name="one" type="xs:ID"/>
                 <xs:attribute name="two" type="xs:ID"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    // Two `id` children of the same `node`, and two ID attributes of one
    // element, all bind a single element each.
    assert_eq!(
        code(
            &schema,
            r#"<doc><node one="a" two="b"><id>z</id><id>z</id><node one="e" two="e"/></node></doc>"#
        ),
        None
    );
    // The same value on a *different* element makes the binding two-membered.
    assert_eq!(
        code(
            &schema,
            r#"<doc><node one="a" two="b"><id>z</id><node one="e" two="z"/></node></doc>"#
        )
        .as_deref(),
        Some("cvc-id.2")
    );
    Ok(())
}

/// §3.3.4.5 clause 1, with the Recommendation's own example: in
/// `<DOC><X>abcd</X></DOC>` the ID binds `DOC`, so when the `ID` element has no
/// parent inside the ·validation root· its ·binding· is empty and the document
/// is invalid (`ibmData/instance_invalid/S3_3_4/s3_3_4ii26`).
#[test]
fn an_id_element_without_a_parent_binds_nothing() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="root" type="xs:ID"/>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(&schema, "<root>abcd</root>").as_deref(),
        Some("cvc-id.1")
    );
    Ok(())
}

/// §3.11.4 clause 1: the ·target node set· omits ·skipped· elements, so a
/// duplicate key inside a `processContents="skip"` wildcard is invisible to
/// `xs:key` (`saxonData/Wild/wild101`).
#[test]
fn identity_constraints_do_not_see_skipped_content() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:choice maxOccurs="unbounded">
                   <xs:element ref="note"/>
                   <xs:element ref="wrapper"/>
                 </xs:choice>
               </xs:complexType>
               <xs:key name="k">
                 <xs:selector xpath=".//note"/>
                 <xs:field xpath="@id"/>
               </xs:key>
             </xs:element>
             <xs:element name="note">
               <xs:complexType>
                 <xs:attribute name="id" type="xs:string"/>
               </xs:complexType>
             </xs:element>
             <xs:element name="wrapper">
               <xs:complexType>
                 <xs:sequence>
                   <xs:any minOccurs="0" maxOccurs="unbounded" processContents="skip"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    // A duplicate `id`, and a `note` with none at all, are both fine inside
    // the skipped wrapper.
    assert_eq!(
        code(
            &schema,
            r#"<doc><note id="n1"/><wrapper><note id="n1"/><note/></wrapper></doc>"#
        ),
        None
    );
    // Outside it, the same duplicate is a key violation.
    assert_eq!(
        code(&schema, r#"<doc><note id="n1"/><note id="n1"/></doc>"#).as_deref(),
        Some("cvc-identity-constraint.4.2.2")
    );
    Ok(())
}

/// The last paragraph of §3.11.4: "single atomic values are not distinguished
/// from lists with single items" (`saxonData/Id/id022`).
#[test]
fn a_singleton_list_matches_an_atomic_key() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element ref="para" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
               <xs:key name="k">
                 <xs:selector xpath="para"/>
                 <xs:field xpath="@key"/>
               </xs:key>
               <xs:keyref name="r" refer="k">
                 <xs:selector xpath="para"/>
                 <xs:field xpath="@ref"/>
               </xs:keyref>
             </xs:element>
             <xs:element name="para">
               <xs:complexType>
                 <xs:attribute name="key" type="xs:Name" use="required"/>
                 <xs:attribute name="ref" type="names"/>
               </xs:complexType>
             </xs:element>
             <xs:simpleType name="names">
               <xs:list itemType="xs:Name"/>
             </xs:simpleType>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(
            &schema,
            r#"<doc><para key="alpha"/><para key="beta" ref="alpha"/></doc>"#
        ),
        None
    );
    // A two-item list is a two-member key sequence and matches nothing.
    assert_eq!(
        code(
            &schema,
            r#"<doc><para key="alpha"/><para key="beta" ref="alpha beta"/></doc>"#
        )
        .as_deref(),
        Some("cvc-identity-constraint.4.3")
    );
    Ok(())
}

/// Datatypes §3.3: a timezoned and an un-timezoned date/time value are
/// incomparable, so they never satisfy a `keyref` (`saxonData/Zone/zone206`).
#[test]
fn an_untimezoned_time_matches_no_timezoned_key() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="target" type="xs:time"/>
                   <xs:element name="equiv" type="xs:time" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
               <xs:key name="t">
                 <xs:selector xpath="target"/>
                 <xs:field xpath="."/>
               </xs:key>
               <xs:keyref name="r" refer="t">
                 <xs:selector xpath="equiv"/>
                 <xs:field xpath="."/>
               </xs:keyref>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(
            &schema,
            "<doc><target>02:00:00-05:00</target><equiv>07:00:00Z</equiv></doc>"
        ),
        None
    );
    assert_eq!(
        code(
            &schema,
            "<doc><target>12:00:00Z</target><equiv>12:00:00</equiv></doc>"
        )
        .as_deref(),
        Some("cvc-identity-constraint.4.3")
    );
    Ok(())
}

/// Datatypes §4.3.13.3 clause 1.2: an `assertion` **facet** is evaluated with
/// no context item, so `.` raises a dynamic error and the assertion is false
/// (`saxonData/Assert/assert-simple008`).
#[test]
fn a_facet_assertion_has_no_context_item() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="a">
               <xs:simpleType>
                 <xs:restriction base="xs:date">
                   <xs:assertion test=". castable as xs:date"/>
                 </xs:restriction>
               </xs:simpleType>
             </xs:element>
             <xs:element name="b">
               <xs:simpleType>
                 <xs:restriction base="xs:date">
                   <xs:assertion test="$value lt xs:date('2100-01-01')"/>
                 </xs:restriction>
               </xs:simpleType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(
        code(&schema, "<a>2001-01-01</a>").as_deref(),
        Some("cvc-assertion")
    );
    // `$value`, which is all a facet assertion gets, still works.
    assert_eq!(code(&schema, "<b>2001-01-01</b>"), None);
    Ok(())
}

/// §3.13.4.1 clause 1.2: the assertion root is only *partially* assessed, so
/// it "will normally have the type annotation anyType" and atomizes to
/// `xs:untypedAtomic`; and comments are excluded from the partial PSVI
/// (`saxonData/Assert/assert014`, `assert023`).
#[test]
fn an_assertion_sees_an_untyped_root_and_no_comments() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="dated">
               <xs:simpleContent>
                 <xs:extension base="xs:date"/>
               </xs:simpleContent>
             </xs:complexType>
             <xs:element name="temp">
               <xs:complexType>
                 <xs:simpleContent>
                   <xs:restriction base="dated">
                     <xs:assert test="not(data(.) instance of xs:date)"/>
                   </xs:restriction>
                 </xs:simpleContent>
               </xs:complexType>
             </xs:element>
             <xs:element name="quiet">
               <xs:complexType>
                 <xs:sequence/>
                 <xs:assert test="empty(.//comment())"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<temp>2008-07-01</temp>"), None);
    assert_eq!(code(&schema, "<quiet><!-- hidden --></quiet>"), None);
    Ok(())
}

/// §3.4.4.2 clause 5: an element admitted by a wildcard is assessed against its
/// *global* declaration, and that ·governing type definition· must be validly
/// substitutable for the ·locally declared type· — which §3.4.4.1 looks for in
/// the base type as well (`saxonData/Wild/wild063`, `wild068`).
#[test]
fn a_wildcard_match_must_agree_with_the_locally_declared_type() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="zing">
               <xs:sequence>
                 <xs:element name="e" type="xs:integer"/>
                 <xs:any namespace="{LOCAL}" processContents="lax"/>
               </xs:sequence>
             </xs:complexType>
             <xs:element name="doc" type="zing"/>
             <xs:element name="e" type="xs:decimal"/>
           </xs:schema>"#
    )])?;
    // `xs:byte` is derived from the locally declared `xs:integer`.
    assert_eq!(
        code(
            &schema,
            r#"<doc><e>1</e><e xmlns:xs="http://www.w3.org/2001/XMLSchema"
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:type="xs:byte">3</e></doc>"#
        ),
        None
    );
    // The global `e` is `xs:decimal`, which is not.
    assert_eq!(
        code(&schema, "<doc><e>1</e><e>2.5</e></doc>").as_deref(),
        Some("cvc-complex-type.5")
    );
    Ok(())
}

/// §3.4.4.2 clause 1.1: empty content admits "no character … [children]" at
/// all, not even white space — unlike element-only content (clause 1.3). An
/// `xs:openContent` promotes the variety to element-only (§3.4.2.3.3 clause
/// 6.2), which restores the white-space tolerance
/// (`saxonData/Open/open012` against `open010`).
#[test]
fn empty_content_rejects_white_space_unless_it_is_opened() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="shut">
               <xs:complexType>
                 <xs:sequence/>
               </xs:complexType>
             </xs:element>
             <xs:element name="open">
               <xs:complexType>
                 <xs:openContent mode="suffix">
                   <xs:any namespace="http://open.example/" processContents="lax"/>
                 </xs:openContent>
                 <xs:sequence/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, "<shut></shut>"), None);
    assert_eq!(
        code(&schema, "<shut>\n  \n</shut>").as_deref(),
        Some("cvc-complex-type.2.1")
    );
    assert_eq!(code(&schema, "<open>\n  \n</open>"), None);
    Ok(())
}

/// §3.4.2.3.3 clause 2.1.3: a branch-less `xs:choice` is ·empty content· only
/// when its `minOccurs` is 0; with `minOccurs` 1 it accepts the empty
/// *language*, so even an empty element is invalid
/// (`saxonData/Complex/complex022`).
#[test]
fn a_branchless_choice_accepts_nothing() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="z">
               <xs:complexType>
                 <xs:choice/>
               </xs:complexType>
             </xs:element>
             <xs:element name="y">
               <xs:complexType>
                 <xs:choice minOccurs="0"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    assert!(code(&schema, "<z/>").is_some());
    assert_eq!(code(&schema, "<y/>"), None);
    Ok(())
}

/// §3.10.6.3: the wildcard union disallows a name that *neither* wildcard
/// allows, which includes one the other's namespace constraint excludes
/// (`saxonData/Wild/wild046`).
#[test]
fn a_wildcard_union_disallows_what_neither_allows() -> Result<(), SchemaError> {
    let xml = "http://www.w3.org/XML/1998/namespace";
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="base">
               <xs:sequence/>
               <xs:anyAttribute namespace="{LOCAL}" notQName="a b c" processContents="skip"/>
             </xs:complexType>
             <xs:complexType name="wider">
               <xs:complexContent>
                 <xs:extension base="base">
                   <xs:anyAttribute notNamespace="http://example.com/x"
                                    notQName="c d e xml:lang" processContents="skip"/>
                 </xs:extension>
               </xs:complexContent>
             </xs:complexType>
             <xs:element name="computer" type="wider"/>
           </xs:schema>"#
    )])?;
    assert_eq!(code(&schema, r#"<computer a="a"/>"#), None);
    assert_eq!(code(&schema, r#"<computer d="d"/>"#), None);
    assert!(code(&schema, r#"<computer c="c"/>"#).is_some());
    assert!(
        code(
            &schema,
            &format!(r#"<computer xmlns:xml="{xml}" xml:lang="de"/>"#)
        )
        .is_some()
    );
    Ok(())
}

/// §3.16.6.3 clause 2.2.4.3: `xsi:type` may name a member of a union only when
/// "the {facets} property of B and of any intervening union datatypes is
/// empty" (`saxonData/Simple/simple016`).
#[test]
fn a_faceted_union_hides_its_members() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="chap" type="chap"/>
             <xs:simpleType name="chap">
               <xs:union memberTypes="dt xs:time"/>
             </xs:simpleType>
             <xs:simpleType name="dt">
               <xs:restriction>
                 <xs:simpleType>
                   <xs:union memberTypes="xs:date xs:dateTime"/>
                 </xs:simpleType>
                 <xs:pattern value=".*Z"/>
               </xs:restriction>
             </xs:simpleType>
           </xs:schema>"#
    )])?;
    let head = r#"xmlns:xs="http://www.w3.org/2001/XMLSchema"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#;
    // `xs:time` is an unfaceted member of `chap` itself.
    assert_eq!(
        code(
            &schema,
            &format!(r#"<chap {head} xsi:type="xs:time">18:18:18Z</chap>"#)
        ),
        None
    );
    // `xs:date` is only reachable through `dt`, which has a pattern.
    assert!(
        code(
            &schema,
            &format!(r#"<chap {head} xsi:type="xs:date">2010-10-12Z</chap>"#)
        )
        .is_some()
    );
    Ok(())
}

/// §3.4.6.2 clause 1.4.3.2.2.1: "Both B and T have {content type} . {variety} =
/// mixed or both have {content type} . {variety} = element-only"
/// (`saxonData/Complex/complex025`, `complex026`).
#[test]
fn an_extension_may_not_change_the_content_variety() {
    let elementwise = format!(
        r#"{HEAD}
             <xs:complexType name="b">
               <xs:choice minOccurs="0" maxOccurs="unbounded">
                 <xs:element name="elem"/>
               </xs:choice>
             </xs:complexType>
             <xs:element name="root">
               <xs:complexType mixed="true">
                 <xs:complexContent>
                   <xs:extension base="b"/>
                 </xs:complexContent>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    );
    assert!(compile(&[&elementwise]).is_err());
    let mixedwise = format!(
        r#"{HEAD}
             <xs:complexType name="b" mixed="true">
               <xs:choice minOccurs="0" maxOccurs="unbounded">
                 <xs:element name="elem"/>
               </xs:choice>
             </xs:complexType>
             <xs:element name="root">
               <xs:complexType>
                 <xs:complexContent>
                   <xs:extension base="b">
                     <xs:sequence>
                       <xs:element name="elem2"/>
                     </xs:sequence>
                   </xs:extension>
                 </xs:complexContent>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    );
    assert!(compile(&[&mixedwise]).is_err());
}

/// §3.4.3 (*Complex Type Definition Representation OK*) clause 5: `mixed`
/// written on both `xs:complexType` and `xs:complexContent` must agree
/// (`saxonData/Complex/complex002`).
#[test]
fn the_two_mixed_attributes_may_not_disagree() {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:complexType name="b" mixed="true">
               <xs:sequence>
                 <xs:element name="a" type="xs:string"/>
               </xs:sequence>
             </xs:complexType>
             <xs:complexType name="r" mixed="true">
               <xs:complexContent mixed="false">
                 <xs:restriction base="b">
                   <xs:sequence>
                     <xs:element name="a" type="xs:string"/>
                   </xs:sequence>
                 </xs:restriction>
               </xs:complexContent>
             </xs:complexType>
           </xs:schema>"#
    )]);
    assert!(schema.is_err());
}

/// §3.3.6.1 clause 7: every `xs:alternative`'s type must be validly
/// substitutable for the element's declared type
/// (`ibmData/schema_invalid/S3_12/s3_12si03`).
#[test]
fn a_type_alternative_must_derive_from_the_declared_type() {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="root" type="xs:integer">
               <xs:alternative test="@kind='string'" type="xs:string"/>
             </xs:element>
           </xs:schema>"#
    )]);
    assert!(schema.is_err());
}

/// §3.8.6.3: a content model ·implicitly contains· the members of the
/// ·substitution group· of every declaration it contains, so a local `n` and a
/// global `n` standing in a referenced substitution group must agree about
/// their type (`saxonData/Subsgroup/subsgroup901`).
#[test]
fn element_declarations_consistent_sees_substitution_groups() {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:element name="back">
               <xs:complexType>
                 <xs:sequence>
                   <xs:element name="n" type="xs:date"/>
                   <xs:element ref="appendixContent" minOccurs="0" maxOccurs="unbounded"/>
                 </xs:sequence>
               </xs:complexType>
             </xs:element>
             <xs:element name="appendixContent" abstract="true"/>
             <xs:element name="n" substitutionGroup="appendixContent" type="xs:string"/>
           </xs:schema>"#
    )]);
    assert!(schema.is_err());
}

/// §3.11.3 (*Identity-constraint Definition Representation OK*) clauses 1, 4
/// and 5 (`saxonData/Id/id041`,
/// `ibmData/schema_invalid/S2_2_4/s2_2_4si02`, `s2_2_4si07`).
#[test]
fn an_identity_constraint_reference_is_checked() {
    let with_wrong_category = format!(
        r#"{HEAD}
             <xs:element name="doc">
               <xs:complexType>
                 <xs:choice maxOccurs="unbounded">
                   <xs:element name="chap" type="chap">
                     <xs:unique name="u">
                       <xs:selector xpath="section"/>
                       <xs:field xpath="@nr"/>
                     </xs:unique>
                   </xs:element>
                   <xs:element name="appx" type="chap">
                     <xs:key ref="u"/>
                   </xs:element>
                 </xs:choice>
               </xs:complexType>
             </xs:element>
             <xs:complexType name="chap">
               <xs:sequence>
                 <xs:element name="section">
                   <xs:complexType>
                     <xs:attribute name="nr" type="xs:decimal"/>
                   </xs:complexType>
                 </xs:element>
               </xs:sequence>
             </xs:complexType>
           </xs:schema>"#
    );
    assert!(compile(&[&with_wrong_category]).is_err());
    let with_extra_attribute = with_wrong_category.replace(
        r#"<xs:key ref="u"/>"#,
        r#"<xs:unique ref="u" name="also"/>"#,
    );
    assert!(compile(&[&with_extra_attribute]).is_err());
}

/// XSD 1.1 §3.6.6 dropped the 1.0 prohibition on an attribute group referring
/// to itself (W3C bug 15795): `msData/attributeGroup/attgC010`, `attgC020`,
/// `attgC031` and `attgD015` are all marked invalid for 1.0 and valid for 1.1.
#[test]
fn a_circular_attribute_group_is_permitted_in_1_1() -> Result<(), SchemaError> {
    const TEXT: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:element name="T" type="test"/>
        <xs:complexType name="test">
            <xs:attributeGroup ref="test"/>
        </xs:complexType>
        <xs:attributeGroup name="test">
            <xs:attributeGroup ref="test"/>
            <xs:attribute name="foo" type="xs:int"/>
            <xs:attribute name="bar"/>
        </xs:attributeGroup>
    </xs:schema>"#;
    let schema = compile(&[TEXT])?;
    // The self-reference contributes nothing new; the two direct attributes stay.
    assert!(schema.validate_str(r#"<T foo="1" bar="x"/>"#).valid);
    // The `foo` use survived the cycle, so a non-integer value is caught.
    assert!(!schema.validate_str(r#"<T foo="notint" bar="x"/>"#).valid);
    // XSD 1.0 still rejects the cycle.
    let mut one_zero = SchemaSet::new().with_version(XsdVersion::V1_0);
    one_zero.add_document(None, TEXT)?;
    assert_eq!(
        one_zero
            .compile()
            .expect_err("XSD 1.0 forbids the cycle")
            .code(),
        "src-attribute_group.3"
    );
    Ok(())
}

/// The {attribute uses} of a group in an XSD 1.1 reference cycle are the least
/// fixed point of the graph, so a complex type that reaches the cycle at either
/// group requires *every* attribute the cycle contributes — the group entered
/// second must not be left short of what lies past the back-edge.
#[test]
fn an_attribute_group_cycle_contributes_every_use_at_either_entry() -> Result<(), SchemaError> {
    let schema = compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:element name="t1" type="ty1"/>
        <xs:element name="t2" type="ty2"/>
        <xs:complexType name="ty1"><xs:attributeGroup ref="A"/></xs:complexType>
        <xs:complexType name="ty2"><xs:attributeGroup ref="B"/></xs:complexType>
        <xs:attributeGroup name="A">
            <xs:attribute name="a1" type="xs:int" use="required"/>
            <xs:attributeGroup ref="B"/>
        </xs:attributeGroup>
        <xs:attributeGroup name="B">
            <xs:attribute name="b1" type="xs:int" use="required"/>
            <xs:attributeGroup ref="A"/>
        </xs:attributeGroup>
    </xs:schema>"#])?;
    // Both entries require both attributes, whichever group the type names.
    assert!(schema.validate_str(r#"<t1 a1="1" b1="2"/>"#).valid);
    assert!(schema.validate_str(r#"<t2 a1="1" b1="2"/>"#).valid);
    assert!(
        !schema.validate_str(r#"<t1 a1="1"/>"#).valid,
        "`b1` is required through the cycle"
    );
    assert!(
        !schema.validate_str(r#"<t2 b1="2"/>"#).valid,
        "`a1` is required through the cycle, from the second-entered group"
    );
    Ok(())
}

/// A `choice` restricting a `choice` by keeping a single sequence branch is a
/// valid restriction under XSD 1.1 (`Particles_w3c/particlesZ023`): once
/// `simplify` rewrites the one-branch restriction as that branch, the pair
/// reaches `map_and_sum`, whose 1.0 sum rejects it but whose 1.1 branch does not.
#[test]
fn a_choice_keeping_one_sequence_branch_restricts_in_1_1() -> Result<(), SchemaError> {
    compile(&[
        r#"<xs:schema targetNamespace="http://myuri" xmlns="http://myuri"
                    xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:element name="A" type="xs:string"/>
        <xs:element name="B" type="xs:string"/>
        <xs:complexType name="eleType">
            <xs:sequence>
                <xs:element ref="A"/>
                <xs:element ref="B"/>
                <xs:choice>
                    <xs:sequence>
                        <xs:element name="AAA" minOccurs="0" maxOccurs="unbounded"/>
                        <xs:element name="BBB" minOccurs="0" maxOccurs="unbounded"/>
                    </xs:sequence>
                    <xs:sequence>
                        <xs:element name="CCC" minOccurs="0" maxOccurs="unbounded"/>
                    </xs:sequence>
                </xs:choice>
            </xs:sequence>
        </xs:complexType>
        <xs:complexType name="eleType2">
            <xs:complexContent>
                <xs:restriction base="eleType">
                    <xs:sequence>
                        <xs:element ref="A"/>
                        <xs:element ref="B"/>
                        <xs:choice>
                            <xs:sequence>
                                <xs:element name="AAA" minOccurs="0" maxOccurs="unbounded"/>
                                <xs:element name="BBB" minOccurs="0" maxOccurs="unbounded"/>
                            </xs:sequence>
                        </xs:choice>
                    </xs:sequence>
                </xs:restriction>
            </xs:complexContent>
        </xs:complexType>
    </xs:schema>"#,
    ])?;
    Ok(())
}

/// The same, keeping one repeated group branch (`Particles_w3c/particlesZ024`),
/// and narrowing a two-branch group choice to one (`Group_w3c/groupH021v`).
#[test]
fn a_choice_narrowed_to_one_group_restricts_in_1_1() -> Result<(), SchemaError> {
    compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:complexType name="A">
            <xs:choice>
                <xs:group ref="x" maxOccurs="unbounded"/>
                <xs:group ref="y" maxOccurs="unbounded"/>
            </xs:choice>
        </xs:complexType>
        <xs:group name="x">
            <xs:sequence>
                <xs:element name="x1"/>
                <xs:element name="x2"/>
            </xs:sequence>
        </xs:group>
        <xs:group name="y">
            <xs:choice>
                <xs:element name="y1"/>
                <xs:element name="y2"/>
            </xs:choice>
        </xs:group>
        <xs:element name="elem">
            <xs:complexType>
                <xs:complexContent>
                    <xs:restriction base="A">
                        <xs:choice>
                            <xs:group ref="x" maxOccurs="unbounded"/>
                        </xs:choice>
                    </xs:restriction>
                </xs:complexContent>
            </xs:complexType>
        </xs:element>
    </xs:schema>"#])?;
    Ok(())
}

/// An optional `choice` restricting a `choice` that is required once but has an
/// emptiable branch is a valid restriction under XSD 1.1
/// (`Additional_w3c/addB118`): the base admits the empty sequence, so a
/// restriction that may be absent is still contained in it.
#[test]
fn an_optional_choice_restricts_an_emptiable_required_choice_in_1_1() -> Result<(), SchemaError> {
    compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
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
    </xs:schema>"#])?;
    Ok(())
}

/// A restriction that lists a repeated substitution-group choice's members
/// inline maps onto the base's unbounded group under XSD 1.1
/// (`Particles_w3c/particlesZ028`): `flatten` splices the base group back in,
/// and an unbounded base particle may be the image of several consecutive
/// restriction particles.
#[test]
fn an_inline_sequence_maps_onto_a_repeated_group_in_1_1() -> Result<(), SchemaError> {
    compile(&[r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:element abstract="true" name="aba" type="xs:string"/>
        <xs:element abstract="true" name="abb" type="xs:int"/>
        <xs:element name="a" substitutionGroup="aba" type="xs:string"/>
        <xs:element name="b" substitutionGroup="abb" type="xs:int"/>
        <xs:element name="d" type="xs:anyURI"/>
        <xs:complexType name="test">
            <xs:sequence>
                <xs:group maxOccurs="unbounded" minOccurs="0" ref="abs"/>
                <xs:element minOccurs="0" ref="d"/>
            </xs:sequence>
        </xs:complexType>
        <xs:complexType name="test4">
            <xs:complexContent>
                <xs:restriction base="test">
                    <xs:sequence>
                        <xs:sequence minOccurs="1" maxOccurs="1">
                            <xs:element ref="a"/>
                            <xs:element ref="b"/>
                        </xs:sequence>
                        <xs:element ref="d"/>
                    </xs:sequence>
                </xs:restriction>
            </xs:complexContent>
        </xs:complexType>
        <xs:group name="abs">
            <xs:choice>
                <xs:element ref="aba"/>
                <xs:element ref="abb"/>
            </xs:choice>
        </xs:group>
    </xs:schema>"#])?;
    Ok(())
}

/// `saxonData/Open/open009`: `mode="suffix"` open content inside an `xs:all`
/// group. §3.4.2.4 makes the wildcard matches a *suffix* — the content model
/// must already be satisfied before the first of them, and nothing but further
/// wildcard matches may follow — which an unordered group has to observe just
/// as an ordered one does.
#[test]
fn open009_suffix_open_content_inside_an_all_group_keeps_its_place() -> Result<(), SchemaError> {
    let schema = compile(&[&format!(
        r#"{HEAD}
             <xs:defaultOpenContent mode="suffix" appliesToEmpty="false">
               <xs:any namespace="http://open.com/" processContents="lax"/>
             </xs:defaultOpenContent>
             <xs:element name="doc">
               <xs:complexType>
                 <xs:all>
                   <xs:element name="a" maxOccurs="2"/>
                   <xs:element name="b" minOccurs="0"/>
                   <xs:element name="c" minOccurs="0"/>
                 </xs:all>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )])?;
    const EXTRA: &str = r#"<extra xmlns="http://open.com/">42</extra>"#;
    // `v1`: the whole model, then the wildcard matches.
    assert_eq!(
        code(&schema, &format!("<doc><c/><a/><b/>{EXTRA}{EXTRA}</doc>")),
        None
    );
    // `v2`: the model alone, in any order and with no wildcard match at all.
    assert_eq!(code(&schema, "<doc><c/><a/><b/><a/></doc>"), None);
    // `n1`: a content-model element written *after* the suffix has begun.
    assert_eq!(
        code(
            &schema,
            &format!("<doc><c/><a/><b/>{EXTRA}{EXTRA}<a/></doc>")
        )
        .as_deref(),
        Some("cvc-complex-type.2.4.a")
    );
    // `n2`: the suffix begins before the model is satisfied — `a` is required.
    assert_eq!(
        code(&schema, &format!("<doc><c/><b/>{EXTRA}{EXTRA}</doc>")).as_deref(),
        Some("cvc-complex-type.2.4.a")
    );
    Ok(())
}

/// `ibmData/instance_invalid/D4_3_15/d4_3_15ii31` and `d4_3_15ii32`: the XDM
/// instance an assertion sees is rooted at the **element**, not at a document
/// node (§3.13.4.2), so `//` — which begins with
/// `fn:root(.) treat as document-node()` — raises `err:XPDY0050` and leaves
/// the assertion unsatisfied. That is how the Recommendation keeps an
/// assertion inside its own subtree; both fixtures are expected invalid even
/// though the subtree does hold exactly one matching node.
#[test]
fn d4_3_15ii31_a_leading_slash_cannot_reach_out_of_the_assertion_subtree() -> Result<(), SchemaError>
{
    const ELEMENTS: &str = r#"
        <xs:element name="root" type="rootType"/>
        <xs:complexType name="rootType">
          <xs:sequence>
            <xs:element name="ele1" type="elementType1" minOccurs="0" maxOccurs="unbounded"/>
            <xs:element name="ele2" type="elementType2" minOccurs="0" maxOccurs="unbounded"/>
          </xs:sequence>
        </xs:complexType>
        <xs:complexType name="elementType1">
          <xs:sequence>
            <xs:element name="subElement1" type="rootType" minOccurs="0" maxOccurs="unbounded"/>
          </xs:sequence>
          <xs:attribute name="attr1" type="xs:string"/>
        </xs:complexType>"#;
    const INSTANCE: &str = r#"<root>
          <ele1 attr1="1">
            <subElement1>
              <ele2 attr2="3"><subElement2><ele1 attr1="2"/></subElement2></ele2>
            </subElement1>
          </ele1>
        </root>"#;
    for test in ["count(//ele1) eq 1", "count(//@attr1) eq 1"] {
        let schema = compile(&[&format!(
            r#"{HEAD}{ELEMENTS}
                 <xs:complexType name="elementType2">
                   <xs:sequence>
                     <xs:element name="subElement2" type="rootType"
                                 minOccurs="0" maxOccurs="unbounded"/>
                   </xs:sequence>
                   <xs:attribute name="attr2" type="xs:string"/>
                   <xs:assert test="{test}"/>
                 </xs:complexType>
               </xs:schema>"#
        )])?;
        assert_eq!(
            code(&schema, INSTANCE).as_deref(),
            Some("cvc-assertion"),
            "`{test}` reached outside the assertion's own subtree"
        );
    }
    // The control: the same counts written *relatively* stay inside the
    // subtree and are answerable, so the rejection above is about the root of
    // the fabricated tree and not about the expression failing to match.
    for test in ["count(.//ele1) eq 1", "count(.//@attr1) eq 1"] {
        let schema = compile(&[&format!(
            r#"{HEAD}{ELEMENTS}
                 <xs:complexType name="elementType2">
                   <xs:sequence>
                     <xs:element name="subElement2" type="rootType"
                                 minOccurs="0" maxOccurs="unbounded"/>
                   </xs:sequence>
                   <xs:attribute name="attr2" type="xs:string"/>
                   <xs:assert test="{test}"/>
                 </xs:complexType>
               </xs:schema>"#
        )])?;
        assert_eq!(
            code(&schema, INSTANCE),
            None,
            "`{test}` should be satisfied inside the subtree"
        );
    }
    Ok(())
}
