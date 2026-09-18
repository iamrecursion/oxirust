//! Assessment rules that depend on the *shape* of the instance rather than on
//! any one schema component: the `ID` uniqueness of §3.3.4.5 over values
//! written as element content, and the `use="prohibited"` corner of §3.4.1.

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

/// The schema of `sunData/ElemDecl/identityConstraintDefs/idConstrDefs00301m`
/// in miniature: three local declarations whose types are `xs:ID`, `xs:IDREF`
/// and `xs:IDREFS`.
const ELEMENT_CONTENT_IDS: &str = r#"
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
      <xs:element name="root">
        <xs:complexType>
          <xs:choice minOccurs="0" maxOccurs="unbounded">
            <xs:element name="Element" type="xs:ID"/>
            <xs:element name="ElementRef" type="xs:IDREF"/>
          </xs:choice>
        </xs:complexType>
      </xs:element>
    </xs:schema>"#;

/// `idConstrDefs00301m2_n`: two `xs:ID`-typed children carry the same value.
/// §3.3.4.5 binds each of them to their *parent*, so XSD 1.0 — which asks that
/// every `ID` value in the document be distinct — rejects the instance, while
/// XSD 1.1 counts the ·binding· and accepts it.
#[test]
fn id_constr_defs_00301m_repeated_element_content_ids_split_by_version() {
    let instance = "<root><Element>e001</Element><ElementRef>e001</ElementRef>\
                    <Element>e001</Element></root>";
    let outcome = schema(ELEMENT_CONTENT_IDS).validate_str(instance);
    assert_eq!(code(&outcome), Some("cvc-id.2"));
    assert!(schema_1_1(ELEMENT_CONTENT_IDS).validate_str(instance).valid);
}

/// `idConstrDefs00301m1_p`: distinct values are fine in either version.
#[test]
fn id_constr_defs_00301m_distinct_element_content_ids_are_accepted() {
    let instance = "<root><Element>e001</Element><ElementRef>e001</ElementRef>\
                    <Element>e002</Element></root>";
    assert!(schema(ELEMENT_CONTENT_IDS).validate_str(instance).valid);
    assert!(schema_1_1(ELEMENT_CONTENT_IDS).validate_str(instance).valid);
}

/// A restriction that keeps a `fixed` value constraint on an attribute use of
/// the given kind. XSD 1.0 lets `use="prohibited"` carry one; §3.4.1 makes the
/// prohibited use contribute nothing whatever, so the constraint is inert.
fn constrained_use(kind: &str) -> String {
    format!(
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:element name="root">
               <xs:complexType>
                 <xs:attribute name="tag" type="xs:string" use="{kind}" fixed="x1"/>
               </xs:complexType>
             </xs:element>
           </xs:schema>"#
    )
}

/// The names a `default` or `fixed` value constraint added to an element.
fn defaults_on_root(text: &str) -> Vec<String> {
    let document = oxixml_dom::build::parse_str("<root/>").expect("a well-formed instance");
    let root = document.document_element().expect("a document element");
    let outcome = schema(text).validate_document(&document);
    assert!(outcome.valid, "unexpected {:?}", outcome.errors);
    outcome
        .psvi
        .defaulted_attributes(root)
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect()
}

/// §3.4.1: an attribute use with `use="prohibited"` contributes no attribute
/// declaration, so a value constraint written beside it supplies nothing — the
/// element is assessed exactly as if the use were not there. §3.4.5 would
/// otherwise add it to the element's \[attributes\], where it would be
/// serialized, feed the ID/IDREF table and be selected by an
/// identity-constraint `field`.
#[test]
fn a_prohibited_attribute_use_supplies_no_default() {
    assert!(defaults_on_root(&constrained_use("prohibited")).is_empty());
}

/// The companion, which differs in one word: an `optional` use with the same
/// value constraint *does* default, so the test above measures the prohibition
/// rather than a missing feature.
#[test]
fn an_optional_attribute_use_still_supplies_its_default() {
    assert_eq!(defaults_on_root(&constrained_use("optional")), ["tag=x1"]);
}
