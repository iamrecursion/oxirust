//! Per-document import licensing: which namespaces one schema document may
//! name in a `QName` reference.
//!
//! The schema representation constraint *QName resolution (Schema Document)*
//! ([XSD 1.0 §3.15.3][r10], [XSD 1.1 §3.17.6.2][r11], clause 4) makes that a
//! question about the document the reference is *written in* and about no
//! other, however completely the components of the other namespace were
//! loaded. W3C bug 4057 settled the reading these groups test; every test here
//! is named after the group of the [W3C XML Schema Test Suite][xsts] it
//! reproduces and carries its documents inline.
//!
//! [r10]: https://www.w3.org/TR/xmlschema-1/#src-resolve
//! [r11]: https://www.w3.org/TR/xmlschema11-1/#src-resolve
//! [xsts]: https://www.w3.org/XML/2004/xml-schema-test-suite/

use std::collections::HashMap;

use oxixml_schema::{ResolvedSchema, SchemaError, SchemaResolver, SchemaSet, XsdVersion};

/// Serves schema documents from a map, so that `xs:import` and `xs:include`
/// are exercised without touching the file system.
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

/// Asserts that both versions accept `text`, which refers to nothing outside
/// itself.
#[track_caller]
fn accepted(what: &str, text: &str) {
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        let mut set = SchemaSet::new().with_version(version);
        if let Err(error) = set
            .add_document(None, text)
            .and_then(|()| set.compile())
            .map(|_| ())
        {
            panic!("{what}: rejected under {version:?}: {error}");
        }
    }
}

/// `msData/element/elemZ006`, `msData/schema/schZ004`: an `xs:import` is a
/// licence granted to the document that writes it, and only to that document.
/// Both fixtures import namespace `a`, whose own document imports `b`, and
/// then refer to `b` themselves — which *QName resolution (Schema Document)*
/// clause 4.2 refuses, however completely the components of `b` were loaded
/// (W3C bug 4057).
#[test]
fn elem_z006_an_import_does_not_license_the_imports_of_what_it_imports() {
    const B: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         targetNamespace="b">
                         <xs:complexType name="b"><xs:sequence/></xs:complexType>
                       </xs:schema>"#;
    const A: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         targetNamespace="a">
                         <xs:import namespace="b" schemaLocation="b.xsd"/>
                         <xs:complexType name="a"><xs:sequence/></xs:complexType>
                       </xs:schema>"#;
    const MAIN: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                            targetNamespace="main" xmlns:a="a" xmlns:b="b">
                            <xs:import namespace="a" schemaLocation="a.xsd"/>
                            <xs:complexType name="ct">
                              <xs:sequence>
                                <xs:element name="a" type="a:a"/>
                                <xs:element name="b" type="b:b"/>
                              </xs:sequence>
                            </xs:complexType>
                          </xs:schema>"#;
    let resolver = MemoryResolver::default().with("a.xsd", A).with("b.xsd", B);
    let error = assemble_with(XsdVersion::V1_0, resolver, MAIN)
        .expect_err("elemZ006: `b` is reachable but not licensed");
    assert_eq!(error.code(), "src-resolve");
    // The control: importing `b` as well makes exactly the same reference
    // legal, so the rule is about the licence and not about reachability.
    let resolver = MemoryResolver::default().with("a.xsd", A).with("b.xsd", B);
    assemble_with(
        XsdVersion::V1_0,
        resolver,
        &MAIN.replace(
            r#"<xs:import namespace="a" schemaLocation="a.xsd"/>"#,
            r#"<xs:import namespace="a" schemaLocation="a.xsd"/>
               <xs:import namespace="b" schemaLocation="b.xsd"/>"#,
        ),
    )
    .expect("elemZ006: the imported namespace is licensed");
}

/// `msData/element/elemZ007`, `msData/schema/schZ005`: the licence does not
/// travel through `xs:include` either. The *included* document writes the
/// `xs:import`; the *including* one makes the reference, and is refused.
#[test]
fn elem_z007_an_include_does_not_inherit_the_included_documents_imports() {
    const B: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         targetNamespace="b">
                         <xs:element name="b"/>
                       </xs:schema>"#;
    const INCLUDED: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                                <xs:import namespace="b" schemaLocation="b.xsd"/>
                                <xs:element name="a"/>
                              </xs:schema>"#;
    const MAIN: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                            targetNamespace="main" xmlns:a="main" xmlns:b="b">
                            <xs:include schemaLocation="included.xsd"/>
                            <xs:complexType name="ct1">
                              <xs:sequence>
                                <xs:element ref="a:a"/>
                                <xs:element ref="b:b"/>
                              </xs:sequence>
                            </xs:complexType>
                          </xs:schema>"#;
    let resolver = MemoryResolver::default()
        .with("included.xsd", INCLUDED)
        .with("b.xsd", B);
    let error = assemble_with(XsdVersion::V1_0, resolver, MAIN)
        .expect_err("elemZ007: the included document's import is not the includer's");
    assert_eq!(error.code(), "src-resolve");
    // The reference into the chameleon-included document's own namespace —
    // `main`, which it was absorbed into — stays legal.
    let resolver = MemoryResolver::default()
        .with("included.xsd", INCLUDED)
        .with("b.xsd", B);
    assemble_with(
        XsdVersion::V1_0,
        resolver,
        &MAIN.replace(r#"<xs:element ref="b:b"/>"#, ""),
    )
    .expect("elemZ007: a chameleon include is licensed for the including namespace");
}

/// `msData/schema/schG9`: a document that imports two namespaces is licensed
/// for those two, and a third reached through one of *their* imports is
/// refused.
#[test]
fn sch_g9_only_the_namespaces_a_document_imports_itself_are_licensed() {
    const OTHER: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             targetNamespace="ns-c">
                             <xs:complexType name="ct-A"><xs:sequence/></xs:complexType>
                           </xs:schema>"#;
    const B: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         targetNamespace="ns-b" xmlns:c="ns-c">
                         <xs:import namespace="ns-c" schemaLocation="c.xsd"/>
                         <xs:complexType name="ct-A"><xs:sequence/></xs:complexType>
                       </xs:schema>"#;
    const MAIN: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                            targetNamespace="ns-a" xmlns:b="ns-b" xmlns:c="ns-c">
                            <xs:import namespace="ns-b" schemaLocation="b.xsd"/>
                            <xs:element name="a-e2" type="b:ct-A"/>
                            <xs:element name="a-e3" type="c:ct-A"/>
                          </xs:schema>"#;
    let resolver = MemoryResolver::default()
        .with("b.xsd", B)
        .with("c.xsd", OTHER);
    let error =
        assemble_with(XsdVersion::V1_1, resolver, MAIN).expect_err("schG9: `ns-c` is not licensed");
    assert_eq!(error.code(), "src-resolve");
    let resolver = MemoryResolver::default()
        .with("b.xsd", B)
        .with("c.xsd", OTHER);
    assemble_with(
        XsdVersion::V1_1,
        resolver,
        &MAIN.replace(r#"<xs:element name="a-e3" type="c:ct-A"/>"#, ""),
    )
    .expect("schG9: the directly imported namespace is licensed");
}

/// `msData/identityConstraint/idC019`: the licence is over an *optional*
/// namespace. The refused reference here is to the **absent** namespace, made
/// from a document that has a `targetNamespace` and imports nothing — clause
/// 4.1, which a set of namespace *names* could not express.
#[test]
fn id_c019_a_target_namespace_does_not_license_an_unqualified_reference() {
    const OTHER: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                             xmlns:o="diffNS" targetNamespace="diffNS">
                             <xs:element name="otherSubRoot">
                               <xs:complexType>
                                 <xs:sequence><xs:element ref="o:otherElement"/></xs:sequence>
                               </xs:complexType>
                               <xs:keyref name="keyrefName" refer="keyName">
                                 <xs:selector xpath=".//otherElement"/>
                                 <xs:field xpath="@keyData"/>
                               </xs:keyref>
                             </xs:element>
                             <xs:element name="otherElement">
                               <xs:complexType>
                                 <xs:attribute name="keyData" type="xs:string"/>
                               </xs:complexType>
                             </xs:element>
                           </xs:schema>"#;
    const MAIN: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                            <xs:import namespace="diffNS" schemaLocation="other.xsd"/>
                            <xs:element name="root">
                              <xs:complexType>
                                <xs:sequence><xs:element ref="keyElement"/></xs:sequence>
                              </xs:complexType>
                              <xs:key name="keyName">
                                <xs:selector xpath=".//keyElement"/>
                                <xs:field xpath="@keyField"/>
                              </xs:key>
                            </xs:element>
                            <xs:element name="keyElement">
                              <xs:complexType>
                                <xs:attribute name="keyField" type="xs:string"/>
                              </xs:complexType>
                            </xs:element>
                          </xs:schema>"#;
    let resolver = MemoryResolver::default().with("other.xsd", OTHER);
    let error = assemble_with(XsdVersion::V1_0, resolver, MAIN)
        .expect_err("idC019: the absent namespace is not licensed for `diffNS`");
    assert_eq!(error.code(), "src-resolve");
    // Clause 4.1.2: an `xs:import` with no `namespace` licenses exactly this.
    let resolver = MemoryResolver::default().with(
        "other.xsd",
        &OTHER.replace(
            r#"<xs:element name="otherSubRoot">"#,
            r#"<xs:import/><xs:element name="otherSubRoot">"#,
        ),
    );
    assemble_with(XsdVersion::V1_0, resolver, MAIN)
        .expect("idC019: an import with no namespace licenses the absent one");
}

/// `ibmData/schema_invalid/S4_2_6/s4_2_6si01`: the same rule reaches the XSD
/// 1.1 pass, and a document that declares no `targetNamespace` at all is
/// licensed for the absent namespace without importing anything.
#[test]
fn s4_2_6si01_the_licence_is_checked_under_both_versions() {
    const B: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                         targetNamespace="b">
                         <xs:element name="b" type="xs:string"/>
                       </xs:schema>"#;
    const MAIN: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
                            targetNamespace="a" xmlns:b="b">
                            <xs:complexType name="ct">
                              <xs:sequence><xs:element ref="b:b"/></xs:sequence>
                            </xs:complexType>
                          </xs:schema>"#;
    for version in [XsdVersion::V1_0, XsdVersion::V1_1] {
        let resolver = MemoryResolver::default().with("b.xsd", B);
        let error = assemble_with(version, resolver, MAIN)
            .expect_err("s4_2_6si01: `b` was never imported here");
        assert_eq!(error.code(), "src-resolve");
    }
    // A document with no `targetNamespace` may name the absent namespace, and
    // every document may name the XSD and XML namespaces without an import
    // (`msData/additional/addB187`).
    accepted(
        "s4_2_6si01",
        r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
             <xs:complexType name="ct">
               <xs:sequence><xs:element name="e" type="xs:string"/></xs:sequence>
               <xs:attribute ref="xml:lang"/>
             </xs:complexType>
             <xs:element name="e2" type="ct"/>
           </xs:schema>"#,
    );
}
