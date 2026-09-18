//! Building the DOM subtree for an `EncryptedData` or `EncryptedKey`.
//!
//! The XML Encryption schema is a `sequence`, so child order is fixed and this module always
//! emits `EncryptionMethod?`, `ds:KeyInfo?`, `CipherData`, `EncryptionProperties?` and — for an
//! `EncryptedKey` — `ReferenceList?`, `CarriedKeyName?` in that order.
//!
//! # Prefixes and declarations
//!
//! Four namespaces can appear: `xenc`, `xenc11`, `ds` and `dsig11`. Working out in advance which
//! ones a given model will actually need would mean a pre-pass that has to stay in sync with the
//! emitter forever, so instead every element declares the prefix it uses on itself and a single
//! sweep afterwards — [`crate::fragment::drop_redundant_namespace_declarations`] — removes the
//! ones an ancestor already provides. The result is the minimal, conventional shape: `xenc` on
//! the `EncryptedData`, `ds` on the `ds:KeyInfo`, `xenc11` on the `xenc11:MGF`, and nowhere
//! else.

use oxixml_dom::{Document, NodeId, build};

use crate::algorithm::{XMLDSIG_NS, XMLDSIG11_NS, XMLENC_NS, XMLENC11_NS};
use crate::error::Result;
use crate::model::agreement::{AgreementMethod, DhKeyValue, EcKeyValue, KeyDerivationMethod};
use crate::model::cipher::{CipherData, CipherReference, Transform, XmlFragment};
use crate::model::encrypted::{EncryptedData, EncryptedKey, EncryptedType};
use crate::model::keyinfo::{KeyInfo, KeyInfoItem, ReferenceList};
use crate::model::properties::EncryptionProperties;

/// The prefix this crate uses for the XML Encryption 1.0 namespace.
pub const XENC_PREFIX: &str = "xenc";
/// The prefix this crate uses for the XML Encryption 1.1 namespace.
pub const XENC11_PREFIX: &str = "xenc11";
/// The prefix this crate uses for the XML Signature namespace.
pub const DS_PREFIX: &str = "ds";
/// The prefix this crate uses for the XML Signature 1.1 namespace.
pub const DSIG11_PREFIX: &str = "dsig11";

/// Creates a namespace-qualified element and declares its prefix on itself.
fn element(doc: &mut Document, namespace: &str, prefix: &str, local: &str) -> Result<NodeId> {
    let node = doc.create_element_ns(Some(namespace), &format!("{prefix}:{local}"))?;
    doc.declare_namespace(node, Some(prefix), namespace)?;
    Ok(node)
}

/// Creates an `xenc:` element.
fn xenc(doc: &mut Document, local: &str) -> Result<NodeId> {
    element(doc, XMLENC_NS, XENC_PREFIX, local)
}

/// Appends a text child to `parent`.
fn text(doc: &mut Document, parent: NodeId, value: &str) -> Result<()> {
    let node = doc.create_text_node(value)?;
    doc.append_child(parent, node)?;
    Ok(())
}

/// Appends a base64-encoded text child to `parent`.
fn base64_text(doc: &mut Document, parent: NodeId, octets: &[u8]) -> Result<()> {
    text(doc, parent, &oxixml_text::base64::base64_encode(octets))
}

/// Appends the verbatim fragments in `fragments` to `parent`.
fn fragments(doc: &mut Document, parent: NodeId, fragments: &[XmlFragment]) -> Result<()> {
    for fragment in fragments {
        match fragment {
            XmlFragment::Text(value) => text(doc, parent, value)?,
            XmlFragment::Element(xml) => {
                let parsed = build::parse_str(xml)?;
                let Some(root) = parsed.document_element() else {
                    continue;
                };
                let imported = doc.import_node(&parsed, root, true)?;
                doc.append_child(parent, imported)?;
            }
        }
    }
    Ok(())
}

/// Writes the `EncryptionMethod`, `ds:KeyInfo`, `CipherData` and `EncryptionProperties`
/// children shared by both structures.
fn write_common(doc: &mut Document, parent: NodeId, common: &EncryptedType) -> Result<()> {
    if let Some(id) = &common.id {
        doc.set_attribute(parent, "Id", id.clone())?;
    }
    if let Some(content_type) = &common.content_type {
        doc.set_attribute(parent, "Type", content_type.uri().to_owned())?;
    }
    if let Some(mime_type) = &common.mime_type {
        doc.set_attribute(parent, "MimeType", mime_type.clone())?;
    }
    if let Some(encoding) = &common.encoding {
        doc.set_attribute(parent, "Encoding", encoding.clone())?;
    }

    if let Some(method) = &common.encryption_method {
        let node = xenc(doc, "EncryptionMethod")?;
        doc.set_attribute(node, "Algorithm", method.algorithm.clone())?;
        if let Some(bits) = method.key_size {
            let child = xenc(doc, "KeySize")?;
            text(doc, child, &bits.to_string())?;
            doc.append_child(node, child)?;
        }
        if let Some(label) = &method.oaep_params {
            let child = xenc(doc, "OAEPparams")?;
            base64_text(doc, child, label)?;
            doc.append_child(node, child)?;
        }
        if let Some(uri) = &method.mgf {
            let child = element(doc, XMLENC11_NS, XENC11_PREFIX, "MGF")?;
            doc.set_attribute(child, "Algorithm", uri.clone())?;
            doc.append_child(node, child)?;
        }
        if let Some(uri) = &method.digest_method {
            let child = element(doc, XMLDSIG_NS, DS_PREFIX, "DigestMethod")?;
            doc.set_attribute(child, "Algorithm", uri.clone())?;
            doc.append_child(node, child)?;
        }
        fragments(doc, node, &method.other)?;
        doc.append_child(parent, node)?;
    }

    if let Some(key_info) = &common.key_info {
        let node = write_key_info(doc, key_info)?;
        doc.append_child(parent, node)?;
    }

    let cipher_data = write_cipher_data(doc, &common.cipher_data)?;
    doc.append_child(parent, cipher_data)?;

    if let Some(properties) = &common.encryption_properties {
        // The schema requires at least one child, so an empty container is dropped rather than
        // written as an invalid element.
        if !properties.is_empty() {
            let node = write_properties(doc, properties)?;
            doc.append_child(parent, node)?;
        }
    }
    Ok(())
}

/// Builds a `xenc:CipherData` element.
fn write_cipher_data(doc: &mut Document, data: &CipherData) -> Result<NodeId> {
    let node = xenc(doc, "CipherData")?;
    match data {
        CipherData::Value(octets) => {
            let value = xenc(doc, "CipherValue")?;
            base64_text(doc, value, octets)?;
            doc.append_child(node, value)?;
        }
        CipherData::Reference(reference) => {
            let child = write_cipher_reference(doc, reference)?;
            doc.append_child(node, child)?;
        }
    }
    Ok(node)
}

/// Builds a `xenc:CipherReference` element.
fn write_cipher_reference(doc: &mut Document, reference: &CipherReference) -> Result<NodeId> {
    let node = xenc(doc, "CipherReference")?;
    doc.set_attribute(node, "URI", reference.uri.clone())?;
    if !reference.transforms.is_empty() {
        let transforms = xenc(doc, "Transforms")?;
        for transform in &reference.transforms {
            let child = write_transform(doc, transform)?;
            doc.append_child(transforms, child)?;
        }
        doc.append_child(node, transforms)?;
    }
    Ok(node)
}

/// Builds a `ds:Transform` element.
fn write_transform(doc: &mut Document, transform: &Transform) -> Result<NodeId> {
    let node = element(doc, XMLDSIG_NS, DS_PREFIX, "Transform")?;
    doc.set_attribute(node, "Algorithm", transform.algorithm.clone())?;
    fragments(doc, node, &transform.content)?;
    Ok(node)
}

/// Builds a `ds:KeyInfo` element.
fn write_key_info(doc: &mut Document, key_info: &KeyInfo) -> Result<NodeId> {
    let node = element(doc, XMLDSIG_NS, DS_PREFIX, "KeyInfo")?;
    fill_key_info(doc, node, key_info)?;
    Ok(node)
}

/// Fills `node` — already created, of whatever name and namespace — with `key_info`'s `Id`
/// attribute and children.
///
/// `OriginatorKeyInfo` and `RecipientKeyInfo` are `ds:KeyInfoType` themselves, just under a
/// different element name in the `xenc` namespace, so they share this with `write_key_info`
/// rather than duplicating the child-writing match.
fn fill_key_info(doc: &mut Document, node: NodeId, key_info: &KeyInfo) -> Result<()> {
    if let Some(id) = &key_info.id {
        doc.set_attribute(node, "Id", id.clone())?;
    }
    for item in &key_info.items {
        match item {
            KeyInfoItem::KeyName(name) => {
                let child = element(doc, XMLDSIG_NS, DS_PREFIX, "KeyName")?;
                text(doc, child, name)?;
                doc.append_child(node, child)?;
            }
            KeyInfoItem::RetrievalMethod { uri, type_uri } => {
                let child = element(doc, XMLDSIG_NS, DS_PREFIX, "RetrievalMethod")?;
                doc.set_attribute(child, "URI", uri.clone())?;
                if let Some(type_uri) = type_uri {
                    doc.set_attribute(child, "Type", type_uri.clone())?;
                }
                doc.append_child(node, child)?;
            }
            KeyInfoItem::EncryptedKey(key) => {
                let child = write_encrypted_key(doc, key)?;
                doc.append_child(node, child)?;
            }
            KeyInfoItem::AgreementMethod(method) => {
                let child = write_agreement_method(doc, method)?;
                doc.append_child(node, child)?;
            }
            KeyInfoItem::EcKeyValue(value) => {
                let outer = element(doc, XMLDSIG_NS, DS_PREFIX, "KeyValue")?;
                let inner = write_ec_key_value(doc, value)?;
                doc.append_child(outer, inner)?;
                doc.append_child(node, outer)?;
            }
            KeyInfoItem::DhKeyValue(value) => {
                let outer = element(doc, XMLDSIG_NS, DS_PREFIX, "KeyValue")?;
                let inner = write_dh_key_value(doc, value)?;
                doc.append_child(outer, inner)?;
                doc.append_child(node, outer)?;
            }
            KeyInfoItem::Other(fragment) => {
                fragments(doc, node, std::slice::from_ref(fragment))?;
            }
        }
    }
    Ok(())
}

/// Builds a `dsig11:ECKeyValue` element: the canonical form of [`EcKeyValue`], with
/// `NamedCurve/@URI`.
///
/// This is the one combination of namespace and `NamedCurve` shape this crate has verified
/// against the XML Signature 1.1 schema itself (`dsig11:ECKeyValueType` is a `choice` of
/// `ECParameters`/`NamedCurve` followed by `PublicKey`, and `dsig11:NamedCurveType` has a
/// required `URI` attribute and no text content). The `xenc11:ECKeyValue` spelling some
/// implementations use is accepted on parse for interoperability — see [`EcKeyValue`] — but is
/// not what this crate writes, since mixing that namespace with an attribute-carrying
/// `NamedCurve` would combine two shapes never confirmed to appear together.
fn write_ec_key_value(doc: &mut Document, value: &EcKeyValue) -> Result<NodeId> {
    let node = element(doc, XMLDSIG11_NS, DSIG11_PREFIX, "ECKeyValue")?;
    if let Some(id) = &value.id {
        doc.set_attribute(node, "Id", id.clone())?;
    }
    let named_curve = element(doc, XMLDSIG11_NS, DSIG11_PREFIX, "NamedCurve")?;
    doc.set_attribute(named_curve, "URI", value.named_curve.clone())?;
    doc.append_child(node, named_curve)?;
    let public_key = element(doc, XMLDSIG11_NS, DSIG11_PREFIX, "PublicKey")?;
    base64_text(doc, public_key, &value.public_key)?;
    doc.append_child(node, public_key)?;
    Ok(node)
}

/// Builds a `xenc:DHKeyValue` element, in the schema's `P`, `Q?`, `Generator`, `Public`, `seed?`
/// order.
fn write_dh_key_value(doc: &mut Document, value: &DhKeyValue) -> Result<NodeId> {
    let node = xenc(doc, "DHKeyValue")?;
    let p = xenc(doc, "P")?;
    base64_text(doc, p, &value.p)?;
    doc.append_child(node, p)?;
    if let Some(q) = &value.q {
        let child = xenc(doc, "Q")?;
        base64_text(doc, child, q)?;
        doc.append_child(node, child)?;
    }
    let generator = xenc(doc, "Generator")?;
    base64_text(doc, generator, &value.generator)?;
    doc.append_child(node, generator)?;
    let public = xenc(doc, "Public")?;
    base64_text(doc, public, &value.public)?;
    doc.append_child(node, public)?;
    if let Some(seed) = &value.seed {
        let child = xenc(doc, "seed")?;
        base64_text(doc, child, seed)?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

/// Builds a `xenc11:KeyDerivationMethod` element, with a `ConcatKDFParams` child when the method
/// carries the fields for one.
fn write_key_derivation_method(doc: &mut Document, method: &KeyDerivationMethod) -> Result<NodeId> {
    let node = element(doc, XMLENC11_NS, XENC11_PREFIX, "KeyDerivationMethod")?;
    doc.set_attribute(node, "Algorithm", method.algorithm.clone())?;
    if method.digest_method.is_some() {
        let params = write_concat_kdf_params(doc, method)?;
        doc.append_child(node, params)?;
    }
    fragments(doc, node, &method.other)?;
    Ok(node)
}

/// Builds a `xenc11:ConcatKDFParams` element from the flattened fields of `method`. An empty
/// octet field is omitted, which is the schema's own default for it.
fn write_concat_kdf_params(doc: &mut Document, method: &KeyDerivationMethod) -> Result<NodeId> {
    let node = element(doc, XMLENC11_NS, XENC11_PREFIX, "ConcatKDFParams")?;
    for (name, octets) in [
        ("AlgorithmID", &method.algorithm_id),
        ("PartyUInfo", &method.party_u_info),
        ("PartyVInfo", &method.party_v_info),
        ("SuppPubInfo", &method.supp_pub_info),
        ("SuppPrivInfo", &method.supp_priv_info),
    ] {
        if !octets.is_empty() {
            doc.set_attribute(node, name, crate::crypto::kdf::encode_kdf_bitstring(octets))?;
        }
    }
    if let Some(uri) = &method.digest_method {
        let child = element(doc, XMLDSIG_NS, DS_PREFIX, "DigestMethod")?;
        doc.set_attribute(child, "Algorithm", uri.clone())?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

/// Builds a `xenc:AgreementMethod` element in schema order: `KA-Nonce?`, then the recognized
/// `##other` children (`ds:DigestMethod`, `xenc11:KeyDerivationMethod`) and any unrecognized
/// ones, then `OriginatorKeyInfo?`, then `RecipientKeyInfo?`.
fn write_agreement_method(doc: &mut Document, method: &AgreementMethod) -> Result<NodeId> {
    let node = xenc(doc, "AgreementMethod")?;
    doc.set_attribute(node, "Algorithm", method.algorithm.clone())?;
    if let Some(nonce) = &method.ka_nonce {
        let child = xenc(doc, "KA-Nonce")?;
        base64_text(doc, child, nonce)?;
        doc.append_child(node, child)?;
    }
    if let Some(uri) = &method.digest_method {
        let child = element(doc, XMLDSIG_NS, DS_PREFIX, "DigestMethod")?;
        doc.set_attribute(child, "Algorithm", uri.clone())?;
        doc.append_child(node, child)?;
    }
    if let Some(derivation) = &method.key_derivation_method {
        let child = write_key_derivation_method(doc, derivation)?;
        doc.append_child(node, child)?;
    }
    fragments(doc, node, &method.other)?;
    if let Some(info) = &method.originator_key_info {
        let child = xenc(doc, "OriginatorKeyInfo")?;
        fill_key_info(doc, child, info)?;
        doc.append_child(node, child)?;
    }
    if let Some(info) = &method.recipient_key_info {
        let child = xenc(doc, "RecipientKeyInfo")?;
        fill_key_info(doc, child, info)?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

/// Builds a `xenc:EncryptionProperties` element.
fn write_properties(doc: &mut Document, properties: &EncryptionProperties) -> Result<NodeId> {
    let node = xenc(doc, "EncryptionProperties")?;
    if let Some(id) = &properties.id {
        doc.set_attribute(node, "Id", id.clone())?;
    }
    for property in &properties.properties {
        let child = xenc(doc, "EncryptionProperty")?;
        if let Some(id) = &property.id {
            doc.set_attribute(child, "Id", id.clone())?;
        }
        if let Some(target) = &property.target {
            doc.set_attribute(child, "Target", target.clone())?;
        }
        fragments(doc, child, &property.content)?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

/// Builds a `xenc:ReferenceList` element.
fn write_reference_list(doc: &mut Document, list: &ReferenceList) -> Result<NodeId> {
    let node = xenc(doc, "ReferenceList")?;
    for reference in &list.references {
        let child = xenc(doc, reference.kind.local_name())?;
        doc.set_attribute(child, "URI", reference.uri.clone())?;
        fragments(doc, child, &reference.extensions)?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

/// Builds a detached `xenc:EncryptedData` element.
fn write_encrypted_data(doc: &mut Document, data: &EncryptedData) -> Result<NodeId> {
    let node = xenc(doc, EncryptedData::LOCAL_NAME)?;
    write_common(doc, node, &data.common)?;
    Ok(node)
}

/// Builds a detached `xenc:EncryptedKey` element.
fn write_encrypted_key(doc: &mut Document, key: &EncryptedKey) -> Result<NodeId> {
    let node = xenc(doc, EncryptedKey::LOCAL_NAME)?;
    if let Some(recipient) = &key.recipient {
        doc.set_attribute(node, "Recipient", recipient.clone())?;
    }
    write_common(doc, node, &key.common)?;
    if let Some(list) = &key.reference_list {
        let child = write_reference_list(doc, list)?;
        doc.append_child(node, child)?;
    }
    if let Some(name) = &key.carried_key_name {
        let child = xenc(doc, "CarriedKeyName")?;
        text(doc, child, name)?;
        doc.append_child(node, child)?;
    }
    Ok(node)
}

impl EncryptedData {
    /// Builds the `xenc:EncryptedData` subtree in `doc`, appending it to `parent` when one is
    /// given and leaving it detached otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Dom`](crate::EncError::Dom) when the tree cannot be built — for
    /// instance because `parent` does not belong to `doc` — and
    /// [`EncError::Build`](crate::EncError::Build) when a stored [`XmlFragment::Element`] is not
    /// well-formed XML.
    pub fn write_into(&self, doc: &mut Document, parent: Option<NodeId>) -> Result<NodeId> {
        let node = write_encrypted_data(doc, self)?;
        crate::fragment::drop_redundant_namespace_declarations(doc, node)?;
        if let Some(parent) = parent {
            doc.append_child(parent, node)?;
        }
        Ok(node)
    }

    /// Serializes the structure to standalone XML text.
    ///
    /// # Errors
    ///
    /// See [`Self::write_into`].
    pub fn to_xml(&self) -> Result<String> {
        let mut doc = Document::new();
        let node = self.write_into(&mut doc, None)?;
        Ok(doc.node_to_xml(node, &oxixml_dom::SerializeOptions::new()))
    }
}

impl EncryptedKey {
    /// Builds the `xenc:EncryptedKey` subtree in `doc`, appending it to `parent` when one is
    /// given and leaving it detached otherwise.
    ///
    /// # Errors
    ///
    /// See [`EncryptedData::write_into`].
    pub fn write_into(&self, doc: &mut Document, parent: Option<NodeId>) -> Result<NodeId> {
        let node = write_encrypted_key(doc, self)?;
        crate::fragment::drop_redundant_namespace_declarations(doc, node)?;
        if let Some(parent) = parent {
            doc.append_child(parent, node)?;
        }
        Ok(node)
    }

    /// Serializes the structure to standalone XML text.
    ///
    /// # Errors
    ///
    /// See [`EncryptedData::write_into`].
    pub fn to_xml(&self) -> Result<String> {
        let mut doc = Document::new();
        let node = self.write_into(&mut doc, None)?;
        Ok(doc.node_to_xml(node, &oxixml_dom::SerializeOptions::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::{BlockEncryption, KeyWrap};
    use crate::model::{ContentType, EncryptionMethod, EncryptionProperty, Reference};

    #[test]
    fn a_minimal_structure_has_exactly_one_child() -> Result<()> {
        // The prefix is declared once, on the apex: the sweep in `write_into` removes the
        // redundant declarations the builder puts on every element.
        let xml = EncryptedData::new(vec![0xde, 0xad]).to_xml()?;
        assert_eq!(
            xml,
            concat!(
                r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#">"#,
                "<xenc:CipherData><xenc:CipherValue>3q0=</xenc:CipherValue>",
                "</xenc:CipherData></xenc:EncryptedData>",
            )
        );
        Ok(())
    }

    #[test]
    fn each_namespace_is_declared_exactly_once() -> Result<()> {
        use crate::algorithm::KeyTransport;
        use crate::crypto::digest::Digest;
        use crate::crypto::mgf1;

        let mut key = EncryptedKey::new(vec![1, 2, 3]);
        key.common.encryption_method = Some(
            EncryptionMethod::key_transport(KeyTransport::RsaOaep)
                .with_digest_method(Digest::Sha256.uri())
                .with_mgf(mgf1::MGF1_SHA256_URI),
        );
        key.common.key_info = Some(KeyInfo::with_key_name("recipient"));

        let mut data = EncryptedData::new(vec![4, 5, 6]);
        data.common.key_info = Some(KeyInfo::with_encrypted_key(key));
        let xml = data.to_xml()?;

        assert_eq!(xml.matches("xmlns:xenc=").count(), 1, "{xml}");
        assert_eq!(xml.matches("xmlns:ds=").count(), 1, "{xml}");
        assert_eq!(xml.matches("xmlns:xenc11=").count(), 1, "{xml}");
        // Still parses back to the same model.
        assert_eq!(EncryptedData::from_xml(&xml)?, data);
        Ok(())
    }

    #[test]
    fn children_are_emitted_in_schema_order() -> Result<()> {
        let mut data = EncryptedData::new(vec![1]);
        data.common.id = Some("ed1".into());
        data.common.content_type = Some(ContentType::Element);
        data.common.mime_type = Some("text/xml".into());
        data.common.encoding = Some("urn:example:enc".into());
        data.common.encryption_method = Some(EncryptionMethod::block(BlockEncryption::Aes256Gcm));
        data.common.key_info = Some(KeyInfo::with_key_name("k"));
        data.common.encryption_properties =
            Some(EncryptionProperties::new().with(EncryptionProperty::text("when")));

        let xml = data.to_xml()?;
        let method = xml.find("EncryptionMethod").expect("method");
        let key_info = xml.find("KeyInfo").expect("key info");
        let cipher = xml.find("CipherData").expect("cipher data");
        let properties = xml.find("EncryptionProperties").expect("properties");
        assert!(method < key_info, "{xml}");
        assert!(key_info < cipher, "{xml}");
        assert!(cipher < properties, "{xml}");
        assert!(xml.contains(r#"Id="ed1""#), "{xml}");
        assert!(xml.contains(r#"MimeType="text/xml""#), "{xml}");
        assert!(xml.contains(r#"Encoding="urn:example:enc""#), "{xml}");
        Ok(())
    }

    #[test]
    fn an_empty_property_container_is_dropped() -> Result<()> {
        let mut data = EncryptedData::new(vec![1]);
        data.common.encryption_properties = Some(EncryptionProperties::new());
        let xml = data.to_xml()?;
        assert!(!xml.contains("EncryptionProperties"), "{xml}");
        Ok(())
    }

    #[test]
    fn encrypted_key_extras_follow_the_common_children() -> Result<()> {
        let mut key = EncryptedKey::new(vec![2])
            .with_recipient("bob")
            .with_carried_key_name("session");
        key.common.encryption_method = Some(EncryptionMethod::key_wrap(KeyWrap::Aes128));
        key.reference_list = Some(ReferenceList {
            references: vec![Reference::data("#ed1"), Reference::key("#ek2")],
        });
        let xml = key.to_xml()?;
        let cipher = xml.find("CipherData").expect("cipher data");
        let list = xml.find("ReferenceList").expect("reference list");
        let carried = xml.find("CarriedKeyName").expect("carried key name");
        assert!(cipher < list, "{xml}");
        assert!(list < carried, "{xml}");
        assert!(xml.contains(r#"Recipient="bob""#), "{xml}");
        assert!(xml.contains(r#"<xenc:DataReference"#), "{xml}");
        assert!(xml.contains(r#"<xenc:KeyReference"#), "{xml}");
        Ok(())
    }

    #[test]
    fn verbatim_fragments_are_reproduced() -> Result<()> {
        let mut data = EncryptedData::new(vec![3]);
        let mut key_info = KeyInfo::new();
        key_info.items.push(KeyInfoItem::Other(XmlFragment::Element(
            r#"<ds:X509Data xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:X509SubjectName>CN=x</ds:X509SubjectName></ds:X509Data>"#.into(),
        )));
        data.common.key_info = Some(key_info);
        let xml = data.to_xml()?;
        assert!(xml.contains("X509SubjectName"), "{xml}");
        assert!(xml.contains("CN=x"), "{xml}");
        Ok(())
    }

    #[test]
    fn a_cipher_reference_is_written_with_its_transforms() -> Result<()> {
        let mut data = EncryptedData::new(Vec::new());
        let mut reference = CipherReference::new("http://example.org/blob");
        reference.transforms.push(Transform::new(
            "http://www.w3.org/TR/1999/REC-xpath-19991116",
        ));
        data.common.cipher_data = CipherData::Reference(reference);
        let xml = data.to_xml()?;
        assert!(xml.contains(r#"URI="http://example.org/blob""#), "{xml}");
        assert!(xml.contains("xenc:Transforms"), "{xml}");
        assert!(xml.contains("ds:Transform"), "{xml}");
        assert!(!xml.contains("CipherValue"), "{xml}");
        Ok(())
    }
}
