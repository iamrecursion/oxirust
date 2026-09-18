//! Reading an `EncryptedData` or `EncryptedKey` back out of a DOM tree.
//!
//! # How strict is this?
//!
//! Strict about what a decryptor must not guess at, permissive about everything else:
//!
//! * The element name and namespace are checked. An `EncryptedData` in the wrong namespace is
//!   not an `EncryptedData`.
//! * `CipherData` is required, exactly as the schema says, and so is the `Algorithm` attribute
//!   of an `EncryptionMethod` and the `URI` of a reference.
//! * Child *order* is not enforced. The schema is a `sequence`, so a conforming document has
//!   them in order anyway, and rejecting a mis-ordered one buys nothing while breaking
//!   interoperability with producers that get it wrong.
//! * Unrecognized children are preserved verbatim rather than rejected, because the schema's
//!   own wildcards permit them.
//! * White space between elements is ignored.
//!
//! Base64 is decoded with the *forgiving* decoder of [`oxixml_text::base64`], which skips XML
//! white space. This is not laxity: XMLEnc §4.5.4 explicitly contemplates line-wrapped
//! `CipherValue` content, and `xsd:base64Binary`'s lexical space allows white space anywhere.

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::algorithm::{XMLDSIG_NS, XMLDSIG11_NS, XMLENC_NS, XMLENC11_NS};
use crate::error::{EncError, Result};
use crate::fragment;
use crate::model::agreement::{AgreementMethod, DhKeyValue, EcKeyValue, KeyDerivationMethod};
use crate::model::cipher::{CipherData, CipherReference, Transform, XmlFragment};
use crate::model::encrypted::{ContentType, EncryptedData, EncryptedKey, EncryptedType};
use crate::model::keyinfo::{KeyInfo, KeyInfoItem, Reference, ReferenceKind, ReferenceList};
use crate::model::method::EncryptionMethod;
use crate::model::properties::{EncryptionProperties, EncryptionProperty};

/// Returns the element children of `node`, ignoring text, comments and processing instructions.
fn element_children(doc: &Document, node: NodeId) -> Vec<NodeId> {
    doc.children(node)
        .filter(|child| doc.kind(*child) == Some(NodeKind::Element))
        .collect()
}

/// Returns `true` when `node` is an element with the given namespace and local name.
fn is_element(doc: &Document, node: NodeId, namespace: &str, local: &str) -> bool {
    doc.element(node)
        .is_some_and(|data| data.namespace_uri() == Some(namespace) && data.local_name() == local)
}

/// Returns the concatenated character data of `node`.
fn text_of(doc: &Document, node: NodeId) -> String {
    doc.text_content(node).unwrap_or_default()
}

/// Decodes the base64 content of `node`, tolerating embedded white space.
fn base64_of(doc: &Document, node: NodeId) -> Result<Vec<u8>> {
    Ok(oxixml_text::base64::base64_decode_forgiving(&text_of(
        doc, node,
    ))?)
}

/// Returns an attribute value, or `None` when the attribute is absent.
///
/// XML Encryption's own attributes are unqualified, so a plain name lookup is right.
fn attribute(doc: &Document, node: NodeId, name: &str) -> Option<String> {
    doc.get_attribute(node, name).map(str::to_owned)
}

/// Returns a required attribute, or an error naming what was missing.
fn required_attribute(doc: &Document, node: NodeId, name: &str, element: &str) -> Result<String> {
    attribute(doc, node, name)
        .ok_or_else(|| EncError::InvalidStructure(format!("`{element}` has no `{name}` attribute")))
}

/// Captures the children of `node` verbatim, as text and self-contained elements.
fn verbatim_children(doc: &Document, node: NodeId) -> Result<Vec<XmlFragment>> {
    let mut captured = Vec::new();
    for child in doc.children(node).collect::<Vec<_>>() {
        match doc.kind(child) {
            Some(NodeKind::Element) => captured.push(XmlFragment::Element(
                fragment::serialize_self_contained(doc, child)?,
            )),
            Some(NodeKind::Text | NodeKind::CdataSection) => {
                if let Some(value) = doc.character_data(child) {
                    captured.push(XmlFragment::Text(value.to_owned()));
                }
            }
            _ => {}
        }
    }
    Ok(captured)
}

/// Captures one element verbatim.
fn verbatim_element(doc: &Document, node: NodeId) -> Result<XmlFragment> {
    Ok(XmlFragment::Element(fragment::serialize_self_contained(
        doc, node,
    )?))
}

/// Returns `true` when `node` is an `ECKeyValue` element in either of the two namespaces this
/// crate accepts; see the [module documentation](crate::model::agreement).
fn is_ec_key_value_element(doc: &Document, node: NodeId) -> bool {
    is_element(doc, node, XMLENC11_NS, "ECKeyValue")
        || is_element(doc, node, XMLDSIG11_NS, "ECKeyValue")
}

/// Returns `true` when `node` is a `xenc:DHKeyValue` element.
fn is_dh_key_value_element(doc: &Document, node: NodeId) -> bool {
    is_element(doc, node, XMLENC_NS, "DHKeyValue")
}

/// Parses an `ECKeyValue` element, accepting the curve URI either as `NamedCurve/@URI` or as
/// `NamedCurve`'s text content, and `NamedCurve`/`PublicKey` in either of the two namespaces an
/// `ECKeyValue` itself may use.
fn parse_ec_key_value(doc: &Document, node: NodeId) -> Result<EcKeyValue> {
    let mut named_curve = None;
    let mut public_key = None;
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC11_NS, "NamedCurve")
            || is_element(doc, child, XMLDSIG11_NS, "NamedCurve")
        {
            named_curve = Some(match attribute(doc, child, "URI") {
                Some(uri) => uri,
                None => text_of(doc, child).trim().to_owned(),
            });
        } else if is_element(doc, child, XMLENC11_NS, "PublicKey")
            || is_element(doc, child, XMLDSIG11_NS, "PublicKey")
        {
            public_key = Some(base64_of(doc, child)?);
        }
    }
    Ok(EcKeyValue {
        id: attribute(doc, node, "Id"),
        named_curve: named_curve.ok_or_else(|| {
            EncError::InvalidStructure(
                "an `ECKeyValue` has no `NamedCurve` (explicit `ECParameters` are not supported)"
                    .to_owned(),
            )
        })?,
        public_key: public_key.ok_or_else(|| {
            EncError::InvalidStructure("an `ECKeyValue` has no `PublicKey`".to_owned())
        })?,
    })
}

/// Parses a `xenc:DHKeyValue` element.
fn parse_dh_key_value(doc: &Document, node: NodeId) -> Result<DhKeyValue> {
    let mut p = None;
    let mut q = None;
    let mut generator = None;
    let mut public = None;
    let mut seed = None;
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC_NS, "P") {
            p = Some(base64_of(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "Q") {
            q = Some(base64_of(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "Generator") {
            generator = Some(base64_of(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "Public") {
            public = Some(base64_of(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "seed") {
            seed = Some(base64_of(doc, child)?);
        }
    }
    Ok(DhKeyValue {
        p: p.ok_or_else(|| {
            EncError::InvalidStructure("a `xenc:DHKeyValue` has no `P`".to_owned())
        })?,
        q,
        generator: generator.ok_or_else(|| {
            EncError::InvalidStructure("a `xenc:DHKeyValue` has no `Generator`".to_owned())
        })?,
        public: public.ok_or_else(|| {
            EncError::InvalidStructure("a `xenc:DHKeyValue` has no `Public`".to_owned())
        })?,
        seed,
    })
}

/// Decodes a `xenc11:ConcatKDFParams` attribute's bit-string `hexBinary` encoding, treating an
/// absent attribute as the schema's default: the empty bit string.
fn concat_kdf_attribute(doc: &Document, node: NodeId, name: &str) -> Result<Vec<u8>> {
    let Some(hex_text) = attribute(doc, node, name) else {
        return Ok(Vec::new());
    };
    let raw = oxixml_text::hex::decode(&hex_text).map_err(|error| {
        EncError::InvalidStructure(format!(
            "`{name}` on `xenc11:ConcatKDFParams` is not valid hexBinary: {error}"
        ))
    })?;
    crate::crypto::kdf::decode_kdf_bitstring(&raw)
}

/// Parses a `xenc11:ConcatKDFParams` element's `ds:DigestMethod` and five attributes into
/// `method`, which the caller has already confirmed has no `ConcatKDFParams` yet.
fn parse_concat_kdf_params_into(
    doc: &Document,
    node: NodeId,
    method: &mut KeyDerivationMethod,
) -> Result<()> {
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLDSIG_NS, "DigestMethod") && method.digest_method.is_none() {
            method.digest_method = Some(required_attribute(
                doc,
                child,
                "Algorithm",
                "ds:DigestMethod",
            )?);
        }
    }
    if method.digest_method.is_none() {
        return Err(EncError::InvalidStructure(
            "a `xenc11:ConcatKDFParams` has no `ds:DigestMethod` child".to_owned(),
        ));
    }
    method.algorithm_id = concat_kdf_attribute(doc, node, "AlgorithmID")?;
    method.party_u_info = concat_kdf_attribute(doc, node, "PartyUInfo")?;
    method.party_v_info = concat_kdf_attribute(doc, node, "PartyVInfo")?;
    method.supp_pub_info = concat_kdf_attribute(doc, node, "SuppPubInfo")?;
    method.supp_priv_info = concat_kdf_attribute(doc, node, "SuppPrivInfo")?;
    Ok(())
}

/// Parses a `xenc11:KeyDerivationMethod` element.
fn parse_key_derivation_method(doc: &Document, node: NodeId) -> Result<KeyDerivationMethod> {
    let mut method = KeyDerivationMethod {
        algorithm: required_attribute(doc, node, "Algorithm", "xenc11:KeyDerivationMethod")?,
        digest_method: None,
        algorithm_id: Vec::new(),
        party_u_info: Vec::new(),
        party_v_info: Vec::new(),
        supp_pub_info: Vec::new(),
        supp_priv_info: Vec::new(),
        other: Vec::new(),
    };
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC11_NS, "ConcatKDFParams") && method.digest_method.is_none()
        {
            parse_concat_kdf_params_into(doc, child, &mut method)?;
        } else {
            method.other.push(verbatim_element(doc, child)?);
        }
    }
    Ok(method)
}

/// Parses a `xenc:AgreementMethod` element.
///
/// The content model is `mixed="true"`, so character data alongside the recognized children is
/// captured into [`AgreementMethod::other`] rather than dropped.
fn parse_agreement_method(doc: &Document, node: NodeId) -> Result<AgreementMethod> {
    let mut method = AgreementMethod::new(required_attribute(
        doc,
        node,
        "Algorithm",
        "xenc:AgreementMethod",
    )?);
    for child in doc.children(node).collect::<Vec<_>>() {
        match doc.kind(child) {
            Some(NodeKind::Element) => {
                if is_element(doc, child, XMLENC_NS, "KA-Nonce") && method.ka_nonce.is_none() {
                    method.ka_nonce = Some(base64_of(doc, child)?);
                } else if is_element(doc, child, XMLENC11_NS, "KeyDerivationMethod")
                    && method.key_derivation_method.is_none()
                {
                    method.key_derivation_method = Some(parse_key_derivation_method(doc, child)?);
                } else if is_element(doc, child, XMLDSIG_NS, "DigestMethod")
                    && method.digest_method.is_none()
                {
                    method.digest_method = Some(required_attribute(
                        doc,
                        child,
                        "Algorithm",
                        "ds:DigestMethod",
                    )?);
                } else if is_element(doc, child, XMLENC_NS, "OriginatorKeyInfo")
                    && method.originator_key_info.is_none()
                {
                    method.originator_key_info = Some(parse_key_info(doc, child)?);
                } else if is_element(doc, child, XMLENC_NS, "RecipientKeyInfo")
                    && method.recipient_key_info.is_none()
                {
                    method.recipient_key_info = Some(parse_key_info(doc, child)?);
                } else {
                    method.other.push(verbatim_element(doc, child)?);
                }
            }
            Some(NodeKind::Text | NodeKind::CdataSection) => {
                if let Some(value) = doc.character_data(child) {
                    method.other.push(XmlFragment::Text(value.to_owned()));
                }
            }
            _ => {}
        }
    }
    Ok(method)
}

/// Parses an `xenc:EncryptionMethod` element.
fn parse_encryption_method(doc: &Document, node: NodeId) -> Result<EncryptionMethod> {
    let mut method = EncryptionMethod::new(required_attribute(
        doc,
        node,
        "Algorithm",
        "EncryptionMethod",
    )?);
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC_NS, "KeySize") {
            let raw = text_of(doc, child);
            let trimmed = raw.trim();
            method.key_size = Some(
                trimmed
                    .parse::<u32>()
                    .map_err(|_| EncryptionMethodError::key_size(trimmed))?,
            );
        } else if is_element(doc, child, XMLENC_NS, "OAEPparams") {
            method.oaep_params = Some(base64_of(doc, child)?);
        } else if is_element(doc, child, XMLDSIG_NS, "DigestMethod") {
            method.digest_method = Some(required_attribute(
                doc,
                child,
                "Algorithm",
                "ds:DigestMethod",
            )?);
        } else if is_element(doc, child, XMLENC11_NS, "MGF") {
            method.mgf = Some(required_attribute(doc, child, "Algorithm", "xenc11:MGF")?);
        } else {
            method.other.push(verbatim_element(doc, child)?);
        }
    }
    Ok(method)
}

/// A tiny helper so the `KeySize` error message stays in one place.
struct EncryptionMethodError;

impl EncryptionMethodError {
    fn key_size(raw: &str) -> EncError {
        EncError::InvalidStructure(format!("`xenc:KeySize` is not an integer: `{raw}`"))
    }
}

/// Parses a `xenc:CipherData` element.
fn parse_cipher_data(doc: &Document, node: NodeId) -> Result<CipherData> {
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC_NS, "CipherValue") {
            return Ok(CipherData::Value(base64_of(doc, child)?));
        }
        if is_element(doc, child, XMLENC_NS, "CipherReference") {
            return Ok(CipherData::Reference(parse_cipher_reference(doc, child)?));
        }
    }
    Err(EncError::InvalidStructure(
        "`xenc:CipherData` has neither a `CipherValue` nor a `CipherReference`".to_owned(),
    ))
}

/// Parses a `xenc:CipherReference` element.
fn parse_cipher_reference(doc: &Document, node: NodeId) -> Result<CipherReference> {
    let mut reference = CipherReference::new(required_attribute(
        doc,
        node,
        "URI",
        "xenc:CipherReference",
    )?);
    for child in element_children(doc, node) {
        if !is_element(doc, child, XMLENC_NS, "Transforms") {
            continue;
        }
        for transform in element_children(doc, child) {
            if !is_element(doc, transform, XMLDSIG_NS, "Transform") {
                continue;
            }
            reference.transforms.push(Transform {
                algorithm: required_attribute(doc, transform, "Algorithm", "ds:Transform")?,
                content: verbatim_children(doc, transform)?,
            });
        }
    }
    Ok(reference)
}

/// Parses a `ds:KeyValue` element, unwrapping a recognized `ECKeyValue`/`DHKeyValue` child into
/// the same structured item a bare, unwrapped one parses to.
///
/// A `ds:KeyValue` with any other single child, with no element child, or with more than one —
/// none of which the schema's own `choice` content model permits, but a lenient parse tolerates
/// rather than rejects — is kept as one opaque [`KeyInfoItem::Other`] fragment, exactly as it
/// round-tripped before this crate understood key agreement at all.
fn parse_key_value(doc: &Document, node: NodeId) -> Result<KeyInfoItem> {
    let children = element_children(doc, node);
    if let [only] = children.as_slice() {
        if is_ec_key_value_element(doc, *only) {
            return Ok(KeyInfoItem::EcKeyValue(parse_ec_key_value(doc, *only)?));
        }
        if is_dh_key_value_element(doc, *only) {
            return Ok(KeyInfoItem::DhKeyValue(parse_dh_key_value(doc, *only)?));
        }
    }
    Ok(KeyInfoItem::Other(verbatim_element(doc, node)?))
}

/// Parses a `ds:KeyInfo` element.
fn parse_key_info(doc: &Document, node: NodeId) -> Result<KeyInfo> {
    let mut info = KeyInfo {
        id: attribute(doc, node, "Id"),
        items: Vec::new(),
    };
    for child in element_children(doc, node) {
        if is_element(doc, child, XMLDSIG_NS, "KeyName") {
            info.items.push(KeyInfoItem::KeyName(text_of(doc, child)));
        } else if is_element(doc, child, XMLDSIG_NS, "RetrievalMethod") {
            info.items.push(KeyInfoItem::RetrievalMethod {
                uri: required_attribute(doc, child, "URI", "ds:RetrievalMethod")?,
                type_uri: attribute(doc, child, "Type"),
            });
        } else if is_element(doc, child, XMLENC_NS, EncryptedKey::LOCAL_NAME) {
            info.items
                .push(KeyInfoItem::EncryptedKey(Box::new(EncryptedKey::parse(
                    doc, child,
                )?)));
        } else if is_element(doc, child, XMLENC_NS, "AgreementMethod") {
            info.items.push(KeyInfoItem::AgreementMethod(Box::new(
                parse_agreement_method(doc, child)?,
            )));
        } else if is_ec_key_value_element(doc, child) {
            info.items
                .push(KeyInfoItem::EcKeyValue(parse_ec_key_value(doc, child)?));
        } else if is_dh_key_value_element(doc, child) {
            info.items
                .push(KeyInfoItem::DhKeyValue(parse_dh_key_value(doc, child)?));
        } else if is_element(doc, child, XMLDSIG_NS, "KeyValue") {
            info.items.push(parse_key_value(doc, child)?);
        } else {
            info.items
                .push(KeyInfoItem::Other(verbatim_element(doc, child)?));
        }
    }
    Ok(info)
}

/// Parses a `xenc:EncryptionProperties` element.
fn parse_properties(doc: &Document, node: NodeId) -> Result<EncryptionProperties> {
    let mut properties = EncryptionProperties {
        id: attribute(doc, node, "Id"),
        properties: Vec::new(),
    };
    for child in element_children(doc, node) {
        if !is_element(doc, child, XMLENC_NS, "EncryptionProperty") {
            continue;
        }
        properties.properties.push(EncryptionProperty {
            id: attribute(doc, child, "Id"),
            target: attribute(doc, child, "Target"),
            content: verbatim_children(doc, child)?,
        });
    }
    Ok(properties)
}

/// Parses a `xenc:ReferenceList` element.
fn parse_reference_list(doc: &Document, node: NodeId) -> Result<ReferenceList> {
    let mut list = ReferenceList::new();
    for child in element_children(doc, node) {
        let kind = if is_element(doc, child, XMLENC_NS, "DataReference") {
            ReferenceKind::Data
        } else if is_element(doc, child, XMLENC_NS, "KeyReference") {
            ReferenceKind::Key
        } else {
            continue;
        };
        list.references.push(Reference {
            kind,
            uri: required_attribute(doc, child, "URI", kind.local_name())?,
            extensions: verbatim_children(doc, child)?,
        });
    }
    Ok(list)
}

/// Parses the attributes and children shared by both structures, returning the children that
/// were *not* consumed so the caller can look for its own extras among them.
fn parse_common(doc: &Document, node: NodeId) -> Result<(EncryptedType, Vec<NodeId>)> {
    let mut encryption_method = None;
    let mut key_info = None;
    let mut cipher_data = None;
    let mut encryption_properties = None;
    let mut leftovers = Vec::new();

    for child in element_children(doc, node) {
        if is_element(doc, child, XMLENC_NS, "EncryptionMethod") && encryption_method.is_none() {
            encryption_method = Some(parse_encryption_method(doc, child)?);
        } else if is_element(doc, child, XMLDSIG_NS, "KeyInfo") && key_info.is_none() {
            key_info = Some(parse_key_info(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "CipherData") && cipher_data.is_none() {
            cipher_data = Some(parse_cipher_data(doc, child)?);
        } else if is_element(doc, child, XMLENC_NS, "EncryptionProperties")
            && encryption_properties.is_none()
        {
            encryption_properties = Some(parse_properties(doc, child)?);
        } else {
            leftovers.push(child);
        }
    }

    let cipher_data = cipher_data.ok_or_else(|| {
        EncError::InvalidStructure("the element has no `xenc:CipherData` child".to_owned())
    })?;

    Ok((
        EncryptedType {
            id: attribute(doc, node, "Id"),
            content_type: attribute(doc, node, "Type").map(|uri| ContentType::from_uri(&uri)),
            mime_type: attribute(doc, node, "MimeType"),
            encoding: attribute(doc, node, "Encoding"),
            encryption_method,
            key_info,
            cipher_data,
            encryption_properties,
        },
        leftovers,
    ))
}

impl EncryptedData {
    /// Parses the `xenc:EncryptedData` element `node` of `doc`.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidStructure`] when `node` is not an `xenc:EncryptedData`
    /// element, when the required `CipherData` child is missing, or when a required attribute
    /// is absent; and [`EncError::Base64`](crate::EncError::Base64) when a `CipherValue` is not
    /// valid base64.
    pub fn parse(doc: &Document, node: NodeId) -> Result<Self> {
        if !is_element(doc, node, XMLENC_NS, Self::LOCAL_NAME) {
            return Err(EncError::InvalidStructure(format!(
                "expected a `{{{XMLENC_NS}}}{}` element",
                Self::LOCAL_NAME
            )));
        }
        let (common, _leftovers) = parse_common(doc, node)?;
        Ok(Self { common })
    }

    /// Parses standalone `xenc:EncryptedData` XML text.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Build`](crate::EncError::Build) when the text is not well-formed,
    /// plus the errors of [`Self::parse`].
    pub fn from_xml(xml: &str) -> Result<Self> {
        let doc = oxixml_dom::build::parse_str(xml)?;
        let root = doc.document_element().ok_or_else(|| {
            EncError::InvalidStructure("the document has no document element".to_owned())
        })?;
        Self::parse(&doc, root)
    }
}

impl EncryptedKey {
    /// Parses the `xenc:EncryptedKey` element `node` of `doc`.
    ///
    /// # Errors
    ///
    /// See [`EncryptedData::parse`].
    pub fn parse(doc: &Document, node: NodeId) -> Result<Self> {
        if !is_element(doc, node, XMLENC_NS, Self::LOCAL_NAME) {
            return Err(EncError::InvalidStructure(format!(
                "expected a `{{{XMLENC_NS}}}{}` element",
                Self::LOCAL_NAME
            )));
        }
        let (common, leftovers) = parse_common(doc, node)?;

        let mut reference_list = None;
        let mut carried_key_name = None;
        for child in leftovers {
            if is_element(doc, child, XMLENC_NS, "ReferenceList") && reference_list.is_none() {
                reference_list = Some(parse_reference_list(doc, child)?);
            } else if is_element(doc, child, XMLENC_NS, "CarriedKeyName")
                && carried_key_name.is_none()
            {
                carried_key_name = Some(text_of(doc, child));
            }
        }

        Ok(Self {
            common,
            recipient: attribute(doc, node, "Recipient"),
            reference_list,
            carried_key_name,
        })
    }

    /// Parses standalone `xenc:EncryptedKey` XML text.
    ///
    /// # Errors
    ///
    /// See [`EncryptedData::from_xml`].
    pub fn from_xml(xml: &str) -> Result<Self> {
        let doc = oxixml_dom::build::parse_str(xml)?;
        let root = doc.document_element().ok_or_else(|| {
            EncError::InvalidStructure("the document has no document element".to_owned())
        })?;
        Self::parse(&doc, root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::BlockEncryption;

    #[test]
    fn a_minimal_encrypted_data_parses() -> Result<()> {
        let data = EncryptedData::from_xml(
            r#"<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#">
                 <CipherData><CipherValue>3q0=</CipherValue></CipherData>
               </EncryptedData>"#,
        )?;
        assert_eq!(data.ciphertext()?, &[0xde, 0xad]);
        assert!(data.common.encryption_method.is_none());
        Ok(())
    }

    #[test]
    fn line_wrapped_base64_is_accepted() -> Result<()> {
        let data = EncryptedData::from_xml(
            "<EncryptedData xmlns=\"http://www.w3.org/2001/04/xmlenc#\"><CipherData>\
             <CipherValue>3q2\n  +7w  ==</CipherValue></CipherData></EncryptedData>",
        )?;
        assert_eq!(data.ciphertext()?, &[0xde, 0xad, 0xbe, 0xef]);
        Ok(())
    }

    #[test]
    fn the_wrong_namespace_is_not_an_encrypted_data() {
        assert!(matches!(
            EncryptedData::from_xml(
                r#"<EncryptedData xmlns="urn:example:not-xmlenc"><CipherData/></EncryptedData>"#
            ),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn a_missing_cipher_data_is_rejected() {
        assert!(matches!(
            EncryptedData::from_xml(
                r#"<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"/>"#
            ),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn an_empty_cipher_data_is_rejected() {
        assert!(matches!(
            EncryptedData::from_xml(
                r#"<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><CipherData/></EncryptedData>"#
            ),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn an_encryption_method_without_an_algorithm_is_rejected() {
        assert!(matches!(
            EncryptedData::from_xml(
                r#"<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><EncryptionMethod/><CipherData><CipherValue></CipherValue></CipherData></EncryptedData>"#
            ),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn a_non_integer_key_size_is_rejected() {
        assert!(matches!(
            EncryptedData::from_xml(
                r#"<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><EncryptionMethod Algorithm="urn:x"><KeySize>lots</KeySize></EncryptionMethod><CipherData><CipherValue></CipherValue></CipherData></EncryptedData>"#
            ),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn a_full_structure_parses_every_part() -> Result<()> {
        let data = EncryptedData::from_xml(
            r##"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                   xmlns:ds="http://www.w3.org/2000/09/xmldsig#"
                                   Id="ed1"
                                   Type="http://www.w3.org/2001/04/xmlenc#Content"
                                   MimeType="text/xml">
                 <xenc:EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes128-cbc">
                   <xenc:KeySize>128</xenc:KeySize>
                 </xenc:EncryptionMethod>
                 <ds:KeyInfo Id="ki"><ds:KeyName>session</ds:KeyName></ds:KeyInfo>
                 <xenc:CipherData><xenc:CipherValue>AAEC</xenc:CipherValue></xenc:CipherData>
                 <xenc:EncryptionProperties Id="props">
                   <xenc:EncryptionProperty Target="#ed1" Id="p1">2026-07-27</xenc:EncryptionProperty>
                 </xenc:EncryptionProperties>
               </xenc:EncryptedData>"##,
        )?;
        assert_eq!(data.common.id.as_deref(), Some("ed1"));
        assert_eq!(data.common.content_type, Some(ContentType::Content));
        assert_eq!(data.common.mime_type.as_deref(), Some("text/xml"));
        let method = data.common.encryption_method.as_ref().expect("a method");
        assert_eq!(method.block_encryption(), Some(BlockEncryption::Aes128Cbc));
        assert_eq!(method.key_size, Some(128));
        let key_info = data.common.key_info.as_ref().expect("key info");
        assert_eq!(key_info.id.as_deref(), Some("ki"));
        assert_eq!(key_info.key_name(), Some("session"));
        assert_eq!(data.ciphertext()?, &[0, 1, 2]);
        let properties = data
            .common
            .encryption_properties
            .as_ref()
            .expect("properties");
        assert_eq!(properties.id.as_deref(), Some("props"));
        assert_eq!(properties.properties.len(), 1);
        assert_eq!(properties.properties[0].text_content(), "2026-07-27");
        Ok(())
    }

    #[test]
    fn an_encrypted_key_parses_its_extras() -> Result<()> {
        let key = EncryptedKey::from_xml(
            r##"<xenc:EncryptedKey xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" Recipient="bob">
                 <xenc:CipherData><xenc:CipherValue>AA==</xenc:CipherValue></xenc:CipherData>
                 <xenc:ReferenceList>
                   <xenc:DataReference URI="#ed1"/>
                   <xenc:KeyReference URI="#ek2"/>
                 </xenc:ReferenceList>
                 <xenc:CarriedKeyName>session</xenc:CarriedKeyName>
               </xenc:EncryptedKey>"##,
        )?;
        assert_eq!(key.recipient.as_deref(), Some("bob"));
        assert_eq!(key.carried_key_name.as_deref(), Some("session"));
        let list = key.reference_list.as_ref().expect("a reference list");
        assert_eq!(list.data_references().collect::<Vec<_>>(), ["#ed1"]);
        assert_eq!(list.key_references().collect::<Vec<_>>(), ["#ek2"]);
        Ok(())
    }

    #[test]
    fn a_nested_encrypted_key_in_key_info_parses() -> Result<()> {
        let data = EncryptedData::from_xml(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                   xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
                 <ds:KeyInfo>
                   <xenc:EncryptedKey>
                     <xenc:EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#kw-aes256"/>
                     <ds:KeyInfo><ds:KeyName>kek</ds:KeyName></ds:KeyInfo>
                     <xenc:CipherData><xenc:CipherValue>AAA=</xenc:CipherValue></xenc:CipherData>
                   </xenc:EncryptedKey>
                 </ds:KeyInfo>
                 <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
               </xenc:EncryptedData>"#,
        )?;
        let key_info = data.common.key_info.as_ref().expect("key info");
        let nested = key_info.encrypted_key().expect("a nested EncryptedKey");
        assert_eq!(
            nested.common.key_info.as_ref().and_then(KeyInfo::key_name),
            Some("kek")
        );
        assert_eq!(nested.ciphertext()?, &[0, 0]);
        Ok(())
    }

    #[test]
    fn unrecognized_key_info_children_survive() -> Result<()> {
        let data = EncryptedData::from_xml(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                   xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
                 <ds:KeyInfo><ds:X509Data><ds:X509IssuerSerial/></ds:X509Data></ds:KeyInfo>
                 <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
               </xenc:EncryptedData>"#,
        )?;
        let key_info = data.common.key_info.as_ref().expect("key info");
        assert_eq!(key_info.items.len(), 1);
        match &key_info.items[0] {
            KeyInfoItem::Other(XmlFragment::Element(xml)) => {
                assert!(xml.contains("X509IssuerSerial"), "{xml}");
                assert!(xml.contains("http://www.w3.org/2000/09/xmldsig#"), "{xml}");
            }
            other => panic!("expected an opaque fragment, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_cipher_reference_parses_with_transforms() -> Result<()> {
        let data = EncryptedData::from_xml(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                   xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
                 <xenc:CipherData>
                   <xenc:CipherReference URI="http://example.org/blob">
                     <xenc:Transforms>
                       <ds:Transform Algorithm="urn:example:t"><ds:XPath>self::text()</ds:XPath></ds:Transform>
                     </xenc:Transforms>
                   </xenc:CipherReference>
                 </xenc:CipherData>
               </xenc:EncryptedData>"#,
        )?;
        let reference = data
            .common
            .cipher_data
            .reference()
            .expect("a cipher reference");
        assert_eq!(reference.uri, "http://example.org/blob");
        assert_eq!(reference.transforms.len(), 1);
        assert_eq!(reference.transforms[0].algorithm, "urn:example:t");
        assert!(
            reference.transforms[0]
                .content
                .iter()
                .any(|fragment| fragment.as_str().contains("XPath"))
        );
        assert!(matches!(
            data.ciphertext(),
            Err(EncError::CipherReferenceNotResolved(_))
        ));
        Ok(())
    }

    #[test]
    fn parsing_and_writing_are_inverses() -> Result<()> {
        let source = r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#" Id="e" Type="http://www.w3.org/2001/04/xmlenc#Element"><xenc:EncryptionMethod Algorithm="http://www.w3.org/2009/xmlenc11#aes256-gcm"><xenc:KeySize>256</xenc:KeySize></xenc:EncryptionMethod><ds:KeyInfo><ds:KeyName>k</ds:KeyName></ds:KeyInfo><xenc:CipherData><xenc:CipherValue>AQIDBA==</xenc:CipherValue></xenc:CipherData></xenc:EncryptedData>"#;
        let parsed = EncryptedData::from_xml(source)?;
        let written = parsed.to_xml()?;
        let reparsed = EncryptedData::from_xml(&written)?;
        assert_eq!(parsed, reparsed);
        Ok(())
    }

    #[test]
    fn an_encrypted_key_is_not_an_encrypted_data() {
        let doc = oxixml_dom::build::parse_str(
            r#"<xenc:EncryptedKey xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"><xenc:CipherData><xenc:CipherValue/></xenc:CipherData></xenc:EncryptedKey>"#,
        )
        .expect("well-formed");
        let root = doc.document_element().expect("a root");
        assert!(matches!(
            EncryptedData::parse(&doc, root),
            Err(EncError::InvalidStructure(_))
        ));
        assert!(EncryptedKey::parse(&doc, root).is_ok());
    }

    /// An `xenc:EncryptedKey` whose `KeyInfo` is a full `ECDH-ES` `AgreementMethod` — `KA-Nonce`,
    /// a `ConcatKDFParams`-bearing `KeyDerivationMethod`, an `ECKeyValue` originator, and a
    /// `KeyName` recipient — parses into the structured model and re-serializes faithfully.
    #[test]
    fn an_agreement_method_with_an_ec_key_value_round_trips() -> Result<()> {
        let source = concat!(
            r#"<xenc:EncryptedKey xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:xenc11="http://www.w3.org/2009/xmlenc11#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#">"#,
            r#"<xenc:EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#kw-aes128"/>"#,
            r#"<ds:KeyInfo><xenc:AgreementMethod Algorithm="http://www.w3.org/2009/xmlenc11#ECDH-ES">"#,
            r#"<xenc:KA-Nonce>AQID</xenc:KA-Nonce>"#,
            r#"<xenc11:KeyDerivationMethod Algorithm="http://www.w3.org/2009/xmlenc11#ConcatKDF">"#,
            r#"<xenc11:ConcatKDFParams AlgorithmID="0006414553313238" PartyUInfo="00416c696365"><ds:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/></xenc11:ConcatKDFParams>"#,
            r#"</xenc11:KeyDerivationMethod>"#,
            r#"<xenc:OriginatorKeyInfo><ds:KeyValue><xenc11:ECKeyValue><xenc11:NamedCurve URI="urn:oid:1.2.840.10045.3.1.7"/><xenc11:PublicKey>BA==</xenc11:PublicKey></xenc11:ECKeyValue></ds:KeyValue></xenc:OriginatorKeyInfo>"#,
            r#"<xenc:RecipientKeyInfo><ds:KeyName>bob</ds:KeyName></xenc:RecipientKeyInfo>"#,
            r#"</xenc:AgreementMethod></ds:KeyInfo>"#,
            r#"<xenc:CipherData><xenc:CipherValue>AAECAwQF</xenc:CipherValue></xenc:CipherData>"#,
            r#"</xenc:EncryptedKey>"#,
        );
        let key = EncryptedKey::from_xml(source)?;
        let key_info = key.common.key_info.as_ref().expect("a key info");
        let agreement = key_info.agreement_method().expect("an agreement method");
        assert_eq!(
            agreement.algorithm,
            "http://www.w3.org/2009/xmlenc11#ECDH-ES"
        );
        assert_eq!(agreement.ka_nonce.as_deref(), Some(&[1u8, 2, 3][..]));

        let derivation = agreement
            .key_derivation_method
            .as_ref()
            .expect("a key derivation method");
        assert_eq!(
            derivation.digest_method.as_deref(),
            Some("http://www.w3.org/2001/04/xmlenc#sha256")
        );
        assert_eq!(derivation.algorithm_id, b"\x06AES128");
        assert_eq!(derivation.party_u_info, b"Alice");
        assert!(derivation.party_v_info.is_empty());

        let originator = agreement
            .originator_key_info
            .as_ref()
            .expect("an originator key info");
        let ec = originator.ec_key_value().expect("an ECKeyValue");
        assert_eq!(ec.named_curve, "urn:oid:1.2.840.10045.3.1.7");
        assert_eq!(ec.public_key, [0x04]);

        let recipient = agreement
            .recipient_key_info
            .as_ref()
            .expect("a recipient key info");
        assert_eq!(recipient.key_name(), Some("bob"));

        let written = key.to_xml()?;
        let reparsed = EncryptedKey::from_xml(&written)?;
        assert_eq!(key, reparsed);
        Ok(())
    }

    /// A foreign child inside the `AgreementMethod`'s own `##other` wildcard survives a round
    /// trip verbatim — the no-regression check for the crate's passthrough guarantee, extended
    /// to the new structured variant.
    #[test]
    fn an_agreement_method_preserves_a_foreign_child_verbatim() -> Result<()> {
        let source = concat!(
            r#"<xenc:EncryptedKey xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#" xmlns:ex="urn:example:ext">"#,
            r#"<xenc:EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#kw-aes128"/>"#,
            r#"<ds:KeyInfo><xenc:AgreementMethod Algorithm="http://www.w3.org/2009/xmlenc11#ECDH-ES">"#,
            r#"<ex:Extension attr="1"><ex:Nested/></ex:Extension>"#,
            r#"</xenc:AgreementMethod></ds:KeyInfo>"#,
            r#"<xenc:CipherData><xenc:CipherValue>AAA=</xenc:CipherValue></xenc:CipherData>"#,
            r#"</xenc:EncryptedKey>"#,
        );
        let key = EncryptedKey::from_xml(source)?;
        let agreement = key
            .common
            .key_info
            .as_ref()
            .and_then(KeyInfo::agreement_method)
            .expect("an agreement method");
        assert_eq!(agreement.other.len(), 1);
        let fragment = agreement.other[0].as_str();
        assert!(fragment.contains("Extension"), "{fragment}");
        assert!(fragment.contains("urn:example:ext"), "{fragment}");
        assert!(fragment.contains("Nested"), "{fragment}");

        let written = key.to_xml()?;
        assert!(written.contains("urn:example:ext"), "{written}");
        let reparsed = EncryptedKey::from_xml(&written)?;
        assert_eq!(key, reparsed);
        Ok(())
    }

    /// A `xenc:DHKeyValue` wrapped in a `ds:KeyValue` — the conventional placement — round-trips
    /// through the same structured item a bare one would, still wrapped in `ds:KeyValue` on the
    /// way back out.
    #[test]
    fn a_dh_key_value_round_trips_wrapped_in_a_key_value() -> Result<()> {
        let source = r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
             <ds:KeyInfo>
               <ds:KeyValue>
                 <xenc:DHKeyValue>
                   <xenc:P>AAE=</xenc:P>
                   <xenc:Q>AAI=</xenc:Q>
                   <xenc:Generator>Ag==</xenc:Generator>
                   <xenc:Public>AAM=</xenc:Public>
                   <xenc:seed>AAQ=</xenc:seed>
                 </xenc:DHKeyValue>
               </ds:KeyValue>
             </ds:KeyInfo>
             <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
           </xenc:EncryptedData>"#;
        let data = EncryptedData::from_xml(source)?;
        let key_info = data.common.key_info.as_ref().expect("a key info");
        let dh = key_info.dh_key_value().expect("a DHKeyValue");
        assert_eq!(dh.p, [0x00, 0x01]);
        assert_eq!(dh.q.as_deref(), Some(&[0x00, 0x02][..]));
        assert_eq!(dh.generator, [0x02]);
        assert_eq!(dh.public, [0x00, 0x03]);
        assert_eq!(dh.seed.as_deref(), Some(&[0x00, 0x04][..]));

        let written = data.to_xml()?;
        assert!(written.contains("ds:KeyValue"), "{written}");
        assert!(written.contains("xenc:DHKeyValue"), "{written}");
        let reparsed = EncryptedData::from_xml(&written)?;
        assert_eq!(data, reparsed);
        Ok(())
    }

    /// An `ECKeyValue` in the XML Signature 1.1 namespace, with the curve as `NamedCurve`'s text
    /// content rather than a `URI` attribute, is accepted leniently — see the
    /// [module documentation](crate::model::agreement) for why both shapes are recognized.
    #[test]
    fn an_ec_key_value_is_accepted_in_either_namespace_and_named_curve_shape() -> Result<()> {
        let dsig11_with_uri_attribute = r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#" xmlns:dsig11="http://www.w3.org/2009/xmldsig11#">
             <ds:KeyInfo><dsig11:ECKeyValue><dsig11:NamedCurve URI="urn:oid:1.2.840.10045.3.1.7"/><dsig11:PublicKey>BAE=</dsig11:PublicKey></dsig11:ECKeyValue></ds:KeyInfo>
             <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
           </xenc:EncryptedData>"#;
        let xenc11_with_text_content = r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#" xmlns:xenc11="http://www.w3.org/2009/xmlenc11#">
             <ds:KeyInfo><xenc11:ECKeyValue><xenc11:NamedCurve>urn:oid:1.2.840.10045.3.1.7</xenc11:NamedCurve><xenc11:PublicKey>BAE=</xenc11:PublicKey></xenc11:ECKeyValue></ds:KeyInfo>
             <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
           </xenc:EncryptedData>"#;

        for source in [dsig11_with_uri_attribute, xenc11_with_text_content] {
            let data = EncryptedData::from_xml(source)?;
            let key_info = data.common.key_info.as_ref().expect("a key info");
            let ec = key_info.ec_key_value().expect("an ECKeyValue");
            assert_eq!(ec.named_curve, "urn:oid:1.2.840.10045.3.1.7");
            assert_eq!(ec.public_key, [0x04, 0x01]);
        }
        Ok(())
    }

    /// An `ECKeyValue` naming `ECParameters` (explicit curve parameters) rather than a
    /// `NamedCurve` is a structural error, not a silently ignored one.
    #[test]
    fn an_ec_key_value_with_explicit_parameters_is_rejected() {
        let source = r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#" xmlns:ds="http://www.w3.org/2000/09/xmldsig#" xmlns:xenc11="http://www.w3.org/2009/xmlenc11#">
             <ds:KeyInfo><xenc11:ECKeyValue><xenc11:ECParameters/><xenc11:PublicKey>BAE=</xenc11:PublicKey></xenc11:ECKeyValue></ds:KeyInfo>
             <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
           </xenc:EncryptedData>"#;
        assert!(matches!(
            EncryptedData::from_xml(source),
            Err(EncError::InvalidStructure(_))
        ));
    }
}
