//! Encrypting and decrypting nodes of a document tree.
//!
//! This is the layer that ties the [model](crate::model), the [ciphers](crate::cipher), the
//! [key resolution](crate::key) and the [fragment handling](crate::fragment) together into the
//! three operations an application actually performs.
//!
//! # Element versus content
//!
//! [XMLEnc §2.2](https://www.w3.org/TR/xmlenc-core1/#sec-Usage) draws a distinction that is easy
//! to get wrong. Given
//!
//! ```xml
//! <PaymentInfo><CreditCard><Number>4019…</Number></CreditCard></PaymentInfo>
//! ```
//!
//! * [`encrypt_element`] on `CreditCard` hides the element *and its name*: the result is
//!   `<PaymentInfo><EncryptedData …/></PaymentInfo>`, and an observer cannot tell there was a
//!   credit card there at all.
//! * [`encrypt_content`] on `CreditCard` hides only what is inside it:
//!   `<PaymentInfo><CreditCard><EncryptedData …/></CreditCard></PaymentInfo>`, which leaks the
//!   fact that a `CreditCard` element exists.
//!
//! The two write different `Type` attributes, and [`decrypt`] does the right thing for either
//! because at the tree level both mean "replace this `EncryptedData` with the nodes that come
//! out" (XMLEnc §4.5.2).
//!
//! # What is *not* protected
//!
//! Encrypting an element hides its name and content but not its *position* or *size*: an
//! observer sees where in the document something was encrypted and roughly how big it was.
//! XMLEnc §6.3 spells this out. Neither is the `EncryptedData` element itself authenticated —
//! see [`crate::cipher`] — so an attacker who can rewrite the document can move, duplicate or
//! delete `EncryptedData` elements even when AES-GCM makes the ciphertext itself untamperable.
//! Combine with XML Signature when integrity of the document structure matters.
//!
//! # Example
//!
//! ```
//! use oxixml_dom::build;
//! use oxixml_enc::key::{KeyStore, SymmetricKey};
//! use oxixml_enc::random::SystemRandom;
//! use oxixml_enc::{EncryptOptions, decrypt, encrypt_element};
//!
//! let mut doc = build::parse_str(
//!     r#"<PaymentInfo xmlns="urn:example:payment">
//!          <Name>John Smith</Name>
//!          <CreditCard Limit="5,000"><Number>4019 2445 0277 5567</Number></CreditCard>
//!        </PaymentInfo>"#,
//! )?;
//! let root = doc.document_element().expect("a document element");
//! let card = doc
//!     .get_elements_by_tag_name_ns(root, Some("urn:example:payment"), "CreditCard")
//!     .next()
//!     .expect("a CreditCard element");
//!
//! let key = SymmetricKey::new(vec![0x2b; 32]);
//! let mut rng = SystemRandom::new();
//! let encrypted = encrypt_element(&mut doc, card, &key, &EncryptOptions::new(), &mut rng)?;
//!
//! let xml = doc.to_xml();
//! assert!(!xml.contains("4019"), "the card number is gone");
//! assert!(!xml.contains("CreditCard"), "so is the element name");
//!
//! // Decrypting puts it back, exactly.
//! let store = KeyStore::new().with_default_key(key);
//! decrypt(&mut doc, encrypted, &store)?;
//! assert!(doc.to_xml().contains("4019 2445 0277 5567"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::algorithm::{BlockEncryption, KeyAgreement, KeyTransport, KeyWrap};
use crate::cipher;
use crate::crypto::ecdh::EcPublicKey;
use crate::crypto::kdf::ConcatKdfParams;
use crate::crypto::rsa::{OaepParams, RsaPublicKey};
use crate::error::{EncError, Result};
use crate::fragment;
use crate::key::{KeyContext, KeyResolver, SymmetricKey};
use crate::model::{
    AgreementMethod, ContentType, EcKeyValue, EncryptedData, EncryptedKey, EncryptionMethod,
    EncryptionProperties, KeyDerivationMethod, KeyInfo, KeyInfoItem,
};
use crate::random::RandomSource;

/// How an `EncryptedData` should be built.
///
/// The defaults are the safe ones: AES-256-GCM, which is the only XML Encryption block cipher
/// immune to the chosen-ciphertext attacks of §6.1.1, and a `KeySize` element that matches it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptOptions {
    /// The block-encryption algorithm.
    pub algorithm: BlockEncryption,
    /// The `Id` attribute to put on the `EncryptedData`, which a `ReferenceList` can point at.
    pub id: Option<String>,
    /// The `ds:KeyInfo` to include, if any.
    pub key_info: Option<KeyInfo>,
    /// The `MimeType` attribute, advisory only.
    pub mime_type: Option<String>,
    /// The `Encoding` attribute, advisory only.
    pub encoding: Option<String>,
    /// The `EncryptionProperties` to include, if any. Remember that these are plaintext.
    pub encryption_properties: Option<EncryptionProperties>,
    /// Whether to emit the `xenc:KeySize` child of the `EncryptionMethod`. It is redundant for
    /// every algorithm this crate implements, but some consumers expect it.
    pub emit_key_size: bool,
}

impl Default for EncryptOptions {
    fn default() -> Self {
        Self {
            algorithm: BlockEncryption::default(),
            id: None,
            key_info: None,
            mime_type: None,
            encoding: None,
            encryption_properties: None,
            emit_key_size: true,
        }
    }
}

impl EncryptOptions {
    /// Returns the defaults: AES-256-GCM and nothing else.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the block-encryption algorithm.
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: BlockEncryption) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Sets the `Id` attribute.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the `ds:KeyInfo`.
    #[must_use]
    pub fn with_key_info(mut self, key_info: KeyInfo) -> Self {
        self.key_info = Some(key_info);
        self
    }

    /// Sets the `EncryptionProperties`.
    #[must_use]
    pub fn with_encryption_properties(mut self, properties: EncryptionProperties) -> Self {
        self.encryption_properties = Some(properties);
        self
    }

    /// Selects whether the redundant `xenc:KeySize` child is emitted.
    #[must_use]
    pub const fn with_key_size(mut self, emit: bool) -> Self {
        self.emit_key_size = emit;
        self
    }

    /// Builds the `EncryptedData` model for `ciphertext` and `content_type`.
    fn build(&self, ciphertext: Vec<u8>, content_type: ContentType) -> EncryptedData {
        let mut method = EncryptionMethod::block(self.algorithm);
        if !self.emit_key_size {
            method.key_size = None;
        }
        let mut data = EncryptedData::new(ciphertext);
        data.common.id = self.id.clone();
        data.common.content_type = Some(content_type);
        data.common.mime_type = self.mime_type.clone();
        data.common.encoding = self.encoding.clone();
        data.common.encryption_method = Some(method);
        data.common.key_info = self.key_info.clone();
        data.common.encryption_properties = self.encryption_properties.clone();
        data
    }
}

/// Encrypts `element` and replaces it with an `xenc:EncryptedData` carrying
/// `Type="…xmlenc#Element"`.
///
/// The element is serialized with the namespace handling of XMLEnc §4.5.3.1, so that decrypting
/// it back into a different context cannot change what it means. Returns the identifier of the
/// new `EncryptedData` element.
///
/// # Errors
///
/// * [`EncError::InvalidStructure`] — `element` is not an element, or has no parent to be
///   replaced in.
/// * [`EncError::InvalidKeyLength`] — the key does not match the algorithm.
/// * [`EncError::Entropy`] — the random source failed.
/// * [`EncError::Dom`] — the tree could not be rewritten.
pub fn encrypt_element(
    doc: &mut Document,
    element: NodeId,
    key: &SymmetricKey,
    options: &EncryptOptions,
    rng: &mut dyn RandomSource,
) -> Result<NodeId> {
    require_element(doc, element)?;
    let parent = doc.parent(element).ok_or_else(|| {
        EncError::InvalidStructure(
            "the element to encrypt has no parent, so it cannot be replaced".to_owned(),
        )
    })?;

    let plaintext = fragment::serialize_self_contained(doc, element)?;
    let ciphertext = cipher::encrypt(options.algorithm, key, plaintext.as_bytes(), rng)?;
    let model = options.build(ciphertext, ContentType::Element);

    let replacement = model.write_into(doc, None)?;
    doc.replace_child(parent, replacement, element)?;
    Ok(replacement)
}

/// Encrypts the *content* of `element` and replaces its children with a single
/// `xenc:EncryptedData` carrying `Type="…xmlenc#Content"`.
///
/// The element itself, its name and its attributes stay in the clear. Returns the identifier of
/// the new `EncryptedData` element.
///
/// # Errors
///
/// The same as [`encrypt_element`], except that no parent is required — an element with no
/// parent still has content.
pub fn encrypt_content(
    doc: &mut Document,
    element: NodeId,
    key: &SymmetricKey,
    options: &EncryptOptions,
    rng: &mut dyn RandomSource,
) -> Result<NodeId> {
    require_element(doc, element)?;

    let plaintext = fragment::serialize_children(doc, element)?;
    let ciphertext = cipher::encrypt(options.algorithm, key, plaintext.as_bytes(), rng)?;
    let model = options.build(ciphertext, ContentType::Content);

    for child in doc.children(element).collect::<Vec<_>>() {
        doc.remove_child(element, child)?;
    }
    let replacement = model.write_into(doc, None)?;
    doc.append_child(element, replacement)?;
    Ok(replacement)
}

/// What came out of [`decrypt`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decrypted {
    /// The `Type` was `Element` or `Content`, so the plaintext was parsed and the
    /// `EncryptedData` element was replaced by these nodes, in order.
    ///
    /// The nodes are already in the tree; the identifiers are returned so the caller can
    /// operate on what it just revealed.
    Nodes(Vec<NodeId>),
    /// The `Type` was absent or named something that is not XML, so the plaintext is returned
    /// as octets and the tree is left untouched.
    Octets(Vec<u8>),
}

impl Decrypted {
    /// Returns the revealed nodes, or an empty slice for octets.
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        match self {
            Self::Nodes(nodes) => nodes,
            Self::Octets(_) => &[],
        }
    }

    /// Returns the octets, or `None` when the plaintext was XML that went back into the tree.
    #[must_use]
    pub fn octets(&self) -> Option<&[u8]> {
        match self {
            Self::Octets(octets) => Some(octets),
            Self::Nodes(_) => None,
        }
    }
}

/// Decrypts the `xenc:EncryptedData` element `node`, replacing it with the plaintext when the
/// plaintext is XML.
///
/// The key comes from `resolver`, which is handed the structure's `EncryptionMethod` and
/// `ds:KeyInfo`; see [`crate::key`] for the resolution rules and the trust model.
///
/// # Errors
///
/// * [`EncError::InvalidStructure`] — `node` is not an `xenc:EncryptedData`, has no
///   `EncryptionMethod`, or has no parent to be replaced in.
/// * [`EncError::UnsupportedAlgorithm`] — the `EncryptionMethod` names something this crate
///   does not implement.
/// * [`EncError::KeyNotFound`] — the resolver could not supply a key.
/// * [`EncError::DecryptionFailed`] — the ciphertext did not decrypt. This one variant covers
///   a wrong key, a failed authentication tag and malformed padding alike, deliberately.
/// * [`EncError::NotUtf8`] / [`EncError::Build`] — the plaintext decrypted but is not
///   well-formed XML, which after an authenticated decryption means a faulty sender.
pub fn decrypt(doc: &mut Document, node: NodeId, resolver: &dyn KeyResolver) -> Result<Decrypted> {
    let model = EncryptedData::parse(doc, node)?;
    let method = model.common.encryption_method.as_ref().ok_or_else(|| {
        EncError::InvalidStructure(
            "the `xenc:EncryptedData` has no `EncryptionMethod`, so the algorithm is unknown"
                .to_owned(),
        )
    })?;
    let algorithm = method
        .block_encryption()
        .ok_or_else(|| EncError::UnsupportedAlgorithm(method.algorithm.clone()))?;
    if method.key_size_disagrees_with_algorithm() {
        return Err(EncError::InvalidStructure(format!(
            "`xenc:KeySize` contradicts `{}`",
            method.algorithm
        )));
    }

    let key = resolver.resolve(&KeyContext {
        document: doc,
        encryption_method: Some(method),
        key_info: model.common.key_info.as_ref(),
        expected_key_len: Some(algorithm.key_len()),
    })?;

    let plaintext = cipher::decrypt(algorithm, &key, model.common.ciphertext()?)?;

    let is_xml = model
        .common
        .content_type
        .as_ref()
        .is_some_and(ContentType::is_xml);
    if !is_xml {
        return Ok(Decrypted::Octets(plaintext));
    }

    let text = String::from_utf8(plaintext)?;
    let inserted = replace_with_parsed(doc, node, &text)?;
    Ok(Decrypted::Nodes(inserted))
}

/// Parses `plaintext` in the context of `node`'s parent and puts the resulting nodes where
/// `node` was, per the decrypt-and-replace algorithm of XMLEnc §4.5.2.
fn replace_with_parsed(doc: &mut Document, node: NodeId, plaintext: &str) -> Result<Vec<NodeId>> {
    let parent = doc.parent(node).ok_or_else(|| {
        EncError::InvalidStructure(
            "the `xenc:EncryptedData` has no parent, so there is nothing to replace it in"
                .to_owned(),
        )
    })?;

    let wrapper_doc = fragment::wrap_in_context(doc, parent, plaintext)?;
    let wrapper = wrapper_doc.document_element().ok_or_else(|| {
        EncError::InvalidStructure("the decryption wrapper has no document element".to_owned())
    })?;

    // The `EncryptedData` has to go *first*. When it is the document element, the tree cannot
    // hold it and the replacement at the same time, and inserting first would be rejected.
    let reference = doc.next_sibling(node);
    doc.remove_child(parent, node)?;

    let mut inserted = Vec::new();
    for child in wrapper_doc.children(wrapper).collect::<Vec<_>>() {
        let imported = doc.import_node(&wrapper_doc, child, true)?;
        doc.insert_before(parent, imported, reference)?;
        fragment::drop_redundant_namespace_declarations(doc, imported)?;
        inserted.push(imported);
    }
    Ok(inserted)
}

/// Builds an `xenc:EncryptedKey` that carries `content_key` wrapped under a symmetric
/// key-encryption key.
///
/// The result is what goes into an [`EncryptOptions::key_info`] so that the recipient can
/// recover the content-encryption key from the document itself.
///
/// # Errors
///
/// See [`cipher::wrap_key`].
pub fn wrap_key_for_key_info(
    algorithm: KeyWrap,
    key_encryption_key: &SymmetricKey,
    content_key: &SymmetricKey,
    key_encryption_key_name: Option<&str>,
) -> Result<EncryptedKey> {
    let wrapped = cipher::wrap_key(algorithm, key_encryption_key, content_key)?;
    let mut key = EncryptedKey::new(wrapped);
    key.common.encryption_method = Some(EncryptionMethod::key_wrap(algorithm));
    key.common.key_info = key_encryption_key_name.map(KeyInfo::with_key_name);
    Ok(key)
}

/// Builds an `xenc:EncryptedKey` that carries `content_key` encrypted under an RSA public key.
///
/// The `EncryptionMethod` records the OAEP parameters that were used — the digest as
/// `ds:DigestMethod`, the mask generation function as `xenc11:MGF` where the algorithm permits
/// it, and the label as `xenc:OAEPparams` — so that the recipient can reconstruct them.
///
/// # Errors
///
/// See [`cipher::transport_key`].
pub fn transport_key_for_key_info(
    algorithm: KeyTransport,
    public_key: &RsaPublicKey,
    content_key: &SymmetricKey,
    params: &OaepParams,
    recipient_key_name: Option<&str>,
    rng: &mut dyn RandomSource,
) -> Result<EncryptedKey> {
    let ciphertext = cipher::transport_key(algorithm, public_key, content_key, params, rng)?;

    let mut method = EncryptionMethod::key_transport(algorithm);
    if algorithm.is_oaep() {
        method.digest_method = Some(params.digest.uri().to_owned());
        if algorithm.allows_mgf_element() {
            method.mgf = Some(crate::crypto::mgf1::mgf1_uri(params.mgf_digest).to_owned());
        }
        if !params.label.is_empty() {
            method.oaep_params = Some(params.label.clone());
        }
    }

    let mut key = EncryptedKey::new(ciphertext);
    key.common.encryption_method = Some(method);
    key.common.key_info = recipient_key_name.map(KeyInfo::with_key_name);
    Ok(key)
}

/// Builds an `xenc:EncryptedKey` that carries `content_key` wrapped under a key-encryption key
/// agreed with `recipient` via `ECDH-ES` — the key-agreement method XMLEnc 1.1 §5.6.4 requires
/// and the only one this crate exposes an encryption-direction helper for; `dh` and `dh-es` are
/// implemented for decryption (see [`crate::key::KeyStore::open_encrypted_key`]) but not, yet,
/// for building a fresh `AgreementMethod`.
///
/// A fresh ephemeral originator key pair is generated on `recipient`'s curve — see
/// [`crate::agreement::ecdh_es_wrap_key`] — the key-encryption key is derived with the ConcatKDF
/// from `params`, and `content_key` is wrapped under it with `wrap`. The result's `KeyInfo` is an
/// `AgreementMethod` carrying the originator's ephemeral public key in its `OriginatorKeyInfo`
/// and `params` in its `KeyDerivationMethod`; `recipient_key_name`, when given, becomes a
/// `ds:KeyName` in the `RecipientKeyInfo` so that the [`KeyStore`](crate::key::KeyStore) that
/// opens it can find the matching registered private key.
///
/// # Errors
///
/// The errors of [`crate::agreement::ecdh_es_wrap_key`].
pub fn ecdh_es_wrap_key_for_key_info(
    wrap: KeyWrap,
    recipient: &EcPublicKey,
    params: &ConcatKdfParams,
    content_key: &SymmetricKey,
    recipient_key_name: Option<&str>,
    rng: &mut dyn RandomSource,
) -> Result<EncryptedKey> {
    let agreed = crate::agreement::ecdh_es_wrap_key(recipient, wrap, params, content_key, rng)?;

    let mut agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
    agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(params));
    agreement.originator_key_info = Some(KeyInfo {
        id: None,
        items: vec![KeyInfoItem::EcKeyValue(EcKeyValue::from_public_key(
            &agreed.originator_public_key,
        )?)],
    });
    agreement.recipient_key_info = recipient_key_name.map(KeyInfo::with_key_name);

    let mut key = EncryptedKey::new(agreed.wrapped_key);
    key.common.encryption_method = Some(EncryptionMethod::key_wrap(wrap));
    key.common.key_info = Some(KeyInfo {
        id: None,
        items: vec![KeyInfoItem::AgreementMethod(Box::new(agreement))],
    });
    Ok(key)
}

/// Rejects a node that is not an element.
fn require_element(doc: &Document, node: NodeId) -> Result<()> {
    match doc.kind(node) {
        Some(NodeKind::Element) => Ok(()),
        Some(kind) => Err(EncError::InvalidStructure(format!(
            "expected an element, found a {kind}"
        ))),
        None => Err(EncError::InvalidStructure(
            "the node does not belong to this document".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{KeyStore, StaticKey};
    use crate::random::PseudoRandom;
    use oxixml_dom::build;

    fn document(xml: &str) -> Result<Document> {
        Ok(build::parse_str(xml)?)
    }

    fn key() -> SymmetricKey {
        SymmetricKey::new(vec![0x2b; 32])
    }

    #[test]
    fn encrypting_an_element_hides_its_name() -> Result<()> {
        let mut doc = document("<a><b><c>secret</c></b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        encrypt_element(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(1),
        )?;
        let xml = doc.to_xml();
        assert!(!xml.contains("secret"), "{xml}");
        assert!(!xml.contains("<b>"), "{xml}");
        assert!(xml.contains("xenc:EncryptedData"), "{xml}");
        assert!(
            xml.contains(r#"Type="http://www.w3.org/2001/04/xmlenc#Element""#),
            "{xml}"
        );
        Ok(())
    }

    #[test]
    fn encrypting_content_keeps_the_element() -> Result<()> {
        let mut doc = document("<a><b attr=\"v\"><c>secret</c></b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        encrypt_content(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(2),
        )?;
        let xml = doc.to_xml();
        assert!(!xml.contains("secret"), "{xml}");
        assert!(xml.contains(r#"<b attr="v">"#), "{xml}");
        assert!(
            xml.contains(r#"Type="http://www.w3.org/2001/04/xmlenc#Content""#),
            "{xml}"
        );
        Ok(())
    }

    #[test]
    fn a_round_trip_reproduces_the_tree() -> Result<()> {
        let source =
            r#"<a xmlns="urn:d" xmlns:p="urn:p"><b p:x="1"><c>text &amp; more</c><!--k--></b></a>"#;
        let original = document(source)?;
        let mut doc = document(source)?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");

        let encrypted = encrypt_element(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(3),
        )?;
        let resolver = StaticKey::new(key());
        let outcome = decrypt(&mut doc, encrypted, &resolver)?;
        assert_eq!(outcome.nodes().len(), 1);
        assert!(
            original.structural_eq(&doc),
            "{:?}",
            original.structural_diff(&doc)
        );
        Ok(())
    }

    #[test]
    fn a_content_round_trip_reproduces_the_children() -> Result<()> {
        let source = r#"<a xmlns="urn:d">text<b/>more<![CDATA[raw]]></a>"#;
        let mut doc = document(source)?;
        let root = doc.document_element().expect("a root");
        let encrypted = encrypt_content(
            &mut doc,
            root,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(4),
        )?;
        decrypt(&mut doc, encrypted, &StaticKey::new(key()))?;
        // The CDATA section serializes as a CDATA section, so the text is preserved verbatim.
        assert_eq!(doc.text_content(root).as_deref(), Some("textmoreraw"));
        Ok(())
    }

    #[test]
    fn every_algorithm_round_trips_through_the_tree() -> Result<()> {
        for algorithm in BlockEncryption::ALL {
            let mut doc = document("<a><b>payload</b></a>")?;
            let root = doc.document_element().expect("a root");
            let target = doc.children(root).next().expect("a child");
            let key = SymmetricKey::new(vec![0x3c; algorithm.key_len()]);
            let options = EncryptOptions::new().with_algorithm(algorithm);
            let encrypted =
                encrypt_element(&mut doc, target, &key, &options, &mut PseudoRandom::new(5))?;
            decrypt(&mut doc, encrypted, &StaticKey::new(key))?;
            assert_eq!(doc.to_xml(), "<a><b>payload</b></a>", "{algorithm:?}");
        }
        Ok(())
    }

    #[test]
    fn a_wrong_key_fails_opaquely() -> Result<()> {
        let mut doc = document("<a><b>payload</b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let encrypted = encrypt_element(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(6),
        )?;
        let wrong = StaticKey::new(SymmetricKey::new(vec![0xff; 32]));
        assert!(matches!(
            decrypt(&mut doc, encrypted, &wrong),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn the_document_element_can_be_encrypted() -> Result<()> {
        let mut doc = document("<a><b/></a>")?;
        let root = doc.document_element().expect("a root");
        let encrypted = encrypt_element(
            &mut doc,
            root,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(7),
        )?;
        assert!(doc.to_xml().starts_with("<xenc:EncryptedData"));
        decrypt(&mut doc, encrypted, &StaticKey::new(key()))?;
        assert_eq!(doc.to_xml(), "<a><b/></a>");
        Ok(())
    }

    #[test]
    fn a_non_element_cannot_be_encrypted() -> Result<()> {
        let mut doc = document("<a>text</a>")?;
        let root = doc.document_element().expect("a root");
        let text = doc.children(root).next().expect("a text node");
        assert!(matches!(
            encrypt_element(
                &mut doc,
                text,
                &key(),
                &EncryptOptions::new(),
                &mut PseudoRandom::new(8)
            ),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    #[test]
    fn a_detached_element_cannot_be_encrypted_in_place() -> Result<()> {
        let mut doc = document("<a/>")?;
        let orphan = doc.create_element("b")?;
        assert!(matches!(
            encrypt_element(
                &mut doc,
                orphan,
                &key(),
                &EncryptOptions::new(),
                &mut PseudoRandom::new(9)
            ),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    #[test]
    fn a_non_xml_type_yields_octets_and_leaves_the_tree_alone() -> Result<()> {
        let mut doc = document("<a><placeholder/></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");

        // Encrypt, then rewrite the Type to something opaque.
        let encrypted = encrypt_element(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(10),
        )?;
        doc.set_attribute(encrypted, "Type", "urn:example:jpeg")?;

        let before = doc.to_xml();
        let outcome = decrypt(&mut doc, encrypted, &StaticKey::new(key()))?;
        assert_eq!(
            outcome.octets().map(<[u8]>::to_vec),
            Some(b"<placeholder/>".to_vec())
        );
        assert!(outcome.nodes().is_empty());
        assert_eq!(doc.to_xml(), before, "the tree must not change");
        Ok(())
    }

    #[test]
    fn a_missing_encryption_method_is_reported() -> Result<()> {
        let doc = document(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"><xenc:CipherData><xenc:CipherValue>AA==</xenc:CipherValue></xenc:CipherData></xenc:EncryptedData>"#,
        )?;
        let mut doc = doc;
        let root = doc.document_element().expect("a root");
        assert!(matches!(
            decrypt(&mut doc, root, &StaticKey::new(key())),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    #[test]
    fn an_unknown_block_algorithm_is_reported() -> Result<()> {
        let mut doc = document(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"><xenc:EncryptionMethod Algorithm="urn:example:rot13"/><xenc:CipherData><xenc:CipherValue>AA==</xenc:CipherValue></xenc:CipherData></xenc:EncryptedData>"#,
        )?;
        let root = doc.document_element().expect("a root");
        assert!(matches!(
            decrypt(&mut doc, root, &StaticKey::new(key())),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
        Ok(())
    }

    #[test]
    fn a_contradictory_key_size_is_refused() -> Result<()> {
        let mut doc = document("<a><b>x</b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let encrypted = encrypt_element(
            &mut doc,
            target,
            &key(),
            &EncryptOptions::new(),
            &mut PseudoRandom::new(11),
        )?;
        // Downgrade the advertised key size; the decryptor must refuse rather than obey.
        let method = doc.children(encrypted).next().expect("the method element");
        let key_size = doc.children(method).next().expect("the KeySize element");
        doc.set_text_content(key_size, "128")?;
        assert!(matches!(
            decrypt(&mut doc, encrypted, &StaticKey::new(key())),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    #[test]
    fn key_size_can_be_suppressed() -> Result<()> {
        let mut doc = document("<a><b/></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let options = EncryptOptions::new().with_key_size(false);
        encrypt_element(
            &mut doc,
            target,
            &key(),
            &options,
            &mut PseudoRandom::new(12),
        )?;
        assert!(!doc.to_xml().contains("KeySize"), "{}", doc.to_xml());
        Ok(())
    }

    #[test]
    fn a_wrapped_key_travels_with_the_document() -> Result<()> {
        let kek = SymmetricKey::new(vec![0x71; 32]);
        let cek = SymmetricKey::generate(32, &mut PseudoRandom::new(13))?;
        let carried = wrap_key_for_key_info(KeyWrap::Aes256, &kek, &cek, Some("kek-2026"))?;

        let mut doc = document("<a><b>payload</b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let options = EncryptOptions::new().with_key_info(KeyInfo::with_encrypted_key(carried));
        let encrypted =
            encrypt_element(&mut doc, target, &cek, &options, &mut PseudoRandom::new(14))?;

        // The recipient knows only the key-encryption key.
        let store = KeyStore::new().with_named_key_encryption_key("kek-2026", kek);
        decrypt(&mut doc, encrypted, &store)?;
        assert_eq!(doc.to_xml(), "<a><b>payload</b></a>");
        Ok(())
    }

    /// The encryption-direction mirror of `a_wrapped_key_travels_with_the_document`: the key
    /// travels agreed under `ECDH-ES` instead of wrapped under a shared secret, and the recipient
    /// only ever holds its own EC private key.
    #[test]
    fn an_agreed_key_travels_with_the_document() -> Result<()> {
        use crate::crypto::digest::Digest;
        use crate::crypto::ecdh::EcPrivateKey;
        use oxixml_support::ec::CurveId;

        let mut rng = PseudoRandom::new(0xec5);
        let recipient_private = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        let cek = SymmetricKey::generate(32, &mut rng)?;
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(KeyWrap::Aes256.uri().as_bytes().to_vec());

        let carried = ecdh_es_wrap_key_for_key_info(
            KeyWrap::Aes256,
            &recipient_private.public_key(),
            &params,
            &cek,
            Some("bob"),
            &mut rng,
        )?;

        let mut doc = document("<a><b>payload</b></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let options = EncryptOptions::new().with_key_info(KeyInfo::with_encrypted_key(carried));
        let encrypted = encrypt_element(&mut doc, target, &cek, &options, &mut rng)?;

        // The recipient knows only its own EC private key.
        let store = KeyStore::new().with_named_key_encryption_key("bob", recipient_private);
        decrypt(&mut doc, encrypted, &store)?;
        assert_eq!(doc.to_xml(), "<a><b>payload</b></a>");
        Ok(())
    }

    #[test]
    fn options_are_reflected_in_the_output() -> Result<()> {
        let mut doc = document("<a><b/></a>")?;
        let root = doc.document_element().expect("a root");
        let target = doc.children(root).next().expect("a child");
        let mut options = EncryptOptions::new()
            .with_algorithm(BlockEncryption::Aes128Cbc)
            .with_id("ed-1")
            .with_key_info(KeyInfo::with_key_name("session"));
        options.mime_type = Some("text/xml".into());
        options.encoding = Some("urn:example:enc".into());
        encrypt_element(
            &mut doc,
            target,
            &SymmetricKey::new(vec![0u8; 16]),
            &options,
            &mut PseudoRandom::new(15),
        )?;
        let xml = doc.to_xml();
        assert!(xml.contains(r#"Id="ed-1""#), "{xml}");
        assert!(xml.contains(r#"MimeType="text/xml""#), "{xml}");
        assert!(xml.contains(r#"Encoding="urn:example:enc""#), "{xml}");
        assert!(xml.contains("aes128-cbc"), "{xml}");
        assert!(xml.contains("session"), "{xml}");
        Ok(())
    }

    #[test]
    fn decrypted_reports_its_shape() {
        let nodes = Decrypted::Nodes(vec![NodeId::from_index(3)]);
        assert_eq!(nodes.nodes().len(), 1);
        assert!(nodes.octets().is_none());
        let octets = Decrypted::Octets(vec![1, 2]);
        assert!(octets.nodes().is_empty());
        assert_eq!(octets.octets(), Some(&[1u8, 2][..]));
    }
}
