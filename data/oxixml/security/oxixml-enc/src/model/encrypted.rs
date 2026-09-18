//! `xenc:EncryptedData`, `xenc:EncryptedKey` and the `EncryptedType` they share.

use crate::algorithm::{TYPE_CONTENT, TYPE_ELEMENT, TYPE_ENCRYPTED_KEY};
use crate::model::cipher::CipherData;
use crate::model::keyinfo::{KeyInfo, ReferenceList};
use crate::model::method::EncryptionMethod;
use crate::model::properties::EncryptionProperties;

/// What the `Type` attribute of an `EncryptedData` says the plaintext is.
///
/// [XMLEnc §3.1](https://www.w3.org/TR/xmlenc-core1/#sec-EncryptedData) makes the attribute
/// optional, and its absence means "arbitrary octets": the decryptor returns the plaintext
/// without trying to parse it. The two values that matter to a document tree are
/// [`Element`](Self::Element) and [`Content`](Self::Content), and confusing them is a
/// correctness bug, not a stylistic one — see [`crate::api`].
///
/// # Example
///
/// ```
/// use oxixml_enc::model::ContentType;
///
/// assert_eq!(
///     ContentType::from_uri("http://www.w3.org/2001/04/xmlenc#Content"),
///     ContentType::Content,
/// );
/// assert_eq!(ContentType::Element.uri(), "http://www.w3.org/2001/04/xmlenc#Element");
/// assert!(ContentType::Element.is_xml());
/// assert!(!ContentType::from_uri("urn:example:jpeg").is_xml());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// `…xmlenc#Element`: the plaintext is a single serialized element, which replaces the
    /// `EncryptedData` element itself.
    Element,
    /// `…xmlenc#Content`: the plaintext is the *content* of an element — a possibly empty
    /// sequence of children — which replaces the `EncryptedData`'s position among the parent's
    /// children.
    Content,
    /// `…xmlenc#EncryptedKey`: the plaintext is key material. Used on an `EncryptedKey`.
    EncryptedKey,
    /// Any other URI: the plaintext is opaque octets of the named media type.
    Other(String),
}

impl ContentType {
    /// Returns the content type named by `uri`; an unrecognized URI becomes
    /// [`Other`](Self::Other).
    #[must_use]
    pub fn from_uri(uri: &str) -> Self {
        match uri {
            TYPE_ELEMENT => Self::Element,
            TYPE_CONTENT => Self::Content,
            TYPE_ENCRYPTED_KEY => Self::EncryptedKey,
            other => Self::Other(other.to_owned()),
        }
    }

    /// Returns the URI for the content type.
    #[must_use]
    pub fn uri(&self) -> &str {
        match self {
            Self::Element => TYPE_ELEMENT,
            Self::Content => TYPE_CONTENT,
            Self::EncryptedKey => TYPE_ENCRYPTED_KEY,
            Self::Other(uri) => uri,
        }
    }

    /// Returns `true` for the two types whose plaintext is XML that replaces nodes in a tree.
    #[must_use]
    pub const fn is_xml(&self) -> bool {
        matches!(self, Self::Element | Self::Content)
    }
}

/// The fields `EncryptedData` and `EncryptedKey` share, i.e. the abstract `xenc:EncryptedType`
/// of the schema.
///
/// The schema orders the children `EncryptionMethod?`, `ds:KeyInfo?`, `CipherData`,
/// `EncryptionProperties?`, and this crate always emits them in that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedType {
    /// The optional `Id` attribute, which same-document references point at.
    pub id: Option<String>,
    /// The optional `Type` attribute.
    pub content_type: Option<ContentType>,
    /// The optional `MimeType` attribute, advisory only.
    pub mime_type: Option<String>,
    /// The optional `Encoding` attribute, advisory only.
    pub encoding: Option<String>,
    /// The optional `EncryptionMethod` child. Absent means the algorithm is known out of band.
    pub encryption_method: Option<EncryptionMethod>,
    /// The optional `ds:KeyInfo` child.
    pub key_info: Option<KeyInfo>,
    /// The required `CipherData` child.
    pub cipher_data: CipherData,
    /// The optional `EncryptionProperties` child.
    pub encryption_properties: Option<EncryptionProperties>,
}

impl EncryptedType {
    /// Creates a structure carrying `ciphertext` inline and nothing else.
    #[must_use]
    pub fn from_ciphertext(ciphertext: Vec<u8>) -> Self {
        Self {
            id: None,
            content_type: None,
            mime_type: None,
            encoding: None,
            encryption_method: None,
            key_info: None,
            cipher_data: CipherData::Value(ciphertext),
            encryption_properties: None,
        }
    }

    /// Returns the inline ciphertext, or an error naming the unresolved reference.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::CipherReferenceNotResolved`](crate::EncError::CipherReferenceNotResolved)
    /// when the `CipherData` is a `CipherReference`; see [`crate::model::CipherReference`] for
    /// why this crate does not dereference it.
    pub fn ciphertext(&self) -> crate::Result<&[u8]> {
        match &self.cipher_data {
            CipherData::Value(octets) => Ok(octets),
            CipherData::Reference(reference) => Err(crate::EncError::CipherReferenceNotResolved(
                reference.uri.clone(),
            )),
        }
    }
}

/// The `xenc:EncryptedData` element of
/// [XMLEnc §3.1](https://www.w3.org/TR/xmlenc-core1/#sec-EncryptedData).
///
/// This is the element that replaces the plaintext in the document tree.
///
/// # Example
///
/// ```
/// use oxixml_enc::algorithm::BlockEncryption;
/// use oxixml_enc::model::{ContentType, EncryptedData, EncryptionMethod, KeyInfo};
/// use oxixml_dom::Document;
///
/// let mut data = EncryptedData::new(vec![0xde, 0xad, 0xbe, 0xef]);
/// data.common.content_type = Some(ContentType::Element);
/// data.common.encryption_method = Some(EncryptionMethod::block(BlockEncryption::Aes128Cbc));
/// data.common.key_info = Some(KeyInfo::with_key_name("session"));
///
/// let mut document = Document::new();
/// let node = data.write_into(&mut document, None)?;
/// let xml = document.node_to_xml(node, &Default::default());
/// assert!(xml.contains("<xenc:CipherValue>3q2+7w==</xenc:CipherValue>"), "{xml}");
///
/// // …and back again.
/// assert_eq!(EncryptedData::parse(&document, node)?, data);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedData {
    /// The attributes and children shared with `EncryptedKey`.
    pub common: EncryptedType,
}

impl EncryptedData {
    /// The element's local name.
    pub const LOCAL_NAME: &'static str = "EncryptedData";

    /// Creates an `EncryptedData` carrying `ciphertext` inline.
    #[must_use]
    pub fn new(ciphertext: Vec<u8>) -> Self {
        Self {
            common: EncryptedType::from_ciphertext(ciphertext),
        }
    }

    /// Returns the inline ciphertext.
    ///
    /// # Errors
    ///
    /// See [`EncryptedType::ciphertext`].
    pub fn ciphertext(&self) -> crate::Result<&[u8]> {
        self.common.ciphertext()
    }

    /// Returns the `Type` attribute, or [`ContentType::Other`] with an empty URI when it is
    /// absent — callers that care about the distinction should read
    /// [`self.common.content_type`](EncryptedType::content_type) directly.
    #[must_use]
    pub fn content_type(&self) -> Option<&ContentType> {
        self.common.content_type.as_ref()
    }
}

/// The `xenc:EncryptedKey` element of
/// [XMLEnc §3.5.1](https://www.w3.org/TR/xmlenc-core1/#sec-EncryptedKey).
///
/// An `EncryptedKey` is an `EncryptedType` whose plaintext is a symmetric key, plus three extra
/// pieces: the `Recipient` attribute, a `ReferenceList` naming what the key decrypts, and a
/// `CarriedKeyName` that lets several `EncryptedKey` elements for different recipients say they
/// carry the *same* key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedKey {
    /// The attributes and children shared with `EncryptedData`.
    pub common: EncryptedType,
    /// The optional `Recipient` attribute: a hint about who this copy of the key is for.
    pub recipient: Option<String>,
    /// The optional `ReferenceList` child.
    pub reference_list: Option<ReferenceList>,
    /// The optional `CarriedKeyName` child.
    pub carried_key_name: Option<String>,
}

impl EncryptedKey {
    /// The element's local name.
    pub const LOCAL_NAME: &'static str = "EncryptedKey";

    /// Creates an `EncryptedKey` carrying `ciphertext` inline.
    #[must_use]
    pub fn new(ciphertext: Vec<u8>) -> Self {
        Self {
            common: EncryptedType::from_ciphertext(ciphertext),
            recipient: None,
            reference_list: None,
            carried_key_name: None,
        }
    }

    /// Returns the inline ciphertext, i.e. the wrapped key material.
    ///
    /// # Errors
    ///
    /// See [`EncryptedType::ciphertext`].
    pub fn ciphertext(&self) -> crate::Result<&[u8]> {
        self.common.ciphertext()
    }

    /// Sets the `CarriedKeyName` and returns the value, for chaining.
    #[must_use]
    pub fn with_carried_key_name(mut self, name: impl Into<String>) -> Self {
        self.carried_key_name = Some(name.into());
        self
    }

    /// Sets the `Recipient` and returns the value, for chaining.
    #[must_use]
    pub fn with_recipient(mut self, recipient: impl Into<String>) -> Self {
        self.recipient = Some(recipient.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cipher::CipherReference;

    #[test]
    fn content_types_round_trip_through_their_uris() {
        for content_type in [
            ContentType::Element,
            ContentType::Content,
            ContentType::EncryptedKey,
            ContentType::Other("urn:example:png".into()),
        ] {
            assert_eq!(ContentType::from_uri(content_type.uri()), content_type);
        }
    }

    #[test]
    fn only_element_and_content_are_xml() {
        assert!(ContentType::Element.is_xml());
        assert!(ContentType::Content.is_xml());
        assert!(!ContentType::EncryptedKey.is_xml());
        assert!(!ContentType::Other("urn:x".into()).is_xml());
    }

    #[test]
    fn a_cipher_reference_is_not_a_ciphertext() {
        let mut data = EncryptedData::new(vec![1, 2, 3]);
        assert_eq!(data.ciphertext().expect("inline"), &[1, 2, 3]);
        data.common.cipher_data = CipherData::Reference(CipherReference::new("http://x/y"));
        assert!(matches!(
            data.ciphertext(),
            Err(crate::EncError::CipherReferenceNotResolved(uri)) if uri == "http://x/y"
        ));
    }

    #[test]
    fn builders_set_the_encrypted_key_extras() {
        let key = EncryptedKey::new(vec![0])
            .with_recipient("alice@example.org")
            .with_carried_key_name("session-2026");
        assert_eq!(key.recipient.as_deref(), Some("alice@example.org"));
        assert_eq!(key.carried_key_name.as_deref(), Some("session-2026"));
        assert!(key.reference_list.is_none());
        assert_eq!(key.ciphertext().expect("inline"), &[0]);
    }

    #[test]
    fn the_content_type_accessor_mirrors_the_field() {
        let mut data = EncryptedData::new(Vec::new());
        assert_eq!(data.content_type(), None);
        data.common.content_type = Some(ContentType::Content);
        assert_eq!(data.content_type(), Some(&ContentType::Content));
    }
}
