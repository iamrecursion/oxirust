//! Keys, and how a decryptor finds the right one.
//!
//! # Where the key comes from
//!
//! Decrypting an `EncryptedData` needs a *content-encryption key*, and XML Encryption offers
//! several ways for the document to say which one:
//!
//! * nothing at all — the recipient is expected to know, out of band;
//! * a `ds:KeyName`, naming a key the recipient already holds;
//! * an `xenc:EncryptedKey` carrying the content-encryption key itself, protected either by a
//!   shared *key-encryption key* (AES Key Wrap) or by the recipient's public key (RSA);
//! * a `ds:RetrievalMethod` pointing at an `EncryptedKey` elsewhere in the same document.
//!
//! [`KeyResolver`] is the hook for all of that, and [`KeyStore`] is an implementation that
//! handles every case above, recursively — an `EncryptedKey` may itself have a `KeyInfo` naming
//! the key-encryption key that protects it.
//!
//! # Trust
//!
//! A `KeyInfo` is **attacker-controlled data**. [`KeyStore`] therefore never *learns* a key from
//! a document: it only ever looks one up among the keys the application registered. A
//! `ds:KeyName` that is not in the store is [`EncError::KeyNotFound`], never a fallback to some
//! other key. A `ds:RetrievalMethod` is followed only when it is a same-document reference
//! (`URI="#id"`), because following an external one would fetch attacker-chosen URLs.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::key::{KeyStore, SymmetricKey};
//!
//! let store = KeyStore::new()
//!     .with_named_key("session-2026", SymmetricKey::new(vec![0x11; 32]))
//!     .with_named_key_encryption_key("kek", SymmetricKey::new(vec![0x22; 32]));
//! assert!(store.named_key("session-2026").is_some());
//! assert!(store.named_key("nope").is_none());
//! ```

use std::collections::BTreeMap;
use std::hint::black_box;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::algorithm::{KeyWrap, XMLENC_NS};
use crate::cipher;
use crate::crypto::dh::DhPrivateKey;
use crate::crypto::digest::Digest;
use crate::crypto::ecdh::EcPrivateKey;
use crate::crypto::rsa::RsaPrivateKey;
use crate::error::{EncError, Result};
use crate::model::{AgreementMethod, EncryptedKey, EncryptionMethod, KeyInfo, KeyInfoItem};
use crate::random::RandomSource;

/// Symmetric key material.
///
/// The octets are zeroed when the value is dropped and never appear in its [`Debug`] output.
/// Both are hygiene, not guarantees: a `SymmetricKey` that was moved leaves its old bytes
/// behind, and nothing stops the allocator or the optimizer from keeping a copy.
#[derive(Clone, PartialEq, Eq)]
pub struct SymmetricKey {
    octets: Vec<u8>,
}

impl core::fmt::Debug for SymmetricKey {
    /// Prints the length and nothing else.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SymmetricKey")
            .field("len", &self.octets.len())
            .finish_non_exhaustive()
    }
}

impl Drop for SymmetricKey {
    fn drop(&mut self) {
        self.octets.fill(0);
        let _ = black_box(&self.octets);
    }
}

impl SymmetricKey {
    /// Takes ownership of `octets` as key material.
    #[must_use]
    pub fn new(octets: Vec<u8>) -> Self {
        Self { octets }
    }

    /// Draws `len` octets from `rng` as a fresh key.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Entropy`] when the random source fails.
    pub fn generate(len: usize, rng: &mut dyn RandomSource) -> Result<Self> {
        let mut octets = vec![0u8; len];
        rng.fill(&mut octets)?;
        Ok(Self { octets })
    }

    /// Returns the key material.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.octets
    }

    /// Returns the key length in octets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.octets.len()
    }

    /// Returns `true` when the key is empty, which no algorithm accepts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.octets.is_empty()
    }
}

/// The information a [`KeyResolver`] gets to work with.
#[derive(Debug, Clone, Copy)]
pub struct KeyContext<'a> {
    /// The document the structure lives in, so that same-document references can be followed.
    pub document: &'a Document,
    /// The `EncryptionMethod` of the `EncryptedData` or `EncryptedKey` being decrypted.
    pub encryption_method: Option<&'a EncryptionMethod>,
    /// The `ds:KeyInfo` of the structure being decrypted.
    pub key_info: Option<&'a KeyInfo>,
    /// The key length the algorithm needs, when it is known.
    pub expected_key_len: Option<usize>,
}

/// Finds the content-encryption key for a structure being decrypted.
pub trait KeyResolver {
    /// Returns the key to decrypt with.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::KeyNotFound`] when no key is available, and any error the
    /// implementation hits while unwrapping one.
    fn resolve(&self, context: &KeyContext<'_>) -> Result<SymmetricKey>;
}

impl<T: KeyResolver + ?Sized> KeyResolver for &T {
    fn resolve(&self, context: &KeyContext<'_>) -> Result<SymmetricKey> {
        (**self).resolve(context)
    }
}

/// A resolver that always answers with the same key.
///
/// This is the right choice when the key is agreed out of band and the document's `KeyInfo`, if
/// any, is not consulted.
#[derive(Debug, Clone)]
pub struct StaticKey {
    key: SymmetricKey,
}

impl StaticKey {
    /// Wraps `key` as a resolver.
    #[must_use]
    pub fn new(key: SymmetricKey) -> Self {
        Self { key }
    }
}

impl KeyResolver for StaticKey {
    fn resolve(&self, _context: &KeyContext<'_>) -> Result<SymmetricKey> {
        Ok(self.key.clone())
    }
}

/// A private key that can open an `EncryptedKey`.
#[derive(Debug, Clone)]
pub enum KeyDecryptionKey {
    /// A shared key-encryption key, for the `kw-aes*` algorithms.
    Symmetric(SymmetricKey),
    /// An RSA private key, for the `rsa-*` key-transport algorithms.
    Rsa(Box<RsaPrivateKey>),
    /// An elliptic-curve private key, for the `ECDH-ES` key-agreement method.
    Ec(Box<EcPrivateKey>),
    /// A finite-field Diffie–Hellman private key, for the `dh` and `dh-es` key-agreement
    /// methods.
    Dh(Box<DhPrivateKey>),
}

impl From<SymmetricKey> for KeyDecryptionKey {
    fn from(key: SymmetricKey) -> Self {
        Self::Symmetric(key)
    }
}

impl From<RsaPrivateKey> for KeyDecryptionKey {
    fn from(key: RsaPrivateKey) -> Self {
        Self::Rsa(Box::new(key))
    }
}

impl From<EcPrivateKey> for KeyDecryptionKey {
    fn from(key: EcPrivateKey) -> Self {
        Self::Ec(Box::new(key))
    }
}

impl From<DhPrivateKey> for KeyDecryptionKey {
    fn from(key: DhPrivateKey) -> Self {
        Self::Dh(Box::new(key))
    }
}

/// A registry of the keys an application holds, and the resolution rules of XMLEnc §3.5.
///
/// See the [module documentation](self) for the trust model. In short: the store answers only
/// with keys that were registered, and never derives one from the document.
#[derive(Debug, Clone, Default)]
pub struct KeyStore {
    /// Content-encryption keys, by `ds:KeyName`.
    named_keys: BTreeMap<String, SymmetricKey>,
    /// Key-decryption keys, by the `ds:KeyName` inside an `EncryptedKey`'s own `KeyInfo`.
    named_key_encryption_keys: BTreeMap<String, KeyDecryptionKey>,
    /// The content-encryption key to use when the document names none.
    default_key: Option<SymmetricKey>,
    /// The key-decryption key to use when an `EncryptedKey` names none.
    default_key_encryption_key: Option<KeyDecryptionKey>,
}

impl KeyStore {
    /// Returns an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a content-encryption key under a `ds:KeyName`.
    #[must_use]
    pub fn with_named_key(mut self, name: impl Into<String>, key: SymmetricKey) -> Self {
        self.named_keys.insert(name.into(), key);
        self
    }

    /// Registers a key-decryption key under a `ds:KeyName`.
    #[must_use]
    pub fn with_named_key_encryption_key(
        mut self,
        name: impl Into<String>,
        key: impl Into<KeyDecryptionKey>,
    ) -> Self {
        self.named_key_encryption_keys
            .insert(name.into(), key.into());
        self
    }

    /// Registers the content-encryption key to use when the document names none.
    #[must_use]
    pub fn with_default_key(mut self, key: SymmetricKey) -> Self {
        self.default_key = Some(key);
        self
    }

    /// Registers the key-decryption key to use when an `EncryptedKey` names none.
    #[must_use]
    pub fn with_default_key_encryption_key(mut self, key: impl Into<KeyDecryptionKey>) -> Self {
        self.default_key_encryption_key = Some(key.into());
        self
    }

    /// Returns a registered content-encryption key by name.
    #[must_use]
    pub fn named_key(&self, name: &str) -> Option<&SymmetricKey> {
        self.named_keys.get(name)
    }

    /// Returns a registered key-decryption key by name.
    #[must_use]
    pub fn named_key_encryption_key(&self, name: &str) -> Option<&KeyDecryptionKey> {
        self.named_key_encryption_keys.get(name)
    }

    /// Opens an `EncryptedKey` and returns the key material it carries.
    ///
    /// When the `EncryptedKey`'s own `ds:KeyInfo` is an `xenc:AgreementMethod`, the
    /// key-encryption key is *derived* rather than looked up: the originator's public value
    /// comes from the `AgreementMethod`'s `OriginatorKeyInfo`, the recipient's private key from
    /// this store (found by the same `ds:KeyName`-then-`Recipient`-then-default precedence used
    /// for any other key-decryption key, but against the `AgreementMethod`'s `RecipientKeyInfo`
    /// instead of the `EncryptedKey`'s own `KeyInfo`), and the two combine through the
    /// key-agreement math of [`crate::agreement`] to produce the key that unwraps the ciphertext
    /// exactly as a directly registered one would.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidStructure`] when the `EncryptedKey` has no
    /// `EncryptionMethod`, or an `AgreementMethod` is missing a part it needs (an
    /// `OriginatorKeyInfo`, a `KeyDerivationMethod` for `dh-es`/`ECDH-ES`, and so on);
    /// [`EncError::UnsupportedAlgorithm`] when it names an algorithm, curve, or key derivation
    /// this crate does not implement; [`EncError::KeyNotFound`] when the key that protects it is
    /// not registered, or is registered under the wrong type (a symmetric key where an EC
    /// private key is needed, and so on); and [`EncError::DecryptionFailed`] when it does not
    /// open.
    pub fn open_encrypted_key(&self, key: &EncryptedKey) -> Result<SymmetricKey> {
        let method = key.common.encryption_method.as_ref().ok_or_else(|| {
            EncError::InvalidStructure(
                "an `xenc:EncryptedKey` needs an `EncryptionMethod` to be opened".to_owned(),
            )
        })?;
        let ciphertext = key.common.ciphertext()?;

        if let Some(agreement) = key
            .common
            .key_info
            .as_ref()
            .and_then(KeyInfo::agreement_method)
        {
            let wrap = method
                .key_wrap_algorithm()
                .ok_or_else(|| EncError::UnsupportedAlgorithm(method.algorithm.clone()))?;
            let kek = self.agreement_key_encryption_key(agreement, key, wrap)?;
            return cipher::unwrap_key(wrap, &kek, ciphertext);
        }

        let protector = self.key_decryption_key(key)?;

        if let Some(algorithm) = method.key_wrap_algorithm() {
            let KeyDecryptionKey::Symmetric(kek) = &protector else {
                return Err(EncError::KeyNotFound(format!(
                    "`{}` needs a symmetric key-encryption key",
                    method.algorithm
                )));
            };
            return cipher::unwrap_key(algorithm, kek, ciphertext);
        }

        if let Some(algorithm) = method.key_transport_algorithm() {
            let KeyDecryptionKey::Rsa(private) = &protector else {
                return Err(EncError::KeyNotFound(format!(
                    "`{}` needs an RSA private key",
                    method.algorithm
                )));
            };
            let params = method.oaep_parameters()?;
            return cipher::untransport_key(algorithm, private, ciphertext, &params);
        }

        Err(EncError::UnsupportedAlgorithm(method.algorithm.clone()))
    }

    /// Picks the key that protects `key`, from its own `ds:KeyName` or from the default.
    fn key_decryption_key(&self, key: &EncryptedKey) -> Result<KeyDecryptionKey> {
        let name = key.common.key_info.as_ref().and_then(KeyInfo::key_name);
        self.resolve_key_decryption_key(name, key.recipient.as_deref())
    }

    /// Picks the key-decryption key named by `name`, falling back to `recipient` and then to the
    /// store's default — the precedence [`Self::key_decryption_key`] uses for an `EncryptedKey`'s
    /// own `KeyInfo`, generalized so [`Self::agreement_key_encryption_key`] can apply the same
    /// precedence to an `AgreementMethod`'s `RecipientKeyInfo`.
    fn resolve_key_decryption_key(
        &self,
        name: Option<&str>,
        recipient: Option<&str>,
    ) -> Result<KeyDecryptionKey> {
        if let Some(name) = name {
            return self
                .named_key_encryption_keys
                .get(name)
                .cloned()
                .ok_or_else(|| EncError::KeyNotFound(format!("the key named `{name}`")));
        }
        if let Some(recipient) = recipient {
            if let Some(found) = self.named_key_encryption_keys.get(recipient) {
                return Ok(found.clone());
            }
        }
        self.default_key_encryption_key
            .clone()
            .ok_or_else(|| EncError::KeyNotFound("an unnamed key-encryption key".to_owned()))
    }

    /// Derives the key-encryption key an `xenc:AgreementMethod` describes: the originator's
    /// public value comes from the document, the recipient's private key from this store, and
    /// `wrap` fixes the length the derivation must produce — the length of the key-wrap key that
    /// unwraps the `EncryptedKey`'s ciphertext, per XMLEnc §5.6.
    ///
    /// Every check that depends only on the document — the algorithm, curve and key derivation
    /// it names, and the shape of its `OriginatorKeyInfo` — runs before the recipient's private
    /// key is looked up. A malformed or unsupported `AgreementMethod` is therefore always
    /// diagnosed the same way regardless of what happens to be registered in this store, rather
    /// than a document defect being masked by an unrelated missing-key one or vice versa.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::UnsupportedAlgorithm`] when the `AgreementMethod`, its curve, or its
    /// key derivation is not one this crate implements; [`EncError::InvalidStructure`] when a
    /// required part is missing, or when a `dh` agreement contradicts itself by also carrying a
    /// `KeyDerivationMethod` (see
    /// [`KeyAgreement::uses_explicit_key_derivation`](crate::algorithm::KeyAgreement::uses_explicit_key_derivation));
    /// [`EncError::KeyNotFound`] when the recipient's private key is not registered, or is
    /// registered as the wrong key type; and the errors of the key-agreement math itself.
    fn agreement_key_encryption_key(
        &self,
        agreement: &AgreementMethod,
        encrypted_key: &EncryptedKey,
        wrap: KeyWrap,
    ) -> Result<SymmetricKey> {
        let algorithm = agreement.key_agreement()?;
        let originator = agreement.originator_key_info.as_ref().ok_or_else(|| {
            EncError::InvalidStructure(
                "an `xenc:AgreementMethod` has no `OriginatorKeyInfo`".to_owned(),
            )
        })?;

        if algorithm.is_elliptic() {
            let peer = originator
                .ec_key_value()
                .ok_or_else(|| {
                    EncError::InvalidStructure(
                        "an `ECDH-ES` `OriginatorKeyInfo` has no `ECKeyValue`".to_owned(),
                    )
                })?
                .public_key()?;
            let params = agreement
                .key_derivation_method
                .as_ref()
                .ok_or_else(|| {
                    EncError::InvalidStructure(
                        "an `ECDH-ES` `AgreementMethod` has no `KeyDerivationMethod`".to_owned(),
                    )
                })?
                .concat_kdf_params()?;
            let recipient_private = self.agreement_recipient_ec_key(agreement, encrypted_key)?;
            return crate::agreement::ecdh_es_key_encryption_key(
                &recipient_private,
                &peer,
                &params,
                wrap.key_len(),
            );
        }

        let peer = originator
            .dh_key_value()
            .ok_or_else(|| {
                EncError::InvalidStructure(format!(
                    "a `{}` `OriginatorKeyInfo` has no `xenc:DHKeyValue`",
                    agreement.algorithm
                ))
            })?
            .public_key()?;

        if algorithm.uses_explicit_key_derivation() {
            let params = agreement
                .key_derivation_method
                .as_ref()
                .ok_or_else(|| {
                    EncError::InvalidStructure(format!(
                        "a `{}` `AgreementMethod` has no `KeyDerivationMethod`",
                        agreement.algorithm
                    ))
                })?
                .concat_kdf_params()?;
            let recipient_private = self.agreement_recipient_dh_key(agreement, encrypted_key)?;
            crate::agreement::dh_es_key_encryption_key(
                &recipient_private,
                &peer,
                &params,
                wrap.key_len(),
            )
        } else {
            if agreement.key_derivation_method.is_some() {
                return Err(EncError::InvalidStructure(
                    "a `dh` `AgreementMethod` must not carry a `KeyDerivationMethod`".to_owned(),
                ));
            }
            let digest = match &agreement.digest_method {
                Some(uri) => Digest::from_uri(uri)
                    .ok_or_else(|| EncError::UnsupportedAlgorithm(uri.clone()))?,
                None => Digest::default(),
            };
            let ka_nonce = agreement.ka_nonce.as_deref().unwrap_or(&[]);
            let recipient_private = self.agreement_recipient_dh_key(agreement, encrypted_key)?;
            crate::agreement::dh_key_encryption_key(
                &recipient_private,
                &peer,
                digest,
                wrap.uri(),
                ka_nonce,
                wrap.key_len(),
            )
        }
    }

    /// Looks up the recipient's EC private key for `agreement`, by the same
    /// `ds:KeyName`-then-`Recipient`-then-default precedence [`Self::key_decryption_key`] uses,
    /// applied to the `AgreementMethod`'s own `RecipientKeyInfo` instead of the `EncryptedKey`'s.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::KeyNotFound`] when nothing is registered under the resolved name, or
    /// when what is registered there is not an EC private key.
    fn agreement_recipient_ec_key(
        &self,
        agreement: &AgreementMethod,
        encrypted_key: &EncryptedKey,
    ) -> Result<EcPrivateKey> {
        let name = agreement
            .recipient_key_info
            .as_ref()
            .and_then(KeyInfo::key_name);
        match self.resolve_key_decryption_key(name, encrypted_key.recipient.as_deref())? {
            KeyDecryptionKey::Ec(key) => Ok(*key),
            _ => Err(EncError::KeyNotFound(format!(
                "`{}` needs an EC private key",
                agreement.algorithm
            ))),
        }
    }

    /// The [`Self::agreement_recipient_ec_key`] of the finite-field methods.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::KeyNotFound`] when nothing is registered under the resolved name, or
    /// when what is registered there is not a Diffie–Hellman private key.
    fn agreement_recipient_dh_key(
        &self,
        agreement: &AgreementMethod,
        encrypted_key: &EncryptedKey,
    ) -> Result<DhPrivateKey> {
        let name = agreement
            .recipient_key_info
            .as_ref()
            .and_then(KeyInfo::key_name);
        match self.resolve_key_decryption_key(name, encrypted_key.recipient.as_deref())? {
            KeyDecryptionKey::Dh(key) => Ok(*key),
            _ => Err(EncError::KeyNotFound(format!(
                "`{}` needs a Diffie-Hellman private key",
                agreement.algorithm
            ))),
        }
    }
}

impl KeyResolver for KeyStore {
    fn resolve(&self, context: &KeyContext<'_>) -> Result<SymmetricKey> {
        let Some(key_info) = context.key_info else {
            return self.default_key.clone().ok_or_else(|| {
                EncError::KeyNotFound("a structure with no `ds:KeyInfo`".to_owned())
            });
        };

        // The order matters: a carried key beats a name, because it is more specific, and a
        // retrieval method is followed only when nothing more direct is present.
        for item in &key_info.items {
            if let KeyInfoItem::EncryptedKey(encrypted) = item {
                return self.open_encrypted_key(encrypted);
            }
        }
        for item in &key_info.items {
            if let KeyInfoItem::KeyName(name) = item {
                return self
                    .named_keys
                    .get(name)
                    .cloned()
                    .ok_or_else(|| EncError::KeyNotFound(format!("the key named `{name}`")));
            }
        }
        for item in &key_info.items {
            if let KeyInfoItem::RetrievalMethod { uri, .. } = item {
                let encrypted = resolve_same_document_encrypted_key(context.document, uri)?;
                return self.open_encrypted_key(&encrypted);
            }
        }

        self.default_key
            .clone()
            .ok_or_else(|| EncError::KeyNotFound("a `ds:KeyInfo` with no usable child".to_owned()))
    }
}

/// Follows a same-document `ds:RetrievalMethod` to an `xenc:EncryptedKey`.
///
/// Only a bare fragment reference (`#id`) is followed. Anything else — an absolute URL, a
/// relative path, a fragment with an XPointer expression — is rejected rather than fetched; see
/// the [module documentation](self).
fn resolve_same_document_encrypted_key(doc: &Document, uri: &str) -> Result<EncryptedKey> {
    let Some(id) = uri.strip_prefix('#') else {
        return Err(EncError::KeyNotFound(format!(
            "the retrieval method `{uri}`, which is not a same-document reference"
        )));
    };
    if id.is_empty() || id.contains(['(', ')', '/']) {
        return Err(EncError::KeyNotFound(format!(
            "the retrieval method `{uri}`, whose fragment is not a bare identifier"
        )));
    }

    // `get_element_by_id` only knows about attributes a DTD or `xml:id` declared as IDs, which a
    // non-validating parse of an XML Encryption document has not seen. Fall back to scanning
    // for an `Id` attribute, which the XML Encryption schema declares as type `ID`.
    //
    // The scan skips detached nodes. The arena keeps a removed subtree alive so that its
    // identifiers stay valid, and an `EncryptedKey` that a previous decryption detached is no
    // longer part of the document — resolving a reference to one would let a stale structure
    // answer for a live one.
    let found = doc
        .get_element_by_id(id)
        .filter(|node| is_encrypted_key(doc, *node))
        .or_else(|| {
            doc.nodes_in_document_order().into_iter().find(|node| {
                doc.tree_root(*node) == Some(doc.root())
                    && is_encrypted_key(doc, *node)
                    && doc.get_attribute(*node, "Id") == Some(id)
            })
        });

    let node = found.ok_or_else(|| {
        EncError::KeyNotFound(format!("an `xenc:EncryptedKey` with `Id=\"{id}\"`"))
    })?;
    EncryptedKey::parse(doc, node)
}

/// Returns `true` when `node` is an `xenc:EncryptedKey` element.
fn is_encrypted_key(doc: &Document, node: NodeId) -> bool {
    doc.kind(node) == Some(NodeKind::Element)
        && doc.element(node).is_some_and(|data| {
            data.namespace_uri() == Some(XMLENC_NS) && data.local_name() == EncryptedKey::LOCAL_NAME
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::{KeyAgreement, KeyWrap};
    use crate::crypto::dh::{DhParameters, DhPrivateKey};
    use crate::crypto::ecdh::EcPrivateKey;
    use crate::crypto::kdf::ConcatKdfParams;
    use crate::model::{
        AgreementMethod, DhKeyValue, EcKeyValue, EncryptionMethod, KeyDerivationMethod,
    };
    use crate::random::{FixedRandom, PseudoRandom};
    use oxixml_support::ec::CurveId;

    fn empty_document() -> Document {
        Document::new()
    }

    fn context<'a>(doc: &'a Document, key_info: Option<&'a KeyInfo>) -> KeyContext<'a> {
        KeyContext {
            document: doc,
            encryption_method: None,
            key_info,
            expected_key_len: None,
        }
    }

    #[test]
    fn a_generated_key_has_the_requested_length() -> Result<()> {
        let key = SymmetricKey::generate(24, &mut FixedRandom::new(&[0x7f]))?;
        assert_eq!(key.len(), 24);
        assert!(!key.is_empty());
        assert_eq!(key.as_bytes(), &[0x7f; 24]);
        Ok(())
    }

    #[test]
    fn debug_does_not_leak_key_material() {
        let key = SymmetricKey::new(vec![0xab; 16]);
        let rendered = format!("{key:?}");
        assert!(rendered.contains("16"), "{rendered}");
        assert!(!rendered.contains("171"), "{rendered}");
        assert!(!rendered.contains("ab"), "{rendered}");
    }

    #[test]
    fn a_static_resolver_ignores_the_key_info() -> Result<()> {
        let doc = empty_document();
        let resolver = StaticKey::new(SymmetricKey::new(vec![1, 2, 3]));
        let info = KeyInfo::with_key_name("irrelevant");
        assert_eq!(
            resolver.resolve(&context(&doc, Some(&info)))?.as_bytes(),
            &[1, 2, 3]
        );
        Ok(())
    }

    #[test]
    fn a_named_key_is_found_and_an_unknown_one_is_not() -> Result<()> {
        let doc = empty_document();
        let store = KeyStore::new().with_named_key("k", SymmetricKey::new(vec![9; 16]));
        let found = KeyInfo::with_key_name("k");
        assert_eq!(store.resolve(&context(&doc, Some(&found)))?.len(), 16);

        let missing = KeyInfo::with_key_name("other");
        assert!(matches!(
            store.resolve(&context(&doc, Some(&missing))),
            Err(EncError::KeyNotFound(_))
        ));
        Ok(())
    }

    #[test]
    fn an_absent_key_info_falls_back_to_the_default() -> Result<()> {
        let doc = empty_document();
        let store = KeyStore::new().with_default_key(SymmetricKey::new(vec![4; 32]));
        assert_eq!(store.resolve(&context(&doc, None))?.len(), 32);

        let empty = KeyStore::new();
        assert!(matches!(
            empty.resolve(&context(&doc, None)),
            Err(EncError::KeyNotFound(_))
        ));
        Ok(())
    }

    #[test]
    fn a_wrapped_key_is_unwrapped_with_the_named_kek() -> Result<()> {
        let kek = SymmetricKey::new(vec![0x31; 32]);
        let cek = SymmetricKey::new(vec![0x42; 16]);
        let wrapped = cipher::wrap_key(KeyWrap::Aes256, &kek, &cek)?;

        let mut encrypted = EncryptedKey::new(wrapped);
        encrypted.common.encryption_method = Some(EncryptionMethod::key_wrap(KeyWrap::Aes256));
        encrypted.common.key_info = Some(KeyInfo::with_key_name("kek"));

        let store = KeyStore::new().with_named_key_encryption_key("kek", kek);
        assert_eq!(
            store.open_encrypted_key(&encrypted)?.as_bytes(),
            &[0x42; 16]
        );

        // Without the key-encryption key registered, the lookup fails rather than guessing.
        assert!(matches!(
            KeyStore::new().open_encrypted_key(&encrypted),
            Err(EncError::KeyNotFound(_))
        ));
        Ok(())
    }

    #[test]
    fn an_encrypted_key_without_a_method_cannot_be_opened() {
        let encrypted = EncryptedKey::new(vec![0; 24]);
        let store =
            KeyStore::new().with_default_key_encryption_key(SymmetricKey::new(vec![0x31; 32]));
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn an_unknown_key_protection_algorithm_is_reported() {
        let mut encrypted = EncryptedKey::new(vec![0; 24]);
        encrypted.common.encryption_method = Some(EncryptionMethod::new("urn:example:wrap"));
        let store =
            KeyStore::new().with_default_key_encryption_key(SymmetricKey::new(vec![0x31; 32]));
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn a_symmetric_key_is_refused_where_rsa_is_required() {
        use crate::algorithm::KeyTransport;
        let mut encrypted = EncryptedKey::new(vec![0; 128]);
        encrypted.common.encryption_method =
            Some(EncryptionMethod::key_transport(KeyTransport::RsaOaep));
        let store =
            KeyStore::new().with_default_key_encryption_key(SymmetricKey::new(vec![0x31; 32]));
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::KeyNotFound(_))
        ));
    }

    #[test]
    fn only_same_document_retrieval_methods_are_followed() {
        let doc = empty_document();
        for uri in [
            "http://example.org/keys#k",
            "keys.xml#k",
            "#",
            "#xpointer(id('k'))",
        ] {
            assert!(
                matches!(
                    resolve_same_document_encrypted_key(&doc, uri),
                    Err(EncError::KeyNotFound(_))
                ),
                "`{uri}` should not be followed"
            );
        }
    }

    #[test]
    fn a_retrieval_method_finds_the_encrypted_key_by_id() -> Result<()> {
        let kek = SymmetricKey::new(vec![0x10; 16]);
        let cek = SymmetricKey::new(vec![0x20; 16]);
        let wrapped = cipher::wrap_key(KeyWrap::Aes128, &kek, &cek)?;
        let mut carried = EncryptedKey::new(wrapped);
        carried.common.id = Some("ek1".into());
        carried.common.encryption_method = Some(EncryptionMethod::key_wrap(KeyWrap::Aes128));

        let mut doc = oxixml_dom::build::parse_str("<root/>")?;
        let root = doc.document_element().expect("a root");
        carried.write_into(&mut doc, Some(root))?;

        let store = KeyStore::new().with_default_key_encryption_key(kek);
        let info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::RetrievalMethod {
                uri: "#ek1".into(),
                type_uri: None,
            }],
        };
        assert_eq!(
            store.resolve(&context(&doc, Some(&info)))?.as_bytes(),
            &[0x20; 16]
        );

        // A dangling reference is an error, not a fallback.
        let dangling = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::RetrievalMethod {
                uri: "#nope".into(),
                type_uri: None,
            }],
        };
        assert!(matches!(
            store.resolve(&context(&doc, Some(&dangling))),
            Err(EncError::KeyNotFound(_))
        ));
        Ok(())
    }

    #[test]
    fn a_key_info_with_nothing_usable_falls_back_to_the_default() -> Result<()> {
        let doc = empty_document();
        let store = KeyStore::new().with_default_key(SymmetricKey::new(vec![7; 16]));
        let info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::Other(crate::model::XmlFragment::Element(
                "<x/>".into(),
            ))],
        };
        assert_eq!(store.resolve(&context(&doc, Some(&info)))?.len(), 16);
        Ok(())
    }

    #[test]
    fn the_recipient_attribute_can_select_a_key() -> Result<()> {
        let kek = SymmetricKey::new(vec![0x51; 16]);
        let cek = SymmetricKey::new(vec![0x52; 16]);
        let wrapped = cipher::wrap_key(KeyWrap::Aes128, &kek, &cek)?;
        let mut encrypted = EncryptedKey::new(wrapped).with_recipient("alice");
        encrypted.common.encryption_method = Some(EncryptionMethod::key_wrap(KeyWrap::Aes128));

        let store = KeyStore::new().with_named_key_encryption_key("alice", kek);
        assert_eq!(
            store.open_encrypted_key(&encrypted)?.as_bytes(),
            &[0x52; 16]
        );
        Ok(())
    }

    /// A 512-bit safe prime, small enough to keep the finite-field tests quick.
    const DH_PRIME: &str = concat!(
        "FEF3B78C842B8406334F4AAF28C8077743921557B93BC3E50E1F1A1688562AC0",
        "242F191536AC32BBE8C3E858D86D6DDCA8B383C6DDCA0BAFC57355BBA8C035B7",
    );

    fn dh_prime() -> Result<Vec<u8>> {
        oxixml_text::hex::decode(DH_PRIME)
            .map_err(|_| EncError::InvalidAgreementKey("bad test prime"))
    }

    /// An `EncryptedKey` whose `KeyInfo` is `agreement`, carrying `wrapped` under `wrap`.
    fn agreement_encrypted_key(
        wrap: KeyWrap,
        agreement: AgreementMethod,
        wrapped: Vec<u8>,
    ) -> EncryptedKey {
        let mut encrypted = EncryptedKey::new(wrapped);
        encrypted.common.encryption_method = Some(EncryptionMethod::key_wrap(wrap));
        encrypted.common.key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::AgreementMethod(Box::new(agreement))],
        });
        encrypted
    }

    #[test]
    fn ecdh_es_agreement_decrypts_an_encrypted_key_end_to_end() -> Result<()> {
        let mut rng = PseudoRandom::new(0xec);
        let recipient_private = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        let content_key = SymmetricKey::new(vec![0x9a; 16]);
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(KeyWrap::Aes128.uri().as_bytes().to_vec());

        let agreed = crate::agreement::ecdh_es_wrap_key(
            &recipient_private.public_key(),
            KeyWrap::Aes128,
            &params,
            &content_key,
            &mut rng,
        )?;

        let mut agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(&params));
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::EcKeyValue(EcKeyValue::from_public_key(
                &agreed.originator_public_key,
            )?)],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("bob"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, agreed.wrapped_key);
        let store = KeyStore::new().with_named_key_encryption_key("bob", recipient_private);
        assert_eq!(
            store.open_encrypted_key(&encrypted)?.as_bytes(),
            content_key.as_bytes()
        );
        Ok(())
    }

    #[test]
    fn dh_es_agreement_decrypts_an_encrypted_key_end_to_end() -> Result<()> {
        let parameters = DhParameters::new(&dh_prime()?, &[0x02], None)?;
        let mut rng = PseudoRandom::new(0xd5);
        let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
        let recipient_private = DhPrivateKey::generate(&parameters, &mut rng)?;
        let content_key = SymmetricKey::new(vec![0x5e; 16]);
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(KeyWrap::Aes128.uri().as_bytes().to_vec());

        let kek = crate::agreement::dh_es_key_encryption_key(
            &originator,
            &recipient_private.public_key()?,
            &params,
            16,
        )?;
        let wrapped = cipher::wrap_key(KeyWrap::Aes128, &kek, &content_key)?;

        let mut agreement = AgreementMethod::new(KeyAgreement::DhEs.uri());
        agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(&params));
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::DhKeyValue(DhKeyValue {
                p: parameters.prime().to_be_bytes(),
                q: None,
                generator: parameters.generator().to_be_bytes(),
                public: originator.public_key()?.to_be_bytes()?,
                seed: None,
            })],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("carol"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, wrapped);
        let store = KeyStore::new().with_named_key_encryption_key("carol", recipient_private);
        assert_eq!(
            store.open_encrypted_key(&encrypted)?.as_bytes(),
            content_key.as_bytes()
        );
        Ok(())
    }

    #[test]
    fn legacy_dh_agreement_decrypts_an_encrypted_key_end_to_end() -> Result<()> {
        let parameters = DhParameters::new(&dh_prime()?, &[0x02], None)?;
        let mut rng = PseudoRandom::new(0xd0);
        let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
        let recipient_private = DhPrivateKey::generate(&parameters, &mut rng)?;
        let content_key = SymmetricKey::new(vec![0x24; 16]);

        let kek = crate::agreement::dh_key_encryption_key(
            &originator,
            &recipient_private.public_key()?,
            Digest::Sha256,
            KeyWrap::Aes128.uri(),
            b"nonce",
            16,
        )?;
        let wrapped = cipher::wrap_key(KeyWrap::Aes128, &kek, &content_key)?;

        let mut agreement = AgreementMethod::new(KeyAgreement::Dh.uri());
        agreement.digest_method = Some(Digest::Sha256.uri().to_owned());
        agreement.ka_nonce = Some(b"nonce".to_vec());
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::DhKeyValue(DhKeyValue {
                p: parameters.prime().to_be_bytes(),
                q: None,
                generator: parameters.generator().to_be_bytes(),
                public: originator.public_key()?.to_be_bytes()?,
                seed: None,
            })],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("dan"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, wrapped);
        let store = KeyStore::new().with_named_key_encryption_key("dan", recipient_private);
        assert_eq!(
            store.open_encrypted_key(&encrypted)?.as_bytes(),
            content_key.as_bytes()
        );
        Ok(())
    }

    /// A `dh` agreement that also carries a `KeyDerivationMethod` contradicts itself — see
    /// [`KeyAgreement::uses_explicit_key_derivation`] — and is rejected rather than silently
    /// obeyed.
    #[test]
    fn a_dh_agreement_with_a_key_derivation_method_is_rejected() -> Result<()> {
        let parameters = DhParameters::new(&dh_prime()?, &[0x02], None)?;
        let mut rng = PseudoRandom::new(0xd1);
        let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
        let recipient_private = DhPrivateKey::generate(&parameters, &mut rng)?;

        let mut agreement = AgreementMethod::new(KeyAgreement::Dh.uri());
        agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(
            &ConcatKdfParams::new(Digest::Sha256),
        ));
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::DhKeyValue(DhKeyValue {
                p: parameters.prime().to_be_bytes(),
                q: None,
                generator: parameters.generator().to_be_bytes(),
                public: originator.public_key()?.to_be_bytes()?,
                seed: None,
            })],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("erin"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, vec![0u8; 24]);
        let store = KeyStore::new().with_named_key_encryption_key("erin", recipient_private);
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    /// An `AgreementMethod` naming a curve this crate does not implement fails closed, typed —
    /// never a silent wrong key.
    #[test]
    fn an_agreement_method_naming_an_unsupported_curve_fails_closed() -> Result<()> {
        let mut rng = PseudoRandom::new(0xec2);
        let recipient_private = EcPrivateKey::generate(CurveId::P256, &mut rng)?;

        let mut agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(
            &ConcatKdfParams::new(Digest::Sha256),
        ));
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::EcKeyValue(EcKeyValue {
                id: None,
                named_curve: crate::crypto::ecdh::P521_NAMED_CURVE_URI.to_owned(),
                public_key: vec![0x04; 133],
            })],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("frank"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, vec![0u8; 24]);
        let store = KeyStore::new().with_named_key_encryption_key("frank", recipient_private);
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
        Ok(())
    }

    /// A recipient private key absent from the store is `KeyNotFound`, and so is one registered
    /// under the right name but the wrong type — a symmetric key can no more open an `ECDH-ES`
    /// agreement than an EC key can open a `kw-aes` wrap.
    #[test]
    fn an_agreement_recipient_key_absent_or_the_wrong_type_is_key_not_found() -> Result<()> {
        let mut rng = PseudoRandom::new(0xec3);
        let recipient_private = EcPrivateKey::generate(CurveId::P256, &mut rng)?;

        let mut agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        agreement.key_derivation_method = Some(KeyDerivationMethod::concat_kdf(
            &ConcatKdfParams::new(Digest::Sha256),
        ));
        agreement.originator_key_info = Some(KeyInfo {
            id: None,
            items: vec![KeyInfoItem::EcKeyValue(EcKeyValue::from_public_key(
                &recipient_private.public_key(),
            )?)],
        });
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("nobody-registered"));
        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, vec![0u8; 24]);

        // Nothing at all is registered under that name.
        let empty = KeyStore::new();
        assert!(matches!(
            empty.open_encrypted_key(&encrypted),
            Err(EncError::KeyNotFound(_))
        ));

        // A symmetric key is registered under that exact name — the wrong type for `ECDH-ES`.
        let wrong_type = KeyStore::new()
            .with_named_key_encryption_key("nobody-registered", SymmetricKey::new(vec![0x11; 16]));
        assert!(matches!(
            wrong_type.open_encrypted_key(&encrypted),
            Err(EncError::KeyNotFound(_))
        ));
        Ok(())
    }

    /// An `AgreementMethod` with no `OriginatorKeyInfo` at all is a structural error, not a
    /// missing-key one — even though the store below has no recipient key registered either,
    /// which would just as validly be `KeyNotFound`. Document defects are diagnosed before the
    /// store is even consulted.
    #[test]
    fn an_agreement_method_without_an_originator_key_info_is_invalid_structure() -> Result<()> {
        let mut rng = PseudoRandom::new(0xec4);
        let recipient_private = EcPrivateKey::generate(CurveId::P256, &mut rng)?;

        let agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, vec![0u8; 24]);
        let store = KeyStore::new().with_named_key_encryption_key("g", recipient_private);
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::InvalidStructure(_))
        ));
        Ok(())
    }

    /// The same precedence holds one layer deeper: an `OriginatorKeyInfo` that is *present* but
    /// carries no `ECKeyValue` is still a structural error even when the recipient key is also
    /// unregistered — the document is diagnosed first, not whichever defect happens to be
    /// checked first by accident of how the store is queried.
    #[test]
    fn a_missing_ec_key_value_is_invalid_structure_even_with_no_recipient_key_registered() {
        let mut agreement = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        // Present, but empty: no ECKeyValue inside it.
        agreement.originator_key_info = Some(KeyInfo::new());
        agreement.recipient_key_info = Some(KeyInfo::with_key_name("nobody-registered"));

        let encrypted = agreement_encrypted_key(KeyWrap::Aes128, agreement, vec![0u8; 24]);
        let store = KeyStore::new(); // nothing registered at all
        assert!(matches!(
            store.open_encrypted_key(&encrypted),
            Err(EncError::InvalidStructure(_))
        ));
    }

    /// The `##other` verbatim passthrough for a non-`AgreementMethod` `ds:KeyValue` child (a
    /// plain `ds:RSAKeyValue`, which this crate has no business interpreting) still holds: it is
    /// neither an `EcKeyValue` nor a `DhKeyValue`, so it round-trips as one opaque fragment.
    #[test]
    fn an_unrecognized_key_value_still_round_trips_verbatim() -> Result<()> {
        let doc = oxixml_dom::build::parse_str(
            r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                    xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
                 <ds:KeyInfo><ds:KeyValue><ds:RSAKeyValue><ds:Modulus>AQ==</ds:Modulus><ds:Exponent>AQ==</ds:Exponent></ds:RSAKeyValue></ds:KeyValue></ds:KeyInfo>
                 <xenc:CipherData><xenc:CipherValue>AQ==</xenc:CipherValue></xenc:CipherData>
               </xenc:EncryptedData>"#,
        )?;
        let root = doc.document_element().ok_or_else(|| {
            EncError::InvalidStructure("the document has no document element".to_owned())
        })?;
        let data = crate::model::EncryptedData::parse(&doc, root)?;
        let key_info = data.common.key_info.as_ref().expect("key info");
        assert_eq!(key_info.items.len(), 1);
        assert!(matches!(&key_info.items[0], KeyInfoItem::Other(_)));
        assert!(key_info.ec_key_value().is_none());
        assert!(key_info.dh_key_value().is_none());
        Ok(())
    }
}
