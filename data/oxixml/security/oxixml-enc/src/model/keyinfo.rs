//! `ds:KeyInfo` as XML Encryption extends it, plus `xenc:ReferenceList`.

use crate::model::agreement::{AgreementMethod, DhKeyValue, EcKeyValue};
use crate::model::cipher::XmlFragment;
use crate::model::encrypted::EncryptedKey;

/// The `ds:KeyInfo` element, modelled only as far as XML Encryption needs.
///
/// XML Signature defines a long list of `KeyInfo` children — `KeyValue`, `X509Data`,
/// `PGPData`, `SPKIData`, `MgmtData` — and XMLEnc §3.5 adds two more: `xenc:EncryptedKey` and
/// `xenc:AgreementMethod`. The children that affect *decryption* are modelled structurally:
///
/// * [`KeyName`](KeyInfoItem::KeyName) — names a key the recipient already has.
/// * [`EncryptedKey`](KeyInfoItem::EncryptedKey) — carries the content-encryption key, wrapped.
/// * [`RetrievalMethod`](KeyInfoItem::RetrievalMethod) — points at an `EncryptedKey` elsewhere
///   in the same document.
/// * [`AgreementMethod`](KeyInfoItem::AgreementMethod) — derives a key-encryption key by
///   agreement instead of transporting one; see [`crate::model::AgreementMethod`].
/// * [`EcKeyValue`](KeyInfoItem::EcKeyValue) and [`DhKeyValue`](KeyInfoItem::DhKeyValue) — the
///   public-key forms an `AgreementMethod`'s `OriginatorKeyInfo`/`RecipientKeyInfo` carry,
///   whether they appear bare or wrapped in a `ds:KeyValue`.
///
/// Everything else round-trips verbatim as [`Other`](KeyInfoItem::Other). That is not a
/// shortcut: an encryption implementation has no business interpreting an X.509 chain, and a
/// deployment that needs one plugs in its own [`KeyResolver`](crate::KeyResolver) which can read
/// the fragment. In particular a `ds:KeyValue` that carries anything other than a recognized
/// `ECKeyValue`/`DHKeyValue` — a `ds:RSAKeyValue`, a `ds:DSAKeyValue`, an unrecognized extension
/// — is kept as one opaque `Other` fragment, exactly as before this crate understood key
/// agreement at all.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeyInfo {
    /// The optional `Id` attribute.
    pub id: Option<String>,
    /// The children, in document order.
    pub items: Vec<KeyInfoItem>,
}

impl KeyInfo {
    /// Returns an empty `KeyInfo`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a `KeyInfo` whose only child is a `ds:KeyName`.
    #[must_use]
    pub fn with_key_name(name: impl Into<String>) -> Self {
        Self {
            id: None,
            items: vec![KeyInfoItem::KeyName(name.into())],
        }
    }

    /// Returns a `KeyInfo` whose only child is an `xenc:EncryptedKey`.
    #[must_use]
    pub fn with_encrypted_key(key: EncryptedKey) -> Self {
        Self {
            id: None,
            items: vec![KeyInfoItem::EncryptedKey(Box::new(key))],
        }
    }

    /// Returns the first `ds:KeyName`, if any.
    #[must_use]
    pub fn key_name(&self) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::KeyName(name) => Some(name.as_str()),
            _ => None,
        })
    }

    /// Returns the first `xenc:EncryptedKey`, if any.
    #[must_use]
    pub fn encrypted_key(&self) -> Option<&EncryptedKey> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::EncryptedKey(key) => Some(key.as_ref()),
            _ => None,
        })
    }

    /// Returns the `URI` of the first `ds:RetrievalMethod`, if any.
    #[must_use]
    pub fn retrieval_uri(&self) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::RetrievalMethod { uri, .. } => Some(uri.as_str()),
            _ => None,
        })
    }

    /// Returns the first `xenc:AgreementMethod`, if any.
    #[must_use]
    pub fn agreement_method(&self) -> Option<&AgreementMethod> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::AgreementMethod(method) => Some(method.as_ref()),
            _ => None,
        })
    }

    /// Returns the first elliptic-curve public-key value, if any.
    #[must_use]
    pub fn ec_key_value(&self) -> Option<&EcKeyValue> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::EcKeyValue(value) => Some(value),
            _ => None,
        })
    }

    /// Returns the first finite-field Diffie–Hellman public-key value, if any.
    #[must_use]
    pub fn dh_key_value(&self) -> Option<&DhKeyValue> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::DhKeyValue(value) => Some(value),
            _ => None,
        })
    }

    /// Returns `true` when there are no children.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// One child of a [`KeyInfo`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyInfoItem {
    /// `ds:KeyName` — a name the recipient can look a key up by.
    KeyName(String),
    /// `ds:RetrievalMethod` — a pointer to key information elsewhere. Only a same-document
    /// reference (`URI="#id"`) is resolvable by this crate.
    RetrievalMethod {
        /// The required `URI` attribute.
        uri: String,
        /// The optional `Type` attribute, e.g. `…xmlenc#EncryptedKey`.
        type_uri: Option<String>,
    },
    /// `xenc:EncryptedKey` — the content-encryption key, itself encrypted.
    EncryptedKey(Box<EncryptedKey>),
    /// `xenc:AgreementMethod` — derives a key-encryption key by agreement rather than
    /// transporting one.
    AgreementMethod(Box<AgreementMethod>),
    /// An elliptic-curve public-key value (`ECKeyValue`), bare or unwrapped from a `ds:KeyValue`.
    EcKeyValue(EcKeyValue),
    /// A finite-field Diffie–Hellman public-key value (`xenc:DHKeyValue`), bare or unwrapped
    /// from a `ds:KeyValue`.
    DhKeyValue(DhKeyValue),
    /// Any other child, kept verbatim.
    Other(XmlFragment),
}

/// The `xenc:ReferenceList` element of
/// [XMLEnc §3.6](https://www.w3.org/TR/xmlenc-core1/#sec-ReferenceList).
///
/// An `EncryptedKey` uses it to point at everything the transported key decrypts, which lets a
/// recipient unwrap the key once and then find its uses.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReferenceList {
    /// The `DataReference` and `KeyReference` children, in document order.
    pub references: Vec<Reference>,
}

impl ReferenceList {
    /// Returns an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the `URI`s of the `DataReference` children.
    pub fn data_references(&self) -> impl Iterator<Item = &str> {
        self.references
            .iter()
            .filter(|reference| reference.kind == ReferenceKind::Data)
            .map(|reference| reference.uri.as_str())
    }

    /// Returns the `URI`s of the `KeyReference` children.
    pub fn key_references(&self) -> impl Iterator<Item = &str> {
        self.references
            .iter()
            .filter(|reference| reference.kind == ReferenceKind::Key)
            .map(|reference| reference.uri.as_str())
    }
}

/// Whether a reference points at encrypted data or at another encrypted key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceKind {
    /// `xenc:DataReference`.
    Data,
    /// `xenc:KeyReference`.
    Key,
}

impl ReferenceKind {
    /// Returns the element's local name.
    #[must_use]
    pub const fn local_name(self) -> &'static str {
        match self {
            Self::Data => "DataReference",
            Self::Key => "KeyReference",
        }
    }
}

/// One `DataReference` or `KeyReference`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// Which of the two elements this is.
    pub kind: ReferenceKind,
    /// The required `URI` attribute.
    pub uri: String,
    /// Any children, kept verbatim.
    pub extensions: Vec<XmlFragment>,
}

impl Reference {
    /// Creates a `DataReference`.
    #[must_use]
    pub fn data(uri: impl Into<String>) -> Self {
        Self {
            kind: ReferenceKind::Data,
            uri: uri.into(),
            extensions: Vec::new(),
        }
    }

    /// Creates a `KeyReference`.
    #[must_use]
    pub fn key(uri: impl Into<String>) -> Self {
        Self {
            kind: ReferenceKind::Key,
            uri: uri.into(),
            extensions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_find_the_first_matching_child() {
        let info = KeyInfo {
            id: Some("ki".into()),
            items: vec![
                KeyInfoItem::Other(XmlFragment::Element("<ds:X509Data/>".into())),
                KeyInfoItem::KeyName("first".into()),
                KeyInfoItem::KeyName("second".into()),
                KeyInfoItem::RetrievalMethod {
                    uri: "#ek".into(),
                    type_uri: None,
                },
            ],
        };
        assert_eq!(info.key_name(), Some("first"));
        assert_eq!(info.retrieval_uri(), Some("#ek"));
        assert!(info.encrypted_key().is_none());
        assert!(!info.is_empty());
    }

    #[test]
    fn an_empty_key_info_reports_nothing() {
        let info = KeyInfo::new();
        assert!(info.is_empty());
        assert_eq!(info.key_name(), None);
        assert_eq!(info.retrieval_uri(), None);
        assert!(info.encrypted_key().is_none());
    }

    #[test]
    fn reference_lists_split_by_kind() {
        let list = ReferenceList {
            references: vec![
                Reference::data("#d1"),
                Reference::key("#k1"),
                Reference::data("#d2"),
            ],
        };
        assert_eq!(list.data_references().collect::<Vec<_>>(), ["#d1", "#d2"]);
        assert_eq!(list.key_references().collect::<Vec<_>>(), ["#k1"]);
        assert!(ReferenceList::new().references.is_empty());
    }

    #[test]
    fn reference_kinds_name_their_elements() {
        assert_eq!(ReferenceKind::Data.local_name(), "DataReference");
        assert_eq!(ReferenceKind::Key.local_name(), "KeyReference");
    }
}
