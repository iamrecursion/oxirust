//! `CipherData` and the opaque-XML carrier the rest of the model uses.

/// A fragment of XML that this crate models opaquely.
///
/// The XML Encryption schema is full of `<any namespace='##other'/>` wildcards: an
/// `EncryptionMethod` may carry algorithm-specific parameters, a `ds:KeyInfo` may carry an
/// `X509Data` or a `KeyValue`, an `EncryptionProperty` may carry anything at all. Modelling
/// every possibility would mean re-implementing XML Signature and half a dozen profiles inside
/// an encryption crate.
///
/// Instead those children are kept **verbatim, as serialized XML**, so that parsing and
/// re-serializing an `EncryptedData` reproduces them exactly. Each stored element is serialized
/// with the namespace fixup of DOM Level 3 Appendix B.1, which means it carries every namespace
/// declaration it needs and can therefore be re-parsed on its own, in any context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmlFragment {
    /// Character data.
    Text(String),
    /// A self-contained serialized element, namespace declarations included.
    Element(String),
}

impl XmlFragment {
    /// Returns the stored text or serialized element.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(text) | Self::Element(text) => text,
        }
    }

    /// Returns `true` when the fragment is an element rather than character data.
    #[must_use]
    pub const fn is_element(&self) -> bool {
        matches!(self, Self::Element(_))
    }
}

/// The `xenc:CipherData` element of [XMLEnc §3.3](https://www.w3.org/TR/xmlenc-core1/#sec-CipherData).
///
/// The schema is a choice: either the ciphertext is inline, base64-encoded in a `CipherValue`,
/// or it lives somewhere else and a `CipherReference` says where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CipherData {
    /// A `xenc:CipherValue` — the ciphertext itself, already base64-decoded.
    Value(Vec<u8>),
    /// A `xenc:CipherReference` — a pointer to the ciphertext.
    Reference(CipherReference),
}

impl CipherData {
    /// Returns the inline ciphertext, or `None` for a `CipherReference`.
    #[must_use]
    pub fn value(&self) -> Option<&[u8]> {
        match self {
            Self::Value(octets) => Some(octets),
            Self::Reference(_) => None,
        }
    }

    /// Returns the reference, or `None` for an inline `CipherValue`.
    #[must_use]
    pub const fn reference(&self) -> Option<&CipherReference> {
        match self {
            Self::Reference(reference) => Some(reference),
            Self::Value(_) => None,
        }
    }
}

/// The `xenc:CipherReference` element of [XMLEnc §3.3.1](https://www.w3.org/TR/xmlenc-core1/#sec-CipherReference).
///
/// Dereferencing one means fetching `uri` and then running the `ds:Transform` chain over what
/// comes back. Both halves are application policy — an XML Encryption implementation that
/// silently fetched arbitrary URIs would be a server-side request forgery engine, and XMLEnc
/// §6.5 warns about exactly that — so this crate models the reference faithfully and refuses to
/// resolve it, returning
/// [`EncError::CipherReferenceNotResolved`](crate::EncError::CipherReferenceNotResolved).
/// An application that wants the behaviour can fetch the octets itself and replace the
/// `CipherData` with a [`CipherData::Value`] before decrypting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CipherReference {
    /// The required `URI` attribute.
    pub uri: String,
    /// The `ds:Transform` children of the optional `Transforms` element, in document order.
    pub transforms: Vec<Transform>,
}

impl CipherReference {
    /// Creates a reference with no transforms.
    #[must_use]
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            transforms: Vec::new(),
        }
    }
}

/// One `ds:Transform` inside a `CipherReference/Transforms`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transform {
    /// The required `Algorithm` attribute.
    pub algorithm: String,
    /// The transform's children, kept verbatim.
    pub content: Vec<XmlFragment>,
}

impl Transform {
    /// Creates a transform with no children.
    #[must_use]
    pub fn new(algorithm: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            content: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cipher_data_accessors_are_exclusive() {
        let value = CipherData::Value(vec![1, 2, 3]);
        assert_eq!(value.value(), Some(&[1u8, 2, 3][..]));
        assert!(value.reference().is_none());

        let reference = CipherData::Reference(CipherReference::new("#target"));
        assert!(reference.value().is_none());
        assert_eq!(
            reference.reference().map(|r| r.uri.as_str()),
            Some("#target")
        );
    }

    #[test]
    fn fragments_expose_their_text() {
        assert_eq!(XmlFragment::Text("a".into()).as_str(), "a");
        assert_eq!(XmlFragment::Element("<a/>".into()).as_str(), "<a/>");
        assert!(XmlFragment::Element("<a/>".into()).is_element());
        assert!(!XmlFragment::Text("a".into()).is_element());
    }

    #[test]
    fn a_transform_starts_empty() {
        let transform = Transform::new("urn:example:t");
        assert_eq!(transform.algorithm, "urn:example:t");
        assert!(transform.content.is_empty());
        assert!(CipherReference::new("#x").transforms.is_empty());
    }
}
