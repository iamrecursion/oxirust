//! The `<ds:KeyInfo>` data model — [XML Signature §4.5](https://www.w3.org/TR/xmldsig-core1/#sec-KeyInfo).
//!
//! `KeyInfo` is deliberately open-ended: it is a bag of hints about which key
//! to use, every one of them optional, and a verifier is free to ignore all of
//! them. This module models the hints faithfully — including the ones this
//! crate cannot act on, such as `<ds:PGPData>` — so that a
//! [`KeySelector`](crate::KeySelector) can look at what the signer actually
//! said instead of at a lossy summary.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{KeyInfo, KeyInfoItem, Signature};
//! use oxixml_dom::build;
//!
//! let doc = build::parse_str(
//!     r##"<Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
//!          <SignedInfo>
//!            <CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>
//!            <SignatureMethod Algorithm="http://www.w3.org/2000/09/xmldsig#rsa-sha1"/>
//!            <Reference URI="#o">
//!              <DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>
//!              <DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</DigestValue>
//!            </Reference>
//!          </SignedInfo>
//!          <SignatureValue>AA==</SignatureValue>
//!          <KeyInfo><KeyName>alice</KeyName></KeyInfo>
//!          <Object Id="o"/>
//!        </Signature>"##,
//! )?;
//! let root = doc.document_element().expect("a document element");
//! let signature = Signature::parse(&doc, root)?;
//! let key_info: &KeyInfo = signature.key_info().expect("a KeyInfo");
//! assert_eq!(key_info.key_names().collect::<Vec<_>>(), ["alice"]);
//! assert!(matches!(
//!     key_info.items().next(),
//!     Some(KeyInfoItem::KeyName(_))
//! ));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::algorithm::DigestAlgorithm;
use crate::key::dsa::DsaPublicKey;
use crate::key::ec::{EcPublicKey, named_curve_from_uri};
use crate::key::rsa::RsaPublicKey;

/// A parsed `<ds:KeyInfo>`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyInfo {
    pub(crate) id: Option<String>,
    pub(crate) items: Vec<KeyInfoItem>,
}

impl KeyInfo {
    /// The `Id` attribute, which is what a `<dsig11:KeyInfoReference>` points at.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Every child hint, in document order.
    pub fn items(&self) -> impl Iterator<Item = &KeyInfoItem> {
        self.items.iter()
    }

    /// The `<ds:KeyName>` hints.
    pub fn key_names(&self) -> impl Iterator<Item = &str> {
        self.items.iter().filter_map(|item| match item {
            KeyInfoItem::KeyName(name) => Some(name.as_str()),
            _ => None,
        })
    }

    /// Every DER X.509 certificate carried by a `<ds:X509Data>`, in document
    /// order. The signer's certificate is conventionally first, but nothing in
    /// the Recommendation requires that.
    pub fn certificates(&self) -> impl Iterator<Item = &[u8]> {
        self.items
            .iter()
            .filter_map(|item| match item {
                KeyInfoItem::X509Data(data) => Some(data.certificates.iter()),
                _ => None,
            })
            .flatten()
            .map(Vec::as_slice)
    }

    /// The first RSA public key that can be recovered from this `KeyInfo`,
    /// trying `<ds:RSAKeyValue>`, then `<dsig11:DEREncodedKeyValue>`, then
    /// `<ds:X509Certificate>`.
    ///
    /// Returns [`None`] when there is no RSA key here, and also when the key
    /// material is present but malformed — a `KeyInfo` is a *hint*, so a broken
    /// hint is simply not a usable key.
    #[must_use]
    pub fn rsa_public_key(&self) -> Option<RsaPublicKey> {
        for item in &self.items {
            let candidate = match item {
                KeyInfoItem::RsaKeyValue { modulus, exponent } => {
                    RsaPublicKey::from_components(modulus, exponent).ok()
                }
                KeyInfoItem::DerEncodedKeyValue(der) => {
                    RsaPublicKey::from_subject_public_key_info(der).ok()
                }
                KeyInfoItem::X509Data(data) => data
                    .certificates
                    .iter()
                    .find_map(|der| RsaPublicKey::from_x509_certificate(der).ok()),
                _ => None,
            };
            if candidate.is_some() {
                return candidate;
            }
        }
        None
    }

    /// The first elliptic-curve public key that can be recovered from this
    /// `KeyInfo`, trying `<dsig11:ECKeyValue>`, then
    /// `<dsig11:DEREncodedKeyValue>`, then `<ds:X509Certificate>`.
    ///
    /// Returns [`None`] when there is no EC key here, when the key material is
    /// malformed, and when it names a curve other than P-256 or P-384 — a
    /// `KeyInfo` is a *hint*, and a hint this crate cannot act on is not a usable
    /// key. A caller that needs to tell those cases apart can read
    /// [`KeyInfoItem::EcKeyValue`] directly.
    #[must_use]
    pub fn ec_public_key(&self) -> Option<EcPublicKey> {
        for item in &self.items {
            let candidate = match item {
                KeyInfoItem::EcKeyValue {
                    named_curve,
                    public_key,
                } => named_curve
                    .as_deref()
                    .and_then(named_curve_from_uri)
                    .and_then(|curve| EcPublicKey::from_sec1_point(curve, public_key).ok()),
                KeyInfoItem::DerEncodedKeyValue(der) => {
                    EcPublicKey::from_subject_public_key_info(der).ok()
                }
                KeyInfoItem::X509Data(data) => data
                    .certificates
                    .iter()
                    .find_map(|der| EcPublicKey::from_x509_certificate(der).ok()),
                _ => None,
            };
            if candidate.is_some() {
                return candidate;
            }
        }
        None
    }

    /// The first DSA public key that can be recovered from this `KeyInfo`,
    /// trying `<ds:DSAKeyValue>`, then `<dsig11:DEREncodedKeyValue>`, then
    /// `<ds:X509Certificate>`.
    ///
    /// A `<ds:DSAKeyValue>` that omits `P`, `Q` or `G` — which the schema permits,
    /// on the theory that the verifier already knows the domain parameters —
    /// yields [`None`], because this crate has nowhere else to get them from.
    #[must_use]
    pub fn dsa_public_key(&self) -> Option<DsaPublicKey> {
        for item in &self.items {
            let candidate = match item {
                KeyInfoItem::DsaKeyValue { p, q, g, y } => {
                    match (p.as_deref(), q.as_deref(), g.as_deref()) {
                        (Some(p), Some(q), Some(g)) => {
                            DsaPublicKey::from_components(p, q, g, y).ok()
                        }
                        _ => None,
                    }
                }
                KeyInfoItem::DerEncodedKeyValue(der) => {
                    DsaPublicKey::from_subject_public_key_info(der).ok()
                }
                KeyInfoItem::X509Data(data) => data
                    .certificates
                    .iter()
                    .find_map(|der| DsaPublicKey::from_x509_certificate(der).ok()),
                _ => None,
            };
            if candidate.is_some() {
                return candidate;
            }
        }
        None
    }

    /// Appends more hints, used when a `<dsig11:KeyInfoReference>` is followed
    /// and the two `KeyInfo` elements are merged.
    pub(crate) fn extend_items(&mut self, items: impl IntoIterator<Item = KeyInfoItem>) {
        self.items.extend(items);
    }

    /// The URI of the first `<dsig11:KeyInfoReference>`, if any.
    ///
    /// The verifier follows same-document references itself before calling a
    /// [`KeySelector`](crate::KeySelector), so a selector normally never sees
    /// one of these.
    #[must_use]
    pub fn key_info_reference(&self) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            KeyInfoItem::KeyInfoReference { uri } => Some(uri.as_str()),
            _ => None,
        })
    }
}

/// One child of a `<ds:KeyInfo>`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyInfoItem {
    /// `<ds:KeyName>` — an application-defined name for the key.
    KeyName(String),
    /// `<ds:KeyValue><ds:RSAKeyValue>` — the modulus and exponent, base64
    /// decoded.
    RsaKeyValue {
        /// The modulus `n`, big-endian.
        modulus: Vec<u8>,
        /// The public exponent `e`, big-endian.
        exponent: Vec<u8>,
    },
    /// `<ds:KeyValue><ds:DSAKeyValue>` — the domain parameters and the public
    /// value, base64 decoded.
    ///
    /// [§4.5.2.1](https://www.w3.org/TR/xmldsig-core1/#sec-DSAKeyValue) makes
    /// `P` and `Q` an optional pair and `G` optional on its own, so all three are
    /// [`Option`]; only `Y` is required. The `J`, `Seed` and `PgenCounter`
    /// children record how the parameters were generated rather than what they
    /// are, and are not retained.
    DsaKeyValue {
        /// `<ds:P>`, the prime modulus, big-endian.
        p: Option<Vec<u8>>,
        /// `<ds:Q>`, the subgroup order, big-endian.
        q: Option<Vec<u8>>,
        /// `<ds:G>`, the generator, big-endian.
        g: Option<Vec<u8>>,
        /// `<ds:Y>`, the public value, big-endian.
        y: Vec<u8>,
    },
    /// `<ds:KeyValue><dsig11:ECKeyValue>` — a named curve and a point on it.
    ///
    /// The `URI` is kept verbatim rather than resolved, so that a curve this
    /// crate does not implement can still be reported by name. It is [`None`]
    /// when the key gave explicit `<dsig11:ECParameters>` instead of naming a
    /// curve, a form this crate does not interpret.
    EcKeyValue {
        /// The `URI` of the `<dsig11:NamedCurve>`, verbatim.
        named_curve: Option<String>,
        /// The `<dsig11:PublicKey>`, base64 decoded — an uncompressed SEC 1
        /// point.
        public_key: Vec<u8>,
    },
    /// `<dsig11:DEREncodedKeyValue>` — a DER `SubjectPublicKeyInfo`.
    DerEncodedKeyValue(Vec<u8>),
    /// `<dsig11:KeyInfoReference URI="…"/>` — an indirection to another
    /// `KeyInfo`.
    KeyInfoReference {
        /// The `URI` attribute.
        uri: String,
    },
    /// `<ds:RetrievalMethod URI="…" Type="…"/>`.
    RetrievalMethod {
        /// The `URI` attribute.
        uri: String,
        /// The `Type` attribute, if present.
        key_type: Option<String>,
    },
    /// `<ds:X509Data>`.
    X509Data(X509Data),
    /// `<ds:PGPData>`. Recognised only.
    PgpData,
    /// `<ds:SPKIData>`. Recognised only.
    SpkiData,
    /// `<ds:MgmtData>` — deprecated by the Recommendation, kept for fidelity.
    MgmtData(String),
    /// Anything else: the content model of `KeyInfo` allows arbitrary elements
    /// from any namespace.
    Other {
        /// The namespace name of the element, if it has one.
        namespace: Option<String>,
        /// The local name of the element.
        local_name: String,
    },
}

/// A parsed `<ds:X509Data>` — [XML Signature §4.5.4](https://www.w3.org/TR/xmldsig-core1/#sec-X509Data).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct X509Data {
    pub(crate) certificates: Vec<Vec<u8>>,
    pub(crate) subject_names: Vec<String>,
    pub(crate) issuer_serials: Vec<IssuerSerial>,
    pub(crate) subject_key_identifiers: Vec<Vec<u8>>,
    pub(crate) crls: Vec<Vec<u8>>,
    pub(crate) digests: Vec<(DigestAlgorithm, Vec<u8>)>,
}

impl X509Data {
    /// The DER certificates, in document order.
    pub fn certificates(&self) -> impl Iterator<Item = &[u8]> {
        self.certificates.iter().map(Vec::as_slice)
    }

    /// The `<ds:X509SubjectName>` values, verbatim (RFC 4514 syntax).
    pub fn subject_names(&self) -> impl Iterator<Item = &str> {
        self.subject_names.iter().map(String::as_str)
    }

    /// The `<ds:X509IssuerSerial>` pairs.
    pub fn issuer_serials(&self) -> impl Iterator<Item = &IssuerSerial> {
        self.issuer_serials.iter()
    }

    /// The `<ds:X509SKI>` values, base64 decoded.
    pub fn subject_key_identifiers(&self) -> impl Iterator<Item = &[u8]> {
        self.subject_key_identifiers.iter().map(Vec::as_slice)
    }

    /// The `<ds:X509CRL>` values, base64 decoded.
    pub fn crls(&self) -> impl Iterator<Item = &[u8]> {
        self.crls.iter().map(Vec::as_slice)
    }

    /// The `<dsig11:X509Digest>` values: a digest of the DER certificate, used
    /// as a compact certificate reference. Resolving one needs a certificate
    /// store, which this crate does not have — see [the gap list](crate#gaps).
    pub fn digests(&self) -> impl Iterator<Item = (DigestAlgorithm, &[u8])> {
        self.digests
            .iter()
            .map(|(algorithm, value)| (*algorithm, value.as_slice()))
    }
}

/// An `<ds:X509IssuerSerial>`: a certificate identified by who issued it and
/// the serial number they gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssuerSerial {
    pub(crate) issuer_name: String,
    pub(crate) serial_number: String,
}

impl IssuerSerial {
    /// The `<ds:X509IssuerName>`, in the RFC 4514 syntax the Recommendation
    /// requires.
    #[must_use]
    pub fn issuer_name(&self) -> &str {
        &self.issuer_name
    }

    /// The `<ds:X509SerialNumber>`, an `xsd:integer` in decimal.
    #[must_use]
    pub fn serial_number(&self) -> &str {
        &self.serial_number
    }
}
