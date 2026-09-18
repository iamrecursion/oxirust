//! `xenc:AgreementMethod` and the elements it carries.
//!
//! [XMLEnc §5.6](https://www.w3.org/TR/xmlenc-core1/#sec-Alg-KeyAgreement) puts an
//! `AgreementMethod` inside a `ds:KeyInfo` in place of a key that travels encrypted: instead of
//! transporting a key-encryption key, the two parties each contribute a public value and derive
//! the same key-encryption key independently. An `AgreementMethod` carries, in schema order:
//!
//! * `KA-Nonce?` — a nonce the legacy `dh` derivation of XMLEnc 1.0 §5.5.2 mixes in.
//! * a `##other` extension point, where XMLEnc 1.1 places `ds:DigestMethod` (for `dh`) and
//!   `xenc11:KeyDerivationMethod` (for `dh-es` and `ECDH-ES`) — both are recognized here, and
//!   this crate additionally accepts a `ConcatKDFParams`-bearing `KeyDerivationMethod`; anything
//!   else in that slot round-trips verbatim as [`other`](AgreementMethod::other).
//! * `OriginatorKeyInfo?` and `RecipientKeyInfo?` — each is a full `ds:KeyInfoType`, so it is
//!   modelled with the same [`KeyInfo`] type a document's own `ds:KeyInfo` uses. The originator's
//!   public value normally arrives here as an
//!   [`EcKeyValue`](crate::model::keyinfo::KeyInfoItem::EcKeyValue) or
//!   [`DhKeyValue`](crate::model::keyinfo::KeyInfoItem::DhKeyValue); the recipient's is usually
//!   just a `ds:KeyName` that says which registered private key to use.
//!
//! # Namespace leniency for `ECKeyValue`
//!
//! Two different XML Signature/Encryption specifications define an `ECKeyValue` element for
//! carrying a named-curve public point: XML Signature 1.1 defines `dsig11:ECKeyValue`
//! (`http://www.w3.org/2009/xmldsig11#`) with a `NamedCurve` child whose curve is a `URI`
//! attribute, and implementations following XML Encryption 1.1 have been observed to emit
//! `xenc11:ECKeyValue` (`http://www.w3.org/2009/xmlenc11#`) instead. Rather than guess which one
//! a given producer used, parsing accepts `ECKeyValue` in either namespace, and accepts the
//! curve URI either as `NamedCurve/@URI` or as `NamedCurve`'s text content. Writing always
//! produces one canonical shape — `dsig11:ECKeyValue` with `NamedCurve/@URI` — the only
//! namespace/shape pairing this crate has confirmed against the XML Signature 1.1 schema
//! itself, rather than a hybrid of the two specifications.
//!
//! Similarly, a `ds:KeyValue` that wraps an `ECKeyValue` or `xenc:DHKeyValue` — the conventional
//! placement, since `ds:KeyValueType` is defined as a choice with a `##other` extension point —
//! is unwrapped into the same structured item a bare, unwrapped `ECKeyValue`/`DHKeyValue` parses
//! to. A `ds:KeyValue` that carries anything else (`ds:RSAKeyValue`, `ds:DSAKeyValue`, or an
//! unrecognized extension) is left exactly as it always has been: one verbatim
//! [`Other`](crate::model::keyinfo::KeyInfoItem::Other) fragment.

use crate::agreement::require_named_curve;
use crate::algorithm::{KeyAgreement, KeyDerivation};
use crate::crypto::dh::{DhParameters, DhPublicKey};
use crate::crypto::digest::Digest;
use crate::crypto::ecdh::{EcPublicKey, named_curve_uri};
use crate::crypto::kdf::ConcatKdfParams;
use crate::error::{EncError, Result};
use crate::model::cipher::XmlFragment;
use crate::model::keyinfo::KeyInfo;

/// The `xenc:AgreementMethod` element: `KA-Nonce?`, an XMLEnc 1.1 extension slot this crate
/// recognizes as `ds:DigestMethod` and `xenc11:KeyDerivationMethod`, then `OriginatorKeyInfo?`
/// and `RecipientKeyInfo?`, in that schema order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgreementMethod {
    /// The required `Algorithm` attribute, kept as the URI so an unrecognized agreement survives
    /// a parse/serialize round trip; resolve it with [`Self::key_agreement`].
    pub algorithm: String,
    /// The optional `KA-Nonce` child, base64-decoded. XMLEnc 1.0 §5.5.2 folds this into the
    /// legacy `dh` key derivation; it is unused by `dh-es` and `ECDH-ES`.
    pub ka_nonce: Option<Vec<u8>>,
    /// The optional `ds:DigestMethod` child's `Algorithm` attribute. This is the digest the
    /// legacy `dh` derivation uses; `dh-es` and `ECDH-ES` name their digest inside
    /// [`key_derivation_method`](Self::key_derivation_method) instead, and a document that
    /// supplies both is contradicting itself — see
    /// [`KeyAgreement::uses_explicit_key_derivation`].
    pub digest_method: Option<String>,
    /// The optional `xenc11:KeyDerivationMethod` child, which `dh-es` and `ECDH-ES` require and
    /// the legacy `dh` must not carry.
    pub key_derivation_method: Option<KeyDerivationMethod>,
    /// The optional `OriginatorKeyInfo` child: the sender's (usually ephemeral) public value.
    pub originator_key_info: Option<KeyInfo>,
    /// The optional `RecipientKeyInfo` child: usually just a `ds:KeyName` naming which
    /// registered private key the recipient should use.
    pub recipient_key_info: Option<KeyInfo>,
    /// Any further children — and any character data, since the element's content model is
    /// mixed — kept verbatim.
    pub other: Vec<XmlFragment>,
}

impl AgreementMethod {
    /// Creates an agreement method that names `algorithm` and carries nothing else.
    #[must_use]
    pub fn new(algorithm: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            ka_nonce: None,
            digest_method: None,
            key_derivation_method: None,
            originator_key_info: None,
            recipient_key_info: None,
            other: Vec::new(),
        }
    }

    /// Resolves the `Algorithm` attribute to a [`KeyAgreement`].
    ///
    /// # Errors
    ///
    /// Returns [`EncError::UnsupportedAlgorithm`] for a URI this crate does not implement.
    pub fn key_agreement(&self) -> Result<KeyAgreement> {
        KeyAgreement::from_uri(&self.algorithm)
            .ok_or_else(|| EncError::UnsupportedAlgorithm(self.algorithm.clone()))
    }
}

/// The `xenc11:KeyDerivationMethod` element of
/// [XMLEnc §5.4](https://www.w3.org/TR/xmlenc-core1/#sec-KeyDerivation).
///
/// The schema gives it one required `Algorithm` attribute and arbitrary children. This crate
/// understands one derivation, the ConcatKDF ([`KeyDerivation::ConcatKdf`]), whose parameters
/// arrive in an `xenc11:ConcatKDFParams` child; that child's fields are flattened onto this type
/// rather than nested, the same way [`EncryptionMethod`](crate::model::EncryptionMethod)
/// flattens `ds:DigestMethod` and `xenc11:MGF`. A `KeyDerivationMethod` naming anything else
/// parses fine — its children land in [`other`](Self::other) — and fails only when
/// [`Self::concat_kdf_params`] is actually asked to resolve it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyDerivationMethod {
    /// The required `Algorithm` attribute.
    pub algorithm: String,
    /// `ConcatKDFParams/ds:DigestMethod/@Algorithm`, required by the schema whenever a
    /// `ConcatKDFParams` child is present.
    pub digest_method: Option<String>,
    /// `ConcatKDFParams/@AlgorithmID`, decoded from its bit-string encoding. Empty when the
    /// attribute is absent, which is the schema's default.
    pub algorithm_id: Vec<u8>,
    /// `ConcatKDFParams/@PartyUInfo`, decoded the same way.
    pub party_u_info: Vec<u8>,
    /// `ConcatKDFParams/@PartyVInfo`, decoded the same way.
    pub party_v_info: Vec<u8>,
    /// `ConcatKDFParams/@SuppPubInfo`, decoded the same way.
    pub supp_pub_info: Vec<u8>,
    /// `ConcatKDFParams/@SuppPrivInfo`, decoded the same way.
    pub supp_priv_info: Vec<u8>,
    /// Any child other than a recognized `ConcatKDFParams` — which is every child, when
    /// `algorithm` names something other than the ConcatKDF — kept verbatim.
    pub other: Vec<XmlFragment>,
}

impl KeyDerivationMethod {
    /// Builds a `KeyDerivationMethod` naming the ConcatKDF and carrying `params`.
    #[must_use]
    pub fn concat_kdf(params: &ConcatKdfParams) -> Self {
        Self {
            algorithm: KeyDerivation::ConcatKdf.uri().to_owned(),
            digest_method: Some(params.digest.uri().to_owned()),
            algorithm_id: params.algorithm_id.clone(),
            party_u_info: params.party_u_info.clone(),
            party_v_info: params.party_v_info.clone(),
            supp_pub_info: params.supp_pub_info.clone(),
            supp_priv_info: params.supp_priv_info.clone(),
            other: Vec::new(),
        }
    }

    /// Resolves this element to the [`ConcatKdfParams`] the crypto layer takes.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::UnsupportedAlgorithm`] when `algorithm` does not name the ConcatKDF,
    /// or when its digest is not one of the digests this crate implements, and
    /// [`EncError::InvalidStructure`] when it names the ConcatKDF but carries no
    /// `ConcatKDFParams` (so there is no digest to resolve).
    pub fn concat_kdf_params(&self) -> Result<ConcatKdfParams> {
        if KeyDerivation::from_uri(&self.algorithm) != Some(KeyDerivation::ConcatKdf) {
            return Err(EncError::UnsupportedAlgorithm(self.algorithm.clone()));
        }
        let digest_uri = self.digest_method.as_deref().ok_or_else(|| {
            EncError::InvalidStructure(
                "a `xenc11:KeyDerivationMethod` naming the ConcatKDF has no `ConcatKDFParams`"
                    .to_owned(),
            )
        })?;
        let digest = Digest::from_uri(digest_uri)
            .ok_or_else(|| EncError::UnsupportedAlgorithm(digest_uri.to_owned()))?;
        Ok(ConcatKdfParams::new(digest)
            .with_algorithm_id(self.algorithm_id.clone())
            .with_party_u_info(self.party_u_info.clone())
            .with_party_v_info(self.party_v_info.clone())
            .with_supp_pub_info(self.supp_pub_info.clone())
            .with_supp_priv_info(self.supp_priv_info.clone()))
    }
}

/// The `dsig11:ECKeyValue` (or, leniently, `xenc11:ECKeyValue`) element: an elliptic-curve public
/// key named by a curve URI.
///
/// Only the named-curve form is modelled; `ECParameters` (explicit curve parameters) is not. The
/// schema makes `ECParameters` and `NamedCurve` a `choice` — exactly one, never both — but this
/// crate's own parse is more forgiving than the schema requires: it looks for a `NamedCurve`
/// child and, if one is present, uses it and ignores any `ECParameters` sitting alongside it
/// rather than rejecting the document as malformed. An `ECKeyValue` with *no* `NamedCurve` at
/// all — the only shape that actually needs `ECParameters` — is rejected by [`Self::public_key`]
/// rather than approximated.
///
/// # Namespace and `NamedCurve` leniency
///
/// Two different XML Signature/Encryption specifications define an `ECKeyValue` element for
/// carrying a named-curve public point: XML Signature 1.1 defines `dsig11:ECKeyValue`
/// (`http://www.w3.org/2009/xmldsig11#`) with a `NamedCurve` child whose curve is a `URI`
/// attribute, and implementations following XML Encryption 1.1 have been observed to emit
/// `xenc11:ECKeyValue` (`http://www.w3.org/2009/xmlenc11#`) instead. Rather than guess which one
/// a given producer used, parsing accepts `ECKeyValue` in either namespace, and accepts the
/// curve URI either as `NamedCurve/@URI` or as `NamedCurve`'s text content. Writing always
/// produces one canonical shape — `dsig11:ECKeyValue` with `NamedCurve/@URI` — because that is
/// the only namespace/shape combination this crate has confirmed against the XML Signature 1.1
/// schema itself, rather than a hybrid of the two specifications' shapes.
///
/// A `ds:KeyValue` that wraps an `ECKeyValue` or `xenc:DHKeyValue` — the conventional placement,
/// since `ds:KeyValueType` is defined as a choice with a `##other` extension point — is unwrapped
/// into the same structured item a bare, unwrapped `ECKeyValue`/`DHKeyValue` parses to. A
/// `ds:KeyValue` that carries anything else (`ds:RSAKeyValue`, `ds:DSAKeyValue`, or an
/// unrecognized extension) is left exactly as it always has been: one verbatim
/// [`Other`](crate::model::keyinfo::KeyInfoItem::Other) fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcKeyValue {
    /// The optional `Id` attribute.
    pub id: Option<String>,
    /// The curve, as the raw `NamedCurve` URI — resolve it with [`Self::public_key`].
    pub named_curve: String,
    /// `PublicKey`, base64-decoded: the point in uncompressed SEC 1 form (`04 ‖ X ‖ Y`).
    pub public_key: Vec<u8>,
}

impl EcKeyValue {
    /// Builds the canonical form of an `ECKeyValue` for `key`.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] only if the point could not be encoded, which
    /// [`EcPublicKey`]'s own invariant rules out.
    pub fn from_public_key(key: &EcPublicKey) -> Result<Self> {
        Ok(Self {
            id: None,
            named_curve: named_curve_uri(key.curve()).to_owned(),
            public_key: key.to_sec1()?,
        })
    }

    /// Resolves this element to a validated [`EcPublicKey`].
    ///
    /// # Errors
    ///
    /// Returns [`EncError::UnsupportedAlgorithm`] when `named_curve` is not one of the curves
    /// this crate implements, and [`EncError::InvalidAgreementKey`] when `public_key` is not a
    /// valid point on it.
    pub fn public_key(&self) -> Result<EcPublicKey> {
        let curve = require_named_curve(&self.named_curve)?;
        EcPublicKey::from_sec1(curve, &self.public_key)
    }
}

/// The `xenc:DHKeyValue` element of
/// [XMLEnc §5.6.1](https://www.w3.org/TR/xmlenc-core1/#sec-DHKeyValue): a finite-field
/// Diffie–Hellman domain and a public value in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DhKeyValue {
    /// `P`, base64-decoded: the prime modulus.
    pub p: Vec<u8>,
    /// `Q`, base64-decoded: the subgroup order, when the document supplies one. See
    /// [`crate::crypto::dh`] for why supplying it matters.
    pub q: Option<Vec<u8>>,
    /// `Generator`, base64-decoded.
    pub generator: Vec<u8>,
    /// `Public`, base64-decoded: the public value itself.
    pub public: Vec<u8>,
    /// `seed`, base64-decoded, when present. This crate does not verify that the domain
    /// parameters were generated from it; the octets are kept only for round-trip fidelity.
    pub seed: Option<Vec<u8>>,
}

impl DhKeyValue {
    /// Resolves `p`, `generator` and `q` to validated [`DhParameters`].
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the values do not form a plausible
    /// Diffie–Hellman domain; see [`DhParameters::new`].
    pub fn parameters(&self) -> Result<DhParameters> {
        DhParameters::new(&self.p, &self.generator, self.q.as_deref())
    }

    /// Resolves this element to a validated [`DhPublicKey`] in its own domain.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::parameters`], plus [`EncError::InvalidAgreementKey`] when `public`
    /// is not a valid public value in that domain.
    pub fn public_key(&self) -> Result<DhPublicKey> {
        DhPublicKey::new(&self.parameters()?, &self.public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ecdh::{EcPrivateKey, P256_NAMED_CURVE_URI, P521_NAMED_CURVE_URI};
    use crate::random::PseudoRandom;
    use oxixml_support::ec::CurveId;

    #[test]
    fn a_new_agreement_method_carries_only_its_algorithm() -> Result<()> {
        let method = AgreementMethod::new(KeyAgreement::EcdhEs.uri());
        assert_eq!(method.key_agreement()?, KeyAgreement::EcdhEs);
        assert!(method.ka_nonce.is_none());
        assert!(method.originator_key_info.is_none());
        assert!(method.other.is_empty());
        Ok(())
    }

    #[test]
    fn an_unrecognized_agreement_uri_is_reported_when_resolved() {
        let method = AgreementMethod::new("urn:example:agree");
        assert!(matches!(
            method.key_agreement(),
            Err(EncError::UnsupportedAlgorithm(uri)) if uri == "urn:example:agree"
        ));
    }

    #[test]
    fn concat_kdf_round_trips_through_the_derivation_method() -> Result<()> {
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(b"alg".to_vec())
            .with_party_u_info(b"u".to_vec())
            .with_party_v_info(b"v".to_vec());
        let method = KeyDerivationMethod::concat_kdf(&params);
        assert_eq!(method.algorithm, KeyDerivation::ConcatKdf.uri());
        assert_eq!(method.concat_kdf_params()?, params);
        Ok(())
    }

    #[test]
    fn a_key_derivation_method_naming_something_else_is_refused_when_resolved() {
        let method = KeyDerivationMethod {
            algorithm: "http://www.w3.org/2009/xmlenc11#pbkdf2".to_owned(),
            digest_method: None,
            algorithm_id: Vec::new(),
            party_u_info: Vec::new(),
            party_v_info: Vec::new(),
            supp_pub_info: Vec::new(),
            supp_priv_info: Vec::new(),
            other: Vec::new(),
        };
        assert!(matches!(
            method.concat_kdf_params(),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn a_concat_kdf_method_with_no_params_is_a_structural_error() {
        let method = KeyDerivationMethod {
            algorithm: KeyDerivation::ConcatKdf.uri().to_owned(),
            digest_method: None,
            algorithm_id: Vec::new(),
            party_u_info: Vec::new(),
            party_v_info: Vec::new(),
            supp_pub_info: Vec::new(),
            supp_priv_info: Vec::new(),
            other: Vec::new(),
        };
        assert!(matches!(
            method.concat_kdf_params(),
            Err(EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn an_ec_key_value_round_trips_through_a_public_key() -> Result<()> {
        let mut rng = PseudoRandom::new(9);
        let key = EcPrivateKey::generate(CurveId::P256, &mut rng)?.public_key();
        let value = EcKeyValue::from_public_key(&key)?;
        assert_eq!(value.named_curve, P256_NAMED_CURVE_URI);
        assert_eq!(value.public_key()?, key);
        Ok(())
    }

    #[test]
    fn an_ec_key_value_naming_an_unsupported_curve_is_refused() {
        let value = EcKeyValue {
            id: None,
            named_curve: P521_NAMED_CURVE_URI.to_owned(),
            public_key: vec![0x04; 133],
        };
        assert!(matches!(
            value.public_key(),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn a_dh_key_value_resolves_its_domain_and_public_value() -> Result<()> {
        // The 512-bit safe prime already used by the crypto-layer tests.
        let prime = oxixml_text::hex::decode(concat!(
            "FEF3B78C842B8406334F4AAF28C8077743921557B93BC3E50E1F1A1688562AC0",
            "242F191536AC32BBE8C3E858D86D6DDCA8B383C6DDCA0BAFC57355BBA8C035B7",
        ))
        .map_err(|_| EncError::InvalidAgreementKey("bad test hex"))?;
        let mut rng = PseudoRandom::new(4);
        let parameters = DhParameters::new(&prime, &[0x02], None)?;
        let public = crate::crypto::dh::DhPrivateKey::generate(&parameters, &mut rng)?
            .public_key()?
            .to_be_bytes()?;
        let value = DhKeyValue {
            p: prime,
            q: None,
            generator: vec![0x02],
            public: public.clone(),
            seed: Some(vec![0xab]),
        };
        assert_eq!(value.parameters()?, parameters);
        assert_eq!(value.public_key()?.to_be_bytes()?, public);
        Ok(())
    }
}
