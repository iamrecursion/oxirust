//! The algorithm identifiers of [XML Encryption 1.1 §5](https://www.w3.org/TR/xmlenc-core1/#sec-Algorithms),
//! and the enumerations that make them exhaustive.
//!
//! XML Encryption names every algorithm by URI in an `Algorithm` attribute. Three families
//! matter, and each gets a type here:
//!
//! * [`BlockEncryption`] — what encrypts the data itself (`EncryptedData/EncryptionMethod`).
//! * [`KeyWrap`] — what protects a content-encryption key with a *symmetric* key-encryption key
//!   (`EncryptedKey/EncryptionMethod`).
//! * [`KeyTransport`] — what protects a content-encryption key with a *public* key
//!   (`EncryptedKey/EncryptionMethod`).
//!
//! Two more name the halves of a key *agreement*, which produces a key-encryption key rather
//! than transporting one (`ds:KeyInfo/xenc:AgreementMethod`):
//!
//! * [`KeyAgreement`] — how the two parties reach a shared secret.
//! * [`KeyDerivation`] — how that shared secret becomes a key.
//!
//! Each type parses from a URI, renders back to the canonical URI, and reports the key size it
//! requires, so an `EncryptionMethod` can be validated without string comparisons leaking into
//! the rest of the crate.
//!
//! # Deliberately absent
//!
//! * **Triple DES** (`tripledes-cbc`, `kw-tripledes`). XMLEnc 1.0 made it mandatory; XMLEnc 1.1
//!   §5.1.1 demotes it and NIST SP 800-131A has since disallowed it outright for encryption.
//!   Implementing it would mean shipping a cipher no new document should use.
//! * **PBKDF2** (`pbkdf2`). It derives a key from a passphrase rather than from a shared secret,
//!   it is OPTIONAL in XMLEnc 1.1, and a passphrase-derived key-encryption key is a different
//!   security model that belongs to the application rather than to this crate.
//! * **`EXI`** as a `Type` value: the compression is a separate specification.
//!
//! An `EncryptionMethod` naming any of these parses fine — the [model](crate::model) keeps the
//! URI verbatim — and fails only when a cipher is actually asked for, with
//! [`EncError::UnsupportedAlgorithm`](crate::EncError::UnsupportedAlgorithm).
//!
//! # Example
//!
//! ```
//! use oxixml_enc::algorithm::{BlockEncryption, KeyTransport, KeyWrap};
//!
//! let algorithm = BlockEncryption::from_uri("http://www.w3.org/2009/xmlenc11#aes256-gcm")
//!     .expect("a known block-encryption URI");
//! assert_eq!(algorithm, BlockEncryption::Aes256Gcm);
//! assert_eq!(algorithm.key_len(), 32);
//! assert!(algorithm.is_authenticated());
//!
//! assert_eq!(KeyWrap::from_uri(algorithm.uri()), None); // the families do not overlap
//! assert_eq!(
//!     KeyTransport::from_uri("http://www.w3.org/2009/xmlenc11#rsa-oaep"),
//!     Some(KeyTransport::RsaOaep),
//! );
//! ```

/// The XML Encryption 1.0 namespace name, `http://www.w3.org/2001/04/xmlenc#`.
pub const XMLENC_NS: &str = "http://www.w3.org/2001/04/xmlenc#";

/// The XML Encryption 1.1 namespace name, `http://www.w3.org/2009/xmlenc11#`.
pub const XMLENC11_NS: &str = "http://www.w3.org/2009/xmlenc11#";

/// The XML Signature namespace name, `http://www.w3.org/2000/09/xmldsig#`, which `ds:KeyInfo`
/// and `ds:DigestMethod` live in.
pub const XMLDSIG_NS: &str = "http://www.w3.org/2000/09/xmldsig#";

/// The XML Signature 1.1 namespace name, `http://www.w3.org/2009/xmldsig11#`.
///
/// XML Signature 1.1 defines `dsig11:ECKeyValue` here for a named-curve public key inside a
/// `ds:KeyValue`. This crate also accepts the same element spelled in the [`XMLENC11_NS`]
/// namespace, which is what its own [`agreement`](crate::agreement) documentation and some
/// XML Encryption 1.1 implementations use; see [`crate::model::EcKeyValue`].
pub const XMLDSIG11_NS: &str = "http://www.w3.org/2009/xmldsig11#";

/// The `Type` attribute value for an encrypted *element*, `…xmlenc#Element`.
pub const TYPE_ELEMENT: &str = "http://www.w3.org/2001/04/xmlenc#Element";

/// The `Type` attribute value for encrypted element *content*, `…xmlenc#Content`.
pub const TYPE_CONTENT: &str = "http://www.w3.org/2001/04/xmlenc#Content";

/// The `Type` attribute value for a transported key, `…xmlenc#EncryptedKey`.
pub const TYPE_ENCRYPTED_KEY: &str = "http://www.w3.org/2001/04/xmlenc#EncryptedKey";

/// A block-encryption algorithm: what protects the data of an `EncryptedData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlockEncryption {
    /// `http://www.w3.org/2001/04/xmlenc#aes128-cbc`.
    Aes128Cbc,
    /// `http://www.w3.org/2001/04/xmlenc#aes192-cbc`.
    Aes192Cbc,
    /// `http://www.w3.org/2001/04/xmlenc#aes256-cbc`.
    Aes256Cbc,
    /// `http://www.w3.org/2009/xmlenc11#aes128-gcm`.
    Aes128Gcm,
    /// `http://www.w3.org/2009/xmlenc11#aes192-gcm`.
    Aes192Gcm,
    /// `http://www.w3.org/2009/xmlenc11#aes256-gcm`. The default, because AES-GCM is the only
    /// XML Encryption block-encryption algorithm not vulnerable to the chosen-ciphertext
    /// attacks of §6.1.1, and 256 bits is the strongest key it defines.
    #[default]
    Aes256Gcm,
}

impl BlockEncryption {
    /// Every block-encryption algorithm this crate implements.
    pub const ALL: [Self; 6] = [
        Self::Aes128Cbc,
        Self::Aes192Cbc,
        Self::Aes256Cbc,
        Self::Aes128Gcm,
        Self::Aes192Gcm,
        Self::Aes256Gcm,
    ];

    /// Returns the algorithm named by `uri`, or `None`.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://www.w3.org/2001/04/xmlenc#aes128-cbc" => Some(Self::Aes128Cbc),
            "http://www.w3.org/2001/04/xmlenc#aes192-cbc" => Some(Self::Aes192Cbc),
            "http://www.w3.org/2001/04/xmlenc#aes256-cbc" => Some(Self::Aes256Cbc),
            "http://www.w3.org/2009/xmlenc11#aes128-gcm" => Some(Self::Aes128Gcm),
            "http://www.w3.org/2009/xmlenc11#aes192-gcm" => Some(Self::Aes192Gcm),
            "http://www.w3.org/2009/xmlenc11#aes256-gcm" => Some(Self::Aes256Gcm),
            _ => None,
        }
    }

    /// Returns the canonical URI for the algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Aes128Cbc => "http://www.w3.org/2001/04/xmlenc#aes128-cbc",
            Self::Aes192Cbc => "http://www.w3.org/2001/04/xmlenc#aes192-cbc",
            Self::Aes256Cbc => "http://www.w3.org/2001/04/xmlenc#aes256-cbc",
            Self::Aes128Gcm => "http://www.w3.org/2009/xmlenc11#aes128-gcm",
            Self::Aes192Gcm => "http://www.w3.org/2009/xmlenc11#aes192-gcm",
            Self::Aes256Gcm => "http://www.w3.org/2009/xmlenc11#aes256-gcm",
        }
    }

    /// Returns the required key length in octets.
    #[must_use]
    pub const fn key_len(self) -> usize {
        match self {
            Self::Aes128Cbc | Self::Aes128Gcm => 16,
            Self::Aes192Cbc | Self::Aes192Gcm => 24,
            Self::Aes256Cbc | Self::Aes256Gcm => 32,
        }
    }

    /// Returns the key length in bits, which is what the optional `xenc:KeySize` element
    /// carries.
    #[must_use]
    pub const fn key_bits(self) -> u32 {
        (self.key_len() as u32) * 8
    }

    /// Returns `true` for the AEAD algorithms, whose ciphertext carries an authentication tag.
    #[must_use]
    pub const fn is_authenticated(self) -> bool {
        matches!(self, Self::Aes128Gcm | Self::Aes192Gcm | Self::Aes256Gcm)
    }

    /// Returns the number of octets that prefix the ciphertext: 16 for the CBC initialization
    /// vector (XMLEnc §5.2.3) and 12 for the GCM one (§5.2.4).
    #[must_use]
    pub const fn iv_len(self) -> usize {
        if self.is_authenticated() { 12 } else { 16 }
    }

    /// Returns the number of octets that suffix the ciphertext: 16 for the GCM authentication
    /// tag, zero for CBC.
    #[must_use]
    pub const fn tag_len(self) -> usize {
        if self.is_authenticated() { 16 } else { 0 }
    }
}

/// A symmetric key-wrap algorithm: what protects a content-encryption key with a shared
/// key-encryption key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyWrap {
    /// `http://www.w3.org/2001/04/xmlenc#kw-aes128` — RFC 3394 with a 128-bit key.
    Aes128,
    /// `http://www.w3.org/2001/04/xmlenc#kw-aes192` — RFC 3394 with a 192-bit key.
    Aes192,
    /// `http://www.w3.org/2001/04/xmlenc#kw-aes256` — RFC 3394 with a 256-bit key. The default.
    #[default]
    Aes256,
    /// `http://www.w3.org/2009/xmlenc11#kw-aes-128-pad` — RFC 5649 with a 128-bit key.
    Aes128Pad,
    /// `http://www.w3.org/2009/xmlenc11#kw-aes-192-pad` — RFC 5649 with a 192-bit key.
    Aes192Pad,
    /// `http://www.w3.org/2009/xmlenc11#kw-aes-256-pad` — RFC 5649 with a 256-bit key.
    Aes256Pad,
}

impl KeyWrap {
    /// Every key-wrap algorithm this crate implements.
    pub const ALL: [Self; 6] = [
        Self::Aes128,
        Self::Aes192,
        Self::Aes256,
        Self::Aes128Pad,
        Self::Aes192Pad,
        Self::Aes256Pad,
    ];

    /// Returns the algorithm named by `uri`, or `None`.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://www.w3.org/2001/04/xmlenc#kw-aes128" => Some(Self::Aes128),
            "http://www.w3.org/2001/04/xmlenc#kw-aes192" => Some(Self::Aes192),
            "http://www.w3.org/2001/04/xmlenc#kw-aes256" => Some(Self::Aes256),
            "http://www.w3.org/2009/xmlenc11#kw-aes-128-pad" => Some(Self::Aes128Pad),
            "http://www.w3.org/2009/xmlenc11#kw-aes-192-pad" => Some(Self::Aes192Pad),
            "http://www.w3.org/2009/xmlenc11#kw-aes-256-pad" => Some(Self::Aes256Pad),
            _ => None,
        }
    }

    /// Returns the canonical URI for the algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Aes128 => "http://www.w3.org/2001/04/xmlenc#kw-aes128",
            Self::Aes192 => "http://www.w3.org/2001/04/xmlenc#kw-aes192",
            Self::Aes256 => "http://www.w3.org/2001/04/xmlenc#kw-aes256",
            Self::Aes128Pad => "http://www.w3.org/2009/xmlenc11#kw-aes-128-pad",
            Self::Aes192Pad => "http://www.w3.org/2009/xmlenc11#kw-aes-192-pad",
            Self::Aes256Pad => "http://www.w3.org/2009/xmlenc11#kw-aes-256-pad",
        }
    }

    /// Returns the required key-encryption-key length in octets.
    #[must_use]
    pub const fn key_len(self) -> usize {
        match self {
            Self::Aes128 | Self::Aes128Pad => 16,
            Self::Aes192 | Self::Aes192Pad => 24,
            Self::Aes256 | Self::Aes256Pad => 32,
        }
    }

    /// Returns `true` for the RFC 5649 variants, which accept key material of any length.
    #[must_use]
    pub const fn is_padded(self) -> bool {
        matches!(self, Self::Aes128Pad | Self::Aes192Pad | Self::Aes256Pad)
    }
}

/// A public-key key-transport algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyTransport {
    /// `http://www.w3.org/2001/04/xmlenc#rsa-1_5` — RSAES-PKCS1-v1_5.
    ///
    /// XMLEnc §5.5.1 marks this NOT RECOMMENDED (Bleichenbacher). It is implemented only so
    /// that existing documents can be read.
    Rsa15,
    /// `http://www.w3.org/2001/04/xmlenc#rsa-oaep-mgf1p` — RSAES-OAEP with the mask generation
    /// function fixed to MGF1 with SHA-1. An `xenc11:MGF` element is forbidden with this URI.
    RsaOaepMgf1p,
    /// `http://www.w3.org/2009/xmlenc11#rsa-oaep` — RSAES-OAEP with the mask generation
    /// function selected by an optional `xenc11:MGF` element. The default.
    #[default]
    RsaOaep,
}

impl KeyTransport {
    /// Every key-transport algorithm this crate implements.
    pub const ALL: [Self; 3] = [Self::Rsa15, Self::RsaOaepMgf1p, Self::RsaOaep];

    /// Returns the algorithm named by `uri`, or `None`.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://www.w3.org/2001/04/xmlenc#rsa-1_5" => Some(Self::Rsa15),
            "http://www.w3.org/2001/04/xmlenc#rsa-oaep-mgf1p" => Some(Self::RsaOaepMgf1p),
            "http://www.w3.org/2009/xmlenc11#rsa-oaep" => Some(Self::RsaOaep),
            _ => None,
        }
    }

    /// Returns the canonical URI for the algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Rsa15 => "http://www.w3.org/2001/04/xmlenc#rsa-1_5",
            Self::RsaOaepMgf1p => "http://www.w3.org/2001/04/xmlenc#rsa-oaep-mgf1p",
            Self::RsaOaep => "http://www.w3.org/2009/xmlenc11#rsa-oaep",
        }
    }

    /// Returns `true` when the algorithm is one of the OAEP variants.
    #[must_use]
    pub const fn is_oaep(self) -> bool {
        matches!(self, Self::RsaOaepMgf1p | Self::RsaOaep)
    }

    /// Returns `true` when an `xenc11:MGF` child of the `EncryptionMethod` is permitted.
    ///
    /// XMLEnc 1.1 §5.5.2 forbids it for `rsa-oaep-mgf1p`, whose mask generation function is part
    /// of the identifier.
    #[must_use]
    pub const fn allows_mgf_element(self) -> bool {
        matches!(self, Self::RsaOaep)
    }
}

/// A key-agreement algorithm: how two parties reach a shared secret that becomes a
/// key-encryption key, rather than one of them encrypting a key for the other.
///
/// This is what an `xenc:AgreementMethod` inside a `ds:KeyInfo` names. The agreement itself only
/// produces a shared secret; a [`KeyDerivation`] turns that into key material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyAgreement {
    /// `http://www.w3.org/2001/04/xmlenc#dh` — finite-field Diffie–Hellman with the fixed digest
    /// construction XMLEnc 1.0 §5.5.2 defines. The key derivation is *implied* by this
    /// identifier; a `KeyDerivationMethod` does not apply to it.
    Dh,
    /// `http://www.w3.org/2009/xmlenc11#dh-es` — finite-field Diffie–Hellman in ephemeral-static
    /// mode, with the key derivation named explicitly by a `xenc11:KeyDerivationMethod`.
    DhEs,
    /// `http://www.w3.org/2009/xmlenc11#ECDH-ES` — elliptic-curve Diffie–Hellman in
    /// ephemeral-static mode, with an explicit key derivation. REQUIRED by XMLEnc 1.1, and the
    /// default here: it is the only one of the three a new document should use.
    #[default]
    EcdhEs,
}

impl KeyAgreement {
    /// Every key-agreement algorithm this crate implements.
    pub const ALL: [Self; 3] = [Self::Dh, Self::DhEs, Self::EcdhEs];

    /// Returns the algorithm named by `uri`, or `None`.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            "http://www.w3.org/2001/04/xmlenc#dh" => Some(Self::Dh),
            "http://www.w3.org/2009/xmlenc11#dh-es" => Some(Self::DhEs),
            "http://www.w3.org/2009/xmlenc11#ECDH-ES" => Some(Self::EcdhEs),
            _ => None,
        }
    }

    /// Returns the canonical URI for the algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Dh => "http://www.w3.org/2001/04/xmlenc#dh",
            Self::DhEs => "http://www.w3.org/2009/xmlenc11#dh-es",
            Self::EcdhEs => "http://www.w3.org/2009/xmlenc11#ECDH-ES",
        }
    }

    /// Returns `true` when the agreement is over an elliptic curve rather than a prime field.
    #[must_use]
    pub const fn is_elliptic(self) -> bool {
        matches!(self, Self::EcdhEs)
    }

    /// Returns `true` when a `xenc11:KeyDerivationMethod` selects the key derivation.
    ///
    /// It is `false` for the legacy `dh` identifier, whose derivation is fixed by XMLEnc 1.0
    /// §5.5.2 — an `AgreementMethod` that names `dh` *and* carries a `KeyDerivationMethod` is
    /// contradicting itself, and a decryptor that quietly obeyed the element would derive a
    /// different key from the sender.
    #[must_use]
    pub const fn uses_explicit_key_derivation(self) -> bool {
        matches!(self, Self::DhEs | Self::EcdhEs)
    }
}

/// A key-derivation algorithm: how a shared secret becomes key material.
///
/// This is what a `xenc11:KeyDerivationMethod` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KeyDerivation {
    /// `http://www.w3.org/2009/xmlenc11#ConcatKDF` — the single-step key-derivation function of
    /// NIST SP 800-56A §5.8.1. REQUIRED by XMLEnc 1.1, and the only one this crate implements;
    /// see [`crate::crypto::kdf`].
    #[default]
    ConcatKdf,
}

impl KeyDerivation {
    /// Every key-derivation algorithm this crate implements.
    pub const ALL: [Self; 1] = [Self::ConcatKdf];

    /// Returns the algorithm named by `uri`, or `None`.
    ///
    /// In particular this returns `None` for `xenc11:pbkdf2`, which is deliberately not
    /// implemented; see the [module documentation](self).
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            crate::crypto::kdf::CONCAT_KDF_URI => Some(Self::ConcatKdf),
            _ => None,
        }
    }

    /// Returns the canonical URI for the algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::ConcatKdf => crate::crypto::kdf::CONCAT_KDF_URI,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_encryption_uri_round_trips() {
        for algorithm in BlockEncryption::ALL {
            assert_eq!(BlockEncryption::from_uri(algorithm.uri()), Some(algorithm));
            assert_eq!(algorithm.key_bits() as usize, algorithm.key_len() * 8);
        }
        assert_eq!(BlockEncryption::from_uri("urn:example:rot13"), None);
    }

    #[test]
    fn every_key_wrap_uri_round_trips() {
        for algorithm in KeyWrap::ALL {
            assert_eq!(KeyWrap::from_uri(algorithm.uri()), Some(algorithm));
        }
        assert_eq!(KeyWrap::from_uri("urn:example:rot13"), None);
    }

    #[test]
    fn every_key_transport_uri_round_trips() {
        for algorithm in KeyTransport::ALL {
            assert_eq!(KeyTransport::from_uri(algorithm.uri()), Some(algorithm));
        }
        assert_eq!(KeyTransport::from_uri("urn:example:rot13"), None);
    }

    #[test]
    fn every_key_agreement_uri_round_trips() {
        for algorithm in KeyAgreement::ALL {
            assert_eq!(KeyAgreement::from_uri(algorithm.uri()), Some(algorithm));
        }
        assert_eq!(KeyAgreement::from_uri("urn:example:rot13"), None);
        // The identifier is case-sensitive, and only the 1.1 spelling is `ECDH-ES`.
        assert_eq!(
            KeyAgreement::from_uri("http://www.w3.org/2009/xmlenc11#ecdh-es"),
            None
        );
    }

    #[test]
    fn every_key_derivation_uri_round_trips() {
        for algorithm in KeyDerivation::ALL {
            assert_eq!(KeyDerivation::from_uri(algorithm.uri()), Some(algorithm));
        }
        // PBKDF2 is named by the specification and deliberately not implemented.
        assert_eq!(
            KeyDerivation::from_uri(crate::crypto::kdf::PBKDF2_URI),
            None
        );
        assert_eq!(KeyDerivation::from_uri("urn:example:kdf"), None);
    }

    #[test]
    fn only_the_1_1_agreements_take_an_explicit_key_derivation() {
        assert!(!KeyAgreement::Dh.uses_explicit_key_derivation());
        assert!(KeyAgreement::DhEs.uses_explicit_key_derivation());
        assert!(KeyAgreement::EcdhEs.uses_explicit_key_derivation());
        assert!(KeyAgreement::EcdhEs.is_elliptic());
        assert!(!KeyAgreement::Dh.is_elliptic());
        assert!(!KeyAgreement::DhEs.is_elliptic());
        assert_eq!(KeyAgreement::default(), KeyAgreement::EcdhEs);
        assert_eq!(KeyDerivation::default(), KeyDerivation::ConcatKdf);
    }

    #[test]
    fn the_families_are_disjoint() {
        for algorithm in BlockEncryption::ALL {
            assert_eq!(KeyWrap::from_uri(algorithm.uri()), None);
            assert_eq!(KeyTransport::from_uri(algorithm.uri()), None);
            assert_eq!(KeyAgreement::from_uri(algorithm.uri()), None);
            assert_eq!(KeyDerivation::from_uri(algorithm.uri()), None);
        }
        for algorithm in KeyWrap::ALL {
            assert_eq!(BlockEncryption::from_uri(algorithm.uri()), None);
            assert_eq!(KeyTransport::from_uri(algorithm.uri()), None);
            assert_eq!(KeyAgreement::from_uri(algorithm.uri()), None);
            assert_eq!(KeyDerivation::from_uri(algorithm.uri()), None);
        }
        for algorithm in KeyAgreement::ALL {
            assert_eq!(BlockEncryption::from_uri(algorithm.uri()), None);
            assert_eq!(KeyWrap::from_uri(algorithm.uri()), None);
            assert_eq!(KeyTransport::from_uri(algorithm.uri()), None);
            assert_eq!(KeyDerivation::from_uri(algorithm.uri()), None);
        }
        for algorithm in KeyDerivation::ALL {
            assert_eq!(BlockEncryption::from_uri(algorithm.uri()), None);
            assert_eq!(KeyWrap::from_uri(algorithm.uri()), None);
            assert_eq!(KeyTransport::from_uri(algorithm.uri()), None);
            assert_eq!(KeyAgreement::from_uri(algorithm.uri()), None);
        }
    }

    #[test]
    fn framing_lengths_match_the_specification() {
        for algorithm in BlockEncryption::ALL {
            if algorithm.is_authenticated() {
                assert_eq!(algorithm.iv_len(), 12);
                assert_eq!(algorithm.tag_len(), 16);
            } else {
                assert_eq!(algorithm.iv_len(), 16);
                assert_eq!(algorithm.tag_len(), 0);
            }
        }
    }

    #[test]
    fn the_defaults_are_the_recommended_algorithms() {
        assert_eq!(BlockEncryption::default(), BlockEncryption::Aes256Gcm);
        assert_eq!(KeyWrap::default(), KeyWrap::Aes256);
        assert_eq!(KeyTransport::default(), KeyTransport::RsaOaep);
    }

    #[test]
    fn only_the_1_1_oaep_identifier_allows_an_mgf_element() {
        assert!(KeyTransport::RsaOaep.allows_mgf_element());
        assert!(!KeyTransport::RsaOaepMgf1p.allows_mgf_element());
        assert!(!KeyTransport::Rsa15.allows_mgf_element());
        assert!(KeyTransport::RsaOaepMgf1p.is_oaep());
        assert!(!KeyTransport::Rsa15.is_oaep());
    }

    #[test]
    fn padded_key_wrap_is_flagged() {
        assert!(KeyWrap::Aes256Pad.is_padded());
        assert!(!KeyWrap::Aes256.is_padded());
        assert_eq!(KeyWrap::Aes192Pad.key_len(), 24);
    }
}
