//! Algorithm identifiers — the URIs that appear in `Algorithm` attributes.
//!
//! [XML Signature §6.1](https://www.w3.org/TR/xmldsig-core1/#sec-AlgID) names
//! every cryptographic and transform algorithm with a URI. This module turns
//! those URIs into closed enums so that the rest of the crate never compares
//! strings, and so that an unimplemented-but-known algorithm ([`DsigError::UnsupportedAlgorithm`])
//! can be told apart from one nobody has heard of ([`DsigError::UnknownAlgorithm`]).
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{DigestAlgorithm, SignatureAlgorithm, TransformAlgorithm};
//!
//! let digest = DigestAlgorithm::from_uri("http://www.w3.org/2001/04/xmlenc#sha256")?;
//! assert_eq!(digest, DigestAlgorithm::Sha256);
//! assert_eq!(digest.output_length(), 32);
//!
//! let method = SignatureAlgorithm::from_uri("http://www.w3.org/2000/09/xmldsig#rsa-sha1")?;
//! assert_eq!(method.digest(), DigestAlgorithm::Sha1);
//! assert!(method.is_rsa());
//!
//! let transform =
//!     TransformAlgorithm::from_uri("http://www.w3.org/2000/09/xmldsig#enveloped-signature")?;
//! assert_eq!(transform, TransformAlgorithm::EnvelopedSignature);
//! # Ok::<(), oxixml_dsig::DsigError>(())
//! ```

use std::fmt;
use std::str::FromStr;

use oxixml_c14n::CanonicalizationAlgorithm;

use crate::error::{DsigError, DsigResult};

/// The `http://www.w3.org/2000/09/xmldsig#` namespace name.
pub const DSIG_NAMESPACE: &str = "http://www.w3.org/2000/09/xmldsig#";
/// The `http://www.w3.org/2009/xmldsig11#` namespace name, new in XML Signature 1.1.
pub const DSIG11_NAMESPACE: &str = "http://www.w3.org/2009/xmldsig11#";
/// The namespace name of the [XPath Filter 2.0](https://www.w3.org/TR/xmldsig-filter2/)
/// transform parameters.
pub const FILTER2_NAMESPACE: &str = "http://www.w3.org/2002/06/xmldsig-filter2";
/// The `http://www.w3.org/2001/10/xml-exc-c14n#` namespace, home of `InclusiveNamespaces`.
pub const EXC_C14N_NAMESPACE: &str = "http://www.w3.org/2001/10/xml-exc-c14n#";
/// The `http://www.w3.org/2007/05/xmldsig-more#` namespace of
/// [RFC 9231](https://www.rfc-editor.org/rfc/rfc9231), home of `RSAPSSParams`.
pub const XMLDSIG_MORE_NAMESPACE: &str = "http://www.w3.org/2007/05/xmldsig-more#";
/// `http://www.w3.org/2007/05/xmldsig-more#MGF1` — the only mask generation
/// function [RFC 8017](https://www.rfc-editor.org/rfc/rfc8017) defines.
pub const MGF1_URI: &str = "http://www.w3.org/2007/05/xmldsig-more#MGF1";

/// The `Type` a `Reference` uses for a `<ds:Object>`.
pub const OBJECT_TYPE: &str = "http://www.w3.org/2000/09/xmldsig#Object";
/// The `Type` a `Reference` uses for a `<ds:Manifest>`.
pub const MANIFEST_TYPE: &str = "http://www.w3.org/2000/09/xmldsig#Manifest";
/// The `Type` a `Reference` uses for a `<ds:SignatureProperties>`.
pub const SIGNATURE_PROPERTIES_TYPE: &str = "http://www.w3.org/2000/09/xmldsig#SignatureProperties";

// --- digest algorithms ----------------------------------------------------------------------

/// `http://www.w3.org/2000/09/xmldsig#sha1`
pub const SHA1_URI: &str = "http://www.w3.org/2000/09/xmldsig#sha1";
/// `http://www.w3.org/2001/04/xmldsig-more#sha224`
pub const SHA224_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#sha224";
/// `http://www.w3.org/2001/04/xmlenc#sha256`
pub const SHA256_URI: &str = "http://www.w3.org/2001/04/xmlenc#sha256";
/// `http://www.w3.org/2001/04/xmldsig-more#sha384`
pub const SHA384_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#sha384";
/// `http://www.w3.org/2001/04/xmlenc#sha512`
pub const SHA512_URI: &str = "http://www.w3.org/2001/04/xmlenc#sha512";

/// A message digest, as named by a `<ds:DigestMethod Algorithm="…">`.
///
/// [XML Signature §6.2](https://www.w3.org/TR/xmldsig-core1/#sec-MessageDigests)
/// makes SHA-256 REQUIRED and the rest of the SHA-2 family plus SHA-1
/// OPTIONAL; all five are implemented here on top of
/// [`oxixml_support::hash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DigestAlgorithm {
    /// SHA-1 — [`SHA1_URI`]. Broken for collision resistance; see [the security notes](crate#security).
    Sha1,
    /// SHA-224 — [`SHA224_URI`].
    Sha224,
    /// SHA-256 — [`SHA256_URI`].
    Sha256,
    /// SHA-384 — [`SHA384_URI`].
    Sha384,
    /// SHA-512 — [`SHA512_URI`].
    Sha512,
}

impl DigestAlgorithm {
    /// Every digest this crate implements, in increasing strength.
    pub const ALL: [Self; 5] = [
        Self::Sha1,
        Self::Sha224,
        Self::Sha256,
        Self::Sha384,
        Self::Sha512,
    ];

    /// Parses a `<ds:DigestMethod>` `Algorithm` URI.
    ///
    /// # Errors
    ///
    /// [`DsigError::UnknownAlgorithm`] if the URI is not a digest identifier
    /// this crate knows.
    pub fn from_uri(uri: &str) -> DsigResult<Self> {
        match uri {
            SHA1_URI => Ok(Self::Sha1),
            SHA224_URI => Ok(Self::Sha224),
            SHA256_URI => Ok(Self::Sha256),
            SHA384_URI => Ok(Self::Sha384),
            SHA512_URI => Ok(Self::Sha512),
            other => Err(DsigError::UnknownAlgorithm(other.to_owned())),
        }
    }

    /// The `Algorithm` URI for this digest.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Sha1 => SHA1_URI,
            Self::Sha224 => SHA224_URI,
            Self::Sha256 => SHA256_URI,
            Self::Sha384 => SHA384_URI,
            Self::Sha512 => SHA512_URI,
        }
    }

    /// The digest length in octets.
    #[must_use]
    pub const fn output_length(self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha224 => 28,
            Self::Sha256 => 32,
            Self::Sha384 => 48,
            Self::Sha512 => 64,
        }
    }

    /// Computes the digest of `data`.
    #[must_use]
    pub fn digest(self, data: &[u8]) -> Vec<u8> {
        crate::crypto::digest::digest(self, data)
    }
}

impl fmt::Display for DigestAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.uri())
    }
}

impl FromStr for DigestAlgorithm {
    type Err = DsigError;

    fn from_str(uri: &str) -> DsigResult<Self> {
        Self::from_uri(uri)
    }
}

// --- signature algorithms -------------------------------------------------------------------

/// `http://www.w3.org/2000/09/xmldsig#hmac-sha1`
pub const HMAC_SHA1_URI: &str = "http://www.w3.org/2000/09/xmldsig#hmac-sha1";
/// `http://www.w3.org/2001/04/xmldsig-more#hmac-sha224`
pub const HMAC_SHA224_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#hmac-sha224";
/// `http://www.w3.org/2001/04/xmldsig-more#hmac-sha256`
pub const HMAC_SHA256_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#hmac-sha256";
/// `http://www.w3.org/2001/04/xmldsig-more#hmac-sha384`
pub const HMAC_SHA384_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#hmac-sha384";
/// `http://www.w3.org/2001/04/xmldsig-more#hmac-sha512`
pub const HMAC_SHA512_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#hmac-sha512";

/// `http://www.w3.org/2000/09/xmldsig#rsa-sha1`
pub const RSA_SHA1_URI: &str = "http://www.w3.org/2000/09/xmldsig#rsa-sha1";
/// `http://www.w3.org/2001/04/xmldsig-more#rsa-sha224`
pub const RSA_SHA224_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha224";
/// `http://www.w3.org/2001/04/xmldsig-more#rsa-sha256`
pub const RSA_SHA256_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256";
/// `http://www.w3.org/2001/04/xmldsig-more#rsa-sha384`
pub const RSA_SHA384_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha384";
/// `http://www.w3.org/2001/04/xmldsig-more#rsa-sha512`
pub const RSA_SHA512_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha512";

/// `http://www.w3.org/2000/09/xmldsig#dsa-sha1`
pub const DSA_SHA1_URI: &str = "http://www.w3.org/2000/09/xmldsig#dsa-sha1";
/// `http://www.w3.org/2009/xmldsig11#dsa-sha256`
pub const DSA_SHA256_URI: &str = "http://www.w3.org/2009/xmldsig11#dsa-sha256";
/// `http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha1`
pub const ECDSA_SHA1_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha1";
/// `http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha224`
pub const ECDSA_SHA224_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha224";
/// `http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha256`
pub const ECDSA_SHA256_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha256";
/// `http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha384`
pub const ECDSA_SHA384_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha384";
/// `http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha512`
pub const ECDSA_SHA512_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha512";
/// `http://www.w3.org/2007/05/xmldsig-more#rsa-pss`
pub const RSA_PSS_URI: &str = "http://www.w3.org/2007/05/xmldsig-more#rsa-pss";

/// The family a [`SignatureAlgorithm`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum SignatureFamily {
    /// Keyed-hash message authentication, [RFC 2104](https://www.rfc-editor.org/rfc/rfc2104).
    /// Symmetric: the verifier can also forge.
    Hmac,
    /// RSASSA-PKCS1-v1_5, [RFC 8017 §8.2](https://www.rfc-editor.org/rfc/rfc8017#section-8.2).
    RsaPkcs1V15,
    /// RSASSA-PSS, [RFC 8017 §8.1](https://www.rfc-editor.org/rfc/rfc8017#section-8.1),
    /// parameterised as [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9)
    /// describes.
    RsaPss,
    /// DSA, [FIPS 186-4 §4](https://doi.org/10.6028/NIST.FIPS.186-4).
    Dsa,
    /// ECDSA over P-256 or P-384,
    /// [FIPS 186-4 §6](https://doi.org/10.6028/NIST.FIPS.186-4).
    Ecdsa,
}

/// A signature or MAC algorithm, as named by `<ds:SignatureMethod Algorithm="…">`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum SignatureAlgorithm {
    /// HMAC over SHA-1 — [`HMAC_SHA1_URI`].
    HmacSha1,
    /// HMAC over SHA-224 — [`HMAC_SHA224_URI`].
    HmacSha224,
    /// HMAC over SHA-256 — [`HMAC_SHA256_URI`].
    HmacSha256,
    /// HMAC over SHA-384 — [`HMAC_SHA384_URI`].
    HmacSha384,
    /// HMAC over SHA-512 — [`HMAC_SHA512_URI`].
    HmacSha512,
    /// RSASSA-PKCS1-v1_5 over SHA-1 — [`RSA_SHA1_URI`].
    RsaSha1,
    /// RSASSA-PKCS1-v1_5 over SHA-224 — [`RSA_SHA224_URI`].
    RsaSha224,
    /// RSASSA-PKCS1-v1_5 over SHA-256 — [`RSA_SHA256_URI`].
    RsaSha256,
    /// RSASSA-PKCS1-v1_5 over SHA-384 — [`RSA_SHA384_URI`].
    RsaSha384,
    /// RSASSA-PKCS1-v1_5 over SHA-512 — [`RSA_SHA512_URI`].
    RsaSha512,
    /// RSASSA-PSS — [`RSA_PSS_URI`]. The digest and the other parameters come
    /// from the method's `<pss:RSAPSSParams>` child, not from the identifier;
    /// see [`PssParameters`](crate::PssParameters).
    RsaPss,
    /// DSA over SHA-1 — [`DSA_SHA1_URI`].
    DsaSha1,
    /// DSA over SHA-256 — [`DSA_SHA256_URI`].
    DsaSha256,
    /// ECDSA over SHA-1 — [`ECDSA_SHA1_URI`].
    EcdsaSha1,
    /// ECDSA over SHA-224 — [`ECDSA_SHA224_URI`].
    EcdsaSha224,
    /// ECDSA over SHA-256 — [`ECDSA_SHA256_URI`].
    EcdsaSha256,
    /// ECDSA over SHA-384 — [`ECDSA_SHA384_URI`].
    EcdsaSha384,
    /// ECDSA over SHA-512 — [`ECDSA_SHA512_URI`].
    EcdsaSha512,
}

impl SignatureAlgorithm {
    /// Parses a `<ds:SignatureMethod>` `Algorithm` URI.
    ///
    /// # Errors
    ///
    /// [`DsigError::UnknownAlgorithm`] for a URI outside the set above.
    /// Recognising a URI does not imply support: check
    /// [`is_supported`](SignatureAlgorithm::is_supported).
    pub fn from_uri(uri: &str) -> DsigResult<Self> {
        match uri {
            HMAC_SHA1_URI => Ok(Self::HmacSha1),
            HMAC_SHA224_URI => Ok(Self::HmacSha224),
            HMAC_SHA256_URI => Ok(Self::HmacSha256),
            HMAC_SHA384_URI => Ok(Self::HmacSha384),
            HMAC_SHA512_URI => Ok(Self::HmacSha512),
            RSA_SHA1_URI => Ok(Self::RsaSha1),
            RSA_SHA224_URI => Ok(Self::RsaSha224),
            RSA_SHA256_URI => Ok(Self::RsaSha256),
            RSA_SHA384_URI => Ok(Self::RsaSha384),
            RSA_SHA512_URI => Ok(Self::RsaSha512),
            RSA_PSS_URI => Ok(Self::RsaPss),
            DSA_SHA1_URI => Ok(Self::DsaSha1),
            DSA_SHA256_URI => Ok(Self::DsaSha256),
            ECDSA_SHA1_URI => Ok(Self::EcdsaSha1),
            ECDSA_SHA224_URI => Ok(Self::EcdsaSha224),
            ECDSA_SHA256_URI => Ok(Self::EcdsaSha256),
            ECDSA_SHA384_URI => Ok(Self::EcdsaSha384),
            ECDSA_SHA512_URI => Ok(Self::EcdsaSha512),
            other => Err(DsigError::UnknownAlgorithm(other.to_owned())),
        }
    }

    /// The `Algorithm` URI for this method.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::HmacSha1 => HMAC_SHA1_URI,
            Self::HmacSha224 => HMAC_SHA224_URI,
            Self::HmacSha256 => HMAC_SHA256_URI,
            Self::HmacSha384 => HMAC_SHA384_URI,
            Self::HmacSha512 => HMAC_SHA512_URI,
            Self::RsaSha1 => RSA_SHA1_URI,
            Self::RsaSha224 => RSA_SHA224_URI,
            Self::RsaSha256 => RSA_SHA256_URI,
            Self::RsaSha384 => RSA_SHA384_URI,
            Self::RsaSha512 => RSA_SHA512_URI,
            Self::RsaPss => RSA_PSS_URI,
            Self::DsaSha1 => DSA_SHA1_URI,
            Self::DsaSha256 => DSA_SHA256_URI,
            Self::EcdsaSha1 => ECDSA_SHA1_URI,
            Self::EcdsaSha224 => ECDSA_SHA224_URI,
            Self::EcdsaSha256 => ECDSA_SHA256_URI,
            Self::EcdsaSha384 => ECDSA_SHA384_URI,
            Self::EcdsaSha512 => ECDSA_SHA512_URI,
        }
    }

    /// The digest the method hashes the canonical `SignedInfo` with.
    ///
    /// [`SignatureAlgorithm::RsaPss`] has no digest baked into its identifier —
    /// the `<ds:DigestMethod>` inside the method's `<pss:RSAPSSParams>` carries
    /// it — so SHA-256, the default
    /// [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9)
    /// names, is reported. Read
    /// [`SignatureMethod::effective_pss_parameters`](crate::SignatureMethod::effective_pss_parameters)
    /// for the digest a particular signature actually declared.
    #[must_use]
    pub const fn digest(self) -> DigestAlgorithm {
        match self {
            Self::HmacSha1 | Self::RsaSha1 | Self::DsaSha1 | Self::EcdsaSha1 => {
                DigestAlgorithm::Sha1
            }
            Self::HmacSha224 | Self::RsaSha224 | Self::EcdsaSha224 => DigestAlgorithm::Sha224,
            Self::HmacSha256
            | Self::RsaSha256
            | Self::DsaSha256
            | Self::EcdsaSha256
            | Self::RsaPss => DigestAlgorithm::Sha256,
            Self::HmacSha384 | Self::RsaSha384 | Self::EcdsaSha384 => DigestAlgorithm::Sha384,
            Self::HmacSha512 | Self::RsaSha512 | Self::EcdsaSha512 => DigestAlgorithm::Sha512,
        }
    }

    /// Which cryptosystem the method belongs to.
    #[must_use]
    pub const fn family(self) -> SignatureFamily {
        match self {
            Self::HmacSha1
            | Self::HmacSha224
            | Self::HmacSha256
            | Self::HmacSha384
            | Self::HmacSha512 => SignatureFamily::Hmac,
            Self::RsaSha1
            | Self::RsaSha224
            | Self::RsaSha256
            | Self::RsaSha384
            | Self::RsaSha512 => SignatureFamily::RsaPkcs1V15,
            Self::RsaPss => SignatureFamily::RsaPss,
            Self::DsaSha1 | Self::DsaSha256 => SignatureFamily::Dsa,
            Self::EcdsaSha1
            | Self::EcdsaSha224
            | Self::EcdsaSha256
            | Self::EcdsaSha384
            | Self::EcdsaSha512 => SignatureFamily::Ecdsa,
        }
    }

    /// Whether the method is a keyed hash rather than a public-key signature.
    #[must_use]
    pub const fn is_hmac(self) -> bool {
        matches!(self.family(), SignatureFamily::Hmac)
    }

    /// Whether the method is RSASSA-PKCS1-v1_5.
    #[must_use]
    pub const fn is_rsa(self) -> bool {
        matches!(self.family(), SignatureFamily::RsaPkcs1V15)
    }

    /// Whether this crate can compute and check the method.
    ///
    /// True for every identifier this enum has, since XML Signature 1.1's whole
    /// algorithm set is implemented. It stays on the public API because support
    /// is not only a property of the *method*: an ECDSA method still needs a key
    /// on P-256 or P-384, and a DSA method a key whose subgroup order fits the
    /// signature width — both of which are decided when the key is read.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(
            self.family(),
            SignatureFamily::Hmac
                | SignatureFamily::RsaPkcs1V15
                | SignatureFamily::RsaPss
                | SignatureFamily::Dsa
                | SignatureFamily::Ecdsa
        )
    }

    /// The octet width of each half of a DSA `SignatureValue`.
    ///
    /// [XML Signature §6.4.1](https://www.w3.org/TR/xmldsig-core1/#sec-DSA)
    /// fixes `r` and `s` at 20 octets each for `#dsa-sha1`, and
    /// [§6.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-DSAwithSHA256) at 32
    /// for `xmldsig11#dsa-sha256` — a property of the *identifier*, not of the
    /// key, so a key with a narrower `q` is left-padded into it.
    ///
    /// Returns [`None`] for methods that are not DSA. ECDSA takes its width from
    /// the curve instead; see [XML Signature §6.4.3](https://www.w3.org/TR/xmldsig-core1/#sec-ECDSA).
    #[must_use]
    pub const fn dsa_signature_half_width(self) -> Option<usize> {
        match self {
            Self::DsaSha1 => Some(20),
            Self::DsaSha256 => Some(32),
            _ => None,
        }
    }

    /// The smallest `HMACOutputLength`, in bits, that
    /// [XML Signature §4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod)
    /// allows for this method: "the larger of (a) half the underlying hash
    /// algorithm's output length, and (b) 80 bits".
    ///
    /// Returns [`None`] for methods that are not HMAC.
    #[must_use]
    pub const fn minimum_hmac_output_length(self) -> Option<usize> {
        if self.is_hmac() {
            let half = self.digest().output_length() * 4;
            Some(if half > 80 { half } else { 80 })
        } else {
            None
        }
    }
}

impl fmt::Display for SignatureAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.uri())
    }
}

impl FromStr for SignatureAlgorithm {
    type Err = DsigError;

    fn from_str(uri: &str) -> DsigResult<Self> {
        Self::from_uri(uri)
    }
}

// --- transforms -----------------------------------------------------------------------------

/// `http://www.w3.org/2000/09/xmldsig#enveloped-signature`
pub const ENVELOPED_SIGNATURE_URI: &str = "http://www.w3.org/2000/09/xmldsig#enveloped-signature";
/// `http://www.w3.org/2000/09/xmldsig#base64`
pub const BASE64_TRANSFORM_URI: &str = "http://www.w3.org/2000/09/xmldsig#base64";
/// `http://www.w3.org/TR/1999/REC-xpath-19991116`
pub const XPATH_TRANSFORM_URI: &str = "http://www.w3.org/TR/1999/REC-xpath-19991116";
/// `http://www.w3.org/2002/06/xmldsig-filter2`
pub const XPATH_FILTER2_URI: &str = "http://www.w3.org/2002/06/xmldsig-filter2";
/// `http://www.w3.org/TR/1999/REC-xslt-19991116` — recognised, not implemented.
pub const XSLT_TRANSFORM_URI: &str = "http://www.w3.org/TR/1999/REC-xslt-19991116";

/// A transform, as named by `<ds:Transform Algorithm="…">`.
///
/// The canonicalization algorithms double as transforms, so
/// [`TransformAlgorithm::Canonicalization`] simply wraps
/// [`oxixml_c14n::CanonicalizationAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformAlgorithm {
    /// One of the canonicalization algorithms of
    /// [XML Signature §6.5](https://www.w3.org/TR/xmldsig-core1/#sec-CanonicalizationAlg),
    /// used in transform position.
    Canonicalization(CanonicalizationAlgorithm),
    /// The enveloped signature transform — [`ENVELOPED_SIGNATURE_URI`].
    EnvelopedSignature,
    /// The base64 decoding transform — [`BASE64_TRANSFORM_URI`].
    Base64,
    /// The XPath filtering transform — [`XPATH_TRANSFORM_URI`].
    XPath,
    /// The XPath Filter 2.0 transform — [`XPATH_FILTER2_URI`].
    XPathFilter2,
    /// The XSLT transform — [`XSLT_TRANSFORM_URI`]. Recognised only; running a
    /// stylesheet is out of scope for this crate. See [the gap list](crate#gaps).
    Xslt,
}

impl TransformAlgorithm {
    /// Parses a `<ds:Transform>` `Algorithm` URI.
    ///
    /// # Errors
    ///
    /// [`DsigError::UnknownAlgorithm`] for a URI that is neither a transform
    /// identifier nor a canonicalization identifier.
    pub fn from_uri(uri: &str) -> DsigResult<Self> {
        match uri {
            ENVELOPED_SIGNATURE_URI => Ok(Self::EnvelopedSignature),
            BASE64_TRANSFORM_URI => Ok(Self::Base64),
            XPATH_TRANSFORM_URI => Ok(Self::XPath),
            XPATH_FILTER2_URI => Ok(Self::XPathFilter2),
            XSLT_TRANSFORM_URI => Ok(Self::Xslt),
            other => CanonicalizationAlgorithm::from_uri(other)
                .map(Self::Canonicalization)
                .map_err(|_| DsigError::UnknownAlgorithm(other.to_owned())),
        }
    }

    /// The `Algorithm` URI for this transform.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Canonicalization(algorithm) => algorithm.uri(),
            Self::EnvelopedSignature => ENVELOPED_SIGNATURE_URI,
            Self::Base64 => BASE64_TRANSFORM_URI,
            Self::XPath => XPATH_TRANSFORM_URI,
            Self::XPathFilter2 => XPATH_FILTER2_URI,
            Self::Xslt => XSLT_TRANSFORM_URI,
        }
    }

    /// Whether this crate can run the transform.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        !matches!(self, Self::Xslt)
    }
}

impl fmt::Display for TransformAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.uri())
    }
}

impl FromStr for TransformAlgorithm {
    type Err = DsigError;

    fn from_str(uri: &str) -> DsigResult<Self> {
        Self::from_uri(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_uris_round_trip() -> DsigResult<()> {
        for algorithm in DigestAlgorithm::ALL {
            assert_eq!(DigestAlgorithm::from_uri(algorithm.uri())?, algorithm);
        }
        Ok(())
    }

    #[test]
    fn hmac_floor_follows_the_recommendation() {
        // max(half the hash length in bits, 80).
        assert_eq!(
            SignatureAlgorithm::HmacSha1.minimum_hmac_output_length(),
            Some(80)
        );
        assert_eq!(
            SignatureAlgorithm::HmacSha256.minimum_hmac_output_length(),
            Some(128)
        );
        assert_eq!(
            SignatureAlgorithm::HmacSha512.minimum_hmac_output_length(),
            Some(256)
        );
        assert_eq!(
            SignatureAlgorithm::RsaSha256.minimum_hmac_output_length(),
            None
        );
    }

    #[test]
    fn every_signature_method_is_implemented() -> DsigResult<()> {
        for uri in [
            HMAC_SHA1_URI,
            RSA_SHA256_URI,
            RSA_PSS_URI,
            DSA_SHA1_URI,
            DSA_SHA256_URI,
            ECDSA_SHA1_URI,
            ECDSA_SHA224_URI,
            ECDSA_SHA256_URI,
            ECDSA_SHA384_URI,
            ECDSA_SHA512_URI,
        ] {
            let algorithm = SignatureAlgorithm::from_uri(uri)?;
            assert!(algorithm.is_supported(), "{uri}");
            assert_eq!(algorithm.uri(), uri);
        }
        // The XSLT transform, by contrast, is recognised and refused.
        let xslt = TransformAlgorithm::from_uri(XSLT_TRANSFORM_URI)?;
        assert!(!xslt.is_supported());
        Ok(())
    }

    #[test]
    fn the_dsa_signature_width_comes_from_the_identifier() {
        assert_eq!(
            SignatureAlgorithm::DsaSha1.dsa_signature_half_width(),
            Some(20)
        );
        assert_eq!(
            SignatureAlgorithm::DsaSha256.dsa_signature_half_width(),
            Some(32)
        );
        assert_eq!(
            SignatureAlgorithm::EcdsaSha256.dsa_signature_half_width(),
            None
        );
    }

    #[test]
    fn canonicalization_doubles_as_a_transform() -> DsigResult<()> {
        let transform = TransformAlgorithm::from_uri(oxixml_c14n::EXCLUSIVE_XML_C14N_1_0)?;
        assert!(matches!(transform, TransformAlgorithm::Canonicalization(_)));
        assert!(transform.is_supported());
        Ok(())
    }

    #[test]
    fn an_unknown_uri_is_reported_as_such() {
        assert!(matches!(
            DigestAlgorithm::from_uri("urn:nope"),
            Err(DsigError::UnknownAlgorithm(_))
        ));
    }
}
