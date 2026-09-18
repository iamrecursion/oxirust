//! The hash functions XML Encryption names, and the URIs that name them.
//!
//! RSA-OAEP takes two hash parameters — the message digest and the hash inside the mask
//! generation function — and XML Encryption carries each as a URI in the `EncryptionMethod`
//! element (`ds:DigestMethod/@Algorithm` and `xenc11:MGF/@Algorithm` respectively). This module
//! is the bridge between those URIs and the streaming digests of
//! [`oxixml_support::hash`], which implement FIPS 180-4 directly.
//!
//! SHA-1 is included because it is the *default* for RSA-OAEP in both XMLEnc 1.0 and 1.1 and is
//! therefore unavoidable for interoperability. Its collision resistance is broken (SHAttered,
//! 2017), but OAEP relies on SHA-1 only as a random oracle for masking and label hashing, where
//! no collision resistance is required, so `rsa-oaep-mgf1p` remains sound. It is nonetheless
//! not what a new deployment should choose; prefer `xenc11:rsa-oaep` with SHA-256.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::digest::Digest;
//!
//! let digest = Digest::from_uri("http://www.w3.org/2001/04/xmlenc#sha256")
//!     .expect("a known digest URI");
//! assert_eq!(digest, Digest::Sha256);
//! assert_eq!(digest.output_len(), 32);
//! assert_eq!(digest.hash(b"abc")[0], 0xba);
//! ```

use oxixml_support::hash;

/// The XML Signature URI for SHA-1.
pub const SHA1_URI: &str = "http://www.w3.org/2000/09/xmldsig#sha1";
/// The XML Signature (`xmldsig-more`) URI for SHA-224.
pub const SHA224_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#sha224";
/// The XML Encryption URI for SHA-256.
pub const SHA256_URI: &str = "http://www.w3.org/2001/04/xmlenc#sha256";
/// The XML Encryption URI for SHA-384, as used by XML Encryption 1.1 §5.1.1.
pub const SHA384_URI: &str = "http://www.w3.org/2001/04/xmlenc#sha384";
/// The `xmldsig-more` URI for SHA-384, which is what XML Signature 1.1 registers and what many
/// deployments emit; accepted as a synonym of [`SHA384_URI`].
pub const SHA384_DSIG_URI: &str = "http://www.w3.org/2001/04/xmldsig-more#sha384";
/// The XML Encryption URI for SHA-512.
pub const SHA512_URI: &str = "http://www.w3.org/2001/04/xmlenc#sha512";

/// One of the hash functions XML Encryption may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Digest {
    /// SHA-1 (FIPS 180-4), the RSA-OAEP default of RFC 8017 and XML Encryption.
    #[default]
    Sha1,
    /// SHA-224 (FIPS 180-4).
    Sha224,
    /// SHA-256 (FIPS 180-4).
    Sha256,
    /// SHA-384 (FIPS 180-4).
    Sha384,
    /// SHA-512 (FIPS 180-4).
    Sha512,
}

impl Digest {
    /// Every digest this crate supports, in ascending output-length order.
    pub const ALL: [Self; 5] = [
        Self::Sha1,
        Self::Sha224,
        Self::Sha256,
        Self::Sha384,
        Self::Sha512,
    ];

    /// Returns the digest named by `uri`, or `None` when the URI is not one of the recognized
    /// ones.
    #[must_use]
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            SHA1_URI => Some(Self::Sha1),
            SHA224_URI => Some(Self::Sha224),
            SHA256_URI => Some(Self::Sha256),
            SHA384_URI | SHA384_DSIG_URI => Some(Self::Sha384),
            SHA512_URI => Some(Self::Sha512),
            _ => None,
        }
    }

    /// Returns the canonical URI this crate emits for the digest.
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

    /// Returns the digest length in octets.
    #[must_use]
    pub const fn output_len(self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha224 => 28,
            Self::Sha256 => 32,
            Self::Sha384 => 48,
            Self::Sha512 => 64,
        }
    }

    /// Hashes `data` in one shot.
    #[must_use]
    pub fn hash(self, data: &[u8]) -> Vec<u8> {
        match self {
            Self::Sha1 => hash::sha1(data).to_vec(),
            Self::Sha224 => hash::sha224(data).to_vec(),
            Self::Sha256 => hash::sha256(data).to_vec(),
            Self::Sha384 => hash::sha384(data).to_vec(),
            Self::Sha512 => hash::sha512(data).to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_digest_round_trips_through_its_uri() {
        for digest in Digest::ALL {
            assert_eq!(Digest::from_uri(digest.uri()), Some(digest));
            assert_eq!(digest.hash(b"").len(), digest.output_len());
        }
    }

    #[test]
    fn the_dsig_spelling_of_sha384_is_accepted() {
        assert_eq!(Digest::from_uri(SHA384_DSIG_URI), Some(Digest::Sha384));
    }

    #[test]
    fn an_unknown_uri_is_rejected() {
        assert_eq!(Digest::from_uri("urn:example:md5"), None);
        assert_eq!(Digest::from_uri(""), None);
    }

    #[test]
    fn known_answers_from_fips_180_4() {
        // The "abc" vectors of FIPS 180-4 Appendix A/B/C/D.
        assert_eq!(Digest::Sha1.hash(b"abc")[..4], [0xa9, 0x99, 0x3e, 0x36]);
        assert_eq!(Digest::Sha224.hash(b"abc")[..4], [0x23, 0x09, 0x7d, 0x22]);
        assert_eq!(Digest::Sha256.hash(b"abc")[..4], [0xba, 0x78, 0x16, 0xbf]);
        assert_eq!(Digest::Sha384.hash(b"abc")[..4], [0xcb, 0x00, 0x75, 0x3f]);
        assert_eq!(Digest::Sha512.hash(b"abc")[..4], [0xdd, 0xaf, 0x35, 0xa1]);
    }

    #[test]
    fn the_default_is_the_oaep_default() {
        assert_eq!(Digest::default(), Digest::Sha1);
    }
}
