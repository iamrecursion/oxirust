//! The MGF1 mask generation function of [RFC 8017 Appendix B.2.1](https://www.rfc-editor.org/rfc/rfc8017#appendix-B.2.1).
//!
//! MGF1 stretches a seed into a mask of arbitrary length by hashing the seed concatenated with
//! a 32-bit big-endian counter and concatenating the results:
//! `MGF1(Z, l) = Hash(Z ‖ 0) ‖ Hash(Z ‖ 1) ‖ … ` truncated to `l` octets. It is the only mask
//! generation function XML Encryption defines, and the `xenc11:MGF/@Algorithm` URIs of
//! XMLEnc 1.1 §5.5.2 select nothing but which hash it uses.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::digest::Digest;
//! use oxixml_enc::crypto::mgf1::{mgf1, MGF1_SHA256_URI, mgf1_digest_from_uri};
//!
//! let mask = mgf1(Digest::Sha1, b"bar", 50);
//! assert_eq!(mask.len(), 50);
//! assert_eq!(&mask[..4], &[0xbc, 0x0c, 0x65, 0x5e]);
//!
//! assert_eq!(mgf1_digest_from_uri(MGF1_SHA256_URI), Some(Digest::Sha256));
//! ```

use crate::crypto::digest::Digest;

/// `MGF1 with SHA-1`, XML Encryption 1.1 §5.5.2.
pub const MGF1_SHA1_URI: &str = "http://www.w3.org/2009/xmlenc11#mgf1sha1";
/// `MGF1 with SHA-224`, XML Encryption 1.1 §5.5.2.
pub const MGF1_SHA224_URI: &str = "http://www.w3.org/2009/xmlenc11#mgf1sha224";
/// `MGF1 with SHA-256`, XML Encryption 1.1 §5.5.2.
pub const MGF1_SHA256_URI: &str = "http://www.w3.org/2009/xmlenc11#mgf1sha256";
/// `MGF1 with SHA-384`, XML Encryption 1.1 §5.5.2.
pub const MGF1_SHA384_URI: &str = "http://www.w3.org/2009/xmlenc11#mgf1sha384";
/// `MGF1 with SHA-512`, XML Encryption 1.1 §5.5.2.
pub const MGF1_SHA512_URI: &str = "http://www.w3.org/2009/xmlenc11#mgf1sha512";
/// The XML Encryption 1.0 spelling `xenc:MGF1withSHA1`, which XMLEnc 1.1 Example 33 still uses;
/// accepted as a synonym of [`MGF1_SHA1_URI`].
pub const MGF1_SHA1_LEGACY_URI: &str = "http://www.w3.org/2001/04/xmlenc#MGF1withSHA1";

/// Returns the hash a mask-generation-function URI selects, or `None` for an unknown URI.
#[must_use]
pub fn mgf1_digest_from_uri(uri: &str) -> Option<Digest> {
    match uri {
        MGF1_SHA1_URI | MGF1_SHA1_LEGACY_URI => Some(Digest::Sha1),
        MGF1_SHA224_URI => Some(Digest::Sha224),
        MGF1_SHA256_URI => Some(Digest::Sha256),
        MGF1_SHA384_URI => Some(Digest::Sha384),
        MGF1_SHA512_URI => Some(Digest::Sha512),
        _ => None,
    }
}

/// Returns the canonical mask-generation-function URI for `digest`.
#[must_use]
pub const fn mgf1_uri(digest: Digest) -> &'static str {
    match digest {
        Digest::Sha1 => MGF1_SHA1_URI,
        Digest::Sha224 => MGF1_SHA224_URI,
        Digest::Sha256 => MGF1_SHA256_URI,
        Digest::Sha384 => MGF1_SHA384_URI,
        Digest::Sha512 => MGF1_SHA512_URI,
    }
}

/// Generates a `length`-octet mask from `seed` (RFC 8017 Appendix B.2.1).
///
/// RFC 8017 requires `length ≤ 2^32 · hLen`, which no allocation on a 64-bit host can violate
/// in practice and which a 32-bit host cannot express at all, so the bound is not checked
/// separately: a `length` that large would fail to allocate first.
#[must_use]
pub fn mgf1(digest: Digest, seed: &[u8], length: usize) -> Vec<u8> {
    let mut mask = Vec::with_capacity(length);
    let mut counter = 0u32;
    let mut buffer = Vec::with_capacity(seed.len() + 4);
    while mask.len() < length {
        buffer.clear();
        buffer.extend_from_slice(seed);
        buffer.extend_from_slice(&counter.to_be_bytes());
        mask.extend_from_slice(&digest.hash(&buffer));
        counter = counter.wrapping_add(1);
    }
    mask.truncate(length);
    mask
}

/// Exclusive-ors a `mask` derived from `seed` into `data`, in place.
///
/// This is the shape OAEP uses the function in four times; folding the allocation and the loop
/// into one place keeps the OAEP code readable.
pub fn mask_in_place(digest: Digest, seed: &[u8], data: &mut [u8]) {
    let mask = mgf1(digest, seed, data.len());
    for (byte, mask_byte) in data.iter_mut().zip(mask.iter()) {
        *byte ^= *mask_byte;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_uri_round_trips() {
        for digest in Digest::ALL {
            assert_eq!(mgf1_digest_from_uri(mgf1_uri(digest)), Some(digest));
        }
        assert_eq!(
            mgf1_digest_from_uri(MGF1_SHA1_LEGACY_URI),
            Some(Digest::Sha1)
        );
        assert_eq!(mgf1_digest_from_uri("urn:example:mgf"), None);
    }

    #[test]
    fn a_mask_is_a_prefix_of_a_longer_mask() {
        // MGF1 is defined by concatenation, so shortening the request truncates the output.
        let long = mgf1(Digest::Sha256, b"seed", 100);
        let short = mgf1(Digest::Sha256, b"seed", 33);
        assert_eq!(&long[..33], short.as_slice());
    }

    #[test]
    fn zero_length_is_empty() {
        assert!(mgf1(Digest::Sha1, b"seed", 0).is_empty());
    }

    #[test]
    fn masking_is_an_involution() {
        let original = *b"the quick brown fox";
        let mut data = original;
        mask_in_place(Digest::Sha384, b"seed", &mut data);
        assert_ne!(data, original);
        mask_in_place(Digest::Sha384, b"seed", &mut data);
        assert_eq!(data, original);
    }
}
