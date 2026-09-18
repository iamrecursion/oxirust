//! Digest dispatch on top of [`oxixml_support::hash`].

use oxixml_support::{bytes, hash};

use crate::algorithm::DigestAlgorithm;

/// Computes the digest of `data` with `algorithm`.
#[must_use]
pub(crate) fn digest(algorithm: DigestAlgorithm, data: &[u8]) -> Vec<u8> {
    match algorithm {
        DigestAlgorithm::Sha1 => hash::sha1(data).to_vec(),
        DigestAlgorithm::Sha224 => hash::sha224(data).to_vec(),
        DigestAlgorithm::Sha256 => hash::sha256(data).to_vec(),
        DigestAlgorithm::Sha384 => hash::sha384(data).to_vec(),
        DigestAlgorithm::Sha512 => hash::sha512(data).to_vec(),
    }
}

/// Computes `HMAC(key, message)` with the hash `algorithm` names.
#[must_use]
pub(crate) fn hmac(algorithm: DigestAlgorithm, key: &[u8], message: &[u8]) -> Vec<u8> {
    match algorithm {
        DigestAlgorithm::Sha1 => hash::hmac_sha1(key, message).to_vec(),
        DigestAlgorithm::Sha224 => {
            let mut mac = hash::Hmac::<hash::Sha224>::new(key);
            mac.update(message);
            mac.finalize()
        }
        DigestAlgorithm::Sha256 => hash::hmac_sha256(key, message).to_vec(),
        DigestAlgorithm::Sha384 => hash::hmac_sha384(key, message).to_vec(),
        DigestAlgorithm::Sha512 => hash::hmac_sha512(key, message).to_vec(),
    }
}

/// Compares two byte strings without an early exit.
///
/// Digest and MAC comparisons are the classic place where a `==` leaks the
/// position of the first differing octet, which is enough to forge a MAC one
/// octet at a time. This delegates to [`oxixml_support::bytes::ct_eq`], the one
/// constant-time comparison every crate in this workspace checks a digest, MAC
/// or padding against, rather than keeping a second implementation here.
#[must_use]
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    bytes::ct_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips_180_4_abc_vectors() {
        // FIPS 180-4 one-block message "abc".
        assert_eq!(
            digest(DigestAlgorithm::Sha1, b"abc"),
            [
                0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50,
                0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d
            ]
        );
        assert_eq!(
            digest(DigestAlgorithm::Sha224, b"abc"),
            [
                0x23, 0x09, 0x7d, 0x22, 0x34, 0x05, 0xd8, 0x22, 0x86, 0x42, 0xa4, 0x77, 0xbd, 0xa2,
                0x55, 0xb3, 0x2a, 0xad, 0xbc, 0xe4, 0xbd, 0xa0, 0xb3, 0xf7, 0xe3, 0x6c, 0x9d, 0xa7
            ]
        );
    }

    #[test]
    fn rfc_4231_hmac_sha224_case_1() {
        // RFC 4231 test case 1: key = 20 * 0x0b, data = "Hi There".
        let mac = hmac(DigestAlgorithm::Sha224, &[0x0b; 20], b"Hi There");
        assert_eq!(
            mac,
            [
                0x89, 0x6f, 0xb1, 0x12, 0x8a, 0xbb, 0xdf, 0x19, 0x68, 0x32, 0x10, 0x7c, 0xd4, 0x9d,
                0xf3, 0x3f, 0x47, 0xb4, 0xb1, 0x16, 0x99, 0x12, 0xba, 0x4f, 0x53, 0x68, 0x4b, 0x22
            ]
        );
    }

    #[test]
    fn constant_time_eq_agrees_with_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }
}
