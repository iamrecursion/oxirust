//! RSASSA-PKCS1-v1_5 — [RFC 8017 §8.2](https://www.rfc-editor.org/rfc/rfc8017#section-8.2).
//!
//! Three pieces are needed and all three are here:
//!
//! * the **DigestInfo prefixes** of
//!   [RFC 8017 §9.2 note 1](https://www.rfc-editor.org/rfc/rfc8017#section-9.2), the DER
//!   encoding of `DigestInfo ::= SEQUENCE { digestAlgorithm AlgorithmIdentifier, digest
//!   OCTET STRING }` with the digest bytes stripped off the end;
//! * **EMSA-PKCS1-v1_5-ENCODE**, which frames that `DigestInfo` as
//!   `0x00 || 0x01 || PS || 0x00 || T` where `PS` is at least eight `0xff` octets;
//! * the **RSA primitives** `RSAVP1` and `RSASP1`, which are one modular
//!   exponentiation each, done with [`oxixml_support::bigint`].
//!
//! Verification follows the *encode-and-compare* formulation the RFC recommends
//! over the *decode* formulation: the expected `EM` is built from scratch and
//! compared with the octets recovered from the signature. That rules out the
//! whole family of Bleichenbacher "signature forgery with e=3" bugs, which all
//! come from a lenient parser skipping over trailing garbage.

use oxixml_support::bigint::BigUint;

use crate::algorithm::DigestAlgorithm;
use crate::crypto::digest::constant_time_eq;

/// The DER prefix of a `DigestInfo` for `algorithm`, as listed in
/// [RFC 8017 §9.2](https://www.rfc-editor.org/rfc/rfc8017#section-9.2).
#[must_use]
pub(crate) const fn digest_info_prefix(algorithm: DigestAlgorithm) -> &'static [u8] {
    match algorithm {
        DigestAlgorithm::Sha1 => &[
            0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04,
            0x14,
        ],
        DigestAlgorithm::Sha224 => &[
            0x30, 0x2d, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
            0x04, 0x05, 0x00, 0x04, 0x1c,
        ],
        DigestAlgorithm::Sha256 => &[
            0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
            0x01, 0x05, 0x00, 0x04, 0x20,
        ],
        DigestAlgorithm::Sha384 => &[
            0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
            0x02, 0x05, 0x00, 0x04, 0x30,
        ],
        DigestAlgorithm::Sha512 => &[
            0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
            0x03, 0x05, 0x00, 0x04, 0x40,
        ],
    }
}

/// Builds the complete `DigestInfo` DER for a digest value.
#[must_use]
pub(crate) fn digest_info(algorithm: DigestAlgorithm, digest: &[u8]) -> Vec<u8> {
    let prefix = digest_info_prefix(algorithm);
    let mut out = Vec::with_capacity(prefix.len() + digest.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(digest);
    out
}

/// EMSA-PKCS1-v1_5-ENCODE from [RFC 8017 §9.2](https://www.rfc-editor.org/rfc/rfc8017#section-9.2).
///
/// Returns [`None`] when `em_len` cannot hold the `DigestInfo` plus the eleven
/// mandatory framing octets — the RFC's "intended encoded message length too
/// short" error.
#[must_use]
pub(crate) fn emsa_pkcs1_v1_5_encode(
    algorithm: DigestAlgorithm,
    message: &[u8],
    em_len: usize,
) -> Option<Vec<u8>> {
    let t = digest_info(
        algorithm,
        &crate::crypto::digest::digest(algorithm, message),
    );
    if em_len < t.len() + 11 {
        return None;
    }
    let padding_len = em_len - t.len() - 3;
    let mut em = Vec::with_capacity(em_len);
    em.push(0x00);
    em.push(0x01);
    em.resize(2 + padding_len, 0xff);
    em.push(0x00);
    em.extend_from_slice(&t);
    Some(em)
}

/// RSAVP1: `s^e mod n`, rendered as exactly `k` octets.
///
/// Returns [`None`] if the signature representative is not in `[0, n-1]`, which
/// [RFC 8017 §5.2.2](https://www.rfc-editor.org/rfc/rfc8017#section-5.2.2) makes
/// a hard error rather than something to reduce modulo `n`.
#[must_use]
pub(crate) fn rsavp1(modulus: &BigUint, exponent: &BigUint, signature: &[u8]) -> Option<Vec<u8>> {
    let k = modulus_octet_length(modulus);
    if signature.len() != k {
        return None;
    }
    let s = BigUint::from_be_bytes(signature);
    if &s >= modulus {
        return None;
    }
    let m = s.modpow(exponent, modulus)?;
    m.to_be_bytes_padded(k)
}

/// RSASP1: `m^d mod n`, rendered as exactly `k` octets.
#[must_use]
pub(crate) fn rsasp1(
    modulus: &BigUint,
    private_exponent: &BigUint,
    message: &[u8],
) -> Option<Vec<u8>> {
    let k = modulus_octet_length(modulus);
    let m = BigUint::from_be_bytes(message);
    if &m >= modulus {
        return None;
    }
    let s = m.modpow(private_exponent, modulus)?;
    s.to_be_bytes_padded(k)
}

/// The octet length `k` of a modulus: `ceil(bit_len / 8)`, which is what
/// [`BigUint::to_be_bytes`] already produces for a non-zero value.
#[must_use]
pub(crate) fn modulus_octet_length(modulus: &BigUint) -> usize {
    modulus.to_be_bytes().len()
}

/// RSASSA-PKCS1-V1_5-VERIFY.
///
/// `message` is the octet string that was signed (for XML Signature: the
/// canonical `SignedInfo`), *not* its digest.
#[must_use]
pub(crate) fn verify(
    modulus: &BigUint,
    exponent: &BigUint,
    algorithm: DigestAlgorithm,
    message: &[u8],
    signature: &[u8],
) -> bool {
    let k = modulus_octet_length(modulus);
    let Some(recovered) = rsavp1(modulus, exponent, signature) else {
        return false;
    };
    let Some(expected) = emsa_pkcs1_v1_5_encode(algorithm, message, k) else {
        return false;
    };
    constant_time_eq(&recovered, &expected)
}

/// RSASSA-PKCS1-V1_5-SIGN.
///
/// Returns [`None`] when the modulus is too small for the chosen digest.
#[must_use]
pub(crate) fn sign(
    modulus: &BigUint,
    private_exponent: &BigUint,
    algorithm: DigestAlgorithm,
    message: &[u8],
) -> Option<Vec<u8>> {
    let k = modulus_octet_length(modulus);
    let em = emsa_pkcs1_v1_5_encode(algorithm, message, k)?;
    rsasp1(modulus, private_exponent, &em)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_info_prefixes_are_self_describing() {
        // Each prefix is a DER SEQUENCE whose declared length equals the rest of
        // the prefix plus the digest that follows it.
        for algorithm in DigestAlgorithm::ALL {
            let prefix = digest_info_prefix(algorithm);
            assert_eq!(prefix.first().copied(), Some(0x30), "{algorithm}");
            let declared = usize::from(prefix.get(1).copied().unwrap_or(0));
            assert_eq!(
                declared,
                prefix.len() - 2 + algorithm.output_length(),
                "{algorithm}"
            );
            // …and the trailing OCTET STRING header announces the digest length.
            let tail = prefix.len();
            assert_eq!(prefix.get(tail - 2).copied(), Some(0x04), "{algorithm}");
            assert_eq!(
                usize::from(prefix.get(tail - 1).copied().unwrap_or(0)),
                algorithm.output_length(),
                "{algorithm}"
            );
        }
    }

    #[test]
    fn emsa_encoding_has_the_rfc_8017_shape() {
        let em = emsa_pkcs1_v1_5_encode(DigestAlgorithm::Sha256, b"abc", 128)
            .expect("128 octets is plenty for SHA-256");
        assert_eq!(em.len(), 128);
        assert_eq!(em.first().copied(), Some(0x00));
        assert_eq!(em.get(1).copied(), Some(0x01));
        let zero = em
            .iter()
            .skip(2)
            .position(|byte| *byte != 0xff)
            .expect("a separator");
        assert!(zero >= 8, "at least eight 0xff padding octets");
        assert_eq!(em.get(2 + zero).copied(), Some(0x00));
        assert_eq!(
            em.get(3 + zero..),
            Some(
                digest_info(
                    DigestAlgorithm::Sha256,
                    &crate::crypto::digest::digest(DigestAlgorithm::Sha256, b"abc")
                )
                .as_slice()
            )
        );
    }

    #[test]
    fn emsa_encoding_refuses_a_modulus_that_is_too_small() {
        assert!(emsa_pkcs1_v1_5_encode(DigestAlgorithm::Sha512, b"abc", 64).is_none());
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let modulus = crate::testkey::modulus();
        let public = crate::testkey::public_exponent();
        let private = crate::testkey::private_exponent();
        for algorithm in DigestAlgorithm::ALL {
            let signature = sign(&modulus, &private, algorithm, b"SignedInfo octets")
                .expect("a 1024-bit modulus holds every SHA-2 DigestInfo");
            assert_eq!(signature.len(), 128, "{algorithm}");
            assert!(verify(
                &modulus,
                &public,
                algorithm,
                b"SignedInfo octets",
                &signature
            ));
            assert!(
                !verify(&modulus, &public, algorithm, b"other octets", &signature),
                "{algorithm}"
            );
        }
    }

    #[test]
    fn a_flipped_signature_bit_does_not_verify() {
        let modulus = crate::testkey::modulus();
        let public = crate::testkey::public_exponent();
        let private = crate::testkey::private_exponent();
        let mut signature = sign(&modulus, &private, DigestAlgorithm::Sha256, b"payload")
            .expect("the key is large enough");
        if let Some(byte) = signature.last_mut() {
            *byte ^= 0x01;
        }
        assert!(!verify(
            &modulus,
            &public,
            DigestAlgorithm::Sha256,
            b"payload",
            &signature
        ));
    }

    #[test]
    fn a_signature_of_the_wrong_length_is_rejected() {
        let modulus = crate::testkey::modulus();
        let public = crate::testkey::public_exponent();
        assert!(rsavp1(&modulus, &public, &[0u8; 127]).is_none());
        assert!(rsavp1(&modulus, &public, &[0u8; 129]).is_none());
    }

    #[test]
    fn a_representative_at_or_above_the_modulus_is_rejected() {
        let modulus = crate::testkey::modulus();
        let public = crate::testkey::public_exponent();
        assert!(
            rsavp1(
                &modulus,
                &public,
                &crate::testkey::bytes(crate::testkey::MODULUS_BASE64)
            )
            .is_none()
        );
    }
}
