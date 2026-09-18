//! The key derivation functions that turn a Diffie–Hellman shared secret into a key.
//!
//! A shared secret is not a key. `Z` is an element of a group, so its octets are not uniformly
//! distributed and every party who can guess the group can distinguish them; a derivation step
//! is what turns it into key material of the right length. XML Encryption names two:
//!
//! * **ConcatKDF** ([XMLEnc 1.1 §5.4.1](https://www.w3.org/TR/xmlenc-core1/#sec-ConcatKDF)),
//!   which is the single-step key-derivation function of NIST SP 800-56A §5.8.1 — hash the
//!   counter, the shared secret and a fixed `OtherInfo` string, repeat until there are enough
//!   octets, truncate. This is what the `ECDH-ES` and `dh-es` agreement methods use, and it is
//!   the one a new deployment should choose.
//! * The **legacy digest construction** of
//!   [XMLEnc 1.0 §5.5.2](https://www.w3.org/TR/2002/REC-xmlenc-core-20021210/#sec-DHKeyAgreement),
//!   which the plain `dh` agreement method carries with it. It hashes the shared secret together
//!   with the *textual* algorithm URI and key size rather than a caller-chosen `OtherInfo`, and
//!   it exists here only so that documents written against XMLEnc 1.0 can be read.
//!
//! # What a KDF does not do
//!
//! Neither function authenticates anything. Deriving a key from a shared secret proves only that
//! the peer's public value went into it; with an ephemeral-static agreement — which is what
//! `ECDH-ES` is — *anyone* can produce a well-formed `EncryptedKey`, because the originator's key
//! is fresh and unauthenticated. Knowing who sent a document needs XML Signature.
//!
//! # Example
//!
//! The worked example of [RFC 7518 Appendix C](https://www.rfc-editor.org/rfc/rfc7518#appendix-C)
//! derives a 128-bit key from a P-256 shared secret with SHA-256:
//!
//! ```
//! use oxixml_enc::crypto::digest::Digest;
//! use oxixml_enc::crypto::kdf::{ConcatKdfParams, concat_kdf};
//!
//! let shared_secret = oxixml_text::hex::decode(
//!     "9E56D91D817135D372834283BF84269CFB316EA3DA806A48F6DAA7798CFE90C4",
//! )?;
//! let params = ConcatKdfParams::new(Digest::Sha256)
//!     .with_algorithm_id(b"\x00\x00\x00\x07A128GCM".to_vec())
//!     .with_party_u_info(b"\x00\x00\x00\x05Alice".to_vec())
//!     .with_party_v_info(b"\x00\x00\x00\x03Bob".to_vec())
//!     .with_supp_pub_info(vec![0x00, 0x00, 0x00, 0x80]);
//!
//! let key = concat_kdf(&params, &shared_secret, 16)?;
//! assert_eq!(oxixml_text::hex::encode_upper(&key), "56AA8DEAF8236D205C2228CD71A7101A");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::crypto::digest::Digest;
use crate::error::{EncError, Result};

/// The `Algorithm` URI of ConcatKDF, `http://www.w3.org/2009/xmlenc11#ConcatKDF`.
pub const CONCAT_KDF_URI: &str = "http://www.w3.org/2009/xmlenc11#ConcatKDF";

/// The `Algorithm` URI of PBKDF2, `http://www.w3.org/2009/xmlenc11#pbkdf2`.
///
/// PBKDF2 derives a key from a *passphrase*, not from a shared secret, and is OPTIONAL in
/// XMLEnc 1.1. It is named here so that a caller can recognize it and refuse it explicitly; this
/// crate does not implement it.
pub const PBKDF2_URI: &str = "http://www.w3.org/2009/xmlenc11#pbkdf2";

/// The most keying material either derivation will produce, in octets.
///
/// The largest key XML Encryption asks for is 32 octets, so this ceiling is generous by two
/// orders of magnitude. It exists because the requested length comes, ultimately, from a
/// `KeySize` element in an attacker-supplied document, and each additional block costs a hash of
/// the whole shared secret and `OtherInfo`.
pub const MAX_DERIVED_LEN: usize = 1024;

/// The `OtherInfo` fields of the NIST SP 800-56A single-step key-derivation function, as the
/// `xenc11:ConcatKDFParams` element carries them.
///
/// Every field defaults to empty. A field left empty simply contributes nothing to `OtherInfo`,
/// which is what an absent attribute means — but note that two different agreements whose
/// `OtherInfo` strings collide derive the same key, so `AlgorithmID`, `PartyUInfo` and
/// `PartyVInfo` should be *unambiguously* encoded (a length prefix, as RFC 7518 does, or a fixed
/// width) rather than simply concatenated text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConcatKdfParams {
    /// The hash function, from the `ds:DigestMethod` child of `xenc11:ConcatKDFParams`.
    pub digest: Digest,
    /// `AlgorithmID` — which algorithm the derived key is for.
    pub algorithm_id: Vec<u8>,
    /// `PartyUInfo` — the originator's contribution. For a static-static agreement this MUST
    /// contain a nonce, because nothing else in the derivation is fresh.
    pub party_u_info: Vec<u8>,
    /// `PartyVInfo` — the recipient's contribution.
    pub party_v_info: Vec<u8>,
    /// `SuppPubInfo` — supplementary public information.
    pub supp_pub_info: Vec<u8>,
    /// `SuppPrivInfo` — supplementary private information, shared out of band.
    pub supp_priv_info: Vec<u8>,
}

impl ConcatKdfParams {
    /// Returns parameters that use `digest` and leave every `OtherInfo` field empty.
    #[must_use]
    pub fn new(digest: Digest) -> Self {
        Self {
            digest,
            ..Self::default()
        }
    }

    /// Sets `AlgorithmID`.
    #[must_use]
    pub fn with_algorithm_id(mut self, value: impl Into<Vec<u8>>) -> Self {
        self.algorithm_id = value.into();
        self
    }

    /// Sets `PartyUInfo`.
    #[must_use]
    pub fn with_party_u_info(mut self, value: impl Into<Vec<u8>>) -> Self {
        self.party_u_info = value.into();
        self
    }

    /// Sets `PartyVInfo`.
    #[must_use]
    pub fn with_party_v_info(mut self, value: impl Into<Vec<u8>>) -> Self {
        self.party_v_info = value.into();
        self
    }

    /// Sets `SuppPubInfo`.
    #[must_use]
    pub fn with_supp_pub_info(mut self, value: impl Into<Vec<u8>>) -> Self {
        self.supp_pub_info = value.into();
        self
    }

    /// Sets `SuppPrivInfo`.
    #[must_use]
    pub fn with_supp_priv_info(mut self, value: impl Into<Vec<u8>>) -> Self {
        self.supp_priv_info = value.into();
        self
    }

    /// Returns the `OtherInfo` octet string: the five fields concatenated in the order
    /// SP 800-56A §5.8.1.2 fixes.
    #[must_use]
    pub fn other_info(&self) -> Vec<u8> {
        let mut other = Vec::with_capacity(
            self.algorithm_id.len()
                + self.party_u_info.len()
                + self.party_v_info.len()
                + self.supp_pub_info.len()
                + self.supp_priv_info.len(),
        );
        other.extend_from_slice(&self.algorithm_id);
        other.extend_from_slice(&self.party_u_info);
        other.extend_from_slice(&self.party_v_info);
        other.extend_from_slice(&self.supp_pub_info);
        other.extend_from_slice(&self.supp_priv_info);
        other
    }
}

/// The single-step key-derivation function of NIST SP 800-56A §5.8.1, which XMLEnc 1.1 §5.4.1
/// names `ConcatKDF`.
///
/// ```text
/// reps = ceil(key_len / hash_len)
/// K(i) = H( counter(i) ‖ Z ‖ OtherInfo )   with counter a 32-bit big-endian integer from 1
/// derived = leftmost key_len octets of K(1) ‖ … ‖ K(reps)
/// ```
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyDerivation`] when `shared_secret` is empty, when `key_len` is
/// zero, or when `key_len` exceeds [`MAX_DERIVED_LEN`].
pub fn concat_kdf(
    params: &ConcatKdfParams,
    shared_secret: &[u8],
    key_len: usize,
) -> Result<Vec<u8>> {
    if shared_secret.is_empty() {
        return Err(EncError::InvalidKeyDerivation(
            "the shared secret is empty, so there is nothing to derive from",
        ));
    }
    if key_len == 0 {
        return Err(EncError::InvalidKeyDerivation(
            "a derived key of zero octets was requested",
        ));
    }
    if key_len > MAX_DERIVED_LEN {
        return Err(EncError::InvalidKeyDerivation(
            "more keying material was requested than ConcatKDF will produce",
        ));
    }

    let hash_len = params.digest.output_len();
    let other_info = params.other_info();
    let reps = key_len.div_ceil(hash_len);

    let mut derived = Vec::with_capacity(reps * hash_len);
    let mut block = Vec::with_capacity(4 + shared_secret.len() + other_info.len());
    for counter in 1..=reps {
        // `reps` is bounded by `MAX_DERIVED_LEN`, far below the 2^32 - 1 that SP 800-56A allows,
        // so the counter always fits its four octets.
        let counter = u32::try_from(counter).map_err(|_| {
            EncError::InvalidKeyDerivation("the ConcatKDF counter would overflow 32 bits")
        })?;
        block.clear();
        block.extend_from_slice(&counter.to_be_bytes());
        block.extend_from_slice(shared_secret);
        block.extend_from_slice(&other_info);
        derived.extend_from_slice(&params.digest.hash(&block));
    }
    // The block held the shared secret; overwrite it rather than leaving it in the allocation.
    block.fill(0);

    derived.truncate(key_len);
    Ok(derived)
}

/// The key derivation function XMLEnc 1.0 §5.5.2 defines for the plain `dh` agreement method.
///
/// ```text
/// KM(counter) = H( ZZ ‖ counter ‖ EncryptionAlg ‖ KA-Nonce ‖ KeySize )
/// derived     = leading key_len octets of KM(1) ‖ KM(2) ‖ …
/// ```
///
/// Three of those fields are *text*, which is easy to get wrong and impossible to detect from a
/// failed decryption:
///
/// * `counter` is a one-octet counter starting at one, written as **two upper-case hexadecimal
///   digits** — the two ASCII characters `01`, not the octet `0x01`.
/// * `EncryptionAlg` is the URI of the algorithm the derived key will be used *for* — the block
///   cipher or the key wrap — and never the URI of the agreement method itself.
/// * `KeySize` is the derived key's size **in bits**, as the decimal digits of that number in
///   ASCII with no leading zeros. It is computed here from `key_len` so that the two cannot
///   disagree.
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyDerivation`] when `shared_secret` is empty, when `key_len` is
/// zero or exceeds [`MAX_DERIVED_LEN`], or when the one-octet counter would have to pass 255.
pub fn dh_legacy_kdf(
    digest: Digest,
    encryption_algorithm: &str,
    ka_nonce: &[u8],
    shared_secret: &[u8],
    key_len: usize,
) -> Result<Vec<u8>> {
    if shared_secret.is_empty() {
        return Err(EncError::InvalidKeyDerivation(
            "the shared secret is empty, so there is nothing to derive from",
        ));
    }
    if key_len == 0 {
        return Err(EncError::InvalidKeyDerivation(
            "a derived key of zero octets was requested",
        ));
    }
    if key_len > MAX_DERIVED_LEN {
        return Err(EncError::InvalidKeyDerivation(
            "more keying material was requested than the XMLEnc 1.0 key derivation will produce",
        ));
    }

    let hash_len = digest.output_len();
    let reps = key_len.div_ceil(hash_len);
    if reps > 255 {
        return Err(EncError::InvalidKeyDerivation(
            "the XMLEnc 1.0 key derivation counter is one octet and would overflow",
        ));
    }
    let key_size = format!("{}", key_len * 8);

    let mut derived = Vec::with_capacity(reps * hash_len);
    let mut block = Vec::new();
    for counter in 1..=reps {
        block.clear();
        block.extend_from_slice(shared_secret);
        block.extend_from_slice(format!("{counter:02X}").as_bytes());
        block.extend_from_slice(encryption_algorithm.as_bytes());
        block.extend_from_slice(ka_nonce);
        block.extend_from_slice(key_size.as_bytes());
        derived.extend_from_slice(&digest.hash(&block));
    }
    block.fill(0);

    derived.truncate(key_len);
    Ok(derived)
}

/// Decodes the bit-string encoding XMLEnc 1.1 §5.4.1 uses for the `xenc11:ConcatKDFParams`
/// attributes.
///
/// The attributes are `xsd:hexBinary`, but what they carry is a *bit* string, so the encoding
/// prepends one octet giving how many padding bits the last octet has: the five-bit string
/// `11011` becomes `03D8` — three bits of padding, then `11011000`. `OtherInfo` is built from
/// the original, unpadded bit strings.
///
/// This crate hashes octets, so a field that is not a whole number of octets cannot be
/// represented and is refused rather than silently padded — silently padding would derive a
/// *different* key from the one the sender derived, and the only symptom would be an opaque
/// decryption failure.
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyDerivation`] when `encoded` is empty (the padding octet is not
/// optional), when the padding count is not in `0..8`, or when it is not zero.
pub fn decode_kdf_bitstring(encoded: &[u8]) -> Result<Vec<u8>> {
    let Some((padding_bits, value)) = encoded.split_first() else {
        return Err(EncError::InvalidKeyDerivation(
            "a ConcatKDF parameter is empty, but its leading padding octet is required",
        ));
    };
    if *padding_bits > 7 {
        return Err(EncError::InvalidKeyDerivation(
            "a ConcatKDF parameter claims more than seven padding bits",
        ));
    }
    if *padding_bits != 0 {
        return Err(EncError::InvalidKeyDerivation(
            "a ConcatKDF parameter is not a whole number of octets",
        ));
    }
    Ok(value.to_vec())
}

/// Encodes `octets` in the bit-string `hexBinary` form [`decode_kdf_bitstring`] reads: a leading
/// zero padding-bits octet, then `octets` unchanged, hex-encoded.
///
/// This crate only ever represents whole-octet `OtherInfo` fields, so the padding-bits octet is
/// always zero; `decode_kdf_bitstring(&decode(&encode_kdf_bitstring(octets))?)` always succeeds
/// and returns `octets`.
#[must_use]
pub fn encode_kdf_bitstring(octets: &[u8]) -> String {
    let mut padded = Vec::with_capacity(1 + octets.len());
    padded.push(0u8);
    padded.extend_from_slice(octets);
    oxixml_text::hex::encode_upper(&padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Z` of the worked example in RFC 7518 Appendix C, which is the shared secret of a P-256
    /// agreement between the two key pairs printed there.
    const RFC_7518_Z: &str = "9E56D91D817135D372834283BF84269CFB316EA3DA806A48F6DAA7798CFE90C4";

    fn rfc_7518_z() -> Vec<u8> {
        oxixml_text::hex::decode(RFC_7518_Z).expect("hexadecimal")
    }

    fn rfc_7518_params() -> ConcatKdfParams {
        ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(b"\x00\x00\x00\x07A128GCM".to_vec())
            .with_party_u_info(b"\x00\x00\x00\x05Alice".to_vec())
            .with_party_v_info(b"\x00\x00\x00\x03Bob".to_vec())
            .with_supp_pub_info(vec![0x00, 0x00, 0x00, 0x80])
    }

    #[test]
    fn the_rfc_7518_appendix_c_vector_is_reproduced() -> Result<()> {
        let derived = concat_kdf(&rfc_7518_params(), &rfc_7518_z(), 16)?;
        assert_eq!(
            oxixml_text::hex::encode_upper(&derived),
            "56AA8DEAF8236D205C2228CD71A7101A"
        );
        Ok(())
    }

    #[test]
    fn other_info_is_the_five_fields_in_order() {
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(vec![1])
            .with_party_u_info(vec![2])
            .with_party_v_info(vec![3])
            .with_supp_pub_info(vec![4])
            .with_supp_priv_info(vec![5]);
        assert_eq!(params.other_info(), vec![1, 2, 3, 4, 5]);
        // Absent fields simply contribute nothing.
        assert!(ConcatKdfParams::new(Digest::Sha1).other_info().is_empty());
    }

    #[test]
    fn a_derived_key_longer_than_the_hash_uses_more_blocks() -> Result<()> {
        let params = rfc_7518_params();
        let secret = rfc_7518_z();
        let long = concat_kdf(&params, &secret, 80)?;
        assert_eq!(long.len(), 80);

        // Each block is the hash of its own counter, so the first 32 octets are the single-block
        // answer and the next 32 are a different block entirely.
        let first = concat_kdf(&params, &secret, 32)?;
        assert_eq!(&long[..32], &first[..]);
        assert_ne!(&long[32..64], &first[..]);

        // The published vector only ever needs one block, so the counter's four big-endian
        // octets are pinned here for the second and third blocks as well.
        let other_info = params.other_info();
        for (index, counter) in [1u32, 2, 3].iter().enumerate() {
            let mut input = Vec::new();
            input.extend_from_slice(&counter.to_be_bytes());
            input.extend_from_slice(&secret);
            input.extend_from_slice(&other_info);
            let block = params.digest.hash(&input);
            let start = index * 32;
            let end = (start + 32).min(80);
            assert_eq!(&long[start..end], &block[..end - start], "block {counter}");
        }
        Ok(())
    }

    #[test]
    fn every_input_changes_the_derived_key() -> Result<()> {
        let secret = rfc_7518_z();
        let base = concat_kdf(&rfc_7518_params(), &secret, 16)?;
        let variants = [
            rfc_7518_params().with_algorithm_id(b"\x00\x00\x00\x07A256GCM".to_vec()),
            rfc_7518_params().with_party_u_info(b"\x00\x00\x00\x05Alicf".to_vec()),
            rfc_7518_params().with_party_v_info(b"\x00\x00\x00\x03Bod".to_vec()),
            rfc_7518_params().with_supp_pub_info(vec![0x00, 0x00, 0x01, 0x00]),
            rfc_7518_params().with_supp_priv_info(vec![0x01]),
            ConcatKdfParams {
                digest: Digest::Sha384,
                ..rfc_7518_params()
            },
        ];
        for variant in variants {
            assert_ne!(concat_kdf(&variant, &secret, 16)?, base);
        }
        // As does the shared secret itself.
        let mut other_secret = secret.clone();
        other_secret[0] ^= 0x01;
        assert_ne!(concat_kdf(&rfc_7518_params(), &other_secret, 16)?, base);
        Ok(())
    }

    #[test]
    fn out_of_range_requests_are_refused() {
        let params = rfc_7518_params();
        let secret = rfc_7518_z();
        assert!(matches!(
            concat_kdf(&params, &secret, 0),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            concat_kdf(&params, &secret, MAX_DERIVED_LEN + 1),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            concat_kdf(&params, &[], 16),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(concat_kdf(&params, &secret, MAX_DERIVED_LEN).is_ok());
    }

    /// The legacy construction is defined only by the specification's own prose, so this test
    /// pins the three fields that are text rather than octets: the counter is the ASCII digits
    /// `01`, the key size is the ASCII digits of the *bit* count, and the algorithm URI goes in
    /// as UTF-8.
    #[test]
    fn the_legacy_derivation_hashes_exactly_what_the_specification_says() -> Result<()> {
        let secret = [0x11u8; 128];
        let uri = "http://www.w3.org/2001/04/xmlenc#aes128-cbc";
        let nonce = [0xab, 0xcd];

        let derived = dh_legacy_kdf(Digest::Sha1, uri, &nonce, &secret, 16)?;

        let mut expected_input = Vec::new();
        expected_input.extend_from_slice(&secret);
        expected_input.extend_from_slice(b"01");
        expected_input.extend_from_slice(uri.as_bytes());
        expected_input.extend_from_slice(&nonce);
        expected_input.extend_from_slice(b"128");
        assert_eq!(derived, Digest::Sha1.hash(&expected_input)[..16]);
        Ok(())
    }

    #[test]
    fn the_legacy_counter_is_two_ascii_hexadecimal_digits() -> Result<()> {
        let secret = [0x22u8; 64];
        // SHA-1 gives twenty octets a block, so forty octets need three blocks.
        let derived = dh_legacy_kdf(Digest::Sha1, "urn:example:aes", &[], &secret, 40)?;
        assert_eq!(derived.len(), 40);

        for (index, counter) in ["01", "02", "03"].iter().enumerate() {
            let mut input = Vec::new();
            input.extend_from_slice(&secret);
            input.extend_from_slice(counter.as_bytes());
            input.extend_from_slice(b"urn:example:aes");
            input.extend_from_slice(b"320");
            let block = Digest::Sha1.hash(&input);
            let start = index * 20;
            let end = (start + 20).min(40);
            assert_eq!(
                &derived[start..end],
                &block[..end - start],
                "block {counter}"
            );
        }
        Ok(())
    }

    #[test]
    fn the_legacy_derivation_distinguishes_the_algorithm_and_the_nonce() -> Result<()> {
        let secret = [0x33u8; 64];
        let base = dh_legacy_kdf(Digest::Sha256, "urn:example:a", &[1], &secret, 32)?;
        assert_ne!(
            dh_legacy_kdf(Digest::Sha256, "urn:example:b", &[1], &secret, 32)?,
            base
        );
        assert_ne!(
            dh_legacy_kdf(Digest::Sha256, "urn:example:a", &[2], &secret, 32)?,
            base
        );
        assert_ne!(
            dh_legacy_kdf(Digest::Sha256, "urn:example:a", &[], &secret, 32)?,
            base
        );
        Ok(())
    }

    #[test]
    fn the_legacy_derivation_refuses_out_of_range_requests() {
        let secret = [0x44u8; 64];
        assert!(matches!(
            dh_legacy_kdf(Digest::Sha1, "urn:example:a", &[], &secret, 0),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            dh_legacy_kdf(Digest::Sha1, "urn:example:a", &[], &[], 16),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            dh_legacy_kdf(
                Digest::Sha1,
                "urn:example:a",
                &[],
                &secret,
                MAX_DERIVED_LEN + 1
            ),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        // Twenty octets a block and a one-octet counter cap SHA-1 at 5100 octets, which is above
        // `MAX_DERIVED_LEN`; SHA-1 therefore never reaches the counter limit, but the check is
        // what keeps a future, shorter digest from silently wrapping.
        assert!(
            dh_legacy_kdf(Digest::Sha1, "urn:example:a", &[], &secret, MAX_DERIVED_LEN).is_ok()
        );
    }

    #[test]
    fn the_bit_string_encoding_is_decoded_when_it_is_octet_aligned() -> Result<()> {
        assert_eq!(decode_kdf_bitstring(&[0x00, 0x41])?, vec![0x41]);
        assert_eq!(decode_kdf_bitstring(&[0x00])?, Vec::<u8>::new());
        Ok(())
    }

    #[test]
    fn encoding_a_bit_string_round_trips_through_decoding() -> Result<()> {
        for octets in [&b""[..], &b"\x41"[..], &[0u8; 40][..], &[0xff; 3][..]] {
            let encoded = encode_kdf_bitstring(octets);
            let hex_decoded = oxixml_text::hex::decode(&encoded)
                .map_err(|_| EncError::InvalidKeyDerivation("bad test hex"))?;
            assert_eq!(decode_kdf_bitstring(&hex_decoded)?, octets);
        }
        assert_eq!(encode_kdf_bitstring(&[0x41]), "0041");
        Ok(())
    }

    #[test]
    fn a_bit_string_that_is_not_octet_aligned_is_refused() {
        // The specification's own example: the five-bit string 11011 encodes as 03D8.
        assert!(matches!(
            decode_kdf_bitstring(&[0x03, 0xd8]),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            decode_kdf_bitstring(&[0x08, 0x00]),
            Err(EncError::InvalidKeyDerivation(_))
        ));
        assert!(matches!(
            decode_kdf_bitstring(&[]),
            Err(EncError::InvalidKeyDerivation(_))
        ));
    }
}
