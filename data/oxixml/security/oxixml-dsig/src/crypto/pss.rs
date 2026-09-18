//! RSASSA-PSS — [RFC 8017 §8.1](https://www.rfc-editor.org/rfc/rfc8017#section-8.1).
//!
//! PSS is the probabilistic signature scheme PKCS#1 v2.1 added alongside the
//! older v1.5 padding, and the two differ in a way that matters here: v1.5
//! encodes a fixed `DigestInfo` and can therefore be checked by re-encoding and
//! comparing, while PSS mixes in a *salt* that only the encoded message reveals.
//! Verification must genuinely decode — recover the salt from the padding, rebuild
//! `H'` from it, and compare — which is why this module implements
//! EMSA-PSS-VERIFY step by step rather than reusing the encode-and-compare shortcut
//! of [`crate::crypto::pkcs1`].
//!
//! The pieces are:
//!
//! * **MGF1** ([§B.2.1](https://www.rfc-editor.org/rfc/rfc8017#appendix-B.2.1)), the
//!   counter-mode mask generation function;
//! * **EMSA-PSS-ENCODE / EMSA-PSS-VERIFY** ([§9.1](https://www.rfc-editor.org/rfc/rfc8017#section-9.1));
//! * the RSA primitives, shared with [`crate::crypto::pkcs1`].
//!
//! # Parameters
//!
//! XML Signature carries the digest, the mask generation function, the salt
//! length and the trailer field as explicit parameters of the
//! `<ds:SignatureMethod>` — see
//! [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9) — so
//! nothing here assumes SHA-1, and the caller passes whatever the document
//! declared. Only trailer field 1 (`0xbc`) exists in RFC 8017, so that is the
//! only one accepted.
//!
//! # Determinism
//!
//! The salt does not have to be secret or unpredictable — a verifier reads it
//! straight out of the padding — and PSS remains a secure signature scheme with
//! a fixed salt, which is why RFC 8017 permits a salt length of zero. This crate
//! has no random number generator and does not want one, so [`sign`] derives the
//! salt from the private key and the message digest with HMAC. Signatures are
//! therefore reproducible, exactly as the ECDSA and DSA paths are.

use oxixml_support::bigint::BigUint;

use crate::algorithm::DigestAlgorithm;
use crate::crypto::digest::{constant_time_eq, digest, hmac};
use crate::crypto::pkcs1::{modulus_octet_length, rsasp1};
use crate::model::PssParameters;

/// The only trailer field [RFC 8017](https://www.rfc-editor.org/rfc/rfc8017#section-9.1)
/// defines, encoded as the single octet `0xbc`.
pub(crate) const TRAILER_FIELD_BC: u32 = 1;

/// MGF1 from [RFC 8017 §B.2.1](https://www.rfc-editor.org/rfc/rfc8017#appendix-B.2.1).
///
/// `T = Hash(seed || 0x00000000) || Hash(seed || 0x00000001) || …`, truncated to
/// `mask_len`.
#[must_use]
pub(crate) fn mgf1(algorithm: DigestAlgorithm, seed: &[u8], mask_len: usize) -> Vec<u8> {
    let hash_len = algorithm.output_length();
    let mut mask = Vec::with_capacity(mask_len + hash_len);
    let mut counter = 0_u32;
    let mut block = Vec::with_capacity(seed.len() + 4);
    while mask.len() < mask_len {
        block.clear();
        block.extend_from_slice(seed);
        block.extend_from_slice(&counter.to_be_bytes());
        mask.extend_from_slice(&digest(algorithm, &block));
        counter = counter.wrapping_add(1);
    }
    mask.truncate(mask_len);
    mask
}

/// EMSA-PSS-ENCODE from [RFC 8017 §9.1.1](https://www.rfc-editor.org/rfc/rfc8017#section-9.1.1),
/// with the salt supplied by the caller.
///
/// Returns [`None`] for the RFC's "encoding error": an `em_bits` too small to
/// hold the hash, the salt and the two framing octets.
#[must_use]
pub(crate) fn emsa_pss_encode(
    algorithm: DigestAlgorithm,
    mask_algorithm: DigestAlgorithm,
    message: &[u8],
    salt: &[u8],
    em_bits: usize,
) -> Option<Vec<u8>> {
    let hash_len = algorithm.output_length();
    let em_len = em_bits.div_ceil(8);
    if em_len < minimum_encoded_length(hash_len, salt.len())? {
        return None;
    }
    let message_hash = digest(algorithm, message);

    // M' = (0x00) * 8 || mHash || salt, and H = Hash(M').
    let mut padded = Vec::with_capacity(8 + hash_len + salt.len());
    padded.extend_from_slice(&[0_u8; 8]);
    padded.extend_from_slice(&message_hash);
    padded.extend_from_slice(salt);
    let h = digest(algorithm, &padded);

    // DB = PS || 0x01 || salt, with PS the zero octets that pad DB out.
    let db_len = em_len - hash_len - 1;
    let mut db = vec![0_u8; db_len];
    let separator = db_len.checked_sub(salt.len() + 1)?;
    *db.get_mut(separator)? = 0x01;
    db.get_mut(separator + 1..)?.copy_from_slice(salt);

    // maskedDB = DB xor MGF(H, emLen - hLen - 1), with the unused leading bits cleared.
    let mask = mgf1(mask_algorithm, &h, db_len);
    for (target, source) in db.iter_mut().zip(mask.iter()) {
        *target ^= *source;
    }
    clear_leading_bits(&mut db, 8 * em_len - em_bits);

    let mut em = db;
    em.extend_from_slice(&h);
    em.push(0xbc);
    Some(em)
}

/// EMSA-PSS-VERIFY from [RFC 8017 §9.1.2](https://www.rfc-editor.org/rfc/rfc8017#section-9.1.2).
///
/// Every failure is the same answer — "inconsistent" — so the caller learns
/// nothing about *which* step of the padding was wrong.
#[must_use]
pub(crate) fn emsa_pss_verify(
    algorithm: DigestAlgorithm,
    mask_algorithm: DigestAlgorithm,
    message: &[u8],
    em: &[u8],
    em_bits: usize,
    salt_len: usize,
) -> bool {
    let hash_len = algorithm.output_length();
    let em_len = em_bits.div_ceil(8);
    // Step 3, with the addition checked: `salt_len` comes straight out of a
    // `<pss:SaltLength>` and is bounded only by `usize`, so computing the
    // minimum length naively would overflow before it could reject anything.
    let Some(minimum) = minimum_encoded_length(hash_len, salt_len) else {
        return false;
    };
    if em.len() != em_len || em_len < minimum {
        return false;
    }
    if em.last().copied() != Some(0xbc) {
        return false;
    }
    let db_len = em_len - hash_len - 1;
    let (Some(masked_db), Some(h)) = (em.get(..db_len), em.get(db_len..db_len + hash_len)) else {
        return false;
    };

    // The leading bits that the encoding cleared must still be clear.
    let spare_bits = 8 * em_len - em_bits;
    if spare_bits != 0 {
        let Some(first) = masked_db.first() else {
            return false;
        };
        if first >> (8 - spare_bits) != 0 {
            return false;
        }
    }

    let mask = mgf1(mask_algorithm, h, db_len);
    let mut db: Vec<u8> = masked_db
        .iter()
        .zip(mask.iter())
        .map(|(left, right)| left ^ right)
        .collect();
    clear_leading_bits(&mut db, spare_bits);

    // DB must be PS || 0x01 || salt.
    let Some(separator) = salt_len
        .checked_add(1)
        .and_then(|offset| db_len.checked_sub(offset))
    else {
        return false;
    };
    if db
        .get(..separator)
        .is_none_or(|ps| ps.iter().any(|byte| *byte != 0))
    {
        return false;
    }
    if db.get(separator).copied() != Some(0x01) {
        return false;
    }
    let Some(salt) = db.get(separator + 1..) else {
        return false;
    };

    let message_hash = digest(algorithm, message);
    let mut padded = Vec::with_capacity(8 + hash_len + salt.len());
    padded.extend_from_slice(&[0_u8; 8]);
    padded.extend_from_slice(&message_hash);
    padded.extend_from_slice(salt);
    constant_time_eq(&digest(algorithm, &padded), h)
}

/// `hLen + sLen + 2`, the shortest encoded message that can hold a digest, a
/// salt and the two framing octets — [RFC 8017 §9.1.1](https://www.rfc-editor.org/rfc/rfc8017#section-9.1.1)
/// step 3.
///
/// Returns [`None`] when the sum does not fit in a `usize`, which only a
/// declared salt length near the top of the range can cause and which means the
/// encoding could never have been long enough anyway.
fn minimum_encoded_length(hash_len: usize, salt_len: usize) -> Option<usize> {
    hash_len.checked_add(salt_len)?.checked_add(2)
}

/// Zeroes the `count` most significant bits of a big-endian magnitude.
fn clear_leading_bits(bytes: &mut [u8], count: usize) {
    if count == 0 || count >= 8 {
        return;
    }
    if let Some(first) = bytes.first_mut() {
        *first &= 0xff_u8 >> count;
    }
}

/// RSASSA-PSS-VERIFY from [RFC 8017 §8.1.2](https://www.rfc-editor.org/rfc/rfc8017#section-8.1.2).
///
/// `message` is the octet string that was signed — for XML Signature the
/// canonical `SignedInfo` — not its digest.
#[must_use]
pub(crate) fn verify(
    modulus: &BigUint,
    exponent: &BigUint,
    parameters: PssParameters,
    message: &[u8],
    signature: &[u8],
) -> bool {
    if parameters.trailer_field() != TRAILER_FIELD_BC {
        return false;
    }
    let Some(em_bits) = modulus.bit_len().checked_sub(1) else {
        return false;
    };
    let Some(em) = rsa_public_transform(modulus, exponent, signature, em_bits.div_ceil(8)) else {
        return false;
    };
    emsa_pss_verify(
        parameters.digest(),
        parameters.mask_digest(),
        message,
        &em,
        em_bits,
        parameters.salt_octets(),
    )
}

/// RSASSA-PSS-SIGN from [RFC 8017 §8.1.1](https://www.rfc-editor.org/rfc/rfc8017#section-8.1.1),
/// with the deterministic salt described in the module documentation.
///
/// Returns [`None`] when the modulus is too small for the chosen digest and salt
/// length, or when the trailer field is one RFC 8017 does not define.
#[must_use]
pub(crate) fn sign(
    modulus: &BigUint,
    private_exponent: &BigUint,
    parameters: PssParameters,
    message: &[u8],
) -> Option<Vec<u8>> {
    if parameters.trailer_field() != TRAILER_FIELD_BC {
        return None;
    }
    let em_bits = modulus.bit_len().checked_sub(1)?;
    // The "encoding error" test of §9.1.1 step 3, hoisted ahead of the salt.
    // The salt length is declared by the `<pss:SaltLength>` of a signature
    // template, so generating the salt first would let a template a few hundred
    // octets long ask for gigabytes of MGF1 output before the encoding got
    // around to refusing it.
    if em_bits.div_ceil(8)
        < minimum_encoded_length(
            parameters.digest().output_length(),
            parameters.salt_octets(),
        )?
    {
        return None;
    }
    let salt = derive_salt(
        parameters.digest(),
        private_exponent,
        message,
        parameters.salt_octets(),
    );
    let em = emsa_pss_encode(
        parameters.digest(),
        parameters.mask_digest(),
        message,
        &salt,
        em_bits,
    )?;
    rsasp1(modulus, private_exponent, &em)
}

/// Derives the salt from the private exponent and the message.
///
/// HMAC keyed with the private exponent is a pseudo-random function, so the salt
/// is unpredictable to anyone who does not hold the key and reveals nothing about
/// it — while being reproducible, which is what makes a signature deterministic.
fn derive_salt(
    algorithm: DigestAlgorithm,
    private_exponent: &BigUint,
    message: &[u8],
    salt_len: usize,
) -> Vec<u8> {
    if salt_len == 0 {
        return Vec::new();
    }
    let seed = hmac(
        algorithm,
        &private_exponent.to_be_bytes(),
        &digest(algorithm, message),
    );
    mgf1(algorithm, &seed, salt_len)
}

/// `s^e mod n`, rendered as exactly `em_len` octets.
///
/// Returns [`None`] when the signature is not `k` octets long, when the
/// representative is not below the modulus, or when the recovered integer is too
/// large for `em_len` — all three of which
/// [RFC 8017 §8.1.2](https://www.rfc-editor.org/rfc/rfc8017#section-8.1.2) makes
/// hard errors.
fn rsa_public_transform(
    modulus: &BigUint,
    exponent: &BigUint,
    signature: &[u8],
    em_len: usize,
) -> Option<Vec<u8>> {
    if signature.len() != modulus_octet_length(modulus) {
        return None;
    }
    let s = BigUint::from_be_bytes(signature);
    if &s >= modulus {
        return None;
    }
    s.modpow(exponent, modulus)?.to_be_bytes_padded(em_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::{self, hex};

    /// The modulus of Example 1 in the PKCS#1 v2.1 test vector set
    /// (`pss-vect.txt`), a 1024-bit key.
    const EXAMPLE_1_MODULUS: &str = "\
        a56e4a0e701017589a5187dc7ea841d156f2ec0e36ad52a44dfeb1e61f7ad991\
        d8c51056ffedb162b4c0f283a12a88a394dff526ab7291cbb307ceabfce0b1df\
        d5cd9508096d5b2b8b6df5d671ef6377c0921cb23c270a70e2598e6ff89d19f1\
        05acc2d3f0cb35f29280e1386b6f64c4ef22e1e1f20d0ce8cffb2249bd9a2137";

    /// Its private exponent.
    const EXAMPLE_1_PRIVATE_EXPONENT: &str = "\
        33a5042a90b27d4f5451ca9bbbd0b44771a101af884340aef9885f2a4bbe92e8\
        94a724ac3c568c8f97853ad07c0266c8c6a3ca0929f1e8f11231884429fc4d9a\
        e55fee896a10ce707c3ed7e734e44727a39574501a532683109c2abacaba283c\
        31b4bd2f53c3ee37e352cee34f9e503bd80c0622ad79c6dcee883547c6a3b325";

    /// The message of Example 1.1.
    const EXAMPLE_1_1_MESSAGE: &str = "\
        cdc87da223d786df3b45e0bbbc721326d1ee2af806cc315475cc6f0d9c66e1b6\
        2371d45ce2392e1ac92844c310102f156a0d8d52c1f4c40ba3aa65095786cb76\
        9757a6563ba958fed0bcc984e8b517a3d5f515b23b8a41e74aa867693f90dfb0\
        61a6e86dfaaee64472c00e5f20945729cbebe77f06ce78e08f4098fba41f9d61\
        93c0317e8b60d4b6084acb42d29e3808a3bc372d85e331170fcbf7cc72d0b71c\
        296648b3a4d10f416295d0807aa625cab2744fd9ea8fd223c42537029828bd16\
        be02546f130fd2e33b936d2676e08aed1b73318b750a0167d0";

    /// Its salt, which makes the encoding — and therefore the signature —
    /// reproducible octet for octet.
    const EXAMPLE_1_1_SALT: &str = "dee959c7e06411361420ff80185ed57f3e6776af";

    /// Its signature.
    const EXAMPLE_1_1_SIGNATURE: &str = "\
        9074308fb598e9701b2294388e52f971faac2b60a5145af185df5287b5ed2887\
        e57ce7fd44dc8634e407c8e0e4360bc226f3ec227f9d9e54638e8d31f5051215\
        df6ebb9c2f9579aa77598a38f914b5b9c1bd83c4e2f9f382a0d0aa3542ffee65\
        984a601bc69eb28deb27dca12c82c2d4c3f66cd500f1ff2b994d8a4e30cbb33c";

    /// The message of Example 1.2.
    const EXAMPLE_1_2_MESSAGE: &str = "\
        851384cdfe819c22ed6c4ccb30daeb5cf059bc8e1166b7e3530c4c233e2b5f8f\
        71a1cca582d43ecc72b1bca16dfc7013226b9e";

    /// Its signature.
    const EXAMPLE_1_2_SIGNATURE: &str = "\
        3ef7f46e831bf92b32274142a585ffcefbdca7b32ae90d10fb0f0c729984f04e\
        f29a9df0780775ce43739b97838390db0a5505e63de927028d9d29b219ca2c45\
        17832558a55d694a6d25b9dab66003c4cccd907802193be5170d26147d37b935\
        90241be51c25055f47ef62752cfbe21418fafe98c22c4d4d47724fdb5669e843";

    fn example_1_modulus() -> BigUint {
        BigUint::from_be_bytes(&hex(EXAMPLE_1_MODULUS))
    }

    #[test]
    fn mgf1_matches_the_rfc_8017_construction() {
        // MGF1 is Hash(seed || counter) concatenated; a request of exactly one
        // hash length is the first block on its own.
        let seed = b"seed";
        let mask = mgf1(DigestAlgorithm::Sha256, seed, 32);
        let mut block = seed.to_vec();
        block.extend_from_slice(&0_u32.to_be_bytes());
        assert_eq!(mask, digest(DigestAlgorithm::Sha256, &block));
        // A longer request continues with counter 1 and is truncated.
        let long = mgf1(DigestAlgorithm::Sha256, seed, 40);
        assert_eq!(long.len(), 40);
        assert_eq!(long.get(..32), Some(mask.as_slice()));
    }

    /// PKCS#1 v2.1 `pss-vect.txt`, Examples 1.1 and 1.2: signatures produced by
    /// an implementation that shares no code with this one, over SHA-1 with MGF1
    /// over SHA-1 and a 20-octet salt.
    #[test]
    fn pkcs_1_v2_1_signature_examples_verify() {
        let modulus = example_1_modulus();
        let exponent = BigUint::from_u64(65_537);
        for (message, signature) in [
            (EXAMPLE_1_1_MESSAGE, EXAMPLE_1_1_SIGNATURE),
            (EXAMPLE_1_2_MESSAGE, EXAMPLE_1_2_SIGNATURE),
        ] {
            let message = hex(message);
            let signature = hex(signature);
            let check = |salt_len: usize, algorithm: DigestAlgorithm, signature: &[u8]| {
                verify(
                    &modulus,
                    &exponent,
                    PssParameters::new(algorithm).salt_length(salt_len),
                    &message,
                    signature,
                )
            };
            assert!(
                check(20, DigestAlgorithm::Sha1, &signature),
                "the published PSS signature must verify"
            );
            // One flipped bit anywhere must break it …
            let mut tampered = signature.clone();
            if let Some(last) = tampered.last_mut() {
                *last ^= 0x01;
            }
            assert!(!check(20, DigestAlgorithm::Sha1, &tampered));
            // … as must the wrong salt length …
            assert!(!check(32, DigestAlgorithm::Sha1, &signature));
            // … and the wrong digest.
            assert!(!check(20, DigestAlgorithm::Sha256, &signature));
        }
    }

    /// The same vector, but reproducing the *encoding*: with the published salt
    /// the encoded message is fully determined, so the signature octets must come
    /// out identical.
    #[test]
    fn pkcs_1_v2_1_encoding_reproduces_the_published_signature() {
        let modulus = example_1_modulus();
        let private_exponent = BigUint::from_be_bytes(&hex(EXAMPLE_1_PRIVATE_EXPONENT));
        let em_bits = modulus.bit_len() - 1;
        let em = emsa_pss_encode(
            DigestAlgorithm::Sha1,
            DigestAlgorithm::Sha1,
            &hex(EXAMPLE_1_1_MESSAGE),
            &hex(EXAMPLE_1_1_SALT),
            em_bits,
        )
        .expect("the modulus is large enough");
        assert_eq!(
            rsasp1(&modulus, &private_exponent, &em),
            Some(hex(EXAMPLE_1_1_SIGNATURE))
        );
    }

    #[test]
    fn sign_and_verify_round_trip_over_every_digest() {
        let modulus = testkey::modulus();
        let public = testkey::public_exponent();
        let private = testkey::private_exponent();
        for algorithm in DigestAlgorithm::ALL {
            let parameters = PssParameters::new(algorithm);
            let Some(signature) = sign(&modulus, &private, parameters, b"SignedInfo octets") else {
                // A 1024-bit modulus cannot hold SHA-512 with a 64-octet salt.
                assert_eq!(algorithm, DigestAlgorithm::Sha512);
                continue;
            };
            assert_eq!(signature.len(), 128, "{algorithm}");
            assert!(verify(
                &modulus,
                &public,
                parameters,
                b"SignedInfo octets",
                &signature,
            ));
            assert!(!verify(
                &modulus,
                &public,
                parameters,
                b"other octets",
                &signature
            ));
            // Signing is deterministic: the same message yields the same octets.
            assert_eq!(
                sign(&modulus, &private, parameters, b"SignedInfo octets"),
                Some(signature)
            );
        }
    }

    #[test]
    fn a_zero_length_salt_round_trips() {
        let modulus = testkey::modulus();
        let parameters = PssParameters::new(DigestAlgorithm::Sha256).salt_length(0);
        let signature = sign(
            &modulus,
            &testkey::private_exponent(),
            parameters,
            b"payload",
        )
        .expect("a 1024-bit modulus holds SHA-256 with no salt");
        assert!(verify(
            &modulus,
            &testkey::public_exponent(),
            parameters,
            b"payload",
            &signature,
        ));
    }

    #[test]
    fn a_mask_over_a_different_hash_is_honoured() {
        let modulus = testkey::modulus();
        let parameters = PssParameters::new(DigestAlgorithm::Sha256)
            .mask_generation_digest(DigestAlgorithm::Sha1);
        let signature = sign(
            &modulus,
            &testkey::private_exponent(),
            parameters,
            b"payload",
        )
        .expect("the modulus is large enough");
        assert!(verify(
            &modulus,
            &testkey::public_exponent(),
            parameters,
            b"payload",
            &signature,
        ));
        // MGF1 over the wrong hash must not verify.
        assert!(!verify(
            &modulus,
            &testkey::public_exponent(),
            PssParameters::new(DigestAlgorithm::Sha256),
            b"payload",
            &signature,
        ));
    }

    #[test]
    fn an_undefined_trailer_field_is_refused() {
        let mut parameters = PssParameters::new(DigestAlgorithm::Sha256);
        parameters.trailer_field = 2;
        assert!(
            sign(
                &testkey::modulus(),
                &testkey::private_exponent(),
                parameters,
                b"payload"
            )
            .is_none()
        );
    }

    /// The salt length is declared by the document — a `<pss:SaltLength>` is
    /// attacker-supplied text with no bound of its own — so it reaches both
    /// paths as an arbitrary `usize`.
    ///
    /// Signing must refuse it before generating a salt, or a template a few
    /// hundred octets long would ask for gigabytes of MGF1 output; verifying
    /// must return "inconsistent" rather than overflowing the `hLen + sLen + 2`
    /// of RFC 8017 §9.1.1 step 3, which in a debug build is a panic and in a
    /// release build wraps to a small number that the length test would then
    /// wrongly accept.
    #[test]
    fn an_absurd_salt_length_is_refused_by_both_directions() {
        let modulus = testkey::modulus();
        for salt_len in [usize::MAX, usize::MAX - 1, usize::MAX / 2, 1 << 40, 4096] {
            let parameters = PssParameters::new(DigestAlgorithm::Sha256).salt_length(salt_len);
            assert_eq!(
                sign(
                    &modulus,
                    &testkey::private_exponent(),
                    parameters,
                    b"payload"
                ),
                None,
                "salt length {salt_len}"
            );
            assert!(
                !verify(
                    &modulus,
                    &testkey::public_exponent(),
                    parameters,
                    b"payload",
                    &[0_u8; 128],
                ),
                "salt length {salt_len}"
            );
            // And the encoding step itself, reached directly.
            assert!(
                !emsa_pss_verify(
                    DigestAlgorithm::Sha256,
                    DigestAlgorithm::Sha256,
                    b"payload",
                    &[0_u8; 128],
                    1023,
                    salt_len,
                ),
                "salt length {salt_len}"
            );
        }
    }

    /// SHA-512 with a 64-octet salt needs `64 + 64 + 2 = 130` octets of encoded
    /// message; a 1024-bit modulus offers 128.
    #[test]
    fn a_modulus_too_small_for_the_digest_and_salt_is_refused() {
        assert!(
            sign(
                &testkey::modulus(),
                &testkey::private_exponent(),
                PssParameters::new(DigestAlgorithm::Sha512),
                b"payload"
            )
            .is_none()
        );
    }
}
