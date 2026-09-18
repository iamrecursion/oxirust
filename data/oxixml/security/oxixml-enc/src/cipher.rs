//! The framing XML Encryption puts around raw ciphertext, and the key-protection operations.
//!
//! [XMLEnc §5.2](https://www.w3.org/TR/xmlenc-core1/#sec-Alg-Block) does not carry the
//! initialization vector in an attribute: it *prefixes* it to the ciphertext, "for ease of
//! availability to the decryption code and to emphasize its association with the cipher text".
//! AES-GCM additionally appends the authentication tag (§5.2.4). So the octets that end up
//! base64-encoded in a `CipherValue` are
//!
//! ```text
//! CBC:  IV(16) ‖ C          where C is the CBC encryption of the padded plaintext
//! GCM:  IV(12) ‖ C ‖ T(16)  where T is the 128-bit authentication tag
//! ```
//!
//! This module is the only place that knows those layouts.
//!
//! # No associated data
//!
//! XML Encryption defines no way to carry additional authenticated data, so [`encrypt`] passes
//! an empty AAD to GCM. That is a real limitation of the *specification*, not of this code:
//! the `Type`, `Id` and `EncryptionMethod` of an `EncryptedData` are **not** authenticated even
//! under AES-GCM, so an attacker who can edit the document can change `Type` from `Element` to
//! `Content` or point `KeyInfo` at a different key. An application that cares must sign the
//! document (XML Signature) as well as encrypt it. [`crate::crypto::gcm`] takes an AAD argument
//! so that a profile which does define one can use it.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::algorithm::BlockEncryption;
//! use oxixml_enc::cipher;
//! use oxixml_enc::key::SymmetricKey;
//! use oxixml_enc::random::FixedRandom;
//!
//! let key = SymmetricKey::new(vec![0x11; 32]);
//! let mut rng = FixedRandom::new(&[0x22; 16]);
//! let framed = cipher::encrypt(BlockEncryption::Aes256Gcm, &key, b"secret", &mut rng)?;
//!
//! // 12 octets of IV, 6 of ciphertext, 16 of tag.
//! assert_eq!(framed.len(), 12 + 6 + 16);
//! assert_eq!(&framed[..12], &[0x22; 12]);
//! assert_eq!(cipher::decrypt(BlockEncryption::Aes256Gcm, &key, &framed)?, b"secret");
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use crate::algorithm::{BlockEncryption, KeyTransport, KeyWrap};
use crate::crypto::aes::{Aes, BLOCK_SIZE};
use crate::crypto::rsa::{OaepParams, RsaPrivateKey, RsaPublicKey};
use crate::crypto::{cbc, gcm, keywrap};
use crate::error::{EncError, Result};
use crate::key::SymmetricKey;
use crate::random::RandomSource;

/// Checks that `key` is the length `algorithm` requires and expands the AES key schedule.
fn cipher_for(algorithm: BlockEncryption, key: &SymmetricKey) -> Result<Aes> {
    if key.len() != algorithm.key_len() {
        return Err(EncError::InvalidKeyLength {
            algorithm: "AES",
            length: key.len(),
        });
    }
    Aes::new(key.as_bytes())
}

/// Encrypts `plaintext` and returns the framed octets a `CipherValue` carries.
///
/// A fresh initialization vector is drawn from `rng` for every call, as XMLEnc §6.4 requires.
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyLength`] when the key does not match the algorithm, and
/// [`EncError::Entropy`] when the random source fails.
pub fn encrypt(
    algorithm: BlockEncryption,
    key: &SymmetricKey,
    plaintext: &[u8],
    rng: &mut dyn RandomSource,
) -> Result<Vec<u8>> {
    let aes = cipher_for(algorithm, key)?;
    let mut iv = vec![0u8; algorithm.iv_len()];
    rng.fill(&mut iv)?;

    let mut framed = iv.clone();
    if algorithm.is_authenticated() {
        let (ciphertext, tag) = gcm::encrypt(&aes, &iv, plaintext, &[])?;
        framed.extend_from_slice(&ciphertext);
        framed.extend_from_slice(&tag);
    } else {
        let mut block_iv = [0u8; BLOCK_SIZE];
        block_iv.copy_from_slice(&iv);
        framed.extend_from_slice(&cbc::encrypt_padded(&aes, &block_iv, plaintext, rng)?);
    }
    Ok(framed)
}

/// Decrypts the framed octets of a `CipherValue`.
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyLength`] when the key does not match the algorithm, and
/// [`EncError::DecryptionFailed`] for a ciphertext that is too short to contain its own
/// framing, a failed GCM tag check or malformed CBC padding — all of them the same opaque
/// error, deliberately.
pub fn decrypt(algorithm: BlockEncryption, key: &SymmetricKey, framed: &[u8]) -> Result<Vec<u8>> {
    let aes = cipher_for(algorithm, key)?;
    let iv_len = algorithm.iv_len();
    let tag_len = algorithm.tag_len();

    // A truncated ciphertext is a decryption failure rather than a structural error: its length
    // is attacker-controlled, and reporting it separately would be one more oracle bit.
    if framed.len() < iv_len + tag_len {
        return Err(EncError::DecryptionFailed);
    }
    let (iv, rest) = framed.split_at(iv_len);

    if algorithm.is_authenticated() {
        let (ciphertext, tag) = rest.split_at(rest.len() - tag_len);
        return gcm::decrypt(&aes, iv, ciphertext, &[], tag);
    }

    if rest.is_empty() || rest.len() % BLOCK_SIZE != 0 {
        return Err(EncError::DecryptionFailed);
    }
    let mut block_iv = [0u8; BLOCK_SIZE];
    block_iv.copy_from_slice(iv);
    cbc::decrypt_padded(&aes, &block_iv, rest)
}

/// Wraps a content-encryption key under a symmetric key-encryption key (XMLEnc §5.7).
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyLength`] when the key-encryption key does not match the
/// algorithm, and [`EncError::InvalidBlockLength`] when the RFC 3394 variants are asked to wrap
/// key material that is not a positive multiple of eight octets — use one of the `-pad`
/// algorithms for that.
pub fn wrap_key(
    algorithm: KeyWrap,
    key_encryption_key: &SymmetricKey,
    content_key: &SymmetricKey,
) -> Result<Vec<u8>> {
    if key_encryption_key.len() != algorithm.key_len() {
        return Err(EncError::InvalidKeyLength {
            algorithm: "AES key wrap",
            length: key_encryption_key.len(),
        });
    }
    let aes = Aes::new(key_encryption_key.as_bytes())?;
    if algorithm.is_padded() {
        keywrap::wrap_padded(&aes, content_key.as_bytes())
    } else {
        keywrap::wrap(&aes, content_key.as_bytes())
    }
}

/// Unwraps a content-encryption key.
///
/// # Errors
///
/// Returns [`EncError::InvalidKeyLength`] when the key-encryption key does not match the
/// algorithm, and [`EncError::DecryptionFailed`] when the integrity check fails.
pub fn unwrap_key(
    algorithm: KeyWrap,
    key_encryption_key: &SymmetricKey,
    wrapped: &[u8],
) -> Result<SymmetricKey> {
    if key_encryption_key.len() != algorithm.key_len() {
        return Err(EncError::InvalidKeyLength {
            algorithm: "AES key wrap",
            length: key_encryption_key.len(),
        });
    }
    let aes = Aes::new(key_encryption_key.as_bytes())?;
    let octets = if algorithm.is_padded() {
        keywrap::unwrap_padded(&aes, wrapped)?
    } else {
        keywrap::unwrap(&aes, wrapped)?
    };
    Ok(SymmetricKey::new(octets))
}

/// Encrypts a content-encryption key under an RSA public key (XMLEnc §5.5).
///
/// # Errors
///
/// Returns [`EncError::MessageTooLong`] when the key does not fit the modulus, and
/// [`EncError::Entropy`] when the random source fails.
pub fn transport_key(
    algorithm: KeyTransport,
    public_key: &RsaPublicKey,
    content_key: &SymmetricKey,
    params: &OaepParams,
    rng: &mut dyn RandomSource,
) -> Result<Vec<u8>> {
    match algorithm {
        KeyTransport::Rsa15 => public_key.encrypt_pkcs1v15(content_key.as_bytes(), rng),
        KeyTransport::RsaOaepMgf1p | KeyTransport::RsaOaep => {
            public_key.encrypt_oaep(content_key.as_bytes(), params, rng)
        }
    }
}

/// Recovers a content-encryption key from an RSA-protected `CipherValue`.
///
/// # Errors
///
/// Returns [`EncError::DecryptionFailed`] for every failure, deliberately without
/// distinguishing them; see [`crate::crypto::rsa`].
pub fn untransport_key(
    algorithm: KeyTransport,
    private_key: &RsaPrivateKey,
    ciphertext: &[u8],
    params: &OaepParams,
) -> Result<SymmetricKey> {
    let octets = match algorithm {
        KeyTransport::Rsa15 => private_key.decrypt_pkcs1v15(ciphertext)?,
        KeyTransport::RsaOaepMgf1p | KeyTransport::RsaOaep => {
            private_key.decrypt_oaep(ciphertext, params)?
        }
    };
    Ok(SymmetricKey::new(octets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{FixedRandom, PseudoRandom};

    #[test]
    fn every_algorithm_round_trips_at_every_length() -> Result<()> {
        for algorithm in BlockEncryption::ALL {
            let key = SymmetricKey::new(vec![0x5a; algorithm.key_len()]);
            let mut rng = PseudoRandom::new(7);
            for length in [0usize, 1, 15, 16, 17, 100] {
                let plaintext: Vec<u8> = (0..length).map(|index| index as u8).collect();
                let framed = encrypt(algorithm, &key, &plaintext, &mut rng)?;
                assert_eq!(decrypt(algorithm, &key, &framed)?, plaintext);
            }
        }
        Ok(())
    }

    #[test]
    fn framing_lengths_are_exactly_what_the_specification_says() -> Result<()> {
        let mut rng = PseudoRandom::new(1);
        for algorithm in BlockEncryption::ALL {
            let key = SymmetricKey::new(vec![0x11; algorithm.key_len()]);
            let framed = encrypt(algorithm, &key, &[0u8; 16], &mut rng)?;
            if algorithm.is_authenticated() {
                assert_eq!(framed.len(), 12 + 16 + 16);
            } else {
                // 16 octets of plaintext are padded up to two blocks.
                assert_eq!(framed.len(), 16 + 32);
            }
        }
        Ok(())
    }

    #[test]
    fn a_fresh_iv_is_drawn_every_time() -> Result<()> {
        let key = SymmetricKey::new(vec![0x33; 16]);
        let mut rng = PseudoRandom::new(99);
        let first = encrypt(BlockEncryption::Aes128Gcm, &key, b"x", &mut rng)?;
        let second = encrypt(BlockEncryption::Aes128Gcm, &key, b"x", &mut rng)?;
        assert_ne!(first, second);
        assert_ne!(first[..12], second[..12]);
        Ok(())
    }

    #[test]
    fn a_wrong_key_length_is_rejected_before_anything_else() {
        let key = SymmetricKey::new(vec![0u8; 16]);
        assert!(matches!(
            encrypt(
                BlockEncryption::Aes256Cbc,
                &key,
                b"x",
                &mut FixedRandom::new(&[0])
            ),
            Err(EncError::InvalidKeyLength { .. })
        ));
        assert!(matches!(
            decrypt(BlockEncryption::Aes256Cbc, &key, &[0u8; 48]),
            Err(EncError::InvalidKeyLength { .. })
        ));
    }

    #[test]
    fn truncated_ciphertexts_fail_opaquely() -> Result<()> {
        for algorithm in BlockEncryption::ALL {
            let key = SymmetricKey::new(vec![0x44; algorithm.key_len()]);
            for length in 0..algorithm.iv_len() + algorithm.tag_len() {
                assert!(
                    matches!(
                        decrypt(algorithm, &key, &vec![0u8; length]),
                        Err(EncError::DecryptionFailed)
                    ),
                    "{algorithm:?} accepted {length} octets"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn a_ragged_cbc_ciphertext_fails_opaquely() {
        let key = SymmetricKey::new(vec![0x55; 16]);
        assert!(matches!(
            decrypt(BlockEncryption::Aes128Cbc, &key, &[0u8; 16 + 9]),
            Err(EncError::DecryptionFailed)
        ));
        // An IV with no ciphertext at all is also a failure: there is no padding to strip.
        assert!(matches!(
            decrypt(BlockEncryption::Aes128Cbc, &key, &[0u8; 16]),
            Err(EncError::DecryptionFailed)
        ));
    }

    #[test]
    fn tampering_with_a_gcm_ciphertext_is_detected() -> Result<()> {
        let key = SymmetricKey::new(vec![0x66; 32]);
        let mut rng = PseudoRandom::new(5);
        let framed = encrypt(BlockEncryption::Aes256Gcm, &key, b"payload", &mut rng)?;
        for index in 0..framed.len() {
            let mut broken = framed.clone();
            broken[index] ^= 0x01;
            assert!(
                matches!(
                    decrypt(BlockEncryption::Aes256Gcm, &key, &broken),
                    Err(EncError::DecryptionFailed)
                ),
                "flipping octet {index} went undetected"
            );
        }
        Ok(())
    }

    #[test]
    fn key_wrap_round_trips_for_every_algorithm() -> Result<()> {
        for algorithm in KeyWrap::ALL {
            let kek = SymmetricKey::new(vec![0x77; algorithm.key_len()]);
            let cek = SymmetricKey::new(vec![0x88; 32]);
            let wrapped = wrap_key(algorithm, &kek, &cek)?;
            assert_eq!(
                unwrap_key(algorithm, &kek, &wrapped)?.as_bytes(),
                cek.as_bytes()
            );
        }
        Ok(())
    }

    #[test]
    fn a_padded_algorithm_wraps_a_ragged_key() -> Result<()> {
        let kek = SymmetricKey::new(vec![0x99; 32]);
        let cek = SymmetricKey::new(vec![0xaa; 5]);
        assert!(matches!(
            wrap_key(KeyWrap::Aes256, &kek, &cek),
            Err(EncError::InvalidBlockLength { .. })
        ));
        let wrapped = wrap_key(KeyWrap::Aes256Pad, &kek, &cek)?;
        assert_eq!(
            unwrap_key(KeyWrap::Aes256Pad, &kek, &wrapped)?.as_bytes(),
            &[0xaa; 5]
        );
        Ok(())
    }

    #[test]
    fn a_wrong_kek_length_is_rejected() {
        let kek = SymmetricKey::new(vec![0u8; 16]);
        let cek = SymmetricKey::new(vec![0u8; 16]);
        assert!(matches!(
            wrap_key(KeyWrap::Aes256, &kek, &cek),
            Err(EncError::InvalidKeyLength { .. })
        ));
        assert!(matches!(
            unwrap_key(KeyWrap::Aes256, &kek, &[0u8; 24]),
            Err(EncError::InvalidKeyLength { .. })
        ));
    }
}
