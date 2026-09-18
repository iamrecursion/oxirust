//! Cipher Block Chaining (NIST SP 800-38A §6.2) and the XML Encryption padding scheme.
//!
//! # The padding XML Encryption uses is *not* PKCS#7
//!
//! [XML Encryption 1.1 §5.2.1](https://www.w3.org/TR/xmlenc-core1/#sec-Padding) specifies a
//! scheme that PKCS#7 is a special case of, but which is strictly weaker: let `N` be the
//! smallest *non-zero* number of octets that brings the plaintext up to a multiple of the block
//! size `B` (so `1 ≤ N ≤ B`, and a plaintext that is already a whole number of blocks still
//! gets a whole block of padding). Append `N - 1` **arbitrary** octets and then a final octet
//! whose value is `N`. Only the last octet carries information; the filler is explicitly *not*
//! required to repeat `N`, and the spec's own worked example writes it as `0x616263????????05`.
//!
//! [`pad`] therefore fills the `N - 1` filler octets from the caller's random source rather
//! than repeating `N`, exactly as the specification illustrates, and [`unpad`] validates only
//! what the specification says is validatable: that the trailing length octet is in `1..=B` and
//! does not exceed the message. Rejecting non-repeating filler would break interoperability
//! with every conforming encryptor.
//!
//! # Padding oracles
//!
//! Because only one octet is checkable, CBC-mode XML Encryption is the textbook setting for
//! Vaudenay's padding-oracle attack, and for the Jager–Somorovsky adaptive chosen-ciphertext
//! attack on XML Encryption itself (CCS 2011) which turns *any* difference in a decryptor's
//! observable behaviour — an error message, an error code, a timing difference, a schema
//! validation failure downstream — into a full plaintext recovery oracle at roughly 14 queries
//! per octet. [XML Encryption 1.1 §6.1.1](https://www.w3.org/TR/xmlenc-core1/#sec-Attacks) and
//! §6.7 respond to it with two rules, both of which this crate follows:
//!
//! * **Prefer AEAD.** AES-GCM ([`crate::crypto::gcm`]) authenticates the ciphertext before any
//!   plaintext is produced, so no oracle exists. It is the default of
//!   [`EncryptOptions`](crate::EncryptOptions).
//! * **Never distinguish failures.** [`unpad`] returns the single opaque
//!   [`EncError::DecryptionFailed`], and the whole decrypt path funnels every failure into that
//!   one variant, so an application that reports the error verbatim still leaks nothing.
//!
//! What this crate *cannot* do for you is hide a timing difference introduced by the layers
//! above it, or the fact that a document either did or did not decrypt. An application that
//! must accept CBC ciphertexts from untrusted senders should treat a decryption failure as
//! fatal to the whole message and answer identically in every case.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::{aes::Aes, cbc};
//! use oxixml_enc::random::FixedRandom;
//!
//! let aes = Aes::new(&[0x2b; 16])?;
//! let iv = [0x00u8; 16];
//! let mut rng = FixedRandom::new(&[0xa5; 64]);
//!
//! let ciphertext = cbc::encrypt_padded(&aes, &iv, b"hello", &mut rng)?;
//! assert_eq!(ciphertext.len(), 16); // 5 octets of plaintext + 11 of padding
//! assert_eq!(cbc::decrypt_padded(&aes, &iv, &ciphertext)?, b"hello");
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use crate::crypto::aes::{Aes, BLOCK_SIZE};
use crate::crypto::ct;
use crate::error::{EncError, Result};
use crate::random::RandomSource;

/// Encrypts a plaintext whose length is already a multiple of the block size (SP 800-38A §6.2,
/// `CBC-AES-Encrypt`).
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when `plaintext` is not a whole number of blocks.
pub fn encrypt(aes: &Aes, iv: &[u8; BLOCK_SIZE], plaintext: &[u8]) -> Result<Vec<u8>> {
    if plaintext.len() % BLOCK_SIZE != 0 {
        return Err(EncError::InvalidBlockLength {
            block_size: BLOCK_SIZE,
            length: plaintext.len(),
        });
    }
    let mut output = Vec::with_capacity(plaintext.len());
    let mut chain = *iv;
    for block in plaintext.chunks_exact(BLOCK_SIZE) {
        for (chain_byte, plain_byte) in chain.iter_mut().zip(block.iter()) {
            *chain_byte ^= *plain_byte;
        }
        aes.encrypt_block(&mut chain);
        output.extend_from_slice(&chain);
    }
    Ok(output)
}

/// Decrypts a ciphertext whose length is a multiple of the block size, without removing padding
/// (SP 800-38A §6.2, `CBC-AES-Decrypt`).
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when `ciphertext` is not a whole number of blocks.
pub fn decrypt(aes: &Aes, iv: &[u8; BLOCK_SIZE], ciphertext: &[u8]) -> Result<Vec<u8>> {
    if ciphertext.len() % BLOCK_SIZE != 0 {
        return Err(EncError::InvalidBlockLength {
            block_size: BLOCK_SIZE,
            length: ciphertext.len(),
        });
    }
    let mut output = Vec::with_capacity(ciphertext.len());
    let mut chain = *iv;
    let mut block = [0u8; BLOCK_SIZE];
    for input in ciphertext.chunks_exact(BLOCK_SIZE) {
        block.copy_from_slice(input);
        let next_chain = block;
        aes.decrypt_block(&mut block);
        for (plain_byte, chain_byte) in block.iter_mut().zip(chain.iter()) {
            *plain_byte ^= *chain_byte;
        }
        output.extend_from_slice(&block);
        chain = next_chain;
    }
    Ok(output)
}

/// Appends XML Encryption padding (XMLEnc 1.1 §5.2.1) to `data`, extending it to the next
/// multiple of `block_size`.
///
/// A plaintext that is already block-aligned gains a whole extra block, because `N` must be
/// non-zero.
///
/// # Errors
///
/// Returns [`EncError::Entropy`] when the random source fails. Panics are impossible: the
/// function asserts nothing about `block_size` beyond the debug assertion below, and a
/// `block_size` of zero would be a programming error inside this crate rather than a runtime
/// condition.
///
/// # Panics
///
/// Panics in debug builds when `block_size` is zero or exceeds 255.
pub fn pad(data: &mut Vec<u8>, block_size: usize, rng: &mut dyn RandomSource) -> Result<()> {
    debug_assert!(
        (1..=255).contains(&block_size),
        "the padding length must fit in one octet"
    );
    let remainder = data.len() % block_size;
    let padding_len = block_size - remainder;
    let start = data.len();
    data.resize(start + padding_len, 0);
    if padding_len > 1 {
        rng.fill(&mut data[start..start + padding_len - 1])?;
    }
    // The specification requires the *last* octet to be the padding length; the filler before
    // it is arbitrary, and is therefore left as the random bytes just written.
    let last = data.len() - 1;
    data[last] = padding_len as u8;
    Ok(())
}

/// Removes XML Encryption padding, returning the plaintext prefix.
///
/// # Errors
///
/// Returns [`EncError::DecryptionFailed`] when the trailing length octet is zero, larger than
/// `block_size`, or larger than the message — deliberately the same error the caller gets for a
/// wrong key, so that nothing distinguishes the two. See the [module documentation](self).
pub fn unpad(data: &[u8], block_size: usize) -> Result<&[u8]> {
    let Some(&declared) = data.last() else {
        return Err(EncError::DecryptionFailed);
    };
    let padding_len = declared as usize;

    // Fold the three conditions into one mask so the check itself does not branch on the
    // (secret) padding octet. `data.len()` and `block_size` are public.
    let non_zero = !ct::mask_is_zero(declared);
    let within_block = ct::mask_le(padding_len, block_size);
    let within_message = ct::mask_le(padding_len, data.len());
    let acceptable = non_zero & within_block & within_message;

    if acceptable != 0xff {
        return Err(EncError::DecryptionFailed);
    }
    Ok(&data[..data.len() - padding_len])
}

/// Encrypts `plaintext` with CBC and XML Encryption padding in one step.
///
/// The initialization vector is *not* prepended; [`crate::cipher`] does that where the
/// specification requires it.
///
/// # Errors
///
/// Returns [`EncError::Entropy`] when the random source fails.
pub fn encrypt_padded(
    aes: &Aes,
    iv: &[u8; BLOCK_SIZE],
    plaintext: &[u8],
    rng: &mut dyn RandomSource,
) -> Result<Vec<u8>> {
    let mut buffer = plaintext.to_vec();
    pad(&mut buffer, BLOCK_SIZE, rng)?;
    encrypt(aes, iv, &buffer)
}

/// Decrypts a CBC ciphertext and strips its XML Encryption padding.
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when the ciphertext is not a whole number of
/// blocks, and [`EncError::DecryptionFailed`] when the padding does not validate.
pub fn decrypt_padded(aes: &Aes, iv: &[u8; BLOCK_SIZE], ciphertext: &[u8]) -> Result<Vec<u8>> {
    if ciphertext.is_empty() {
        return Err(EncError::DecryptionFailed);
    }
    let padded = decrypt(aes, iv, ciphertext)?;
    let plaintext = unpad(&padded, BLOCK_SIZE)?;
    Ok(plaintext.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::FixedRandom;

    #[test]
    fn padding_always_adds_between_one_and_a_whole_block() -> Result<()> {
        let mut rng = FixedRandom::new(&[0x5a; 128]);
        for length in 0..40usize {
            let mut data = vec![0xccu8; length];
            pad(&mut data, BLOCK_SIZE, &mut rng)?;
            assert!(data.len() > length);
            assert!(data.len() - length <= BLOCK_SIZE);
            assert_eq!(data.len() % BLOCK_SIZE, 0);
            assert_eq!(unpad(&data, BLOCK_SIZE)?, vec![0xccu8; length].as_slice());
        }
        Ok(())
    }

    #[test]
    fn a_block_aligned_plaintext_gains_a_whole_block() -> Result<()> {
        let mut rng = FixedRandom::new(&[0x11; 64]);
        let mut data = vec![0u8; BLOCK_SIZE];
        pad(&mut data, BLOCK_SIZE, &mut rng)?;
        assert_eq!(data.len(), 2 * BLOCK_SIZE);
        assert_eq!(data[data.len() - 1], BLOCK_SIZE as u8);
        Ok(())
    }

    #[test]
    fn filler_octets_are_not_required_to_repeat_the_length() -> Result<()> {
        // A conforming encryptor may emit anything for the filler; we must accept it.
        let mut data = vec![0x61, 0x62, 0x63];
        data.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, 0x00, 0x99, 0x77, 0x12]);
        data.extend_from_slice(&[0x34, 0x56, 0x78, 0x9a, 0x0d]);
        assert_eq!(data.len(), BLOCK_SIZE);
        assert_eq!(unpad(&data, BLOCK_SIZE)?, &[0x61, 0x62, 0x63]);
        Ok(())
    }

    #[test]
    fn rejects_impossible_padding_lengths() {
        let mut data = vec![0u8; BLOCK_SIZE];
        data[BLOCK_SIZE - 1] = 0;
        assert!(matches!(
            unpad(&data, BLOCK_SIZE),
            Err(EncError::DecryptionFailed)
        ));
        data[BLOCK_SIZE - 1] = 17;
        assert!(matches!(
            unpad(&data, BLOCK_SIZE),
            Err(EncError::DecryptionFailed)
        ));
        assert!(matches!(
            unpad(&[], BLOCK_SIZE),
            Err(EncError::DecryptionFailed)
        ));
        // Longer than the message even though it fits in a block.
        assert!(matches!(
            unpad(&[0x08], BLOCK_SIZE),
            Err(EncError::DecryptionFailed)
        ));
    }

    #[test]
    fn rejects_ragged_inputs() -> Result<()> {
        let aes = Aes::new(&[0u8; 16])?;
        assert!(matches!(
            encrypt(&aes, &[0u8; BLOCK_SIZE], &[0u8; 17]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        assert!(matches!(
            decrypt(&aes, &[0u8; BLOCK_SIZE], &[0u8; 17]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        Ok(())
    }

    #[test]
    fn round_trips_at_every_length() -> Result<()> {
        let aes = Aes::new(&[0x42u8; 32])?;
        let iv = [0x9eu8; BLOCK_SIZE];
        let mut rng = FixedRandom::new(&[0x77; 256]);
        for length in 0..48usize {
            let plaintext: Vec<u8> = (0..length).map(|index| index as u8).collect();
            let ciphertext = encrypt_padded(&aes, &iv, &plaintext, &mut rng)?;
            assert_eq!(decrypt_padded(&aes, &iv, &ciphertext)?, plaintext);
        }
        Ok(())
    }

    #[test]
    fn an_empty_ciphertext_is_a_decryption_failure() -> Result<()> {
        let aes = Aes::new(&[0u8; 16])?;
        assert!(matches!(
            decrypt_padded(&aes, &[0u8; BLOCK_SIZE], &[]),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }
}
