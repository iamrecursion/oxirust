//! Galois/Counter Mode, implemented from
//! [NIST SP 800-38D](https://doi.org/10.6028/NIST.SP.800-38D).
//!
//! GCM is the authenticated encryption mode XML Encryption 1.1 §5.2.4 introduced precisely to
//! escape the CBC padding-oracle attacks of §6.1.1: the tag is verified *before* any plaintext
//! is handed back, so a decryptor exposes no oracle no matter how it reports failure.
//!
//! # Structure
//!
//! GCM is `GCTR` (counter mode with a 32-bit counter, [`crate::crypto::ctr`]) for
//! confidentiality plus `GHASH` — a universal hash in `GF(2^128)` — for authenticity:
//!
//! * `H = CIPH_K(0^128)`, the hash subkey.
//! * `J0`, the pre-counter block: for the 96-bit initialization vector that XML Encryption
//!   mandates, `J0 = IV ‖ 0^31 ‖ 1`; for any other length, `J0 = GHASH_H(IV ‖ 0^s ‖ [0]_64 ‖
//!   [len(IV)]_64)`. Both are implemented.
//! * `C = GCTR_K(inc32(J0), P)`.
//! * `S = GHASH_H(A ‖ 0^v ‖ C ‖ 0^u ‖ [len(A)]_64 ‖ [len(C)]_64)`.
//! * `T = MSB_t(GCTR_K(J0, S))`.
//!
//! # The field
//!
//! `GHASH` works in `GF(2^128)` modulo `x^128 + x^7 + x^2 + x + 1`, but with SP 800-38D's
//! *reflected* bit convention: the leftmost bit of a block is the coefficient of `x^0`. `gf_mul`
//! implements the right-shift algorithm of §6.3 directly on that convention, which is why the
//! reduction constant appears as `0xe1` in the top octet rather than `0x87` in the bottom one.
//! The multiplication is a fixed 128-iteration loop with no data-dependent branch or memory
//! access, so it is inherently free of the cache-timing channel that table-driven `GHASH`
//! implementations have; it is correspondingly slow.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::{aes::Aes, gcm};
//!
//! let aes = Aes::new(&[0u8; 16])?;
//! let iv = [0u8; 12];
//! // McGrew–Viega / SP 800-38D test case 2.
//! let (ciphertext, tag) = gcm::encrypt(&aes, &iv, &[0u8; 16], &[])?;
//! assert_eq!(&ciphertext[..4], &[0x03, 0x88, 0xda, 0xce]);
//! assert_eq!(gcm::decrypt(&aes, &iv, &ciphertext, &[], &tag)?, vec![0u8; 16]);
//!
//! // Flipping one bit of the tag makes decryption fail.
//! let mut broken = tag;
//! broken[0] ^= 1;
//! assert!(gcm::decrypt(&aes, &iv, &ciphertext, &[], &broken).is_err());
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use crate::crypto::aes::{Aes, BLOCK_SIZE};
use crate::crypto::ct;
use crate::crypto::ctr;
use crate::error::{EncError, Result};

/// The initialization-vector length in octets that XML Encryption 1.1 §5.2.4 mandates.
pub const IV_LEN: usize = 12;

/// The authentication-tag length in octets that XML Encryption 1.1 §5.2.4 mandates.
pub const TAG_LEN: usize = 16;

/// The reduction constant `R = 11100001 ‖ 0^120` of SP 800-38D §6.3.
const R: u128 = 0xe100_0000_0000_0000_0000_0000_0000_0000;

/// Multiplies two blocks in `GF(2^128)` under the reflected convention of SP 800-38D §6.3,
/// Algorithm 1.
///
/// Blocks are carried as `u128` in big-endian bit order, so bit 127 of the integer is the
/// leftmost bit of the block, which is the coefficient of `x^0`.
#[must_use]
fn gf_mul(x: u128, y: u128) -> u128 {
    let mut product = 0u128;
    let mut value = y;
    for bit in 0..128u32 {
        // Branch-free conditional accumulate: the mask is all-ones when bit `i` of `x` is set.
        let mask = 0u128.wrapping_sub((x >> (127 - bit)) & 1);
        product ^= value & mask;
        // V ← V >> 1, then V ← V ⊕ R when the bit shifted out was set.
        let carry = 0u128.wrapping_sub(value & 1);
        value = (value >> 1) ^ (R & carry);
    }
    product
}

/// The `GHASH` function of SP 800-38D §6.4, streamed one block at a time.
#[derive(Clone)]
struct GHash {
    subkey: u128,
    accumulator: u128,
}

impl GHash {
    /// Creates a hasher from the hash subkey `H`.
    const fn new(subkey: u128) -> Self {
        Self {
            subkey,
            accumulator: 0,
        }
    }

    /// Absorbs one 16-octet block.
    fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        self.accumulator = gf_mul(self.accumulator ^ u128::from_be_bytes(*block), self.subkey);
    }

    /// Absorbs `data`, zero-padding the final partial block on the right as §6.4 requires.
    fn update(&mut self, data: &[u8]) {
        for chunk in data.chunks(BLOCK_SIZE) {
            let mut block = [0u8; BLOCK_SIZE];
            block[..chunk.len()].copy_from_slice(chunk);
            self.update_block(&block);
        }
    }

    /// Returns the accumulated hash.
    const fn finish(self) -> [u8; BLOCK_SIZE] {
        self.accumulator.to_be_bytes()
    }
}

/// Computes the hash subkey `H = CIPH_K(0^128)`.
fn hash_subkey(aes: &Aes) -> u128 {
    let mut block = [0u8; BLOCK_SIZE];
    aes.encrypt_block(&mut block);
    u128::from_be_bytes(block)
}

/// Derives the pre-counter block `J0` (SP 800-38D §7.1, step 2).
fn pre_counter_block(subkey: u128, iv: &[u8]) -> [u8; BLOCK_SIZE] {
    if iv.len() == IV_LEN {
        let mut block = [0u8; BLOCK_SIZE];
        block[..IV_LEN].copy_from_slice(iv);
        block[BLOCK_SIZE - 1] = 1;
        return block;
    }
    let mut hash = GHash::new(subkey);
    hash.update(iv);
    let mut length_block = [0u8; BLOCK_SIZE];
    let bits = (iv.len() as u64).wrapping_mul(8);
    length_block[BLOCK_SIZE - 8..].copy_from_slice(&bits.to_be_bytes());
    hash.update_block(&length_block);
    hash.finish()
}

/// Computes `S = GHASH_H(A ‖ 0^v ‖ C ‖ 0^u ‖ [len(A)]_64 ‖ [len(C)]_64)`.
fn ghash_lengths(subkey: u128, aad: &[u8], ciphertext: &[u8]) -> [u8; BLOCK_SIZE] {
    let mut hash = GHash::new(subkey);
    hash.update(aad);
    hash.update(ciphertext);
    let mut length_block = [0u8; BLOCK_SIZE];
    let aad_bits = (aad.len() as u64).wrapping_mul(8);
    let ciphertext_bits = (ciphertext.len() as u64).wrapping_mul(8);
    length_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
    length_block[8..].copy_from_slice(&ciphertext_bits.to_be_bytes());
    hash.update_block(&length_block);
    hash.finish()
}

/// Computes the tag `T = MSB_128(GCTR_K(J0, S))`.
fn compute_tag(aes: &Aes, j0: &[u8; BLOCK_SIZE], s: &[u8; BLOCK_SIZE]) -> [u8; TAG_LEN] {
    let mut tag = *s;
    ctr::apply(aes, j0, &mut tag, ctr::increment32);
    tag
}

/// `GCM-AE_K(IV, P, A)` of SP 800-38D §7.1, returning the ciphertext and the 128-bit tag.
///
/// The initialization vector is *not* prepended and the tag is *not* appended; the XML
/// Encryption framing that does both lives in [`crate::cipher`].
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when `iv` is empty, which SP 800-38D forbids
/// (`1 ≤ len(IV) ≤ 2^64 - 1`).
pub fn encrypt(
    aes: &Aes,
    iv: &[u8],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<(Vec<u8>, [u8; TAG_LEN])> {
    if iv.is_empty() {
        return Err(EncError::InvalidBlockLength {
            block_size: IV_LEN,
            length: 0,
        });
    }
    let subkey = hash_subkey(aes);
    let j0 = pre_counter_block(subkey, iv);

    let mut ciphertext = plaintext.to_vec();
    let mut counter = j0;
    ctr::increment32(&mut counter);
    ctr::apply(aes, &counter, &mut ciphertext, ctr::increment32);

    let s = ghash_lengths(subkey, aad, &ciphertext);
    let tag = compute_tag(aes, &j0, &s);
    Ok((ciphertext, tag))
}

/// `GCM-AD_K(IV, C, A, T)` of SP 800-38D §7.2.
///
/// The tag is verified before any plaintext is returned, and the comparison is the
/// data-independent [`ct::eq`], so neither the answer nor the time taken reveals where the
/// first differing octet was.
///
/// # Errors
///
/// Returns [`EncError::DecryptionFailed`] when the tag does not verify, and
/// [`EncError::InvalidBlockLength`] when `iv` is empty.
pub fn decrypt(aes: &Aes, iv: &[u8], ciphertext: &[u8], aad: &[u8], tag: &[u8]) -> Result<Vec<u8>> {
    if iv.is_empty() {
        return Err(EncError::InvalidBlockLength {
            block_size: IV_LEN,
            length: 0,
        });
    }
    let subkey = hash_subkey(aes);
    let j0 = pre_counter_block(subkey, iv);
    let s = ghash_lengths(subkey, aad, ciphertext);
    let expected = compute_tag(aes, &j0, &s);

    if !ct::eq(&expected, tag) {
        return Err(EncError::DecryptionFailed);
    }

    let mut plaintext = ciphertext.to_vec();
    let mut counter = j0;
    ctr::increment32(&mut counter);
    ctr::apply(aes, &counter, &mut plaintext, ctr::increment32);
    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_has_an_identity() {
        // Under the reflected convention the multiplicative identity is `1 ‖ 0^127`, i.e. the
        // integer with only bit 127 set.
        let one = 1u128 << 127;
        for value in [
            0u128,
            1,
            u128::MAX,
            0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
        ] {
            assert_eq!(gf_mul(value, one), value);
            assert_eq!(gf_mul(one, value), value);
            assert_eq!(gf_mul(value, 0), 0);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_distributive() {
        let a = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeffu128;
        let b = 0xfedc_ba98_7654_3210_0f1e_2d3c_4b5a_6978u128;
        let c = 0xdead_beef_cafe_babe_1234_5678_9abc_def0u128;
        assert_eq!(gf_mul(a, b), gf_mul(b, a));
        assert_eq!(gf_mul(a, b ^ c), gf_mul(a, b) ^ gf_mul(a, c));
    }

    #[test]
    fn subkey_matches_the_published_value_for_the_all_zero_key() -> Result<()> {
        // McGrew–Viega test case 1 publishes H for K = 0^128.
        let aes = Aes::new(&[0u8; 16])?;
        assert_eq!(
            hash_subkey(&aes).to_be_bytes(),
            [
                0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b, 0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34,
                0x2b, 0x2e
            ]
        );
        Ok(())
    }

    #[test]
    fn a_non_standard_iv_length_uses_the_ghash_derivation() -> Result<()> {
        let aes = Aes::new(&[0x11u8; 16])?;
        // An 8-octet IV takes the GHASH branch; the round trip must still work.
        let (ciphertext, tag) = encrypt(&aes, &[0x22u8; 8], b"payload", b"header")?;
        assert_eq!(
            decrypt(&aes, &[0x22u8; 8], &ciphertext, b"header", &tag)?,
            b"payload"
        );
        Ok(())
    }

    #[test]
    fn an_empty_iv_is_rejected() -> Result<()> {
        let aes = Aes::new(&[0u8; 16])?;
        assert!(matches!(
            encrypt(&aes, &[], b"x", &[]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        assert!(matches!(
            decrypt(&aes, &[], b"x", &[], &[0u8; TAG_LEN]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        Ok(())
    }

    #[test]
    fn tampering_with_the_aad_is_detected() -> Result<()> {
        let aes = Aes::new(&[0x33u8; 32])?;
        let iv = [0x44u8; IV_LEN];
        let (ciphertext, tag) = encrypt(&aes, &iv, b"secret", b"authenticated")?;
        assert!(matches!(
            decrypt(&aes, &iv, &ciphertext, b"authenticatee", &tag),
            Err(EncError::DecryptionFailed)
        ));
        assert!(matches!(
            decrypt(&aes, &iv, &ciphertext, b"", &tag),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn a_truncated_tag_never_verifies() -> Result<()> {
        let aes = Aes::new(&[0x55u8; 16])?;
        let iv = [0x66u8; IV_LEN];
        let (ciphertext, tag) = encrypt(&aes, &iv, b"data", &[])?;
        assert!(matches!(
            decrypt(&aes, &iv, &ciphertext, &[], &tag[..8]),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }
}
