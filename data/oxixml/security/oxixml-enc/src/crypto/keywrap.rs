//! The AES Key Wrap algorithm of [RFC 3394](https://www.rfc-editor.org/rfc/rfc3394) and the
//! padded variant of [RFC 5649](https://www.rfc-editor.org/rfc/rfc5649).
//!
//! XML Encryption uses key wrapping to protect a content-encryption key with a key-encryption
//! key that both parties already share: the `kw-aes128`/`kw-aes192`/`kw-aes256` algorithms of
//! XMLEnc §5.7 are exactly RFC 3394, and the reserved `kw-aes-*-pad` identifiers of XMLEnc 1.1
//! Appendix A are RFC 5649.
//!
//! # The algorithm
//!
//! RFC 3394 §2.2.1 wraps `n` 64-bit blocks `P₁ … Pₙ` under a key-encryption key with six passes
//! over the data. `A` starts as the fixed *initial value* `A6A6A6A6A6A6A6A6`; each step encrypts
//! `A ‖ Rᵢ` with AES-ECB, keeps the left half as the new `A` after exclusive-oring in the step
//! counter `t = (j·n) + i`, and keeps the right half as the new `Rᵢ`. Unwrapping runs the same
//! schedule backwards and succeeds only if `A` comes back equal to the initial value, which is
//! what makes key wrap an *integrity-protecting* transform rather than plain encryption: a
//! wrong key-encryption key is caught with probability `1 - 2^-64`.
//!
//! RFC 5649 keeps that machinery but replaces the fixed initial value with
//! `A65959A6 ‖ MLI`, where `MLI` is the true (unpadded) length of the key material, and
//! zero-pads the material to a multiple of eight octets. That makes it usable for key material
//! whose length is not a multiple of eight — which RFC 3394 cannot handle at all — and adds the
//! degenerate single-block case that is a single AES encryption rather than six passes.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::{aes::Aes, keywrap};
//!
//! // RFC 3394 §4.1.
//! let kek = Aes::new(&[
//!     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
//!     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
//! ])?;
//! let key = [
//!     0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
//!     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
//! ];
//! let wrapped = keywrap::wrap(&kek, &key)?;
//! assert_eq!(&wrapped[..4], &[0x1f, 0xa6, 0x8b, 0x0a]);
//! assert_eq!(keywrap::unwrap(&kek, &wrapped)?, key);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use crate::crypto::aes::{Aes, BLOCK_SIZE};
use crate::crypto::ct;
use crate::error::{EncError, Result};

/// One semiblock: half an AES block, the unit RFC 3394 operates on.
const SEMIBLOCK: usize = 8;

/// The default initial value `A6A6A6A6A6A6A6A6` of RFC 3394 §2.2.3.1.
const DEFAULT_IV: [u8; SEMIBLOCK] = [0xa6; SEMIBLOCK];

/// The alternative initial value prefix `A65959A6` of RFC 5649 §3.
const PADDED_IV_PREFIX: [u8; 4] = [0xa6, 0x59, 0x59, 0xa6];

/// The smallest number of semiblocks RFC 3394 accepts (`n ≥ 2`).
const MIN_SEMIBLOCKS: usize = 2;

/// Wraps `key_material` with the key-encryption key `kek` (RFC 3394 §2.2.1).
///
/// The result is `key_material.len() + 8` octets long.
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] unless `key_material` is a multiple of eight octets
/// long and at least sixteen octets — RFC 3394 is defined only for `n ≥ 2` semiblocks. Use
/// [`wrap_padded`] for key material of any other length.
pub fn wrap(kek: &Aes, key_material: &[u8]) -> Result<Vec<u8>> {
    if key_material.len() % SEMIBLOCK != 0 || key_material.len() / SEMIBLOCK < MIN_SEMIBLOCKS {
        return Err(EncError::InvalidBlockLength {
            block_size: SEMIBLOCK,
            length: key_material.len(),
        });
    }
    Ok(wrap_with_iv(kek, key_material, &DEFAULT_IV))
}

/// The shared body of [`wrap`] and [`wrap_padded`]: six passes over `n ≥ 2` semiblocks.
fn wrap_with_iv(kek: &Aes, key_material: &[u8], iv: &[u8; SEMIBLOCK]) -> Vec<u8> {
    let count = key_material.len() / SEMIBLOCK;
    let mut a = *iv;
    let mut registers = key_material.to_vec();

    let mut block = [0u8; BLOCK_SIZE];
    for pass in 0..6u64 {
        for index in 0..count {
            block[..SEMIBLOCK].copy_from_slice(&a);
            block[SEMIBLOCK..]
                .copy_from_slice(&registers[index * SEMIBLOCK..(index + 1) * SEMIBLOCK]);
            kek.encrypt_block(&mut block);

            // t = (n · j) + i, with i counted from one.
            let step = pass * count as u64 + index as u64 + 1;
            a.copy_from_slice(&block[..SEMIBLOCK]);
            for (offset, byte) in step.to_be_bytes().iter().enumerate() {
                a[offset] ^= *byte;
            }
            registers[index * SEMIBLOCK..(index + 1) * SEMIBLOCK]
                .copy_from_slice(&block[SEMIBLOCK..]);
        }
    }

    let mut output = Vec::with_capacity(key_material.len() + SEMIBLOCK);
    output.extend_from_slice(&a);
    output.extend_from_slice(&registers);
    output
}

/// Unwraps `wrapped` with the key-encryption key `kek` and verifies the integrity check
/// (RFC 3394 §2.2.2).
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when `wrapped` is not a multiple of eight octets or
/// is shorter than three semiblocks, and [`EncError::DecryptionFailed`] when the recovered
/// initial value is not `A6A6A6A6A6A6A6A6` — the same opaque error a wrong key produces
/// everywhere else in this crate.
pub fn unwrap(kek: &Aes, wrapped: &[u8]) -> Result<Vec<u8>> {
    let (recovered_iv, key_material) = unwrap_raw(kek, wrapped)?;
    if !ct::eq(&recovered_iv, &DEFAULT_IV) {
        return Err(EncError::DecryptionFailed);
    }
    Ok(key_material)
}

/// The shared body of [`unwrap`] and [`unwrap_padded`]: runs the schedule backwards and returns
/// the recovered initial value alongside the key material, without judging either.
fn unwrap_raw(kek: &Aes, wrapped: &[u8]) -> Result<([u8; SEMIBLOCK], Vec<u8>)> {
    if wrapped.len() % SEMIBLOCK != 0 || wrapped.len() / SEMIBLOCK < MIN_SEMIBLOCKS + 1 {
        return Err(EncError::InvalidBlockLength {
            block_size: SEMIBLOCK,
            length: wrapped.len(),
        });
    }
    let count = wrapped.len() / SEMIBLOCK - 1;
    let mut a = [0u8; SEMIBLOCK];
    a.copy_from_slice(&wrapped[..SEMIBLOCK]);
    let mut registers = wrapped[SEMIBLOCK..].to_vec();

    let mut block = [0u8; BLOCK_SIZE];
    for pass in (0..6u64).rev() {
        for index in (0..count).rev() {
            let step = pass * count as u64 + index as u64 + 1;
            block[..SEMIBLOCK].copy_from_slice(&a);
            for (offset, byte) in step.to_be_bytes().iter().enumerate() {
                block[offset] ^= *byte;
            }
            block[SEMIBLOCK..]
                .copy_from_slice(&registers[index * SEMIBLOCK..(index + 1) * SEMIBLOCK]);
            kek.decrypt_block(&mut block);

            a.copy_from_slice(&block[..SEMIBLOCK]);
            registers[index * SEMIBLOCK..(index + 1) * SEMIBLOCK]
                .copy_from_slice(&block[SEMIBLOCK..]);
        }
    }
    Ok((a, registers))
}

/// Wraps key material of *any* non-empty length with the padded variant of RFC 5649 §4.1.
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when `key_material` is empty, and
/// [`EncError::MessageTooLong`] when it is longer than `2^32 - 1` octets, the largest length
/// the message-length indicator can express.
pub fn wrap_padded(kek: &Aes, key_material: &[u8]) -> Result<Vec<u8>> {
    if key_material.is_empty() {
        return Err(EncError::InvalidBlockLength {
            block_size: SEMIBLOCK,
            length: 0,
        });
    }
    let Ok(length) = u32::try_from(key_material.len()) else {
        return Err(EncError::MessageTooLong {
            length: key_material.len(),
            maximum: u32::MAX as usize,
        });
    };

    let mut iv = [0u8; SEMIBLOCK];
    iv[..4].copy_from_slice(&PADDED_IV_PREFIX);
    iv[4..].copy_from_slice(&length.to_be_bytes());

    let padded_len = key_material.len().next_multiple_of(SEMIBLOCK);
    let mut padded = key_material.to_vec();
    padded.resize(padded_len, 0);

    if padded_len == SEMIBLOCK {
        // RFC 5649 §4.1 step 2: a single semiblock is encrypted directly.
        let mut block = [0u8; BLOCK_SIZE];
        block[..SEMIBLOCK].copy_from_slice(&iv);
        block[SEMIBLOCK..].copy_from_slice(&padded);
        kek.encrypt_block(&mut block);
        return Ok(block.to_vec());
    }
    Ok(wrap_with_iv(kek, &padded, &iv))
}

/// Unwraps key material wrapped by [`wrap_padded`] (RFC 5649 §4.2).
///
/// # Errors
///
/// Returns [`EncError::InvalidBlockLength`] when the input is not a multiple of eight octets or
/// is shorter than two semiblocks, and [`EncError::DecryptionFailed`] when the alternative
/// initial value, the message-length indicator or the zero padding does not check out.
pub fn unwrap_padded(kek: &Aes, wrapped: &[u8]) -> Result<Vec<u8>> {
    if wrapped.len() % SEMIBLOCK != 0 || wrapped.len() < 2 * SEMIBLOCK {
        return Err(EncError::InvalidBlockLength {
            block_size: SEMIBLOCK,
            length: wrapped.len(),
        });
    }

    let (recovered_iv, padded) = if wrapped.len() == BLOCK_SIZE {
        let mut block = [0u8; BLOCK_SIZE];
        block.copy_from_slice(wrapped);
        kek.decrypt_block(&mut block);
        let mut iv = [0u8; SEMIBLOCK];
        iv.copy_from_slice(&block[..SEMIBLOCK]);
        (iv, block[SEMIBLOCK..].to_vec())
    } else {
        unwrap_raw(kek, wrapped)?
    };

    if !ct::eq(&recovered_iv[..4], &PADDED_IV_PREFIX) {
        return Err(EncError::DecryptionFailed);
    }
    let mut length_bytes = [0u8; 4];
    length_bytes.copy_from_slice(&recovered_iv[4..]);
    let length = u32::from_be_bytes(length_bytes) as usize;

    // The declared length must land in the last semiblock, and the padding must be zero.
    if length == 0 || length > padded.len() || padded.len() - length >= SEMIBLOCK {
        return Err(EncError::DecryptionFailed);
    }
    let mut padding_difference = 0u8;
    for byte in &padded[length..] {
        padding_difference |= *byte;
    }
    if padding_difference != 0 {
        return Err(EncError::DecryptionFailed);
    }
    Ok(padded[..length].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kek(length: usize) -> Result<Aes> {
        let key: Vec<u8> = (0..length as u8).collect();
        Aes::new(&key)
    }

    #[test]
    fn rejects_key_material_that_is_not_semiblock_aligned() -> Result<()> {
        let kek = kek(16)?;
        assert!(matches!(
            wrap(&kek, &[0u8; 17]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        assert!(matches!(
            wrap(&kek, &[0u8; 8]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        Ok(())
    }

    #[test]
    fn a_wrong_key_encryption_key_fails_the_integrity_check() -> Result<()> {
        let wrapped = wrap(&kek(16)?, &[0x11u8; 32])?;
        let wrong = Aes::new(&[0xffu8; 16])?;
        assert!(matches!(
            unwrap(&wrong, &wrapped),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn a_flipped_bit_fails_the_integrity_check() -> Result<()> {
        let kek = kek(32)?;
        let mut wrapped = wrap(&kek, &[0x22u8; 24])?;
        wrapped[9] ^= 0x01;
        assert!(matches!(
            unwrap(&kek, &wrapped),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn padded_wrap_round_trips_at_every_length() -> Result<()> {
        let kek = kek(24)?;
        for length in 1..40usize {
            let material: Vec<u8> = (0..length).map(|index| index as u8 ^ 0x5a).collect();
            let wrapped = wrap_padded(&kek, &material)?;
            assert_eq!(
                wrapped.len(),
                length.next_multiple_of(SEMIBLOCK) + SEMIBLOCK
            );
            assert_eq!(unwrap_padded(&kek, &wrapped)?, material);
        }
        Ok(())
    }

    #[test]
    fn padded_wrap_rejects_empty_material_and_bad_ciphertexts() -> Result<()> {
        let kek = kek(16)?;
        assert!(matches!(
            wrap_padded(&kek, &[]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        assert!(matches!(
            unwrap_padded(&kek, &[0u8; 8]),
            Err(EncError::InvalidBlockLength { .. })
        ));
        let mut wrapped = wrap_padded(&kek, b"abc")?;
        wrapped[0] ^= 0x80;
        assert!(matches!(
            unwrap_padded(&kek, &wrapped),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn the_two_variants_do_not_accept_each_others_output() -> Result<()> {
        let kek = kek(16)?;
        let wrapped = wrap(&kek, &[0x77u8; 16])?;
        assert!(matches!(
            unwrap_padded(&kek, &wrapped),
            Err(EncError::DecryptionFailed)
        ));
        let padded = wrap_padded(&kek, &[0x77u8; 16])?;
        assert!(matches!(
            unwrap(&kek, &padded),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }
}
