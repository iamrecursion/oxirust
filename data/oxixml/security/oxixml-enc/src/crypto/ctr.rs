//! Counter mode (NIST SP 800-38A §6.5) and the 32-bit-counter variant GCM builds on
//! (NIST SP 800-38D §6.5, `GCTR`).
//!
//! Counter mode turns a block cipher into a stream cipher: the cipher is applied to a sequence
//! of counter blocks and the results are exclusive-ored with the data, so encryption and
//! decryption are the same operation. **A counter block must never repeat under one key.** GCM
//! guarantees that by deriving the initial block from a unique initialization vector; a caller
//! using [`apply`] directly is responsible for it.
//!
//! Two counter conventions are provided because the two standards differ:
//!
//! * [`increment_full`] — SP 800-38A's "standard incrementing function" over the whole
//!   128-bit block.
//! * [`increment32`] — SP 800-38D's `inc32`, which increments only the rightmost 32 bits and
//!   leaves the leading 96 bits (the nonce) alone.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::{aes::Aes, ctr};
//!
//! // NIST SP 800-38A §F.5.1, CTR-AES128.Encrypt, first block.
//! let key = [
//!     0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
//!     0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
//! ];
//! let counter = [
//!     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
//!     0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
//! ];
//! let mut data = [
//!     0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
//!     0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
//! ];
//! ctr::apply(&Aes::new(&key)?, &counter, &mut data, ctr::increment_full);
//! assert_eq!(&data[..4], &[0x87, 0x4d, 0x61, 0x91]);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use crate::crypto::aes::{Aes, BLOCK_SIZE};

/// The standard incrementing function of SP 800-38A §B.1 applied to the whole block.
pub fn increment_full(counter: &mut [u8; BLOCK_SIZE]) {
    for byte in counter.iter_mut().rev() {
        let (next, carried) = byte.overflowing_add(1);
        *byte = next;
        if !carried {
            return;
        }
    }
}

/// `inc32` of SP 800-38D §6.2: increment the rightmost 32 bits modulo `2^32`, leaving the
/// leading 96 bits untouched.
pub fn increment32(counter: &mut [u8; BLOCK_SIZE]) {
    for byte in counter[BLOCK_SIZE - 4..].iter_mut().rev() {
        let (next, carried) = byte.overflowing_add(1);
        *byte = next;
        if !carried {
            return;
        }
    }
}

/// Applies the counter-mode keystream starting at `initial_counter` to `data`, in place.
///
/// `increment` selects the counter convention; pass [`increment_full`] for SP 800-38A CTR and
/// [`increment32`] for the `GCTR` of GCM. The final partial block uses only as much keystream
/// as it needs, as both standards require.
pub fn apply(
    aes: &Aes,
    initial_counter: &[u8; BLOCK_SIZE],
    data: &mut [u8],
    increment: fn(&mut [u8; BLOCK_SIZE]),
) {
    let mut counter = *initial_counter;
    for chunk in data.chunks_mut(BLOCK_SIZE) {
        let mut keystream = counter;
        aes.encrypt_block(&mut keystream);
        for (byte, key_byte) in chunk.iter_mut().zip(keystream.iter()) {
            *byte ^= *key_byte;
        }
        increment(&mut counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn full_increment_carries_across_the_whole_block() {
        let mut counter = [0xffu8; BLOCK_SIZE];
        increment_full(&mut counter);
        assert_eq!(counter, [0u8; BLOCK_SIZE]);

        let mut counter = [0u8; BLOCK_SIZE];
        counter[BLOCK_SIZE - 1] = 0xff;
        increment_full(&mut counter);
        assert_eq!(counter[BLOCK_SIZE - 1], 0x00);
        assert_eq!(counter[BLOCK_SIZE - 2], 0x01);
    }

    #[test]
    fn inc32_leaves_the_nonce_alone() {
        let mut counter = [0x11u8; BLOCK_SIZE];
        counter[BLOCK_SIZE - 4..].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        increment32(&mut counter);
        assert_eq!(&counter[..BLOCK_SIZE - 4], &[0x11u8; BLOCK_SIZE - 4]);
        assert_eq!(&counter[BLOCK_SIZE - 4..], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn counter_mode_is_its_own_inverse() -> Result<()> {
        let aes = Aes::new(&[0x24u8; 24])?;
        let counter = [0x07u8; BLOCK_SIZE];
        let original: Vec<u8> = (0..70u8).collect();
        let mut data = original.clone();
        apply(&aes, &counter, &mut data, increment_full);
        assert_ne!(data, original);
        apply(&aes, &counter, &mut data, increment_full);
        assert_eq!(data, original);
        Ok(())
    }
}
