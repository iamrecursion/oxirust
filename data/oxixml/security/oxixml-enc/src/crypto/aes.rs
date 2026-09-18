//! The AES block cipher, implemented directly from
//! [FIPS 197](https://doi.org/10.6028/NIST.FIPS.197-upd1).
//!
//! # No transcribed magic tables
//!
//! Every table this module uses — the S-box, its inverse and the six
//! `GF(2^8)` multiplication tables needed by `MixColumns`/`InvMixColumns` — is *derived at
//! compile time* from the field definition in FIPS 197 §4 (the irreducible polynomial
//! `x^8 + x^4 + x^3 + x + 1`, i.e. `0x11b`) by `const fn`s in this file. Nothing is copied out
//! of the standard by hand, so there is no transcription to get wrong, and the tables are
//! provably the ones the standard defines rather than the ones someone typed.
//!
//! # State layout
//!
//! FIPS 197 §3.4 maps the 16 input octets onto the state column by column: `in[r + 4c]` is
//! `s[r][c]`. This module keeps the block in exactly that order, so a column of the state is a
//! contiguous four-octet window `block[4c .. 4c + 4]`, and `ShiftRows` — which moves whole rows
//! — is the strided permutation `s'[r + 4c] = s[r + 4·((c + r) mod 4)]`.
//!
//! # Side channels
//!
//! `SubBytes` is a 256-entry table lookup indexed by a secret octet, and `MixColumns` uses
//! further 256-entry tables. On a machine with a data cache these lookups are *not*
//! constant-time; a local attacker who can observe cache state can mount the classical
//! Bernstein/Osvik–Shamir–Tromer timing attacks. A bitsliced or vector-permute implementation
//! would be required to close that channel and neither is expressible without `unsafe` or
//! target-specific intrinsics, both of which this workspace forbids. Everything that does *not*
//! need a table — key comparison, tag comparison, padding checks — is written branch-free and
//! data-independent; see [`crate::crypto::ct`].
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::aes::{Aes, BLOCK_SIZE};
//!
//! // FIPS 197 Appendix C.1: the AES-128 known-answer test.
//! let key: [u8; 16] = [
//!     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
//!     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
//! ];
//! let aes = Aes::new(&key)?;
//! let mut block: [u8; BLOCK_SIZE] = [
//!     0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
//!     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
//! ];
//! aes.encrypt_block(&mut block);
//! assert_eq!(
//!     block,
//!     [
//!         0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
//!         0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
//!     ],
//! );
//! aes.decrypt_block(&mut block);
//! assert_eq!(block[0], 0x00);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use core::hint::black_box;

use crate::error::{EncError, Result};

/// The AES block size in octets. AES is defined for a 128-bit block only.
pub const BLOCK_SIZE: usize = 16;

/// The largest number of round keys any AES variant needs (`Nr + 1` for AES-256).
const MAX_ROUND_KEYS: usize = 15;

/// The largest number of expanded key words any AES variant needs (`4 · (Nr + 1)`).
const MAX_WORDS: usize = 4 * MAX_ROUND_KEYS;

// ---------------------------------------------------------------------------------------------
// GF(2^8) arithmetic and the compile-time tables derived from it.
// ---------------------------------------------------------------------------------------------

/// Multiplies by `x` in `GF(2^8)` modulo `x^8 + x^4 + x^3 + x + 1` (FIPS 197 §4.2.1).
///
/// The reduction is written branch-free: `(byte >> 7) & 1` is the coefficient of `x^7`, and
/// multiplying it by `0x1b` conditionally folds in the low half of the modulus.
const fn xtime(byte: u8) -> u8 {
    (byte << 1) ^ (((byte >> 7) & 1) * 0x1b)
}

/// Multiplies two field elements (FIPS 197 §4.2) by the peasant/`xtime` algorithm.
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    let mut step = 0;
    while step < 8 {
        product ^= a * (b & 1);
        a = xtime(a);
        b >>= 1;
        step += 1;
    }
    product
}

/// Returns the multiplicative inverse of `byte` in `GF(2^8)`, with `0` mapped to `0` as FIPS 197
/// §5.1.1 specifies.
///
/// The field has `255` non-zero elements, so `b^254 = b^-1` for every `b != 0`, and `0^254 = 0`
/// falls out for free.
const fn ginv(byte: u8) -> u8 {
    let mut result = 1u8;
    let mut base = byte;
    let mut exponent = 254u32;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = gmul(result, base);
        }
        base = gmul(base, base);
        exponent >>= 1;
    }
    result
}

/// The affine transformation of FIPS 197 §5.1.1, written with rotations:
/// `b' = b ⊕ (b <<< 1) ⊕ (b <<< 2) ⊕ (b <<< 3) ⊕ (b <<< 4) ⊕ 0x63`.
const fn affine(byte: u8) -> u8 {
    byte ^ byte.rotate_left(1)
        ^ byte.rotate_left(2)
        ^ byte.rotate_left(3)
        ^ byte.rotate_left(4)
        ^ 0x63
}

/// Builds the substitution box of FIPS 197 §5.1.1 from the field definition.
const fn build_sbox() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0usize;
    while index < 256 {
        table[index] = affine(ginv(index as u8));
        index += 1;
    }
    table
}

/// Inverts a bijective 256-entry table.
const fn invert_table(table: &[u8; 256]) -> [u8; 256] {
    let mut inverse = [0u8; 256];
    let mut index = 0usize;
    while index < 256 {
        inverse[table[index] as usize] = index as u8;
        index += 1;
    }
    inverse
}

/// Builds the table of `factor · x` over every field element `x`.
const fn build_mul_table(factor: u8) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0usize;
    while index < 256 {
        table[index] = gmul(factor, index as u8);
        index += 1;
    }
    table
}

/// The AES substitution box (FIPS 197 Figure 7), derived rather than transcribed.
const SBOX: [u8; 256] = build_sbox();
/// The inverse substitution box (FIPS 197 Figure 14).
const INV_SBOX: [u8; 256] = invert_table(&SBOX);

/// `x ↦ 02 · x`, used by `MixColumns`.
const MUL2: [u8; 256] = build_mul_table(0x02);
/// `x ↦ 03 · x`, used by `MixColumns`.
const MUL3: [u8; 256] = build_mul_table(0x03);
/// `x ↦ 09 · x`, used by `InvMixColumns`.
const MUL9: [u8; 256] = build_mul_table(0x09);
/// `x ↦ 0b · x`, used by `InvMixColumns`.
const MUL11: [u8; 256] = build_mul_table(0x0b);
/// `x ↦ 0d · x`, used by `InvMixColumns`.
const MUL13: [u8; 256] = build_mul_table(0x0d);
/// `x ↦ 0e · x`, used by `InvMixColumns`.
const MUL14: [u8; 256] = build_mul_table(0x0e);

// ---------------------------------------------------------------------------------------------
// The cipher.
// ---------------------------------------------------------------------------------------------

/// An AES cipher with an expanded key schedule.
///
/// Construct one with [`Aes::new`] from a 16-, 24- or 32-octet key (AES-128, AES-192 or
/// AES-256) and then encrypt or decrypt individual blocks. Modes of operation live in the
/// sibling modules: [`cbc`](crate::crypto::cbc), [`ctr`](crate::crypto::ctr),
/// [`gcm`](crate::crypto::gcm) and [`keywrap`](crate::crypto::keywrap).
///
/// The round keys are overwritten with zeros when the value is dropped. That is a best-effort
/// hygiene measure, not a guarantee: nothing stops the optimizer from having kept a copy in a
/// register or a spill slot, and a value that was moved before being dropped leaves the old
/// bytes behind.
#[derive(Clone)]
pub struct Aes {
    /// `round_keys[r]` is the round key added at round `r`; entries past `rounds` are zero.
    round_keys: [[u8; BLOCK_SIZE]; MAX_ROUND_KEYS],
    /// `Nr`, the number of rounds: 10, 12 or 14.
    rounds: usize,
}

impl core::fmt::Debug for Aes {
    /// Never prints the key schedule.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Aes")
            .field("key_bits", &(((self.rounds - 6) * 4) * 8))
            .finish_non_exhaustive()
    }
}

impl Drop for Aes {
    fn drop(&mut self) {
        for key in &mut self.round_keys {
            *key = [0u8; BLOCK_SIZE];
        }
        // Keep the optimizer from deleting the writes above as dead stores.
        let _ = black_box(&self.round_keys);
    }
}

impl Aes {
    /// Expands `key` into a round-key schedule.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidKeyLength`] unless `key` is 16, 24 or 32 octets long.
    pub fn new(key: &[u8]) -> Result<Self> {
        let nk = match key.len() {
            16 => 4usize,
            24 => 6usize,
            32 => 8usize,
            length => {
                return Err(EncError::InvalidKeyLength {
                    algorithm: "AES",
                    length,
                });
            }
        };
        let rounds = nk + 6;
        let word_count = 4 * (rounds + 1);

        // FIPS 197 §5.2, KeyExpansion.
        let mut words = [[0u8; 4]; MAX_WORDS];
        for (index, word) in words.iter_mut().take(nk).enumerate() {
            word.copy_from_slice(&key[index * 4..index * 4 + 4]);
        }
        let mut rcon = 0x01u8;
        for index in nk..word_count {
            let mut temp = words[index - 1];
            if index % nk == 0 {
                // RotWord then SubWord, then exclusive-or with Rcon[i/Nk].
                temp = [
                    SBOX[temp[1] as usize] ^ rcon,
                    SBOX[temp[2] as usize],
                    SBOX[temp[3] as usize],
                    SBOX[temp[0] as usize],
                ];
                rcon = xtime(rcon);
            } else if nk > 6 && index % nk == 4 {
                // AES-256 only: an extra SubWord in the middle of each group.
                temp = [
                    SBOX[temp[0] as usize],
                    SBOX[temp[1] as usize],
                    SBOX[temp[2] as usize],
                    SBOX[temp[3] as usize],
                ];
            }
            let previous = words[index - nk];
            words[index] = [
                previous[0] ^ temp[0],
                previous[1] ^ temp[1],
                previous[2] ^ temp[2],
                previous[3] ^ temp[3],
            ];
        }

        let mut round_keys = [[0u8; BLOCK_SIZE]; MAX_ROUND_KEYS];
        for (round, round_key) in round_keys.iter_mut().take(rounds + 1).enumerate() {
            for column in 0..4 {
                let word = words[round * 4 + column];
                round_key[column * 4..column * 4 + 4].copy_from_slice(&word);
            }
        }

        Ok(Self { round_keys, rounds })
    }

    /// Returns the key length in octets that this instance was created with (16, 24 or 32).
    #[must_use]
    pub const fn key_len(&self) -> usize {
        (self.rounds - 6) * 4
    }

    /// Encrypts one block in place (FIPS 197 §5.1, `Cipher`).
    pub fn encrypt_block(&self, block: &mut [u8; BLOCK_SIZE]) {
        add_round_key(block, &self.round_keys[0]);
        for round_key in &self.round_keys[1..self.rounds] {
            sub_bytes(block);
            shift_rows(block);
            mix_columns(block);
            add_round_key(block, round_key);
        }
        sub_bytes(block);
        shift_rows(block);
        add_round_key(block, &self.round_keys[self.rounds]);
    }

    /// Decrypts one block in place (FIPS 197 §5.3, `InvCipher`).
    pub fn decrypt_block(&self, block: &mut [u8; BLOCK_SIZE]) {
        add_round_key(block, &self.round_keys[self.rounds]);
        for round in (1..self.rounds).rev() {
            inv_shift_rows(block);
            inv_sub_bytes(block);
            add_round_key(block, &self.round_keys[round]);
            inv_mix_columns(block);
        }
        inv_shift_rows(block);
        inv_sub_bytes(block);
        add_round_key(block, &self.round_keys[0]);
    }
}

/// `AddRoundKey` (FIPS 197 §5.1.4).
fn add_round_key(block: &mut [u8; BLOCK_SIZE], round_key: &[u8; BLOCK_SIZE]) {
    for (byte, key_byte) in block.iter_mut().zip(round_key.iter()) {
        *byte ^= *key_byte;
    }
}

/// `SubBytes` (FIPS 197 §5.1.1).
fn sub_bytes(block: &mut [u8; BLOCK_SIZE]) {
    for byte in block.iter_mut() {
        *byte = SBOX[*byte as usize];
    }
}

/// `InvSubBytes` (FIPS 197 §5.3.2).
fn inv_sub_bytes(block: &mut [u8; BLOCK_SIZE]) {
    for byte in block.iter_mut() {
        *byte = INV_SBOX[*byte as usize];
    }
}

/// `ShiftRows` (FIPS 197 §5.1.2): row `r` rotates left by `r` columns.
fn shift_rows(block: &mut [u8; BLOCK_SIZE]) {
    let source = *block;
    for row in 1..4usize {
        for column in 0..4usize {
            block[row + 4 * column] = source[row + 4 * ((column + row) % 4)];
        }
    }
}

/// `InvShiftRows` (FIPS 197 §5.3.1): row `r` rotates right by `r` columns.
fn inv_shift_rows(block: &mut [u8; BLOCK_SIZE]) {
    let source = *block;
    for row in 1..4usize {
        for column in 0..4usize {
            block[row + 4 * ((column + row) % 4)] = source[row + 4 * column];
        }
    }
}

/// `MixColumns` (FIPS 197 §5.1.3): each column is multiplied by the fixed polynomial
/// `{03}x^3 + {01}x^2 + {01}x + {02}`.
fn mix_columns(block: &mut [u8; BLOCK_SIZE]) {
    for column in block.chunks_exact_mut(4) {
        let (a0, a1, a2, a3) = (column[0], column[1], column[2], column[3]);
        column[0] = MUL2[a0 as usize] ^ MUL3[a1 as usize] ^ a2 ^ a3;
        column[1] = a0 ^ MUL2[a1 as usize] ^ MUL3[a2 as usize] ^ a3;
        column[2] = a0 ^ a1 ^ MUL2[a2 as usize] ^ MUL3[a3 as usize];
        column[3] = MUL3[a0 as usize] ^ a1 ^ a2 ^ MUL2[a3 as usize];
    }
}

/// `InvMixColumns` (FIPS 197 §5.3.3): each column is multiplied by
/// `{0b}x^3 + {0d}x^2 + {09}x + {0e}`.
fn inv_mix_columns(block: &mut [u8; BLOCK_SIZE]) {
    for column in block.chunks_exact_mut(4) {
        let (a0, a1, a2, a3) = (column[0], column[1], column[2], column[3]);
        column[0] =
            MUL14[a0 as usize] ^ MUL11[a1 as usize] ^ MUL13[a2 as usize] ^ MUL9[a3 as usize];
        column[1] =
            MUL9[a0 as usize] ^ MUL14[a1 as usize] ^ MUL11[a2 as usize] ^ MUL13[a3 as usize];
        column[2] =
            MUL13[a0 as usize] ^ MUL9[a1 as usize] ^ MUL14[a2 as usize] ^ MUL11[a3 as usize];
        column[3] =
            MUL11[a0 as usize] ^ MUL13[a1 as usize] ^ MUL9[a2 as usize] ^ MUL14[a3 as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox_matches_the_standard_at_the_documented_points() {
        // FIPS 197 Figure 7 spot checks, plus the two fixed points of the affine map.
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x01], 0x7c);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0xff], 0x16);
        assert_eq!(INV_SBOX[0x63], 0x00);
        assert_eq!(INV_SBOX[0xed], 0x53);
    }

    #[test]
    fn sbox_is_a_bijection() {
        let mut seen = [false; 256];
        for value in SBOX {
            assert!(!seen[value as usize], "S-box is not injective");
            seen[value as usize] = true;
        }
        for value in 0..=255u8 {
            assert_eq!(INV_SBOX[SBOX[value as usize] as usize], value);
        }
    }

    #[test]
    fn field_inverse_is_an_inverse() {
        for value in 1..=255u8 {
            assert_eq!(gmul(value, ginv(value)), 1, "inverse of {value:#04x}");
        }
        assert_eq!(ginv(0), 0);
    }

    #[test]
    fn mix_columns_round_trips() -> Result<()> {
        // FIPS 197 §5.1.3 worked example.
        let mut block = [0u8; BLOCK_SIZE];
        block[..4].copy_from_slice(&[0xdb, 0x13, 0x53, 0x45]);
        block[4..8].copy_from_slice(&[0xf2, 0x0a, 0x22, 0x5c]);
        mix_columns(&mut block);
        assert_eq!(&block[..4], &[0x8e, 0x4d, 0xa1, 0xbc]);
        assert_eq!(&block[4..8], &[0x9f, 0xdc, 0x58, 0x9d]);
        inv_mix_columns(&mut block);
        assert_eq!(&block[..4], &[0xdb, 0x13, 0x53, 0x45]);
        Ok(())
    }

    #[test]
    fn shift_rows_round_trips() {
        let mut block = [0u8; BLOCK_SIZE];
        for (index, byte) in block.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let original = block;
        shift_rows(&mut block);
        assert_ne!(block, original);
        inv_shift_rows(&mut block);
        assert_eq!(block, original);
    }

    #[test]
    fn rejects_a_bad_key_length() {
        assert!(matches!(
            Aes::new(&[0u8; 20]),
            Err(EncError::InvalidKeyLength {
                algorithm: "AES",
                length: 20
            })
        ));
    }

    #[test]
    fn key_len_is_reported() -> Result<()> {
        assert_eq!(Aes::new(&[0u8; 16])?.key_len(), 16);
        assert_eq!(Aes::new(&[0u8; 24])?.key_len(), 24);
        assert_eq!(Aes::new(&[0u8; 32])?.key_len(), 32);
        Ok(())
    }

    #[test]
    fn debug_does_not_leak_the_schedule() -> Result<()> {
        let aes = Aes::new(&[0xabu8; 16])?;
        let rendered = format!("{aes:?}");
        assert!(rendered.contains("128"), "{rendered}");
        assert!(!rendered.contains("ab"), "{rendered}");
        Ok(())
    }
}
