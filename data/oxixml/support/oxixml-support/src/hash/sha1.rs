//! SHA-1 (FIPS 180-4, section 6.1).

use super::{Digest, LengthError};

/// Block size in bytes.
const BLOCK_SIZE: usize = 64;
/// Digest size in bytes.
const OUTPUT_SIZE: usize = 20;

/// Initial hash value `H(0)` (FIPS 180-4 §5.3.1).
const INITIAL_STATE: [u32; 5] = [
    0x6745_2301,
    0xEFCD_AB89,
    0x98BA_DCFE,
    0x1032_5476,
    0xC3D2_E1F0,
];

/// The SHA-1 hash function.
///
/// Streaming: create with [`Sha1::new`], feed input with [`Sha1::update`], then obtain the
/// 20-byte digest with [`Sha1::finalize`].
///
/// SHA-1 is broken for collision resistance; it is provided only for verifying legacy data.
#[derive(Clone)]
pub struct Sha1 {
    state: [u32; 5],
    buffer: [u8; BLOCK_SIZE],
    buffer_len: usize,
    total_len: u64,
}

impl Sha1 {
    /// Create a new SHA-1 hasher in its initial state.
    #[inline]
    pub fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0u8; BLOCK_SIZE],
            buffer_len: 0,
            total_len: 0,
        }
    }

    /// Absorb `data` into the running digest. May be called any number of times.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buffer_len > 0 {
            let needed = BLOCK_SIZE - self.buffer_len;
            let take = needed.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == BLOCK_SIZE {
                let block = self.buffer;
                compress(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }
        while data.len() >= BLOCK_SIZE {
            let mut block = [0u8; BLOCK_SIZE];
            block.copy_from_slice(&data[..BLOCK_SIZE]);
            compress(&mut self.state, &block);
            data = &data[BLOCK_SIZE..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    /// Finish hashing and return the 20-byte digest.
    pub fn finalize(mut self) -> [u8; OUTPUT_SIZE] {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut block = self.buffer;
        let mut len = self.buffer_len;
        block[len] = 0x80;
        len += 1;
        if len > BLOCK_SIZE - 8 {
            for byte in block.iter_mut().skip(len) {
                *byte = 0;
            }
            compress(&mut self.state, &block);
            block = [0u8; BLOCK_SIZE];
            len = 0;
        }
        for byte in block.iter_mut().take(BLOCK_SIZE - 8).skip(len) {
            *byte = 0;
        }
        block[BLOCK_SIZE - 8..].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut self.state, &block);

        let mut out = [0u8; OUTPUT_SIZE];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state.iter()) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

impl Default for Sha1 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Digest for Sha1 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;
    const OUTPUT_SIZE: usize = OUTPUT_SIZE;

    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < OUTPUT_SIZE {
            return Err(LengthError {
                expected: OUTPUT_SIZE,
                actual: out.len(),
            });
        }
        out[..OUTPUT_SIZE].copy_from_slice(&self.finalize());
        Ok(())
    }
}

/// Process one 64-byte block, updating the five state words in place.
#[allow(clippy::many_single_char_names)] // matches the FIPS 180-4 working-variable names a..e.
fn compress(state: &mut [u32; 5], block: &[u8; BLOCK_SIZE]) {
    let mut w = [0u32; 80];
    for (word, chunk) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..80 {
        w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
    }

    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for (i, &wi) in w.iter().enumerate() {
        let (f, k) = match i {
            0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
            20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
            40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
            _ => (b ^ c ^ d, 0xCA62_C1D6),
        };
        let temp = a
            .rotate_left(5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(wi);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = temp;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn digest(data: &[u8]) -> String {
        let mut hasher = Sha1::new();
        hasher.update(data);
        hex(&hasher.finalize())
    }

    #[test]
    fn empty_string() {
        assert_eq!(digest(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn abc() {
        assert_eq!(digest(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn two_block_message() {
        assert_eq!(
            digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn million_a() {
        let mut hasher = Sha1::new();
        // Feed in awkward chunk sizes to exercise the buffering path.
        let chunk = vec![b'a'; 999];
        let mut remaining = 1_000_000usize;
        while remaining > 0 {
            let take = chunk.len().min(remaining);
            hasher.update(&chunk[..take]);
            remaining -= take;
        }
        assert_eq!(
            hex(&hasher.finalize()),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }

    #[test]
    fn one_shot_matches() {
        assert_eq!(hex(&crate::hash::sha1(b"abc")), digest(b"abc"));
    }

    #[test]
    fn digest_into_reports_undersized_buffer_instead_of_panicking() {
        let mut hasher = Sha1::new();
        hasher.update(b"abc");
        let mut short = [0u8; 19]; // OUTPUT_SIZE is 20.
        let err = hasher
            .digest_into(&mut short)
            .expect_err("19 bytes is one short of OUTPUT_SIZE");
        assert_eq!(err.expected, 20);
        assert_eq!(err.actual, 19);
    }
}
