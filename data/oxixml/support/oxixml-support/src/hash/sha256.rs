//! SHA-224 and SHA-256 (FIPS 180-4, sections 6.2 / 5.3.2–5.3.3).
//!
//! Both share the 32-bit compression function and differ only in the initial hash value and the
//! number of output words retained.

use super::{Digest, LengthError};

/// Block size in bytes (shared by SHA-224 and SHA-256).
const BLOCK_SIZE: usize = 64;

/// SHA-256 initial hash value `H(0)` (FIPS 180-4 §5.3.3).
const INITIAL_STATE_256: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// SHA-224 initial hash value `H(0)` (FIPS 180-4 §5.3.2).
const INITIAL_STATE_224: [u32; 8] = [
    0xc105_9ed8,
    0x367c_d507,
    0x3070_dd17,
    0xf70e_5939,
    0xffc0_0b31,
    0x6858_1511,
    0x64f9_8fa7,
    0xbefa_4fa4,
];

/// Round constants `K` (FIPS 180-4 §4.2.2): the first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes.
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Shared SHA-224/256 streaming state.
#[derive(Clone)]
struct Core {
    state: [u32; 8],
    buffer: [u8; BLOCK_SIZE],
    buffer_len: usize,
    total_len: u64,
}

impl Core {
    #[inline]
    fn new(initial_state: [u32; 8]) -> Self {
        Self {
            state: initial_state,
            buffer: [0u8; BLOCK_SIZE],
            buffer_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
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

    fn finalize_state(mut self) -> [u32; 8] {
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
        self.state
    }
}

/// Serialize the eight state words as big-endian bytes into `out` (which selects the length).
fn state_to_bytes(state: &[u32; 8], out: &mut [u8]) {
    for (chunk, word) in out.chunks_exact_mut(4).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
}

/// The SHA-256 hash function.
///
/// Create with [`Sha256::new`], absorb input with [`Sha256::update`], and obtain the 32-byte
/// digest with [`Sha256::finalize`].
#[derive(Clone)]
pub struct Sha256 {
    core: Core,
}

impl Sha256 {
    /// Create a new SHA-256 hasher in its initial state.
    #[inline]
    pub fn new() -> Self {
        Self {
            core: Core::new(INITIAL_STATE_256),
        }
    }

    /// Absorb `data` into the running digest. May be called any number of times.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.core.update(data);
    }

    /// Finish hashing and return the 32-byte digest.
    #[inline]
    pub fn finalize(self) -> [u8; 32] {
        let state = self.core.finalize_state();
        let mut out = [0u8; 32];
        state_to_bytes(&state, &mut out);
        out
    }
}

impl Default for Sha256 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Digest for Sha256 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;
    const OUTPUT_SIZE: usize = 32;

    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < 32 {
            return Err(LengthError {
                expected: 32,
                actual: out.len(),
            });
        }
        out[..32].copy_from_slice(&self.finalize());
        Ok(())
    }
}

/// The SHA-224 hash function.
///
/// SHA-224 is SHA-256 with a different initial value, truncated to the leftmost 224 bits.
#[derive(Clone)]
pub struct Sha224 {
    core: Core,
}

impl Sha224 {
    /// Create a new SHA-224 hasher in its initial state.
    #[inline]
    pub fn new() -> Self {
        Self {
            core: Core::new(INITIAL_STATE_224),
        }
    }

    /// Absorb `data` into the running digest. May be called any number of times.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.core.update(data);
    }

    /// Finish hashing and return the 28-byte digest.
    #[inline]
    pub fn finalize(self) -> [u8; 28] {
        let state = self.core.finalize_state();
        let mut full = [0u8; 32];
        state_to_bytes(&state, &mut full);
        let mut out = [0u8; 28];
        out.copy_from_slice(&full[..28]);
        out
    }
}

impl Default for Sha224 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Digest for Sha224 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;
    const OUTPUT_SIZE: usize = 28;

    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < 28 {
            return Err(LengthError {
                expected: 28,
                actual: out.len(),
            });
        }
        out[..28].copy_from_slice(&self.finalize());
        Ok(())
    }
}

/// Process one 64-byte block, updating the eight state words in place.
#[allow(clippy::many_single_char_names)] // matches the FIPS 180-4 working-variable names a..h.
fn compress(state: &mut [u32; 8], block: &[u8; BLOCK_SIZE]) {
    let mut w = [0u32; 64];
    for (word, chunk) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (&ki, &wi) in K.iter().zip(w.iter()) {
        let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(ki)
            .wrapping_add(wi);
        let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = big_s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn digest256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex(&hasher.finalize())
    }

    fn digest224(data: &[u8]) -> String {
        let mut hasher = Sha224::new();
        hasher.update(data);
        hex(&hasher.finalize())
    }

    #[test]
    fn sha256_empty() {
        assert_eq!(
            digest256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        assert_eq!(
            digest256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_two_block() {
        assert_eq!(
            digest256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_million_a() {
        let mut hasher = Sha256::new();
        let chunk = vec![b'a'; 1_000];
        for _ in 0..1_000 {
            hasher.update(&chunk);
        }
        assert_eq!(
            hex(&hasher.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha224_abc() {
        assert_eq!(
            digest224(b"abc"),
            "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"
        );
    }

    #[test]
    fn sha224_empty() {
        assert_eq!(
            digest224(b""),
            "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
        );
    }

    #[test]
    fn sha224_two_block() {
        assert_eq!(
            digest224(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525"
        );
    }

    #[test]
    fn sha256_digest_into_reports_undersized_buffer_instead_of_panicking() {
        let mut hasher = Sha256::new();
        hasher.update(b"abc");
        let mut short = [0u8; 31]; // OUTPUT_SIZE is 32.
        let err = hasher
            .digest_into(&mut short)
            .expect_err("31 bytes is one short of Sha256::OUTPUT_SIZE");
        assert_eq!(err.expected, 32);
        assert_eq!(err.actual, 31);
    }

    #[test]
    fn sha224_digest_into_reports_undersized_buffer_instead_of_panicking() {
        let mut hasher = Sha224::new();
        hasher.update(b"abc");
        let mut short = [0u8; 27]; // OUTPUT_SIZE is 28.
        let err = hasher
            .digest_into(&mut short)
            .expect_err("27 bytes is one short of Sha224::OUTPUT_SIZE");
        assert_eq!(err.expected, 28);
        assert_eq!(err.actual, 27);
    }
}
