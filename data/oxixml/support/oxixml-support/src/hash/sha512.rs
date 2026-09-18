//! SHA-384 and SHA-512 (FIPS 180-4, sections 6.4 / 5.3.4–5.3.5).
//!
//! Both share the 64-bit compression function over 1024-bit blocks, differing in the initial
//! hash value and the number of output words retained.

use super::{Digest, LengthError};

/// Block size in bytes (shared by SHA-384 and SHA-512).
const BLOCK_SIZE: usize = 128;
/// Number of trailing bytes reserved for the 128-bit message-length field.
const LENGTH_BYTES: usize = 16;

/// SHA-512 initial hash value `H(0)` (FIPS 180-4 §5.3.5).
const INITIAL_STATE_512: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

/// SHA-384 initial hash value `H(0)` (FIPS 180-4 §5.3.4).
const INITIAL_STATE_384: [u64; 8] = [
    0xcbbb_9d5d_c105_9ed8,
    0x629a_292a_367c_d507,
    0x9159_015a_3070_dd17,
    0x152f_ecd8_f70e_5939,
    0x6733_2667_ffc0_0b31,
    0x8eb4_4a87_6858_1511,
    0xdb0c_2e0d_64f9_8fa7,
    0x47b5_481d_befa_4fa4,
];

/// Round constants `K` (FIPS 180-4 §4.2.3): the first 64 bits of the fractional parts of the
/// cube roots of the first 80 primes.
const K: [u64; 80] = [
    0x428a_2f98_d728_ae22,
    0x7137_4491_23ef_65cd,
    0xb5c0_fbcf_ec4d_3b2f,
    0xe9b5_dba5_8189_dbbc,
    0x3956_c25b_f348_b538,
    0x59f1_11f1_b605_d019,
    0x923f_82a4_af19_4f9b,
    0xab1c_5ed5_da6d_8118,
    0xd807_aa98_a303_0242,
    0x1283_5b01_4570_6fbe,
    0x2431_85be_4ee4_b28c,
    0x550c_7dc3_d5ff_b4e2,
    0x72be_5d74_f27b_896f,
    0x80de_b1fe_3b16_96b1,
    0x9bdc_06a7_25c7_1235,
    0xc19b_f174_cf69_2694,
    0xe49b_69c1_9ef1_4ad2,
    0xefbe_4786_384f_25e3,
    0x0fc1_9dc6_8b8c_d5b5,
    0x240c_a1cc_77ac_9c65,
    0x2de9_2c6f_592b_0275,
    0x4a74_84aa_6ea6_e483,
    0x5cb0_a9dc_bd41_fbd4,
    0x76f9_88da_8311_53b5,
    0x983e_5152_ee66_dfab,
    0xa831_c66d_2db4_3210,
    0xb003_27c8_98fb_213f,
    0xbf59_7fc7_beef_0ee4,
    0xc6e0_0bf3_3da8_8fc2,
    0xd5a7_9147_930a_a725,
    0x06ca_6351_e003_826f,
    0x1429_2967_0a0e_6e70,
    0x27b7_0a85_46d2_2ffc,
    0x2e1b_2138_5c26_c926,
    0x4d2c_6dfc_5ac4_2aed,
    0x5338_0d13_9d95_b3df,
    0x650a_7354_8baf_63de,
    0x766a_0abb_3c77_b2a8,
    0x81c2_c92e_47ed_aee6,
    0x9272_2c85_1482_353b,
    0xa2bf_e8a1_4cf1_0364,
    0xa81a_664b_bc42_3001,
    0xc24b_8b70_d0f8_9791,
    0xc76c_51a3_0654_be30,
    0xd192_e819_d6ef_5218,
    0xd699_0624_5565_a910,
    0xf40e_3585_5771_202a,
    0x106a_a070_32bb_d1b8,
    0x19a4_c116_b8d2_d0c8,
    0x1e37_6c08_5141_ab53,
    0x2748_774c_df8e_eb99,
    0x34b0_bcb5_e19b_48a8,
    0x391c_0cb3_c5c9_5a63,
    0x4ed8_aa4a_e341_8acb,
    0x5b9c_ca4f_7763_e373,
    0x682e_6ff3_d6b2_b8a3,
    0x748f_82ee_5def_b2fc,
    0x78a5_636f_4317_2f60,
    0x84c8_7814_a1f0_ab72,
    0x8cc7_0208_1a64_39ec,
    0x90be_fffa_2363_1e28,
    0xa450_6ceb_de82_bde9,
    0xbef9_a3f7_b2c6_7915,
    0xc671_78f2_e372_532b,
    0xca27_3ece_ea26_619c,
    0xd186_b8c7_21c0_c207,
    0xeada_7dd6_cde0_eb1e,
    0xf57d_4f7f_ee6e_d178,
    0x06f0_67aa_7217_6fba,
    0x0a63_7dc5_a2c8_98a6,
    0x113f_9804_bef9_0dae,
    0x1b71_0b35_131c_471b,
    0x28db_77f5_2304_7d84,
    0x32ca_ab7b_40c7_2493,
    0x3c9e_be0a_15c9_bebc,
    0x431d_67c4_9c10_0d4c,
    0x4cc5_d4be_cb3e_42b6,
    0x597f_299c_fc65_7e2a,
    0x5fcb_6fab_3ad6_faec,
    0x6c44_198c_4a47_5817,
];

/// Shared SHA-384/512 streaming state.
#[derive(Clone)]
struct Core {
    state: [u64; 8],
    buffer: [u8; BLOCK_SIZE],
    buffer_len: usize,
    total_len: u128,
}

impl Core {
    #[inline]
    fn new(initial_state: [u64; 8]) -> Self {
        Self {
            state: initial_state,
            buffer: [0u8; BLOCK_SIZE],
            buffer_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u128);
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

    fn finalize_state(mut self) -> [u64; 8] {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut block = self.buffer;
        let mut len = self.buffer_len;
        block[len] = 0x80;
        len += 1;
        if len > BLOCK_SIZE - LENGTH_BYTES {
            for byte in block.iter_mut().skip(len) {
                *byte = 0;
            }
            compress(&mut self.state, &block);
            block = [0u8; BLOCK_SIZE];
            len = 0;
        }
        for byte in block.iter_mut().take(BLOCK_SIZE - LENGTH_BYTES).skip(len) {
            *byte = 0;
        }
        block[BLOCK_SIZE - LENGTH_BYTES..].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut self.state, &block);
        self.state
    }
}

/// Serialize the eight 64-bit state words as big-endian bytes into `out`.
fn state_to_bytes(state: &[u64; 8], out: &mut [u8]) {
    for (chunk, word) in out.chunks_exact_mut(8).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
}

/// The SHA-512 hash function.
#[derive(Clone)]
pub struct Sha512 {
    core: Core,
}

impl Sha512 {
    /// Create a new SHA-512 hasher in its initial state.
    #[inline]
    pub fn new() -> Self {
        Self {
            core: Core::new(INITIAL_STATE_512),
        }
    }

    /// Absorb `data` into the running digest. May be called any number of times.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.core.update(data);
    }

    /// Finish hashing and return the 64-byte digest.
    #[inline]
    pub fn finalize(self) -> [u8; 64] {
        let state = self.core.finalize_state();
        let mut out = [0u8; 64];
        state_to_bytes(&state, &mut out);
        out
    }
}

impl Default for Sha512 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Digest for Sha512 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;
    const OUTPUT_SIZE: usize = 64;

    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < 64 {
            return Err(LengthError {
                expected: 64,
                actual: out.len(),
            });
        }
        out[..64].copy_from_slice(&self.finalize());
        Ok(())
    }
}

/// The SHA-384 hash function.
///
/// SHA-384 is SHA-512 with a different initial value, truncated to the leftmost 384 bits.
#[derive(Clone)]
pub struct Sha384 {
    core: Core,
}

impl Sha384 {
    /// Create a new SHA-384 hasher in its initial state.
    #[inline]
    pub fn new() -> Self {
        Self {
            core: Core::new(INITIAL_STATE_384),
        }
    }

    /// Absorb `data` into the running digest. May be called any number of times.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.core.update(data);
    }

    /// Finish hashing and return the 48-byte digest.
    #[inline]
    pub fn finalize(self) -> [u8; 48] {
        let state = self.core.finalize_state();
        let mut full = [0u8; 64];
        state_to_bytes(&state, &mut full);
        let mut out = [0u8; 48];
        out.copy_from_slice(&full[..48]);
        out
    }
}

impl Default for Sha384 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Digest for Sha384 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;
    const OUTPUT_SIZE: usize = 48;

    #[inline]
    fn absorb(&mut self, data: &[u8]) {
        self.update(data);
    }

    #[inline]
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < 48 {
            return Err(LengthError {
                expected: 48,
                actual: out.len(),
            });
        }
        out[..48].copy_from_slice(&self.finalize());
        Ok(())
    }
}

/// Process one 128-byte block, updating the eight 64-bit state words in place.
#[allow(clippy::many_single_char_names)] // matches the FIPS 180-4 working-variable names a..h.
fn compress(state: &mut [u64; 8], block: &[u8; BLOCK_SIZE]) {
    let mut w = [0u64; 80];
    for (word, chunk) in w.iter_mut().zip(block.chunks_exact(8)) {
        *word = u64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
    }
    for i in 16..80 {
        let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
        let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (&ki, &wi) in K.iter().zip(w.iter()) {
        let big_s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(ki)
            .wrapping_add(wi);
        let big_s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
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

    fn digest512(data: &[u8]) -> String {
        let mut hasher = Sha512::new();
        hasher.update(data);
        hex(&hasher.finalize())
    }

    fn digest384(data: &[u8]) -> String {
        let mut hasher = Sha384::new();
        hasher.update(data);
        hex(&hasher.finalize())
    }

    #[test]
    fn sha512_empty() {
        assert_eq!(
            digest512(b""),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
    }

    #[test]
    fn sha512_abc() {
        assert_eq!(
            digest512(b"abc"),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn sha512_two_block() {
        assert_eq!(
            digest512(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn\
                  hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018\
             501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
        );
    }

    #[test]
    fn sha512_million_a() {
        let mut hasher = Sha512::new();
        let chunk = vec![b'a'; 1_000];
        for _ in 0..1_000 {
            hasher.update(&chunk);
        }
        assert_eq!(
            hex(&hasher.finalize()),
            "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb\
             de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"
        );
    }

    #[test]
    fn sha384_abc() {
        assert_eq!(
            digest384(b"abc"),
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed\
             8086072ba1e7cc2358baeca134c825a7"
        );
    }

    #[test]
    fn sha384_empty() {
        assert_eq!(
            digest384(b""),
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da\
             274edebfe76f65fbd51ad2f14898b95b"
        );
    }

    #[test]
    fn sha384_two_block() {
        assert_eq!(
            digest384(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn\
                  hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712\
             fcc7c71a557e2db966c3e9fa91746039"
        );
    }

    #[test]
    fn sha512_digest_into_reports_undersized_buffer_instead_of_panicking() {
        let mut hasher = Sha512::new();
        hasher.update(b"abc");
        let mut short = [0u8; 63]; // OUTPUT_SIZE is 64.
        let err = hasher
            .digest_into(&mut short)
            .expect_err("63 bytes is one short of Sha512::OUTPUT_SIZE");
        assert_eq!(err.expected, 64);
        assert_eq!(err.actual, 63);
    }

    #[test]
    fn sha384_digest_into_reports_undersized_buffer_instead_of_panicking() {
        let mut hasher = Sha384::new();
        hasher.update(b"abc");
        let mut short = [0u8; 47]; // OUTPUT_SIZE is 48.
        let err = hasher
            .digest_into(&mut short)
            .expect_err("47 bytes is one short of Sha384::OUTPUT_SIZE");
        assert_eq!(err.expected, 48);
        assert_eq!(err.actual, 47);
    }
}
