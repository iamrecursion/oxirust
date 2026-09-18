//! Where initialization vectors, OAEP seeds and PKCS#1 padding strings come from.
//!
//! [XML Encryption 1.1 §6.4](https://www.w3.org/TR/xmlenc-core1/#sec-Nonce) requires a
//! *different* initialization vector for every encryption under a key, and for AES-GCM the
//! consequence of getting that wrong is not merely a loss of confidentiality but a recovery of
//! the authentication subkey, which forges arbitrary messages. Every operation in this crate
//! that needs randomness therefore takes an explicit `&mut dyn RandomSource` rather than
//! reaching for a hidden global: the caller decides, and the decision is visible at the call
//! site.
//!
//! Three implementations are provided:
//!
//! * [`SystemRandom`] — the operating system's entropy, for real use.
//! * [`PseudoRandom`] — a deterministic PRNG, for reproducible tests. **Never** for real keys.
//! * [`FixedRandom`] — a fixed, cycling octet string, for known-answer tests where the
//!   initialization vector must be pinned. **Never** for real keys.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::random::{FixedRandom, RandomSource, SystemRandom};
//!
//! let mut system = SystemRandom::new();
//! let mut iv = [0u8; 12];
//! system.fill(&mut iv)?;
//! assert!(iv.iter().any(|byte| *byte != 0), "vanishingly unlikely to be all zeros");
//!
//! // A deterministic source reproduces a known ciphertext exactly.
//! let mut fixed = FixedRandom::new(&[0xde, 0xad, 0xbe, 0xef]);
//! let mut buffer = [0u8; 6];
//! fixed.fill(&mut buffer)?;
//! assert_eq!(buffer, [0xde, 0xad, 0xbe, 0xef, 0xde, 0xad]);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use oxixml_support::hash::sha512;
use oxixml_support::rng::{Pcg32, Rng};

use crate::crypto::aes::{Aes, BLOCK_SIZE};
use crate::crypto::ctr;
use crate::error::Result;

/// A source of octets for initialization vectors, OAEP seeds and padding.
pub trait RandomSource {
    /// Fills `dest` completely.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Entropy`](crate::EncError::Entropy) when the underlying source
    /// cannot supply the octets. Deterministic implementations never fail.
    fn fill(&mut self, dest: &mut [u8]) -> Result<()>;
}

impl<T: RandomSource + ?Sized> RandomSource for &mut T {
    fn fill(&mut self, dest: &mut [u8]) -> Result<()> {
        (**self).fill(dest)
    }
}

/// The path this crate reads operating-system entropy from on Unix-like targets.
const URANDOM_PATH: &str = "/dev/urandom";

/// The operating system's entropy.
///
/// # How it works, and what it does not do
///
/// On any target where `/dev/urandom` can be opened — Linux, the BSDs, macOS, illumos — the
/// octets come straight from it, freshly on every call. That is the whole implementation, and
/// it is the right one: `/dev/urandom` is the kernel CSPRNG.
///
/// On a target where that file does not exist (notably Windows) there is no way to reach
/// `BCryptGenRandom` from pure Rust with no dependencies and no `unsafe`, so this type falls
/// back to seeding an **AES-256-CTR generator** from every source of unpredictability the
/// standard library exposes: [`RandomState`], which the standard library itself seeds from the
/// operating system's CSPRNG; the wall clock and a monotonic clock read; the process
/// identifier; and heap and stack addresses, which address-space layout randomization makes
/// unpredictable. Those are hashed with SHA-512 into a key and a counter, and the generator is
/// re-seeded on every call so that a single weak component cannot pin the output.
///
/// The fallback is a genuine best effort, not a promise. If you deploy on such a target and
/// need a defensible entropy story, supply your own [`RandomSource`] backed by the platform
/// CSPRNG — the whole API takes one explicitly for exactly that reason.
#[derive(Debug, Default)]
pub struct SystemRandom {
    /// A counter folded into the fallback seed so that two calls in the same nanosecond, from
    /// the same thread, still differ.
    sequence: u64,
}

impl SystemRandom {
    /// Creates a handle to the operating system's entropy.
    #[must_use]
    pub const fn new() -> Self {
        Self { sequence: 0 }
    }

    /// Tries to read `dest` from `/dev/urandom`, returning `false` when it is unavailable.
    fn try_urandom(dest: &mut [u8]) -> bool {
        let Ok(mut file) = std::fs::File::open(URANDOM_PATH) else {
            return false;
        };
        file.read_exact(dest).is_ok()
    }

    /// Gathers a 64-octet seed from everything the standard library makes unpredictable.
    fn gather_seed(&mut self) -> [u8; 64] {
        self.sequence = self.sequence.wrapping_add(1);

        let mut material: Vec<u8> = Vec::with_capacity(128);
        material.extend_from_slice(&self.sequence.to_le_bytes());

        // `RandomState` is seeded by the standard library from the operating system's CSPRNG.
        // Hashing distinct inputs with two independent instances extracts that seed material.
        for salt in [0u64, 0x9e37_79b9_7f4a_7c15] {
            let state = RandomState::new();
            let mut hasher = state.build_hasher();
            hasher.write_u64(salt);
            hasher.write_u64(self.sequence);
            material.extend_from_slice(&hasher.finish().to_le_bytes());
        }

        // Wall-clock and monotonic-clock reads.
        if let Ok(since_epoch) = SystemTime::now().duration_since(UNIX_EPOCH) {
            material.extend_from_slice(&since_epoch.as_nanos().to_le_bytes());
        }
        let start = Instant::now();
        material.extend_from_slice(&start.elapsed().as_nanos().to_le_bytes());

        material.extend_from_slice(&std::process::id().to_le_bytes());
        material.extend_from_slice(format!("{:?}", std::thread::current().id()).as_bytes());

        // Heap and stack addresses, which ASLR randomizes.
        let heap = Box::new(0u8);
        let heap_address = std::ptr::from_ref::<u8>(heap.as_ref()) as usize;
        material.extend_from_slice(&heap_address.to_le_bytes());
        let stack_address = std::ptr::from_ref(&material) as usize;
        material.extend_from_slice(&stack_address.to_le_bytes());

        sha512(&material)
    }
}

impl RandomSource for SystemRandom {
    fn fill(&mut self, dest: &mut [u8]) -> Result<()> {
        if dest.is_empty() {
            return Ok(());
        }
        if Self::try_urandom(dest) {
            return Ok(());
        }

        // Fallback: AES-256-CTR keyed from the gathered seed.
        let seed = self.gather_seed();
        let aes = Aes::new(&seed[..32])?;
        let mut counter = [0u8; BLOCK_SIZE];
        counter.copy_from_slice(&seed[32..48]);
        dest.fill(0);
        ctr::apply(&aes, &counter, dest, ctr::increment_full);
        Ok(())
    }
}

/// A deterministic pseudo-random source for reproducible tests.
///
/// Wraps the PCG generator of [`oxixml_support::rng`], which is explicitly documented as
/// **non-cryptographic**: its state is recoverable from a handful of outputs. Using it for a
/// real initialization vector would let anyone who sees one ciphertext predict every later one.
/// It exists so that a test can assert on an exact ciphertext.
#[derive(Debug, Clone)]
pub struct PseudoRandom {
    inner: Pcg32,
}

impl PseudoRandom {
    /// Creates a generator from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Pcg32::new(seed),
        }
    }
}

impl RandomSource for PseudoRandom {
    fn fill(&mut self, dest: &mut [u8]) -> Result<()> {
        self.inner.fill_bytes(dest);
        Ok(())
    }
}

/// A source that repeats a fixed octet string forever.
///
/// This is how a known-answer test pins an initialization vector or an OAEP seed to the value a
/// published vector used. It is not random in any sense.
#[derive(Debug, Clone)]
pub struct FixedRandom {
    bytes: Vec<u8>,
    cursor: usize,
}

impl FixedRandom {
    /// Creates a source that cycles through `bytes`.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` is empty, because there would be nothing to cycle through and every
    /// later call would have to fail or hang. This is a programming error in a test, not a
    /// runtime condition, so it is reported as a panic rather than an error.
    #[must_use]
    pub fn new(bytes: &[u8]) -> Self {
        assert!(!bytes.is_empty(), "FixedRandom needs at least one octet");
        Self {
            bytes: bytes.to_vec(),
            cursor: 0,
        }
    }
}

impl RandomSource for FixedRandom {
    fn fill(&mut self, dest: &mut [u8]) -> Result<()> {
        for byte in dest.iter_mut() {
            *byte = self.bytes[self.cursor];
            self.cursor = (self.cursor + 1) % self.bytes.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_source_produces_different_octets_each_time() -> Result<()> {
        let mut source = SystemRandom::new();
        let mut first = [0u8; 32];
        let mut second = [0u8; 32];
        source.fill(&mut first)?;
        source.fill(&mut second)?;
        assert_ne!(first, second);
        assert_ne!(first, [0u8; 32]);
        Ok(())
    }

    #[test]
    fn an_empty_request_succeeds() -> Result<()> {
        SystemRandom::new().fill(&mut [])?;
        Ok(())
    }

    #[test]
    fn the_fallback_generator_is_usable_on_its_own() -> Result<()> {
        // Exercise the fallback path directly, since `/dev/urandom` exists on the test host.
        let mut source = SystemRandom::new();
        let first = source.gather_seed();
        let second = source.gather_seed();
        assert_ne!(first, second);

        let aes = Aes::new(&first[..32])?;
        let mut counter = [0u8; BLOCK_SIZE];
        counter.copy_from_slice(&first[32..48]);
        let mut output = [0u8; 64];
        ctr::apply(&aes, &counter, &mut output, ctr::increment_full);
        assert_ne!(output, [0u8; 64]);
        Ok(())
    }

    #[test]
    fn the_fixed_source_cycles() -> Result<()> {
        let mut source = FixedRandom::new(&[1, 2, 3]);
        let mut buffer = [0u8; 7];
        source.fill(&mut buffer)?;
        assert_eq!(buffer, [1, 2, 3, 1, 2, 3, 1]);
        // The cursor persists across calls.
        let mut next = [0u8; 2];
        source.fill(&mut next)?;
        assert_eq!(next, [2, 3]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "at least one octet")]
    fn an_empty_fixed_source_is_a_programming_error() {
        let _ = FixedRandom::new(&[]);
    }

    #[test]
    fn the_pseudo_source_is_reproducible() -> Result<()> {
        let mut first = PseudoRandom::new(42);
        let mut second = PseudoRandom::new(42);
        let mut third = PseudoRandom::new(43);
        let (mut a, mut b, mut c) = ([0u8; 24], [0u8; 24], [0u8; 24]);
        first.fill(&mut a)?;
        second.fill(&mut b)?;
        third.fill(&mut c)?;
        assert_eq!(a, b);
        assert_ne!(a, c);
        Ok(())
    }

    #[test]
    fn a_mutable_reference_is_itself_a_source() -> Result<()> {
        fn take(source: &mut dyn RandomSource) -> Result<u8> {
            let mut byte = [0u8; 1];
            source.fill(&mut byte)?;
            Ok(byte[0])
        }
        let mut source = FixedRandom::new(&[9]);
        let mut borrowed = &mut source;
        assert_eq!(take(&mut borrowed)?, 9);
        Ok(())
    }
}
