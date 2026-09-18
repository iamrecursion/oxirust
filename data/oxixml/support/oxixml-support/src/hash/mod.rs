//! Streaming SHA-1 / SHA-2 digests and HMAC, implemented directly from the specifications.
//!
//! The digest algorithms follow [FIPS 180-4] (Secure Hash Standard); [`Hmac`] follows
//! [RFC 2104] / [FIPS 198-1]. Every algorithm is validated in the crate's test suite against
//! official vectors — NIST SHA examples (`""`, `"abc"`, the two-block message, and one million
//! `'a'`s), [RFC 4231] for HMAC-SHA-256/384/512, and [RFC 2202] for HMAC-SHA-1.
//!
//! Each hash type follows the same streaming shape:
//!
//! * [`Sha256::new`] (and the [`Default`] impl) create a fresh state,
//! * [`Sha256::update`] absorbs a chunk of input (call it as many times as you like), and
//! * [`Sha256::finalize`] consumes the state and returns a fixed-size digest array.
//!
//! For convenience, one-shot free functions ([`sha256`], [`hmac_sha256`], …) wrap the common
//! "hash this whole slice" case.
//!
//! # Example
//!
//! ```
//! use oxixml_support::hash::{sha256, Sha256};
//!
//! // One-shot.
//! let a = sha256(b"abc");
//!
//! // Streaming, in two pieces — identical result.
//! let mut hasher = Sha256::new();
//! hasher.update(b"a");
//! hasher.update(b"bc");
//! let b = hasher.finalize();
//!
//! assert_eq!(a, b);
//! ```
//!
//! [FIPS 180-4]: https://doi.org/10.6028/NIST.FIPS.180-4
//! [FIPS 198-1]: https://doi.org/10.6028/NIST.FIPS.198-1
//! [RFC 2104]: https://www.rfc-editor.org/rfc/rfc2104
//! [RFC 2202]: https://www.rfc-editor.org/rfc/rfc2202
//! [RFC 4231]: https://www.rfc-editor.org/rfc/rfc4231

mod hmac;
mod sha1;
mod sha256;
mod sha512;

pub use hmac::Hmac;
pub use sha1::Sha1;
pub use sha256::{Sha224, Sha256};
pub use sha512::{Sha384, Sha512};

/// The buffer passed to [`Digest::digest_into`] or [`Hmac::finalize_into`] was shorter than the
/// algorithm's fixed output size.
///
/// Every call site inside this crate sizes its buffer correctly and never observes this error;
/// it exists so a caller that builds an output buffer from untrusted or parsed data (for
/// example a digest length read from a document) gets a checkable [`Result`] instead of a
/// panic when that data turns out to be wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthError {
    /// The number of bytes the algorithm actually produces.
    pub expected: usize,
    /// The number of bytes the caller's buffer actually had.
    pub actual: usize,
}

impl std::fmt::Display for LengthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "output buffer too small: need {} bytes, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for LengthError {}

/// A fixed-output streaming hash algorithm, used to make [`Hmac`] generic over the concrete
/// digests in this module.
///
/// Implementors expose their [`BLOCK_SIZE`](Digest::BLOCK_SIZE) and
/// [`OUTPUT_SIZE`](Digest::OUTPUT_SIZE) as associated constants (both in bytes) and provide the
/// two streaming operations. The method names ([`absorb`](Digest::absorb),
/// [`digest_into`](Digest::digest_into)) are deliberately distinct from the inherent
/// `update` / `finalize` methods to keep call resolution unambiguous.
pub trait Digest: Clone + Default {
    /// The internal block size in bytes (64 for SHA-1/224/256, 128 for SHA-384/512).
    const BLOCK_SIZE: usize;
    /// The digest output size in bytes.
    const OUTPUT_SIZE: usize;

    /// Absorb `data` into the running state.
    fn absorb(&mut self, data: &[u8]);

    /// Finish hashing and write exactly [`OUTPUT_SIZE`](Digest::OUTPUT_SIZE) bytes into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`LengthError`] if `out` is shorter than [`OUTPUT_SIZE`](Digest::OUTPUT_SIZE),
    /// leaving `out` unmodified, rather than panicking.
    fn digest_into(self, out: &mut [u8]) -> Result<(), LengthError>;
}

/// Compute the SHA-1 digest of `data` in one call.
///
/// SHA-1 is retained for legacy interoperability (for example verifying pre-existing XML
/// signatures); it is **not** collision-resistant and must not be used for new security
/// guarantees.
#[inline]
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut hasher = Sha1::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute the SHA-224 digest of `data` in one call.
#[inline]
pub fn sha224(data: &[u8]) -> [u8; 28] {
    let mut hasher = Sha224::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute the SHA-256 digest of `data` in one call.
#[inline]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute the SHA-384 digest of `data` in one call.
#[inline]
pub fn sha384(data: &[u8]) -> [u8; 48] {
    let mut hasher = Sha384::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute the SHA-512 digest of `data` in one call.
#[inline]
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute `HMAC-SHA-1(key, message)` in one call.
#[inline]
pub fn hmac_sha1(key: &[u8], message: &[u8]) -> [u8; 20] {
    let mut mac = Hmac::<Sha1>::new(key);
    mac.update(message);
    let mut out = [0u8; 20];
    mac.finalize_into(&mut out)
        .expect("out is exactly Sha1::OUTPUT_SIZE bytes long");
    out
}

/// Compute `HMAC-SHA-256(key, message)` in one call.
#[inline]
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new(key);
    mac.update(message);
    let mut out = [0u8; 32];
    mac.finalize_into(&mut out)
        .expect("out is exactly Sha256::OUTPUT_SIZE bytes long");
    out
}

/// Compute `HMAC-SHA-384(key, message)` in one call.
#[inline]
pub fn hmac_sha384(key: &[u8], message: &[u8]) -> [u8; 48] {
    let mut mac = Hmac::<Sha384>::new(key);
    mac.update(message);
    let mut out = [0u8; 48];
    mac.finalize_into(&mut out)
        .expect("out is exactly Sha384::OUTPUT_SIZE bytes long");
    out
}

/// Compute `HMAC-SHA-512(key, message)` in one call.
#[inline]
pub fn hmac_sha512(key: &[u8], message: &[u8]) -> [u8; 64] {
    let mut mac = Hmac::<Sha512>::new(key);
    mac.update(message);
    let mut out = [0u8; 64];
    mac.finalize_into(&mut out)
        .expect("out is exactly Sha512::OUTPUT_SIZE bytes long");
    out
}
