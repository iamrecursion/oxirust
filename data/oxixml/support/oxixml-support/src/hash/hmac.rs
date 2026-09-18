//! Keyed-hash message authentication (HMAC), generic over the [`Digest`] implementations.
//!
//! Follows [RFC 2104] / [FIPS 198-1]:
//!
//! ```text
//! HMAC(K, m) = H( (K' ^ opad) || H( (K' ^ ipad) || m ) )
//! ```
//!
//! where `K'` is the key padded (or first hashed, then padded) to the digest's block size,
//! `ipad` is the byte `0x36` repeated, and `opad` is the byte `0x5c` repeated.
//!
//! [RFC 2104]: https://www.rfc-editor.org/rfc/rfc2104
//! [FIPS 198-1]: https://doi.org/10.6028/NIST.FIPS.198-1

use super::{Digest, LengthError};

/// The largest block size among the supported digests (SHA-512 uses 128-byte blocks).
const MAX_BLOCK_SIZE: usize = 128;
/// The largest digest output among the supported digests (SHA-512 produces 64 bytes).
const MAX_OUTPUT_SIZE: usize = 64;

/// Inner padding byte.
const IPAD: u8 = 0x36;
/// Outer padding byte.
const OPAD: u8 = 0x5c;

/// An HMAC computation over the digest `D`.
///
/// Construct with [`Hmac::new`], feed the message with [`Hmac::update`], and read the tag with
/// [`Hmac::finalize`] (allocating) or [`Hmac::finalize_into`] (into a caller-provided buffer).
///
/// # Example
///
/// ```
/// use oxixml_support::hash::{Hmac, Sha256};
///
/// let mut mac = Hmac::<Sha256>::new(b"key");
/// mac.update(b"The quick brown fox jumps over the lazy dog");
/// let tag = mac.finalize();
/// assert_eq!(tag.len(), 32);
/// ```
#[derive(Clone)]
pub struct Hmac<D: Digest> {
    /// Digest instance already primed with `K' ^ ipad`, now absorbing the message.
    inner: D,
    /// `K' ^ opad`; only the first `D::BLOCK_SIZE` bytes are meaningful.
    outer_pad: [u8; MAX_BLOCK_SIZE],
}

impl<D: Digest> Hmac<D> {
    /// Create a new HMAC keyed with `key`.
    ///
    /// Keys longer than the digest block size are first hashed, as required by the standard;
    /// shorter keys are zero-padded.
    pub fn new(key: &[u8]) -> Self {
        let block = D::BLOCK_SIZE;
        let mut key_block = [0u8; MAX_BLOCK_SIZE];
        if key.len() > block {
            let mut hasher = D::default();
            hasher.absorb(key);
            hasher.digest_into(&mut key_block[..D::OUTPUT_SIZE]).expect(
                "key_block is MAX_BLOCK_SIZE bytes long, which is at least D::OUTPUT_SIZE \
                 for every Digest implementation in this crate",
            );
        } else {
            key_block[..key.len()].copy_from_slice(key);
        }

        let mut inner_pad = [0u8; MAX_BLOCK_SIZE];
        let mut outer_pad = [0u8; MAX_BLOCK_SIZE];
        for ((inner_byte, outer_byte), &key_byte) in inner_pad[..block]
            .iter_mut()
            .zip(outer_pad[..block].iter_mut())
            .zip(key_block[..block].iter())
        {
            *inner_byte = key_byte ^ IPAD;
            *outer_byte = key_byte ^ OPAD;
        }

        let mut inner = D::default();
        inner.absorb(&inner_pad[..block]);
        Self { inner, outer_pad }
    }

    /// Absorb `data` into the message. May be called any number of times.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.inner.absorb(data);
    }

    /// Finish and write the tag (exactly `D::OUTPUT_SIZE` bytes) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`LengthError`] if `out` is shorter than `D::OUTPUT_SIZE`, leaving `out`
    /// unmodified, rather than panicking. Prefer [`Hmac::finalize`] when an allocation is
    /// acceptable and this precondition would otherwise need checking by hand.
    pub fn finalize_into(self, out: &mut [u8]) -> Result<(), LengthError> {
        if out.len() < D::OUTPUT_SIZE {
            return Err(LengthError {
                expected: D::OUTPUT_SIZE,
                actual: out.len(),
            });
        }
        let block = D::BLOCK_SIZE;
        let mut inner_digest = [0u8; MAX_OUTPUT_SIZE];
        self.inner
            .digest_into(&mut inner_digest[..D::OUTPUT_SIZE])
            .expect(
                "inner_digest is MAX_OUTPUT_SIZE bytes long, which is at least D::OUTPUT_SIZE \
                 for every Digest implementation in this crate",
            );

        let mut outer = D::default();
        outer.absorb(&self.outer_pad[..block]);
        outer.absorb(&inner_digest[..D::OUTPUT_SIZE]);
        outer
            .digest_into(&mut out[..D::OUTPUT_SIZE])
            .expect("checked out.len() >= D::OUTPUT_SIZE above");
        Ok(())
    }

    /// Finish and return the tag as a freshly allocated `Vec<u8>` of length `D::OUTPUT_SIZE`.
    pub fn finalize(self) -> Vec<u8> {
        let mut out = vec![0u8; D::OUTPUT_SIZE];
        self.finalize_into(&mut out)
            .expect("out was just allocated with exactly D::OUTPUT_SIZE bytes");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Sha1, Sha256, Sha384, Sha512, hmac_sha1, hmac_sha256};
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn mac<D: Digest>(key: &[u8], data: &[u8]) -> String {
        let mut hmac = Hmac::<D>::new(key);
        hmac.update(data);
        hex(&hmac.finalize())
    }

    // RFC 2202 §3, HMAC-SHA-1 test cases 1–7.
    #[test]
    fn rfc2202_hmac_sha1() {
        let cases: [(Vec<u8>, Vec<u8>, &str); 7] = [
            (
                vec![0x0b; 20],
                b"Hi There".to_vec(),
                "b617318655057264e28bc0b6fb378c8ef146be00",
            ),
            (
                b"Jefe".to_vec(),
                b"what do ya want for nothing?".to_vec(),
                "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79",
            ),
            (
                vec![0xaa; 20],
                vec![0xdd; 50],
                "125d7342b9ac11cd91a39af48aa17b4f63f175d3",
            ),
            (
                (1u8..=25).collect(),
                vec![0xcd; 50],
                "4c9007f4026250c6bc8414f9bf50c86c2d7235da",
            ),
            (
                vec![0x0c; 20],
                b"Test With Truncation".to_vec(),
                "4c1a03424b55e07fe7f27be1d58bb9324a9a5a04",
            ),
            (
                vec![0xaa; 80],
                b"Test Using Larger Than Block-Size Key - Hash Key First".to_vec(),
                "aa4ae5e15272d00e95705637ce8a3b55ed402112",
            ),
            (
                vec![0xaa; 80],
                b"Test Using Larger Than Block-Size Key and Larger Than One Block-Size Data"
                    .to_vec(),
                "e8e99d0f45237d786d6bbaa7965c7808bbff1a91",
            ),
        ];
        for (key, data, expected) in &cases {
            assert_eq!(&mac::<Sha1>(key, data), expected);
            assert_eq!(hex(&hmac_sha1(key, data)), *expected);
        }
    }

    /// RFC 4231 test data. Case 5 ("Test With Truncation") is verified against the full digest
    /// rather than the RFC's 128-bit truncation.
    fn rfc4231_cases() -> Vec<(Vec<u8>, Vec<u8>)> {
        vec![
            (vec![0x0b; 20], b"Hi There".to_vec()),
            (b"Jefe".to_vec(), b"what do ya want for nothing?".to_vec()),
            (vec![0xaa; 20], vec![0xdd; 50]),
            ((1u8..=25).collect(), vec![0xcd; 50]),
            (vec![0x0c; 20], b"Test With Truncation".to_vec()),
            (vec![0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First".to_vec()),
            (
                vec![0xaa; 131],
                b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.".to_vec(),
            ),
        ]
    }

    #[test]
    fn rfc4231_hmac_sha256() {
        let expected = [
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
            "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
            "a3b6167473100ee06e0c796c2955552bfa6f7c0a6a8aef8b93f860aab0cd20c5",
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
        ];
        for ((key, data), want) in rfc4231_cases().iter().zip(expected.iter()) {
            assert_eq!(&mac::<Sha256>(key, data), want);
            assert_eq!(hex(&hmac_sha256(key, data)), *want);
        }
    }

    #[test]
    fn rfc4231_hmac_sha384() {
        let expected = [
            "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6",
            "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649",
            "88062608d3e6ad8a0aa2ace014c8a86f0aa635d947ac9febe83ef4e55966144b2a5ab39dc13814b94e3ab6e101a34f27",
            "3e8a69b7783c25851933ab6290af6ca77a9981480850009cc5577c6e1f573b4e6801dd23c4a7d679ccf8a386c674cffb",
            "3abf34c3503b2a23a46efc619baef897f4c8e42c934ce55ccbae9740fcbc1af4ca62269e2a37cd88ba926341efe4aeea",
            "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952",
            "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e",
        ];
        for ((key, data), want) in rfc4231_cases().iter().zip(expected.iter()) {
            assert_eq!(&mac::<Sha384>(key, data), want);
        }
    }

    #[test]
    fn rfc4231_hmac_sha512() {
        let expected = [
            "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854",
            "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737",
            "fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb",
            "b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3dba91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd",
            "415fad6271580a531d4179bc891d87a650188707922a4fbb36663a1eb16da008711c5b50ddd0fc235084eb9d3364a1454fb2ef67cd1d29fe6773068ea266e96b",
            "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598",
            "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58",
        ];
        for ((key, data), want) in rfc4231_cases().iter().zip(expected.iter()) {
            assert_eq!(&mac::<Sha512>(key, data), want);
        }
    }

    #[test]
    fn streaming_matches_one_shot() {
        let mut mac = Hmac::<Sha256>::new(b"secret");
        mac.update(b"hello ");
        mac.update(b"world");
        assert_eq!(
            hex(&mac.finalize()),
            hex(&hmac_sha256(b"secret", b"hello world"))
        );
    }

    // A caller that sizes its output buffer from untrusted or parsed data (for example a
    // digest length read out of a document) must get a checkable error, not a panic, when
    // that data turns out to be too short. See `Digest::digest_into`'s own equivalent test in
    // `sha256.rs`/`sha512.rs` for the same guarantee one level down.
    #[test]
    fn finalize_into_reports_undersized_buffer_instead_of_panicking() {
        let mut mac = Hmac::<Sha256>::new(b"key");
        mac.update(b"message");
        let mut short = [0u8; 31]; // Sha256::OUTPUT_SIZE is 32.
        let err = mac
            .finalize_into(&mut short)
            .expect_err("31 bytes is one short of Sha256::OUTPUT_SIZE");
        assert_eq!(err.expected, 32);
        assert_eq!(err.actual, 31);
    }

    #[test]
    fn finalize_into_reports_empty_buffer_instead_of_panicking() {
        let mut mac = Hmac::<Sha1>::new(b"key");
        mac.update(b"message");
        let mut empty: [u8; 0] = [];
        let err = mac
            .finalize_into(&mut empty)
            .expect_err("an empty buffer can never hold a digest");
        assert_eq!(err.expected, 20);
        assert_eq!(err.actual, 0);
    }

    #[test]
    fn finalize_into_accepts_an_oversized_buffer_and_leaves_the_tail_untouched() {
        // The documented contract is "at least OUTPUT_SIZE bytes"; a longer buffer is fine,
        // and only the first OUTPUT_SIZE bytes are written.
        let mut mac = Hmac::<Sha256>::new(b"key");
        mac.update(b"message");
        let mut oversized = [0xAAu8; 40];
        mac.finalize_into(&mut oversized)
            .expect("a 40-byte buffer is more than enough for a 32-byte SHA-256 tag");
        assert_eq!(&oversized[32..], [0xAA; 8]);
    }
}
