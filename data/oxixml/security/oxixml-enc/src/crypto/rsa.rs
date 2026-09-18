//! RSA key transport: the RSAEP/RSADP primitives of
//! [RFC 8017](https://www.rfc-editor.org/rfc/rfc8017) together with the RSAES-OAEP and
//! RSAES-PKCS1-v1_5 encodings that XML Encryption §5.5 names.
//!
//! # Scope and honest limitations
//!
//! Modular exponentiation comes from [`oxixml_support::bigint`], which is a schoolbook
//! implementation with no Montgomery arithmetic, no CRT and **no constant-time discipline**.
//! That has two consequences a caller must understand:
//!
//! * **Performance.** A 2048-bit private-key operation costs on the order of a few thousand
//!   schoolbook multiplications of 32 limbs plus a division each. It is fine for unwrapping one
//!   content-encryption key per document, which is all XML Encryption ever asks for. It is not
//!   fine as a server-side bulk primitive.
//! * **Side channels.** `modpow` is square-and-multiply with a data-dependent branch on the
//!   exponent bits, so a *private-key* operation leaks the exponent to anyone who can time it
//!   or observe the instruction cache. Public-key operations use only public values and are
//!   safe. **Do not use [`RsaPrivateKey`] where a local attacker can measure it.** Closing this
//!   properly requires a constant-time bignum, which belongs in `oxixml-support` rather than
//!   here; until it exists, this is a documented limitation rather than a hidden one.
//!
//! # Key generation, and the entropy it needs
//!
//! XML Encryption itself never generates an RSA key — it consumes long-lived ones a PKI
//! produced — so [`RsaPrivateKey::new`] takes `(n, e, d)` directly and is the usual entry point.
//! [`RsaPrivateKey::generate`] exists for the cases that have no PKI to consume: tests, tools
//! and self-contained deployments. It follows FIPS 186-5 Appendix A.1.1 (random probable
//! primes), with the primality machinery in [`crate::crypto::primes`].
//!
//! There is no random number generator anywhere in this workspace, and there is deliberately
//! none here: generation takes a `&mut dyn RandomSource`, exactly as every other operation in
//! this crate does, and **the caller must supply cryptographic randomness** — the operating
//! system's, via [`SystemRandom`](crate::random::SystemRandom), or its own equivalent. Handing it
//! [`PseudoRandom`](crate::random::PseudoRandom) produces a perfectly well-formed key that
//! anyone who guesses the seed can reproduce.
//!
//! Only the plain `(n, e, d)` representation is stored, because that is all
//! [`oxixml_support::bigint`] can exploit; the primes are used to derive `d` and then dropped,
//! and no CRT parameters are kept.
//!
//! # Manger's attack
//!
//! [Manger, CRYPTO 2001](https://doi.org/10.1007/3-540-44647-8_14) showed that an OAEP decryptor
//! which distinguishes "the integer was ≥ n" from "the leading octet was not zero" from "the
//! label hash did not match" is a full plaintext-recovery oracle in about `k` queries. XMLEnc
//! §6.1.2 repeats the warning. [`RsaPrivateKey::decrypt_oaep`] accordingly computes all three checks, combines
//! them into a single mask, and returns the one opaque [`EncError::DecryptionFailed`]; it never
//! returns early on any of them.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::digest::Digest;
//! use oxixml_enc::crypto::rsa::{OaepParams, RsaPrivateKey};
//! use oxixml_enc::random::FixedRandom;
//!
//! // A 1024-bit key. Real deployments use 2048 bits or more.
//! # let n = oxixml_text::hex::decode(
//! #     "B81C0417F43CDFB4E4975A5A4DBA570A153A435BE3AD3136F27C1F0A9F21C8CA\
//! #      5222F013827F1772BA91F97F8A2AA9C6C063336E875B91FBBE5731E0E72A1175\
//! #      02FAE4E912FBB0400B0D3C79F7265DE81AF04761486DDDCCC5572A6816692180\
//! #      E8E3DB2A1EAD6F75B1CFDA98A7F2A53AA2C71594E6FD83B9CBF8151DABF06B75").expect("hex");
//! # let d = oxixml_text::hex::decode(
//! #     "9680A0800C726ED56B40A6E5C7409AC26D53F1987E68282128DC7A6BAE6669C3\
//! #      AA9D50F041EF2EF5E4BA4BE10CCB8F7A1FEF782DD48CF0AB808FBB0085380887\
//! #      E0AEC90D2D5608B5F469E170D0C5A8A9A04C53A2C867E39BFFECFAFDB134192D\
//! #      A043814CCC4745AFF78F2A008A3784963133986CACABDB9555598B2C4F94B409").expect("hex");
//! let private = RsaPrivateKey::new(&n, &[0x01, 0x00, 0x01], &d)?;
//! let public = private.public_key();
//!
//! let params = OaepParams::new(Digest::Sha256);
//! let mut rng = FixedRandom::new(&[0x9c; 64]);
//! let ciphertext = public.encrypt_oaep(b"a 256-bit content key ..", &params, &mut rng)?;
//! assert_eq!(private.decrypt_oaep(&ciphertext, &params)?, b"a 256-bit content key ..");
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use oxixml_support::bigint::BigUint;

use crate::crypto::ct;
use crate::crypto::digest::Digest;
use crate::crypto::mgf1;
use crate::crypto::primes;
use crate::error::{EncError, Result};
use crate::random::RandomSource;

/// The public exponent [`RsaPrivateKey::generate`] uses, `65537` (`F₄`).
///
/// FIPS 186-5 permits any odd `e` in `65537..2^256`, and this crate offers no choice: 65537 is
/// what every deployment uses, it is prime — which makes the `gcd(e, p−1) = 1` condition a
/// single divisibility test — and its two set bits make the public operation cheap. A smaller
/// exponent invites Coppersmith and Håstad; a larger one buys nothing.
pub const GENERATED_PUBLIC_EXPONENT: u64 = 65537;

/// The narrowest modulus [`RsaPrivateKey::generate`] will produce, in bits.
///
/// This is the floor [`RsaPublicKey::new`] already enforces on a modulus it is *given*. It is
/// the point below which the value cannot be a key at all, not a recommendation: 2048 bits is
/// the minimum for anything that matters, and 512-bit RSA has been factorable since 1999.
pub const MIN_GENERATED_MODULUS_BITS: usize = 512;

/// The widest modulus [`RsaPrivateKey::generate`] will produce, in bits.
///
/// Two primes, each capped by [`primes::MAX_PRIME_BITS`] so that the modular arithmetic has room
/// for its intermediate products.
pub const MAX_GENERATED_MODULUS_BITS: usize = 2 * primes::MAX_PRIME_BITS;

/// How many times generation will start over with a fresh pair of primes.
///
/// Each attempt fails only when `d` comes out below the FIPS 186-5 floor of `2^(nlen/2)`, which
/// for `e = 65537` is vanishingly rare.
const MAX_GENERATION_ATTEMPTS: usize = 16;

/// The parameters of RSAES-OAEP that XML Encryption lets an `EncryptionMethod` carry.
///
/// RFC 8017 allows the message digest, the mask generation function and the label ("P source")
/// to be chosen independently. XML Encryption exposes all three: `ds:DigestMethod` sets
/// [`digest`](Self::digest), `xenc11:MGF` sets [`mgf_digest`](Self::mgf_digest) and
/// `xenc:OAEPparams` sets [`label`](Self::label). The defaults are RFC 8017's: SHA-1 for both
/// hashes and the empty label.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OaepParams {
    /// The hash used for the label digest and inside the encoding.
    pub digest: Digest,
    /// The hash used inside MGF1.
    pub mgf_digest: Digest,
    /// The OAEP label, `P` in RFC 8017's notation — the decoded content of `xenc:OAEPparams`.
    pub label: Vec<u8>,
}

impl OaepParams {
    /// Returns parameters that use `digest` for both the message digest and MGF1, with an empty
    /// label. This is the shape `xenc11:rsa-oaep` takes when only `ds:DigestMethod` is given.
    #[must_use]
    pub fn new(digest: Digest) -> Self {
        Self {
            digest,
            mgf_digest: digest,
            label: Vec::new(),
        }
    }

    /// Returns the RFC 8017 defaults, which are also the fixed parameters of the XML Encryption
    /// 1.0 algorithm `rsa-oaep-mgf1p`: SHA-1 everywhere and an empty label.
    #[must_use]
    pub fn sha1() -> Self {
        Self::new(Digest::Sha1)
    }

    /// Sets the mask-generation-function hash independently of the message digest.
    #[must_use]
    pub fn with_mgf_digest(mut self, digest: Digest) -> Self {
        self.mgf_digest = digest;
        self
    }

    /// Sets the OAEP label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<Vec<u8>>) -> Self {
        self.label = label.into();
        self
    }
}

/// The largest public exponent [`RsaPublicKey::new`] will accept, in bits.
///
/// [FIPS 186-5 Appendix A.1.1](https://doi.org/10.6028/NIST.FIPS.186-5) requires every RSA
/// public exponent it certifies to satisfy `65537 ≤ e < 2^256`; no interoperable key uses one
/// wider than a handful of bits. The bound is deliberately far tighter than
/// [`BigUint::MAX_BITS`], the ceiling on the modulus, because `modpow`'s cost is *linear in the
/// exponent's bit length* regardless of the modulus — so a modulus-only ceiling would still let
/// a modulus and exponent both sitting at [`BigUint::MAX_BITS`] cost on the order of a minute of
/// CPU per operation, which is not a fix for the denial of service this exists to prevent.
pub const MAXIMUM_RSA_PUBLIC_EXPONENT_BITS: usize = 256;

/// An RSA public key: the modulus `n` and the public exponent `e`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsaPublicKey {
    modulus: BigUint,
    exponent: BigUint,
    /// `k` in RFC 8017: the modulus length in octets.
    modulus_len: usize,
}

impl RsaPublicKey {
    /// Builds a public key from the big-endian octet strings of `n` and `e`.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidRsaKey`] when the modulus is shorter than 64 octets
    /// (512 bits — below any plausible key, and the smallest size for which OAEP with SHA-1 has
    /// any capacity at all) or wider than [`BigUint::MAX_BITS`], when it is even, or when the
    /// exponent is not an odd integer of at least three or is wider than
    /// [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`].
    pub fn new(modulus: &[u8], exponent: &[u8]) -> Result<Self> {
        let n = BigUint::from_be_bytes(modulus);
        let e = BigUint::from_be_bytes(exponent);
        let modulus_len = n.to_be_bytes().len();

        if modulus_len < 64 {
            return Err(EncError::InvalidRsaKey("the modulus is implausibly short"));
        }
        // `<xenc:CipherData>`/`KeyInfo` key material is attacker-supplied text with no
        // length limit of its own, and `modpow`'s cost grows with the square of the
        // operand width — so an oversized modulus or exponent is refused here, before
        // any exponentiation runs, rather than costing seconds of CPU per operation.
        if n.bit_len() > BigUint::MAX_BITS {
            return Err(EncError::InvalidRsaKey("the modulus is implausibly large"));
        }
        // The product of two odd primes is odd; an even modulus is malformed.
        if n.to_be_bytes().last().is_some_and(|byte| byte % 2 == 0) {
            return Err(EncError::InvalidRsaKey("the modulus is even"));
        }
        if e < BigUint::from_u64(3) {
            return Err(EncError::InvalidRsaKey("the public exponent is too small"));
        }
        // The exponent bound is the load-bearing one: it is cheaper for an attacker to
        // grow than the modulus, and `modpow`'s cost is linear in the exponent's bit
        // length regardless of the modulus width — so it gets its own, much narrower
        // ceiling rather than sharing `BigUint::MAX_BITS` with the modulus.
        if e.bit_len() > MAXIMUM_RSA_PUBLIC_EXPONENT_BITS {
            return Err(EncError::InvalidRsaKey(
                "the public exponent is implausibly large",
            ));
        }
        if e.to_be_bytes().last().is_some_and(|byte| byte % 2 == 0) {
            return Err(EncError::InvalidRsaKey("the public exponent is even"));
        }
        Ok(Self {
            modulus: n,
            exponent: e,
            modulus_len,
        })
    }

    /// Returns `k`, the modulus length in octets.
    #[must_use]
    pub const fn modulus_len(&self) -> usize {
        self.modulus_len
    }

    /// Returns the modulus as a big-endian octet string.
    #[must_use]
    pub fn modulus_bytes(&self) -> Vec<u8> {
        self.modulus.to_be_bytes()
    }

    /// Returns the public exponent as a big-endian octet string.
    #[must_use]
    pub fn exponent_bytes(&self) -> Vec<u8> {
        self.exponent.to_be_bytes()
    }

    /// `RSAEP((n, e), m)` of RFC 8017 §5.1.1.
    fn rsaep(&self, message: &BigUint) -> Result<Vec<u8>> {
        if *message >= self.modulus {
            return Err(EncError::InvalidRsaKey(
                "the encoded message is not smaller than the modulus",
            ));
        }
        let value = message
            .modpow(&self.exponent, &self.modulus)
            .ok_or(EncError::InvalidRsaKey("modular exponentiation failed"))?;
        value
            .to_be_bytes_padded(self.modulus_len)
            .ok_or(EncError::InvalidRsaKey("the result overflowed the modulus"))
    }

    /// `RSAES-OAEP-ENCRYPT((n, e), M, L)` of RFC 8017 §7.1.1.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::MessageTooLong`] when `message` exceeds `k - 2·hLen - 2` octets, and
    /// [`EncError::Entropy`] when the random source fails.
    pub fn encrypt_oaep(
        &self,
        message: &[u8],
        params: &OaepParams,
        rng: &mut dyn RandomSource,
    ) -> Result<Vec<u8>> {
        let encoded = oaep_encode(message, params, self.modulus_len, rng)?;
        self.rsaep(&BigUint::from_be_bytes(&encoded))
    }

    /// `RSAES-PKCS1-V1_5-ENCRYPT((n, e), M)` of RFC 8017 §7.2.1.
    ///
    /// XML Encryption §5.5.1 marks this algorithm NOT RECOMMENDED because of Bleichenbacher's
    /// 1998 adaptive chosen-ciphertext attack; it is implemented for interoperability with
    /// existing `rsa-1_5` deployments only.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::MessageTooLong`] when `message` exceeds `k - 11` octets, and
    /// [`EncError::Entropy`] when the random source fails.
    pub fn encrypt_pkcs1v15(&self, message: &[u8], rng: &mut dyn RandomSource) -> Result<Vec<u8>> {
        let maximum = self.modulus_len.saturating_sub(11);
        if message.len() > maximum {
            return Err(EncError::MessageTooLong {
                length: message.len(),
                maximum,
            });
        }
        // EM = 0x00 ‖ 0x02 ‖ PS ‖ 0x00 ‖ M, with PS non-zero and at least eight octets long.
        let padding_len = self.modulus_len - message.len() - 3;
        let mut encoded = Vec::with_capacity(self.modulus_len);
        encoded.push(0x00);
        encoded.push(0x02);
        let start = encoded.len();
        encoded.resize(start + padding_len, 0);
        fill_non_zero(&mut encoded[start..start + padding_len], rng)?;
        encoded.push(0x00);
        encoded.extend_from_slice(message);
        self.rsaep(&BigUint::from_be_bytes(&encoded))
    }
}

/// An RSA private key: the modulus `n`, the public exponent `e` and the private exponent `d`.
///
/// Only the plain (non-CRT) representation is kept, because that is all
/// [`oxixml_support::bigint`] can exploit. See the [module documentation](self) for the
/// side-channel consequences.
#[derive(Clone, PartialEq, Eq)]
pub struct RsaPrivateKey {
    public: RsaPublicKey,
    private_exponent: BigUint,
}

impl core::fmt::Debug for RsaPrivateKey {
    /// Never prints the private exponent.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RsaPrivateKey")
            .field("modulus_bits", &(self.public.modulus_len * 8))
            .finish_non_exhaustive()
    }
}

impl RsaPrivateKey {
    /// Builds a private key from the big-endian octet strings of `n`, `e` and `d`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`RsaPublicKey::new`], and [`EncError::InvalidRsaKey`] when the
    /// private exponent is zero or is not smaller than the modulus.
    pub fn new(modulus: &[u8], exponent: &[u8], private_exponent: &[u8]) -> Result<Self> {
        let public = RsaPublicKey::new(modulus, exponent)?;
        let d = BigUint::from_be_bytes(private_exponent);
        if d.is_zero() {
            return Err(EncError::InvalidRsaKey("the private exponent is zero"));
        }
        if d >= public.modulus {
            return Err(EncError::InvalidRsaKey(
                "the private exponent is not smaller than the modulus",
            ));
        }
        Ok(Self {
            public,
            private_exponent: d,
        })
    }

    /// Generates a fresh key pair of `modulus_bits` bits, drawing every random octet from `rng`.
    ///
    /// This is FIPS 186-5 Appendix A.1.1, "random probable primes":
    ///
    /// * `p` and `q` are random primes of half the modulus width with their **top two bits
    ///   set**, so the product has exactly `modulus_bits` bits.
    /// * Each is checked against [`GENERATED_PUBLIC_EXPONENT`]: `gcd(e, p−1) = gcd(e, q−1) = 1`,
    ///   without which `e` has no inverse and the key would not be one.
    /// * They are kept apart — `|p − q| > 2^(modulus_bits/2 − 100)` — because two primes close
    ///   together let Fermat's method split the modulus by hand.
    /// * `d = e⁻¹ mod λ(n)` with `λ(n) = lcm(p−1, q−1)`, and `d` is required to exceed
    ///   `2^(modulus_bits/2)`; a `d` below that root is recoverable by Wiener's attack.
    ///
    /// The primes are dropped once `d` is derived. Only `(n, e, d)` is kept, so this key costs
    /// the same to use as one read from a PKI, and no CRT parameters exist to leak.
    ///
    /// # The random source is the whole security of the result
    ///
    /// `rng` **must** be a cryptographic source — see the [module documentation](self). A key
    /// generated from a predictable source is predictable, and nothing downstream can detect it:
    /// the key is perfectly well-formed.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidRsaKey`] when `modulus_bits` is odd or outside
    /// `MIN_GENERATED_MODULUS_BITS..=MAX_GENERATED_MODULUS_BITS`, and when the search does not
    /// converge — which means the random source is not producing what it claims to. Returns
    /// [`EncError::Entropy`] when the random source fails outright.
    pub fn generate(modulus_bits: usize, rng: &mut dyn RandomSource) -> Result<Self> {
        if modulus_bits % 2 != 0 {
            return Err(EncError::InvalidRsaKey(
                "the modulus width must be an even number of bits",
            ));
        }
        if !(MIN_GENERATED_MODULUS_BITS..=MAX_GENERATED_MODULUS_BITS).contains(&modulus_bits) {
            return Err(EncError::InvalidRsaKey(
                "the requested modulus width is out of range",
            ));
        }

        let half = modulus_bits / 2;
        let rounds = primes::miller_rabin_rounds(half);
        let exponent = BigUint::from_u64(GENERATED_PUBLIC_EXPONENT);
        let one = BigUint::one();
        let separation_bits = half.saturating_sub(100);

        for _ in 0..MAX_GENERATION_ATTEMPTS {
            let p = usable_prime(half, rounds, &exponent, None, separation_bits, rng)?;
            let q = usable_prime(half, rounds, &exponent, Some(&p), separation_bits, rng)?;

            let modulus = p
                .checked_mul(&q)
                .ok_or(EncError::InvalidRsaKey("the modulus overflowed"))?;
            if modulus.bit_len() != modulus_bits {
                continue;
            }

            // λ(n) = lcm(p−1, q−1) = (p−1)(q−1) / gcd(p−1, q−1).
            let p_minus_one = p
                .checked_sub(&one)
                .ok_or(EncError::InvalidRsaKey("a prime was zero"))?;
            let q_minus_one = q
                .checked_sub(&one)
                .ok_or(EncError::InvalidRsaKey("a prime was zero"))?;
            let product = p_minus_one
                .checked_mul(&q_minus_one)
                .ok_or(EncError::InvalidRsaKey("the totient overflowed"))?;
            let common = gcd(&p_minus_one, &q_minus_one)?;
            let (lambda, _) = product.div_rem(&common).ok_or(EncError::InvalidRsaKey(
                "computing the Carmichael function failed",
            ))?;

            let Some(private_exponent) = exponent.mod_inverse(&lambda) else {
                continue;
            };
            // FIPS 186-5 A.1.1: 2^(nlen/2) < d. `d` is odd, so this comparison of bit lengths is
            // exact.
            if private_exponent.bit_len() <= half {
                continue;
            }

            return Self::new(
                &modulus.to_be_bytes(),
                &exponent.to_be_bytes(),
                &private_exponent.to_be_bytes(),
            );
        }

        Err(EncError::InvalidRsaKey("key generation did not converge"))
    }

    /// Returns the matching public key.
    #[must_use]
    pub fn public_key(&self) -> RsaPublicKey {
        self.public.clone()
    }

    /// Returns `k`, the modulus length in octets.
    #[must_use]
    pub const fn modulus_len(&self) -> usize {
        self.public.modulus_len
    }

    /// `RSADP((n, d), c)` of RFC 8017 §5.1.2.
    fn rsadp(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        // A ciphertext of the wrong length, or one representing an integer ≥ n, is a
        // *structural* error that a passive observer already knows; it is still folded into the
        // opaque failure so that a Manger oracle cannot be built out of the distinction.
        if ciphertext.len() != self.public.modulus_len {
            return Err(EncError::DecryptionFailed);
        }
        let value = BigUint::from_be_bytes(ciphertext);
        if value >= self.public.modulus {
            return Err(EncError::DecryptionFailed);
        }
        let plain = value
            .modpow(&self.private_exponent, &self.public.modulus)
            .ok_or(EncError::DecryptionFailed)?;
        plain
            .to_be_bytes_padded(self.public.modulus_len)
            .ok_or(EncError::DecryptionFailed)
    }

    /// `RSAES-OAEP-DECRYPT(K, C, L)` of RFC 8017 §7.1.2.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::DecryptionFailed`] for every failure, without distinguishing them —
    /// see the [module documentation](self).
    pub fn decrypt_oaep(&self, ciphertext: &[u8], params: &OaepParams) -> Result<Vec<u8>> {
        let encoded = self.rsadp(ciphertext)?;
        oaep_decode(&encoded, params)
    }

    /// `RSAES-PKCS1-V1_5-DECRYPT(K, C)` of RFC 8017 §7.2.2.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::DecryptionFailed`] for every failure. This is necessary but *not*
    /// sufficient to stop Bleichenbacher's attack, which also exploits timing and the behaviour
    /// of whatever consumes the recovered key; prefer OAEP.
    pub fn decrypt_pkcs1v15(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        let encoded = self.rsadp(ciphertext)?;
        if encoded.len() < 11 {
            return Err(EncError::DecryptionFailed);
        }

        // Walk the whole buffer, recording where the first zero octet after the padding is,
        // without ever branching out early.
        let mut valid = ct::mask_eq(encoded[0], 0x00) & ct::mask_eq(encoded[1], 0x02);
        let mut separator_index = 0usize;
        let mut found = 0u8;
        for (index, byte) in encoded.iter().enumerate().skip(2) {
            let is_zero = ct::mask_is_zero(*byte);
            let is_first = is_zero & !found;
            // `separator_index` moves only on the first zero octet.
            separator_index |= index & usize::from(is_first != 0).wrapping_neg();
            found |= is_zero;
        }
        // The padding string must be at least eight octets, so the separator cannot be before
        // index 10, and it must exist at all.
        valid &= found;
        valid &= !ct::mask_le(separator_index, 9);

        if valid != 0xff {
            return Err(EncError::DecryptionFailed);
        }
        Ok(encoded[separator_index + 1..].to_vec())
    }
}

/// The greatest common divisor of `left` and `right`, by the Euclidean algorithm.
fn gcd(left: &BigUint, right: &BigUint) -> Result<BigUint> {
    let mut a = left.clone();
    let mut b = right.clone();
    while !b.is_zero() {
        let (_, remainder) = a
            .div_rem(&b)
            .ok_or(EncError::InvalidRsaKey("a division failed"))?;
        a = b;
        b = remainder;
    }
    Ok(a)
}

/// `|left − right|`.
fn absolute_difference(left: &BigUint, right: &BigUint) -> Result<BigUint> {
    let (larger, smaller) = if left >= right {
        (left, right)
    } else {
        (right, left)
    };
    larger
        .checked_sub(smaller)
        .ok_or(EncError::InvalidRsaKey("a subtraction failed"))
}

/// Draws a prime that can actually serve as one half of an RSA modulus.
///
/// Beyond primality that means two things: `e` must be invertible modulo `prime − 1`, and — when
/// the other prime is already in hand — the two must be far enough apart that Fermat
/// factorisation cannot split their product.
fn usable_prime(
    bits: usize,
    rounds: usize,
    exponent: &BigUint,
    other: Option<&BigUint>,
    separation_bits: usize,
    rng: &mut dyn RandomSource,
) -> Result<BigUint> {
    let one = BigUint::one();
    for _ in 0..MAX_GENERATION_ATTEMPTS {
        let prime = primes::generate_prime(bits, rounds, rng)?;
        let minus_one = prime
            .checked_sub(&one)
            .ok_or(EncError::InvalidRsaKey("a prime was zero"))?;
        if !gcd(exponent, &minus_one)?.is_one() {
            continue;
        }
        if let Some(other) = other {
            if absolute_difference(&prime, other)?.bit_len() <= separation_bits {
                continue;
            }
        }
        return Ok(prime);
    }
    Err(EncError::InvalidRsaKey(
        "no prime usable with this public exponent was found",
    ))
}

/// `EME-OAEP-ENCODE` of RFC 8017 §7.1.1 steps 2a–2i.
fn oaep_encode(
    message: &[u8],
    params: &OaepParams,
    modulus_len: usize,
    rng: &mut dyn RandomSource,
) -> Result<Vec<u8>> {
    let hash_len = params.digest.output_len();
    let maximum = modulus_len
        .checked_sub(2 * hash_len + 2)
        .ok_or(EncError::InvalidRsaKey(
            "the modulus is too short for the chosen OAEP digest",
        ))?;
    if message.len() > maximum {
        return Err(EncError::MessageTooLong {
            length: message.len(),
            maximum,
        });
    }

    // DB = lHash ‖ PS ‖ 0x01 ‖ M
    let mut db = Vec::with_capacity(modulus_len - hash_len - 1);
    db.extend_from_slice(&params.digest.hash(&params.label));
    db.resize(modulus_len - hash_len - 1 - message.len() - 1, 0);
    db.push(0x01);
    db.extend_from_slice(message);

    let mut seed = vec![0u8; hash_len];
    rng.fill(&mut seed)?;

    mgf1::mask_in_place(params.mgf_digest, &seed, &mut db);
    mgf1::mask_in_place(params.mgf_digest, &db, &mut seed);

    // EM = 0x00 ‖ maskedSeed ‖ maskedDB
    let mut encoded = Vec::with_capacity(modulus_len);
    encoded.push(0x00);
    encoded.extend_from_slice(&seed);
    encoded.extend_from_slice(&db);
    Ok(encoded)
}

/// `EME-OAEP-DECODE` of RFC 8017 §7.1.2 step 3.
///
/// Every check contributes to one mask and the function returns at a single point, so no
/// control-flow difference distinguishes "the leading octet was not zero" from "the label hash
/// did not match" from "there was no `0x01` separator" — the three distinctions Manger's attack
/// needs.
fn oaep_decode(encoded: &[u8], params: &OaepParams) -> Result<Vec<u8>> {
    let hash_len = params.digest.output_len();
    if encoded.len() < 2 * hash_len + 2 {
        return Err(EncError::DecryptionFailed);
    }

    let mut valid = ct::mask_is_zero(encoded[0]);

    let mut seed = encoded[1..1 + hash_len].to_vec();
    let mut db = encoded[1 + hash_len..].to_vec();
    mgf1::mask_in_place(params.mgf_digest, &db, &mut seed);
    mgf1::mask_in_place(params.mgf_digest, &seed, &mut db);

    let expected_label_hash = params.digest.hash(&params.label);
    let mut label_difference = 0u8;
    for (left, right) in db[..hash_len].iter().zip(expected_label_hash.iter()) {
        label_difference |= left ^ right;
    }
    valid &= ct::mask_is_zero(label_difference);

    // Find the 0x01 separator that ends the zero padding, scanning the whole buffer.
    let mut separator_index = 0usize;
    let mut found = 0u8;
    let mut padding_difference = 0u8;
    for (index, byte) in db.iter().enumerate().skip(hash_len) {
        let is_one = ct::mask_eq(*byte, 0x01);
        let is_first = is_one & !found;
        separator_index |= index & usize::from(is_first != 0).wrapping_neg();
        // Before the separator every octet must be zero.
        padding_difference |= *byte & !found & !is_one;
        found |= is_one;
    }
    valid &= found;
    valid &= ct::mask_is_zero(padding_difference);

    if valid != 0xff {
        return Err(EncError::DecryptionFailed);
    }
    Ok(db[separator_index + 1..].to_vec())
}

/// Fills `buffer` with non-zero octets, as the PKCS#1 v1.5 padding string requires.
///
/// Draws in bulk and re-draws only the zeros, so the expected number of calls into the random
/// source is one plus a vanishing tail.
fn fill_non_zero(buffer: &mut [u8], rng: &mut dyn RandomSource) -> Result<()> {
    rng.fill(buffer)?;
    // 255/256 of the octets survive each pass; sixteen passes leaves a probability below 2^-100
    // of any zero remaining, and the loop below simply keeps going until none does.
    let mut replacement = [0u8; 32];
    let mut cursor = replacement.len();
    for byte in buffer.iter_mut() {
        while *byte == 0 {
            if cursor == replacement.len() {
                rng.fill(&mut replacement)?;
                cursor = 0;
            }
            *byte = replacement[cursor];
            cursor += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{FixedRandom, PseudoRandom};

    /// The 1024-bit key used throughout the unit tests; the integration tests reuse it.
    fn test_key() -> Result<RsaPrivateKey> {
        let n = oxixml_text::hex::decode(concat!(
            "B81C0417F43CDFB4E4975A5A4DBA570A153A435BE3AD3136F27C1F0A9F21C8CA",
            "5222F013827F1772BA91F97F8A2AA9C6C063336E875B91FBBE5731E0E72A1175",
            "02FAE4E912FBB0400B0D3C79F7265DE81AF04761486DDDCCC5572A6816692180",
            "E8E3DB2A1EAD6F75B1CFDA98A7F2A53AA2C71594E6FD83B9CBF8151DABF06B75",
        ))
        .map_err(|_| EncError::InvalidRsaKey("bad test modulus"))?;
        let d = oxixml_text::hex::decode(concat!(
            "9680A0800C726ED56B40A6E5C7409AC26D53F1987E68282128DC7A6BAE6669C3",
            "AA9D50F041EF2EF5E4BA4BE10CCB8F7A1FEF782DD48CF0AB808FBB0085380887",
            "E0AEC90D2D5608B5F469E170D0C5A8A9A04C53A2C867E39BFFECFAFDB134192D",
            "A043814CCC4745AFF78F2A008A3784963133986CACABDB9555598B2C4F94B409",
        ))
        .map_err(|_| EncError::InvalidRsaKey("bad test exponent"))?;
        RsaPrivateKey::new(&n, &[0x01, 0x00, 0x01], &d)
    }

    #[test]
    fn the_key_is_consistent() -> Result<()> {
        let key = test_key()?;
        assert_eq!(key.modulus_len(), 128);
        // (m^e)^d ≡ m (mod n) is the defining property of an RSA key pair.
        let message = BigUint::from_u64(0x1234_5678_9abc_def0);
        let public = key.public_key();
        let ciphertext = public.rsaep(&message)?;
        let recovered = key.rsadp(&ciphertext)?;
        assert_eq!(BigUint::from_be_bytes(&recovered), message);
        Ok(())
    }

    #[test]
    fn oaep_round_trips_with_every_digest() -> Result<()> {
        let key = test_key()?;
        let public = key.public_key();
        for digest in Digest::ALL {
            // A 1024-bit modulus has no OAEP capacity at all with SHA-512.
            if 2 * digest.output_len() + 2 > public.modulus_len() {
                continue;
            }
            let params = OaepParams::new(digest);
            let mut rng = FixedRandom::new(&[0x3c; 128]);
            let ciphertext = public.encrypt_oaep(b"content key", &params, &mut rng)?;
            assert_eq!(ciphertext.len(), 128);
            assert_eq!(key.decrypt_oaep(&ciphertext, &params)?, b"content key");
        }
        Ok(())
    }

    #[test]
    fn oaep_honours_the_label() -> Result<()> {
        let key = test_key()?;
        let public = key.public_key();
        let encrypting = OaepParams::new(Digest::Sha256).with_label(b"label-a".to_vec());
        let decrypting = OaepParams::new(Digest::Sha256).with_label(b"label-b".to_vec());
        let mut rng = FixedRandom::new(&[0x71; 64]);
        let ciphertext = public.encrypt_oaep(b"key", &encrypting, &mut rng)?;
        assert_eq!(key.decrypt_oaep(&ciphertext, &encrypting)?, b"key");
        assert!(matches!(
            key.decrypt_oaep(&ciphertext, &decrypting),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn oaep_honours_an_independent_mgf_digest() -> Result<()> {
        let key = test_key()?;
        let public = key.public_key();
        let params = OaepParams::new(Digest::Sha1).with_mgf_digest(Digest::Sha256);
        let mut rng = FixedRandom::new(&[0x22; 64]);
        let ciphertext = public.encrypt_oaep(b"key", &params, &mut rng)?;
        assert_eq!(key.decrypt_oaep(&ciphertext, &params)?, b"key");
        assert!(matches!(
            key.decrypt_oaep(&ciphertext, &OaepParams::sha1()),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn oaep_rejects_an_over_long_message() -> Result<()> {
        let public = test_key()?.public_key();
        let params = OaepParams::new(Digest::Sha256);
        // k - 2·hLen - 2 = 128 - 64 - 2 = 62.
        assert!(
            public
                .encrypt_oaep(&[0u8; 62], &params, &mut FixedRandom::new(&[1; 64]))
                .is_ok()
        );
        assert!(matches!(
            public.encrypt_oaep(&[0u8; 63], &params, &mut FixedRandom::new(&[1; 64])),
            Err(EncError::MessageTooLong {
                length: 63,
                maximum: 62
            })
        ));
        Ok(())
    }

    #[test]
    fn a_tampered_oaep_ciphertext_fails_opaquely() -> Result<()> {
        let key = test_key()?;
        let params = OaepParams::sha1();
        let mut ciphertext =
            key.public_key()
                .encrypt_oaep(b"key", &params, &mut FixedRandom::new(&[0x40; 64]))?;
        ciphertext[64] ^= 0x01;
        assert!(matches!(
            key.decrypt_oaep(&ciphertext, &params),
            Err(EncError::DecryptionFailed)
        ));
        // A ciphertext of the wrong length is the same opaque failure.
        assert!(matches!(
            key.decrypt_oaep(&ciphertext[1..], &params),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn pkcs1v15_round_trips() -> Result<()> {
        let key = test_key()?;
        let public = key.public_key();
        let mut rng = FixedRandom::new(&[0x8e; 256]);
        for length in [1usize, 16, 32, 117] {
            let message: Vec<u8> = (0..length).map(|index| (index as u8) | 1).collect();
            let ciphertext = public.encrypt_pkcs1v15(&message, &mut rng)?;
            assert_eq!(key.decrypt_pkcs1v15(&ciphertext)?, message);
        }
        assert!(matches!(
            public.encrypt_pkcs1v15(&[0u8; 118], &mut rng),
            Err(EncError::MessageTooLong {
                length: 118,
                maximum: 117
            })
        ));
        Ok(())
    }

    #[test]
    fn pkcs1v15_rejects_a_tampered_ciphertext() -> Result<()> {
        let key = test_key()?;
        let mut ciphertext = key
            .public_key()
            .encrypt_pkcs1v15(b"key", &mut FixedRandom::new(&[0x19; 256]))?;
        ciphertext[3] ^= 0x40;
        assert!(matches!(
            key.decrypt_pkcs1v15(&ciphertext),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn malformed_keys_are_rejected() {
        assert!(matches!(
            RsaPublicKey::new(&[0xffu8; 32], &[0x01, 0x00, 0x01]),
            Err(EncError::InvalidRsaKey(_))
        ));
        let mut even = [0xffu8; 128];
        even[127] = 0xfe;
        assert!(matches!(
            RsaPublicKey::new(&even, &[0x01, 0x00, 0x01]),
            Err(EncError::InvalidRsaKey(_))
        ));
        assert!(matches!(
            RsaPublicKey::new(&[0xffu8; 128], &[0x01]),
            Err(EncError::InvalidRsaKey(_))
        ));
        assert!(matches!(
            RsaPublicKey::new(&[0xffu8; 128], &[0x04]),
            Err(EncError::InvalidRsaKey(_))
        ));
        assert!(matches!(
            RsaPrivateKey::new(&[0xffu8; 128], &[0x01, 0x00, 0x01], &[]),
            Err(EncError::InvalidRsaKey(_))
        ));
        assert!(matches!(
            RsaPrivateKey::new(&[0xffu8; 128], &[0x01, 0x00, 0x01], &[0xffu8; 129]),
            Err(EncError::InvalidRsaKey(_))
        ));
    }

    /// `xenc:CipherData`/`KeyInfo` key material is attacker-supplied text with no length
    /// limit of its own, and `modpow`'s cost grows with the square of the operand width.
    /// An oversized modulus or exponent must be refused here, in microseconds, rather than
    /// reaching `modpow` and costing seconds of CPU per operation.
    #[test]
    fn an_oversized_modulus_is_refused_before_any_arithmetic_runs() {
        // One bit past BigUint::MAX_BITS, odd, and otherwise shaped like a real modulus.
        let mut modulus = vec![0x01u8; BigUint::MAX_BITS / 8 + 1];
        if let Some(last) = modulus.last_mut() {
            *last |= 1;
        }
        assert!(matches!(
            RsaPublicKey::new(&modulus, &[0x01, 0x00, 0x01]),
            Err(EncError::InvalidRsaKey(_))
        ));
    }

    /// The exponent bound is the load-bearing one: it is cheaper for an attacker to grow
    /// than the modulus, and `modpow`'s cost is linear in the exponent's bit length
    /// regardless of the modulus width, so a modulus-only ceiling would leave this open.
    #[test]
    fn an_oversized_exponent_is_refused_before_any_arithmetic_runs() {
        let modulus = [0xffu8; 128];
        let mut exponent = vec![0x01u8; MAXIMUM_RSA_PUBLIC_EXPONENT_BITS / 8 + 1];
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        assert!(matches!(
            RsaPublicKey::new(&modulus, &exponent),
            Err(EncError::InvalidRsaKey(_))
        ));
    }

    /// An exponent between [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`] and [`BigUint::MAX_BITS`]
    /// used to be accepted when the exponent shared the modulus's ceiling — and `modpow`'s
    /// cost is linear in the exponent's bit length regardless of the modulus, so a modulus
    /// and exponent both that wide cost on the order of a minute of CPU per operation. It
    /// must be refused now.
    #[test]
    fn an_exponent_within_the_old_shared_ceiling_is_still_refused() {
        let modulus = [0xffu8; 128];
        let mut exponent = vec![0x01u8; BigUint::MAX_BITS / 8];
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        assert!(exponent.len() * 8 > MAXIMUM_RSA_PUBLIC_EXPONENT_BITS);
        assert!(matches!(
            RsaPublicKey::new(&modulus, &exponent),
            Err(EncError::InvalidRsaKey(_))
        ));
    }

    #[test]
    fn a_modulus_exactly_at_the_ceiling_is_still_accepted() -> Result<()> {
        let mut modulus = vec![0u8; BigUint::MAX_BITS / 8];
        if let Some(first) = modulus.first_mut() {
            *first = 0x80;
        }
        if let Some(last) = modulus.last_mut() {
            *last |= 1;
        }
        let key = RsaPublicKey::new(&modulus, &[0x01, 0x00, 0x01])?;
        assert_eq!(key.modulus_len() * 8, BigUint::MAX_BITS);
        Ok(())
    }

    #[test]
    fn an_exponent_exactly_at_its_own_ceiling_is_still_accepted() -> Result<()> {
        let modulus = [0xffu8; 128];
        let mut exponent = vec![0u8; MAXIMUM_RSA_PUBLIC_EXPONENT_BITS / 8];
        if let Some(first) = exponent.first_mut() {
            *first = 0x80;
        }
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        let key = RsaPublicKey::new(&modulus, &exponent)?;
        assert_eq!(
            key.exponent_bytes().len() * 8,
            MAXIMUM_RSA_PUBLIC_EXPONENT_BITS
        );
        Ok(())
    }

    #[test]
    fn debug_does_not_leak_the_private_exponent() -> Result<()> {
        let key = test_key()?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("1024"), "{rendered}");
        assert!(!rendered.contains("9680"), "{rendered}");
        Ok(())
    }

    /// Generation is exercised at the crate's floor, 512 bits. That is not a key anyone should
    /// use — it is a key size an attacker factors — but it is a *valid* one, and the arithmetic
    /// it exercises is the same at every width.
    #[test]
    fn a_generated_key_is_a_working_key() -> Result<()> {
        let mut rng = PseudoRandom::new(0x0117);
        let key = RsaPrivateKey::generate(512, &mut rng)?;
        let public = key.public_key();

        assert_eq!(public.modulus_len(), 64);
        assert_eq!(public.exponent_bytes(), vec![0x01, 0x00, 0x01]);
        let modulus = BigUint::from_be_bytes(&public.modulus_bytes());
        assert_eq!(
            modulus.bit_len(),
            512,
            "the top two bits of each prime were set"
        );
        assert!(modulus.bit(0), "a product of two odd primes is odd");

        // (m^e)^d ≡ m (mod n): the defining property, on a value that spans the modulus.
        let message = BigUint::from_be_bytes(&[0x5a; 60]);
        let ciphertext = public.rsaep(&message)?;
        assert_eq!(BigUint::from_be_bytes(&key.rsadp(&ciphertext)?), message);

        // And it works through the key-transport machinery it exists for.
        let params = OaepParams::sha1();
        let content_key = [0x2c; 16];
        let wrapped = public.encrypt_oaep(&content_key, &params, &mut rng)?;
        assert_eq!(key.decrypt_oaep(&wrapped, &params)?, content_key);
        Ok(())
    }

    #[test]
    fn out_of_range_modulus_widths_are_refused() {
        let mut rng = PseudoRandom::new(1);
        for bits in [
            511usize,
            MIN_GENERATED_MODULUS_BITS - 2,
            MAX_GENERATED_MODULUS_BITS + 2,
        ] {
            assert!(
                matches!(
                    RsaPrivateKey::generate(bits, &mut rng),
                    Err(EncError::InvalidRsaKey(_))
                ),
                "{bits} bits was accepted"
            );
        }
    }

    #[test]
    fn the_greatest_common_divisor_is_euclid() -> Result<()> {
        assert_eq!(
            gcd(&BigUint::from_u64(462), &BigUint::from_u64(1071))?,
            BigUint::from_u64(21)
        );
        assert_eq!(
            gcd(&BigUint::from_u64(1071), &BigUint::from_u64(462))?,
            BigUint::from_u64(21)
        );
        assert_eq!(
            gcd(&BigUint::from_u64(17), &BigUint::zero())?,
            BigUint::from_u64(17)
        );
        // 65537 is prime, so it shares a factor only with its own multiples.
        assert!(gcd(&BigUint::from_u64(65537), &BigUint::from_u64(65536))?.is_one());
        assert_eq!(
            gcd(&BigUint::from_u64(65537), &BigUint::from_u64(131_074))?,
            BigUint::from_u64(65537)
        );
        Ok(())
    }

    #[test]
    fn the_absolute_difference_is_symmetric() -> Result<()> {
        let small = BigUint::from_u64(7);
        let large = BigUint::from_u64(19);
        assert_eq!(absolute_difference(&small, &large)?, BigUint::from_u64(12));
        assert_eq!(absolute_difference(&large, &small)?, BigUint::from_u64(12));
        assert!(absolute_difference(&large, &large)?.is_zero());
        Ok(())
    }

    #[test]
    fn the_padding_string_is_never_zero() -> Result<()> {
        // A random source that emits a great many zeros still yields non-zero padding.
        let mut zeros = FixedRandom::new(&[0x00, 0x00, 0x00, 0x2a]);
        let mut buffer = [0u8; 40];
        fill_non_zero(&mut buffer, &mut zeros)?;
        assert!(buffer.iter().all(|byte| *byte != 0));
        Ok(())
    }
}
