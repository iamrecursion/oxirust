//! Finite-field Diffie–Hellman, the primitive behind XML Encryption's `dh` and `dh-es` key
//! agreement methods.
//!
//! The primitive is the one NIST SP 800-56A §5.7.1.1 calls FFC DH: with a shared prime `p`, a
//! generator `g` and private exponents `x` and `x'`, both parties reach
//! `ZZ = g^(x·x') mod p` from the other's public value. XMLEnc 1.0 §5.5.2 fixes the encoding of
//! that shared secret: **leading zero octets are kept, so `ZZ` is exactly as long as `p`**.
//! Dropping them — which is what a naive big-integer serialization does — derives a different
//! key one time in 256 and produces a decryption failure nobody can reproduce.
//!
//! # Validating what the peer sent
//!
//! The peer's public value is attacker-controlled, and two classic attacks follow from failing
//! to check it:
//!
//! * `y ∈ {0, 1, p−1}` generates a subgroup of order 1 or 2, so `ZZ` is one of a handful of
//!   values regardless of the private exponent. [`DhPublicKey::new`] refuses those.
//! * When the group has a small subgroup — which it does whenever `p−1` has small factors, i.e.
//!   almost always — a peer value of small order leaks the private exponent modulo that order,
//!   and enough of them recover it (van Oorschot–Wiener). The defence is to know the subgroup
//!   order `q` and check `y^q ≡ 1 (mod p)`. [`DhParameters`] carries `q` when the document
//!   supplies it (`xenc:DHKeyValue/Q`) and the check then runs on every peer value. **Without
//!   `q` there is no such check to run**, which is a limitation of the transmitted parameters,
//!   not of this code; a static private exponent used with parameters that carry no `q` is
//!   exposed, and [`DhPrivateKey::generate`] documents what to do instead.
//!
//! # Timing
//!
//! [`BigUint::modpow`] is square-and-multiply with a data-dependent branch on the exponent bits,
//! so the private exponent leaks to anyone who can time this. As with the crate's RSA private
//! operations, that is a documented property of [`oxixml_support::bigint`] and makes this
//! suitable for processing documents, not for serving a co-located attacker.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::dh::{DhParameters, DhPrivateKey};
//! use oxixml_enc::random::PseudoRandom;
//!
//! // A 512-bit safe prime, small enough to be quick; a real deployment uses 2048 bits or more.
//! let prime = oxixml_text::hex::decode(concat!(
//!     "FEF3B78C842B8406334F4AAF28C8077743921557B93BC3E50E1F1A1688562AC0",
//!     "242F191536AC32BBE8C3E858D86D6DDCA8B383C6DDCA0BAFC57355BBA8C035B7",
//! ))?;
//! let parameters = DhParameters::new(&prime, &[0x02], None)?;
//!
//! let mut rng = PseudoRandom::new(2026);
//! let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
//! let recipient = DhPrivateKey::generate(&parameters, &mut rng)?;
//!
//! let from_originator = originator.shared_secret(&recipient.public_key()?)?;
//! let from_recipient = recipient.shared_secret(&originator.public_key()?)?;
//! assert_eq!(from_originator, from_recipient);
//! assert_eq!(from_originator.len(), 64, "as long as the prime, leading zeros kept");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_support::bigint::BigUint;

use crate::error::{EncError, Result};
use crate::random::RandomSource;

/// The narrowest prime modulus this module accepts, in bits.
///
/// This matches the floor [`RsaPublicKey::new`](crate::crypto::rsa::RsaPublicKey::new) puts on an
/// RSA modulus, and for the same reason: anything below it is not a key, it is a typo. It is a
/// floor for *reading old documents*, not a recommendation — NIST SP 800-56A has required 2048
/// bits since 2019, and a new deployment that can choose should be using `ECDH-ES` instead.
pub const MIN_MODULUS_BITS: usize = 512;

/// The widest prime modulus this module accepts, in bits.
///
/// Modular multiplication needs room for the intermediate product, so the modulus cannot exceed
/// half of [`BigUint::MAX_BITS`]. Refusing a wider one here means an oversized modulus costs a
/// comparison rather than a multi-second exponentiation.
pub const MAX_MODULUS_BITS: usize = BigUint::MAX_BITS / 2;

/// How many candidates [`DhPrivateKey::generate`] will draw before giving up.
const MAX_DRAWS: usize = 64;

/// The domain parameters of a finite-field Diffie–Hellman group: `p`, `g` and, when it is known,
/// the subgroup order `q`.
///
/// These are what `xenc:DHKeyValue` transmits. Supplying `q` is optional in the schema and
/// strongly advisable in practice; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DhParameters {
    prime: BigUint,
    generator: BigUint,
    subgroup_order: Option<BigUint>,
    /// The prime's length in octets, which is also the length of every public value and of the
    /// shared secret.
    prime_len: usize,
}

impl DhParameters {
    /// Builds domain parameters from the big-endian octet strings of `p`, `g` and optionally `q`.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the prime is even or outside
    /// `MIN_MODULUS_BITS..=MAX_MODULUS_BITS`, when the generator is not in `2..p-1`, or when a
    /// subgroup order is supplied that is even, below three, or not smaller than the prime.
    pub fn new(prime: &[u8], generator: &[u8], subgroup_order: Option<&[u8]>) -> Result<Self> {
        let p = BigUint::from_be_bytes(prime);
        let bits = p.bit_len();
        if bits < MIN_MODULUS_BITS {
            return Err(EncError::InvalidAgreementKey(
                "the Diffie-Hellman prime is implausibly short",
            ));
        }
        if bits > MAX_MODULUS_BITS {
            return Err(EncError::InvalidAgreementKey(
                "the Diffie-Hellman prime is implausibly large",
            ));
        }
        if !p.bit(0) {
            return Err(EncError::InvalidAgreementKey(
                "the Diffie-Hellman prime is even",
            ));
        }

        let g = BigUint::from_be_bytes(generator);
        let upper = p
            .checked_sub(&BigUint::one())
            .ok_or(EncError::InvalidAgreementKey("the prime is zero"))?;
        if g < BigUint::from_u64(2) || g >= upper {
            return Err(EncError::InvalidAgreementKey(
                "the Diffie-Hellman generator is not in 2..p-1",
            ));
        }

        let q = match subgroup_order {
            None => None,
            Some(bytes) => {
                let q = BigUint::from_be_bytes(bytes);
                if q < BigUint::from_u64(3) {
                    return Err(EncError::InvalidAgreementKey(
                        "the Diffie-Hellman subgroup order is too small",
                    ));
                }
                if !q.bit(0) {
                    return Err(EncError::InvalidAgreementKey(
                        "the Diffie-Hellman subgroup order is even",
                    ));
                }
                if q >= p {
                    return Err(EncError::InvalidAgreementKey(
                        "the Diffie-Hellman subgroup order is not smaller than the prime",
                    ));
                }
                Some(q)
            }
        };

        let prime_len = p.to_be_bytes().len();
        Ok(Self {
            prime: p,
            generator: g,
            subgroup_order: q,
            prime_len,
        })
    }

    /// The prime modulus `p`.
    #[must_use]
    pub const fn prime(&self) -> &BigUint {
        &self.prime
    }

    /// The generator `g`.
    #[must_use]
    pub const fn generator(&self) -> &BigUint {
        &self.generator
    }

    /// The subgroup order `q`, when the parameters carry one.
    #[must_use]
    pub const fn subgroup_order(&self) -> Option<&BigUint> {
        self.subgroup_order.as_ref()
    }

    /// The length in octets of the prime, of every public value, and of the shared secret.
    #[must_use]
    pub const fn prime_len(&self) -> usize {
        self.prime_len
    }

    /// Encodes `value` at the fixed width XMLEnc 1.0 §5.5.2 requires: as long as `p`, leading
    /// zero octets kept.
    fn fixed_width(&self, value: &BigUint) -> Result<Vec<u8>> {
        value
            .to_be_bytes_padded(self.prime_len)
            .ok_or(EncError::InvalidAgreementKey(
                "a Diffie-Hellman value is wider than the prime",
            ))
    }
}

/// A peer's Diffie–Hellman public value `y`, checked against the domain parameters it came with.
///
/// A value of this type has already passed every check the parameters make possible: it is in
/// `2..p-1`, and when the parameters carry a subgroup order it lies in that subgroup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DhPublicKey {
    parameters: DhParameters,
    value: BigUint,
}

impl DhPublicKey {
    /// Validates a peer public value and pairs it with its parameters.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when `y` is outside `2..p-1` — which covers 0,
    /// 1, `p−1` and anything at or above `p` — or when a subgroup order is known and
    /// `y^q ≢ 1 (mod p)`.
    pub fn new(parameters: &DhParameters, value: &[u8]) -> Result<Self> {
        let y = BigUint::from_be_bytes(value);
        let upper = parameters
            .prime
            .checked_sub(&BigUint::one())
            .ok_or(EncError::InvalidAgreementKey("the prime is zero"))?;
        if y < BigUint::from_u64(2) || y >= upper {
            return Err(EncError::InvalidAgreementKey(
                "the peer public value is not in 2..p-1",
            ));
        }
        if let Some(order) = &parameters.subgroup_order {
            let residue =
                y.modpow(order, &parameters.prime)
                    .ok_or(EncError::InvalidAgreementKey(
                        "modular exponentiation failed",
                    ))?;
            if !residue.is_one() {
                return Err(EncError::InvalidAgreementKey(
                    "the peer public value is not in the subgroup of order q",
                ));
            }
        }
        Ok(Self {
            parameters: parameters.clone(),
            value: y,
        })
    }

    /// The domain parameters this value belongs to.
    #[must_use]
    pub const fn parameters(&self) -> &DhParameters {
        &self.parameters
    }

    /// Encodes the public value as an octet string as long as the prime.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] only if the value is wider than the prime,
    /// which this type's invariant rules out.
    pub fn to_be_bytes(&self) -> Result<Vec<u8>> {
        self.parameters.fixed_width(&self.value)
    }
}

/// A Diffie–Hellman private exponent `x`, with the parameters it belongs to.
///
/// The exponent never appears in the [`Debug`] output. It is not wiped on drop: it lives inside
/// a [`BigUint`], whose limbs this crate cannot reach.
#[derive(Clone, PartialEq, Eq)]
pub struct DhPrivateKey {
    parameters: DhParameters,
    value: BigUint,
}

impl core::fmt::Debug for DhPrivateKey {
    /// Prints the modulus width and nothing else.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DhPrivateKey")
            .field("modulus_bits", &self.parameters.prime.bit_len())
            .finish_non_exhaustive()
    }
}

impl DhPrivateKey {
    /// Reads a private exponent from its big-endian octet string.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the exponent is below two or is not less
    /// than its upper bound — `q−1` when a subgroup order is known, `p−1` otherwise.
    pub fn from_be_bytes(parameters: &DhParameters, value: &[u8]) -> Result<Self> {
        let x = BigUint::from_be_bytes(value);
        let upper = Self::upper_bound(parameters)?;
        if x < BigUint::from_u64(2) || x > upper {
            return Err(EncError::InvalidAgreementKey(
                "the Diffie-Hellman private exponent is out of range",
            ));
        }
        Ok(Self {
            parameters: parameters.clone(),
            value: x,
        })
    }

    /// Draws a fresh private exponent from `rng`.
    ///
    /// The exponent is drawn uniformly from `2..=q-2` when the parameters carry a subgroup order
    /// and from `2..=p-2` when they do not, by rejection sampling — which is unbiased, unlike
    /// reducing a wider draw.
    ///
    /// Prefer using the result **once**, as `dh-es` does. A static exponent reused with
    /// parameters that carry no `q` cannot be defended against a peer that sends small-order
    /// public values; see the [module documentation](self).
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Entropy`] when the random source fails and
    /// [`EncError::InvalidAgreementKey`] when it never lands in range.
    pub fn generate(parameters: &DhParameters, rng: &mut dyn RandomSource) -> Result<Self> {
        let upper = Self::upper_bound(parameters)?;
        let two = BigUint::from_u64(2);
        let width = upper.to_be_bytes().len();
        let mut bytes = vec![0u8; width];
        for _ in 0..MAX_DRAWS {
            rng.fill(&mut bytes)?;
            let candidate = BigUint::from_be_bytes(&bytes);
            if candidate >= two && candidate <= upper {
                bytes.fill(0);
                return Ok(Self {
                    parameters: parameters.clone(),
                    value: candidate,
                });
            }
        }
        bytes.fill(0);
        Err(EncError::InvalidAgreementKey(
            "the random source did not produce a private exponent in range",
        ))
    }

    /// The largest private exponent these parameters admit: `q−2` when a subgroup order is
    /// known, `p−2` otherwise.
    fn upper_bound(parameters: &DhParameters) -> Result<BigUint> {
        let ceiling = parameters
            .subgroup_order
            .as_ref()
            .unwrap_or(&parameters.prime);
        ceiling
            .checked_sub(&BigUint::from_u64(2))
            .ok_or(EncError::InvalidAgreementKey(
                "the Diffie-Hellman group is too small to hold a private exponent",
            ))
    }

    /// The domain parameters this exponent belongs to.
    #[must_use]
    pub const fn parameters(&self) -> &DhParameters {
        &self.parameters
    }

    /// Returns the matching public value, `g^x mod p`.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the exponentiation fails or when the
    /// result is degenerate — which for valid parameters cannot happen, and for parameters whose
    /// generator is not what it claims to be must not be published.
    pub fn public_key(&self) -> Result<DhPublicKey> {
        let value = self
            .parameters
            .generator
            .modpow(&self.value, &self.parameters.prime)
            .ok_or(EncError::InvalidAgreementKey(
                "modular exponentiation failed",
            ))?;
        let encoded = self.parameters.fixed_width(&value)?;
        DhPublicKey::new(&self.parameters, &encoded)
    }

    /// Computes the shared secret `ZZ = y^x mod p`, encoded at the width of `p` with its leading
    /// zero octets kept, as XMLEnc 1.0 §5.5.2 requires.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the peer's value belongs to different
    /// domain parameters, when the exponentiation fails, or when the result is 0, 1 or `p−1` —
    /// the degenerate values that carry no secret at all.
    pub fn shared_secret(&self, peer: &DhPublicKey) -> Result<Vec<u8>> {
        if self.parameters != peer.parameters {
            return Err(EncError::InvalidAgreementKey(
                "the private exponent and the peer public value use different domain parameters",
            ));
        }
        let shared = peer
            .value
            .modpow(&self.value, &self.parameters.prime)
            .ok_or(EncError::InvalidAgreementKey(
                "modular exponentiation failed",
            ))?;

        let upper = self
            .parameters
            .prime
            .checked_sub(&BigUint::one())
            .ok_or(EncError::InvalidAgreementKey("the prime is zero"))?;
        if shared.is_zero() || shared.is_one() || shared == upper {
            return Err(EncError::InvalidAgreementKey(
                "the key agreement produced a degenerate shared secret",
            ));
        }
        self.parameters.fixed_width(&shared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{FixedRandom, PseudoRandom};

    /// A 512-bit **safe** prime: `p = 2·q + 1` with `q` prime too, so `q` is the order of the
    /// squares modulo `p` and the subgroup check has something real to check. `p ≡ 7 (mod 8)`,
    /// which makes 2 a square and therefore a generator of that subgroup.
    const TEST_PRIME: &str = concat!(
        "FEF3B78C842B8406334F4AAF28C8077743921557B93BC3E50E1F1A1688562AC0",
        "242F191536AC32BBE8C3E858D86D6DDCA8B383C6DDCA0BAFC57355BBA8C035B7",
    );

    /// `q = (p − 1) / 2` for [`TEST_PRIME`].
    const TEST_SUBGROUP_ORDER: &str = concat!(
        "7F79DBC64215C20319A7A557946403BBA1C90AABDC9DE1F2870F8D0B442B1560",
        "12178C8A9B56195DF461F42C6C36B6EE5459C1E36EE505D7E2B9AADDD4601ADB",
    );

    fn hex(text: &str) -> Result<Vec<u8>> {
        oxixml_text::hex::decode(text)
            .map_err(|_| EncError::InvalidAgreementKey("bad test parameter"))
    }

    /// The test group, with the subgroup order supplied only when the test needs it: supplying
    /// it makes every public value cost an extra exponentiation.
    fn safe_prime_parameters(with_order: bool) -> Result<DhParameters> {
        let prime = hex(TEST_PRIME)?;
        let order = if with_order {
            Some(hex(TEST_SUBGROUP_ORDER)?)
        } else {
            None
        };
        DhParameters::new(&prime, &[0x02], order.as_deref())
    }

    #[test]
    fn the_subgroup_check_runs_when_the_order_is_known() -> Result<()> {
        let parameters = safe_prime_parameters(true)?;
        assert!(parameters.subgroup_order().is_some());

        // Two generates the subgroup of order q, so every power of it is a member.
        let mine = DhPrivateKey::from_be_bytes(&parameters, &[0x03])?;
        let public = mine.public_key()?;
        assert_eq!(public.to_be_bytes()?.len(), parameters.prime_len());

        // Seventeen is a quadratic non-residue modulo this prime, so it has order 2q and lies
        // outside the subgroup. Without the check it would be accepted; with it, it is not.
        assert!(matches!(
            DhPublicKey::new(&parameters, &[0x11]),
            Err(EncError::InvalidAgreementKey(_))
        ));
        // The same value is accepted when the parameters carry no order to check against, which
        // is exactly the exposure the module documentation warns about.
        assert!(DhPublicKey::new(&safe_prime_parameters(false)?, &[0x11]).is_ok());
        Ok(())
    }

    #[test]
    fn a_generated_pair_agrees_with_itself() -> Result<()> {
        let parameters = safe_prime_parameters(false)?;
        let mut rng = PseudoRandom::new(21);
        let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
        let recipient = DhPrivateKey::generate(&parameters, &mut rng)?;

        let from_originator = originator.shared_secret(&recipient.public_key()?)?;
        let from_recipient = recipient.shared_secret(&originator.public_key()?)?;
        assert_eq!(from_originator, from_recipient);
        assert_eq!(from_originator.len(), parameters.prime_len());

        let stranger = DhPrivateKey::generate(&parameters, &mut rng)?;
        assert_ne!(
            stranger.shared_secret(&recipient.public_key()?)?,
            from_originator
        );
        Ok(())
    }

    #[test]
    fn degenerate_peer_values_are_refused() -> Result<()> {
        let parameters = safe_prime_parameters(false)?;
        let prime_bytes = parameters.prime().to_be_bytes();
        let one_less = parameters
            .prime()
            .checked_sub(&BigUint::one())
            .ok_or(EncError::InvalidAgreementKey("the prime is not zero"))?
            .to_be_bytes();
        for bad in [
            vec![0x00],
            vec![0x01],
            one_less,
            prime_bytes.clone(),
            Vec::new(),
        ] {
            assert!(
                matches!(
                    DhPublicKey::new(&parameters, &bad),
                    Err(EncError::InvalidAgreementKey(_))
                ),
                "accepted {bad:02x?}"
            );
        }
        Ok(())
    }

    #[test]
    fn malformed_parameters_are_refused() -> Result<()> {
        let good = safe_prime_parameters(false)?.prime().to_be_bytes();
        // Too short, even, and a generator outside 2..p-1.
        assert!(matches!(
            DhParameters::new(&[0xffu8; 32], &[0x02], None),
            Err(EncError::InvalidAgreementKey(_))
        ));
        let mut even = good.clone();
        if let Some(last) = even.last_mut() {
            *last &= 0xfe;
        }
        assert!(matches!(
            DhParameters::new(&even, &[0x02], None),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(matches!(
            DhParameters::new(&good, &[0x01], None),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(matches!(
            DhParameters::new(&good, &good, None),
            Err(EncError::InvalidAgreementKey(_))
        ));
        // An oversized prime is refused before any exponentiation runs.
        let mut huge = vec![0xffu8; MAX_MODULUS_BITS / 8 + 1];
        if let Some(last) = huge.last_mut() {
            *last |= 1;
        }
        assert!(matches!(
            DhParameters::new(&huge, &[0x02], None),
            Err(EncError::InvalidAgreementKey(_))
        ));
        Ok(())
    }

    #[test]
    fn parameters_from_different_groups_never_agree() -> Result<()> {
        let first = safe_prime_parameters(false)?;
        let second = DhParameters::new(&first.prime().to_be_bytes(), &[0x03], None)?;
        let mut rng = PseudoRandom::new(22);
        let mine = DhPrivateKey::generate(&first, &mut rng)?;
        let theirs = DhPrivateKey::generate(&second, &mut rng)?;
        assert!(matches!(
            mine.shared_secret(&theirs.public_key()?),
            Err(EncError::InvalidAgreementKey(_))
        ));
        Ok(())
    }

    #[test]
    fn an_out_of_range_private_exponent_is_refused() -> Result<()> {
        let parameters = safe_prime_parameters(false)?;
        assert!(matches!(
            DhPrivateKey::from_be_bytes(&parameters, &[0x01]),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(matches!(
            DhPrivateKey::from_be_bytes(&parameters, &parameters.prime().to_be_bytes()),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(DhPrivateKey::from_be_bytes(&parameters, &[0x02]).is_ok());
        Ok(())
    }

    #[test]
    fn a_random_source_that_never_lands_in_range_is_reported() -> Result<()> {
        let parameters = safe_prime_parameters(false)?;
        let mut zeros = FixedRandom::new(&[0x00]);
        assert!(matches!(
            DhPrivateKey::generate(&parameters, &mut zeros),
            Err(EncError::InvalidAgreementKey(_))
        ));
        Ok(())
    }

    #[test]
    fn debug_does_not_leak_the_private_exponent() -> Result<()> {
        let parameters = safe_prime_parameters(false)?;
        let key = DhPrivateKey::from_be_bytes(&parameters, &[0x12, 0x34, 0x56])?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("512"), "{rendered}");
        assert!(!rendered.contains("1193046"), "{rendered}");
        assert!(!rendered.contains("0x12"), "{rendered}");
        Ok(())
    }
}
