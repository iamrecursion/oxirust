//! DSA key material.
//!
//! A DSA key is four integers: the domain parameters `p`, `q` and `g`, and the
//! public value `y = g^x mod p`. XML Signature carries them in a
//! `<ds:DSAKeyValue>` —
//! [§4.5.2.1](https://www.w3.org/TR/xmldsig-core1/#sec-DSAKeyValue) — where `P`
//! and `Q` are an optional pair, `G` is optional on its own, and only `Y` is
//! required, on the theory that a verifier already knows the parameters. This
//! crate needs all four to compute anything, so a `<ds:DSAKeyValue>` that omits
//! the parameters is a key hint it cannot act on rather than a key.
//!
//! # Validation
//!
//! Constructing a key runs the cheap structural checks that catch a garbled or
//! nonsensical parameter set:
//!
//! * `p` is odd and between [`MINIMUM_DSA_PRIME_BITS`] and
//!   [`MAXIMUM_DSA_PRIME_BITS`] — the upper bound mirrors
//!   [`BigUint::MAX_BITS`], because every operation here is a modular
//!   exponentiation whose cost grows with the square of the operand width and
//!   `p` arrives as attacker-supplied base64 with no length of its own;
//! * `q` is between [`MINIMUM_DSA_SUBGROUP_BITS`] and
//!   [`MAXIMUM_DSA_SUBGROUP_BITS`] and smaller than `p`;
//! * `q` divides `p - 1`, so that an order-`q` subgroup exists at all;
//! * `g` lies in `2..p` and `y` in `1..p`.
//!
//! What is deliberately *not* done is the full partial validation of
//! [SP 800-89 §5.3.1](https://doi.org/10.6028/NIST.SP.800-89) — checking
//! `g^q mod p = 1` and `y^q mod p = 1`. Each is a modular exponentiation as
//! expensive as the verification itself, and it protects only against a *trusted*
//! source handing over a small-subgroup parameter set: a caller who takes the
//! whole key out of the document being verified has already lost, whatever this
//! module checks. Callers who obtain parameters from an untrusted place and
//! reuse them across many signatures should validate them once, themselves.

use oxixml_support::bigint::BigUint;

use crate::crypto::{der, dsa};
use crate::error::{DsigError, DsigResult};

/// The smallest prime modulus this crate will accept, in bits.
///
/// 512 bits is the floor the original DSS allowed and is far below anything
/// safe; it is here so that historical documents still parse, not as a
/// recommendation. Callers that care should check
/// [`DsaPublicKey::prime_bits`] themselves.
pub const MINIMUM_DSA_PRIME_BITS: usize = 512;

/// The largest prime modulus this crate will accept, in bits.
///
/// Mirrors [`BigUint::MAX_BITS`], the ceiling the arithmetic already enforces,
/// so that an oversized parameter set is refused here — with a message that says
/// why — rather than surfacing as an opaque failure deep inside a modular
/// exponentiation.
pub const MAXIMUM_DSA_PRIME_BITS: usize = BigUint::MAX_BITS;

/// The smallest subgroup order this crate will accept, in bits.
pub const MINIMUM_DSA_SUBGROUP_BITS: usize = 160;

/// The largest subgroup order this crate will accept, in bits.
///
/// FIPS 186-4 defines `N` as 160, 224 or 256; the ceiling is generous rather
/// than exact so that it bounds the work without second-guessing a future
/// parameter size.
pub const MAXIMUM_DSA_SUBGROUP_BITS: usize = 512;

/// A DSA public key: the domain parameters and the public value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaPublicKey {
    p: BigUint,
    q: BigUint,
    g: BigUint,
    y: BigUint,
}

impl DsaPublicKey {
    /// Builds a key from big-endian octets, exactly as `<ds:P>`, `<ds:Q>`,
    /// `<ds:G>` and `<ds:Y>` carry them after base64 decoding.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] when any of the checks in the module documentation
    /// fails, with a message naming the one that did.
    pub fn from_components(p: &[u8], q: &[u8], g: &[u8], y: &[u8]) -> DsigResult<Self> {
        let p = BigUint::from_be_bytes(p);
        let q = BigUint::from_be_bytes(q);
        let g = BigUint::from_be_bytes(g);
        let y = BigUint::from_be_bytes(y);

        let prime_bits = p.bit_len();
        if prime_bits < MINIMUM_DSA_PRIME_BITS {
            return Err(DsigError::Key(format!(
                "DSA prime is {prime_bits} bits, below the {MINIMUM_DSA_PRIME_BITS}-bit floor"
            )));
        }
        if prime_bits > MAXIMUM_DSA_PRIME_BITS {
            return Err(DsigError::Key(format!(
                "DSA prime is {prime_bits} bits, above the {MAXIMUM_DSA_PRIME_BITS}-bit ceiling \
                 this crate accepts"
            )));
        }
        let subgroup_bits = q.bit_len();
        if !(MINIMUM_DSA_SUBGROUP_BITS..=MAXIMUM_DSA_SUBGROUP_BITS).contains(&subgroup_bits) {
            return Err(DsigError::Key(format!(
                "DSA subgroup order is {subgroup_bits} bits, outside the \
                 {MINIMUM_DSA_SUBGROUP_BITS}..={MAXIMUM_DSA_SUBGROUP_BITS} bits this crate accepts"
            )));
        }
        if q >= p {
            return Err(DsigError::Key(
                "the DSA subgroup order is not smaller than the prime".to_owned(),
            ));
        }
        let below_prime = p
            .checked_sub(&BigUint::one())
            .ok_or_else(|| DsigError::Key("the DSA prime is zero".to_owned()))?;
        let divides = below_prime
            .div_rem(&q)
            .ok_or_else(|| DsigError::Key("the DSA parameters are too wide".to_owned()))?
            .1;
        if !divides.is_zero() {
            return Err(DsigError::Key(
                "the DSA subgroup order does not divide p - 1, so no subgroup of that order exists"
                    .to_owned(),
            ));
        }
        if g < BigUint::from_u64(2) || g >= p {
            return Err(DsigError::Key(
                "the DSA generator must lie in 2..p".to_owned(),
            ));
        }
        if y.is_zero() || y >= p {
            return Err(DsigError::Key(
                "the DSA public value must lie in 1..p".to_owned(),
            ));
        }
        Ok(Self { p, q, g, y })
    }

    /// Parses a DER `SubjectPublicKeyInfo` holding a DSA key.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the key is not DSA.
    pub fn from_subject_public_key_info(input: &[u8]) -> DsigResult<Self> {
        match der::parse_public_key_info(input)
            .map_err(|error| DsigError::Key(format!("SubjectPublicKeyInfo: {error}")))?
        {
            der::PublicKeyComponents::Dsa(components) => {
                Self::from_components(&components.p, &components.q, &components.g, &components.y)
            }
            _ => Err(DsigError::Key(
                "SubjectPublicKeyInfo: the key is not a DSA key".to_owned(),
            )),
        }
    }

    /// Extracts the `subjectPublicKey` of a DER X.509 certificate.
    ///
    /// **Nothing about the certificate is validated** — see
    /// [`RsaPublicKey::from_x509_certificate`](crate::RsaPublicKey::from_x509_certificate).
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the certified key is not
    /// DSA.
    pub fn from_x509_certificate(input: &[u8]) -> DsigResult<Self> {
        match der::parse_certificate_key(input)
            .map_err(|error| DsigError::Key(format!("X509Certificate: {error}")))?
        {
            der::PublicKeyComponents::Dsa(components) => {
                Self::from_components(&components.p, &components.q, &components.g, &components.y)
            }
            _ => Err(DsigError::Key(
                "X509Certificate: the certified key is not a DSA key".to_owned(),
            )),
        }
    }

    /// The prime modulus `p`, big-endian, without leading zero octets.
    #[must_use]
    pub fn prime(&self) -> Vec<u8> {
        self.p.to_be_bytes()
    }

    /// The subgroup order `q`, big-endian, without leading zero octets.
    #[must_use]
    pub fn subgroup_order(&self) -> Vec<u8> {
        self.q.to_be_bytes()
    }

    /// The generator `g`, big-endian, without leading zero octets.
    #[must_use]
    pub fn generator(&self) -> Vec<u8> {
        self.g.to_be_bytes()
    }

    /// The public value `y`, big-endian, without leading zero octets.
    #[must_use]
    pub fn public_value(&self) -> Vec<u8> {
        self.y.to_be_bytes()
    }

    /// The size of the prime modulus in bits — FIPS 186-4's `L`.
    #[must_use]
    pub fn prime_bits(&self) -> usize {
        self.p.bit_len()
    }

    /// The size of the subgroup order in bits — FIPS 186-4's `N`.
    #[must_use]
    pub fn subgroup_bits(&self) -> usize {
        self.q.bit_len()
    }

    /// Whether `q` fits in the fixed half-width a signature value uses, which is
    /// what makes a key usable with a given `SignatureMethod`.
    #[must_use]
    pub fn fits_signature_width(&self, half_width: usize) -> bool {
        self.subgroup_bits().div_ceil(8) <= half_width
    }

    pub(crate) fn parameters(&self) -> dsa::PublicParameters<'_> {
        dsa::PublicParameters {
            p: &self.p,
            q: &self.q,
            g: &self.g,
            y: &self.y,
        }
    }
}

/// A DSA private key: the domain parameters and the private value `x`.
#[derive(Clone, PartialEq, Eq)]
pub struct DsaPrivateKey {
    public: DsaPublicKey,
    x: BigUint,
}

impl DsaPrivateKey {
    /// Builds a private key from big-endian octets.
    ///
    /// The public value is recomputed as `g^x mod p` rather than taken as an
    /// argument, so a key built this way cannot sign under a public value that
    /// does not match it.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the parameters fail the checks of
    /// [`DsaPublicKey::from_components`], or if `x` is not in `1..q`.
    pub fn from_components(p: &[u8], q: &[u8], g: &[u8], x: &[u8]) -> DsigResult<Self> {
        let x = BigUint::from_be_bytes(x);
        let prime = BigUint::from_be_bytes(p);
        let order = BigUint::from_be_bytes(q);
        let generator = BigUint::from_be_bytes(g);
        if x.is_zero() || x >= order {
            return Err(DsigError::Key(
                "the DSA private value must lie in 1..q".to_owned(),
            ));
        }
        let y = generator.modpow(&x, &prime).ok_or_else(|| {
            DsigError::Key("the DSA parameters are too wide for this crate".to_owned())
        })?;
        let public = DsaPublicKey::from_components(p, q, g, &y.to_be_bytes())?;
        Ok(Self { public, x })
    }

    /// Parses a PKCS#8 `PrivateKeyInfo` holding a DSA key, in DER.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the key is not DSA.
    pub fn from_der(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_dsa_private_key(input)
            .map_err(|error| DsigError::Key(format!("DSA PrivateKeyInfo: {error}")))?;
        Self::from_components(&components.p, &components.q, &components.g, &components.x)
    }

    /// The matching public key.
    #[must_use]
    pub fn public_key(&self) -> DsaPublicKey {
        self.public.clone()
    }

    /// The size of the prime modulus in bits.
    #[must_use]
    pub fn prime_bits(&self) -> usize {
        self.public.prime_bits()
    }

    /// Whether `q` fits in the fixed half-width a signature value uses.
    #[must_use]
    pub fn fits_signature_width(&self, half_width: usize) -> bool {
        self.public.fits_signature_width(half_width)
    }

    pub(crate) const fn p(&self) -> &BigUint {
        &self.public.p
    }

    pub(crate) const fn q(&self) -> &BigUint {
        &self.public.q
    }

    pub(crate) const fn g(&self) -> &BigUint {
        &self.public.g
    }

    pub(crate) const fn x(&self) -> &BigUint {
        &self.x
    }
}

impl std::fmt::Debug for DsaPrivateKey {
    /// Prints the parameter sizes and nothing else.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DsaPrivateKey")
            .field("prime_bits", &self.prime_bits())
            .field("private_value", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::hex;

    /// The FIPS 186-2 Appendix 5 example key, which XML Signature §6.4.1 uses as
    /// its own worked example.
    const EXAMPLE_P: &str = "\
        8DF2A494492276AA3D25759BB06869CBEAC0D83AFB8D0CF7CBB8324F0D7882E5\
        D0762FC5B7210EAFC2E9ADAC32AB7AAC49693DFBF83724C2EC0736EE31C80291";
    const EXAMPLE_Q: &str = "C773218C737EC8EE993B4F2DED30F48EDACE915F";
    const EXAMPLE_G: &str = "\
        626D027839EA0A13413163A55B4CB500299D5522956CEFCB3BFF10F399CE2C2E\
        71CB9DE5FA24BABF58E5B79521925C9CC42E9F6F464B088CC572AF53E6D78802";
    const EXAMPLE_X: &str = "2070B3223DBA372FDE1C0FFC7B2E3B498B260614";
    const EXAMPLE_Y: &str = "\
        19131871D75B1612A819F29D78D1B0D7346F7AA77BB62A859BFD6C5675DA9D21\
        2D3A36EF1672EF660B8C7C255CC0EC74858FBA33F44C06699630A76B030EE333";

    fn example_key() -> DsigResult<DsaPublicKey> {
        DsaPublicKey::from_components(
            &hex(EXAMPLE_P),
            &hex(EXAMPLE_Q),
            &hex(EXAMPLE_G),
            &hex(EXAMPLE_Y),
        )
    }

    #[test]
    fn the_published_example_key_is_accepted() -> DsigResult<()> {
        let key = example_key()?;
        // The original DSS example predates FIPS 186-3 and uses L = 512, which
        // is exactly the floor this crate accepts.
        assert_eq!(key.prime_bits(), MINIMUM_DSA_PRIME_BITS);
        assert_eq!(key.subgroup_bits(), 160);
        assert_eq!(key.prime(), hex(EXAMPLE_P));
        assert_eq!(key.public_value(), hex(EXAMPLE_Y));
        assert!(key.fits_signature_width(20));
        assert!(key.fits_signature_width(32));
        Ok(())
    }

    #[test]
    fn a_private_key_generates_its_own_public_value() -> DsigResult<()> {
        let key = DsaPrivateKey::from_components(
            &hex(EXAMPLE_P),
            &hex(EXAMPLE_Q),
            &hex(EXAMPLE_G),
            &hex(EXAMPLE_X),
        )?;
        assert_eq!(key.public_key(), example_key()?);
        Ok(())
    }

    #[test]
    fn the_private_value_never_reaches_the_debug_output() -> DsigResult<()> {
        let key = DsaPrivateKey::from_components(
            &hex(EXAMPLE_P),
            &hex(EXAMPLE_Q),
            &hex(EXAMPLE_G),
            &hex(EXAMPLE_X),
        )?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("redacted"), "{rendered}");
        assert!(!rendered.contains("2070"), "{rendered}");
        Ok(())
    }

    /// `q` must divide `p - 1`; without that there is no subgroup of order `q`
    /// and every verification would be meaningless arithmetic.
    #[test]
    fn a_subgroup_order_that_does_not_divide_p_minus_one_is_refused() {
        let mut q = hex(EXAMPLE_Q);
        if let Some(last) = q.last_mut() {
            *last ^= 0x02;
        }
        let error =
            DsaPublicKey::from_components(&hex(EXAMPLE_P), &q, &hex(EXAMPLE_G), &hex(EXAMPLE_Y))
                .expect_err("q no longer divides p - 1");
        assert!(error.to_string().contains("divide"), "{error}");
    }

    #[test]
    fn degenerate_parameters_are_refused() {
        let p = hex(EXAMPLE_P);
        let q = hex(EXAMPLE_Q);
        let g = hex(EXAMPLE_G);
        let y = hex(EXAMPLE_Y);
        // g = 1 generates nothing.
        assert!(DsaPublicKey::from_components(&p, &q, &[0x01], &y).is_err());
        // y = 0 makes every signature verify to zero.
        assert!(DsaPublicKey::from_components(&p, &q, &g, &[0x00]).is_err());
        // y = p is out of range.
        assert!(DsaPublicKey::from_components(&p, &q, &g, &p).is_err());
        // A prime far below the floor.
        assert!(DsaPublicKey::from_components(&[0x0b], &q, &g, &y).is_err());
    }

    /// The prime arrives as attacker-supplied base64 with no length of its own,
    /// and every operation on it is a modular exponentiation. An oversized one
    /// must be refused in microseconds rather than reaching the arithmetic.
    #[test]
    fn an_oversized_prime_is_refused_before_any_arithmetic_runs() {
        let mut p = vec![0x01_u8; MAXIMUM_DSA_PRIME_BITS / 8 + 1];
        if let Some(last) = p.last_mut() {
            *last |= 1;
        }
        let error = DsaPublicKey::from_components(&p, &hex(EXAMPLE_Q), &hex(EXAMPLE_G), &[0x02])
            .expect_err("a prime this wide must be rejected");
        let message = error.to_string();
        assert!(message.contains("above"), "{message}");
        assert!(
            message.contains(&MAXIMUM_DSA_PRIME_BITS.to_string()),
            "{message}"
        );
    }

    #[test]
    fn a_private_value_outside_the_subgroup_is_refused() {
        let p = hex(EXAMPLE_P);
        let q = hex(EXAMPLE_Q);
        let g = hex(EXAMPLE_G);
        assert!(DsaPrivateKey::from_components(&p, &q, &g, &[0x00]).is_err());
        assert!(DsaPrivateKey::from_components(&p, &q, &g, &q).is_err());
    }
}
