//! RSA key material.
//!
//! XML Signature can hand you an RSA key in four different spellings, and all
//! four end up here:
//!
//! * `<ds:RSAKeyValue>` with base64 `<ds:Modulus>` and `<ds:Exponent>` —
//!   [XML Signature §4.5.2.2](https://www.w3.org/TR/xmldsig-core1/#sec-RSAKeyValue);
//! * `<dsig11:DEREncodedKeyValue>`, a DER `SubjectPublicKeyInfo` —
//!   [§4.5.9](https://www.w3.org/TR/xmldsig-core1/#sec-DEREncodedKeyValue);
//! * `<ds:X509Certificate>`, from which only the `subjectPublicKey` is taken;
//! * PKCS#1 / PKCS#8 DER, for a private key handed to [`crate::sign`].
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::RsaPublicKey;
//!
//! // The modulus and exponent exactly as they appear in a `<ds:RSAKeyValue>`.
//! let modulus = oxixml_text::base64::base64_decode(
//!     "gIb6nAB9oS/AI5jIj6WymvQhRxiMlE07G4abmMliYi5zWzvaFE2tnU+RZIBgtoXcgDEIU/vsLQut\
//!      7nzCn9mHxC8JEaV4D4U91j64AyZakShqJw7qjJfqUxxPL0yJv2oFiouPDjGuJ9JPi0NrsZq+yfWf\
//!      M54s4b9SNkcOIVMybZU=",
//! )?;
//! let key = RsaPublicKey::from_components(&modulus, &[0x01, 0x00, 0x01])?;
//! assert_eq!(key.modulus_bits(), 1024);
//! assert_eq!(key.exponent(), [0x01, 0x00, 0x01]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_support::bigint::BigUint;

use crate::crypto::der;
use crate::error::{DsigError, DsigResult};

/// An RSA public key: the modulus `n` and the public exponent `e`.
///
/// Constructing one validates that `n` is odd, at least 512 bits and at most
/// [`MAXIMUM_RSA_MODULUS_BITS`], and that `e` is odd, at least 3 and at most
/// [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`] — the cheap structural checks that keep
/// a degenerate "key" (`n = 1`, `e = 1`) from making every signature verify,
/// and that keep an oversized one from turning [`modpow`](BigUint::modpow)
/// into a denial of service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsaPublicKey {
    modulus: Vec<u8>,
    exponent: Vec<u8>,
    n: BigUint,
    e: BigUint,
}

/// The smallest modulus this crate will accept, in bits.
///
/// 512 bits is already far below anything safe; it is the floor at which the
/// arithmetic still makes sense, not a recommendation. Callers that care should
/// check [`RsaPublicKey::modulus_bits`] themselves.
pub const MINIMUM_RSA_MODULUS_BITS: usize = 512;

/// The largest modulus this crate will accept, in bits.
///
/// `<ds:RSAKeyValue>` carries the modulus as attacker-supplied base64 text with
/// no length limit of its own, and [`BigUint::modpow`]'s cost grows with the
/// square of the modulus width it is asked to work with — so a verifier that
/// accepted an arbitrarily large modulus could be made to spend minutes of CPU
/// on a single signature from a document a few hundred kilobytes long. This
/// mirrors [`BigUint::MAX_BITS`], the ceiling [`BigUint::modpow`] and
/// [`BigUint::div_rem`] already enforce internally, so that an oversized key is
/// refused here — with a message that says why — rather than surfacing as an
/// opaque failure deep inside the arithmetic.
///
/// The public exponent has its own, much narrower ceiling — see
/// [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`] for why the modulus bound alone is not
/// enough to close this off.
pub const MAXIMUM_RSA_MODULUS_BITS: usize = BigUint::MAX_BITS;

/// The largest public exponent this crate will accept, in bits.
///
/// [FIPS 186-5 Appendix A.1.1](https://doi.org/10.6028/NIST.FIPS.186-5) requires
/// every RSA public exponent it certifies to satisfy `65537 ≤ e < 2^256`; no
/// interoperable key uses one wider than a handful of bits (`65537` itself is
/// 17 bits). The bound is deliberately far tighter than
/// [`MAXIMUM_RSA_MODULUS_BITS`] because [`BigUint::modpow`]'s cost is *linear
/// in the exponent's bit length* regardless of the modulus: a modulus and
/// exponent that both merely sat at the modulus ceiling would still cost on
/// the order of a minute of CPU per verification, which is not a fix for the
/// denial of service this exists to prevent. Capping the exponent here keeps
/// an oversized one rejected in microseconds while still leaving orders of
/// magnitude of headroom over anything a real key ever uses.
pub const MAXIMUM_RSA_PUBLIC_EXPONENT_BITS: usize = 256;

impl RsaPublicKey {
    /// Builds a key from big-endian `modulus` and `exponent` octets, exactly as
    /// `<ds:Modulus>` and `<ds:Exponent>` carry them after base64 decoding.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the modulus is even, outside
    /// [`MINIMUM_RSA_MODULUS_BITS`]..=[`MAXIMUM_RSA_MODULUS_BITS`], or if the
    /// exponent is even, less than 3, or wider than
    /// [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`].
    pub fn from_components(modulus: &[u8], exponent: &[u8]) -> DsigResult<Self> {
        let n = BigUint::from_be_bytes(modulus);
        let e = BigUint::from_be_bytes(exponent);
        let bits = n.bit_len();
        if bits < MINIMUM_RSA_MODULUS_BITS {
            return Err(DsigError::Key(format!(
                "RSA modulus is {bits} bits, below the {MINIMUM_RSA_MODULUS_BITS}-bit floor"
            )));
        }
        if bits > MAXIMUM_RSA_MODULUS_BITS {
            return Err(DsigError::Key(format!(
                "RSA modulus is {bits} bits, above the {MAXIMUM_RSA_MODULUS_BITS}-bit ceiling \
                 this crate accepts"
            )));
        }
        if !is_odd(modulus) {
            return Err(DsigError::Key("RSA modulus is even".to_owned()));
        }
        let exponent_bits = e.bit_len();
        if exponent_bits > MAXIMUM_RSA_PUBLIC_EXPONENT_BITS {
            return Err(DsigError::Key(format!(
                "RSA public exponent is {exponent_bits} bits, above the \
                 {MAXIMUM_RSA_PUBLIC_EXPONENT_BITS}-bit ceiling this crate accepts"
            )));
        }
        if !is_odd(exponent) || e < BigUint::from_u64(3) {
            return Err(DsigError::Key(
                "RSA public exponent must be an odd integer of at least 3".to_owned(),
            ));
        }
        Ok(Self {
            modulus: n.to_be_bytes(),
            exponent: e.to_be_bytes(),
            n,
            e,
        })
    }

    /// Parses a DER `SubjectPublicKeyInfo`, the payload of a
    /// `<dsig11:DEREncodedKeyValue>`.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the key is not RSA.
    pub fn from_subject_public_key_info(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_subject_public_key_info(input)
            .map_err(|error| DsigError::Key(format!("SubjectPublicKeyInfo: {error}")))?;
        Self::from_components(&components.modulus, &components.exponent)
    }

    /// Parses a bare PKCS#1 `RSAPublicKey`.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed.
    pub fn from_pkcs1(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_rsa_public_key(input)
            .map_err(|error| DsigError::Key(format!("RSAPublicKey: {error}")))?;
        Self::from_components(&components.modulus, &components.exponent)
    }

    /// Extracts the `subjectPublicKey` of a DER X.509 certificate — the payload
    /// of a `<ds:X509Certificate>`.
    ///
    /// **Nothing about the certificate is validated *here*.** Its signature is
    /// not checked, its validity dates are ignored, no chain is built: the
    /// certificate is a container, and deciding whether to trust the key inside
    /// it is the [`KeySelector`](crate::KeySelector)'s job. To have
    /// that decision made properly, parse the certificate with
    /// [`Certificate`](crate::Certificate) and validate a certification path
    /// through it with [`PathValidator`](crate::PathValidator) — which is what
    /// [`TrustedCertificates`](crate::TrustedCertificates) does when it is given
    /// one.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the certified key is not
    /// RSA.
    pub fn from_x509_certificate(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_certificate_public_key(input)
            .map_err(|error| DsigError::Key(format!("X509Certificate: {error}")))?;
        Self::from_components(&components.modulus, &components.exponent)
    }

    /// The modulus `n`, big-endian, without leading zero octets.
    #[must_use]
    pub fn modulus(&self) -> &[u8] {
        &self.modulus
    }

    /// The public exponent `e`, big-endian, without leading zero octets.
    #[must_use]
    pub fn exponent(&self) -> &[u8] {
        &self.exponent
    }

    /// The size of the modulus in bits.
    #[must_use]
    pub fn modulus_bits(&self) -> usize {
        self.n.bit_len()
    }

    pub(crate) const fn n(&self) -> &BigUint {
        &self.n
    }

    pub(crate) const fn e(&self) -> &BigUint {
        &self.e
    }
}

/// An RSA private key: the modulus `n` and the private exponent `d`.
///
/// The CRT parameters are parsed but not kept: this crate performs the plain
/// `m^d mod n` exponentiation, which is slower but has no CRT fault-injection
/// surface and needs no blinding to be constant-time-ish for the modest key
/// sizes XML Signature uses.
#[derive(Clone, PartialEq, Eq)]
pub struct RsaPrivateKey {
    public: RsaPublicKey,
    private_exponent: BigUint,
}

impl RsaPrivateKey {
    /// Builds a private key from big-endian octets.
    ///
    /// `public_exponent` is kept only so that
    /// [`public_key`](RsaPrivateKey::public_key) can hand back a usable
    /// verification key; the signing operation never touches it.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the modulus or public exponent fails the checks of
    /// [`RsaPublicKey::from_components`], or if `d` is zero or not less than
    /// `n`.
    pub fn from_components(
        modulus: &[u8],
        public_exponent: &[u8],
        private_exponent: &[u8],
    ) -> DsigResult<Self> {
        let public = RsaPublicKey::from_components(modulus, public_exponent)?;
        let d = BigUint::from_be_bytes(private_exponent);
        if d.is_zero() || d >= *public.n() {
            return Err(DsigError::Key(
                "RSA private exponent must lie in (0, n)".to_owned(),
            ));
        }
        Ok(Self {
            public,
            private_exponent: d,
        })
    }

    /// Parses a PKCS#1 `RSAPrivateKey` or a PKCS#8 `PrivateKeyInfo` wrapping
    /// one, both in DER.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the key is not RSA.
    pub fn from_der(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_rsa_private_key(input)
            .map_err(|error| DsigError::Key(format!("RSAPrivateKey: {error}")))?;
        Self::from_components(
            &components.modulus,
            &components.public_exponent,
            &components.private_exponent,
        )
    }

    /// The matching public key.
    #[must_use]
    pub fn public_key(&self) -> RsaPublicKey {
        self.public.clone()
    }

    /// The size of the modulus in bits.
    #[must_use]
    pub fn modulus_bits(&self) -> usize {
        self.public.modulus_bits()
    }

    pub(crate) const fn n(&self) -> &BigUint {
        self.public.n()
    }

    pub(crate) const fn d(&self) -> &BigUint {
        &self.private_exponent
    }
}

impl std::fmt::Debug for RsaPrivateKey {
    /// Prints the modulus size and nothing else — a private exponent must never
    /// reach a log file through a stray `{:?}`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RsaPrivateKey")
            .field("modulus_bits", &self.modulus_bits())
            .field("private_exponent", &"<redacted>")
            .finish()
    }
}

/// Whether a big-endian magnitude is odd.
fn is_odd(bytes: &[u8]) -> bool {
    bytes.last().is_some_and(|byte| byte % 2 == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;

    #[test]
    fn every_encoding_yields_the_same_public_key() -> DsigResult<()> {
        let from_components = RsaPublicKey::from_components(
            &testkey::bytes(testkey::MODULUS_BASE64),
            &testkey::bytes(testkey::PUBLIC_EXPONENT_BASE64),
        )?;
        assert_eq!(
            RsaPublicKey::from_subject_public_key_info(&testkey::bytes(testkey::SPKI_BASE64))?,
            from_components
        );
        assert_eq!(
            RsaPublicKey::from_pkcs1(&testkey::bytes(testkey::RSA_PUBLIC_KEY_BASE64))?,
            from_components
        );
        assert_eq!(
            RsaPublicKey::from_x509_certificate(&testkey::bytes(testkey::CERTIFICATE_BASE64))?,
            from_components
        );
        assert_eq!(from_components.modulus_bits(), 1024);
        Ok(())
    }

    #[test]
    fn private_keys_round_trip_to_their_public_half() -> DsigResult<()> {
        let key = RsaPrivateKey::from_der(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))?;
        assert_eq!(
            key.public_key().modulus(),
            testkey::bytes(testkey::MODULUS_BASE64)
        );
        assert_eq!(key.modulus_bits(), 1024);
        Ok(())
    }

    #[test]
    fn a_degenerate_key_is_refused() {
        assert!(RsaPublicKey::from_components(&[0x03], &[0x03]).is_err());
        let modulus = testkey::bytes(testkey::MODULUS_BASE64);
        assert!(RsaPublicKey::from_components(&modulus, &[0x01]).is_err());
        assert!(RsaPublicKey::from_components(&modulus, &[0x02]).is_err());
    }

    #[test]
    fn an_even_modulus_is_refused() {
        let mut modulus = testkey::bytes(testkey::MODULUS_BASE64);
        if let Some(byte) = modulus.last_mut() {
            *byte &= 0xfe;
        }
        assert!(RsaPublicKey::from_components(&modulus, &[0x01, 0x00, 0x01]).is_err());
    }

    #[test]
    fn the_private_exponent_never_reaches_the_debug_output() -> DsigResult<()> {
        let key = RsaPrivateKey::from_der(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("redacted"), "{rendered}");
        assert!(rendered.contains("1024"), "{rendered}");
        // The private exponent's most significant limb must not be anywhere in it.
        assert!(!rendered.contains("2503"), "{rendered}");
        Ok(())
    }

    /// A `<ds:RSAKeyValue>` is attacker-supplied text with no length limit of its
    /// own, and `BigUint::modpow`'s cost grows with the square of the operand
    /// width. An oversized modulus must be refused here, in microseconds, rather
    /// than reaching `modpow` and costing seconds to minutes of CPU per
    /// verification — see [`MAXIMUM_RSA_MODULUS_BITS`].
    #[test]
    fn an_oversized_modulus_is_refused_before_any_arithmetic_runs() {
        // One bit past the ceiling: (MAXIMUM_RSA_MODULUS_BITS + 1) bits, odd, and
        // otherwise shaped like a legitimate modulus.
        let mut modulus = vec![0x01u8; MAXIMUM_RSA_MODULUS_BITS / 8 + 1];
        if let Some(last) = modulus.last_mut() {
            *last |= 1;
        }
        let error = RsaPublicKey::from_components(&modulus, &[0x01, 0x00, 0x01])
            .expect_err("a modulus this wide must be rejected");
        let message = error.to_string();
        assert!(message.contains("above"), "{message}");
        assert!(
            message.contains(&MAXIMUM_RSA_MODULUS_BITS.to_string()),
            "{message}"
        );
    }

    /// The exponent bound is the load-bearing one: it is cheaper for an attacker
    /// to grow (a small modulus keeps the `SignatureValue` small) and
    /// `modpow`'s cost is linear in the exponent's bit length regardless of the
    /// modulus, so a modulus-only ceiling would leave this fully open.
    #[test]
    fn an_oversized_exponent_is_refused_before_any_arithmetic_runs() {
        let modulus = testkey::bytes(testkey::MODULUS_BASE64);
        let mut exponent = vec![0x01u8; MAXIMUM_RSA_PUBLIC_EXPONENT_BITS / 8 + 1];
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        let error = RsaPublicKey::from_components(&modulus, &exponent)
            .expect_err("an exponent this wide must be rejected");
        let message = error.to_string();
        assert!(message.contains("exponent"), "{message}");
        assert!(message.contains("above"), "{message}");
    }

    /// An exponent between [`MAXIMUM_RSA_PUBLIC_EXPONENT_BITS`] and
    /// [`MAXIMUM_RSA_MODULUS_BITS`] used to be accepted when the two shared one
    /// ceiling — and `modpow`'s cost is linear in the exponent's bit length
    /// regardless of the modulus, so a modulus and exponent both that wide cost
    /// on the order of a minute of CPU per verification. It must be refused now.
    #[test]
    fn an_exponent_within_the_old_shared_ceiling_is_still_refused() {
        let modulus = testkey::bytes(testkey::MODULUS_BASE64);
        let mut exponent = vec![0x01u8; MAXIMUM_RSA_MODULUS_BITS / 8];
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        assert!(exponent.len() * 8 > MAXIMUM_RSA_PUBLIC_EXPONENT_BITS);
        let error = RsaPublicKey::from_components(&modulus, &exponent)
            .expect_err("an exponent this wide must still be rejected");
        assert!(error.to_string().contains("exponent"), "{error}");
    }

    #[test]
    fn a_modulus_exactly_at_the_ceiling_is_still_accepted() {
        // MAXIMUM_RSA_MODULUS_BITS bits exactly: the top bit set, everything else
        // zero except the low bit, which keeps it odd.
        let mut modulus = vec![0u8; MAXIMUM_RSA_MODULUS_BITS / 8];
        if let Some(first) = modulus.first_mut() {
            *first = 0x80;
        }
        if let Some(last) = modulus.last_mut() {
            *last |= 1;
        }
        let key =
            RsaPublicKey::from_components(&modulus, &[0x01, 0x00, 0x01]).expect("at the ceiling");
        assert_eq!(key.modulus_bits(), MAXIMUM_RSA_MODULUS_BITS);
    }

    #[test]
    fn an_exponent_exactly_at_its_own_ceiling_is_still_accepted() {
        let modulus = testkey::bytes(testkey::MODULUS_BASE64);
        // MAXIMUM_RSA_PUBLIC_EXPONENT_BITS bits exactly, odd.
        let mut exponent = vec![0u8; MAXIMUM_RSA_PUBLIC_EXPONENT_BITS / 8];
        if let Some(first) = exponent.first_mut() {
            *first = 0x80;
        }
        if let Some(last) = exponent.last_mut() {
            *last |= 1;
        }
        let key = RsaPublicKey::from_components(&modulus, &exponent).expect("at the ceiling");
        assert_eq!(key.exponent().len() * 8, MAXIMUM_RSA_PUBLIC_EXPONENT_BITS);
    }
}
