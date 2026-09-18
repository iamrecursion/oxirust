//! Elliptic-curve Diffie–Hellman over the NIST prime curves, which XML Encryption 1.1 §5.6.4
//! names `ECDH-ES`.
//!
//! The primitive is the one NIST SP 800-56A §5.7.1.2 calls ECC CDH: given a private scalar `d`
//! and the peer's public point `Q`, the shared secret `Z` is the **x coordinate** of `d·Q`,
//! encoded big-endian at the curve's field width. Both supported curves have
//! [cofactor](oxixml_support::ec::COFACTOR) one, so the "cofactor multiplication" of the
//! standard's general form is the identity and is omitted.
//!
//! # Invalid-curve attacks
//!
//! The peer's public key arrives inside the document, which means an attacker chooses it. If a
//! decryptor multiplies its private scalar by a point that is *not* on the curve, the arithmetic
//! silently moves to a different curve — one the attacker picked to have a smooth group order —
//! and a few hundred decryption attempts recover the private key one small factor at a time
//! (Biehl–Meyer–Müller, CRYPTO 2000). Every entry point here therefore validates the peer point
//! completely — canonical coordinates, on the curve, not the point at infinity — before the
//! private scalar goes anywhere near it, and rejects a shared point that comes out as the point
//! at infinity.
//!
//! An [`AffinePoint`] does not record which curve it belongs to,
//! so [`EcPublicKey`] and [`EcPrivateKey`] carry the [`CurveId`] themselves and every operation
//! checks that the two agree. Mixing a P-384 scalar with a P-256 point would otherwise compute a
//! meaningless answer rather than fail.
//!
//! # Timing
//!
//! [`Curve::scalar_mul`](oxixml_support::ec::Curve::scalar_mul) is a Montgomery ladder, so the
//! *sequence* of point operations does not depend on the scalar, but the underlying big-integer
//! arithmetic is variable-time. This is sound for the ephemeral-static agreement `ECDH-ES`
//! specifies, where the originator's scalar is used once and discarded; a long-lived recipient
//! scalar is a secret that this arithmetic does not defend against a local attacker who can
//! measure it. That limitation belongs to [`oxixml_support::bigint`] and is stated, not hidden.
//!
//! # Example
//!
//! ```
//! use oxixml_enc::crypto::ecdh::{EcPrivateKey, EcPublicKey};
//! use oxixml_enc::random::PseudoRandom;
//! use oxixml_support::ec::CurveId;
//!
//! let mut rng = PseudoRandom::new(2026);
//! let recipient = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
//! let published = recipient.public_key().to_sec1()?;
//!
//! // The originator makes a fresh key pair, agrees with the published one, and sends its own
//! // public point along with the ciphertext.
//! let originator = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
//! let peer = EcPublicKey::from_sec1(CurveId::P256, &published)?;
//! let sender_secret = originator.shared_secret(&peer)?;
//!
//! // The recipient reaches the same secret from the other side.
//! let recipient_secret =
//!     recipient.shared_secret(&EcPublicKey::from_sec1(CurveId::P256, &originator.public_key().to_sec1()?)?)?;
//! assert_eq!(sender_secret, recipient_secret);
//! assert_eq!(sender_secret.len(), 32);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use oxixml_support::bigint::BigUint;
use oxixml_support::ec::{AffinePoint, CurveId};

use crate::error::{EncError, Result};
use crate::random::RandomSource;

/// The `NamedCurve` URI XMLEnc 1.1 §5.6.3 uses for NIST P-256, `urn:oid:1.2.840.10045.3.1.7`.
pub const P256_NAMED_CURVE_URI: &str = "urn:oid:1.2.840.10045.3.1.7";

/// The `NamedCurve` URI XMLEnc 1.1 §5.6.3 uses for NIST P-384, `urn:oid:1.3.132.0.34`.
pub const P384_NAMED_CURVE_URI: &str = "urn:oid:1.3.132.0.34";

/// The `NamedCurve` URI for NIST P-521, `urn:oid:1.3.132.0.35`.
///
/// P-521 is named so that a caller can recognize it and refuse it explicitly. This crate
/// implements only the two curves [`oxixml_support::ec`] provides, so
/// [`curve_from_named_curve_uri`] returns `None` for it and every operation fails closed.
pub const P521_NAMED_CURVE_URI: &str = "urn:oid:1.3.132.0.35";

/// Returns the curve named by an `xenc11:NamedCurve` URI, or `None` when it is one this crate
/// does not implement.
#[must_use]
pub fn curve_from_named_curve_uri(uri: &str) -> Option<CurveId> {
    match uri {
        P256_NAMED_CURVE_URI => Some(CurveId::P256),
        P384_NAMED_CURVE_URI => Some(CurveId::P384),
        _ => None,
    }
}

/// Returns the `xenc11:NamedCurve` URI for `curve`.
#[must_use]
pub const fn named_curve_uri(curve: CurveId) -> &'static str {
    match curve {
        CurveId::P256 => P256_NAMED_CURVE_URI,
        CurveId::P384 => P384_NAMED_CURVE_URI,
    }
}

/// A peer's elliptic-curve public key: a curve, and a validated point on it.
///
/// A value of this type is always a usable public key. There is no way to build one that is off
/// the curve, has a non-canonical coordinate, or is the point at infinity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcPublicKey {
    curve: CurveId,
    point: AffinePoint,
}

impl EcPublicKey {
    /// Decodes an uncompressed SEC 1 point (`04 ‖ X ‖ Y`), validating it completely.
    ///
    /// This is the form an `ECKeyValue`'s `PublicKey` child carries, base64-decoded — see
    /// [`crate::model::EcKeyValue`].
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] for a point that is not an uncompressed
    /// encoding of the right length, whose coordinates are not canonical field elements, or that
    /// does not satisfy the curve equation.
    pub fn from_sec1(curve: CurveId, encoded: &[u8]) -> Result<Self> {
        let point = curve.curve().decode_point(encoded)?;
        // `decode_point` rejects the single-octet encoding of the point at infinity, so this
        // cannot fire today; it is kept so that the invariant of this type stays true if a
        // future encoding could produce one.
        if point.is_identity() {
            return Err(EncError::InvalidAgreementKey(
                "the point at infinity is not a valid public key",
            ));
        }
        Ok(Self { curve, point })
    }

    /// Returns the curve this key belongs to.
    #[must_use]
    pub const fn curve(&self) -> CurveId {
        self.curve
    }

    /// Encodes the key as an uncompressed SEC 1 point.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] only if the point could not be encoded, which
    /// the type's invariant rules out.
    pub fn to_sec1(&self) -> Result<Vec<u8>> {
        Ok(self.curve.curve().encode_point(&self.point)?)
    }
}

/// An elliptic-curve private key: a curve, and a scalar in `1..n`.
///
/// # What is and is not protected
///
/// The scalar never appears in the [`Debug`] output. It is *not* wiped on drop: it lives inside a
/// [`BigUint`], whose limbs this crate cannot reach. Treat an `EcPrivateKey` as an ephemeral
/// value — which is exactly what `ECDH-ES` asks the originator for — rather than as a
/// long-lived secret this type will protect for you.
#[derive(Clone, PartialEq, Eq)]
pub struct EcPrivateKey {
    curve: CurveId,
    scalar: BigUint,
}

impl core::fmt::Debug for EcPrivateKey {
    /// Prints the curve and nothing else.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EcPrivateKey")
            .field("curve", &self.curve)
            .finish_non_exhaustive()
    }
}

/// How many candidate scalars [`EcPrivateKey::generate`] will draw before giving up.
///
/// A draw fails only when the octets land outside `1..n`, which for both supported curves has
/// probability below 2⁻³²; a hundred consecutive failures means the random source is broken, not
/// that the caller was unlucky.
const MAX_SCALAR_ATTEMPTS: usize = 100;

impl EcPrivateKey {
    /// Reads a private key from its fixed-width big-endian octet string.
    ///
    /// The width must be exactly the curve's scalar width — 32 octets for P-256, 48 for P-384 —
    /// and the value must lie in `1..n`. A short encoding is refused rather than zero-extended,
    /// because a caller that produced one has a bug that would otherwise surface as a silently
    /// different key.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] for the wrong width, for zero, and for a value
    /// that is not less than the group order.
    pub fn from_be_bytes(curve: CurveId, bytes: &[u8]) -> Result<Self> {
        let scalar = curve.curve().scalar_from_be_bytes(bytes)?;
        Ok(Self { curve, scalar })
    }

    /// Draws a fresh private key from `rng`.
    ///
    /// Uses the rejection ("discard") method of FIPS 186-5 Appendix A.4.2: draw the scalar's
    /// exact width and keep it only if it already lies in `1..n`. That is unbiased, unlike
    /// reducing a draw modulo the order.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::Entropy`] when the random source fails, and
    /// [`EncError::InvalidAgreementKey`] when a hundred consecutive draws all landed outside the
    /// range, which means the source is not producing what it claims to.
    pub fn generate(curve: CurveId, rng: &mut dyn RandomSource) -> Result<Self> {
        let mut bytes = vec![0u8; curve.field_bytes()];
        for _ in 0..MAX_SCALAR_ATTEMPTS {
            rng.fill(&mut bytes)?;
            if let Ok(scalar) = curve.curve().scalar_from_be_bytes(&bytes) {
                bytes.fill(0);
                return Ok(Self { curve, scalar });
            }
        }
        bytes.fill(0);
        Err(EncError::InvalidAgreementKey(
            "the random source did not produce a scalar in range",
        ))
    }

    /// Returns the curve this key belongs to.
    #[must_use]
    pub const fn curve(&self) -> CurveId {
        self.curve
    }

    /// Returns the matching public key, `d·G`.
    #[must_use]
    pub fn public_key(&self) -> EcPublicKey {
        EcPublicKey {
            curve: self.curve,
            point: self.curve.curve().mul_base(&self.scalar),
        }
    }

    /// Computes the ECDH shared secret `Z`: the x coordinate of `d·Q`, at the curve's field
    /// width.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::InvalidAgreementKey`] when the two keys are on different curves, when
    /// the peer's point fails re-validation, or when the product is the point at infinity — the
    /// degenerate case in which the agreement has produced no secret at all and must be
    /// abandoned rather than papered over.
    pub fn shared_secret(&self, peer: &EcPublicKey) -> Result<Vec<u8>> {
        if self.curve != peer.curve {
            return Err(EncError::InvalidAgreementKey(
                "the private key and the peer public key are on different curves",
            ));
        }
        let curve = self.curve.curve();

        // The peer point was validated when it was decoded, and this type cannot hold an invalid
        // one — but the check is repeated here because it is the *only* thing standing between a
        // hostile document and an invalid-curve attack, and it costs one field operation.
        if peer.point.is_identity() || !curve.is_on_curve(&peer.point) {
            return Err(EncError::InvalidAgreementKey(
                "the peer public key is not a point on the curve",
            ));
        }

        let shared = curve.scalar_mul(&peer.point, &self.scalar);
        if shared.is_identity() {
            return Err(EncError::InvalidAgreementKey(
                "the key agreement produced the point at infinity",
            ));
        }
        Ok(curve.encode_x_coordinate(&shared)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{FixedRandom, PseudoRandom};
    use oxixml_support::ec::Curve;

    fn hex(text: &str) -> Vec<u8> {
        oxixml_text::hex::decode(text).expect("hexadecimal")
    }

    /// Builds the uncompressed SEC 1 encoding of a point from its two coordinates.
    fn point(x: &str, y: &str) -> Vec<u8> {
        let mut encoded = vec![0x04];
        encoded.extend_from_slice(&hex(x));
        encoded.extend_from_slice(&hex(y));
        encoded
    }

    /// The 256-bit test vector of [RFC 5903 §8.1](https://www.rfc-editor.org/rfc/rfc5903#section-8.1).
    #[test]
    fn the_rfc_5903_p256_vector_is_reproduced() -> Result<()> {
        let initiator = EcPrivateKey::from_be_bytes(
            CurveId::P256,
            &hex("C88F01F510D9AC3F70A292DAA2316DE544E9AAB8AFE84049C62A9C57862D1433"),
        )?;
        let initiator_public = point(
            "DAD0B65394221CF9B051E1FECA5787D098DFE637FC90B9EF945D0C3772581180",
            "5271A0461CDB8252D61F1C456FA3E59AB1F45B33ACCF5F58389E0577B8990BB3",
        );
        let responder = EcPrivateKey::from_be_bytes(
            CurveId::P256,
            &hex("C6EF9C5D78AE012A011164ACB397CE2088685D8F06BF9BE0B283AB46476BEE53"),
        )?;
        let responder_public = point(
            "D12DFB5289C8D4F81208B70270398C342296970A0BCCB74C736FC7554494BF63",
            "56FBF3CA366CC23E8157854C13C58D6AAC23F046ADA30F8353E74F33039872AB",
        );

        // Each private key produces the published public key.
        assert_eq!(initiator.public_key().to_sec1()?, initiator_public);
        assert_eq!(responder.public_key().to_sec1()?, responder_public);

        // And both sides reach the published shared secret.
        let expected = hex("D6840F6B42F6EDAFD13116E0E12565202FEF8E9ECE7DCE03812464D04B9442DE");
        assert_eq!(
            initiator.shared_secret(&EcPublicKey::from_sec1(CurveId::P256, &responder_public)?)?,
            expected
        );
        assert_eq!(
            responder.shared_secret(&EcPublicKey::from_sec1(CurveId::P256, &initiator_public)?)?,
            expected
        );
        Ok(())
    }

    /// The 384-bit test vector of [RFC 5903 §8.2](https://www.rfc-editor.org/rfc/rfc5903#section-8.2).
    #[test]
    fn the_rfc_5903_p384_vector_is_reproduced() -> Result<()> {
        let initiator = EcPrivateKey::from_be_bytes(
            CurveId::P384,
            &hex(concat!(
                "099F3C7034D4A2C699884D73A375A67F7624EF7C6B3C0F160647B67414DCE655",
                "E35B538041E649EE3FAEF896783AB194",
            )),
        )?;
        let initiator_public = point(
            concat!(
                "667842D7D180AC2CDE6F74F37551F55755C7645C20EF73E31634FE72B4C55EE6",
                "DE3AC808ACB4BDB4C88732AEE95F41AA",
            ),
            concat!(
                "9482ED1FC0EEB9CAFC4984625CCFC23F65032149E0E144ADA024181535A0F38E",
                "EB9FCFF3C2C947DAE69B4C634573A81C",
            ),
        );
        let responder = EcPrivateKey::from_be_bytes(
            CurveId::P384,
            &hex(concat!(
                "41CB0779B4BDB85D47846725FBEC3C9430FAB46CC8DC5060855CC9BDA0AA2942",
                "E0308312916B8ED2960E4BD55A7448FC",
            )),
        )?;
        let responder_public = point(
            concat!(
                "E558DBEF53EECDE3D3FCCFC1AEA08A89A987475D12FD950D83CFA41732BC509D",
                "0D1AC43A0336DEF96FDA41D0774A3571",
            ),
            concat!(
                "DCFBEC7AACF3196472169E838430367F66EEBE3C6E70C416DD5F0C68759DD1FF",
                "F83FA40142209DFF5EAAD96DB9E6386C",
            ),
        );

        assert_eq!(initiator.public_key().to_sec1()?, initiator_public);
        assert_eq!(responder.public_key().to_sec1()?, responder_public);

        let expected = hex(concat!(
            "11187331C279962D93D604243FD592CB9D0A926F422E47187521287E7156C5C4",
            "D603135569B9E9D09CF5D4A270F59746",
        ));
        assert_eq!(
            initiator.shared_secret(&EcPublicKey::from_sec1(CurveId::P384, &responder_public)?)?,
            expected
        );
        assert_eq!(
            responder.shared_secret(&EcPublicKey::from_sec1(CurveId::P384, &initiator_public)?)?,
            expected
        );
        Ok(())
    }

    #[test]
    fn a_generated_pair_agrees_with_itself_on_both_curves() -> Result<()> {
        for curve in [CurveId::P256, CurveId::P384] {
            let mut rng = PseudoRandom::new(7);
            let originator = EcPrivateKey::generate(curve, &mut rng)?;
            let recipient = EcPrivateKey::generate(curve, &mut rng)?;
            assert_ne!(originator, recipient);
            assert_eq!(originator.curve(), curve);

            let from_originator = originator.shared_secret(&recipient.public_key())?;
            let from_recipient = recipient.shared_secret(&originator.public_key())?;
            assert_eq!(from_originator, from_recipient);
            assert_eq!(from_originator.len(), curve.field_bytes());

            // A third party agrees with neither of them.
            let stranger = EcPrivateKey::generate(curve, &mut rng)?;
            assert_ne!(
                stranger.shared_secret(&recipient.public_key())?,
                from_originator
            );
        }
        Ok(())
    }

    #[test]
    fn a_point_that_is_not_on_the_curve_is_refused() {
        // The RFC 5903 responder point with one coordinate bit flipped: still canonical field
        // elements, no longer a solution of the curve equation.
        let mut encoded = point(
            "D12DFB5289C8D4F81208B70270398C342296970A0BCCB74C736FC7554494BF63",
            "56FBF3CA366CC23E8157854C13C58D6AAC23F046ADA30F8353E74F33039872AB",
        );
        encoded[40] ^= 0x01;
        assert!(matches!(
            EcPublicKey::from_sec1(CurveId::P256, &encoded),
            Err(EncError::InvalidAgreementKey(_))
        ));
    }

    #[test]
    fn malformed_point_encodings_are_refused() {
        let valid = point(
            "D12DFB5289C8D4F81208B70270398C342296970A0BCCB74C736FC7554494BF63",
            "56FBF3CA366CC23E8157854C13C58D6AAC23F046ADA30F8353E74F33039872AB",
        );
        // A compressed point, the point at infinity, a truncated encoding and an empty one are
        // all refused rather than guessed at.
        let mut compressed = valid.clone();
        compressed[0] = 0x02;
        for bad in [compressed, vec![0x00], valid[..64].to_vec(), Vec::new()] {
            assert!(
                matches!(
                    EcPublicKey::from_sec1(CurveId::P256, &bad),
                    Err(EncError::InvalidAgreementKey(_))
                ),
                "P-256 accepted {bad:02x?}"
            );
        }
        // A perfectly good P-256 point is still not a P-384 point.
        assert!(matches!(
            EcPublicKey::from_sec1(CurveId::P384, &valid),
            Err(EncError::InvalidAgreementKey(_))
        ));
    }

    #[test]
    fn a_scalar_outside_the_group_order_is_refused() -> Result<()> {
        assert!(matches!(
            EcPrivateKey::from_be_bytes(CurveId::P256, &[0u8; 32]),
            Err(EncError::InvalidAgreementKey(_))
        ));
        let order = Curve::p256()
            .order()
            .to_be_bytes_padded(32)
            .ok_or(EncError::InvalidAgreementKey("the order is 256 bits"))?;
        assert!(matches!(
            EcPrivateKey::from_be_bytes(CurveId::P256, &order),
            Err(EncError::InvalidAgreementKey(_))
        ));
        // The wrong width is refused rather than zero-extended.
        assert!(matches!(
            EcPrivateKey::from_be_bytes(CurveId::P256, &[1u8; 31]),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(matches!(
            EcPrivateKey::from_be_bytes(CurveId::P384, &[1u8; 32]),
            Err(EncError::InvalidAgreementKey(_))
        ));
        Ok(())
    }

    #[test]
    fn keys_on_different_curves_never_agree() -> Result<()> {
        let mut rng = PseudoRandom::new(11);
        let p256 = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        let p384 = EcPrivateKey::generate(CurveId::P384, &mut rng)?;
        assert!(matches!(
            p256.shared_secret(&p384.public_key()),
            Err(EncError::InvalidAgreementKey(_))
        ));
        assert!(matches!(
            p384.shared_secret(&p256.public_key()),
            Err(EncError::InvalidAgreementKey(_))
        ));
        Ok(())
    }

    #[test]
    fn a_random_source_that_never_lands_in_range_is_reported() {
        // Every draw is the all-zero scalar, which is out of range on every curve.
        let mut zeros = FixedRandom::new(&[0x00]);
        assert!(matches!(
            EcPrivateKey::generate(CurveId::P256, &mut zeros),
            Err(EncError::InvalidAgreementKey(_))
        ));
    }

    #[test]
    fn named_curve_uris_round_trip_and_p521_is_refused() {
        for curve in [CurveId::P256, CurveId::P384] {
            assert_eq!(
                curve_from_named_curve_uri(named_curve_uri(curve)),
                Some(curve)
            );
        }
        assert_eq!(curve_from_named_curve_uri(P521_NAMED_CURVE_URI), None);
        assert_eq!(curve_from_named_curve_uri("urn:example:curve"), None);
    }

    #[test]
    fn debug_does_not_leak_the_private_scalar() -> Result<()> {
        let key = EcPrivateKey::from_be_bytes(
            CurveId::P256,
            &hex("C88F01F510D9AC3F70A292DAA2316DE544E9AAB8AFE84049C62A9C57862D1433"),
        )?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("P256"), "{rendered}");
        assert!(!rendered.contains("C88F"), "{rendered}");
        assert!(!rendered.to_lowercase().contains("c88f"), "{rendered}");
        Ok(())
    }
}
