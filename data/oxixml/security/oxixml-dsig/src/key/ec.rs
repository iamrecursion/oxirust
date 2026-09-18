//! Elliptic-curve key material.
//!
//! XML Signature 1.1 spells an EC public key three ways, and all three end up
//! here:
//!
//! * `<dsig11:ECKeyValue>` with a `<dsig11:NamedCurve URI="urn:oid:…"/>` and a
//!   base64 `<dsig11:PublicKey>` holding an uncompressed SEC 1 point —
//!   [§4.5.2.3](https://www.w3.org/TR/xmldsig-core1/#sec-ECKeyValue);
//! * `<dsig11:DEREncodedKeyValue>`, a DER `SubjectPublicKeyInfo` with
//!   `id-ecPublicKey` and a named curve;
//! * `<ds:X509Certificate>`, from which only the `subjectPublicKey` is taken.
//!
//! Only the two curves [XML Signature 1.1
//! §6.4.3](https://www.w3.org/TR/xmldsig-core1/#sec-ECDSA) makes REQUIRED are
//! implemented — P-256 and P-384 — and only the uncompressed point encoding,
//! which is the only one the Recommendation permits. A key on any other curve is
//! refused by name rather than silently mishandled.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{EcPublicKey, NamedCurve};
//!
//! // The `<dsig11:PublicKey>` of an `<dsig11:ECKeyValue>`, base64 decoded.
//! let point = oxixml_text::base64::base64_decode(
//!     "BJ/yaXNlq4FRObyJCBhb5jAz8GVzinK3bBGLjSDfjbJwNfydtgjnlS4EsDmxSRhWyJWq6GIqy5wvnaiA\
//!      RK04uB4=",
//! )?;
//! let key = EcPublicKey::from_sec1_point(NamedCurve::P256, &point)?;
//! assert_eq!(key.curve(), NamedCurve::P256);
//! assert_eq!(key.named_curve_uri(), "urn:oid:1.2.840.10045.3.1.7");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_support::bigint::BigUint;
use oxixml_support::ec::{AffinePoint, Curve, CurveId};

use crate::crypto::der;
use crate::error::{DsigError, DsigResult};

/// Which elliptic curve a key lives on.
///
/// This is [`oxixml_support::ec::CurveId`] re-exported under the name the
/// Recommendation uses for the `<dsig11:NamedCurve>` element, so that a caller
/// never has to name the support crate.
pub type NamedCurve = CurveId;

/// The `<dsig11:NamedCurve>` `URI` of NIST P-256 — the OID `1.2.840.10045.3.1.7`,
/// also known as `prime256v1` and `secp256r1`.
pub const P256_NAMED_CURVE_URI: &str = "urn:oid:1.2.840.10045.3.1.7";

/// The `<dsig11:NamedCurve>` `URI` of NIST P-384 — the OID `1.3.132.0.34`, also
/// known as `secp384r1`.
pub const P384_NAMED_CURVE_URI: &str = "urn:oid:1.3.132.0.34";

/// The curve a `<dsig11:NamedCurve>` `URI` names, or [`None`] for one this crate
/// does not implement.
///
/// The `urn:oid:` form is what
/// [XML Signature 1.1 §4.5.2.3.1](https://www.w3.org/TR/xmldsig-core1/#sec-ECParameters)
/// prescribes; the two OIDs are also accepted bare, because some producers write
/// them that way.
#[must_use]
pub fn named_curve_from_uri(uri: &str) -> Option<NamedCurve> {
    match uri {
        P256_NAMED_CURVE_URI | "1.2.840.10045.3.1.7" => Some(NamedCurve::P256),
        P384_NAMED_CURVE_URI | "1.3.132.0.34" => Some(NamedCurve::P384),
        _ => None,
    }
}

/// The `<dsig11:NamedCurve>` `URI` for a curve.
#[must_use]
pub const fn named_curve_uri(curve: NamedCurve) -> &'static str {
    match curve {
        NamedCurve::P256 => P256_NAMED_CURVE_URI,
        NamedCurve::P384 => P384_NAMED_CURVE_URI,
    }
}

/// An ECDSA public key: a curve and a point on it.
///
/// Constructing one validates that the point decodes as an uncompressed SEC 1
/// encoding of the curve's width, that both coordinates are canonical field
/// elements, that it satisfies the curve equation, and that it is not the point
/// at infinity. A point that fails any of those is not a key and is refused
/// here, so that nothing downstream has to wonder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcPublicKey {
    curve: NamedCurve,
    point: AffinePoint,
    encoded: Vec<u8>,
}

impl EcPublicKey {
    /// Builds a key from an uncompressed SEC 1 point, `0x04 ‖ X ‖ Y`.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the encoding is not the uncompressed form of the
    /// right width, or the point is not on the curve.
    pub fn from_sec1_point(curve: NamedCurve, point: &[u8]) -> DsigResult<Self> {
        let parameters = curve.curve();
        let decoded = parameters
            .decode_point(point)
            .map_err(|error| DsigError::Key(format!("{curve} public key: {error}")))?;
        Self::from_point(curve, decoded)
    }

    /// Builds a key from the two affine coordinates, big-endian.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the coordinates are not canonical field elements or
    /// the point is not on the curve.
    pub fn from_coordinates(curve: NamedCurve, x: &[u8], y: &[u8]) -> DsigResult<Self> {
        let parameters = curve.curve();
        let point = parameters
            .point_from_coordinates(BigUint::from_be_bytes(x), BigUint::from_be_bytes(y))
            .map_err(|error| DsigError::Key(format!("{curve} public key: {error}")))?;
        Self::from_point(curve, point)
    }

    /// Parses a DER `SubjectPublicKeyInfo`, the payload of a
    /// `<dsig11:DEREncodedKeyValue>`.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed, the key is not an
    /// elliptic-curve key, or the curve is not supported.
    pub fn from_subject_public_key_info(input: &[u8]) -> DsigResult<Self> {
        match der::parse_public_key_info(input)
            .map_err(|error| DsigError::Key(format!("SubjectPublicKeyInfo: {error}")))?
        {
            der::PublicKeyComponents::Ec(components) => {
                Self::from_sec1_point(components.curve, &components.point)
            }
            _ => Err(DsigError::Key(
                "SubjectPublicKeyInfo: the key is not an elliptic-curve key".to_owned(),
            )),
        }
    }

    /// Extracts the `subjectPublicKey` of a DER X.509 certificate.
    ///
    /// **Nothing about the certificate is validated** — see
    /// [`RsaPublicKey::from_x509_certificate`](crate::RsaPublicKey::from_x509_certificate)
    /// for what that does and does not mean.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed or the certified key is not an
    /// elliptic-curve key on a supported curve.
    pub fn from_x509_certificate(input: &[u8]) -> DsigResult<Self> {
        match der::parse_certificate_key(input)
            .map_err(|error| DsigError::Key(format!("X509Certificate: {error}")))?
        {
            der::PublicKeyComponents::Ec(components) => {
                Self::from_sec1_point(components.curve, &components.point)
            }
            _ => Err(DsigError::Key(
                "X509Certificate: the certified key is not an elliptic-curve key".to_owned(),
            )),
        }
    }

    /// The curve the key lives on.
    #[must_use]
    pub const fn curve(&self) -> NamedCurve {
        self.curve
    }

    /// The `<dsig11:NamedCurve>` `URI` of that curve.
    #[must_use]
    pub const fn named_curve_uri(&self) -> &'static str {
        named_curve_uri(self.curve)
    }

    /// The uncompressed SEC 1 encoding of the point, as
    /// `<dsig11:PublicKey>` carries it.
    #[must_use]
    pub fn encoded_point(&self) -> &[u8] {
        &self.encoded
    }

    /// The domain parameters of the key's curve.
    pub(crate) fn parameters(&self) -> &'static Curve {
        self.curve.curve()
    }

    pub(crate) const fn point(&self) -> &AffinePoint {
        &self.point
    }

    /// Re-encodes a validated point, which cannot be the identity here.
    fn from_point(curve: NamedCurve, point: AffinePoint) -> DsigResult<Self> {
        let encoded = curve
            .curve()
            .encode_point(&point)
            .map_err(|error| DsigError::Key(format!("{curve} public key: {error}")))?;
        Ok(Self {
            curve,
            point,
            encoded,
        })
    }
}

/// An ECDSA private key: a curve and a scalar in `1..n`.
#[derive(Clone, PartialEq, Eq)]
pub struct EcPrivateKey {
    public: EcPublicKey,
    scalar: BigUint,
}

impl EcPrivateKey {
    /// Builds a private key from the scalar, big-endian and exactly as wide as
    /// the curve's order.
    ///
    /// The public half is recomputed from the scalar rather than taken on trust,
    /// so a key built this way can never sign under a public key that does not
    /// match it.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the scalar is the wrong width or outside `1..n`.
    pub fn from_scalar(curve: NamedCurve, scalar: &[u8]) -> DsigResult<Self> {
        let parameters = curve.curve();
        let scalar = parameters
            .scalar_from_be_bytes(scalar)
            .map_err(|error| DsigError::Key(format!("{curve} private key: {error}")))?;
        let public = EcPublicKey::from_point(curve, parameters.mul_base(&scalar))?;
        Ok(Self { public, scalar })
    }

    /// Parses a SEC 1 `ECPrivateKey` or a PKCS#8 `PrivateKeyInfo` wrapping one,
    /// both in DER.
    ///
    /// # Errors
    ///
    /// [`DsigError::Key`] if the DER is malformed, names no curve, or names one
    /// this crate does not implement.
    pub fn from_der(input: &[u8]) -> DsigResult<Self> {
        let components = der::parse_ec_private_key(input)
            .map_err(|error| DsigError::Key(format!("ECPrivateKey: {error}")))?;
        let curve = components.curve.ok_or_else(|| {
            DsigError::Key(
                "ECPrivateKey: the encoding names no curve, so the key cannot be interpreted"
                    .to_owned(),
            )
        })?;
        let key = Self::from_scalar(curve, &components.private_key)?;
        // When the encoding also carries the public point, it must be the one
        // the scalar generates; anything else is a corrupt or crafted key.
        if let Some(point) = components.public_key {
            if point != key.public.encoded_point() {
                return Err(DsigError::Key(
                    "ECPrivateKey: the public point does not match the private scalar".to_owned(),
                ));
            }
        }
        Ok(key)
    }

    /// The matching public key.
    #[must_use]
    pub fn public_key(&self) -> EcPublicKey {
        self.public.clone()
    }

    /// The curve the key lives on.
    #[must_use]
    pub const fn curve(&self) -> NamedCurve {
        self.public.curve()
    }

    /// The domain parameters of that curve.
    pub(crate) fn parameters(&self) -> &'static Curve {
        self.public.parameters()
    }

    pub(crate) const fn scalar(&self) -> &BigUint {
        &self.scalar
    }
}

impl std::fmt::Debug for EcPrivateKey {
    /// Prints the curve and nothing else — a private scalar must never reach a
    /// log file through a stray `{:?}`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EcPrivateKey")
            .field("curve", &self.curve())
            .field("scalar", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::hex;

    /// The P-256 key pair of RFC 6979 Appendix A.2.5.
    const P256_PRIVATE_KEY: &str =
        "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721";
    const P256_POINT: &str = "\
        04\
        60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6\
        7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299";

    #[test]
    fn a_point_can_be_read_as_sec1_or_as_coordinates() -> DsigResult<()> {
        let encoded = hex(P256_POINT);
        let from_point = EcPublicKey::from_sec1_point(NamedCurve::P256, &encoded)?;
        let from_coordinates = EcPublicKey::from_coordinates(
            NamedCurve::P256,
            encoded.get(1..33).unwrap_or_default(),
            encoded.get(33..).unwrap_or_default(),
        )?;
        assert_eq!(from_point, from_coordinates);
        assert_eq!(from_point.encoded_point(), encoded);
        assert_eq!(from_point.named_curve_uri(), P256_NAMED_CURVE_URI);
        Ok(())
    }

    #[test]
    fn a_private_key_generates_its_own_public_half() -> DsigResult<()> {
        let key = EcPrivateKey::from_scalar(NamedCurve::P256, &hex(P256_PRIVATE_KEY))?;
        assert_eq!(key.public_key().encoded_point(), hex(P256_POINT));
        assert_eq!(key.curve(), NamedCurve::P256);
        Ok(())
    }

    #[test]
    fn the_private_scalar_never_reaches_the_debug_output() -> DsigResult<()> {
        let key = EcPrivateKey::from_scalar(NamedCurve::P256, &hex(P256_PRIVATE_KEY))?;
        let rendered = format!("{key:?}");
        assert!(rendered.contains("redacted"), "{rendered}");
        assert!(!rendered.contains("C9AF"), "{rendered}");
        Ok(())
    }

    #[test]
    fn named_curve_uris_round_trip() {
        for curve in [NamedCurve::P256, NamedCurve::P384] {
            assert_eq!(named_curve_from_uri(named_curve_uri(curve)), Some(curve));
        }
        // secp521r1, which this crate does not implement.
        assert_eq!(named_curve_from_uri("urn:oid:1.3.132.0.35"), None);
        assert_eq!(named_curve_from_uri("http://example.invalid/curve"), None);
    }

    #[test]
    fn a_point_off_the_curve_is_refused() {
        let mut encoded = hex(P256_POINT);
        if let Some(last) = encoded.last_mut() {
            *last ^= 0x01;
        }
        let error = EcPublicKey::from_sec1_point(NamedCurve::P256, &encoded)
            .expect_err("a point off the curve is not a key");
        assert!(error.to_string().contains("curve equation"), "{error}");
    }

    /// Only the uncompressed form is permitted: a compressed point, the identity
    /// and a point of the wrong curve's width are all refused.
    #[test]
    fn only_uncompressed_points_are_accepted() {
        let encoded = hex(P256_POINT);
        let mut compressed = vec![0x02];
        compressed.extend_from_slice(encoded.get(1..33).unwrap_or_default());
        assert!(EcPublicKey::from_sec1_point(NamedCurve::P256, &compressed).is_err());
        assert!(EcPublicKey::from_sec1_point(NamedCurve::P256, &[0x00]).is_err());
        assert!(EcPublicKey::from_sec1_point(NamedCurve::P384, &encoded).is_err());
    }

    #[test]
    fn a_scalar_outside_the_group_order_is_refused() {
        assert!(EcPrivateKey::from_scalar(NamedCurve::P256, &[0_u8; 32]).is_err());
        assert!(EcPrivateKey::from_scalar(NamedCurve::P256, &[0_u8; 31]).is_err());
        let order = NamedCurve::P256
            .curve()
            .order()
            .to_be_bytes_padded(32)
            .expect("the order is 256 bits");
        assert!(EcPrivateKey::from_scalar(NamedCurve::P256, &order).is_err());
    }
}
