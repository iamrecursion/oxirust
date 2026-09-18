//! The SEC 1 octet-string encoding of curve points.
//!
//! Only the **uncompressed** form is implemented: the tag byte `0x04` followed by the x and y
//! coordinates, each padded to the curve's field width. That is the form XML Signature and XML
//! Encryption carry — the `ECKeyValue` public key of XML Signature 1.1 and the originator key
//! of ECDH key agreement are both uncompressed points — so point decompression, which would
//! need a modular square root, is deliberately absent.
//!
//! [SEC 1]: https://www.secg.org/sec1-v2.pdf

use super::{AffinePoint, Curve, EcError};
use crate::bigint::BigUint;

/// The SEC 1 tag byte introducing an uncompressed point.
const UNCOMPRESSED_TAG: u8 = 0x04;

impl Curve {
    /// Encode `point` as an uncompressed SEC 1 octet string, `04 || X || Y`.
    ///
    /// Both coordinates are padded to [`field_bytes`](Curve::field_bytes), so the result is
    /// always `1 + 2 · field_bytes` long.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::IdentityPoint`] for the point at infinity, which has no uncompressed
    /// encoding.
    pub fn encode_point(&self, point: &AffinePoint) -> Result<Vec<u8>, EcError> {
        let (Some(x), Some(y)) = (point.x(), point.y()) else {
            return Err(EcError::IdentityPoint);
        };
        let width = self.field_bytes();
        let mut encoded = Vec::with_capacity(1 + 2 * width);
        encoded.push(UNCOMPRESSED_TAG);
        encoded.extend_from_slice(&Self::field_element_bytes(x, width)?);
        encoded.extend_from_slice(&Self::field_element_bytes(y, width)?);
        Ok(encoded)
    }

    /// Decode an uncompressed SEC 1 octet string into a validated point.
    ///
    /// The encoding is checked completely: the tag byte, the length, that both coordinates are
    /// canonical field elements, and that they satisfy the curve equation. A point that decodes
    /// successfully is a usable public key — the [cofactor](super::COFACTOR) is one, so there is
    /// no small subgroup for it to fall into.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::UnsupportedPointEncoding`] for a tag other than `0x04` — including the
    /// compressed forms `0x02` and `0x03` and the single-byte encoding `0x00` of the point at
    /// infinity — [`EcError::InvalidPointLength`] for the wrong length,
    /// [`EcError::CoordinateOutOfRange`] for a coordinate at or above the field prime, and
    /// [`EcError::NotOnCurve`] for a coordinate pair that is not a point of this curve.
    pub fn decode_point(&self, encoded: &[u8]) -> Result<AffinePoint, EcError> {
        let width = self.field_bytes();
        let expected = 1 + 2 * width;
        match encoded.first() {
            None => {
                return Err(EcError::InvalidPointLength {
                    expected,
                    actual: 0,
                });
            }
            Some(&tag) if tag != UNCOMPRESSED_TAG => {
                return Err(EcError::UnsupportedPointEncoding(tag));
            }
            Some(_) => {}
        }
        if encoded.len() != expected {
            return Err(EcError::InvalidPointLength {
                expected,
                actual: encoded.len(),
            });
        }
        let x = BigUint::from_be_bytes(&encoded[1..=width]);
        let y = BigUint::from_be_bytes(&encoded[1 + width..]);
        self.point_from_coordinates(x, y)
    }

    /// Encode only the x coordinate of `point`, padded to the field width.
    ///
    /// This is the conversion elliptic-curve Diffie-Hellman applies to the shared point before
    /// the result is fed to a key derivation function.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::IdentityPoint`] for the point at infinity, which as a shared secret
    /// would mean the key agreement had degenerated and must be abandoned.
    pub fn encode_x_coordinate(&self, point: &AffinePoint) -> Result<Vec<u8>, EcError> {
        match point.x() {
            None => Err(EcError::IdentityPoint),
            Some(x) => Self::field_element_bytes(x, self.field_bytes()),
        }
    }

    /// Encode `scalar` as a big-endian octet string of the curve's scalar width.
    ///
    /// This is the inverse of [`scalar_from_be_bytes`](Curve::scalar_from_be_bytes), and the
    /// form the specifications transmit the two halves of an ECDSA signature value in.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::ScalarOutOfRange`] if the value is too wide for that width.
    pub fn encode_scalar(&self, scalar: &BigUint) -> Result<Vec<u8>, EcError> {
        scalar
            .to_be_bytes_padded(self.field_bytes())
            .ok_or(EcError::ScalarOutOfRange)
    }

    /// Pad one field element to `width` bytes.
    ///
    /// A coordinate of a valid point always fits; anything else is reported rather than
    /// truncated.
    fn field_element_bytes(value: &BigUint, width: usize) -> Result<Vec<u8>, EcError> {
        value
            .to_be_bytes_padded(width)
            .ok_or(EcError::CoordinateOutOfRange)
    }
}

#[cfg(test)]
mod tests {
    use super::super::CurveId;
    use super::*;

    /// Both curves, each with a point to exercise the encoding on.
    fn samples() -> [(&'static Curve, AffinePoint); 2] {
        [CurveId::P256.curve(), CurveId::P384.curve()].map(|curve| {
            let point = curve.mul_base(&BigUint::from_u64(3));
            (curve, point)
        })
    }

    #[test]
    fn round_trip() {
        for (curve, point) in samples() {
            let width = curve.field_bytes();
            let encoded = curve.encode_point(&point).expect("3G is not the identity");
            assert_eq!(encoded.len(), 1 + 2 * width);
            assert_eq!(encoded[0], 0x04);
            assert_eq!(
                curve.decode_point(&encoded).expect("a valid encoding"),
                point
            );

            // The halves are the padded coordinates.
            assert_eq!(
                &encoded[1..=width],
                curve
                    .encode_x_coordinate(&point)
                    .expect("3G is not the identity")
                    .as_slice()
            );
            assert_eq!(
                &encoded[1 + width..],
                point
                    .y()
                    .expect("3G is not the identity")
                    .to_be_bytes_padded(width)
                    .expect("a coordinate fits the field width")
                    .as_slice()
            );
        }
    }

    #[test]
    fn small_coordinates_are_left_padded() {
        // The base point of P-256 has a full-width x coordinate, but a small value must still
        // occupy the whole field width rather than the shortest encoding.
        let curve = Curve::p256();
        assert_eq!(
            curve.encode_scalar(&BigUint::from_u64(1)),
            Ok({
                let mut expected = vec![0u8; 32];
                expected[31] = 1;
                expected
            })
        );
        let too_wide = BigUint::from_be_bytes(&[0xFF; 33]);
        assert_eq!(
            curve.encode_scalar(&too_wide),
            Err(EcError::ScalarOutOfRange)
        );
    }

    #[test]
    fn the_identity_has_no_encoding() {
        for (curve, _) in samples() {
            let identity = AffinePoint::identity();
            assert_eq!(curve.encode_point(&identity), Err(EcError::IdentityPoint));
            assert_eq!(
                curve.encode_x_coordinate(&identity),
                Err(EcError::IdentityPoint)
            );
            // Nor does its one-byte SEC 1 encoding decode back.
            assert_eq!(
                curve.decode_point(&[0x00]),
                Err(EcError::UnsupportedPointEncoding(0x00))
            );
        }
    }

    #[test]
    fn decoding_rejects_malformed_input() {
        for (curve, point) in samples() {
            let width = curve.field_bytes();
            let expected = 1 + 2 * width;
            let encoded = curve.encode_point(&point).expect("3G is not the identity");

            // Empty input, and input truncated by one byte.
            assert_eq!(
                curve.decode_point(&[]),
                Err(EcError::InvalidPointLength {
                    expected,
                    actual: 0
                })
            );
            assert_eq!(
                curve.decode_point(&encoded[..expected - 1]),
                Err(EcError::InvalidPointLength {
                    expected,
                    actual: expected - 1
                })
            );
            // Trailing junk is not ignored either.
            let mut overlong = encoded.clone();
            overlong.push(0);
            assert_eq!(
                curve.decode_point(&overlong),
                Err(EcError::InvalidPointLength {
                    expected,
                    actual: expected + 1
                })
            );

            // Compressed points are a valid SEC 1 encoding that this module does not implement.
            for tag in [0x02u8, 0x03, 0x06, 0xFF] {
                let mut tagged = encoded.clone();
                tagged[0] = tag;
                assert_eq!(
                    curve.decode_point(&tagged),
                    Err(EcError::UnsupportedPointEncoding(tag))
                );
            }

            // A coordinate at the field prime is not a canonical field element.
            let prime_bytes = curve
                .prime()
                .to_be_bytes_padded(width)
                .expect("the prime is exactly the field width");
            let mut out_of_range = encoded.clone();
            out_of_range[1..=width].copy_from_slice(&prime_bytes);
            assert_eq!(
                curve.decode_point(&out_of_range),
                Err(EcError::CoordinateOutOfRange)
            );
            let mut out_of_range = encoded.clone();
            out_of_range[1 + width..].copy_from_slice(&prime_bytes);
            assert_eq!(
                curve.decode_point(&out_of_range),
                Err(EcError::CoordinateOutOfRange)
            );

            // A point that is in range but not on the curve.
            let mut off_curve = encoded.clone();
            off_curve[expected - 1] ^= 1;
            assert_eq!(curve.decode_point(&off_curve), Err(EcError::NotOnCurve));
            // Including the all-zero pair, which the non-zero coefficient `b` excludes.
            let mut origin = vec![0u8; expected];
            origin[0] = 0x04;
            assert_eq!(curve.decode_point(&origin), Err(EcError::NotOnCurve));
        }
    }
}
