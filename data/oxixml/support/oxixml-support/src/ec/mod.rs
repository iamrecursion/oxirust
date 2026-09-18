//! Elliptic-curve arithmetic over the NIST prime curves **P-256** and **P-384**.
//!
//! # Scope
//!
//! This module is the curve-and-field layer that the elliptic-curve parts of the XML security
//! specifications are built on: ECDSA signatures in `oxixml-dsig` and ECDH key agreement in
//! `oxixml-enc`. It provides
//!
//! * the [domain parameters](P256_PRIME) of the two curves XML Signature and XML Encryption
//!   interoperability actually requires,
//! * [`AffinePoint`], a curve point with the point at infinity representable,
//! * point addition, doubling, scalar multiplication and the double-scalar product
//!   `u₁·G + u₂·Q` that signature verification needs,
//! * the SEC 1 **uncompressed** point encoding (`04 || X || Y`) with full validation on decode,
//!   and
//! * the scalar range helpers a caller needs to check a private key, an `r`/`s` pair or a
//!   shared secret.
//!
//! Deliberately **not** here: the signature and key-agreement schemes themselves, key
//! serialisation formats, compressed point encodings, and any curve other than these two.
//!
//! # Timing
//!
//! [`Curve::scalar_mul`] is a Montgomery ladder: it performs one addition and one doubling per
//! scalar bit, over a bit count fixed by the group order, so the *sequence* of point operations
//! does not depend on the scalar. The underlying [`BigUint`] arithmetic
//! is nonetheless variable time — its operands are trimmed to their significant limbs — so this
//! module is **not** a constant-time implementation and does not claim resistance to a
//! side-channel attacker who can observe the arithmetic. It is written for verification, where
//! every input is public, and for ephemeral key agreement, where a recovered secret expires
//! with the message. Long-lived secret scalars need a hardened implementation.
//! [`Curve::double_scalar_mul`] makes no timing effort at all: it is for verification only.
//!
//! # Example
//!
//! ```
//! use oxixml_support::bigint::BigUint;
//! use oxixml_support::ec::Curve;
//!
//! let curve = Curve::p256();
//!
//! // Multiply the base point: `2·G` is the published second multiple of the P-256 generator.
//! let doubled = curve.mul_base(&BigUint::from_u64(2));
//! let encoded = curve.encode_point(&doubled).expect("2G is not the point at infinity");
//! assert_eq!(encoded.len(), 65);
//! assert_eq!(encoded[0], 0x04);
//! assert_eq!(
//!     encoded[1..33],
//!     [
//!         0x7c, 0xf2, 0x7b, 0x18, 0x8d, 0x03, 0x4f, 0x7e, 0x8a, 0x52, 0x38, 0x03, 0x04, 0xb5,
//!         0x1a, 0xc3, 0xc0, 0x89, 0x69, 0xe2, 0x77, 0xf2, 0x1b, 0x35, 0xa6, 0x0b, 0x48, 0xfc,
//!         0x47, 0x66, 0x99, 0x78,
//!     ]
//! );
//!
//! // Decoding validates the encoding and returns the same point.
//! let decoded = curve.decode_point(&encoded).expect("a valid uncompressed point");
//! assert_eq!(decoded, doubled);
//!
//! // The base point has order `n`, so `n·G` is the point at infinity.
//! assert!(curve.mul_base(curve.order()).is_identity());
//! ```

use core::fmt;
use std::sync::OnceLock;

use crate::bigint::BigUint;

mod params;
mod point;
mod sec1;

pub use params::{
    COFACTOR, P256_A, P256_B, P256_GENERATOR_X, P256_GENERATOR_Y, P256_ORDER, P256_PRIME, P384_A,
    P384_B, P384_GENERATOR_X, P384_GENERATOR_Y, P384_ORDER, P384_PRIME,
};
pub use point::AffinePoint;

/// Everything that can go wrong when a curve point or scalar is read from an untrusted source.
///
/// The variants are deliberately coarse: a caller reporting a failure to a document author must
/// not be able to turn this module into an oracle that distinguishes *why* a point was rejected
/// beyond what the shape of the encoding already reveals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EcError {
    /// An encoded point did not have the length an uncompressed point of this curve must have.
    InvalidPointLength {
        /// The length `1 + 2 · field_bytes` the curve requires.
        expected: usize,
        /// The length actually supplied.
        actual: usize,
    },
    /// The leading tag byte selected an encoding this module does not implement.
    ///
    /// Only the uncompressed form (`0x04`) is supported; compressed points (`0x02` / `0x03`)
    /// and the single-byte encoding of the point at infinity (`0x00`) are rejected here.
    UnsupportedPointEncoding(u8),
    /// A coordinate was not a canonical field element, i.e. not less than the field prime.
    CoordinateOutOfRange,
    /// The coordinates are in range but do not satisfy the curve equation.
    NotOnCurve,
    /// The operation was given, or would have produced, the point at infinity, which has no
    /// uncompressed encoding and is never a valid public key.
    IdentityPoint,
    /// A scalar octet string did not have the curve's scalar width.
    InvalidScalarLength {
        /// The width in bytes the curve requires.
        expected: usize,
        /// The width actually supplied.
        actual: usize,
    },
    /// A scalar was zero or was not less than the group order, so it is outside `1..n`.
    ScalarOutOfRange,
}

impl fmt::Display for EcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPointLength { expected, actual } => write!(
                f,
                "encoded elliptic-curve point has {actual} bytes, expected {expected}"
            ),
            Self::UnsupportedPointEncoding(tag) => write!(
                f,
                "unsupported elliptic-curve point encoding 0x{tag:02x}: only uncompressed points (0x04) are supported"
            ),
            Self::CoordinateOutOfRange => {
                f.write_str("elliptic-curve point coordinate is not less than the field prime")
            }
            Self::NotOnCurve => {
                f.write_str("elliptic-curve point does not satisfy the curve equation")
            }
            Self::IdentityPoint => {
                f.write_str("the point at infinity is not a valid elliptic-curve point here")
            }
            Self::InvalidScalarLength { expected, actual } => write!(
                f,
                "elliptic-curve scalar has {actual} bytes, expected {expected}"
            ),
            Self::ScalarOutOfRange => {
                f.write_str("elliptic-curve scalar is zero or not less than the group order")
            }
        }
    }
}

impl std::error::Error for EcError {}

/// Which of the two supported curves a value belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveId {
    /// NIST P-256, also known as `secp256r1` and `prime256v1`.
    P256,
    /// NIST P-384, also known as `secp384r1`.
    P384,
}

impl CurveId {
    /// The curve's name as the NIST and XML security specifications spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::P256 => "P-256",
            Self::P384 => "P-384",
        }
    }

    /// The width in bytes of a field element — and, for these two curves, of a scalar as well.
    #[must_use]
    pub const fn field_bytes(self) -> usize {
        match self {
            Self::P256 => 32,
            Self::P384 => 48,
        }
    }

    /// The domain parameters of this curve.
    #[must_use]
    pub fn curve(self) -> &'static Curve {
        match self {
            Self::P256 => Curve::p256(),
            Self::P384 => Curve::p384(),
        }
    }
}

impl fmt::Display for CurveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The domain parameters of one curve, and the entry point for every operation on it.
///
/// A `Curve` is a process-wide singleton obtained from [`Curve::p256`], [`Curve::p384`] or
/// [`CurveId::curve`]; the parameters are parsed once on first use. Points are only ever
/// produced by the curve they belong to — by [`decode_point`](Curve::decode_point),
/// [`point_from_coordinates`](Curve::point_from_coordinates) or an arithmetic operation — so
/// the curve is always the context an [`AffinePoint`] is interpreted in.
pub struct Curve {
    /// Which curve this is.
    id: CurveId,
    /// The field prime `p`.
    prime: BigUint,
    /// The curve coefficient `a` (`p - 3` for both curves).
    a: BigUint,
    /// The curve coefficient `b`.
    b: BigUint,
    /// The order `n` of the base point.
    order: BigUint,
    /// The base point `G`.
    generator: AffinePoint,
}

/// The lazily parsed P-256 parameters.
static P256: OnceLock<Curve> = OnceLock::new();
/// The lazily parsed P-384 parameters.
static P384: OnceLock<Curve> = OnceLock::new();

impl Curve {
    /// The NIST P-256 (`secp256r1`) domain parameters.
    #[must_use]
    pub fn p256() -> &'static Curve {
        P256.get_or_init(|| {
            Curve::from_parameters(
                CurveId::P256,
                &P256_PRIME,
                &P256_A,
                &P256_B,
                &P256_GENERATOR_X,
                &P256_GENERATOR_Y,
                &P256_ORDER,
            )
        })
    }

    /// The NIST P-384 (`secp384r1`) domain parameters.
    #[must_use]
    pub fn p384() -> &'static Curve {
        P384.get_or_init(|| {
            Curve::from_parameters(
                CurveId::P384,
                &P384_PRIME,
                &P384_A,
                &P384_B,
                &P384_GENERATOR_X,
                &P384_GENERATOR_Y,
                &P384_ORDER,
            )
        })
    }

    /// Which curve this is.
    #[must_use]
    pub fn id(&self) -> CurveId {
        self.id
    }

    /// The width in bytes of a field element or a scalar on this curve.
    #[must_use]
    pub fn field_bytes(&self) -> usize {
        self.id.field_bytes()
    }

    /// The field prime `p`.
    #[must_use]
    pub fn prime(&self) -> &BigUint {
        &self.prime
    }

    /// The curve coefficient `a` of `y² = x³ + a·x + b`.
    #[must_use]
    pub fn a(&self) -> &BigUint {
        &self.a
    }

    /// The curve coefficient `b` of `y² = x³ + a·x + b`.
    #[must_use]
    pub fn b(&self) -> &BigUint {
        &self.b
    }

    /// The order `n` of the base point, which for these curves is the order of the whole group.
    #[must_use]
    pub fn order(&self) -> &BigUint {
        &self.order
    }

    /// The base point `G`.
    #[must_use]
    pub fn generator(&self) -> &AffinePoint {
        &self.generator
    }

    /// The cofactor, which is [`one`](COFACTOR) for both supported curves.
    #[must_use]
    pub fn cofactor(&self) -> u64 {
        COFACTOR
    }

    /// Whether `scalar` is a valid private key or signature component, i.e. in `1..n`.
    #[must_use]
    pub fn is_valid_scalar(&self, scalar: &BigUint) -> bool {
        !scalar.is_zero() && scalar < &self.order
    }

    /// Reduce `value` modulo the group order `n`.
    ///
    /// This is the conversion signature verification applies to a message digest before using
    /// it as a scalar. The result can be zero.
    ///
    /// Returns [`None`] if `value` is wider than [`BigUint::MAX_BITS`]; every digest and every
    /// decoded scalar is far below that.
    #[must_use]
    pub fn reduce_scalar(&self, value: &BigUint) -> Option<BigUint> {
        Some(value.div_rem(&self.order)?.1)
    }

    /// Decode a fixed-width big-endian scalar and check that it lies in `1..n`.
    ///
    /// This is the reader for a private key, for either half of an ECDSA signature value, and
    /// for anything else the specifications transmit as an integer of the curve's scalar width.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::InvalidScalarLength`] unless `bytes` is exactly
    /// [`field_bytes`](Curve::field_bytes) long, and [`EcError::ScalarOutOfRange`] if the value
    /// is zero or not less than the group order.
    pub fn scalar_from_be_bytes(&self, bytes: &[u8]) -> Result<BigUint, EcError> {
        let expected = self.field_bytes();
        if bytes.len() != expected {
            return Err(EcError::InvalidScalarLength {
                expected,
                actual: bytes.len(),
            });
        }
        let scalar = BigUint::from_be_bytes(bytes);
        if self.is_valid_scalar(&scalar) {
            Ok(scalar)
        } else {
            Err(EcError::ScalarOutOfRange)
        }
    }

    /// Parse the fixed-width parameter octet strings into working values.
    fn from_parameters(
        id: CurveId,
        prime: &[u8],
        a: &[u8],
        b: &[u8],
        generator_x: &[u8],
        generator_y: &[u8],
        order: &[u8],
    ) -> Curve {
        Curve {
            id,
            prime: BigUint::from_be_bytes(prime),
            a: BigUint::from_be_bytes(a),
            b: BigUint::from_be_bytes(b),
            order: BigUint::from_be_bytes(order),
            generator: AffinePoint::from_coordinates_unchecked(
                BigUint::from_be_bytes(generator_x),
                BigUint::from_be_bytes(generator_y),
            ),
        }
    }
}

impl fmt::Debug for Curve {
    /// Print the curve by name; the parameters are fixed by it and would only be noise.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Curve").field(&self.id).finish()
    }
}

/// Parse a big-endian hexadecimal string into a [`BigUint`], for the test vectors below.
#[cfg(test)]
fn big_from_hex(hex: &str) -> BigUint {
    let bytes: Vec<u8> = hex
        .as_bytes()
        .chunks(2)
        .map(|pair| {
            let high = char::from(pair[0]).to_digit(16).expect("hexadecimal digit") as u8;
            let low = char::from(pair[1]).to_digit(16).expect("hexadecimal digit") as u8;
            (high << 4) | low
        })
        .collect();
    BigUint::from_be_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_are_self_consistent() {
        for curve in [Curve::p256(), Curve::p384()] {
            let bits = curve.field_bytes() * 8;
            assert_eq!(curve.prime().bit_len(), bits, "{curve:?} prime width");
            assert_eq!(curve.order().bit_len(), bits, "{curve:?} order width");
            // `a = p - 3` on both curves.
            assert_eq!(
                curve.a(),
                &curve
                    .prime()
                    .checked_sub(&BigUint::from_u64(3))
                    .expect("the prime exceeds three"),
            );
            assert_eq!(curve.cofactor(), 1);
            // The base point is on the curve and is not the point at infinity.
            assert!(!curve.generator().is_identity());
            assert!(curve.is_on_curve(curve.generator()));
        }
    }

    #[test]
    fn curve_identification() {
        assert_eq!(CurveId::P256.name(), "P-256");
        assert_eq!(CurveId::P384.to_string(), "P-384");
        assert_eq!(CurveId::P256.field_bytes(), 32);
        assert_eq!(CurveId::P384.field_bytes(), 48);
        assert_eq!(CurveId::P256.curve().id(), CurveId::P256);
        assert_eq!(CurveId::P384.curve().id(), CurveId::P384);
        assert_eq!(format!("{:?}", Curve::p256()), "Curve(P256)");
        // The parameters are parsed once and shared.
        assert!(core::ptr::eq(Curve::p256(), CurveId::P256.curve()));
    }

    #[test]
    fn scalar_range_helpers() {
        let curve = Curve::p256();
        let order = curve.order();

        assert!(!curve.is_valid_scalar(&BigUint::zero()));
        assert!(curve.is_valid_scalar(&BigUint::one()));
        assert!(!curve.is_valid_scalar(order));
        assert!(
            curve.is_valid_scalar(
                &order
                    .checked_sub(&BigUint::one())
                    .expect("the order is not zero")
            )
        );

        // Reduction wraps at the order.
        assert_eq!(curve.reduce_scalar(order), Some(BigUint::zero()));
        assert_eq!(
            curve.reduce_scalar(&order.checked_add(&BigUint::one()).expect("no overflow")),
            Some(BigUint::one())
        );

        // Fixed-width decoding, with both failure modes.
        let mut encoded = vec![0u8; 32];
        encoded[31] = 9;
        assert_eq!(
            curve.scalar_from_be_bytes(&encoded),
            Ok(BigUint::from_u64(9))
        );
        assert_eq!(
            curve.scalar_from_be_bytes(&encoded[..31]),
            Err(EcError::InvalidScalarLength {
                expected: 32,
                actual: 31
            })
        );
        assert_eq!(
            curve.scalar_from_be_bytes(&[0u8; 32]),
            Err(EcError::ScalarOutOfRange)
        );
        let order_bytes = order.to_be_bytes_padded(32).expect("the order is 256 bits");
        assert_eq!(
            curve.scalar_from_be_bytes(&order_bytes),
            Err(EcError::ScalarOutOfRange)
        );
    }

    #[test]
    fn error_messages_name_the_problem() {
        assert_eq!(
            EcError::InvalidPointLength {
                expected: 65,
                actual: 12
            }
            .to_string(),
            "encoded elliptic-curve point has 12 bytes, expected 65"
        );
        assert!(
            EcError::UnsupportedPointEncoding(0x02)
                .to_string()
                .contains("0x02")
        );
        assert!(EcError::NotOnCurve.to_string().contains("curve equation"));
    }
}
