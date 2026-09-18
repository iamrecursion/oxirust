//! Curve points and the group law.
//!
//! The public surface is affine: a point is a pair of field elements, or the point at infinity.
//! Internally the group law runs in Jacobian projective coordinates, where an addition costs a
//! handful of field multiplications instead of a modular inversion; a whole scalar
//! multiplication therefore needs exactly one inversion, at the very end.

use super::{Curve, EcError};
use crate::bigint::BigUint;

/// A point on one of the supported curves, in affine coordinates.
///
/// The point at infinity — the identity of the group law — is representable and is what
/// [`AffinePoint::identity`] returns. Every other value is a pair `(x, y)` of canonical field
/// elements, both less than the curve's prime, satisfying `y² = x³ + a·x + b`.
///
/// A point carries no reference to its curve: it is produced by a [`Curve`] and passed back to
/// the same one, which is the context that gives its coordinates meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    /// The coordinates, or [`None`] for the point at infinity.
    coordinates: Option<(BigUint, BigUint)>,
}

impl AffinePoint {
    /// The point at infinity, the identity element of the curve group.
    #[must_use]
    pub fn identity() -> Self {
        Self { coordinates: None }
    }

    /// Whether this is the point at infinity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.coordinates.is_none()
    }

    /// The x coordinate, or [`None`] for the point at infinity.
    #[must_use]
    pub fn x(&self) -> Option<&BigUint> {
        self.coordinates.as_ref().map(|(x, _)| x)
    }

    /// The y coordinate, or [`None`] for the point at infinity.
    #[must_use]
    pub fn y(&self) -> Option<&BigUint> {
        self.coordinates.as_ref().map(|(_, y)| y)
    }

    /// Assemble a point from coordinates that are already known to be valid.
    ///
    /// Reserved for the parameter tables and for results of the group law, both of which are
    /// on the curve by construction. Anything derived from input goes through
    /// [`Curve::point_from_coordinates`] instead.
    pub(super) fn from_coordinates_unchecked(x: BigUint, y: BigUint) -> Self {
        Self {
            coordinates: Some((x, y)),
        }
    }
}

/// A curve point in Jacobian projective coordinates.
///
/// `(X, Y, Z)` with a non-zero `Z` denotes the affine point `(X/Z², Y/Z³)`; a zero `Z` denotes
/// the point at infinity, whatever `X` and `Y` hold.
#[derive(Clone)]
struct Jacobian {
    /// The projective x coordinate.
    x: BigUint,
    /// The projective y coordinate.
    y: BigUint,
    /// The projective scale factor, zero for the point at infinity.
    z: BigUint,
}

impl Jacobian {
    /// The point at infinity.
    fn identity() -> Self {
        Self {
            x: BigUint::one(),
            y: BigUint::one(),
            z: BigUint::zero(),
        }
    }

    /// Whether this is the point at infinity.
    fn is_identity(&self) -> bool {
        self.z.is_zero()
    }
}

impl Curve {
    /// Build a point from coordinates read from a document or any other untrusted source.
    ///
    /// # Errors
    ///
    /// Returns [`EcError::CoordinateOutOfRange`] if either coordinate is not less than the
    /// field prime, and [`EcError::NotOnCurve`] if the pair does not satisfy the curve
    /// equation. Because the [cofactor](super::COFACTOR) is one, a point that passes these
    /// checks is guaranteed to have order `n`.
    pub fn point_from_coordinates(&self, x: BigUint, y: BigUint) -> Result<AffinePoint, EcError> {
        if x >= self.prime || y >= self.prime {
            return Err(EcError::CoordinateOutOfRange);
        }
        let point = AffinePoint::from_coordinates_unchecked(x, y);
        if self.is_on_curve(&point) {
            Ok(point)
        } else {
            Err(EcError::NotOnCurve)
        }
    }

    /// Whether `point` satisfies `y² = x³ + a·x + b` with both coordinates in range.
    ///
    /// The point at infinity is on every curve, so it answers `true`.
    #[must_use]
    pub fn is_on_curve(&self, point: &AffinePoint) -> bool {
        let Some((x, y)) = point.coordinates.as_ref() else {
            return true;
        };
        if x >= &self.prime || y >= &self.prime {
            return false;
        }
        let left = self.field_sqr(y);
        let cube = self.field_mul(&self.field_sqr(x), x);
        let right = self.field_add(&self.field_add(&cube, &self.field_mul(&self.a, x)), &self.b);
        left == right
    }

    /// The additive inverse `-P`, which shares the x coordinate and negates y.
    #[must_use]
    pub fn negate(&self, point: &AffinePoint) -> AffinePoint {
        match point.coordinates.as_ref() {
            None => AffinePoint::identity(),
            Some((x, y)) => {
                // `-0` is `0`; every other y has `p - y` as its negative.
                let negated = match self.prime.checked_sub(y) {
                    Some(value) if !y.is_zero() => value,
                    _ => BigUint::zero(),
                };
                AffinePoint::from_coordinates_unchecked(x.clone(), negated)
            }
        }
    }

    /// The group law: `lhs + rhs`.
    ///
    /// Total for every pair of points on this curve, including the point at infinity on either
    /// side, two equal points (which doubles), and a point plus its negation (which yields the
    /// point at infinity).
    #[must_use]
    pub fn add(&self, lhs: &AffinePoint, rhs: &AffinePoint) -> AffinePoint {
        let sum = self.jacobian_add(&self.to_jacobian(lhs), &self.to_jacobian(rhs));
        self.to_affine(&sum)
    }

    /// The group law applied to one point twice: `2·P`.
    #[must_use]
    pub fn double(&self, point: &AffinePoint) -> AffinePoint {
        let doubled = self.jacobian_double(&self.to_jacobian(point));
        self.to_affine(&doubled)
    }

    /// Scalar multiplication `scalar · point`, by a left-to-right Montgomery ladder.
    ///
    /// The result is exact for any `scalar`, including values at or above the group order —
    /// there is no reduction step, so `n · P` really is the point at infinity. The ladder runs
    /// over `max(scalar.bit_len(), n.bit_len())` bits and performs one addition and one
    /// doubling per bit whatever that bit is, so its cost depends on the *width* of the scalar
    /// but not on its value. See the [module documentation](super) for the timing posture, and
    /// note that a caller taking a scalar from untrusted input should bound it first with
    /// [`is_valid_scalar`](Curve::is_valid_scalar) or [`reduce_scalar`](Curve::reduce_scalar).
    #[must_use]
    pub fn scalar_mul(&self, point: &AffinePoint, scalar: &BigUint) -> AffinePoint {
        if point.is_identity() {
            return AffinePoint::identity();
        }
        // `accumulator` holds `k·P` for the scalar bits consumed so far and `successor` holds
        // `(k + 1)·P`; their difference stays `P` through every step, which is what makes the
        // two branches below cost the same.
        let mut accumulator = Jacobian::identity();
        let mut successor = self.to_jacobian(point);
        for index in (0..scalar.bit_len().max(self.order.bit_len())).rev() {
            if scalar.bit(index) {
                accumulator = self.jacobian_add(&accumulator, &successor);
                successor = self.jacobian_double(&successor);
            } else {
                successor = self.jacobian_add(&accumulator, &successor);
                accumulator = self.jacobian_double(&accumulator);
            }
        }
        self.to_affine(&accumulator)
    }

    /// Scalar multiplication of the base point, `scalar · G`.
    #[must_use]
    pub fn mul_base(&self, scalar: &BigUint) -> AffinePoint {
        self.scalar_mul(&self.generator, scalar)
    }

    /// The double-scalar product `base_scalar · G + point_scalar · point`.
    ///
    /// This is the shape signature verification needs, and it is computed with Shamir's trick:
    /// both scalars are consumed together over a single sequence of doublings, roughly halving
    /// the work compared with two separate scalar multiplications. Additions are skipped when
    /// both scalar bits are zero, so unlike [`scalar_mul`](Curve::scalar_mul) this makes no
    /// effort to be regular — it is meant for verification, where both scalars are public.
    #[must_use]
    pub fn double_scalar_mul(
        &self,
        base_scalar: &BigUint,
        point_scalar: &BigUint,
        point: &AffinePoint,
    ) -> AffinePoint {
        let base = self.to_jacobian(&self.generator);
        let other = self.to_jacobian(point);
        // Indexed by the pair of scalar bits: nothing, G, Q, G + Q.
        let table = [
            Jacobian::identity(),
            base.clone(),
            other.clone(),
            self.jacobian_add(&base, &other),
        ];
        let mut accumulator = Jacobian::identity();
        for index in (0..base_scalar.bit_len().max(point_scalar.bit_len())).rev() {
            accumulator = self.jacobian_double(&accumulator);
            let selector =
                usize::from(base_scalar.bit(index)) | (usize::from(point_scalar.bit(index)) << 1);
            if selector != 0 {
                accumulator = self.jacobian_add(&accumulator, &table[selector]);
            }
        }
        self.to_affine(&accumulator)
    }

    // --- projective coordinates ----------------------------------------------------------

    /// Lift an affine point into Jacobian coordinates with `Z = 1`.
    fn to_jacobian(&self, point: &AffinePoint) -> Jacobian {
        match point.coordinates.as_ref() {
            None => Jacobian::identity(),
            Some((x, y)) => Jacobian {
                x: x.clone(),
                y: y.clone(),
                z: BigUint::one(),
            },
        }
    }

    /// Project a Jacobian point back onto the affine plane, dividing out `Z`.
    fn to_affine(&self, point: &Jacobian) -> AffinePoint {
        // A zero `Z` is exactly the point at infinity, and is the only value in `0..p` that the
        // prime field leaves without an inverse.
        let Some(z_inverse) = point.z.mod_inverse(&self.prime) else {
            return AffinePoint::identity();
        };
        let z_inverse_squared = self.field_sqr(&z_inverse);
        let z_inverse_cubed = self.field_mul(&z_inverse_squared, &z_inverse);
        AffinePoint::from_coordinates_unchecked(
            self.field_mul(&point.x, &z_inverse_squared),
            self.field_mul(&point.y, &z_inverse_cubed),
        )
    }

    /// The group law in Jacobian coordinates.
    fn jacobian_add(&self, lhs: &Jacobian, rhs: &Jacobian) -> Jacobian {
        if lhs.is_identity() {
            return rhs.clone();
        }
        if rhs.is_identity() {
            return lhs.clone();
        }
        // Bring both points onto the common scale `(Z1·Z2)` before comparing or subtracting.
        let lhs_zz = self.field_sqr(&lhs.z);
        let rhs_zz = self.field_sqr(&rhs.z);
        let lhs_x = self.field_mul(&lhs.x, &rhs_zz);
        let rhs_x = self.field_mul(&rhs.x, &lhs_zz);
        let lhs_y = self.field_mul(&lhs.y, &self.field_mul(&rhs_zz, &rhs.z));
        let rhs_y = self.field_mul(&rhs.y, &self.field_mul(&lhs_zz, &lhs.z));
        if lhs_x == rhs_x {
            // Equal x: either the same point, which the chord formula cannot express, or a
            // point and its negation, whose sum is the point at infinity.
            return if lhs_y == rhs_y {
                self.jacobian_double(lhs)
            } else {
                Jacobian::identity()
            };
        }
        let h = self.field_sub(&rhs_x, &lhs_x);
        let r = self.field_sub(&rhs_y, &lhs_y);
        let h_squared = self.field_sqr(&h);
        let h_cubed = self.field_mul(&h, &h_squared);
        let scaled = self.field_mul(&lhs_x, &h_squared);
        let x = self.field_sub(
            &self.field_sub(&self.field_sqr(&r), &h_cubed),
            &self.field_double(&scaled),
        );
        let y = self.field_sub(
            &self.field_mul(&r, &self.field_sub(&scaled, &x)),
            &self.field_mul(&lhs_y, &h_cubed),
        );
        let z = self.field_mul(&h, &self.field_mul(&lhs.z, &rhs.z));
        Jacobian { x, y, z }
    }

    /// Point doubling in Jacobian coordinates, for the general coefficient `a`.
    fn jacobian_double(&self, point: &Jacobian) -> Jacobian {
        if point.is_identity() {
            return Jacobian::identity();
        }
        let y_squared = self.field_sqr(&point.y);
        // `s = 4·X·Y²`, the value the tangent slope is measured against.
        let s = self.field_double(&self.field_double(&self.field_mul(&point.x, &y_squared)));
        let x_squared = self.field_sqr(&point.x);
        let z_squared = self.field_sqr(&point.z);
        // `m = 3·X² + a·Z⁴`, the numerator of the tangent slope.
        let m = self.field_add(
            &self.field_add(&self.field_double(&x_squared), &x_squared),
            &self.field_mul(&self.a, &self.field_sqr(&z_squared)),
        );
        let x = self.field_sub(&self.field_sqr(&m), &self.field_double(&s));
        let y_fourth_times_eight =
            self.field_double(&self.field_double(&self.field_double(&self.field_sqr(&y_squared))));
        let y = self.field_sub(
            &self.field_mul(&m, &self.field_sub(&s, &x)),
            &y_fourth_times_eight,
        );
        // A point of order two would land here with `Y = 0`, giving `Z = 0`: the point at
        // infinity, which is the right answer. These curves have no such point.
        let z = self.field_double(&self.field_mul(&point.y, &point.z));
        Jacobian { x, y, z }
    }

    // --- prime field arithmetic ----------------------------------------------------------
    //
    // Every operand below is already reduced into `0..p`, and so is every result.

    /// `a + b` in the field.
    fn field_add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        // Both addends are below the prime, so one conditional subtraction reduces the sum.
        let sum = a + b;
        match sum.checked_sub(&self.prime) {
            Some(reduced) => reduced,
            None => sum,
        }
    }

    /// `2·a` in the field.
    fn field_double(&self, a: &BigUint) -> BigUint {
        self.field_add(a, a)
    }

    /// `a - b` in the field.
    fn field_sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        match a.checked_sub(b) {
            Some(difference) => difference,
            None => (a + &self.prime)
                .checked_sub(b)
                .expect("field operands are below the prime, so `a + p` is at least `b`"),
        }
    }

    /// `a · b` in the field.
    fn field_mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.field_reduce(&(a * b))
    }

    /// `a²` in the field.
    fn field_sqr(&self, a: &BigUint) -> BigUint {
        self.field_reduce(&(a * a))
    }

    /// Reduce a product back into `0..p`.
    fn field_reduce(&self, value: &BigUint) -> BigUint {
        value
            .div_rem(&self.prime)
            .expect("the field prime is non-zero and far narrower than BigUint::MAX_BITS")
            .1
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CurveId, big_from_hex};
    use super::*;
    use crate::rng::{Rng, SplitMix64};

    /// The first ten multiples of the P-256 base point, `1·G` through `10·G`.
    const P256_MULTIPLES: [(&str, &str); 10] = [
        (
            "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
            "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
        ),
        (
            "7cf27b188d034f7e8a52380304b51ac3c08969e277f21b35a60b48fc47669978",
            "07775510db8ed040293d9ac69f7430dbba7dade63ce982299e04b79d227873d1",
        ),
        (
            "5ecbe4d1a6330a44c8f7ef951d4bf165e6c6b721efada985fb41661bc6e7fd6c",
            "8734640c4998ff7e374b06ce1a64a2ecd82ab036384fb83d9a79b127a27d5032",
        ),
        (
            "e2534a3532d08fbba02dde659ee62bd0031fe2db785596ef509302446b030852",
            "e0f1575a4c633cc719dfee5fda862d764efc96c3f30ee0055c42c23f184ed8c6",
        ),
        (
            "51590b7a515140d2d784c85608668fdfef8c82fd1f5be52421554a0dc3d033ed",
            "e0c17da8904a727d8ae1bf36bf8a79260d012f00d4d80888d1d0bb44fda16da4",
        ),
        (
            "b01a172a76a4602c92d3242cb897dde3024c740debb215b4c6b0aae93c2291a9",
            "e85c10743237dad56fec0e2dfba703791c00f7701c7e16bdfd7c48538fc77fe2",
        ),
        (
            "8e533b6fa0bf7b4625bb30667c01fb607ef9f8b8a80fef5b300628703187b2a3",
            "73eb1dbde03318366d069f83a6f5900053c73633cb041b21c55e1a86c1f400b4",
        ),
        (
            "62d9779dbee9b0534042742d3ab54cadc1d238980fce97dbb4dd9dc1db6fb393",
            "ad5accbd91e9d8244ff15d771167cee0a2ed51f6bbe76a78da540a6a0f09957e",
        ),
        (
            "ea68d7b6fedf0b71878938d51d71f8729e0acb8c2c6df8b3d79e8a4b90949ee0",
            "2a2744c972c9fce787014a964a8ea0c84d714feaa4de823fe85a224a4dd048fa",
        ),
        (
            "cef66d6b2a3a993e591214d1ea223fb545ca6c471c48306e4c36069404c5723f",
            "878662a229aaae906e123cdd9d3b4c10590ded29fe751eeeca34bbaa44af0773",
        ),
    ];

    /// The first ten multiples of the P-384 base point, `1·G` through `10·G`.
    const P384_MULTIPLES: [(&str, &str); 10] = [
        (
            "aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7",
            "3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f",
        ),
        (
            "08d999057ba3d2d969260045c55b97f089025959a6f434d651d207d19fb96e9e4fe0e86ebe0e64f85b96a9c75295df61",
            "8e80f1fa5b1b3cedb7bfe8dffd6dba74b275d875bc6cc43e904e505f256ab4255ffd43e94d39e22d61501e700a940e80",
        ),
        (
            "077a41d4606ffa1464793c7e5fdc7d98cb9d3910202dcd06bea4f240d3566da6b408bbae5026580d02d7e5c70500c831",
            "c995f7ca0b0c42837d0bbe9602a9fc998520b41c85115aa5f7684c0edc111eacc24abd6be4b5d298b65f28600a2f1df1",
        ),
        (
            "138251cd52ac9298c1c8aad977321deb97e709bd0b4ca0aca55dc8ad51dcfc9d1589a1597e3a5120e1efd631c63e1835",
            "cacae29869a62e1631e8a28181ab56616dc45d918abc09f3ab0e63cf792aa4dced7387be37bba569549f1c02b270ed67",
        ),
        (
            "11de24a2c251c777573cac5ea025e467f208e51dbff98fc54f6661cbe56583b037882f4a1ca297e60abcdbc3836d84bc",
            "8fa696c77440f92d0f5837e90a00e7c5284b447754d5dee88c986533b6901aeb3177686d0ae8fb33184414abe6c1713a",
        ),
        (
            "627be1acd064d2b2226fe0d26f2d15d3c33ebcbb7f0f5da51cbd41f26257383021317d7202ff30e50937f0854e35c5df",
            "09766a4cb3f8b1c21be6dda6c14f1575b2c95352644f774c99864f613715441604c45b8d84e165311733a408d3f0f934",
        ),
        (
            "283c1d7365ce4788f29f8ebf234edffead6fe997fbea5ffa2d58cc9dfa7b1c508b05526f55b9ebb2040f05b48fb6d0e1",
            "9475c99061e41b88ba52efdb8c1690471a61d867ed799729d9c92cd01dbd225630d84ede32a78f9e64664cdac512ef8c",
        ),
        (
            "1692778ea596e0be75114297a6fa383445bf227fbe58190a900c3c73256f11fb5a3258d6f403d5ece6e9b269d822c87d",
            "dcd2365700d4106a835388ba3db8fd0e22554adc6d521cd4bd1c30c2ec0eec196bade1e9cdd1708d6f6abfa4022b0ad2",
        ),
        (
            "8f0a39a4049bcb3ef1bf29b8b025b78f2216f7291e6fd3bac6cb1ee285fb6e21c388528bfee2b9535c55e4461079118b",
            "62c77e1438b601d6452c4a5322c3a9799a9b3d7ca3c400c6b7678854aed9b3029e743efedfd51b68262da4f9ac664af8",
        ),
        (
            "a669c5563bd67eec678d29d6ef4fde864f372d90b79b9e88931d5c29291238cced8e85ab507bf91aa9cb2d13186658fb",
            "a988b72ae7c1279f22d9083db5f0ecddf70119550c183c31c502df78c3b705a8296d8195248288d997784f6ab73a21dd",
        ),
    ];

    /// The P-256 key pair of RFC 6979 appendix A.2.5: the private key and `d·G`.
    const P256_KEY_PAIR: (&str, &str, &str) = (
        "c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721",
        "60fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb6",
        "7903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e9f5177a3c294d4462299",
    );

    /// The P-384 key pair of RFC 6979 appendix A.2.6: the private key and `d·G`.
    const P384_KEY_PAIR: (&str, &str, &str) = (
        "6b9d3dad2e1b8c1c05b19875b6659f4de23c3b667bf297ba9aa47740787137d896d5724e4c70a825f872c9ea60d2edf5",
        "ec3a4e415b4e19a4568618029f427fa5da9a8bc4ae92e02e06aae5286b300c64def8f0ea9055866064a254515480bc13",
        "8015d9b72d7d57244ea8ef9ac0c621896708a59367f9dfb9f54ca84b3f1c9db1288b231c3ae0d4fe7344fd2533264720",
    );

    /// Rebuild an expected point from a hexadecimal coordinate pair.
    fn expected(curve: &Curve, coordinates: (&str, &str)) -> AffinePoint {
        curve
            .point_from_coordinates(big_from_hex(coordinates.0), big_from_hex(coordinates.1))
            .expect("published multiples of the base point are on the curve")
    }

    /// Both curves with their published table of base-point multiples.
    fn curves() -> [(&'static Curve, &'static [(&'static str, &'static str)]); 2] {
        [
            (CurveId::P256.curve(), &P256_MULTIPLES),
            (CurveId::P384.curve(), &P384_MULTIPLES),
        ]
    }

    #[test]
    fn repeated_addition_walks_the_published_multiples() {
        for (curve, multiples) in curves() {
            let generator = curve.generator();
            let mut running = generator.clone();
            for (index, coordinates) in multiples.iter().enumerate() {
                assert_eq!(
                    running,
                    expected(curve, *coordinates),
                    "{curve:?}: {}G by repeated addition",
                    index + 1
                );
                running = curve.add(&running, generator);
            }
        }
    }

    #[test]
    fn doubling_matches_the_published_multiples() {
        for (curve, multiples) in curves() {
            let generator = curve.generator();
            // 2G, 4G and 8G are the doublings reachable from the table.
            let two = curve.double(generator);
            assert_eq!(two, expected(curve, multiples[1]), "{curve:?}: 2G");
            let four = curve.double(&two);
            assert_eq!(four, expected(curve, multiples[3]), "{curve:?}: 4G");
            assert_eq!(
                curve.double(&four),
                expected(curve, multiples[7]),
                "{curve:?}: 8G"
            );
            // Doubling through the generic addition path must take the same branch.
            assert_eq!(curve.add(generator, generator), two, "{curve:?}: G + G");
        }
    }

    #[test]
    fn scalar_multiplication_matches_the_published_multiples() {
        for (curve, multiples) in curves() {
            for index in [0usize, 4, 9] {
                let scalar = BigUint::from_u64(index as u64 + 1);
                assert_eq!(
                    curve.mul_base(&scalar),
                    expected(curve, multiples[index]),
                    "{curve:?}: {}G by scalar multiplication",
                    index + 1
                );
            }
            // Zero and the group order both give the point at infinity.
            assert!(curve.mul_base(&BigUint::zero()).is_identity());
            assert!(curve.mul_base(curve.order()).is_identity());
        }
    }

    #[test]
    fn published_key_pairs_are_reproduced() {
        for (curve, (private, x, y)) in [
            (CurveId::P256.curve(), P256_KEY_PAIR),
            (CurveId::P384.curve(), P384_KEY_PAIR),
        ] {
            let private_key = big_from_hex(private);
            assert!(curve.is_valid_scalar(&private_key));
            assert_eq!(
                curve.mul_base(&private_key),
                expected(curve, (x, y)),
                "{curve:?}: published public key"
            );
        }
    }

    #[test]
    fn identity_and_negation_behave() {
        for (curve, _) in curves() {
            let identity = AffinePoint::identity();
            let generator = curve.generator();
            assert!(identity.is_identity());
            assert_eq!(identity.x(), None);
            assert_eq!(identity.y(), None);
            assert!(curve.is_on_curve(&identity));

            assert_eq!(&curve.add(&identity, generator), generator);
            assert_eq!(&curve.add(generator, &identity), generator);
            assert!(curve.add(&identity, &identity).is_identity());
            assert!(curve.double(&identity).is_identity());
            assert!(
                curve
                    .scalar_mul(&identity, &BigUint::from_u64(7))
                    .is_identity()
            );
            assert!(curve.negate(&identity).is_identity());

            // `-G` is on the curve, and `G + (-G)` is the point at infinity.
            let negated = curve.negate(generator);
            assert!(curve.is_on_curve(&negated));
            assert_eq!(negated.x(), generator.x());
            assert_ne!(negated.y(), generator.y());
            assert!(curve.add(generator, &negated).is_identity());
            assert_eq!(&curve.negate(&negated), generator);

            // `(n - 1)·G` is `-G`, so adding G once more wraps to the point at infinity.
            let order_minus_one = curve
                .order()
                .checked_sub(&BigUint::one())
                .expect("the order is not zero");
            assert_eq!(curve.mul_base(&order_minus_one), negated);
        }
    }

    #[test]
    fn scalar_multiplication_is_linear() {
        let mut rng = SplitMix64::new(0x0EC0_1A5E_5CA1_A125);
        for (curve, _) in curves() {
            let mut bytes = vec![0u8; curve.field_bytes()];
            rng.fill_bytes(&mut bytes);
            let a = curve
                .reduce_scalar(&BigUint::from_be_bytes(&bytes))
                .expect("a field-width value is far below MAX_BITS");
            rng.fill_bytes(&mut bytes);
            let b = curve
                .reduce_scalar(&BigUint::from_be_bytes(&bytes))
                .expect("a field-width value is far below MAX_BITS");

            let point_a = curve.mul_base(&a);
            let point_b = curve.mul_base(&b);
            let sum = a.checked_add(&b).expect("addition never fails");

            // `(a + b)·G = a·G + b·G`, with the sum deliberately left unreduced so the scalar
            // multiplication has to handle a scalar above the group order.
            assert_eq!(
                curve.mul_base(&sum),
                curve.add(&point_a, &point_b),
                "{curve:?}: distributivity over scalar addition"
            );
            // And the same product reached the other way round.
            assert_eq!(
                curve.scalar_mul(&point_a, &b),
                curve.scalar_mul(&point_b, &a),
                "{curve:?}: associativity of the scalar product"
            );
            // Every point cancels against its negation.
            assert!(
                curve.add(&point_a, &curve.negate(&point_a)).is_identity(),
                "{curve:?}: a point plus its negation"
            );
            // Encoding a random point and reading it back is lossless.
            let encoded = curve
                .encode_point(&point_b)
                .expect("a random multiple of G is not the identity");
            assert_eq!(
                curve.decode_point(&encoded),
                Ok(point_b),
                "{curve:?}: encode then decode"
            );
        }
    }

    #[test]
    fn double_scalar_product_matches_two_separate_products() {
        let mut rng = SplitMix64::new(0x5EED_0EC0_D0B1_E5CA);
        for (curve, _) in curves() {
            let mut bytes = vec![0u8; curve.field_bytes()];
            rng.fill_bytes(&mut bytes);
            let u1 = curve
                .reduce_scalar(&BigUint::from_be_bytes(&bytes))
                .expect("a field-width value is far below MAX_BITS");
            rng.fill_bytes(&mut bytes);
            let u2 = curve
                .reduce_scalar(&BigUint::from_be_bytes(&bytes))
                .expect("a field-width value is far below MAX_BITS");
            let q = curve.mul_base(&BigUint::from_u64(6));

            assert_eq!(
                curve.double_scalar_mul(&u1, &u2, &q),
                curve.add(&curve.mul_base(&u1), &curve.scalar_mul(&q, &u2)),
                "{curve:?}: u1·G + u2·Q"
            );
            // Degenerate scalars still land on the right point.
            assert_eq!(
                curve.double_scalar_mul(&BigUint::zero(), &BigUint::one(), &q),
                q,
                "{curve:?}: 0·G + 1·Q"
            );
            assert_eq!(
                curve.double_scalar_mul(&BigUint::one(), &BigUint::zero(), &q),
                *curve.generator(),
                "{curve:?}: 1·G + 0·Q"
            );
            assert!(
                curve
                    .double_scalar_mul(&BigUint::zero(), &BigUint::zero(), &q)
                    .is_identity()
            );
            assert_eq!(
                curve.double_scalar_mul(&BigUint::one(), &BigUint::one(), &AffinePoint::identity()),
                *curve.generator(),
                "{curve:?}: the point at infinity contributes nothing"
            );
        }
    }

    #[test]
    fn coordinate_validation_rejects_bad_points() {
        for (curve, multiples) in curves() {
            let (x, y) = multiples[2];
            let (x, y) = (big_from_hex(x), big_from_hex(y));
            assert!(curve.point_from_coordinates(x.clone(), y.clone()).is_ok());

            // A y coordinate off by one leaves the curve.
            assert_eq!(
                curve.point_from_coordinates(
                    x.clone(),
                    y.checked_add(&BigUint::one())
                        .expect("addition never fails")
                ),
                Err(EcError::NotOnCurve)
            );
            // Coordinates must be canonical field elements.
            assert_eq!(
                curve.point_from_coordinates(curve.prime().clone(), y.clone()),
                Err(EcError::CoordinateOutOfRange)
            );
            assert_eq!(
                curve.point_from_coordinates(x, curve.prime().clone()),
                Err(EcError::CoordinateOutOfRange)
            );
        }
    }
}
