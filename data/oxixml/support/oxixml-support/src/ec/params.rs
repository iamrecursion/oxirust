//! Domain parameters of the NIST prime curves P-256 and P-384.
//!
//! The values are the ones standardised in [FIPS 186-4] appendix D.1.2 (and, identically, in
//! [SEC 2] as `secp256r1` and `secp384r1`). Each is given as a fixed-width big-endian octet
//! string of the curve's field width — 32 bytes for P-256, 48 for P-384 — which is exactly the
//! form [`BigUint::from_be_bytes`](crate::bigint::BigUint::from_be_bytes) and the SEC 1 point
//! encoding both work in.
//!
//! Both curves are defined by `y² = x³ + a·x + b` over the prime field `GF(p)` with `a = p - 3`,
//! and both have a prime group order `n` and a [cofactor](COFACTOR) of one, so every point on
//! the curve except the point at infinity generates the whole group.
//!
//! [FIPS 186-4]: https://doi.org/10.6028/NIST.FIPS.186-4
//! [SEC 2]: https://www.secg.org/sec2-v2.pdf

/// The cofactor `h = #E(GF(p)) / n` of both curves.
///
/// A cofactor of one means the curve group has exactly `n` elements, so there are no small
/// subgroups: validating that a point satisfies the curve equation is enough to know that its
/// order is `n`, and no additional `n·Q = O` check is needed on a received public key.
pub const COFACTOR: u64 = 1;

/// The P-256 field prime `p = 2²⁵⁶ - 2²²⁴ + 2¹⁹² + 2⁹⁶ - 1`.
pub const P256_PRIME: [u8; 32] =
    hex_bytes("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");

/// The P-256 curve coefficient `a`, which is `p - 3`.
pub const P256_A: [u8; 32] =
    hex_bytes("ffffffff00000001000000000000000000000000fffffffffffffffffffffffc");

/// The P-256 curve coefficient `b`.
pub const P256_B: [u8; 32] =
    hex_bytes("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");

/// The x coordinate of the P-256 base point `G`.
pub const P256_GENERATOR_X: [u8; 32] =
    hex_bytes("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");

/// The y coordinate of the P-256 base point `G`.
pub const P256_GENERATOR_Y: [u8; 32] =
    hex_bytes("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");

/// The order `n` of the P-256 base point, a 256-bit prime.
pub const P256_ORDER: [u8; 32] =
    hex_bytes("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");

/// The P-384 field prime `p = 2³⁸⁴ - 2¹²⁸ - 2⁹⁶ + 2³² - 1`.
pub const P384_PRIME: [u8; 48] = hex_bytes(concat!(
    "ffffffffffffffffffffffffffffffffffffffffffffffff",
    "fffffffffffffffeffffffff0000000000000000ffffffff"
));

/// The P-384 curve coefficient `a`, which is `p - 3`.
pub const P384_A: [u8; 48] = hex_bytes(concat!(
    "ffffffffffffffffffffffffffffffffffffffffffffffff",
    "fffffffffffffffeffffffff0000000000000000fffffffc"
));

/// The P-384 curve coefficient `b`.
pub const P384_B: [u8; 48] = hex_bytes(concat!(
    "b3312fa7e23ee7e4988e056be3f82d19181d9c6efe814112",
    "0314088f5013875ac656398d8a2ed19d2a85c8edd3ec2aef"
));

/// The x coordinate of the P-384 base point `G`.
pub const P384_GENERATOR_X: [u8; 48] = hex_bytes(concat!(
    "aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b98",
    "59f741e082542a385502f25dbf55296c3a545e3872760ab7"
));

/// The y coordinate of the P-384 base point `G`.
pub const P384_GENERATOR_Y: [u8; 48] = hex_bytes(concat!(
    "3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147c",
    "e9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f"
));

/// The order `n` of the P-384 base point, a 384-bit prime.
pub const P384_ORDER: [u8; 48] = hex_bytes(concat!(
    "ffffffffffffffffffffffffffffffffffffffffffffffff",
    "c7634d81f4372ddf581a0db248b0a77aecec196accc52973"
));

/// Decode a lowercase hexadecimal literal into a fixed-size byte array.
///
/// This runs during constant evaluation for every parameter above, so a mistyped literal is a
/// compilation error rather than anything a caller could ever observe at run time.
const fn hex_bytes<const N: usize>(hex: &str) -> [u8; N] {
    let digits = hex.as_bytes();
    assert!(
        digits.len() == 2 * N,
        "curve parameter literal has the wrong width"
    );
    let mut out = [0u8; N];
    let mut index = 0;
    while index < N {
        out[index] = (hex_digit(digits[2 * index]) << 4) | hex_digit(digits[2 * index + 1]);
        index += 1;
    }
    out
}

/// The numeric value of one lowercase hexadecimal digit.
const fn hex_digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("curve parameter literal contains a non-hexadecimal digit"),
    }
}
