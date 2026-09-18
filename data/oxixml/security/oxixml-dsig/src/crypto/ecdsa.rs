//! ECDSA over the NIST prime curves — [FIPS 186-4 §6.4](https://doi.org/10.6028/NIST.FIPS.186-4).
//!
//! The curve arithmetic lives in [`oxixml_support::ec`]; what is here is the
//! signature scheme on top of it, plus the two conversions that are easy to get
//! subtly wrong:
//!
//! * **digest truncation.** The digest is turned into a scalar by taking its
//!   leftmost `min(hashlen, orderlen)` *bits* — never the rightmost, never
//!   padded on the left. Both supported orders are a whole number of octets
//!   wide, so this is a byte truncation in practice, but it is implemented as
//!   the general bit rule in [`crate::crypto::rfc6979::bits2int`] so that
//!   SHA-1 with P-384 (a digest *narrower* than the order) is handled by the
//!   same code path as SHA-512 with P-256.
//! * **the signature value.**
//!   [XML Signature §6.4.3](https://www.w3.org/TR/xmldsig-core1/#sec-ECDSA)
//!   transmits `r` and `s` as the plain concatenation of two `I2OSP` encodings,
//!   each exactly as wide as the base point order — 32 octets for P-256, 48 for
//!   P-384. It is emphatically **not** the ASN.1 `SEQUENCE { r, s }` that X.509
//!   and TLS use, and a verifier that accepted DER here would accept signatures
//!   with attacker-chosen padding around them.
//!
//! # Timing
//!
//! Verification runs entirely on public values. Signing multiplies the base
//! point by a secret nonce through [`Curve::mul_base`], a Montgomery ladder over
//! [`BigUint`] arithmetic that is itself variable time — so this is not a
//! constant-time implementation, and a caller who must resist an attacker
//! measuring the signing process needs hardened arithmetic this workspace does
//! not provide. The nonce is derived deterministically
//! ([RFC 6979](https://www.rfc-editor.org/rfc/rfc6979)), which removes the far
//! more common failure of a repeated or biased `k`.

use oxixml_support::bigint::BigUint;
use oxixml_support::ec::{AffinePoint, Curve};

use crate::algorithm::DigestAlgorithm;
use crate::crypto::digest;
use crate::crypto::rfc6979::{NonceGenerator, bits2int};

/// How many nonces are tried before signing gives up.
///
/// A candidate is rejected only when it makes `r` or `s` come out zero, which
/// happens with probability around `2^-128`; the bound turns an impossible loop
/// into a bounded failure rather than a hang.
const MAX_SIGNATURE_ATTEMPTS: usize = 32;

/// Converts a message to the scalar `e` that the signature equations use.
fn message_scalar(curve: &Curve, algorithm: DigestAlgorithm, message: &[u8]) -> Option<BigUint> {
    let order = curve.order();
    let hash = digest::digest(algorithm, message);
    Some(bits2int(&hash, order.bit_len()).div_rem(order)?.1)
}

/// ECDSA verification.
///
/// `signature` is the raw `r ‖ s` of XML Signature §6.4.3; anything of a
/// different length, or with either half outside `1..n`, is rejected before any
/// curve arithmetic runs.
#[must_use]
pub(crate) fn verify(
    curve: &Curve,
    public_key: &AffinePoint,
    algorithm: DigestAlgorithm,
    message: &[u8],
    signature: &[u8],
) -> bool {
    let width = curve.field_bytes();
    let (Some(r_octets), Some(s_octets)) =
        (signature.get(..width), signature.get(width..2 * width))
    else {
        return false;
    };
    if signature.len() != 2 * width {
        return false;
    }
    // `scalar_from_be_bytes` enforces both the width and the 1..n range.
    let (Ok(r), Ok(s)) = (
        curve.scalar_from_be_bytes(r_octets),
        curve.scalar_from_be_bytes(s_octets),
    ) else {
        return false;
    };
    // The public key is validated where it is parsed; re-checking here keeps
    // this function safe to call with a point from anywhere.
    if public_key.is_identity() || !curve.is_on_curve(public_key) {
        return false;
    }

    let order = curve.order();
    let (Some(e), Some(w)) = (
        message_scalar(curve, algorithm, message),
        s.mod_inverse(order),
    ) else {
        return false;
    };
    let (Some(u1), Some(u2)) = (e.mul_mod(&w, order), r.mul_mod(&w, order)) else {
        return false;
    };
    let point = curve.double_scalar_mul(&u1, &u2, public_key);
    let Some(x) = point.x() else {
        // R is the point at infinity: the signature is invalid.
        return false;
    };
    let Some((_, v)) = x.div_rem(order) else {
        return false;
    };
    v == r
}

/// ECDSA signing with the deterministic nonce of RFC 6979.
///
/// Returns the raw `r ‖ s` of XML Signature §6.4.3, or [`None`] if
/// `private_key` is not a scalar in `1..n`.
#[must_use]
pub(crate) fn sign(
    curve: &Curve,
    private_key: &BigUint,
    algorithm: DigestAlgorithm,
    message: &[u8],
) -> Option<Vec<u8>> {
    let order = curve.order();
    if !curve.is_valid_scalar(private_key) {
        return None;
    }
    let hash = digest::digest(algorithm, message);
    let e = bits2int(&hash, order.bit_len()).div_rem(order)?.1;
    let mut nonces = NonceGenerator::new(algorithm, order, private_key, &hash)?;

    for _ in 0..MAX_SIGNATURE_ATTEMPTS {
        let k = nonces.next_nonce(order)?;
        let point = curve.mul_base(&k);
        let Some(x) = point.x() else {
            continue;
        };
        let r = x.div_rem(order)?.1;
        if r.is_zero() {
            continue;
        }
        let s = {
            let k_inverse = k.mod_inverse(order)?;
            let masked = r.mul_mod(private_key, order)?;
            k_inverse.mul_mod(&e.add_mod(&masked, order)?, order)?
        };
        if s.is_zero() {
            continue;
        }
        let mut value = curve.encode_scalar(&r).ok()?;
        value.extend_from_slice(&curve.encode_scalar(&s).ok()?);
        return Some(value);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::hex;
    use oxixml_support::ec::CurveId;

    /// The P-256 private key of RFC 6979 Appendix A.2.5.
    const P256_PRIVATE_KEY: &str =
        "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721";

    /// The P-384 private key of RFC 6979 Appendix A.2.6.
    const P384_PRIVATE_KEY: &str = "\
        6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D8\
        96D5724E4C70A825F872C9EA60D2EDF5";

    /// `(digest, message, r, s)` — every row of RFC 6979 Appendix A.2.5.
    const P256_VECTORS: [(DigestAlgorithm, &[u8], &str, &str); 10] = [
        (
            DigestAlgorithm::Sha1,
            b"sample",
            "61340C88C3AAEBEB4F6D667F672CA9759A6CCAA9FA8811313039EE4A35471D32",
            "6D7F147DAC089441BB2E2FE8F7A3FA264B9C475098FDCF6E00D7C996E1B8B7EB",
        ),
        (
            DigestAlgorithm::Sha224,
            b"sample",
            "53B2FFF5D1752B2C689DF257C04C40A587FABABB3F6FC2702F1343AF7CA9AA3F",
            "B9AFB64FDC03DC1A131C7D2386D11E349F070AA432A4ACC918BEA988BF75C74C",
        ),
        (
            DigestAlgorithm::Sha256,
            b"sample",
            "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
            "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8",
        ),
        (
            DigestAlgorithm::Sha384,
            b"sample",
            "0EAFEA039B20E9B42309FB1D89E213057CBF973DC0CFC8F129EDDDC800EF7719",
            "4861F0491E6998B9455193E34E7B0D284DDD7149A74B95B9261F13ABDE940954",
        ),
        (
            DigestAlgorithm::Sha512,
            b"sample",
            "8496A60B5E9B47C825488827E0495B0E3FA109EC4568FD3F8D1097678EB97F00",
            "2362AB1ADBE2B8ADF9CB9EDAB740EA6049C028114F2460F96554F61FAE3302FE",
        ),
        (
            DigestAlgorithm::Sha1,
            b"test",
            "0CBCC86FD6ABD1D99E703E1EC50069EE5C0B4BA4B9AC60E409E8EC5910D81A89",
            "01B9D7B73DFAA60D5651EC4591A0136F87653E0FD780C3B1BC872FFDEAE479B1",
        ),
        (
            DigestAlgorithm::Sha224,
            b"test",
            "C37EDB6F0AE79D47C3C27E962FA269BB4F441770357E114EE511F662EC34A692",
            "C820053A05791E521FCAAD6042D40AEA1D6B1A540138558F47D0719800E18F2D",
        ),
        (
            DigestAlgorithm::Sha256,
            b"test",
            "F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
            "019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083",
        ),
        (
            DigestAlgorithm::Sha384,
            b"test",
            "83910E8B48BB0C74244EBDF7F07A1C5413D61472BD941EF3920E623FBCCEBEB6",
            "8DDBEC54CF8CD5874883841D712142A56A8D0F218F5003CB0296B6B509619F2C",
        ),
        (
            DigestAlgorithm::Sha512,
            b"test",
            "461D93F31B6540894788FD206C07CFA0CC35F46FA3C91816FFF1040AD1581A04",
            "39AF9F15DE0DB8D97E72719C74820D304CE5226E32DEDAE67519E840D1194E55",
        ),
    ];

    /// `(digest, message, r, s)` — every row of RFC 6979 Appendix A.2.6.
    const P384_VECTORS: [(DigestAlgorithm, &[u8], &str, &str); 10] = [
        (
            DigestAlgorithm::Sha1,
            b"sample",
            "EC748D839243D6FBEF4FC5C4859A7DFFD7F3ABDDF72014540C16D73309834FA3\
             7B9BA002899F6FDA3A4A9386790D4EB2",
            "A3BCFA947BEEF4732BF247AC17F71676CB31A847B9FF0CBC9C9ED4C1A5B3FACF\
             26F49CA031D4857570CCB5CA4424A443",
        ),
        (
            DigestAlgorithm::Sha224,
            b"sample",
            "42356E76B55A6D9B4631C865445DBE54E056D3B3431766D0509244793C3F9366\
             450F76EE3DE43F5A125333A6BE060122",
            "9DA0C81787064021E78DF658F2FBB0B042BF304665DB721F077A4298B095E483\
             4C082C03D83028EFBF93A3C23940CA8D",
        ),
        (
            DigestAlgorithm::Sha256,
            b"sample",
            "21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33\
             BDE1E888E63355D92FA2B3C36D8FB2CD",
            "F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEB\
             EFDC63ECCD1AC42EC0CB8668A4FA0AB0",
        ),
        (
            DigestAlgorithm::Sha384,
            b"sample",
            "94EDBB92A5ECB8AAD4736E56C691916B3F88140666CE9FA73D64C4EA95AD133C\
             81A648152E44ACF96E36DD1E80FABE46",
            "99EF4AEB15F178CEA1FE40DB2603138F130E740A19624526203B6351D0A3A94F\
             A329C145786E679E7B82C71A38628AC8",
        ),
        (
            DigestAlgorithm::Sha512,
            b"sample",
            "ED0959D5880AB2D869AE7F6C2915C6D60F96507F9CB3E047C0046861DA4A799C\
             FE30F35CC900056D7C99CD7882433709",
            "512C8CCEEE3890A84058CE1E22DBC2198F42323CE8ACA9135329F03C068E5112\
             DC7CC3EF3446DEFCEB01A45C2667FDD5",
        ),
        (
            DigestAlgorithm::Sha1,
            b"test",
            "4BC35D3A50EF4E30576F58CD96CE6BF638025EE624004A1F7789A8B8E43D0678\
             ACD9D29876DAF46638645F7F404B11C7",
            "D5A6326C494ED3FF614703878961C0FDE7B2C278F9A65FD8C4B7186201A29916\
             95BA1C84541327E966FA7B50F7382282",
        ),
        (
            DigestAlgorithm::Sha224,
            b"test",
            "E8C9D0B6EA72A0E7837FEA1D14A1A9557F29FAA45D3E7EE888FC5BF954B5E624\
             64A9A817C47FF78B8C11066B24080E72",
            "07041D4A7A0379AC7232FF72E6F77B6DDB8F09B16CCE0EC3286B2BD43FA8C614\
             1C53EA5ABEF0D8231077A04540A96B66",
        ),
        (
            DigestAlgorithm::Sha256,
            b"test",
            "6D6DEFAC9AB64DABAFE36C6BF510352A4CC27001263638E5B16D9BB51D451559\
             F918EEDAF2293BE5B475CC8F0188636B",
            "2D46F3BECBCC523D5F1A1256BF0C9B024D879BA9E838144C8BA6BAEB4B53B47D\
             51AB373F9845C0514EEFB14024787265",
        ),
        (
            DigestAlgorithm::Sha384,
            b"test",
            "8203B63D3C853E8D77227FB377BCF7B7B772E97892A80F36AB775D509D7A5FEB\
             0542A7F0812998DA8F1DD3CA3CF023DB",
            "DDD0760448D42D8A43AF45AF836FCE4DE8BE06B485E9B61B827C2F13173923E0\
             6A739F040649A667BF3B828246BAA5A5",
        ),
        (
            DigestAlgorithm::Sha512,
            b"test",
            "A0D5D090C9980FAF3C2CE57B7AE951D31977DD11C775D314AF55F76C676447D0\
             6FB6495CD21B4B6E340FC236584FB277",
            "976984E59B4C77B0E8E4460DCA3D9F20E07B9BB1F63BEEFAF576F6B2E8B22463\
             4A2092CD3792E0159AD9CEE37659C736",
        ),
    ];

    /// Signing must reproduce the published `(r, s)` exactly — which checks the
    /// RFC 6979 nonce, the digest truncation and the signature equation all at
    /// once — and the result must verify under the matching public key.
    ///
    /// The P-384 rows with SHA-1 and SHA-224 are the ones that discriminate: the
    /// digest is *narrower* than the group order there, so no truncation happens
    /// and an implementation that shifted instead would fail only on these.
    #[test]
    fn rfc_6979_signatures_are_reproduced_and_verify() {
        let cases = [
            (CurveId::P256, P256_PRIVATE_KEY, P256_VECTORS.as_slice()),
            (CurveId::P384, P384_PRIVATE_KEY, P384_VECTORS.as_slice()),
        ];
        for (id, private_key, vectors) in cases {
            let curve = id.curve();
            let scalar = BigUint::from_be_bytes(&hex(private_key));
            let public_key = curve.mul_base(&scalar);
            for (index, (algorithm, message, r, s)) in vectors.iter().enumerate() {
                let signature =
                    sign(curve, &scalar, *algorithm, message).expect("the key is in range");
                let mut expected = hex(r);
                expected.extend_from_slice(&hex(s));
                assert_eq!(
                    signature,
                    expected,
                    "{id} with {algorithm} over {:?}",
                    std::str::from_utf8(message)
                );
                assert!(verify(curve, &public_key, *algorithm, message, &signature));
                // One negative per curve is enough here; the dedicated tests
                // below cover the rejection paths, and each verification is an
                // expensive double scalar multiplication.
                if index == 0 {
                    assert!(!verify(
                        curve,
                        &public_key,
                        *algorithm,
                        b"a different message",
                        &signature
                    ));
                }
            }
        }
    }

    /// The public keys RFC 6979 publishes alongside the private ones, as a check
    /// that base point multiplication agrees with the RFC.
    #[test]
    fn rfc_6979_public_keys_match() {
        let p256 = Curve::p256();
        let public = p256.mul_base(&BigUint::from_be_bytes(&hex(P256_PRIVATE_KEY)));
        assert_eq!(
            p256.encode_point(&public).ok(),
            Some(hex("04\
                 60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6\
                 7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299"))
        );

        let p384 = Curve::p384();
        let public = p384.mul_base(&BigUint::from_be_bytes(&hex(P384_PRIVATE_KEY)));
        assert_eq!(
            p384.encode_point(&public).ok(),
            Some(hex("04\
                 EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64\
                 DEF8F0EA9055866064A254515480BC13\
                 8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1\
                 288B231C3AE0D4FE7344FD2533264720"))
        );
    }

    /// XML Signature §6.4.3 fixes the signature value at two `I2OSP` encodings of
    /// the order's width. Nothing else may be accepted — in particular not the
    /// ASN.1 `SEQUENCE { r, s }` that X.509 and TLS use, which is the encoding a
    /// generic ECDSA library would hand over.
    #[test]
    fn a_signature_of_the_wrong_length_is_rejected() {
        let curve = Curve::p256();
        let scalar = BigUint::from_be_bytes(&hex(P256_PRIVATE_KEY));
        let public_key = curve.mul_base(&scalar);
        let signature =
            sign(curve, &scalar, DigestAlgorithm::Sha256, b"sample").expect("the key is in range");
        assert_eq!(signature.len(), 64);

        let mut der = vec![0x30, 0x44, 0x02, 0x20];
        der.extend_from_slice(signature.get(..32).unwrap_or_default());
        der.extend_from_slice(&[0x02, 0x20]);
        der.extend_from_slice(signature.get(32..).unwrap_or_default());
        assert!(!verify(
            curve,
            &public_key,
            DigestAlgorithm::Sha256,
            b"sample",
            &der
        ));

        assert!(!verify(
            curve,
            &public_key,
            DigestAlgorithm::Sha256,
            b"sample",
            signature.get(..63).unwrap_or_default()
        ));
        let mut padded = vec![0_u8];
        padded.extend_from_slice(&signature);
        assert!(!verify(
            curve,
            &public_key,
            DigestAlgorithm::Sha256,
            b"sample",
            &padded
        ));
    }

    #[test]
    fn a_component_outside_the_group_order_is_rejected() {
        let curve = Curve::p256();
        let scalar = BigUint::from_be_bytes(&hex(P256_PRIVATE_KEY));
        let public_key = curve.mul_base(&scalar);
        let width = curve.field_bytes();
        let order = curve
            .order()
            .to_be_bytes_padded(width)
            .expect("the order is 256 bits");

        let mut zero_r = vec![0_u8; width];
        zero_r.extend_from_slice(&order);
        assert!(!verify(
            curve,
            &public_key,
            DigestAlgorithm::Sha256,
            b"sample",
            &zero_r
        ));

        let mut order_s = order.clone();
        order_s.extend_from_slice(&order);
        assert!(!verify(
            curve,
            &public_key,
            DigestAlgorithm::Sha256,
            b"sample",
            &order_s
        ));
    }

    #[test]
    fn a_signature_does_not_verify_under_a_different_key() {
        let curve = Curve::p256();
        let scalar = BigUint::from_be_bytes(&hex(P256_PRIVATE_KEY));
        let signature =
            sign(curve, &scalar, DigestAlgorithm::Sha256, b"sample").expect("the key is in range");
        let other = curve.mul_base(&BigUint::from_u64(12_345));
        assert!(!verify(
            curve,
            &other,
            DigestAlgorithm::Sha256,
            b"sample",
            &signature
        ));
    }

    #[test]
    fn the_identity_is_never_a_usable_public_key() {
        let curve = Curve::p256();
        let scalar = BigUint::from_be_bytes(&hex(P256_PRIVATE_KEY));
        let signature =
            sign(curve, &scalar, DigestAlgorithm::Sha256, b"sample").expect("the key is in range");
        assert!(!verify(
            curve,
            &AffinePoint::identity(),
            DigestAlgorithm::Sha256,
            b"sample",
            &signature
        ));
    }

    #[test]
    fn a_private_key_outside_the_group_order_is_refused() {
        let curve = Curve::p256();
        assert!(sign(curve, &BigUint::zero(), DigestAlgorithm::Sha256, b"sample").is_none());
        assert!(sign(curve, curve.order(), DigestAlgorithm::Sha256, b"sample").is_none());
    }
}
