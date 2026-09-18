//! Deterministic nonces — [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979).
//!
//! DSA and ECDSA both need a per-signature secret `k`. Getting it wrong is
//! catastrophic in a way that is invisible from the outside: two signatures made
//! with the same `k`, or with `k` values that are merely *related*, hand the
//! private key to anyone holding both. That is not a theoretical concern — it is
//! how the PlayStation 3 and several Bitcoin wallets lost their keys.
//!
//! RFC 6979 removes the randomness instead of trying to get it right: `k` is
//! derived from the private key and the message with HMAC in an
//! [RFC 8017 HMAC\_DRBG](https://www.rfc-editor.org/rfc/rfc6979#section-3.1)
//! construction, so it is unpredictable to anyone who does not already know the
//! private key, and reproducible for anyone who does. Signatures become
//! deterministic — the same message signed twice yields the same octets — which
//! is also what makes them testable against the vectors in the RFC's appendix.
//!
//! This module implements §3.2 verbatim, with the three conversion primitives of
//! §2.3.2–§2.3.4 ([`bits2int`], `int2octets`, `bits2octets`) as separate
//! functions because their asymmetry is the part implementations get wrong:
//! `bits2int` truncates a digest that is *wider* than the group order and leaves
//! a *narrower* one alone, so a signature scheme that only ever pairs SHA-256
//! with a 256-bit order never exercises either branch.

use oxixml_support::bigint::BigUint;

use crate::algorithm::DigestAlgorithm;
use crate::crypto::digest::hmac;

/// How many candidate `k` values the generator will try before giving up.
///
/// [RFC 6979 §3.2 step h](https://www.rfc-editor.org/rfc/rfc6979#section-3.2)
/// loops until a candidate lands in `1..q`, which for the group orders in use
/// happens on the first try with probability better than `1 - 2^-32`. The bound
/// exists so that a degenerate order — one this crate should never see, but which
/// a caller could construct — cannot turn the loop into a hang.
const MAX_CANDIDATE_ROUNDS: usize = 1024;

/// `bits2int` from [RFC 6979 §2.3.2](https://www.rfc-editor.org/rfc/rfc6979#section-2.3.2).
///
/// Takes the leftmost `min(8 · bytes.len(), qlen)` bits of `bytes` as a
/// big-endian integer. When the input is wider than the group order the low bits
/// are dropped; when it is narrower it is used unchanged, *not* left-shifted.
///
/// This is also the digest-truncation rule of
/// [FIPS 186-4 §6.4](https://doi.org/10.6028/NIST.FIPS.186-4), which is why the
/// verification paths call it too.
pub(crate) fn bits2int(bytes: &[u8], qlen: usize) -> BigUint {
    let blen = bytes.len() * 8;
    if blen <= qlen {
        return BigUint::from_be_bytes(bytes);
    }
    BigUint::from_be_bytes(&shift_right(bytes, blen - qlen))
}

/// Shifts a big-endian magnitude right by `shift` bits.
fn shift_right(bytes: &[u8], shift: usize) -> Vec<u8> {
    let dropped_octets = shift / 8;
    let dropped_bits = shift % 8;
    let Some(kept) = bytes.get(..bytes.len().saturating_sub(dropped_octets)) else {
        return Vec::new();
    };
    if dropped_bits == 0 {
        return kept.to_vec();
    }
    let mut out = Vec::with_capacity(kept.len());
    let mut carry = 0_u8;
    for byte in kept {
        out.push((byte >> dropped_bits) | carry);
        carry = byte << (8 - dropped_bits);
    }
    out
}

/// `int2octets` from [RFC 6979 §2.3.3](https://www.rfc-editor.org/rfc/rfc6979#section-2.3.3):
/// the value as exactly `rlen` big-endian octets.
///
/// Returns [`None`] when the value does not fit, which for a scalar already
/// known to be below the group order cannot happen.
fn int2octets(value: &BigUint, rlen: usize) -> Option<Vec<u8>> {
    value.to_be_bytes_padded(rlen)
}

/// `bits2octets` from [RFC 6979 §2.3.4](https://www.rfc-editor.org/rfc/rfc6979#section-2.3.4):
/// `int2octets(bits2int(bytes) mod q)`.
fn bits2octets(bytes: &[u8], order: &BigUint, qlen: usize, rlen: usize) -> Option<Vec<u8>> {
    let reduced = bits2int(bytes, qlen).div_rem(order)?.1;
    int2octets(&reduced, rlen)
}

/// The HMAC\_DRBG of [RFC 6979 §3.2](https://www.rfc-editor.org/rfc/rfc6979#section-3.2),
/// seeded with a private key and a message digest.
///
/// Each call to [`next_nonce`](NonceGenerator::next_nonce) yields the next
/// candidate `k`. A caller that rejects one — because it made `r` or `s` come out
/// zero — simply asks for another, which is exactly the "loop" of step h.3.
pub(crate) struct NonceGenerator {
    /// The hash the HMAC is built on: the same one the signature uses.
    digest: DigestAlgorithm,
    /// The DRBG key `K`.
    k: Vec<u8>,
    /// The DRBG state `V`.
    v: Vec<u8>,
    /// `qlen`, the bit length of the group order.
    qlen: usize,
    /// Whether a candidate has already been handed out, in which case the state
    /// must be stepped before the next one is generated.
    exhausted: bool,
}

impl NonceGenerator {
    /// Seeds the generator, performing steps b–g of §3.2.
    ///
    /// `message_hash` is `H(m)`, the digest of the message being signed, *not*
    /// the message. Returns [`None`] if `private_key` is not below `order`,
    /// which the callers have already checked.
    pub(crate) fn new(
        digest: DigestAlgorithm,
        order: &BigUint,
        private_key: &BigUint,
        message_hash: &[u8],
    ) -> Option<Self> {
        let qlen = order.bit_len();
        let rlen = qlen.div_ceil(8);
        let hlen = digest.output_length();
        let private_octets = int2octets(private_key, rlen)?;
        let hash_octets = bits2octets(message_hash, order, qlen, rlen)?;

        // b. V = 0x01 0x01 … 0x01   c. K = 0x00 0x00 … 0x00
        let mut v = vec![0x01_u8; hlen];
        let mut k = vec![0x00_u8; hlen];
        // d. K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
        k = hmac(digest, &k, &seed(&v, 0x00, &private_octets, &hash_octets));
        // e. V = HMAC_K(V)
        v = hmac(digest, &k, &v);
        // f. K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
        k = hmac(digest, &k, &seed(&v, 0x01, &private_octets, &hash_octets));
        // g. V = HMAC_K(V)
        v = hmac(digest, &k, &v);

        Some(Self {
            digest,
            k,
            v,
            qlen,
            exhausted: false,
        })
    }

    /// The next candidate `k`, guaranteed to lie in `1..order`.
    ///
    /// Returns [`None`] only if [`MAX_CANDIDATE_ROUNDS`] candidates in a row fall
    /// outside the range.
    pub(crate) fn next_nonce(&mut self, order: &BigUint) -> Option<BigUint> {
        for _ in 0..MAX_CANDIDATE_ROUNDS {
            if self.exhausted {
                self.step();
            }
            let mut t = Vec::new();
            while t.len() * 8 < self.qlen {
                self.v = hmac(self.digest, &self.k, &self.v);
                t.extend_from_slice(&self.v);
            }
            self.exhausted = true;
            let candidate = bits2int(&t, self.qlen);
            if !candidate.is_zero() && &candidate < order {
                return Some(candidate);
            }
        }
        None
    }

    /// Step h.3: `K = HMAC_K(V || 0x00)`, `V = HMAC_K(V)`.
    fn step(&mut self) {
        let mut input = self.v.clone();
        input.push(0x00);
        self.k = hmac(self.digest, &self.k, &input);
        self.v = hmac(self.digest, &self.k, &self.v);
    }
}

/// `V || separator || int2octets(x) || bits2octets(h1)`, the input of steps d and f.
fn seed(v: &[u8], separator: u8, private_octets: &[u8], hash_octets: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() + 1 + private_octets.len() + hash_octets.len());
    out.extend_from_slice(v);
    out.push(separator);
    out.extend_from_slice(private_octets);
    out.extend_from_slice(hash_octets);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::hex;
    use oxixml_support::ec::Curve;

    /// The `k` RFC 6979 Appendix A.2.5 (P-256) and A.2.6 (P-384) publish for the
    /// message `"sample"`.
    ///
    /// The P-384 rows are the load-bearing ones: with SHA-1 and SHA-224 the
    /// digest is *narrower* than the 384-bit group order, so `bits2int` must use
    /// it unchanged. An implementation that shifted it left — or that truncated
    /// unconditionally — still reproduces every P-256 row.
    #[test]
    fn rfc_6979_appendix_a_2_nonces() {
        let p256 = Curve::p256();
        let p384 = Curve::p384();
        let x256 = hex("C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721");
        let x384 = hex(
            "6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D8\
             96D5724E4C70A825F872C9EA60D2EDF5",
        );

        let cases: [(&Curve, &[u8], DigestAlgorithm, &str); 10] = [
            (
                p256,
                &x256,
                DigestAlgorithm::Sha1,
                "882905F1227FD620FBF2ABF21244F0BA83D0DC3A9103DBBEE43A1FB858109DB4",
            ),
            (
                p256,
                &x256,
                DigestAlgorithm::Sha224,
                "103F90EE9DC52E5E7FB5132B7033C63066D194321491862059967C715985D473",
            ),
            (
                p256,
                &x256,
                DigestAlgorithm::Sha256,
                "A6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60",
            ),
            (
                p256,
                &x256,
                DigestAlgorithm::Sha384,
                "09F634B188CEFD98E7EC88B1AA9852D734D0BC272F7D2A47DECC6EBEB375AAD4",
            ),
            (
                p256,
                &x256,
                DigestAlgorithm::Sha512,
                "5FA81C63109BADB88C1F367B47DA606DA28CAD69AA22C4FE6AD7DF73A7173AA5",
            ),
            (
                p384,
                &x384,
                DigestAlgorithm::Sha1,
                "4471EF7518BB2C7C20F62EAE1C387AD0C5E8E470995DB4ACF694466E6AB09663\
                 0F29E5938D25106C3C340045A2DB01A7",
            ),
            (
                p384,
                &x384,
                DigestAlgorithm::Sha224,
                "A4E4D2F0E729EB786B31FC20AD5D849E304450E0AE8E3E341134A5C1AFA03CAB\
                 8083EE4E3C45B06A5899EA56C51B5879",
            ),
            (
                p384,
                &x384,
                DigestAlgorithm::Sha256,
                "180AE9F9AEC5438A44BC159A1FCB277C7BE54FA20E7CF404B490650A8ACC414E\
                 375572342863C899F9F2EDF9747A9B60",
            ),
            (
                p384,
                &x384,
                DigestAlgorithm::Sha384,
                "94ED910D1A099DAD3254E9242AE85ABDE4BA15168EAF0CA87A555FD56D10FBCA\
                 2907E3E83BA95368623B8C4686915CF9",
            ),
            (
                p384,
                &x384,
                DigestAlgorithm::Sha512,
                "92FC3C7183A883E24216D1141F1A8976C5B0DD797DFA597E3D7B32198BD35331\
                 A4E966532593A52980D0E3AAA5E10EC3",
            ),
        ];

        for (curve, private_key, digest, expected) in cases {
            let order = curve.order();
            let scalar = BigUint::from_be_bytes(private_key);
            let mut generator = NonceGenerator::new(
                digest,
                order,
                &scalar,
                &crate::crypto::digest::digest(digest, b"sample"),
            )
            .expect("the private key is in range");
            let nonce = generator.next_nonce(order).expect("a candidate");
            assert_eq!(
                nonce.to_be_bytes_padded(curve.field_bytes()),
                Some(hex(expected)),
                "{:?} with {digest}",
                curve.id()
            );
        }
    }

    #[test]
    fn bits2int_truncates_only_a_wider_input() {
        // Wider than the order: the low bits go.
        assert_eq!(bits2int(&[0xff, 0xff], 8), BigUint::from_u64(0xff));
        // Exactly as wide: unchanged.
        assert_eq!(bits2int(&[0xab, 0xcd], 16), BigUint::from_u64(0xabcd));
        // Narrower: unchanged, *not* shifted up.
        assert_eq!(bits2int(&[0xab, 0xcd], 32), BigUint::from_u64(0xabcd));
        // A shift that is not a whole number of octets.
        assert_eq!(bits2int(&[0b1010_1010, 0b1100_0000], 12), {
            BigUint::from_u64(0b1010_1010_1100)
        });
    }

    /// Two consecutive candidates must differ: the generator has to step its
    /// state between them, as step h.3 requires, rather than replaying `V`.
    #[test]
    fn successive_candidates_differ() {
        let curve = Curve::p256();
        let order = curve.order();
        let scalar = BigUint::from_u64(0x1234_5678);
        let mut generator = NonceGenerator::new(
            DigestAlgorithm::Sha256,
            order,
            &scalar,
            &crate::crypto::digest::digest(DigestAlgorithm::Sha256, b"message"),
        )
        .expect("the private key is in range");
        let first = generator.next_nonce(order).expect("a candidate");
        let second = generator.next_nonce(order).expect("another candidate");
        assert_ne!(first, second);
    }
}
