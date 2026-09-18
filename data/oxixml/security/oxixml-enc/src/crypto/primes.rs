//! Primality testing and prime generation, as FIPS 186-5 Appendix B.3 specifies them.
//!
//! This exists for one caller — [`RsaPrivateKey::generate`](crate::crypto::rsa::RsaPrivateKey::generate)
//! — but it is a standard of its own, so it lives in its own module and is tested against its
//! own known answers.
//!
//! # How a prime is found
//!
//! Random candidates, sieved and then tested:
//!
//! 1. Draw a random odd candidate of the requested width with its **top two bits set**, so that
//!    the product of two such primes has exactly twice the width and no leading zero octet.
//! 2. Reduce it modulo every prime below [`TRIAL_DIVISION_LIMIT`] once, then walk a window of
//!    consecutive odd offsets, updating those residues by addition instead of by division. A
//!    candidate divisible by any small prime is discarded before any exponentiation runs. Most
//!    candidates die here, and they die for the price of an addition.
//! 3. Test what survives with Miller–Rabin (Appendix B.3.1) for [`miller_rabin_rounds`] rounds
//!    with random bases.
//!
//! # Timing
//!
//! Nothing here is constant-time, and it cannot be: the *number* of candidates rejected before a
//! prime is found is data, and it is observable in the running time. That is inherent to
//! probabilistic prime generation, and it is why key generation belongs offline, on values that
//! are about to become secret, rather than in a request handler.

use std::sync::OnceLock;

use oxixml_support::bigint::BigUint;

use crate::error::{EncError, Result};
use crate::random::RandomSource;

/// Candidates are divided by every prime below this limit before any exponentiation runs.
///
/// FIPS 186-5 Appendix B.3 asks for a limit between 10³ and 10⁵. Two thousand is at the low end
/// deliberately: the sieve makes each additional prime almost free during the walk, but the
/// initial reduction costs one division apiece, and the fraction of composites removed grows
/// only as `1/ln(limit)`.
pub const TRIAL_DIVISION_LIMIT: u32 = 2048;

/// How many consecutive odd numbers the sieve walks before drawing a fresh candidate.
///
/// Primes near `2^k` are about `k · ln 2 / 2` odd numbers apart on average — 177 of them for a
/// 512-bit prime — so a window of a thousand odd numbers almost always contains one, and the
/// window is far narrower than the candidate itself, so walking it cannot change the bit length.
const SIEVE_WINDOW: u64 = 2048;

/// The narrowest prime this module will generate.
///
/// Below this the top-two-bits-set discipline and the sieve window stop making sense, and
/// nothing in this crate wants one.
pub const MIN_PRIME_BITS: usize = 64;

/// The widest prime this module will generate.
///
/// A modulus is the product of two primes, and modular multiplication needs room for the
/// intermediate product, so the modulus cannot be wider than half of [`BigUint::MAX_BITS`] —
/// which caps each prime at a quarter of it.
pub const MAX_PRIME_BITS: usize = BigUint::MAX_BITS / 4;

/// How many candidate draws [`generate_prime`] will make, and how many times a Miller–Rabin base
/// will be redrawn, before giving up.
///
/// Each draw sieves a whole window, so exhausting even a handful of them means the random source
/// is not producing what it claims to.
const MAX_DRAWS: usize = 64;

/// The primes below [`TRIAL_DIVISION_LIMIT`], sieved once on first use.
fn small_primes() -> &'static [u32] {
    static PRIMES: OnceLock<Vec<u32>> = OnceLock::new();
    PRIMES.get_or_init(|| {
        let limit = TRIAL_DIVISION_LIMIT as usize;
        let mut is_composite = vec![false; limit];
        let mut primes = Vec::new();
        for candidate in 2..limit {
            if is_composite[candidate] {
                continue;
            }
            primes.push(candidate as u32);
            let mut multiple = candidate * candidate;
            while multiple < limit {
                is_composite[multiple] = true;
                multiple += candidate;
            }
        }
        primes
    })
}

/// The minimum number of Miller–Rabin rounds to run on a prime of `prime_bits` bits.
///
/// The three tabulated sizes are FIPS 186-5 Table B.1's stronger column — the one whose target
/// error probability is 2⁻¹¹² for 1024-bit primes, 2⁻¹²⁸ for 1536-bit and 2⁻¹⁴⁴ for 2048-bit.
/// Those counts are small because the table assumes the candidate was drawn at random, which
/// makes a composite far less likely to survive than the worst case does.
///
/// The table does not cover the shorter primes a 1024-bit or 512-bit modulus needs, and
/// Appendix C's procedure for computing a count rests on that same distributional argument.
/// Rather than extrapolate it, this uses 50 rounds below 1024 bits: the distribution-free
/// Monier–Rabin bound says `t` rounds leave at most a `4⁻ᵗ` chance of accepting *any* odd
/// composite, so 50 rounds reach 2⁻¹⁰⁰ while assuming nothing about how the candidate was
/// chosen. It costs more exponentiations than an extrapolated count would; a key is generated
/// once.
#[must_use]
pub const fn miller_rabin_rounds(prime_bits: usize) -> usize {
    if prime_bits >= 1536 {
        4
    } else if prime_bits >= 1024 {
        5
    } else {
        50
    }
}

/// Returns `value` as a `u64`, or `None` when it does not fit.
fn narrow(value: &BigUint) -> Option<u64> {
    let bytes = value.to_be_bytes();
    if bytes.len() > 8 {
        return None;
    }
    let mut buffer = [0u8; 8];
    buffer[8 - bytes.len()..].copy_from_slice(&bytes);
    Some(u64::from_be_bytes(buffer))
}

/// Returns `value` reduced modulo the small `divisor`.
fn small_remainder(value: &BigUint, divisor: u32) -> Result<u64> {
    let (_, remainder) = value
        .div_rem(&BigUint::from_u64(u64::from(divisor)))
        .ok_or(EncError::InvalidRsaKey("a trial division failed"))?;
    narrow(&remainder).ok_or(EncError::InvalidRsaKey("a trial division failed"))
}

/// Returns `true` when `candidate` has no prime factor below [`TRIAL_DIVISION_LIMIT`].
///
/// A candidate that *is* one of those small primes is reported as prime.
///
/// # Errors
///
/// Returns [`EncError::InvalidRsaKey`] when the candidate is too wide for the arithmetic to
/// divide, which [`MAX_PRIME_BITS`] already rules out for anything this module produced.
pub fn passes_trial_division(candidate: &BigUint) -> Result<bool> {
    for prime in small_primes() {
        if small_remainder(candidate, *prime)? == 0 {
            // Divisible by this prime: composite, unless the candidate *is* this prime.
            return Ok(candidate == &BigUint::from_u64(u64::from(*prime)));
        }
    }
    Ok(true)
}

/// Writes `value = 2^a · m` with `m` odd and returns `(a, m)`.
fn factor_out_twos(value: &BigUint) -> Result<(usize, BigUint)> {
    let bit_len = value.bit_len();
    let mut twos = 0usize;
    while twos < bit_len && !value.bit(twos) {
        twos += 1;
    }
    if twos == 0 {
        return Ok((0, value.clone()));
    }
    // `BigUint` has no shift, so divide by `2^twos` built from its own bit pattern.
    let mut power_bytes = vec![0u8; twos / 8 + 1];
    let index = power_bytes.len() - 1 - twos / 8;
    power_bytes[index] = 1u8 << (twos % 8);
    let power = BigUint::from_be_bytes(&power_bytes);
    let (odd, _) = value.div_rem(&power).ok_or(EncError::InvalidRsaKey(
        "factoring out powers of two failed",
    ))?;
    Ok((twos, odd))
}

/// Draws a base uniformly from `2..=upper` for one Miller–Rabin round.
///
/// Rejection sampling, as FIPS 186-5 Appendix B.3.1 step 4.1 describes it: draw, and redraw when
/// the value is outside the interval. The draw is masked down to the candidate's **bit** width
/// rather than its octet width, so at least a quarter of all draws land in range whatever the
/// candidate is. Sampling at octet width would be fine for a 512-bit prime and hopeless for a
/// one-octet one, where only two of 256 values are in range.
fn draw_base(upper: &BigUint, bits: usize, rng: &mut dyn RandomSource) -> Result<BigUint> {
    let two = BigUint::from_u64(2);
    let width = bits.div_ceil(8);
    let mask = 0xffu8 >> (width * 8 - bits);
    let mut bytes = vec![0u8; width];
    for _ in 0..MAX_DRAWS {
        rng.fill(&mut bytes)?;
        if let Some(first) = bytes.first_mut() {
            *first &= mask;
        }
        let base = BigUint::from_be_bytes(&bytes);
        if base >= two && &base <= upper {
            return Ok(base);
        }
    }
    Err(EncError::InvalidRsaKey(
        "the random source did not produce a Miller-Rabin base in range",
    ))
}

/// The Miller–Rabin probabilistic primality test of FIPS 186-5 Appendix B.3.1.
///
/// Returns `false` as soon as a base proves `candidate` composite, and `true` when `rounds`
/// random bases have all failed to. A `true` answer is a probability, not a proof; see
/// [`miller_rabin_rounds`] for how many rounds make it a good enough one.
///
/// # Errors
///
/// Returns [`EncError::InvalidRsaKey`] when `candidate` is too wide for the modular arithmetic,
/// and [`EncError::Entropy`] when the random source fails.
pub fn is_probably_prime(
    candidate: &BigUint,
    rounds: usize,
    rng: &mut dyn RandomSource,
) -> Result<bool> {
    let three = BigUint::from_u64(3);
    if candidate <= &three {
        // Two and three are prime; zero and one are not.
        return Ok(candidate == &BigUint::from_u64(2) || candidate == &three);
    }
    if !candidate.bit(0) {
        return Ok(false);
    }
    // `mul_mod` needs room for the intermediate product, so a candidate wider than half of
    // `MAX_BITS` cannot be squared and is refused rather than silently mis-tested.
    if candidate.bit_len() > BigUint::MAX_BITS / 2 {
        return Err(EncError::InvalidRsaKey(
            "the candidate is too wide to test for primality",
        ));
    }

    let one = BigUint::one();
    let minus_one = candidate
        .checked_sub(&one)
        .ok_or(EncError::InvalidRsaKey("the candidate is zero"))?;
    let upper = minus_one
        .checked_sub(&one)
        .ok_or(EncError::InvalidRsaKey("the candidate is one"))?;
    let (twos, odd) = factor_out_twos(&minus_one)?;
    let bits = candidate.bit_len();

    'rounds: for _ in 0..rounds {
        let base = draw_base(&upper, bits, rng)?;
        let mut value = base
            .modpow(&odd, candidate)
            .ok_or(EncError::InvalidRsaKey("modular exponentiation failed"))?;
        if value == one || value == minus_one {
            continue;
        }
        for _ in 1..twos {
            value = value
                .mul_mod(&value, candidate)
                .ok_or(EncError::InvalidRsaKey("modular squaring failed"))?;
            if value == minus_one {
                continue 'rounds;
            }
            if value == one {
                // A square root of one that is neither 1 nor -1: composite, with a factor of the
                // candidate within reach of a greatest common divisor.
                return Ok(false);
            }
        }
        return Ok(false);
    }
    Ok(true)
}

/// Generates a random prime of exactly `bits` bits with its **top two bits set**.
///
/// The top two bits matter: two primes each above `3·2^(k-2)` have a product above `2^(2k-1)`,
/// so a modulus built from them has exactly `2k` bits and a fixed octet length. It also
/// satisfies FIPS 186-5's requirement that each prime exceed `√2 · 2^(k-1)`.
///
/// # Errors
///
/// Returns [`EncError::InvalidRsaKey`] when `bits` is outside `MIN_PRIME_BITS..=MAX_PRIME_BITS`
/// or when the search exhausts its candidate draws, and [`EncError::Entropy`] when the random
/// source fails.
pub fn generate_prime(bits: usize, rounds: usize, rng: &mut dyn RandomSource) -> Result<BigUint> {
    if !(MIN_PRIME_BITS..=MAX_PRIME_BITS).contains(&bits) {
        return Err(EncError::InvalidRsaKey(
            "the requested prime width is out of range",
        ));
    }

    let width = bits.div_ceil(8);
    let excess = width * 8 - bits;
    let mut bytes = vec![0u8; width];
    let primes = small_primes();
    let mut residues = vec![0u64; primes.len()];

    for _ in 0..MAX_DRAWS {
        rng.fill(&mut bytes)?;
        // Force the exact bit length, the top two bits and oddness. Every width this crate asks
        // for is a whole number of octets, but the general case is handled so that an odd width
        // cannot silently yield a narrower prime.
        if let Some(first) = bytes.first_mut() {
            *first &= 0xffu8 >> excess;
            *first |= 0xc0u8 >> excess;
        }
        if let Some(last) = bytes.last_mut() {
            *last |= 1;
        }
        let base = BigUint::from_be_bytes(&bytes);
        if base.bit_len() != bits {
            continue;
        }

        for (residue, prime) in residues.iter_mut().zip(primes) {
            *residue = small_remainder(&base, *prime)?;
        }

        let mut offset = 0u64;
        while offset < SIEVE_WINDOW {
            let divisible = residues.iter().zip(primes).any(|(residue, prime)| {
                let prime = u64::from(*prime);
                (*residue + offset) % prime == 0
            });
            if !divisible {
                let candidate = base
                    .checked_add(&BigUint::from_u64(offset))
                    .ok_or(EncError::InvalidRsaKey("a candidate overflowed"))?;
                // Walking the window must never widen the candidate; the window is far narrower
                // than the value, so this only guards the arithmetic.
                if candidate.bit_len() == bits && is_probably_prime(&candidate, rounds, rng)? {
                    bytes.fill(0);
                    return Ok(candidate);
                }
            }
            offset += 2;
        }
    }

    bytes.fill(0);
    Err(EncError::InvalidRsaKey(
        "no prime was found in the sieved candidates",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::PseudoRandom;

    fn big(value: u64) -> BigUint {
        BigUint::from_u64(value)
    }

    #[test]
    fn the_sieve_produces_the_primes_below_the_limit() {
        let primes = small_primes();
        assert_eq!(&primes[..6], &[2, 3, 5, 7, 11, 13]);
        assert_eq!(primes.last(), Some(&2039));
        // π(2048) = 309.
        assert_eq!(primes.len(), 309);
        assert!(!primes.contains(&1));
        assert!(!primes.contains(&2047)); // 23 · 89
    }

    #[test]
    fn small_numbers_are_classified_correctly() -> Result<()> {
        let mut rng = PseudoRandom::new(1);
        for value in [2u64, 3, 5, 7, 11, 13, 97, 65537] {
            assert!(
                is_probably_prime(&big(value), 8, &mut rng)?,
                "{value} is prime"
            );
        }
        for value in [0u64, 1, 4, 9, 15, 21, 25, 91, 65535, 65536] {
            assert!(
                !is_probably_prime(&big(value), 8, &mut rng)?,
                "{value} is composite"
            );
        }
        Ok(())
    }

    /// The Miller–Rabin base must be drawn at the candidate's *bit* width. Drawing at its octet
    /// width leaves only two of 256 values in range for a three-bit candidate, so the search for
    /// a base runs out long before the primality question is ever asked.
    #[test]
    fn a_base_is_found_even_for_the_smallest_candidates() -> Result<()> {
        let mut rng = PseudoRandom::new(12);
        for value in [5u64, 7, 11, 13, 17, 19, 23, 29, 31] {
            assert!(
                is_probably_prime(&big(value), 50, &mut rng)?,
                "{value} is prime"
            );
        }
        for value in [9u64, 15, 21, 25, 27, 33, 35] {
            assert!(
                !is_probably_prime(&big(value), 50, &mut rng)?,
                "{value} is composite"
            );
        }
        Ok(())
    }

    /// The strong pseudoprimes to base two are exactly the numbers one fixed base would let
    /// through. Random bases catch them, which is why the test draws its bases.
    #[test]
    fn strong_pseudoprimes_to_base_two_are_rejected() -> Result<()> {
        let mut rng = PseudoRandom::new(2);
        for value in [2047u64, 3277, 4033, 4681, 8321, 15841, 29341] {
            assert!(
                !is_probably_prime(&big(value), 20, &mut rng)?,
                "{value} is a base-two pseudoprime, not a prime"
            );
        }
        Ok(())
    }

    /// A Carmichael number passes every Fermat test; Miller–Rabin is what distinguishes it.
    #[test]
    fn carmichael_numbers_are_rejected() -> Result<()> {
        let mut rng = PseudoRandom::new(3);
        for value in [561u64, 1105, 1729, 2465, 2821, 6601, 8911] {
            assert!(
                !is_probably_prime(&big(value), 20, &mut rng)?,
                "{value} is a Carmichael number, not a prime"
            );
        }
        Ok(())
    }

    #[test]
    fn trial_division_agrees_with_the_full_test_below_the_squared_limit() -> Result<()> {
        let mut rng = PseudoRandom::new(4);
        for value in 2u64..600 {
            let sieved = passes_trial_division(&big(value))?;
            let tested = is_probably_prime(&big(value), 12, &mut rng)?;
            assert_eq!(sieved, tested, "{value}");
        }
        // A prime above the limit survives the sieve; an even multiple of it does not.
        assert!(passes_trial_division(&big(1_000_003))?);
        assert!(!passes_trial_division(&big(2_000_006))?);
        Ok(())
    }

    #[test]
    fn a_generated_prime_has_the_right_shape() -> Result<()> {
        let mut rng = PseudoRandom::new(5);
        let bits = 128;
        let prime = generate_prime(bits, miller_rabin_rounds(bits), &mut rng)?;

        assert_eq!(prime.bit_len(), bits);
        // The top two bits and the bottom one are all set.
        assert!(prime.bit(bits - 1));
        assert!(prime.bit(bits - 2));
        assert!(prime.bit(0));
        // And it really is prime, by an independent test with fresh bases.
        assert!(passes_trial_division(&prime)?);
        assert!(is_probably_prime(&prime, 40, &mut PseudoRandom::new(6))?);
        Ok(())
    }

    #[test]
    fn generation_is_reproducible_from_the_seed() -> Result<()> {
        let bits = 128;
        let first = generate_prime(bits, 20, &mut PseudoRandom::new(9))?;
        let again = generate_prime(bits, 20, &mut PseudoRandom::new(9))?;
        let other = generate_prime(bits, 20, &mut PseudoRandom::new(10))?;
        assert_eq!(first, again);
        assert_ne!(first, other);
        Ok(())
    }

    #[test]
    fn out_of_range_widths_are_refused() {
        let mut rng = PseudoRandom::new(11);
        assert!(matches!(
            generate_prime(MIN_PRIME_BITS - 1, 4, &mut rng),
            Err(EncError::InvalidRsaKey(_))
        ));
        assert!(matches!(
            generate_prime(MAX_PRIME_BITS + 1, 4, &mut rng),
            Err(EncError::InvalidRsaKey(_))
        ));
    }

    #[test]
    fn the_round_counts_follow_the_table() {
        assert_eq!(miller_rabin_rounds(2048), 4);
        assert_eq!(miller_rabin_rounds(1536), 4);
        assert_eq!(miller_rabin_rounds(1024), 5);
        assert_eq!(miller_rabin_rounds(512), 50);
        assert_eq!(miller_rabin_rounds(256), 50);
    }

    #[test]
    fn factoring_out_twos_is_exact() -> Result<()> {
        // 2^5 · 7 = 224.
        let (twos, odd) = factor_out_twos(&big(224))?;
        assert_eq!(twos, 5);
        assert_eq!(odd, big(7));
        // An odd number has no twos to factor out.
        let (twos, odd) = factor_out_twos(&big(9))?;
        assert_eq!(twos, 0);
        assert_eq!(odd, big(9));
        Ok(())
    }
}
