/// A simple deterministic Linear Congruential Generator (LCG) RNG.
///
/// This is intentionally lightweight and avoids pulling in the `rand` crate.
/// Constants follow Knuth's MMIX multiplier.
#[derive(Debug, Clone)]
pub struct AugRng {
    state: u64,
}

impl AugRng {
    /// Seed the RNG.
    pub fn new(seed: u64) -> Self {
        // Mix seed so that seed=0 does not stay stuck.
        let state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        Self { state }
    }

    /// Advance the LCG and return the next raw 64-bit value.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // LCG: state = a * state + c  (mod 2^64)
        // Constants from Knuth's MMIX.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for a clean mantissa.
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal sample via Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        // Box-Muller: requires two uniform samples in (0, 1].
        let u1 = (self.next_f64() + 1e-300).min(1.0); // avoid ln(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Return `true` with probability `p`.
    pub fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Uniform integer in [0, max).
    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        // Rejection-free scaling via 128-bit trick is overkill here; modulo is fine
        // for non-cryptographic use.
        (self.next_u64() as usize) % max
    }
}

/// Sample λ ~ Beta(alpha, alpha).
///
/// For alpha == 1 this degenerates to Uniform(0,1).
/// For other alpha values we use the normal approximation:
///   λ ≈ clip( 0.5 + N(0,1) * 0.5 / sqrt(2*alpha), 0, 1 )
/// which matches the median and spread of Beta(alpha, alpha) reasonably well.
pub(crate) fn sample_beta_symmetric(alpha: f64, rng: &mut AugRng) -> f64 {
    if (alpha - 1.0).abs() < 1e-9 {
        rng.next_f64()
    } else {
        let sigma = 0.5 / (2.0 * alpha).sqrt();
        let z = rng.next_normal();
        (0.5 + z * sigma).clamp(0.0, 1.0)
    }
}
