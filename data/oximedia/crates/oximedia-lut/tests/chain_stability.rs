//! Long-chain numerical-stability tests for [`oximedia_lut::lut_chain`].
//!
//! These tests push 8-16 LUT stages into a [`LutChain`] and verify that the
//! accumulated output stays within a documented tolerance of the analytic
//! expectation (identity for identity / even-count-swap chains; a looser
//! stability bound for the nonlinear gamma/inverse pair).
//!
//! Sample points are drawn from an inline seeded LCG (no `proptest` / `rand`)
//! so the tests are fully deterministic and dependency-free.

use oximedia_lut::lut_chain::{LutChain, LutChainEntry};
use oximedia_lut::{Lut3d, LutSize, Rgb};

/// 64-bit linear congruential generator (Knuth/MMIX constants).
///
/// `next_unit` returns an `f64` in `[0, 1)` taken from the top 53 bits of the
/// state, which avoids the low-order-bit non-randomness of LCGs.
struct Lcg {
    state: u64,
}

impl Lcg {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Wrapping arithmetic = implicit mod 2^64.
        self.state = self.state.wrapping_mul(Self::A).wrapping_add(Self::C);
        self.state
    }

    /// Uniform `f64` in `[0, 1)` from the high 53 bits.
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // keep top 53 bits
        bits as f64 / ((1u64 << 53) as f64)
    }

    /// A random RGB triple in `[0, 1)^3`.
    fn next_rgb(&mut self) -> Rgb {
        [self.next_unit(), self.next_unit(), self.next_unit()]
    }
}

/// Build a row-major (`r`-outer, `g`-mid, `b`-inner) identity lattice of the
/// given size as the `Vec<Rgb>` expected by [`LutChainEntry::new_3d`].
fn identity_lattice(size: usize) -> Vec<Rgb> {
    let scale = (size - 1) as f64;
    let mut data = Vec::with_capacity(size * size * size);
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                data.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
            }
        }
    }
    data
}

/// Extract a [`Lut3d`] (built via `from_fn`) into the row-major `Vec<Rgb>`
/// lattice consumed by [`LutChainEntry::new_3d`]. `Lut3d` stores data R-outer,
/// matching the chain's `r * size * size + g * size + b` indexing.
fn lattice_from_lut3d(lut: &Lut3d) -> Vec<Rgb> {
    let size = lut.size();
    let mut data = Vec::with_capacity(size * size * size);
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                data.push(lut.get(r, g, b));
            }
        }
    }
    data
}

#[test]
fn chain_15_identity_within_1e_4() {
    let size = 17usize;
    let lattice = identity_lattice(size);

    let mut chain = LutChain::new();
    for _ in 0..15 {
        chain.push(LutChainEntry::new_3d("id", size, lattice.clone()));
    }
    assert_eq!(chain.depth(), 15);

    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    for _ in 0..1000 {
        let p = rng.next_rgb();
        let out = chain.apply_rgb(p);
        for ch in 0..3 {
            assert!(
                (out[ch] - p[ch]).abs() < 1e-4,
                "15x identity drift at ch {ch}: in {} out {}",
                p[ch],
                out[ch]
            );
        }
    }
}

#[test]
fn chain_channelswap_even_count_identity() {
    let size = 17usize;
    // Channel-swap lattice: output = [b, r, g]. Applying it an even number of
    // times cycles r->g->b->r twice = net identity (the swap has order 3, but
    // 2 applications give [g, b, r]; we therefore need the count to be a
    // multiple of 3 OR rely on the lattice being exact at sample points).
    //
    // NOTE: [b, r, g] is a 3-cycle, so the net-identity count is a multiple of
    // 3, NOT simply "even". To honor the slice intent (an even push count that
    // nets to identity) we instead use a self-inverse swap: output = [b, g, r]
    // (swap R<->B, leave G). That involution nets to identity after any even
    // count of applications.
    let swap = Lut3d::from_fn(LutSize::Size17, |[r, g, b]| [b, g, r]);
    let lattice = lattice_from_lut3d(&swap);

    let mut chain = LutChain::new();
    for _ in 0..8 {
        chain.push(LutChainEntry::new_3d("swap_rb", size, lattice.clone()));
    }
    assert_eq!(chain.depth(), 8);

    let mut rng = Lcg::new(0x0fed_cba9_8765_4321);
    for _ in 0..1000 {
        let p = rng.next_rgb();
        let out = chain.apply_rgb(p);
        for ch in 0..3 {
            assert!(
                (out[ch] - p[ch]).abs() < 2e-4,
                "8x R<->B swap should net to identity at ch {ch}: in {} out {}",
                p[ch],
                out[ch]
            );
        }
    }
}

#[test]
fn chain_gamma_then_inverse_8x() {
    // STABILITY (accumulated-error) check, NOT an exactness check: trilinear
    // interpolation of a nonlinear gamma lattice has large interior error, so a
    // forward-gamma stage followed by its inverse does NOT cancel. The dominant
    // error source is the steep interior of x^2 / x^0.5 on a 17^3 grid (a SINGLE
    // forward/inverse pair already peaks at ~0.062 over 1000 samples), and it
    // grows monotonically with repetition. Measured worst-case max-abs error
    // over 1000 LCG samples on this exact lattice pair:
    //   1 pair  (2 stages)  -> ~0.062
    //   2 pairs (4 stages)  -> ~0.112
    //   4 pairs (8 stages)  -> ~0.191
    //   8 pairs (16 stages) -> ~0.238
    // We therefore bound 8 pairs at 0.30 (headroom above the observed 0.238 for
    // seed-dependent sampling), which still catches genuine divergence/blowup
    // while honoring that this is a stability, not exactness, property.
    let size = 17usize;
    let forward = Lut3d::from_fn(LutSize::Size17, |[r, g, b]| {
        [r.powf(2.0), g.powf(2.0), b.powf(2.0)]
    });
    let inverse = Lut3d::from_fn(LutSize::Size17, |[r, g, b]| {
        [r.powf(0.5), g.powf(0.5), b.powf(0.5)]
    });
    let fwd_lat = lattice_from_lut3d(&forward);
    let inv_lat = lattice_from_lut3d(&inverse);

    let mut chain = LutChain::new();
    for _ in 0..8 {
        chain.push(LutChainEntry::new_3d("gamma2", size, fwd_lat.clone()));
        chain.push(LutChainEntry::new_3d("gamma05", size, inv_lat.clone()));
    }
    assert_eq!(chain.depth(), 16);

    let mut rng = Lcg::new(0xdead_beef_cafe_babe);
    for _ in 0..1000 {
        let p = rng.next_rgb();
        let out = chain.apply_rgb(p);
        for ch in 0..3 {
            assert!(
                (out[ch] - p[ch]).abs() < 0.30,
                "gamma/inverse 8x stability bound exceeded at ch {ch}: in {} out {}",
                p[ch],
                out[ch]
            );
        }
    }
}
