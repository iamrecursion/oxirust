//! Regression tests for the verified audit findings F4174-F4181 on the dense
//! material point method.
//!
//! * `stencil_tests`: quadratic B-spline stencil and the P2G/G2P transfers (F4181)
//! * `svd_tests`: signed SVD and polar decomposition (F4175, F4174)
//! * `plasticity_tests`: snow, von Mises and Drucker-Prager plasticity (F4175, F4176, F4174)
//! * `force_tests`: Kirchhoff stress and the P2G elastic force (F4178)
//! * `level_set_tests`: level-set coupling (F4179)
//! * `pressure_tests`: pressure projection (F4180)

use super::functions::*;
use super::types::*;

mod force_tests;
mod level_set_tests;
mod plasticity_tests;
mod pressure_tests;
mod stencil_tests;
mod svd_tests;

/// Deterministic xorshift64* generator, so the tests need no RNG dependency and
/// every run sees the same "random" inputs.
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn vec3(&mut self, lo: f64, hi: f64) -> [f64; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }

    fn mat3(&mut self, lo: f64, hi: f64) -> [[f64; 3]; 3] {
        [self.vec3(lo, hi), self.vec3(lo, hi), self.vec3(lo, hi)]
    }
}

/// Largest absolute entry of `a - b`.
fn max_diff(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

fn vec_diff(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
}

fn frobenius(a: [[f64; 3]; 3]) -> f64 {
    a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

fn rot_x(theta: f64) -> [[f64; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(theta: f64) -> [[f64; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn rot_z(theta: f64) -> [[f64; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn random_rotation(rng: &mut TestRng) -> [[f64; 3]; 3] {
    let pi = std::f64::consts::PI;
    mat3_mul(
        rot_z(rng.range(-pi, pi)),
        mat3_mul(rot_y(rng.range(-pi, pi)), rot_x(rng.range(-pi, pi))),
    )
}

/// Largest deviation of `QᵀQ` from the identity.
fn orthogonality_error(q: [[f64; 3]; 3]) -> f64 {
    max_diff(mat3_mul(mat3_transpose(q), q), mat3_identity())
}

fn symmetry_error(a: [[f64; 3]; 3]) -> f64 {
    max_diff(a, mat3_transpose(a))
}

/// `U diag(σ) Vᵀ`.
fn reconstruct(u: [[f64; 3]; 3], sigma: [f64; 3], vt: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    mat3_mul(mat3_mul(u, mat3_diag(sigma)), vt)
}

/// `R diag(d) Rᵀ`.
fn rotate_diagonal(r: [[f64; 3]; 3], d: [f64; 3]) -> [[f64; 3]; 3] {
    mat3_mul(mat3_mul(r, mat3_diag(d)), mat3_transpose(r))
}

const THETA_C: f64 = 0.025;
const THETA_S: f64 = 0.0075;

const GRID_H: f64 = 0.1;
const VOLUME0: f64 = 1.0e-3;
