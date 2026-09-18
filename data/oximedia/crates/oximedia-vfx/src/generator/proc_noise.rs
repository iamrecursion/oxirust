//! Procedural noise generation: value noise, fractal Brownian motion, Perlin.
//!
//! All functions are deterministic given the same inputs — no global RNG state
//! is used.  Everything is pure-Rust with no external crate dependencies
//! beyond the standard library.

// ── Hash helpers ───────────────────────────────────────────────────────────────

/// Deterministic hash of a `u64` seed to the range [0.0, 1.0).
///
/// Uses a simple finalisation step from the splitmix64 / Murmur3 family.
#[must_use]
pub fn hash_u64(v: u64) -> f64 {
    let mut x = v;
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    // Map to [0, 1)
    (x as f64) / (u64::MAX as f64)
}

/// Hash two i64 grid coordinates and a seed to [0.0, 1.0).
fn hash_2d(ix: i64, iy: i64, seed: u64) -> f64 {
    let h = (ix as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((iy as u64).wrapping_mul(0x6c62_272e_07bb_0142))
        .wrapping_add(seed);
    hash_u64(h)
}

/// Hash a single i64 grid coordinate and seed to [0.0, 1.0).
fn hash_1d(ix: i64, seed: u64) -> f64 {
    let h = (ix as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(seed);
    hash_u64(h)
}

// ── Smoothstep ─────────────────────────────────────────────────────────────────

/// Cubic smoothstep: `3t² − 2t³` for `t` in [0, 1].
#[inline]
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ── Value Noise ────────────────────────────────────────────────────────────────

/// 1-D value noise in [0.0, 1.0] for coordinate `x` and `seed`.
///
/// Returns a smooth, deterministic value.
#[must_use]
pub fn value_noise_1d(x: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let fx = x - x.floor();
    let u = smoothstep(fx);

    let v0 = hash_1d(ix, seed);
    let v1 = hash_1d(ix + 1, seed);
    lerp(v0, v1, u)
}

/// 2-D value noise in [0.0, 1.0] for coordinate `(x, y)` and `seed`.
#[must_use]
pub fn value_noise_2d(x: f64, y: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let u = smoothstep(fx);
    let v = smoothstep(fy);

    let v00 = hash_2d(ix, iy, seed);
    let v10 = hash_2d(ix + 1, iy, seed);
    let v01 = hash_2d(ix, iy + 1, seed);
    let v11 = hash_2d(ix + 1, iy + 1, seed);

    let top = lerp(v00, v10, u);
    let bot = lerp(v01, v11, u);
    lerp(top, bot, v)
}

// ── Fractal Brownian Motion ────────────────────────────────────────────────────

/// 1-D fractal Brownian motion (fBm) built from value noise.
///
/// * `octaves` — number of noise layers (typically 4–8).
/// * `lacunarity` — frequency multiplier per octave (typically ≈ 2.0).
/// * `gain` — amplitude multiplier per octave (typically ≈ 0.5).
/// * `seed` — deterministic seed.
///
/// Returns a value approximately in [0.0, 1.0] (exact range depends on
/// `octaves` and `gain`).
#[must_use]
pub fn fbm_1d(x: f64, octaves: u32, lacunarity: f64, gain: f64, seed: u64) -> f64 {
    let mut value = 0.0_f64;
    let mut amplitude = 1.0_f64;
    let mut frequency = 1.0_f64;
    let mut max_value = 0.0_f64;

    for i in 0..octaves {
        // Use a per-octave seed derivation so octaves are independent
        let oct_seed = seed.wrapping_add((i as u64).wrapping_mul(0xdeadbeef_12345678));
        value += value_noise_1d(x * frequency, oct_seed) * amplitude;
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    if max_value > 0.0 {
        value / max_value
    } else {
        0.0
    }
}

/// 2-D fractal Brownian motion (fBm) built from 2-D value noise.
#[must_use]
pub fn fbm_2d(x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64, seed: u64) -> f64 {
    let mut value = 0.0_f64;
    let mut amplitude = 1.0_f64;
    let mut frequency = 1.0_f64;
    let mut max_value = 0.0_f64;

    for i in 0..octaves {
        let oct_seed = seed.wrapping_add((i as u64).wrapping_mul(0xdeadbeef_12345678));
        value += value_noise_2d(x * frequency, y * frequency, oct_seed) * amplitude;
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    if max_value > 0.0 {
        value / max_value
    } else {
        0.0
    }
}

// ── Perlin Noise ──────────────────────────────────────────────────────────────

/// Classic Perlin noise with a shuffled permutation table.
pub struct PerlinNoise {
    /// Doubled permutation table (512 entries for wrap-around).
    permutation: [u8; 512],
}

impl PerlinNoise {
    /// Build a new `PerlinNoise` instance from `seed`.
    ///
    /// The permutation table is initialised by a Fisher-Yates shuffle seeded
    /// with the hash of `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut perm = [0u8; 256];
        for (i, v) in perm.iter_mut().enumerate() {
            *v = i as u8;
        }

        // Fisher-Yates shuffle using the splitmix64 stream derived from seed
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = || -> u64 {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };

        for i in (1..256usize).rev() {
            let j = (next() as usize) % (i + 1);
            perm.swap(i, j);
        }

        let mut permutation = [0u8; 512];
        for i in 0..256 {
            permutation[i] = perm[i];
            permutation[i + 256] = perm[i];
        }

        Self { permutation }
    }

    /// Look up the permutation table.
    fn p(&self, i: usize) -> usize {
        self.permutation[i & 511] as usize
    }

    /// Perlin gradient function for 1-D.
    fn grad_1d(hash: usize, x: f64) -> f64 {
        if hash & 1 == 0 {
            x
        } else {
            -x
        }
    }

    /// Perlin gradient function for 2-D.
    fn grad_2d(hash: usize, x: f64, y: f64) -> f64 {
        match hash & 3 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            _ => -x - y,
        }
    }

    /// 1-D Perlin noise in approximately [-1.0, 1.0], shifted to [0.0, 1.0].
    #[must_use]
    pub fn noise_1d(&self, x: f64) -> f64 {
        let ix = x.floor() as i64;
        let fx = x - x.floor();
        let u = fade(fx);

        let x0 = (ix & 255) as usize;
        let x1 = (x0 + 1) & 255;

        let g0 = Self::grad_1d(self.p(x0), fx);
        let g1 = Self::grad_1d(self.p(x1), fx - 1.0);

        let raw = lerp(g0, g1, u);
        // Map from approximately [-1, 1] to [0, 1]
        (raw + 1.0) * 0.5
    }

    /// 2-D Perlin noise in approximately [0.0, 1.0].
    #[must_use]
    pub fn noise_2d(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();

        let u = fade(fx);
        let v = fade(fy);

        let x0 = (ix & 255) as usize;
        let y0 = (iy & 255) as usize;
        let x1 = (x0 + 1) & 255;
        let y1 = (y0 + 1) & 255;

        let aa = self.p(self.p(x0) + y0);
        let ab = self.p(self.p(x0) + y1);
        let ba = self.p(self.p(x1) + y0);
        let bb = self.p(self.p(x1) + y1);

        let g00 = Self::grad_2d(aa, fx, fy);
        let g10 = Self::grad_2d(ba, fx - 1.0, fy);
        let g01 = Self::grad_2d(ab, fx, fy - 1.0);
        let g11 = Self::grad_2d(bb, fx - 1.0, fy - 1.0);

        let x_interp_top = lerp(g00, g10, u);
        let x_interp_bot = lerp(g01, g11, u);
        let raw = lerp(x_interp_top, x_interp_bot, v);

        // Clamp and normalise to [0, 1]
        ((raw + 1.0) * 0.5).clamp(0.0, 1.0)
    }
}

/// Perlin quintic fade: `6t⁵ − 15t⁴ + 10t³`.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_hash_u64_range() {
        for seed in [0u64, 1, 42, u64::MAX, 0xdeadbeef] {
            let h = hash_u64(seed);
            assert!(h >= 0.0 && h < 1.0, "hash should be in [0,1), got {h}");
        }
    }

    #[test]
    fn test_hash_u64_deterministic() {
        assert!(
            approx(hash_u64(12345), hash_u64(12345), 1e-15),
            "hash must be deterministic"
        );
    }

    #[test]
    fn test_value_noise_1d_range() {
        for xi in 0..20 {
            let v = value_noise_1d(xi as f64 * 0.37, 42);
            assert!(v >= 0.0 && v <= 1.0, "value_noise_1d out of range: {v}");
        }
    }

    #[test]
    fn test_value_noise_1d_deterministic() {
        let v1 = value_noise_1d(1.5, 99);
        let v2 = value_noise_1d(1.5, 99);
        assert!(
            approx(v1, v2, 1e-15),
            "value_noise_1d should be deterministic"
        );
    }

    #[test]
    fn test_value_noise_2d_range() {
        for yi in 0..5 {
            for xi in 0..5 {
                let v = value_noise_2d(xi as f64 * 0.4, yi as f64 * 0.4, 7);
                assert!(v >= 0.0 && v <= 1.0, "value_noise_2d out of range: {v}");
            }
        }
    }

    #[test]
    fn test_value_noise_2d_different_seeds_differ() {
        let v1 = value_noise_2d(3.7, 2.1, 0);
        let v2 = value_noise_2d(3.7, 2.1, 1);
        assert!(
            (v1 - v2).abs() > 1e-6,
            "different seeds should produce different values"
        );
    }

    #[test]
    fn test_fbm_1d_range() {
        for i in 0..30 {
            let v = fbm_1d(i as f64 * 0.13, 6, 2.0, 0.5, 123);
            assert!(v >= 0.0 && v <= 1.0, "fbm_1d should be in [0,1], got {v}");
        }
    }

    #[test]
    fn test_fbm_2d_range() {
        for j in 0..5 {
            for i in 0..5 {
                let v = fbm_2d(i as f64 * 0.2, j as f64 * 0.2, 4, 2.0, 0.5, 0);
                assert!(v >= 0.0 && v <= 1.0, "fbm_2d out of range: {v}");
            }
        }
    }

    #[test]
    fn test_fbm_1d_deterministic() {
        let v1 = fbm_1d(2.5, 5, 2.0, 0.5, 777);
        let v2 = fbm_1d(2.5, 5, 2.0, 0.5, 777);
        assert!(approx(v1, v2, 1e-15), "fbm_1d must be deterministic");
    }

    #[test]
    fn test_perlin_new_different_seeds() {
        let pn1 = PerlinNoise::new(0);
        let pn2 = PerlinNoise::new(42);
        // They should produce different noise (almost certainly)
        let v1 = pn1.noise_2d(1.3, 2.7);
        let v2 = pn2.noise_2d(1.3, 2.7);
        assert!(
            (v1 - v2).abs() > 1e-8,
            "different seeds should give different noise"
        );
    }

    #[test]
    fn test_perlin_noise_1d_range() {
        let pn = PerlinNoise::new(1);
        for i in 0..50 {
            let v = pn.noise_1d(i as f64 * 0.17);
            assert!(v >= 0.0 && v <= 1.0, "perlin_1d out of range: {v} at i={i}");
        }
    }

    #[test]
    fn test_perlin_noise_2d_range() {
        let pn = PerlinNoise::new(2);
        for j in 0..5 {
            for i in 0..10 {
                let v = pn.noise_2d(i as f64 * 0.23, j as f64 * 0.31);
                assert!(
                    v >= 0.0 && v <= 1.0,
                    "perlin_2d out of range: {v} at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_perlin_noise_2d_deterministic() {
        let pn = PerlinNoise::new(99);
        let v1 = pn.noise_2d(3.14, 2.71);
        let v2 = pn.noise_2d(3.14, 2.71);
        assert!(approx(v1, v2, 1e-15), "perlin_2d must be deterministic");
    }

    #[test]
    fn test_fbm_zero_octaves() {
        // Zero octaves → 0.0 (no contribution)
        let v = fbm_1d(1.0, 0, 2.0, 0.5, 0);
        assert!(approx(v, 0.0, 1e-10), "zero octaves should give 0.0");
    }
}
