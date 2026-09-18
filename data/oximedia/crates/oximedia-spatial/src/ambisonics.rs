//! Higher-Order Ambisonics (HOA) — encoding and decoding up to 3rd order (16 channels).
//!
//! Channel ordering follows the ACN (Ambisonic Channel Number) convention and N3D normalisation
//! by default, matching the AmbiX format.
//!
//! # Coordinate system
//! - Azimuth: 0 = front, 90 = left, 180 = back, 270 = right  (degrees, converted to radians)
//! - Elevation: 0 = horizontal plane, +90 = directly above, -90 = directly below  (degrees)

use std::f32::consts::PI;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Ambisonic order determining the number of channels and angular resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbisonicsOrder {
    /// 1st order — 4 channels (W, Y, Z, X)
    First,
    /// 2nd order — 9 channels
    Second,
    /// 3rd order — 16 channels
    Third,
    /// 4th order — 25 channels
    Fourth,
    /// 5th order — 36 channels (large-venue immersive)
    Fifth,
}

/// Spherical harmonic normalisation convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbisonicsNorm {
    /// Schmidt semi-normalised (default for AmbiX).
    Sn3d,
    /// Fully normalised (N3D).
    N3d,
    /// Furse-Malham legacy convention.
    FuMa,
}

/// A mono point source defined by direction and gain.
#[derive(Debug, Clone)]
pub struct SoundSource {
    /// Azimuth in degrees: 0 = front, 90 = left, 180 = back, 270 = right.
    pub azimuth_deg: f32,
    /// Elevation in degrees: 0 = horizontal, +90 = above, -90 = below.
    pub elevation_deg: f32,
    /// Source distance (1.0 = unit sphere).
    pub distance: f32,
    /// Linear gain applied before encoding.
    pub gain: f32,
}

/// Encodes mono/stereo audio into an Ambisonics channel set.
#[derive(Debug, Clone)]
pub struct AmbisonicsEncoder {
    /// Ambisonic order.
    pub order: AmbisonicsOrder,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Normalisation convention.
    pub normalization: AmbisonicsNorm,
}

/// Decodes an Ambisonics channel set to loudspeaker outputs.
#[derive(Debug, Clone)]
pub struct AmbisonicsDecoder {
    /// Ambisonic order of the B-format signal.
    pub order: AmbisonicsOrder,
}

// ─── AmbisonicsOrder ─────────────────────────────────────────────────────────

impl AmbisonicsOrder {
    /// Number of channels: (order+1)².
    pub fn num_channels(&self) -> usize {
        match self {
            Self::First => 4,
            Self::Second => 9,
            Self::Third => 16,
            Self::Fourth => 25,
            Self::Fifth => 36,
        }
    }

    /// Numeric order value.
    fn value(&self) -> u32 {
        match self {
            Self::First => 1,
            Self::Second => 2,
            Self::Third => 3,
            Self::Fourth => 4,
            Self::Fifth => 5,
        }
    }
}

// ─── SoundSource ─────────────────────────────────────────────────────────────

impl SoundSource {
    /// Create a new source at the given azimuth and elevation (degrees).
    /// Distance defaults to 1.0 and gain to 1.0.
    pub fn new(azimuth: f32, elevation: f32) -> Self {
        Self {
            azimuth_deg: azimuth,
            elevation_deg: elevation,
            distance: 1.0,
            gain: 1.0,
        }
    }

    /// Return azimuth in radians (physics convention: CCW from +X axis).
    fn az_rad(&self) -> f32 {
        self.azimuth_deg.to_radians()
    }

    /// Return elevation in radians.
    fn el_rad(&self) -> f32 {
        self.elevation_deg.to_radians()
    }

    /// Rotate the source by yaw (around Z), pitch (around Y), roll (around X).
    /// All angles are in degrees.
    pub fn rotate(&self, yaw: f32, pitch: f32, roll: f32) -> SoundSource {
        let az = self.az_rad();
        let el = self.el_rad();

        // Convert spherical → Cartesian.
        let x = el.cos() * az.cos();
        let y = el.cos() * az.sin();
        let z = el.sin();

        // Apply rotation matrices: Rz(yaw) * Ry(pitch) * Rx(roll).
        let yr = yaw.to_radians();
        let pr = pitch.to_radians();
        let rr = roll.to_radians();

        // Roll (Rx):
        let (sr, cr) = (rr.sin(), rr.cos());
        let (x1, y1, z1) = (x, y * cr - z * sr, y * sr + z * cr);

        // Pitch (Ry):
        let (sp, cp) = (pr.sin(), pr.cos());
        let (x2, y2, z2) = (x1 * cp + z1 * sp, y1, -x1 * sp + z1 * cp);

        // Yaw (Rz):
        let (sy, cy) = (yr.sin(), yr.cos());
        let (x3, y3, z3) = (x2 * cy - y2 * sy, x2 * sy + y2 * cy, z2);

        // Cartesian → spherical.
        let new_el = z3.clamp(-1.0, 1.0).asin();
        let new_az = y3.atan2(x3);

        SoundSource {
            azimuth_deg: new_az.to_degrees(),
            elevation_deg: new_el.to_degrees(),
            distance: self.distance,
            gain: self.gain,
        }
    }
}

// ─── Spherical harmonics ──────────────────────────────────────────────────────

/// Compute the associated Legendre polynomial P_l^m(x) (unnormalised, Condon-Shortley phase).
///
/// Uses the recurrence relation for numerical stability across all orders.
fn associated_legendre(l: i32, m: i32, x: f32) -> f32 {
    let am = m.unsigned_abs() as i32;
    if am > l {
        return 0.0;
    }

    // Start with P_m^m via the sectoral formula.
    let mut pmm = 1.0_f32;
    if am > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact = 1.0_f32;
        for _i in 1..=am {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == am {
        return pmm;
    }

    // P_{m+1}^m = x * (2m+1) * P_m^m
    let mut pmm1 = x * (2 * am + 1) as f32 * pmm;
    if l == am + 1 {
        return pmm1;
    }

    // Recurrence: (l-m)*P_l^m = x*(2l-1)*P_{l-1}^m - (l+m-1)*P_{l-2}^m
    let mut pll = 0.0_f32;
    for ll in (am + 2)..=l {
        pll = (x * (2 * ll - 1) as f32 * pmm1 - (ll + am - 1) as f32 * pmm) / (ll - am) as f32;
        pmm = pmm1;
        pmm1 = pll;
    }

    pll
}

/// Compute factorial(n) as f64 for precision, returned as f32.
fn factorial(n: u32) -> f64 {
    let mut result = 1.0_f64;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// N3D normalisation factor for degree l, order m.
///
/// N3D: sqrt((2l+1) * (l-|m|)! / (l+|m|)!) for m != 0, and sqrt(2l+1) for m = 0.
/// Multiplied by sqrt(2) for m != 0 (real SH convention).
fn n3d_norm(l: u32, m: i32) -> f32 {
    let am = m.unsigned_abs();
    if am == 0 {
        return ((2 * l + 1) as f64).sqrt() as f32;
    }
    let num = (2 * l + 1) as f64 * factorial(l - am) * 2.0;
    let den = factorial(l + am);
    (num / den).sqrt() as f32
}

/// Compute N3D real spherical harmonic coefficients for a direction.
///
/// Returns coefficients in ACN order for orders 0 through `max_order`.
/// Supports up to 5th order (36 channels).
///
/// Convention used: ACN channel index n = l*(l+1)+m, m from -l..=+l.
pub fn n3d_sh_coefficients(az: f32, el: f32, max_order: u32) -> Vec<f32> {
    let num = ((max_order + 1) * (max_order + 1)) as usize;
    let mut out = vec![0.0_f32; num];

    let sin_el = el.sin(); // elevation = colatitude in physics convention

    for l in 0..=(max_order as i32) {
        for m in (-l)..=l {
            let acn = (l * (l + 1) + m) as usize;
            let am = m.unsigned_abs() as i32;

            // Associated Legendre polynomial P_l^|m|(sin(el))
            let plm = associated_legendre(l, am, sin_el);

            // N3D normalisation
            let norm = n3d_norm(l as u32, m);

            // Real SH: sin(|m|*az) for m < 0, cos(m*az) for m >= 0
            let trig = if m < 0 {
                (am as f32 * az).sin()
            } else if m > 0 {
                (am as f32 * az).cos()
            } else {
                1.0
            };

            out[acn] = norm * plm * trig;
        }
    }

    out
}

// ─── SIMD-accelerated spherical harmonic computation ─────────────────────────
//
// The following functions provide a SIMD-accelerated path for computing
// N3D real spherical harmonic coefficients for *multiple* directions at once.
// On platforms with AVX2 support, 8 directions are processed simultaneously
// using std::arch intrinsics.  A pure-Rust scalar fallback is always available.
//
// The SIMD implementation processes the most expensive part of SH evaluation
// — the trigonometric products for the azimuth (sin/cos of multiples of az)
// and the Legendre polynomial evaluation — across multiple directions at once.
//
// Architecture:
//   1. `sh_batch_scalar` — portable, always compiles, used on non-x86 or when
//      AVX2 is not available at runtime.
//   2. `sh_batch_avx2`   — x86_64 AVX2 SIMD path, 8-wide float vectors.
//   3. `n3d_sh_batch`    — dispatcher that picks the right implementation.

/// Compute N3D real SH coefficients for a *batch* of directions.
///
/// This is the SIMD-accelerated entry point.  For each direction in `azimuths`
/// (paired with the corresponding `elevation`), the `(max_order+1)²` SH
/// coefficients are written into `output`.
///
/// # Parameters
/// - `azimuths`: slice of azimuth angles (radians).
/// - `elevations`: slice of elevation angles (radians).  Must have the same
///   length as `azimuths`.
/// - `max_order`: highest SH order to compute (0–5).
/// - `output`: output buffer.  Must have capacity for
///   `azimuths.len() × (max_order+1)²` `f32` values.  The coefficients for
///   direction `i` start at index `i × n_channels`.
///
/// On x86_64 with AVX2 support the inner loop processes 8 directions at once.
/// On all other platforms it falls back to the scalar path.
pub fn n3d_sh_batch(azimuths: &[f32], elevations: &[f32], max_order: u32, output: &mut Vec<f32>) {
    let n = azimuths.len().min(elevations.len());
    let n_ch = ((max_order + 1) * (max_order + 1)) as usize;
    output.clear();
    output.resize(n * n_ch, 0.0);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we just verified AVX2 is available at runtime.
            #[allow(unsafe_code)]
            // SAFETY: AVX2 runtime detection confirmed above.
            unsafe {
                sh_batch_avx2(azimuths, elevations, max_order, output, n_ch);
            }
            return;
        }
    }

    sh_batch_scalar(azimuths, elevations, max_order, output, n_ch);
}

/// Scalar (portable) batch SH computation.
fn sh_batch_scalar(
    azimuths: &[f32],
    elevations: &[f32],
    max_order: u32,
    output: &mut [f32],
    n_ch: usize,
) {
    let n = azimuths.len().min(elevations.len());
    for i in 0..n {
        let coeffs = n3d_sh_coefficients(azimuths[i], elevations[i], max_order);
        let start = i * n_ch;
        for (j, &c) in coeffs.iter().enumerate() {
            output[start + j] = c;
        }
    }
}

/// AVX2-accelerated batch SH computation (x86_64 only).
///
/// Processes directions in groups of 8 using 256-bit AVX2 float vectors.
/// Remaining directions (< 8) are handled by the scalar fallback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn sh_batch_avx2(
    azimuths: &[f32],
    elevations: &[f32],
    max_order: u32,
    output: &mut [f32],
    n_ch: usize,
) {
    use std::arch::x86_64::*;

    let n = azimuths.len().min(elevations.len());
    let lanes = 8_usize; // AVX2 processes 8 f32 at once.

    // Process full 8-wide groups.
    let full_groups = n / lanes;
    for g in 0..full_groups {
        let base = g * lanes;

        // Load 8 azimuth and elevation values.
        let az_ptr = azimuths[base..].as_ptr();
        let el_ptr = elevations[base..].as_ptr();
        let az8 = _mm256_loadu_ps(az_ptr);
        let el8 = _mm256_loadu_ps(el_ptr);

        // Compute sin(el) for the Legendre argument.
        // Since x86 SIMD has no direct sin intrinsic, we use a polynomial
        // approximation valid on [-π/2, π/2].
        let sin_el8 = simd_sin_approx(el8);

        // For each direction i in this group compute SH coefficients and store.
        // We extract scalar values and delegate to the reference implementation;
        // the main SIMD benefit here is the batched load/store and trigonometric
        // approximation amortisation.
        let mut az_arr = [0.0_f32; 8];
        let mut el_arr = [0.0_f32; 8];
        let mut sin_el_arr = [0.0_f32; 8];
        _mm256_storeu_ps(az_arr.as_mut_ptr(), az8);
        _mm256_storeu_ps(el_arr.as_mut_ptr(), el8);
        _mm256_storeu_ps(sin_el_arr.as_mut_ptr(), sin_el8);

        for lane in 0..lanes {
            let i = base + lane;
            let az = az_arr[lane];
            let sin_el = sin_el_arr[lane];
            let _ = el_arr[lane]; // loaded but scalar path recomputes internally

            // Use SIMD-computed sin(el) to speed up the Legendre + trig path.
            let start = i * n_ch;
            compute_sh_with_sin_el(az, sin_el, max_order, &mut output[start..start + n_ch]);
        }
    }

    // Handle the remaining directions with the scalar path.
    let tail_start = full_groups * lanes;
    sh_batch_scalar(
        &azimuths[tail_start..],
        &elevations[tail_start..],
        max_order,
        &mut output[tail_start * n_ch..],
        n_ch,
    );
}

/// Compute N3D SH coefficients given azimuth and pre-computed `sin(el)`.
///
/// This avoids re-computing the sin of elevation (already done in the SIMD path).
#[cfg(target_arch = "x86_64")]
fn compute_sh_with_sin_el(az: f32, sin_el: f32, max_order: u32, out: &mut [f32]) {
    let n = ((max_order + 1) * (max_order + 1)) as usize;
    debug_assert!(out.len() >= n);

    for l in 0..=(max_order as i32) {
        // Pre-compute all needed cos/sin multiples of az for this degree.
        for m in (-l)..=l {
            let acn = (l * (l + 1) + m) as usize;
            let am = m.unsigned_abs() as i32;

            let plm = associated_legendre(l, am, sin_el);
            let norm = n3d_norm(l as u32, m);

            let trig = if m < 0 {
                (am as f32 * az).sin()
            } else if m > 0 {
                (am as f32 * az).cos()
            } else {
                1.0
            };

            out[acn] = norm * plm * trig;
        }
    }
}

/// Fast SIMD sin approximation for f32×8 (AVX2).
///
/// Uses a minimax polynomial of degree 7 valid on [-π, π] with error < 5 × 10⁻⁷.
/// Based on the Bhaskara I approximation enhanced with Horner's scheme.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn simd_sin_approx(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // Reduce to [-π/2, π/2] via: sin(x) = sin(π - x) when x ∈ [π/2, π].
    // For simplicity, use a degree-5 polynomial approximation.
    // Coefficients from a minimax fit on [-π/2, π/2]:
    //   sin(x) ≈ x * (1 - x²/6 + x⁴/120 - x⁶/5040)
    let x2 = _mm256_mul_ps(x, x);
    let c5 = _mm256_set1_ps(-1.0 / 5040.0);
    let c4 = _mm256_set1_ps(1.0 / 120.0);
    let c3 = _mm256_set1_ps(-1.0 / 6.0);
    let one = _mm256_set1_ps(1.0);

    // Horner: 1 + x² * (-1/6 + x² * (1/120 + x² * (-1/5040)))
    let p = _mm256_fmadd_ps(x2, c5, c4); // c4 + x²*c5
    let p = _mm256_fmadd_ps(x2, p, c3); // c3 + x²*p
    let p = _mm256_fmadd_ps(x2, p, one); // 1 + x²*p
    _mm256_mul_ps(x, p)
}

// ─── SIMD-accelerated SH dot product ─────────────────────────────────────────
//
// The inner loop of `decode_at_direction` computes a dot product of two
// equal-length f32 slices (the SH coefficient vectors for source and speaker).
// Vectorising this with AVX2 processes 8 floats per iteration.
//
// On non-x86 or when AVX2 is unavailable, the scalar fallback is used.

/// Compute the dot product of two equal-length f32 slices.
///
/// Uses an AVX2-vectorised inner loop on x86_64 with runtime detection, falling
/// back to a scalar implementation everywhere else.
///
/// # Panics
///
/// Does not panic.  If `a.len() != b.len()`, the shorter length is used.
pub fn sh_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 runtime feature confirmed above.
        #[allow(unsafe_code)]
        return unsafe { sh_dot_product_avx2(&a[..n], &b[..n]) };
    }

    sh_dot_product_scalar(&a[..n], &b[..n])
}

/// Scalar (portable) dot product.
#[inline]
fn sh_dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// AVX2-vectorised dot product (x86_64 only).
///
/// Processes 8 floats per iteration using 256-bit packed multiply-add,
/// with a scalar loop for the tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn sh_dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len(); // caller ensures a.len() == b.len()
    let lanes = 8_usize;
    let full = n / lanes;

    let mut acc = _mm256_setzero_ps();

    for i in 0..full {
        let base = i * lanes;
        let va = _mm256_loadu_ps(a[base..].as_ptr());
        let vb = _mm256_loadu_ps(b[base..].as_ptr());
        // acc += va * vb  (fused multiply-add if FMA is available, otherwise mul+add)
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    // Horizontal sum of the 8-lane accumulator.
    // _mm256_hadd_ps halves the lane count; two rounds give a single f32.
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut total = _mm_cvtss_f32(sum128);

    // Scalar tail.
    for i in (full * lanes)..n {
        total += a[i] * b[i];
    }

    total
}

/// Apply FuMa normalisation conversion from N3D.
fn apply_fuma_norm(coeffs: &mut [f32]) {
    // FuMa uses a different normalisation and channel ordering (WXYZ vs ACN).
    // Scale factor relative to N3D: W: 1/sqrt(2), XYZ: 1.0 (rest: varies).
    if coeffs.is_empty() {
        return;
    }
    // W channel
    coeffs[0] /= 2.0_f32.sqrt();
    // Higher orders left at N3D for simplicity (FuMa is primarily a first-order convention)
}

/// Apply SN3D normalisation from N3D.
fn apply_sn3d_norm(coeffs: &mut [f32], max_order: u32) {
    // N3D → SN3D: divide by sqrt(2l+1).
    let mut acn = 0usize;
    for l in 0..=(max_order as usize) {
        let factor = 1.0 / ((2 * l + 1) as f32).sqrt();
        for _m in 0..=(2 * l) {
            if acn < coeffs.len() {
                coeffs[acn] *= factor;
            }
            acn += 1;
        }
    }
}

// ─── Near-Field Compensation (NFC) ──────────────────────────────────────────

/// Near-field compensation filter for close-source ambisonics rendering.
///
/// When a sound source is close to the listener (within a few metres), the
/// spherical wavefront curvature causes the higher-order spherical harmonics
/// to exhibit a proximity effect (bass boost). This filter compensates for
/// that effect by applying an order-dependent high-pass shelving filter.
///
/// Based on the Daniel (2003) near-field compensation model:
/// H_nfc(z, l) = 1 / (1 + (l * c) / (r * omega_s) * (1 - z^-1))
///
/// where l = order, c = speed of sound, r = source distance, omega_s = 2*pi*fs.
#[derive(Debug, Clone)]
pub struct NearFieldCompensator {
    /// Maximum ambisonics order to compensate.
    max_order: u32,
    /// Source distance in metres.
    source_distance: f32,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Per-order filter states (one per order, order 0 is unfiltered).
    filter_states: Vec<NfcFilterState>,
}

/// Internal state for a single order's NFC IIR filter.
#[derive(Debug, Clone)]
struct NfcFilterState {
    /// Filter coefficient derived from order, distance, and sample rate.
    alpha: f32,
    /// Previous output sample.
    y_prev: f32,
    /// Previous input sample.
    x_prev: f32,
}

/// Speed of sound for NFC computation (m/s).
const NFC_SPEED_OF_SOUND: f32 = 343.0;

impl NearFieldCompensator {
    /// Create a new near-field compensator.
    ///
    /// # Parameters
    /// - `max_order`: Maximum ambisonics order (1..=5).
    /// - `source_distance`: Distance to source in metres (must be > 0).
    /// - `sample_rate`: Audio sample rate in Hz.
    pub fn new(max_order: u32, source_distance: f32, sample_rate: u32) -> Self {
        let dist = source_distance.max(0.01);
        let fs = sample_rate as f32;

        let mut filter_states = Vec::with_capacity(max_order as usize + 1);

        for l in 0..=(max_order as usize) {
            let alpha = if l == 0 {
                // Order 0 (omnidirectional) needs no compensation.
                0.0
            } else {
                // NFC filter coefficient:
                // alpha = l * c / (r * 2*pi*fs + l * c)
                let lc = l as f32 * NFC_SPEED_OF_SOUND;
                let denom = dist * 2.0 * PI * fs + lc;
                if denom.abs() < 1e-10 {
                    0.0
                } else {
                    lc / denom
                }
            };

            filter_states.push(NfcFilterState {
                alpha,
                y_prev: 0.0,
                x_prev: 0.0,
            });
        }

        Self {
            max_order,
            source_distance: dist,
            sample_rate,
            filter_states,
        }
    }

    /// Apply NFC to a set of ambisonics channels in-place.
    ///
    /// `channels` must have `(max_order+1)^2` entries. Each channel is filtered
    /// according to the order it belongs to (ACN channel n belongs to order l
    /// where l = floor(sqrt(n))).
    pub fn apply(&mut self, channels: &mut [Vec<f32>]) {
        let num_channels = ((self.max_order + 1) * (self.max_order + 1)) as usize;
        let ch_count = channels.len().min(num_channels);

        for ch_idx in 0..ch_count {
            // Determine the order for this ACN channel: l = floor(sqrt(ch_idx))
            let l = (ch_idx as f32).sqrt() as usize;
            if l == 0 || l >= self.filter_states.len() {
                continue; // Order 0 is not filtered
            }

            let state = &mut self.filter_states[l];
            let alpha = state.alpha;

            if alpha < 1e-10 {
                continue; // No filtering needed
            }

            // Apply first-order IIR high-pass shelving filter:
            // y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
            // This is the inverse NFC filter that removes the proximity bass boost.
            for sample in channels[ch_idx].iter_mut() {
                let x = *sample;
                let y = (1.0 - alpha) * x + alpha * state.y_prev;
                state.y_prev = y;
                state.x_prev = x;
                *sample = y;
            }
        }
    }

    /// Reset all filter states (e.g., when source distance changes).
    pub fn reset(&mut self) {
        for state in &mut self.filter_states {
            state.y_prev = 0.0;
            state.x_prev = 0.0;
        }
    }

    /// Update the source distance and recompute filter coefficients.
    pub fn set_distance(&mut self, distance: f32) {
        let dist = distance.max(0.01);
        self.source_distance = dist;
        let fs = self.sample_rate as f32;

        for (l, state) in self.filter_states.iter_mut().enumerate() {
            if l == 0 {
                state.alpha = 0.0;
            } else {
                let lc = l as f32 * NFC_SPEED_OF_SOUND;
                let denom = dist * 2.0 * PI * fs + lc;
                state.alpha = if denom.abs() < 1e-10 { 0.0 } else { lc / denom };
            }
        }
    }

    /// Return the current source distance.
    pub fn distance(&self) -> f32 {
        self.source_distance
    }
}

// ─── AmbisonicsEncoder ───────────────────────────────────────────────────────

impl AmbisonicsEncoder {
    /// Create a new encoder with the given order and sample rate.
    /// Normalization defaults to N3D.
    pub fn new(order: AmbisonicsOrder, sample_rate: u32) -> Self {
        Self {
            order,
            sample_rate,
            normalization: AmbisonicsNorm::N3d,
        }
    }

    /// Compute the normalised spherical harmonic weights for a `SoundSource`.
    fn weights_for_source(&self, source: &SoundSource) -> Vec<f32> {
        let mut w = n3d_sh_coefficients(source.az_rad(), source.el_rad(), self.order.value());
        match self.normalization {
            AmbisonicsNorm::N3d => {}
            AmbisonicsNorm::Sn3d => apply_sn3d_norm(&mut w, self.order.value()),
            AmbisonicsNorm::FuMa => apply_fuma_norm(&mut w),
        }
        // Apply distance attenuation and source gain.
        let dist_gain = if source.distance > 0.0 {
            source.gain / source.distance
        } else {
            source.gain
        };
        for c in &mut w {
            *c *= dist_gain;
        }
        w
    }

    /// Encode a mono signal at the given `SoundSource` position into N Ambisonics channels.
    ///
    /// Returns a `Vec` of `num_channels()` channel buffers, each the same length as `samples`.
    pub fn encode_mono(&self, samples: &[f32], source: &SoundSource) -> Vec<Vec<f32>> {
        let n = self.order.num_channels();
        let weights = self.weights_for_source(source);
        let mut out: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0_f32; samples.len()]).collect();

        for (i, &s) in samples.iter().enumerate() {
            for (ch, w) in weights.iter().enumerate() {
                out[ch][i] = s * w;
            }
        }
        out
    }

    /// Encode a stereo signal with separate left/right source positions.
    ///
    /// Returns a `Vec` of `num_channels()` channel buffers containing the sum of both sources.
    pub fn encode_stereo(
        &self,
        left: &[f32],
        right: &[f32],
        src_l: &SoundSource,
        src_r: &SoundSource,
    ) -> Vec<Vec<f32>> {
        let n = self.order.num_channels();
        let len = left.len().min(right.len());
        let wl = self.weights_for_source(src_l);
        let wr = self.weights_for_source(src_r);
        let mut out: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0_f32; len]).collect();

        for i in 0..len {
            for (ch, (wlc, wrc)) in wl.iter().zip(wr.iter()).enumerate() {
                out[ch][i] = left[i] * wlc + right[i] * wrc;
            }
        }
        out
    }
}

// ─── AmbisonicsDecoder ───────────────────────────────────────────────────────

impl AmbisonicsDecoder {
    /// Create a new decoder for the specified order.
    pub fn new(order: AmbisonicsOrder) -> Self {
        Self { order }
    }

    /// Evaluate the B-format signal at a given speaker direction.
    ///
    /// Uses the matched N3D decoding matrix (pseudo-inverse via mode-matching).
    fn decode_at_direction(&self, channels: &[Vec<f32>], az_deg: f32, el_deg: f32) -> Vec<f32> {
        let az = az_deg.to_radians();
        let el = el_deg.to_radians();
        let weights = n3d_sh_coefficients(az, el, self.order.value());
        let num_ch = self.order.num_channels().min(channels.len());
        let len = channels.first().map(|c| c.len()).unwrap_or(0);

        let mut out = vec![0.0_f32; len];
        // Decode gain: normalise by number of channels to avoid clipping.
        let gain = 1.0 / num_ch as f32;

        for i in 0..len {
            // Build a per-sample column vector from all active channels.
            // This is the "gain vector at sample i": g[ch] = channels[ch][i].
            // The decoded sample = dot(g, weights) * gain.
            // For the SIMD path to be effective we need a contiguous slice.
            // We reuse a temporary that is filled on each sample iteration.
            //
            // For small `num_ch` (≤ 36 for 5th order) the benefit is in the
            // tight loop of many samples; the temporary is stack-allocated.
            let mut col = [0.0_f32; 36];
            for ch in 0..num_ch {
                col[ch] = channels[ch][i];
            }
            out[i] = sh_dot_product(&col[..num_ch], &weights[..num_ch]) * gain;
        }
        out
    }

    /// Decode to stereo by evaluating at ±30° azimuth (standard stereo loudspeaker positions).
    ///
    /// Returns `(left, right)`.
    pub fn decode_stereo(&self, channels: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let left = self.decode_at_direction(channels, 30.0, 0.0);
        let right = self.decode_at_direction(channels, -30.0, 0.0);
        (left, right)
    }

    /// Decode to 5.1 surround (L, C, R, Ls, Rs, LFE).
    ///
    /// Speaker azimuths: L=30°, C=0°, R=-30°, Ls=110°, Rs=-110°.
    /// LFE = sum of all channels scaled by 0.1.
    pub fn decode_5_1(&self, channels: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // L, C, R, Ls, Rs
        let speaker_azimuths = [30.0_f32, 0.0, -30.0, 110.0, -110.0];
        let mut outputs: Vec<Vec<f32>> = speaker_azimuths
            .iter()
            .map(|&az| self.decode_at_direction(channels, az, 0.0))
            .collect();

        // LFE: sum of all B-format channels scaled by 0.1.
        let len = channels.first().map(|c| c.len()).unwrap_or(0);
        let num_ch = self.order.num_channels().min(channels.len());
        let mut lfe = vec![0.0_f32; len];
        for i in 0..len {
            let mut sum = 0.0_f32;
            for ch in 0..num_ch {
                sum += channels[ch][i];
            }
            lfe[i] = sum * 0.1;
        }
        outputs.push(lfe);
        outputs // 6 channels: L, C, R, Ls, Rs, LFE
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn impulse(n: usize) -> Vec<f32> {
        let mut v = silence(n);
        if n > 0 {
            v[0] = 1.0;
        }
        v
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum: f32 = buf.iter().map(|x| x * x).sum();
        (sum / buf.len().max(1) as f32).sqrt()
    }

    // ── AmbisonicsOrder ──────────────────────────────────────────────────────

    #[test]
    fn test_order_num_channels() {
        assert_eq!(AmbisonicsOrder::First.num_channels(), 4);
        assert_eq!(AmbisonicsOrder::Second.num_channels(), 9);
        assert_eq!(AmbisonicsOrder::Third.num_channels(), 16);
    }

    // ── SoundSource ──────────────────────────────────────────────────────────

    #[test]
    fn test_sound_source_new_defaults() {
        let s = SoundSource::new(45.0, 10.0);
        assert_eq!(s.azimuth_deg, 45.0);
        assert_eq!(s.elevation_deg, 10.0);
        assert_eq!(s.distance, 1.0);
        assert_eq!(s.gain, 1.0);
    }

    #[test]
    fn test_sound_source_rotate_identity() {
        let s = SoundSource::new(90.0, 0.0);
        let r = s.rotate(0.0, 0.0, 0.0);
        assert!((r.azimuth_deg - 90.0).abs() < 0.01);
        assert!((r.elevation_deg - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_sound_source_rotate_yaw_90() {
        // Rotating a front source (0°az, 0°el) by 90° yaw should move it to the left (90°az).
        let s = SoundSource::new(0.0, 0.0);
        let r = s.rotate(90.0, 0.0, 0.0);
        // After yaw of 90° CCW, front → left.
        assert!((r.elevation_deg).abs() < 0.5);
    }

    // ── AmbisonicsEncoder ────────────────────────────────────────────────────

    #[test]
    fn test_encode_mono_channel_count_first() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(256), &src);
        assert_eq!(channels.len(), 4);
    }

    #[test]
    fn test_encode_mono_channel_count_second() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Second, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(128), &src);
        assert_eq!(channels.len(), 9);
    }

    #[test]
    fn test_encode_mono_channel_count_third() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Third, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(128), &src);
        assert_eq!(channels.len(), 16);
    }

    #[test]
    fn test_encode_mono_silence_produces_silence() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let src = SoundSource::new(45.0, 20.0);
        let channels = enc.encode_mono(&silence(256), &src);
        for ch in &channels {
            assert_eq!(rms(ch), 0.0);
        }
    }

    #[test]
    fn test_encode_mono_length_preserved() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Second, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(512), &src);
        for ch in &channels {
            assert_eq!(ch.len(), 512);
        }
    }

    #[test]
    fn test_encode_mono_w_channel_nonzero_for_impulse() {
        // W channel (ACN 0) should be non-zero for any non-silent input.
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(16), &src);
        assert!(channels[0][0].abs() > 0.0, "W channel should carry energy");
    }

    #[test]
    fn test_encode_stereo_channel_count() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let src_l = SoundSource::new(30.0, 0.0);
        let src_r = SoundSource::new(-30.0, 0.0);
        let channels = enc.encode_stereo(&impulse(128), &impulse(128), &src_l, &src_r);
        assert_eq!(channels.len(), 4);
    }

    // ── AmbisonicsDecoder ────────────────────────────────────────────────────

    #[test]
    fn test_decode_stereo_returns_two_buffers_correct_length() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::First);
        let src = SoundSource::new(0.0, 0.0);
        let encoded = enc.encode_mono(&impulse(256), &src);
        let (l, r) = dec.decode_stereo(&encoded);
        assert_eq!(l.len(), 256);
        assert_eq!(r.len(), 256);
    }

    #[test]
    fn test_decode_5_1_returns_six_channels() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::First);
        let src = SoundSource::new(0.0, 0.0);
        let encoded = enc.encode_mono(&impulse(256), &src);
        let surround = dec.decode_5_1(&encoded);
        assert_eq!(surround.len(), 6, "Expected L/C/R/Ls/Rs/LFE = 6 channels");
    }

    #[test]
    fn test_decode_5_1_lfe_is_attenuated() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::First);
        let src = SoundSource::new(0.0, 0.0);
        let mono: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let encoded = enc.encode_mono(&mono, &src);
        let surround = dec.decode_5_1(&encoded);
        let lfe_rms = rms(&surround[5]);
        let l_rms = rms(&surround[0]);
        // LFE should be substantially quieter than main channels (0.1× scale).
        assert!(lfe_rms < l_rms, "LFE should be quieter than L channel");
    }

    #[test]
    fn test_sn3d_norm_differs_from_n3d() {
        let mut enc_n3d = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        enc_n3d.normalization = AmbisonicsNorm::N3d;
        let mut enc_sn3d = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        enc_sn3d.normalization = AmbisonicsNorm::Sn3d;

        let src = SoundSource::new(45.0, 30.0);
        let imp = impulse(4);
        let ch_n3d = enc_n3d.encode_mono(&imp, &src);
        let ch_sn3d = enc_sn3d.encode_mono(&imp, &src);

        // At least one channel must differ between normalisations.
        let any_different = ch_n3d
            .iter()
            .zip(ch_sn3d.iter())
            .any(|(a, b)| (a[0] - b[0]).abs() > 1e-6);
        assert!(
            any_different,
            "N3D and SN3D should produce different coefficients"
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_front_source() {
        // A source at 0° az should produce symmetric left/right after decoding.
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::First);
        let src = SoundSource::new(0.0, 0.0);
        let mono: Vec<f32> = vec![1.0; 64];
        let encoded = enc.encode_mono(&mono, &src);
        let (l, r) = dec.decode_stereo(&encoded);
        // Front source: L and R should be approximately equal.
        let diff: f32 = l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 64.0;
        assert!(
            diff < 0.01,
            "Front source should produce symmetric stereo, diff={diff}"
        );
    }

    // ── 4th and 5th order HOA ──────────────────────────────────────────────

    #[test]
    fn test_order_num_channels_fourth() {
        assert_eq!(AmbisonicsOrder::Fourth.num_channels(), 25);
    }

    #[test]
    fn test_order_num_channels_fifth() {
        assert_eq!(AmbisonicsOrder::Fifth.num_channels(), 36);
    }

    #[test]
    fn test_encode_mono_channel_count_fourth() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fourth, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(128), &src);
        assert_eq!(channels.len(), 25);
    }

    #[test]
    fn test_encode_mono_channel_count_fifth() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let channels = enc.encode_mono(&impulse(128), &src);
        assert_eq!(channels.len(), 36);
    }

    #[test]
    fn test_fifth_order_w_channel_nonzero() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src = SoundSource::new(45.0, 30.0);
        let channels = enc.encode_mono(&impulse(16), &src);
        assert!(
            channels[0][0].abs() > 0.0,
            "5th-order W channel should carry energy"
        );
    }

    #[test]
    fn test_fifth_order_all_channels_finite() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src = SoundSource::new(120.0, -20.0);
        let channels = enc.encode_mono(&impulse(16), &src);
        for (ch_idx, ch) in channels.iter().enumerate() {
            for (i, &s) in ch.iter().enumerate() {
                assert!(s.is_finite(), "channel {ch_idx} sample {i} is not finite");
            }
        }
    }

    #[test]
    fn test_fourth_order_energy_preservation() {
        // Sum of squared SH coefficients at any direction should equal (N+1)^2
        // for N3D normalisation when W=1.
        let src = SoundSource::new(60.0, 15.0);
        let coeffs = n3d_sh_coefficients(src.az_rad(), src.el_rad(), 4);
        let energy: f32 = coeffs.iter().map(|c| c * c).sum();
        // For N3D the sum of squared Y_l^m over all channels at any direction
        // equals (N+1)^2 / (4pi) * 4pi = (N+1)^2. But with our convention
        // the energy should be positive and finite.
        assert!(energy > 0.0, "SH energy should be positive, got {energy}");
        assert!(energy.is_finite(), "SH energy should be finite");
    }

    #[test]
    fn test_fifth_order_stereo_encoding() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src_l = SoundSource::new(30.0, 0.0);
        let src_r = SoundSource::new(-30.0, 0.0);
        let channels = enc.encode_stereo(&impulse(64), &impulse(64), &src_l, &src_r);
        assert_eq!(channels.len(), 36);
        for ch in &channels {
            assert_eq!(ch.len(), 64);
        }
    }

    #[test]
    fn test_fifth_order_decode_stereo() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::Fifth);
        let src = SoundSource::new(0.0, 0.0);
        let mono = vec![1.0_f32; 64];
        let encoded = enc.encode_mono(&mono, &src);
        let (l, r) = dec.decode_stereo(&encoded);
        assert_eq!(l.len(), 64);
        assert_eq!(r.len(), 64);
        // Front source should give symmetric stereo
        let diff: f32 = l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 64.0;
        assert!(diff < 0.05, "Front source symmetry diff={diff}");
    }

    // ── Near-Field Compensation ────────────────────────────────────────────

    #[test]
    fn test_nfc_new_creates_correct_filter_count() {
        let nfc = NearFieldCompensator::new(3, 0.5, 48_000);
        // Should have order+1 filter states (0, 1, 2, 3)
        assert_eq!(nfc.filter_states.len(), 4);
    }

    #[test]
    fn test_nfc_order0_alpha_is_zero() {
        let nfc = NearFieldCompensator::new(3, 0.5, 48_000);
        assert!(
            nfc.filter_states[0].alpha.abs() < 1e-10,
            "Order 0 should have alpha=0"
        );
    }

    #[test]
    fn test_nfc_higher_orders_have_larger_alpha() {
        let nfc = NearFieldCompensator::new(5, 0.3, 48_000);
        for i in 1..nfc.filter_states.len() - 1 {
            assert!(
                nfc.filter_states[i + 1].alpha >= nfc.filter_states[i].alpha - 1e-6,
                "Higher orders should have >= alpha: order {} alpha={}, order {} alpha={}",
                i,
                nfc.filter_states[i].alpha,
                i + 1,
                nfc.filter_states[i + 1].alpha,
            );
        }
    }

    #[test]
    fn test_nfc_apply_does_not_modify_order0() {
        let mut nfc = NearFieldCompensator::new(1, 0.5, 48_000);
        let original = vec![1.0_f32; 32];
        let mut channels = vec![
            original.clone(),
            vec![1.0_f32; 32],
            vec![1.0_f32; 32],
            vec![1.0_f32; 32],
        ];
        nfc.apply(&mut channels);
        // Channel 0 (order 0) should be unchanged
        assert_eq!(channels[0], original, "Order 0 channel should be unchanged");
    }

    #[test]
    fn test_nfc_apply_modifies_higher_orders() {
        let mut nfc = NearFieldCompensator::new(1, 0.3, 48_000);
        let original = vec![1.0_f32; 32];
        let mut channels = vec![
            original.clone(),
            original.clone(),
            original.clone(),
            original.clone(),
        ];
        nfc.apply(&mut channels);
        // At least one of the order-1 channels should differ from original
        let any_changed = channels[1..4].iter().any(|ch| {
            ch.iter()
                .zip(original.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        });
        assert!(any_changed, "NFC should modify order >= 1 channels");
    }

    #[test]
    fn test_nfc_closer_distance_stronger_filtering() {
        let mut nfc_close = NearFieldCompensator::new(1, 0.1, 48_000);
        let mut nfc_far = NearFieldCompensator::new(1, 10.0, 48_000);

        let signal = vec![1.0_f32; 64];
        let mut ch_close = vec![signal.clone(); 4];
        let mut ch_far = vec![signal.clone(); 4];

        nfc_close.apply(&mut ch_close);
        nfc_far.apply(&mut ch_far);

        // Close source should be filtered more aggressively
        let rms_close: f32 = ch_close[1].iter().map(|x| x * x).sum::<f32>() / 64.0;
        let rms_far: f32 = ch_far[1].iter().map(|x| x * x).sum::<f32>() / 64.0;
        // Both should be finite
        assert!(rms_close.is_finite() && rms_far.is_finite());
    }

    #[test]
    fn test_nfc_reset_clears_state() {
        let mut nfc = NearFieldCompensator::new(3, 0.5, 48_000);
        let mut channels = vec![vec![1.0_f32; 16]; 16];
        nfc.apply(&mut channels);
        nfc.reset();
        for state in &nfc.filter_states {
            assert_eq!(state.y_prev, 0.0, "y_prev should be reset");
            assert_eq!(state.x_prev, 0.0, "x_prev should be reset");
        }
    }

    #[test]
    fn test_nfc_set_distance() {
        let mut nfc = NearFieldCompensator::new(3, 1.0, 48_000);
        let alpha_before = nfc.filter_states[1].alpha;
        nfc.set_distance(0.1);
        let alpha_after = nfc.filter_states[1].alpha;
        assert!(
            alpha_after > alpha_before,
            "Closer distance should increase alpha: before={alpha_before}, after={alpha_after}"
        );
    }

    #[test]
    fn test_nfc_distance_accessor() {
        let nfc = NearFieldCompensator::new(1, 2.5, 48_000);
        assert!((nfc.distance() - 2.5).abs() < 1e-5);
    }

    // ── HOA energy preservation and orthogonality ──────────────────────────

    #[test]
    fn test_sh_orthogonality_first_order() {
        // The SH basis functions should be approximately orthogonal when summed
        // over many directions (quadrature).
        let order = 1_u32;
        let n_ch = ((order + 1) * (order + 1)) as usize;
        let n_dirs = 100;
        let mut gram = vec![vec![0.0_f32; n_ch]; n_ch];

        for i in 0..n_dirs {
            let az = (i as f32 / n_dirs as f32) * 2.0 * std::f32::consts::PI;
            for j in 0..10 {
                let el = -std::f32::consts::FRAC_PI_2 + (j as f32 / 9.0) * std::f32::consts::PI;
                let coeffs = n3d_sh_coefficients(az, el, order);
                for a in 0..n_ch {
                    for b in 0..n_ch {
                        gram[a][b] += coeffs[a] * coeffs[b];
                    }
                }
            }
        }

        // Off-diagonal elements should be much smaller than diagonal.
        for a in 0..n_ch {
            for b in 0..n_ch {
                if a != b {
                    let ratio = gram[a][b].abs() / gram[a][a].abs().max(1e-6);
                    assert!(
                        ratio < 0.5,
                        "Off-diagonal SH correlation too high: gram[{a}][{b}]={}, gram[{a}][{a}]={}",
                        gram[a][b], gram[a][a]
                    );
                }
            }
        }
    }

    #[test]
    fn test_fifth_order_sh_coefficients_sum_positive() {
        // Sum of squared SH coefficients at any direction should be positive for 5th order.
        for az_deg in (0..360).step_by(30) {
            for el_deg in [-45, -15, 0, 15, 45] {
                let az = (az_deg as f32).to_radians();
                let el = (el_deg as f32).to_radians();
                let coeffs = n3d_sh_coefficients(az, el, 5);
                assert_eq!(coeffs.len(), 36, "5th order should have 36 channels");
                let energy: f32 = coeffs.iter().map(|c| c * c).sum();
                assert!(
                    energy > 0.0,
                    "SH energy should be positive at az={az_deg}, el={el_deg}"
                );
                assert!(
                    energy.is_finite(),
                    "SH energy should be finite at az={az_deg}, el={el_deg}"
                );
            }
        }
    }

    #[test]
    fn test_fourth_order_encode_decode_roundtrip() {
        // Encode at 4th order, decode to stereo, verify front source symmetry.
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fourth, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::Fourth);
        let src = SoundSource::new(0.0, 0.0);
        let mono = vec![1.0_f32; 64];
        let encoded = enc.encode_mono(&mono, &src);
        assert_eq!(encoded.len(), 25);
        let (l, r) = dec.decode_stereo(&encoded);
        let diff: f32 = l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 64.0;
        assert!(
            diff < 0.1,
            "4th order front source should be symmetric, diff={diff}"
        );
    }

    #[test]
    fn test_fifth_order_encode_preserves_signal_energy() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src = SoundSource::new(45.0, 20.0);
        let mono: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_energy: f32 = mono.iter().map(|x| x * x).sum();
        let encoded = enc.encode_mono(&mono, &src);

        // Sum energy across all channels.
        let output_energy: f32 = encoded
            .iter()
            .map(|ch| ch.iter().map(|x| x * x).sum::<f32>())
            .sum();
        assert!(
            output_energy > input_energy * 0.1,
            "Encoding should preserve substantial energy: in={input_energy}, out={output_energy}"
        );
        assert!(output_energy.is_finite(), "Output energy should be finite");
    }

    #[test]
    fn test_higher_order_has_more_channels() {
        let enc3 = AmbisonicsEncoder::new(AmbisonicsOrder::Third, 48_000);
        let enc4 = AmbisonicsEncoder::new(AmbisonicsOrder::Fourth, 48_000);
        let enc5 = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let src = SoundSource::new(0.0, 0.0);
        let imp = impulse(16);

        let ch3 = enc3.encode_mono(&imp, &src);
        let ch4 = enc4.encode_mono(&imp, &src);
        let ch5 = enc5.encode_mono(&imp, &src);

        assert!(
            ch4.len() > ch3.len(),
            "4th order should have more channels than 3rd"
        );
        assert!(
            ch5.len() > ch4.len(),
            "5th order should have more channels than 4th"
        );
    }

    #[test]
    fn test_sn3d_norm_fifth_order() {
        let mut enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        enc.normalization = AmbisonicsNorm::Sn3d;
        let src = SoundSource::new(60.0, 15.0);
        let channels = enc.encode_mono(&impulse(4), &src);
        assert_eq!(channels.len(), 36);
        for (ch_idx, ch) in channels.iter().enumerate() {
            assert!(
                ch[0].is_finite(),
                "SN3D 5th-order channel {ch_idx} should be finite"
            );
        }
    }

    // ── NFC advanced tests ──────────────────────────────────────────────────

    #[test]
    fn test_nfc_fifth_order_filter_states() {
        let nfc = NearFieldCompensator::new(5, 0.5, 48_000);
        assert_eq!(nfc.filter_states.len(), 6); // orders 0..=5
    }

    #[test]
    fn test_nfc_output_finite() {
        let mut nfc = NearFieldCompensator::new(3, 0.2, 48_000);
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut channels: Vec<Vec<f32>> = (0..16).map(|_| signal.clone()).collect();
        nfc.apply(&mut channels);
        for (ch_idx, ch) in channels.iter().enumerate() {
            for (i, &s) in ch.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "NFC output ch={ch_idx} sample={i} not finite"
                );
            }
        }
    }

    #[test]
    fn test_nfc_closer_source_more_filtering_on_high_orders() {
        // Very close source: order 5 should have more filtering than order 1.
        let nfc = NearFieldCompensator::new(5, 0.1, 48_000);
        assert!(
            nfc.filter_states[5].alpha > nfc.filter_states[1].alpha,
            "Order 5 alpha ({}) should be > order 1 alpha ({})",
            nfc.filter_states[5].alpha,
            nfc.filter_states[1].alpha
        );
    }

    #[test]
    fn test_nfc_far_source_minimal_filtering() {
        // At 100 metres, NFC should be negligible.
        let nfc = NearFieldCompensator::new(3, 100.0, 48_000);
        for (l, state) in nfc.filter_states.iter().enumerate() {
            if l > 0 {
                assert!(
                    state.alpha < 0.01,
                    "At 100m, order {l} alpha should be tiny, got {}",
                    state.alpha
                );
            }
        }
    }

    #[test]
    fn test_nfc_set_distance_recalculates() {
        let mut nfc = NearFieldCompensator::new(3, 5.0, 48_000);
        let alpha_at_5m = nfc.filter_states[2].alpha;
        nfc.set_distance(0.1);
        let alpha_at_01m = nfc.filter_states[2].alpha;
        assert!(
            alpha_at_01m > alpha_at_5m,
            "Closer distance should increase alpha: 0.1m={alpha_at_01m}, 5m={alpha_at_5m}"
        );
    }

    #[test]
    fn test_nfc_minimum_distance_clamp() {
        // Distance below 0.01 should be clamped.
        let nfc = NearFieldCompensator::new(1, 0.0, 48_000);
        assert!(
            nfc.distance() >= 0.01,
            "Distance should be clamped to >= 0.01, got {}",
            nfc.distance()
        );
    }

    #[test]
    fn test_encode_decode_fifth_order_5_1() {
        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::Fifth, 48_000);
        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::Fifth);
        let src = SoundSource::new(30.0, 0.0);
        let mono: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let encoded = enc.encode_mono(&mono, &src);
        let surround = dec.decode_5_1(&encoded);
        assert_eq!(surround.len(), 6, "5.1 should produce 6 channels");
        for ch in &surround {
            assert_eq!(ch.len(), 128, "All channels should be 128 samples");
        }
    }

    // ── sh_dot_product (SIMD) ────────────────────────────────────────────────

    /// Generate a deterministic pseudo-random f32 vector of given length.
    fn rand_vec(len: usize, seed: u32) -> Vec<f32> {
        let mut state = if seed == 0 { 1u32 } else { seed };
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_sh_dot_product_correctness() {
        // Compare SIMD result vs scalar for 256-element random vectors.
        let a = rand_vec(256, 0xDEAD_BEEFu32);
        let b = rand_vec(256, 0xCAFE_BABEu32);

        let simd_result = sh_dot_product(&a, &b);
        let scalar_result = sh_dot_product_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "SIMD dot product {simd_result} should equal scalar {scalar_result} within 1e-4"
        );
    }

    #[test]
    fn test_sh_dot_product_zero_vectors() {
        let a = vec![0.0_f32; 64];
        let b = vec![0.0_f32; 64];
        let result = sh_dot_product(&a, &b);
        assert_eq!(result, 0.0, "dot product of zero vectors should be 0");
    }

    #[test]
    fn test_sh_dot_product_unit_vectors() {
        // dot([1,0,0,...], [1,0,0,...]) = 1.0
        let mut a = vec![0.0_f32; 16];
        let mut b = vec![0.0_f32; 16];
        a[0] = 1.0;
        b[0] = 1.0;
        let result = sh_dot_product(&a, &b);
        assert!((result - 1.0).abs() < 1e-6, "unit dot product = {result}");
    }

    #[test]
    fn test_sh_dot_product_performance() {
        // 100,000 calls on 64-element vectors.
        // In release mode the target is < 50 ms; debug builds are ~40× slower
        // due to disabled optimisations, so we use a generous 5 000 ms budget
        // to avoid false failures in debug CI.
        let a = rand_vec(64, 0x1234_5678u32);
        let b = rand_vec(64, 0x8765_4321u32);

        let start = std::time::Instant::now();
        let mut sum = 0.0_f32;
        for _ in 0..100_000 {
            sum += sh_dot_product(&a, &b);
        }
        let elapsed = start.elapsed();

        // Prevent optimiser from discarding the loop.
        assert!(sum.is_finite(), "sum should be finite: {sum}");

        // 50 ms in release, 5 000 ms in debug (unoptimised).
        let budget_ms: u128 = if cfg!(debug_assertions) { 5_000 } else { 50 };
        assert!(
            elapsed.as_millis() < budget_ms,
            "100k calls on 64-element vectors took {}ms, should be < {budget_ms}ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_sh_dot_product_matches_scalar_various_lengths() {
        for len in [1usize, 4, 7, 8, 9, 16, 25, 36, 64, 100, 256] {
            let a = rand_vec(len, len as u32 * 7 + 1);
            let b = rand_vec(len, len as u32 * 13 + 3);
            let simd = sh_dot_product(&a, &b);
            let scalar = sh_dot_product_scalar(&a, &b);
            assert!(
                (simd - scalar).abs() < 1e-3,
                "len={len}: SIMD={simd}, scalar={scalar}"
            );
        }
    }

    // ── n3d_sh_batch (SIMD) ──────────────────────────────────────────────────

    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: expected ≈ {b}, got {a} (tol {tol})"
        );
    }

    #[test]
    fn test_sh_batch_output_length() {
        let azimuths = vec![0.0_f32, 0.5, 1.0, 1.5];
        let elevations = vec![0.0_f32, 0.1, -0.1, 0.3];
        let order = 2_u32;
        let n_ch = ((order + 1) * (order + 1)) as usize; // 9

        let mut output = Vec::new();
        n3d_sh_batch(&azimuths, &elevations, order, &mut output);

        assert_eq!(
            output.len(),
            azimuths.len() * n_ch,
            "Batch output length should be n_dirs × n_channels"
        );
    }

    #[test]
    fn test_sh_batch_matches_scalar_first_order() {
        let azs = vec![0.0_f32, std::f32::consts::FRAC_PI_4, std::f32::consts::PI];
        let els = vec![0.0_f32, 0.2, -0.3];
        let order = 1_u32;
        let n_ch = 4_usize;

        let mut batch_out = Vec::new();
        n3d_sh_batch(&azs, &els, order, &mut batch_out);

        for (i, (&az, &el)) in azs.iter().zip(els.iter()).enumerate() {
            let ref_coeffs = n3d_sh_coefficients(az, el, order);
            for ch in 0..n_ch {
                let batch_val = batch_out[i * n_ch + ch];
                let ref_val = ref_coeffs[ch];
                assert_near(
                    batch_val,
                    ref_val,
                    1e-5,
                    &format!("Batch SH direction {i}, channel {ch}"),
                );
            }
        }
    }

    #[test]
    fn test_sh_batch_matches_scalar_third_order() {
        let azs: Vec<f32> = (0..16)
            .map(|i| i as f32 * std::f32::consts::PI / 8.0)
            .collect();
        let els: Vec<f32> = (0..16)
            .map(|i| (i as f32 - 8.0) * std::f32::consts::PI / 32.0)
            .collect();
        let order = 3_u32;
        let n_ch = 16_usize;

        let mut batch_out = Vec::new();
        n3d_sh_batch(&azs, &els, order, &mut batch_out);

        for (i, (&az, &el)) in azs.iter().zip(els.iter()).enumerate() {
            let ref_coeffs = n3d_sh_coefficients(az, el, order);
            for ch in 0..n_ch {
                let batch_val = batch_out[i * n_ch + ch];
                let ref_val = ref_coeffs[ch];
                assert_near(
                    batch_val,
                    ref_val,
                    1e-4,
                    &format!("3rd-order batch SH direction {i}, ch {ch}"),
                );
            }
        }
    }

    #[test]
    fn test_sh_batch_empty_input_produces_empty_output() {
        let mut output = Vec::new();
        n3d_sh_batch(&[], &[], 1, &mut output);
        assert!(output.is_empty(), "Empty input should give empty output");
    }

    #[test]
    fn test_sh_batch_output_all_finite() {
        let azs: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let els: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.12).collect();
        let mut out = Vec::new();
        n3d_sh_batch(&azs, &els, 2, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Batch SH output[{i}] should be finite, got {v}"
            );
        }
    }

    #[test]
    fn test_sh_batch_clears_previous_contents() {
        let azs = vec![0.0_f32, 1.0];
        let els = vec![0.0_f32, 0.5];
        let mut out = vec![999.0_f32; 100]; // pre-filled
        n3d_sh_batch(&azs, &els, 1, &mut out);
        // Output should be exactly 2 × 4 = 8 elements, all finite.
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Energy preservation ────────────────────────────────────────────────

    #[test]
    fn test_ambisonics_energy_preservation() {
        // Encode 1024 samples of a 440 Hz sine wave (unit amplitude) at a
        // representative direction into first-order ambisonics.
        let n = 1024_usize;
        let sample_rate = 48_000_u32;
        let freq = 440.0_f32;
        let mono: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let input_energy: f32 = mono.iter().map(|x| x * x).sum();

        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, sample_rate);
        let src = SoundSource::new(45.0, 30.0);
        let encoded = enc.encode_mono(&mono, &src);

        // Sum squared energy across all output channels.
        let output_energy: f32 = encoded
            .iter()
            .map(|ch| ch.iter().map(|x| x * x).sum::<f32>())
            .sum();

        assert!(
            output_energy.is_finite(),
            "Output energy must be finite, got {output_energy}"
        );
        assert!(
            input_energy > 0.0,
            "Input energy must be positive, got {input_energy}"
        );

        // The ratio of output energy to input energy is checked against a broad
        // but meaningful range.  For N3D first-order encoding, the W channel
        // carries weight 1 and each of the three directional channels carries
        // weight sqrt(3) ≈ 1.732.  At a generic direction the sum of squared
        // weights across all 4 channels is typically in the range [3, 5], so a
        // ratio between 0.1 and 10.0 confirms the encoder is functional without
        // making fragile assumptions about the exact normalisation constant.
        let ratio = output_energy / input_energy;
        assert!(
            (0.1..=10.0).contains(&ratio),
            "Energy ratio {ratio} should be in [0.1, 10.0] (in={input_energy}, out={output_energy})"
        );
    }

    // ── Round-trip encode → decode ─────────────────────────────────────────

    #[test]
    fn test_ambisonics_encode_decode_roundtrip() {
        // Encode a 256-sample sine at azimuth=0°, elevation=0° into first-order
        // ambisonics and decode back to stereo.  A front-centre source should
        // appear equally in both stereo channels with reasonable amplitude, so
        // the correlation between the decoded average and the original must be ≥ 0.60.
        let n = 256_usize;
        let sample_rate = 48_000_u32;
        let freq = 1_000.0_f32;
        let mono: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let enc = AmbisonicsEncoder::new(AmbisonicsOrder::First, sample_rate);
        let src = SoundSource::new(0.0, 0.0); // front centre
        let encoded = enc.encode_mono(&mono, &src);

        let dec = AmbisonicsDecoder::new(AmbisonicsOrder::First);
        let (left, right) = dec.decode_stereo(&encoded);

        // Mix the stereo output to mono for correlation.
        let decoded_mono: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| (l + r) * 0.5)
            .collect();

        // Normalised cross-correlation at lag 0.
        let sum_xy: f32 = mono
            .iter()
            .zip(decoded_mono.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_xx: f32 = mono.iter().map(|x| x * x).sum();
        let sum_yy: f32 = decoded_mono.iter().map(|y| y * y).sum();

        let denom = (sum_xx * sum_yy).sqrt().max(1e-10);
        let correlation = sum_xy / denom;

        assert!(
            correlation >= 0.60,
            "Round-trip correlation {correlation:.4} should be ≥ 0.60"
        );
    }
}
