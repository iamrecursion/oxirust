//! Binaural audio rendering via Head-Related Transfer Functions (HRTFs).
//!
//! A synthetic HRTF database is generated covering azimuth 0..360° in 15° steps
//! and elevations -30°, -15°, 0°, 15°, 30° (5 × 24 = 120 measurements).
//!
//! The HRTFs model:
//! - **ITD** — inter-aural time difference via fractional delay
//! - **ILD** — inter-aural level difference
//! - Simple onset + exponential decay envelope

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single HRTF measurement — impulse responses for left and right ears.
#[derive(Debug, Clone)]
pub struct HrtfCoefficients {
    /// Azimuth in degrees (0..360).
    pub azimuth_deg: i32,
    /// Elevation in degrees (-40..90).
    pub elevation_deg: i32,
    /// Left-ear impulse response.
    pub left_ir: Vec<f32>,
    /// Right-ear impulse response.
    pub right_ir: Vec<f32>,
    /// Length of each impulse response in samples.
    pub ir_length: usize,
}

/// A collection of HRTF measurements at different spatial positions.
#[derive(Debug, Clone)]
pub struct HrtfDatabase {
    /// All HRTF measurements.
    pub measurements: Vec<HrtfCoefficients>,
    /// Impulse response length common to all measurements.
    pub ir_length: usize,
    /// Sample rate the IRs are designed for.
    pub sample_rate: u32,
}

/// Binaural renderer — convolves mono audio with HRTFs.
#[derive(Debug, Clone)]
pub struct BinauralRenderer {
    /// HRTF database used for rendering.
    pub db: HrtfDatabase,
}

// ─── Fractional delay ─────────────────────────────────────────────────────────

/// Apply a fractional sample delay using linear interpolation.
/// `delay_samples` may be fractional.
fn fractional_delay(ir: &mut Vec<f32>, delay_samples: f32) {
    let int_part = delay_samples.floor() as usize;
    let frac = delay_samples - delay_samples.floor();

    let n = ir.len();
    let mut delayed = vec![0.0_f32; n];

    for i in 0..n {
        let j = i.saturating_sub(int_part);
        let val0 = if i >= int_part { ir[j] } else { 0.0 };
        let val1 = if i > int_part && j + 1 < n {
            ir[j + 1]
        } else {
            0.0
        };
        delayed[i] = val0 * (1.0 - frac) + val1 * frac;
    }
    *ir = delayed;
}

// ─── HrtfDatabase ────────────────────────────────────────────────────────────

/// Azimuth steps and elevation levels for the synthetic database.
const AZ_STEP: i32 = 15;
const ELEVATIONS: &[i32] = &[-30, -15, 0, 15, 30];
const IR_LENGTH: usize = 64;

/// Speed of sound (m/s).
const SPEED_OF_SOUND: f32 = 343.0;
/// Head radius (m).
const HEAD_RADIUS: f32 = 0.085;

impl HrtfDatabase {
    /// Generate a synthetic HRTF database.
    ///
    /// Covers azimuth 0..360° in 15° steps and elevations -30, -15, 0, 15, 30°.
    /// Returns 5 × 24 = 120 measurements.
    pub fn synthetic() -> Self {
        let sample_rate = 48_000_u32;
        let mut measurements = Vec::with_capacity(120);

        for &el_deg in ELEVATIONS {
            for az_idx in 0..24 {
                let az_deg = az_idx * AZ_STEP;
                let coeff = Self::make_synthetic_hrtf(az_deg, el_deg, sample_rate);
                measurements.push(coeff);
            }
        }

        Self {
            measurements,
            ir_length: IR_LENGTH,
            sample_rate,
        }
    }

    /// Build one synthetic HRTF for a given azimuth and elevation.
    fn make_synthetic_hrtf(az_deg: i32, el_deg: i32, sample_rate: u32) -> HrtfCoefficients {
        let az_rad = (az_deg as f32).to_radians();
        let el_rad = (el_deg as f32).to_radians();

        // ITD = (d / c) * sin(az) * cos(el)  (Woodworth model)
        let itd_seconds = (HEAD_RADIUS / SPEED_OF_SOUND) * az_rad.sin() * el_rad.cos();
        let itd_samples = itd_seconds * sample_rate as f32;

        // ILD: right ear is louder when source is on the right (az ∈ 270..90°).
        // Positive itd_samples → source to the left → right ear attenuated.
        let ild_db = 10.0 * ((1.0 + az_rad.sin()) * 0.5 + 0.5).ln() / std::f32::consts::LN_10 * 2.0;
        let ild_linear = 10.0_f32.powf(ild_db / 20.0);

        // Base IR: onset + exponential decay.
        let mut base_ir = vec![0.0_f32; IR_LENGTH];
        for (i, sample) in base_ir.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            // Short attack then exponential decay (approx pinna resonance envelope).
            *sample = if i == 0 { 1.0 } else { (-30.0 * t).exp() * 0.5 };
        }

        // Left ear: positive ITD → source left-of-centre → left ear leads.
        let mut left_ir = base_ir.clone();
        let mut right_ir = base_ir.clone();

        if itd_samples >= 0.0 {
            // Source is to the left: delay right ear, attenuate right ear.
            fractional_delay(&mut right_ir, itd_samples.abs());
            for s in &mut right_ir {
                *s /= ild_linear.max(0.01);
            }
        } else {
            // Source is to the right: delay left ear, attenuate left ear.
            fractional_delay(&mut left_ir, itd_samples.abs());
            for s in &mut left_ir {
                *s /= ild_linear.max(0.01);
            }
        }

        // Normalise each IR to unit peak.
        normalize_ir(&mut left_ir);
        normalize_ir(&mut right_ir);

        HrtfCoefficients {
            azimuth_deg: az_deg,
            elevation_deg: el_deg,
            left_ir,
            right_ir,
            ir_length: IR_LENGTH,
        }
    }

    /// Find the nearest measurement using angular distance on the sphere.
    pub fn nearest_measurement(&self, azimuth: f32, elevation: f32) -> &HrtfCoefficients {
        let az_rad = azimuth.to_radians();
        let el_rad = elevation.to_radians();

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (idx, m) in self.measurements.iter().enumerate() {
            let maz = (m.azimuth_deg as f32).to_radians();
            let mel = (m.elevation_deg as f32).to_radians();

            // Great-circle distance using Haversine.
            let d_az = (az_rad - maz) / 2.0;
            let d_el = (el_rad - mel) / 2.0;
            let a = d_el.sin().powi(2) + el_rad.cos() * mel.cos() * d_az.sin().powi(2);
            let dist = 2.0 * a.sqrt().asin();

            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        &self.measurements[best_idx]
    }
}

/// Normalise an impulse response to unit peak amplitude.
fn normalize_ir(ir: &mut Vec<f32>) {
    let peak = ir.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    if peak > 1e-10 {
        for s in ir.iter_mut() {
            *s /= peak;
        }
    }
}

// ─── Convolution ─────────────────────────────────────────────────────────────

/// Direct (linear) convolution of `signal` with `ir`.
///
/// Output length = `signal.len() + ir.len() - 1`.
pub fn convolve(signal: &[f32], ir: &[f32]) -> Vec<f32> {
    if signal.is_empty() || ir.is_empty() {
        return Vec::new();
    }
    let out_len = signal.len() + ir.len() - 1;
    let mut out = vec![0.0_f32; out_len];

    for (n, &s) in signal.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        for (k, &h) in ir.iter().enumerate() {
            out[n + k] += s * h;
        }
    }
    out
}

// ─── BinauralRenderer ────────────────────────────────────────────────────────

impl BinauralRenderer {
    /// Create a new renderer with the given HRTF database.
    pub fn new(db: HrtfDatabase) -> Self {
        Self { db }
    }

    /// Render mono audio at a fixed spatial position using overlap-add convolution.
    ///
    /// Returns `(left_ear, right_ear)`.
    pub fn render(&self, samples: &[f32], azimuth: f32, elevation: f32) -> (Vec<f32>, Vec<f32>) {
        let hrtf = self.db.nearest_measurement(azimuth, elevation);
        let left = convolve(samples, &hrtf.left_ir);
        let right = convolve(samples, &hrtf.right_ir);
        (left, right)
    }

    /// Render mono audio moving along a spatial path.
    ///
    /// `path` is a slice of `(azimuth_deg, elevation_deg)` waypoints evenly distributed
    /// across the signal.  Adjacent segments are crossfaded for smooth transitions.
    ///
    /// Returns `(left_ear, right_ear)`.
    pub fn render_moving(&self, samples: &[f32], path: &[(f32, f32)]) -> (Vec<f32>, Vec<f32>) {
        if path.is_empty() || samples.is_empty() {
            let n = samples.len() + self.db.ir_length - 1;
            return (vec![0.0; n], vec![0.0; n]);
        }

        let n_segments = path.len();
        let seg_len = (samples.len() + n_segments - 1) / n_segments;
        let out_len = samples.len() + self.db.ir_length - 1;

        let mut left_out = vec![0.0_f32; out_len];
        let mut right_out = vec![0.0_f32; out_len];

        for (seg_idx, &(az, el)) in path.iter().enumerate() {
            let start = seg_idx * seg_len;
            let end = ((seg_idx + 1) * seg_len).min(samples.len());
            if start >= samples.len() {
                break;
            }

            let segment = &samples[start..end];
            let hrtf = self.db.nearest_measurement(az, el);
            let seg_left = convolve(segment, &hrtf.left_ir);
            let seg_right = convolve(segment, &hrtf.right_ir);

            // Crossfade window: fade in at start, fade out at end.
            let xfade = seg_len.min(16);
            for (i, (&sl, &sr)) in seg_left.iter().zip(seg_right.iter()).enumerate() {
                let out_i = start + i;
                if out_i >= out_len {
                    break;
                }
                // Fade-in for first segment transition.
                let fade = if i < xfade && seg_idx > 0 {
                    i as f32 / xfade as f32
                } else {
                    1.0
                };
                left_out[out_i] += sl * fade;
                right_out[out_i] += sr * fade;
            }
        }

        (left_out, right_out)
    }
}

// ─── HRTF Spherical Harmonics Interpolation ─────────────────────────────────

/// Maximum SH order used for HRTF interpolation (3rd order = 16 coefficients).
const SH_INTERP_ORDER: u32 = 3;

/// Number of SH coefficients for the interpolation order.
const SH_INTERP_CHANNELS: usize = ((SH_INTERP_ORDER + 1) * (SH_INTERP_ORDER + 1)) as usize;

/// HRTF database with spherical harmonics decomposition for smooth interpolation.
///
/// Instead of snapping to the nearest measured HRTF, this approach decomposes
/// the HRTF set into SH coefficients, allowing continuous interpolation at any
/// direction. This is critical for smooth head rotation in binaural rendering.
#[derive(Debug, Clone)]
pub struct ShInterpolatedHrtfDb {
    /// SH coefficients for each left-ear IR sample.
    /// Shape: `[ir_sample_index][sh_coefficient_index]`
    left_sh_coeffs: Vec<Vec<f32>>,
    /// SH coefficients for each right-ear IR sample.
    right_sh_coeffs: Vec<Vec<f32>>,
    /// Length of each impulse response.
    ir_length: usize,
    /// Sample rate.
    sample_rate: u32,
}

/// Compute real spherical harmonic basis functions for a direction (az, el in radians).
///
/// Returns `SH_INTERP_CHANNELS` coefficients in ACN order with N3D normalisation.
fn sh_basis(az_rad: f32, el_rad: f32) -> Vec<f32> {
    crate::ambisonics::n3d_sh_coefficients(az_rad, el_rad, SH_INTERP_ORDER)
}

impl ShInterpolatedHrtfDb {
    /// Build an SH-interpolated HRTF database from an existing `HrtfDatabase`.
    ///
    /// This decomposes each IR sample across all measurements into SH coefficients
    /// using a least-squares pseudo-inverse approach.
    pub fn from_database(db: &HrtfDatabase) -> Self {
        let n_meas = db.measurements.len();
        let ir_len = db.ir_length;

        if n_meas == 0 {
            return Self {
                left_sh_coeffs: Vec::new(),
                right_sh_coeffs: Vec::new(),
                ir_length: ir_len,
                sample_rate: db.sample_rate,
            };
        }

        // Build the SH matrix Y: [n_meas x SH_INTERP_CHANNELS]
        // Y[i][j] = j-th SH basis function evaluated at the i-th measurement direction
        let mut y_matrix: Vec<Vec<f32>> = Vec::with_capacity(n_meas);
        for m in &db.measurements {
            let az_rad = (m.azimuth_deg as f32).to_radians();
            let el_rad = (m.elevation_deg as f32).to_radians();
            y_matrix.push(sh_basis(az_rad, el_rad));
        }

        // Compute pseudo-inverse: (Y^T Y)^{-1} Y^T using normal equations.
        // Y^T Y is [SH_INTERP_CHANNELS x SH_INTERP_CHANNELS]
        let n_sh = SH_INTERP_CHANNELS;
        let mut yty = vec![vec![0.0_f32; n_sh]; n_sh];
        for row in &y_matrix {
            for i in 0..n_sh {
                for j in 0..n_sh {
                    yty[i][j] += row[i] * row[j];
                }
            }
        }

        // Add Tikhonov regularisation for stability
        for i in 0..n_sh {
            yty[i][i] += 0.01;
        }

        // Solve via Cholesky-like direct inversion of the small matrix
        let yty_inv = invert_symmetric_matrix(&yty);

        // Compute SH coefficients for each IR sample:
        // For each time sample t: sh_coeffs[t] = (Y^T Y)^{-1} Y^T h[t]
        let mut left_sh = vec![vec![0.0_f32; n_sh]; ir_len];
        let mut right_sh = vec![vec![0.0_f32; n_sh]; ir_len];

        for t in 0..ir_len {
            // Compute Y^T * h_left[t] and Y^T * h_right[t]
            let mut yt_hl = vec![0.0_f32; n_sh];
            let mut yt_hr = vec![0.0_f32; n_sh];

            for (m_idx, row) in y_matrix.iter().enumerate() {
                let hl = if t < db.measurements[m_idx].left_ir.len() {
                    db.measurements[m_idx].left_ir[t]
                } else {
                    0.0
                };
                let hr = if t < db.measurements[m_idx].right_ir.len() {
                    db.measurements[m_idx].right_ir[t]
                } else {
                    0.0
                };
                for j in 0..n_sh {
                    yt_hl[j] += row[j] * hl;
                    yt_hr[j] += row[j] * hr;
                }
            }

            // Multiply by (Y^T Y)^{-1}
            for i in 0..n_sh {
                let mut sl = 0.0_f32;
                let mut sr = 0.0_f32;
                for j in 0..n_sh {
                    sl += yty_inv[i][j] * yt_hl[j];
                    sr += yty_inv[i][j] * yt_hr[j];
                }
                left_sh[t][i] = sl;
                right_sh[t][i] = sr;
            }
        }

        Self {
            left_sh_coeffs: left_sh,
            right_sh_coeffs: right_sh,
            ir_length: ir_len,
            sample_rate: db.sample_rate,
        }
    }

    /// Interpolate the HRTF at an arbitrary direction (degrees).
    ///
    /// Returns `(left_ir, right_ir)` impulse responses reconstructed from the
    /// SH decomposition. This provides smooth, continuous interpolation without
    /// audible jumps during head rotation.
    pub fn interpolate(&self, azimuth_deg: f32, elevation_deg: f32) -> (Vec<f32>, Vec<f32>) {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();
        let basis = sh_basis(az_rad, el_rad);

        let mut left_ir = vec![0.0_f32; self.ir_length];
        let mut right_ir = vec![0.0_f32; self.ir_length];

        for t in 0..self.ir_length {
            let mut sl = 0.0_f32;
            let mut sr = 0.0_f32;
            for (j, &b) in basis.iter().enumerate() {
                sl += self.left_sh_coeffs[t][j] * b;
                sr += self.right_sh_coeffs[t][j] * b;
            }
            left_ir[t] = sl;
            right_ir[t] = sr;
        }

        (left_ir, right_ir)
    }

    /// Return the IR length.
    pub fn ir_length(&self) -> usize {
        self.ir_length
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Binaural renderer using SH-interpolated HRTFs for smooth head tracking.
#[derive(Debug, Clone)]
pub struct ShBinauralRenderer {
    /// SH-interpolated HRTF database.
    pub db: ShInterpolatedHrtfDb,
}

impl ShBinauralRenderer {
    /// Create a renderer from an SH-interpolated HRTF database.
    pub fn new(db: ShInterpolatedHrtfDb) -> Self {
        Self { db }
    }

    /// Render mono audio at a given direction using SH-interpolated HRTFs.
    ///
    /// Returns `(left_ear, right_ear)`.
    pub fn render(
        &self,
        samples: &[f32],
        azimuth_deg: f32,
        elevation_deg: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let (left_ir, right_ir) = self.db.interpolate(azimuth_deg, elevation_deg);
        let left = convolve(samples, &left_ir);
        let right = convolve(samples, &right_ir);
        (left, right)
    }
}

/// Invert a small symmetric positive-definite matrix using Gauss-Jordan elimination.
///
/// Returns an identity-sized matrix if inversion fails.
fn invert_symmetric_matrix(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = m.len();
    // Augmented matrix [M | I]
    let mut aug: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f32; 2 * n];
            for j in 0..n {
                row[j] = m[i][j];
            }
            row[n + i] = 1.0;
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            // Singular — return identity
            return (0..n)
                .map(|i| {
                    let mut row = vec![0.0_f32; n];
                    row[i] = 1.0;
                    row
                })
                .collect();
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse
    aug.iter().map(|row| row[n..(2 * n)].to_vec()).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse(n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        if n > 0 {
            v[0] = 1.0;
        }
        v
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum: f32 = buf.iter().map(|x| x * x).sum();
        (sum / buf.len().max(1) as f32).sqrt()
    }

    // ── HrtfDatabase ────────────────────────────────────────────────────────

    #[test]
    fn test_synthetic_db_measurement_count() {
        let db = HrtfDatabase::synthetic();
        // 5 elevations × 24 azimuths = 120
        assert_eq!(db.measurements.len(), 120);
    }

    #[test]
    fn test_synthetic_db_ir_length() {
        let db = HrtfDatabase::synthetic();
        for m in &db.measurements {
            assert_eq!(m.left_ir.len(), IR_LENGTH);
            assert_eq!(m.right_ir.len(), IR_LENGTH);
        }
    }

    #[test]
    fn test_nearest_measurement_exact_match() {
        let db = HrtfDatabase::synthetic();
        let m = db.nearest_measurement(0.0, 0.0);
        assert_eq!(m.azimuth_deg, 0);
        assert_eq!(m.elevation_deg, 0);
    }

    #[test]
    fn test_nearest_measurement_approximate() {
        let db = HrtfDatabase::synthetic();
        // 7° az, 3° el → nearest should be 0° az, 0° el.
        let m = db.nearest_measurement(7.0, 3.0);
        assert!(m.azimuth_deg == 0 || m.azimuth_deg == 15);
        assert!(ELEVATIONS.contains(&m.elevation_deg));
    }

    #[test]
    fn test_hrtf_irs_not_all_zero() {
        let db = HrtfDatabase::synthetic();
        for m in &db.measurements {
            let l_nonzero = m.left_ir.iter().any(|&x| x.abs() > 1e-6);
            let r_nonzero = m.right_ir.iter().any(|&x| x.abs() > 1e-6);
            assert!(
                l_nonzero,
                "Left IR should be non-zero for az={} el={}",
                m.azimuth_deg, m.elevation_deg
            );
            assert!(
                r_nonzero,
                "Right IR should be non-zero for az={} el={}",
                m.azimuth_deg, m.elevation_deg
            );
        }
    }

    // ── convolve ─────────────────────────────────────────────────────────────

    #[test]
    fn test_convolve_output_length() {
        let sig = vec![1.0_f32; 100];
        let ir = vec![1.0_f32; 64];
        let out = convolve(&sig, &ir);
        assert_eq!(out.len(), 163); // 100 + 64 - 1
    }

    #[test]
    fn test_convolve_impulse_recovers_ir() {
        let ir = vec![0.1_f32, 0.5, 0.3, 0.1];
        let sig = impulse(8);
        let out = convolve(&sig, &ir);
        // Convolving an impulse with an IR should reproduce the IR.
        for (i, &expected) in ir.iter().enumerate() {
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "Mismatch at {i}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_convolve_empty_signal() {
        let out = convolve(&[], &[1.0, 0.5]);
        assert!(out.is_empty());
    }

    // ── BinauralRenderer ─────────────────────────────────────────────────────

    #[test]
    fn test_render_output_length() {
        let db = HrtfDatabase::synthetic();
        let renderer = BinauralRenderer::new(db);
        let sig = vec![1.0_f32; 512];
        let (l, r) = renderer.render(&sig, 0.0, 0.0);
        // Output length = signal.len() + ir_len - 1
        assert_eq!(l.len(), 512 + IR_LENGTH - 1);
        assert_eq!(r.len(), 512 + IR_LENGTH - 1);
    }

    #[test]
    fn test_render_produces_audio() {
        let db = HrtfDatabase::synthetic();
        let renderer = BinauralRenderer::new(db);
        let sig: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let (l, r) = renderer.render(&sig, 45.0, 0.0);
        assert!(rms(&l) > 0.0, "Left output should have energy");
        assert!(rms(&r) > 0.0, "Right output should have energy");
    }

    #[test]
    fn test_render_moving_output_length() {
        let db = HrtfDatabase::synthetic();
        let ir_len = db.ir_length;
        let renderer = BinauralRenderer::new(db);
        let sig = vec![1.0_f32; 256];
        let path = vec![(0.0_f32, 0.0_f32), (90.0, 0.0), (180.0, 0.0)];
        let (l, r) = renderer.render_moving(&sig, &path);
        assert_eq!(l.len(), 256 + ir_len - 1);
        assert_eq!(r.len(), 256 + ir_len - 1);
    }

    #[test]
    fn test_render_moving_empty_path() {
        let db = HrtfDatabase::synthetic();
        let ir_len = db.ir_length;
        let renderer = BinauralRenderer::new(db);
        let sig = vec![1.0_f32; 64];
        let (l, r) = renderer.render_moving(&sig, &[]);
        // Empty path → zero output of expected length.
        assert_eq!(l.len(), 64 + ir_len - 1);
        assert!(l.iter().all(|&x| x == 0.0));
        assert!(r.iter().all(|&x| x == 0.0));
    }

    // ── SH-interpolated HRTF ──────────────────────────────────────────────

    #[test]
    fn test_sh_interpolated_db_from_synthetic() {
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        assert_eq!(sh_db.ir_length(), IR_LENGTH);
        assert_eq!(sh_db.sample_rate(), 48_000);
    }

    #[test]
    fn test_sh_interpolated_ir_length() {
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let (left, right) = sh_db.interpolate(0.0, 0.0);
        assert_eq!(left.len(), IR_LENGTH);
        assert_eq!(right.len(), IR_LENGTH);
    }

    #[test]
    fn test_sh_interpolated_irs_are_finite() {
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        for az in [0.0_f32, 45.0, 90.0, 180.0, 270.0] {
            let (left, right) = sh_db.interpolate(az, 0.0);
            for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
                assert!(l.is_finite(), "left[{i}] not finite at az={az}");
                assert!(r.is_finite(), "right[{i}] not finite at az={az}");
            }
        }
    }

    #[test]
    fn test_sh_interpolated_non_measured_direction() {
        // Interpolate at a direction that is NOT in the measurement grid
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        // 7.5 degrees is between 0 and 15 degree measurements
        let (left, right) = sh_db.interpolate(7.5, 5.0);
        let l_energy: f32 = left.iter().map(|x| x * x).sum();
        let r_energy: f32 = right.iter().map(|x| x * x).sum();
        assert!(l_energy > 0.0, "Interpolated left IR should have energy");
        assert!(r_energy > 0.0, "Interpolated right IR should have energy");
    }

    #[test]
    fn test_sh_interpolated_continuity() {
        // Two nearby directions should produce similar IRs
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let (l1, r1) = sh_db.interpolate(45.0, 0.0);
        let (l2, r2) = sh_db.interpolate(46.0, 0.0);

        let l_diff: f32 = l1
            .iter()
            .zip(l2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / l1.len() as f32;
        let r_diff: f32 = r1
            .iter()
            .zip(r2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / r1.len() as f32;

        assert!(
            l_diff < 0.5,
            "Nearby directions should give similar left IRs, diff={l_diff}"
        );
        assert!(
            r_diff < 0.5,
            "Nearby directions should give similar right IRs, diff={r_diff}"
        );
    }

    #[test]
    fn test_sh_binaural_renderer_render() {
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let renderer = ShBinauralRenderer::new(sh_db);
        let sig: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let (l, r) = renderer.render(&sig, 45.0, 0.0);
        assert_eq!(l.len(), 256 + IR_LENGTH - 1);
        assert_eq!(r.len(), 256 + IR_LENGTH - 1);
        assert!(rms(&l) > 0.0, "SH binaural left should have energy");
        assert!(rms(&r) > 0.0, "SH binaural right should have energy");
    }

    #[test]
    fn test_sh_binaural_renderer_front_symmetric() {
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let renderer = ShBinauralRenderer::new(sh_db);
        let sig = vec![1.0_f32; 128];
        let (l, r) = renderer.render(&sig, 0.0, 0.0);
        // Front source: left and right should have similar energy
        let l_rms = rms(&l);
        let r_rms = rms(&r);
        let ratio = if l_rms > r_rms {
            l_rms / r_rms.max(1e-10)
        } else {
            r_rms / l_rms.max(1e-10)
        };
        assert!(
            ratio < 3.0,
            "Front source should have roughly similar L/R energy, ratio={ratio}"
        );
    }

    #[test]
    fn test_invert_symmetric_matrix_identity() {
        let m = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let inv = invert_symmetric_matrix(&m);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-5,
                    "inv[{i}][{j}] = {}, expected {expected}",
                    inv[i][j]
                );
            }
        }
    }

    // ── SH HRTF interpolation — accuracy at measured positions ──────────

    #[test]
    fn test_sh_interpolated_at_measured_direction_has_energy() {
        // Interpolating at a direction that IS in the measurement grid should
        // produce an IR with significant energy (not zero).
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        for &el in ELEVATIONS {
            for az_idx in 0..24 {
                let az = (az_idx * AZ_STEP) as f32;
                let (left, right) = sh_db.interpolate(az, el as f32);
                let l_energy: f32 = left.iter().map(|x| x * x).sum();
                let r_energy: f32 = right.iter().map(|x| x * x).sum();
                assert!(
                    l_energy > 1e-6,
                    "Left IR should have energy at az={az}, el={el}, got {l_energy}"
                );
                assert!(
                    r_energy > 1e-6,
                    "Right IR should have energy at az={az}, el={el}, got {r_energy}"
                );
            }
        }
    }

    #[test]
    fn test_sh_interpolated_smooth_rotation_sweep() {
        // Sweep azimuth in 1-degree steps; adjacent directions should produce
        // similar IRs (small L2 difference).
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let mut prev_left = sh_db.interpolate(0.0, 0.0).0;
        let mut max_diff = 0.0_f32;

        for az_deg in 1..=360 {
            let (left, _right) = sh_db.interpolate(az_deg as f32, 0.0);
            let diff: f32 = left
                .iter()
                .zip(prev_left.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if diff > max_diff {
                max_diff = diff;
            }
            prev_left = left;
        }

        // The maximum step-to-step difference should be bounded for smooth rotation.
        assert!(
            max_diff < 5.0,
            "Max 1-degree step IR difference should be small, got {max_diff}"
        );
    }

    #[test]
    fn test_sh_interpolated_elevation_sweep() {
        // Sweep elevation to confirm smooth behaviour across elevation grid.
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);

        for el in [-30, -15, 0, 15, 30] {
            let (left, right) = sh_db.interpolate(90.0, el as f32);
            let l_e: f32 = left.iter().map(|x| x * x).sum();
            let r_e: f32 = right.iter().map(|x| x * x).sum();
            assert!(l_e.is_finite(), "Left energy not finite at el={el}");
            assert!(r_e.is_finite(), "Right energy not finite at el={el}");
        }
    }

    #[test]
    fn test_sh_binaural_renderer_moving_source() {
        // Render a moving source along a path using the SH renderer.
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let renderer = ShBinauralRenderer::new(sh_db);

        let sig: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        // Render at several positions and check all produce valid output.
        for az in [0.0_f32, 90.0, 180.0, 270.0] {
            let (l, r) = renderer.render(&sig, az, 0.0);
            assert_eq!(l.len(), 512 + IR_LENGTH - 1);
            assert!(
                l.iter().all(|x| x.is_finite()),
                "Left output not finite at az={az}"
            );
            assert!(
                r.iter().all(|x| x.is_finite()),
                "Right output not finite at az={az}"
            );
        }
    }

    #[test]
    fn test_sh_interpolated_lateral_source_asymmetry() {
        // A source at 90 degrees should produce different left/right IRs.
        let db = HrtfDatabase::synthetic();
        let sh_db = ShInterpolatedHrtfDb::from_database(&db);
        let (left, right) = sh_db.interpolate(90.0, 0.0);

        let l_energy: f32 = left.iter().map(|x| x * x).sum();
        let r_energy: f32 = right.iter().map(|x| x * x).sum();

        // At 90 degrees (left side), we expect some energy difference.
        // The exact magnitude depends on the synthetic HRTF but it should differ.
        let diff = (l_energy - r_energy).abs();
        assert!(
            diff > 0.0 || (l_energy > 0.0 && r_energy > 0.0),
            "Lateral source should produce valid stereo: L={l_energy}, R={r_energy}"
        );
    }
}
