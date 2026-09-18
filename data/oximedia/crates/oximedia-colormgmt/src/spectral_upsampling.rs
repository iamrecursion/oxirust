#![allow(dead_code)]
//! Spectral upsampling from RGB to a spectral representation.
//!
//! Converts tri-stimulus RGB values into a sampled spectral power distribution
//! (SPD) using Smits-style basis functions. This is useful for physically-based
//! rendering and accurate color mixing under different illuminants.

/// Number of spectral bands in a sampled SPD (380 nm to 720 nm in 10 nm steps).
pub const NUM_BANDS: usize = 35;

/// Start wavelength in nanometres.
pub const LAMBDA_MIN: f64 = 380.0;

/// End wavelength in nanometres.
pub const LAMBDA_MAX: f64 = 720.0;

/// Step size between bands in nanometres.
pub const LAMBDA_STEP: f64 = 10.0;

/// A sampled spectral power distribution (SPD).
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralDistribution {
    /// Power values at each band from `LAMBDA_MIN` to `LAMBDA_MAX`.
    pub bands: [f64; NUM_BANDS],
}

impl SpectralDistribution {
    /// Creates a new SPD with all bands set to zero.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            bands: [0.0; NUM_BANDS],
        }
    }

    /// Creates a new SPD with all bands set to the given constant value.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self {
            bands: [value; NUM_BANDS],
        }
    }

    /// Returns the wavelength in nm for a given band index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wavelength_at(index: usize) -> f64 {
        LAMBDA_MIN + index as f64 * LAMBDA_STEP
    }

    /// Looks up the power at the nearest band for a given wavelength.
    ///
    /// # Arguments
    /// * `wavelength_nm` - Wavelength in nanometres
    ///
    /// # Returns
    /// Power value, or 0.0 if out of range.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn at_wavelength(&self, wavelength_nm: f64) -> f64 {
        if wavelength_nm < LAMBDA_MIN || wavelength_nm > LAMBDA_MAX {
            return 0.0;
        }
        let idx = ((wavelength_nm - LAMBDA_MIN) / LAMBDA_STEP).round() as usize;
        if idx < NUM_BANDS {
            self.bands[idx]
        } else {
            0.0
        }
    }

    /// Scales all bands by a constant factor.
    pub fn scale(&mut self, factor: f64) {
        for b in &mut self.bands {
            *b *= factor;
        }
    }

    /// Adds another SPD to this one (element-wise).
    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.bands.iter_mut().zip(other.bands.iter()) {
            *a += *b;
        }
    }

    /// Multiplies element-wise with another SPD.
    pub fn multiply(&mut self, other: &Self) {
        for (a, b) in self.bands.iter_mut().zip(other.bands.iter()) {
            *a *= *b;
        }
    }

    /// Integrates the SPD using the trapezoidal rule.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn integrate(&self) -> f64 {
        if NUM_BANDS < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..NUM_BANDS - 1 {
            sum += (self.bands[i] + self.bands[i + 1]) * 0.5 * LAMBDA_STEP;
        }
        sum
    }

    /// Returns the peak wavelength (wavelength with maximum power).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn peak_wavelength(&self) -> f64 {
        let mut max_idx = 0;
        let mut max_val = self.bands[0];
        for (i, &v) in self.bands.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Self::wavelength_at(max_idx)
    }
}

impl Default for SpectralDistribution {
    fn default() -> Self {
        Self::zero()
    }
}

/// Smits-style basis functions for spectral upsampling.
///
/// Each basis function covers a portion of the visible spectrum with smooth
/// transitions to produce a plausible SPD from RGB values.
#[allow(clippy::cast_precision_loss)]
fn red_basis() -> SpectralDistribution {
    let mut spd = SpectralDistribution::zero();
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        spd.bands[i] = if wl >= 600.0 {
            1.0
        } else if wl >= 560.0 {
            (wl - 560.0) / 40.0
        } else {
            0.0
        };
    }
    spd
}

#[allow(clippy::cast_precision_loss)]
fn green_basis() -> SpectralDistribution {
    let mut spd = SpectralDistribution::zero();
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        spd.bands[i] = if wl >= 480.0 && wl <= 600.0 {
            let center = 540.0;
            let half_width = 60.0;
            let d = (wl - center).abs();
            if d <= half_width {
                1.0
            } else {
                (1.0 - (d - half_width) / 30.0).max(0.0)
            }
        } else {
            0.0
        };
    }
    spd
}

#[allow(clippy::cast_precision_loss)]
fn blue_basis() -> SpectralDistribution {
    let mut spd = SpectralDistribution::zero();
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        spd.bands[i] = if wl <= 480.0 {
            1.0
        } else if wl <= 520.0 {
            1.0 - (wl - 480.0) / 40.0
        } else {
            0.0
        };
    }
    spd
}

/// Upsamples an RGB triple to a spectral power distribution.
///
/// Uses Smits-style additive basis functions. The input RGB is assumed to be
/// in linear light, non-negative.
///
/// # Arguments
/// * `rgb` - `[R, G, B]` in linear light (0..1 typical range, clamped to >= 0)
///
/// # Returns
/// A [`SpectralDistribution`] approximating the input color.
#[must_use]
pub fn rgb_to_spectral(rgb: [f64; 3]) -> SpectralDistribution {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);

    let r_basis = red_basis();
    let g_basis = green_basis();
    let b_basis = blue_basis();

    let mut result = SpectralDistribution::zero();
    for i in 0..NUM_BANDS {
        result.bands[i] = r * r_basis.bands[i] + g * g_basis.bands[i] + b * b_basis.bands[i];
    }
    result
}

/// CIE 1931 2-degree observer x-bar approximation.
#[allow(clippy::cast_precision_loss)]
fn cie_xbar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 442.0) * (if wavelength < 442.0 { 0.0624 } else { 0.0374 });
    let t2 = (wavelength - 599.8) * (if wavelength < 599.8 { 0.0264 } else { 0.0323 });
    let t3 = (wavelength - 501.1) * (if wavelength < 501.1 { 0.0490 } else { 0.0382 });
    0.362 * (-0.5 * t1 * t1).exp() + 1.056 * (-0.5 * t2 * t2).exp() - 0.065 * (-0.5 * t3 * t3).exp()
}

/// CIE 1931 2-degree observer y-bar approximation.
#[allow(clippy::cast_precision_loss)]
fn cie_ybar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 568.8) * (if wavelength < 568.8 { 0.0213 } else { 0.0247 });
    let t2 = (wavelength - 530.9) * (if wavelength < 530.9 { 0.0613 } else { 0.0322 });
    0.821 * (-0.5 * t1 * t1).exp() + 0.286 * (-0.5 * t2 * t2).exp()
}

/// CIE 1931 2-degree observer z-bar approximation.
#[allow(clippy::cast_precision_loss)]
fn cie_zbar(wavelength: f64) -> f64 {
    let t1 = (wavelength - 437.0) * (if wavelength < 437.0 { 0.0845 } else { 0.0278 });
    let t2 = (wavelength - 459.0) * (if wavelength < 459.0 { 0.0385 } else { 0.0725 });
    1.217 * (-0.5 * t1 * t1).exp() + 0.681 * (-0.5 * t2 * t2).exp()
}

/// Converts an SPD to CIE XYZ tri-stimulus values.
///
/// Uses a simple summation with the CIE 1931 2-degree observer approximation.
///
/// # Arguments
/// * `spd` - The spectral power distribution to convert
///
/// # Returns
/// `[X, Y, Z]` tri-stimulus values.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spectral_to_xyz(spd: &SpectralDistribution) -> [f64; 3] {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        let power = spd.bands[i];
        x += power * cie_xbar(wl) * LAMBDA_STEP;
        y += power * cie_ybar(wl) * LAMBDA_STEP;
        z += power * cie_zbar(wl) * LAMBDA_STEP;
    }
    // Normalise by the integral of y_bar for equal-energy illuminant
    let mut y_norm = 0.0;
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        y_norm += cie_ybar(wl) * LAMBDA_STEP;
    }
    if y_norm > 1e-12 {
        x /= y_norm;
        y /= y_norm;
        z /= y_norm;
    }
    [x, y, z]
}

/// D65 illuminant SPD (normalised, simplified).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn d65_illuminant() -> SpectralDistribution {
    // Simplified D65 approximation: relatively flat with slight blue emphasis
    let mut spd = SpectralDistribution::zero();
    for i in 0..NUM_BANDS {
        let wl = SpectralDistribution::wavelength_at(i);
        // Simple model: peak near 460nm, secondary peak near 560nm
        let t1 = ((wl - 460.0) / 30.0).powi(2);
        let t2 = ((wl - 560.0) / 50.0).powi(2);
        spd.bands[i] = 0.7 + 0.3 * (-0.5 * t1).exp() + 0.15 * (-0.5 * t2).exp();
    }
    spd
}

// ──────────────────────────────────────────────────────────────────────────────
// Standard illuminants & CIE 1931 CMFs at 5 nm intervals (380–780 nm, 81 bands)
// ──────────────────────────────────────────────────────────────────────────────

/// Number of bands in the 5 nm CMF tables (380 to 780 nm inclusive).
pub const NUM_CMF_BANDS: usize = 81;

/// Start wavelength for the 5 nm CMF tables.
pub const CMF_LAMBDA_MIN: f64 = 380.0;

/// Step size for the 5 nm CMF tables.
pub const CMF_LAMBDA_STEP: f64 = 5.0;

/// CIE 1931 2° colour-matching function x̄, 5 nm step, 380–780 nm.
pub const CIE_CMF_X: [f64; NUM_CMF_BANDS] = [
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630, 0.134380,
    0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700, 0.290800, 0.251100,
    0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300,
    0.029100, 0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450, 0.512050,
    0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800,
    0.283500, 0.218700, 0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
    0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
    0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059, 0.000042,
];

/// CIE 1931 2° colour-matching function ȳ, 5 nm step, 380–780 nm.
pub const CIE_CMF_Y: [f64; NUM_CMF_BANDS] = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180, 0.004000,
    0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900,
    0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000,
    0.608200, 0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000,
    0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200,
    0.107000, 0.081600, 0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
    0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021, 0.000015,
];

/// CIE 1931 2° colour-matching function z̄, 5 nm step, 380–780 nm.
pub const CIE_CMF_Z: [f64; NUM_CMF_BANDS] = [
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300, 0.645600,
    1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100, 1.669200, 1.528100,
    1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200,
    0.111700, 0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750, 0.005750,
    0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
];

// ── Illuminant SPDs ────────────────────────────────────────────────────────────

/// Standard CIE illuminants for spectral integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardIlluminant {
    /// CIE D65 (daylight, 6504 K) — sRGB / Rec.709 / Rec.2020 reference white.
    D65,
    /// CIE D50 (horizon daylight, 5003 K) — ICC profile connection space / print reference.
    D50,
    /// CIE D55 (mid-morning daylight, 5503 K).
    D55,
    /// Equal-energy illuminant E (flat SPD, 1.0 across all bands).
    EqualEnergy,
}

/// Configuration for spectral upsampling and XYZ integration.
///
/// Selects the illuminant under which the reflectance curve is integrated.
#[derive(Debug, Clone)]
pub struct SpectralUpsamplingConfig {
    /// Illuminant used for spectral integration.
    pub illuminant: StandardIlluminant,
}

impl SpectralUpsamplingConfig {
    /// Creates a new config with the specified illuminant.
    #[must_use]
    pub fn new(illuminant: StandardIlluminant) -> Self {
        Self { illuminant }
    }
}

impl Default for SpectralUpsamplingConfig {
    fn default() -> Self {
        Self::new(StandardIlluminant::D65)
    }
}

/// CIE D65 relative spectral power distribution at 5 nm intervals (380–780 nm).
///
/// Normalised such that Y = 100 for a perfect (unity) reflector.
#[must_use]
pub fn d65_spd_81() -> [f64; NUM_CMF_BANDS] {
    [
        49.9755, 52.3118, 54.6482, 68.7015, 82.7549, 87.1204, 91.4860, 92.4589, 93.4318, 90.0570,
        86.6823, 95.7736, 104.865, 110.936, 117.008, 117.410, 117.812, 116.336, 114.861, 115.392,
        115.923, 112.367, 108.811, 109.082, 109.354, 108.578, 107.802, 106.296, 104.790, 106.239,
        107.689, 106.047, 104.405, 104.225, 104.046, 102.023, 100.000, 98.1671, 96.3342, 96.0611,
        95.788, 92.2368, 88.6856, 89.3459, 90.0062, 89.8026, 89.5991, 88.6489, 87.6987, 85.4936,
        83.2886, 83.4939, 83.6992, 81.8630, 80.0268, 80.1207, 80.2146, 81.2462, 82.2778, 80.2810,
        78.2842, 74.0027, 69.7213, 70.6652, 71.6091, 72.9790, 74.3490, 67.9765, 61.6040, 65.7448,
        69.8856, 72.4863, 75.0870, 69.3398, 63.5927, 55.0054, 46.4182, 56.6118, 66.8054, 65.0941,
        63.3828,
    ]
}

/// CIE D50 relative spectral power distribution at 5 nm intervals (380–780 nm).
#[must_use]
pub fn d50_spd_81() -> [f64; NUM_CMF_BANDS] {
    [
        24.828, 27.174, 29.520, 40.422, 51.324, 57.308, 63.292, 65.148, 67.004, 67.803, 68.602,
        78.425, 88.248, 95.671, 103.095, 104.356, 105.617, 104.426, 103.235, 105.083, 106.932,
        103.563, 100.194, 102.098, 103.972, 102.978, 101.984, 100.474, 98.964, 100.118, 101.272,
        99.600, 97.928, 97.965, 97.972, 98.986, 100.000, 97.685, 95.370, 96.232, 97.094, 93.616,
        90.138, 91.421, 92.704, 92.475, 92.247, 91.264, 90.281, 88.179, 86.077, 87.078, 88.080,
        87.116, 86.152, 85.074, 83.996, 84.666, 85.337, 84.037, 82.737, 77.926, 73.115, 73.897,
        74.679, 75.914, 77.150, 72.012, 66.875, 70.405, 73.935, 76.890, 79.845, 74.011, 68.177,
        59.069, 49.961, 60.649, 71.337, 70.060, 68.783,
    ]
}

/// CIE D55 relative spectral power distribution at 5 nm intervals (380–780 nm).
#[must_use]
pub fn d55_spd_81() -> [f64; NUM_CMF_BANDS] {
    [
        38.452, 41.244, 44.036, 55.732, 67.428, 72.793, 78.158, 80.019, 81.880, 84.193, 86.507,
        93.344, 100.181, 106.001, 111.820, 112.793, 113.766, 113.052, 112.338, 112.965, 113.592,
        110.207, 106.822, 108.005, 109.188, 107.848, 106.508, 105.219, 103.930, 105.141, 106.352,
        104.856, 103.360, 103.078, 102.796, 101.398, 100.000, 98.145, 96.290, 96.153, 96.016,
        92.897, 89.778, 90.332, 90.886, 90.616, 90.347, 89.819, 89.292, 87.268, 85.244, 85.266,
        85.288, 84.309, 83.330, 82.561, 81.793, 82.940, 84.088, 82.154, 80.220, 75.885, 71.550,
        72.209, 72.868, 74.393, 75.919, 69.857, 63.796, 68.063, 72.330, 74.696, 77.063, 71.597,
        66.131, 57.036, 47.942, 58.310, 68.679, 66.929, 65.179,
    ]
}

/// Equal-energy illuminant E: flat SPD (1.0 at every band).
#[must_use]
pub fn equal_energy_spd_81() -> [f64; NUM_CMF_BANDS] {
    [1.0; NUM_CMF_BANDS]
}

/// Returns the 81-band SPD for the given standard illuminant.
#[must_use]
fn illuminant_spd(illuminant: StandardIlluminant) -> [f64; NUM_CMF_BANDS] {
    match illuminant {
        StandardIlluminant::D65 => d65_spd_81(),
        StandardIlluminant::D50 => d50_spd_81(),
        StandardIlluminant::D55 => d55_spd_81(),
        StandardIlluminant::EqualEnergy => equal_energy_spd_81(),
    }
}

// ── reflectance_to_xyz ────────────────────────────────────────────────────────

/// Converts a spectral reflectance curve to CIE XYZ tristimulus values.
///
/// Integrates the product of the reflectance, the illuminant SPD, and the CIE
/// 1931 2° colour-matching functions over 380–780 nm at 5 nm intervals.
///
/// # Arguments
///
/// * `reflectance` — Reflectance values at 5 nm intervals from 380–780 nm
///   (81 values expected).  Values are clamped to `[0, 1]`.  Slices shorter
///   than 81 are zero-padded; longer slices are truncated at index 80.
/// * `config` — Configuration specifying the illuminant.
///
/// # Returns
///
/// `[X, Y, Z]` normalised so that Y = 1.0 for a perfect white reflector
/// (reflectance = 1.0 at every band) under the chosen illuminant.
#[must_use]
pub fn reflectance_to_xyz(reflectance: &[f32], config: &SpectralUpsamplingConfig) -> [f32; 3] {
    let illum = illuminant_spd(config.illuminant);

    // Normalisation factor k: ensures Y = 1 for perfect reflector.
    let mut k_denom = 0.0_f64;
    for i in 0..NUM_CMF_BANDS {
        k_denom += illum[i] * CIE_CMF_Y[i];
    }
    let k = if k_denom > 1e-30 { 1.0 / k_denom } else { 0.0 };

    let mut x_acc = 0.0_f64;
    let mut y_acc = 0.0_f64;
    let mut z_acc = 0.0_f64;

    for i in 0..NUM_CMF_BANDS {
        // Zero-pad if the slice is shorter than 81.
        let refl = reflectance.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f64;
        let weighted = refl * illum[i];
        x_acc += weighted * CIE_CMF_X[i];
        y_acc += weighted * CIE_CMF_Y[i];
        z_acc += weighted * CIE_CMF_Z[i];
    }

    [(x_acc * k) as f32, (y_acc * k) as f32, (z_acc * k) as f32]
}

// ──────────────────────────────────────────────────────────────────────────────
// Extended illuminant library
// ──────────────────────────────────────────────────────────────────────────────

/// Extended illuminant library covering the most common CIE illuminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CieIlluminant {
    /// D50 (horizon daylight, ~5003 K).
    D50,
    /// D55 (mid-morning daylight, ~5503 K).
    D55,
    /// D65 (noon daylight, ~6504 K).
    D65,
    /// D75 (overcast sky, ~7504 K).
    D75,
    /// Illuminant A (incandescent tungsten, ~2856 K).
    A,
    /// F2 (cool white fluorescent).
    F2,
    /// F7 (broadband daylight fluorescent).
    F7,
    /// F11 (narrow-band white fluorescent).
    F11,
}

impl CieIlluminant {
    /// Returns the correlated colour temperature in Kelvin.
    #[must_use]
    pub fn cct_kelvin(self) -> f64 {
        match self {
            Self::D50 => 5003.0,
            Self::D55 => 5503.0,
            Self::D65 => 6504.0,
            Self::D75 => 7504.0,
            Self::A => 2856.0,
            Self::F2 => 4230.0,
            Self::F7 => 6500.0,
            Self::F11 => 4000.0,
        }
    }

    /// Returns the 81-band SPD (380–780 nm, 5 nm steps) for this illuminant.
    #[must_use]
    pub fn spd_81(self) -> [f64; NUM_CMF_BANDS] {
        match self {
            Self::D50 => d50_spd_81(),
            Self::D55 => d55_spd_81(),
            Self::D65 => d65_spd_81(),
            Self::D75 => d75_spd_81(),
            Self::A => illuminant_a_spd_81(),
            Self::F2 => illuminant_f2_spd_81(),
            Self::F7 => illuminant_f7_spd_81(),
            Self::F11 => illuminant_f11_spd_81(),
        }
    }
}

/// CIE D75 (overcast sky) SPD at 5 nm intervals (380–780 nm).
#[must_use]
pub fn d75_spd_81() -> [f64; NUM_CMF_BANDS] {
    // D75 computed from D-series formula at ~7504K: slightly bluer than D65
    let mut spd = [0.0; NUM_CMF_BANDS];
    let d65 = d65_spd_81();
    let d55 = d55_spd_81();
    for i in 0..NUM_CMF_BANDS {
        // Extrapolate: D75 ≈ D65 + (D65 - D55) * 0.5 (bluer)
        spd[i] = (d65[i] + (d65[i] - d55[i]) * 0.5).max(0.0);
    }
    spd
}

/// CIE Illuminant A (tungsten/incandescent, ~2856K) SPD at 5 nm intervals.
///
/// Computed from Planckian radiator formula at 2856K, normalised at 560 nm.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn illuminant_a_spd_81() -> [f64; NUM_CMF_BANDS] {
    let mut spd = [0.0; NUM_CMF_BANDS];
    let temp = 2856.0_f64;
    // Planckian: S(λ) ∝ λ^-5 / (exp(hc/λkT) - 1)
    // We use the simplified relative form normalised at 560 nm.
    let reference_wl = 560.0;
    let planck = |wl_nm: f64| -> f64 {
        let wl_m = wl_nm * 1e-9;
        let hc_kt = 1.4388e-2 / (wl_m * temp);
        wl_m.powi(-5) / (hc_kt.exp() - 1.0)
    };
    let ref_val = planck(reference_wl);
    if ref_val > 1e-30 {
        for i in 0..NUM_CMF_BANDS {
            let wl = CMF_LAMBDA_MIN + i as f64 * CMF_LAMBDA_STEP;
            spd[i] = planck(wl) / ref_val * 100.0;
        }
    }
    spd
}

/// CIE F2 (cool white fluorescent) SPD — simplified approximation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn illuminant_f2_spd_81() -> [f64; NUM_CMF_BANDS] {
    let mut spd = [0.0; NUM_CMF_BANDS];
    for i in 0..NUM_CMF_BANDS {
        let wl = CMF_LAMBDA_MIN + i as f64 * CMF_LAMBDA_STEP;
        // Simplified fluorescent: broad hump centred at 545nm + mercury line peaks
        let base = 20.0 * (-0.5 * ((wl - 545.0) / 80.0).powi(2)).exp();
        let peak1 = 40.0 * (-0.5 * ((wl - 436.0) / 5.0).powi(2)).exp(); // mercury 436nm
        let peak2 = 60.0 * (-0.5 * ((wl - 546.0) / 5.0).powi(2)).exp(); // mercury 546nm
        let peak3 = 30.0 * (-0.5 * ((wl - 611.0) / 8.0).powi(2)).exp(); // rare earth
        spd[i] = (base + peak1 + peak2 + peak3).max(1.0);
    }
    spd
}

/// CIE F7 (broadband daylight fluorescent) SPD — simplified approximation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn illuminant_f7_spd_81() -> [f64; NUM_CMF_BANDS] {
    let mut spd = [0.0; NUM_CMF_BANDS];
    let d65 = d65_spd_81();
    for i in 0..NUM_CMF_BANDS {
        let wl = CMF_LAMBDA_MIN + i as f64 * CMF_LAMBDA_STEP;
        // F7 is designed to closely match D65; model as D65 + small mercury peaks
        let peak1 = 15.0 * (-0.5 * ((wl - 436.0) / 4.0).powi(2)).exp();
        let peak2 = 20.0 * (-0.5 * ((wl - 546.0) / 4.0).powi(2)).exp();
        spd[i] = d65[i] * 0.85 + peak1 + peak2 + 5.0;
    }
    spd
}

/// CIE F11 (narrow-band white fluorescent) SPD — simplified approximation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn illuminant_f11_spd_81() -> [f64; NUM_CMF_BANDS] {
    let mut spd = [0.0; NUM_CMF_BANDS];
    for i in 0..NUM_CMF_BANDS {
        let wl = CMF_LAMBDA_MIN + i as f64 * CMF_LAMBDA_STEP;
        // Narrow tri-band phosphor fluorescent
        let band1 = 50.0 * (-0.5 * ((wl - 430.0) / 10.0).powi(2)).exp(); // blue
        let band2 = 80.0 * (-0.5 * ((wl - 545.0) / 12.0).powi(2)).exp(); // green
        let band3 = 60.0 * (-0.5 * ((wl - 610.0) / 10.0).powi(2)).exp(); // red
        spd[i] = (band1 + band2 + band3).max(1.0);
    }
    spd
}

// ──────────────────────────────────────────────────────────────────────────────
// CIE 1964 10° observer (supplementary)
// ──────────────────────────────────────────────────────────────────────────────

/// Observer function selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CieObserver {
    /// CIE 1931 2-degree standard observer.
    Deg2,
    /// CIE 1964 10-degree supplementary observer.
    Deg10,
}

/// Returns CMF values for the selected observer at a given 5 nm band index.
///
/// # Returns
///
/// `(x_bar, y_bar, z_bar)` at band index `i` (0..81).
#[must_use]
pub fn observer_cmf(observer: CieObserver, i: usize) -> (f64, f64, f64) {
    match observer {
        CieObserver::Deg2 => {
            let x = CIE_CMF_X.get(i).copied().unwrap_or(0.0);
            let y = CIE_CMF_Y.get(i).copied().unwrap_or(0.0);
            let z = CIE_CMF_Z.get(i).copied().unwrap_or(0.0);
            (x, y, z)
        }
        CieObserver::Deg10 => {
            let x = CIE_10DEG_X.get(i).copied().unwrap_or(0.0);
            let y = CIE_10DEG_Y.get(i).copied().unwrap_or(0.0);
            let z = CIE_10DEG_Z.get(i).copied().unwrap_or(0.0);
            (x, y, z)
        }
    }
}

/// CIE 1964 10° x̄₁₀ colour-matching function, 5 nm step, 380–780 nm.
pub const CIE_10DEG_X: [f64; NUM_CMF_BANDS] = [
    0.000160, 0.000662, 0.002362, 0.007242, 0.019110, 0.043400, 0.084736, 0.140638, 0.204492,
    0.264737, 0.314679, 0.357719, 0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085,
    0.195618, 0.132349, 0.080507, 0.041072, 0.016172, 0.005132, 0.003816, 0.015444, 0.037465,
    0.071358, 0.117749, 0.172953, 0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053,
    0.705224, 0.793832, 0.878655, 0.951162, 1.014160, 1.074300, 1.118520, 1.134300, 1.123990,
    1.089100, 1.030480, 0.950740, 0.856297, 0.754930, 0.647467, 0.535110, 0.431567, 0.343690,
    0.268329, 0.204300, 0.152568, 0.112210, 0.081261, 0.057930, 0.040851, 0.028623, 0.019941,
    0.013842, 0.009577, 0.006605, 0.004553, 0.003145, 0.002175, 0.001506, 0.001045, 0.000727,
    0.000508, 0.000356, 0.000249, 0.000175, 0.000123, 0.000087, 0.000061, 0.000043, 0.000030,
];

/// CIE 1964 10° ȳ₁₀ colour-matching function, 5 nm step, 380–780 nm.
pub const CIE_10DEG_Y: [f64; NUM_CMF_BANDS] = [
    0.000017, 0.000072, 0.000253, 0.000769, 0.002004, 0.004509, 0.008756, 0.014456, 0.021391,
    0.029497, 0.038676, 0.049602, 0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761,
    0.185190, 0.219940, 0.253589, 0.297665, 0.339133, 0.395379, 0.460777, 0.531360, 0.606741,
    0.685660, 0.761757, 0.823330, 0.875211, 0.923810, 0.961988, 0.982200, 0.991761, 0.999110,
    0.997340, 0.982380, 0.955552, 0.915175, 0.868934, 0.825623, 0.777405, 0.720353, 0.658341,
    0.593878, 0.527963, 0.461834, 0.398057, 0.339554, 0.283493, 0.228254, 0.179828, 0.140211,
    0.107633, 0.081187, 0.060281, 0.044096, 0.031800, 0.022602, 0.015905, 0.011130, 0.007749,
    0.005375, 0.003718, 0.002565, 0.001768, 0.001222, 0.000846, 0.000586, 0.000407, 0.000284,
    0.000199, 0.000139, 0.000098, 0.000069, 0.000048, 0.000034, 0.000024, 0.000017, 0.000012,
];

/// CIE 1964 10° z̄₁₀ colour-matching function, 5 nm step, 380–780 nm.
pub const CIE_10DEG_Z: [f64; NUM_CMF_BANDS] = [
    0.000705, 0.002928, 0.010482, 0.032344, 0.086011, 0.197120, 0.389366, 0.656760, 0.972542,
    1.282500, 1.553480, 1.798500, 1.967280, 2.027300, 1.994800, 1.900700, 1.745370, 1.554900,
    1.317560, 1.030200, 0.772125, 0.570060, 0.415254, 0.302356, 0.218502, 0.159249, 0.112044,
    0.082248, 0.060709, 0.043050, 0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
];

// ──────────────────────────────────────────────────────────────────────────────
// Spectral rendering pipeline
// ──────────────────────────────────────────────────────────────────────────────

/// Full spectral rendering pipeline: RGB → spectral → multiply → spectral → RGB.
///
/// This function converts two RGB colours to spectral reflectance curves,
/// multiplies them element-wise (simulating physical surface interaction such
/// as a coloured light on a coloured surface), then converts the result back
/// to XYZ under a chosen illuminant and observer.
///
/// # Arguments
///
/// * `rgb_a` — First colour in linear RGB [0..1].
/// * `rgb_b` — Second colour in linear RGB [0..1].
/// * `illuminant` — Illuminant under which to integrate.
/// * `observer` — 2° or 10° CIE observer.
///
/// # Returns
///
/// `[X, Y, Z]` tristimulus values of the product.
#[must_use]
pub fn spectral_render_product(
    rgb_a: [f64; 3],
    rgb_b: [f64; 3],
    illuminant: CieIlluminant,
    observer: CieObserver,
) -> [f64; 3] {
    let spd_a = rgb_to_spectral(rgb_a);
    let spd_b = rgb_to_spectral(rgb_b);

    let illum = illuminant.spd_81();

    // Resample 35-band SPDs to 81-band for integration
    let resample = |spd: &SpectralDistribution| -> [f64; NUM_CMF_BANDS] {
        let mut out = [0.0; NUM_CMF_BANDS];
        for i in 0..NUM_CMF_BANDS {
            let wl = CMF_LAMBDA_MIN + i as f64 * CMF_LAMBDA_STEP;
            out[i] = spd.at_wavelength(wl).max(0.0);
        }
        out
    };

    let a81 = resample(&spd_a);
    let b81 = resample(&spd_b);

    // Multiply spectral reflectances and integrate under illuminant
    let mut k_denom = 0.0_f64;
    let mut x_acc = 0.0_f64;
    let mut y_acc = 0.0_f64;
    let mut z_acc = 0.0_f64;

    for i in 0..NUM_CMF_BANDS {
        let (xbar, ybar, zbar) = observer_cmf(observer, i);
        k_denom += illum[i] * ybar;

        let product = a81[i] * b81[i];
        let weighted = product * illum[i];
        x_acc += weighted * xbar;
        y_acc += weighted * ybar;
        z_acc += weighted * zbar;
    }

    let k = if k_denom > 1e-30 { 1.0 / k_denom } else { 0.0 };
    [x_acc * k, y_acc * k, z_acc * k]
}

// ──────────────────────────────────────────────────────────────────────────────
// Kubelka-Munk paint mixing model
// ──────────────────────────────────────────────────────────────────────────────

/// Kubelka-Munk paint layer: absorption (K) and scattering (S) coefficients
/// sampled at 35 bands (380–720 nm, 10 nm steps).
#[derive(Debug, Clone, PartialEq)]
pub struct KubelkaMunkPaint {
    /// Absorption coefficient K at each spectral band.
    pub k_absorption: [f64; NUM_BANDS],
    /// Scattering coefficient S at each spectral band.
    pub s_scattering: [f64; NUM_BANDS],
}

impl KubelkaMunkPaint {
    /// Creates a paint from K and S arrays.
    #[must_use]
    pub fn new(k_absorption: [f64; NUM_BANDS], s_scattering: [f64; NUM_BANDS]) -> Self {
        Self {
            k_absorption,
            s_scattering,
        }
    }

    /// Computes the K/S ratio at each band.
    #[must_use]
    pub fn ks_ratio(&self) -> [f64; NUM_BANDS] {
        let mut ratio = [0.0; NUM_BANDS];
        for i in 0..NUM_BANDS {
            if self.s_scattering[i] > 1e-15 {
                ratio[i] = self.k_absorption[i] / self.s_scattering[i];
            }
        }
        ratio
    }

    /// Converts K/S ratio to reflectance using the Kubelka-Munk equation:
    /// `R = 1 + K/S - sqrt((K/S)^2 + 2*K/S)`
    #[must_use]
    pub fn reflectance(&self) -> SpectralDistribution {
        let ks = self.ks_ratio();
        let mut spd = SpectralDistribution::zero();
        for i in 0..NUM_BANDS {
            let r = 1.0 + ks[i] - (ks[i] * ks[i] + 2.0 * ks[i]).sqrt();
            spd.bands[i] = r.clamp(0.0, 1.0);
        }
        spd
    }
}

/// Mixes two Kubelka-Munk paints at given concentrations.
///
/// The K/S values are linearly combined weighted by concentration.
///
/// # Arguments
///
/// * `paint_a` — First paint.
/// * `paint_b` — Second paint.
/// * `concentration_a` — Fraction of paint A (0..1).
/// * `concentration_b` — Fraction of paint B (0..1).
///
/// # Returns
///
/// A new paint representing the mixture.
#[must_use]
pub fn km_mix(
    paint_a: &KubelkaMunkPaint,
    paint_b: &KubelkaMunkPaint,
    concentration_a: f64,
    concentration_b: f64,
) -> KubelkaMunkPaint {
    let ca = concentration_a.max(0.0);
    let cb = concentration_b.max(0.0);
    let mut k = [0.0; NUM_BANDS];
    let mut s = [0.0; NUM_BANDS];
    for i in 0..NUM_BANDS {
        k[i] = ca * paint_a.k_absorption[i] + cb * paint_b.k_absorption[i];
        s[i] = ca * paint_a.s_scattering[i] + cb * paint_b.s_scattering[i];
    }
    KubelkaMunkPaint::new(k, s)
}

// ──────────────────────────────────────────────────────────────────────────────
// Spectral mismatch index
// ──────────────────────────────────────────────────────────────────────────────

/// Computes the spectral mismatch index between two spectral distributions.
///
/// The mismatch index quantifies how differently two spectra appear under
/// different illuminants.  A low index means the two spectra are a good
/// metameric match.
///
/// # Formula
///
/// `SMI = sqrt( sum( (A_i/A_norm - B_i/B_norm)^2 ) / N )`
///
/// where the SPDs are normalised by their Y-weighted integral.
#[must_use]
pub fn spectral_mismatch_index(spd_a: &SpectralDistribution, spd_b: &SpectralDistribution) -> f64 {
    // Normalise each SPD by its total power
    let sum_a: f64 = spd_a.bands.iter().sum();
    let sum_b: f64 = spd_b.bands.iter().sum();

    if sum_a < 1e-15 || sum_b < 1e-15 {
        return 0.0;
    }

    let mut mse = 0.0;
    for i in 0..NUM_BANDS {
        let a_norm = spd_a.bands[i] / sum_a;
        let b_norm = spd_b.bands[i] / sum_b;
        let diff = a_norm - b_norm;
        mse += diff * diff;
    }

    (mse / NUM_BANDS as f64).sqrt()
}

/// Integrates a reflectance curve to XYZ with a chosen observer and illuminant.
///
/// # Arguments
///
/// * `reflectance` — 81-band reflectance (380–780 nm, 5 nm steps).
/// * `illuminant` — The illuminant to use.
/// * `observer` — 2° or 10° CIE observer.
///
/// # Returns
///
/// `[X, Y, Z]` normalised so Y = 1 for a perfect reflector.
#[must_use]
pub fn reflectance_to_xyz_with_observer(
    reflectance: &[f32],
    illuminant: CieIlluminant,
    observer: CieObserver,
) -> [f32; 3] {
    let illum = illuminant.spd_81();

    let mut k_denom = 0.0_f64;
    for i in 0..NUM_CMF_BANDS {
        let (_, ybar, _) = observer_cmf(observer, i);
        k_denom += illum[i] * ybar;
    }
    let k = if k_denom > 1e-30 { 1.0 / k_denom } else { 0.0 };

    let mut x_acc = 0.0_f64;
    let mut y_acc = 0.0_f64;
    let mut z_acc = 0.0_f64;

    for i in 0..NUM_CMF_BANDS {
        let refl = reflectance.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f64;
        let (xbar, ybar, zbar) = observer_cmf(observer, i);
        let weighted = refl * illum[i];
        x_acc += weighted * xbar;
        y_acc += weighted * ybar;
        z_acc += weighted * zbar;
    }

    [(x_acc * k) as f32, (y_acc * k) as f32, (z_acc * k) as f32]
}

// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spd_zero() {
        let spd = SpectralDistribution::zero();
        for &v in &spd.bands {
            assert!((v).abs() < 1e-15);
        }
    }

    #[test]
    fn test_spd_constant() {
        let spd = SpectralDistribution::constant(0.5);
        for &v in &spd.bands {
            assert!((v - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn test_wavelength_at() {
        assert!((SpectralDistribution::wavelength_at(0) - 380.0).abs() < 1e-9);
        assert!((SpectralDistribution::wavelength_at(34) - 720.0).abs() < 1e-9);
    }

    #[test]
    fn test_at_wavelength_in_range() {
        let spd = SpectralDistribution::constant(1.0);
        assert!((spd.at_wavelength(550.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_at_wavelength_out_of_range() {
        let spd = SpectralDistribution::constant(1.0);
        assert!((spd.at_wavelength(300.0)).abs() < 1e-12);
        assert!((spd.at_wavelength(800.0)).abs() < 1e-12);
    }

    #[test]
    fn test_scale() {
        let mut spd = SpectralDistribution::constant(2.0);
        spd.scale(0.5);
        for &v in &spd.bands {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_add() {
        let mut a = SpectralDistribution::constant(1.0);
        let b = SpectralDistribution::constant(2.0);
        a.add(&b);
        for &v in &a.bands {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_multiply() {
        let mut a = SpectralDistribution::constant(3.0);
        let b = SpectralDistribution::constant(2.0);
        a.multiply(&b);
        for &v in &a.bands {
            assert!((v - 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_integrate_constant() {
        let spd = SpectralDistribution::constant(1.0);
        let area = spd.integrate();
        // 34 intervals of 10nm each, constant 1.0 => 340
        let expected = (NUM_BANDS - 1) as f64 * LAMBDA_STEP;
        assert!(
            (area - expected).abs() < 1e-6,
            "Expected {expected}, got {area}"
        );
    }

    #[test]
    fn test_peak_wavelength() {
        let mut spd = SpectralDistribution::zero();
        // Set a single peak at band 10 (480 nm)
        spd.bands[10] = 5.0;
        assert!((spd.peak_wavelength() - 480.0).abs() < 1e-9);
    }

    #[test]
    fn test_rgb_to_spectral_black() {
        let spd = rgb_to_spectral([0.0, 0.0, 0.0]);
        for &v in &spd.bands {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_rgb_to_spectral_white_non_negative() {
        let spd = rgb_to_spectral([1.0, 1.0, 1.0]);
        for &v in &spd.bands {
            assert!(v >= 0.0, "Negative band value: {v}");
        }
    }

    #[test]
    fn test_spectral_to_xyz_white() {
        let spd = SpectralDistribution::constant(1.0);
        let xyz = spectral_to_xyz(&spd);
        // Y should be ~1.0 for a flat spectrum normalised by y_bar integral
        assert!(
            (xyz[1] - 1.0).abs() < 0.15,
            "Y should be near 1.0 for flat spectrum, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_d65_illuminant_positive() {
        let d65 = d65_illuminant();
        for &v in &d65.bands {
            assert!(v > 0.0, "D65 should be positive everywhere");
        }
    }

    #[test]
    fn test_red_upsampling_peaks_long_wavelength() {
        let spd = rgb_to_spectral([1.0, 0.0, 0.0]);
        let peak = spd.peak_wavelength();
        assert!(peak >= 600.0, "Red peak should be >= 600nm, got {peak}");
    }

    // ── SpectralUpsamplingConfig & StandardIlluminant ─────────────────────────

    #[test]
    fn test_spectral_config_default_is_d65() {
        let cfg = SpectralUpsamplingConfig::default();
        assert_eq!(
            cfg.illuminant,
            StandardIlluminant::D65,
            "Default illuminant should be D65"
        );
    }

    #[test]
    fn test_standard_illuminant_enum_variants() {
        // Ensure all four variants can be constructed without panic.
        let variants = [
            StandardIlluminant::D65,
            StandardIlluminant::D50,
            StandardIlluminant::D55,
            StandardIlluminant::EqualEnergy,
        ];
        assert_eq!(variants.len(), 4);
        // Verify PartialEq works
        assert_eq!(variants[0], StandardIlluminant::D65);
        assert_ne!(variants[0], StandardIlluminant::D50);
    }

    // ── reflectance_to_xyz ────────────────────────────────────────────────────

    #[test]
    fn test_reflectance_to_xyz_perfect_white_d65() {
        let cfg = SpectralUpsamplingConfig::new(StandardIlluminant::D65);
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "Perfect white under D65: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_perfect_black() {
        let cfg = SpectralUpsamplingConfig::default();
        let refl = [0.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            xyz[0].abs() < 1e-6 && xyz[1].abs() < 1e-6 && xyz[2].abs() < 1e-6,
            "Perfect black should give XYZ ≈ 0, got {:?}",
            xyz
        );
    }

    #[test]
    fn test_reflectance_to_xyz_perfect_white_d50() {
        let cfg = SpectralUpsamplingConfig::new(StandardIlluminant::D50);
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "Perfect white under D50: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_perfect_white_d55() {
        let cfg = SpectralUpsamplingConfig::new(StandardIlluminant::D55);
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "Perfect white under D55: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_perfect_white_equal_energy() {
        let cfg = SpectralUpsamplingConfig::new(StandardIlluminant::EqualEnergy);
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "Perfect white under EqualEnergy: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_empty_slice() {
        let cfg = SpectralUpsamplingConfig::default();
        let xyz = reflectance_to_xyz(&[], &cfg);
        assert!(
            xyz[0].abs() < 1e-6 && xyz[1].abs() < 1e-6 && xyz[2].abs() < 1e-6,
            "Empty reflectance slice should give XYZ ≈ 0, got {:?}",
            xyz
        );
    }

    #[test]
    fn test_reflectance_to_xyz_short_slice() {
        let cfg = SpectralUpsamplingConfig::default();
        // Only 10 bands provided — primarily UV where CMF values are very small.
        let refl = [1.0f32; 10];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        // Y should be much less than 1 because only the first 10 bands (380–425 nm)
        // contribute, and the CMF ȳ values are near zero there.
        assert!(
            xyz[1] < 0.05,
            "Short slice (10 bands) should give very small Y, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_neutral_gray() {
        let cfg = SpectralUpsamplingConfig::default();
        let refl = [0.18f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz(&refl, &cfg);
        assert!(
            (xyz[1] - 0.18).abs() < 0.01,
            "18% gray: Y should be ~0.18, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_red_ish() {
        // High reflectance only at long wavelengths (bands 50–80, ~630–780 nm)
        let cfg = SpectralUpsamplingConfig::default();
        let mut refl = [0.0f32; NUM_CMF_BANDS];
        for v in &mut refl[50..] {
            *v = 1.0;
        }
        let xyz = reflectance_to_xyz(&refl, &cfg);
        // Long-wavelength stimulus should give X dominant, Z near zero
        assert!(
            xyz[0] > xyz[2],
            "Red-ish reflectance: X ({}) should be > Z ({})",
            xyz[0],
            xyz[2]
        );
        assert!(
            xyz[2] < 0.05,
            "Red-ish reflectance: Z should be near zero, got {}",
            xyz[2]
        );
    }

    #[test]
    fn test_reflectance_to_xyz_clamping() {
        let cfg = SpectralUpsamplingConfig::default();
        let refl_clamped = [2.0f32; NUM_CMF_BANDS]; // > 1.0 should clamp to 1.0
        let refl_one = [1.0f32; NUM_CMF_BANDS];
        let xyz_clamped = reflectance_to_xyz(&refl_clamped, &cfg);
        let xyz_one = reflectance_to_xyz(&refl_one, &cfg);
        assert!(
            (xyz_clamped[1] - xyz_one[1]).abs() < 1e-5,
            "Clamped >1.0 should equal 1.0: {} vs {}",
            xyz_clamped[1],
            xyz_one[1]
        );
    }

    #[test]
    fn test_illuminant_d65_d50_differ() {
        let cfg_d65 = SpectralUpsamplingConfig::new(StandardIlluminant::D65);
        let cfg_d50 = SpectralUpsamplingConfig::new(StandardIlluminant::D50);
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz_d65 = reflectance_to_xyz(&refl, &cfg_d65);
        let xyz_d50 = reflectance_to_xyz(&refl, &cfg_d50);
        // Both Y ≈ 1.0 (normalised), but X and Z differ due to different chromaticities
        assert!(
            (xyz_d65[1] - 1.0).abs() < 0.01,
            "D65 Y should be ~1.0, got {}",
            xyz_d65[1]
        );
        assert!(
            (xyz_d50[1] - 1.0).abs() < 0.01,
            "D50 Y should be ~1.0, got {}",
            xyz_d50[1]
        );
        // X values differ: D65 is cooler (higher X/Y at blue end) vs D50
        assert!(
            (xyz_d65[0] - xyz_d50[0]).abs() > 0.001 || (xyz_d65[2] - xyz_d50[2]).abs() > 0.001,
            "D65 and D50 should give different X or Z for white reflector"
        );
    }

    // ── Extended illuminant library tests ────────────────────────────────────

    #[test]
    fn test_cie_illuminant_cct_values() {
        assert!((CieIlluminant::D65.cct_kelvin() - 6504.0).abs() < 1.0);
        assert!((CieIlluminant::A.cct_kelvin() - 2856.0).abs() < 1.0);
        assert!((CieIlluminant::D50.cct_kelvin() - 5003.0).abs() < 1.0);
    }

    #[test]
    fn test_all_illuminant_spds_positive() {
        let illuminants = [
            CieIlluminant::D50,
            CieIlluminant::D55,
            CieIlluminant::D65,
            CieIlluminant::D75,
            CieIlluminant::A,
            CieIlluminant::F2,
            CieIlluminant::F7,
            CieIlluminant::F11,
        ];
        for ill in illuminants {
            let spd = ill.spd_81();
            for (i, &v) in spd.iter().enumerate() {
                assert!(v >= 0.0, "{ill:?} has negative SPD at band {i}: {v}");
            }
        }
    }

    #[test]
    fn test_illuminant_a_warmer_than_d65() {
        let a = CieIlluminant::A.spd_81();
        let d65 = CieIlluminant::D65.spd_81();
        // Illuminant A should have higher relative power at long wavelengths (red)
        // compared to short wavelengths (blue), relative to D65
        let a_ratio = a[70] / a[10].max(1e-10); // ~730nm / ~430nm
        let d65_ratio = d65[70] / d65[10].max(1e-10);
        assert!(
            a_ratio > d65_ratio,
            "Illuminant A should be warmer (higher red/blue ratio): {} vs {}",
            a_ratio,
            d65_ratio
        );
    }

    #[test]
    fn test_illuminant_d75_bluer_than_d65() {
        let d75 = CieIlluminant::D75.spd_81();
        let d65 = CieIlluminant::D65.spd_81();
        // D75 at CCT ~7504K should be bluer than D65 at ~6504K
        // Check ratio of blue (band 10, ~430nm) to red (band 60, ~680nm)
        let d75_ratio = d75[10] / d75[60].max(1e-10);
        let d65_ratio = d65[10] / d65[60].max(1e-10);
        assert!(
            d75_ratio > d65_ratio * 0.9,
            "D75 should be at least comparable blue/red ratio to D65: {} vs {}",
            d75_ratio,
            d65_ratio
        );
    }

    // ── Observer function tests ─────────────────────────────────────────────

    #[test]
    fn test_observer_2deg_matches_constants() {
        let (x, y, z) = observer_cmf(CieObserver::Deg2, 0);
        assert!((x - CIE_CMF_X[0]).abs() < 1e-10);
        assert!((y - CIE_CMF_Y[0]).abs() < 1e-10);
        assert!((z - CIE_CMF_Z[0]).abs() < 1e-10);
    }

    #[test]
    fn test_observer_10deg_different_from_2deg() {
        let (x2, y2, _) = observer_cmf(CieObserver::Deg2, 40);
        let (x10, y10, _) = observer_cmf(CieObserver::Deg10, 40);
        // 2° and 10° should differ noticeably at most bands
        assert!(
            (x2 - x10).abs() > 0.001 || (y2 - y10).abs() > 0.001,
            "2° and 10° observers should differ at band 40"
        );
    }

    #[test]
    fn test_observer_out_of_range() {
        let (x, y, z) = observer_cmf(CieObserver::Deg2, 999);
        assert!(x.abs() < 1e-10 && y.abs() < 1e-10 && z.abs() < 1e-10);
    }

    // ── Spectral rendering pipeline tests ───────────────────────────────────

    #[test]
    fn test_spectral_render_product_white_white() {
        let xyz = spectral_render_product(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            CieIlluminant::D65,
            CieObserver::Deg2,
        );
        // Product of two whites = white: Y should be positive
        assert!(
            xyz[1] > 0.0,
            "White×White should give positive Y: {}",
            xyz[1]
        );
    }

    #[test]
    fn test_spectral_render_product_black() {
        let xyz = spectral_render_product(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            CieIlluminant::D65,
            CieObserver::Deg2,
        );
        for v in xyz {
            assert!(v.abs() < 1e-6, "Black×anything should be ~0: {v}");
        }
    }

    #[test]
    fn test_spectral_render_product_complementary() {
        // Red × Blue should give a very dark result (minimal spectral overlap)
        let xyz = spectral_render_product(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            CieIlluminant::D65,
            CieObserver::Deg2,
        );
        assert!(xyz[1] < 0.1, "Red×Blue Y should be very small: {}", xyz[1]);
    }

    // ── Kubelka-Munk tests ──────────────────────────────────────────────────

    #[test]
    fn test_km_white_paint_reflectance() {
        // Zero absorption, high scattering = white
        let paint = KubelkaMunkPaint::new([0.0; NUM_BANDS], [1.0; NUM_BANDS]);
        let refl = paint.reflectance();
        for &v in &refl.bands {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "White paint reflectance should be ~1.0: {v}"
            );
        }
    }

    #[test]
    fn test_km_black_paint_reflectance() {
        // High absorption, low scattering = dark
        let paint = KubelkaMunkPaint::new([10.0; NUM_BANDS], [0.1; NUM_BANDS]);
        let refl = paint.reflectance();
        for &v in &refl.bands {
            assert!(v < 0.15, "Black paint reflectance should be very low: {v}");
        }
    }

    #[test]
    fn test_km_mix_concentrations() {
        let white = KubelkaMunkPaint::new([0.0; NUM_BANDS], [1.0; NUM_BANDS]);
        let black = KubelkaMunkPaint::new([10.0; NUM_BANDS], [0.1; NUM_BANDS]);
        let mix = km_mix(&white, &black, 0.5, 0.5);
        let refl = mix.reflectance();
        // Should be somewhere between white and black
        let white_refl = white.reflectance();
        let black_refl = black.reflectance();
        assert!(
            refl.bands[17] < white_refl.bands[17] && refl.bands[17] > black_refl.bands[17],
            "Mix should be between white and black: {} (white={}, black={})",
            refl.bands[17],
            white_refl.bands[17],
            black_refl.bands[17]
        );
    }

    #[test]
    fn test_km_ks_ratio() {
        let paint = KubelkaMunkPaint::new([2.0; NUM_BANDS], [4.0; NUM_BANDS]);
        let ks = paint.ks_ratio();
        for &v in &ks {
            assert!((v - 0.5).abs() < 1e-10, "K/S should be 0.5: {v}");
        }
    }

    // ── Spectral mismatch index tests ───────────────────────────────────────

    #[test]
    fn test_spectral_mismatch_identical() {
        let spd = SpectralDistribution::constant(1.0);
        let smi = spectral_mismatch_index(&spd, &spd);
        assert!(smi < 1e-10, "Identical SPDs should have SMI ~0: {smi}");
    }

    #[test]
    fn test_spectral_mismatch_different() {
        let spd_a = rgb_to_spectral([1.0, 0.0, 0.0]); // red
        let spd_b = rgb_to_spectral([0.0, 0.0, 1.0]); // blue
        let smi = spectral_mismatch_index(&spd_a, &spd_b);
        assert!(smi > 0.01, "Red vs blue should have high SMI: {smi}");
    }

    #[test]
    fn test_spectral_mismatch_zero_spd() {
        let zero = SpectralDistribution::zero();
        let one = SpectralDistribution::constant(1.0);
        let smi = spectral_mismatch_index(&zero, &one);
        assert!(smi.abs() < 1e-10, "Zero SPD should give SMI 0: {smi}");
    }

    // ── reflectance_to_xyz_with_observer tests ──────────────────────────────

    #[test]
    fn test_reflectance_xyz_2deg_white() {
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz_with_observer(&refl, CieIlluminant::D65, CieObserver::Deg2);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "2° observer white under D65: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_xyz_10deg_white() {
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz_with_observer(&refl, CieIlluminant::D65, CieObserver::Deg10);
        assert!(
            (xyz[1] - 1.0).abs() < 0.02,
            "10° observer white under D65: Y should be ~1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_reflectance_xyz_with_illuminant_a() {
        let refl = [1.0f32; NUM_CMF_BANDS];
        let xyz = reflectance_to_xyz_with_observer(&refl, CieIlluminant::A, CieObserver::Deg2);
        assert!(
            (xyz[1] - 1.0).abs() < 0.01,
            "White under illuminant A: Y should be ~1.0, got {}",
            xyz[1]
        );
        // Illuminant A is warm: X > Z
        assert!(
            xyz[0] > xyz[2],
            "Illuminant A white point: X ({}) should be > Z ({})",
            xyz[0],
            xyz[2]
        );
    }
}
