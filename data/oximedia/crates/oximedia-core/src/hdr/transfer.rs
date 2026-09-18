//! Transfer characteristics (EOTF/OETF) for HDR and SDR video.
//!
//! This module defines transfer functions that map between electrical signal
//! values and optical (linear) light values. The transfer characteristic
//! determines how the signal encodes light levels.
//!
//! # Transfer Functions
//!
//! - **EOTF (Electro-Optical Transfer Function)**: Converts signal to linear light
//! - **OETF (Opto-Electronic Transfer Function)**: Converts linear light to signal
#![allow(clippy::match_same_arms)]
//!
//! # Examples
//!
//! ```
//! use oximedia_core::hdr::TransferCharacteristic;
//!
//! let transfer = TransferCharacteristic::Pq;
//! let signal = 0.5;
//!
//! // Convert signal to linear light (EOTF)
//! let linear = transfer.eotf(signal);
//! assert!(linear >= 0.0 && linear <= 1.0);
//!
//! // Convert back to signal (OETF)
//! let roundtrip = transfer.oetf(linear);
//! assert!((roundtrip - signal).abs() < 0.001);
//! ```

/// Transfer characteristic (EOTF) for video signals.
///
/// The transfer characteristic determines how electrical signal values
/// map to optical (linear) light values. Different transfer functions
/// are optimized for different use cases.
///
/// # Variants
///
/// - **Pq**: ST.2084 Perceptual Quantizer (HDR10)
/// - **Hlg**: Hybrid Log-Gamma (broadcast HDR)
/// - **Bt709**: BT.709/sRGB gamma (SDR)
/// - **Bt2020**: BT.2020 (same as BT.709, just wider primaries)
/// - **Linear**: No transfer function (linear light)
/// - **Srgb**: sRGB transfer function (nearly identical to BT.709)
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::TransferCharacteristic;
///
/// let pq = TransferCharacteristic::Pq;
/// assert!(pq.is_hdr());
/// assert_eq!(pq.reference_peak_nits(), 10000.0);
///
/// let bt709 = TransferCharacteristic::Bt709;
/// assert!(!bt709.is_hdr());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TransferCharacteristic {
    /// ST.2084 (PQ) - Perceptual Quantizer.
    ///
    /// Used in HDR10 and HDR10+. Maps to 0-10000 nits.
    /// Optimized for perceptual uniformity across the entire dynamic range.
    Pq,

    /// HLG (Hybrid Log-Gamma) - ARIB STD-B67.
    ///
    /// Used in broadcast HDR. Backward compatible with SDR displays.
    /// Combines gamma curve (for SDR compatibility) with logarithmic curve (for HDR).
    Hlg,

    /// BT.709 - Rec.709 transfer function.
    ///
    /// Standard for HD television and sRGB-like content.
    /// Uses gamma ~2.4 with linear segment near black.
    #[default]
    Bt709,

    /// BT.2020 - Rec.2020 transfer function.
    ///
    /// Uses the same transfer function as BT.709, but with wider color primaries.
    /// Often paired with PQ or HLG for HDR content.
    Bt2020,

    /// Linear - No transfer function.
    ///
    /// Signal values are directly proportional to light intensity.
    /// Used in rendering and intermediate processing.
    Linear,

    /// sRGB transfer function.
    ///
    /// Nearly identical to BT.709, with slight differences in the linear segment.
    /// Standard for computer graphics and web content.
    Srgb,
}

impl TransferCharacteristic {
    /// Applies the Electro-Optical Transfer Function (EOTF).
    ///
    /// Converts electrical signal value [0, 1] to linear light [0, 1].
    ///
    /// # Arguments
    ///
    /// * `signal` - Normalized signal value [0, 1]
    ///
    /// # Returns
    ///
    /// Linear light value [0, 1] where 1.0 represents the reference peak luminance.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::TransferCharacteristic;
    ///
    /// let pq = TransferCharacteristic::Pq;
    /// let linear = pq.eotf(0.5);
    /// assert!(linear >= 0.0 && linear <= 1.0);
    /// ```
    #[must_use]
    pub fn eotf(&self, signal: f64) -> f64 {
        match self {
            Self::Pq => pq_eotf(signal),
            Self::Hlg => hlg_eotf(signal),
            Self::Bt709 | Self::Bt2020 => bt709_eotf(signal),
            Self::Srgb => srgb_eotf(signal),
            Self::Linear => signal.clamp(0.0, 1.0),
        }
    }

    /// Applies the Opto-Electronic Transfer Function (OETF).
    ///
    /// Converts linear light value [0, 1] to electrical signal [0, 1].
    ///
    /// # Arguments
    ///
    /// * `linear` - Linear light value [0, 1]
    ///
    /// # Returns
    ///
    /// Normalized signal value [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::TransferCharacteristic;
    ///
    /// let pq = TransferCharacteristic::Pq;
    /// let signal = pq.oetf(0.5);
    /// assert!(signal >= 0.0 && signal <= 1.0);
    /// ```
    #[must_use]
    pub fn oetf(&self, linear: f64) -> f64 {
        match self {
            Self::Pq => pq_oetf(linear),
            Self::Hlg => hlg_oetf(linear),
            Self::Bt709 | Self::Bt2020 => bt709_oetf(linear),
            Self::Srgb => srgb_oetf(linear),
            Self::Linear => linear.clamp(0.0, 1.0),
        }
    }

    /// Returns true if this is an HDR transfer function.
    ///
    /// PQ and HLG are considered HDR transfer functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::TransferCharacteristic;
    ///
    /// assert!(TransferCharacteristic::Pq.is_hdr());
    /// assert!(TransferCharacteristic::Hlg.is_hdr());
    /// assert!(!TransferCharacteristic::Bt709.is_hdr());
    /// ```
    #[must_use]
    pub const fn is_hdr(&self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    /// Returns the reference peak luminance in nits.
    ///
    /// This is the luminance that corresponds to signal value 1.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::TransferCharacteristic;
    ///
    /// assert_eq!(TransferCharacteristic::Pq.reference_peak_nits(), 10000.0);
    /// assert_eq!(TransferCharacteristic::Bt709.reference_peak_nits(), 100.0);
    /// ```
    #[must_use]
    pub const fn reference_peak_nits(&self) -> f64 {
        match self {
            Self::Pq => 10000.0,   // ST.2084 reference
            Self::Hlg => 1000.0,   // HLG nominal peak
            Self::Bt709 => 100.0,  // SDR reference white
            Self::Bt2020 => 100.0, // Same as BT.709
            Self::Srgb => 80.0,    // sRGB reference (lower than BT.709)
            Self::Linear => 100.0, // Assume SDR range
        }
    }

    /// Returns a human-readable name for this transfer function.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::TransferCharacteristic;
    ///
    /// assert_eq!(TransferCharacteristic::Pq.name(), "ST.2084 (PQ)");
    /// assert_eq!(TransferCharacteristic::Hlg.name(), "HLG (Hybrid Log-Gamma)");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Pq => "ST.2084 (PQ)",
            Self::Hlg => "HLG (Hybrid Log-Gamma)",
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
            Self::Linear => "Linear",
            Self::Srgb => "sRGB",
        }
    }
}

/// ST.2084 (PQ) EOTF - converts PQ signal to linear light.
///
/// Returns linear light in range [0, 1] where 1.0 represents 10000 nits.
///
/// Reference: SMPTE ST 2084:2014
#[must_use]
fn pq_eotf(e: f64) -> f64 {
    // Constants from ST.2084
    const M1: f64 = 2610.0 / 16_384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let e = e.clamp(0.0, 1.0);
    if e == 0.0 {
        return 0.0;
    }

    let e_m2 = e.powf(1.0 / M2);
    let num = (e_m2 - C1).max(0.0);
    let den = C2 - C3 * e_m2;

    if den.abs() < 1e-10 {
        0.0
    } else {
        (num / den).powf(1.0 / M1)
    }
}

/// ST.2084 (PQ) inverse EOTF - converts linear light to PQ signal.
///
/// Input is linear light in range [0, 1] where 1.0 represents 10000 nits.
///
/// Reference: SMPTE ST 2084:2014
#[must_use]
fn pq_oetf(y: f64) -> f64 {
    const M1: f64 = 2610.0 / 16_384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    let y = y.clamp(0.0, 1.0);
    if y == 0.0 {
        return 0.0;
    }

    let y_m1 = y.powf(M1);
    let num = C1 + C2 * y_m1;
    let den = 1.0 + C3 * y_m1;

    (num / den).powf(M2)
}

/// HLG EOTF - converts HLG signal to linear light.
///
/// Returns linear light in range [0, 1].
///
/// Reference: ARIB STD-B67
#[must_use]
fn hlg_eotf(e: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;

    let e = e.clamp(0.0, 1.0);

    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - C) / A).exp() + B) / 12.0
    }
}

/// HLG inverse EOTF - converts linear light to HLG signal.
///
/// Input is linear light in range [0, 1].
///
/// Reference: ARIB STD-B67
#[must_use]
fn hlg_oetf(y: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;

    let y = y.clamp(0.0, 1.0);

    if y <= 1.0 / 12.0 {
        (3.0 * y).sqrt()
    } else {
        A * (12.0 * y - B).ln() + C
    }
}

/// BT.709 EOTF (gamma 2.4 with linear segment).
///
/// Reference: ITU-R BT.709
#[must_use]
fn bt709_eotf(e: f64) -> f64 {
    const BETA: f64 = 0.018_053_968_510_807;
    const ALPHA: f64 = 1.099_296_826_809_44;
    const GAMMA: f64 = 1.0 / 0.45;

    let e = e.clamp(0.0, 1.0);

    if e < BETA * 4.5 {
        e / 4.5
    } else {
        ((e + (ALPHA - 1.0)) / ALPHA).powf(GAMMA)
    }
}

/// BT.709 inverse EOTF.
///
/// Reference: ITU-R BT.709
#[must_use]
fn bt709_oetf(y: f64) -> f64 {
    const BETA: f64 = 0.018_053_968_510_807;
    const ALPHA: f64 = 1.099_296_826_809_44;
    const GAMMA: f64 = 0.45;

    let y = y.clamp(0.0, 1.0);

    if y < BETA {
        4.5 * y
    } else {
        ALPHA * y.powf(GAMMA) - (ALPHA - 1.0)
    }
}

/// sRGB EOTF (gamma 2.2 with linear segment).
///
/// Very similar to BT.709 but with slightly different parameters.
#[must_use]
fn srgb_eotf(e: f64) -> f64 {
    const THRESHOLD: f64 = 0.04045;

    let e = e.clamp(0.0, 1.0);

    if e <= THRESHOLD {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB inverse EOTF.
#[must_use]
fn srgb_oetf(y: f64) -> f64 {
    const THRESHOLD: f64 = 0.003_130_8;

    let y = y.clamp(0.0, 1.0);

    if y <= THRESHOLD {
        12.92 * y
    } else {
        1.055 * y.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_roundtrip() {
        let signal = 0.5;
        let linear = pq_eotf(signal);
        let roundtrip = pq_oetf(linear);
        assert!((roundtrip - signal).abs() < 0.001);
    }

    #[test]
    fn test_pq_black_and_white() {
        assert_eq!(pq_eotf(0.0), 0.0);
        let white = pq_eotf(1.0);
        assert!((white - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hlg_roundtrip() {
        let signal = 0.5;
        let linear = hlg_eotf(signal);
        let roundtrip = hlg_oetf(linear);
        assert!((roundtrip - signal).abs() < 0.001);
    }

    #[test]
    fn test_hlg_black_and_white() {
        assert_eq!(hlg_eotf(0.0), 0.0);
        let white = hlg_eotf(1.0);
        // HLG EOTF can return values slightly above 1.0 for signal value 1.0
        assert!(white > 0.0 && white < 1.1);
    }

    #[test]
    fn test_bt709_roundtrip() {
        let signal = 0.5;
        let linear = bt709_eotf(signal);
        let roundtrip = bt709_oetf(linear);
        assert!((roundtrip - signal).abs() < 0.001);
    }

    #[test]
    fn test_srgb_roundtrip() {
        let signal = 0.5;
        let linear = srgb_eotf(signal);
        let roundtrip = srgb_oetf(linear);
        assert!((roundtrip - signal).abs() < 0.001);
    }

    #[test]
    fn test_transfer_eotf() {
        let pq = TransferCharacteristic::Pq;
        let linear = pq.eotf(0.5);
        assert!(linear >= 0.0 && linear <= 1.0);

        let hlg = TransferCharacteristic::Hlg;
        let linear_hlg = hlg.eotf(0.5);
        assert!(linear_hlg >= 0.0 && linear_hlg <= 1.0);
    }

    #[test]
    fn test_transfer_oetf() {
        let pq = TransferCharacteristic::Pq;
        let signal = pq.oetf(0.5);
        assert!(signal >= 0.0 && signal <= 1.0);
    }

    #[test]
    fn test_is_hdr() {
        assert!(TransferCharacteristic::Pq.is_hdr());
        assert!(TransferCharacteristic::Hlg.is_hdr());
        assert!(!TransferCharacteristic::Bt709.is_hdr());
        assert!(!TransferCharacteristic::Srgb.is_hdr());
        assert!(!TransferCharacteristic::Linear.is_hdr());
    }

    #[test]
    fn test_reference_peak_nits() {
        assert_eq!(TransferCharacteristic::Pq.reference_peak_nits(), 10000.0);
        assert_eq!(TransferCharacteristic::Hlg.reference_peak_nits(), 1000.0);
        assert_eq!(TransferCharacteristic::Bt709.reference_peak_nits(), 100.0);
        assert_eq!(TransferCharacteristic::Srgb.reference_peak_nits(), 80.0);
    }

    #[test]
    fn test_transfer_names() {
        assert_eq!(TransferCharacteristic::Pq.name(), "ST.2084 (PQ)");
        assert_eq!(TransferCharacteristic::Hlg.name(), "HLG (Hybrid Log-Gamma)");
        assert_eq!(TransferCharacteristic::Bt709.name(), "BT.709");
    }

    #[test]
    fn test_linear_passthrough() {
        let linear = TransferCharacteristic::Linear;
        assert_eq!(linear.eotf(0.5), 0.5);
        assert_eq!(linear.oetf(0.5), 0.5);
    }

    #[test]
    fn test_clamping() {
        let pq = TransferCharacteristic::Pq;
        assert_eq!(pq.eotf(-0.1), pq.eotf(0.0));
        assert_eq!(pq.eotf(1.5), pq.eotf(1.0));
    }
}
