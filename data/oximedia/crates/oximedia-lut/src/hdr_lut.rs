#![allow(dead_code)]
//! HDR transfer-function LUT pair (forward and inverse) for OETF/EOTF workflows.

/// Supported HDR electro-optical transfer functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransfer {
    /// SMPTE ST 2084 Perceptual Quantizer — used by HDR10 and Dolby Vision.
    Pq,
    /// Hybrid Log-Gamma (BBC/NHK) — HDR broadcast standard.
    Hlg,
    /// Sony S-Log3 camera log encoding.
    Slog3,
    /// ARRI `LogC` (EI 800 reference).
    LogC,
}

impl HdrTransfer {
    /// Typical peak luminance of a display targeted by this transfer function, in nits.
    #[must_use]
    pub fn peak_luminance_nits(self) -> f32 {
        match self {
            Self::Pq => 10_000.0,
            Self::Hlg => 1_000.0,
            Self::Slog3 => 800.0,
            Self::LogC => 800.0,
        }
    }

    /// Returns `true` when the transfer function is scene-referred (camera log).
    #[must_use]
    pub fn is_scene_referred(self) -> bool {
        matches!(self, Self::Slog3 | Self::LogC)
    }

    /// Short label used in file names and metadata.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Pq => "PQ",
            Self::Hlg => "HLG",
            Self::Slog3 => "SLog3",
            Self::LogC => "LogC",
        }
    }

    /// Apply the OETF (scene-linear → encoded signal) to a normalised linear value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn encode(self, linear: f32) -> f32 {
        match self {
            Self::Pq => Self::pq_encode(linear),
            Self::Hlg => Self::hlg_encode(linear),
            Self::Slog3 => Self::slog3_encode(linear),
            Self::LogC => Self::logc_encode(linear),
        }
    }

    /// Apply the EOTF (encoded signal → scene-linear) — the inverse of `encode`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn decode(self, signal: f32) -> f32 {
        match self {
            Self::Pq => Self::pq_decode(signal),
            Self::Hlg => Self::hlg_decode(signal),
            Self::Slog3 => Self::slog3_decode(signal),
            Self::LogC => Self::logc_decode(signal),
        }
    }

    // ---- PQ (ST 2084) ----
    fn pq_encode(y: f32) -> f32 {
        const M1: f32 = 0.159_301_76;
        const M2: f32 = 78.843_75;
        const C1: f32 = 0.835_937_5;
        const C2: f32 = 18.851_563;
        const C3: f32 = 18.686_875;
        let y = y.max(0.0);
        let yp = y.powf(M1);
        ((C1 + C2 * yp) / (1.0 + C3 * yp)).powf(M2)
    }

    fn pq_decode(e: f32) -> f32 {
        const M1: f32 = 0.159_301_76;
        const M2: f32 = 78.843_75;
        const C1: f32 = 0.835_937_5;
        const C2: f32 = 18.851_563;
        const C3: f32 = 18.686_875;
        let e = e.clamp(0.0, 1.0);
        let ep = e.powf(1.0 / M2);
        let num = (ep - C1).max(0.0);
        let den = C2 - C3 * ep;
        (num / den.max(1e-8)).powf(1.0 / M1)
    }

    // ---- HLG ----
    fn hlg_encode(y: f32) -> f32 {
        const A: f32 = 0.178_832_77;
        const B: f32 = 0.284_668_92;
        const C: f32 = 0.559_910_7;
        if y < 0.0 {
            return 0.0;
        }
        if y <= 1.0 / 12.0 {
            (3.0 * y).sqrt()
        } else {
            A * (12.0 * y - B).ln() + C
        }
    }

    fn hlg_decode(e: f32) -> f32 {
        const A: f32 = 0.178_832_77;
        const B: f32 = 0.284_668_92;
        const C: f32 = 0.559_910_7;
        let e = e.clamp(0.0, 1.0);
        if e <= 0.5 {
            e * e / 3.0
        } else {
            (((e - C) / A).exp() + B) / 12.0
        }
    }

    // ---- S-Log3 ----
    fn slog3_encode(x: f32) -> f32 {
        let x = x.max(-0.014_f32);
        let ratio = (x + 0.01) / (0.18 + 0.01);
        (420.0 + ratio.max(1e-10).log10() * 261.5) / 1023.0
    }

    fn slog3_decode(e: f32) -> f32 {
        let y = e * 1023.0;
        if y >= 171.210_3 {
            10.0_f32.powf((y - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
        } else {
            (y - 95.0) / 1030.0
        }
    }

    // ---- LogC (simplified) ----
    fn logc_encode(x: f32) -> f32 {
        if x >= 0.010_591 {
            0.247_190 * (5.555_556 * x + 0.052_272).log10() + 0.385_537
        } else {
            5.367_655 * x + 0.092_809
        }
    }

    fn logc_decode(e: f32) -> f32 {
        if e >= 0.149_658 {
            (10.0_f32.powf((e - 0.385_537) / 0.247_190) - 0.052_272) / 5.555_556
        } else {
            (e - 0.092_809) / 5.367_655
        }
    }
}

/// A 1-D LUT that maps an encoded HDR signal to scene-linear (or vice-versa).
#[derive(Debug, Clone)]
pub struct HdrLut {
    /// Transfer function this LUT was built for.
    pub transfer: HdrTransfer,
    /// Whether this is the inverse (decode) direction.
    inverse: bool,
    /// LUT entries (normalised 0..=1 output values).
    entries: Vec<f32>,
}

impl HdrLut {
    /// Build a forward (OETF: linear → encoded) LUT with `size` entries.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn build_forward(transfer: HdrTransfer, size: usize) -> Self {
        let entries = (0..size)
            .map(|i| {
                let x = i as f32 / (size - 1).max(1) as f32;
                transfer.encode(x)
            })
            .collect();
        Self {
            transfer,
            inverse: false,
            entries,
        }
    }

    /// Build an inverse (EOTF: encoded → linear) LUT with `size` entries.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn build_inverse(transfer: HdrTransfer, size: usize) -> Self {
        let entries = (0..size)
            .map(|i| {
                let x = i as f32 / (size - 1).max(1) as f32;
                transfer.decode(x)
            })
            .collect();
        Self {
            transfer,
            inverse: true,
            entries,
        }
    }

    /// Returns `true` when this LUT maps encoded → linear (inverse direction).
    #[must_use]
    pub fn is_inverse(&self) -> bool {
        self.inverse
    }

    /// Apply the LUT to a single normalised value using linear interpolation.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    #[must_use]
    pub fn apply(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let n = self.entries.len();
        if n == 0 {
            return x;
        }
        let pos = x * (n - 1) as f32;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = pos.fract();
        self.entries[lo] * (1.0 - frac) + self.entries[hi] * frac
    }

    /// Number of entries in this LUT.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the LUT has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A matched pair of forward and inverse HDR LUTs for the same transfer function.
#[derive(Debug, Clone)]
pub struct HdrLutPair {
    /// Forward (encode) LUT.
    pub forward: HdrLut,
    /// Inverse (decode) LUT.
    pub inverse: HdrLut,
}

impl HdrLutPair {
    /// Build a matched pair for the given transfer function and LUT size.
    #[must_use]
    pub fn build(transfer: HdrTransfer, size: usize) -> Self {
        Self {
            forward: HdrLut::build_forward(transfer, size),
            inverse: HdrLut::build_inverse(transfer, size),
        }
    }

    /// Apply the forward LUT (linear → encoded signal).
    #[must_use]
    pub fn forward_apply(&self, x: f32) -> f32 {
        self.forward.apply(x)
    }

    /// Apply the inverse LUT (encoded signal → linear).
    #[must_use]
    pub fn inverse_apply(&self, x: f32) -> f32 {
        self.inverse.apply(x)
    }

    /// Verify that `forward_apply` followed by `inverse_apply` is close to identity.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn roundtrip_error(&self, samples: usize) -> f32 {
        let n = samples.max(2);
        (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                let encoded = self.forward_apply(x);
                let decoded = self.inverse_apply(encoded);
                (decoded - x).abs()
            })
            .fold(0.0_f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_peak_luminance() {
        assert!((HdrTransfer::Pq.peak_luminance_nits() - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_hlg_peak_luminance() {
        assert!((HdrTransfer::Hlg.peak_luminance_nits() - 1_000.0).abs() < 1.0);
    }

    #[test]
    fn test_scene_referred_flags() {
        assert!(HdrTransfer::Slog3.is_scene_referred());
        assert!(HdrTransfer::LogC.is_scene_referred());
        assert!(!HdrTransfer::Pq.is_scene_referred());
        assert!(!HdrTransfer::Hlg.is_scene_referred());
    }

    #[test]
    fn test_labels() {
        assert_eq!(HdrTransfer::Pq.label(), "PQ");
        assert_eq!(HdrTransfer::Hlg.label(), "HLG");
        assert_eq!(HdrTransfer::Slog3.label(), "SLog3");
        assert_eq!(HdrTransfer::LogC.label(), "LogC");
    }

    #[test]
    fn test_forward_lut_is_not_inverse() {
        let lut = HdrLut::build_forward(HdrTransfer::Pq, 64);
        assert!(!lut.is_inverse());
    }

    #[test]
    fn test_inverse_lut_is_inverse() {
        let lut = HdrLut::build_inverse(HdrTransfer::Hlg, 64);
        assert!(lut.is_inverse());
    }

    #[test]
    fn test_lut_correct_len() {
        let lut = HdrLut::build_forward(HdrTransfer::Pq, 1024);
        assert_eq!(lut.len(), 1024);
        assert!(!lut.is_empty());
    }

    #[test]
    fn test_lut_apply_clamped_below_zero() {
        let lut = HdrLut::build_forward(HdrTransfer::Pq, 64);
        let out = lut.apply(-0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_lut_apply_clamped_above_one() {
        let lut = HdrLut::build_forward(HdrTransfer::Pq, 64);
        let out = lut.apply(2.0);
        assert!(out.is_finite());
    }

    #[test]
    fn test_pair_forward_inverse_roundtrip_logc() {
        let pair = HdrLutPair::build(HdrTransfer::LogC, 1024);
        let err = pair.roundtrip_error(64);
        // Expect reasonable round-trip accuracy for 1024-entry LUT.
        assert!(err < 0.1, "round-trip error too large: {err}");
    }

    #[test]
    fn test_pair_forward_apply_increases_for_hlg() {
        let pair = HdrLutPair::build(HdrTransfer::Hlg, 256);
        // For HLG at moderate brightness the encoded value should be > 0.
        let out = pair.forward_apply(0.18);
        assert!(out > 0.0);
    }

    #[test]
    fn test_pair_inverse_apply_nonnegative() {
        let pair = HdrLutPair::build(HdrTransfer::Slog3, 256);
        let out = pair.inverse_apply(0.5);
        assert!(out >= 0.0);
    }
}
