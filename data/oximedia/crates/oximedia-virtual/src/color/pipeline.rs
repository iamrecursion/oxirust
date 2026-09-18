//! Color pipeline management
//!
//! Manages the complete color pipeline from camera input to LED output,
//! including ACES color management with ACEScg working space and
//! RRT (Reference Rendering Transform) + ODT (Output Display Transform).

use super::{lut::LutProcessor, match_color::ColorMatcher, ColorTransform};
use crate::Result;
use serde::{Deserialize, Serialize};

/// ACES color space identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcesColorSpace {
    /// ACES2065-1 (AP0 primaries, linear) -- the archival interchange space
    Aces2065,
    /// ACEScg (AP1 primaries, linear) -- the CG working space
    AcesCg,
    /// ACEScc (AP1 primaries, logarithmic) -- for color grading
    AcesCc,
    /// ACEScct (AP1 primaries, logarithmic with toe) -- grading with shadow control
    AcesCct,
}

/// Output display transform target
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdtTarget {
    /// sRGB display (Rec.709 primaries, 2.2 gamma)
    SRgb,
    /// Rec.709 broadcast (Rec.709 primaries, BT.1886 EOTF)
    Rec709,
    /// Rec.2020 HDR (PQ EOTF, 1000 nit)
    Rec2020Pq,
    /// DCI-P3 theatrical (2.6 gamma, D65 whitepoint)
    DciP3,
}

/// ACES pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcesConfig {
    /// Working color space (typically ACEScg for CG compositing)
    pub working_space: AcesColorSpace,
    /// Output display transform target
    pub odt_target: OdtTarget,
    /// Enable tone mapping (RRT)
    pub rrt_enabled: bool,
    /// Exposure adjustment in stops (applied before RRT)
    pub exposure_stops: f32,
    /// Enable gamut mapping from working space to output
    pub gamut_mapping: bool,
}

impl Default for AcesConfig {
    fn default() -> Self {
        Self {
            working_space: AcesColorSpace::AcesCg,
            odt_target: OdtTarget::SRgb,
            rrt_enabled: true,
            exposure_stops: 0.0,
            gamut_mapping: true,
        }
    }
}

/// ACES color management processor.
///
/// Implements the ACES color pipeline:
///   Input (camera) -> IDT -> ACEScg working space -> RRT -> ODT -> display
///
/// The IDT (Input Device Transform) converts from input color space to ACES.
/// The RRT applies a film-like S-curve tone map.
/// The ODT converts from ACES to the display's color space.
pub struct AcesProcessor {
    config: AcesConfig,
    /// Pre-computed exposure multiplier
    exposure_gain: f32,
    /// Rec.709 -> AP1 (ACEScg) 3x3 matrix
    rec709_to_ap1: ColorTransform,
    /// AP1 (ACEScg) -> Rec.709 3x3 matrix
    ap1_to_rec709: ColorTransform,
}

impl AcesProcessor {
    /// Create a new ACES processor.
    #[must_use]
    pub fn new(config: AcesConfig) -> Self {
        let exposure_gain = 2.0_f32.powf(config.exposure_stops);

        // Rec.709 linear -> ACEScg (AP1) matrix
        // This is the standard ACES IDT for Rec.709
        let rec709_to_ap1 = ColorTransform {
            matrix: [
                [0.613_097_3, 0.339_523_1, 0.047_379_6],
                [0.070_194_2, 0.916_353_8, 0.013_451_9],
                [0.020_616_1, 0.109_569_6, 0.869_814_3],
            ],
            offset: [0.0, 0.0, 0.0],
        };

        // ACEScg (AP1) -> Rec.709 linear matrix (inverse of above)
        let ap1_to_rec709 = ColorTransform {
            matrix: [
                [1.704_858_7, -0.621_716_1, -0.083_299_0],
                [-0.130_076_8, 1.140_867_2, -0.010_790_3],
                [-0.023_964_1, -0.128_975_5, 1.152_939_6],
            ],
            offset: [0.0, 0.0, 0.0],
        };

        Self {
            config,
            exposure_gain,
            rec709_to_ap1,
            ap1_to_rec709,
        }
    }

    /// Process a single pixel through the full ACES pipeline.
    ///
    /// Input is linear Rec.709 RGB in [0, 1+]. Output depends on the ODT target.
    #[must_use]
    pub fn process_pixel(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Step 1: IDT -- convert from Rec.709 linear to ACEScg (AP1 linear)
        let acescg = self.rec709_to_acescg(rgb);

        // Step 2: Exposure adjustment
        let exposed = [
            acescg[0] * self.exposure_gain,
            acescg[1] * self.exposure_gain,
            acescg[2] * self.exposure_gain,
        ];

        // Step 3: RRT (Reference Rendering Transform) -- filmic tone mapping
        let tonemapped = if self.config.rrt_enabled {
            self.rrt(exposed)
        } else {
            exposed
        };

        // Step 4: ODT -- convert to display color space
        self.odt(tonemapped)
    }

    /// Process a frame buffer of RGB u8 pixels through the ACES pipeline.
    pub fn process_frame(&self, frame: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let expected = width * height * 3;
        if frame.len() != expected {
            return Err(crate::VirtualProductionError::Color(format!(
                "Frame size mismatch: expected {expected}, got {}",
                frame.len()
            )));
        }

        let mut output = vec![0u8; expected];
        for i in 0..(width * height) {
            let idx = i * 3;
            // Convert from sRGB u8 to linear f32
            let r_lin = srgb_to_linear(f32::from(frame[idx]) / 255.0);
            let g_lin = srgb_to_linear(f32::from(frame[idx + 1]) / 255.0);
            let b_lin = srgb_to_linear(f32::from(frame[idx + 2]) / 255.0);

            let result = self.process_pixel([r_lin, g_lin, b_lin]);

            // Depending on ODT target, the output may already be in display space
            output[idx] = (result[0].clamp(0.0, 1.0) * 255.0) as u8;
            output[idx + 1] = (result[1].clamp(0.0, 1.0) * 255.0) as u8;
            output[idx + 2] = (result[2].clamp(0.0, 1.0) * 255.0) as u8;
        }

        Ok(output)
    }

    /// Convert from Rec.709 linear to ACEScg (AP1 linear).
    #[must_use]
    fn rec709_to_acescg(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.rec709_to_ap1.apply(rgb)
    }

    /// Convert from ACEScg (AP1 linear) to Rec.709 linear.
    #[must_use]
    fn acescg_to_rec709(&self, acescg: [f32; 3]) -> [f32; 3] {
        self.ap1_to_rec709.apply(acescg)
    }

    /// ACES Reference Rendering Transform (RRT).
    ///
    /// Applies a filmic S-curve tone mapping that compresses highlights
    /// while preserving shadow detail. This is an approximation of the
    /// official ACES RRT using the fitted curve from Stephen Hill / Narkowicz.
    ///
    /// The curve: f(x) = (x*(a*x + b)) / (x*(c*x + d) + e)
    /// where a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
    #[must_use]
    fn rrt(&self, acescg: [f32; 3]) -> [f32; 3] {
        [
            aces_tonemap(acescg[0]),
            aces_tonemap(acescg[1]),
            aces_tonemap(acescg[2]),
        ]
    }

    /// Output Display Transform (ODT).
    ///
    /// Converts from tone-mapped ACEScg to the target display color space.
    #[must_use]
    fn odt(&self, tonemapped: [f32; 3]) -> [f32; 3] {
        // Convert back to Rec.709 linear
        let rec709_lin = if self.config.gamut_mapping {
            let raw = self.acescg_to_rec709(tonemapped);
            // Soft-clip negative values from gamut mapping
            [raw[0].max(0.0), raw[1].max(0.0), raw[2].max(0.0)]
        } else {
            tonemapped
        };

        match self.config.odt_target {
            OdtTarget::SRgb => [
                linear_to_srgb(rec709_lin[0]),
                linear_to_srgb(rec709_lin[1]),
                linear_to_srgb(rec709_lin[2]),
            ],
            OdtTarget::Rec709 => [
                bt1886_eotf_inverse(rec709_lin[0]),
                bt1886_eotf_inverse(rec709_lin[1]),
                bt1886_eotf_inverse(rec709_lin[2]),
            ],
            OdtTarget::Rec2020Pq => {
                // Simplified: just apply PQ EOTF inverse
                // In reality, this would also need Rec.709 -> Rec.2020 gamut conversion
                [
                    pq_oetf(rec709_lin[0]),
                    pq_oetf(rec709_lin[1]),
                    pq_oetf(rec709_lin[2]),
                ]
            }
            OdtTarget::DciP3 => [
                rec709_lin[0].max(0.0).powf(1.0 / 2.6),
                rec709_lin[1].max(0.0).powf(1.0 / 2.6),
                rec709_lin[2].max(0.0).powf(1.0 / 2.6),
            ],
        }
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &AcesConfig {
        &self.config
    }
}

/// ACES filmic tone map curve (Narkowicz 2015 fit).
///
/// f(x) = (x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14)
#[must_use]
fn aces_tonemap(x: f32) -> f32 {
    let x = x.max(0.0);
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    let num = x * (a * x + b);
    let den = x * (c * x + d) + e;
    if den.abs() < 1e-10 {
        return 0.0;
    }
    (num / den).clamp(0.0, 1.0)
}

/// sRGB EOTF: convert sRGB encoded value to linear.
#[must_use]
fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB OETF: convert linear to sRGB encoded value.
#[must_use]
fn linear_to_srgb(l: f32) -> f32 {
    let l = l.max(0.0);
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// BT.1886 inverse EOTF (used for Rec.709 broadcast displays).
///
/// V = L^(1/2.4) (simplified)
#[must_use]
fn bt1886_eotf_inverse(l: f32) -> f32 {
    l.max(0.0).powf(1.0 / 2.4)
}

/// ST 2084 PQ OETF (Perceptual Quantizer) for HDR displays.
///
/// Simplified PQ curve for [0, 1] normalised input.
#[must_use]
fn pq_oetf(l: f32) -> f32 {
    let l = l.max(0.0);
    let m1: f32 = 0.159_301_76;
    let m2: f32 = 78.843_75;
    let c1: f32 = 0.835_937_5;
    let c2: f32 = 18.851_563;
    let c3: f32 = 18.6875;

    let lm1 = l.powf(m1);
    let num = c1 + c2 * lm1;
    let den = 1.0 + c3 * lm1;
    if den.abs() < 1e-10 {
        return 0.0;
    }
    (num / den).powf(m2).clamp(0.0, 1.0)
}

/// Color pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPipelineConfig {
    /// Enable color matching
    pub color_matching: bool,
    /// Enable LUT application
    pub lut_enabled: bool,
    /// Input color space
    pub input_color_space: String,
    /// Output color space
    pub output_color_space: String,
    /// ACES pipeline configuration (None = bypass ACES)
    pub aces: Option<AcesConfig>,
}

impl Default for ColorPipelineConfig {
    fn default() -> Self {
        Self {
            color_matching: true,
            lut_enabled: false,
            input_color_space: "Rec709".to_string(),
            output_color_space: "Rec709".to_string(),
            aces: None,
        }
    }
}

/// Color pipeline
pub struct ColorPipeline {
    config: ColorPipelineConfig,
    color_matcher: Option<ColorMatcher>,
    lut_processor: Option<LutProcessor>,
    aces_processor: Option<AcesProcessor>,
}

impl ColorPipeline {
    /// Create new color pipeline
    pub fn new(config: ColorPipelineConfig) -> Result<Self> {
        let color_matcher = if config.color_matching {
            Some(ColorMatcher::new()?)
        } else {
            None
        };

        let lut_processor = if config.lut_enabled {
            Some(LutProcessor::new()?)
        } else {
            None
        };

        let aces_processor = config
            .aces
            .as_ref()
            .map(|aces_cfg| AcesProcessor::new(aces_cfg.clone()));

        Ok(Self {
            config,
            color_matcher,
            lut_processor,
            aces_processor,
        })
    }

    /// Process frame through color pipeline
    pub fn process(&mut self, frame: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        let mut output = frame.to_vec();

        // Apply color matching
        if let Some(matcher) = &mut self.color_matcher {
            output = matcher.process(&output, width, height)?;
        }

        // Apply ACES pipeline
        if let Some(aces) = &self.aces_processor {
            output = aces.process_frame(&output, width, height)?;
        }

        // Apply LUT
        if let Some(lut) = &mut self.lut_processor {
            output = lut.apply(&output, width, height)?;
        }

        Ok(output)
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ColorPipelineConfig {
        &self.config
    }

    /// Get the ACES processor (if configured).
    #[must_use]
    pub fn aces_processor(&self) -> Option<&AcesProcessor> {
        self.aces_processor.as_ref()
    }

    /// Update the ACES configuration at runtime.
    pub fn set_aces_config(&mut self, config: AcesConfig) {
        self.aces_processor = Some(AcesProcessor::new(config.clone()));
        self.config.aces = Some(config);
    }

    /// Disable the ACES pipeline.
    pub fn disable_aces(&mut self) {
        self.aces_processor = None;
        self.config.aces = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_pipeline() {
        let config = ColorPipelineConfig::default();
        let pipeline = ColorPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    // --- ACES pipeline tests ---

    #[test]
    fn test_aces_processor_creation() {
        let config = AcesConfig::default();
        let proc = AcesProcessor::new(config);
        assert_eq!(proc.config().working_space, AcesColorSpace::AcesCg);
    }

    #[test]
    fn test_aces_tonemap_black_stays_black() {
        let result = aces_tonemap(0.0);
        assert!(result.abs() < 1e-4, "black should stay black: {result}");
    }

    #[test]
    fn test_aces_tonemap_monotonic() {
        // Tone map should be monotonically increasing
        let mut prev = aces_tonemap(0.0);
        for i in 1..=100 {
            let x = i as f32 * 0.1;
            let y = aces_tonemap(x);
            assert!(y >= prev, "tonemap not monotonic at x={x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn test_aces_tonemap_compresses_highlights() {
        // Values above 1.0 should be compressed below 1.0
        let y = aces_tonemap(5.0);
        assert!(y < 1.0, "highlights should be compressed: {y}");
        assert!(y > 0.9, "shouldn't lose all highlight detail: {y}");
    }

    #[test]
    fn test_aces_tonemap_midtone_preservation() {
        // 18% grey (0.18) should map to approximately 0.21 (Narkowicz fit)
        let y = aces_tonemap(0.18);
        assert!(y > 0.05 && y < 0.5, "18% grey output: {y}");
    }

    #[test]
    fn test_srgb_linear_roundtrip() {
        for i in 0..=10 {
            let v = i as f32 * 0.1;
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            assert!(
                (back - v).abs() < 0.002,
                "sRGB roundtrip failed at {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_srgb_linear_zero_and_one() {
        assert!(srgb_to_linear(0.0).abs() < 1e-6);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-4);
        assert!(linear_to_srgb(0.0).abs() < 1e-6);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_aces_process_pixel_black() {
        let config = AcesConfig::default();
        let proc = AcesProcessor::new(config);
        let result = proc.process_pixel([0.0, 0.0, 0.0]);
        // Black should stay near black
        for ch in &result {
            assert!(*ch < 0.05, "black pixel should stay dark: {result:?}");
        }
    }

    #[test]
    fn test_aces_process_pixel_white_compressed() {
        let config = AcesConfig::default();
        let proc = AcesProcessor::new(config);
        let result = proc.process_pixel([1.0, 1.0, 1.0]);
        // White should be bright but tone-mapped
        for ch in &result {
            assert!(*ch > 0.5, "white should be bright: {result:?}");
            assert!(*ch <= 1.0, "output should be in [0,1]: {result:?}");
        }
    }

    #[test]
    fn test_aces_exposure_adjustment() {
        let config_neutral = AcesConfig {
            exposure_stops: 0.0,
            ..AcesConfig::default()
        };
        let config_bright = AcesConfig {
            exposure_stops: 2.0,
            ..AcesConfig::default()
        };

        let proc_neutral = AcesProcessor::new(config_neutral);
        let proc_bright = AcesProcessor::new(config_bright);

        let input = [0.18, 0.18, 0.18]; // 18% grey
        let r_neutral = proc_neutral.process_pixel(input);
        let r_bright = proc_bright.process_pixel(input);

        // +2 stops should be brighter
        assert!(
            r_bright[0] > r_neutral[0],
            "+2 stops should be brighter: {r_bright:?} vs {r_neutral:?}"
        );
    }

    #[test]
    fn test_aces_no_rrt() {
        let config = AcesConfig {
            rrt_enabled: false,
            ..AcesConfig::default()
        };
        let proc = AcesProcessor::new(config);
        // Without RRT, values above 1.0 in ACEScg can clip
        let result = proc.process_pixel([0.5, 0.5, 0.5]);
        for ch in &result {
            assert!(*ch <= 1.0 && *ch >= 0.0, "should be in range: {result:?}");
        }
    }

    #[test]
    fn test_aces_process_frame() {
        let config = AcesConfig::default();
        let proc = AcesProcessor::new(config);

        // 2x2 frame, all mid-grey sRGB
        let frame = vec![128u8; 2 * 2 * 3];
        let result = proc.process_frame(&frame, 2, 2);
        assert!(result.is_ok());
        let out = result.expect("should succeed in test");
        assert_eq!(out.len(), 12);
        // Output is u8, values inherently in [0, 255]
    }

    #[test]
    fn test_aces_process_frame_size_mismatch() {
        let config = AcesConfig::default();
        let proc = AcesProcessor::new(config);
        let frame = vec![0u8; 10]; // wrong size
        let result = proc.process_frame(&frame, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_odt_rec709() {
        let config = AcesConfig {
            odt_target: OdtTarget::Rec709,
            ..AcesConfig::default()
        };
        let proc = AcesProcessor::new(config);
        let result = proc.process_pixel([0.5, 0.5, 0.5]);
        for ch in &result {
            assert!(*ch >= 0.0 && *ch <= 1.0, "Rec709 output: {result:?}");
        }
    }

    #[test]
    fn test_odt_pq() {
        let config = AcesConfig {
            odt_target: OdtTarget::Rec2020Pq,
            ..AcesConfig::default()
        };
        let proc = AcesProcessor::new(config);
        let result = proc.process_pixel([0.5, 0.5, 0.5]);
        for ch in &result {
            assert!(*ch >= 0.0 && *ch <= 1.0, "PQ output: {result:?}");
        }
    }

    #[test]
    fn test_odt_dci_p3() {
        let config = AcesConfig {
            odt_target: OdtTarget::DciP3,
            ..AcesConfig::default()
        };
        let proc = AcesProcessor::new(config);
        let result = proc.process_pixel([0.5, 0.5, 0.5]);
        for ch in &result {
            assert!(*ch >= 0.0 && *ch <= 1.0, "DCI-P3 output: {result:?}");
        }
    }

    #[test]
    fn test_pipeline_with_aces() {
        let config = ColorPipelineConfig {
            color_matching: false,
            lut_enabled: false,
            input_color_space: "Rec709".to_string(),
            output_color_space: "sRGB".to_string(),
            aces: Some(AcesConfig::default()),
        };
        let mut pipeline = ColorPipeline::new(config).expect("should succeed in test");
        let frame = vec![128u8; 4 * 4 * 3];
        let result = pipeline.process(&frame, 4, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_disable_aces() {
        let config = ColorPipelineConfig {
            aces: Some(AcesConfig::default()),
            ..ColorPipelineConfig::default()
        };
        let mut pipeline = ColorPipeline::new(config).expect("should succeed in test");
        assert!(pipeline.aces_processor().is_some());

        pipeline.disable_aces();
        assert!(pipeline.aces_processor().is_none());
    }

    #[test]
    fn test_pipeline_set_aces_config() {
        let config = ColorPipelineConfig::default();
        let mut pipeline = ColorPipeline::new(config).expect("should succeed in test");
        assert!(pipeline.aces_processor().is_none());

        pipeline.set_aces_config(AcesConfig {
            exposure_stops: 1.0,
            ..AcesConfig::default()
        });
        assert!(pipeline.aces_processor().is_some());
    }

    #[test]
    fn test_pq_oetf_black() {
        let v = pq_oetf(0.0);
        assert!(v < 0.1, "PQ of black should be near zero: {v}");
    }

    #[test]
    fn test_bt1886_inverse_monotonic() {
        let mut prev = bt1886_eotf_inverse(0.0);
        for i in 1..=10 {
            let x = i as f32 * 0.1;
            let y = bt1886_eotf_inverse(x);
            assert!(y >= prev, "BT.1886 not monotonic at {x}");
            prev = y;
        }
    }

    #[test]
    fn test_aces_color_matrix_invertibility() {
        // Rec.709 -> AP1 -> Rec.709 should roundtrip
        let config = AcesConfig {
            rrt_enabled: false,
            gamut_mapping: false,
            ..AcesConfig::default()
        };
        let proc = AcesProcessor::new(config);
        let input = [0.3, 0.5, 0.7];
        let acescg = proc.rec709_to_acescg(input);
        let back = proc.acescg_to_rec709(acescg);
        for i in 0..3 {
            assert!(
                (back[i] - input[i]).abs() < 0.01,
                "matrix roundtrip ch{i}: {} vs {}",
                back[i],
                input[i]
            );
        }
    }
}
