//! Film stock emulation effects.
//!
//! Simulates the characteristic grain, colour palette, and tonal response of
//! classic photographic film stocks for cinematic VFX grading.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Known photographic film stocks and their characteristic responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilmStock {
    /// Kodak Vision3 500T – fine grain, rich shadows, warm highlights.
    KodakVision3,
    /// Fuji Eterna 500T – cool midtones, very fine grain.
    FujiEterna,
    /// Kodak Tri-X 400 – high-contrast monochrome with visible grain.
    KodakTriX,
    /// Kodak Ektachrome 100 – slide film, high saturation, fine grain.
    KodakEktachrome,
    /// Generic negative film (neutral baseline).
    GenericNegative,
    /// Generic reversal film (neutral baseline, high contrast).
    GenericReversal,
}

impl FilmStock {
    /// Normalised grain level characteristic of this stock (0.0 = smooth, 1.0 = very grainy).
    pub fn grain_level(self) -> f32 {
        match self {
            Self::KodakVision3 => 0.15,
            Self::FujiEterna => 0.10,
            Self::KodakTriX => 0.60,
            Self::KodakEktachrome => 0.08,
            Self::GenericNegative => 0.25,
            Self::GenericReversal => 0.20,
        }
    }

    /// ISO rating of this stock.
    pub fn iso(self) -> u32 {
        match self {
            Self::KodakVision3 => 500,
            Self::FujiEterna => 500,
            Self::KodakTriX => 400,
            Self::KodakEktachrome => 100,
            Self::GenericNegative => 400,
            Self::GenericReversal => 200,
        }
    }

    /// Whether this stock produces a monochrome (B&W) image.
    pub fn is_monochrome(self) -> bool {
        matches!(self, Self::KodakTriX)
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::KodakVision3 => "Kodak Vision3 500T",
            Self::FujiEterna => "Fuji Eterna 500T",
            Self::KodakTriX => "Kodak Tri-X 400",
            Self::KodakEktachrome => "Kodak Ektachrome 100",
            Self::GenericNegative => "Generic Negative",
            Self::GenericReversal => "Generic Reversal",
        }
    }
}

/// A colour shift applied per-channel during film emulation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorShift {
    /// Red channel delta (-1.0 – 1.0).
    pub red: f32,
    /// Green channel delta (-1.0 – 1.0).
    pub green: f32,
    /// Blue channel delta (-1.0 – 1.0).
    pub blue: f32,
}

impl Default for ColorShift {
    fn default() -> Self {
        Self {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
        }
    }
}

impl ColorShift {
    /// Create a new colour shift.
    pub fn new(red: f32, green: f32, blue: f32) -> Self {
        Self { red, green, blue }
    }

    /// Identity (no colour shift).
    pub fn identity() -> Self {
        Self::default()
    }

    /// Magnitude of the overall colour shift (Euclidean distance from origin).
    pub fn magnitude(self) -> f32 {
        (self.red * self.red + self.green * self.green + self.blue * self.blue).sqrt()
    }

    /// Returns `true` if the shift is effectively zero.
    pub fn is_identity(self) -> bool {
        self.magnitude() < 1e-6
    }

    /// Apply the colour shift to an RGB triple (values 0.0 – 1.0), clamping output.
    pub fn apply(self, r: f32, g: f32, b: f32) -> [f32; 3] {
        [
            (r + self.red).clamp(0.0, 1.0),
            (g + self.green).clamp(0.0, 1.0),
            (b + self.blue).clamp(0.0, 1.0),
        ]
    }
}

/// Film grain and colour emulation effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilmEffect {
    /// Film stock being emulated.
    pub stock: FilmStock,
    /// Grain intensity multiplier (0.0 = no grain, 2.0 = double grain).
    pub grain_intensity: f32,
    /// Colour shift applied on top of the stock's characteristic palette.
    pub color_shift: ColorShift,
    /// Whether to apply temporal grain variation (grain changes per frame).
    pub temporal_variation: bool,
    /// Lift (black point) adjustment (0.0 = no lift).
    pub lift: f32,
    /// Gain (white point) adjustment (1.0 = no change).
    pub gain: f32,
}

impl Default for FilmEffect {
    fn default() -> Self {
        Self {
            stock: FilmStock::GenericNegative,
            grain_intensity: 1.0,
            color_shift: ColorShift::identity(),
            temporal_variation: true,
            lift: 0.0,
            gain: 1.0,
        }
    }
}

impl FilmEffect {
    /// Create a film effect for a specific stock with default parameters.
    pub fn for_stock(stock: FilmStock) -> Self {
        Self {
            stock,
            ..Self::default()
        }
    }

    /// Effective grain amount (stock grain × intensity multiplier), clamped to 0 – 1.
    pub fn effective_grain(&self) -> f32 {
        (self.stock.grain_level() * self.grain_intensity).clamp(0.0, 1.0)
    }

    /// Apply photochemical grain to a single normalised luminance sample.
    ///
    /// `luma` is 0.0 – 1.0, `noise` is a pre-generated random value in -1.0 – 1.0.
    pub fn apply_grain(&self, luma: f32, noise: f32) -> f32 {
        // Grain is strongest in midtones, reduced in deep shadows and bright highlights
        let midtone_weight = 1.0 - (luma * 2.0 - 1.0).abs();
        let grain_amount = self.effective_grain() * midtone_weight * noise;
        (luma + grain_amount).clamp(0.0, 1.0)
    }

    /// Apply the colour shift to an RGB triple (0.0 – 1.0).
    pub fn apply_color_shift(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        self.color_shift.apply(r, g, b)
    }

    /// Apply the lift and gain tonal operators to a single channel value.
    pub fn apply_lift_gain(&self, value: f32) -> f32 {
        ((value + self.lift) * self.gain).clamp(0.0, 1.0)
    }

    /// Process a flat RGBA pixel buffer, applying grain, colour shift, and lift/gain.
    ///
    /// `noise_values` must have at least `width * height` pre-generated values
    /// in the range -1.0 – 1.0.
    pub fn process_buffer(&self, pixels: &mut [u8], noise_values: &[f32]) {
        let pixel_count = pixels.len() / 4;
        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= pixels.len() {
                break;
            }
            let noise = noise_values.get(i).copied().unwrap_or(0.0);

            let rf = f32::from(pixels[base]) / 255.0;
            let gf = f32::from(pixels[base + 1]) / 255.0;
            let bf = f32::from(pixels[base + 2]) / 255.0;

            // Apply colour shift
            let [rs, gs, bs] = self.apply_color_shift(rf, gf, bf);

            // Apply lift/gain
            let rl = self.apply_lift_gain(rs);
            let gl = self.apply_lift_gain(gs);
            let bl = self.apply_lift_gain(bs);

            // Apply grain using luminance
            let luma = 0.299 * rl + 0.587 * gl + 0.114 * bl;
            let grain_delta = self.apply_grain(luma, noise) - luma;

            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            {
                pixels[base] = ((rl + grain_delta).clamp(0.0, 1.0) * 255.0) as u8;
                pixels[base + 1] = ((gl + grain_delta).clamp(0.0, 1.0) * 255.0) as u8;
                pixels[base + 2] = ((bl + grain_delta).clamp(0.0, 1.0) * 255.0) as u8;
            }
            // Alpha unchanged
        }
    }
}

/// Preset factory for well-known film looks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilmEffectPreset;

impl FilmEffectPreset {
    /// Classic Kodak Vision3 cinematic look.
    pub fn create_kodak() -> FilmEffect {
        FilmEffect {
            stock: FilmStock::KodakVision3,
            grain_intensity: 1.0,
            color_shift: ColorShift::new(0.02, 0.0, -0.01), // slight warmth
            temporal_variation: true,
            lift: 0.02,
            gain: 0.98,
        }
    }

    /// Cool-toned Fuji Eterna look.
    pub fn create_fuji() -> FilmEffect {
        FilmEffect {
            stock: FilmStock::FujiEterna,
            grain_intensity: 0.8,
            color_shift: ColorShift::new(-0.01, 0.0, 0.02), // slight cool
            temporal_variation: true,
            lift: 0.01,
            gain: 0.99,
        }
    }

    /// High-contrast B&W Kodak Tri-X look.
    pub fn create_trix() -> FilmEffect {
        FilmEffect {
            stock: FilmStock::KodakTriX,
            grain_intensity: 1.5,
            color_shift: ColorShift::identity(),
            temporal_variation: true,
            lift: 0.0,
            gain: 1.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_film_stock_grain_level_range() {
        let stocks = [
            FilmStock::KodakVision3,
            FilmStock::FujiEterna,
            FilmStock::KodakTriX,
            FilmStock::KodakEktachrome,
            FilmStock::GenericNegative,
            FilmStock::GenericReversal,
        ];
        for s in stocks {
            let g = s.grain_level();
            assert!(g >= 0.0 && g <= 1.0, "{}: grain {g} out of range", s.name());
        }
    }

    #[test]
    fn test_film_stock_iso_positive() {
        assert!(FilmStock::KodakVision3.iso() > 0);
        assert!(FilmStock::KodakEktachrome.iso() > 0);
    }

    #[test]
    fn test_film_stock_monochrome() {
        assert!(FilmStock::KodakTriX.is_monochrome());
        assert!(!FilmStock::KodakVision3.is_monochrome());
    }

    #[test]
    fn test_color_shift_identity_magnitude_zero() {
        assert!(ColorShift::identity().magnitude() < 1e-6);
    }

    #[test]
    fn test_color_shift_magnitude_positive() {
        let cs = ColorShift::new(1.0, 0.0, 0.0);
        assert!((cs.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_shift_apply_clamps() {
        let cs = ColorShift::new(1.0, 1.0, 1.0);
        let result = cs.apply(0.9, 0.9, 0.9);
        for v in result {
            assert!(v <= 1.0);
        }
    }

    #[test]
    fn test_color_shift_is_identity() {
        assert!(ColorShift::identity().is_identity());
        assert!(!ColorShift::new(0.5, 0.0, 0.0).is_identity());
    }

    #[test]
    fn test_film_effect_effective_grain() {
        let effect = FilmEffect::for_stock(FilmStock::KodakVision3);
        let g = effect.effective_grain();
        assert!(g >= 0.0 && g <= 1.0);
    }

    #[test]
    fn test_film_effect_grain_zero_at_extremes() {
        let effect = FilmEffect::for_stock(FilmStock::KodakVision3);
        // At black (0.0) and white (1.0) midtone_weight = 0 so grain = 0 regardless of noise
        let black = effect.apply_grain(0.0, 1.0);
        assert!((black - 0.0).abs() < 1e-5, "expected ~0, got {black}");
        let white = effect.apply_grain(1.0, 1.0);
        assert!((white - 1.0).abs() < 1e-5, "expected ~1, got {white}");
    }

    #[test]
    fn test_film_effect_apply_color_shift_identity() {
        let effect = FilmEffect::default();
        let result = effect.apply_color_shift(0.5, 0.5, 0.5);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_film_effect_apply_lift_gain() {
        let effect = FilmEffect {
            lift: 0.1,
            gain: 1.0,
            ..Default::default()
        };
        let v = effect.apply_lift_gain(0.0);
        assert!((v - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_preset_create_kodak_stock() {
        let p = FilmEffectPreset::create_kodak();
        assert_eq!(p.stock, FilmStock::KodakVision3);
    }

    #[test]
    fn test_preset_create_trix_grain_intensity() {
        let p = FilmEffectPreset::create_trix();
        assert!(
            p.grain_intensity > 1.0,
            "Tri-X should have high grain intensity"
        );
    }

    #[test]
    fn test_process_buffer_length_preserved() {
        let effect = FilmEffect::for_stock(FilmStock::FujiEterna);
        let mut pixels = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
        let noise = vec![0.0f32; 16];
        let original_len = pixels.len();
        effect.process_buffer(&mut pixels, &noise);
        assert_eq!(pixels.len(), original_len);
    }

    #[test]
    fn test_film_effect_names_non_empty() {
        let stocks = [
            FilmStock::KodakVision3,
            FilmStock::FujiEterna,
            FilmStock::KodakTriX,
            FilmStock::KodakEktachrome,
            FilmStock::GenericNegative,
            FilmStock::GenericReversal,
        ];
        for s in stocks {
            assert!(!s.name().is_empty());
        }
    }
}
