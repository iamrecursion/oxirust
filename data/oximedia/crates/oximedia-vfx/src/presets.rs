//! Effect presets for common use cases.

use crate::{
    generator::{
        BarsType, ColorBars, Gradient, GradientType, Noise, NoiseType, Pattern, PatternType,
    },
    keying::{AdvancedKey, KeyColor},
    light::{Bloom, BloomQuality, FlareType, Glow, GlowMode, LensFlare},
    particle::{Dust, DustMode, Rain, RainIntensity, Snow, SnowStyle, SparkType, Sparks},
    style::{
        Cartoon, CartoonStyle, Halftone, HalftonePattern, Mosaic, MosaicMode, OilPaint, PaintStyle,
        Sketch, SketchStyle,
    },
    transition::{
        Dissolve, Push, PushDirection, Slide, SlideDirection, Wipe, WipePattern, Zoom, ZoomMode,
    },
    Color, EasingFunction,
};

/// Preset collection for transitions.
pub struct TransitionPresets;

impl TransitionPresets {
    /// Create a smooth cross-dissolve.
    #[must_use]
    pub fn smooth_dissolve() -> Dissolve {
        Dissolve::new().with_power(1.0)
    }

    /// Create an ease-in dissolve.
    #[must_use]
    pub fn ease_in_dissolve() -> Dissolve {
        Dissolve::new().with_power(2.0)
    }

    /// Create an ease-out dissolve.
    #[must_use]
    pub fn ease_out_dissolve() -> Dissolve {
        Dissolve::new().with_power(0.5)
    }

    /// Create dip to black dissolve.
    #[must_use]
    pub fn dip_to_black() -> Dissolve {
        Dissolve::new().with_dip_to_black(true)
    }

    /// Create classic horizontal wipe.
    #[must_use]
    pub fn horizontal_wipe() -> Wipe {
        Wipe::new(WipePattern::LeftToRight).with_feather(0.05)
    }

    /// Create vertical wipe.
    #[must_use]
    pub fn vertical_wipe() -> Wipe {
        Wipe::new(WipePattern::TopToBottom).with_feather(0.05)
    }

    /// Create diagonal wipe.
    #[must_use]
    pub fn diagonal_wipe() -> Wipe {
        Wipe::new(WipePattern::DiagonalTLBR).with_feather(0.05)
    }

    /// Create circular wipe.
    #[must_use]
    pub fn circular_wipe() -> Wipe {
        Wipe::new(WipePattern::CircleIn).with_feather(0.1)
    }

    /// Create clock wipe.
    #[must_use]
    pub fn clock_wipe() -> Wipe {
        Wipe::new(WipePattern::ClockWipe).with_feather(0.05)
    }

    /// Create barn door wipe.
    #[must_use]
    pub fn barn_door() -> Wipe {
        Wipe::new(WipePattern::BarnDoorHorizontal).with_feather(0.05)
    }

    /// Create push left transition.
    #[must_use]
    pub fn push_left() -> Push {
        Push::new(PushDirection::Left)
    }

    /// Create push right transition.
    #[must_use]
    pub fn push_right() -> Push {
        Push::new(PushDirection::Right)
    }

    /// Create slide in from left.
    #[must_use]
    pub fn slide_in_left() -> Slide {
        Slide::new(SlideDirection::FromLeft).with_easing(EasingFunction::EaseOut)
    }

    /// Create slide in from right.
    #[must_use]
    pub fn slide_in_right() -> Slide {
        Slide::new(SlideDirection::FromRight).with_easing(EasingFunction::EaseOut)
    }

    /// Create zoom in transition.
    #[must_use]
    pub fn zoom_in() -> Zoom {
        Zoom::new(ZoomMode::In)
    }

    /// Create zoom out transition.
    #[must_use]
    pub fn zoom_out() -> Zoom {
        Zoom::new(ZoomMode::Out)
    }

    /// Create cross zoom transition.
    #[must_use]
    pub fn cross_zoom() -> Zoom {
        Zoom::new(ZoomMode::Cross)
    }
}

/// Preset collection for generators.
pub struct GeneratorPresets;

impl GeneratorPresets {
    /// Create SMPTE color bars.
    #[must_use]
    pub fn smpte_bars() -> ColorBars {
        ColorBars::new(BarsType::Smpte)
    }

    /// Create EBU color bars.
    #[must_use]
    pub fn ebu_bars() -> ColorBars {
        ColorBars::new(BarsType::Ebu)
    }

    /// Create checkerboard pattern.
    #[must_use]
    pub fn checkerboard() -> Pattern {
        Pattern::new(PatternType::Checkerboard).with_size(32)
    }

    /// Create grid pattern.
    #[must_use]
    pub fn grid() -> Pattern {
        Pattern::new(PatternType::Grid).with_size(64)
    }

    /// Create zone plate test pattern.
    #[must_use]
    pub fn zone_plate() -> Pattern {
        Pattern::new(PatternType::ZonePlate).with_size(50)
    }

    /// Create white noise.
    #[must_use]
    pub fn white_noise() -> Noise {
        Noise::new(NoiseType::White).with_amplitude(0.5)
    }

    /// Create Perlin noise.
    #[must_use]
    pub fn perlin_noise() -> Noise {
        Noise::new(NoiseType::Perlin)
            .with_amplitude(1.0)
            .with_scale(1.0)
    }

    /// Create linear gradient.
    #[must_use]
    pub fn linear_gradient() -> Gradient {
        Gradient::new(GradientType::Linear)
            .with_start_color(Color::black())
            .with_end_color(Color::white())
    }

    /// Create radial gradient.
    #[must_use]
    pub fn radial_gradient() -> Gradient {
        Gradient::new(GradientType::Radial)
            .with_start_color(Color::white())
            .with_end_color(Color::black())
            .with_center(0.5, 0.5)
    }

    /// Create sunset gradient.
    #[must_use]
    pub fn sunset_gradient() -> Gradient {
        Gradient::new(GradientType::Linear)
            .with_start_color(Color::rgb(255, 94, 77))
            .with_end_color(Color::rgb(245, 175, 25))
            .with_angle(90.0)
    }

    /// Create ocean gradient.
    #[must_use]
    pub fn ocean_gradient() -> Gradient {
        Gradient::new(GradientType::Radial)
            .with_start_color(Color::rgb(0, 119, 190))
            .with_end_color(Color::rgb(0, 40, 83))
    }
}

/// Preset collection for stylization effects.
pub struct StylePresets;

impl StylePresets {
    /// Create cartoon effect with default settings.
    #[must_use]
    pub fn cartoon() -> Cartoon {
        Cartoon::new(CartoonStyle::CelShading)
            .with_levels(4)
            .with_edge_threshold(0.3)
    }

    /// Create comic book effect.
    #[must_use]
    pub fn comic_book() -> Cartoon {
        Cartoon::new(CartoonStyle::ComicBook)
            .with_levels(6)
            .with_edge_threshold(0.2)
    }

    /// Create pencil sketch.
    #[must_use]
    pub fn pencil_sketch() -> Sketch {
        Sketch::new(SketchStyle::Pencil).with_intensity(1.0)
    }

    /// Create pen and ink.
    #[must_use]
    pub fn pen_ink() -> Sketch {
        Sketch::new(SketchStyle::PenInk).with_intensity(1.2)
    }

    /// Create oil painting effect.
    #[must_use]
    pub fn oil_painting() -> OilPaint {
        OilPaint::new(PaintStyle::Oil)
            .with_radius(5)
            .with_levels(25)
    }

    /// Create watercolor effect.
    #[must_use]
    pub fn watercolor() -> OilPaint {
        OilPaint::new(PaintStyle::Watercolor)
            .with_radius(3)
            .with_levels(15)
    }

    /// Create mosaic effect.
    #[must_use]
    pub fn mosaic() -> Mosaic {
        Mosaic::new(MosaicMode::Square).with_block_size(16)
    }

    /// Create pixelate effect.
    #[must_use]
    pub fn pixelate() -> Mosaic {
        Mosaic::new(MosaicMode::Square).with_block_size(8)
    }

    /// Create halftone dots.
    #[must_use]
    pub fn halftone_dots() -> Halftone {
        Halftone::new(HalftonePattern::Dots)
            .with_dot_size(4)
            .with_angle(45.0)
    }

    /// Create halftone lines.
    #[must_use]
    pub fn halftone_lines() -> Halftone {
        Halftone::new(HalftonePattern::Lines)
            .with_dot_size(3)
            .with_angle(0.0)
    }

    /// Create crosshatch effect.
    #[must_use]
    pub fn crosshatch() -> Halftone {
        Halftone::new(HalftonePattern::Crosshatch)
            .with_dot_size(5)
            .with_angle(45.0)
    }
}

/// Preset collection for light effects.
pub struct LightPresets;

impl LightPresets {
    /// Create subtle lens flare.
    #[must_use]
    pub fn subtle_flare() -> LensFlare {
        LensFlare::new(FlareType::Anamorphic)
            .with_position(0.8, 0.2)
            .with_intensity(0.5)
            .with_color(Color::rgb(255, 220, 180))
    }

    /// Create bright lens flare.
    #[must_use]
    pub fn bright_flare() -> LensFlare {
        LensFlare::new(FlareType::Spherical)
            .with_position(0.7, 0.3)
            .with_intensity(1.5)
            .with_color(Color::white())
    }

    /// Create star burst.
    #[must_use]
    pub fn star_burst() -> LensFlare {
        LensFlare::new(FlareType::StarBurst)
            .with_position(0.5, 0.5)
            .with_intensity(1.0)
            .with_color(Color::rgb(255, 255, 200))
    }

    /// Create soft glow.
    #[must_use]
    pub fn soft_glow() -> Glow {
        Glow::new(GlowMode::Soft)
            .with_radius(15)
            .with_intensity(0.6)
            .with_threshold(180)
    }

    /// Create hard glow.
    #[must_use]
    pub fn hard_glow() -> Glow {
        Glow::new(GlowMode::Hard)
            .with_radius(8)
            .with_intensity(1.2)
            .with_threshold(200)
    }

    /// Create HDR bloom.
    #[must_use]
    pub fn hdr_bloom() -> Bloom {
        Bloom::new(BloomQuality::High)
            .with_threshold(0.8)
            .with_intensity(1.5)
            .with_radius(25)
    }

    /// Create subtle bloom.
    #[must_use]
    pub fn subtle_bloom() -> Bloom {
        Bloom::new(BloomQuality::Medium)
            .with_threshold(0.9)
            .with_intensity(0.8)
            .with_radius(15)
    }
}

/// Preset collection for particle effects.
pub struct ParticlePresets;

impl ParticlePresets {
    /// Create light snow.
    #[must_use]
    pub fn light_snow() -> Snow {
        Snow::new(SnowStyle::Light).with_wind(0.1)
    }

    /// Create heavy snow.
    #[must_use]
    pub fn heavy_snow() -> Snow {
        Snow::new(SnowStyle::Heavy).with_wind(0.3)
    }

    /// Create blizzard.
    #[must_use]
    pub fn blizzard() -> Snow {
        Snow::new(SnowStyle::Blizzard).with_wind(0.7)
    }

    /// Create light rain.
    #[must_use]
    pub fn light_rain() -> Rain {
        Rain::new(RainIntensity::Drizzle).with_angle(10.0)
    }

    /// Create heavy rain.
    #[must_use]
    pub fn heavy_rain() -> Rain {
        Rain::new(RainIntensity::Heavy).with_angle(15.0)
    }

    /// Create storm.
    #[must_use]
    pub fn storm() -> Rain {
        Rain::new(RainIntensity::Storm).with_angle(25.0)
    }

    /// Create fire sparks.
    #[must_use]
    pub fn fire_sparks() -> Sparks {
        Sparks::new(SparkType::Fire)
            .with_source(0.5, 0.9)
            .with_color(Color::rgb(255, 140, 0))
    }

    /// Create electric sparks.
    #[must_use]
    pub fn electric_sparks() -> Sparks {
        Sparks::new(SparkType::Electric)
            .with_source(0.5, 0.5)
            .with_color(Color::rgb(100, 200, 255))
    }

    /// Create magic sparks.
    #[must_use]
    pub fn magic_sparks() -> Sparks {
        Sparks::new(SparkType::Magic)
            .with_source(0.5, 0.3)
            .with_color(Color::rgb(200, 100, 255))
    }

    /// Create floating dust.
    #[must_use]
    pub fn floating_dust() -> Dust {
        Dust::new(DustMode::Floating).with_density(0.5)
    }

    /// Create rising dust.
    #[must_use]
    pub fn rising_dust() -> Dust {
        Dust::new(DustMode::Rising).with_density(0.7)
    }

    /// Create swirling dust.
    #[must_use]
    pub fn swirling_dust() -> Dust {
        Dust::new(DustMode::Swirling).with_density(0.6)
    }
}

/// Preset collection for keying.
pub struct KeyingPresets;

impl KeyingPresets {
    /// Create standard green screen key.
    #[must_use]
    pub fn green_screen() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Green)
            .with_threshold(0.4)
            .with_tolerance(0.2)
            .with_feather(0.1)
            .with_despill(0.5)
    }

    /// Create aggressive green key.
    #[must_use]
    pub fn aggressive_green() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Green)
            .with_threshold(0.3)
            .with_tolerance(0.3)
            .with_feather(0.05)
            .with_despill(0.8)
    }

    /// Create soft green key.
    #[must_use]
    pub fn soft_green() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Green)
            .with_threshold(0.5)
            .with_tolerance(0.15)
            .with_feather(0.15)
            .with_despill(0.3)
    }

    /// Create standard blue screen key.
    #[must_use]
    pub fn blue_screen() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Blue)
            .with_threshold(0.4)
            .with_tolerance(0.2)
            .with_feather(0.1)
            .with_despill(0.5)
    }

    /// Create aggressive blue key.
    #[must_use]
    pub fn aggressive_blue() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Blue)
            .with_threshold(0.3)
            .with_tolerance(0.3)
            .with_feather(0.05)
            .with_despill(0.8)
    }

    /// Create soft blue key.
    #[must_use]
    pub fn soft_blue() -> AdvancedKey {
        AdvancedKey::new(KeyColor::Blue)
            .with_threshold(0.5)
            .with_tolerance(0.15)
            .with_feather(0.15)
            .with_despill(0.3)
    }
}

/// Effect preset manager.
pub struct PresetManager {
    presets: Vec<EffectPreset>,
}

/// A saved effect preset.
#[derive(Debug, Clone)]
pub struct EffectPreset {
    /// Preset name.
    pub name: String,
    /// Preset category.
    pub category: String,
    /// Preset description.
    pub description: String,
    /// Preset tags for search.
    pub tags: Vec<String>,
}

impl PresetManager {
    /// Create a new preset manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            presets: Vec::new(),
        }
    }

    /// Load default presets.
    pub fn load_defaults(&mut self) {
        // Transition presets
        self.add_preset(
            "Smooth Dissolve",
            "Transitions",
            "Smooth cross-dissolve",
            vec!["dissolve", "fade"],
        );
        self.add_preset(
            "Dip to Black",
            "Transitions",
            "Dissolve through black",
            vec!["dissolve", "black"],
        );
        self.add_preset(
            "Horizontal Wipe",
            "Transitions",
            "Classic horizontal wipe",
            vec!["wipe", "horizontal"],
        );
        self.add_preset(
            "Clock Wipe",
            "Transitions",
            "Clock-style wipe",
            vec!["wipe", "clock"],
        );
        self.add_preset(
            "Push Left",
            "Transitions",
            "Push frame to left",
            vec!["push", "slide"],
        );
        self.add_preset(
            "Zoom In",
            "Transitions",
            "Zoom in transition",
            vec!["zoom", "scale"],
        );

        // Generator presets
        self.add_preset(
            "SMPTE Bars",
            "Generators",
            "SMPTE color bars",
            vec!["bars", "test"],
        );
        self.add_preset(
            "Checkerboard",
            "Generators",
            "Checkerboard pattern",
            vec!["pattern", "test"],
        );
        self.add_preset(
            "Perlin Noise",
            "Generators",
            "Smooth Perlin noise",
            vec!["noise", "texture"],
        );
        self.add_preset(
            "Sunset Gradient",
            "Generators",
            "Warm sunset colors",
            vec!["gradient", "color"],
        );

        // Style presets
        self.add_preset(
            "Cartoon",
            "Stylization",
            "Cel-shaded cartoon",
            vec!["cartoon", "artistic"],
        );
        self.add_preset(
            "Pencil Sketch",
            "Stylization",
            "Pencil drawing",
            vec!["sketch", "artistic"],
        );
        self.add_preset(
            "Oil Painting",
            "Stylization",
            "Oil paint effect",
            vec!["paint", "artistic"],
        );
        self.add_preset(
            "Halftone Dots",
            "Stylization",
            "Comic book dots",
            vec!["halftone", "comic"],
        );

        // Light presets
        self.add_preset(
            "Lens Flare",
            "Light",
            "Subtle lens flare",
            vec!["flare", "lens"],
        );
        self.add_preset(
            "Soft Glow",
            "Light",
            "Soft glow effect",
            vec!["glow", "bloom"],
        );
        self.add_preset(
            "HDR Bloom",
            "Light",
            "HDR bloom effect",
            vec!["bloom", "hdr"],
        );

        // Particle presets
        self.add_preset("Snow", "Particles", "Falling snow", vec!["snow", "weather"]);
        self.add_preset("Rain", "Particles", "Rainfall", vec!["rain", "weather"]);
        self.add_preset(
            "Fire Sparks",
            "Particles",
            "Fire embers",
            vec!["sparks", "fire"],
        );

        // Keying presets
        self.add_preset(
            "Green Screen",
            "Keying",
            "Standard green key",
            vec!["key", "chroma"],
        );
        self.add_preset(
            "Blue Screen",
            "Keying",
            "Standard blue key",
            vec!["key", "chroma"],
        );
    }

    fn add_preset(&mut self, name: &str, category: &str, description: &str, tags: Vec<&str>) {
        self.presets.push(EffectPreset {
            name: name.to_string(),
            category: category.to_string(),
            description: description.to_string(),
            tags: tags.iter().map(|s| (*s).to_string()).collect(),
        });
    }

    /// Get all presets.
    #[must_use]
    pub fn presets(&self) -> &[EffectPreset] {
        &self.presets
    }

    /// Search presets by tag.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&EffectPreset> {
        let query = query.to_lowercase();
        self.presets
            .iter()
            .filter(|p| {
                p.name.to_lowercase().contains(&query)
                    || p.tags.iter().any(|t| t.contains(&query))
                    || p.description.to_lowercase().contains(&query)
            })
            .collect()
    }

    /// Get presets by category.
    #[must_use]
    pub fn by_category(&self, category: &str) -> Vec<&EffectPreset> {
        self.presets
            .iter()
            .filter(|p| p.category == category)
            .collect()
    }

    /// Get all categories.
    #[must_use]
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<_> = self.presets.iter().map(|p| p.category.clone()).collect();
        cats.sort();
        cats.dedup();
        cats
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        let mut manager = Self::new();
        manager.load_defaults();
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_presets() {
        let _dissolve = TransitionPresets::smooth_dissolve();
        let _wipe = TransitionPresets::horizontal_wipe();
        let _push = TransitionPresets::push_left();
        let _slide = TransitionPresets::slide_in_left();
        let _zoom = TransitionPresets::zoom_in();
    }

    #[test]
    fn test_generator_presets() {
        let _bars = GeneratorPresets::smpte_bars();
        let _pattern = GeneratorPresets::checkerboard();
        let _noise = GeneratorPresets::perlin_noise();
        let _gradient = GeneratorPresets::linear_gradient();
    }

    #[test]
    fn test_style_presets() {
        let _cartoon = StylePresets::cartoon();
        let _sketch = StylePresets::pencil_sketch();
        let _paint = StylePresets::oil_painting();
        let _halftone = StylePresets::halftone_dots();
    }

    #[test]
    fn test_light_presets() {
        let _flare = LightPresets::subtle_flare();
        let _glow = LightPresets::soft_glow();
        let _bloom = LightPresets::hdr_bloom();
    }

    #[test]
    fn test_particle_presets() {
        let _snow = ParticlePresets::light_snow();
        let _rain = ParticlePresets::light_rain();
        let _sparks = ParticlePresets::fire_sparks();
        let _dust = ParticlePresets::floating_dust();
    }

    #[test]
    fn test_keying_presets() {
        let _green = KeyingPresets::green_screen();
        let _blue = KeyingPresets::blue_screen();
    }

    #[test]
    fn test_preset_manager() {
        let manager = PresetManager::default();
        assert!(!manager.presets().is_empty());

        let results = manager.search("dissolve");
        assert!(!results.is_empty());

        let transitions = manager.by_category("Transitions");
        assert!(!transitions.is_empty());

        let categories = manager.categories();
        assert!(categories.contains(&"Transitions".to_string()));
    }
}
