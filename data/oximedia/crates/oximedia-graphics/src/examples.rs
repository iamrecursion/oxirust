//! Example graphics and demonstrations

use crate::animation::{AnimationClip, AnimationTrack, Easing, Keyframe};
use crate::color::{Color, Gradient};
use crate::effects::{Blur, ColorAdjustment, DropShadow, EffectStack, Glow};
use crate::elements::{BreakingNews, Bug, Scoreboard, SocialFeed, Ticker, WeatherGraphic};
use crate::error::Result;
use crate::overlay::SafeArea;
use crate::primitives::{Circle, Fill, Path, Point, Rect, Stroke};
use crate::professional::{AspectRatio, BroadcastLimiter, Framerate, ResolutionPreset};
use crate::render::{RenderTarget, SoftwareRenderer};
use crate::text::{FontManager, TextStyle};
use crate::transitions::{SlideDirection, Transition};
use std::time::Duration;

/// Example collection
pub struct Examples;

impl Examples {
    /// Create a simple lower third example
    pub fn simple_lower_third() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Background rectangle
        let bg_rect = Rect::new(50.0, 900.0, 800.0, 120.0);
        let bg_fill = Fill::Solid(Color::new(0, 0, 0, 200));
        renderer.render_rect(&mut target, bg_rect, &bg_fill, None)?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a weather graphic example
    #[must_use]
    pub fn weather_graphic_example() -> WeatherGraphic {
        WeatherGraphic::new(
            Point::new(1500.0, 50.0),
            350.0,
            250.0,
            "New York".to_string(),
            72,
            "Partly Cloudy".to_string(),
        )
        .with_high_low(75, 65)
    }

    /// Create a scoreboard example
    #[must_use]
    pub fn scoreboard_example() -> Scoreboard {
        let mut scoreboard = Scoreboard::new(
            Point::new(660.0, 50.0),
            600.0,
            120.0,
            "Lakers".to_string(),
            "Celtics".to_string(),
        );
        scoreboard.set_score(102, 98);
        scoreboard.set_time("2:45".to_string());
        scoreboard
    }

    /// Create a ticker example
    #[must_use]
    pub fn ticker_example() -> Ticker {
        Ticker::new(
            1000.0,
            50.0,
            "Breaking: Major announcement expected within the hour...".to_string(),
            TextStyle::default(),
            1920.0,
        )
        .with_speed(150.0)
    }

    /// Create a bug/watermark example
    #[must_use]
    pub fn bug_example() -> Bug {
        Bug::new(Point::new(1750.0, 50.0), (150.0, 150.0)).with_opacity(0.6)
    }

    /// Create a social feed example
    #[must_use]
    pub fn social_feed_example() -> SocialFeed {
        SocialFeed::new(
            Point::new(50.0, 500.0),
            450.0,
            200.0,
            "Twitter".to_string(),
            "@user".to_string(),
            "This is an example tweet to demonstrate social feed integration!".to_string(),
        )
    }

    /// Create a breaking news example
    #[must_use]
    pub fn breaking_news_example() -> BreakingNews {
        BreakingNews::new(
            0.0,
            100.0,
            "BREAKING NEWS".to_string(),
            "Major event happening now - stay tuned for updates".to_string(),
            1920.0,
        )
    }

    /// Create a gradient background example
    pub fn gradient_background() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        let gradient = Gradient::linear(
            (0.0, 0.0),
            (1920.0, 1080.0),
            vec![
                (0.0, Color::rgb(0, 50, 150)),
                (0.5, Color::rgb(0, 100, 200)),
                (1.0, Color::rgb(0, 150, 250)),
            ],
        );

        let rect = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        renderer.render_rect(&mut target, rect, &Fill::Gradient(gradient), None)?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a radial gradient example
    pub fn radial_gradient_example() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        let gradient = Gradient::radial(
            (960.0, 540.0),
            600.0,
            vec![
                (0.0, Color::WHITE),
                (0.5, Color::rgb(200, 200, 200)),
                (1.0, Color::BLACK),
            ],
        );

        let rect = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        renderer.render_rect(&mut target, rect, &Fill::Gradient(gradient), None)?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create an animation example
    #[must_use]
    pub fn animation_example() -> AnimationClip {
        let mut clip = AnimationClip::new(Duration::from_secs(2));

        // Position animation
        let mut position_track = AnimationTrack::new();
        position_track.add_keyframe(Keyframe::new(
            0.0,
            Point::new(-500.0, 900.0),
            Easing::EaseInOut,
        ));
        position_track.add_keyframe(Keyframe::new(
            1.0,
            Point::new(50.0, 900.0),
            Easing::EaseInOut,
        ));
        clip.position = Some(position_track);

        // Opacity animation
        let mut opacity_track = AnimationTrack::new();
        opacity_track.add_keyframe(Keyframe::new(0.0, 0.0, Easing::Linear));
        opacity_track.add_keyframe(Keyframe::new(1.0, 1.0, Easing::Linear));
        clip.opacity = Some(opacity_track);

        clip
    }

    /// Create a transition example
    #[must_use]
    pub fn transition_example() -> Transition {
        Transition::slide(SlideDirection::Left, Duration::from_millis(500))
    }

    /// Create an effect stack example
    #[must_use]
    pub fn effect_stack_example() -> EffectStack {
        EffectStack::new()
            .with_drop_shadow(DropShadow::new(5.0, 5.0, 10.0, Color::new(0, 0, 0, 128)))
            .with_glow(Glow::new(10.0, 0.7, Color::WHITE))
            .with_blur(Blur::gaussian(3.0))
            .with_color_adjustment(ColorAdjustment {
                brightness: 0.1,
                contrast: 0.2,
                saturation: 0.1,
                hue_shift: 0.0,
            })
    }

    /// Create a safe area guide example
    #[must_use]
    pub fn safe_area_example() -> SafeArea {
        SafeArea::calculate(1920, 1080)
    }

    /// Create a broadcast limiter example
    #[must_use]
    pub fn broadcast_limiter_example() -> BroadcastLimiter {
        BroadcastLimiter::legal_range()
    }

    /// Create a multi-layer composition example
    pub fn multi_layer_composition() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Layer 1: Background
        let bg_rect = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        renderer.render_rect(&mut target, bg_rect, &Fill::Solid(Color::BLACK), None)?;

        // Layer 2: Geometric shapes
        let circle1 = Circle::new(480.0, 270.0, 200.0);
        renderer.render_circle(
            &mut target,
            circle1,
            &Fill::Solid(Color::new(255, 0, 0, 100)),
            Some(&Stroke::new(Color::RED, 3.0)),
        )?;

        let circle2 = Circle::new(1440.0, 270.0, 200.0);
        renderer.render_circle(
            &mut target,
            circle2,
            &Fill::Solid(Color::new(0, 255, 0, 100)),
            Some(&Stroke::new(Color::GREEN, 3.0)),
        )?;

        let circle3 = Circle::new(960.0, 810.0, 200.0);
        renderer.render_circle(
            &mut target,
            circle3,
            &Fill::Solid(Color::new(0, 0, 255, 100)),
            Some(&Stroke::new(Color::BLUE, 3.0)),
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a path drawing example
    #[must_use]
    pub fn path_drawing_example() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(100.0, 100.0));
        path.line_to(Point::new(200.0, 100.0));
        path.line_to(Point::new(200.0, 200.0));
        path.line_to(Point::new(100.0, 200.0));
        path.close();
        path
    }

    /// Create a rounded rectangle path example
    #[must_use]
    pub fn rounded_rectangle_example() -> Path {
        let rect = Rect::new(100.0, 100.0, 400.0, 300.0);
        Path::rounded_rect(rect, 20.0)
    }

    /// Create a bezier curve example
    #[must_use]
    pub fn bezier_curve_example() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(100.0, 300.0));
        path.cubic_to(
            Point::new(200.0, 100.0),
            Point::new(300.0, 500.0),
            Point::new(400.0, 300.0),
        );
        path
    }

    /// Create a complex sports scoreboard
    pub fn complex_sports_scoreboard() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Main background
        let bg = Rect::new(660.0, 50.0, 600.0, 180.0);
        renderer.render_rect(
            &mut target,
            bg,
            &Fill::Solid(Color::new(0, 0, 0, 220)),
            None,
        )?;

        // Team sections
        let home_bg = Rect::new(660.0, 50.0, 300.0, 90.0);
        renderer.render_rect(
            &mut target,
            home_bg,
            &Fill::Solid(Color::new(50, 0, 0, 255)),
            None,
        )?;

        let away_bg = Rect::new(960.0, 50.0, 300.0, 90.0);
        renderer.render_rect(
            &mut target,
            away_bg,
            &Fill::Solid(Color::new(0, 0, 50, 255)),
            None,
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a weather map overlay
    pub fn weather_map_overlay() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Temperature zones
        for i in 0..5 {
            let y = 200.0 + i as f32 * 150.0;
            let temp_zone = Rect::new(100.0, y, 800.0, 120.0);
            let color = match i {
                0 => Color::new(255, 0, 0, 150),   // Hot
                1 => Color::new(255, 128, 0, 150), // Warm
                2 => Color::new(255, 255, 0, 150), // Mild
                3 => Color::new(0, 255, 0, 150),   // Cool
                _ => Color::new(0, 0, 255, 150),   // Cold
            };
            renderer.render_rect(&mut target, temp_zone, &Fill::Solid(color), None)?;
        }

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create an election results graphic
    pub fn election_results_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Background
        let bg = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        renderer.render_rect(&mut target, bg, &Fill::Solid(Color::rgb(20, 20, 40)), None)?;

        // Results bars
        let dem_bar = Rect::new(100.0, 400.0, 800.0, 100.0);
        renderer.render_rect(
            &mut target,
            dem_bar,
            &Fill::Solid(Color::rgb(0, 100, 200)),
            None,
        )?;

        let rep_bar = Rect::new(100.0, 550.0, 600.0, 100.0);
        renderer.render_rect(
            &mut target,
            rep_bar,
            &Fill::Solid(Color::rgb(200, 0, 0)),
            None,
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a stock ticker graphic
    pub fn stock_ticker_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Ticker background
        let ticker_bg = Rect::new(0.0, 1020.0, 1920.0, 60.0);
        renderer.render_rect(
            &mut target,
            ticker_bg,
            &Fill::Solid(Color::new(0, 50, 0, 220)),
            None,
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a podcast overlay
    pub fn podcast_overlay() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Guest name plate
        let name_plate = Rect::new(50.0, 900.0, 600.0, 150.0);
        renderer.render_rect(
            &mut target,
            name_plate,
            &Fill::Solid(Color::new(80, 40, 120, 200)),
            Some(&Stroke::new(Color::new(150, 100, 200, 255), 3.0)),
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create an esports overlay
    pub fn esports_overlay() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Team 1 panel
        let team1_panel = Rect::new(50.0, 50.0, 400.0, 150.0);
        let gradient1 = Gradient::linear(
            (50.0, 50.0),
            (450.0, 50.0),
            vec![
                (0.0, Color::new(255, 0, 0, 220)),
                (1.0, Color::new(150, 0, 0, 220)),
            ],
        );
        renderer.render_rect(&mut target, team1_panel, &Fill::Gradient(gradient1), None)?;

        // Team 2 panel
        let team2_panel = Rect::new(1470.0, 50.0, 400.0, 150.0);
        let gradient2 = Gradient::linear(
            (1470.0, 50.0),
            (1870.0, 50.0),
            vec![
                (0.0, Color::new(0, 0, 255, 220)),
                (1.0, Color::new(0, 0, 150, 220)),
            ],
        );
        renderer.render_rect(&mut target, team2_panel, &Fill::Gradient(gradient2), None)?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a countdown timer graphic
    pub fn countdown_timer_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Timer background circle
        let timer_circle = Circle::new(960.0, 540.0, 300.0);
        renderer.render_circle(
            &mut target,
            timer_circle,
            &Fill::Solid(Color::new(0, 0, 0, 200)),
            Some(&Stroke::new(Color::new(255, 200, 0, 255), 10.0)),
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a donation alert graphic
    pub fn donation_alert_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        target.clear(Color::TRANSPARENT);

        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Alert panel
        let alert_panel = Rect::new(560.0, 400.0, 800.0, 280.0);
        let gradient = Gradient::radial(
            (960.0, 540.0),
            400.0,
            vec![
                (0.0, Color::new(255, 215, 0, 240)),
                (1.0, Color::new(200, 150, 0, 240)),
            ],
        );
        renderer.render_rect(
            &mut target,
            alert_panel,
            &Fill::Gradient(gradient),
            Some(&Stroke::new(Color::new(255, 215, 0, 255), 5.0)),
        )?;

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a poll results graphic
    pub fn poll_results_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Background
        let bg = Rect::new(0.0, 0.0, 1920.0, 1080.0);
        renderer.render_rect(&mut target, bg, &Fill::Solid(Color::rgb(30, 30, 50)), None)?;

        // Poll bars
        let heights = [650.0, 450.0, 350.0, 550.0];
        let colors = [
            Color::rgb(100, 150, 255),
            Color::rgb(150, 100, 255),
            Color::rgb(255, 100, 150),
            Color::rgb(150, 255, 100),
        ];

        for (i, (&height, &color)) in heights.iter().zip(colors.iter()).enumerate() {
            let x = 200.0 + i as f32 * 400.0;
            let bar = Rect::new(x, 900.0 - height, 300.0, height);
            renderer.render_rect(&mut target, bar, &Fill::Solid(color), None)?;
        }

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Create a leaderboard graphic
    pub fn leaderboard_graphic() -> Result<RenderTarget> {
        let mut target = RenderTarget::new(1920, 1080)?;
        let font_manager = FontManager::new();
        let mut renderer = SoftwareRenderer::new(font_manager);

        // Background panel
        let panel = Rect::new(100.0, 100.0, 800.0, 800.0);
        renderer.render_rect(
            &mut target,
            panel,
            &Fill::Solid(Color::new(0, 0, 0, 220)),
            Some(&Stroke::new(Color::new(255, 200, 0, 255), 4.0)),
        )?;

        // Leaderboard entries
        for i in 0..10 {
            let y = 150.0 + i as f32 * 70.0;
            let entry = Rect::new(120.0, y, 760.0, 60.0);
            let color = if i == 0 {
                Color::new(255, 215, 0, 180)
            } else if i < 3 {
                Color::new(192, 192, 192, 150)
            } else {
                Color::new(50, 50, 70, 150)
            };
            renderer.render_rect(&mut target, entry, &Fill::Solid(color), None)?;
        }

        renderer.flush(&mut target);
        Ok(target)
    }

    /// Get example resolution presets
    #[must_use]
    pub fn resolution_presets() -> Vec<(ResolutionPreset, (u32, u32))> {
        vec![
            (
                ResolutionPreset::HD720,
                ResolutionPreset::HD720.dimensions(),
            ),
            (
                ResolutionPreset::FullHD,
                ResolutionPreset::FullHD.dimensions(),
            ),
            (
                ResolutionPreset::UHD4K,
                ResolutionPreset::UHD4K.dimensions(),
            ),
            (
                ResolutionPreset::UHD8K,
                ResolutionPreset::UHD8K.dimensions(),
            ),
        ]
    }

    /// Get example framerates
    #[must_use]
    pub fn framerate_examples() -> Vec<(Framerate, f32)> {
        vec![
            (Framerate::Film23976, Framerate::Film23976.as_float()),
            (Framerate::Film24, Framerate::Film24.as_float()),
            (Framerate::PAL25, Framerate::PAL25.as_float()),
            (Framerate::NTSC2997, Framerate::NTSC2997.as_float()),
            (Framerate::FPS30, Framerate::FPS30.as_float()),
            (Framerate::FPS60, Framerate::FPS60.as_float()),
        ]
    }

    /// Get example aspect ratios
    #[must_use]
    pub fn aspect_ratio_examples() -> Vec<(AspectRatio, f32)> {
        vec![
            (AspectRatio::Ratio16x9, AspectRatio::Ratio16x9.as_float()),
            (AspectRatio::Ratio4x3, AspectRatio::Ratio4x3.as_float()),
            (AspectRatio::Ratio1x1, AspectRatio::Ratio1x1.as_float()),
            (AspectRatio::Ratio21x9, AspectRatio::Ratio21x9.as_float()),
            (AspectRatio::Ratio9x16, AspectRatio::Ratio9x16.as_float()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transitions::TransitionType;

    #[test]
    fn test_simple_lower_third() {
        let result = Examples::simple_lower_third();
        assert!(result.is_ok());
    }

    #[test]
    fn test_weather_graphic_example() {
        let weather = Examples::weather_graphic_example();
        assert_eq!(weather.location, "New York");
        assert_eq!(weather.temperature, 72);
    }

    #[test]
    fn test_scoreboard_example() {
        let scoreboard = Examples::scoreboard_example();
        assert_eq!(scoreboard.home_score, 102);
        assert_eq!(scoreboard.away_score, 98);
    }

    #[test]
    fn test_ticker_example() {
        let ticker = Examples::ticker_example();
        assert!(ticker.text.contains("Breaking"));
    }

    #[test]
    fn test_bug_example() {
        let bug = Examples::bug_example();
        assert_eq!(bug.opacity, 0.6);
    }

    #[test]
    fn test_animation_example() {
        let clip = Examples::animation_example();
        assert!(clip.position.is_some());
        assert!(clip.opacity.is_some());
    }

    #[test]
    fn test_transition_example() {
        let transition = Examples::transition_example();
        assert!(matches!(
            transition.transition_type,
            TransitionType::Slide { .. }
        ));
    }

    #[test]
    fn test_effect_stack_example() {
        let stack = Examples::effect_stack_example();
        assert!(stack.drop_shadow.is_some());
        assert!(stack.glow.is_some());
        assert!(stack.blur.is_some());
    }

    #[test]
    fn test_path_drawing_example() {
        let path = Examples::path_drawing_example();
        assert!(!path.commands.is_empty());
    }

    #[test]
    fn test_rounded_rectangle_example() {
        let path = Examples::rounded_rectangle_example();
        assert!(!path.commands.is_empty());
    }

    #[test]
    fn test_bezier_curve_example() {
        let path = Examples::bezier_curve_example();
        assert!(!path.commands.is_empty());
    }

    #[test]
    fn test_gradient_background() {
        let result = Examples::gradient_background();
        assert!(result.is_ok());
    }

    #[test]
    fn test_radial_gradient_example() {
        let result = Examples::radial_gradient_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolution_presets() {
        let presets = Examples::resolution_presets();
        assert_eq!(presets.len(), 4);
    }

    #[test]
    fn test_framerate_examples() {
        let framerates = Examples::framerate_examples();
        assert_eq!(framerates.len(), 6);
    }

    #[test]
    fn test_aspect_ratio_examples() {
        let ratios = Examples::aspect_ratio_examples();
        assert_eq!(ratios.len(), 5);
    }
}
