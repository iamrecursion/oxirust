//! Broadcast graphics elements (lower thirds, tickers, bugs, scoreboards, etc.)

use crate::animation::{AnimationClip, Timeline, Transform};
use crate::color::Color;
use crate::primitives::{Point, Rect};
use crate::text::TextStyle;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Graphics element trait
pub trait GraphicsElement {
    /// Get bounding rectangle
    fn bounds(&self) -> Rect;

    /// Update element with delta time
    fn update(&mut self, delta: Duration);

    /// Is element visible
    fn is_visible(&self) -> bool;

    /// Set visibility
    fn set_visible(&mut self, visible: bool);

    /// Get transform
    fn transform(&self) -> &Transform;

    /// Get mutable transform
    fn transform_mut(&mut self) -> &mut Transform;
}

/// Lower third graphic (name, title overlay)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerThird {
    /// Position
    pub position: Point,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
    /// Background color
    pub background: Color,
    /// Main text (name)
    pub main_text: String,
    /// Main text style
    pub main_style: TextStyle,
    /// Secondary text (title)
    pub secondary_text: Option<String>,
    /// Secondary text style
    pub secondary_style: Option<TextStyle>,
    /// Corner radius
    pub corner_radius: f32,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
    /// Animation timeline
    pub timeline: Option<Timeline>,
    /// Animation clip
    pub animation: Option<AnimationClip>,
}

impl LowerThird {
    /// Create a new lower third
    #[must_use]
    pub fn new(
        position: Point,
        width: f32,
        height: f32,
        main_text: String,
        main_style: TextStyle,
    ) -> Self {
        Self {
            position,
            width,
            height,
            background: Color::new(0, 0, 0, 200),
            main_text,
            main_style,
            secondary_text: None,
            secondary_style: None,
            corner_radius: 5.0,
            transform: Transform::identity(),
            visible: true,
            timeline: None,
            animation: None,
        }
    }

    /// Set secondary text
    #[must_use]
    pub fn with_secondary(mut self, text: String, style: TextStyle) -> Self {
        self.secondary_text = Some(text);
        self.secondary_style = Some(style);
        self
    }

    /// Set background color
    #[must_use]
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Animate in (slide from left)
    pub fn animate_in(&mut self, duration: Duration) {
        let mut timeline = Timeline::new(duration);
        timeline.play();
        self.timeline = Some(timeline);
        // Animation implementation would be more complex
    }
}

impl GraphicsElement for LowerThird {
    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.width, self.height)
    }

    fn update(&mut self, delta: Duration) {
        if let Some(ref mut timeline) = self.timeline {
            timeline.update(delta);
            if let Some(ref animation) = self.animation {
                self.transform = animation.evaluate(timeline);
            }
        }
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Ticker/crawl (scrolling text at bottom)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Y position
    pub y: f32,
    /// Height
    pub height: f32,
    /// Background color
    pub background: Color,
    /// Text content
    pub text: String,
    /// Text style
    pub text_style: TextStyle,
    /// Scroll speed (pixels per second)
    pub scroll_speed: f32,
    /// Current scroll position
    pub scroll_position: f32,
    /// Screen width
    pub screen_width: f32,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl Ticker {
    /// Create a new ticker
    #[must_use]
    pub fn new(
        y: f32,
        height: f32,
        text: String,
        text_style: TextStyle,
        screen_width: f32,
    ) -> Self {
        Self {
            y,
            height,
            background: Color::new(0, 0, 0, 200),
            text,
            text_style,
            scroll_speed: 100.0,
            scroll_position: screen_width,
            screen_width,
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Set scroll speed
    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.scroll_speed = speed;
        self
    }

    /// Reset scroll position
    pub fn reset(&mut self) {
        self.scroll_position = self.screen_width;
    }
}

impl GraphicsElement for Ticker {
    fn bounds(&self) -> Rect {
        Rect::new(0.0, self.y, self.screen_width, self.height)
    }

    fn update(&mut self, delta: Duration) {
        if self.visible {
            self.scroll_position -= self.scroll_speed * delta.as_secs_f32();
            // Reset when fully scrolled off screen
            if self.scroll_position < -self.screen_width {
                self.scroll_position = self.screen_width;
            }
        }
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Bug (station logo/watermark)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bug {
    /// Position
    pub position: Point,
    /// Size
    pub size: (f32, f32),
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
    /// Image path or content
    pub image_path: Option<String>,
    /// Background color (if no image)
    pub background: Option<Color>,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl Bug {
    /// Create a new bug
    #[must_use]
    pub fn new(position: Point, size: (f32, f32)) -> Self {
        Self {
            position,
            size,
            opacity: 0.7,
            image_path: None,
            background: None,
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Set image
    #[must_use]
    pub fn with_image(mut self, path: String) -> Self {
        self.image_path = Some(path);
        self
    }

    /// Set opacity
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

impl GraphicsElement for Bug {
    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.size.0, self.size.1)
    }

    fn update(&mut self, _delta: Duration) {
        // Bugs typically don't animate
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Scoreboard for sports graphics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scoreboard {
    /// Position
    pub position: Point,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
    /// Home team name
    pub home_team: String,
    /// Away team name
    pub away_team: String,
    /// Home score
    pub home_score: u32,
    /// Away score
    pub away_score: u32,
    /// Period/quarter
    pub period: String,
    /// Time remaining
    pub time: String,
    /// Background color
    pub background: Color,
    /// Text style
    pub text_style: TextStyle,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl Scoreboard {
    /// Create a new scoreboard
    #[must_use]
    pub fn new(
        position: Point,
        width: f32,
        height: f32,
        home_team: String,
        away_team: String,
    ) -> Self {
        Self {
            position,
            width,
            height,
            home_team,
            away_team,
            home_score: 0,
            away_score: 0,
            period: "1".to_string(),
            time: "00:00".to_string(),
            background: Color::new(0, 0, 0, 220),
            text_style: TextStyle::default(),
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Update score
    pub fn set_score(&mut self, home: u32, away: u32) {
        self.home_score = home;
        self.away_score = away;
    }

    /// Update time
    pub fn set_time(&mut self, time: String) {
        self.time = time;
    }
}

impl GraphicsElement for Scoreboard {
    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.width, self.height)
    }

    fn update(&mut self, _delta: Duration) {
        // Scoreboards typically update via external data
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Breaking news banner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingNews {
    /// Y position
    pub y: f32,
    /// Height
    pub height: f32,
    /// Title (e.g., "BREAKING NEWS")
    pub title: String,
    /// News text
    pub text: String,
    /// Background color
    pub background: Color,
    /// Title color
    pub title_color: Color,
    /// Text color
    pub text_color: Color,
    /// Title style
    pub title_style: TextStyle,
    /// Text style
    pub text_style: TextStyle,
    /// Screen width
    pub screen_width: f32,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl BreakingNews {
    /// Create a new breaking news banner
    #[must_use]
    pub fn new(y: f32, height: f32, title: String, text: String, screen_width: f32) -> Self {
        Self {
            y,
            height,
            title,
            text,
            background: Color::rgb(204, 0, 0),
            title_color: Color::WHITE,
            text_color: Color::WHITE,
            title_style: TextStyle::default(),
            text_style: TextStyle::default(),
            screen_width,
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Set colors
    #[must_use]
    pub fn with_colors(mut self, background: Color, title: Color, text: Color) -> Self {
        self.background = background;
        self.title_color = title;
        self.text_color = text;
        self
    }
}

impl GraphicsElement for BreakingNews {
    fn bounds(&self) -> Rect {
        Rect::new(0.0, self.y, self.screen_width, self.height)
    }

    fn update(&mut self, _delta: Duration) {
        // Breaking news typically doesn't animate (or slides in once)
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Social media feed overlay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialFeed {
    /// Position
    pub position: Point,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
    /// Platform (Twitter, etc.)
    pub platform: String,
    /// Username
    pub username: String,
    /// Post text
    pub text: String,
    /// Avatar path
    pub avatar_path: Option<String>,
    /// Background color
    pub background: Color,
    /// Text style
    pub text_style: TextStyle,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl SocialFeed {
    /// Create a new social feed element
    #[must_use]
    pub fn new(
        position: Point,
        width: f32,
        height: f32,
        platform: String,
        username: String,
        text: String,
    ) -> Self {
        Self {
            position,
            width,
            height,
            platform,
            username,
            text,
            avatar_path: None,
            background: Color::new(0, 0, 0, 200),
            text_style: TextStyle::default(),
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Set avatar
    #[must_use]
    pub fn with_avatar(mut self, path: String) -> Self {
        self.avatar_path = Some(path);
        self
    }
}

impl GraphicsElement for SocialFeed {
    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.width, self.height)
    }

    fn update(&mut self, _delta: Duration) {
        // Social feeds typically update via external data
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

/// Weather graphic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherGraphic {
    /// Position
    pub position: Point,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
    /// Location
    pub location: String,
    /// Temperature
    pub temperature: i32,
    /// Condition (sunny, cloudy, etc.)
    pub condition: String,
    /// High temperature
    pub high: Option<i32>,
    /// Low temperature
    pub low: Option<i32>,
    /// Background color
    pub background: Color,
    /// Text style
    pub text_style: TextStyle,
    /// Transform
    pub transform: Transform,
    /// Visible
    pub visible: bool,
}

impl WeatherGraphic {
    /// Create a new weather graphic
    #[must_use]
    pub fn new(
        position: Point,
        width: f32,
        height: f32,
        location: String,
        temperature: i32,
        condition: String,
    ) -> Self {
        Self {
            position,
            width,
            height,
            location,
            temperature,
            condition,
            high: None,
            low: None,
            background: Color::new(0, 120, 200, 200),
            text_style: TextStyle::default(),
            transform: Transform::identity(),
            visible: true,
        }
    }

    /// Set high/low temperatures
    #[must_use]
    pub fn with_high_low(mut self, high: i32, low: i32) -> Self {
        self.high = Some(high);
        self.low = Some(low);
        self
    }
}

impl GraphicsElement for WeatherGraphic {
    fn bounds(&self) -> Rect {
        Rect::new(self.position.x, self.position.y, self.width, self.height)
    }

    fn update(&mut self, _delta: Duration) {
        // Weather graphics update via external data
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_third() {
        let style = TextStyle::default();
        let lt = LowerThird::new(
            Point::new(50.0, 900.0),
            800.0,
            120.0,
            "John Doe".to_string(),
            style,
        );

        assert_eq!(lt.bounds(), Rect::new(50.0, 900.0, 800.0, 120.0));
        assert!(lt.is_visible());
    }

    #[test]
    fn test_ticker() {
        let style = TextStyle::default();
        let ticker = Ticker::new(1000.0, 50.0, "Breaking news...".to_string(), style, 1920.0);

        assert_eq!(ticker.scroll_position, 1920.0);
        assert!(ticker.is_visible());
    }

    #[test]
    fn test_ticker_update() {
        let style = TextStyle::default();
        let mut ticker = Ticker::new(1000.0, 50.0, "Test".to_string(), style, 1920.0);

        let initial_pos = ticker.scroll_position;
        ticker.update(Duration::from_secs(1));
        assert!(ticker.scroll_position < initial_pos);
    }

    #[test]
    fn test_bug() {
        let bug = Bug::new(Point::new(1800.0, 50.0), (100.0, 100.0));
        assert_eq!(bug.bounds(), Rect::new(1800.0, 50.0, 100.0, 100.0));
        assert_eq!(bug.opacity, 0.7);
    }

    #[test]
    fn test_scoreboard() {
        let mut scoreboard = Scoreboard::new(
            Point::new(100.0, 50.0),
            400.0,
            100.0,
            "Home".to_string(),
            "Away".to_string(),
        );

        scoreboard.set_score(21, 14);
        assert_eq!(scoreboard.home_score, 21);
        assert_eq!(scoreboard.away_score, 14);
    }

    #[test]
    fn test_breaking_news() {
        let news = BreakingNews::new(
            100.0,
            80.0,
            "BREAKING NEWS".to_string(),
            "Important update".to_string(),
            1920.0,
        );

        assert_eq!(news.title, "BREAKING NEWS");
        assert_eq!(news.text, "Important update");
    }

    #[test]
    fn test_social_feed() {
        let feed = SocialFeed::new(
            Point::new(50.0, 500.0),
            400.0,
            200.0,
            "Twitter".to_string(),
            "@user".to_string(),
            "Tweet text".to_string(),
        );

        assert_eq!(feed.platform, "Twitter");
        assert_eq!(feed.username, "@user");
    }

    #[test]
    fn test_weather_graphic() {
        let weather = WeatherGraphic::new(
            Point::new(1500.0, 50.0),
            300.0,
            200.0,
            "New York".to_string(),
            72,
            "Sunny".to_string(),
        )
        .with_high_low(75, 65);

        assert_eq!(weather.temperature, 72);
        assert_eq!(weather.high, Some(75));
        assert_eq!(weather.low, Some(65));
    }
}
