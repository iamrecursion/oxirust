//! EAS crawl (scrolling text) generation.

use serde::{Deserialize, Serialize};
use tracing::info;

/// Crawl configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlConfig {
    /// Crawl text
    pub text: String,
    /// Scroll speed (pixels per second)
    pub speed: f32,
    /// Font size
    pub font_size: u32,
    /// Background color (RGBA)
    pub background_color: (u8, u8, u8, u8),
    /// Text color (RGBA)
    pub text_color: (u8, u8, u8, u8),
    /// Vertical position (0.0 = top, 1.0 = bottom)
    pub vertical_position: f32,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            text: String::new(),
            speed: 100.0,
            font_size: 24,
            background_color: (0, 0, 0, 200),
            text_color: (255, 255, 255, 255),
            vertical_position: 0.9,
        }
    }
}

/// Crawl generator for EAS alerts.
pub struct CrawlGenerator {
    config: CrawlConfig,
    position: f32,
}

impl CrawlGenerator {
    /// Create a new crawl generator.
    pub fn new(config: CrawlConfig) -> Self {
        info!("Creating crawl generator");

        Self {
            config,
            position: 0.0,
        }
    }

    /// Update crawl position.
    pub fn update(&mut self, delta_time: f32) {
        self.position += self.config.speed * delta_time;
    }

    /// Reset crawl to start.
    pub fn reset(&mut self) {
        self.position = 0.0;
    }

    /// Get current position.
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Set crawl text.
    pub fn set_text(&mut self, text: String) {
        info!("Setting crawl text: {}", text);
        self.config.text = text;
    }

    /// Get crawl text.
    pub fn text(&self) -> &str {
        &self.config.text
    }

    /// Set scroll speed.
    pub fn set_speed(&mut self, speed: f32) {
        self.config.speed = speed;
    }

    /// Generate crawl for rendering.
    pub fn generate(&self) -> CrawlData {
        CrawlData {
            text: self.config.text.clone(),
            position: self.position,
            font_size: self.config.font_size,
            background_color: self.config.background_color,
            text_color: self.config.text_color,
            vertical_position: self.config.vertical_position,
        }
    }
}

/// Crawl data for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlData {
    /// Crawl text
    pub text: String,
    /// Horizontal position
    pub position: f32,
    /// Font size
    pub font_size: u32,
    /// Background color (RGBA)
    pub background_color: (u8, u8, u8, u8),
    /// Text color (RGBA)
    pub text_color: (u8, u8, u8, u8),
    /// Vertical position
    pub vertical_position: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crawl_generator() {
        let config = CrawlConfig {
            text: "Emergency Alert".to_string(),
            speed: 100.0,
            ..Default::default()
        };

        let mut generator = CrawlGenerator::new(config);
        assert_eq!(generator.position(), 0.0);

        generator.update(1.0); // 1 second
        assert_eq!(generator.position(), 100.0);

        generator.reset();
        assert_eq!(generator.position(), 0.0);
    }

    #[test]
    fn test_crawl_text() {
        let config = CrawlConfig::default();
        let mut generator = CrawlGenerator::new(config);

        generator.set_text("Test Message".to_string());
        assert_eq!(generator.text(), "Test Message");
    }

    #[test]
    fn test_generate_crawl_data() {
        let config = CrawlConfig {
            text: "Test".to_string(),
            ..Default::default()
        };

        let generator = CrawlGenerator::new(config);
        let data = generator.generate();

        assert_eq!(data.text, "Test");
        assert_eq!(data.position, 0.0);
    }
}
