//! Scrolling ticker management.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Style of ticker animation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TickerStyle {
    /// Scrolls from right to left.
    ScrollLeft,
    /// Scrolls from left to right.
    ScrollRight,
    /// Crawl along the bottom.
    Crawl,
    /// Static display (no scrolling).
    Static,
}

/// Scrolling ticker/crawl configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Unique identifier.
    pub id: String,

    /// Ticker text content.
    pub text: String,

    /// Style of animation.
    pub style: TickerStyle,

    /// Y position (pixels from top).
    pub y: i32,

    /// Scroll speed (pixels per second).
    pub speed: f32,

    /// Font size.
    pub font_size: u32,

    /// Font color (RGBA).
    pub color: (u8, u8, u8, u8),

    /// Background color (RGBA).
    pub background_color: Option<(u8, u8, u8, u8)>,

    /// Whether the ticker is currently visible.
    pub visible: bool,

    /// Duration to show (None = infinite).
    pub duration: Option<Duration>,

    /// Whether to loop the text.
    pub looping: bool,
}

impl Ticker {
    /// Creates a new ticker.
    #[must_use]
    pub fn new<S: Into<String>>(text: S, style: TickerStyle, y: i32) -> Self {
        Self {
            id: generate_id(),
            text: text.into(),
            style,
            y,
            speed: 100.0,
            font_size: 24,
            color: (255, 255, 255, 255),
            background_color: Some((0, 0, 0, 180)),
            visible: false,
            duration: None,
            looping: true,
        }
    }

    /// Sets the scroll speed.
    #[must_use]
    pub const fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Sets the font size.
    #[must_use]
    pub const fn with_font_size(mut self, size: u32) -> Self {
        self.font_size = size;
        self
    }

    /// Sets the text color.
    #[must_use]
    pub const fn with_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.color = (r, g, b, a);
        self
    }

    /// Sets the background color.
    #[must_use]
    pub const fn with_background(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.background_color = Some((r, g, b, a));
        self
    }

    /// Removes the background.
    #[must_use]
    pub const fn without_background(mut self) -> Self {
        self.background_color = None;
        self
    }

    /// Sets the duration.
    #[must_use]
    pub const fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Sets whether to loop.
    #[must_use]
    pub const fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Shows the ticker.
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hides the ticker.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Updates the ticker text.
    pub fn set_text<S: Into<String>>(&mut self, text: S) {
        self.text = text.into();
    }
}

/// Manager for tickers.
#[derive(Debug, Default)]
pub struct TickerManager {
    tickers: Vec<Ticker>,
}

impl TickerManager {
    /// Creates a new ticker manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a ticker.
    pub fn add_ticker(&mut self, ticker: Ticker) {
        self.tickers.push(ticker);
    }

    /// Removes a ticker by ID.
    pub fn remove_ticker(&mut self, ticker_id: &str) {
        self.tickers.retain(|t| t.id != ticker_id);
    }

    /// Shows a ticker by ID.
    pub fn show_ticker(&mut self, ticker_id: &str) {
        if let Some(ticker) = self.tickers.iter_mut().find(|t| t.id == ticker_id) {
            ticker.show();
        }
    }

    /// Hides a ticker by ID.
    pub fn hide_ticker(&mut self, ticker_id: &str) {
        if let Some(ticker) = self.tickers.iter_mut().find(|t| t.id == ticker_id) {
            ticker.hide();
        }
    }

    /// Updates the text of a ticker.
    pub fn update_ticker_text<S: Into<String>>(&mut self, ticker_id: &str, text: S) {
        if let Some(ticker) = self.tickers.iter_mut().find(|t| t.id == ticker_id) {
            ticker.set_text(text);
        }
    }

    /// Gets all visible tickers.
    #[must_use]
    pub fn get_visible_tickers(&self) -> Vec<&Ticker> {
        self.tickers.iter().filter(|t| t.visible).collect()
    }

    /// Returns the number of tickers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tickers.len()
    }

    /// Returns true if there are no tickers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tickers.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("ticker_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticker() {
        let mut ticker = Ticker::new("Breaking News", TickerStyle::ScrollLeft, 1000)
            .with_speed(150.0)
            .with_font_size(28);

        assert!(!ticker.visible);
        ticker.show();
        assert!(ticker.visible);
    }

    #[test]
    fn test_ticker_manager() {
        let mut manager = TickerManager::new();
        let ticker = Ticker::new("News", TickerStyle::Crawl, 1000);
        let ticker_id = ticker.id.clone();

        manager.add_ticker(ticker);
        assert_eq!(manager.len(), 1);

        manager.show_ticker(&ticker_id);
        assert_eq!(manager.get_visible_tickers().len(), 1);

        manager.update_ticker_text(&ticker_id, "Updated News");
    }
}
