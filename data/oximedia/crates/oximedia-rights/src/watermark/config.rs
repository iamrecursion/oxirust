//! Watermark configuration

use serde::{Deserialize, Serialize};

/// Watermark type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatermarkType {
    /// Visible watermark (text/logo)
    Visible,
    /// Invisible/digital watermark
    Invisible,
}

/// Watermark position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatermarkPosition {
    /// Top left
    TopLeft,
    /// Top right
    TopRight,
    /// Bottom left
    BottomLeft,
    /// Bottom right
    BottomRight,
    /// Center
    Center,
}

/// Watermark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkConfig {
    /// Watermark type
    pub watermark_type: WatermarkType,
    /// Text content (for visible watermarks)
    pub text: Option<String>,
    /// Position
    pub position: WatermarkPosition,
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
    /// Font size (for text watermarks)
    pub font_size: Option<u32>,
}

impl WatermarkConfig {
    /// Create a new visible text watermark
    pub fn visible_text(text: impl Into<String>) -> Self {
        Self {
            watermark_type: WatermarkType::Visible,
            text: Some(text.into()),
            position: WatermarkPosition::BottomRight,
            opacity: 0.5,
            font_size: Some(24),
        }
    }

    /// Create a new invisible watermark
    pub fn invisible() -> Self {
        Self {
            watermark_type: WatermarkType::Invisible,
            text: None,
            position: WatermarkPosition::Center,
            opacity: 1.0,
            font_size: None,
        }
    }

    /// Set position
    pub fn with_position(mut self, position: WatermarkPosition) -> Self {
        self.position = position;
        self
    }

    /// Set opacity
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visible_watermark_config() {
        let config = WatermarkConfig::visible_text("Copyright 2024")
            .with_position(WatermarkPosition::TopRight)
            .with_opacity(0.7);

        assert_eq!(config.text, Some("Copyright 2024".to_string()));
        assert_eq!(config.opacity, 0.7);
    }

    #[test]
    fn test_invisible_watermark_config() {
        let config = WatermarkConfig::invisible();
        assert!(config.text.is_none());
    }
}
