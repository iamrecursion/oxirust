//! LED wall as green screen alternative

use super::GreenScreenConfig;
use crate::Result;

/// LED green screen alternative
pub struct LedGreenScreen {
    #[allow(dead_code)]
    config: GreenScreenConfig,
}

impl LedGreenScreen {
    /// Create new LED green screen
    #[must_use]
    pub fn new(config: GreenScreenConfig) -> Self {
        Self { config }
    }

    /// Process frame
    pub fn process(&mut self, frame: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(frame.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_led_green_screen() {
        let config = GreenScreenConfig::default();
        let _screen = LedGreenScreen::new(config);
    }
}
