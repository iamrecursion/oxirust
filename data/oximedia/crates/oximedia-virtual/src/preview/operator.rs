//! Operator preview display

use super::PreviewConfig;
use crate::Result;

/// Operator preview
pub struct OperatorPreview {
    #[allow(dead_code)]
    config: PreviewConfig,
}

impl OperatorPreview {
    /// Create new operator preview
    #[must_use]
    pub fn new(config: PreviewConfig) -> Self {
        Self { config }
    }

    /// Generate preview frame
    pub fn generate(&mut self, source: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(source.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_preview() {
        let config = PreviewConfig::default();
        let _preview = OperatorPreview::new(config);
    }
}
