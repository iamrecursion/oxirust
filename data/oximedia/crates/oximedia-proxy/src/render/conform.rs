//! Render conforming for final output.

use crate::Result;

/// Render conform engine.
pub struct RenderConform;

impl RenderConform {
    /// Create a new render conform engine.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Conform a render to use original media.
    pub fn conform(
        &self,
        _timeline: &std::path::Path,
        _output: &std::path::Path,
    ) -> Result<ConformResult> {
        // Placeholder: would conform timeline to originals
        Ok(ConformResult {
            output_path: std::path::PathBuf::new(),
            clips_conformed: 0,
        })
    }
}

impl Default for RenderConform {
    fn default() -> Self {
        Self::new()
    }
}

/// Render conform result.
#[derive(Debug, Clone)]
pub struct ConformResult {
    /// Output file path.
    pub output_path: std::path::PathBuf,

    /// Number of clips conformed.
    pub clips_conformed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_conform() {
        let conformer = RenderConform::new();
        let result = conformer.conform(
            std::path::Path::new("timeline.xml"),
            std::path::Path::new("output.mov"),
        );
        assert!(result.is_ok());
    }
}
