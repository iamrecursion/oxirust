//! Section labeling for musical structure.

use crate::types::Segment;

/// Section labeler.
pub struct SectionLabeler;

impl SectionLabeler {
    /// Create a new section labeler.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Label segments with section names.
    #[must_use]
    pub fn label_segments(&self, boundaries: &[f32], duration: f32) -> Vec<Segment> {
        let mut segments = Vec::new();

        for i in 0..boundaries.len().saturating_sub(1) {
            let start = boundaries[i];
            let end = boundaries[i + 1].min(duration);

            if start >= end {
                continue;
            }

            let label = self.infer_label(i, boundaries.len());
            let confidence = 0.7; // Simplified confidence

            segments.push(Segment {
                start,
                end,
                label,
                confidence,
            });
        }

        segments
    }

    /// Infer section label based on position.
    fn infer_label(&self, index: usize, total: usize) -> String {
        if index == 0 {
            "intro".to_string()
        } else if index == total - 2 {
            "outro".to_string()
        } else if index % 2 == 0 {
            format!("verse_{}", index / 2)
        } else {
            "chorus".to_string()
        }
    }
}

impl Default for SectionLabeler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_labeler_creation() {
        let _labeler = SectionLabeler::new();
    }

    #[test]
    fn test_label_segments() {
        let labeler = SectionLabeler::new();
        let boundaries = vec![0.0, 10.0, 20.0, 30.0];
        let segments = labeler.label_segments(&boundaries, 40.0);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].label, "intro");
    }
}
