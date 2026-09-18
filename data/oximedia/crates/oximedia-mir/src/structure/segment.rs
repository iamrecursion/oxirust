//! Structural segmentation using self-similarity.

use crate::structure::labels::SectionLabeler;
use crate::structure::similarity::SimilarityMatrix;
use crate::types::{Segment, StructureResult};
use crate::utils::stft;
use crate::MirResult;

/// Structure analyzer.
pub struct StructureAnalyzer {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl StructureAnalyzer {
    /// Create a new structure analyzer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Analyze musical structure.
    ///
    /// # Errors
    ///
    /// Returns error if structure analysis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, signal: &[f32]) -> MirResult<StructureResult> {
        // Compute features for similarity
        let features = self.compute_features(signal)?;

        // Compute self-similarity matrix
        let sim_matrix = SimilarityMatrix::new();
        let (similarity_matrix, matrix_size) = sim_matrix.compute(&features)?;

        // Find segment boundaries
        let boundaries = self.find_boundaries(&similarity_matrix, matrix_size)?;

        // Label sections
        let labeler = SectionLabeler::new();
        let segments = labeler.label_segments(&boundaries, signal.len() as f32 / self.sample_rate);

        // Compute structural complexity
        let complexity = self.compute_complexity(&segments);

        Ok(StructureResult {
            segments,
            similarity_matrix,
            matrix_size,
            complexity,
        })
    }

    /// Compute feature vectors for similarity analysis.
    fn compute_features(&self, signal: &[f32]) -> MirResult<Vec<Vec<f32>>> {
        let frames = stft(signal, self.window_size, self.hop_size)?;

        let features: Vec<Vec<f32>> = frames
            .iter()
            .map(|frame| {
                let mag = crate::utils::magnitude_spectrum(frame);
                // Downsample magnitude spectrum for efficiency
                mag.iter().step_by(4).copied().collect()
            })
            .collect();

        Ok(features)
    }

    /// Find segment boundaries from similarity matrix.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::unnecessary_wraps)]
    fn find_boundaries(
        &self,
        _similarity_matrix: &[f32],
        matrix_size: usize,
    ) -> MirResult<Vec<f32>> {
        // Simplified boundary detection
        // In practice, would use novelty curve from similarity matrix
        let segment_duration = 10.0; // 10 seconds per segment
        let num_segments = (matrix_size as f32 * self.hop_size as f32
            / self.sample_rate
            / segment_duration)
            .ceil() as usize;

        let boundaries: Vec<f32> = (0..=num_segments)
            .map(|i| i as f32 * segment_duration)
            .collect();

        Ok(boundaries)
    }

    /// Compute structural complexity.
    fn compute_complexity(&self, segments: &[Segment]) -> f32 {
        if segments.len() <= 1 {
            return 0.0;
        }

        // Count unique section labels
        let mut unique_labels = std::collections::HashSet::new();
        for segment in segments {
            unique_labels.insert(&segment.label);
        }

        // Complexity based on number of unique sections
        (unique_labels.len() as f32 / segments.len() as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_analyzer_creation() {
        let analyzer = StructureAnalyzer::new(44100.0, 2048, 512);
        assert_eq!(analyzer.sample_rate, 44100.0);
    }
}
