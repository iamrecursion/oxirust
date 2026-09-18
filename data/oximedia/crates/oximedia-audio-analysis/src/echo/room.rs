//! Room acoustics analysis.

use crate::{AnalysisConfig, Result};

/// Room analyzer for analyzing room acoustics.
pub struct RoomAnalyzer {
    #[allow(dead_code)]
    config: AnalysisConfig,
}

impl RoomAnalyzer {
    /// Create a new room analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze room characteristics.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<RoomCharacteristics> {
        // Estimate room size from reverb time
        let rt60 = super::rt60::measure_rt60(samples, sample_rate);

        // Estimate room size (larger rooms have longer RT60)
        let room_size = if rt60 > 2.0 {
            RoomSize::Large
        } else if rt60 > 0.8 {
            RoomSize::Medium
        } else if rt60 > 0.3 {
            RoomSize::Small
        } else {
            RoomSize::Anechoic
        };

        // Estimate absorption (inverse of RT60)
        let absorption = (1.0 / (rt60 + 0.1)).min(1.0);

        Ok(RoomCharacteristics {
            rt60,
            room_size,
            absorption,
        })
    }
}

/// Room size classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomSize {
    /// Anechoic (no reflections)
    Anechoic,
    /// Small room
    Small,
    /// Medium room
    Medium,
    /// Large room/hall
    Large,
}

/// Room acoustic characteristics.
#[derive(Debug, Clone)]
pub struct RoomCharacteristics {
    /// RT60 reverberation time in seconds
    pub rt60: f32,
    /// Estimated room size
    pub room_size: RoomSize,
    /// Absorption coefficient (0-1)
    pub absorption: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = RoomAnalyzer::new(config);

        let samples = vec![0.1; 44100];
        let result = analyzer.analyze(&samples, 44100.0);
        assert!(result.is_ok());
    }
}
