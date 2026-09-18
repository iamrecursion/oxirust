//! Pre-configured restoration presets for common scenarios.

use crate::azimuth::{AzimuthCorrector, AzimuthCorrectorConfig};
use crate::click::{ClickDetector, ClickDetectorConfig, ClickRemover};
use crate::clip::{BasicDeclipper, ClipDetector, ClipDetectorConfig, DeclipConfig};
use crate::crackle::{CrackleDetector, CrackleRemover};
use crate::dc::DcRemover;
use crate::hiss::{HissRemover, HissRemoverConfig};
use crate::hum::HumRemover;
use crate::noise::{
    NoiseGate, NoiseGateConfig, NoiseProfile, SpectralSubtraction, SpectralSubtractionConfig,
};
use crate::phase::PhaseCorrector;
use crate::utils::interpolation::InterpolationMethod;
use crate::wow::WowFlutterCorrector;
use crate::RestorationStep;

/// Vinyl restoration preset.
///
/// Removes clicks, crackle, and hum typical of vinyl records.
#[derive(Debug, Clone)]
pub struct VinylRestoration {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Enable click removal.
    pub click_removal: bool,
    /// Enable crackle removal.
    pub crackle_removal: bool,
    /// Enable hum removal.
    pub hum_removal: bool,
}

impl VinylRestoration {
    /// Create a new vinyl restoration preset.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            click_removal: true,
            crackle_removal: true,
            hum_removal: true,
        }
    }
}

impl Default for VinylRestoration {
    fn default() -> Self {
        Self::new(44100)
    }
}

impl From<VinylRestoration> for Vec<RestorationStep> {
    fn from(preset: VinylRestoration) -> Self {
        let mut steps = Vec::new();

        // DC offset removal
        steps.push(RestorationStep::DcRemoval(DcRemover::new(
            10.0,
            preset.sample_rate,
        )));

        // Click removal
        if preset.click_removal {
            steps.push(RestorationStep::ClickRemoval {
                detector: ClickDetector::new(ClickDetectorConfig::default()),
                remover: ClickRemover::new(InterpolationMethod::Cubic, 2),
            });
        }

        // Crackle removal
        if preset.crackle_removal {
            steps.push(RestorationStep::CrackleRemoval {
                detector: CrackleDetector::new(0.3, 1),
                remover: CrackleRemover::new(5),
            });
        }

        // Hum removal (50/60 Hz)
        if preset.hum_removal {
            steps.push(RestorationStep::HumRemoval(HumRemover::new_standard(
                50.0,
                preset.sample_rate,
                5,
                10.0,
            )));
            steps.push(RestorationStep::HumRemoval(HumRemover::new_standard(
                60.0,
                preset.sample_rate,
                5,
                10.0,
            )));
        }

        steps
    }
}

/// Tape restoration preset.
///
/// Corrects azimuth, removes wow/flutter and hiss.
#[derive(Debug, Clone)]
pub struct TapeRestoration {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Enable azimuth correction.
    pub azimuth_correction: bool,
    /// Enable wow/flutter correction.
    pub wow_flutter_correction: bool,
    /// Enable hiss removal.
    pub hiss_removal: bool,
}

impl TapeRestoration {
    /// Create a new tape restoration preset.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            azimuth_correction: true,
            wow_flutter_correction: true,
            hiss_removal: true,
        }
    }
}

impl Default for TapeRestoration {
    fn default() -> Self {
        Self::new(44100)
    }
}

impl From<TapeRestoration> for Vec<RestorationStep> {
    fn from(preset: TapeRestoration) -> Self {
        let mut steps = Vec::new();

        // DC offset removal
        steps.push(RestorationStep::DcRemoval(DcRemover::new(
            10.0,
            preset.sample_rate,
        )));

        // Azimuth correction
        if preset.azimuth_correction {
            steps.push(RestorationStep::AzimuthCorrection(AzimuthCorrector::new(
                AzimuthCorrectorConfig::default(),
            )));
        }

        // Wow/flutter correction
        if preset.wow_flutter_correction {
            steps.push(RestorationStep::WowFlutterCorrection(
                WowFlutterCorrector::new(4),
            ));
        }

        // Hiss removal
        if preset.hiss_removal {
            steps.push(RestorationStep::HissRemoval(HissRemover::new(
                HissRemoverConfig::default(),
                2048,
                1024,
            )));
        }

        steps
    }
}

/// Broadcast cleanup preset.
///
/// Removes clipping, noise, and DC offset for broadcast material.
#[derive(Debug, Clone)]
pub struct BroadcastCleanup {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Noise profile for reduction.
    pub noise_profile: Option<NoiseProfile>,
}

impl BroadcastCleanup {
    /// Create a new broadcast cleanup preset.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            noise_profile: None,
        }
    }

    /// Set noise profile for reduction.
    pub fn with_noise_profile(mut self, profile: NoiseProfile) -> Self {
        self.noise_profile = Some(profile);
        self
    }
}

impl Default for BroadcastCleanup {
    fn default() -> Self {
        Self::new(44100)
    }
}

impl From<BroadcastCleanup> for Vec<RestorationStep> {
    fn from(preset: BroadcastCleanup) -> Self {
        let mut steps = Vec::new();

        // DC offset removal
        steps.push(RestorationStep::DcRemoval(DcRemover::new(
            10.0,
            preset.sample_rate,
        )));

        // Declipping
        steps.push(RestorationStep::Declipping {
            detector: ClipDetector::new(ClipDetectorConfig::default()),
            declipper: BasicDeclipper::new(DeclipConfig::default()),
        });

        // Noise reduction if profile available
        if let Some(profile) = preset.noise_profile {
            steps.push(RestorationStep::NoiseReduction(SpectralSubtraction::new(
                profile,
                1024,
                SpectralSubtractionConfig::default(),
            )));
        } else {
            // Use noise gate as fallback
            steps.push(RestorationStep::NoiseGate(NoiseGate::new(
                NoiseGateConfig::default(),
            )));
        }

        // Phase correction for stereo
        steps.push(RestorationStep::PhaseCorrection(PhaseCorrector::new(0.5)));

        steps
    }
}

/// Archival restoration preset.
///
/// Full restoration chain for preservation work.
#[derive(Debug, Clone)]
pub struct ArchivalRestoration {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Noise profile for reduction.
    pub noise_profile: Option<NoiseProfile>,
}

impl ArchivalRestoration {
    /// Create a new archival restoration preset.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            noise_profile: None,
        }
    }

    /// Set noise profile for reduction.
    pub fn with_noise_profile(mut self, profile: NoiseProfile) -> Self {
        self.noise_profile = Some(profile);
        self
    }
}

impl Default for ArchivalRestoration {
    fn default() -> Self {
        Self::new(44100)
    }
}

impl From<ArchivalRestoration> for Vec<RestorationStep> {
    #[allow(clippy::vec_init_then_push)]
    fn from(preset: ArchivalRestoration) -> Self {
        let mut steps = Vec::new();

        // DC offset removal
        steps.push(RestorationStep::DcRemoval(DcRemover::new(
            10.0,
            preset.sample_rate,
        )));

        // Azimuth correction
        steps.push(RestorationStep::AzimuthCorrection(AzimuthCorrector::new(
            AzimuthCorrectorConfig::default(),
        )));

        // Click removal
        steps.push(RestorationStep::ClickRemoval {
            detector: ClickDetector::new(ClickDetectorConfig::default()),
            remover: ClickRemover::new(InterpolationMethod::Cubic, 2),
        });

        // Crackle removal
        steps.push(RestorationStep::CrackleRemoval {
            detector: CrackleDetector::new(0.3, 1),
            remover: CrackleRemover::new(5),
        });

        // Declipping
        steps.push(RestorationStep::Declipping {
            detector: ClipDetector::new(ClipDetectorConfig::default()),
            declipper: BasicDeclipper::new(DeclipConfig::default()),
        });

        // Hum removal
        steps.push(RestorationStep::HumRemoval(HumRemover::new_standard(
            50.0,
            preset.sample_rate,
            5,
            10.0,
        )));
        steps.push(RestorationStep::HumRemoval(HumRemover::new_standard(
            60.0,
            preset.sample_rate,
            5,
            10.0,
        )));

        // Noise reduction
        if let Some(profile) = preset.noise_profile {
            steps.push(RestorationStep::NoiseReduction(SpectralSubtraction::new(
                profile,
                1024,
                SpectralSubtractionConfig::default(),
            )));
        }

        // Hiss removal
        steps.push(RestorationStep::HissRemoval(HissRemover::new(
            HissRemoverConfig::default(),
            2048,
            1024,
        )));

        // Wow/flutter correction
        steps.push(RestorationStep::WowFlutterCorrection(
            WowFlutterCorrector::new(4),
        ));

        // Phase correction
        steps.push(RestorationStep::PhaseCorrection(PhaseCorrector::new(0.5)));

        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vinyl_restoration() {
        let preset = VinylRestoration::default();
        let steps: Vec<RestorationStep> = preset.into();
        assert!(!steps.is_empty());
    }

    #[test]
    fn test_tape_restoration() {
        let preset = TapeRestoration::default();
        let steps: Vec<RestorationStep> = preset.into();
        assert!(!steps.is_empty());
    }

    #[test]
    fn test_broadcast_cleanup() {
        let preset = BroadcastCleanup::default();
        let steps: Vec<RestorationStep> = preset.into();
        assert!(!steps.is_empty());
    }

    #[test]
    fn test_archival_restoration() {
        let preset = ArchivalRestoration::default();
        let steps: Vec<RestorationStep> = preset.into();
        assert!(steps.len() > 5); // Should have many steps
    }

    #[test]
    fn test_with_noise_profile() {
        let profile = NoiseProfile::new(2048);
        let preset = BroadcastCleanup::new(44100).with_noise_profile(profile);
        assert!(preset.noise_profile.is_some());
    }
}
