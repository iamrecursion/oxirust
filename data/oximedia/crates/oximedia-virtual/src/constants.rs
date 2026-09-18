//! Constants and configuration values for virtual production
//!
//! Provides commonly used constants, default values, and configuration presets.

use std::time::Duration;

/// Default target frame rate (60 FPS)
pub const DEFAULT_TARGET_FPS: f64 = 60.0;

/// Maximum target frame rate (240 FPS)
pub const MAX_TARGET_FPS: f64 = 240.0;

/// Minimum target frame rate (24 FPS)
pub const MIN_TARGET_FPS: f64 = 24.0;

/// Default sync accuracy in milliseconds
pub const DEFAULT_SYNC_ACCURACY_MS: f64 = 0.5;

/// Maximum acceptable latency in milliseconds
pub const MAX_LATENCY_MS: u64 = 20;

/// Default camera tracking rate (120 Hz)
pub const DEFAULT_TRACKING_RATE_HZ: f64 = 120.0;

/// Default LED wall brightness (nits)
pub const DEFAULT_LED_BRIGHTNESS_NITS: u32 = 1000;

/// Maximum LED wall brightness (nits)
pub const MAX_LED_BRIGHTNESS_NITS: u32 = 5000;

/// Default color bit depth
pub const DEFAULT_COLOR_BIT_DEPTH: u8 = 10;

/// Standard frame rates
pub mod frame_rates {
    /// 23.976 FPS (film)
    pub const FILM_23_976: f64 = 23.976;
    /// 24 FPS (film)
    pub const FILM_24: f64 = 24.0;
    /// 25 FPS (PAL)
    pub const PAL_25: f64 = 25.0;
    /// 29.97 FPS (NTSC)
    pub const NTSC_29_97: f64 = 29.97;
    /// 30 FPS
    pub const FPS_30: f64 = 30.0;
    /// 50 FPS (PAL high frame rate)
    pub const PAL_50: f64 = 50.0;
    /// 59.94 FPS (NTSC high frame rate)
    pub const NTSC_59_94: f64 = 59.94;
    /// 60 FPS
    pub const FPS_60: f64 = 60.0;
    /// 120 FPS (high frame rate)
    pub const FPS_120: f64 = 120.0;
    /// 240 FPS (high speed)
    pub const FPS_240: f64 = 240.0;
}

/// Standard resolutions
pub mod resolutions {
    /// HD resolution (1920x1080)
    pub const HD_1080P: (usize, usize) = (1920, 1080);
    /// 4K UHD resolution (3840x2160)
    pub const UHD_4K: (usize, usize) = (3840, 2160);
    /// 8K UHD resolution (7680x4320)
    pub const UHD_8K: (usize, usize) = (7680, 4320);
    /// 2K DCI resolution (2048x1080)
    pub const DCI_2K: (usize, usize) = (2048, 1080);
    /// 4K DCI resolution (4096x2160)
    pub const DCI_4K: (usize, usize) = (4096, 2160);
}

/// Color space constants
pub mod color_spaces {
    /// sRGB gamma
    pub const SRGB_GAMMA: f32 = 2.2;
    /// Rec.709 gamma
    pub const REC709_GAMMA: f32 = 2.4;
    /// Rec.2020 gamma
    pub const REC2020_GAMMA: f32 = 2.4;
    /// Linear gamma
    pub const LINEAR_GAMMA: f32 = 1.0;
}

/// LED wall constants
pub mod led_wall {
    /// Minimum pixel pitch (mm)
    pub const MIN_PIXEL_PITCH_MM: f64 = 0.5;
    /// Maximum pixel pitch (mm)
    pub const MAX_PIXEL_PITCH_MM: f64 = 10.0;
    /// Typical viewing distance (meters)
    pub const TYPICAL_VIEWING_DISTANCE_M: f64 = 5.0;
    /// Default refresh rate (Hz)
    pub const DEFAULT_REFRESH_RATE_HZ: u32 = 3840;
}

/// Camera tracking constants
pub mod tracking {
    /// Minimum confidence threshold
    pub const MIN_CONFIDENCE: f32 = 0.3;
    /// Good confidence threshold
    pub const GOOD_CONFIDENCE: f32 = 0.8;
    /// Excellent confidence threshold
    pub const EXCELLENT_CONFIDENCE: f32 = 0.95;
    /// Default smoothing window
    pub const DEFAULT_SMOOTHING_WINDOW: usize = 5;
}

/// Timing constants
pub mod timing {
    use super::Duration;

    /// One millisecond
    pub const ONE_MS: Duration = Duration::from_millis(1);
    /// One microsecond
    pub const ONE_US: Duration = Duration::from_micros(1);
    /// Frame time for 60 FPS
    pub const FRAME_TIME_60FPS: Duration = Duration::from_micros(16667);
    /// Frame time for 120 FPS
    pub const FRAME_TIME_120FPS: Duration = Duration::from_micros(8333);
}

/// Metric thresholds
pub mod metrics {
    /// Good FPS threshold
    pub const GOOD_FPS_THRESHOLD: f64 = 58.0;
    /// Acceptable frame drop rate (percentage)
    pub const ACCEPTABLE_DROP_RATE_PCT: f64 = 1.0;
    /// Critical latency threshold (ms)
    pub const CRITICAL_LATENCY_MS: u64 = 50;
}

/// Configuration presets
pub mod presets {
    use crate::{QualityMode, VirtualProductionConfig, WorkflowType};

    /// High quality LED wall preset
    #[must_use]
    pub fn led_wall_high_quality() -> VirtualProductionConfig {
        VirtualProductionConfig {
            workflow: WorkflowType::LedWall,
            target_fps: super::DEFAULT_TARGET_FPS,
            sync_accuracy_ms: 0.5,
            quality: QualityMode::Final,
            color_calibration: true,
            lens_correction: true,
            num_cameras: 1,
            motion_capture: false,
            unreal_integration: false,
        }
    }

    /// Real-time preview preset
    #[must_use]
    pub fn realtime_preview() -> VirtualProductionConfig {
        VirtualProductionConfig {
            workflow: WorkflowType::LedWall,
            target_fps: super::DEFAULT_TARGET_FPS,
            sync_accuracy_ms: 1.0,
            quality: QualityMode::Preview,
            color_calibration: false,
            lens_correction: false,
            num_cameras: 1,
            motion_capture: false,
            unreal_integration: false,
        }
    }

    /// Multi-camera production preset
    #[must_use]
    pub fn multi_camera_production() -> VirtualProductionConfig {
        VirtualProductionConfig {
            workflow: WorkflowType::LedWall,
            target_fps: super::DEFAULT_TARGET_FPS,
            sync_accuracy_ms: 0.5,
            quality: QualityMode::Final,
            color_calibration: true,
            lens_correction: true,
            num_cameras: 4,
            motion_capture: false,
            unreal_integration: false,
        }
    }

    /// Unreal Engine integration preset
    #[must_use]
    pub fn unreal_integration() -> VirtualProductionConfig {
        VirtualProductionConfig {
            workflow: WorkflowType::LedWall,
            target_fps: super::DEFAULT_TARGET_FPS,
            sync_accuracy_ms: 0.5,
            quality: QualityMode::Final,
            color_calibration: true,
            lens_correction: true,
            num_cameras: 1,
            motion_capture: true,
            unreal_integration: true,
        }
    }

    /// AR/VR preset
    #[must_use]
    pub fn ar_vr() -> VirtualProductionConfig {
        VirtualProductionConfig {
            workflow: WorkflowType::AugmentedReality,
            target_fps: 90.0, // Higher FPS for VR
            sync_accuracy_ms: 0.3,
            quality: QualityMode::Final,
            color_calibration: true,
            lens_correction: true,
            num_cameras: 1,
            motion_capture: true,
            unreal_integration: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rates() {
        assert_eq!(frame_rates::FPS_60, 60.0);
        assert_eq!(frame_rates::FPS_30, 30.0);
    }

    #[test]
    fn test_resolutions() {
        assert_eq!(resolutions::HD_1080P, (1920, 1080));
        assert_eq!(resolutions::UHD_4K, (3840, 2160));
    }

    #[test]
    fn test_presets() {
        let config = presets::led_wall_high_quality();
        assert_eq!(config.quality, crate::QualityMode::Final);

        let config = presets::realtime_preview();
        assert_eq!(config.quality, crate::QualityMode::Preview);

        let config = presets::multi_camera_production();
        assert_eq!(config.num_cameras, 4);
    }

    #[test]
    fn test_timing_constants() {
        assert_eq!(timing::ONE_MS.as_millis(), 1);
        assert_eq!(timing::ONE_US.as_micros(), 1);
    }
}
