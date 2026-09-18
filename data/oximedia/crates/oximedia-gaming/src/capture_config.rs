//! Game capture configuration and session management.
//!
//! Provides types for configuring screen/game capture sources, defining
//! capture regions, and tracking per-session frame statistics.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// CaptureSource
// ---------------------------------------------------------------------------

/// The origin of a game capture stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureSource {
    /// Capture an entire monitor by index.
    Monitor(u32),
    /// Capture a specific application window by title.
    Window(String),
    /// Use the OS game-capture hook for minimal overhead.
    GameCapture,
    /// Capture a webcam/video device by index.
    Webcam(u32),
}

impl CaptureSource {
    /// Returns `true` for sources that represent a full display output
    /// (`Monitor` or `GameCapture`).
    #[must_use]
    pub fn is_display(&self) -> bool {
        matches!(self, Self::Monitor(_) | Self::GameCapture)
    }

    /// Human-readable description of the capture source.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Monitor(idx) => format!("Monitor {idx}"),
            Self::Window(title) => format!("Window: {title}"),
            Self::GameCapture => "Game Capture (hook)".to_string(),
            Self::Webcam(idx) => format!("Webcam {idx}"),
        }
    }
}

// ---------------------------------------------------------------------------
// CaptureConfig
// ---------------------------------------------------------------------------

/// Full configuration for a game capture session.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Where to capture from.
    pub source: CaptureSource,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target frames per second.
    pub fps: f32,
    /// Capture in HDR mode.
    pub hdr: bool,
    /// Enable compatibility with anti-cheat systems.
    pub anti_cheat_compat: bool,
}

impl CaptureConfig {
    /// Returns `true` for HD resolutions (width ≥ 1280 and height ≥ 720).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.width >= 1280 && self.height >= 720
    }

    /// Returns `true` for 4 K (UHD) resolutions (width ≥ 3840 and height ≥ 2160).
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width >= 3840 && self.height >= 2160
    }

    /// Total pixel throughput per second (`width × height × fps`).
    #[must_use]
    pub fn pixel_rate(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * self.fps as u64
    }
}

// ---------------------------------------------------------------------------
// CaptureRegion
// ---------------------------------------------------------------------------

/// A rectangular sub-region of a capture source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRegion {
    /// Left edge offset in pixels.
    pub x: u32,
    /// Top edge offset in pixels.
    pub y: u32,
    /// Region width in pixels.
    pub width: u32,
    /// Region height in pixels.
    pub height: u32,
}

impl CaptureRegion {
    /// Region area in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` when the region covers the entire `total_w × total_h`
    /// surface and starts at the origin.
    #[must_use]
    pub fn is_fullscreen(&self, total_w: u32, total_h: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width == total_w && self.height == total_h
    }
}

// ---------------------------------------------------------------------------
// CaptureSession
// ---------------------------------------------------------------------------

/// A running capture session that tracks frame delivery statistics.
#[derive(Debug, Clone)]
pub struct CaptureSession {
    /// Capture configuration for this session.
    pub config: CaptureConfig,
    /// Optional capture sub-region (full frame if `None`).
    pub region: Option<CaptureRegion>,
    /// Total frames successfully captured.
    pub frames_captured: u64,
    /// Total frames dropped by the capture pipeline.
    pub dropped_frames: u64,
}

impl CaptureSession {
    /// Create a new capture session.
    #[must_use]
    pub fn new(config: CaptureConfig, region: Option<CaptureRegion>) -> Self {
        Self {
            config,
            region,
            frames_captured: 0,
            dropped_frames: 0,
        }
    }

    /// Record a successfully delivered frame.
    pub fn record_frame(&mut self) {
        self.frames_captured += 1;
    }

    /// Record a dropped frame.
    pub fn drop_frame(&mut self) {
        self.dropped_frames += 1;
    }

    /// Fraction of frames dropped (0.0–1.0).
    ///
    /// Returns `0.0` when no frames have been attempted.
    #[must_use]
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames_captured + self.dropped_frames;
        if total == 0 {
            return 0.0;
        }
        self.dropped_frames as f64 / total as f64
    }

    /// Effective capture rate adjusted for dropped frames.
    ///
    /// `effective_fps = configured_fps × (1 − drop_rate)`
    #[must_use]
    pub fn effective_fps(&self) -> f32 {
        self.config.fps * (1.0 - self.drop_rate() as f32)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CaptureConfig {
        CaptureConfig {
            source: CaptureSource::Monitor(0),
            width: 1920,
            height: 1080,
            fps: 60.0,
            hdr: false,
            anti_cheat_compat: false,
        }
    }

    // CaptureSource

    #[test]
    fn test_monitor_is_display() {
        assert!(CaptureSource::Monitor(0).is_display());
    }

    #[test]
    fn test_game_capture_is_display() {
        assert!(CaptureSource::GameCapture.is_display());
    }

    #[test]
    fn test_window_not_display() {
        assert!(!CaptureSource::Window("Game".to_string()).is_display());
    }

    #[test]
    fn test_webcam_not_display() {
        assert!(!CaptureSource::Webcam(0).is_display());
    }

    #[test]
    fn test_monitor_description() {
        assert_eq!(CaptureSource::Monitor(1).description(), "Monitor 1");
    }

    #[test]
    fn test_window_description() {
        let d = CaptureSource::Window("MyGame".to_string()).description();
        assert!(d.contains("MyGame"));
    }

    // CaptureConfig

    #[test]
    fn test_is_hd_1080p() {
        assert!(default_config().is_hd());
    }

    #[test]
    fn test_is_hd_false_sd() {
        let cfg = CaptureConfig {
            width: 640,
            height: 480,
            ..default_config()
        };
        assert!(!cfg.is_hd());
    }

    #[test]
    fn test_is_4k() {
        let cfg = CaptureConfig {
            width: 3840,
            height: 2160,
            ..default_config()
        };
        assert!(cfg.is_4k());
    }

    #[test]
    fn test_is_not_4k_1080p() {
        assert!(!default_config().is_4k());
    }

    #[test]
    fn test_pixel_rate() {
        // 1920 × 1080 × 60 = 124_416_000
        assert_eq!(default_config().pixel_rate(), 1920 * 1080 * 60);
    }

    // CaptureRegion

    #[test]
    fn test_capture_region_area() {
        let r = CaptureRegion {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert_eq!(r.area(), 1920 * 1080);
    }

    #[test]
    fn test_capture_region_is_fullscreen_true() {
        let r = CaptureRegion {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert!(r.is_fullscreen(1920, 1080));
    }

    #[test]
    fn test_capture_region_is_fullscreen_false_offset() {
        let r = CaptureRegion {
            x: 10,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert!(!r.is_fullscreen(1920, 1080));
    }

    // CaptureSession

    #[test]
    fn test_capture_session_record_frame() {
        let mut s = CaptureSession::new(default_config(), None);
        s.record_frame();
        s.record_frame();
        assert_eq!(s.frames_captured, 2);
    }

    #[test]
    fn test_capture_session_drop_rate_zero() {
        let s = CaptureSession::new(default_config(), None);
        assert_eq!(s.drop_rate(), 0.0);
    }

    #[test]
    fn test_capture_session_drop_rate_half() {
        let mut s = CaptureSession::new(default_config(), None);
        s.record_frame();
        s.drop_frame();
        assert!((s.drop_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_capture_session_effective_fps() {
        let mut s = CaptureSession::new(default_config(), None);
        // No drops → effective fps == configured fps
        s.record_frame();
        assert!((s.effective_fps() - 60.0).abs() < 0.001);
    }
}
