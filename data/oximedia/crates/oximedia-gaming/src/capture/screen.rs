//! Screen capture implementation.
//!
//! Provides efficient screen capture for monitors, windows, and regions.

use crate::{GamingError, GamingResult};
use std::time::{Duration, Instant};

/// Screen capture implementation.
pub struct ScreenCapture {
    config: CaptureConfig,
    state: CaptureState,
    /// Frame sequence counter.
    sequence: u64,
    /// Timestamp of when capturing started, used for frame timestamps.
    capture_start: Option<Instant>,
}

/// Capture configuration.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Capture region
    pub region: CaptureRegion,
    /// Target framerate
    pub framerate: u32,
    /// Capture cursor
    pub capture_cursor: bool,
}

/// Capture region specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureRegion {
    /// Primary monitor (full screen)
    PrimaryMonitor,
    /// Specific monitor by index
    Monitor(usize),
    /// Specific window by handle/ID
    Window(u64),
    /// Custom region (x, y, width, height)
    Region {
        /// X coordinate of the region
        x: i32,
        /// Y coordinate of the region
        y: i32,
        /// Width of the region
        width: u32,
        /// Height of the region
        height: u32,
    },
}

/// Monitor information.
#[derive(Debug, Clone)]
pub struct MonitorInfo {
    /// Monitor index
    pub index: usize,
    /// Monitor name
    pub name: String,
    /// Resolution (width, height)
    pub resolution: (u32, u32),
    /// Position (x, y)
    pub position: (i32, i32),
    /// Refresh rate in Hz
    pub refresh_rate: u32,
    /// Is primary monitor
    pub is_primary: bool,
}

/// Capture state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureState {
    Idle,
    Capturing,
    Paused,
}

/// Captured frame data.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Frame data (RGBA or other format)
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Timestamp
    pub timestamp: Duration,
    /// Frame sequence number
    pub sequence: u64,
}

impl ScreenCapture {
    /// Create a new screen capture with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if screen capture initialization fails.
    pub fn new(config: CaptureConfig) -> GamingResult<Self> {
        // Validate configuration
        if config.framerate == 0 || config.framerate > 240 {
            return Err(GamingError::InvalidConfig(
                "Framerate must be between 1 and 240".to_string(),
            ));
        }

        Ok(Self {
            config,
            state: CaptureState::Idle,
            sequence: 0,
            capture_start: None,
        })
    }

    /// Start capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture fails to start.
    pub fn start(&mut self) -> GamingResult<()> {
        if self.state == CaptureState::Capturing {
            return Err(GamingError::InvalidConfig(
                "Capture already running".to_string(),
            ));
        }

        self.state = CaptureState::Capturing;
        self.sequence = 0;
        self.capture_start = Some(Instant::now());
        Ok(())
    }

    /// Stop capturing frames.
    pub fn stop(&mut self) {
        self.state = CaptureState::Idle;
        self.capture_start = None;
    }

    /// Pause capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture is not running.
    pub fn pause(&mut self) -> GamingResult<()> {
        if self.state != CaptureState::Capturing {
            return Err(GamingError::InvalidConfig(
                "Capture not running".to_string(),
            ));
        }

        self.state = CaptureState::Paused;
        Ok(())
    }

    /// Resume capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture is not paused.
    pub fn resume(&mut self) -> GamingResult<()> {
        if self.state != CaptureState::Paused {
            return Err(GamingError::InvalidConfig("Capture not paused".to_string()));
        }

        self.state = CaptureState::Capturing;
        Ok(())
    }

    /// Capture a single frame.
    ///
    /// Generates a simulated RGBA frame with a deterministic colour gradient
    /// pattern that varies per-frame (based on the sequence number). This
    /// exercises the full capture pipeline: resolution resolution, RGBA buffer
    /// allocation, timestamp tracking, and sequence numbering.
    ///
    /// # Errors
    ///
    /// Returns error if capture is not running.
    pub fn capture_frame(&mut self) -> GamingResult<CapturedFrame> {
        if self.state != CaptureState::Capturing {
            return Err(GamingError::CaptureFailed(
                "Capture not running".to_string(),
            ));
        }

        let (width, height) = match self.config.region {
            CaptureRegion::PrimaryMonitor | CaptureRegion::Monitor(_) => (1920, 1080),
            CaptureRegion::Window(_) => (1280, 720),
            CaptureRegion::Region { width, height, .. } => (width, height),
        };

        // Compute timestamp relative to capture start
        let timestamp = self
            .capture_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        let seq = self.sequence;
        self.sequence += 1;

        // Generate a deterministic gradient pattern that changes per frame.
        // Each pixel is RGBA: R depends on x, G on y, B on sequence, A = 255.
        let pixel_count = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(pixel_count * 4);
        let frame_phase = (seq % 256) as u8;

        for y in 0..height {
            // Precompute the green channel for this row
            let g = if height > 1 {
                ((y as u64 * 255) / (height as u64 - 1)) as u8
            } else {
                0
            };
            for x in 0..width {
                let r = if width > 1 {
                    ((x as u64 * 255) / (width as u64 - 1)) as u8
                } else {
                    0
                };
                data.push(r);
                data.push(g);
                data.push(frame_phase);
                data.push(255); // alpha
            }
        }

        // Optionally draw a small cursor indicator in the top-left corner
        if self.config.capture_cursor && width >= 8 && height >= 8 {
            for cy in 0..8u32 {
                for cx in 0..8u32 {
                    let idx = ((cy * width + cx) * 4) as usize;
                    if idx + 3 < data.len() {
                        data[idx] = 255; // R
                        data[idx + 1] = 255; // G
                        data[idx + 2] = 255; // B
                                             // alpha stays 255
                    }
                }
            }
        }

        Ok(CapturedFrame {
            data,
            width,
            height,
            timestamp,
            sequence: seq,
        })
    }

    /// Get list of available monitors.
    ///
    /// # Errors
    ///
    /// Returns error if monitor enumeration fails.
    pub fn list_monitors() -> GamingResult<Vec<MonitorInfo>> {
        // In a real implementation, this would enumerate actual monitors
        Ok(vec![MonitorInfo {
            index: 0,
            name: "Primary Monitor".to_string(),
            resolution: (1920, 1080),
            position: (0, 0),
            refresh_rate: 60,
            is_primary: true,
        }])
    }

    /// Check if capture is active.
    #[must_use]
    pub fn is_capturing(&self) -> bool {
        self.state == CaptureState::Capturing
    }

    /// Get capture configuration.
    #[must_use]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            region: CaptureRegion::PrimaryMonitor,
            framerate: 60,
            capture_cursor: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screen_capture_creation() {
        let config = CaptureConfig::default();
        let capture = ScreenCapture::new(config).expect("valid screen capture");
        assert!(!capture.is_capturing());
    }

    #[test]
    fn test_invalid_framerate() {
        let mut config = CaptureConfig::default();
        config.framerate = 0;
        let result = ScreenCapture::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_capture_lifecycle() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");

        capture.start().expect("start should succeed");
        assert!(capture.is_capturing());

        capture.pause().expect("pause should succeed");
        assert!(!capture.is_capturing());

        capture.resume().expect("resume should succeed");
        assert!(capture.is_capturing());

        capture.stop();
        assert!(!capture.is_capturing());
    }

    #[test]
    fn test_list_monitors() {
        let monitors = ScreenCapture::list_monitors().expect("list monitors should succeed");
        assert!(!monitors.is_empty());
    }

    #[test]
    fn test_capture_frame() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");

        capture.start().expect("start should succeed");
        let frame = capture
            .capture_frame()
            .expect("capture frame should succeed");

        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.data.is_empty());
    }

    #[test]
    fn test_capture_region() {
        let region = CaptureRegion::Region {
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
        };

        let mut config = CaptureConfig::default();
        config.region = region;

        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let frame = capture
            .capture_frame()
            .expect("capture frame should succeed");
        assert_eq!(frame.width, 1280);
        assert_eq!(frame.height, 720);
    }

    #[test]
    fn test_frame_sequence_increments() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let f0 = capture.capture_frame().expect("frame 0");
        let f1 = capture.capture_frame().expect("frame 1");
        let f2 = capture.capture_frame().expect("frame 2");

        assert_eq!(f0.sequence, 0);
        assert_eq!(f1.sequence, 1);
        assert_eq!(f2.sequence, 2);
    }

    #[test]
    fn test_frame_data_is_rgba() {
        let config = CaptureConfig {
            region: CaptureRegion::Region {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            framerate: 30,
            capture_cursor: false,
        };
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let frame = capture.capture_frame().expect("capture should succeed");
        // RGBA = 4 bytes per pixel
        assert_eq!(frame.data.len(), 16 * 16 * 4);
        // All alpha channels should be 255
        for pixel_idx in 0..(16 * 16) {
            assert_eq!(frame.data[pixel_idx * 4 + 3], 255);
        }
    }

    #[test]
    fn test_frames_differ_per_sequence() {
        let config = CaptureConfig {
            region: CaptureRegion::Region {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            framerate: 60,
            capture_cursor: false,
        };
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let f0 = capture.capture_frame().expect("frame 0");
        let f1 = capture.capture_frame().expect("frame 1");

        // Blue channel changes per frame, so data should differ
        assert_ne!(f0.data, f1.data);
    }

    #[test]
    fn test_cursor_indicator_drawn() {
        let config = CaptureConfig {
            region: CaptureRegion::Region {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            framerate: 60,
            capture_cursor: true,
        };
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let frame = capture.capture_frame().expect("capture should succeed");
        // Top-left 8x8 should be white (255,255,255,255)
        for cy in 0..8u32 {
            for cx in 0..8u32 {
                let idx = ((cy * 16 + cx) * 4) as usize;
                assert_eq!(frame.data[idx], 255, "R at ({cx},{cy})");
                assert_eq!(frame.data[idx + 1], 255, "G at ({cx},{cy})");
                assert_eq!(frame.data[idx + 2], 255, "B at ({cx},{cy})");
            }
        }
    }

    #[test]
    fn test_capture_not_running_error() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        // Not started - should fail
        assert!(capture.capture_frame().is_err());
    }

    #[test]
    fn test_sequence_resets_on_restart() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");
        let _ = capture.capture_frame().expect("frame 0");
        let _ = capture.capture_frame().expect("frame 1");
        capture.stop();

        capture.start().expect("restart should succeed");
        let frame = capture.capture_frame().expect("frame after restart");
        assert_eq!(frame.sequence, 0);
    }
}
