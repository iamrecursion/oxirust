#![allow(dead_code)]
//! NDI bridge for virtual production.
//!
//! Provides an abstraction layer for sending and receiving video frames
//! over NDI (Network Device Interface) within a virtual production pipeline.
//! Supports source discovery, frame format conversion, and health monitoring.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Pixel format used for NDI frame transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiPixelFormat {
    /// 8-bit BGRA (32 bpp, alpha pre-multiplied).
    Bgra8,
    /// 8-bit UYVY (16 bpp, 4:2:2 chroma sub-sampling).
    Uyvy,
    /// 8-bit RGBA (32 bpp, straight alpha).
    Rgba8,
    /// 10-bit packed (v210 layout).
    V210,
    /// 16-bit linear light per channel (64 bpp).
    Rgba16f,
}

impl NdiPixelFormat {
    /// Bytes per pixel for this format.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bytes_per_pixel(&self) -> f64 {
        match self {
            Self::Bgra8 | Self::Rgba8 => 4.0,
            Self::Uyvy => 2.0,
            Self::V210 => 8.0 / 3.0,
            Self::Rgba16f => 8.0,
        }
    }

    /// Returns `true` if the format carries an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        matches!(self, Self::Bgra8 | Self::Rgba8 | Self::Rgba16f)
    }

    /// Returns the human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bgra8 => "BGRA8",
            Self::Uyvy => "UYVY",
            Self::Rgba8 => "RGBA8",
            Self::V210 => "V210",
            Self::Rgba16f => "RGBA16F",
        }
    }
}

/// Frame rate descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NdiFrameRate {
    /// Numerator (e.g. 30000).
    pub num: u32,
    /// Denominator (e.g. 1001).
    pub den: u32,
}

impl NdiFrameRate {
    /// Creates a new frame rate.
    #[must_use]
    pub fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// Frames per second as `f64`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.den == 0 {
            return 0.0;
        }
        f64::from(self.num) / f64::from(self.den)
    }

    /// Common 29.97 fps (NTSC).
    #[must_use]
    pub fn ntsc_30() -> Self {
        Self {
            num: 30000,
            den: 1001,
        }
    }

    /// Common 59.94 fps.
    #[must_use]
    pub fn ntsc_60() -> Self {
        Self {
            num: 60000,
            den: 1001,
        }
    }

    /// 25 fps (PAL).
    #[must_use]
    pub fn pal_25() -> Self {
        Self { num: 25, den: 1 }
    }
}

/// A single NDI video frame.
#[derive(Debug, Clone)]
pub struct NdiFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub pixel_format: NdiPixelFormat,
    /// Frame rate.
    pub frame_rate: NdiFrameRate,
    /// Row stride in bytes.
    pub stride: u32,
    /// Raw pixel data.
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
}

impl NdiFrame {
    /// Creates a new frame with allocated but zeroed pixel data.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        pixel_format: NdiPixelFormat,
        frame_rate: NdiFrameRate,
    ) -> Self {
        let bpp = pixel_format.bytes_per_pixel();
        let stride = (f64::from(width) * bpp).ceil() as u32;
        let data_len = (stride * height) as usize;
        Self {
            width,
            height,
            pixel_format,
            frame_rate,
            stride,
            data: vec![0u8; data_len],
            pts_us: 0,
        }
    }

    /// Total data size in bytes.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when data length matches expected stride * height.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == (self.stride as usize) * (self.height as usize)
    }
}

/// Health status of an NDI connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdiBridgeHealth {
    /// Healthy – frames arriving on time.
    Healthy,
    /// Degraded – occasional dropped frames.
    Degraded,
    /// Disconnected – no frames for a long time.
    Disconnected,
}

/// Discovered NDI source on the network.
#[derive(Debug, Clone)]
pub struct NdiSource {
    /// Human-readable source name.
    pub name: String,
    /// IP address or hostname.
    pub address: String,
    /// Port number.
    pub port: u16,
    /// When the source was last seen.
    pub last_seen: Instant,
}

impl NdiSource {
    /// Creates a new source entry.
    pub fn new(name: impl Into<String>, address: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
            port,
            last_seen: Instant::now(),
        }
    }

    /// Returns `true` if the source has not been seen for longer than `timeout`.
    #[must_use]
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_seen.elapsed() > timeout
    }
}

/// NDI bridge configuration.
#[derive(Debug, Clone)]
pub struct NdiBridgeConfig {
    /// Preferred pixel format for receiving frames.
    pub preferred_format: NdiPixelFormat,
    /// Target frame rate.
    pub target_frame_rate: NdiFrameRate,
    /// Discovery timeout before marking sources stale.
    pub discovery_timeout: Duration,
    /// Maximum receive queue depth (frames).
    pub max_queue_depth: usize,
    /// Health check interval.
    pub health_check_interval: Duration,
}

impl Default for NdiBridgeConfig {
    fn default() -> Self {
        Self {
            preferred_format: NdiPixelFormat::Bgra8,
            target_frame_rate: NdiFrameRate::ntsc_60(),
            discovery_timeout: Duration::from_secs(10),
            max_queue_depth: 4,
            health_check_interval: Duration::from_secs(2),
        }
    }
}

/// Statistics for an NDI bridge session.
#[derive(Debug, Clone)]
pub struct NdiBridgeStats {
    /// Total frames received.
    pub frames_received: u64,
    /// Total frames sent.
    pub frames_sent: u64,
    /// Total frames dropped.
    pub frames_dropped: u64,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Current health.
    pub health: NdiBridgeHealth,
    /// Uptime.
    pub uptime: Duration,
}

impl NdiBridgeStats {
    /// Creates zeroed stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames_received: 0,
            frames_sent: 0,
            frames_dropped: 0,
            bytes_transferred: 0,
            health: NdiBridgeHealth::Disconnected,
            uptime: Duration::ZERO,
        }
    }

    /// Drop ratio as a percentage.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn drop_ratio(&self) -> f64 {
        let total = self.frames_received + self.frames_dropped;
        if total == 0 {
            return 0.0;
        }
        self.frames_dropped as f64 / total as f64 * 100.0
    }

    /// Average throughput in megabytes per second.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        let secs = self.uptime.as_secs_f64();
        if secs < f64::EPSILON {
            return 0.0;
        }
        (self.bytes_transferred as f64) / secs / (1024.0 * 1024.0)
    }
}

impl Default for NdiBridgeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// NDI bridge – manages sending and receiving NDI streams.
pub struct NdiBridge {
    /// Bridge configuration.
    config: NdiBridgeConfig,
    /// Known NDI sources on the local subnet.
    sources: HashMap<String, NdiSource>,
    /// Accumulated statistics.
    stats: NdiBridgeStats,
    /// Session start time.
    started_at: Instant,
    /// Whether the bridge is currently running.
    running: bool,
}

impl NdiBridge {
    /// Creates a new NDI bridge with the given config.
    #[must_use]
    pub fn new(config: NdiBridgeConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            stats: NdiBridgeStats::new(),
            started_at: Instant::now(),
            running: false,
        }
    }

    /// Starts the bridge (begins source discovery).
    pub fn start(&mut self) {
        self.running = true;
        self.started_at = Instant::now();
        self.stats.health = NdiBridgeHealth::Healthy;
    }

    /// Stops the bridge.
    pub fn stop(&mut self) {
        self.running = false;
        self.stats.health = NdiBridgeHealth::Disconnected;
    }

    /// Returns whether the bridge is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Registers a discovered NDI source.
    pub fn register_source(&mut self, source: NdiSource) {
        self.sources.insert(source.name.clone(), source);
    }

    /// Removes stale sources that exceed the discovery timeout.
    pub fn prune_stale_sources(&mut self) -> usize {
        let timeout = self.config.discovery_timeout;
        let before = self.sources.len();
        self.sources.retain(|_, s| !s.is_stale(timeout));
        before - self.sources.len()
    }

    /// Returns a slice of currently known sources.
    #[must_use]
    pub fn sources(&self) -> Vec<&NdiSource> {
        self.sources.values().collect()
    }

    /// Simulates receiving a frame and updating statistics.
    pub fn receive_frame(&mut self, frame: &NdiFrame) {
        if !self.running {
            return;
        }
        self.stats.frames_received += 1;
        self.stats.bytes_transferred += frame.data_size() as u64;
        self.stats.uptime = self.started_at.elapsed();
    }

    /// Simulates sending a frame and updating statistics.
    pub fn send_frame(&mut self, frame: &NdiFrame) {
        if !self.running {
            return;
        }
        self.stats.frames_sent += 1;
        self.stats.bytes_transferred += frame.data_size() as u64;
        self.stats.uptime = self.started_at.elapsed();
    }

    /// Records a dropped frame.
    pub fn record_drop(&mut self) {
        self.stats.frames_dropped += 1;
        if self.stats.drop_ratio() > 5.0 {
            self.stats.health = NdiBridgeHealth::Degraded;
        }
    }

    /// Returns current statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> &NdiBridgeStats {
        &self.stats
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &NdiBridgeConfig {
        &self.config
    }

    /// Evaluates health based on recent statistics.
    pub fn evaluate_health(&mut self) -> NdiBridgeHealth {
        if !self.running {
            self.stats.health = NdiBridgeHealth::Disconnected;
        } else if self.stats.drop_ratio() > 10.0 {
            self.stats.health = NdiBridgeHealth::Degraded;
        } else {
            self.stats.health = NdiBridgeHealth::Healthy;
        }
        self.stats.health
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_bytes_per_pixel() {
        assert!((NdiPixelFormat::Bgra8.bytes_per_pixel() - 4.0).abs() < f64::EPSILON);
        assert!((NdiPixelFormat::Uyvy.bytes_per_pixel() - 2.0).abs() < f64::EPSILON);
        assert!((NdiPixelFormat::Rgba16f.bytes_per_pixel() - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pixel_format_has_alpha() {
        assert!(NdiPixelFormat::Bgra8.has_alpha());
        assert!(NdiPixelFormat::Rgba8.has_alpha());
        assert!(!NdiPixelFormat::Uyvy.has_alpha());
        assert!(!NdiPixelFormat::V210.has_alpha());
    }

    #[test]
    fn test_pixel_format_name() {
        assert_eq!(NdiPixelFormat::V210.name(), "V210");
        assert_eq!(NdiPixelFormat::Rgba16f.name(), "RGBA16F");
    }

    #[test]
    fn test_frame_rate_fps() {
        let ntsc = NdiFrameRate::ntsc_30();
        assert!((ntsc.fps() - 29.97).abs() < 0.1);
        let pal = NdiFrameRate::pal_25();
        assert!((pal.fps() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_rate_zero_den() {
        let fr = NdiFrameRate::new(30000, 0);
        assert!((fr.fps()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_creation_and_validity() {
        let frame = NdiFrame::new(1920, 1080, NdiPixelFormat::Bgra8, NdiFrameRate::ntsc_60());
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert!(frame.is_valid());
        assert_eq!(frame.data_size(), 1920 * 4 * 1080);
    }

    #[test]
    fn test_frame_uyvy_stride() {
        let frame = NdiFrame::new(1920, 1080, NdiPixelFormat::Uyvy, NdiFrameRate::pal_25());
        assert_eq!(frame.stride, 3840);
        assert!(frame.is_valid());
    }

    #[test]
    fn test_ndi_source_staleness() {
        let source = NdiSource::new("Cam1", "192.168.1.100", 5961);
        assert!(!source.is_stale(Duration::from_secs(5)));
    }

    #[test]
    fn test_bridge_start_stop() {
        let mut bridge = NdiBridge::new(NdiBridgeConfig::default());
        assert!(!bridge.is_running());
        bridge.start();
        assert!(bridge.is_running());
        assert_eq!(bridge.evaluate_health(), NdiBridgeHealth::Healthy);
        bridge.stop();
        assert!(!bridge.is_running());
        assert_eq!(bridge.evaluate_health(), NdiBridgeHealth::Disconnected);
    }

    #[test]
    fn test_bridge_receive_send() {
        let mut bridge = NdiBridge::new(NdiBridgeConfig::default());
        bridge.start();
        let frame = NdiFrame::new(1920, 1080, NdiPixelFormat::Bgra8, NdiFrameRate::ntsc_60());
        bridge.receive_frame(&frame);
        bridge.send_frame(&frame);
        assert_eq!(bridge.stats().frames_received, 1);
        assert_eq!(bridge.stats().frames_sent, 1);
    }

    #[test]
    fn test_bridge_source_management() {
        let mut bridge = NdiBridge::new(NdiBridgeConfig::default());
        bridge.register_source(NdiSource::new("Cam1", "10.0.0.1", 5961));
        bridge.register_source(NdiSource::new("Cam2", "10.0.0.2", 5961));
        assert_eq!(bridge.sources().len(), 2);
    }

    #[test]
    fn test_stats_drop_ratio() {
        let mut stats = NdiBridgeStats::new();
        assert!((stats.drop_ratio()).abs() < f64::EPSILON);
        stats.frames_received = 95;
        stats.frames_dropped = 5;
        assert!((stats.drop_ratio() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_throughput() {
        let stats = NdiBridgeStats {
            frames_received: 100,
            frames_sent: 0,
            frames_dropped: 0,
            bytes_transferred: 1024 * 1024 * 100,
            health: NdiBridgeHealth::Healthy,
            uptime: Duration::from_secs(10),
        };
        assert!((stats.throughput_mbps() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_bridge_degraded_health_on_drops() {
        let mut bridge = NdiBridge::new(NdiBridgeConfig::default());
        bridge.start();
        bridge.stats.frames_received = 80;
        bridge.stats.frames_dropped = 20;
        let health = bridge.evaluate_health();
        assert_eq!(health, NdiBridgeHealth::Degraded);
    }

    #[test]
    fn test_default_config() {
        let cfg = NdiBridgeConfig::default();
        assert_eq!(cfg.preferred_format, NdiPixelFormat::Bgra8);
        assert_eq!(cfg.max_queue_depth, 4);
        assert_eq!(cfg.discovery_timeout, Duration::from_secs(10));
    }
}
