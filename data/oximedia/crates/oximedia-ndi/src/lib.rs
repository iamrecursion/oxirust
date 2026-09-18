//! NDI (Network Device Interface) implementation for OxiMedia
//!
//! This is a clean-room implementation of the NDI protocol that doesn't rely on
//! the official NDI SDK. It provides mDNS-based discovery, low-latency streaming,
//! and support for tally lights and PTZ control.
//!
//! # Features
//!
//! - mDNS-based source discovery
//! - Low-latency video streaming (<1 frame)
//! - Full HD and 4K support
//! - Tally light support (program/preview indicators)
//! - PTZ (Pan-Tilt-Zoom) control
//! - Audio/video synchronization
//! - Bandwidth adaptation
//!
//! # Example
//!
//! ```no_run
//! use oximedia_ndi::{NdiSource, NdiSender, SenderConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an NDI sender
//! let sender = NdiSender::new(SenderConfig::default()).await?;
//!
//! // Send a video frame
//! // sender.send_video_frame(frame).await?;
//!
//! // Discover NDI sources
//! let sources = NdiSource::discover_sources(std::time::Duration::from_secs(5)).await?;
//! for source in sources {
//!     println!("Found: {} at {}", source.name(), source.address());
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod alpha_channel;
pub mod audio_config;
pub mod audio_format;
pub mod av_buffer;
pub mod bandwidth;
pub mod bandwidth_limiter;
pub mod bandwidth_probe;
pub mod bridge;
pub mod channel_map;
pub mod clock_sync;
pub mod color_space_ndi;
pub mod connection_config;
pub mod connection_pool;
pub mod connection_state;
pub mod connection_stats;
pub mod failover;
pub mod frame_buffer;
pub mod frame_buffer_pool;
pub mod frame_pool;
pub mod frame_rate_converter;
pub mod frame_slice_encoder;
pub mod frame_sync;
pub mod genlock;
pub mod group;
pub mod hx2;
pub mod kvm;
pub mod latency_monitor;
pub mod local_bypass;
pub mod mdns_advertiser;
pub mod metadata;
pub mod metadata_channel;
pub mod metadata_frame;
pub mod metadata_serializer;
pub mod ndi_ext;
pub mod ndi_stats;
pub mod ptp_clock;
pub mod ptz;
pub mod quality;
pub mod recording;
pub mod routing;
pub mod sender_config;
pub mod source_filter;
pub mod source_registry;
pub mod statistics;
pub mod stream_info;
pub mod stream_mux;
pub mod tally_bridge;
pub mod tally_bus;
pub mod tally_manager;
pub mod transport;
pub mod video_format;
pub mod web_preview;

pub mod codec;
mod discovery;
pub mod protocol;
mod receiver;
mod sender;
mod tally;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub use codec::{SpeedHqCodec, YuvFormat};
pub use discovery::{DiscoveryService, NdiSourceInfo};
pub use protocol::{NdiFrame, NdiFrameType, NdiMetadata};
pub use receiver::{NdiReceiver, ReceiverConfig};
pub use sender::{NdiSender, SenderConfig};
pub use tally::{TallyServer, TallyState};

use oximedia_core::OxiError;
use thiserror::Error;

/// NDI-specific errors
#[derive(Error, Debug)]
pub enum NdiError {
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Discovery error: {0}")]
    Discovery(String),

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("Timeout")]
    Timeout,

    #[error("Source not found: {0}")]
    SourceNotFound(String),

    #[error("Invalid frame format")]
    InvalidFrameFormat,

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Core error: {0}")]
    Core(#[from] OxiError),
}

/// Result type for NDI operations
pub type Result<T> = std::result::Result<T, NdiError>;

/// NDI configuration
#[derive(Debug, Clone)]
pub struct NdiConfig {
    /// Source name
    pub name: String,

    /// Source groups (for filtering)
    pub groups: Vec<String>,

    /// Enable low bandwidth mode
    pub low_bandwidth: bool,

    /// Buffer size in frames
    pub buffer_size: usize,

    /// Enable tally support
    pub enable_tally: bool,

    /// Enable PTZ support
    pub enable_ptz: bool,

    /// Discovery timeout
    pub discovery_timeout: Duration,

    /// Connection timeout
    pub connection_timeout: Duration,
}

impl Default for NdiConfig {
    fn default() -> Self {
        Self {
            name: "OxiMedia NDI Source".to_string(),
            groups: vec!["public".to_string()],
            low_bandwidth: false,
            buffer_size: 16,
            enable_tally: true,
            enable_ptz: true,
            discovery_timeout: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(10),
        }
    }
}

/// Represents an NDI source that can be received from
#[derive(Debug, Clone)]
pub struct NdiSource {
    info: Arc<NdiSourceInfo>,
    receiver: Option<Arc<NdiReceiver>>,
}

impl NdiSource {
    /// Create a new NDI source from source information
    pub fn new(info: NdiSourceInfo) -> Self {
        Self {
            info: Arc::new(info),
            receiver: None,
        }
    }

    /// Get the source name
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the source address
    pub fn address(&self) -> SocketAddr {
        self.info.address
    }

    /// Get the source groups
    pub fn groups(&self) -> &[String] {
        &self.info.groups
    }

    /// Discover available NDI sources on the network
    ///
    /// # Arguments
    ///
    /// * `timeout` - How long to wait for discovery responses
    ///
    /// # Returns
    ///
    /// A list of discovered NDI sources
    pub async fn discover_sources(timeout: Duration) -> Result<Vec<NdiSource>> {
        let discovery = DiscoveryService::new()?;
        let sources = discovery.discover(timeout).await?;
        Ok(sources.into_iter().map(NdiSource::new).collect())
    }

    /// Connect to this NDI source for receiving
    ///
    /// # Arguments
    ///
    /// * `config` - Receiver configuration
    ///
    /// # Returns
    ///
    /// A receiver that can be used to receive frames from this source
    pub async fn connect(&mut self, config: ReceiverConfig) -> Result<Arc<NdiReceiver>> {
        let receiver = Arc::new(NdiReceiver::new(self.info.clone(), config).await?);
        self.receiver = Some(receiver.clone());
        Ok(receiver)
    }

    /// Get the connected receiver, if any
    pub fn receiver(&self) -> Option<Arc<NdiReceiver>> {
        self.receiver.clone()
    }

    /// Disconnect from the source
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(receiver) = self.receiver.take() {
            receiver.disconnect().await?;
        }
        Ok(())
    }
}

/// PTZ (Pan-Tilt-Zoom) command
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PtzCommand {
    /// Pan left at specified speed (0.0 to 1.0)
    PanLeft(f32),

    /// Pan right at specified speed (0.0 to 1.0)
    PanRight(f32),

    /// Tilt up at specified speed (0.0 to 1.0)
    TiltUp(f32),

    /// Tilt down at specified speed (0.0 to 1.0)
    TiltDown(f32),

    /// Zoom in at specified speed (0.0 to 1.0)
    ZoomIn(f32),

    /// Zoom out at specified speed (0.0 to 1.0)
    ZoomOut(f32),

    /// Focus near at specified speed (0.0 to 1.0)
    FocusNear(f32),

    /// Focus far at specified speed (0.0 to 1.0)
    FocusFar(f32),

    /// Auto focus
    AutoFocus,

    /// Store preset at index
    StorePreset(u8),

    /// Recall preset at index
    RecallPreset(u8),

    /// Stop all motion
    Stop,
}

/// Video frame format information
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoFormat {
    /// Width in pixels
    pub width: u32,

    /// Height in pixels
    pub height: u32,

    /// Frame rate numerator
    pub fps_num: u32,

    /// Frame rate denominator
    pub fps_den: u32,

    /// Progressive (true) or interlaced (false)
    pub progressive: bool,

    /// Aspect ratio (width/height)
    pub aspect_ratio: f32,
}

impl VideoFormat {
    /// Create a new video format
    pub fn new(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Self {
        Self {
            width,
            height,
            fps_num,
            fps_den,
            progressive: true,
            aspect_ratio: width as f32 / height as f32,
        }
    }

    /// Create a Full HD 1080p30 format
    pub fn full_hd_30p() -> Self {
        Self::new(1920, 1080, 30, 1)
    }

    /// Create a Full HD 1080p60 format
    pub fn full_hd_60p() -> Self {
        Self::new(1920, 1080, 60, 1)
    }

    /// Create a 4K UHD 30p format
    pub fn uhd_4k_30p() -> Self {
        Self::new(3840, 2160, 30, 1)
    }

    /// Create a 4K UHD 60p format
    pub fn uhd_4k_60p() -> Self {
        Self::new(3840, 2160, 60, 1)
    }

    /// Get the frame rate as a floating point number
    pub fn frame_rate(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }
}

/// Audio format information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Bits per sample
    pub bits_per_sample: u16,
}

impl AudioFormat {
    /// Create a new audio format
    pub fn new(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Self {
        Self {
            sample_rate,
            channels,
            bits_per_sample,
        }
    }

    /// Create a standard 48kHz stereo 16-bit format
    pub fn stereo_48k() -> Self {
        Self::new(48000, 2, 16)
    }

    /// Create a standard 48kHz mono 16-bit format
    pub fn mono_48k() -> Self {
        Self::new(48000, 1, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_format() {
        let format = VideoFormat::full_hd_60p();
        assert_eq!(format.width, 1920);
        assert_eq!(format.height, 1080);
        assert_eq!(format.fps_num, 60);
        assert_eq!(format.fps_den, 1);
        assert!((format.frame_rate() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_format() {
        let format = AudioFormat::stereo_48k();
        assert_eq!(format.sample_rate, 48000);
        assert_eq!(format.channels, 2);
        assert_eq!(format.bits_per_sample, 16);
    }

    #[test]
    fn test_ndi_config_default() {
        let config = NdiConfig::default();
        assert!(config.enable_tally);
        assert!(config.enable_ptz);
        assert_eq!(config.buffer_size, 16);
    }

    // --- Smoke tests for newly-registered orphan modules ---

    #[test]
    fn smoke_alpha_channel() {
        // Verify AlphaMode is accessible (AlphaFrame constructed via module types)
        use alpha_channel::AlphaMode;
        let mode = AlphaMode::default();
        let _ = std::hint::black_box(mode);
    }

    #[test]
    fn smoke_bandwidth_limiter() {
        use bandwidth_limiter::{BandwidthLimiter, LimiterConfig};
        let limiter = BandwidthLimiter::new(LimiterConfig::default());
        let _ = std::hint::black_box(limiter);
    }

    #[test]
    fn smoke_bandwidth_probe() {
        use bandwidth_probe::{BandwidthProber, ProbeConfig};
        let prober = BandwidthProber::new(ProbeConfig::default());
        let _ = std::hint::black_box(prober);
    }

    #[test]
    fn smoke_bridge() {
        use bridge::{BridgeRouteTable, SubnetId};
        use std::time::Duration;
        // BridgeRouteTable requires a TTL
        let b = BridgeRouteTable::new(Duration::from_secs(30));
        let _ = std::hint::black_box(b);
        let id = SubnetId::new("10.0.0.0/24");
        let _ = std::hint::black_box(id);
    }

    #[test]
    fn smoke_color_space_ndi() {
        use color_space_ndi::NdiColorSpace;
        let cs = NdiColorSpace::bt709();
        let _ = std::hint::black_box(cs);
    }

    #[test]
    fn smoke_connection_pool() {
        use connection_pool::{ConnectionPool, ConnectionPoolConfig};
        let pool = ConnectionPool::new(ConnectionPoolConfig::default());
        let _ = std::hint::black_box(pool);
    }

    #[test]
    fn smoke_connection_stats() {
        use connection_stats::ConnectionStatsRegistry;
        let reg = ConnectionStatsRegistry::new();
        let _ = std::hint::black_box(reg);
    }

    #[test]
    fn smoke_frame_buffer_pool() {
        use frame_buffer_pool::{BufferPool, PixelFormat};
        let pool = BufferPool::new(4, 1280, 720, PixelFormat::Bgra8);
        assert_eq!(pool.stats().total, 4);
        let _ = std::hint::black_box(pool);
    }

    #[test]
    fn smoke_frame_rate_converter() {
        use frame_rate_converter::{ConverterConfig, FrameRateConverter};
        let conv = FrameRateConverter::new(ConverterConfig::default());
        let _ = std::hint::black_box(conv);
    }

    #[test]
    fn smoke_hx2() {
        use hx2::{Hx2Config, Hx2Preset};
        let cfg = Hx2Config::from_preset(Hx2Preset::Balanced);
        let _ = std::hint::black_box(cfg);
    }

    #[test]
    fn smoke_kvm() {
        use kvm::{encode_kvm_event, KvmEvent, MouseMoveEvent};
        let evt = KvmEvent::MouseMove(MouseMoveEvent {
            x: 100.0,
            y: 200.0,
            absolute: true,
        });
        let xml = encode_kvm_event(&evt);
        assert!(xml.contains("ndi_kvm"));
        let _ = std::hint::black_box(xml);
    }

    #[test]
    fn smoke_latency_monitor() {
        use latency_monitor::LatencyMonitor;
        let mut m = LatencyMonitor::new("smoke");
        m.record(5.0);
        assert_eq!(m.sample_count(), 1);
        let _ = std::hint::black_box(m);
    }

    #[test]
    fn smoke_mdns_advertiser() {
        use mdns_advertiser::MdnsAdvertisement;
        let ad = MdnsAdvertisement::new("TestSource", 5960);
        assert_eq!(ad.port, 5960);
        let _ = std::hint::black_box(ad);
    }

    #[test]
    fn smoke_metadata_channel() {
        use metadata_channel::MetadataChannelQueue;
        let q = MetadataChannelQueue::new(32);
        assert_eq!(q.len(), 0);
        let _ = std::hint::black_box(q);
    }

    #[test]
    fn smoke_metadata_serializer() {
        use metadata_serializer::{
            MetadataPayload, MetadataSerializer, NdiMetadataFrame, NdiMetadataKind, TallyPayload,
        };
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::TallyState,
            source_name: "smoke".to_string(),
            timecode: 0,
            payload: MetadataPayload::Tally(TallyPayload {
                program: true,
                preview: false,
            }),
        };
        let xml = MetadataSerializer::to_xml(&frame);
        assert!(xml.contains("program"));
        let _ = std::hint::black_box(xml);
    }

    #[test]
    fn smoke_ndi_ext() {
        use ndi_ext::SpeedHqEncoder;
        let block = [0i32; 64];
        let out = SpeedHqEncoder::encode_intra_prediction(&block, 0);
        assert_eq!(out.len(), 64);
        let _ = std::hint::black_box(out);
    }

    #[test]
    fn smoke_ptp_clock() {
        use ptp_clock::{PtpClock, PtpServoConfig};
        let clock = PtpClock::new(PtpServoConfig::default());
        let _ = std::hint::black_box(clock);
    }

    #[test]
    fn smoke_recording() {
        use recording::{RecordingConfig, RecordingSession};
        let cfg = RecordingConfig::default();
        let session = RecordingSession::new(cfg);
        let _ = std::hint::black_box(session);
    }

    #[test]
    fn smoke_stream_mux() {
        use stream_mux::{MuxStrategy, StreamMux};
        let mux = StreamMux::new(MuxStrategy::default(), 16);
        let _ = std::hint::black_box(mux);
    }

    #[test]
    fn smoke_tally_bridge() {
        use tally_bridge::{GenericTallyState, MultiSourceTallyAggregator};
        let agg = MultiSourceTallyAggregator::new();
        let _ = std::hint::black_box(agg);
        let _ = std::hint::black_box(GenericTallyState::Off);
    }

    #[test]
    fn smoke_web_preview() {
        use web_preview::MjpegBroadcaster;
        // MjpegBroadcaster::new takes a capacity usize
        let broadcaster = MjpegBroadcaster::new(8);
        assert!(broadcaster.is_empty());
        let _ = std::hint::black_box(broadcaster);
    }

    // --- Frame pool smoke test ---

    #[test]
    fn smoke_frame_pool() {
        use frame_pool::FramePool;
        let mut pool = FramePool::new(4, 1024);
        assert_eq!(pool.capacity(), 4);
        let f = pool.acquire().expect("acquire frame");
        assert_eq!(pool.in_use(), 1);
        pool.release(f);
        assert_eq!(pool.in_use(), 0);
    }

    // --- Loopback test: encode a video frame, decode it, verify dimensions ---

    /// In-process loopback: create an NdiVideoFrame, serialize it via the NDI
    /// protocol wire format, deserialize it and verify that the dimensions and
    /// pixel data are preserved.
    #[test]
    fn loopback_encode_decode_video_frame() {
        use protocol::{NdiFrame, NdiVideoFrame};

        let width = 320_u32;
        let height = 240_u32;
        let fps_num = 30_u32;
        let fps_den = 1_u32;
        let format = VideoFormat::new(width, height, fps_num, fps_den);

        // Create a synthetic UYVY422 frame (2 bytes per pixel, filled with 128).
        let stride = (width * 2) as u32;
        let pixel_data = vec![128u8; (stride * height) as usize];
        let bytes_data = bytes::Bytes::from(pixel_data.clone());

        let sequence = 42_u32;
        let timestamp = 1_000_000_i64;

        let video_frame = NdiVideoFrame::new(sequence, timestamp, format, bytes_data, stride);
        let ndi_frame = NdiFrame::Video(video_frame);

        // Encode to wire bytes.
        let wire_bytes = ndi_frame.encode().expect("encode frame");

        // Decode from wire bytes.
        let decoded_frame = NdiFrame::decode(&wire_bytes).expect("decode frame");

        // Verify the decoded frame.
        if let NdiFrame::Video(vf) = decoded_frame {
            assert_eq!(vf.format.width, width, "width mismatch");
            assert_eq!(vf.format.height, height, "height mismatch");
            assert_eq!(vf.format.fps_num, fps_num, "fps_num mismatch");
            assert_eq!(vf.format.fps_den, fps_den, "fps_den mismatch");
            assert_eq!(vf.stride, stride, "stride mismatch");
            assert_eq!(vf.header.sequence, sequence, "sequence mismatch");
            assert_eq!(
                vf.data.len(),
                pixel_data.len(),
                "pixel data length mismatch"
            );
        } else {
            panic!("decoded frame is not a Video frame");
        }
    }

    // --- Parallel SpeedHQ encoding smoke test ---

    #[test]
    fn parallel_speedhq_single_vs_multi_slice() {
        use codec::{encode_speedhq_parallel, VideoFrame};

        let frame = VideoFrame::new_test(320, 240);

        let single = encode_speedhq_parallel(&frame, 1);
        let multi = encode_speedhq_parallel(&frame, 4);

        assert_eq!(
            single.len(),
            1,
            "single-slice should return 1 encoded chunk"
        );
        assert_eq!(multi.len(), 4, "4-slice should return 4 encoded chunks");

        // Each slice should be non-empty.
        for (i, chunk) in single.iter().enumerate() {
            assert!(!chunk.is_empty(), "single-slice chunk {i} is empty");
        }
        for (i, chunk) in multi.iter().enumerate() {
            assert!(!chunk.is_empty(), "4-slice chunk {i} is empty");
        }

        // Total bytes in multi-slice should be in the same order of magnitude
        // as single-slice (allow up to 10x difference due to slice overhead).
        let total_single: usize = single.iter().map(|s| s.len()).sum();
        let total_multi: usize = multi.iter().map(|s| s.len()).sum();
        assert!(
            total_multi <= total_single * 10,
            "multi-slice total {total_multi} is suspiciously larger than single {total_single}"
        );
    }
}
