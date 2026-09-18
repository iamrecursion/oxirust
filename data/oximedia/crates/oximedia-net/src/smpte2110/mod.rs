//! SMPTE ST 2110 professional media over IP.
//!
//! This module implements the SMPTE ST 2110 suite of standards for professional
//! media transport over IP networks. It provides broadcast-quality video, audio,
//! and ancillary data transmission with precise timing and synchronization.
//!
//! # Standards
//!
//! - **ST 2110-10**: System timing and definitions
//! - **ST 2110-20**: Uncompressed video
//! - **ST 2110-21**: Traffic shaping and delivery timing
//! - **ST 2110-30**: PCM audio
//! - **ST 2110-40**: Ancillary data
//!
//! # Features
//!
//! - Uncompressed video transmission (SD, HD, 4K, 8K)
//! - PCM audio up to 64 channels
//! - Ancillary data (captions, timecode, etc.)
//! - PTP (IEEE 1588) synchronization
//! - Narrow/wide timing modes
//! - Gapped/linear transmission
//! - SDP session description
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::smpte2110::{St2110Source, VideoConfig};
//! use oximedia_net::smpte2110::video::PixelFormat;
//!
//! let config = VideoConfig {
//!     width: 1920,
//!     height: 1080,
//!     frame_rate: FrameRate::FPS_25,
//!     pixel_format: PixelFormat::YCbCr422_10bit,
//!     ..Default::default()
//! };
//!
//! let source = St2110Source::new_video(config);
//! ```

use crate::error::NetResult;
use bytes::Bytes;
use parking_lot::RwLock;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

pub mod ancillary;
pub mod audio;
pub mod ptp;
pub mod rtp;
pub mod sdp;
pub mod timing;
pub mod video;

// Re-export commonly used types
pub use ancillary::{AncillaryConfig, AncillaryData, AncillaryPacket};
pub use audio::{AudioConfig, AudioFormat, AudioPacket, AudioSampleRate};
pub use ptp::{PtpClock, PtpTimestamp};
pub use rtp::{RtpHeader, RtpPacket, RtpSession};
pub use sdp::{MediaType, SdpSession};
pub use timing::{FrameRate, ScanType, TimingMode, TransmissionMode};
pub use video::{PixelFormat, VideoConfig, VideoPacket};

/// ST 2110 stream type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// Video stream (ST 2110-20).
    Video,
    /// Audio stream (ST 2110-30).
    Audio,
    /// Ancillary data stream (ST 2110-40).
    Ancillary,
}

/// ST 2110 source configuration.
#[derive(Debug, Clone)]
pub struct St2110Config {
    /// Stream type.
    pub stream_type: StreamType,
    /// Multicast/unicast destination address.
    pub destination: IpAddr,
    /// Destination port.
    pub port: u16,
    /// Source IP address.
    pub source_ip: IpAddr,
    /// SSRC identifier.
    pub ssrc: u32,
    /// Timing mode.
    pub timing_mode: TimingMode,
    /// Transmission mode.
    pub transmission_mode: TransmissionMode,
    /// Enable PTP synchronization.
    pub enable_ptp: bool,
    /// PTP domain number.
    pub ptp_domain: u8,
}

impl Default for St2110Config {
    fn default() -> Self {
        Self {
            stream_type: StreamType::Video,
            destination: IpAddr::V4(Ipv4Addr::new(239, 0, 0, 1)),
            port: 5004,
            source_ip: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            ssrc: rand::random(),
            timing_mode: TimingMode::Wide,
            transmission_mode: TransmissionMode::Gapped,
            enable_ptp: true,
            ptp_domain: ptp::PTP_DOMAIN_DEFAULT,
        }
    }
}

/// ST 2110 source for transmitting media.
#[derive(Debug)]
pub struct St2110Source {
    /// Configuration.
    config: St2110Config,
    /// RTP session.
    rtp_session: Arc<RwLock<RtpSession>>,
    /// PTP clock (if enabled).
    ptp_clock: Option<Arc<RwLock<PtpClock>>>,
    /// Video configuration (if video stream).
    video_config: Option<VideoConfig>,
    /// Audio configuration (if audio stream).
    audio_config: Option<AudioConfig>,
    /// Ancillary configuration (if ancillary stream).
    ancillary_config: Option<AncillaryConfig>,
}

impl St2110Source {
    /// Creates a new ST 2110 video source.
    #[must_use]
    pub fn new_video(video_config: VideoConfig) -> Self {
        let config = St2110Config {
            stream_type: StreamType::Video,
            ..Default::default()
        };

        let clock_rate = 90000; // Standard video clock rate
        let rtp_session = Arc::new(RwLock::new(RtpSession::new(config.ssrc, clock_rate)));

        let ptp_clock = if config.enable_ptp {
            let clock_id = ptp::ClockIdentity::new([0; 8]); // Would use MAC address
            let port_id = ptp::PortIdentity::new(clock_id, 1);
            Some(Arc::new(RwLock::new(PtpClock::new(
                port_id,
                config.ptp_domain,
            ))))
        } else {
            None
        };

        Self {
            config,
            rtp_session,
            ptp_clock,
            video_config: Some(video_config),
            audio_config: None,
            ancillary_config: None,
        }
    }

    /// Creates a new ST 2110 audio source.
    #[must_use]
    pub fn new_audio(audio_config: AudioConfig) -> Self {
        let config = St2110Config {
            stream_type: StreamType::Audio,
            ..Default::default()
        };

        let clock_rate = audio_config.sample_rate as u32;
        let rtp_session = Arc::new(RwLock::new(RtpSession::new(config.ssrc, clock_rate)));

        let ptp_clock = if config.enable_ptp {
            let clock_id = ptp::ClockIdentity::new([0; 8]);
            let port_id = ptp::PortIdentity::new(clock_id, 1);
            Some(Arc::new(RwLock::new(PtpClock::new(
                port_id,
                config.ptp_domain,
            ))))
        } else {
            None
        };

        Self {
            config,
            rtp_session,
            ptp_clock,
            video_config: None,
            audio_config: Some(audio_config),
            ancillary_config: None,
        }
    }

    /// Creates a new ST 2110 ancillary data source.
    #[must_use]
    pub fn new_ancillary(ancillary_config: AncillaryConfig) -> Self {
        let config = St2110Config {
            stream_type: StreamType::Ancillary,
            ..Default::default()
        };

        let clock_rate = 90000; // Standard clock rate for ancillary
        let rtp_session = Arc::new(RwLock::new(RtpSession::new(config.ssrc, clock_rate)));

        let ptp_clock = if config.enable_ptp {
            let clock_id = ptp::ClockIdentity::new([0; 8]);
            let port_id = ptp::PortIdentity::new(clock_id, 1);
            Some(Arc::new(RwLock::new(PtpClock::new(
                port_id,
                config.ptp_domain,
            ))))
        } else {
            None
        };

        Self {
            config,
            rtp_session,
            ptp_clock,
            video_config: None,
            audio_config: None,
            ancillary_config: Some(ancillary_config),
        }
    }

    /// Gets the stream configuration.
    #[must_use]
    pub const fn config(&self) -> &St2110Config {
        &self.config
    }

    /// Gets the video configuration (if video stream).
    #[must_use]
    pub const fn video_config(&self) -> Option<&VideoConfig> {
        self.video_config.as_ref()
    }

    /// Gets the audio configuration (if audio stream).
    #[must_use]
    pub const fn audio_config(&self) -> Option<&AudioConfig> {
        self.audio_config.as_ref()
    }

    /// Gets the ancillary configuration (if ancillary stream).
    #[must_use]
    pub const fn ancillary_config(&self) -> Option<&AncillaryConfig> {
        self.ancillary_config.as_ref()
    }

    /// Gets the RTP session.
    #[must_use]
    pub fn rtp_session(&self) -> Arc<RwLock<RtpSession>> {
        Arc::clone(&self.rtp_session)
    }

    /// Gets the PTP clock (if enabled).
    #[must_use]
    pub fn ptp_clock(&self) -> Option<Arc<RwLock<PtpClock>>> {
        self.ptp_clock.as_ref().map(Arc::clone)
    }

    /// Generates an SDP description for this source.
    #[must_use]
    pub fn generate_sdp(&self) -> String {
        let mut session = SdpSession::new("SMPTE ST 2110 Stream", self.config.source_ip);

        match self.config.stream_type {
            StreamType::Video => {
                if let Some(video_cfg) = &self.video_config {
                    session.add_video_media(self.config.destination, self.config.port, video_cfg);
                }
            }
            StreamType::Audio => {
                if let Some(audio_cfg) = &self.audio_config {
                    session.add_audio_media(self.config.destination, self.config.port, audio_cfg);
                }
            }
            StreamType::Ancillary => {
                if let Some(anc_cfg) = &self.ancillary_config {
                    session.add_ancillary_media(self.config.destination, self.config.port, anc_cfg);
                }
            }
        }

        session.to_string()
    }

    /// Gets the destination socket address.
    #[must_use]
    pub fn destination(&self) -> SocketAddr {
        SocketAddr::new(self.config.destination, self.config.port)
    }
}

/// ST 2110 sink for receiving media.
#[derive(Debug)]
pub struct St2110Sink {
    /// Configuration.
    config: St2110Config,
    /// RTP session.
    rtp_session: Arc<RwLock<RtpSession>>,
    /// PTP clock (if enabled).
    ptp_clock: Option<Arc<RwLock<PtpClock>>>,
    /// Video decoder state (if video stream).
    video_decoder: Option<video::VideoDecoder>,
    /// Audio decoder state (if audio stream).
    audio_decoder: Option<audio::AudioDecoder>,
}

impl St2110Sink {
    /// Creates a new ST 2110 video sink.
    #[must_use]
    pub fn new_video(video_config: VideoConfig) -> Self {
        let config = St2110Config {
            stream_type: StreamType::Video,
            ..Default::default()
        };

        let clock_rate = 90000;
        let rtp_session = Arc::new(RwLock::new(RtpSession::new(config.ssrc, clock_rate)));

        let ptp_clock = if config.enable_ptp {
            let clock_id = ptp::ClockIdentity::new([0; 8]);
            let port_id = ptp::PortIdentity::new(clock_id, 1);
            Some(Arc::new(RwLock::new(PtpClock::new(
                port_id,
                config.ptp_domain,
            ))))
        } else {
            None
        };

        Self {
            config,
            rtp_session,
            ptp_clock,
            video_decoder: Some(video::VideoDecoder::new(video_config)),
            audio_decoder: None,
        }
    }

    /// Creates a new ST 2110 audio sink.
    #[must_use]
    pub fn new_audio(audio_config: AudioConfig) -> Self {
        let config = St2110Config {
            stream_type: StreamType::Audio,
            ..Default::default()
        };

        let clock_rate = audio_config.sample_rate as u32;
        let rtp_session = Arc::new(RwLock::new(RtpSession::new(config.ssrc, clock_rate)));

        let ptp_clock = if config.enable_ptp {
            let clock_id = ptp::ClockIdentity::new([0; 8]);
            let port_id = ptp::PortIdentity::new(clock_id, 1);
            Some(Arc::new(RwLock::new(PtpClock::new(
                port_id,
                config.ptp_domain,
            ))))
        } else {
            None
        };

        Self {
            config,
            rtp_session,
            ptp_clock,
            video_decoder: None,
            audio_decoder: Some(audio::AudioDecoder::new(audio_config)),
        }
    }

    /// Processes a received RTP packet.
    pub fn process_packet(&mut self, data: Bytes) -> NetResult<()> {
        let packet = RtpPacket::parse(data)?;

        // Update RTP statistics
        self.rtp_session.write().process_packet(&packet);

        // Process based on stream type
        match self.config.stream_type {
            StreamType::Video => {
                if let Some(decoder) = &mut self.video_decoder {
                    decoder.process_rtp_packet(&packet)?;
                }
            }
            StreamType::Audio => {
                if let Some(decoder) = &mut self.audio_decoder {
                    decoder.process_rtp_packet(&packet)?;
                }
            }
            StreamType::Ancillary => {
                // Process ancillary data
            }
        }

        Ok(())
    }

    /// Gets the stream configuration.
    #[must_use]
    pub const fn config(&self) -> &St2110Config {
        &self.config
    }

    /// Checks if PTP is synchronized.
    #[must_use]
    pub fn is_ptp_synced(&self) -> bool {
        self.ptp_clock
            .as_ref()
            .map_or(false, |clock| clock.read().is_synchronized())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_st2110_config_default() {
        let config = St2110Config::default();
        assert_eq!(config.stream_type, StreamType::Video);
        assert_eq!(config.port, 5004);
        assert_eq!(config.timing_mode, TimingMode::Wide);
    }

    #[test]
    fn test_video_source_creation() {
        let video_config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            ..Default::default()
        };

        let source = St2110Source::new_video(video_config);
        assert_eq!(source.config().stream_type, StreamType::Video);
        assert!(source.video_config().is_some());
    }

    #[test]
    fn test_audio_source_creation() {
        let audio_config = AudioConfig {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            ..Default::default()
        };

        let source = St2110Source::new_audio(audio_config);
        assert_eq!(source.config().stream_type, StreamType::Audio);
        assert!(source.audio_config().is_some());
    }

    #[test]
    fn test_sdp_generation() {
        let video_config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            ..Default::default()
        };

        let source = St2110Source::new_video(video_config);
        let sdp = source.generate_sdp();

        assert!(sdp.contains("v=0"));
        assert!(sdp.contains("m=video"));
    }
}
