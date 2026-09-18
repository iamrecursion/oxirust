//! Network streaming for `OxiMedia`.
//!
//! This crate provides network streaming protocols for the `OxiMedia` multimedia
//! framework. It supports various streaming protocols including:
//!
//! - **HLS** (HTTP Live Streaming) - Apple's adaptive streaming protocol
//! - **DASH** (Dynamic Adaptive Streaming over HTTP) - MPEG-DASH streaming
//! - **RTMP** (Real-Time Messaging Protocol) - Flash streaming protocol
//! - **SRT** (Secure Reliable Transport) - Low-latency streaming
//! - **WebRTC** - Real-time browser communication
//! - **SMPTE ST 2110** - Professional media over IP (uncompressed video/audio/ANC)
//! - **CDN** - Multi-CDN failover and load balancing
//!
//! # Overview
//!
//! Each streaming protocol module provides:
//! - Protocol-specific packet/message types
//! - Parsing and serialization
//! - Session management
//! - Adaptive bitrate support where applicable
//!
//! The CDN module provides:
//! - Multi-CDN provider support (Cloudflare, Fastly, Akamai, CloudFront, Custom)
//! - Real-time health monitoring
//! - Automatic failover with circuit breaker pattern
//! - Intelligent routing strategies
//! - Performance metrics and SLA monitoring
//!
//! The SMPTE ST 2110 module provides:
//! - Uncompressed video transport (ST 2110-20)
//! - PCM audio transport (ST 2110-30)
//! - Ancillary data transport (ST 2110-40)
//! - PTP synchronization (IEEE 1588)
//! - SDP session description
//! - Broadcast-quality professional media over IP
//!
//! # Example
//!
//! ```no_run
//! use oximedia_net::hls::MasterPlaylist;
//! use oximedia_net::error::NetResult;
//!
//! fn main() -> NetResult<()> {
//!     // Parse an HLS master playlist from a string
//!     let hls_data = concat!(
//!         "#EXTM3U\n",
//!         "#EXT-X-VERSION:3\n",
//!         "#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=1280x720\n",
//!         "720p.m3u8\n",
//!     );
//!     let master = MasterPlaylist::parse(hls_data)?;
//!     println!("HLS: {} variant stream(s)", master.variants.len());
//!     Ok(())
//! }
//! ```
//!
//! # Protocol Support Matrix
//!
//! | Protocol | Latency | Reliability | Encryption | ABR | Typical Use |
//! |---|---|---|---|---|---|
//! | HLS | 6–30 s | High (TCP/HTTP) | HTTPS, AES-128 | Yes | VOD + live CDN distribution |
//! | DASH | 3–30 s | High (TCP/HTTP) | HTTPS, ClearKey | Yes | VOD + live CDN (MPEG standard) |
//! | RTMP | 0.5–3 s | Medium (TCP) | RTMPS (TLS) | No | Ingest / contribution |
//! | SRT | 0.1–1 s | High (UDP+ARQ) | AES-128/256-GCM | No | Broadcast contribution, WAN |
//! | WebRTC | < 100 ms | Medium (DTLS/ICE) | DTLS-SRTP (mandatory) | Via REMB | Browser real-time, conferencing |
//! | SMPTE ST 2110 | < 1 ms | High (PTP-synced) | None native | No | Professional broadcast studio LAN |
//! | RIST | 0.1–0.5 s | High (UDP+ARQ) | AES-128 | No | Broadcast transport, WAN |
//! | QUIC | < 50 ms | High (QUIC/HTTP3) | TLS 1.3 (mandatory) | Yes | Next-gen ABR streaming |
//!
//! # Protocol Selection Guide
//!
//! Choose the right streaming protocol for your use case:
//!
//! ## CDN/ABR Distribution — HLS or DASH
//!
//! **HLS** is the best choice for Apple devices (iPhone, iPad, Apple TV) and
//! any CDN-based adaptive bitrate delivery. Its wide CDN support, proven
//! compatibility, and LL-HLS variant for low latency (2–4 s) make it the
//! default for OTT and broadcaster VOD.
//!
//! **DASH** (MPEG-DASH) is the MPEG standard equivalent — better suited for
//! Android and Smart TV ecosystems, or when interoperability with DVB/HbbTV
//! is required. LL-DASH can achieve sub-4-second latency via chunked transfer.
//!
//! ## Ultra-Low Latency — SRT, WebRTC, or QUIC
//!
//! **SRT** (Secure Reliable Transport) is ideal for professional contribution
//! links over the public internet: it achieves sub-second latency with ARQ
//! retransmission and optional AES-256-GCM encryption. Use it for encoder to
//! production switcher transport, remote sports feeds, and cloud ingest.
//!
//! **WebRTC** is browser-native and achieves sub-100 ms latency via UDP with
//! ICE/STUN/TURN NAT traversal and mandatory DTLS-SRTP encryption. Best for
//! interactive video (video conferencing, watch parties, WHIP/WHEP ingest).
//!
//! **QUIC** (HTTP/3 Datagrams) combines the deployment advantages of HTTP
//! with near-UDP latency and TLS 1.3. Use for next-generation ABR streaming
//! where head-of-line blocking in HTTP/2 is a concern.
//!
//! ## Broadcast Contribution — SMPTE ST 2110 or RIST
//!
//! **SMPTE ST 2110** carries uncompressed video (ST 2110-20), PCM audio
//! (ST 2110-30), and ancillary data (ST 2110-40) over PTP-synchronized
//! IP fabric. Sub-millisecond latency on a dedicated LAN; not suitable for
//! internet transport. Use in broadcast studios and production facilities.
//!
//! **RIST** is an SMPTE-standardized protocol that adds ARQ reliability over
//! UDP, similar to SRT but with broader vendor support. Good for long-haul
//! broadcast contribution when SRT is unavailable.
//!
//! # CDN Configuration
//!
//! Configure multi-CDN failover with automatic circuit breaking. The
//! [`cdn::FailoverManager`] opens a provider's circuit after a configurable
//! failure threshold and transparently routes to the next provider in the
//! fallback chain until the primary recovers.
//!
//! ```no_run
//! use oximedia_net::cdn::{CdnManager, CdnProvider, CdnConfig};
//!
//! fn main() {
//!     let config = CdnConfig::default();
//!     let manager = CdnManager::new(config);
//!
//!     // Register Cloudflare as primary (highest priority)
//!     manager.add_provider(CdnProvider::cloudflare("https://cdn.example.com", 100));
//!
//!     // Register Fastly as secondary fallback
//!     manager.add_provider(CdnProvider::fastly("https://fastly.example.com", 80));
//!
//!     // The failover module will automatically open the Cloudflare circuit
//!     // after 5 consecutive failures and route requests to Fastly until
//!     // Cloudflare recovers (circuit moves to half-open after 60 s).
//! }
//! ```

#![warn(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    dead_code,
    clippy::pedantic,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_map_unit_fn
)]

pub mod abr;
pub mod abr_buffer;
pub mod bandwidth_adaptation;
pub mod bandwidth_estimator;
pub mod bandwidth_probe;
pub mod bandwidth_throttle;
pub mod bandwidth_trigger;
pub mod buffer_model;
pub mod cdn;
pub mod connection_pool;
pub mod dash;
pub mod error;
pub mod fec;
pub mod fec_interleave;
pub mod flow_control;
pub mod hls;
pub mod http2;
pub mod ice;
pub mod live;
pub mod ll_dash;
pub mod ll_dash_config;
pub mod manifest_cache;
pub mod mdns;
pub mod multicast;
pub mod multicast_manager;
pub mod multipath;
pub mod network_path;
pub mod network_simulator;
pub mod pacing;
pub mod packet_buffer;
pub mod playlist_parser;
pub mod protocol_detect;
pub mod qos_monitor;
pub mod quic;
pub mod quic_datagram;
pub mod relay;
pub mod retry_policy;
pub mod rist;
pub mod rtmp;
pub mod rtp_session;
pub mod rtsp;
pub mod session_tracker;
pub mod smpte2022_7;
pub mod smpte2110;
pub mod srt;
pub mod srt_aes256gcm;
pub mod srt_config;
pub mod srt_group;
pub mod srt_pacing;
pub mod stream_health_monitor;
pub mod stream_mux;
pub mod tls_provider;
pub mod webrtc;
pub mod websocket;
pub mod whep_client;
pub mod whip;
pub mod whip_whep;
pub mod zero_copy_serve;
pub mod zixi;

// Re-export commonly used items
pub use error::{NetError, NetResult};

// Re-export the process-wide Pure-Rust TLS crypto provider bootstrap — call
// this once, as early as possible, from any binary/library entry point that
// may open a TLS connection. See [`tls_provider`] for details.
pub use tls_provider::install_default_crypto_provider;

// Re-export SRT stats and key exchange types
pub use srt::{DirectionStats, RttStats, SrtStreamStats, StreamQuality};
pub use srt::{EncryptionSession, EncryptionState, KmxKeyMaterial, KwAlgorithm};

// Re-export streaming ABR types
pub use abr::streaming::{
    AbrBandwidthEstimator as BandwidthEstimator, AbrController, AbrSwitchReason, AbrVariant,
    BandwidthSample, BufferedSegment, SegmentFetcher, SelectionResult,
};
