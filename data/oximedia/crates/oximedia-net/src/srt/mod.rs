//! SRT (Secure Reliable Transport) protocol implementation.
//!
//! This module provides a comprehensive implementation of the SRT protocol for
//! low-latency, secure, and reliable video streaming over UDP.
//!
//! # Key Types
//!
//! - [`SrtPacket`] - SRT packet structure (data or control)
//! - [`ControlPacket`] - Control packet types (handshake, ACK, NAK, etc.)
//! - [`DataPacket`] - Data packet with sequence numbers and timestamps
//! - [`SrtSocket`] - SRT socket state machine
//! - [`SrtConnection`] - High-level connection with async I/O
//! - [`SrtConfig`] - Configuration options
//!
//! # Protocol Features
//!
//! SRT provides UDP with reliability features:
//! - **Automatic Repeat Request (ARQ)** - Packet retransmission on loss
//! - **Congestion Control** - AIMD-based window management
//! - **Encryption** - AES-128/192/256 payload encryption
//! - **Loss Recovery** - Out-of-order packet handling and gap detection
//! - **Latency Control** - Configurable buffering and delivery
//!
//! # Architecture
//!
//! The implementation is organized into several modules:
//!
//! - `packet` - Packet encoding/decoding
//! - `socket` - State machine and protocol logic
//! - `connection` - UDP transport and async I/O
//! - `congestion` - Congestion control algorithm
//! - `crypto` - AES encryption
//! - `loss` - Loss detection and tracking
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::srt::{SrtConnection, SrtConfig};
//! use std::net::SocketAddr;
//!
//! async fn stream_video() -> NetResult<()> {
//!     let local_addr: SocketAddr = "0.0.0.0:0".parse().expect("valid addr");
//!     let peer_addr: SocketAddr = "127.0.0.1:9000".parse().expect("valid addr");
//!
//!     let config = SrtConfig::new()
//!         .with_latency(120)
//!         .with_passphrase("secret");
//!
//!     let conn = SrtConnection::new(local_addr, peer_addr, config).await?;
//!     conn.connect(std::time::Duration::from_secs(3)).await?;
//!
//!     conn.send(b"Hello, SRT!").await?;
//!
//!     let mut buf = vec![0u8; 1316];
//!     let len = conn.recv(&mut buf).await?;
//!
//!     conn.close().await?;
//!     Ok(())
//! }
//! ```

mod congestion;
mod connection;
pub mod connection_mode;
mod crypto;
pub mod hsv5;
pub mod key_exchange;
mod loss;
mod monitor;
mod packet;
mod socket;
mod stats;
mod stream;

pub use congestion::CongestionControl;
pub use connection::SrtConnection;
pub use connection_mode::{
    CallerState, ConnectionMode, ListenerState, PendingConnection, RendezvousPhase, RendezvousState,
};
pub use crypto::{
    derive_session_key, AesContext, KeyMaterial, KeyMaterialPacket, KeySchedule, PassphraseAuth,
    SrtCryptoContext, SrtPacketBuffer,
};
pub use hsv5::{SrtEncryption, SrtExtensionBlock, SrtHandshake, SrtPacketType, SrtStreamConfig};
pub use key_exchange::KeyMaterial as KmxKeyMaterial;
pub use key_exchange::{EncryptionSession, EncryptionState, KwAlgorithm};
pub use loss::{LossList, LossRange, ReceiveBuffer};
pub use monitor::{
    BandwidthEstimator, ConnectionMonitor, JitterCalculator, LossRateEstimator, QualityMetrics,
};
pub use packet::{
    ControlPacket, ControlType, DataPacket, EncryptionFlag, HandshakeExtension, HandshakeInfo,
    PacketFlags, PacketPosition, SrtPacket,
};
pub use socket::{ConnectionState, SrtConfig, SrtSocket, SrtStats};
pub use stats::{BufferStats, DirectionStats, RttStats, SrtStreamStats, StreamQuality};
pub use stream::{SrtListener, SrtMultiplexer, SrtReceiver, SrtSender};
