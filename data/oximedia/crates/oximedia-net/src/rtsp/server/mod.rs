//! RTSP 1.0 server implementation.
//!
//! Provides a complete async RTSP server that:
//! - Accepts TCP connections on a configurable bind address
//! - Handles OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, GET_PARAMETER
//! - Forwards RTP packets to playing clients via TCP-interleaved transport
//! - Manages per-connection session state (Init → Ready → Playing → Paused)
//! - Exposes a [`MountPointRegistry`] for registering stream sources
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use oximedia_net::rtsp::server::{
//!     RtspServer, RtspServerConfig, MountPoint, MountPointRegistry,
//! };
//! use oximedia_net::rtsp::SessionDescription;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let server = RtspServer::new(RtspServerConfig {
//!     bind_address: "0.0.0.0:8554".into(),
//!     ..Default::default()
//! });
//!
//! // Register a stream source.
//! let sdp = SessionDescription::for_rtsp_stream(
//!     "127.0.0.1", 96, "H264", 90000, None, None, None, None,
//! ).to_string();
//! let (mp, _rx) = MountPoint::new("/stream".into(), sdp);
//! let mp = server.registry().register(mp);
//!
//! // Publish RTP packets (e.g. from an encoder thread):
//! // mp.publish(Arc::new(rtp_bytes));
//! let _ = mp;
//!
//! server.run().await?;
//! # Ok(()) }
//! ```

mod auth_server;
mod connection;
mod registry;
mod rtsp_server;
mod state;

pub use auth_server::ServerChallenge;
pub use connection::ServerConnection;
pub use registry::{MountPoint, MountPointRegistry};
pub use rtsp_server::RtspServer;
pub use state::{RtspServerConfig, RtspSession, RtspSessionState};
