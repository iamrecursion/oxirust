//! DASH (Dynamic Adaptive Streaming over HTTP) server implementation.
//!
//! This module provides a complete DASH server with support for:
//! - MPD (Media Presentation Description) generation
//! - Segment templates
//! - Multi-period support
//! - Low latency DASH
//! - Multiple representations

pub mod ll_dash;
pub mod mpd;
pub mod server;

pub use ll_dash::LlDashConfig;
pub use mpd::{MpdBuilder, Representation};
pub use server::DashServer;
