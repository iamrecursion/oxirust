//! Streaming protocol encoding presets.
//!
//! This module contains presets for adaptive bitrate streaming:
//! - HLS: HTTP Live Streaming (Apple)
//! - DASH: Dynamic Adaptive Streaming over HTTP (MPEG)
//! - SmoothStreaming: Microsoft Smooth Streaming
//! - RTMP: Real-Time Messaging Protocol
//! - SRT: Secure Reliable Transport

pub mod dash;
pub mod hls;
pub mod rtmp;
pub mod smooth;
pub mod srt;
