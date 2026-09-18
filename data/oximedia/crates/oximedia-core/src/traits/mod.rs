//! Traits for multimedia components.
//!
//! This module provides traits that define the interfaces for:
//!
//! - [`VideoDecoder`] - Video decoder implementations
//! - [`AudioDecoder`] - Audio decoder implementations
//! - [`Demuxer`] - Container format demuxers
//!
//! # Design Philosophy
//!
//! These traits use a push-pull model similar to `FFmpeg`'s libavcodec:
//!
//! 1. Push compressed data with `send_packet()`
//! 2. Pull decoded frames with `receive_frame()`
//!
//! This allows decoders to buffer frames internally when needed.

mod decoder;
mod demuxer;

pub use decoder::{
    AudioDecoder, AudioFrame, HorizontalAlign, SubtitleDecoder, SubtitleFrame, SubtitleSettings,
    VerticalAlign, VideoDecoder, VideoFrame,
};
pub use demuxer::{AudioStreamInfo, ContainerInfo, Demuxer, Packet, StreamInfo, VideoStreamInfo};
