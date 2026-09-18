//! Streaming demuxing and muxing.
//!
//! This module provides streaming capabilities for both demuxing and muxing,
//! optimized for live streaming and network sources. Includes CMAF chunked
//! transfer encoding for low-latency delivery.

#![forbid(unsafe_code)]

pub mod demux;
pub mod mux;

#[cfg(not(target_arch = "wasm32"))]
pub use demux::{spawn_demuxer, PacketReceiver};
pub use demux::{ProgressiveBuffer, StreamingDemuxer, StreamingDemuxerConfig, StreamingState};
#[cfg(not(target_arch = "wasm32"))]
pub use mux::{spawn_muxer, PacketSender, StreamingMuxer};
pub use mux::{
    ChunkSample, CmafChunk, CmafChunkMode, CmafChunkOwned, CmafChunkWriter, CmafChunkedConfig,
    CmafChunkedEncoder, CmafSample, CmafSegment, LatencyMonitor, MuxingStats, StreamingMuxerConfig,
};
