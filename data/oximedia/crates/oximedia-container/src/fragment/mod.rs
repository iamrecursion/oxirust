//! Fragmented container support.
//!
//! Provides fragmented MP4 and segment writing for adaptive streaming.

#![forbid(unsafe_code)]

pub mod init;
pub mod mp4;
#[cfg(not(target_arch = "wasm32"))]
pub mod segment;

pub use init::InitSegmentBuilder;
pub use mp4::{
    CmafChunk, CmafChunkBuilder, CmafChunkType, CmafConfig, FragmentBoundary,
    FragmentBoundaryDetector, FragmentType, FragmentedMp4Builder, FragmentedMp4Config,
    FragmentedMp4Ingest, FragmentedTrack, IngestResult, Mp4Fragment,
};
#[cfg(not(target_arch = "wasm32"))]
pub use segment::{DashManifestGenerator, SegmentInfo, SegmentWriter, SegmentWriterConfig};
