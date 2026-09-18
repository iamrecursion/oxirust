//! DASH live streaming server implementation.
//!
//! This module provides a complete implementation of MPEG-DASH live streaming
//! with support for:
//!
//! - Dynamic MPD generation with SegmentTimeline
//! - On-the-fly segment creation from live input
//! - Low-latency DASH (LL-DASH) with chunked transfer
//! - DVR/time-shift buffer management
//! - Multi-quality representations
//! - Adaptive bitrate streaming support
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::dash::live::{DashLiveServer, DashLiveConfig};
//! use std::time::Duration;
//!
//! async fn run_live_server() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = DashLiveConfig {
//!         segment_duration: Duration::from_secs(2),
//!         min_buffer_time: Duration::from_secs(4),
//!         time_shift_buffer: Duration::from_secs(60),
//!         low_latency: true,
//!     };
//!
//!     let mut server = DashLiveServer::start(config).await?;
//!
//!     // Ingest live packets
//!     loop {
//!         let packet = receive_packet().await?;
//!         server.ingest_packet(packet).await?;
//!     }
//! }
//! ```

mod chunked;
mod dvr;
mod mpd_gen;
mod segment;
mod timeline;

pub use chunked::{Chunk, ChunkCoordinator, ChunkedConfig, ChunkedTransfer, ProducerReferenceTime};
pub use dvr::{DvrBuffer, DvrSegment, DvrStats};
pub use mpd_gen::{AdaptationSetBuilder, DynamicMpdGenerator, MpdConfig, RepresentationBuilder};
pub use segment::{
    CodecInfo, GeneratedSegment, LiveSegmentGenerator, MultiRepresentationGenerator,
    SegmentAlignment,
};
pub use timeline::{SegmentInfo, TimelineManager};

use crate::dash::mpd::{Representation, SegmentTemplate};
use crate::error::{NetError, NetResult};
use bytes::Bytes;
use oximedia_container::Packet;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Configuration for DASH live streaming.
#[derive(Debug, Clone)]
pub struct DashLiveConfig {
    /// Target segment duration.
    pub segment_duration: Duration,
    /// Minimum buffer time.
    pub min_buffer_time: Duration,
    /// Time-shift buffer depth (DVR window).
    pub time_shift_buffer: Duration,
    /// Enable low-latency mode.
    pub low_latency: bool,
}

impl Default for DashLiveConfig {
    fn default() -> Self {
        Self {
            segment_duration: Duration::from_secs(2),
            min_buffer_time: Duration::from_secs(4),
            time_shift_buffer: Duration::from_secs(60),
            low_latency: false,
        }
    }
}

/// DASH live streaming server.
///
/// The server manages:
/// - Multiple quality representations
/// - Segment generation and buffering
/// - Dynamic MPD updates
/// - DVR buffer management
/// - Optional low-latency chunked delivery
pub struct DashLiveServer {
    /// Server configuration.
    config: DashLiveConfig,
    /// Internal state.
    state: Arc<RwLock<ServerState>>,
}

/// Internal server state.
struct ServerState {
    /// MPD generator.
    mpd_generator: DynamicMpdGenerator,
    /// Segment generators by representation ID.
    segment_generators: MultiRepresentationGenerator,
    /// Timeline managers by representation ID.
    timelines: HashMap<String, TimelineManager>,
    /// DVR buffers by representation ID.
    dvr_buffers: HashMap<String, DvrBuffer>,
    /// Chunk coordinator (if low-latency enabled).
    chunk_coordinator: Option<ChunkCoordinator>,
    /// Stream availability start time.
    availability_start: SystemTime,
    /// Current period ID.
    current_period_id: String,
    /// Representation metadata.
    representations: HashMap<String, RepresentationMetadata>,
    /// Segment number mapping (representation -> current segment).
    segment_numbers: HashMap<String, u64>,
}

/// Metadata for a representation.
#[derive(Debug, Clone)]
struct RepresentationMetadata {
    /// Representation ID.
    id: String,
    /// Bandwidth in bits per second.
    bandwidth: u64,
    /// Codec information.
    codec: CodecInfo,
    /// Timescale.
    timescale: u32,
    /// Adaptation set ID.
    adaptation_set_id: u32,
    /// Width (for video).
    width: Option<u32>,
    /// Height (for video).
    height: Option<u32>,
}

impl DashLiveServer {
    /// Starts a new DASH live streaming server.
    ///
    /// # Arguments
    ///
    /// * `config` - Server configuration
    ///
    /// # Returns
    ///
    /// A running DASH live server
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub async fn start(config: DashLiveConfig) -> NetResult<Self> {
        let availability_start = SystemTime::now();

        let mpd_config = MpdConfig {
            min_buffer_time: config.min_buffer_time,
            suggested_presentation_delay: if config.low_latency {
                Duration::from_secs(2)
            } else {
                Duration::from_secs(6)
            },
            time_shift_buffer_depth: config.time_shift_buffer,
            availability_start_time: availability_start,
            minimum_update_period: config.segment_duration,
        };

        let mut mpd_generator = DynamicMpdGenerator::new(mpd_config);

        // Add UTCTiming element
        mpd_generator.add_utc_timing(
            "urn:mpeg:dash:utc:http-iso:2014".to_string(),
            "https://time.akamai.com/?iso".to_string(),
        );

        // Create initial period
        let period_id = mpd_generator.add_period(None, None);

        let chunk_coordinator = if config.low_latency {
            Some(ChunkCoordinator::new(Duration::from_secs(2)))
        } else {
            None
        };

        let state = ServerState {
            mpd_generator,
            segment_generators: MultiRepresentationGenerator::new(),
            timelines: HashMap::new(),
            dvr_buffers: HashMap::new(),
            chunk_coordinator,
            availability_start,
            current_period_id: period_id,
            representations: HashMap::new(),
            segment_numbers: HashMap::new(),
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Adds a representation to the live stream.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique representation identifier
    /// * `bandwidth` - Bandwidth in bits per second
    /// * `codec` - Codec information
    /// * `timescale` - Timescale in units per second
    /// * `width` - Video width (optional)
    /// * `height` - Video height (optional)
    ///
    /// # Errors
    ///
    /// Returns an error if the representation cannot be added
    pub async fn add_representation(
        &mut self,
        id: impl Into<String>,
        bandwidth: u64,
        codec: CodecInfo,
        timescale: u32,
        width: Option<u32>,
        height: Option<u32>,
    ) -> NetResult<()> {
        let id = id.into();
        let mut state = self.state.write().await;

        // Determine adaptation set ID based on codec type
        let adaptation_set_id = if codec.is_video { 0 } else { 1 };

        // Create segment generator
        let segment_generator =
            LiveSegmentGenerator::new(&id, timescale, self.config.segment_duration, codec.clone());

        state
            .segment_generators
            .add_representation(segment_generator);

        // Create timeline manager
        let timeline = TimelineManager::new(
            timescale,
            self.config.segment_duration,
            state.availability_start,
        );
        state.timelines.insert(id.clone(), timeline);

        // Create DVR buffer
        let dvr_buffer = DvrBuffer::new(self.config.time_shift_buffer);
        state.dvr_buffers.insert(id.clone(), dvr_buffer);

        // Add to chunk coordinator if enabled
        if let Some(ref mut coordinator) = state.chunk_coordinator {
            coordinator.add_representation(id.clone(), ChunkedConfig::default());
        }

        // Store representation metadata
        let metadata = RepresentationMetadata {
            id: id.clone(),
            bandwidth,
            codec: codec.clone(),
            timescale,
            adaptation_set_id,
            width,
            height,
        };
        state.representations.insert(id.clone(), metadata);
        state.segment_numbers.insert(id.clone(), 1);

        // Update MPD
        self.update_mpd_representations(&mut state).await?;

        Ok(())
    }

    /// Ingests a packet into the live stream.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - The representation to add the packet to
    /// * `packet` - The packet to ingest
    ///
    /// # Errors
    ///
    /// Returns an error if the packet cannot be processed
    pub async fn ingest_packet(
        &mut self,
        representation_id: impl AsRef<str>,
        packet: Packet,
    ) -> NetResult<()> {
        let representation_id = representation_id.as_ref();
        let mut state = self.state.write().await;

        // Add packet to segment generator
        if let Some(segment) = state
            .segment_generators
            .add_packet(representation_id, packet)
        {
            // Segment completed
            self.handle_completed_segment(&mut state, representation_id, segment)
                .await?;
        }

        Ok(())
    }

    /// Retrieves the current MPD manifest.
    ///
    /// # Returns
    ///
    /// The MPD as an XML string
    pub async fn get_mpd(&self) -> String {
        let mut state = self.state.write().await;
        state.mpd_generator.generate_xml()
    }

    /// Retrieves a segment by number and representation ID.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - Representation identifier
    /// * `segment_number` - Segment number
    ///
    /// # Returns
    ///
    /// The segment data if found
    ///
    /// # Errors
    ///
    /// Returns an error if the segment is not found
    pub async fn get_segment(
        &self,
        representation_id: &str,
        segment_number: u64,
    ) -> NetResult<Bytes> {
        let state = self.state.read().await;

        let dvr_buffer = state.dvr_buffers.get(representation_id).ok_or_else(|| {
            NetError::not_found(format!("Representation not found: {representation_id}"))
        })?;

        let segment = dvr_buffer
            .get_segment(segment_number, representation_id)
            .ok_or_else(|| NetError::not_found(format!("Segment {segment_number} not found")))?;

        Ok(segment.data.clone())
    }

    /// Retrieves the initialization segment for a representation.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - Representation identifier
    ///
    /// # Returns
    ///
    /// The initialization segment data
    ///
    /// # Errors
    ///
    /// Returns an error if the initialization segment is not found
    pub async fn get_init_segment(&self, representation_id: &str) -> NetResult<Bytes> {
        let state = self.state.read().await;

        let generator = state
            .segment_generators
            .generator(representation_id)
            .ok_or_else(|| {
                NetError::not_found(format!("Representation not found: {representation_id}"))
            })?;

        generator
            .init_segment()
            .cloned()
            .ok_or_else(|| NetError::not_found("Initialization segment not generated"))
    }

    /// Generates initialization segments for all representations.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails
    pub async fn generate_init_segments(&mut self) -> NetResult<()> {
        let mut state = self.state.write().await;

        let repr_ids: Vec<String> = state.representations.keys().cloned().collect();

        for repr_id in repr_ids {
            if let Some(generator) = state.segment_generators.generator_mut(&repr_id) {
                generator.generate_init_segment(None);
            }
        }

        Ok(())
    }

    /// Retrieves a chunk for low-latency streaming.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - Representation identifier
    /// * `segment_number` - Segment number
    /// * `chunk_sequence` - Chunk sequence number
    ///
    /// # Returns
    ///
    /// The chunk data if found
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk is not found or low-latency mode is disabled
    pub async fn get_chunk(
        &self,
        representation_id: &str,
        _segment_number: u64,
        chunk_sequence: u32,
    ) -> NetResult<Chunk> {
        let state = self.state.read().await;

        let coordinator = state
            .chunk_coordinator
            .as_ref()
            .ok_or_else(|| NetError::invalid_state("Low-latency mode not enabled"))?;

        let transfer = coordinator.transfer(representation_id).ok_or_else(|| {
            NetError::not_found(format!("Representation not found: {representation_id}"))
        })?;

        // Note: In a real implementation, we'd need to track which segment's chunks are available
        transfer
            .get_chunk(chunk_sequence)
            .cloned()
            .ok_or_else(|| NetError::not_found(format!("Chunk {chunk_sequence} not found")))
    }

    /// Returns server statistics.
    #[must_use]
    pub async fn stats(&self) -> ServerStats {
        let state = self.state.read().await;

        let mut representation_stats = Vec::new();

        for (repr_id, dvr_buffer) in &state.dvr_buffers {
            let timeline = state.timelines.get(repr_id);
            let current_segment = state.segment_numbers.get(repr_id).copied().unwrap_or(0);

            representation_stats.push(RepresentationStats {
                id: repr_id.clone(),
                current_segment_number: current_segment,
                buffered_segments: dvr_buffer.segment_count(),
                buffer_size: dvr_buffer.total_size(),
                current_time: timeline.map(|t| t.current_time_secs()).unwrap_or(0.0),
            });
        }

        ServerStats {
            uptime: SystemTime::now()
                .duration_since(state.availability_start)
                .unwrap_or(Duration::ZERO),
            representation_count: state.representations.len(),
            representations: representation_stats,
            low_latency_enabled: state.chunk_coordinator.is_some(),
        }
    }

    /// Forces segment finalization for all representations.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails
    pub async fn finalize_segments(&mut self) -> NetResult<()> {
        let mut state = self.state.write().await;

        let segments = state.segment_generators.finalize_all();

        for (repr_id, segment) in segments {
            self.handle_completed_segment(&mut state, &repr_id, segment)
                .await?;
        }

        Ok(())
    }

    /// Handles a completed segment.
    async fn handle_completed_segment(
        &self,
        state: &mut ServerState,
        representation_id: &str,
        segment: GeneratedSegment,
    ) -> NetResult<()> {
        let metadata = state
            .representations
            .get(representation_id)
            .ok_or_else(|| {
                NetError::not_found(format!("Representation not found: {representation_id}"))
            })?;

        // Update timeline
        if let Some(timeline) = state.timelines.get_mut(representation_id) {
            let duration = Duration::from_secs_f64(segment.duration_secs(metadata.timescale));
            timeline.add_segment(duration);

            // Trim old segments based on DVR window
            timeline.trim_old_segments(self.config.time_shift_buffer);

            // Update MPD timeline
            state.mpd_generator.update_timeline(
                &state.current_period_id,
                metadata.adaptation_set_id,
                representation_id,
                timeline.to_segment_timeline(),
            );
        }

        // Add to DVR buffer
        if let Some(dvr_buffer) = state.dvr_buffers.get_mut(representation_id) {
            let dvr_segment = DvrSegment::new(
                segment.number,
                representation_id,
                segment.data.clone(),
                Duration::from_secs_f64(segment.start_time_secs(metadata.timescale)),
                Duration::from_secs_f64(segment.duration_secs(metadata.timescale)),
                metadata.timescale,
            );
            dvr_buffer.add_segment(dvr_segment);
        }

        // Update segment number tracking
        state
            .segment_numbers
            .insert(representation_id.to_string(), segment.number + 1);

        Ok(())
    }

    /// Updates MPD representations.
    async fn update_mpd_representations(&self, state: &mut ServerState) -> NetResult<()> {
        // Group representations by adaptation set
        let mut video_reprs = Vec::new();
        let mut audio_reprs = Vec::new();

        for metadata in state.representations.values() {
            let mut repr = Representation::new(&metadata.id, metadata.bandwidth);
            repr.codecs = Some(metadata.codec.codec.clone());
            repr.mime_type = Some(metadata.codec.mime_type.clone());

            if metadata.codec.is_video {
                repr.width = metadata.width;
                repr.height = metadata.height;
            }

            // Create segment template
            let template = SegmentTemplate::new(metadata.timescale)
                .with_media("$RepresentationID$/$Number$.m4s")
                .with_initialization("$RepresentationID$/init.mp4");

            repr.segment_template = Some(template);

            if metadata.codec.is_video {
                video_reprs.push(repr);
            } else {
                audio_reprs.push(repr);
            }
        }

        // Clear existing adaptation sets
        if let Some(period) = state.mpd_generator.current_period_mut() {
            period.adaptation_sets.clear();

            // Add video adaptation set
            if !video_reprs.is_empty() {
                let mut video_as = AdaptationSetBuilder::new()
                    .id(0)
                    .content_type("video")
                    .mime_type("video/mp4")
                    .segment_alignment(true)
                    .build();

                video_as.representations = video_reprs;
                period.adaptation_sets.push(video_as);
            }

            // Add audio adaptation set
            if !audio_reprs.is_empty() {
                let mut audio_as = AdaptationSetBuilder::new()
                    .id(1)
                    .content_type("audio")
                    .mime_type("audio/mp4")
                    .segment_alignment(true)
                    .build();

                audio_as.representations = audio_reprs;
                period.adaptation_sets.push(audio_as);
            }
        }

        Ok(())
    }
}

/// Server statistics.
#[derive(Debug, Clone)]
pub struct ServerStats {
    /// Uptime since stream start.
    pub uptime: Duration,
    /// Number of representations.
    pub representation_count: usize,
    /// Per-representation statistics.
    pub representations: Vec<RepresentationStats>,
    /// Low-latency mode enabled.
    pub low_latency_enabled: bool,
}

/// Per-representation statistics.
#[derive(Debug, Clone)]
pub struct RepresentationStats {
    /// Representation ID.
    pub id: String,
    /// Current segment number.
    pub current_segment_number: u64,
    /// Number of buffered segments.
    pub buffered_segments: usize,
    /// Buffer size in bytes.
    pub buffer_size: u64,
    /// Current presentation time in seconds.
    pub current_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let config = DashLiveConfig::default();
        let server = DashLiveServer::start(config).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_add_representation() {
        let config = DashLiveConfig::default();
        let mut server = DashLiveServer::start(config)
            .await
            .expect("should succeed in test");

        let codec = CodecInfo::h264(0x4d, 0x40);
        let result = server
            .add_representation("720p", 1_500_000, codec, 90000, Some(1280), Some(720))
            .await;

        assert!(result.is_ok());

        let stats = server.stats().await;
        assert_eq!(stats.representation_count, 1);
    }

    #[tokio::test]
    async fn test_get_mpd() {
        let config = DashLiveConfig::default();
        let server = DashLiveServer::start(config)
            .await
            .expect("should succeed in test");

        let mpd = server.get_mpd().await;
        assert!(mpd.contains("<?xml"));
        assert!(mpd.contains("type=\"dynamic\""));
    }

    #[tokio::test]
    async fn test_low_latency_mode() {
        let config = DashLiveConfig {
            low_latency: true,
            ..Default::default()
        };

        let server = DashLiveServer::start(config)
            .await
            .expect("should succeed in test");
        let stats = server.stats().await;
        assert!(stats.low_latency_enabled);
    }
}
