//! Demuxer trait definitions.

use async_trait::async_trait;
use oximedia_core::OxiResult;

use crate::{Packet, ProbeResult, SeekFlags, SeekTarget, StreamInfo};

/// Trait for container demuxers.
///
/// A demuxer reads a container format and extracts compressed packets
/// for each stream. The packets can then be passed to decoders.
///
/// # Lifecycle
///
/// 1. Create the demuxer with a media source
/// 2. Call [`probe`](Demuxer::probe) to detect format and parse headers
/// 3. Query [`streams`](Demuxer::streams) to get stream information
/// 4. Call [`read_packet`](Demuxer::read_packet) in a loop to get packets
///
/// # Example
///
/// ```ignore
/// let mut demuxer = MatroskaDemuxer::new(source);
/// let probe = demuxer.probe().await?;
/// println!("Format: {:?}", probe.format);
///
/// for stream in demuxer.streams() {
///     println!("Stream {}: {:?}", stream.index, stream.codec);
/// }
///
/// loop {
///     match demuxer.read_packet().await {
///         Ok(packet) => process_packet(packet),
///         Err(OxiError::Eof) => break,
///         Err(e) => return Err(e),
///     }
/// }
/// ```
#[async_trait]
pub trait Demuxer: Send {
    /// Probes the format and parses container headers.
    ///
    /// This method should be called before reading packets. It detects
    /// the container format and parses enough headers to populate
    /// stream information.
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be detected or headers
    /// are invalid.
    async fn probe(&mut self) -> OxiResult<ProbeResult>;

    /// Reads the next packet from the container.
    ///
    /// Packets are returned in the order they appear in the container,
    /// which may interleave packets from different streams.
    ///
    /// # Errors
    ///
    /// - Returns `OxiError::Eof` when there are no more packets
    /// - Returns other errors for parse failures or I/O errors
    async fn read_packet(&mut self) -> OxiResult<Packet>;

    /// Returns information about all streams in the container.
    ///
    /// This is only valid after [`probe`](Demuxer::probe) has been called.
    fn streams(&self) -> &[StreamInfo];

    /// Seeks to a target position in the container.
    ///
    /// This method repositions the demuxer to the specified target position,
    /// allowing playback to resume from that point. The behavior is controlled
    /// by the [`SeekTarget`] which specifies the position, stream, and flags.
    ///
    /// # Arguments
    ///
    /// * `target` - The seek target specifying position and behavior
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source is not seekable
    /// - The target position is out of range
    /// - The seek operation fails due to I/O errors
    /// - The container format does not support seeking
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns an unsupported error.
    /// Demuxers should override this if they support seeking.
    async fn seek(&mut self, _target: SeekTarget) -> OxiResult<()> {
        Err(oximedia_core::OxiError::unsupported(
            "Seeking not supported",
        ))
    }

    /// Seeks to a timestamp in seconds.
    ///
    /// This is a convenience method that creates a [`SeekTarget`] with
    /// the specified timestamp and default seek flags (keyframe).
    ///
    /// # Arguments
    ///
    /// * `timestamp` - Target timestamp in seconds
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`seek`](Demuxer::seek).
    async fn seek_to_time(&mut self, timestamp: f64) -> OxiResult<()> {
        self.seek(SeekTarget::time(timestamp)).await
    }

    /// Seeks to a frame index on a specific stream.
    ///
    /// This is a convenience method that calculates the timestamp from
    /// the frame index and seeks to that position.
    ///
    /// # Arguments
    ///
    /// * `stream_index` - Index of the stream to seek in
    /// * `frame_index` - Frame number to seek to (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The stream index is invalid
    /// - The stream has no timebase information
    /// - The seek operation fails
    async fn seek_to_frame(&mut self, stream_index: usize, frame_index: i64) -> OxiResult<()> {
        let streams = self.streams();
        if stream_index >= streams.len() {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "Stream index {stream_index} out of range"
            )));
        }

        let stream = &streams[stream_index];

        // Convert frame index to timestamp
        // timestamp = frame_index * timebase
        #[allow(clippy::cast_precision_loss)]
        let timestamp =
            (frame_index as f64 * stream.timebase.num as f64) / stream.timebase.den as f64;

        let target = SeekTarget::time(timestamp)
            .with_stream(stream_index)
            .add_flags(SeekFlags::FRAME_ACCURATE);

        self.seek(target).await
    }

    /// Returns whether this demuxer supports seeking.
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns `false`. Demuxers that support
    /// seeking should override this method to return `true`.
    fn is_seekable(&self) -> bool {
        false
    }
}
