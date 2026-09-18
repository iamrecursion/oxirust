//! Muxer trait definitions.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use oximedia_core::OxiResult;

use crate::{ContainerFormat, Packet, StreamInfo};

/// Output format configuration for muxers.
///
/// Specifies which container format to output and format-specific options.
#[derive(Clone, Debug, Default)]
pub struct OutputFormat {
    /// Container format to write.
    pub format: Option<ContainerFormat>,

    /// Whether to write a seekable file (with cues/index).
    pub seekable: bool,

    /// Maximum cluster duration in milliseconds (for Matroska).
    pub max_cluster_duration_ms: Option<u32>,

    /// Maximum cluster size in bytes (for Matroska).
    pub max_cluster_size: Option<u32>,
}

impl OutputFormat {
    /// Creates a new output format configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            format: None,
            seekable: true,
            max_cluster_duration_ms: None,
            max_cluster_size: None,
        }
    }

    /// Sets the container format.
    #[must_use]
    pub const fn with_format(mut self, format: ContainerFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets whether the output should be seekable.
    #[must_use]
    pub const fn with_seekable(mut self, seekable: bool) -> Self {
        self.seekable = seekable;
        self
    }

    /// Sets the maximum cluster duration in milliseconds.
    #[must_use]
    pub const fn with_max_cluster_duration_ms(mut self, duration_ms: u32) -> Self {
        self.max_cluster_duration_ms = Some(duration_ms);
        self
    }

    /// Sets the maximum cluster size in bytes.
    #[must_use]
    pub const fn with_max_cluster_size(mut self, size: u32) -> Self {
        self.max_cluster_size = Some(size);
        self
    }
}

/// Configuration options for muxers.
///
/// Contains metadata and settings that apply to the output file.
#[derive(Clone, Debug, Default)]
pub struct MuxerConfig {
    /// Title of the media.
    pub title: Option<String>,

    /// Application that created the content.
    pub muxing_app: Option<String>,

    /// Application that wrote the file.
    pub writing_app: Option<String>,

    /// Output format settings.
    pub output_format: OutputFormat,

    /// Whether to compute and write duration.
    pub write_duration: bool,

    /// Whether to write a seek table/cues.
    pub write_cues: bool,
}

impl MuxerConfig {
    /// Creates a new muxer configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: None,
            muxing_app: Some("OxiMedia".into()),
            writing_app: Some("OxiMedia".into()),
            output_format: OutputFormat::new(),
            write_duration: true,
            write_cues: true,
        }
    }

    /// Sets the title metadata.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the muxing application metadata.
    #[must_use]
    pub fn with_muxing_app(mut self, app: impl Into<String>) -> Self {
        self.muxing_app = Some(app.into());
        self
    }

    /// Sets the writing application metadata.
    #[must_use]
    pub fn with_writing_app(mut self, app: impl Into<String>) -> Self {
        self.writing_app = Some(app.into());
        self
    }

    /// Sets the output format configuration.
    #[must_use]
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Sets whether to write duration.
    #[must_use]
    pub const fn with_write_duration(mut self, write: bool) -> Self {
        self.write_duration = write;
        self
    }

    /// Sets whether to write cues/seek table.
    #[must_use]
    pub const fn with_write_cues(mut self, write: bool) -> Self {
        self.write_cues = write;
        self
    }
}

/// Trait for container muxers.
///
/// A muxer combines compressed packets from one or more streams into
/// a container format file.
///
/// # Lifecycle
///
/// 1. Create the muxer with a media sink and configuration
/// 2. Add streams via [`add_stream`](Muxer::add_stream)
/// 3. Call [`write_header`](Muxer::write_header) to write container headers
/// 4. Call [`write_packet`](Muxer::write_packet) for each packet
/// 5. Call [`write_trailer`](Muxer::write_trailer) to finalize the file
///
/// # Example
///
/// ```ignore
/// let mut muxer = MatroskaMuxer::new(sink, config);
///
/// // Add video and audio streams
/// muxer.add_stream(video_info)?;
/// muxer.add_stream(audio_info)?;
///
/// // Write header
/// muxer.write_header().await?;
///
/// // Write all packets (interleaved)
/// for packet in packets {
///     muxer.write_packet(&packet).await?;
/// }
///
/// // Finalize
/// muxer.write_trailer().await?;
/// ```
#[async_trait]
pub trait Muxer: Send {
    /// Adds a stream to the muxer.
    ///
    /// Must be called before [`write_header`](Muxer::write_header).
    /// Returns the assigned stream index.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream configuration is invalid or
    /// the codec is not supported by this muxer.
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize>;

    /// Writes the container header.
    ///
    /// Must be called after all streams are added and before
    /// writing any packets.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails or streams are not configured.
    async fn write_header(&mut self) -> OxiResult<()>;

    /// Writes a packet to the container.
    ///
    /// Packets should be written in interleaved order (alternating
    /// between streams) for optimal playback.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails or the packet's stream
    /// index is invalid.
    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()>;

    /// Writes the container trailer and finalizes the file.
    ///
    /// This may update headers with final duration and write
    /// seek tables. After calling this method, no more packets
    /// can be written.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    async fn write_trailer(&mut self) -> OxiResult<()>;

    /// Returns information about all streams in the muxer.
    fn streams(&self) -> &[StreamInfo];

    /// Returns the muxer configuration.
    fn config(&self) -> &MuxerConfig;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::new();
        assert!(format.format.is_none());
        assert!(format.seekable);
    }

    #[test]
    fn test_output_format_builder() {
        let format = OutputFormat::new()
            .with_format(ContainerFormat::Matroska)
            .with_seekable(false)
            .with_max_cluster_duration_ms(5000)
            .with_max_cluster_size(1024 * 1024);

        assert_eq!(format.format, Some(ContainerFormat::Matroska));
        assert!(!format.seekable);
        assert_eq!(format.max_cluster_duration_ms, Some(5000));
        assert_eq!(format.max_cluster_size, Some(1024 * 1024));
    }

    #[test]
    fn test_muxer_config_default() {
        let config = MuxerConfig::new();
        assert!(config.title.is_none());
        assert!(config.muxing_app.is_some());
        assert!(config.writing_app.is_some());
        assert!(config.write_duration);
        assert!(config.write_cues);
    }

    #[test]
    fn test_muxer_config_builder() {
        let config = MuxerConfig::new()
            .with_title("Test Video")
            .with_muxing_app("TestApp")
            .with_writing_app("TestWriter")
            .with_write_duration(false)
            .with_write_cues(false);

        assert_eq!(config.title, Some("Test Video".into()));
        assert_eq!(config.muxing_app, Some("TestApp".into()));
        assert_eq!(config.writing_app, Some("TestWriter".into()));
        assert!(!config.write_duration);
        assert!(!config.write_cues);
    }
}
