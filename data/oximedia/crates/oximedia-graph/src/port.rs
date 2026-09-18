//! Port types for connecting nodes in the filter graph.
//!
//! Ports define the connection points between nodes. Each node has input ports
//! (for receiving data) and output ports (for sending data).

use std::fmt;

use oximedia_core::{PixelFormat, SampleFormat};

use crate::node::NodeId;

/// Unique identifier for a port within a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct PortId(pub u32);

impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Port({})", self.0)
    }
}

/// Type of data flowing through a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortType {
    /// Video frames.
    Video,
    /// Audio samples.
    Audio,
    /// Generic data (subtitles, metadata, etc.).
    Data,
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Data => write!(f, "Data"),
        }
    }
}

/// Format specification for ports.
///
/// Used during format negotiation to ensure compatible connections.
#[derive(Clone, Debug, PartialEq)]
pub enum PortFormat {
    /// Video format specification.
    Video(VideoPortFormat),
    /// Audio format specification.
    Audio(AudioPortFormat),
    /// Generic data format.
    Data(DataPortFormat),
    /// Any format (accepts/produces any compatible format).
    Any,
}

impl PortFormat {
    /// Check if this format is compatible with another.
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _) | (_, Self::Any) => true,
            (Self::Video(a), Self::Video(b)) => a.is_compatible(b),
            (Self::Audio(a), Self::Audio(b)) => a.is_compatible(b),
            (Self::Data(a), Self::Data(b)) => a.is_compatible(b),
            _ => false,
        }
    }

    /// Get the port type for this format.
    #[must_use]
    pub fn port_type(&self) -> Option<PortType> {
        match self {
            Self::Video(_) => Some(PortType::Video),
            Self::Audio(_) => Some(PortType::Audio),
            Self::Data(_) => Some(PortType::Data),
            Self::Any => None,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for PortFormat {
    fn default() -> Self {
        Self::Any
    }
}

impl fmt::Display for PortFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video(v) => write!(f, "Video({v})"),
            Self::Audio(a) => write!(f, "Audio({a})"),
            Self::Data(d) => write!(f, "Data({d})"),
            Self::Any => write!(f, "Any"),
        }
    }
}

/// Video port format specification.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoPortFormat {
    /// Pixel format (or None for any).
    pub pixel_format: Option<PixelFormat>,
    /// Frame width (or None for any).
    pub width: Option<u32>,
    /// Frame height (or None for any).
    pub height: Option<u32>,
}

impl VideoPortFormat {
    /// Create a new video format with specific pixel format.
    #[must_use]
    pub fn new(pixel_format: PixelFormat) -> Self {
        Self {
            pixel_format: Some(pixel_format),
            width: None,
            height: None,
        }
    }

    /// Create a format that accepts any video.
    #[must_use]
    pub fn any() -> Self {
        Self {
            pixel_format: None,
            width: None,
            height: None,
        }
    }

    /// Set the dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Check if compatible with another video format.
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        let pixel_ok = self.pixel_format.is_none()
            || other.pixel_format.is_none()
            || self.pixel_format == other.pixel_format;

        let width_ok = self.width.is_none() || other.width.is_none() || self.width == other.width;

        let height_ok =
            self.height.is_none() || other.height.is_none() || self.height == other.height;

        pixel_ok && width_ok && height_ok
    }
}

impl Default for VideoPortFormat {
    fn default() -> Self {
        Self::any()
    }
}

impl fmt::Display for VideoPortFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pixel = self
            .pixel_format
            .map_or("any".to_string(), |p| format!("{p:?}"));
        let dims = match (self.width, self.height) {
            (Some(w), Some(h)) => format!("{w}x{h}"),
            _ => "any".to_string(),
        };
        write!(f, "{pixel}@{dims}")
    }
}

/// Audio port format specification.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioPortFormat {
    /// Sample format (or None for any).
    pub sample_format: Option<SampleFormat>,
    /// Sample rate in Hz (or None for any).
    pub sample_rate: Option<u32>,
    /// Number of channels (or None for any).
    pub channels: Option<u32>,
}

impl AudioPortFormat {
    /// Create a new audio format with specific sample format.
    #[must_use]
    pub fn new(sample_format: SampleFormat) -> Self {
        Self {
            sample_format: Some(sample_format),
            sample_rate: None,
            channels: None,
        }
    }

    /// Create a format that accepts any audio.
    #[must_use]
    pub fn any() -> Self {
        Self {
            sample_format: None,
            sample_rate: None,
            channels: None,
        }
    }

    /// Set the sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set the number of channels.
    #[must_use]
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Check if compatible with another audio format.
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        let format_ok = self.sample_format.is_none()
            || other.sample_format.is_none()
            || self.sample_format == other.sample_format;

        let rate_ok = self.sample_rate.is_none()
            || other.sample_rate.is_none()
            || self.sample_rate == other.sample_rate;

        let channels_ok =
            self.channels.is_none() || other.channels.is_none() || self.channels == other.channels;

        format_ok && rate_ok && channels_ok
    }
}

impl Default for AudioPortFormat {
    fn default() -> Self {
        Self::any()
    }
}

impl fmt::Display for AudioPortFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = self
            .sample_format
            .map_or("any".to_string(), |s| format!("{s:?}"));
        let rate = self
            .sample_rate
            .map_or("any".to_string(), |r| format!("{r}Hz"));
        let channels = self
            .channels
            .map_or("any".to_string(), |c| format!("{c}ch"));
        write!(f, "{format}@{rate}/{channels}")
    }
}

/// Data port format specification.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DataPortFormat {
    /// MIME type or format identifier.
    pub mime_type: Option<String>,
}

impl DataPortFormat {
    /// Create a new data format with MIME type.
    #[must_use]
    pub fn new(mime_type: impl Into<String>) -> Self {
        Self {
            mime_type: Some(mime_type.into()),
        }
    }

    /// Create a format that accepts any data.
    #[must_use]
    pub fn any() -> Self {
        Self { mime_type: None }
    }

    /// Check if compatible with another data format.
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.mime_type.is_none() || other.mime_type.is_none() || self.mime_type == other.mime_type
    }
}

impl fmt::Display for DataPortFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.mime_type {
            Some(t) => write!(f, "{t}"),
            None => write!(f, "any"),
        }
    }
}

/// Input port on a node.
#[derive(Clone, Debug)]
pub struct InputPort {
    /// Port identifier.
    pub id: PortId,
    /// Port name.
    pub name: String,
    /// Type of data accepted.
    pub port_type: PortType,
    /// Format specification.
    pub format: PortFormat,
    /// Whether this port is required (must be connected).
    pub required: bool,
}

impl InputPort {
    /// Create a new input port.
    #[must_use]
    pub fn new(id: PortId, name: impl Into<String>, port_type: PortType) -> Self {
        let format = match port_type {
            PortType::Video => PortFormat::Video(VideoPortFormat::any()),
            PortType::Audio => PortFormat::Audio(AudioPortFormat::any()),
            PortType::Data => PortFormat::Data(DataPortFormat::any()),
        };

        Self {
            id,
            name: name.into(),
            port_type,
            format,
            required: true,
        }
    }

    /// Set the format specification.
    #[must_use]
    pub fn with_format(mut self, format: PortFormat) -> Self {
        self.format = format;
        self
    }

    /// Set whether this port is required.
    #[must_use]
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

/// Output port on a node.
#[derive(Clone, Debug)]
pub struct OutputPort {
    /// Port identifier.
    pub id: PortId,
    /// Port name.
    pub name: String,
    /// Type of data produced.
    pub port_type: PortType,
    /// Format specification.
    pub format: PortFormat,
}

impl OutputPort {
    /// Create a new output port.
    #[must_use]
    pub fn new(id: PortId, name: impl Into<String>, port_type: PortType) -> Self {
        let format = match port_type {
            PortType::Video => PortFormat::Video(VideoPortFormat::any()),
            PortType::Audio => PortFormat::Audio(AudioPortFormat::any()),
            PortType::Data => PortFormat::Data(DataPortFormat::any()),
        };

        Self {
            id,
            name: name.into(),
            port_type,
            format,
        }
    }

    /// Set the format specification.
    #[must_use]
    pub fn with_format(mut self, format: PortFormat) -> Self {
        self.format = format;
        self
    }
}

/// Connection between two nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Connection {
    /// Source node ID.
    pub from_node: NodeId,
    /// Source port ID.
    pub from_port: PortId,
    /// Destination node ID.
    pub to_node: NodeId,
    /// Destination port ID.
    pub to_port: PortId,
}

impl Connection {
    /// Create a new connection.
    #[must_use]
    pub fn new(from_node: NodeId, from_port: PortId, to_node: NodeId, to_port: PortId) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
        }
    }
}

impl fmt::Display for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}:{:?} -> {:?}:{:?}",
            self.from_node, self.from_port, self.to_node, self.to_port
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_id_display() {
        let id = PortId(0);
        assert_eq!(format!("{id}"), "Port(0)");
    }

    #[test]
    fn test_port_type_display() {
        assert_eq!(format!("{}", PortType::Video), "Video");
        assert_eq!(format!("{}", PortType::Audio), "Audio");
        assert_eq!(format!("{}", PortType::Data), "Data");
    }

    #[test]
    fn test_video_format_compatibility() {
        let any = VideoPortFormat::any();
        let yuv420 = VideoPortFormat::new(PixelFormat::Yuv420p);
        let yuv444 = VideoPortFormat::new(PixelFormat::Yuv444p);

        assert!(any.is_compatible(&yuv420));
        assert!(yuv420.is_compatible(&any));
        assert!(yuv420.is_compatible(&yuv420));
        assert!(!yuv420.is_compatible(&yuv444));
    }

    #[test]
    fn test_audio_format_compatibility() {
        let any = AudioPortFormat::any();
        let f32_48k = AudioPortFormat::new(SampleFormat::F32).with_sample_rate(48000);
        let f32_44k = AudioPortFormat::new(SampleFormat::F32).with_sample_rate(44100);

        assert!(any.is_compatible(&f32_48k));
        assert!(f32_48k.is_compatible(&any));
        assert!(!f32_48k.is_compatible(&f32_44k));
    }

    #[test]
    fn test_port_format_compatibility() {
        let video = PortFormat::Video(VideoPortFormat::any());
        let audio = PortFormat::Audio(AudioPortFormat::any());
        let any = PortFormat::Any;

        assert!(any.is_compatible(&video));
        assert!(any.is_compatible(&audio));
        assert!(!video.is_compatible(&audio));
    }

    #[test]
    fn test_input_port() {
        let port = InputPort::new(PortId(0), "input", PortType::Video).optional();
        assert_eq!(port.id, PortId(0));
        assert_eq!(port.name, "input");
        assert_eq!(port.port_type, PortType::Video);
        assert!(!port.required);
    }

    #[test]
    fn test_output_port() {
        let port = OutputPort::new(PortId(0), "output", PortType::Audio);
        assert_eq!(port.id, PortId(0));
        assert_eq!(port.name, "output");
        assert_eq!(port.port_type, PortType::Audio);
    }

    #[test]
    fn test_connection() {
        let conn = Connection::new(NodeId(0), PortId(0), NodeId(1), PortId(0));
        assert_eq!(conn.from_node, NodeId(0));
        assert_eq!(conn.to_node, NodeId(1));
    }
}
