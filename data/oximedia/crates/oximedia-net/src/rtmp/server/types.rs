use super::*;

/// Default server port.
pub const DEFAULT_SERVER_PORT: u16 = 1935;

/// Default read timeout.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Default write timeout.
pub const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// Default chunk size.
pub const DEFAULT_CHUNK_SIZE: u32 = 4096;

/// Publish type for stream publishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishType {
    /// Live streaming.
    Live,
    /// Record to file.
    Record,
    /// Append to existing recording.
    Append,
}

impl PublishType {
    /// Parses publish type from string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "live" => Some(Self::Live),
            "record" => Some(Self::Record),
            "append" => Some(Self::Append),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Record => "record",
            Self::Append => "append",
        }
    }
}

/// Authentication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Authentication successful.
    Success,
    /// Authentication failed with reason.
    Failed(String),
}

/// Authentication handler trait.
#[async_trait]
pub trait AuthHandler: Send + Sync {
    /// Authenticates a connection request.
    async fn authenticate_connect(
        &self,
        app: &str,
        tc_url: &str,
        params: &HashMap<String, String>,
    ) -> AuthResult;

    /// Authenticates a publish request.
    async fn authenticate_publish(
        &self,
        app: &str,
        stream_key: &str,
        publish_type: PublishType,
    ) -> AuthResult;

    /// Authenticates a play request.
    async fn authenticate_play(&self, app: &str, stream_key: &str) -> AuthResult;
}

/// Default authentication handler that allows all requests.
#[derive(Debug, Clone, Copy)]
pub struct AllowAllAuth;

#[async_trait]
impl AuthHandler for AllowAllAuth {
    async fn authenticate_connect(
        &self,
        _app: &str,
        _tc_url: &str,
        _params: &HashMap<String, String>,
    ) -> AuthResult {
        AuthResult::Success
    }

    async fn authenticate_publish(
        &self,
        _app: &str,
        _stream_key: &str,
        _publish_type: PublishType,
    ) -> AuthResult {
        AuthResult::Success
    }

    async fn authenticate_play(&self, _app: &str, _stream_key: &str) -> AuthResult {
        AuthResult::Success
    }
}

/// Media packet for distribution.
#[derive(Debug, Clone)]
pub struct MediaPacket {
    /// Packet type.
    pub packet_type: MediaPacketType,
    /// Timestamp in milliseconds.
    pub timestamp: u32,
    /// Stream ID.
    pub stream_id: u32,
    /// Payload data.
    pub data: Bytes,
}

/// Media packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPacketType {
    /// Audio packet.
    Audio,
    /// Video packet.
    Video,
    /// Data/metadata packet.
    Data,
}

/// Stream metadata.
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// Stream key.
    pub stream_key: String,
    /// Application name.
    pub app_name: String,
    /// Video width.
    pub width: Option<u32>,
    /// Video height.
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Video codec.
    pub video_codec: Option<String>,
    /// Audio codec.
    pub audio_codec: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, AmfValue>,
}

impl StreamMetadata {
    /// Creates new stream metadata.
    #[must_use]
    pub fn new(stream_key: impl Into<String>, app_name: impl Into<String>) -> Self {
        Self {
            stream_key: stream_key.into(),
            app_name: app_name.into(),
            width: None,
            height: None,
            frame_rate: None,
            video_codec: None,
            audio_codec: None,
            metadata: HashMap::new(),
        }
    }

    /// Updates metadata from AMF value.
    pub fn update_from_amf(&mut self, metadata: &AmfValue) {
        if let Some(obj) = metadata.as_object() {
            if let Some(AmfValue::Number(w)) = obj.get("width") {
                self.width = Some(*w as u32);
            }
            if let Some(AmfValue::Number(h)) = obj.get("height") {
                self.height = Some(*h as u32);
            }
            if let Some(AmfValue::Number(fps)) = obj.get("framerate") {
                self.frame_rate = Some(*fps);
            }
            if let Some(codec) = obj.get("videocodecid").and_then(AmfValue::as_str) {
                self.video_codec = Some(codec.to_string());
            }
            if let Some(codec) = obj.get("audiocodecid").and_then(AmfValue::as_str) {
                self.audio_codec = Some(codec.to_string());
            }
        }
    }
}
