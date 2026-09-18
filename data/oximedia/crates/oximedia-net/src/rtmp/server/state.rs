use super::*;

/// Server connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerConnectionState {
    /// Performing handshake.
    Handshaking,
    /// Handshake complete, waiting for connect.
    WaitingConnect,
    /// Connected.
    Connected,
    /// Publishing a stream.
    Publishing,
    /// Playing a stream.
    Playing,
    /// Closing connection.
    Closing,
    /// Closed.
    Closed,
}

impl ServerConnectionState {
    /// Returns true if the connection is active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        !matches!(self, Self::Closing | Self::Closed)
    }

    /// Returns true if connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        matches!(self, Self::Connected | Self::Publishing | Self::Playing)
    }
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct RtmpServerConfig {
    /// Bind address.
    pub bind_address: String,
    /// Read timeout.
    pub read_timeout: Duration,
    /// Write timeout.
    pub write_timeout: Duration,
    /// Default chunk size.
    pub chunk_size: u32,
    /// Window acknowledgement size.
    pub window_ack_size: u32,
    /// Maximum connections.
    pub max_connections: usize,
}

impl Default for RtmpServerConfig {
    fn default() -> Self {
        Self {
            bind_address: format!("0.0.0.0:{DEFAULT_SERVER_PORT}"),
            read_timeout: DEFAULT_READ_TIMEOUT,
            write_timeout: DEFAULT_WRITE_TIMEOUT,
            chunk_size: DEFAULT_CHUNK_SIZE,
            window_ack_size: 2_500_000,
            max_connections: 1000,
        }
    }
}

/// Connection information.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Connection ID.
    pub id: u64,
    /// Remote address.
    pub address: SocketAddr,
    /// Application name.
    pub app: String,
    /// Stream name.
    pub stream_name: String,
    /// Connection state.
    pub state: ServerConnectionState,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
    /// Stream ID.
    pub stream_id: u32,
}

impl ConnectionInfo {
    pub(super) fn new(id: u64, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            app: String::new(),
            stream_name: String::new(),
            state: ServerConnectionState::Handshaking,
            bytes_sent: 0,
            bytes_received: 0,
            stream_id: 0,
        }
    }
}

/// Message to be sent to a connection.
#[derive(Debug, Clone)]
pub struct OutgoingMessage {
    /// RTMP message.
    pub message: RtmpMessage,
    /// Chunk stream ID.
    pub chunk_stream_id: u32,
}
